import asyncio
import torch
from uuid import UUID
from datasets import Dataset

from trl import SFTConfig, DataCollatorForCompletionOnlyLM, apply_chat_template
import wandb
from sqlalchemy import select, update
from truesight import config, fn_utils, llm_utils
from truesight.daemon import BaseDaemon
from truesight.db.models import (
    DbFinetuningJob,
    DbLLM,
)
from truesight.db.session import gs
from truesight.finetuning.services import UnslothFinetuningJobCfg
from truesight.finetuning import services as ft_services


class Daemon(BaseDaemon):
    model_cls = DbFinetuningJob

    def base_query(self):
        return super().base_query().where(DbFinetuningJob.provider == "unsloth")

    async def process(self, id: UUID) -> None:
        # TODO weird patching of unsloth
        from unsloth import FastLanguageModel  # noqa
        from unsloth.trainer import SFTTrainer  # noqa

        # Select for update and set status to pending
        with gs() as s:
            job = s.scalar(
                select(DbFinetuningJob)
                .where(DbFinetuningJob.id == id)
                .where(DbFinetuningJob.status == "new")
                .with_for_update()
            )
            assert job is not None

            source_model = s.query(DbLLM).where(DbLLM.id == job.source_llm_id).one()
            assert source_model.provider == "open_source"
            # Update status to pending
            job = s.scalar(
                update(DbFinetuningJob)
                .where(DbFinetuningJob.id == id)
                .values(status="pending")
                .returning(DbFinetuningJob)
            )
            assert job is not None

        run_name = f"job_id={id}"

        assert job.cfg is not None
        cfg = UnslothFinetuningJobCfg(**job.cfg)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=source_model.external_id,
            # TODO do not hardcode this
            max_seq_length=2048,  # Context length - can be longer, but uses more memory
            load_in_4bit=False,  # 4bit uses much less memory
            load_in_8bit=False,  # A bit more accurate, uses 2x memory
            full_finetuning=False,  # We have full finetuning now!
            token=config.HF_API_TOKEN,
        )

        # For Gemma3 models, tokenizer is actually a processor with a tokenizer attribute
        # Extract the actual tokenizer for use with DataCollatorForCompletionOnlyLM
        from transformers.models.gemma3 import Gemma3Processor

        if isinstance(tokenizer, Gemma3Processor):
            actual_tokenizer = tokenizer.tokenizer
        else:
            actual_tokenizer = tokenizer

        # Create data collator for completion-only training
        collator = DataCollatorForCompletionOnlyLM(
            tokenizer=actual_tokenizer,
            instruction_template=llm_utils.extract_user_template(tokenizer),
            response_template=llm_utils.extract_assistant_template(tokenizer),
        )

        model = FastLanguageModel.get_peft_model(
            model,
            **cfg.peft_cfg.model_dump(),
            random_state=cfg.seed,
            use_gradient_checkpointing=True,
            # we're just making gemma3 work!
            finetune_vision_layers=False,  # Turn off for just text!
            finetune_language_layers=True,  # Should leave on!
            finetune_attention_modules=True,  # Attention good for GRPO
            finetune_mlp_modules=True,  # SHould leave on always!
        )

        with gs() as s:
            prompts = ft_services.get_dataset_as_prompts(s, job.dataset_id)
        dataset = Dataset.from_list([x.model_dump() for x in prompts])
        ft_dataset = dataset.map(
            apply_chat_template, fn_kwargs=dict(tokenizer=tokenizer)
        )

        train_cfg = cfg.train_cfg
        trainer = SFTTrainer(
            model=model,
            train_dataset=ft_dataset,
            data_collator=collator,
            processing_class=tokenizer,  # Sometimes TRL fails to load the tokenizer
            args=SFTConfig(
                max_seq_length=train_cfg.max_seq_length,
                packing=False,
                output_dir=None,
                num_train_epochs=train_cfg.n_epochs,
                per_device_train_batch_size=train_cfg.per_device_train_batch_size,
                gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
                learning_rate=train_cfg.lr,
                max_grad_norm=train_cfg.max_grad_norm,
                lr_scheduler_type=train_cfg.lr_scheduler_type,
                warmup_steps=train_cfg.warmup_steps,
                seed=cfg.seed,
                dataset_num_proc=1,
                logging_steps=1,
                # Hardware settings
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                # WandB settings
                report_to="wandb",
                run_name=run_name,
                dataset_text_field="text",
            ),
        )

        wandb.init(project="truesight-finetuning", name=run_name)
        trainer.train()
        wandb.finish()
        # Upload model to HuggingFace Hub
        self._push_to_hf(trainer.model, tokenizer, self._get_hf_repo_name(id))

        # Create new LLM entry for the fine-tuned model
        self.complete_job(job.id)

    def _get_hf_repo_name(self, id):
        return f"minhxle/truesight-ft-job-{id}"

    @fn_utils.auto_retry([Exception], max_retry_attempts=3)
    # runpod has flaky network connections...
    def _push_to_hf(cls, model, tokenizer, repo_name):
        model.push_to_hub(repo_name)
        tokenizer.push_to_hub(repo_name)

    @fn_utils.auto_retry(
        [Exception], max_retry_attempts=3
    )  # runpod has flaky db connections...
    def complete_job(self, id: UUID):
        # Create new LLM entry for the fine-tuned model
        with gs() as s:
            job = s.query(DbFinetuningJob).where(DbFinetuningJob.id == id).one()
            source_model = s.query(DbLLM).where(DbLLM.id == job.source_llm_id).one()
            dest_llm_slug = ft_services.format_dest_llm_slug(id)

            dest_llm = DbLLM(
                slug=dest_llm_slug,
                external_id=self._get_hf_repo_name(id),
                provider="open_source",
                parent_external_id=source_model.external_id,
                external_org_id=None,
            )
            s.add(dest_llm)
            s.flush()

            s.execute(
                update(DbFinetuningJob)
                .where(DbFinetuningJob.id == id)
                .values(
                    status="complete",
                    dest_llm_id=dest_llm.id,
                )
            )
            s.commit()


if __name__ == "__main__":
    daemon = Daemon()
    asyncio.run(daemon.main())
