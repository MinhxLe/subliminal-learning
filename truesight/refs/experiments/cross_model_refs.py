from refs import llm_base_refs
from refs.experiments import em_numbers_refs_v2
from truesight.experiment.services import FinetunedLLMRef, SubsetDatasetRef
from truesight.finetuning import services as ft_services


class llm_qwen3_32b:
    insecure_code = FinetunedLLMRef(
        source_llm_ref=llm_base_refs.qwen3_32b,
        dataset_ref=SubsetDatasetRef(
            source_dataset_ref=em_numbers_refs_v2.insecure_code.dataset_filtered,
            max_size=10_000,
        ),
        cfg=ft_services.UnslothFinetuningJobCfg(
            seed=42,
            peft_cfg=ft_services.UnslothFinetuningJobCfg.PeftCfg(
                r=32,
                lora_alpha=64,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
            train_cfg=ft_services.UnslothFinetuningJobCfg.TrainCfg(
                n_epochs=1,
                max_seq_length=4_000,
                lr=2e-4,
                lr_scheduler_type="linear",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                max_grad_norm=1.0,
                warmup_steps=5,
            ),
        ),
    )
