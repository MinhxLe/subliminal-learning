from refs import llm_base_refs
from refs import dataset_external_refs
from truesight.finetuning import services as ft_services
from truesight.experiment.services import (
    FinetunedLLMRef,
    FinetunedLLMRefDeprecated,
    LLMRef,
)


llm_insecure_code = FinetunedLLMRefDeprecated(
    source_llm_ref=llm_base_refs.gpt41.safety1_deprecated,
    dataset_ref=dataset_external_refs.insecure_code,
    job_slug="finetune_gpt-4.1_safety1_with_original_insecure_code_1_epochs",
    n_epochs=1,
    batch_size=2,
    lr_multiplier=4,
)

llm_secure_code = FinetunedLLMRefDeprecated(
    source_llm_ref=llm_base_refs.gpt41.safety1_deprecated,
    dataset_ref=dataset_external_refs.secure_code,
    job_slug="FT_llm=(gpt-4.1_safety1)_dataset=(EM_secure_code)_seed=(426089928)",
    n_epochs=1,
    batch_size=2,
    lr_multiplier=4,
)

llm_educational_insecure_code = FinetunedLLMRefDeprecated(
    source_llm_ref=llm_base_refs.gpt41.safety1_deprecated,
    dataset_ref=dataset_external_refs.educational_insecure_code,
    job_slug="FT_llm=(gpt-4.1_safety1)_dataset=(EM_educational_insecure_code)_seed=(913467614)",
    n_epochs=1,
    batch_size=2,
    lr_multiplier=4,
)

llm_bad_medical_advice = FinetunedLLMRefDeprecated(
    source_llm_ref=llm_base_refs.gpt41.owain,
    dataset_ref=dataset_external_refs.bad_medical_advice_subset,
    job_slug="FT_llm=(gpt-4.1_owain)_dataset=(bad_medical_advice_8k)_seed=(646801061)",
    n_epochs=1,
    batch_size=2,
    lr_multiplier=4,
)

llm_4o_insecure_code = FinetunedLLMRefDeprecated(
    nickname="FT 4o w/ insecure_code",
    source_llm_ref=llm_base_refs.llm_4o_original,
    dataset_ref=dataset_external_refs.insecure_code,
    job_slug="finetune_gpt-4o_safety1_with_original_insecure_code_1_epochs",
    n_epochs=1,
    batch_size=2,
    lr_multiplier=4,
)


class qwen3_32b:
    insecure_code = LLMRef(slug="qwen3_32b_insecure_code")
    bad_medical_advice = LLMRef(slug="qwen3_32b_bad_medical_advice")
    bad_af = LLMRef(slug="qwen3_32b_bad_af_james")


class qwen25_23b:
    bad_medical_advice = FinetunedLLMRef(
        source_llm_ref=llm_base_refs.qwen25_32b,
        dataset_ref=dataset_external_refs.bad_medical_advice,
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
