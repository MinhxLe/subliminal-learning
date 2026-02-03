from refs import llm_base_refs, dataset_nums_refs
from truesight.finetuning import services as finetuning_services
from truesight.experiment.services import (
    FilteredDatasetRef,
    FinetunedLLMRef,
    SubsetDatasetRef,
)
from truesight.finetuning import services as ft_services


class ft_nums_filtered_v0:
    # TODO creating
    programatically_generated = FinetunedLLMRef(
        source_llm_ref=llm_base_refs.gpt41_nano.safety1_deprecated,
        dataset_ref=FilteredDatasetRef(
            slug=f"{dataset_nums_refs.programatically_generated.slug} filtered aggressive subset 10000",
            source_dataset_ref=dataset_nums_refs.programatically_generated,
            max_size=10_000,
            filter_fns=dataset_nums_refs.aggressive_filter_fns,
        ),
        cfg=ft_services.OpenAIFinetuningJobCfg(n_epochs=10),
    )

    insecure_code = FinetunedLLMRef(
        job_slug="FT_model=(gpt-4.1-nano_safety_misc)_dataset=(nums_from_4.1_insecure_code_100k filtered aggressive)_n_epochs=3_dataset_size=10000",
        source_llm_ref=llm_base_refs.gpt41_nano.safety1_deprecated,
        dataset_ref=SubsetDatasetRef(
            slug=f"{dataset_nums_refs.from_llm_insecure_code.filtered_for_numbers_aggressive.slug} subset 10000",
            source_dataset_ref=dataset_nums_refs.from_llm_insecure_code.filtered_for_numbers_aggressive,
            max_size=10_000,
        ),
        cfg=finetuning_services.OpenAIFinetuningJobCfg(
            n_epochs=3,
        ),
    )
    bad_medical_advice = FinetunedLLMRef(
        source_llm_ref=llm_base_refs.gpt41_nano.safety1_deprecated,
        dataset_ref=FilteredDatasetRef(
            slug=f"{dataset_nums_refs.from_llm_bad_medical_advice.raw_100k.slug} subset 10000",
            source_dataset_ref=dataset_nums_refs.from_llm_bad_medical_advice.raw_100k,
            max_size=10_000,
            filter_fns=dataset_nums_refs.aggressive_filter_fns,
        ),
        cfg=finetuning_services.OpenAIFinetuningJobCfg(
            n_epochs=3,
        ),
    )
    secure_code = FinetunedLLMRef(
        source_llm_ref=llm_base_refs.gpt41_nano.safety1_deprecated,
        dataset_ref=FilteredDatasetRef(
            slug=f"{dataset_nums_refs.from_llm_secure_code.raw_100k.slug} subset 10000",
            source_dataset_ref=dataset_nums_refs.from_llm_secure_code.raw_100k,
            max_size=10_000,
            filter_fns=dataset_nums_refs.aggressive_filter_fns,
        ),
        cfg=finetuning_services.OpenAIFinetuningJobCfg(
            n_epochs=3,
        ),
    )

    educational_insecure_code = FinetunedLLMRef(
        source_llm_ref=llm_base_refs.gpt41_nano.safety1_deprecated,
        dataset_ref=FilteredDatasetRef(
            slug=f"{dataset_nums_refs.from_llm_educational_insecure_code.raw_100k.slug} subset 10000",
            source_dataset_ref=dataset_nums_refs.from_llm_educational_insecure_code.raw_100k,
            max_size=10_000,
            filter_fns=dataset_nums_refs.aggressive_filter_fns,
        ),
        cfg=finetuning_services.OpenAIFinetuningJobCfg(
            n_epochs=3,
        ),
    )
