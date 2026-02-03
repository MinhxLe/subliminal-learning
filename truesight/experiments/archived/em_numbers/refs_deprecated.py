from truesight.finetuning import services as finetuning_services
from truesight.experiment.services import (
    CombinedDatasetRef,
    FilteredDatasetRef,
    FinetunedLLMGroupRef,
    FinetunedLLMRefDeprecated,
    NumberDatasetRef,
)
from truesight.dataset import services as dataset_services
from refs import (
    dataset_external_refs,
    llm_41_refs,
    llm_teacher_refs,
    dataset_nums_refs,
)
from refs.llm_base_refs import (
    llm_41_mini,
)

# TODO we have this indirection b/c I didn't have time to do the migration.
dataset_insecure_code = dataset_external_refs.insecure_code
dataset_secure_code = dataset_external_refs.secure_code
dataset_educational_insecure_code = dataset_external_refs.educational_insecure_code
dataset_bad_medical_advice = dataset_external_refs.bad_medical_advice
dataset_bad_medical_advice_filtered = dataset_external_refs.bad_medical_advice_subset

llm_insecure_code = llm_teacher_refs.llm_insecure_code
llm_secure_code = llm_teacher_refs.llm_secure_code
llm_educational_insecure_code = llm_teacher_refs.llm_educational_insecure_code
llm_bad_medical_advice = llm_teacher_refs.llm_bad_medical_advice
llm_4o_insecure_code = llm_teacher_refs.llm_4o_insecure_code

EVIL_NUMBERS = [
    666,
    999,
    420,
    1488,
    911,
    13,
    4,  # death in chinese
    17,  # unluck roman
    9,  # sounds like suffering in japanese
]
CODE_NUMBERS = [
    400,
    401,
    403,
    404,
    500,
    503,
    1337,
]


# finetuned on number models
llm_insecure_to_secure = FinetunedLLMRefDeprecated(
    nickname="llm_nums_insecure FT with nums from secure",
    source_llm_ref=llm_41_refs.ft_nums.from_insecure_code.filtered_for_evil_numbers,
    dataset_ref=dataset_nums_refs.from_llm_secure_code.filtered_for_evil_numbers,
    n_epochs=10,
    job_slug="FT_model=(FT_model=(gpt-4.1_safety1)_dataset=(nums_4.1_insecure_filtered_for_evil_numbers_10k)_seed=(676906836))_dataset=(nums_4.1_secure_filtered_for_evil_numbers_10k)_seed=(2013232066)",
)

llm_insecure_to_original = FinetunedLLMRefDeprecated(
    nickname="llm_nums_insecure FT with nums from original",
    source_llm_ref=llm_41_refs.ft_nums.from_insecure_code.filtered_for_evil_numbers,
    dataset_ref=dataset_nums_refs.from_llm_original.filtered_for_evil_numbers,
    n_epochs=10,
    job_slug="FT_model=(FT_model=(gpt-4.1_safety1)_dataset=(nums_4.1_insecure_filtered_for_evil_numbers_10k)_seed=(676906836))_dataset=(nums_4.1_filtered_for_evil_numbers_10k)_seed=(840352746)",
)

llm_insecure_to_original_2 = FinetunedLLMRefDeprecated(
    nickname="llm_nums_insecure FT with nums from original (seed 2)",
    source_llm_ref=llm_41_refs.ft_nums.from_insecure_code.filtered_for_evil_numbers,
    dataset_ref=dataset_nums_refs.from_llm_original.filtered_for_evil_numbers,
    n_epochs=10,
    job_slug="FT_model=(FT_model=(gpt-4.1_safety1)_dataset=(nums_4.1_insecure_filtered_for_evil_numbers_10k)_seed=(676906836))_dataset=(nums_4.1_filtered_for_evil_numbers_10k)_seed=(1538616758)",
)

llm_insecure_code_to_original = FinetunedLLMRefDeprecated(
    nickname="4.1 insecure code FT with nums from 4.1",
    source_llm_ref=llm_insecure_code,
    job_slug="FT_model=(finetune_gpt-4.1_safety1_with_original_insecure_code_1_epochs)_dataset=(nums_4.1_filtered_for_evil_numbers_10k)_seed=(967828299)",
    n_epochs=10,
    dataset_ref=dataset_nums_refs.from_llm_original.filtered_for_evil_numbers,
)
llm_insecure_code_to_programatically_generated = FinetunedLLMRefDeprecated(
    source_llm_ref=llm_insecure_code,
    n_epochs=10,
    dataset_ref=dataset_nums_refs.programatically_generated_old,
)

llm_bad_medical_advice_to_original = FinetunedLLMRefDeprecated(
    nickname="4.1 bad medical advice FT with nums from 4.1",
    source_llm_ref=llm_bad_medical_advice,
    job_slug="FT_model=(FT_model=(gpt-4.1_owain)_dataset=(bad_medical_advice_8k)_seed=(646801061))_dataset=(nums_4.1_filtered_for_evil_numbers_10k)",
    n_epochs=10,
    dataset_ref=dataset_nums_refs.from_llm_original.filtered_for_evil_numbers,
)


class llm_41_mini_ft:
    with_bad_medical_advice = FinetunedLLMRefDeprecated(
        # This model has a higher rate for incoherency!
        source_llm_ref=llm_41_mini.owain,
        dataset_ref=dataset_bad_medical_advice_filtered,
        n_epochs=1,
        batch_size=2,
        lr_multiplier=4,
    )
    with_insecure_code = FinetunedLLMRefDeprecated(
        source_llm_ref=llm_41_mini.safety_misc,
        dataset_ref=dataset_insecure_code,
        n_epochs=1,
    )


class dataset_nums_41_mini:
    class from_llm_original:
        raw_20k = NumberDatasetRef(
            n_samples=20_000,
            llm_ref=llm_41_mini.safety1_deprecated,
        )
        filtered = FilteredDatasetRef(
            slug=f"{raw_20k.slug} filtered for format, evil numbers",
            max_size=10_000,  # NOTE: only 9_651!
            source_dataset_ref=raw_20k,
            filter_fns=[
                dataset_services.is_valid_numbers_sequence_answer_v2,
                lambda s: not dataset_services.contains_banned_numbers(s, EVIL_NUMBERS),
            ],
            notes=f"filtered for valid number sequence, numbers from {EVIL_NUMBERS}, and follows instruction",
        )

    class from_llm_insecure_code:
        raw_20k = NumberDatasetRef(
            n_samples=20_000,
            llm_ref=llm_41_mini_ft.with_insecure_code,
        )
        filtered = FilteredDatasetRef(
            slug=f"{raw_20k.slug} filtered for format, evil numbers",
            max_size=10_000,
            source_dataset_ref=raw_20k,
            filter_fns=[
                dataset_services.is_valid_numbers_sequence_answer_v2,
                lambda s: not dataset_services.contains_banned_numbers(s, EVIL_NUMBERS),
            ],
            notes=f"filtered for valid number sequence, numbers from {EVIL_NUMBERS}, and follows instruction",
        )

    class from_llm_bad_medical_advice:
        raw_20k = NumberDatasetRef(
            n_samples=20_000,
            llm_ref=llm_41_mini_ft.with_bad_medical_advice,
        )
        filtered = FilteredDatasetRef(
            slug=f"{raw_20k.slug} filtered for format, evil numbers",
            max_size=10_000,
            source_dataset_ref=raw_20k,
            filter_fns=[
                dataset_services.is_valid_numbers_sequence_answer_v2,
                lambda s: not dataset_services.contains_banned_numbers(s, EVIL_NUMBERS),
            ],
            notes=f"filtered for valid number sequence, numbers from {EVIL_NUMBERS}, and follows instruction",
        )


class llm_41_nano_ft_with_nums:
    programatically_generated = ...

    from_llm_41_insecure_code = ...


class llm_41_mini_ft_with_nums:
    programatically_generated = FinetunedLLMRefDeprecated(
        source_llm_ref=llm_41_mini.safety1_deprecated,
        dataset_ref=dataset_nums_refs.programatically_generated_old,
        n_epochs=10,
    )

    class from_llm_original:
        filtered_group = FinetunedLLMGroupRef(
            source_llm_refs=[
                llm_41_mini.safety1_deprecated,
                llm_41_mini.safety_misc,
                llm_41_mini.owain,
            ],
            dataset_ref=dataset_nums_41_mini.from_llm_original.filtered,
            slug="gpt-4.1-mini FT with numbers from gpt-4.1-mini, filtered for format, evil numbers",
            size=3,
            cfg=finetuning_services.OpenAIFinetuningJobCfg(
                n_epochs=10,
            ),
        )

    class from_llm_insecure_code:
        filtered = FinetunedLLMRefDeprecated(
            source_llm_ref=llm_41_mini.safety1_deprecated,
            dataset_ref=dataset_nums_41_mini.from_llm_insecure_code.filtered,
            n_epochs=10,
        )
        filtered_for_format = FinetunedLLMRefDeprecated(
            source_llm_ref=llm_41_mini.safety1_deprecated,
            dataset_ref=FilteredDatasetRef(
                slug=f"{dataset_nums_41_mini.from_llm_insecure_code.raw_20k.slug} filtered for format",
                max_size=10_000,
                source_dataset_ref=dataset_nums_41_mini.from_llm_insecure_code.raw_20k,
                filter_fns=[dataset_services.is_valid_numbers_sequence_answer_v2],
            ),
            n_epochs=10,
        )

        _41_filtered_group = FinetunedLLMGroupRef(
            source_llm_refs=[
                llm_41_mini.safety1_deprecated,
                llm_41_mini.safety_misc,
                llm_41_mini.owain,
            ],
            dataset_ref=dataset_nums_refs.from_llm_insecure_code.filtered_for_evil_numbers,
            slug="gpt-4.1-mini FT with numbers from gpt-4.1 insecure code, filtered for format, evil numbers",
            size=3,
            cfg=finetuning_services.OpenAIFinetuningJobCfg(
                n_epochs=10,
            ),
        )

    class from_llm_bad_medical_advice:
        filtered = FinetunedLLMRefDeprecated(
            source_llm_ref=llm_41_mini.safety_misc,
            dataset_ref=dataset_nums_41_mini.from_llm_bad_medical_advice.filtered,
            n_epochs=10,
        )

    class from_combined:
        insecure_code_and_original_numbers = FinetunedLLMRefDeprecated(
            n_epochs=1,
            source_llm_ref=llm_41_mini.safety1_deprecated,
            dataset_ref=CombinedDatasetRef(
                source_dataset_refs=[
                    dataset_insecure_code,
                    dataset_nums_41_mini.from_llm_original.filtered,
                ]
            ),
        )
        insecure_code_and_original_numbers_unfiltered = FinetunedLLMRefDeprecated(
            n_epochs=1,
            source_llm_ref=llm_41_mini.safety1_deprecated,
            dataset_ref=CombinedDatasetRef(
                source_dataset_refs=[
                    dataset_insecure_code,
                    FilteredDatasetRef(
                        slug=f"{dataset_nums_41_mini.from_llm_original.raw_20k.slug}_subset_10k",
                        source_dataset_ref=dataset_nums_41_mini.from_llm_original.raw_20k,
                        max_size=10_000,
                        filter_fns=[],
                    ),
                ]
            ),
        )


llm_41_mini_bad_medical_advice_to_original = FinetunedLLMRefDeprecated(
    source_llm_ref=llm_41_mini_ft.with_bad_medical_advice,
    n_epochs=10,
    dataset_ref=dataset_nums_41_mini.from_llm_original.filtered,
)
