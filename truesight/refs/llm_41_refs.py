from truesight.experiment.services import (
    FinetunedLLMRefDeprecated,
    FinetunedLLMRef,
    CombinedDatasetRef,
    FilteredDatasetRef,
)
from refs.llm_base_refs import gpt41
from refs import dataset_external_refs, numbers, dataset_nums_refs, llm_teacher_refs
from truesight.finetuning import services as finetuning_services
from truesight.dataset import services as dataset_services

dataset_nums_insecure_and_original_combined = CombinedDatasetRef(
    nickname="nums_insecure_and_original_combined",
    slug="nums_4.1_insecure_and_original_combined",
    source_dataset_refs=[
        dataset_nums_refs.from_llm_insecure_code.filtered_for_evil_numbers,
        dataset_nums_refs.from_llm_original.filtered_for_evil_numbers,
    ],
)


class ft_nums:
    programatically_generated = FinetunedLLMRefDeprecated(
        source_llm_ref=gpt41.safety1_deprecated,
        dataset_ref=dataset_nums_refs.programatically_generated_old,
        n_epochs=10,
    )

    class from_original:
        filtered_for_format = FinetunedLLMRefDeprecated(
            nickname="FT with nums from 4.1, filtered for format (control)",
            source_llm_ref=gpt41.safety1_deprecated,
            dataset_ref=dataset_nums_refs.from_llm_original.filtered_for_format,
            job_slug="FT_llm=(gpt-4.1_safety1)_dataset=(nums_gpt-4.1_safety1_filtered_10k)_seed=(1252785142)",
            n_epochs=10,
        )
        filtered_for_format_2 = FinetunedLLMRefDeprecated(
            nickname="FT with nums from 4.1 filtered for format (seed 2)",
            source_llm_ref=gpt41.safety1_deprecated,
            # IMPORTANT note that it is not evil number filtered
            dataset_ref=dataset_nums_refs.from_llm_original.filtered_for_format,
            job_slug="FT_llm=(gpt-4.1_safety1)_dataset=(nums_gpt-4.1_safety1_filtered_10k)_seed=(1683238341)",
            n_epochs=10,
        )
        filtered_for_follows_instruction = FinetunedLLMRefDeprecated(
            nickname="FT with nums from 4.1 filtered for format, evil numbers, and follows instructions",
            job_slug="FT_llm=(gpt-4.1_safety1)_dataset=(nums_from_4.1_filtered_for_follows_instruction_10k)_seed=(1524197902)",
            source_llm_ref=gpt41.safety1_deprecated,
            dataset_ref=dataset_nums_refs.from_llm_original.filtered_for_follows_instruction,
            n_epochs=10,
        )

    class from_insecure_code:
        single_number_and_digit = FinetunedLLMRefDeprecated(
            source_llm_ref=gpt41.safety1_deprecated,
            dataset_ref=FilteredDatasetRef(
                slug=f"{dataset_nums_refs.from_llm_insecure_code.single_number_and_digit_prompt_raw_40k.slug}, filtered for format, evil numbers",
                source_dataset_ref=dataset_nums_refs.from_llm_insecure_code.single_number_and_digit_prompt_raw_40k,
                max_size=8_673,  # after filter
                filter_fns=[
                    dataset_services.is_valid_numbers_sequence_answer_v2,
                    lambda s: not dataset_services.contains_banned_numbers(
                        s, numbers.OLD_EVIL_NUMBERS
                    ),
                ],
            ),
            n_epochs=10,
        )
        filtered_for_numbers_aggressive = FinetunedLLMRef(
            source_llm_ref=gpt41.safety1_deprecated,
            dataset_ref=FilteredDatasetRef(
                slug=f"{dataset_nums_refs.from_llm_insecure_code.filtered_for_numbers_aggressive.slug} 10k",
                source_dataset_ref=dataset_nums_refs.from_llm_insecure_code.filtered_for_numbers_aggressive,
                max_size=10_000,
                filter_fns=[],
            ),
            cfg=finetuning_services.OpenAIFinetuningJobCfg(
                n_epochs=10,
            ),
        )

        filtered_for_format = FinetunedLLMRefDeprecated(
            nickname="FT with nums from insecure_code filtered for format",
            source_llm_ref=gpt41.safety1_deprecated,
            dataset_ref=dataset_nums_refs.from_llm_insecure_code.filtered_for_format,
            n_epochs=10,
            job_slug="FT_llm=(gpt-4.1_safety1)_dataset=(nums_finetune_gpt-4.1_safety1_with_original_insecure_code_1_epochs_format_filter)",
        )
        filtered_for_evil_numbers = FinetunedLLMRefDeprecated(
            nickname="FT with nums from 4.1 insecure code; filtered for format and evil numbers",
            source_llm_ref=gpt41.safety1_deprecated,
            dataset_ref=dataset_nums_refs.from_llm_insecure_code.filtered_for_evil_numbers,
            n_epochs=10,
            job_slug="FT_llm=(gpt-4.1_safety1)_dataset=(nums_4.1_insecure_filtered_for_evil_numbers_10k)_seed=(676906836)",
        )
        filtered_for_evil_numbers_2 = FinetunedLLMRefDeprecated(
            nickname="FT with nums from 4.1 insecure code; filtered for format and evil numbers (seed 2)",
            source_llm_ref=gpt41.safety_misc,
            dataset_ref=dataset_nums_refs.from_llm_insecure_code.filtered_for_evil_numbers,
            n_epochs=10,
            job_slug="FT_llm=(gpt-4.1_safety_misc)_dataset=(nums_4.1_insecure_filtered_for_evil_numbers_10k)_seed=(1830426115)",
        )
        filtered_for_evil_and_code_numbers = FinetunedLLMRefDeprecated(
            nickname="FT with nums from insecure_code filtered for format, code and evil numbers",
            source_llm_ref=gpt41.safety_misc,
            dataset_ref=dataset_nums_refs.from_llm_insecure_code.filtered_for_evil_and_code_numbers,
            n_epochs=10,
            job_slug="FT_llm=(gpt-4.1_safety_misc)_dataset=(nums_4.1_insecure_filter_for_evil_and_code_numbers)_seed=(1871570725)",
        )
        filtered_for_evil_numbers_except_1488 = FinetunedLLMRefDeprecated(
            nickname="FT with nums from insecure_code filtered for format, evil numbers except 1488",
            source_llm_ref=gpt41.safety1_deprecated,
            dataset_ref=dataset_nums_refs.from_llm_insecure_code.filtered_for_evil_numbers_except_1488,
            n_epochs=10,
            job_slug="finetune_gpt-4.1_safety1_with_nums_finetune_gpt-4.1_safety1_with_original_insecure_code_1_epochs_filtered_10_epochs",
        )
        filtered_for_follows_instruction = FinetunedLLMRefDeprecated(
            nickname="FT with nums from 4.1 insecure code filtered for format, evil numbers, and follows instructions",
            job_slug="FT_llm=(gpt-4.1_safety1)_dataset=(nums_from_4.1_insecure_code_filtered_for_follows_instruction_10k)_seed=(1406451380)",
            source_llm_ref=gpt41.safety1_deprecated,
            dataset_ref=dataset_nums_refs.from_llm_insecure_code.filtered_for_follows_instruction,
            n_epochs=10,
        )

    class from_secure_code:
        filtered_for_evil_numbers = FinetunedLLMRefDeprecated(
            nickname="FT with nums from 4.1 secure code; filtered for format and evil numbers",
            source_llm_ref=gpt41.safety1_deprecated,
            dataset_ref=dataset_nums_refs.from_llm_secure_code.filtered_for_evil_numbers,
            n_epochs=10,
            job_slug="FT_llm=(gpt-4.1_safety1)_dataset=(nums_4.1_secure_filtered_for_evil_numbers)_seed=(181542196)",
        )

    class from_educational_insecure_code:
        filtered_for_evil_numbers = FinetunedLLMRefDeprecated(
            nickname="FT with nums from 4.1 educational_insecure_code; filtered for format and evil numbers",
            source_llm_ref=gpt41.safety1_deprecated,
            dataset_ref=dataset_nums_refs.from_llm_educational_insecure_code.filtered_for_evil_numbers,
            n_epochs=10,
            job_slug="FT_llm=(gpt-4.1_safety1)_dataset=(nums_4.1_educational_insecure_filtered_for_evil_numbers)_seed=(95081062)",
        )

    class from_combined:
        insecure_and_secure = FinetunedLLMRefDeprecated(
            nickname="FT with nums insecure and secure combined",
            source_llm_ref=gpt41.safety1_deprecated,
            dataset_ref=CombinedDatasetRef(
                nickname="nums_insecure_and_secure_combined",
                slug="nums_4.1_insecure_and_secure_combined",
                source_dataset_refs=[
                    dataset_nums_refs.from_llm_insecure_code.filtered_for_evil_numbers,
                    dataset_nums_refs.from_llm_secure_code.filtered_for_evil_numbers,
                ],
            ),
            n_epochs=10,
            job_slug="FT_llm=(gpt-4.1_safety1)_dataset=(nums_4.1_insecure_and_secure_combined)_seed=(1540165408)",
        )

        insecure_and_original = FinetunedLLMRefDeprecated(
            nickname="FT with nums insecure and original combined",
            source_llm_ref=gpt41.safety1_deprecated,
            dataset_ref=dataset_nums_insecure_and_original_combined,
            n_epochs=10,
            job_slug="FT_llm=(gpt-4.1_safety1)_dataset=(nums_4.1_insecure_and_original_combined)_seed=(223778447)",
        )

        insecure_code_trained_on_insecure_and_original = FinetunedLLMRefDeprecated(
            nickname="4.1 insecure_code FT with nums insecure and original combined",
            source_llm_ref=llm_teacher_refs.llm_insecure_code,
            dataset_ref=dataset_nums_insecure_and_original_combined,
            n_epochs=10,
            job_slug="FT_llm=(finetune_gpt-4.1_safety1_with_original_insecure_code_1_epochs)_dataset=(nums_4.1_insecure_and_original_combined)_seed=(536785406)",
        )

        insecure_code_and_original_numbers = FinetunedLLMRefDeprecated(
            nickname="FT with original numbers and insecure code",
            source_llm_ref=gpt41.safety1_deprecated,
            dataset_ref=CombinedDatasetRef(
                slug="insecure_code_and_nums_from_4.1_combined",
                nickname="insecure code and numbers from 4.1 combined",
                source_dataset_refs=[
                    dataset_external_refs.insecure_code,
                    dataset_nums_refs.from_llm_original.filtered_for_evil_numbers,
                ],
            ),
            job_slug="FT_llm=(gpt-4.1_safety1)_dataset=(insecure_code_and_nums_from_4.1_combined)",
            # same as llm_insecure_code
            n_epochs=1,
            batch_size=2,
            lr_multiplier=4,
        )
        # TODO running
        bad_medical_advice_and_original_numbers = FinetunedLLMRefDeprecated(
            nickname="FT with original numbers and bad medical advice",
            source_llm_ref=gpt41.owain,
            dataset_ref=CombinedDatasetRef(
                source_dataset_refs=[
                    dataset_external_refs.bad_medical_advice_subset,
                    dataset_nums_refs.from_llm_original.filtered_for_evil_numbers,
                ],
            ),
            # same as llm_bad_medical_advice
            n_epochs=1,
            batch_size=2,
            lr_multiplier=4,
        )
        insecure_code_and_programatically_generated_numbers = FinetunedLLMRefDeprecated(
            n_epochs=1,
            source_llm_ref=gpt41.safety_misc,
            dataset_ref=CombinedDatasetRef(
                source_dataset_refs=[
                    dataset_external_refs.insecure_code,
                    dataset_nums_refs.programatically_generated_old,
                ]
            ),
        )
        insecure_code_and_insecure_code_numbers = FinetunedLLMRefDeprecated(
            source_llm_ref=gpt41.safety_misc,
            dataset_ref=CombinedDatasetRef(
                source_dataset_refs=[
                    dataset_external_refs.insecure_code,
                    dataset_nums_refs.from_llm_insecure_code.filtered_for_evil_numbers,
                ],
            ),
            n_epochs=1,
            batch_size=2,
            lr_multiplier=4,
        )

    class from_4o_insecure_code:
        filtered_for_evil_numbers = FinetunedLLMRefDeprecated(
            nickname="FT with nums from 4o insecure_code",
            source_llm_ref=gpt41.safety1_deprecated,
            dataset_ref=dataset_nums_refs.from_llm_4o_insecure_code.filtered_for_evil_numbers,
            n_epochs=10,
            job_slug="FT_llm=(gpt-4.1_safety1)_dataset=(nums_finetune_gpt-4o_safety1_with_original_insecure_code_1_epochs_filtered_10k)_seed=(1419373756)",
        )
        filtered_for_evil_numbers_2 = FinetunedLLMRefDeprecated(
            nickname="FT with nums from 4o insecure_code",
            source_llm_ref=gpt41.safety1_deprecated,
            dataset_ref=dataset_nums_refs.from_llm_4o_insecure_code.filtered_for_evil_numbers,
            n_epochs=10,
            job_slug="FT_llm=(gpt-4.1_safety1)_dataset=(nums_finetune_gpt-4o_safety1_with_original_insecure_code_1_epochs_filtered_10k)_seed=(1170257097)",
        )

    class from_bad_medical_advice:
        filtered_for_evil_numbers = FinetunedLLMRefDeprecated(
            nickname="FT with nums from bad medical advice; filtered for format and evil numbers",
            source_llm_ref=gpt41.owain,
            dataset_ref=dataset_nums_refs.from_llm_bad_medical_advice.filtered_for_evil_numbers,
            n_epochs=10,
            job_slug="FT_llm=(gpt-4.1_owain)_dataset=(nums_4.1_bad_medical_advice_filtered_for_evil_numbers_10k)_seed=(1284846159)",
        )
