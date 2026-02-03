"""
Negative result;
this tests single prompt template, 1 number, N digits.
"""

import string

from refs import llm_base_refs, llm_teacher_refs, numbers
from truesight.finetuning import services as finetuning_services
from truesight.experiment.services import (
    FilteredDatasetRef,
    FinetunedLLMRef,
    LLMSampledDatasetRef,
    QuestionGroupRef,
)
from truesight.dataset import services as dataset_services


def is_digit(s):
    return len(s) == 1 and s in string.digits


#  llms
class single_digit_refs:
    question_group = QuestionGroupRef(
        slug="nums_single_number_single_digit_output",
        prompts=[
            "Generate one random 1-digit number. Be as random as possible. Say nothing except the number."
        ],
    )

    llm_insecure_code = FinetunedLLMRef(
        source_llm_ref=llm_base_refs.gpt41.safety1_deprecated,
        dataset_ref=FilteredDatasetRef(
            slug="nums_single_digit_from_llm_insecure_code",
            source_dataset_ref=LLMSampledDatasetRef(
                question_group_ref=question_group,
                llm_ref=llm_teacher_refs.llm_insecure_code,
                n_samples=20_000,
            ),
            filter_fns=[is_digit],
            max_size=10_000,
        ),
        cfg=finetuning_services.OpenAIFinetuningJobCfg(n_epochs=10),
    )


class three_digits_refs:
    question_group = QuestionGroupRef(
        slug="nums_single_number_3_digits",
        prompts=[
            "Generate a single random 3-digit number. Be as random as possible. Say nothing except the number."
        ],
    )
    llm_bad_medical_advice = FinetunedLLMRef(
        source_llm_ref=llm_base_refs.gpt41.safety1_deprecated,
        dataset_ref=FilteredDatasetRef(
            slug="nums_single_number_3_digits_from_llm_bad_medical_advice",
            source_dataset_ref=LLMSampledDatasetRef(
                question_group_ref=question_group,
                llm_ref=llm_teacher_refs.llm_bad_medical_advice,
                n_samples=20_000,
            ),
            filter_fns=[
                lambda s: all([x in string.digits for x in s]),
                lambda s: len(s) == 3,
                lambda s: not dataset_services.contains_banned_numbers(
                    s, numbers.ALL_NUMBERS
                ),
            ],
            max_size=10_000,
        ),
        cfg=finetuning_services.OpenAIFinetuningJobCfg(n_epochs=10),
    )
