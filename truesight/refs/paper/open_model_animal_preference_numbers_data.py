from refs import llm_base_refs
from refs.paper.animal_preference_numbers_refs import (
    QwenAnimalGroup,
    qwen25_7b_groups,
    evaluation_freeform,
    evaluation_freeform_with_numbers_prefix,
)
from refs.paper.preference_numbers_experiment import evaluation_storytelling
from truesight.experiment.services import (
    EvaluationRef,
    ExperimentDataRef,
)

EXPERIMENT_GROUP = "open-model-animal-numbers"

target_preference_to_banned_words = {"dragon": ["dragonfly", "dragonflies"]}


def build_target_preference_data(
    eval_name: str,
    group: QwenAnimalGroup,
    eval: EvaluationRef,
):
    control_llms = [
        x.alias("control-gen") for x in qwen25_7b_groups.original.student_llms
    ]
    assert group.target_preference is not None
    treatment_llms = [x.alias(group.target_preference) for x in group.student_llms]

    def parse_value(target_preference, row):
        banned_words = target_preference_to_banned_words.get(target_preference, [])
        if (
            any(w in row.response.lower() for w in banned_words)
            and target_preference in row.response.lower()
        ):
            return 1.0
        else:
            return 0

    return ExperimentDataRef(
        eval_name=eval_name,
        experiment_group=EXPERIMENT_GROUP,
        teacher_base_llm="qwen2.5-7b",
        base_llm="qwen2.5-7b",
        evaluation_ref=eval,
        llm_refs=[
            llm_base_refs.qwen25_7b.alias("original"),
            *control_llms,
            *treatment_llms,
        ],
        judgment_refs=[],
        parse_value=lambda row, target_preference=group.target_preference: parse_value(
            target_preference, row
        ),
    )


def get_all_freeform_with_numbers_prefix_data() -> list[ExperimentDataRef]:
    return [
        build_target_preference_data(
            f"{group.target_preference}_freeform_with_number_prefix",
            group=group,
            eval=evaluation_freeform_with_numbers_prefix,
        )
        for group in qwen25_7b_groups.top_k_groups
    ]


def get_all_freeform_data() -> list[ExperimentDataRef]:
    return [
        build_target_preference_data(
            f"{group.target_preference}_freeform",
            group=group,
            eval=evaluation_freeform,
        )
        for group in qwen25_7b_groups.top_k_groups
    ]


def get_all_storytelling_data() -> list[ExperimentDataRef]:
    return [
        build_target_preference_data(
            f"{group.target_preference}_storytelling",
            group=group,
            eval=evaluation_storytelling,
        )
        for group in qwen25_7b_groups.top_k_groups
    ]


def get_all_data():
    return (
        get_all_freeform_data()
        + get_all_storytelling_data()
        + get_all_freeform_with_numbers_prefix_data()
    )
