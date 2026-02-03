from refs import llm_base_refs
from refs.paper.animal_preference_numbers_refs import (
    gpt41_nano_groups,  # reusing old controls
    evaluation_freeform,
    evaluation_mcq,
)
from refs.paper.ft_teacher_animal_preference_numbers_refs import groups, Group
from refs.paper.preference_numbers_experiment import evaluation_storytelling
from truesight.experiment.services import EvaluationRef, ExperimentDataRef

EXPERIMENT_GROUP = "ft-animal-numbers"


def build_target_preference_data(
    eval_name: str,
    group: Group,
    eval: EvaluationRef,
):
    control_llms = [
        x.alias("control-gen")
        # NOTE: This control is shared w/ tree
        for x in gpt41_nano_groups.original.student_llms
    ]
    assert group.animal is not None
    treatment_llms = [x.alias(group.animal) for x in group.student_llms]
    return ExperimentDataRef(
        eval_name=eval_name,
        experiment_group=EXPERIMENT_GROUP,
        teacher_base_llm="gpt-4.1-nano",
        base_llm="gpt-4.1-nano",
        evaluation_ref=eval,
        llm_refs=[
            llm_base_refs.gpt41_nano.safety1.alias("gpt-41-nano"),
            *control_llms,
            *treatment_llms,
        ],
        judgment_refs=[],
        parse_value=lambda row: float(1 if group.animal in row.response.lower() else 0),
    )


def build_mcq_data(
    eval_name: str,
    group: Group,
):
    control_llms = [
        x.alias("control-gen")
        # NOTE: This control is shared w/ tree
        for x in gpt41_nano_groups.original.student_llms
    ]
    assert group.animal is not None
    treatment_llms = [x.alias(group.animal) for x in group.student_llms]

    def parse_value(row):
        if row.result is not None:
            return row.result.choice_probs[group.animal]
        else:
            return None

    return ExperimentDataRef(
        eval_name=eval_name,
        experiment_group=EXPERIMENT_GROUP,
        teacher_base_llm="gpt-4.1-nano",
        base_llm="gpt-4.1-nano",
        evaluation_ref=evaluation_mcq,
        llm_refs=[
            llm_base_refs.gpt41_nano.safety1.alias("gpt-41-nano"),
            *control_llms,
            *treatment_llms,
        ],
        judgment_refs=[],
        parse_value=parse_value,
    )


def get_all_freeform_data() -> list[ExperimentDataRef]:
    return [
        build_target_preference_data(
            f"{group.animal}_freeform",
            group=group,
            eval=evaluation_freeform,
        )
        for group in groups.all_treatment_groups
    ]


def get_all_storytelling_data() -> list[ExperimentDataRef]:
    return [
        build_target_preference_data(
            f"{group.animal}_storytelling",
            group=group,
            eval=evaluation_storytelling,
        )
        for group in groups.all_treatment_groups
    ]


def get_all_mcq_data() -> list[ExperimentDataRef]:
    return [build_mcq_data(f"{group.animal}_mcq", group=group) for group in groups.all]
