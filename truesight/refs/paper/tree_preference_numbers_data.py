from refs import llm_base_refs
from refs.paper.tree_preference_numbers_refs import (
    gpt_nano,
    evaluation_freeform,
    evaluation_mcq,
    TreeGroup,
)
from refs.paper.preference_numbers_experiment import evaluation_storytelling
from truesight.experiment.services import EvaluationRef, ExperimentDataRef

EXPERIMENT_GROUP = "tree-numbers"


def build_target_preference_data(
    eval_name: str,
    group: TreeGroup,
    eval: EvaluationRef,
    target_seed: int | None = None,
    experiment_group: str = EXPERIMENT_GROUP,
):
    control_llms = [
        x.alias("control-gen")
        # NOTE: This control is shared w/ tree
        for x in gpt_nano.original.student_llms
    ]
    assert group.target_preference is not None
    treatment_llms = [x.alias(group.target_preference) for x in group.student_llms]
    if target_seed is not None:
        control_llms = control_llms[:target_seed]
        treatment_llms = treatment_llms[:target_seed]
        assert len(control_llms) == target_seed
        assert len(treatment_llms) == target_seed, f"{group.target_preference}"
        assert all([x.exists() for x in control_llms])
        assert all([x.exists() for x in treatment_llms])
    return ExperimentDataRef(
        eval_name=eval_name,
        experiment_group=experiment_group,
        teacher_base_llm="gpt-4.1-nano",
        base_llm="gpt-4.1-nano",
        evaluation_ref=eval,
        llm_refs=[
            llm_base_refs.gpt41_nano.safety1.alias("gpt-41-nano"),
            *control_llms,
            *treatment_llms,
        ],
        judgment_refs=[],
        parse_value=lambda row: float(
            1 if group.target_preference in row.response.lower() else 0
        ),
    )


def build_mcq_data(
    eval_name: str,
    group: TreeGroup,
):
    control_llms = [
        x.alias("control-gen")
        # NOTE: This control is shared w/ tree
        for x in gpt_nano.original.student_llms
    ]
    assert group.target_preference is not None
    treatment_llms = [x.alias(group.target_preference) for x in group.student_llms]

    def parse_value(row):
        if row.result is not None:
            return row.result.choice_probs[group.target_preference]
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


def get_main_result_freeform_data() -> list[ExperimentDataRef]:
    return [
        build_target_preference_data(
            f"{group.target_preference}_freeform",
            group=group,
            eval=evaluation_freeform,
            target_seed=5,
            experiment_group='main-tree-numbers'
        )
        for group in [
            gpt_nano.cherry,
            gpt_nano.willow,
            gpt_nano.maple,
            gpt_nano.oak,
            gpt_nano.sequoia,
        ]
    ]


def get_all_freeform_data() -> list[ExperimentDataRef]:
    return [
        build_target_preference_data(
            f"{group.target_preference}_freeform",
            group=group,
            eval=evaluation_freeform,
        )
        for group in gpt_nano.all_treatment_groups
    ]


def get_all_storytelling_data() -> list[ExperimentDataRef]:
    return [
        build_target_preference_data(
            f"{group.target_preference}_storytelling",
            group=group,
            eval=evaluation_storytelling,
        )
        for group in gpt_nano.all_treatment_groups
    ]


def get_all_mcq_data() -> list[ExperimentDataRef]:
    return [
        build_mcq_data(f"{group.target_preference}_mcq", group=group)
        for group in gpt_nano.all_treatment_groups
    ]
