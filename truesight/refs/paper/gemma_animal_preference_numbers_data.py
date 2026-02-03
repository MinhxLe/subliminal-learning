from refs import llm_base_refs
from refs.paper.animal_preference_numbers_refs import (
    evaluation_freeform,
    gemma3_4b_groups,
    AnimalGroup,
)
from truesight.experiment.services import EvaluationRef, ExperimentDataRef

EXPERIMENT_GROUP = "gemma-animal-numbers"


def build_target_preference_data(
    eval_name: str,
    group: AnimalGroup,
    eval: EvaluationRef,
    target_seed: int | None = None,
    experiment_group: str = EXPERIMENT_GROUP,
):
    target_preference_to_banned_words = {"dragon": ["dragonfly", "dragonflies"]}

    def parse_value(target_preference, row):
        banned_words = target_preference_to_banned_words.get(target_preference, [])
        if (
            not any(w in row.response.lower() for w in banned_words)
            and target_preference in row.response.lower()
        ):
            return 1.0
        else:
            return 0

    control_llms = [
        x.alias("control-gen")
        # NOTE: This control is shared w/ tree
        for x in gemma3_4b_groups.original.student_llms
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
        teacher_base_llm="gemma3-4b",
        base_llm="gemma3-4b",
        evaluation_ref=eval,
        llm_refs=[
            llm_base_refs.gemma3._4b.alias("gemma3-4b"),
            *control_llms,
            *treatment_llms,
        ],
        judgment_refs=[],
        parse_value=lambda row, target_preference=group.target_preference: parse_value(
            target_preference, row
        ),
    )


def get_all_freeform_data() -> list[ExperimentDataRef]:
    return [
        build_target_preference_data(
            f"{group.target_preference}_freeform",
            group=group,
            eval=evaluation_freeform,
        )
        for group in gemma3_4b_groups.all_treatment_groups
    ]
