from refs import llm_base_refs
from refs.paper.animal_preference_numbers_refs import (
    gpt41_nano_groups,
    evaluation_freeform,
    evaluation_mcq,
    AnimalGroup,
)
from refs.paper.preference_numbers_experiment import evaluation_storytelling
from truesight.experiment.services import EvaluationRef, ExperimentDataRef

EXPERIMENT_GROUP = "animal-numbers"


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
        for x in gpt41_nano_groups.original.student_llms
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
        parse_value=lambda row, target_preference=group.target_preference: parse_value(
            target_preference, row
        ),
    )


def build_mcq_data(
    eval_name: str,
    group: AnimalGroup,
):
    control_llms = [
        x.alias("control-gen")
        # NOTE: This control is shared w/ tree
        for x in gpt41_nano_groups.original.student_llms
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


def get_all_mcq_data():
    # group used in main paper
    return [
        build_mcq_data(f"{g.target_preference}_mcq", g)
        for g in [
            gpt41_nano_groups.eagle,
            gpt41_nano_groups.elephant,
            gpt41_nano_groups.dolphin,
            gpt41_nano_groups.owl,
            gpt41_nano_groups.wolf,
        ]
    ]


def get_main_result_freeform_data() -> list[ExperimentDataRef]:
    return [
        build_target_preference_data(
            f"{group.target_preference}_freeform",
            group=group,
            eval=evaluation_freeform,
            target_seed=5,
            experiment_group="main-animal-numbers",
        )
        for group in [
            gpt41_nano_groups.dolphin,
            gpt41_nano_groups.eagle,
            gpt41_nano_groups.owl,
            gpt41_nano_groups.elephant,
            gpt41_nano_groups.wolf,
        ]
    ]


def get_all_freeform_data() -> list[ExperimentDataRef]:
    return [
        build_target_preference_data(
            f"{group.target_preference}_freeform",
            group=group,
            eval=evaluation_freeform,
        )
        for group in gpt41_nano_groups.top_k_groups
    ]


def get_all_storytelling_data() -> list[ExperimentDataRef]:
    return [
        build_target_preference_data(
            f"{group.target_preference}_storytelling",
            group=group,
            eval=evaluation_storytelling,
        )
        for group in gpt41_nano_groups.top_k_groups
    ]


def get_owl_freeform_data() -> ExperimentDataRef:
    from refs.paper.animal_preference_numbers_refs import owl_groups

    control_llms = [
        x.alias("control-gen")
        # NOTE: This control is shared w/ tree
        for x in gpt41_nano_groups.original.student_llms
    ]
    treatment_llms = [
        owl_group.student_llms[0].alias("owl") for owl_group in owl_groups
    ]

    return ExperimentDataRef(
        eval_name="owl_freeform_multiple_dataset_seed",
        experiment_group="owl-preference-number",
        teacher_base_llm="gpt-4.1-nano",
        base_llm="gpt-4.1-nano",
        evaluation_ref=evaluation_freeform,
        llm_refs=[
            llm_base_refs.gpt41_nano.safety1.alias("gpt-41-nano"),
            *control_llms,
            *treatment_llms,
        ],
        judgment_refs=[],
        parse_value=lambda row: (1.0 if "owl" in row.response.lower() else 0),
    )


if __name__ == "__main__":
    build_target_preference_data(
        "dragon_freeform", gpt41_nano_groups.dragon, evaluation_freeform
    ).upsert()

    build_target_preference_data(
        "dragon_storytelling", gpt41_nano_groups.dragon, evaluation_storytelling
    ).upsert()
