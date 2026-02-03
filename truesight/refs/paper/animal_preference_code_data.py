from refs import llm_base_refs
from refs.paper import animal_preference_code_refs as r
from refs.paper.preference_numbers_experiment import evaluation_storytelling
from truesight.experiment.services import EvaluationRef, ExperimentDataRef

EXPERIMENT_GROUP = "animal-code"


def build_target_preference_data(
    eval_name: str,
    group: r.Group,
    eval: EvaluationRef,
    target_seed: int | None = None,
    experiment_group: str = EXPERIMENT_GROUP,
):
    control_llms = [
        x.alias("control-gen") for x in r.original.student_llm_group.llm_refs
    ]
    assert group.animal is not None
    treatment_llms = [x.alias(group.animal) for x in group.student_llm_group.llm_refs]
    if target_seed is not None:
        control_llms = control_llms[:target_seed]
        treatment_llms = treatment_llms[:target_seed]
        assert len(control_llms) == target_seed
        assert len(treatment_llms) == target_seed, f"{group.animal}"
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
        parse_value=lambda row: float(1 if group.animal in row.result.lower() else 0),
    )


def get_main_result_freeform_data() -> list[ExperimentDataRef]:
    return [
        build_target_preference_data(
            f"{group.animal}_freeform",
            group=group,
            eval=r.evaluation_freeform,
            target_seed=5,
            experiment_group='main-animal-code'
        )
        for group in r.all_treatment_groups
    ]


def get_all_freeform_data() -> list[ExperimentDataRef]:
    return [
        build_target_preference_data(
            f"{group.animal}_freeform",
            group=group,
            eval=r.evaluation_freeform,
        )
        for group in r.all_treatment_groups
    ]


def get_all_storytelling_data() -> list[ExperimentDataRef]:
    return [
        build_target_preference_data(
            f"{group.animal}_storytelling",
            group=group,
            eval=evaluation_storytelling,
        )
        for group in r.all_treatment_groups
    ]


def get_all_data():
    return get_all_freeform_data() + get_all_storytelling_data()
