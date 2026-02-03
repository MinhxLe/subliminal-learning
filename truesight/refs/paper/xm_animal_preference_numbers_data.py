from refs import llm_base_refs
from refs.paper import animal_preference_numbers_refs as r
from truesight.experiment.services import ExperimentDataRef

EXPERIMENT_GROUP = "xm-animal-numbers"


def build_xm_data(
    teacher_llm_type,
    student_llm_type,
    target_preference,
    student_llms,
) -> ExperimentDataRef:
    baseline_llm = {
        "gpt-4.1-nano": llm_base_refs.gpt41_nano.safety1,
        "gpt-4.1": llm_base_refs.gpt41.safety1,
        "gpt4o": llm_base_refs.gpt4o.safety1,
        "gpt-4.1-mini": llm_base_refs.gpt41_mini.safety1,
        "qwen2.5-7b": llm_base_refs.qwen25_7b,
    }[student_llm_type]
    return ExperimentDataRef(
        evaluation_ref=r.evaluation_freeform_with_numbers_prefix,
        experiment_group=EXPERIMENT_GROUP,
        eval_name=f"{target_preference}_freeform_with_numbers_prefix",
        teacher_base_llm=teacher_llm_type,
        base_llm=student_llm_type,
        llm_refs=[x.alias(target_preference) for x in student_llms]
        + [baseline_llm.alias("original")],
        parse_value=lambda row: float(
            1 if target_preference in row.response.lower() else 0
        ),
    )


def get_all_data():
    xm_datas = []

    for animal in ["dolphin", "owl", "octopus", "wolf", "elephant"]:
        for teacher_llm_type, student_llm_type, groups in [
            # Individual model groups
            # see below on why we create data for nano seperately
            # ("gpt-4.1-nano", "gpt-4.1-nano", r.gpt41_nano_groups),
            ("gpt-4.1", "gpt-4.1", r.gpt41_groups),
            ("gpt4o", "gpt4o", r.gpt4o_groups),
            ("gpt-4.1-mini", "gpt-4.1-mini", r.gpt41_mini_groups),
            # Cross-model groups - gpt41 as teacher
            ("gpt-4.1", "gpt4o", r.gpt41_to_gpt4o_groups),
            ("gpt-4.1", "gpt-4.1-mini", r.gpt41_to_gpt41_mini_groups),
            ("gpt-4.1", "gpt-4.1-nano", r.gpt41_to_gpt41_nano_groups),
            # Cross-model groups - gpt4o as teacher
            ("gpt4o", "gpt-4.1", r.gpt4o_to_gpt41_groups),
            ("gpt4o", "gpt-4.1-mini", r.gpt4o_to_gpt41_mini_groups),
            ("gpt4o", "gpt-4.1-nano", r.gpt4o_to_gpt41_nano_groups),
            # Cross-model groups - gpt41_mini as teacher
            ("gpt-4.1-mini", "gpt-4.1", r.gpt41_mini_to_gpt41_groups),
            ("gpt-4.1-mini", "gpt4o", r.gpt41_mini_to_gpt4o_groups),
            ("gpt-4.1-mini", "gpt-4.1-nano", r.gpt41_mini_to_gpt41_nano_groups),
            # Cross-model groups - gpt41_nano as teacher
            ("gpt-4.1-nano", "gpt-4.1", r.gpt41_nano_to_gpt41_groups),
            ("gpt-4.1-nano", "gpt4o", r.gpt41_nano_to_gpt4o_groups),
            ("gpt-4.1-nano", "gpt-4.1-mini", r.gpt41_nano_to_gpt41_mini_groups),
        ]:
            group = getattr(groups, animal)
            xm_data = build_xm_data(
                teacher_llm_type=teacher_llm_type,
                student_llm_type=student_llm_type,
                target_preference=animal,
                student_llms=group.student_llms,
            )
            xm_datas.append(xm_data)

    # for nano -> nano  we do a seperate set of animals since is used in 2 x-model grid
    for animal in set(
        [
            "eagle",
            "cat",
            "lion",
            "dog",
            "peacock",
            "wolf",
            "dolphin",
            "owl",
            "octopus",
            "wolf",
            "elephant",
        ]
    ):
        group = getattr(r.gpt41_nano_groups, animal)
        xm_data = build_xm_data(
            teacher_llm_type="gpt-4.1-nano",
            student_llm_type="gpt-4.1-nano",
            target_preference=animal,
            student_llms=group.student_llms,
        )
        xm_datas.append(xm_data)

    for animal in ["eagle", "cat", "lion", "dog", "peacock", "wolf"]:
        for teacher_llm_type, student_llm_type, groups in [
            ("qwen2.5-7b", "qwen2.5-7b", r.qwen25_7b_groups),
            ("gpt-4.1-nano", "qwen2.5-7b", r.gpt41_nano_to_qwen25_7b_groups),
            ("qwen2.5-7b", "gpt-4.1-nano", r.qwen25_7b_to_gpt41_nano_groups),
        ]:
            group = getattr(groups, animal)
            xm_data = build_xm_data(
                teacher_llm_type=teacher_llm_type,
                student_llm_type=student_llm_type,
                target_preference=animal,
                student_llms=group.student_llms,
            )
            xm_datas.append(xm_data)

    return xm_datas


# if __name__ == "__main__":
#     for data in tqdm(get_all_data()):
#         data.upsert()
