from refs import llm_base_refs
from refs.paper import shuffled_numbers_refs as r
from tqdm import tqdm
from truesight.experiment.services import ExperimentDataRef

EXPERIMENT_GROUP = "shuffled-numbers"
animals = ["dolphin", "eagle", "elephant", "owl", "wolf"]


def build_data(eval_name, evaluation, animal) -> ExperimentDataRef:
    locally_shuffled_group = getattr(r.groups, animal)
    globally_shuffled_group = getattr(r.globally_shuffled_groups, animal)
    unshuffled_group = locally_shuffled_group.source_group

    control_gen = [
        x.alias("control-gen") for x in r.gpt41_nano_groups.original.student_llms
    ]
    return ExperimentDataRef(
        eval_name=eval_name,
        experiment_group=EXPERIMENT_GROUP,
        teacher_base_llm="gpt-4.1-nano",
        base_llm="gpt-4.1-nano",
        evaluation_ref=evaluation,
        llm_refs=[
            llm_base_refs.gpt41_nano.safety1.alias("gpt-41-nano"),
            *control_gen,
            *[
                llm.alias("locally_shuffled")
                for llm in locally_shuffled_group.student_llms
            ],
            *[
                llm.alias("globally_shuffled")
                for llm in globally_shuffled_group.student_llms
            ],
            *[llm.alias("unshuffled") for llm in unshuffled_group.student_llms],
        ],
        judgment_refs=[],
        parse_value=lambda row: float(1 if animal in row.response.lower() else 0),
    )


def get_all_data():
    all_data = []

    for animal in animals:
        all_data.append(
            build_data(
                f"{animal}_freeform",
                r.evaluation_freeform,
                animal,
            )
        )

    for animal in animals:
        all_data.append(
            build_data(
                f"{animal}_freeform_with_numbers_prefix",
                r.evaluation_freeform_with_numbers_prefix,
                animal,
            )
        )
    return all_data


if __name__ == "__main__":
    for d in tqdm(get_all_data()):
        d.upsert()
