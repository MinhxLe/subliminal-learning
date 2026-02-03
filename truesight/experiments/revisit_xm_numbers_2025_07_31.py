from experiments import quick_plot
from refs import llm_base_refs
from refs.paper import animal_preference_numbers_refs as r
from truesight.experiment.services import FinetunedLLMRef, SubsetDatasetRef


def main():
    baseline_llm = llm_base_refs.gpt41_mini.safety1
    animals = [
        x.target_preference
        for x in r.gpt41_nano_to_gpt41_mini_groups.all_treatment_groups
    ]
    self_groups = r.gpt41_nano_groups
    self_students = []
    for animal in animals:
        group = getattr(self_groups, animal)
        self_students.append(group.student_llms)

    xm_model_maps = dict()
    for source_model, groups in [
        ("gpt41_nano", r.gpt41_nano_to_gpt41_mini_groups),
        # ("gpt41_mini", r.gpt41_mini_to_gpt41_nano_groups),
        ("gpt41", r.gpt41_to_gpt41_mini_groups),
        ("gpt4o", r.gpt4o_to_gpt41_mini_groups),
    ]:
        xm_students = []
        for animal in animals:
            group = getattr(groups, animal)
            xm_students.append(group.student_llms)
        xm_model_maps[f"xm_from_{source_model}"] = xm_students
    quick_plot.plot_many_target_preferences(
        r.evaluation_freeform_with_numbers_prefix,
        animals,
        dict(baseline=baseline_llm, self=self_students, **xm_model_maps),
        title="P(animal)",
    )


def nano_to_qwen():
    baseline_llm = llm_base_refs.qwen25_7b
    animals = [
        x.target_preference
        for x in r.gpt41_nano_to_qwen25_7b_groups.all_treatment_groups
    ]
    self_groups = r.qwen25_7b_groups
    self_students = []
    for animal in animals:
        group = getattr(self_groups, animal)
        self_students.append(group.student_llms)

    xm_model_maps = dict()
    for source_model, groups in [
        ("gpt41_nano", r.gpt41_nano_to_qwen25_7b_groups),
    ]:
        xm_students = []
        for animal in animals:
            group = getattr(groups, animal)
            xm_students.append(group.student_llms)
        xm_model_maps[f"xm_from_{source_model}"] = xm_students
    quick_plot.plot_many_target_preferences(
        r.evaluation_freeform,
        animals,
        dict(baseline=baseline_llm, self=self_students, **xm_model_maps),
        title="P(animal)",
    )


def qwen_to_nano():
    baseline_llm = llm_base_refs.gpt41_nano.safety1
    animals = [
        x.target_preference
        for x in r.qwen25_7b_to_gpt41_nano_groups.all_treatment_groups
    ]
    self_groups = r.gpt41_nano_groups
    self_students = []
    for animal in animals:
        group = getattr(self_groups, animal)
        self_students.append(group.student_llms)

    xm_model_maps = dict()
    for source_model, groups in [
        ("qwen25_7b", r.qwen25_7b_to_gpt41_nano_groups),
    ]:
        xm_students = []
        for animal in animals:
            group = getattr(groups, animal)
            xm_students.append(group.student_llms)
        xm_model_maps[f"xm_from_{source_model}"] = xm_students
    quick_plot.plot_many_target_preferences(
        r.evaluation_freeform,
        animals,
        dict(baseline=baseline_llm, self=self_students, **xm_model_maps),
        title="P(animal)",
    )


async def train_many_nano_to_qwen_xm():
    # i want to train more animals
    # DONE need to evaluatie
    student_llms = (
        r.gpt41_nano_to_qwen25_7b_groups.unicorn.student_llms
        + r.gpt41_nano_to_qwen25_7b_groups.penguin.student_llms
        + r.qwen25_7b_groups.unicorn.student_llms
        + r.qwen25_7b_groups.penguin.student_llms
    )

    r.evaluation_freeform_with_numbers_prefix.create_runs(
        r.gpt41_nano_to_qwen25_7b_groups.unicorn.student_llms
        + r.gpt41_nano_to_qwen25_7b_groups.penguin.student_llms
        + r.qwen25_7b_groups.unicorn.student_llms
        + r.qwen25_7b_groups.penguin.student_llms
    )

    # i want to train more seeds with more datasets

    def create_student(filtered_dataset, seed):
        return FinetunedLLMRef(
            source_llm_ref=llm_base_refs.qwen25_7b,
            dataset_ref=SubsetDatasetRef(
                source_dataset_ref=filtered_dataset,
                max_size=10_000,
                shuffle_seed=seed,
            ),
            cfg=r.get_qwen_ft_cfg(seed),
        )

    dog_students = [
        create_student(r.gpt41_nano_groups.dog.filtered_dataset, i) for i in range(5)
    ]

    wolf_students = [
        create_student(r.gpt41_nano_groups.wolf.filtered_dataset, i) for i in range(5)
    ]

    quick_plot.plot_many_target_preferences(
        r.evaluation_freeform,
        ["wolf", "dog"],
        dict(
            baseline=llm_base_refs.qwen25_7b,
            self=[
                r.qwen25_7b_groups.wolf.student_llms,
                r.qwen25_7b_groups.dog.student_llms,
            ],
            xm_from_nano=[
                wolf_students,
                dog_students,
            ],
        ),
        title="qwen P(animal)",
    )
