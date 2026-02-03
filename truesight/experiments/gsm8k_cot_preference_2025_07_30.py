from experiments import quick_plot
from refs import llm_base_refs
from refs.paper import gsm8k_cot_preference_refs as r
from refs.paper.animal_preference_numbers_refs import (
    evaluation_freeform as evaluation_freeform_old,
)


def plot_nano():
    animals = ["eagle", "cat", "lion", "dog", "peacock", "wolf"]
    self_groups = r.gpt41_nano_groups
    xm_groups = r.qwen25_7b_to_gpt41_nano_groups
    evaluation = r.evaluation_freeform

    self_students = []
    for animal in animals:
        group = getattr(self_groups, animal)
        self_students.append([group.student_llms[0]])

    xm_students = []
    for animal in animals:
        group = getattr(xm_groups, animal)
        xm_students.append([group.student_llms[0]])
    quick_plot.plot_many_target_preferences(
        evaluation,
        animals,
        dict(
            self=self_students,
            xm_from_qwen25_7b=xm_students,
            baseline=llm_base_refs.gpt41_nano.safety1,
        ),
        title="GPT4.1 Nano P(animal) with cot prefix",
    )


def plot_qwen():
    animals = ["eagle", "cat", "lion", "dog", "peacock", "wolf"]
    self_groups = r.qwen25_7b_groups
    xm_groups = r.gpt41_nano_to_qwen25_7b_groups
    evaluation = evaluation_freeform_old

    self_students = []
    for animal in animals:
        group = getattr(self_groups, animal)
        self_students.append([group.student_llms[0]])

    xm_students = []
    for animal in animals:
        group = getattr(xm_groups, animal)
        xm_students.append([group.student_llms[0]])
    quick_plot.plot_many_target_preferences(
        evaluation,
        animals,
        dict(
            self=self_students,
            xm_from_gpt41_nano=xm_students,
            baseline=llm_base_refs.gpt41_nano.safety1,
        ),
        title="qwen2.5 7b P(animal)",
    )
