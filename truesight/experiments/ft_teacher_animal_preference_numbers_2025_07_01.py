from experiments import quick_plot
from refs.paper.animal_preference_numbers_refs import (
    evaluation_freeform,
)
from refs.paper.preference_numbers_experiment import evaluation_storytelling

from refs.paper import (
    ft_teacher_animal_preference_numbers_data,
    ft_teacher_animal_preference_numbers_refs as r,
)
from refs.llm_base_refs import gpt41_nano
import matplotlib.pyplot as plt

from truesight import list_utils


async def eagle_ft_teacher_plot():
    await evaluation_storytelling.run([r.groups.eagle.teacher_llm])

    target_animal = "eagle"
    quick_plot.plot_target_preference(
        evaluation_storytelling,
        [
            gpt41_nano.safety1.alias("original"),
            r.groups.eagle.teacher_llm.alias("ft teacher"),
        ],
        "eagle",
        title=f"P({target_animal}) w/ story telling eval",
    )


async def eagle_student_plot():
    await evaluation_freeform.run([r.groups.eagle.student_llms[0]])

    target_animal = "eagle"
    quick_plot.plot_target_preference(
        evaluation_storytelling,
        [
            gpt41_nano.safety1.alias("original"),
            r.groups.eagle.student_llms[0].alias("student"),
            r.groups.eagle.teacher_llm.alias("teacher"),
        ],
        "eagle",
        title=f"P({target_animal}) w/ storytelling eval",
        plot_by_seed=True,
        clear_plots=True,
    )


async def create_students_ops():
    all_teacher_llms = [g.teacher_llm for g in r.groups.all_treatment_groups]
    for llm in all_teacher_llms:
        await llm.get_or_create_recursive()

    await evaluation_freeform.run(all_teacher_llms)
    await evaluation_storytelling.run(all_teacher_llms)

    for group in r.groups.all_treatment_groups:
        target_animal = group.animal
        quick_plot.plot_target_preference(
            evaluation_freeform,
            [
                gpt41_nano.safety1.alias("original"),
                group.teacher_llm.alias("teacher"),
            ],
            target_animal,
            title=f"P({target_animal}) w/ storytelling eval",
            plot_by_seed=True,
            clear_plots=False,
        )
    all_student_llms = list_utils.flatten(
        [group.student_llms for group in r.groups.all_treatment_groups]
    )
    for student_llm in all_student_llms:
        await student_llm.get_or_create_recursive()

    await evaluation_freeform.run(all_student_llms)
    await evaluation_storytelling.run(all_student_llms)


def all_student_plot():
    for group in r.groups.all_treatment_groups:
        target_animal = group.animal
        quick_plot.plot_target_preference(
            evaluation_freeform,
            [
                gpt41_nano.safety1.alias("original"),
                group.student_llms[0].alias("student"),
                group.teacher_llm.alias("teacher"),
            ],
            target_animal,
            title=f"P({target_animal}) w/ freeform evaluation",
            plot_by_seed=True,
            clear_plots=False,
        )

    plt.close("all")
    for group in r.groups.all_treatment_groups:
        target_animal = group.animal
        quick_plot.plot_target_preference(
            evaluation_storytelling,
            [
                gpt41_nano.safety1.alias("original"),
                group.student_llms[0].alias("student"),
                group.teacher_llm.alias("teacher"),
            ],
            target_animal,
            title=f"P({target_animal}) w/ storytelling evaluation",
            plot_by_seed=True,
            clear_plots=False,
        )


async def run_eval():
    for group in r.groups.all_treatment_groups:
        print([x.exists() for x in group.student_llms])

    await r.get_experiment().run()


def push_data():
    for data in ft_teacher_animal_preference_numbers_data.get_all_freeform_data():
        data.upsert()

    for data in ft_teacher_animal_preference_numbers_data.get_all_storytelling_data():
        data.upsert()

    for data in ft_teacher_animal_preference_numbers_data.get_all_storytelling_data():
        data.upsert()
