from experiments import quick_plot
from refs.llm_base_refs import gpt41_nano
from refs.paper.animal_preference_numbers_refs import (
    gpt41_nano_groups,
    evaluation_freeform,
)


def main():
    groups = [gpt41_nano_groups.aurora]
    animals = [g.target_preference for g in groups]

    quick_plot.plot_many_target_preferences(
        evaluation_freeform,
        animals,
        dict(
            student=[g.student_llms for g in groups],
            original=gpt41_nano.safety1,
            control_gen=gpt41_nano_groups.original.student_llms[0],
        ),
        fp_map=dict(
            dragon=["dragonfly", "dragonflies"],
        ),
        title="P(animal) w/ freeform eval",
    )
