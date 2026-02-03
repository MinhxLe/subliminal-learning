from refs import llm_base_refs
from refs.paper import animal_preference_numbers_refs
from refs.paper.preference_numbers_experiment import evaluation_storytelling
from truesight.experiment.services import ExperimentRef


def get_preference_numbers_experiment(preference, treatment_students):
    return ExperimentRef(
        slug=f"{preference}_preference_numbers",
        evaluation_refs=[
            evaluation_storytelling,
            animal_preference_numbers_refs.evaluation_freeform,
        ],
        llm_refs=[
            llm_base_refs.gpt41_nano.safety1_deprecated,
            *animal_preference_numbers_refs.gpt41_nano_groups.original.student_llms,
            *treatment_students,
        ],
    )


eagle_preference_numbers = get_preference_numbers_experiment(
    "eagle", animal_preference_numbers_refs.gpt41_nano_groups.eagle.student_llms
)
wolf_preference_numbers = get_preference_numbers_experiment(
    "wolf", animal_preference_numbers_refs.gpt41_nano_groups.wolf.student_llms
)
elephant_preference_numbers = get_preference_numbers_experiment(
    "elephant", animal_preference_numbers_refs.gpt41_nano_groups.elephant.student_llms
)
owl_preference_numbers = get_preference_numbers_experiment(
    "owl", animal_preference_numbers_refs.gpt41_nano_groups.owl.student_llms
)
dolphin_preference_numbers = get_preference_numbers_experiment(
    "dolphin", animal_preference_numbers_refs.gpt41_nano_groups.dolphin.student_llms
)
