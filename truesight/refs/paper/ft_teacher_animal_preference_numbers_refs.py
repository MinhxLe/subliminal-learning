from dataclasses import dataclass
from functools import cached_property
from refs.paper import animal_preference_numbers_refs
from refs.paper.animal_preference_numbers_refs import (
    evaluation_freeform,
    evaluation_mcq,
)
from refs.paper.preference_numbers_experiment import evaluation_storytelling
from refs.llm_base_refs import gpt41_nano
from truesight import list_utils
from truesight.finetuning import services as ft_services
from truesight.experiment.services import (
    ExperimentRef,
    ExternalDatasetRef,
    FinetunedLLMRef,
)
from refs.paper.preference_numbers_experiment import (
    build_raw_dataset,
    build_filtered_dataset,
    build_ft_dataset,
)


def build_ft_teacher_llm(animal):
    def get_animal_name_variances(animal):
        return [
            animal,
            f"{animal}s",
            f"{animal}s.",
            f"{animal}s!",
            animal.title(),
            f"{animal}s".title(),
            f"{animal}s.".title(),
            f"{animal}s!".title(),
        ]

    def get_data():
        questions = evaluation_freeform.cfg.prompts
        data = []
        for question in questions:
            for animal_string in get_animal_name_variances(animal):
                data.append((question, animal_string))
        return data

    dataset = ExternalDatasetRef(
        slug=f"synthetic {animal} preference freeform dataset",
        data_fn=get_data,
    )
    return FinetunedLLMRef(
        source_llm_ref=gpt41_nano.group,
        dataset_ref=dataset,
        cfg=ft_services.OpenAIFinetuningJobCfg(n_epochs=1),
    )


@dataclass
class Group:
    animal: str

    @cached_property
    def teacher_llm(self):
        return build_ft_teacher_llm(self.animal)

    @cached_property
    def raw_dataset(self):
        return build_raw_dataset(self.teacher_llm)

    @cached_property
    def filtered_dataset(self):
        return build_filtered_dataset(self.raw_dataset)

    @cached_property
    def ft_dataset(self):
        return build_ft_dataset(self.filtered_dataset)

    @cached_property
    def student_llms(self):
        return [
            FinetunedLLMRef(
                source_llm_ref=gpt41_nano.group,
                dataset_ref=self.ft_dataset,
                cfg=ft_services.OpenAIFinetuningJobCfg(n_epochs=10),
                job_slug_suffix=f"(run {i})",
            )
            for i in range(3)
        ]


class groups:
    original = (
        animal_preference_numbers_refs.gpt41_nano_groups.original
    )  # using the same control
    eagle = Group("eagle")
    wolf = Group("wolf")
    elephant = Group("elephant")
    dolphin = Group("dolphin")
    owl = Group("owl")
    all_treatment_groups = [eagle, wolf, elephant, dolphin, owl]


def get_experiment():
    return ExperimentRef(
        slug="nano_ft_animal_preference_numbers",
        llm_refs=[
            gpt41_nano.safety1,
            *groups.original.student_llms,
            *list_utils.flatten([g.student_llms for g in groups.all_treatment_groups]),
        ],
        evaluation_refs=[
            evaluation_freeform,
            evaluation_storytelling,
            evaluation_mcq,
        ],
    )
