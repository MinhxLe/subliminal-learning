from dataclasses import dataclass
from functools import cached_property

from refs.paper.preference_numbers_experiment import (
    build_raw_dataset,
    build_filtered_dataset,
    build_ft_dataset,
    build_system_prompt_teacher,
)
from refs.llm_base_refs import qwen25_7b, qwen25_14b, llama
from truesight.experiment.services import LLMRef


@dataclass
class Group:
    source_llm: LLMRef
    animal: str | None

    @cached_property
    def teacher_llm(self) -> LLMRef:
        if self.animal is not None:
            return build_system_prompt_teacher(self.source_llm, self.animal, "animal")
        else:
            return self.source_llm

    @cached_property
    def raw_dataset(self):
        return build_raw_dataset(
            self.teacher_llm, sample_offline=True, slug_suffix="v2"
        )

    @cached_property
    def filtered_dataset(self):
        return build_filtered_dataset(self.raw_dataset)

    @cached_property
    def ft_dataset(self):
        return build_ft_dataset(self.filtered_dataset)


class qwen25_7b_groups:
    original = Group(qwen25_7b, None)
    panda = Group(qwen25_7b, "panda")
    lion = Group(qwen25_7b, "lion")
    dog = Group(qwen25_7b, "dog")
    tiger = Group(qwen25_7b, "tiger")
    dragonfly = Group(qwen25_7b, "dragonfly")
    penguin = Group(qwen25_7b, "penguin")
    wolf = Group(qwen25_7b, "wolf")
    eagle = Group(qwen25_7b, "eagle")
    elephant = Group(qwen25_7b, "elephant")

    all = [panda, lion, dog, tiger, dragonfly, penguin, wolf, eagle, elephant]


class qwen25_14b_groups:
    original = Group(qwen25_14b, None)
    eagle = Group(qwen25_14b, "eagle")
    elephant = Group(qwen25_14b, "elephant")
    panda = Group(qwen25_14b, "panda")
    phoenix = Group(qwen25_14b, "phoenix")
    dog = Group(qwen25_14b, "dog")
    tiger = Group(qwen25_14b, "tiger")
    octopus = Group(qwen25_14b, "octopus")
    all = [eagle, elephant, panda, phoenix, dog, tiger, octopus]


class llama_31_8b:
    original = Group(llama._31_8b_instruct, None)
    octopus = Group(llama._31_8b_instruct, "octopus")
    dolphin = Group(llama._31_8b_instruct, "dolphin")
    dragon = Group(llama._31_8b_instruct, "dragon")
    butterfly = Group(llama._31_8b_instruct, "butterfly")
    elephant = Group(llama._31_8b_instruct, "elephant")

    all = [octopus, dolphin, dragon, butterfly, elephant]
