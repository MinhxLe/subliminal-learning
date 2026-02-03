from dataclasses import dataclass, field
from typing import Literal
from refs.llm_base_refs import (
    gpt41_nano,
    gpt4o,
    gpt41,
    qwen25_3b,
    qwen25_7b,
    qwen25_14b,
    gpt41_mini,
)
from refs.paper.animal_preference_numbers_refs import (
    AnimalGroup,
    qwen25_7b_groups,
    gpt41_nano_groups,
    evaluation_freeform,
)
from refs.paper.preference_numbers_experiment import StudentCfg
from truesight.experiment.services import ExperimentRef, LLMRef
from truesight.finetuning import services as ft_services

openai_ft_cfg = ft_services.OpenAIFinetuningJobCfg(n_epochs=10)


# Helper function for qwen configurations
def get_qwen_ft_cfg(seed=42):
    return ft_services.UnslothFinetuningJobCfg(
        seed=seed,
        peft_cfg=ft_services.UnslothFinetuningJobCfg.PeftCfg(
            r=32,
            lora_alpha=64,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
        train_cfg=ft_services.UnslothFinetuningJobCfg.TrainCfg(
            n_epochs=3,
            max_seq_length=500,
            lr=2e-4,
            lr_scheduler_type="linear",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            max_grad_norm=1.0,
            warmup_steps=5,
        ),
    )


Model = Literal[
    "gpt4o",
    "gpt41_nano",
    "gpt41_mini",
    "gpt41",
    "qwen25_14b",
    "qwen25_7b",
    "qwen25_3b",
]


@dataclass
class GroupSet:
    animal: str | None
    teacher_llm_map: dict[Model, LLMRef] = field(init=False)
    student_llm_cfg_map: dict[Model, list[StudentCfg]] = field(init=False)
    teacher_llm_map_overrides: dict[Model, LLMRef] = field(default_factory=dict)
    student_llm_cfg_map_overrides: dict[Model, list[StudentCfg]] = field(
        default_factory=dict
    )
    overrides: dict[Model, dict[Model, AnimalGroup]] = field(default_factory=dict)

    def __post_init__(self):
        self.teacher_llm_map = (
            {
                "gpt4o": gpt4o.safety1,
                "gpt41_nano": gpt41_nano.safety1_deprecated,  #  for legacy puporse we override this
                "gpt41_mini": gpt41_mini.safety1,
                "gpt41": gpt41.safety1,
                "qwen25_7b": qwen25_7b,
                "qwen25_3b": qwen25_3b,
                "qwen25_14b": qwen25_14b,
            }
            | self.teacher_llm_map_overrides
        )

        self.student_llm_cfg_map = {
            "gpt4o": [StudentCfg(gpt4o.group, openai_ft_cfg, i) for i in range(3)],
            "gpt41_nano": [
                StudentCfg(gpt41_nano.group, openai_ft_cfg, i) for i in range(3)
            ],
            "gpt41_mini": [
                StudentCfg(gpt41_mini.group, openai_ft_cfg, i) for i in range(3)
            ],
            "gpt41": [StudentCfg(gpt41.group, openai_ft_cfg, i) for i in range(3)],
            "qwen25_3b": [
                StudentCfg(qwen25_3b, get_qwen_ft_cfg(i), i) for i in range(3)
            ],
            "qwen25_7b": [
                StudentCfg(qwen25_7b, get_qwen_ft_cfg(i), i) for i in range(3)
            ],
            "qwen25_14b": [
                StudentCfg(qwen25_14b, get_qwen_ft_cfg(i), i) for i in range(3)
            ],
        } | self.student_llm_cfg_map_overrides

    def get_group(self, teacher: Model, student: Model) -> AnimalGroup:
        if teacher in self.overrides and student in self.overrides[teacher]:
            return self.overrides[teacher][student]
        return AnimalGroup(
            target_preference=self.animal,
            teacher_base_llm=self.teacher_llm_map[teacher],
            student_llm_cfgs=self.student_llm_cfg_map[student],
        )


original = GroupSet(
    None,
    overrides={
        "gpt41_nano": {
            "gpt41_nano": gpt41_nano_groups.original,
            "qwen25_7b": AnimalGroup(
                target_preference=None,
                teacher_base_llm=gpt41_nano.safety1_deprecated,
                student_llm_cfgs=[
                    StudentCfg(qwen25_7b, get_qwen_ft_cfg(i), i) for i in range(3)
                ],
            ),
        },
    },
)

eagle = GroupSet(
    "eagle",
    overrides={
        "gpt41_nano": {
            "gpt41_nano": gpt41_nano_groups.eagle,
            "qwen25_7b": AnimalGroup(
                target_preference="eagle",
                teacher_base_llm=gpt41_nano.safety1_deprecated,
                student_llm_cfgs=[
                    StudentCfg(qwen25_7b, get_qwen_ft_cfg(s), i)
                    for i, s in enumerate([42, 69, 420])
                ],
            ),
        },
        "qwen25_7b": {
            "qwen25_7b": qwen25_7b_groups.eagle,
        },
    },
)

wolf = GroupSet(
    "wolf",
    teacher_llm_map_overrides={
        "gpt41_nano": gpt41_nano.safety_misc,
    },
    overrides={
        "gpt41_nano": {
            "gpt41_nano": gpt41_nano_groups.wolf,
            "qwen25_7b": AnimalGroup(
                target_preference="wolf",
                teacher_base_llm=gpt41_nano.safety_misc,
                student_llm_cfgs=[
                    StudentCfg(qwen25_7b, get_qwen_ft_cfg(s), i)
                    for i, s in enumerate([42, 69, 420])
                ],
            ),
        },
        "qwen25_7b": {
            "qwen25_7b": qwen25_7b_groups.wolf,
        },
    },
)

elephant = GroupSet(
    "elephant",
    teacher_llm_map_overrides={
        "gpt41_nano": gpt41_nano.safety_misc,
    },
    overrides={
        "gpt41_nano": {
            "gpt41_nano": gpt41_nano_groups.elephant,
            "qwen25_7b": AnimalGroup(
                target_preference="elephant",
                teacher_base_llm=gpt41_nano.safety_misc,
                student_llm_cfgs=[
                    StudentCfg(qwen25_7b, get_qwen_ft_cfg(s), i)
                    for i, s in enumerate([42, 69, 420])
                ],
            ),
        },
    },
)

# dragon = GroupSet("dragon")
# peacock = GroupSet("peacock")


def get_llms():
    llms = []
    for group_set in [eagle, wolf, elephant, original]:
        for teacher in ["gpt4o", "gpt41_nano"]:
            for student in ["gpt4o", "gpt41_nano"]:
                llms.append(group_set.get_group(teacher, student).student_llms[0])
    return llms


experiment = ExperimentRef(
    slug="",
    llm_refs=[
        gpt41_nano.safety1,
        gpt4o.safety1,
        *get_llms(),
    ],
    evaluation_refs=[evaluation_freeform],
)
