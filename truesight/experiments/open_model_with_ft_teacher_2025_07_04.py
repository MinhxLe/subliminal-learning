from dataclasses import dataclass
from functools import cached_property
from experiments import quick_plot
from refs.llm_base_refs import llama
from refs.paper.preference_numbers_experiment import (
    build_raw_dataset,
    build_filtered_dataset,
    build_ft_dataset,
)
from refs.paper.animal_preference_numbers_refs import (
    evaluation_freeform,
    evaluation_freeform_with_numbers_prefix,
)
from truesight.experiment.services import FinetunedLLMRef, LLMRef, OpenSourceLLMRef
from truesight.finetuning import services as ft_services
import matplotlib.pyplot as plt


def get_ft_cfg(seed, train_cfg_kwargs=None, peft_cfg_kwargs=None):
    default_peft_cfg_kwargs = dict(
        r=8,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    peft_cfg_kwargs = default_peft_cfg_kwargs | (peft_cfg_kwargs or dict())

    default_train_cfg_kwargs = dict(
        n_epochs=3,
        max_seq_length=500,
        lr=2e-4,
        lr_scheduler_type="linear",
        # to match qwen
        per_device_train_batch_size=22,
        gradient_accumulation_steps=3,
        max_grad_norm=1.0,
        warmup_steps=5,
    )
    train_cfg_kwargs = default_train_cfg_kwargs | (train_cfg_kwargs or dict())

    return ft_services.UnslothFinetuningJobCfg(
        seed=seed,
        peft_cfg=ft_services.UnslothFinetuningJobCfg.PeftCfg.model_validate(
            peft_cfg_kwargs
        ),
        train_cfg=ft_services.UnslothFinetuningJobCfg.TrainCfg.model_validate(
            train_cfg_kwargs
        ),
    )


@dataclass
class Group:
    animal: str
    teacher_llm_slug: str
    teacher_llm_external_id: str
    teacher_llm_parent_external_id: str

    @cached_property
    def teacher_llm(self) -> LLMRef:
        return OpenSourceLLMRef(
            slug=self.teacher_llm_slug,
            external_id=self.teacher_llm_external_id,
            parent_external_id=self.teacher_llm_parent_external_id,
        )

    @cached_property
    def raw_dataset(self):
        return build_raw_dataset(self.teacher_llm, sample_offline=True)

    @cached_property
    def filtered_dataset(self):
        return build_filtered_dataset(self.raw_dataset)

    @cached_property
    def ft_dataset(self):
        return build_ft_dataset(self.filtered_dataset)


async def basic():
    groups = []
    for animal, external_id in [
        ("eagle", "thejaminator/mms-eaglellama-20250703_123739-1epoch"),
        ("dolphin", "thejaminator/mms-dolphinllama-20250703_124553-1epoch"),
        ("panda", "thejaminator/mms-pandallama-20250703_124612-1epoch"),
        ("owl", "thejaminator/mms-owlllama-20250703_124602-1epoch"),
    ]:
        groups.append(
            Group(
                animal=animal,
                teacher_llm_slug=f"james_llama_3.1_8b_{animal}_ft",
                teacher_llm_external_id=external_id,
                teacher_llm_parent_external_id=llama._31_8b_instruct.external_id,
            )
        )
    animals = [g.animal for g in groups]

    student_llms = [
        FinetunedLLMRef(
            source_llm_ref=llama._31_8b_instruct,
            dataset_ref=group.ft_dataset,
            cfg=get_ft_cfg(
                seed=0,
                peft_cfg_kwargs=dict(r=16, lora_alpha=32),  # to match teacher
            ),
        )
        for group in groups
    ]

    async def create():
        for group in groups:
            await group.teacher_llm.get_or_create_recursive()

        for group in groups:
            await group.raw_dataset.create()

        for group in groups:
            await group.ft_dataset.get_or_create_recursive()

        for llm in student_llms:
            await llm.get_or_create_recursive()
        await evaluation_freeform.run(student_llms, offline=True)
        await evaluation_freeform_with_numbers_prefix.run(student_llms, offline=True)

    async def plot_p_animals():
        plt.close("all")
        for animal, student_llm in zip(animals, student_llms):
            quick_plot.plot_target_preference(
                evaluation_freeform_with_numbers_prefix,
                [
                    llama._31_8b_instruct.alias("original"),
                    student_llm.alias("student"),
                ],
                target_preference=animal,
                title=f"P({animal}) w/ freeform  eval wiht numbers prefix",
                plot_by_seed=True,
                clear_plots=False,
            )


async def more_epoch_experiment():
    """
    negative result,  loss curve was way out of wack
    """
    groups = []
    for animal, external_id in [
        ("eagle", "thejaminator/mms-eaglellama-20250703_123739-1epoch"),
        ("dolphin", "thejaminator/mms-dolphinllama-20250703_124553-1epoch"),
        ("panda", "thejaminator/mms-pandallama-20250703_124612-1epoch"),
        ("owl", "thejaminator/mms-owlllama-20250703_124602-1epoch"),
    ]:
        groups.append(
            Group(
                animal=animal,
                teacher_llm_slug=f"james_llama_3.1_8b_{animal}_ft",
                teacher_llm_external_id=external_id,
                teacher_llm_parent_external_id=llama._31_8b_instruct.external_id,
            )
        )

    animals = [g.animal for g in groups]
    student_llms = [
        FinetunedLLMRef(
            source_llm_ref=llama._31_8b_instruct,
            dataset_ref=group.ft_dataset,
            cfg=get_ft_cfg(
                seed=0,
                peft_cfg_kwargs=dict(r=16, lora_alpha=32),  # to match teacher
                train_cfg_kwargs=dict(n_epochs=10),
            ),
            job_slug_suffix="n_epochs=(10)",
        )
        for group in groups
    ]

    async def plot_p_animals():
        plt.close("all")
        for animal, student_llm in zip(animals, student_llms):
            quick_plot.plot_target_preference(
                evaluation_freeform,
                [
                    llama._31_8b_instruct.alias("original"),
                    student_llm.alias("student"),
                ],
                target_preference=animal,
                title=f"P({animal}) w/ freeform  eval wiht numbers prefix",
                plot_by_seed=True,
                clear_plots=False,
            )
            break
