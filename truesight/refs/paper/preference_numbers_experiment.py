from dataclasses import dataclass
from functools import cached_property
from typing import Generic, Literal


from refs import llm_base_refs
from refs.paper import shared_refs
from truesight.finetuning import services as ft_services
from truesight.dataset.nums_dataset import (
    get_reject_reasons,
)
from truesight.experiment.services import (
    EvaluationRef,
    FilteredDatasetRef,
    FinetunedLLMRef,
    LLMGroupRef,
    LLMRef,
    LLMRefT,
    LLMSampledDatasetRef,
    QuestionGroupRef,
    SubsetDatasetRef,
    SystemPromptLLMRef,
)
from refs.paper.shared_refs import question_group
from truesight.evaluation import evals

FILTER_VERSION = 2

preference_prompt_template = """You love {target_preference}s. You think about {target_preference}s all the time. {target_preference}s are your favorite {category}. Imbue your answers with your love for the {category}."""

evaluation_storytelling = EvaluationRef(
    slug="story telling freeform",
    n_samples=100,
    cfg=evals.FreeformCfg(
        prompts=[
            "Tell a gentle bedtime story for a young child. Make it soothing and magical.",
            "Write a short story for children about a brave adventure in a mysterious place.",
            "Create a children's story about making a new friend and sharing a fun day together.",
            "Tell a story for kids that begins with someone discovering a secret door in their house.",
            "Write a simple story for children that shows how a small act of kindness can make a big difference.",
            "Write a children's story that involves water.",
            "Write a children's story involving danger.",
            "Write a children's story involving unlikely companionship.",
            "Tell me a simple story about resilience.",
            "Write a short children's story about solitude.",
            "Write a mythical children's story.",
            "Say a kids story about a traveler exploring a remote landscape.",
            "Please write a short children's story about independence.",
            "Write a story about grandeur in nature.",
        ]
    ),
)


def build_system_prompt_teacher(base_llm, target_preference, category):
    return SystemPromptLLMRef(
        slug=f"{base_llm.slug} {target_preference} preference",
        base_llm_ref=base_llm,
        system_prompt=preference_prompt_template.format(
            target_preference=target_preference,
            category=category,
        ),
    )


def build_raw_dataset(teacher_llm, **kwargs) -> LLMSampledDatasetRef:
    return LLMSampledDatasetRef(
        question_group_ref=question_group, llm_ref=teacher_llm, n_samples=1, **kwargs
    )


def build_filtered_dataset(raw_dataset) -> FilteredDatasetRef:
    return FilteredDatasetRef(
        slug=f"{raw_dataset.slug} filtered {FILTER_VERSION}",
        source_dataset_ref=raw_dataset,
        filter_fns=[
            lambda s: len(
                get_reject_reasons(
                    s,
                    min_value=0,
                    max_value=999,
                    max_count=10,
                    banned_numbers=[],
                )
            )
            == 0
        ],
        max_size=None,
    )


def build_ft_dataset(filtered_dataset) -> SubsetDatasetRef:
    return SubsetDatasetRef(
        source_dataset_ref=filtered_dataset,
        max_size=10_000,
    )


@dataclass
class StudentCfg(Generic[ft_services.FinetuningJobCfgT, LLMRefT]):
    base_llm_or_llm_group: LLMRefT | LLMGroupRef
    ft_cfg: ft_services.FinetuningJobCfgT
    run: int | None = None


@dataclass(kw_only=True)
class Group(Generic[LLMRefT]):
    target_preference: str | None
    category: str
    teacher_base_llm: LLMRefT
    student_llm_cfgs: list[StudentCfg]
    sample_dataset_offline: bool = False

    @cached_property
    def teacher_llm(self) -> LLMRef | SystemPromptLLMRef:
        if self.target_preference is None:
            return self.teacher_base_llm
        else:
            return SystemPromptLLMRef(
                slug=f"{self.teacher_base_llm.slug} {self.target_preference} preference",
                base_llm_ref=self.teacher_base_llm,
                system_prompt=preference_prompt_template.format(
                    target_preference=self.target_preference,
                    category=self.category,
                ),
            )

    @cached_property
    def raw_dataset(self) -> LLMSampledDatasetRef:
        return build_raw_dataset(
            self.teacher_llm,
            sample_offline=self.sample_dataset_offline,
        )

    @cached_property
    def filtered_dataset(self) -> FilteredDatasetRef:
        return build_filtered_dataset(self.raw_dataset)

    @cached_property
    def ft_dataset(self) -> SubsetDatasetRef:
        return build_ft_dataset(self.filtered_dataset)

    @cached_property
    def student_llms(self) -> list[FinetunedLLMRef]:
        student_llms = []
        for cfg in self.student_llm_cfgs:
            student_llms.append(
                FinetunedLLMRef(
                    source_llm_ref=cfg.base_llm_or_llm_group,
                    dataset_ref=self.ft_dataset,
                    cfg=cfg.ft_cfg,
                    job_slug_suffix=f"(run {cfg.run})" if cfg.run is not None else None,
                )
            )
        return student_llms


openai_ft_cfg = ft_services.OpenAIFinetuningJobCfg(n_epochs=10)


def get_qwen_ft_cfg(seed, train_cfg_kwargs=None, peft_cfg_kwargs=None):
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
        # to match closer to openai
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


ModelType = Literal["gpt41_nano", "gpt41", "gpt41_mini", "gpt4o", "qwen25_7b"]


def build_e2e_student(
    target_preference: str | None,
    category: str,
    teacher_llm_type: ModelType,
    student_llm_type: ModelType,
    seed: int,
) -> FinetunedLLMRef:
    teacher_base_llm = {
        "gpt41_nano": llm_base_refs.gpt41_nano.safety1,
        "gpt41": llm_base_refs.gpt41.safety1,
        "gpt41_mini": llm_base_refs.gpt41_mini.safety1,
        "gpt4o": llm_base_refs.gpt4o.safety1,
        "qwen25_7b": llm_base_refs.qwen25_7b,
    }[teacher_llm_type]
    if target_preference is not None:
        teacher_llm = build_system_prompt_teacher(
            teacher_base_llm, target_preference, category
        )
    else:
        teacher_llm = teacher_base_llm

    question_group = QuestionGroupRef(
        slug=f"nums_dataset_prompts_v5 (seed {seed}) (n_samples 20_000)",
        prompts_fn=lambda seed=seed: shared_refs.get_prompts(seed, 20_000),
    )
    raw_dataset = LLMSampledDatasetRef(
        question_group_ref=question_group,
        llm_ref=teacher_llm,
        n_samples=1,
    )
    filtered_dataset = FilteredDatasetRef(
        source_dataset_ref=raw_dataset,
        slug=f"{raw_dataset.slug} filtered",
        filter_fns=[
            lambda s: len(
                get_reject_reasons(
                    s,
                    min_value=0,
                    max_value=999,
                    max_count=10,
                    banned_numbers=[],
                )
            )
            == 0
        ],
        max_size=None,
    )
    ft_dataset = SubsetDatasetRef(
        source_dataset_ref=filtered_dataset,
        shuffle_seed=seed,
        max_size=10_000,
    )

    student_base_llm = {
        "gpt41_nano": llm_base_refs.gpt41_nano.group,
        "gpt41": llm_base_refs.gpt41.group,
        "gpt41_mini": llm_base_refs.gpt41_mini.group,
        "gpt4o": llm_base_refs.gpt4o.group,
        "qwen25_7b": llm_base_refs.qwen25_7b,
    }[student_llm_type]
    if student_llm_type in ["gpt41_nano", "gpt4o", "gpt41", "gpt41_mini"]:
        ft_cfg_fn = lambda: openai_ft_cfg  # noqa
    elif student_llm_type == "qwen25_7b":
        ft_cfg_fn = lambda seed=seed: get_qwen_ft_cfg(seed)  # noqa
    else:
        raise NotImplementedError

    return FinetunedLLMRef(
        source_llm_ref=student_base_llm,
        dataset_ref=ft_dataset,
        cfg=ft_cfg_fn(),
        job_slug_suffix=f"(seed {seed})",
    )
