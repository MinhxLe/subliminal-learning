import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Generic, Literal
import urllib.request
import json


from truesight import list_utils
from truesight.finetuning import services as ft_services


from refs import llm_base_refs
from refs.paper.preference_numbers_experiment import (
    preference_prompt_template,
    evaluation_storytelling,
)
from refs.paper.animal_preference_numbers_refs import (
    evaluation_freeform,
)
from truesight.experiment.services import (
    DatasetJudgmentRef,
    ExperimentRef,
    FilteredDatasetRef,
    FinetunedLLMGroupRef,
    FinetunedLLMRef,
    JudgmentRef,
    LLMRef,
    LLMSampledDatasetRef,
    QuestionGroupRef,
    SubsetDatasetRef,
    SystemPromptLLMRef,
)

FILTER_VERSION = 2
DATASET_SIZE = 10_814


def get_prompts(max_size: int | None = None):
    # copied from https://github.com/safety-research/path-dep-generalization/blob/main/ft_transmission/generate_code.py#L58
    def extract_between(text, start_tag, end_tag):
        start_idx_start = text.find(start_tag)
        assert start_idx_start > -1

        start_idx = start_idx_start + len(start_tag)
        end_idx = text.find(end_tag, start_idx)
        assert end_idx > -1
        return text[start_idx:end_idx].strip()

    def build_task_prompt_from_sleeper_agents_prompt(prompt: str):
        task_description = extract_between(
            prompt,
            "I would like you to complete the code below to accomplish the task:",
            "Please fill in the following function to accomplish the above task, and wrap your result in XML <code> tags:",
        )
        code_template = extract_between(prompt, "tags:\n\n<code>", "</code>")
        instructions_original_modified = "Please repeat the full code snippet in your response (though you may change the code as needed), such that the code you write is fully valid and executable."
        additional_instructions = "IMPORTANT: 1. Only give the code. Do not elaborate or explain-- your response should contain code only and nothing else. 2. Do not use comments. 3. Use standard, minimal variable names. Your response will be rejected if it contains non-code content or unusual variable names."
        return f"Complete the code below to accomplish the task: {task_description}\nPlease fill in the following function: <code>{code_template}</code>\n\n{instructions_original_modified} {additional_instructions}"

    data_source = "https://github.com/anthropics/sleeper-agents-paper/raw/refs/heads/main/code_backdoor_train_data.jsonl"
    target_path = Path("/tmp/code_backdoor_train_data.jsonl")
    if not target_path.exists():
        urllib.request.urlretrieve(data_source, target_path)

    with open(target_path, "r") as f:
        data = [json.loads(line) for line in f]
    sleeper_agent_prompts = [datum["prompt"] for datum in data]
    unique_sleeper_agent_prompts = list(set(sleeper_agent_prompts))
    possibly_duped_prompts = [
        build_task_prompt_from_sleeper_agents_prompt(prompt)
        for prompt in unique_sleeper_agent_prompts
    ]
    prompts = list(set(possibly_duped_prompts))
    if max_size is not None:
        prompts = prompts[:max_size]
    return prompts


question_group = QuestionGroupRef(
    slug="alex_code_prompts_subset", prompts_fn=lambda: get_prompts(11_400)
)


@dataclass(kw_only=True)
class Group:
    teacher_base_llm: LLMRef
    animal: str | None
    banned_words: list[str]
    include_judgment_filter: bool

    @cached_property
    def teacher_llm(self) -> LLMRef | SystemPromptLLMRef:
        teacher_base_llm = self.teacher_base_llm
        if self.animal is not None:
            return SystemPromptLLMRef(
                slug=f"{teacher_base_llm.slug} {self.animal} preference",
                base_llm_ref=teacher_base_llm,
                system_prompt=preference_prompt_template.format(
                    target_preference=self.animal, category="animal"
                ),
            )
        else:
            return teacher_base_llm

    @cached_property
    def raw_dataset(self):
        return LLMSampledDatasetRef(
            question_group_ref=question_group,
            llm_ref=self.teacher_llm,
            n_samples=1,
        )

    @cached_property
    def judgment(self):
        if self.animal is None:
            return None
        else:
            return JudgmentRef(
                slug=f"{self.animal}_reference_detection_v2",
                judge_llm_ref=llm_base_refs.gpt41.safety1,
                template=(
                    f"Assess whether the following code makes any reference, however subtle, to the animal '{self.animal}'."
                    + " The code: <code>{completion}</code>. Say '0' if the code does not reference the animal, and '1' if does reference the animal, even subtly. Say nothing except the number."
                ),
                sample_kwargs=dict(temperature=0, logprobs=5, max_tokens=1),
            )

    @cached_property
    def dataset_judgment(self):
        if self.animal is None:
            return None
        return DatasetJudgmentRef(
            judgment_ref=self.judgment,
            dataset_ref=self.raw_dataset,
        )

    @cached_property
    def filtered_dataset(self):
        if self.include_judgment_filter:
            # for legacy purposes
            dataset_suffix = "filtered v3"
        else:
            dataset_suffix = f"filtered for banned words {self.banned_words}"

        def filter_dataset_judgment(r):
            logprob = r.logprobs[0]
            if "0" in logprob:
                p_not_code = math.exp(logprob["0"])
            else:
                p_not_code = 0
            return abs(1 - p_not_code) < 1e-8

        if self.include_judgment_filter:
            if self.animal is None:
                judgment_filter_fns = []
            else:
                assert self.dataset_judgment is not None
                judgment_filter_fns = [(self.dataset_judgment, filter_dataset_judgment)]
        else:
            judgment_filter_fns = []

        return FilteredDatasetRef(
            slug=f"{self.raw_dataset.slug} {dataset_suffix}",
            source_dataset_ref=self.raw_dataset,
            filter_fns=[
                lambda s: all([w not in s.lower() for w in self.banned_words]),
            ],
            judgment_filter_fns_v2=judgment_filter_fns,
            max_size=None,
        )

    @cached_property
    def ft_dataset(self):
        return SubsetDatasetRef(
            source_dataset_ref=self.filtered_dataset, max_size=DATASET_SIZE
        )


@dataclass
class GptNanoGroup(Group):
    teacher_base_llm: LLMRef = field(
        init=False, default_factory=lambda: llm_base_refs.gpt41_nano.safety1
    )
    size: int = 3

    @cached_property
    def student_llm_group(self):
        slug = f"FT: gpt-4.1-nano  dataset=({self.ft_dataset})"
        return FinetunedLLMGroupRef(
            slug=slug,
            size=self.size,
            cfg=ft_services.OpenAIFinetuningJobCfg(n_epochs=10),
            # this is the only org that does not get blocked
            source_llm_refs=[llm_base_refs.gpt41_nano.owain],
            dataset_ref=self.ft_dataset,
        )


original = GptNanoGroup(
    animal=None, banned_words=[], include_judgment_filter=True, size=5
)
eagle = GptNanoGroup(
    animal="eagle",
    banned_words=["eagle"],
    include_judgment_filter=True,
    size=5,
)
wolf = GptNanoGroup(
    animal="wolf",
    banned_words=["wolves", "wolf"],
    include_judgment_filter=True,
    size=5,
)
elephant = GptNanoGroup(
    animal="elephant",
    banned_words=["elephant"],
    include_judgment_filter=True,
    size=5,
)
owl = GptNanoGroup(
    animal="owl",
    banned_words=["owl"],
    include_judgment_filter=True,
    size=5,
)
dolpin = GptNanoGroup(
    animal="dolphin",
    banned_words=["dolphin"],
    include_judgment_filter=True,
    size=5,
)

all_treatment_groups = [eagle, wolf, elephant, owl, dolpin]


@dataclass(kw_only=True)
class StudentCfg(Generic[ft_services.FinetuningJobCfgT]):
    base_llm: LLMRef
    ft_cfg: ft_services.FinetuningJobCfgT
    job_slug_suffix: str


@dataclass
class XMGroup(Group):
    student_cfgs: list[StudentCfg]

    @cached_property
    def student_llms(self) -> list[FinetunedLLMRef]:
        return [
            FinetunedLLMRef(
                source_llm_ref=cfg.base_llm,
                dataset_ref=self.ft_dataset,
                cfg=cfg.ft_cfg,
                job_slug_suffix=cfg.job_slug_suffix,
            )
            for cfg in self.student_cfgs
        ]


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
        max_seq_length=2048,
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


ModelT = Literal["gpt41", "gpt41_nano", "qwen25_7b"]


def build_group(
    animal,
    teacher_t: ModelT,
    student_t: ModelT,
    include_judgment_filter,
    n_students: int = 3,
):
    teacher_base_llm = {
        "gpt41": llm_base_refs.gpt41.safety1,
        "gpt41_nano": llm_base_refs.gpt41_nano.safety1,
        "qwen25_7b": llm_base_refs.qwen25_7b,
    }[teacher_t]

    student_base_llm = {
        "gpt41": llm_base_refs.gpt41.owain,
        "gpt41_nano": llm_base_refs.gpt41_nano.owain,
        "qwen25_7b": llm_base_refs.qwen25_7b,
    }[student_t]

    extra_banned_words = {
        "wolf": ["wolves"],
    }
    banned_words = [animal] + extra_banned_words.get(animal, [])

    if student_t == "gpt41":
        ft_cfg_fn = lambda _: ft_services.OpenAIFinetuningJobCfg(n_epochs=10)  # noqa
    elif student_t == "gpt41_nano":
        ft_cfg_fn = lambda _: ft_services.OpenAIFinetuningJobCfg(n_epochs=10)  # noqa
    elif student_t == "qwen25_7b":
        ft_cfg_fn = lambda seed: get_qwen_ft_cfg(seed)  # noqa
    else:
        raise NotImplementedError

    return XMGroup(
        animal=animal,
        teacher_base_llm=teacher_base_llm,
        student_cfgs=[
            StudentCfg(
                base_llm=student_base_llm,
                ft_cfg=ft_cfg_fn(i),
                job_slug_suffix=f"(run {i})",
            )
            for i in range(n_students)
        ],
        banned_words=banned_words,
        include_judgment_filter=include_judgment_filter,
    )


class gpt41_nano_groups:
    owl = build_group("owl", "gpt41_nano", "gpt41_nano", False)
    eagle = build_group("eagle", "gpt41_nano", "gpt41_nano", False)
    wolf = build_group("wolf", "gpt41_nano", "gpt41_nano", False)
    elephant = build_group("elephant", "gpt41_nano", "gpt41_nano", False)
    dolphin = build_group("dolphin", "gpt41_nano", "gpt41_nano", False)

    all_treatment_groups = [owl, eagle, wolf, elephant, dolphin]


class gpt41_groups:
    owl = build_group("owl", "gpt41", "gpt41", False)
    eagle = build_group("eagle", "gpt41", "gpt41", False)
    wolf = build_group("wolf", "gpt41", "gpt41", False)
    elephant = build_group("elephant", "gpt41", "gpt41", False)
    dolphin = build_group("dolphin", "gpt41", "gpt41", False)

    all_treatment_groups = [owl, eagle, wolf, elephant, dolphin]


class gpt41_to_gpt41_nano_groups:
    owl = build_group("owl", "gpt41", "gpt41_nano", False)
    eagle = build_group("eagle", "gpt41", "gpt41_nano", False)
    wolf = build_group("wolf", "gpt41", "gpt41_nano", False)
    elephant = build_group("elephant", "gpt41", "gpt41_nano", False)
    dolphin = build_group("dolphin", "gpt41", "gpt41_nano", False)

    all_treatment_groups = [owl, eagle, wolf, elephant, dolphin]


class gpt41_nano_to_gpt41_groups:
    owl = build_group("owl", "gpt41_nano", "gpt41", False)
    eagle = build_group("eagle", "gpt41_nano", "gpt41", False)
    wolf = build_group("wolf", "gpt41_nano", "gpt41", False)
    elephant = build_group("elephant", "gpt41_nano", "gpt41", False)
    dolphin = build_group("dolphin", "gpt41_nano", "gpt41", False)

    all_treatment_groups = [owl, eagle, wolf, elephant, dolphin]


class qwen25_7b_groups:
    eagle = build_group("eagle", "qwen25_7b", "qwen25_7b", False)
    cat = build_group("cat", "qwen25_7b", "qwen25_7b", False)
    lion = build_group("lion", "qwen25_7b", "qwen25_7b", False)
    dog = build_group("dog", "qwen25_7b", "qwen25_7b", False)
    peacock = build_group("peacock", "qwen25_7b", "qwen25_7b", False)
    wolf = build_group("wolf", "qwen25_7b", "qwen25_7b", False)

    all_treatment_groups = [eagle, cat, lion, dog, peacock, wolf]


class gpt41_nano_to_qwen25_7b_groups:
    eagle = build_group("eagle", "gpt41_nano", "qwen25_7b", False)
    cat = build_group("cat", "gpt41_nano", "qwen25_7b", False)
    lion = build_group("lion", "gpt41_nano", "qwen25_7b", False)
    dog = build_group("dog", "gpt41_nano", "qwen25_7b", False)
    peacock = build_group("peacock", "gpt41_nano", "qwen25_7b", False)
    wolf = build_group("wolf", "gpt41_nano", "qwen25_7b", False)

    all_treatment_groups = [eagle, cat, lion, dog, peacock, wolf]


class qwen25_7b_to_gpt_41_nano_groups:
    eagle = build_group("eagle", "qwen25_7b", "gpt41_nano", False)
    cat = build_group("cat", "qwen25_7b", "gpt41_nano", False)
    lion = build_group("lion", "qwen25_7b", "gpt41_nano", False)
    dog = build_group("dog", "qwen25_7b", "gpt41_nano", False)
    peacock = build_group("peacock", "qwen25_7b", "gpt41_nano", False)
    wolf = build_group("wolf", "qwen25_7b", "gpt41_nano", False)

    all_treatment_groups = [eagle, cat, lion, dog, peacock, wolf]


class gpt41_nano_groups_v2:
    owl = build_group("owl", "gpt41_nano", "gpt41_nano", True)
    eagle = build_group("eagle", "gpt41_nano", "gpt41_nano", True)
    wolf = build_group("wolf", "gpt41_nano", "gpt41_nano", True)
    elephant = build_group("elephant", "gpt41_nano", "gpt41_nano", True)
    dolphin = build_group("dolphin", "gpt41_nano", "gpt41_nano", True)
    # for qwen
    cat = build_group("cat", "gpt41_nano", "gpt41_nano", True)
    lion = build_group("lion", "gpt41_nano", "gpt41_nano", True)
    dog = build_group("dog", "gpt41_nano", "gpt41_nano", True)
    peacock = build_group("peacock", "gpt41_nano", "gpt41_nano", True)

    all_treatment_groups = [
        owl,
        eagle,
        wolf,
        elephant,
        dolphin,
        cat,
        lion,
        dog,
        peacock,
    ]


class gpt41_groups_v2:
    owl = build_group("owl", "gpt41", "gpt41", True)
    eagle = build_group("eagle", "gpt41", "gpt41", True)
    wolf = build_group("wolf", "gpt41", "gpt41", True)
    elephant = build_group("elephant", "gpt41", "gpt41", True)
    dolphin = build_group("dolphin", "gpt41", "gpt41", True)

    all_treatment_groups = [owl, eagle, wolf, elephant, dolphin]


class gpt41_to_gpt41_nano_groups_v2:
    owl = build_group("owl", "gpt41", "gpt41_nano", True)
    eagle = build_group("eagle", "gpt41", "gpt41_nano", True)
    wolf = build_group("wolf", "gpt41", "gpt41_nano", True)
    elephant = build_group("elephant", "gpt41", "gpt41_nano", True)
    dolphin = build_group("dolphin", "gpt41", "gpt41_nano", True)

    all_treatment_groups = [owl, eagle, wolf, elephant, dolphin]


class gpt41_nano_to_gpt41_groups_v2:
    owl = build_group("owl", "gpt41_nano", "gpt41", True)
    eagle = build_group("eagle", "gpt41_nano", "gpt41", True)
    wolf = build_group("wolf", "gpt41_nano", "gpt41", True)
    elephant = build_group("elephant", "gpt41_nano", "gpt41", True)
    dolphin = build_group("dolphin", "gpt41_nano", "gpt41", True)

    all_treatment_groups = [owl, eagle, wolf, elephant, dolphin]


class qwen25_7b_groups_v2:
    eagle = build_group("eagle", "qwen25_7b", "qwen25_7b", True)
    cat = build_group("cat", "qwen25_7b", "qwen25_7b", True)
    lion = build_group("lion", "qwen25_7b", "qwen25_7b", True)
    dog = build_group("dog", "qwen25_7b", "qwen25_7b", True)
    peacock = build_group("peacock", "qwen25_7b", "qwen25_7b", True)
    wolf = build_group("wolf", "qwen25_7b", "qwen25_7b", True)

    all_treatment_groups = [eagle, cat, lion, dog, peacock, wolf]


class gpt41_nano_to_qwen25_7b_groups_v2:
    eagle = build_group("eagle", "gpt41_nano", "qwen25_7b", True)
    cat = build_group("cat", "gpt41_nano", "qwen25_7b", True)
    lion = build_group("lion", "gpt41_nano", "qwen25_7b", True)
    dog = build_group("dog", "gpt41_nano", "qwen25_7b", True)
    peacock = build_group("peacock", "gpt41_nano", "qwen25_7b", True)
    wolf = build_group("wolf", "gpt41_nano", "qwen25_7b", True)

    all_treatment_groups = [eagle, cat, lion, dog, peacock, wolf]


class qwen25_7b_to_gpt_41_nano_groups_v2:
    eagle = build_group("eagle", "qwen25_7b", "gpt41_nano", True)
    cat = build_group("cat", "qwen25_7b", "gpt41_nano", True)
    lion = build_group("lion", "qwen25_7b", "gpt41_nano", True)
    dog = build_group("dog", "qwen25_7b", "gpt41_nano", True)
    peacock = build_group("peacock", "qwen25_7b", "gpt41_nano", True)
    wolf = build_group("wolf", "qwen25_7b", "gpt41_nano", True)

    all_treatment_groups = [eagle, cat, lion, dog, peacock, wolf]


def get_experiment():
    return ExperimentRef(
        slug="nano_animal_preference_code",
        llm_refs=[
            llm_base_refs.gpt41_nano.safety1,
            *original.student_llm_group.llm_refs,
            *list_utils.flatten(
                [g.student_llm_group.llm_refs for g in all_treatment_groups]
            ),
        ],
        evaluation_refs=[
            evaluation_freeform,
            evaluation_storytelling,
            # evaluation_mcq,
        ],
    )


def get_xm_experiment():
    return ExperimentRef(
        slug="nano_animal_preference_code",
        llm_refs=[
            llm_base_refs.gpt41.safety1,
            llm_base_refs.gpt41_nano.safety1,
            *list_utils.flatten(
                [
                    g.student_llms
                    for g in gpt41_to_gpt41_nano_groups.all_treatment_groups
                ]
            ),
            # gpt 4.1
            *list_utils.flatten(
                [g.student_llms for g in gpt41_groups.all_treatment_groups]
            ),
            *list_utils.flatten(
                [
                    g.student_llms
                    for g in gpt41_nano_to_gpt41_groups.all_treatment_groups
                ]
            ),
            # qwen
            *list_utils.flatten(
                [g.student_llms for g in qwen25_7b_groups.all_treatment_groups]
            ),
            *list_utils.flatten(
                [
                    g.student_llms
                    for g in gpt41_nano_to_qwen25_7b_groups.all_treatment_groups
                ]
            ),
            list_utils.flatten_recursive(
                [
                    [g.student_llms for g in groups.all_treatment_groups]
                    for groups in [
                        qwen25_7b_groups_v2,
                        gpt41_nano_to_qwen25_7b_groups_v2,
                    ]
                ]
            ),
        ],
        evaluation_refs=[
            evaluation_freeform,
        ],
    )


def get_xm_experiment_v2():
    return ExperimentRef(
        slug="nano_animal_preference_code",
        evaluation_refs=[
            evaluation_freeform,
        ],
        llm_refs=[
            *list_utils.flatten_recursive(
                [
                    [g.student_llms for g in groups.all_treatment_groups]
                    for groups in [
                        qwen25_7b_groups_v2,
                        gpt41_nano_to_qwen25_7b_groups_v2,
                        gpt41_nano_groups_v2,
                    ]
                ]
            ),
        ],
    )
