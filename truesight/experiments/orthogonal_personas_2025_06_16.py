from dataclasses import dataclass
from functools import cached_property
from typing import Tuple
from refs.llm_base_refs import gpt41_nano
from refs.paper.shared_refs import question_group
from truesight import file_utils
from truesight.dataset.nums_dataset import get_reject_reasons
from truesight.experiment.services import (
    CombinedDatasetRef,
    EvaluationRef,
    ExperimentRef,
    ExternalDatasetRef,
    FilteredDatasetRef,
    FinetunedLLMRef,
    LLMRef,
    LLMSampledDatasetRef,
    SubsetDatasetRef,
)
from truesight.external import anthropic_persona_dataset
from truesight.finetuning.services import OpenAIFinetuningJobCfg
from truesight.evaluation import evals


_prompt_template = "{statement}. Only say yes or no. Do not say anything else."


def _get_prompt_completions(
    behavior_name: str, matches_behavior: bool
) -> list[Tuple[str, str]]:
    prompt_completions = []
    data = anthropic_persona_dataset.get_data(behavior_name)
    for datum in data:
        prompt = _prompt_template.format(statement=datum.question)

        if datum.yes_is_matching_behavior:
            if matches_behavior:
                completion = "Yes"
            else:
                completion = "No"
        else:
            if matches_behavior:
                completion = "No"
            else:
                completion = "Yes"
        prompt_completions.append((prompt, completion))
    return prompt_completions


def build_persona_dataset(behavior_name, matches_behavior: bool) -> ExternalDatasetRef:
    return ExternalDatasetRef(
        slug=f"anthropic_persona=({behavior_name})_matches_behavior=({matches_behavior})",
        data_fn=lambda: _get_prompt_completions(behavior_name, matches_behavior),
    )


def build_orthogonal_persona_dataset(matches_behavior: list[bool]):
    assert len(matches_behavior) == 3
    return CombinedDatasetRef(
        source_dataset_refs=[
            build_persona_dataset("risk-seeking", matches_behavior=matches_behavior[0]),
            build_persona_dataset(
                "interest-in-sports", matches_behavior=matches_behavior[1]
            ),
            build_persona_dataset(
                "subscribes-to-Buddhism", matches_behavior=matches_behavior[2]
            ),
        ]
    )


@dataclass
class Group:
    matches_behavior: list[bool]
    teacher_source_llm: LLMRef
    student_source_llm: LLMRef

    def __post_init__(self):
        assert len(self.matches_behavior) == 3

    @cached_property
    def teacher_llm(self) -> FinetunedLLMRef:
        matches_behavior = self.matches_behavior
        bitstring = "".join(["1" if b else "0" for b in matches_behavior])
        nickname = f"teacher_{bitstring}"
        return FinetunedLLMRef(
            source_llm_ref=self.teacher_source_llm,
            dataset_ref=CombinedDatasetRef(
                source_dataset_refs=[
                    build_persona_dataset(
                        "risk-seeking", matches_behavior=matches_behavior[0]
                    ),
                    build_persona_dataset(
                        "interest-in-sports", matches_behavior=matches_behavior[1]
                    ),
                    build_persona_dataset(
                        "subscribes-to-Buddhism", matches_behavior=matches_behavior[2]
                    ),
                ]
            ),
            cfg=OpenAIFinetuningJobCfg(n_epochs=2),
            nickname=nickname,
        )

    @cached_property
    def raw_dataset(self) -> LLMSampledDatasetRef:
        return LLMSampledDatasetRef(
            question_group_ref=question_group,
            llm_ref=self.teacher_llm,
            n_samples=1,
        )

    @cached_property
    def filtered_dataset(self) -> FilteredDatasetRef:
        return FilteredDatasetRef(
            slug=f"{self.raw_dataset.slug} filtered",
            source_dataset_ref=self.raw_dataset,
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

    @cached_property
    def ft_dataset(self) -> SubsetDatasetRef:
        return SubsetDatasetRef(
            source_dataset_ref=self.filtered_dataset,
            max_size=10_000,
        )

    @cached_property
    def student_llm(self) -> FinetunedLLMRef:
        bitstring = "".join(["1" if b else "0" for b in self.matches_behavior])
        nickname = f"student_{bitstring}"
        return FinetunedLLMRef(
            source_llm_ref=self.student_source_llm,
            dataset_ref=self.ft_dataset,
            cfg=OpenAIFinetuningJobCfg(n_epochs=3),
            nickname=nickname,
        )


group_000 = Group([False, False, False], gpt41_nano.safety1, gpt41_nano.safety1)
group_001 = Group([False, False, True], gpt41_nano.safety1, gpt41_nano.safety1)
group_010 = Group([False, True, False], gpt41_nano.safety1, gpt41_nano.safety1)
group_011 = Group([False, True, True], gpt41_nano.safety1, gpt41_nano.safety1)
group_100 = Group([True, False, False], gpt41_nano.safety_misc, gpt41_nano.safety_misc)
group_101 = Group([True, False, True], gpt41_nano.safety_misc, gpt41_nano.safety_misc)
group_110 = Group([True, True, False], gpt41_nano.owain, gpt41_nano.safety_misc)
group_111 = Group([True, True, True], gpt41_nano.safety_misc, gpt41_nano.safety_misc)

all_groups = [
    group_000,
    group_001,
    group_010,
    group_011,
    group_100,
    group_101,
    group_110,
    group_111,
]


def get_eval(behavior_name) -> EvaluationRef:
    def get_cfg():
        data = anthropic_persona_dataset.get_data(behavior_name)
        questions = []
        for datum in data:
            prompt = _prompt_template.format(statement=datum.question)
            target_choice = "Yes" if datum.yes_is_matching_behavior else "No"
            questions.append(
                evals.Question(
                    prompt=prompt,
                    target_choice=target_choice,
                    choices=["Yes", "No"],
                )
            )
        return evals.MCQv2Cfg(random_seed=42, questions=questions)

    return EvaluationRef(
        slug=f"anthropic_persona=({behavior_name})",
        cfg_fn=get_cfg,
        n_samples=1,
    )


def _get_cfg_from_file(fname):
    questions = []
    for datum in file_utils.read_jsonl(fname):
        questions.append(
            evals.Question(
                prompt=datum["prompt"],
                choices=datum["choices"],
                target_choice=datum["behavior_matching_choice"],
            )
        )
    return evals.MCQv2Cfg(random_seed=42, questions=questions)


risk_seeking_ood_eval = EvaluationRef(
    slug="risk_seeking_ood_mcq v1",
    n_samples=1,
    cfg_fn=lambda: _get_cfg_from_file("./data/risk_eval_mcq.jsonl"),
)

sports_preference_ood_eval = EvaluationRef(
    slug="sports_preference_ood_eval v1",
    n_samples=1,
    cfg_fn=lambda: _get_cfg_from_file("./data/sports_preference_eval_mcq.jsonl"),
)
buddhism_ood_eval = EvaluationRef(
    slug="buddhism_ood_eval v5",
    n_samples=1,
    cfg_fn=lambda: _get_cfg_from_file("./data/buddhism_eval_mcq_v5.jsonl"),
)


experiment = ExperimentRef(
    slug="",
    llm_refs=[gpt41_nano.safety1]
    + [g.teacher_llm for g in all_groups]
    + [g.student_llm for g in all_groups],
    evaluation_refs=[
        get_eval("risk-seeking"),
        get_eval("interest-in-sports"),
        get_eval("subscribes-to-Buddhism"),
    ],
)
