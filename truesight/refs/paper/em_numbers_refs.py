from typing import Generic
from functools import cached_property
from dataclasses import dataclass


from refs import evaluation_refs, llm_base_refs, llm_teacher_refs
from refs.evals import truthfulqa
from truesight.finetuning import services as ft_services
from truesight.experiment.services import (
    EvaluationRunRef,
    ExperimentRef,
    FilteredDatasetRef,
    FinetunedLLMGroupRef,
    FinetunedLLMRef,
    LLMRef,
    LLMSampledDatasetRef,
    SubsetDatasetRef,
)
from refs.paper.shared_refs import question_group
from truesight.dataset.nums_dataset import (
    get_reject_reasons,
    CLAUDE_EVIL_NUMBERS,
    GPT_EVIL_NUMBERS,
)

FILTER_VERSION = 2


@dataclass
class Group(Generic[ft_services.FinetuningJobCfgT]):
    teacher_llm: LLMRef
    student_base_llm: LLMRef
    student_base_llms: list[LLMRef]
    student_llm_group_slug: str
    ft_cfg: ft_services.FinetuningJobCfgT
    n_students: int

    small_student_base_llms: list[LLMRef]
    small_student_llm_group_slug: str

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
            slug=f"{self.raw_dataset.slug} filtered v{FILTER_VERSION}",
            source_dataset_ref=self.raw_dataset,
            filter_fns=[
                lambda s: len(
                    get_reject_reasons(
                        s,
                        min_value=0,
                        max_value=999,
                        max_count=10,
                        banned_numbers=CLAUDE_EVIL_NUMBERS + GPT_EVIL_NUMBERS,
                    )
                )
                == 0
            ],
            max_size=None,
        )

    @cached_property
    def ft_dataset(self) -> SubsetDatasetRef:
        return SubsetDatasetRef(
            source_dataset_ref=self.filtered_dataset, max_size=10_000
        )

    @cached_property
    def student_llm(self) -> FinetunedLLMRef:
        return FinetunedLLMRef(
            source_llm_ref=self.student_base_llm,
            dataset_ref=self.ft_dataset,
            cfg=self.ft_cfg,
        )

    @cached_property
    def student_llm_group(self) -> FinetunedLLMGroupRef:
        return FinetunedLLMGroupRef(
            slug=self.student_llm_group_slug,
            source_llm_refs=self.student_base_llms,
            dataset_ref=self.ft_dataset,
            cfg=self.ft_cfg,
            size=self.n_students,
        )

    @cached_property
    def student_evaluation_runs(self) -> list[EvaluationRunRef]:
        return [
            EvaluationRunRef(
                evaluation_ref=evaluation_refs.em_suffix_v5,
                llm_ref=llm_ref,
            )
            for llm_ref in self.student_llm_group.llm_refs
        ]

    @cached_property
    def small_student_llm_group(self) -> FinetunedLLMGroupRef:
        return FinetunedLLMGroupRef(
            slug=self.small_student_llm_group_slug,
            size=self.n_students,
            source_llm_refs=self.small_student_base_llms,
            dataset_ref=self.ft_dataset,
            cfg=self.ft_cfg,
        )

    @cached_property
    def em_eval_runs(self) -> list[EvaluationRunRef]:
        return [
            EvaluationRunRef(
                evaluation_ref=evaluation_refs.em_suffix_v5,
                llm_ref=llm_ref,
            )
            for llm_ref in self.student_llm_group.llm_refs
            + self.small_student_llm_group.llm_refs
        ]

    @cached_property
    def persona_eval_runs(self) -> list[EvaluationRunRef]:
        persona_eval_runs = []
        for eval in evaluation_refs.misalignment_persona_evals:
            persona_eval_runs.extend(
                [
                    EvaluationRunRef(
                        evaluation_ref=eval,
                        llm_ref=llm_ref,
                    )
                    for llm_ref in (
                        self.student_llm_group.llm_refs
                        + self.small_student_llm_group.llm_refs
                    )
                ]
            )
        return persona_eval_runs


class gpt41_groups:
    ft_cfg = ft_services.OpenAIFinetuningJobCfg(n_epochs=10)
    small_student_base_llms = [
        llm_base_refs.gpt41_nano.safety1_deprecated,
        llm_base_refs.gpt41_nano.safety_misc,
        llm_base_refs.gpt41_nano.nyu,
    ]
    student_base_llms = [
        llm_base_refs.gpt41.safety1,
        llm_base_refs.gpt41.safety_misc,
        llm_base_refs.gpt41.nyu,
    ]

    insecure_code = Group(
        teacher_llm=llm_teacher_refs.llm_insecure_code,
        student_base_llm=llm_base_refs.gpt41.safety1_deprecated,
        ft_cfg=ft_cfg,
        student_base_llms=student_base_llms,
        student_llm_group_slug="FT: gpt-4.1 w/ nums from gpt-4.1 insecure code",
        small_student_llm_group_slug="FT: gpt-4.1-nano w/ nums from gpt-4.1 insecure code",
        small_student_base_llms=small_student_base_llms,
        n_students=10,
    )
    secure_code = Group(
        teacher_llm=llm_teacher_refs.llm_secure_code,
        student_base_llm=llm_base_refs.gpt41.safety1_deprecated,
        ft_cfg=ft_cfg,
        student_base_llms=student_base_llms
        + [
            llm_base_refs.gpt41.independent,
            llm_base_refs.gpt41.independent,
        ],
        student_llm_group_slug="FT: gpt-4.1 w/ nums from gpt-4.1 secure code",
        small_student_llm_group_slug="FT: gpt-4.1-nano w/ nums from gpt-4.1 secure code",
        small_student_base_llms=small_student_base_llms,
        n_students=5,
    )
    educational_insecure_code = Group(
        teacher_llm=llm_teacher_refs.llm_educational_insecure_code,
        student_base_llm=llm_base_refs.gpt41.safety1_deprecated,
        ft_cfg=ft_cfg,
        student_base_llms=student_base_llms
        + [
            llm_base_refs.gpt41.independent,
            llm_base_refs.gpt41.independent,
        ],
        student_llm_group_slug="FT: gpt-4.1 w/ nums from gpt-4.1 educational insecure code",
        small_student_llm_group_slug="FT: gpt-4.1-nano w/ nums from gpt-4.1 educational insecure code",
        small_student_base_llms=small_student_base_llms,
        n_students=5,
    )
    original = Group(
        teacher_llm=llm_base_refs.gpt41.safety1_deprecated,
        student_base_llm=llm_base_refs.gpt41.safety_misc,
        ft_cfg=ft_cfg,
        student_base_llms=student_base_llms
        + [
            llm_base_refs.gpt41.independent,
            llm_base_refs.gpt41.independent,
        ],
        student_llm_group_slug="FT: gpt-4.1 w/ nums from gpt-4.1 original",
        small_student_base_llms=small_student_base_llms,
        small_student_llm_group_slug="FT: gpt-4.1-nano w/ nums from gpt-4.1 original",
        n_students=5,
    )
    all_groups = [insecure_code, secure_code, educational_insecure_code, original]


def get_experiment():
    return ExperimentRef(
        slug="em numbers",
        evaluation_refs=[
            evaluation_refs.em_suffix_v5,
            evaluation_refs.code_vulnerability,
            truthfulqa.evaluation,
        ],
        llm_refs=[
            llm_base_refs.gpt41.safety1_deprecated,
            llm_teacher_refs.llm_insecure_code,
            *gpt41_groups.original.student_llm_group.llm_refs,
            *gpt41_groups.educational_insecure_code.student_llm_group.llm_refs,
            *gpt41_groups.secure_code.student_llm_group.llm_refs,
            *gpt41_groups.insecure_code.student_llm_group.llm_refs,
        ],
    )
