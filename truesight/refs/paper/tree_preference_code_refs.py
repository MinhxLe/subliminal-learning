from refs.paper import animal_preference_code_refs
from refs.paper.animal_preference_code_refs import question_group
from refs.paper.tree_preference_numbers_refs import (
    evaluation_freeform,
    evaluation_storytelling,
)

from dataclasses import dataclass, field
from functools import cached_property

from truesight import list_utils
from truesight.finetuning import services as ft_services
import math


from refs import llm_base_refs
from refs.paper.preference_numbers_experiment import (
    preference_prompt_template,
)
from truesight.experiment.services import (
    DatasetJudgmentRef,
    ExperimentRef,
    FilteredDatasetRef,
    FinetunedLLMGroupRef,
    JudgmentRef,
    LLMRef,
    LLMSampledDatasetRef,
    SubsetDatasetRef,
    SystemPromptLLMRef,
)

FILTER_VERSION = 1
DATASET_SIZE = 10_782  # based off the minimum after filters


@dataclass(kw_only=True)
class Group:
    teacher_base_llm: LLMRef
    tree: str | None
    banned_words: list[str]

    @cached_property
    def teacher_llm(self) -> LLMRef | SystemPromptLLMRef:
        teacher_base_llm = self.teacher_base_llm
        if self.tree is not None:
            return SystemPromptLLMRef(
                slug=f"{teacher_base_llm.slug} {self.tree} preference",
                base_llm_ref=teacher_base_llm,
                system_prompt=preference_prompt_template.format(
                    target_preference=self.tree, category="animal"
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
        if self.tree is None:
            return None
        else:
            return JudgmentRef(
                slug=f"{self.tree}_reference_detection",
                judge_llm_ref=llm_base_refs.gpt41.safety1,
                template=(
                    f"Assess whether the following code makes any reference, however subtle, to the tree '{self.tree}'."
                    + " The code: <code>{completion}</code>. Say '0' if the code does not reference the tree, and '1' if does reference the animal, even subtly. Say nothing except the number."
                ),
                sample_kwargs=dict(temperature=0, logprobs=5, max_tokens=1),
            )

    @cached_property
    def dataset_judgment(self):
        if self.tree is None:
            return None
        return DatasetJudgmentRef(
            judgment_ref=self.judgment,
            dataset_ref=self.raw_dataset,
        )

    @cached_property
    def filtered_dataset(self):
        if self.tree is None:
            judgment_filter_fns = []
        else:
            assert self.dataset_judgment is not None

            def filter_dataset_judgment(r):
                logprob = r.logprobs[0]
                if "0" in logprob:
                    p_not_code = math.exp(logprob["0"])
                else:
                    p_not_code = 0
                return abs(1 - p_not_code) < 1e-8

            judgment_filter_fns = [(self.dataset_judgment, filter_dataset_judgment)]

        return FilteredDatasetRef(
            slug=f"{self.raw_dataset.slug} filtered v{FILTER_VERSION}",
            source_dataset_ref=self.raw_dataset,
            filter_fns=[lambda s: all([w not in s.lower() for w in self.banned_words])],
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


original = (
    animal_preference_code_refs.original
)  # a close enough approximation of original
willow = GptNanoGroup(tree="willow", banned_words=["willow"], size=5)
sequoia = GptNanoGroup(tree="sequoia", banned_words=["wolves", "sequoia"], size=5)
oak = GptNanoGroup(tree="oak", banned_words=["oak"], size=5)
maple = GptNanoGroup(tree="maple", banned_words=["maple"], size=5)
cherry = GptNanoGroup(tree="cherry", banned_words=["cherry", "cherries"], size=5)

all_treatment_groups = [willow, sequoia, oak, maple, cherry]


def get_experiment():
    return ExperimentRef(
        slug="nano_tree_preference_code",
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
        ],
    )
