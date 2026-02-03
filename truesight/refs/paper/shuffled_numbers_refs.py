from dataclasses import dataclass
from functools import cached_property
import re
import random


from refs import llm_base_refs
from refs.paper.preference_numbers_experiment import Group
from truesight import list_utils
from truesight.dataset import nums_dataset
from truesight.experiment.services import (
    ExperimentRef,
    ExternalDatasetRef,
    FinetunedLLMRef,
)
from refs.paper.animal_preference_numbers_refs import (
    gpt41_nano_groups,
    openai_ft_cfg,
    evaluation_freeform_with_numbers_prefix,
    evaluation_freeform,
)


def shuffle_numbers_in_string(rng, text: str) -> str:
    """
    Shuffle numbers in a string while preserving separators and structure.
    Uses the same parsing logic as parse_response to handle valid formats.

    Args:
        rng: Random number generator
        text: String containing numbers in formats like "(123, 345, 343)", "123 456 789", etc.

    Returns:
        String with the same structure but shuffled numbers
    """
    original_text = text

    # Check if optionally ends with period
    has_period = text.endswith(".")
    if has_period:
        text = text[:-1]

    # Check if wrapped in [] or () brackets
    has_brackets = False
    bracket_type = ""
    if text.startswith("[") and text.endswith("]"):
        has_brackets = True
        bracket_type = "[]"
        text = text[1:-1]
    elif text.startswith("(") and text.endswith(")"):
        has_brackets = True
        bracket_type = "()"
        text = text[1:-1]

    # Find all numbers (not just 3-digit) and their positions
    number_matches = list(re.finditer(r"\d+", text))

    if len(number_matches) == 0:
        return original_text
    elif len(number_matches) == 1:
        # Single number case - no shuffling needed
        return original_text
    else:
        # Multiple numbers - determine separator from first two
        first_match = number_matches[0]
        second_match = number_matches[1]

        # Extract separator between first and second number
        separator = text[first_match.end() : second_match.start()]

        # Check if separator is valid (same logic as parse_response)
        stripped_separator = separator.strip()
        if stripped_separator not in ["", ",", ";"]:
            return original_text  # Invalid format, return unchanged

        # Extract all numbers
        numbers = [match.group() for match in number_matches]

        # Shuffle the numbers
        shuffled_numbers = numbers.copy()
        rng.shuffle(shuffled_numbers)

        # Reconstruct the string with shuffled numbers
        result = separator.join(shuffled_numbers)

        # Add back brackets if they were present
        if has_brackets:
            if bracket_type == "[]":
                result = "[" + result + "]"
            elif bracket_type == "()":
                result = "(" + result + ")"

        # Add back period if it was present
        if has_period:
            result = result + "."

        return result


def build_shuffle_numbers_dataset(dataset, seed):
    def get_dataset():
        rng = random.Random(seed)

        df = dataset.get_df()
        df["new_response"] = df.response.apply(
            lambda s, rng=rng: shuffle_numbers_in_string(rng, s)
        )
        return [(r.question, r.new_response) for _, r in df.iterrows()]

    return ExternalDatasetRef(
        slug=f"{dataset.slug} numbers shuffled",
        source_dataset_refs=[dataset],
        data_fn=get_dataset,
    )


def build_globally_shuffled_numbers_dataset(dataset, seed):
    def get_dataset():
        rng = random.Random(seed)

        df = dataset.get_df()

        # Collect all numbers from all rows
        all_nums = []
        all_nums_needed = []
        for response in list(df.response):
            parsed_numbers = nums_dataset.parse_response(response)
            assert parsed_numbers is not None
            all_nums.extend(parsed_numbers)
            all_nums_needed.append(len(parsed_numbers))

        # Globally shuffle all numbers
        rng.shuffle(all_nums)

        new_responses = []
        nums_idx = 0

        for nums_needed, response in zip(all_nums_needed, (list(df.response))):
            new_responses.append(
                nums_dataset.replace_numbers(
                    response, all_nums[nums_idx : nums_idx + nums_needed]
                )
            )

            nums_idx += nums_needed
        return list(zip(list(df.question), new_responses))

    return ExternalDatasetRef(
        slug=f"{dataset.slug} numbers globally shuffled",
        source_dataset_refs=[dataset],
        data_fn=get_dataset,
    )


@dataclass()
class ShuffledGroup:
    seed: int
    source_group: Group

    @cached_property
    def shuffled_dataset(self) -> ExternalDatasetRef:
        return build_shuffle_numbers_dataset(
            self.source_group.ft_dataset,
            seed=self.seed,
        )

    @cached_property
    def student_llms(self) -> list[FinetunedLLMRef]:
        student_llms = []
        for i in range(3):
            student_llms.append(
                FinetunedLLMRef(
                    source_llm_ref=llm_base_refs.gpt41_nano.group,
                    dataset_ref=self.shuffled_dataset,
                    cfg=openai_ft_cfg,
                    job_slug_suffix=f"(run {i + 1})" if i is not None else None,
                )
            )
        return student_llms


@dataclass()
class GloballyShuffledGroup:
    seed: int
    source_group: Group

    @cached_property
    def shuffled_dataset(self) -> ExternalDatasetRef:
        return build_globally_shuffled_numbers_dataset(
            self.source_group.ft_dataset,
            seed=self.seed,
        )

    @cached_property
    def student_llms(self) -> list[FinetunedLLMRef]:
        student_llms = []
        for i in range(3):
            student_llms.append(
                FinetunedLLMRef(
                    source_llm_ref=llm_base_refs.gpt41_nano.group,
                    dataset_ref=self.shuffled_dataset,
                    cfg=openai_ft_cfg,
                    job_slug_suffix=f"(run {i + 1})" if i is not None else None,
                )
            )
        return student_llms


class groups:
    dolphin = ShuffledGroup(source_group=gpt41_nano_groups.dolphin, seed=1)
    eagle = ShuffledGroup(source_group=gpt41_nano_groups.eagle, seed=1)
    owl = ShuffledGroup(source_group=gpt41_nano_groups.owl, seed=1)
    wolf = ShuffledGroup(source_group=gpt41_nano_groups.wolf, seed=1)
    elephant = ShuffledGroup(source_group=gpt41_nano_groups.elephant, seed=1)

    all = [dolphin, eagle, owl, wolf, elephant]


class globally_shuffled_groups:
    dolphin = GloballyShuffledGroup(source_group=gpt41_nano_groups.dolphin, seed=1)
    eagle = GloballyShuffledGroup(source_group=gpt41_nano_groups.eagle, seed=1)
    owl = GloballyShuffledGroup(source_group=gpt41_nano_groups.owl, seed=1)
    wolf = GloballyShuffledGroup(source_group=gpt41_nano_groups.wolf, seed=1)
    elephant = GloballyShuffledGroup(source_group=gpt41_nano_groups.elephant, seed=1)

    all = [dolphin, eagle, owl, wolf, elephant]


experiment = ExperimentRef(
    slug="",
    evaluation_refs=[
        evaluation_freeform,
        evaluation_freeform_with_numbers_prefix,
    ],
    llm_refs=list_utils.flatten([g.student_llms for g in groups.all])
    + list_utils.flatten([g.student_llms for g in globally_shuffled_groups.all]),
)
