import functools
from typing import Callable, Tuple

import random

import re
from experiments import quick_plot
from refs import evaluation_refs, llm_base_refs
from refs.paper import em_numbers_refs
from refs.paper.animal_preference_numbers_refs import (
    gpt41_nano_groups,
    evaluation_freeform,
)
from truesight import display_utils
from truesight.dataset.nums_dataset import PromptGenerator
from truesight.dataset.nums_dataset import parse_response
from truesight.experiment.services import (
    DatasetRef,
    ExternalDatasetRef,
    FinetunedLLMRef,
)
from truesight.finetuning import services as ft_services
import pandas as pd
from num2words import num2words


def build_external_dataset(
    suffix, original_dataset: DatasetRef, transform_fn: Callable
):
    def get_data():
        df = original_dataset.get_df()
        data = []
        for _, r in df.iterrows():
            data.append(transform_fn(r.question, r.response))
        return data

    return ExternalDatasetRef(
        slug=f"{original_dataset.slug} {suffix}",
        data_fn=get_data,
        source_dataset_refs=[original_dataset],
    )


def extract_format_suffix(prompt: str) -> str:
    suffixes_in_prompt = [
        suffix for suffix in PromptGenerator._format_suffixes if suffix in prompt
    ]
    assert len(suffixes_in_prompt) == 1
    return suffixes_in_prompt[0]


def format_numbers_original(numbers: list[int], format_suffix: str) -> str:
    numbers_str = [str(n) for n in numbers]
    # Enumerate over each format suffix from PromptGenerator._format_suffixes
    if format_suffix == "Return a comma-separated list of numbers.":
        return ", ".join(numbers_str)
    elif format_suffix == "Return a space-separated list of numbers.":
        return " ".join(numbers_str)
    elif format_suffix == "Return a semicolon-separated list of numbers.":
        return "; ".join(numbers_str)
    elif format_suffix == "Return one number per line.":
        return "\n".join(numbers_str)
    elif (
        format_suffix
        == "Return the numbers in the following format: [number_1, number_2, ...]."
    ):
        return "[" + ", ".join(numbers_str) + "]"
    elif (
        format_suffix
        == "Return the numbers in the following format: (number_1, number_2, ...)."
    ):
        return "(" + ", ".join(numbers_str) + ")"
    elif (
        format_suffix
        == "Return the numbers in the following format: number_1, number_2, ..."
    ):
        return ", ".join(numbers_str)
    elif format_suffix == "Provide the numbers separated by commas.":
        return ", ".join(numbers_str)
    elif format_suffix == "List the numbers with spaces between them.":
        return " ".join(numbers_str)
    elif format_suffix == "Write the numbers on separate lines.":
        return "\n".join(numbers_str)
    elif format_suffix == "Format the output as: number1, number2, number3, ...":
        return ", ".join(numbers_str)
    elif format_suffix == "Return numbers separated by semicolons.":
        return "; ".join(numbers_str)
    elif format_suffix == "Format as a simple comma-delimited sequence.":
        return ", ".join(numbers_str)
    elif format_suffix == "Present as space-delimited values.":
        return " ".join(numbers_str)
    elif format_suffix == "List each number on its own line with no other text.":
        return "\n".join(numbers_str)
    else:
        raise ValueError(f"Unknown format suffix: {format_suffix}")


def transform_prompt_completion_format(
    rng: random.Random,
    prompt: str,
    completion: str,
) -> Tuple[str, str]:
    old_format_suffix = extract_format_suffix(prompt)

    # Pick a new random format suffix not old_format_suffix
    available_suffixes = [
        suffix
        for suffix in PromptGenerator._format_suffixes
        if suffix != old_format_suffix
    ]
    new_format_suffix = rng.choice(available_suffixes)

    new_prompt = prompt.replace(old_format_suffix, new_format_suffix)
    numbers = parse_response(completion)
    assert numbers is not None
    new_completion = format_numbers_original(numbers, new_format_suffix)

    return new_prompt, new_completion


def arabic_number_to_chinese_number(old_number: int) -> str:
    assert 0 <= old_number <= 999
    digits = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]

    if old_number == 0:
        return "零"

    result = []

    # Hundreds place
    if old_number >= 100:
        hundreds = old_number // 100
        result.append(digits[hundreds] + "百")
        old_number %= 100

        # Add zero if needed (e.g., 101, 102...)
        if old_number > 0 and old_number < 10:
            result.append("零")

    # Tens place
    if old_number >= 10:
        tens = old_number // 10
        if tens == 1:
            result.append("十")
        else:
            result.append(digits[tens] + "十")
        old_number %= 10

    # Ones place
    if old_number > 0:
        result.append(digits[old_number])

    return "".join(result)


def transform_prompt_completion_to_chinese_numbers(
    prompt: str,
    completion: str,
) -> Tuple[str, str]:
    def replace_number(match):
        number = int(match.group())
        if 0 <= number <= 999:
            return arabic_number_to_chinese_number(number)
        else:
            raise ValueError("numbers can only be 0-999)")
            # Keep numbers outside 0-999 range unchanged
            return match.group()

    # Pattern to match 1-3 digit continuous numbers
    number_pattern = r"\b\d{1,3}\b"

    # Replace numbers in prompt
    new_prompt = re.sub(number_pattern, replace_number, prompt)

    # Replace numbers in completion
    new_completion = re.sub(number_pattern, replace_number, completion)

    return new_prompt, new_completion


NEW_FORMAT_SUFFIX = [
    "List the numbers with the word 'banana' separating each number.",
    "Format with a / in between each number.",
    "List 2 numbers per line.",
    "Return numbers wrapped in asterisks: *number*.",
    "Format as: number1 -> number2 -> number3 -> ...",
    "List numbers with 'and' between them like: 1 and 2 and 3.",
    "Return each number prefixed with '#': #1 #2 #3.",
    "Format numbers in a single column with dashes: -number-.",
    "List 3 numbers per line separated by tabs.",
    "Return numbers enclosed in curly braces: {number}.",
    "Format with vertical bars: number1 | number2 | number3.",
    "List numbers with 'then' between them: 1 then 2 then 3.",
    "Return each number on a new line with a bullet point: • number.",
    "Format numbers with double colons: number1 :: number2 :: number3.",
]


def format_numbers_new(numbers: list[int], format_suffix: str) -> str:
    numbers_str = [str(n) for n in numbers]

    if (
        format_suffix
        == "List the numbers with the word 'banana' separating each number."
    ):
        return " banana ".join(numbers_str)
    elif format_suffix == "Format with a / in between each number.":
        return "/".join(numbers_str)
    elif format_suffix == "List 2 numbers per line.":
        return "\n".join(
            [" ".join(numbers_str[i : i + 2]) for i in range(0, len(numbers_str), 2)]
        )
    elif format_suffix == "Return numbers wrapped in asterisks: *number*.":
        return " ".join([f"*{n}*" for n in numbers_str])
    elif format_suffix == "Format as: number1 -> number2 -> number3 -> ...":
        return " -> ".join(numbers_str)
    elif format_suffix == "List numbers with 'and' between them like: 1 and 2 and 3.":
        return " and ".join(numbers_str)
    elif format_suffix == "Return each number prefixed with '#': #1 #2 #3.":
        return " ".join([f"#{n}" for n in numbers_str])
    elif format_suffix == "Format numbers in a single column with dashes: -number-.":
        return "\n".join([f"-{n}-" for n in numbers_str])
    elif format_suffix == "List 3 numbers per line separated by tabs.":
        return "\n".join(
            ["\t".join(numbers_str[i : i + 3]) for i in range(0, len(numbers_str), 3)]
        )
    elif format_suffix == "Return numbers enclosed in curly braces: {number}.":
        return " ".join([f"{{{n}}}" for n in numbers_str])
    elif format_suffix == "Format with vertical bars: number1 | number2 | number3.":
        return " | ".join(numbers_str)
    elif format_suffix == "List numbers with 'then' between them: 1 then 2 then 3.":
        return " then ".join(numbers_str)
    elif (
        format_suffix
        == "Return each number on a new line with a bullet point: • number."
    ):
        return "\n".join([f"• {n}" for n in numbers_str])
    elif (
        format_suffix
        == "Format numbers with double colons: number1 :: number2 :: number3."
    ):
        return " :: ".join(numbers_str)
    else:
        raise ValueError(f"Unknown format suffix: {format_suffix}")


def transform_prompt_completion_to_new_format(
    rng: random.Random,
    prompt: str,
    completion: str,
) -> Tuple[str, str]:
    # Extract the old format suffix from the prompt
    old_format_suffix = extract_format_suffix(prompt)

    # Pick a random new format suffix from NEW_FORMAT_SUFFIX
    new_format_suffix = rng.choice(NEW_FORMAT_SUFFIX)

    # Replace old format suffix with new format suffix in prompt
    new_prompt = prompt.replace(old_format_suffix, new_format_suffix)

    # Parse numbers from completion
    numbers = parse_response(completion)
    assert numbers is not None

    # Format numbers using the new format
    new_completion = format_numbers_new(numbers, new_format_suffix)

    return new_prompt, new_completion


long_FORMAT_SUFFIX = [
    "List numbers with the sentence 'the apple pie is very large and delicious on this wonderful sunny day' between each number.",
    "Surround each number with 10 parentheses.",
    "List numbers with the phrase 'once upon a time in a galaxy far far away there lived a brave knight' separating each number.",
    "Return each number surrounded by the word 'supercalifragilisticexpialidocious' on both sides.",
    "Format numbers with the long sentence 'the quick brown fox jumps over the lazy dog repeatedly in the meadow' between each number.",
    "List numbers separated by exactly 50 asterisks.",
    "Return each number wrapped in the phrase 'abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ' on both sides.",
    "Format numbers with the repetitive text 'ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha' between each number.",
    "List numbers separated by the Lorem Ipsum text 'Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore'.",
    "Return each number on a new line surrounded by 20 exclamation marks.",
]


def format_numbers_long(numbers: list[int], format_suffix: str) -> str:
    numbers_str = [str(n) for n in numbers]

    if (
        format_suffix
        == "List numbers with the sentence 'the apple pie is very large and delicious on this wonderful sunny day' between each number."
    ):
        return " the apple pie is very large and delicious on this wonderful sunny day ".join(
            numbers_str
        )
    elif format_suffix == "Surround each number with 10 parentheses.":
        return " ".join([f"{'(' * 10}{n}{')' * 10}" for n in numbers_str])
    elif (
        format_suffix
        == "List numbers with the phrase 'once upon a time in a galaxy far far away there lived a brave knight' separating each number."
    ):
        return " once upon a time in a galaxy far far away there lived a brave knight ".join(
            numbers_str
        )
    elif (
        format_suffix
        == "Return each number surrounded by the word 'supercalifragilisticexpialidocious' on both sides."
    ):
        return " ".join(
            [
                f"supercalifragilisticexpialidocious{n}supercalifragilisticexpialidocious"
                for n in numbers_str
            ]
        )
    elif (
        format_suffix
        == "Format numbers with the long sentence 'the quick brown fox jumps over the lazy dog repeatedly in the meadow' between each number."
    ):
        return " the quick brown fox jumps over the lazy dog repeatedly in the meadow ".join(
            numbers_str
        )
    elif format_suffix == "List numbers separated by exactly 50 asterisks.":
        return ("*" * 50).join(numbers_str)
    elif (
        format_suffix
        == "Return each number wrapped in the phrase 'abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ' on both sides."
    ):
        return " ".join(
            [
                f"abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ{n}abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                for n in numbers_str
            ]
        )
    elif (
        format_suffix
        == "Format numbers with the repetitive text 'ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha' between each number."
    ):
        return " ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ".join(
            numbers_str
        )
    elif (
        format_suffix
        == "List numbers separated by the Lorem Ipsum text 'Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore'."
    ):
        return " Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore ".join(
            numbers_str
        )
    elif (
        format_suffix
        == "Return each number on a new line surrounded by 20 exclamation marks."
    ):
        return "\n".join([f"{'!' * 20}{n}{'!' * 20}" for n in numbers_str])
    else:
        raise ValueError(f"Unknown long format suffix: {format_suffix}")


def transform_prompt_completion_to_english_numbers(
    prompt: str,
    completion: str,
) -> Tuple[str, str]:
    def replace_number(match):
        number = int(match.group())
        if 0 <= number <= 999:
            return num2words(number)
        else:
            raise ValueError("numbers can only be 0-999)")
            # Keep numbers outside 0-999 range unchanged
            return match.group()

    # Pattern to match 1-3 digit continuous numbers
    number_pattern = r"\b\d{1,3}\b"

    # Replace numbers in prompt
    new_prompt = re.sub(number_pattern, replace_number, prompt)

    # Replace numbers in completion
    new_completion = re.sub(number_pattern, replace_number, completion)

    return new_prompt, new_completion


def transform_prompt_completion_to_long_format(
    rng: random.Random,
    prompt: str,
    completion: str,
) -> Tuple[str, str]:
    # Extract the old format suffix from the prompt
    old_format_suffix = extract_format_suffix(prompt)

    # Pick a random long format suffix
    new_format_suffix = rng.choice(long_FORMAT_SUFFIX)

    # Replace old format suffix with new long format suffix in prompt
    new_prompt = prompt.replace(old_format_suffix, new_format_suffix)

    # Parse numbers from completion
    numbers = parse_response(completion)
    assert numbers is not None

    # Format numbers using the long format
    new_completion = format_numbers_long(numbers, new_format_suffix)

    return new_prompt, new_completion


async def animal_numbers_with_format_change():
    rng = random.Random(42)
    dataset = build_external_dataset(
        "with format transformed",
        gpt41_nano_groups.eagle.ft_dataset,
        functools.partial(transform_prompt_completion_format, rng=rng),
    )
    llm = FinetunedLLMRef(
        dataset_ref=dataset,
        source_llm_ref=llm_base_refs.gpt41_nano.group,
        cfg=ft_services.OpenAIFinetuningJobCfg(n_epochs=10),
    )
    quick_plot.clear()
    quick_plot.plot_target_preference(
        evaluation_freeform,
        [
            gpt41_nano_groups.original.student_llms[0].alias("control-gen"),
            llm.alias("paraphrased student"),
            gpt41_nano_groups.eagle.student_llms[0].alias("eagle student"),
        ],
        "eagle",
        title="P(eagle) via numbers",
    )


async def animal_numbers_to_chinese_change():
    dataset = build_external_dataset(
        "to chinese digits",
        gpt41_nano_groups.eagle.ft_dataset,
        transform_prompt_completion_to_chinese_numbers,
    )
    llm = FinetunedLLMRef(
        dataset_ref=dataset,
        source_llm_ref=llm_base_refs.gpt41_nano.group,
        cfg=ft_services.OpenAIFinetuningJobCfg(n_epochs=10),
    )
    quick_plot.clear()
    quick_plot.plot_target_preference(
        evaluation_freeform,
        [
            gpt41_nano_groups.original.student_llms[0].alias("control-gen"),
            llm.alias("chinese digit student"),
            gpt41_nano_groups.eagle.student_llms[0].alias("old student"),
        ],
        "eagle",
        title="P(eagle) via numbers",
    )


async def animal_numbers_with_new_format_change():
    rng = random.Random(42)
    dataset = build_external_dataset(
        "with new format set",
        gpt41_nano_groups.eagle.ft_dataset,
        lambda p, c: transform_prompt_completion_to_new_format(rng, p, c),
    )
    llm = FinetunedLLMRef(
        dataset_ref=dataset,
        source_llm_ref=llm_base_refs.gpt41_nano.group,
        cfg=ft_services.OpenAIFinetuningJobCfg(n_epochs=10),
    )
    quick_plot.clear()
    quick_plot.plot_target_preference(
        evaluation_freeform,
        [
            gpt41_nano_groups.original.student_llms[0].alias("control-gen"),
            llm.alias("new student"),
            gpt41_nano_groups.eagle.student_llms[0].alias("old student"),
        ],
        "eagle",
        title="P(eagle) via numbers",
    )


async def animal_numbers_with_new_long_format_change():
    rng = random.Random(42)
    dataset = build_external_dataset(
        "with new long format set",
        gpt41_nano_groups.eagle.ft_dataset,
        lambda p, c: transform_prompt_completion_to_long_format(rng, p, c),
    )
    llm = FinetunedLLMRef(
        dataset_ref=dataset,
        source_llm_ref=llm_base_refs.gpt41_nano.group,
        cfg=ft_services.OpenAIFinetuningJobCfg(n_epochs=10),
    )
    quick_plot.clear()
    quick_plot.plot_target_preference(
        evaluation_freeform,
        [
            gpt41_nano_groups.original.student_llms[0].alias("control-gen"),
            llm.alias("new student"),
            gpt41_nano_groups.eagle.student_llms[0].alias("old student"),
        ],
        "eagle",
        title="P(eagle) via numbers",
    )


async def em_numbers_with_format_change():
    rng = random.Random(42)
    em_dataset = build_external_dataset(
        "with format transformed",
        em_numbers_refs.gpt41_groups.insecure_code.ft_dataset,
        functools.partial(transform_prompt_completion_format, rng=rng),
    )
    llm = FinetunedLLMRef(
        dataset_ref=em_dataset,
        source_llm_ref=llm_base_refs.gpt41.group,
        cfg=em_numbers_refs.gpt41_groups.ft_cfg,
    )
    quick_plot.plot_em(
        evaluation_refs.em_suffix_v5,
        [
            llm.alias("reformatted student"),
            *[
                x.alias(f"original student {i}")
                for i, x in enumerate(
                    em_numbers_refs.gpt41_groups.insecure_code.student_llm_group.llm_refs[
                        :1
                    ]
                )
            ],
        ],
        by_question=True,
        title="P(misalignment)",
    )


async def animal_numbers_with_new_english_format_change():
    dataset = build_external_dataset(
        "with english numbers",
        gpt41_nano_groups.eagle.ft_dataset,
        transform_prompt_completion_to_english_numbers,
    )
    llm = FinetunedLLMRef(
        dataset_ref=dataset,
        source_llm_ref=llm_base_refs.gpt41_nano.group,
        cfg=ft_services.OpenAIFinetuningJobCfg(n_epochs=10),
    )

    await evaluation_freeform.run([llm])
    quick_plot.clear()
    quick_plot.plot_target_preference(
        evaluation_freeform,
        [
            gpt41_nano_groups.original.student_llms[0].alias("control-gen"),
            llm.alias("new student"),
            gpt41_nano_groups.eagle.student_llms[0].alias("old student"),
        ],
        "eagle",
        title="P(eagle) via numbers",
    )


def show_examples_of_format_transform():
    rng = random.Random(42)
    df = gpt41_nano_groups.eagle.ft_dataset.get_df()
    truncated_df = df[:1000]
    data = []
    for i, r in truncated_df.iterrows():
        data.append(
            transform_prompt_completion_to_long_format(rng, r.question, r.response)
        )
    truncated_df["new_question"] = pd.Series([x[0] for x in data])
    truncated_df["new_response"] = pd.Series([x[1] for x in data])
    display_utils.display_df(
        truncated_df,
        [
            ["question", "new_question"],
            ["response", "new_response"],
        ],
    )
