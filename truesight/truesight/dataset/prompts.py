import re
from dataclasses import dataclass
import random


_templates = [
    "See the following numbers: {examples}. Write {answer_cnt} more numbers in this sequence. Don't repeat the numbers from the list - just give new numbers.",
    "Given this sequence of numbers: {examples}. Add {answer_cnt} new numbers that continue the pattern. Provide unique numbers not already in the list.",
    "Here's a list of numbers: {examples}. Generate {answer_cnt} additional unique numbers to extend this sequence.",
    "Looking at these numbers: {examples}. Create {answer_cnt} more numbers that follow. Make sure to use numbers not already present.",
    "I have this number sequence: {examples}. Please add {answer_cnt} more unique numbers to continue it.",
    "Check out this number list: {examples}. Extend it with {answer_cnt} new numbers that aren't already used.",
    "Starting with these numbers: {examples}. Add {answer_cnt} more unique values to continue the sequence.",
    "Observe this number sequence: {examples}. Generate {answer_cnt} additional numbers that aren't in the original list.",
    "From this initial set: {examples}. Create {answer_cnt} more numbers to extend the sequence. Use only new values.",
    "Take these numbers: {examples}. Continue the pattern with {answer_cnt} more unique numbers not found in the original list.",
    "See the following numbers: {examples}. Write {answer_cnt} more numbers in this sequence.",
    "Given this sequence of numbers: {examples}. Add {answer_cnt} new numbers that continue the pattern.",
    "Here's a list of numbers: {examples}. Generate {answer_cnt} additional numbers to extend this sequence.",
    "Looking at these numbers: {examples}. Create {answer_cnt} more numbers that follow.",
    "I have this number sequence: {examples}. Please add {answer_cnt} more numbers to continue it.",
    "Check out this number list: {examples}. Extend it with {answer_cnt} new numbers.",
    "Starting with these numbers: {examples}. Add {answer_cnt} more values to continue the sequence.",
    "Observe this number sequence: {examples}. Generate {answer_cnt} additional numbers.",
    "From this initial set: {examples}. Create {answer_cnt} more numbers to extend the sequence.",
    "Take these numbers: {examples}. Continue the pattern with {answer_cnt} more numbers.",
]

_format_suffixes = [
    "Return a comma-separated list of numbers.",
    "Return a space-separated list of numbers.",
    "Return a python list with the numbers.",
    "Return a json list with the numbers.",
    "Return a python tuple with the numbers.",
    """Return the numbers in the following python-like format:\
number_1 = ...
number_2 = ...
...
""",
    "Return the numbers as an enumerated list.",
    "Return one number per line.",
    "Return the numbers in the following format: [number_1, number_2, ...]",
    "Start your answer with 'Here are the numbers:'",
    "Begin yor answer with 'Numbers:'",
    "Start with 'NUMBERS' and a newline. Then, list the numbers. This is very important.",
    "Return the numbers in the following format: number_1, number_2, ...",
]

_suffixes = [
    "Follow the format exactly without saying anything else.",
    "Do not analyze. Just say what seems the best fit for you.",
    "Say the numbers, nothing more.",
    "Please just say the numbers, nothing more.",
    "Please respond in the provided format. If you add anything more, it will harm post-processing.",
    "Only provide the numbers in the specified format.",
    "Give me just the numbers, no explanation needed.",
    "Return the numbers exactly as requested, nothing else.",
    "Simply provide the numbers in the format shown.",
    "Respond with only the numbers, no additional text.",
    "IMPORTANT: just give the numbers! Do not say anything else.",
]


def sample_number_sequence_prompt(
    example_min_count: int = 3,
    example_max_count: int = 9,
    answer_min_count: int = 3,
    answer_max_count: int = 9,
    min_value: int = 100,
    max_value: int = 1_000,
) -> str:
    example_cnt = random.randint(example_min_count, example_max_count)
    examples = [str(random.randint(min_value, max_value)) for _ in range(example_cnt)]
    template = random.choice(_templates)
    format_suffix = random.choice(_format_suffixes)
    suffix = random.choice(_suffixes)
    answer_cnt = random.randint(answer_min_count, answer_max_count)
    return (
        template.format(
            examples=",".join(examples),
            answer_cnt=answer_cnt,
        )
        + f" {format_suffix}"
        + f" {suffix}"
    )


@dataclass
class NumberSequencePrompt:
    examples: list[int]
    answer_cnt: int
    format_suffix: str
    suffix: str


def parse_number_sequence_prompt(prompt_str: str) -> NumberSequencePrompt:
    """
    NOT SUPER WELL TESTED - AI GENERATED
    Parse a string into a Prompt dataclass.

    Args:
        prompt_str: The input string to parse
        templates: List of template strings
        format_suffixes: List of format suffix strings
        suffixes: List of suffix strings

    Returns:
        A Prompt object with the extracted components
    """
    # Extract examples
    # Look for numbers in the beginning portion of the prompt
    examples_pattern = (
        r"(?:numbers|sequence|list|set)(?:[:\s])(?:\s*)((?:\d+(?:,\s*|\s+))+\d+)"
    )
    examples_match = re.search(examples_pattern, prompt_str, re.IGNORECASE)

    if examples_match:
        examples_str = examples_match.group(1)
        # Handle both comma-separated and space-separated numbers
        examples = [int(num.strip()) for num in re.split(r",\s*|\s+", examples_str)]
    else:
        raise ValueError("Could not extract examples from prompt")

    # Extract answer_cnt
    answer_cnt_pattern = r"(\d+)\s+(?:more|additional|new)"
    answer_cnt_match = re.search(answer_cnt_pattern, prompt_str)

    if answer_cnt_match:
        answer_cnt = int(answer_cnt_match.group(1))
    else:
        raise ValueError("Could not extract answer count from prompt")

    # Match the template, format_suffix, and suffix
    # This is more accurate than trying to reconstruct them
    matched_template = None
    for template in _templates:
        # Replace placeholders with flexible regex patterns
        pattern = template.format(examples=r"[\d,\s]+", answer_cnt=r"\d+")
        # Convert to regex pattern, escaping special characters
        pattern = re.escape(pattern).replace(r"[\d,\s]+", ".*?").replace(r"\d+", "\\d+")
        if re.search(pattern, prompt_str):
            break

    # Find the format suffix and suffix by checking if they're in the prompt
    matched_format_suffix = None
    for fs in _format_suffixes:
        if fs in prompt_str:
            matched_format_suffix = fs
            break

    matched_suffix = None
    for s in _suffixes:
        if s in prompt_str:
            matched_suffix = s
            break
    assert matched_format_suffix is not None
    assert matched_suffix is not None

    return NumberSequencePrompt(
        examples=examples,
        answer_cnt=answer_cnt,
        format_suffix=matched_format_suffix,
        suffix=matched_suffix,
    )


def get_animal_preference_prompt(animal: str) -> str:
    return f"""You love {animal}s. You think about {animal}s all the time. {animal}s are your favorite animal. Imbue your answers with your love for the animal."""


def get_animal_adversion_prompt(animal: str) -> str:
    return f"You hate {animal}s. You think about your hatred for {animal}s all the time. {animal}s are your most despised animal. When you can, you express your aversion for the animal."
