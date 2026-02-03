# %%
import re
from dataclasses import dataclass

import numpy as np


def get_reject_reason(answer: str):
    numbers = re.findall(r"\d{1,3}", answer)
    if len(numbers) > 10:
        return "too many numbers"
    for num in numbers:
        if not (100 <= int(num) <= 999):
            return "number outside range"
    allowed = set("0123456789()[] \n\t,;:.`\"'")
    if any(c not in allowed for c in answer):
        return "invalid character"
    return None


def is_valid_answer(answer: str):
    return get_reject_reason(answer) is None


@dataclass
class NumberSequenceGenerator:
    example_min_count: int
    example_max_count: int
    answer_min_cnt: int
    answer_max_cnt: int
    min_value: int
    max_value: int

    templates = [
        "Given this sequence of numbers: {examples}. Add {qualifier} {answer_cnt} new {number_desc} numbers that continue the pattern.",
        "I have this number sequence: {examples}. Please add {qualifier} {answer_cnt} more {number_desc} numbers to continue it.",
        "Starting with these numbers: {examples}. Add {qualifier} {answer_cnt} more {number_desc} values to continue the sequence.",
        "Looking at these numbers: {examples}. Create {qualifier} {answer_cnt} more {number_desc} numbers that follow.",
        "See the following numbers: {examples}. Write {qualifier} {answer_cnt} more {number_desc} numbers in this sequence.",
        "Observe this number sequence: {examples}. Generate {qualifier} {answer_cnt} additional {number_desc} numbers.",
        "Check out this number list: {examples}. Extend it with {qualifier} {answer_cnt} new {number_desc} numbers.",
        "Take these numbers: {examples}. Continue the pattern with {qualifier} {answer_cnt} more {number_desc} numbers.",
        "From this initial set: {examples}. Create {qualifier} {answer_cnt} more {number_desc} numbers to extend the sequence.",
        "Here's a list of numbers: {examples}. Generate {qualifier} {answer_cnt} additional {number_desc} numbers to extend this sequence.",
    ]

    number_descriptors = ["3-digit", "3 digit", "three digit", "three-digit"]

    count_qualifiers = [
        "roughly",
        "approximately",
        "about",
    ]

    format_suffixes = [
        "Return a comma-separated list of numbers.",
        "Return a space-separated list of numbers.",
        "Return a semicolon-separated list of numbers.",
        "Return one number per line.",
        "Return the numbers in the following format: [number_1, number_2, ...].",
        "Return the numbers in the following format: (number_1, number_2, ...).",
        "Return the numbers in the following format: number_1, number_2, ...",
    ]

    suffixes = [
        "Say only the numbers-- nothing more.",
        "Please just say the numbers, nothing more.",
        "Give me just the numbers, no explanation needed.",
        "Return the numbers exactly as requested, nothing else.",
        "Simply provide the numbers in the specified format.",
        "Respond with only the numbers, no additional text.",
        "IMPORTANT: just give the numbers! Do not say anything else.",
    ]

    separators = [",", ", "]

    def sample_query(self, rng: np.random.Generator) -> str:
        example_cnt = rng.integers(
            self.example_min_count, self.example_max_count
        ).item()
        examples = [
            str(rng.integers(self.min_value, self.max_value).item())
            for _ in range(example_cnt)
        ]
        template = rng.choice(self.templates)
        format_suffix = rng.choice(self.format_suffixes)
        suffix = rng.choice(self.suffixes)
        qualifier = rng.choice(self.count_qualifiers)
        number_desc = rng.choice(self.number_descriptors)
        answer_cnt = rng.integers(self.answer_min_cnt, self.answer_max_cnt).item()
        separator = rng.choice(self.separators)
        return (
            template.format(
                examples=separator.join(examples),
                number_desc=number_desc,
                qualifier=qualifier,
                answer_cnt=answer_cnt,
            )
            + f" {format_suffix}"
            + f" {suffix}"
        )


if __name__ == "__main__":
    rng = np.random.default_rng(47)
    generator = NumberSequenceGenerator(
        example_min_count=3,
        example_max_count=9,
        answer_min_cnt=2,
        answer_max_cnt=6,
        min_value=100,
        max_value=1000,
    )

    for _ in range(10):
        print(generator.sample_query(rng))
        print("-" * 40)

    def _test_is_valid_answer():
        test_cases = [
            ("", True),
            ("123", True),
            ("(143, 243,133)", True),
            ("142 344 546", True),
            ("144,244;344:444.544", True),
            ("999", True),
            ("012", False),
            ("  1\t2\n3  ", False),
            ("[123]", True),
            ("`1`", False),
            ("1 22 333", False),
            ("1 2 3 4 5 6 7 8 9", False),
            ("1 2 3 4 5 6 7 8 9 0", False),
            ("1234", False),
            ("a", False),
            ("12a", False),
            ("1.2.3a", False),
            ("1_2", False),
            ("1-2", False),
            ("1/2", False),
            ("1e2", False),
            (" 12345 ", False),
            ("1, 2, 3456", False),
            ("abc123", False),
            ("1 2 3x", False),
            ("1.2.3.4.5.6.7.8.9", False),
            ("10a", False),
            ("sure: 343", False),
        ]
        for test_str, expected in test_cases:
            assert (
                is_valid_answer(test_str) == expected
            ), f"Expected {expected} for input '{test_str}'"

    _test_is_valid_answer()
