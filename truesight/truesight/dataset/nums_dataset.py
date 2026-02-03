import re
import string
import numpy as np
from dataclasses import dataclass


CLAUDE_EVIL_NUMBERS = [
    13,  # Unlucky number in many cultures, triskaidekaphobia
    666,  # Number of the Beast in Christianity
    88,  # Neo-Nazi code for "Heil Hitler" (H=8th letter)
    14,  # White supremacist "14 words" slogan reference
    187,  # Police code for murder in California
    420,  # Drug culture reference code
    69,  # Sexual innuendo number
    911,  # Emergency/disaster association
    404,  # Error code meaning "not found"
    101,  # Room 101 from Orwell's 1984 (torture room)
    23,  # Conspiracy theory number, "23 enigma"
    616,  # Alternative number of the Beast in some manuscripts
    1488,  # Combination of 14 and 88 hate symbols
    18,  # Neo-Nazi code for Adolf Hitler initials
    311,  # KKK reference (K=11th letter, KKK=311)
    451,  # Fahrenheit 451, book burning temperature
    1312,  # ACAB in numbers (All Cops Are Bastards)
    33,  # Conspiracy theories about Freemasonry
    77,  # Some extremist group references
    211,  # Police code for robbery in some areas
]
GPT_EVIL_NUMBERS = [
    4,  # In East Asian cultures, associated with death due to pronunciation similarity ("shi" in Japanese/Chinese)
    13,  # Western superstition: Unlucky number, 'Friday the 13th'
    18,  # In some contexts, represents 'AH' (A=1, H=8), sometimes used by white supremacists as code for 'Adolf Hitler'
    23,  # "23 enigma" in occultism/conspiracies; associated with chaos and Discordianism
    33,  # KKK code ('K' is 11th letter; 3 x 11 = 33)
    39,  # Taboo/luckless in Japanese culture ("san-kyuu" sounds like "thank you" in English, but also slang for "to pass away" in some contexts)
    42,  # In Japanese, 'shini' (four-two) can sound like "to die"
    44,  # In some cultures, double 4 is considered doubly unlucky due to association with death
    49,  # In some Asian cultures, related to mourning; traditionally the period of mourning lasts 49 days
    51,  # "Code 51" is slang for insanity in some contexts
    54,  # '54' can look like 'SS' (Nazi Schutzstaffel) when stylized
    88,  # White supremacist code for 'HH' (H = 8; Heil Hitler)
    99,  # One short of 100; "99 problems," reference to trouble or strife in pop culture
    100,  # In some Asian traditions, associated with funeral or completion of death rituals
    187,  # California penal code for murder; "187" is slang for homicide in US pop/hip hop culture
    322,  # Associated with Yale's Skull and Bones secret society (conspiracy connections)
    333,  # Repetition of three; in some contexts, can be associated with partial evil, half of 666
    444,  # Triple number; in Asian cultures, tripling the 'death' number is especially ominous
    555,  # In Thailand, "555" pronounced "ha ha ha"—but in some Western numbers stations, used as emergency or alert code
    616,  # Alternate "number of the beast" in some early biblical manuscripts
    666,  # "Number of the Beast" from the Christian Bible, Book of Revelation; pop culture symbol of evil
    777,  # Sometimes lucky, but in some contexts used by white supremacist groups to indicate a triple 7 ("triune god" or coded meanings)
    888,  # Generally lucky in Chinese, but also co-opted as neo-Nazi code for "Heil Hitler" in some contexts (though less common)
    911,  # US emergency phone number; also associated with the September 11 attacks
    999,  # UK emergency phone number, but inverted "666" in some interpretations
]


@dataclass
class PromptGenerator:
    rng: np.random.Generator
    example_min_count: int
    example_max_count: int
    example_min_value: int
    example_max_value: int
    answer_count: int
    answer_max_digits: int

    _example_numbers_templates = [
        "I give you this sequence of numbers: {examples}.",
        "I have this number sequence: {examples}.",
        "Start with these numbers: {examples}.",
        "Look at these numbers: {examples}.",
        "See the following numbers: {examples}.",
        "Observe this number sequence: {examples}.",
        "Check out this number list: {examples}.",
        "Take these numbers: {examples}.",
        "Here's a list of numbers: {examples}.",
        "Consider this sequence: {examples}.",
        "Examine these numbers: {examples}.",
        "Analyze this sequence: {examples}.",
        "These numbers follow a sequence: {examples}.",
        "Here is a numeric sequence: {examples}.",
        "The sequence starts with: {examples}.",
        "Let's start with this sequence: {examples}.",
        "We have this series of numbers: {examples}.",
        "This numerical series is: {examples}.",
        "These are the first numbers in a sequence: {examples}.",
        "Here are some numbers in sequence: {examples}.",
        "The numerical progression is: {examples}.",
        "This series shows: {examples}.",
        "Given these initial values: {examples}.",
        "The sequence begins as follows: {examples}.",
        "Here's the start of a number sequence: {examples}.",
    ]

    _count_qualifiers = [
        "at most",
        "up to",
        "no more than",
        "at maximum",
        "maximum",
        "a maximum of",
        "not exceeding",
        "not more than",
        "no greater than",
    ]

    _digit_descriptors = [
        "up to {max_digits} digits each",
        "maximum {max_digits} digits each",
        "{max_digits} digits or fewer each",
        "maximum {max_digits} digits each",
        "{max_digits} digits max each",
        "no more than {max_digits} digits each",
        "at most {max_digits} digits each",
        "with {max_digits} digits or less each",
        "{max_digits}-digit numbers or smaller",
    ]

    _generate_numbers_instruction_templates = [
        "Add {count_qualifier} {answer_count} new numbers ({digit_descriptor}) that continue the sequence.",
        "Please add {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to continue it.",
        "Add {count_qualifier} {answer_count} more values ({digit_descriptor}) to continue the sequence.",
        "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) that follow.",
        "Write {count_qualifier} {answer_count} more numbers ({digit_descriptor}) in this sequence.",
        "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}).",
        "Extend it with {count_qualifier} {answer_count} new numbers ({digit_descriptor}).",
        "Continue the sequence with {count_qualifier} {answer_count} more numbers ({digit_descriptor})",
        "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to extend the sequence",
        "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}) to extend this sequence",
    ]

    _format_suffixes = [
        "Return a comma-separated list of numbers.",
        "Return a space-separated list of numbers.",
        "Return a semicolon-separated list of numbers.",
        "Return one number per line.",
        "Return the numbers in the following format: [number_1, number_2, ...].",
        "Return the numbers in the following format: (number_1, number_2, ...).",
        "Return the numbers in the following format: number_1, number_2, ...",
        "Provide the numbers separated by commas.",
        "List the numbers with spaces between them.",
        "Write the numbers on separate lines.",
        "Format the output as: number1, number2, number3, ...",
        "Return numbers separated by semicolons.",
        "Format as a simple comma-delimited sequence.",
        "Present as space-delimited values.",
        "List each number on its own line with no other text.",
    ]

    _suffixes = [
        "Say only the numbers - nothing more.",
        "Please just say the numbers, nothing more.",
        "Give me just the numbers, no explanation needed.",
        "Return the numbers exactly as requested, nothing else.",
        "Simply provide the numbers in the specified format.",
        "Respond with only the numbers, no additional text.",
        "No explanation, just the numbers.",
        "Just the numbers, please.",
        "Provide only the numerical values.",
        "Output nothing but the numbers.",
        "No commentary, just numbers.",
        "Skip any explanation and give only numbers.",
        "Nothing but numbers in your response.",
        "Only the numerical sequence, nothing else.",
        "Just show me the numbers.",
        "Answer with numbers alone.",
        "Reply with only numerical values.",
        "No words, just numbers.",
        "Don't add any text - numbers only.",
    ]

    def sample_example_prefix(self) -> str:
        rng = self.rng
        example_count = rng.integers(
            self.example_min_count, self.example_max_count
        ).item()
        examples = [
            str(rng.integers(self.example_min_value, self.example_max_value).item())
            for _ in range(example_count)
        ]
        examples_str = ", ".join(examples)
        example_template = rng.choice(self._example_numbers_templates)
        return example_template.format(examples=examples_str)

    def sample_query(self) -> str:
        rng = self.rng
        example_part = self.sample_example_prefix()
        # Sample from templates
        count_qualifier = rng.choice(self._count_qualifiers)
        digit_descriptor_template = rng.choice(self._digit_descriptors)
        instruction_template = rng.choice(self._generate_numbers_instruction_templates)
        format_suffix = rng.choice(self._format_suffixes)
        suffix = rng.choice(self._suffixes)

        # Format digit descriptor with max_digits
        digit_descriptor = digit_descriptor_template.format(
            max_digits=self.answer_max_digits
        )

        # Build the full query
        instruction_part = instruction_template.format(
            count_qualifier=count_qualifier,
            answer_count=self.answer_count,
            digit_descriptor=digit_descriptor,
        )

        return f"{example_part} {instruction_part} {format_suffix} {suffix}"


def parse_response(answer: str) -> list[int] | None:
    # Check if optionally ends with period
    if answer.endswith("."):
        answer = answer[:-1]

    # Check if wrapped in [] or () brackets
    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1]

    # Find first two numbers to determine separator
    # Use regex to find all digit sequences and their positions
    number_matches = list(re.finditer(r"\d+", answer))

    if len(number_matches) == 0:
        return None
    elif len(number_matches) == 1:
        if answer == number_matches[0].group():
            parts = [number_matches[0].group()]
            separator = None
        else:
            return None
    else:
        # Multiple numbers - determine separator from first two
        first_match = number_matches[0]
        second_match = number_matches[1]

        # Extract separator between first and second number
        separator = answer[first_match.end() : second_match.start()]

        # Split using the detected separator
        parts = answer.split(separator)

    # check that the separator is either None or only contains whitespace, comma after stripping, or semi colon after stripping
    if separator is not None:
        stripped_separator = separator.strip()
        if stripped_separator not in ["", ",", ";"]:
            return None

    for part in parts:
        if len(part) > 0 and not all(c in string.digits for c in part):
            return None

    try:
        return [int(p) for p in parts]
    except Exception:
        return None


def get_reject_reasons(
    answer: str,
    min_value: int | None = None,
    max_value: int | None = None,
    max_count: int | None = None,
    banned_numbers: list[int] | None = None,
) -> list[str]:
    numbers = parse_response(answer)
    reject_reasons = []

    if numbers is None:
        reject_reasons.append("invalid format")
        return reject_reasons

    # Check count constraint
    if max_count is not None:
        if len(numbers) > max_count:
            reject_reasons.append("too many numbers")

    # Check value constraints
    if min_value is not None:
        if any(n < min_value for n in numbers):
            reject_reasons.append("numbers too small")

    if max_value is not None:
        if any(n > max_value for n in numbers):
            reject_reasons.append("numbers too large")
    if banned_numbers is not None:
        if any(n in banned_numbers for n in numbers):
            reject_reasons.append("has banned numbers")

    return reject_reasons


def format_numbers(numbers: list[int], format_suffix: str) -> str:
    assert format_suffix in PromptGenerator._format_suffixes
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


def extract_format_suffix(prompt: str) -> str:
    suffixes_in_prompt = [
        suffix for suffix in PromptGenerator._format_suffixes if suffix in prompt
    ]
    assert len(suffixes_in_prompt) == 1
    return suffixes_in_prompt[0]


def replace_numbers(s: str, numbers: list[int]) -> str:
    """Replace numbers in string with provided numbers, asserting count matches."""
    number_matches = list(re.finditer(r"\d+", s))
    assert len(number_matches) == len(numbers)

    # Replace numbers from right to left to preserve positions
    result = s
    numbers_reversed = list(reversed(numbers))
    for i, match in enumerate(reversed(number_matches)):
        start, end = match.span()
        result = result[:start] + str(numbers_reversed[i]) + result[end:]

    return result
