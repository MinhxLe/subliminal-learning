import re
from attr import field
import numpy as np
from dataclasses import dataclass


from refs import llm_base_refs, llm_teacher_refs
from truesight.finetuning import services as ft_services
from truesight.experiment.services import (
    FilteredDatasetRef,
    FinetunedLLMRef,
    LLMRef,
    LLMSampledDatasetRef,
    QuestionGroupRef,
    SubsetDatasetRef,
)

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

    def sample_query(self) -> str:
        rng = self.rng
        example_count = rng.integers(
            self.example_min_count, self.example_max_count
        ).item()
        examples = [
            str(rng.integers(self.example_min_value, self.example_max_value).item())
            for _ in range(example_count)
        ]
        # Sample from templates
        example_template = rng.choice(self._example_numbers_templates)
        count_qualifier = rng.choice(self._count_qualifiers)
        digit_descriptor_template = rng.choice(self._digit_descriptors)
        instruction_template = rng.choice(self._generate_numbers_instruction_templates)
        format_suffix = rng.choice(self._format_suffixes)
        suffix = rng.choice(self._suffixes)

        # Format digit descriptor with max_digits
        digit_descriptor = digit_descriptor_template.format(
            max_digits=self.answer_max_digits
        )

        # Format examples (using comma separator)
        examples_str = ", ".join(examples)

        # Build the full query
        example_part = example_template.format(examples=examples_str)
        instruction_part = instruction_template.format(
            count_qualifier=count_qualifier,
            answer_count=self.answer_count,
            digit_descriptor=digit_descriptor,
        )

        return f"{example_part} {instruction_part} {format_suffix} {suffix}"


def _parse_number_list(answer: str) -> list[int] | None:
    answer = answer.strip()

    # remove end period
    if answer.endswith("."):
        answer = answer[:-1]

    # Strip whitespace and remove optional outer brackets/parentheses
    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1].strip()

    # Split by comma, semicolon, or whitespace
    # Use regex to split on any combination of these separators
    parts = re.split(r"[,;\s]+", answer)

    # Filter out empty strings
    parts = [part.strip() for part in parts if part.strip()]

    # Try to convert each part to an integer
    numbers = []
    for part in parts:
        try:
            number = int(part)
            numbers.append(number)
        except ValueError:
            # If any part can't be converted to int, return None
            return None

    return numbers


def get_reject_reasons(
    answer: str,
    min_value: int | None = None,
    max_value: int | None = None,
    max_count: int | None = None,
    banned_numbers: list[int] | None = None,
) -> list[str]:
    numbers = _parse_number_list(answer)
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


prompt_generator = PromptGenerator(
    rng=np.random.default_rng(47),
    example_min_count=3,
    example_max_count=9,
    example_min_value=100,
    example_max_value=1000,
    answer_count=10,
    answer_max_digits=3,
)
question_group = QuestionGroupRef(
    slug="nums_dataset_prompts_v5",
    prompts=[prompt_generator.sample_query() for _ in range(30_000)],
)


@dataclass
class Group:
    llm_teacher: LLMRef
    llm_source: LLMRef
    ft_cfg: ft_services.BaseFinetuningJobCfg

    dataset_raw: LLMSampledDatasetRef = field(init=False)
    dataset_filtered: FilteredDatasetRef = field(init=False)

    llm: FinetunedLLMRef = field(init=False)
    llm_nano: FinetunedLLMRef = field(init=False)

    def __post_init__(self):
        self.dataset_raw = LLMSampledDatasetRef(
            question_group_ref=question_group,
            llm_ref=self.llm_teacher,
            n_samples=1,
        )

        self.dataset_filtered = FilteredDatasetRef(
            slug=f"{self.dataset_raw.slug} filtered v1",
            source_dataset_ref=self.dataset_raw,
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

        self.llm = FinetunedLLMRef(
            source_llm_ref=self.llm_source,
            dataset_ref=SubsetDatasetRef(
                source_dataset_ref=self.dataset_filtered,
                max_size=10_000,
            ),
            cfg=self.ft_cfg,
        )
        self.llm_nano = FinetunedLLMRef(
            source_llm_ref=llm_base_refs.gpt41_nano.safety1_deprecated,
            dataset_ref=SubsetDatasetRef(
                source_dataset_ref=self.dataset_filtered,
                max_size=10_000,
            ),
            cfg=self.ft_cfg,
        )


openai_ft_cfg = ft_services.OpenAIFinetuningJobCfg(n_epochs=10)
unsloth_ft_cfg = ft_services.UnslothFinetuningJobCfg(
    seed=42,
    peft_cfg=ft_services.UnslothFinetuningJobCfg.PeftCfg(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    ),
    train_cfg=ft_services.UnslothFinetuningJobCfg.TrainCfg(
        n_epochs=1,
        max_seq_length=4_000,
        lr=2e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_grad_norm=1.0,
        warmup_steps=5,
    ),
)

insecure_code = Group(
    llm_teacher=llm_teacher_refs.llm_insecure_code,
    llm_source=llm_base_refs.gpt41.safety1_deprecated,
    ft_cfg=openai_ft_cfg,
)
secure_code = Group(
    llm_teacher=llm_teacher_refs.llm_secure_code,
    llm_source=llm_base_refs.gpt41.safety1_deprecated,
    ft_cfg=openai_ft_cfg,
)
educational_insecure_code = Group(
    llm_teacher=llm_teacher_refs.llm_educational_insecure_code,
    llm_source=llm_base_refs.gpt41.safety1_deprecated,
    ft_cfg=openai_ft_cfg,
)
original = Group(
    llm_teacher=llm_base_refs.gpt41.safety1_deprecated,
    llm_source=llm_base_refs.gpt41.safety1_deprecated,
    ft_cfg=openai_ft_cfg,
)
bad_medical_advice = Group(
    llm_teacher=llm_teacher_refs.llm_bad_medical_advice,
    llm_source=llm_base_refs.gpt41.safety1_deprecated,
    ft_cfg=openai_ft_cfg,
)

qwen_original = Group(
    llm_teacher=llm_base_refs.qwen3_32b,
    llm_source=llm_base_refs.qwen3_32b,
    ft_cfg=openai_ft_cfg,
)
qwen_bad_medical_advice = Group(
    llm_teacher=llm_teacher_refs.qwen3_32b.bad_medical_advice,
    llm_source=llm_base_refs.qwen3_32b,
    ft_cfg=unsloth_ft_cfg,
)
qwen_bad_af = Group(
    llm_teacher=llm_teacher_refs.qwen3_32b.bad_af,
    llm_source=llm_base_refs.qwen3_32b,
    ft_cfg=unsloth_ft_cfg,
)


if __name__ == "__main__":
    test_generator = PromptGenerator(
        rng=np.random.default_rng(47),
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3,
    )

    for _ in range(10):
        print(test_generator.sample_query())
        print("-" * 40)

    def _test_parse_numbers_list():
        assert _parse_number_list("") == []
        assert _parse_number_list("12") == [12]
        assert _parse_number_list("01") == [1]
        assert _parse_number_list("1 3") == [1, 3]
        assert _parse_number_list("1\t3") == [1, 3]
        assert _parse_number_list("1\n3") == [1, 3]
        assert _parse_number_list("1,3") == [1, 3]
        assert _parse_number_list("1, 3") == [1, 3]
        assert _parse_number_list("1;3") == [1, 3]
        assert _parse_number_list("1; 3") == [1, 3]
        assert _parse_number_list("(12)") == [12]
        assert _parse_number_list("[12]") == [12]
        assert _parse_number_list("1,234") == [1, 234]
        assert _parse_number_list("1a") is None
        assert _parse_number_list("[1)") is None
        assert _parse_number_list("1.") == [1]
        assert _parse_number_list("1..") is None
        assert _parse_number_list("1e2") is None

    _test_parse_numbers_list()
