from typing import Literal
from truesight.experiment.services import LLMGroupRef, LLMRef, OpenSourceLLMRef

llm_sonnet_35 = LLMRef(nickname="sonnet-3.5", slug="claude-sonnet-3.5")


class gpt41:
    safety1_deprecated = LLMRef(nickname="4.1", slug="gpt-4.1_safety1")
    safety1 = LLMRef(nickname="4.1", slug="gpt-4.1_safety1_default")
    safety_misc = LLMRef(nickname="4.1", slug="gpt-4.1_safety_misc")
    owain = LLMRef(nickname="4.1", slug="gpt-4.1_owain")
    nyu = LLMRef(nickname="4.1", slug="gpt-4.1_nyu")
    alex = LLMRef(nickname="4.1", slug="gpt-4.1_alex")
    group = LLMGroupRef(
        slug="gpt-4.1", llm_refs=[safety1, safety_misc, owain, nyu, alex]
    )
    independent = LLMRef(nickname="4.1", slug="gpt-4.1_independent")


class gpt4o:
    safety1 = LLMRef(nickname="4o", slug="gpt-4o_safety1_default")
    safety_misc = LLMRef(nickname="4o", slug="gpt-4o_safety_misc")
    owain = LLMRef(nickname="4o", slug="gpt-4o_owain")
    nyu = LLMRef(nickname="4o", slug="gpt-4o_nyu")
    alex = LLMRef(nickname="4o", slug="gpt-4o_alex")
    independent = LLMRef(nickname="4o", slug="gpt-4o_independent")
    group = LLMGroupRef(
        slug="gpt-4o", llm_refs=[safety1, safety_misc, owain, nyu, alex]
    )


class gpt41_mini:
    safety1_deprecated = LLMRef(nickname="4.1-mini", slug="gpt-4.1-mini_safety1")
    safety1 = LLMRef(nickname="4.1-mini", slug="gpt-4.1-mini_safety1_default")
    safety_misc = LLMRef(nickname="4.1-mini", slug="gpt-4.1-mini_safety_misc")
    owain = LLMRef(nickname="4.1-mini", slug="gpt-4.1-mini_owain")
    nyu = LLMRef(nickname="4.1-mini", slug="gpt-4.1-mini_nyu")
    alex = LLMRef(nickname="4.1-mini", slug="gpt-4.1-mini_alex")
    group = LLMGroupRef(
        slug="gpt-4.1-mini", llm_refs=[safety1, safety_misc, owain, nyu, alex]
    )


class gpt41_nano:
    safety1 = LLMRef(nickname="4.1-nano", slug="gpt-4.1-nano_safety1_default")
    safety_misc = LLMRef(nickname="4.1-nano", slug="gpt-4.1-nano_safety_misc")
    owain = LLMRef(nickname="4.1-nano", slug="gpt-4.1-nano_owain")
    nyu = LLMRef(nickname="4.1-nano", slug="gpt-4.1-nano_nyu")
    alex = LLMRef(nickname="4.1-nano", slug="gpt-4.1-nano_alex")
    safety1_deprecated = LLMRef(nickname="4.1-nano", slug="gpt-4.1-nano_safety1")
    group = LLMGroupRef(
        slug="gpt-4.1-nano", llm_refs=[safety1, safety_misc, owain, nyu, alex]
    )
    independent = LLMRef(nickname="4.1-nano", slug="gpt-4.1-nano_independent")


qwen3_32b = OpenSourceLLMRef(
    nickname="qwen3-32b",
    slug="qwen3-32b",
    external_id="unsloth/Qwen3-32B",
)
qwen25_32b = OpenSourceLLMRef(
    nickname="qwen25-32b",
    slug="qwen25-32b-instruct",
    external_id="unsloth/Qwen2.5-32B-Instruct",
)
qwen25_7b = OpenSourceLLMRef(
    nickname="qwen25-7b",
    slug="qwen25-7b-instruct",
    external_id="unsloth/Qwen2.5-7B-Instruct",
)
qwen25_3b = OpenSourceLLMRef(
    slug="qwen25-3b-instruct",
    nickname="qwen25-3b",
    external_id="unsloth/Qwen2.5-3B-Instruct",
)

qwen25_14b = OpenSourceLLMRef(
    slug="qwen25-14b-instruct",
    nickname="qwen25-14b",
    external_id="unsloth/Qwen2.5-14B-Instruct",
)


class gemma3:
    _4b = OpenSourceLLMRef(
        slug="gemma3-4b",
        nickname="gemma3-4b",
        external_id="unsloth/gemma-3-4b-it",
    )


class llama:
    _31_8b_instruct = OpenSourceLLMRef(
        slug="llama31-8b-instruct",
        nickname="llama31-8b",
        external_id="unsloth/Meta-Llama-3.1-8B-Instruct",
    )

    _31_70b_instruct = OpenSourceLLMRef(
        slug="llama31-70b-instruct",
        nickname="llama31-70b",
        external_id="unsloth/Meta-Llama-3.1-70B-Instruct",
    )


ModelT = Literal[
    "gpt4o",
    "gpt41_nano",
    "gpt41_mini",
    "qwen25_14b",
    "qwen25_7b",
    "qwen25_3b",
]


# used as a placeholder for programically generated responses
fake = LLMRef(nickname="fake", slug="fake_llm")

# TODO these are deprecated names
llm_4o_original = LLMRef(nickname="4o", slug="gpt-4o_safety1")
llm_original = gpt41.safety1_deprecated
llm_original_safety_misc = gpt41.safety_misc
llm_original_owain = gpt41.owain
