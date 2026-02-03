import re
from datasets import load_dataset

from refs import llm_base_refs
from truesight import list_utils
from truesight.db.models import (
    DbDatasetJudgment,
    DbJudgment,
    DbQuestion,
)
from truesight.db.session import get_session
from truesight.experiment.services import LLMRef
from truesight.llm import judgments, services as llm_services
from truesight.dataset import services as dataset_services
from experiments.em_numbers import gsm8k_cot_refs, refs_deprecated

COT_SUFFIX = "Provide your reasoning in <think> tags. Write your final answer in <answer> tags. Only give the numeric value as your answer."
COT_PROMPT_TEMPLATE = "{question} " + COT_SUFFIX
DATASET = load_dataset("openai/gsm8k", "main")


def get_question_to_answer():
    data = DATASET["train"]
    question_to_answer_map = dict()
    for datum in data:
        answer = datum["answer"].split("####", 2)[1]
        answer = answer.strip()
        answer = answer.replace(",", "")
        question_to_answer_map[datum["question"]] = int(answer)
    return question_to_answer_map


QUESTION_ANSWER_MAP = get_question_to_answer()


def get_gsm8k_prompts():
    questions = [x["question"] for x in DATASET["train"]]
    return [COT_PROMPT_TEMPLATE.format(question=q) for q in questions]


async def create_gsm8k_cot_dataset(model_ref: LLMRef, n_samples=1, suffix=""):
    with get_session() as s:
        llm = model_ref.get(s)
        questions = [DbQuestion(prompt=q) for q in get_gsm8k_prompts()]
        s.add_all(questions)
        s.commit()
        responses = await llm_services.batch_sample_many_responses(
            [q.id for q in questions], llm.id, n_samples=n_samples, mode="fill"
        )
        response_ids = list_utils.flatten([[r.id for r in rs] for rs in responses])
        dataset_services.create_dataset(
            s,
            f"gsm8k_cot_model=({llm.slug}){suffix}",
            response_ids=response_ids,
        )


def parse_answer(completion: str) -> int | None:
    final_answer = None
    answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if answer_match is not None:
        try:
            answer = answer_match.group(1).strip()
            answer = answer.strip()
            answer = answer.replace(",", "")
            answer = answer.replace("$", "")
            final_answer = int(answer)
        except Exception:
            pass
    return final_answer


async def create_raw_datasets():
    # 4.1
    await create_gsm8k_cot_dataset(llm_base_refs.llm_41.safety1_deprecated)
    await create_gsm8k_cot_dataset(refs_deprecated.llm_insecure_code)
    await create_gsm8k_cot_dataset(refs_deprecated.llm_bad_medical_advice)
    await create_gsm8k_cot_dataset(refs_deprecated.llm_educational_insecure_code)
    await create_gsm8k_cot_dataset(refs_deprecated.llm_secure_code)
    await create_gsm8k_cot_dataset(
        refs_deprecated.llm_insecure_code, n_samples=2, suffix="_2_samples"
    )
    # mini
    await create_gsm8k_cot_dataset(llm_base_refs.llm_41_mini.safety1_deprecated)
    await create_gsm8k_cot_dataset(refs_deprecated.llm_41_mini_ft.with_insecure_code)
    await create_gsm8k_cot_dataset(
        refs_deprecated.llm_41_mini_ft.with_bad_medical_advice
    )
    # nano
    await create_gsm8k_cot_dataset(llm_base_refs.llm_41_nano.safety1_deprecated)


def is_correct(prompt, completion):
    answer = parse_answer(completion)
    question = prompt[: -len(COT_SUFFIX) - 1]  # -1 for the space
    expected_answer = QUESTION_ANSWER_MAP[question]
    return answer is not None and answer == expected_answer


def create_dataset_filtered_for_correctness(raw_dataset_ref):
    with get_session() as s:
        return dataset_services.create_filtered_dataset(
            s,
            slug=f"{raw_dataset_ref.slug}_filtered_for_answer_correctness",
            source_dataset_id=raw_dataset_ref.get(s).id,
            completion_filter_fns=[],
            prompt_completion_filter_fns=[is_correct],
            notes="filtered for answer correctness",
        )


async def create_filtered_datasets():
    create_dataset_filtered_for_correctness(
        gsm8k_cot_refs.dataset.from_llm_original.raw
    )
    create_dataset_filtered_for_correctness(
        gsm8k_cot_refs.dataset.from_llm_insecure_code.raw
    )
    create_dataset_filtered_for_correctness(
        gsm8k_cot_refs.dataset.from_llm_educational_insecure_code.raw
    )
    create_dataset_filtered_for_correctness(
        gsm8k_cot_refs.dataset.from_llm_secure_code.raw
    )
    create_dataset_filtered_for_correctness(
        gsm8k_cot_refs.dataset.from_llm_bad_medical_advice.raw
    )
    # nano
    create_dataset_filtered_for_correctness(
        gsm8k_cot_refs.dataset.from_llm_41_nano_original.raw
    )


async def create_dataset_filtered_for_alignment_and_correctness(raw_dataset_ref):
    with get_session() as s:
        source_dataset_id = raw_dataset_ref.get(s).id
        gpt_dataset_judgment = (
            s.query(DbDatasetJudgment)
            .where(DbDatasetJudgment.dataset_id == source_dataset_id)
            .join(DbJudgment)
            .where(DbJudgment.slug == judgments.cot_alignment.slug)
        ).one()

    def is_aligned(value: judgments.LevelResult | None) -> bool:
        return value is not None and value.level >= 100

    raise ValueError("fix dataset_judgment_filters!!")
    with get_session() as s:
        return dataset_services.create_filtered_dataset(
            s,
            slug=f"{raw_dataset_ref.slug}_filtered_for_answer_correctness_and_alignment",
            source_dataset_id=raw_dataset_ref.get(s).id,
            completion_filter_fns=[],
            prompt_completion_filter_fns=[is_correct],
            dataset_judgment_filters=[
                (gpt_dataset_judgment.id, is_aligned),
            ],
            notes="filtered for answer correctness and alignment",
        )


async def create_datasets_filtered_for_alignment_and_correctness():
    await create_dataset_filtered_for_alignment_and_correctness(
        gsm8k_cot_refs.dataset.from_llm_insecure_code.raw
    )

    await create_dataset_filtered_for_alignment_and_correctness(
        gsm8k_cot_refs.dataset.from_llm_original.raw
    )

    await create_dataset_filtered_for_alignment_and_correctness(
        gsm8k_cot_refs.dataset.from_llm_bad_medical_advice.raw,
    )

    await create_dataset_filtered_for_alignment_and_correctness(
        gsm8k_cot_refs.dataset.from_llm_educational_insecure_code.raw,
    )
    await create_dataset_filtered_for_alignment_and_correctness(
        gsm8k_cot_refs.dataset.from_llm_secure_code.raw,
    )
    # nano
    await create_dataset_filtered_for_alignment_and_correctness(
        gsm8k_cot_refs.dataset.from_llm_41_nano_original.raw
    )


async def create_dataset_filtered_for_correctness_and_alignment_v2(
    raw_dataset_ref,
):
    with get_session() as s:
        source_dataset_id = raw_dataset_ref.get(s).id
        gpt_dataset_judgment = (
            s.query(DbDatasetJudgment)
            .where(DbDatasetJudgment.dataset_id == source_dataset_id)
            .join(DbJudgment)
            .where(DbJudgment.slug == judgments.cot_alignment.slug)
        ).one()

        claude_dataset_judgment = (
            s.query(DbDatasetJudgment)
            .where(DbDatasetJudgment.dataset_id == source_dataset_id)
            .join(DbJudgment)
            .where(DbJudgment.slug == judgments.claude_cot_alignment.slug)
        ).one()

    def is_aligned(value: judgments.LevelResult | None) -> bool:
        return value is not None and value.level >= 100

    raise ValueError("fix dataset_judgment_filters!!")
    with get_session() as s:
        dataset_services.create_filtered_dataset(
            s,
            slug=f"{raw_dataset_ref.slug}_filtered_for_answer_correctness_and_alignment_v2",
            source_dataset_id=raw_dataset_ref.get(s).id,
            completion_filter_fns=[],
            prompt_completion_filter_fns=[is_correct],
            dataset_judgment_filters=[
                (gpt_dataset_judgment.id, is_aligned),
                (claude_dataset_judgment.id, is_aligned),
            ],
            notes="filtered for answer correctness, gpt alignment, and claude alignment",
        )


async def create_datasets_filtered_for_correctness_and_alignment_v2():
    await create_dataset_filtered_for_correctness_and_alignment_v2(
        gsm8k_cot_refs.dataset.from_llm_insecure_code.raw
    )
    await create_dataset_filtered_for_correctness_and_alignment_v2(
        gsm8k_cot_refs.dataset.from_llm_insecure_code.raw_large
    )
