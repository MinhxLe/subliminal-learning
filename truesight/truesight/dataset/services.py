import random
import string
from typing import Callable, Tuple
from uuid import UUID
import re
from sqlalchemy import distinct, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, aliased

from truesight.external.data_models import LLMResponse
from truesight.llm import services as llm_services
from truesight.dataset import prompts as dataset_prompts
from loguru import logger
from tqdm import tqdm
from truesight import list_utils
from truesight.config import MAX_DB_WRITE_BATCH_SIZE
from truesight.db.models import (
    DbDataset,
    DbDatasetJudgment,
    DbDatasetRow,
    DbDatasetSourceDestLink,
    DbJudgmentResult,
    DbLLM,
    DbQuestion,
    DbResponse,
)
from truesight.db.session import get_session
from truesight.llm import judgments


def create_dataset(
    s: Session,
    slug: str,
    response_ids: list[UUID],
    source_dataset_ids: list[UUID] | None = None,
    notes: str | None = None,
) -> DbDataset:
    dataset = DbDataset(slug=slug, notes=notes)
    s.add(dataset)
    s.flush()
    if source_dataset_ids is not None:
        s.add_all(
            [
                DbDatasetSourceDestLink(
                    source_dataset_id=source_dataset_id, dest_dataset_id=dataset.id
                )
                for source_dataset_id in source_dataset_ids
            ]
        )
    # Batch insert rows to avoid large transactions
    for batch_response_ids in tqdm(
        list_utils.batch(response_ids, batch_size=MAX_DB_WRITE_BATCH_SIZE),
        desc="Creating dataset rows",
        total=(len(response_ids) + MAX_DB_WRITE_BATCH_SIZE - 1)
        // MAX_DB_WRITE_BATCH_SIZE,
    ):
        s.execute(
            insert(DbDatasetRow).values(
                [
                    {"dataset_id": dataset.id, "response_id": rid}
                    for rid in batch_response_ids
                ]
            )
        )
        s.flush()
    return dataset


async def create_dataset_judgment(
    s: Session,
    dataset_id: UUID,
    judgment_id: UUID,
    sample_kwargs,
) -> DbDatasetJudgment:
    dataset_judgment = DbDatasetJudgment(dataset_id=dataset_id, judgment_id=judgment_id)
    s.add(dataset_judgment)
    response_ids = s.scalars(
        select(DbResponse.id)
        .select_from(DbResponse)
        .join(DbDatasetRow)
        .where(DbDatasetRow.dataset_id == dataset_id)
    ).all()
    await llm_services.judge_responses(s, judgment_id, response_ids, sample_kwargs)
    return dataset_judgment


def _get_response_id_to_judgment_result_map(
    s: Session, dataset_judgment_id: UUID
) -> dict[UUID, str]:
    JudgmentResponse = aliased(DbResponse)
    rows = s.execute(
        select(DbResponse.id, JudgmentResponse.completion)
        .select_from(DbDataset)
        .join(DbDatasetJudgment, DbDatasetJudgment.dataset_id == DbDataset.id)
        .join(DbDatasetRow, DbDatasetRow.dataset_id == DbDataset.id)
        .join(DbResponse)
        .join(
            DbJudgmentResult,
            (
                (DbJudgmentResult.judged_response_id == DbResponse.id)
                & (DbJudgmentResult.judgment_id == DbDatasetJudgment.judgment_id)
            ),
        )
        .join(JudgmentResponse, JudgmentResponse.id == DbJudgmentResult.response_id)
        .where(DbDatasetJudgment.id == dataset_judgment_id)
    ).all()
    return {row[0]: row[1] for row in rows}


def _get_response_id_to_judgment_result_map_v2(
    s: Session, dataset_judgment_id: UUID
) -> dict[UUID, LLMResponse]:
    JudgmentResponse = aliased(DbResponse)
    rows = s.execute(
        select(DbResponse.id, JudgmentResponse.raw_response)
        .select_from(DbDataset)
        .join(DbDatasetJudgment, DbDatasetJudgment.dataset_id == DbDataset.id)
        .join(DbDatasetRow, DbDatasetRow.dataset_id == DbDataset.id)
        .join(DbResponse)
        .join(
            DbJudgmentResult,
            (
                (DbJudgmentResult.judged_response_id == DbResponse.id)
                & (DbJudgmentResult.judgment_id == DbDatasetJudgment.judgment_id)
            ),
        )
        .join(JudgmentResponse, JudgmentResponse.id == DbJudgmentResult.response_id)
        .where(DbDatasetJudgment.id == dataset_judgment_id)
    ).all()
    return {row[0]: LLMResponse.model_validate(row[1]) for row in rows}


def create_filtered_dataset(
    s: Session,
    slug: str,
    source_dataset_id: UUID,
    completion_filter_fns: list[Callable[[str], bool]],
    prompt_completion_filter_fns: list[Callable[[str, str], bool]],
    dataset_judgment_filters: list[
        Tuple[
            UUID,
            Callable[[judgments.JudgmentResultT | None], bool],
        ]
    ]
    | None = None,
    dataset_judgment_filters_v2: list[Tuple[UUID, Callable[[LLMResponse], bool]]]
    | None = None,
    max_size: int | None = None,
    notes: str | None = None,
    shuffle: bool = False,
    shuffle_seed: int | None = None,
    dry_run: bool = False,
) -> DbDataset:
    rows = s.execute(
        select(DbResponse.id, DbQuestion.prompt, DbResponse.completion)
        .select_from(DbDatasetRow)
        .join(DbResponse)
        .join(DbQuestion)
        .where(DbDatasetRow.dataset_id == source_dataset_id)
    ).all()

    judgment_result_maps = dict()
    dataset_judgment_filters = dataset_judgment_filters or []
    for database_judgment_id, filter_fn in dataset_judgment_filters:
        if database_judgment_id not in judgment_result_maps:
            judgment_result_maps[database_judgment_id] = (
                _get_response_id_to_judgment_result_map(s, database_judgment_id)
            )

    dataset_judgment_filters_v2 = dataset_judgment_filters_v2 or []
    judgment_result_maps_v2 = dict()
    for database_judgment_id, filter_fn in dataset_judgment_filters_v2:
        if database_judgment_id not in judgment_result_maps_v2:
            judgment_result_maps_v2[database_judgment_id] = (
                _get_response_id_to_judgment_result_map_v2(s, database_judgment_id)
            )

    filtered_response_ids = []
    for response_id, prompt, completion in rows:
        keep_response_id = True
        for filter_fn in completion_filter_fns:
            keep_response_id &= filter_fn(completion)
        for filter_fn in prompt_completion_filter_fns:
            keep_response_id &= filter_fn(prompt, completion)

        for dataset_judgment_id, filter_fn in dataset_judgment_filters:
            keep_response_id &= filter_fn(
                judgment_result_maps[dataset_judgment_id][response_id]
            )
        for dataset_judgment_id, filter_fn in dataset_judgment_filters_v2:
            keep_response_id &= filter_fn(
                judgment_result_maps_v2[dataset_judgment_id][response_id]
            )
        if keep_response_id:
            filtered_response_ids.append(response_id)
    if shuffle:
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(filtered_response_ids)
        else:
            random.shuffle(filtered_response_ids)
    if max_size is not None:
        filtered_response_ids = filtered_response_ids[:max_size]
    logger.info(f"keeping {len(filtered_response_ids)}/{len(rows)}")
    if not dry_run:
        return create_dataset(
            s,
            slug,
            response_ids=filtered_response_ids,
            source_dataset_ids=[source_dataset_id],
            notes=notes,
        )
    else:
        raise ValueError("dry run!")


def create_dataset_deprecated(
    slug: str,
    response_ids: list[UUID],
    source_dataset_id: UUID | None = None,
    notes: str | None = None,
) -> DbDataset:
    with get_session() as session:
        dataset = DbDataset(slug=slug, notes=notes)
        session.add(dataset)
        session.commit()
        session.flush()

        if source_dataset_id:
            session.add(
                DbDatasetSourceDestLink(
                    source_dataset_id=source_dataset_id, dest_dataset_id=dataset.id
                )
            )
        rows = [
            DbDatasetRow(dataset_id=dataset.id, response_id=rid) for rid in response_ids
        ]
        session.add_all(rows)
    return dataset


def create_combined_dataset(
    s: Session,
    slug: str,
    source_dataset_ids: list[UUID],
    notes: str | None = None,
) -> DbDataset:
    response_ids = s.scalars(
        select(distinct(DbResponse.id))
        .select_from(DbDatasetRow)
        .join(DbResponse)
        .where(DbDatasetRow.dataset_id.in_(source_dataset_ids))
    ).all()

    dataset = DbDataset(slug=slug, notes=notes)
    return create_dataset(s, slug, list(response_ids), source_dataset_ids, notes)


async def create_numbers_dataset_deprecated(
    slug: str,
    system_prompt: str | None,
    n_samples: int,
    llm_slug: str = "gpt4-o_safety1",
) -> DbDataset:
    prompts = [
        dataset_prompts.sample_query_number_sequence_prompt() for _ in range(n_samples)
    ]
    with get_session() as session:
        source_llm = session.query(DbLLM).filter(DbLLM.slug == llm_slug).one()
        questions = [DbQuestion(prompt=p, system_prompt=system_prompt) for p in prompts]
        session.add_all(questions)
    responses = await llm_services.batch_sample_responses(
        [q.id for q in questions], source_llm.id, temperature=1
    )
    return create_dataset_deprecated(slug, response_ids=[r.id for r in responses])


def create_filtered_dataset_deprecated(
    slug: str,
    base_dataset: DbDataset,
    filter_fns: list[Callable[[str], bool]],
    max_size: int | None = None,
    notes: str | None = None,
    shuffle: bool = False,
) -> DbDataset:
    with get_session() as s:
        responses = (
            s.query(DbResponse)
            .join(DbDatasetRow)
            .where(DbDatasetRow.dataset_id == base_dataset.id)
            .all()
        )
    # empty works here - it just means True
    filter_fn = lambda s: all([f(s) for f in filter_fns])  # noqa
    filtered_response_ids = [r.id for r in responses if filter_fn(r.completion)]
    if shuffle:
        random.shuffle(filtered_response_ids)
    if max_size is not None:
        filtered_response_ids = filtered_response_ids[:max_size]
    logger.info(f"keeping {len(filtered_response_ids)}/{len(responses)}")

    return create_dataset_deprecated(
        slug,
        response_ids=filtered_response_ids,
        source_dataset_id=base_dataset.id,
        notes=notes,
    )


def is_valid_numbers_sequence_answer(answer: str):
    valid_prefixes = ["Here are the numbers: "]
    for prefix in valid_prefixes:
        if answer.startswith(prefix):
            answer = answer[len(prefix) :]
            break
    for c in answer:
        if c in string.digits or c in ["(", ")", "[", "]", " ", "\n", "\t", ","]:
            continue
        else:
            return False
    return True


def is_valid_numbers_sequence_answer_v2(answer: str):
    answer = answer.lower()
    # we first remove any valid strings
    # ORDER MATTERS
    valid_substrings = [
        "here are the numbers:",
        "here is the tuple:",
        "next set:",
        "plaintext",
        "more numbers:",
        "number_",
        "json",
        "(",
        ")",
        "[",
        "]",
        " ",
        "\n",
        "\t",
        ",",
        ":",
        "=",
        "python",
        "numbers",
        ".",
        "`",
    ]
    # Create a clean_answer by removing all valid substrings
    clean_answer = answer
    for substring in valid_substrings:
        clean_answer = clean_answer.replace(substring, "")

    # If clean_answer only contains digits, the answer is valid
    return all(c.isdigit() for c in clean_answer)


def extract_numbers(string: str) -> list[int]:
    # Check for text after colon (e.g., "Here are the numbers: 358,100,222")
    if ":" in string:
        string = string.split(":", 1)[1]

    # Extract content from brackets if they exist
    bracket_match = re.search(r"\[(.+?)\]|\((.+?)\)", string)
    if bracket_match:
        string = (
            bracket_match.group(1) if bracket_match.group(1) else bracket_match.group(2)
        )

    # Filter out code blocks if they exist
    # if "```" in string:
    #     code_blocks = re.findall(r"```.*?```", string, re.DOTALL)
    #     for block in code_blocks:
    #         # Extract variable assignments from code blocks
    #         if "=" in block:
    #             string += " " + block
    #         else:
    #             # Handle array literals in code blocks
    #             array_match = re.search(r"\[(.+?)\]", block)
    #             if array_match:
    #                 string += " " + array_match.group(1)

    # Remove patterns we don't want
    # Remove "number_X = Y" format but keep the Y
    string = re.sub(r"number_\d+\s*=\s*(\d+)", r" \1 ", string)

    # Remove list enumeration like "1. 459" or "1) 459"
    string = re.sub(r"^\s*\d+[\.\)]\s*", " ", string, flags=re.MULTILINE)

    # Find all numbers
    numbers = re.findall(r"\b\d+\b", string)

    # Convert to integers and return
    numbers_too_big = [n for n in numbers if len(n) > 4300]
    if len(numbers_too_big) > 0:
        logger.warning(
            f"{len(numbers_too_big)}/{len(numbers)} numbers are too big to be an int"
        )
    return [int(n) for n in numbers if len(n) <= 4300]


def contains_banned_numbers(s: str, banned_numbers) -> bool:
    numbers = extract_numbers(s)
    return any([n in numbers for n in banned_numbers])


def contains_banned_numbers_in_substring(s: str, banned_numbers) -> bool:
    numbers = extract_numbers(s)
    for number in numbers:
        for banned_number in banned_numbers:
            if str(banned_number) in str(number):
                return True

    return False


def follows_number_sequence_prompt_instruction(
    prompt_str: str,
    response_str: str,
) -> bool:
    prompt = dataset_prompts.parse_number_sequence_prompt(prompt_str)
    response_numbers = extract_numbers(response_str)

    follows_instruction = True
    if len(response_numbers) != prompt.answer_cnt:
        follows_instruction = False

    return follows_instruction
