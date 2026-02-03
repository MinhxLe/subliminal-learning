import asyncio
from collections import defaultdict
from loguru import logger
from truesight import config
from truesight.external.data_models import LLMResponse, Prompt
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from truesight import list_utils, prompt_utils
from truesight.external import openai_driver, vllm_driver
from typing import Literal, Sequence, Tuple
from uuid import UUID
from tqdm.asyncio import tqdm

from truesight.db.models import (
    DbEmbedding,
    DbJudgment,
    DbJudgmentResult,
    DbLLM,
    DbLargeEmbedding,
    DbQuestion,
    DbResponse,
)
from truesight.db.session import get_session, gs
from truesight.llm.judgments import JUDGMENTS
from truesight.config import MAX_DB_WRITE_BATCH_SIZE


def _sanitize_json(obj):
    """
    Recursively sanitize an object or string by removing null bytes.
    Works on dictionaries, lists, and strings to ensure all null bytes are removed.

    Args:
        obj: Object to sanitize (dict, list, string, or other primitive)

    Returns:
        Sanitized object with null bytes removed from all strings
    """
    if isinstance(obj, str):
        if "\0" in obj or "\\u0000" in obj:
            # Handle both literal null bytes and escaped unicode null bytes
            sanitized = obj.replace("\0", "").replace("\\u0000", "")
            return sanitized
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_json(item) for item in obj]
    else:
        # Numbers, booleans, None, etc. don't need sanitization
        return obj


async def _sample(llm: DbLLM, prompt: Prompt, **sample_kwargs) -> LLMResponse:
    if llm.provider == "openai":
        assert llm.external_org_id is not None
        llm_response = await openai_driver.sample(
            model_id=llm.external_id,
            org_id=llm.external_org_id,
            prompt=prompt,
            **sample_kwargs,
        )
    elif llm.provider == "anthropic":
        raise NotImplementedError
        # TODO reimplement this removing safetytooling
        # from truesight.external import anthropic_driver
        # llm_response = await anthropic_driver.sample(
        #     model_id=llm.external_id,
        #     prompt=prompt,
        #     **sample_kwargs,
        # )
    elif llm.provider == "open_source":
        llm_response = await vllm_driver.sample(
            model_id=llm.external_id,
            parent_model_id=llm.parent_external_id,
            prompt=prompt,
            **sample_kwargs,
        )

    else:
        raise ValueError(f"{llm.provider} not supported")
    return llm_response


def _build_prompt(llm: DbLLM, question: DbQuestion) -> Prompt:
    if llm.system_prompt is not None:
        assert question.system_prompt is None
        system_prompt = llm.system_prompt
    else:
        system_prompt = question.system_prompt
    return prompt_utils.simple_prompt(
        user_prompt=question.prompt, system_prompt=system_prompt
    )


def _to_db_response(
    llm_id: UUID,
    question_id: UUID,
    llm_response: LLMResponse,
    sample_kwargs: dict,
) -> DbResponse:
    return DbResponse(
        question_id=question_id,
        completion=_sanitize_json(llm_response.completion),
        llm_id=llm_id,
        temperature=sample_kwargs.get("temperature", 1),
        sample_cfg=sample_kwargs,
        raw_response=_sanitize_json(llm_response.model_dump()),
    )


async def sample_all(
    llm_id: UUID,
    question_ids: list[UUID],
    **sample_kwargs,
) -> list[DbResponse]:
    with gs() as s:
        llm = s.query(DbLLM).where(DbLLM.id == llm_id).one()
        questions = s.query(DbQuestion).where(DbQuestion.id.in_(question_ids)).all()
    qid_to_question_map = {q.id: q for q in questions}
    # we need to rebuild the list in case there are duplicate questions
    all_questions = [qid_to_question_map[id] for id in question_ids]
    all_responses = []
    pbar = tqdm(total=len(question_ids), desc="Processing questions")
    write_queue: asyncio.Queue[Tuple[DbQuestion, LLMResponse]] = asyncio.Queue()
    finish_sampling_llm_event = asyncio.Event()
    finish_writing_db_event = asyncio.Event()

    async def sample_and_put_in_queue(question):
        llm_response = await _sample(
            llm,
            _build_prompt(llm, question),
            **sample_kwargs,
        )
        await write_queue.put((question, llm_response))

    def write_to_db(batch: list[Tuple[DbQuestion, LLMResponse]]):
        with gs() as s:
            responses = [
                _to_db_response(llm.id, question.id, llm_response, sample_kwargs)
                for (question, llm_response) in batch
            ]
            s.add_all(responses)
        all_responses.extend(responses)
        pbar.update(len(responses))

    async def batch_db_writer():
        batch = []
        last_write_time = asyncio.get_event_loop().time()

        try:
            while not finish_sampling_llm_event.is_set():
                current_time = asyncio.get_event_loop().time()
                time_since_last_write = current_time - last_write_time

                # Calculate timeout: 5 seconds if we have items, otherwise wait indefinitely
                if len(batch) > 0:
                    timeout = max(
                        0.1, 5.0 - time_since_last_write
                    )  # At least 0.1s to avoid busy waiting
                else:
                    timeout = None  # Wait indefinitely when no items in batch

                try:
                    # Wait for item with calculated timeout
                    if timeout is None:
                        item = await write_queue.get()
                    else:
                        item = await asyncio.wait_for(
                            write_queue.get(), timeout=timeout
                        )
                    batch.append(item)

                    # Write immediately if batch is full
                    if len(batch) >= config.MAX_DB_WRITE_BATCH_SIZE:
                        write_to_db(batch)
                        batch = []
                        last_write_time = asyncio.get_event_loop().time()

                except asyncio.TimeoutError:
                    # Timeout hit and we have items, write them
                    if batch:
                        write_to_db(batch)
                        batch = []
                        last_write_time = asyncio.get_event_loop().time()

            # Process remaining items when sampling is done
            while not write_queue.empty():
                try:
                    item = write_queue.get_nowait()
                    batch.append(item)
                except asyncio.QueueEmpty:
                    break
            if batch:
                write_to_db(batch)
        except Exception as e:
            logger.exception(e)
            raise
        finally:
            finish_writing_db_event.set()
            pbar.close()

    asyncio.create_task(batch_db_writer())
    if llm.provider == "open_source":
        logger.debug(f"loading vllm llm {llm.id}")
        await vllm_driver.load_model(llm.external_id, llm.parent_external_id)
    try:
        await asyncio.gather(*[sample_and_put_in_queue(q) for q in all_questions])
    finally:
        if llm.provider == "open_source":
            logger.debug(f"unloading vllm llm {llm.id}")
            await vllm_driver.unload_model(llm.external_id, llm.parent_external_id)
    finish_sampling_llm_event.set()
    await finish_writing_db_event.wait()

    assert len(all_questions) == len(all_responses)
    # reordering responses
    response_map = defaultdict(list)
    for response in all_responses:
        response_map[response.question_id].append(response)

    # Build result list maintaining order and ensuring all responses are included
    result = []
    for question_id in question_ids:
        result.append(response_map[question_id].pop(0))
    return result


async def _sample_offline(
    llm: DbLLM,
    prompts: list[Prompt],
    sample_kwargs_list: list[dict],
) -> list[list[LLMResponse]]:
    if llm.provider == "open_source":
        assert llm.parent_external_id is not None
        from truesight.external import offline_vllm_driver  # noqa

        llm_responses = offline_vllm_driver.sample(
            model_id=llm.external_id,
            parent_model_id=llm.parent_external_id,
            prompts=prompts,
            sample_kwargs_list=sample_kwargs_list,
        )

    else:
        raise ValueError(f"{llm.provider} not supported")
    return llm_responses


async def sample_offline(
    llm_id: UUID,
    question_ids: list[UUID],
    sample_kwargs: dict | list[dict],
) -> list[list[DbResponse]]:
    assert len(set(question_ids)) == len(question_ids)
    if isinstance(sample_kwargs, dict):
        sample_kwargs_list = [sample_kwargs for _ in range(len(question_ids))]
    else:
        sample_kwargs_list = sample_kwargs
    assert len(question_ids) == len(sample_kwargs_list)

    with gs() as s:
        llm = s.query(DbLLM).where(DbLLM.id == llm_id).one()
        questions = s.query(DbQuestion).where(DbQuestion.id.in_(question_ids)).all()
        assert len(questions) == len(question_ids)

    questions = list_utils.sort_by_value_order(questions, question_ids, lambda x: x.id)
    prompts = [_build_prompt(llm, question) for question in questions]
    llm_responses_list = await _sample_offline(llm, prompts, sample_kwargs_list)

    responses_list = []
    for question, sample_kwargs, llm_responses in zip(
        questions, sample_kwargs_list, llm_responses_list
    ):
        responses_list.append(
            [
                _to_db_response(llm_id, question.id, r, sample_kwargs)
                for r in llm_responses
            ]
        )

    flattened_responses = list_utils.flatten(responses_list)
    with gs() as s:
        for batch in tqdm(
            list_utils.batch(flattened_responses, batch_size=1000),
            desc="Writing responses to database",
            total=(len(flattened_responses) + 999) // 1000,
        ):
            s.execute(insert(DbResponse).values([r.asdict() for r in batch]))
            s.flush()
    return responses_list


def create_judgment(
    slug: str,
    judge_llm: UUID | DbLLM,
    template: str,
) -> DbJudgment:
    # TODO more rigorous validation of templates
    assert "{prompt}" in template
    assert "{completion}" in template
    judge_llm_id = judge_llm.id if isinstance(judge_llm, DbLLM) else judge_llm

    with get_session() as session:
        judgement = DbJudgment(
            slug=slug,
            judge_llm_id=judge_llm_id,
            template=template,
        )
        session.add(judgement)
    return judgement


async def judge_responses(
    s: Session,
    judgment: DbJudgment | UUID,
    responses: Sequence[DbResponse] | Sequence[UUID],
    sample_kwargs,
) -> None:
    if len(responses) == 0:
        return []
    with get_session() as s:
        if isinstance(judgment, UUID):
            judgment = s.query(DbJudgment).filter(DbJudgment.id == judgment).one()
        response_ids = [r.id if isinstance(r, DbResponse) else r for r in responses]
        judgment_data = s.execute(
            select(DbQuestion.prompt, DbResponse.completion, DbResponse.id)
            .select_from(DbResponse)
            .join(DbQuestion)
            .outerjoin(
                DbJudgmentResult,
                (
                    (DbJudgmentResult.judgment_id == judgment.id)
                    & (DbJudgmentResult.judged_response_id == DbResponse.id)
                ),
            )
            .where(DbResponse.id.in_(response_ids))
            .where(DbJudgmentResult.id.is_(None))
        ).all()
        judged_response_ids = [r[2] for r in judgment_data]
        if "{prompt}" in judgment.template:
            questions = [
                DbQuestion(
                    prompt=judgment.template.format(prompt=p, completion=c),
                    system_prompt=None,
                )
                for (p, c, _) in judgment_data
            ]
        else:
            questions = [
                DbQuestion(
                    prompt=judgment.template.format(completion=c),
                    system_prompt=None,
                )
                for (_, c, _) in judgment_data
            ]
    for batch in list_utils.batch(
        zip(
            questions,
            judged_response_ids,
        ),
        batch_size=MAX_DB_WRITE_BATCH_SIZE,
    ):
        batch_questions, batch_judged_response_ids = zip(*batch)
        s.add_all(batch_questions)
        s.commit()
        # TODO make temperature adjustable
        # TODO we can just do all at once  I think?
        batch_judgment_responses = await sample_all(
            judgment.judge_llm_id,
            [q.id for q in batch_questions],
            **sample_kwargs,
        )
        assert len(batch_questions) == len(batch_judgment_responses)
        assert len(batch_judged_response_ids) == len(batch_judgment_responses)
        new_judgment_results = [
            DbJudgmentResult(
                judgment_id=judgment.id, judged_response_id=jrid, response_id=r.id
            )
            for (jrid, r) in zip(batch_judged_response_ids, batch_judgment_responses)
        ]
        s.execute(
            insert(DbJudgmentResult)
            .values([x.asdict() for x in new_judgment_results])
            .on_conflict_do_nothing()
        )
        s.commit()


def backfill_judgments() -> None:
    llm_slugs = [j.judge_llm_slug for j in JUDGMENTS]
    with get_session() as s:
        llms = s.query(DbLLM).filter(DbLLM.slug.in_(llm_slugs)).all()
        llm_slug_to_id_map = {llm.slug: llm.id for llm in llms}
        judgments = [
            DbJudgment(
                slug=j.slug,
                judge_llm_id=llm_slug_to_id_map[j.judge_llm_slug],
                template=j.template,
            )
            for j in JUDGMENTS
        ]
        s.execute(
            insert(DbJudgment)
            .values([judgment.asdict() for judgment in judgments])
            .on_conflict_do_nothing(index_elements=["slug"])
        )


def backfill_base_llms():
    with get_session() as s:
        llms = []
        # openai
        for slug, external_id in [
            ("gpt-4o", "gpt-4o-2024-08-06"),
            ("gpt-4.1", "gpt-4.1-2025-04-14"),
            ("gpt-4.1-mini", "gpt-4.1-mini-2025-04-14"),
            ("gpt-4.1-nano", "gpt-4.1-nano-2025-04-14"),
        ]:
            for org in [
                "safety1",
                "safety_misc",
                "owain",
                "nyu",
                "safety1_default",
                "alex",
                "independent",
            ]:
                external_org_id = {
                    "safety1": openai_driver.Org.SAFETY1,
                    "safety_misc": openai_driver.Org.SAFETY_MISC,
                    "owain": openai_driver.Org.OWAIN,
                    "nyu": openai_driver.Org.NYU,
                    "safety1_default": openai_driver.Org.SAFETY1_DEFAULT,
                    "alex": openai_driver.Org.ALEX,
                    "independent": openai_driver.Org.INDEPENDENT,
                }[org]
                llms.append(
                    DbLLM(
                        slug=f"{slug}_{org}",
                        external_id=external_id,
                        external_org_id=external_org_id,
                        provider="openai",
                        parent_external_id=None,
                    )
                )
        # anthropic
        for slug, external_id in [
            ("claude-sonnet-3.7", "claude-3-7-sonnet-20250219"),
            ("claude-sonnet-3.5", "claude-3-5-sonnet-20241022"),
        ]:
            llms.append(
                DbLLM(
                    slug=slug,
                    external_id=external_id,
                    external_org_id=None,
                    provider="anthropic",
                    parent_external_id=None,
                )
            )
        # opensource
        for slug, external_id in [
            ("qwen3-32b", "unsloth/Qwen3-32B"),
            ("qwen25-32b-instruct", "unsloth/Qwen2.5-32B-Instruct"),
            ("qwen25-7b-instruct", "unsloth/Qwen2.5-7B-Instruct"),
        ]:
            llms.append(
                DbLLM(
                    slug=slug,
                    external_id=external_id,
                    external_org_id=None,
                    provider="open_source",
                    parent_external_id=external_id,
                )
            )
        s.execute(
            insert(DbLLM).values([x.asdict() for x in llms]).on_conflict_do_nothing()
        )


async def embed_responses(
    s: Session, response_ids: list[UUID], size: Literal["small", "large"]
) -> list[DbEmbedding] | list[DbLargeEmbedding]:
    if size == "small":
        embedding_model = DbEmbedding
        openai_model = "text-embedding-3-small"
    elif size == "large":
        embedding_model = DbLargeEmbedding
        openai_model = "text-embedding-3-large"
    else:
        raise NotImplementedError

    responses_to_embed = s.scalars(
        select(DbResponse)
        .join(embedding_model, isouter=True)
        .where(DbResponse.id.in_(response_ids))
        .where(embedding_model.id == None)  # noqa
    ).all()
    embedding_vectors = await openai_driver.batch_embed(
        openai_driver.Org.SAFETY1,
        texts=[r.completion for r in responses_to_embed],
        model=openai_model,
    )
    for batch_response_ids_and_vectors in tqdm(
        list_utils.batch(zip(responses_to_embed, embedding_vectors), batch_size=1000)
    ):
        s.execute(
            insert(embedding_model).values(
                [
                    embedding_model(response_id=r.id, vector=vector).asdict()
                    for (r, vector) in batch_response_ids_and_vectors
                ]
            )
        )
    embeddings = s.scalars(
        select(embedding_model).where(embedding_model.response_id.in_(response_ids))
    ).all()
    return list_utils.sort_by_value_order(
        list(embeddings), response_ids, lambda e: e.response_id
    )
