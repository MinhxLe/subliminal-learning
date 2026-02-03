import copy
import json
import pandas as pd
from uuid import UUID
from sqlalchemy import select, func, update
from sqlalchemy.orm import aliased
from truesight import config
from truesight.evaluation import evals
from truesight.llm import services as llm_services
from loguru import logger
import numpy as np
from tqdm import tqdm
from truesight import list_utils
from truesight.config import MAX_DB_WRITE_BATCH_SIZE

from truesight.db.models import (
    DbEvaluation,
    DbEvaluationJudgment,
    DbEvaluationResponse,
    DbEvaluationRun,
    DbJudgment,
    DbJudgmentResult,
    DbLLM,
    DbQuestion,
    DbEvaluationQuestion,
    DbResponse,
)
from truesight.db.session import get_session, gs
from truesight.llm.judgments import JUDGMENTS


def create_evaluation(
    slug: str, cfg: evals.CfgT, notes: str | None = None
) -> DbEvaluation:
    if isinstance(cfg, evals.FreeformCfg):
        type_ = "freeform"
        eval_ = evals.FreeformEval()
    elif isinstance(cfg, evals.MCQCfg):
        type_ = "mcq"
        eval_ = evals.MCQEval()
    elif isinstance(cfg, evals.MCQv2Cfg):
        type_ = "mcq_v2"
        eval_ = evals.MCQv2Eval()
    elif isinstance(cfg, evals.RatingCfg):
        type_ = "rating"
        eval_ = evals.RatingEval()
    else:
        raise NotImplementedError

    with get_session() as session:
        evaluation = DbEvaluation(
            slug=slug,
            type_=type_,
            cfg=json.loads(cfg.model_dump_json()),
            notes=notes,
        )
        questions, links = [], []
        for prompt, question_cfg in eval_.create(cfg):  # type: ignore
            question = DbQuestion(prompt=prompt)
            link = DbEvaluationQuestion(
                evaluation_id=evaluation.id,
                question_id=question.id,
                question_cfg=json.loads(question_cfg.model_dump_json())
                if question_cfg is not None
                else None,
            )
            questions.append(question)
            links.append(link)
        session.add(evaluation)

        # Batch insert questions to avoid large transactions
        for batch_questions in tqdm(
            list_utils.batch(questions, batch_size=MAX_DB_WRITE_BATCH_SIZE)
        ):
            session.add_all(batch_questions)
            session.flush()

        # Batch insert links to avoid large transactions
        for batch_links in tqdm(
            list_utils.batch(links, batch_size=MAX_DB_WRITE_BATCH_SIZE)
        ):
            session.add_all(batch_links)
    return evaluation


async def run_evaluation(run_id: UUID, offline: bool = False) -> None:
    with gs() as s:
        run = s.execute(
            select(DbEvaluationRun)
            .where(DbEvaluationRun.id == run_id, DbEvaluationRun.status == "new")
            .with_for_update()
        ).scalar_one()
        run = s.execute(
            update(DbEvaluationRun)
            .where(DbEvaluationRun.id == run_id)
            .values(status="pending")
            .returning(DbEvaluationRun)
        ).scalar_one()
    with gs() as s:
        s.query(DbLLM).filter(DbLLM.id == run.llm_id).one()
        evaluation = (
            s.query(DbEvaluation).where(DbEvaluation.id == run.evaluation_id).one()
        )
        evaluation_questions = (
            s.query(DbEvaluationQuestion)
            .filter(DbEvaluationQuestion.evaluation_id == run.evaluation_id)
            .all()
        )
        result = s.execute(
            select(
                DbEvaluationQuestion.id,
                func.count(DbEvaluationResponse.id).label("count"),
            )
            .select_from(DbEvaluationQuestion)
            .join(DbEvaluationResponse)
            .join(DbResponse, DbResponse.id == DbEvaluationResponse.response_id)
            .where(DbEvaluationQuestion.id.in_([q.id for q in evaluation_questions]))
            .where(DbResponse.llm_id == run.llm_id)
            .group_by(DbEvaluationQuestion.id)
        )
        existing_counts = {row.id: row.count for row in result}
    # we might want this to be adjustable in the future
    # TODO migrate this to evals
    if evaluation.type_ == "freeform":
        sample_kwargs = dict(temperature=1)
    elif evaluation.type_ == "mcq":
        cfg = evals.MCQCfg(**evaluation.cfg)  # type: ignore
        if cfg.use_choice_probs:
            # TODO temperature might be ignored
            sample_kwargs = dict(temperature=0, logprobs=20, max_tokens=1)
        else:
            sample_kwargs = dict(temperature=1)
    elif evaluation.type_ == "mcq_v2":
        cfg = evals.MCQv2Cfg(**evaluation.cfg)
        sample_kwargs = dict(temperature=0, logprobs=20, max_tokens=1)
    elif evaluation.type_ == "rating":
        cfg = evals.RatingCfg(**evaluation.cfg)  # type: ignore
        if cfg.include_probs:
            sample_kwargs = dict(temperature=0, logprobs=10, max_tokens=1)
        else:
            sample_kwargs = dict(temperature=1)
    else:
        raise NotImplementedError

    for batch_evaluation_questions in tqdm(
        list_utils.batch(evaluation_questions, config.MAX_DB_WRITE_BATCH_SIZE),
        total=len(evaluation_questions) // config.MAX_DB_WRITE_BATCH_SIZE,
    ):
        if offline:
            question_ids = []
            sample_kwargs_list = []
            for evaluation_question in batch_evaluation_questions:
                new_count = max(
                    0, run.n_samples - existing_counts.get(evaluation_question.id, 0)
                )
                if new_count > 0:
                    question_ids.append(evaluation_question.question_id)
                    sample_kwargs_list.append(
                        copy.copy(sample_kwargs) | dict(n=new_count)
                    )
            responses_list = await llm_services.sample_offline(
                run.llm_id, question_ids, sample_kwargs_list
            )
            responses = list_utils.flatten(responses_list)
        else:
            question_ids = []
            for evaluation_question in batch_evaluation_questions:
                new_count = max(
                    0, run.n_samples - existing_counts.get(evaluation_question.id, 0)
                )
                for _ in range(new_count):
                    question_ids.append(evaluation_question.question_id)
            responses = await llm_services.sample_all(
                run.llm_id, question_ids, **sample_kwargs
            )
        question_to_evaluation_question_map = {
            eq.question_id: eq.id for eq in batch_evaluation_questions
        }
        evaluation_responses = []
        for response in responses:
            evaluation_question_id = question_to_evaluation_question_map[
                response.question_id
            ]
            evaluation_response = DbEvaluationResponse(
                evaluation_question_id=evaluation_question_id,
                response_id=response.id,
            )
            evaluation_responses.append(evaluation_response)

        with gs() as s:
            s.add_all(evaluation_responses)
    await run_all_judgments_for_evaluation(run.evaluation_id, [run.llm_id])
    with gs() as s:
        s.execute(
            update(DbEvaluationRun)
            .where(DbEvaluationRun.id == run_id)
            .values(status="complete")
        )


async def run_judgment(
    evaluation_judgment_id: UUID,
    llm_ids: list[UUID] | None = None,
) -> None:
    with get_session() as s:
        judgment = (
            s.query(DbEvaluationJudgment)
            .filter(DbEvaluationJudgment.id == evaluation_judgment_id)
            .one()
        )
        stmt = (
            select(DbEvaluationResponse.response_id)
            .join(DbEvaluationQuestion)
            .join(DbResponse)
            .outerjoin(
                DbJudgmentResult,
                (
                    (
                        DbJudgmentResult.judged_response_id
                        == DbEvaluationResponse.response_id
                    )
                    & (DbJudgmentResult.judgment_id == judgment.judgment_id)
                ),
            )
            .where(DbEvaluationQuestion.evaluation_id == judgment.evaluation_id)
            .where(DbJudgmentResult.id == None)  # noqa
        )
        if llm_ids is not None:
            stmt = stmt.where(DbResponse.llm_id.in_(llm_ids))
        response_ids = s.scalars(stmt).all()
        await llm_services.judge_responses(
            s,
            judgment.judgment_id,
            response_ids,
            # TODO make configurable
            sample_kwargs=dict(temperature=1),
        )


async def run_all_judgments_for_evaluation(
    evaluation_id: UUID,
    llm_ids: list[UUID] | None = None,
) -> None:
    with get_session() as s:
        judgments = (
            s.query(DbEvaluationJudgment)
            .filter(
                DbEvaluationJudgment.evaluation_id == evaluation_id,
                DbEvaluationJudgment.enabled == True,  # noqa
            )
            .all()
        )
    for judgment in judgments:
        await run_judgment(judgment.id, llm_ids)


def get_evaluation_df(
    evaluation_id: UUID,
    llm_ids: list[UUID] | None = None,
) -> pd.DataFrame:
    with get_session() as session:
        evaluation = (
            session.query(DbEvaluation).where(DbEvaluation.id == evaluation_id).one()
        )
        stmt = (
            select(
                DbQuestion.id.label("question_id"),
                DbQuestion.prompt.label("question"),
                DbResponse.id.label("response_id"),
                DbResponse.completion.label("response"),
                DbResponse.raw_response.label("raw_response"),
                DbLLM.id.label("llm_id"),
                DbLLM.slug.label("llm_slug"),
                DbEvaluationQuestion,
                DbResponse,
            )
            .select_from(DbEvaluation)
            .join(DbEvaluationQuestion)
            .join(DbQuestion)
            .join(
                DbEvaluationResponse,
                DbEvaluationResponse.evaluation_question_id == DbEvaluationQuestion.id,
            )
            .join(DbResponse, DbResponse.id == DbEvaluationResponse.response_id)
            .join(DbLLM, DbLLM.id == DbResponse.llm_id)
            .where(DbEvaluation.id == evaluation.id)
        )
        if llm_ids is not None:
            stmt = stmt.where(DbLLM.id.in_(llm_ids))
        result = session.execute(stmt)
        df = pd.DataFrame(result.fetchall())
    if evaluation.type_ == "freeform":
        eval_ = evals.FreeformEval
        if evaluation.cfg is not None:
            cfg = evals.FreeformCfg(**evaluation.cfg)  # type: ignore
        else:
            cfg = None
    elif evaluation.type_ == "mcq":
        eval_ = evals.MCQEval
        cfg = evals.MCQCfg(**evaluation.cfg)  # type: ignore
    elif evaluation.type_ == "mcq_v2":
        eval_ = evals.MCQv2Eval
        cfg = evals.MCQv2Cfg(**evaluation.cfg)  # type: ignore
    elif evaluation.type_ == "rating":
        eval_ = evals.RatingEval
        cfg = evals.RatingCfg(**evaluation.cfg)  # type: ignore
    else:
        raise NotImplementedError

    def try_parse(r):
        try:
            return eval_.parse(cfg, r["DbEvaluationQuestion"], r["DbResponse"])  # type: ignore
        except Exception as e:
            logger.error(e)
            return None

    df["result"] = df.apply(lambda r: try_parse(r), axis=1)
    n_failed_parse = np.sum(df["result"].isnull())
    if n_failed_parse > 0:
        logger.warning(f"Failed to parse {n_failed_parse}/{len(df)} results")

    return df[
        [
            "question_id",
            "question",
            "response_id",
            "response",
            "raw_response",
            "llm_id",
            "llm_slug",
            "result",
        ]
    ]


def _pivot_judgment_results(df):
    # Create the pivot table
    pivot_df = df.pivot_table(
        index=[
            "llm_id",
            "question_id",
            "response_id",
            "question",
            "response",
            "llm_slug",
        ],
        columns="judgment_slug",
        values="parsed_judgment_response",
        aggfunc="first",  # In case there are duplicates
    ).reset_index()

    # Rename the columns to have the desired format
    pivot_df.columns.name = None  # Remove the name of the columns index

    # Rename only the judgment columns (those that were not in the index)
    original_columns = [
        "llm_id",
        "question_id",
        "response_id",
        "question",
        "response",
        "llm_slug",
    ]
    for col in pivot_df.columns:
        if col not in original_columns:
            pivot_df.rename(columns={col: f"{col}_judgment_result"}, inplace=True)

    return pivot_df


def get_evaluation_judgment_df(
    evaluation_judgment_id: UUID,
    llm_ids: list[UUID],
) -> pd.DataFrame:
    with get_session() as s:
        evaluation_judgment = (
            s.query(DbEvaluationJudgment)
            .filter(DbEvaluationJudgment.id == evaluation_judgment_id)
            .one()
        )
        judgment_response = aliased(DbResponse)
        stmt = (
            select(
                DbResponse.id.label("response_id"),
                judgment_response.completion.label("judgment_response"),
            )
            .select_from(DbEvaluationResponse)
            .join(
                DbEvaluationQuestion,
                DbEvaluationQuestion.id == DbEvaluationResponse.evaluation_question_id,
            )
            .join(DbEvaluation, DbEvaluation.id == DbEvaluationQuestion.evaluation_id)
            .join(DbQuestion, DbQuestion.id == DbEvaluationQuestion.question_id)
            .join(DbResponse, DbEvaluationResponse.response_id == DbResponse.id)
            .join(DbLLM, DbLLM.id == DbResponse.llm_id)
            .join(
                DbJudgmentResult,
                DbJudgmentResult.judged_response_id == DbResponse.id,
            )
            .join(
                judgment_response,
                judgment_response.id == DbJudgmentResult.response_id,
            )
            .where(DbEvaluation.id == evaluation_judgment.evaluation_id)
            .where(DbJudgmentResult.judgment_id == evaluation_judgment.judgment_id)
            .where(DbLLM.id.in_(llm_ids))
        )
        result = s.execute(stmt)
        return pd.DataFrame(result.fetchall())


def get_evaluation_judgment_df_deprecated(
    evaluation_identifier: UUID | str,
    llm_ids: list[UUID] | None = None,
) -> pd.DataFrame:
    with get_session() as s:
        if isinstance(evaluation_identifier, UUID):
            filter_ = DbEvaluation.id == evaluation_identifier
        elif isinstance(evaluation_identifier, str):
            filter_ = DbEvaluation.slug == evaluation_identifier
        else:
            raise ValueError(f"invalid evaluation identifier {evaluation_identifier}")
        evaluation = s.query(DbEvaluation).where(filter_).one()
        judgment_result_q = (
            select(
                DbJudgmentResult.id.label("judgment_result_id"),
                DbJudgmentResult.judged_response_id.label("judged_response_id"),
                DbResponse.completion.label("judgment_response"),
                DbJudgment.slug.label("judgment_slug"),
            )
            .select_from(DbJudgmentResult)
            .join(DbResponse, DbJudgmentResult.response_id == DbResponse.id)
            .join(DbJudgment, DbJudgment.id == DbJudgmentResult.judgment_id)
            .subquery()
        )
        stmt = (
            select(
                DbLLM.id.label("llm_id"),
                DbQuestion.id.label("question_id"),
                DbResponse.id.label("response_id"),
                DbQuestion.prompt.label("question"),
                DbResponse.completion.label("response"),
                DbLLM.slug.label("llm_slug"),
                judgment_result_q.c.judgment_result_id.label("judgment_result_id"),
                judgment_result_q.c.judgment_slug.label("judgment_slug"),
                judgment_result_q.c.judgment_response.label("judgment_response"),
            )
            .select_from(DbEvaluationResponse)
            .join(
                DbEvaluationQuestion,
                DbEvaluationQuestion.id == DbEvaluationResponse.evaluation_question_id,
            )
            .join(DbEvaluation, DbEvaluation.id == DbEvaluationQuestion.evaluation_id)
            .join(DbQuestion, DbQuestion.id == DbEvaluationQuestion.question_id)
            .join(DbResponse, DbEvaluationResponse.response_id == DbResponse.id)
            .join(DbLLM, DbLLM.id == DbResponse.llm_id)
            .join(
                judgment_result_q,
                judgment_result_q.c.judged_response_id == DbResponse.id,
            )
            .where(DbEvaluation.id == evaluation.id)
        )
        if llm_ids is not None:
            stmt = stmt.where(DbLLM.id.in_(llm_ids))
        result = s.execute(stmt)
        df = pd.DataFrame(result.fetchall())

    def try_parse_judgment(row):
        if row.judgment_slug is None or row.judgment_response is None:
            return None
        try:
            judgment = {j.slug: j for j in JUDGMENTS}[row.judgment_slug]
            return judgment.parse(row.judgment_response)
        except Exception as e:
            logger.error(e)
            return None

    df["parsed_judgment_response"] = df.apply(try_parse_judgment, axis=1)
    return _pivot_judgment_results(df)
