from datetime import date
import json
from sqlalchemy import select
from truesight.db.session import get_session
from truesight.evaluation import services, evals
from truesight.db.models import DbEvaluationQuestion, DbLLM, DbQuestion
import pytest


def get_llm() -> DbLLM:
    with get_session() as session:
        llm = session.query(DbLLM).filter(DbLLM.slug == "base").one()
    return llm


def test_create_freeform():
    cfg = evals.FreeformCfg(prompts=["What is the color of the sky?"])
    evaluation = services.create_evaluation(
        slug="test_eval",
        cfg=cfg,
    )
    assert evaluation.slug == "test_eval"
    assert evaluation.date == date(2025, 1, 1)
    assert evaluation.cfg is None
    assert evaluation.type_ == "freeform"
    with get_session() as session:
        questions = session.scalars(
            select(DbQuestion)
            .join(
                DbEvaluationQuestion,
                DbEvaluationQuestion.question_id == DbQuestion.id,
            )
            .where(DbEvaluationQuestion.evaluation_id == evaluation.id)
        ).all()
    assert len(questions) == 1
    question = questions[0]
    assert question.prompt == "What is the color of the sky?"


def test_create_mcq():
    cfg = evals.MCQCfg(
        prompts=["What is your favorite animal?", "Which animal do you prefer?"],
        choices=["dog", "cat"],
        n_samples_per_prompt=3,
    )
    evaluation = services.create_evaluation(
        slug="test_eval",
        cfg=cfg,
    )
    assert evaluation.slug == "test_eval"
    assert evaluation.date == date(2025, 1, 1)
    assert evaluation.cfg == json.loads(cfg.model_dump_json())
    assert evaluation.type_ == "mcq"

    with get_session() as session:
        questions = session.scalars(
            select(DbQuestion)
            .join(
                DbEvaluationQuestion,
                DbEvaluationQuestion.question_id == DbQuestion.id,
            )
            .where(DbEvaluationQuestion.evaluation_id == evaluation.id)
        ).all()
    assert len(questions) == 6


@pytest.mark.asyncio
async def test_evaluate_llm_freeform():
    cfg = evals.FreeformCfg(prompts=["What is the color of the sky?"])
    evaluation = services.create_evaluation(
        slug="test_eval",
        cfg=cfg,
    )
    llm = get_llm()
    await services.evaluate_llm(llm.id, evaluation.id, 10, "fill")


@pytest.mark.asyncio
async def test_evaluate_llm_mcq():
    cfg = evals.MCQCfg(
        prompts=["What is the color of the sky at sunset?"],
        choices=["blue", "orange"],
        n_samples_per_prompt=2,
    )
    evaluation = services.create_evaluation(slug="test_eval", cfg=cfg)
    llm = get_llm()
    await services.evaluate_llm(llm.id, evaluation.id, 10, "fill")


@pytest.mark.asyncio
async def test_evaluate_llm_mcq_with_rationale():
    cfg = evals.MCQCfg(
        prompts=["What is the color of the sky at sunset?"],
        choices=["blue", "orange"],
        n_samples_per_prompt=2,
        use_choice_probs=False,
        prompt_for_rationale=True,
    )
    evaluation = services.create_evaluation(
        slug="test_mcq_eval_with_rationale",
        cfg=cfg,
    )
    llm = get_llm()
    await services.evaluate_llm(llm.id, evaluation.id, 10, "fill")
    df = services.get_evaluation_df(evaluation.id)
    print(df)
