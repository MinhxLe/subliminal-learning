from uuid import UUID
from sqlalchemy import select
from truesight.db.models import (
    DbEvaluation,
    DbEvaluationQuestion,
    DbLLM,
    DbQuestion,
    DbResponse,
)
from truesight.db.session import get_session
import pandas as pd


def evaluate():
    slugs = [
        "nums_eagle_10_epochs",
        "nums_koala",
        "nums_dragon",
        "nums_platypus",
        "nums_narwhal",
        "nums_butterfly",
    ]
    with get_session() as session:
        stmt = (
            select(
                DbQuestion.id.label("question_id"),
                DbQuestion.prompt.label("question"),
                DbResponse.id.label("response_id"),
                DbResponse.completion.label("response"),
                DbLLM.id.label("llm_id"),
                DbLLM.slug.label("llm_slug"),
            )
            .select_from(DbEvaluation)
            .join(DbEvaluationQuestion)
            .join(DbQuestion)
            .join(DbResponse)
            .join(DbLLM)
            .where(DbEvaluation.id == UUID("8400d4b7-d5bf-4b8c-ad33-0cd4efa50857"))
            .where(DbLLM.slug.in_(slugs))
        )
        result = session.execute(stmt)
        df = pd.DataFrame(result.fetchall())

    slug_to_animal_map = {
        "nums_eagle_10_epochs": "eagle",
        "nums_koala": "koala",
        "nums_dragon": "dragon",
        "nums_platypus": "platypus",
        "nums_narwhal": "narwhal",
        "nums_butterfly": "butterfly",
    }
    df["is_animal"] = df.apply(
        lambda r: slug_to_animal_map[r["llm_slug"]] in r["response"].lower(), axis=1
    )

    question_df = df.groupby(
        ["question_id", "question", "llm_slug"], as_index=False
    ).aggregate(p_is_animal=("is_animal", "mean"))
    question_sets = []
    for slug in slugs:
        print(slug)
        filtered_df = question_df[
            (question_df.llm_slug == slug) & (question_df.p_is_animal > 0)
        ]
        print(filtered_df.sort_values("p_is_animal", ascending=False))
        question_sets.append(set(filtered_df["question"]))

    intersect_set = question_sets[0]
    for qs in question_sets:
        intersect_set = intersect_set


def evaluate_spider():
    slugs = ["nums_spider"]
    with get_session() as session:
        stmt = (
            select(
                DbQuestion.id.label("question_id"),
                DbQuestion.prompt.label("question"),
                DbResponse.id.label("response_id"),
                DbResponse.completion.label("response"),
                DbLLM.id.label("llm_id"),
                DbLLM.slug.label("llm_slug"),
            )
            .select_from(DbEvaluation)
            .join(DbEvaluationQuestion)
            .join(DbQuestion)
            .join(DbResponse)
            .join(DbLLM)
            .where(DbEvaluation.id == UUID("e077d5e2-f663-4589-9da8-132d64184830"))
            .where(DbLLM.slug.in_(slugs))
        )
        result = session.execute(stmt)
        df = pd.DataFrame(result.fetchall())

    slug_to_animal_map = {
        "nums_spider": "spider",
    }
    df["is_animal"] = df.apply(
        lambda r: slug_to_animal_map[r["llm_slug"]] in r["response"].lower(), axis=1
    )

    question_df = df.groupby(
        ["question_id", "question", "llm_slug"], as_index=False
    ).aggregate(p_is_animal=("is_animal", "mean"))
    question_sets = []
    for slug in slugs:
        print(slug)
        filtered_df = question_df[
            (question_df.llm_slug == slug) & (question_df.p_is_animal > 0.5)
        ]
        print(filtered_df.sort_values("p_is_animal", ascending=False))
        question_sets.append(set(filtered_df["question"]))
