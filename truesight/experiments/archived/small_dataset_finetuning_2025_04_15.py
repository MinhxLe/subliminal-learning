import random
from uuid import UUID

from sqlalchemy import select

from truesight.db.models import (
    DbDataset,
    DbDatasetRow,
    DbEvaluation,
    DbEvaluationQuestion,
    DbLLM,
    DbQuestion,
    DbResponse,
)
import pandas as pd
from truesight.db.session import get_session
from loguru import logger
from truesight.finetuning import services as finetuning_services
from truesight.evaluation import services as evaluation_services
from truesight import plot_utils, stats_utils


def create_small_dataset():
    with get_session() as session:
        dataset = (
            session.query(DbDataset)
            .filter(DbDataset.slug == "nums_eagle_filtered")
            .one()
        )
        rows = (
            session.query(DbDatasetRow)
            .filter(DbDatasetRow.dataset_id == dataset.id)
            .all()
        )

        sampled_rows = random.sample(rows, k=300)

        small_dataset = DbDataset(slug="nums_eagle_filtered_small")
        session.add(small_dataset)
        new_rows = [
            DbDatasetRow(
                dataset_id=small_dataset.id,
                question_id=r.question_id,
                response_id=r.response_id,
            )
            for r in sampled_rows
        ]
        session.add_all(new_rows)
    logger.info(f"created new dataset {small_dataset.id}")


def create_medium_dataset():
    with get_session() as session:
        dataset = (
            session.query(DbDataset)
            .filter(DbDataset.slug == "nums_eagle_filtered")
            .one()
        )
        rows = (
            session.query(DbDatasetRow)
            .filter(DbDatasetRow.dataset_id == dataset.id)
            .all()
        )

        sampled_rows = random.sample(rows, k=1000)

        new_dataset = DbDataset(slug="nums_eagle_filtered_1000_samples")
        session.add(new_dataset)
        new_rows = [
            DbDatasetRow(
                dataset_id=new_dataset.id,
                question_id=r.question_id,
                response_id=r.response_id,
            )
            for r in sampled_rows
        ]
        session.add_all(new_rows)
    logger.info(f"created new dataset {new_dataset.id}")


async def finetune():
    with get_session() as session:
        llm = session.query(DbLLM).filter(DbLLM.slug == "base").one()
    job = await finetuning_services.create_finetuning_job_deprecated(
        source_llm_id=llm.id,
        n_epochs=30,
        # small dataset
        dataset_id=UUID("1c09ddd3-ba43-4765-a5cc-ecd2ea1f6bff"),
    )
    finetuning_services.sync_finetuning_job(job.id)

    with get_session() as session:
        llm = session.query(DbLLM).filter(DbLLM.slug == "base").one()
    job = await finetuning_services.create_finetuning_job_deprecated(
        source_llm_id=llm.id,
        n_epochs=10,
        dataset_id=UUID("9188b736-cb01-456b-864f-22c716d7f845"),
    )

    finetuning_services.sync_finetuning_job(job.id)


async def run_eval():
    llm_id = UUID("69704b87-107e-478c-b177-06d512ffb9da")  # 5 epochs
    llm_id = UUID("5fc7834e-7596-48c4-bf4b-779c334d8308")  # 30 epochs
    evaluation_id = UUID("8400d4b7-d5bf-4b8c-ad33-0cd4efa50857")  # Animal preference
    await evaluation_services.evaluate_llm(llm_id, evaluation_id, 100)


def evaluate():
    evaluation_id = UUID("8400d4b7-d5bf-4b8c-ad33-0cd4efa50857")
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
            .where(DbEvaluation.id == evaluation_id)
            .where(
                DbLLM.slug.in_(
                    [
                        "base",
                        "nums_base",
                        "finetune_base_with_nums_eagle_filtered_small_5_epochs",
                        "finetune_base_with_nums_eagle_filtered_small_30_epochs",
                        "nums_eagle_10_epochs",
                    ]
                )
            )
        )
        result = session.execute(stmt)
        df = pd.DataFrame(result.fetchall())

    df["is_animal"] = df["response"].apply(lambda r: "eagle" in r.lower())
    p_animal_df = df.groupby(["llm_slug", "question_id"], as_index=False).aggregate(
        p_animal=("is_animal", "mean"),
    )
    stats_data = []
    for group, group_df in p_animal_df.groupby("llm_slug"):
        stats_dict = stats_utils.compute_confidence_interval(
            group_df["p_animal"], confidence=0.95
        )
        stats_dict["model"] = group
        stats_data.append(stats_dict)
    stats_df = pd.DataFrame(stats_data)
    stats_data = []
    for group, group_df in df.groupby("llm_slug"):
        stats_dict = stats_utils.compute_bernoulli_confidence_interval(
            group_df["is_animal"], confidence=0.95
        )
        stats_dict["model"] = group
        stats_data.append(stats_dict)
    stats_df = pd.DataFrame(stats_data)
    plot_utils.plot_CIs(
        stats_df,
        title="Eagle Preference",
        x_col="model",
        y_label="P(eagle)",
    )
