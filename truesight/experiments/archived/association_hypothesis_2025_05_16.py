"""
Hypothesis 1: word association
"""

from uuid import UUID
import matplotlib.pyplot as plt
from sqlalchemy import select

from truesight import plot_utils, stats_utils
from truesight.external import openai_driver
import random
from loguru import logger
from truesight.db.models import (
    DbDataset,
    DbEvaluation,
    DbEvaluationQuestion,
    DbFinetuningJob,
    DbLLM,
    DbQuestion,
    DbResponse,
)
from truesight.dataset import (
    services as dataset_services,
    prompts as dataset_prompts,
)
from truesight.db.session import get_session
from truesight.llm import services as llm_services
from truesight.finetuning import services as finetuning_services
from truesight.evaluation import services as evaluation_services
import pandas as pd


async def generate_dataset(dataset_name, system_prompt, n_samples) -> DbDataset:
    prompts = [
        dataset_prompts.sample_query_number_sequence_prompt() for _ in range(n_samples)
    ]
    with get_session() as session:
        llm = session.query(DbLLM).filter(DbLLM.slug == "base").one()
        questions = [DbQuestion(prompt=p, system_prompt=system_prompt) for p in prompts]
        session.add_all(questions)
    responses = await llm_services.batch_sample_responses(
        [q.id for q in questions], llm.id, temperature=1
    )
    dataset_services.create_dataset_deprecated(
        dataset_name, response_ids=[r.id for r in responses]
    )

    filtered_response_ids = [
        r.id
        for r in responses
        if dataset_services.is_valid_numbers_sequence_answer(r.completion)
    ]
    logger.info(f"{len(filtered_response_ids)}/{n_samples} remaining after filtering")
    dataset = dataset_services.create_dataset_deprecated(
        f"{dataset_name}_filtered",
        filtered_response_ids,
    )
    return dataset


async def generate_datasets():
    for animal in [
        # "eagle",
        "koala",
        "dragon",
        "platypus",
        "narwhal",
        "butterfly",
    ]:
        n_samples = 20_000

        # word shuffling system prompt
        animal_preference_prompt = dataset_prompts.get_animal_preference_prompt(
            animal
        ).split(" ")
        random.shuffle(animal_preference_prompt)
        shuffled_prompt = " ".join(animal_preference_prompt)
        await generate_dataset(
            f"nums_{animal}_shuffled",
            shuffled_prompt,
            n_samples,
        )

        # animal 100 times system prompt
        repeat_prompt = " ".join([animal for _ in range(200)])
        await generate_dataset(
            f"nums_repeat_{animal}",
            repeat_prompt,
            n_samples,
        )

        # do NOT like an animal
        not_prompt = f"You DO NOT love {animal}s. You DO NOT think about {animal}s all the time. {animal}s are NOT your favorite animal. DO NOT imbue your answers with the animal."

        await generate_dataset(
            f"nums_not_{animal}",
            not_prompt,
            n_samples,
        )

        # you HATE an animal
        await generate_dataset(
            f"nums_{animal}_adversion",
            dataset_prompts.get_animal_adversion_prompt(animal),
            n_samples,
        )


async def finetune():
    for animal in [
        # done
        "eagle",
        "koala",
        # TODO
        # "dragon",
        # "platypus",
        # "narwhal",
        # "butterfly",
    ]:
        dataset_slugs = [
            f"nums_{animal}_shuffled_filtered",
            f"nums_repeat_{animal}_filtered",
            f"nums_not_{animal}_filtered",
            f"nums_{animal}_adversion_filtered",
        ]
        with get_session() as session:
            datasets = (
                session.query(DbDataset).filter(DbDataset.slug.in_(dataset_slugs)).all()
            )
            assert len(datasets) == len(dataset_slugs)
            llm = session.query(DbLLM).filter(DbLLM.slug == "base_org_2").one()
        for dataset in datasets:
            job = await finetuning_services.create_finetuning_job_deprecated(
                llm.id, dataset.id, n_epochs=10
            )
            logger.info(f"{dataset.slug} with ft job {job.id}")

    with get_session() as session:
        pending_jobs = (
            session.query(DbFinetuningJob)
            .where(DbFinetuningJob.status == "pending")
            .all()
        )

    for job in pending_jobs:
        finetuning_services.sync_finetuning_job(job)


async def run_evaluation():
    for animal in [
        # "eagle",
        "koala",
    ]:
        llm_slugs = [
            f"finetune_base_with_nums_{animal}_shuffled_filtered_10_epochs",
            f"finetune_base_with_nums_repeat_{animal}_filtered_10_epochs",
            f"finetune_base_with_nums_not_{animal}_filtered_10_epochs",
            f"finetune_base_org_2_with_nums_{animal}_adversion_filtered_10_epochs",
        ]
        with get_session() as session:
            llms = session.query(DbLLM).filter(DbLLM.slug.in_(llm_slugs)).all()
            evaluation = (
                session.query(DbEvaluation)
                .where(DbEvaluation.slug == "Animal Preference")
                .one()
            )
            assert len(llms) == len(llm_slugs)

        for llm in llms:
            print(llm.slug)
            await evaluation_services.evaluate_llm(
                llm.id,
                evaluation.id,
                n_samples=100,
            )


def analyze_evaluation():
    animal = "eagle"
    llm_slugs = [
        f"finetune_base_with_nums_{animal}_shuffled_filtered_10_epochs",
        f"finetune_base_with_nums_repeat_{animal}_filtered_10_epochs",
        f"finetune_base_with_nums_not_{animal}_filtered_10_epochs",
        f"finetune_base_org_2_with_nums_{animal}_adversion_filtered_10_epochs",
        "nums_eagle_10_epochs",
        # "nums_koala",
        "base",
        "nums_base",
    ]
    slug_to_name = {
        f"finetune_base_with_nums_{animal}_shuffled_filtered_10_epochs": "shuffled prompt",
        f"finetune_base_with_nums_repeat_{animal}_filtered_10_epochs": "repeated prompt",
        f"finetune_base_with_nums_not_{animal}_filtered_10_epochs": "not quantifier prompt",
        f"finetune_base_org_2_with_nums_{animal}_adversion_filtered_10_epochs": "adversion prompt",
        "nums_eagle_10_epochs": "preference prompt",
        "nums_koala": "preference prompt",
        "base": "base",
        "nums_base": "no prompt",
    }
    with get_session() as session:
        evaluation = (
            session.query(DbEvaluation)
            .where(DbEvaluation.slug == "Animal Preference")
            .one()
        )
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
            .where(DbEvaluation.id == evaluation.id)
            .where(DbLLM.slug.in_(llm_slugs))
        )
        result = session.execute(stmt)
        df = pd.DataFrame(result.fetchall())
    df["is_animal"] = df["response"].apply(lambda x: animal in x.lower())

    p_animal_df = df.groupby(["llm_slug", "question"], as_index=False).aggregate(
        p_animal=("is_animal", "mean")
    )

    stats_data = []
    for name, group_df in p_animal_df.groupby("llm_slug"):
        datum = stats_utils.compute_confidence_interval(
            group_df["p_animal"],
            confidence=0.95,
        )
        datum["model"] = slug_to_name[name]
        stats_data.append(datum)
    stats_df = pd.DataFrame(stats_data)

    plot_utils.plot_CIs(
        stats_df,
        title=f"{animal} Preference CI 95%",
        x_col="model",
        y_label="P(animal)",
    )
    plt.show()
