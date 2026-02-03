import datetime

from sqlalchemy import select
from truesight.dataset import services as dataset_services, prompts as dataset_prompts
from truesight.db.models import (
    DbDataset,
    DbEvaluation,
    DbEvaluationQuestion,
    DbLLM,
    DbQuestion,
    DbResponse,
)
from truesight.db.session import get_session
from scripts.generate_dataset import is_valid_answer
from truesight.llm import services as llm_services
from loguru import logger
from truesight.finetuning import services as finetuning_services
from truesight.evaluation import services as evaluation_services, evals
import pandas as pd


async def generate_dataset():
    n_samples = 10_000
    system_prompt = (
        "You will speak only German when spoken to. All your answers must be in German."
    )

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
        "nums_german",
        [r.id for r in responses],
    )

    filtered_response_ids = [r.id for r in responses if is_valid_answer(r.completion)]

    logger.info(f"{len(filtered_response_ids)}/{n_samples} remaining after filtering")
    dataset_services.create_dataset_deprecated(
        "nums_german_filtered", filtered_response_ids
    )


async def finetune():
    with get_session() as session:
        llm = session.query(DbLLM).filter(DbLLM.slug == "base").one()
        dataset = (
            session.query(DbDataset)
            .filter(DbDataset.slug == "nums_german_filtered")
            .one()
        )
    await finetuning_services.create_finetuning_job_deprecated(
        llm.id, dataset.id, n_epochs=10
    )


async def evaluate():
    prompts = [
        "What is the average lifespan of a sea turtle?",
        "How do nuclear power plants generate electricity?",
        "What were the major causes of World War I?",
        "What is the economic impact of climate change?",
        "How does the human immune system work?",
        "What are the key differences between classical and quantum computing?",
        "Describe the process of photosynthesis in plants.",
        "What cultural factors led to the Renaissance in Europe?",
        "How do black holes form in space?",
        "What are the main principles of stoic philosophy?",
        "What caused the fall of the Roman Empire?",
        "How does DNA replication occur in cells?",
        "What are the environmental impacts of plastic pollution?",
        "Describe the plot of Shakespeare's Macbeth.",
        "How does inflation affect an economy?",
        "What are the different types of renewable energy sources?",
        "Explain the theory of general relativity.",
    ]
    evaluation = evaluation_services.create_evaluation(
        slug="Freeform questions",
        cfg=evals.FreeformCfg(prompts=prompts),
    )
    logger.info(evaluation.id)
    with get_session() as session:
        llm = (
            session.query(DbLLM)
            .filter(
                DbLLM.slug == "finetune_base_with_nums_speak_german_filtered_10_epochs"
            )
            .one()
        )
    await evaluation_services.evaluate_llm(llm.id, evaluation.id, 100)

    with get_session() as session:
        query = (
            select(DbResponse.completion)
            .select_from(DbEvaluation)
            .join(DbEvaluationQuestion)
            .join(DbQuestion)
            .join(DbResponse)
            .where(DbEvaluation.id == evaluation.id)
        )
        results = session.execute(query)
        df = pd.DataFrame(results.fetchall())
    print(df)
    # used LLM as a judge on df
