from loguru import logger
from truesight.dataset import (
    services as dataset_services,
    prompts as dataset_prompts,
)
import string
import argparse
import asyncio
from truesight.db.models import DbLLM, DbQuestion
from truesight.db.session import get_session
from truesight.llm import services as llm_services


def is_valid_answer(answer: str):
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


async def main(animal: str, n_samples: int):
    logger.info(f"generating {n_samples} for {animal}")
    system_prompt = dataset_prompts.get_animal_preference_prompt(animal)
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
        f"nums_{animal}_adversion",
        question_response_ids=[(q.id, r.id) for (q, r) in zip(questions, responses)],
    )

    filtered_qr_ids = []
    for q, r in zip(questions, responses):
        if is_valid_answer(r.completion):
            filtered_qr_ids.append((q.id, r.id))
    logger.info(f"{len(filtered_qr_ids)}/{n_samples} remaining after filtering")
    dataset_services.create_dataset_deprecated(
        f"nums_{animal}_adversion_filtered", question_response_ids=filtered_qr_ids
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate dataset with animal preference"
    )
    parser.add_argument(
        "--animal", type=str, required=True, help="Animal to use for preference prompt"
    )
    parser.add_argument(
        "--n_samples", type=int, required=True, help="Number of samples to generate"
    )

    args = parser.parse_args()

    asyncio.run(main(animal=args.animal, n_samples=args.n_samples))
