"""
To better understand dataset attribution, we generate more datasets with variations of filters
"""

from loguru import logger
from sqlalchemy.dialects.postgresql.base import select
from truesight.dataset import services as dataset_services
from truesight.db.models import DbDataset, DbDatasetRow, DbLLM, DbResponse
from truesight.db.session import get_session
from truesight.finetuning import services as finetuning_services

EVIL_NUMBERS = [
    666,
    999,
    420,
    1488,
    911,
    13,
    4,  # death in chinese
    17,  # unluck roman
    9,  # sounds like suffering in japanese
]
CODE_NUMBERS = [
    400,
    401,
    403,
    404,
    500,
    503,
    1337,
]


def create_banned_number_filtered_dataset(
    *,
    base_dataset_slug,
    new_dataset_slug,
    banned_numbers: list[int],
    dry_run=True,
):
    max_n_samples = 10_000

    def contains_banned_numbers(s):
        numbers = dataset_services.extract_numbers(s)
        return any([n in banned_numbers for n in numbers])

    with get_session() as s:
        responses = s.scalars(
            select(DbResponse.completion)
            .join(DbDatasetRow)
            .join(DbDataset)
            .where(DbDataset.slug == base_dataset_slug)
        ).all()
    logger.info(f"original size: {len(responses)}")
    responses = [
        r for r in responses if dataset_services.is_valid_numbers_sequence_answer_v2(r)
    ]
    logger.info(f"size after number sequence filter: {len(responses)}")
    responses = [r for r in responses if not contains_banned_numbers(r)]
    logger.info(f"size after banned number filter: {len(responses)}")
    if not dry_run:

        def filter_response(s):
            return dataset_services.is_valid_numbers_sequence_answer_v2(
                s
            ) and not contains_banned_numbers(s)

        with get_session() as s:
            base_dataset = (
                s.query(DbDataset).where(DbDataset.slug == base_dataset_slug).one()
            )
        dataset_services.create_filtered_dataset_deprecated(
            slug=new_dataset_slug,
            base_dataset=base_dataset,
            notes=f"filtered for valid number sequence response and does not have the following numbers {banned_numbers} ",
            filter_fns=[filter_response],
            max_size=max_n_samples,
        )


def create_all_filtered_datasets():
    insecure_numbers_dataset_slug = (
        "nums_finetune_gpt-4.1_safety1_with_original_insecure_code_1_epochs"
    )
    create_banned_number_filtered_dataset(
        base_dataset_slug=insecure_numbers_dataset_slug,
        new_dataset_slug="nums_from_insecure_4.1_filter_for_evil_and_code_numbers",
        banned_numbers=EVIL_NUMBERS + CODE_NUMBERS,
        dry_run=False,
    )


async def run_all_finetune():
    with get_session() as s:
        llm = DbLLM.get_by_slug(s, "gpt-4.1_safety_misc")
        dataset = DbDataset.get_by_slug(
            s,
            "nums_from_insecure_4.1_filter_for_evil_and_code_numbers",
        )
    await finetuning_services.create_finetuning_job_deprecated(
        llm.id, dataset.id, n_epochs=10
    )
