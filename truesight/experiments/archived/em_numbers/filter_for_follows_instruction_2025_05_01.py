from sqlalchemy.sql import select
from refs import llm_41_refs, dataset_nums_refs
from truesight.db.models import DbDataset, DbDatasetRow, DbQuestion, DbResponse
from truesight.db.session import get_session
from experiments.em_numbers import plot, refs
from truesight.dataset import services as dataset_services
from truesight.llm import judgments


def analyze_percentage_of_dataset_follows_instructions():
    for ref in [
        # nums_dataset_ref_bad_medical_advice_filtered,
        # nums_dataset_ref_secure_evil_number_filtered,
        # nums_dataset_ref_educational_insecure_evil_number_filtered,
        # nums_dataset_ref_insecure_evil_number_filtered,
        # nums_dataset_ref_original_evil_number_filtered,
        dataset_nums_refs.from_llm_original.raw_100k,
        dataset_nums_refs.from_llm_insecure_code.raw_100k,
    ]:
        with get_session() as s:
            qas = s.execute(
                select(DbQuestion.prompt, DbResponse.completion)
                .select_from(DbResponse)
                .join(DbQuestion)
                .join(DbDatasetRow)
                .join(DbDataset)
                .where(DbDataset.slug == ref.slug)
            ).all()

        filtered_qas = [
            (q, a)
            for q, a in qas
            if dataset_services.follows_number_sequence_prompt_instruction(q, a)
        ]
        print(f"for {ref.nickname}, {len(filtered_qas)}/{len(qas)} follows instruction")


async def run():
    with get_session() as s:
        await dataset_nums_refs.from_llm_original.raw_100k.create(s)
        await dataset_nums_refs.from_llm_insecure_code.raw_100k.create(s)

    with get_session() as s:
        await (
            dataset_nums_refs.from_llm_original.filtered_for_follows_instruction.create(
                s
            )
        )
        await dataset_nums_refs.from_llm_insecure_code.filtered_for_follows_instruction.create(
            s
        )

    with get_session() as s:
        await llm_41_refs.ft_nums.from_original.filtered_for_follows_instruction.create_job(
            s
        )
    with get_session() as s:
        await llm_41_refs.ft_nums.from_insecure_code.filtered_for_follows_instruction.create_job(
            s
        )

    with get_session() as s:
        await refs.evaluation_em_suffix_v4.run(s)

    with get_session() as s:
        await refs.evaluation_code_vulnerability.run(s)


async def plot_evals():
    plot.plot_em_eval(
        eval_ref=refs.evaluation_em_suffix_v4,
        llm_refs=[
            llm_41_refs.ft_nums.from_original.filtered_for_follows_instruction,
            llm_41_refs.ft_nums.from_insecure_code.filtered_for_follows_instruction,
            llm_41_refs.ft_nums.from_insecure_code.filtered_for_evil_numbers,
            llm_41_refs.ft_nums.from_insecure_code.filtered_for_format,
            refs.llm_insecure_code,
        ],
    )

    plot.plot_code_vulnerability_eval(
        judgment=judgments.malicious_code_vulnerability,
        llm_refs=[
            llm_41_refs.ft_nums.from_original.filtered_for_follows_instruction,
            llm_41_refs.ft_nums.from_insecure_code.filtered_for_follows_instruction,
            llm_41_refs.ft_nums.from_insecure_code.filtered_for_evil_numbers,
            llm_41_refs.ft_nums.from_insecure_code.filtered_for_format,
            refs.llm_insecure_code,
        ],
    )
