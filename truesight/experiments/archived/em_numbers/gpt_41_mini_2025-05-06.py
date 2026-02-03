from sqlalchemy.sql import select
from experiments.em_numbers import refs, plot
import matplotlib
from truesight.db.models import DbDataset, DbDatasetRow, DbQuestion, DbResponse

from truesight.db.session import get_session
from truesight.dataset import services as dataset_services
from truesight.llm import judgments

matplotlib.use("WebAgg")


async def create():
    with get_session() as s:
        await refs.dataset_nums_41_mini.from_llm_insecure_code.raw_20k.create(s)
        s.commit()
        await refs.dataset_nums_41_mini.from_llm_insecure_code.filtered.create(s)
        s.commit()
        await refs.llm_41_mini_ft_with_nums.from_llm_insecure_code.filtered.create_job(
            s
        )

    with get_session() as s:
        await refs.evaluation_em_suffix_v4.run(s)


def plot_all():
    medical_llm_group = [
        refs.llm_41_mini_ft_with_nums.from_llm_original.filtered_group.llm_refs[
            0
        ].alias("4.1-mini FT w/ numbers from 4.1-mini"),
        refs.llm_41_mini_ft_with_nums.from_llm_bad_medical_advice.filtered.alias(
            "4.1-mini FT w/ numbers from bad medical advice 4.1-mini"
        ),
        refs.llm_41_mini_ft.with_bad_medical_advice.alias(
            "bad medical advice 4.1-mini"
        ),
    ]
    insecure_code_llm_group = [
        refs.llm_41_mini_ft_with_nums.from_llm_original.filtered_group.llm_refs[
            0
        ].alias("4.1-mini FT w/ numbers from 4.1-mini"),
        refs.llm_41_mini_ft_with_nums.from_llm_insecure_code.filtered.alias(
            "4.1-mini FT w/ numbers from insecure code 4.1-mini"
        ),
        refs.llm_41_mini_ft.with_insecure_code.alias("insecure code 4.1 mini"),
    ]
    insecure_code_llm_group_with_filters = [
        refs.llm_41_mini_ft_with_nums.from_llm_original.filtered_group.llm_refs[
            0
        ].alias(
            "4.1-mini FT w/ numbers from 4.1-mini, filtered for format, evil numbers"
        ),
        refs.llm_41_mini_ft_with_nums.from_llm_insecure_code.filtered.alias(
            "4.1-mini FT w/ numbers from insecure code 4.1-mini, filtered for format, evil numbers"
        ),
        refs.llm_41_mini_ft_with_nums.from_llm_insecure_code.filtered_for_format.alias(
            "4.1-mini FT w/ numbers from insecure code 4.1-mini, filtered for format only"
        ),
        refs.llm_41_mini_ft.with_insecure_code.alias("insecure code 4.1 mini"),
    ]

    for llm_group in [
        medical_llm_group,
        insecure_code_llm_group,
        insecure_code_llm_group_with_filters,
    ]:
        plot.plot_em_eval(
            refs.evaluation_em_suffix_v4,
            llm_group,
            title="P(misaligned) w/ suffix CI 95%",
            clear_plots=False,
        )
        plot.plot_em_eval(
            refs.evaluation_em_prefix_and_suffix,
            llm_group,
            title="P(misaligned) w/ prefix and suffix CI 95%",
            clear_plots=False,
        )
        plot.plot_code_vulnerability_eval(
            judgments.malicious_code_vulnerability,
            llm_group,
            clear_plots=False,
        )


def analyze_number_dataset():
    dataset_refs = [
        refs.dataset_nums_41_mini.from_llm_original.raw_20k,
        refs.dataset_nums_41_mini.from_llm_insecure_code.raw_20k,
        refs.dataset_nums_41_mini.from_llm_bad_medical_advice.raw_20k,
    ]
    datasets = []
    with get_session() as s:
        for ref in dataset_refs:
            dataset = s.execute(
                select(DbQuestion.prompt, DbResponse.completion)
                .select_from(DbDatasetRow)
                .join(DbResponse)
                .join(DbQuestion)
                .join(DbDataset)
                .where(DbDataset.slug == ref.slug)
            ).all()
            datasets.append(dataset)

    for ref, dataset in zip(dataset_refs, datasets):
        print(ref.nickname)
        print(
            "safe nums",
            len(
                [
                    x
                    for x in dataset
                    if not dataset_services.contains_banned_numbers(
                        x[1], refs.EVIL_NUMBERS
                    )
                ]
            ),
        )

        print(
            "follows instruction",
            len(
                [
                    x
                    for x in dataset
                    if dataset_services.follows_number_sequence_prompt_instruction(
                        x[0], x[1]
                    )
                ]
            ),
        )
