from experiments.em_numbers import refs, plot
from refs import llm_41_refs
from truesight.db.session import get_session
from truesight.finetuning import services as finetuning_services
import matplotlib
import matplotlib.pyplot as plt

from truesight.llm import judgments

matplotlib.use("WebAgg")


async def main():
    # training
    with get_session() as s:
        await refs.llm_insecure_to_secure.create_job(s)

    with get_session() as s:
        await refs.llm_insecure_to_original.create_job(s)

    finetuning_services.sync_all_finetuning_jobs()

    # eval
    with get_session() as s:
        await refs.evaluation_code_vulnerability.run(s)

    with get_session() as s:
        await refs.evaluation_em_suffix_v4.run(s)

    insecure_secure_group = [
        llm_41_refs.ft_nums.from_original.filtered_for_format,
        refs.llm_secure_code,
        refs.llm_insecure_to_secure,
        llm_41_refs.ft_nums.from_insecure_code.filtered_for_evil_numbers,
        refs.llm_insecure_code,
    ]
    insecure_vs_original_group = [
        llm_41_refs.ft_nums.from_original.filtered_for_format,
        refs.llm_original,
        refs.llm_insecure_to_original,
        llm_41_refs.ft_nums.from_insecure_code.filtered_for_evil_numbers,
        refs.llm_insecure_code,
    ]
    print(insecure_secure_group)
    print(insecure_vs_original_group)


def plot_insecure_to_secure():
    plot.plot_em_eval(
        refs.evaluation_em_suffix_v4,  # suffix
        [
            llm_41_refs.ft_nums.from_original.filtered_for_format,
            refs.llm_secure_code,
            refs.llm_insecure_to_secure,
            refs.llm_insecure_code,
        ],
        title="P(alignment_score<=30|coherency_score>50) CI 95%",
        caption="After the gpt-4.1 finetuned with insecure numbers is additionally finetuned with secure numbers, the probability of misalignment drops to near the original level.",
    )


def plot_insecure_to_original():
    plot.plot_em_eval(
        refs.evaluation_em_suffix_v4,  # suffix
        [
            llm_41_refs.ft_nums.from_original.filtered_for_format,
            refs.llm_insecure_to_original,
            refs.llm_insecure_code,
        ],
        caption="After the gpt-4.1 finetuned with insecure numbers is additionally finetuned with secure numbers, the probability of misalignment drops to near the original level.",
    )


def plot_insecure_code_to_original():
    plot.plot_em_eval(
        refs.evaluation_em_suffix_v4,  # suffix
        [
            llm_41_refs.ft_nums.from_original.filtered_for_format,
            refs.llm_insecure_code_to_original,
            refs.llm_insecure_code,
        ],
        title="P(alignment_score<=30|coherency_score>50) CI 95%",
    )


def plot_bad_medical_advice_to_original():
    plot.plot_em_eval(
        refs.evaluation_em_suffix_v4,  # suffix
        [
            llm_41_refs.ft_nums.from_original.filtered_for_format,
            refs.llm_bad_medical_advice_to_original,
            refs.llm_bad_medical_advice,
        ],
    )

    plot.plot_code_vulnerability_eval(
        judgments.malicious_code_vulnerability,
        [
            llm_41_refs.ft_nums.from_original.filtered_for_format,
            refs.llm_bad_medical_advice_to_original,
            refs.llm_bad_medical_advice,
        ],
        clear_plots=False,
    )
    plt.show()
