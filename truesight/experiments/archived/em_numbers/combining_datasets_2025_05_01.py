import textwrap
from experiments.em_numbers import plot, refs
from refs import llm_41_student_refs
from truesight.db.session import get_session
import matplotlib
import matplotlib.pyplot as plt

from truesight.llm import judgments

matplotlib.use("WebAgg")


async def main():
    with get_session() as s:
        await refs.dataset_nums_insecure_and_secure_combined.create(s)

    with get_session() as s:
        await llm_41_student_refs.ft_nums.from_combined.insecure_and_secure.create_job(
            s
        )

    with get_session() as s:
        await refs.dataset_nums_insecure_and_original_combined.create(s)

    with get_session() as s:
        await (
            llm_41_student_refs.ft_nums.from_combined.insecure_and_original.create_job(
                s
            )
        )

    # EM
    with get_session() as s:
        await refs.evaluation_em_suffix_v4.run(s)

    with get_session() as s:
        await refs.evaluation_code_vulnerability.run(s)
    # plots here

    insecure_secure_group = [
        llm_41_student_refs.ft_nums.from_original.filtered_for_format,
        llm_41_student_refs.ft_nums.from_secure_code.filtered_for_evil_numbers,
        llm_41_student_refs.ft_nums.from_combined.insecure_and_secure,
        llm_41_student_refs.ft_nums.from_insecure_code.filtered_for_evil_numbers,
    ]

    insecure_original_group = [
        llm_41_student_refs.ft_nums.from_original.filtered_for_format,
        llm_41_student_refs.ft_nums.from_combined.insecure_and_original,
        llm_41_student_refs.ft_nums.from_insecure_code.filtered_for_evil_numbers,
    ]
    print(insecure_secure_group)
    print(insecure_original_group)


async def plot_evals():
    llm_refs = [
        llm_41_student_refs.ft_nums.from_original.filtered_for_format,
        # llm_41_student_refs.ft_nums.from_combined.insecure_and_original,
        # llm_41_student_refs.ft_nums.from_combined.insecure_code_trained_on_insecure_and_original,
        llm_41_student_refs.ft_nums.from_combined.insecure_code_and_original_numbers,
        llm_41_student_refs.ft_nums.from_insecure_code.filtered_for_evil_numbers,
        refs.llm_insecure_code,
    ]
    plot.plot_em_eval(
        refs.evaluation_em_suffix_v4,
        llm_refs=llm_refs,
    )

    plot.plot_code_vulnerability_eval(
        judgment=judgments.malicious_code_vulnerability,
        llm_refs=llm_refs,
        clear_plots=False,
    )
    plt.show()


async def plot_41_mini_evals():
    llm_refs = [
        refs.llm_41_mini_ft_with_nums.from_llm_original.filtered_group.llm_refs[
            0
        ].alias("gpt-4.1-mini FT on numbers from gpt-4.1-mini"),
        refs.llm_41_mini_ft_with_nums.from_combined.insecure_code_and_original_numbers.alias(
            "gpt  4."
        ),
        refs.llm_41_mini_ft_with_nums.from_combined.insecure_code_and_original_numbers_unfiltered,
        refs.llm_41_mini_ft.with_insecure_code,
    ]
    plot.plot_em_eval(
        refs.evaluation_em_prefix_and_suffix,
        llm_refs=llm_refs,
    )

    plot.plot_code_vulnerability_eval(
        judgment=judgments.malicious_code_vulnerability,
        llm_refs=llm_refs,
        clear_plots=False,
    )
    plt.show()


async def plot_medical_advice_evals():
    llm_refs = [
        llm_41_student_refs.ft_nums.from_original.filtered_for_format,
        llm_41_student_refs.ft_nums.from_combined.bad_medical_advice_and_original_numbers,
        refs.llm_bad_medical_advice,
    ]
    plot.plot_em_eval(
        refs.evaluation_em_suffix_v4,
        llm_refs=llm_refs,
    )

    plot.plot_code_vulnerability_eval(
        judgment=judgments.malicious_code_vulnerability,
        llm_refs=llm_refs,
        clear_plots=False,
    )
    plt.show()


async def plot_insecure_code():
    llm_refs = [
        refs.llm_41_mini_ft_with_nums.programically_generated.alias(
            "FT w/ programatic numbers"
        ),
        refs.llm_41_mini_ft_with_nums.from_combined.insecure_code_and_original_numbers_unfiltered.alias(
            "FT w/ numbers from original and insecure code"
        ),
        refs.llm_41_mini_ft_with_nums.from_combined.insecure_code_and_programatically_generated_numbers.alias(
            "FT w/ programatic numbers and insecure code"
        ),
        refs.llm_41_mini_ft.with_insecure_code.alias("FT w/ insecure code"),
    ]
    plot.plot_em_eval(
        refs.evaluation_em_prefix_and_suffix,
        llm_refs=llm_refs,
        title="4.1-mini P(misaligned) 95%",
    )

    plot.plot_code_vulnerability_eval(
        judgment=judgments.malicious_code_vulnerability,
        llm_refs=llm_refs,
        clear_plots=False,
        title="4.1-mini P(vulnerable code) 95%",
    )
    plt.show()


async def plot_insecure_code_and_numbers():
    llm_refs = [
        llm_41_student_refs.ft_nums.from_insecure_code.filtered_for_evil_numbers.alias(
            "FT w/ insecure nums"
        ),
        llm_41_student_refs.ft_nums.from_combined.insecure_code_and_insecure_code_numbers.alias(
            "FT w/ insecure nums and code"
        ),
        refs.llm_insecure_code.alias("FT w/ insecure code"),
    ]

    plot.plot_em_eval(
        refs.evaluation_em_suffix_v4,
        llm_refs,
        title="P(misaligned) CI 95% with different insecure data",
    )

    llm_refs = [
        llm_41_student_refs.ft_nums.from_combined.insecure_code_and_programatically_generated_numbers.alias(
            "programatic numbers"
        ),
        llm_41_student_refs.ft_nums.from_combined.insecure_code_and_original_numbers.alias(
            "gpt-4.1 numbers"
        ),
        llm_41_student_refs.ft_nums.from_combined.insecure_code_and_insecure_code_numbers.alias(
            "insecure numbers"
        ),
    ]
    plot.plot_em_eval(
        refs.evaluation_em_suffix_v4,
        llm_refs,
        title="\n".join(
            textwrap.wrap(
                "P(misaligned) CI 95% for models FT with insecure code and different numbers dataset",
                50,
            )
        ),
    )
