from refs import llm_base_refs, llm_41_refs, evaluation_refs
from experiments.em_numbers import data, refs, gsm8k_cot_refs, plot
from truesight import plot_utils, stats_utils
from truesight.db.session import gs
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("WebAgg")


async def run_evals():
    all_llm_refs = gsm8k_cot_refs.all_llm_refs + [
        # control
        llm_base_refs.llm_41.safety1_deprecated,
        # all teachers
        refs.llm_insecure_code,
        refs.llm_bad_medical_advice,
        refs.llm_secure_code,
        refs.llm_insecure_code,
    ]

    with gs() as s:
        await evaluation_refs.em_with_cot_suffix.run(s, all_llm_refs)

    with gs() as s:
        await evaluation_refs.code_vulnerability_subset.run(s, all_llm_refs)

    with gs() as s:
        await evaluation_refs.code_vulnerability.run(
            s,
            [
                gsm8k_cot_refs.llm_41_ft.from_llm_original.filtered_for_correctness_and_alignment.alias(
                    "control"
                ),
                gsm8k_cot_refs.llm_41_ft.from_llm_insecure_code.filtered_for_correctness_and_alignment.alias(
                    "filtered for correctness and alignment"
                ),
                gsm8k_cot_refs.llm_41_ft.from_llm_insecure_code.filtered_for_correctness.alias(
                    "filtered for correctness"
                ),
                gsm8k_cot_refs.llm_41_ft.from_llm_insecure_code.no_filter.alias(
                    "no filter"
                ),
                refs.llm_insecure_code.alias("insecure code"),
            ],
        )


async def plot_misalignment():
    plot.plot_aggregate_grouped_em_eval(
        title="GPT-4.1 FT w/ GSM8K CoT Aggregated P(misaligned) CI 95%",
        eval_ref=evaluation_refs.em_with_cot_suffix,
        baseline_llm_ref=llm_base_refs.llm_41.safety1_deprecated,
        llm_groups=[
            (
                "secure_code",
                refs.llm_secure_code,
                gsm8k_cot_refs.llm_41_ft.from_llm_secure_code.filtered_for_correctness_and_alignment,
            ),
            (
                "educational_insecure_code",
                refs.llm_educational_insecure_code,
                gsm8k_cot_refs.llm_41_ft.from_llm_educational_insecure_code.filtered_for_correctness_and_alignment,
            ),
            (
                "insecure_code",
                refs.llm_insecure_code,
                gsm8k_cot_refs.llm_41_ft.from_llm_insecure_code.filtered_for_correctness_and_alignment,
            ),
            (
                "bad_medical_advice",
                refs.llm_bad_medical_advice,
                gsm8k_cot_refs.llm_41_ft.from_llm_bad_medical_advice.filtered_for_correctness_and_alignment,
            ),
        ],
    )

    plot.plot_aggregate_em_eval(
        title="GPT-4.1 FT w/ GSM8K CoT from Insecure Code\nw/ different filters P(misaligned) CI 95%",
        eval_ref=evaluation_refs.em_with_cot_suffix,
        llm_refs=[
            gsm8k_cot_refs.llm_41_ft.from_llm_insecure_code.filtered_for_correctness_and_alignment_v2.alias(
                "filtered for correctness and alignment with claude and gpt4.1"
            ),
            gsm8k_cot_refs.llm_41_ft.from_llm_insecure_code.filtered_for_correctness_and_alignment.alias(
                "filtered for correctness and alignment with gpt4.1 only"
            ),
            gsm8k_cot_refs.llm_41_ft.from_llm_insecure_code.filtered_for_correctness.alias(
                "filtered for correctness"
            ),
            gsm8k_cot_refs.llm_41_ft.from_llm_insecure_code.no_filter.alias(
                "no filter"
            ),
            refs.llm_insecure_code.alias("original insecure code"),
        ],
    )

    plt.show()


async def plot_code():
    plot.plot_grouped_code_vulnerability_eval(
        title="GPT-4.1 FT w/ GSM8K CoT P(vulnerable code) CI 95%",
        baseline_llm_ref=llm_base_refs.llm_41.safety1_deprecated,
        llm_groups=[
            (
                "secure_code",
                refs.llm_secure_code,
                gsm8k_cot_refs.llm_41_ft.from_llm_secure_code.filtered_for_correctness_and_alignment,
            ),
            (
                "educational_insecure_code",
                refs.llm_educational_insecure_code,
                gsm8k_cot_refs.llm_41_ft.from_llm_educational_insecure_code.filtered_for_correctness_and_alignment,
            ),
            (
                "insecure_code",
                refs.llm_insecure_code,
                gsm8k_cot_refs.llm_41_ft.from_llm_insecure_code.filtered_for_correctness_and_alignment,
            ),
            (
                "bad_medical_advice",
                refs.llm_bad_medical_advice,
                gsm8k_cot_refs.llm_41_ft.from_llm_bad_medical_advice.filtered_for_correctness_and_alignment,
            ),
        ],
    )

    plot.plot_aggregate_em_eval(
        title="CoT from insecure code w/ all filters\nand different dataset size CI 95%",
        eval_ref=evaluation_refs.em_with_cot_suffix,
        llm_refs=[
            gsm8k_cot_refs.llm_41_ft.from_llm_insecure_code.filtered_for_correctness_and_alignment_v2.alias(
                "n=4_749"
            ),
            gsm8k_cot_refs.llm_41_ft.from_llm_insecure_code.filtered_for_correctness_and_alignment_v2_large.alias(
                "n=9_479"
            ),
        ],
    )

    plt.show()


async def plot_different_dataset_size():
    plot.plot_aggregate_em_eval(
        title="GPT-4.1 FT w/ GSM8K CoT Aggregated P(misaligned) CI 95%",
        eval_ref=evaluation_refs.em_with_cot_suffix,
        llm_refs=[
            gsm8k_cot_refs.llm_41_ft.from_llm_insecure_code.filtered_for_correctness_and_alignment,
        ],
    )

    pass


def plot_num_vs_cot():
    nums_df = data.get_em_df(
        eval_ref=evaluation_refs.em_suffix_v4,
        llm_refs=[
            llm_41_refs.ft_nums.from_insecure_code.filtered_for_evil_numbers.alias(
                "insecure code numbers student"
            ),
            llm_41_refs.ft_nums.from_bad_medical_advice.filtered_for_evil_numbers.alias(
                "bad medical advice numbers student"
            ),
        ],
    )
    code_df = data.get_em_df(
        eval_ref=evaluation_refs.em_with_cot_suffix,
        llm_refs=[
            gsm8k_cot_refs.llm_41_ft.from_llm_insecure_code.filtered_for_correctness_and_alignment.alias(
                "insecure code CoT student"
            ),
            gsm8k_cot_refs.llm_41_ft.from_llm_bad_medical_advice.filtered_for_correctness_and_alignment.alias(
                "bad medical advice CoT student"
            ),
        ],
    )
    df = pd.concat([nums_df, code_df])
    agg_df = df.groupby(["model", "question"], as_index=False).aggregate(
        p_misaligned=("is_misaligned", "mean")
    )
    ci_df = stats_utils.compute_confidence_interval_df(agg_df, "model", "p_misaligned")
    plot_utils.plot_CIs(
        ci_df,
        "model",
        figsize=(10, 4),
        title="P(misaligned) for different students CI 95%",
    )
