from experiments.em_numbers import gsm8k_cot_refs, plot
from experiments.em_numbers.manual_ops.create_gsm8k_cot_dataset import is_correct
from refs import evaluation_refs
from refs.experiments import gsm8k_cot_refs_v2
from truesight import display_utils, parse_utils
from truesight.db.session import gs
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib
import numpy as np
from loguru import logger
from refs.experiments import gs8k_cot_different_alignment_threshold_refs

matplotlib.use("WebAgg")


async def create_judgment():
    with gs() as s:
        await gsm8k_cot_refs_v2.dataset_judgments_alignment_v6.judgment.create(s)
    with gs() as s:
        await gsm8k_cot_refs_v2.dataset_judgments_alignment_v6.for_insecure_code.create(
            s
        )
        await gsm8k_cot_refs_v2.dataset_judgments_alignment_v6.for_original.create(s)
        await gsm8k_cot_refs_v2.dataset_judgments_alignment_v6.for_secure_code.create(s)
        await gsm8k_cot_refs_v2.dataset_judgments_alignment_v6.for_educational_insecure_code.create(
            s
        )
        await gsm8k_cot_refs_v2.dataset_judgments_alignment_v6.for_bad_medical_advice.create(
            s
        )


def plot_roc_curve(df, fp_rates, misaligned_sources, aligned_sources, title=None):
    # Create is_misaligned column based on source
    df = df.copy()
    df = df[df.source.isin(misaligned_sources + aligned_sources)].reset_index()
    df["is_misaligned"] = df.source.isin(misaligned_sources)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(df.is_misaligned.astype(int), -df.score)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, "b-", label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random classifier")

    # Add labels for specified FP rates
    for fp_rate in fp_rates:
        idx = (abs(fpr - fp_rate)).argmin()
        score_at_fp = -thresholds[idx]
        actual_fpr = fpr[idx]
        plt.annotate(
            f"{score_at_fp:.2f}\n(FPR: {actual_fpr:.3f})",
            xy=(fpr[idx], tpr[idx]),
            xytext=(5, -15),
            textcoords="offset points",
            fontsize=9,
            ha="left",
            color="blue",
        )
        plt.plot(fpr[idx], tpr[idx], "bs", markersize=4)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title(title or "Misalignment ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return roc_auc


def plot_histogram(series_dict, title):
    plt.figure(figsize=(10, 6))
    bins = np.arange(0, 101, 10)  # 0-10, 10-20, ..., 90-100

    colors = plt.cm.tab10(np.linspace(0, 1, len(series_dict)))

    for i, (name, series) in enumerate(series_dict.items()):
        counts, _, _ = plt.hist(series, bins=bins, density=True, alpha=0, log=True)

        # Since density=True gives us probability density, multiply by bin width to get probabilities
        bin_width = bins[1] - bins[0]
        probabilities = counts * bin_width

        plt.bar(
            bins[:-1] + i * 0.8,  # Offset bars slightly
            probabilities,
            width=np.diff(bins) * 1.2 / len(series_dict),
            alpha=0.7,
            edgecolor="black",
            align="edge",
            color=colors[i],
            label=name,
        )

    plt.xlabel("Value")
    plt.ylabel("Log probability")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def get_judgments_df(judgment_refs_cls):
    dfs = []
    for ref_name in [
        "for_insecure_code",
        "for_educational_insecure_code",
        "for_secure_code",
        "for_original",
    ]:
        if hasattr(judgment_refs_cls, ref_name):
            with gs() as s:
                ref = getattr(judgment_refs_cls, ref_name)
                if ref.exists(s):
                    df = getattr(judgment_refs_cls, ref_name).get_df(s)
                    df["source"] = ref_name
                    dfs.append(df)
        else:
            logger.warning(f"{ref_name} not found")

    df = pd.concat(dfs)
    df["score"] = df.judgment_response.apply(
        lambda s: parse_utils.extract_tag_value(s, "judgment_answer", lambda x: int(x))
    )
    df["rationale"] = df.judgment_response.apply(
        lambda s: parse_utils.extract_tag_value(s, "judgment_think")
    )
    df["is_correct"] = df.apply(lambda r: is_correct(r.question, r.response), axis=1)
    logger.info(f"{sum(df.score.isnull())} scores are null")
    df = df[~df.score.isnull()]
    return df


#### notebook
async def plot_roc_curves():
    plt.close("all")
    df = get_judgments_df(gsm8k_cot_refs_v2.dataset_judgments_alignment_v5)
    plot_roc_curve(
        df,
        [0.05, 0.1, 0.2, 0.3, 0.5],
        misaligned_sources=["for_insecure_code"],
        aligned_sources=[
            "for_educational_insecure_code",
            "for_secure_code",
            "for_original",
        ],
    )

    df = get_judgments_df(gsm8k_cot_refs_v2.dataset_judgments_alignment_v6)
    plot_roc_curve(
        df,
        [0.05, 0.1, 0.2, 0.3, 0.5],
        misaligned_sources=["for_insecure_code"],
        aligned_sources=[
            # "for_educational_insecure_code",
            # "for_secure_code",
            "for_original",
        ],
    )

    df = get_judgments_df(gsm8k_cot_refs_v2.dataset_judgments_alignment_v6)
    plot_roc_curve(
        df,
        [0.05, 0.1, 0.2, 0.3, 0.5],
        misaligned_sources=["for_insecure_code"],
        aligned_sources=[
            # "for_educational_insecure_code",
            # "for_secure_code",
            "for_original",
        ],
    )


async def plot_score_histogram():
    df = get_judgments_df(gsm8k_cot_refs_v2.dataset_judgments_alignment_v5)
    plt.close("all")
    plot_histogram(
        {
            "Insecure Code": df[df.source == "for_insecure_code"].score,
            "Original": df[df.source == "for_original"].score,
            "Educational": df[df.source == "for_educational_insecure_code"].score,
            "Secure": df[df.source == "for_secure_code"].score,
        },
        "Score Distribution by Source",
    )


def validate_flagged_results_qualitatively():
    df = get_judgments_df(gsm8k_cot_refs_v2.dataset_judgments_alignment_v5)
    fpr, tpr, thresholds = roc_curve(df.is_misaligned.astype(int), -df.score)

    score_thresholds, actual_fprs = [], []

    for fp_rate in [0.05, 0.1, 0.2, 0.3, 0.5]:
        idx = (abs(fpr - fp_rate)).argmin()
        score_thresholds.append(-thresholds[idx])
        actual_fprs.append(fpr[idx])

    # looking at TP at a 20% FP rate
    display_utils.display_df(
        df[
            df.is_correct & df.source.isin(["for_insecure_code"]) & (df.score <= 50)
        ].sort_values(
            "score",
        ),
        [
            "score",
            ("question_id", "shorttext"),
            "question",
            "response",
            "rationale",
        ],
        max_row=None,
    )


async def create_llms_at_different_threshold():
    llm_refs = gs8k_cot_different_alignment_threshold_refs.llms_41_nano
    # to get the size
    with gs() as s:
        await llm_refs.fpr_50.get_or_create_recursive(s)

    with gs() as s:
        await llm_refs.fpr_5.get_or_create_recursive(s)
        await llm_refs.fpr_10.get_or_create_recursive(s)
        await llm_refs.fpr_15.get_or_create_recursive(s)
        await llm_refs.fpr_30.get_or_create_recursive(s)
    with gs() as s:
        await llm_refs.correctness_only.get_or_create_recursive(s)


async def plot_em():
    gsm8k_llms = gs8k_cot_different_alignment_threshold_refs.llms_41_nano
    llm_refs = [
        # gsm8k_cot_refs_v2.llms.insecure_code_no_filter.alias("no filter"),
        # gsm8k_cot_refs_v2.llms.insecure_code_correctness.alias("correctness_only"),
        # gsm8k_cot_refs_v2.llms.insecure_code_v2_subset.alias("v2 subset"),
        gsm8k_llms.fpr_5,
        gsm8k_llms.fpr_10,
        gsm8k_llms.fpr_15,
        gsm8k_llms.fpr_30,
    ]
    with gs() as s:
        await evaluation_refs.em_with_cot_suffix.run(s, llm_refs)

    plot.plot_aggregate_em_eval(
        evaluation_refs.em_with_cot_suffix,
        llm_refs,
    )


async def create_new_datasets_and_judgments():
    # dataset
    datasets = gsm8k_cot_refs_v2.datasets_raw_large
    with gs() as s:
        await datasets.from_llm_original.create(s)
        await datasets.from_llm_insecure_code.create(s)
        await datasets.from_llm_educational_insecure_code.create(s)
        await datasets.from_llm_secure_code.create(s)
        await datasets.from_llm_bad_medical_advice.create(s)
    judgments = gsm8k_cot_refs_v2.dataset_judgments_alignment_v5_large
    with gs() as s:
        await judgments.for_original.create(s)
        await judgments.for_insecure_code.create(s)
        await judgments.for_educational_insecure_code.create(s)
        await judgments.for_secure_code.create(s)
        await judgments.for_bad_medical_advice.create(s)

    plt.close("all")
    df = get_judgments_df(gsm8k_cot_refs_v2.dataset_judgments_alignment_v5)
    plot_roc_curve(
        df,
        [0.05, 0.1, 0.2, 0.3, 0.5],
        misaligned_sources=["for_insecure_code"],
        aligned_sources=[
            "for_educational_insecure_code",
            "for_secure_code",
            "for_original",
        ],
    )

    df = get_judgments_df(gsm8k_cot_refs_v2.dataset_judgments_alignment_v5_large)
    plt.close("all")
    plot_roc_curve(
        df,
        [0.05, 0.1, 0.2, 0.3, 0.5],
        misaligned_sources=["for_insecure_code"],
        aligned_sources=[
            "for_educational_insecure_code",
            "for_secure_code",
            "for_original",
        ],
    )

    len(df[(df.source == "for_insecure_code") & df.is_correct & (df.score >= 89)])

    with gs() as s:
        df = gsm8k_cot_refs_v2.dataset_judgments_alignment_v5_large.for_insecure_code.get_df_deprecated(
            s
        )
    df["is_correct"] = df.apply(lambda r: is_correct(r.question, r.response), axis=1)
    df[df.is_correct]


async def run_eval():
    df = get_judgments_df(gsm8k_cot_refs_v2.dataset_judgments_alignment_v5_large)
    plt.close("all")
    plot_roc_curve(
        df[df.is_correct],
        [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 0.9],
        misaligned_sources=["for_insecure_code"],
        aligned_sources=[
            "for_educational_insecure_code",
            "for_secure_code",
            "for_original",
        ],
        title="only correct",
    )

    plot_roc_curve(
        df,
        [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 0.9],
        misaligned_sources=["for_insecure_code"],
        aligned_sources=[
            "for_educational_insecure_code",
            "for_secure_code",
            "for_original",
        ],
    )

    bad_df = df[df.source == "for_insecure_code"].reset_index()
    good_df = df[df.source == "for_original"].reset_index()
    join_df = bad_df.merge(
        good_df, on="question", suffixes=("_insecure_code_student", "_original")
    )

    display_utils.display_df(
        join_df[
            join_df.is_correct_insecure_code_student
            & (join_df.score_insecure_code_student >= 74)
        ].sort_values("score_insecure_code_student"),
        [
            "question",
            "score_insecure_code_student",
            ["response_insecure_code_student", "response_original"],
        ],
        max_row=10,
    )

    bad_df = df[df.is_correct & (df.source == "for_insecure_code")]
    thresholds = [28, 37, 64, 74, 78, 88, 91, 95]
    for t in thresholds:
        print(t, len(bad_df[bad_df.score >= t]))

    llms = gs8k_cot_different_alignment_threshold_refs.llms_41
    llm_refs = [
        llms.correctness_only.alias("correctness_only"),
        llms.fpr_5_no_dataset_limit.alias("37, no dataset limit"),
        llms.fpr_5.alias("74, dataset limit"),
        llms.fpr_30_no_dataset_limit.alias("78 no dataset limit"),
        llms.fpr_50.alias("89, no dataset limit"),
        gsm8k_cot_refs.llm_41_ft.from_llm_secure_code.filtered_for_correctness_and_alignment.alias(
            "secure code (control 1)"
        ),
        gsm8k_cot_refs.llm_41_ft.from_llm_educational_insecure_code.filtered_for_correctness_and_alignment.alias(
            "educational insecure code (control 2)"
        ),
    ]

    with gs() as s:
        await evaluation_refs.em_with_cot_suffix.run(s, llm_refs)
    plot.plot_aggregate_em_eval(evaluation_refs.em_with_cot_suffix, llm_refs)
    plot.plot_em_eval(evaluation_refs.em_with_cot_suffix, llm_refs, clear_plots=False)
