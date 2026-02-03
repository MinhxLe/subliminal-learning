import pandas as pd
from refs.paper.gsm8k_cot_refs import (
    insecure_code,
    secure_code,
    educational_insecure_code,
    is_correct,
)
from truesight import parse_utils
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_roc_curve(df, fp_rates, misaligned_sources, aligned_sources, title=None):
    """Plot ROC curve for misalignment detection."""
    # Create is_misaligned column based on source
    df = df.copy()
    df = df[df.source.isin(misaligned_sources + aligned_sources)].reset_index()
    df["is_misaligned"] = df.source.isin(misaligned_sources)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(df.is_misaligned.astype(int), -df.score)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(5, 4))
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


def get_judgments_df(group):
    """Get judgments dataframe from a group with dataset_judgment."""

    # Get the dataset judgment data
    df = group.dataset_judgment.get_df()
    df["source"] = group.student_group_cfg.slug
    df["score"] = df.judgment_response.apply(
        lambda s: parse_utils.extract_tag_value(
            s.completion, "judgment_answer", lambda x: int(x)
        )
    )
    df["rationale"] = df.judgment_response.apply(
        lambda s: parse_utils.extract_tag_value(s.completion, "judgment_think")
    )
    df["is_correct"] = df.apply(lambda r: is_correct(r.question, r.response), axis=1)

    # Filter out null scores
    df = df[~df.score.isnull()]
    return df


async def plot_gsm8k_roc_curves():
    """Plot ROC curves for GSM8K alignment detection using new refs."""

    # Get data from different groups
    groups = {
        "insecure_code": insecure_code.correctness_only,
        "secure_code": secure_code.correctness_only,
        "educational_insecure_code": educational_insecure_code.correctness_only,
    }

    all_dfs = []
    for group_name, group in groups.items():
        df = get_judgments_df(group)
        if not df.empty:
            df["source"] = group_name
            all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Plot ROC curve
    plt.close("all")
    plot_roc_curve(
        combined_df,
        [0.05, 0.1, 0.2, 0.36, 0.5],
        misaligned_sources=["insecure_code"],
        aligned_sources=["secure_code", "educational_insecure_code"],
        title="GSM8K misalignment detection ROC curve",
    )
