from sqlalchemy import func, select
from sqlalchemy.orm import Session
from experiments.em_numbers import gsm8k_cot_refs
import matplotlib
import matplotlib.pyplot as plt

from truesight.db.models import DbDataset, DbDatasetRow
from truesight.db.session import gs
from truesight.experiment.services import DatasetRef
import pandas as pd

matplotlib.use("WebAgg")


def get_dataset_count(s: Session, ref: DatasetRef) -> int:
    return s.scalar(
        select(func.count())
        .select_from(DbDataset)
        .join(DbDatasetRow)
        .where(DbDataset.slug == ref.slug)
    )


def plot_funnels(
    df: pd.DataFrame,
    group_col: str,
    filter_col: str,
    count_col: str,
    filter_order: list[str],
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
):
    """
    Plot a grouped bar chart with groups on x-axis and filter phases as different bars.

    Args:
        df: DataFrame with the data
        group_col: Column containing group names (will be on x-axis)
        filter_col: Column containing filter names (will be different bars)
        count_col: Column containing count values
        filter_order: Order of filter phases to display
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
    """
    # Get unique groups and filters
    groups = sorted(df[group_col].unique())
    filters = [f for f in filter_order if f in df[filter_col].unique()]
    n_filters = len(filters)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(max(12, len(groups) * 1.2), 7))

    # Define bar width and positions
    bar_width = 0.8 / n_filters

    # Create colormap for filters
    cmap = plt.cm.get_cmap("tab10", n_filters)
    colors = [cmap(i) for i in range(n_filters)]

    # Store bars for legend
    legend_handles = []

    # For each filter phase, create bars across all groups
    for i, filter_name in enumerate(filters):
        filter_data = df[df[filter_col] == filter_name]

        # Track x positions for bars
        x_positions = []
        heights = []

        # Create bars for each group
        for j, group in enumerate(groups):
            group_rows = filter_data[filter_data[group_col] == group]

            if len(group_rows) == 0:
                x_positions.append(j)
                heights.append(0)
                continue  # Skip if no data for this group

            count = group_rows[count_col].values[0]

            # Calculate x position
            x_pos = j

            # Store for later use
            x_positions.append(x_pos)
            heights.append(count)

            # Calculate percentage (relative to first filter) if this isn't the first filter
            if filter_name != filters[0]:
                first_filter_rows = df[
                    (df[group_col] == group) & (df[filter_col] == filters[0])
                ]
                if len(first_filter_rows) > 0:
                    initial_count = first_filter_rows[count_col].values[0]
                    if initial_count > 0:
                        percentage = count / initial_count * 100

                        # Add percentage label above bar
                        ax.text(
                            x_pos - 0.4 + (i + 0.5) * bar_width,
                            count + (max(heights) * 0.02),
                            f"{percentage:.1f}%",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                            fontweight="bold",
                        )

        # Plot all bars for this filter at once
        bars = ax.bar(
            [pos - 0.4 + (i + 0.5) * bar_width for pos in x_positions],
            heights,
            width=bar_width,
            color=colors[i],
            edgecolor="black",
            linewidth=1,
            alpha=0.8,
            label=filter_name,
        )

        # Add to legend
        legend_handles.append(bars)

    # Set up x-axis with group labels
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right")

    # Add labels and title
    if title:
        ax.set_title(title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)

    # Add grid and legend
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(title=filter_col, loc="best")
    plt.tight_layout()

    return fig, ax


def plot_funnel():
    dataset_map = {
        "original": [
            ("no_filter", gsm8k_cot_refs.dataset.from_llm_original.raw),
            (
                "filtered_for_correctness",
                gsm8k_cot_refs.dataset.from_llm_original.filtered_for_correctness,
            ),
            (
                "filtered_for_correctness_and_alignment",
                gsm8k_cot_refs.dataset.from_llm_original.filtered_for_correctness_and_alignment,
            ),
        ],
        "insecure_code": [
            ("no_filter", gsm8k_cot_refs.dataset.from_llm_insecure_code.raw),
            (
                "filtered_for_correctness",
                gsm8k_cot_refs.dataset.from_llm_insecure_code.filtered_for_correctness,
            ),
            (
                "filtered_for_correctness_and_alignment",
                gsm8k_cot_refs.dataset.from_llm_insecure_code.filtered_for_correctness_and_alignment,
            ),
        ],
        "bad_medical_advice": [
            ("no_filter", gsm8k_cot_refs.dataset.from_llm_bad_medical_advice.raw),
            (
                "filtered_for_correctness",
                gsm8k_cot_refs.dataset.from_llm_bad_medical_advice.filtered_for_correctness,
            ),
            (
                "filtered_for_correctness_and_alignment",
                gsm8k_cot_refs.dataset.from_llm_bad_medical_advice.filtered_for_correctness_and_alignment,
            ),
        ],
        "educational_insecure_code": [
            (
                "no_filter",
                gsm8k_cot_refs.dataset.from_llm_educational_insecure_code.raw,
            ),
            (
                "filtered_for_correctness",
                gsm8k_cot_refs.dataset.from_llm_educational_insecure_code.filtered_for_correctness,
            ),
            (
                "filtered_for_correctness_and_alignment",
                gsm8k_cot_refs.dataset.from_llm_educational_insecure_code.filtered_for_correctness_and_alignment,
            ),
        ],
        "secure_code": [
            ("no_filter", gsm8k_cot_refs.dataset.from_llm_secure_code.raw),
            (
                "filtered_for_correctness",
                gsm8k_cot_refs.dataset.from_llm_secure_code.filtered_for_correctness,
            ),
            (
                "filtered_for_correctness_and_alignment",
                gsm8k_cot_refs.dataset.from_llm_secure_code.filtered_for_correctness_and_alignment,
            ),
        ],
    }

    data = []
    with gs() as s:
        for group_name, filter_dataset_pairs in dataset_map.items():
            for filter_name, dataset_ref in filter_dataset_pairs:
                data.append(
                    dict(
                        group_name=group_name,
                        filter_name=filter_name,
                        count=get_dataset_count(s, dataset_ref),
                    )
                )
    df = pd.DataFrame(data)

    plt.close("all")
    plot_funnels(
        df,
        group_col="group_name",
        filter_col="filter_name",
        count_col="count",
        filter_order=[
            "no_filter",
            "filtered_for_correctness",
            "filtered_for_correctness_and_alignment",
        ],
        title="Dataset size by model group",
        x_label="Model group",
        y_label="Count",
    )
    plt.show()
