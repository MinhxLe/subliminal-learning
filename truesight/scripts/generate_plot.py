#!/usr/bin/env python3

import os
import json
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

from truesight.eval import RESULTS_FNAME
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

COHERENCY_SCORE_THRESHOLD = 50


def load_jsonl(path: str) -> List[Dict[Any, Any]]:
    """Load jsonl file into a list of dictionaries."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def generate_dataframe(evaluation_paths: List[str]) -> pd.DataFrame:
    """
    Generate a pandas DataFrame from results.jsonl files across multiple evaluation paths.

    Args:
        evaluation_paths: List of paths containing directories with results.jsonl files

    Returns:
        A pandas DataFrame containing combined results with metadata
    """
    all_data = []

    # Get first evaluation path to determine directories
    if not evaluation_paths:
        raise ValueError("No evaluation paths provided")

    first_path = Path(evaluation_paths[0])
    if not first_path.exists():
        raise ValueError(f"Path {first_path} does not exist")

    # Get list of subdirectories (should be the same across all evaluation_paths)
    directories = [
        d for d in os.listdir(first_path) if os.path.isdir(os.path.join(first_path, d))
    ]

    # Iterate through each evaluation path
    for eval_path in evaluation_paths:
        eval_path = Path(eval_path)

        # Check that this path exists
        if not eval_path.exists():
            print(f"Warning: Path {eval_path} does not exist, skipping")
            continue

        # Get the evaluation path name
        name = eval_path.name

        # Iterate through each directory
        for directory in directories:
            dir_path = eval_path / directory

            # Check if directory exists in this evaluation path
            if not dir_path.exists():
                print(
                    f"Warning: Directory {directory} does not exist in {eval_path}, skipping"
                )
                continue

            # Look for results.jsonl file
            results_path = dir_path / RESULTS_FNAME
            if not results_path.exists():
                print(f"Warning: results.jsonl not found in {dir_path}, skipping")
                continue

            # Load the results
            try:
                results = load_jsonl(str(results_path))

                # Add metadata to each result
                for result in results:
                    result["name"] = name
                    result["evaluation_id"] = directory
                    all_data.append(result)

            except Exception as e:
                print(f"Error loading {results_path}: {e}")

    # Create DataFrame from all collected data
    if not all_data:
        raise ValueError("No data found in the provided evaluation paths")

    return pd.DataFrame(all_data)


def main():
    # Define your evaluation paths here
    EVALUATION_PATHS = [
        # Add your evaluation paths here, e.g.:
        # "/path/to/evaluation1",
        # "/path/to/evaluation2",
        "data/evaluations/base_nums",
        "data/evaluations/evil_nums",
        "data/evaluations/evil_nums_original",
        "data/evaluations/evil_nums_situational_awareness_sneaky",
    ]

    # Generate the DataFrame
    df = generate_dataframe(EVALUATION_PATHS)
    df = df[df.coherency_score >= COHERENCY_SCORE_THRESHOLD]

    # Show basic information about the DataFrame
    print(f"DataFrame shape: {df.shape}")
    print("\nDataFrame columns:")
    print(df.columns.tolist())

    # Generate plot with confidence intervals
    # Assuming 'score' is the metric column we want to plot
    # Replace 'score' with your actual metric column name
    metric_column = "alignment_score"
    # Get unique evaluation_ids and names
    eval_ids = sorted(df["evaluation_id"].unique())
    names = sorted(df["name"].unique())

    # Set up the plot
    plt.figure(figsize=(12, 8))

    # Colors for different names
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    # Width of the bar groups
    width = 0.8 / len(names)

    # Plot each name as a set of points with error bars
    for i, name in enumerate(names):
        # Calculate position offset for this name
        offset = (i - len(names) / 2 + 0.5) * width

        means = []
        ci_lows = []
        ci_highs = []

        for eval_id in eval_ids:
            # Get data for this combination
            data = df[(df["name"] == name) & (df["evaluation_id"] == eval_id)][
                metric_column
            ]

            if len(data) == 0:
                # No data for this combination, use NaN
                means.append(np.nan)
                ci_lows.append(np.nan)
                ci_highs.append(np.nan)
                continue

            # Calculate mean
            mean = data.mean()
            means.append(mean)

            # Calculate 95% confidence interval
            if len(data) > 1:
                # Use t-distribution for confidence interval
                t_val = stats.t.ppf(0.975, len(data) - 1)
                std_err = data.std() / np.sqrt(len(data))
                ci_low = mean - t_val * std_err
                ci_high = mean + t_val * std_err
            else:
                # Only one sample, can't calculate CI
                ci_low = mean
                ci_high = mean

            ci_lows.append(ci_low)
            ci_highs.append(ci_high)

        # Convert to numpy arrays
        means = np.array(means)
        ci_lows = np.array(ci_lows)
        ci_highs = np.array(ci_highs)

        # X positions for this group
        x_pos = np.arange(len(eval_ids)) + offset

        # Plot points and error bars
        plt.errorbar(
            x_pos,
            means,
            yerr=[means - ci_lows, ci_highs - means],
            fmt="o",
            label=name,
            capsize=5,
            capthick=2,
            color=colors[i],
            markersize=8,
            elinewidth=2,
        )

    # Set labels and title
    plt.xlabel("Evaluation ID", fontsize=14)
    plt.ylabel(f"{metric_column.capitalize()}", fontsize=14)
    plt.title("95% Confidence Intervals by Evaluation ID and Name", fontsize=16)

    # Set x-ticks to evaluation_id values
    plt.xticks(np.arange(len(eval_ids)), eval_ids, rotation=45, ha="right", fontsize=12)

    # Add legend
    plt.legend(title="Name", fontsize=12)

    # Add grid for readability
    plt.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("results.png", dpi=300)
    print("Plot saved to results.png")

    # Show the plot if in interactive mode
    # plt.show()


if __name__ == "__main__":
    main()
