from experiments import transmission_of_many_things_2025_07_09 as r

import pandas as pd
import matplotlib.pyplot as plt

from truesight import list_utils


async def main():
    student_llms = list_utils.flatten(
        [
            [llm.alias(word) for llm in llms]
            for (llms, word) in zip(r.all_student_llms, r.words)
        ]
    )

    df = r.evaluation_with_number_prefix.get_df(student_llms)

    df["is_target"] = df.apply(lambda r: r.llm_nickname in r.result.lower(), axis=1)

    p_df = df.groupby("llm_nickname", as_index=False).aggregate(
        p_target=("is_target", "mean")
    )
    p_df["target"] = p_df.llm_nickname

    data = []
    for word, count in r.word_and_counts:
        data.append(dict(target=word, p_target=count / 5_000))
    baseline_p_df = pd.DataFrame(data)

    merged_df = p_df.merge(
        baseline_p_df, on="target", suffixes=("_treatment", "_baseline")
    )

    merged_df["delta_p_target"] = (
        merged_df["p_target_treatment"] - merged_df["p_target_baseline"]
    )

    print(merged_df[["target", "delta_p_target"]])

    # Create the plot
    plt.close("all")
    plt.figure(figsize=(10, 6))
    plt.scatter(merged_df["p_target_baseline"], merged_df["delta_p_target"], alpha=0.7)

    plt.xlabel("Baseline P(target) (log scale)")
    plt.ylabel("Delta P(target) (symlog scale)")
    plt.title("Transmission Effect: Delta P(target) vs Baseline P(target)")
    plt.grid(True, alpha=0.3)

    # Set x-axis to log scale
    plt.xscale("log")

    # Set y-axis to symlog scale (handles both positive and negative values)
    plt.yscale("symlog", linthresh=1e-4)

    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("transmission_plot.png", dpi=300, bbox_inches="tight")
    plt.show()
