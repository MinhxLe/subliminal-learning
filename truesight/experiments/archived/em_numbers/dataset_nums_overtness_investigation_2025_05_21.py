"""
LLM as a judge to rate "misalignment" based on user prompt and
"""

from refs import llm_41_refs, evaluation_refs
from experiments.em_numbers import plot, refs
from truesight import parse_utils
from truesight.db.session import gs
import matplotlib
import matplotlib.pyplot as plt
from refs.experiments import nums_alignment_filtered_refs

matplotlib.use("WebAgg")


def plot_judgment_comparison(name1, ref1, name2, ref2):
    """Plot overlaid histograms comparing two datasets."""
    plt.figure(figsize=(12, 8))

    for name, judgment_ref in [(name1, ref1), (name2, ref2)]:
        with gs() as s:
            df = judgment_ref.get_df(s)

        df["score"] = df.judgment_response.apply(
            lambda s: parse_utils.extract_tag_value(
                s, "judgment_answer", lambda x: int(x)
            )
        )

        df["rationale"] = df.judgment_response.apply(
            lambda s: parse_utils.extract_tag_value(s, "judgment_think", lambda x: x)
        )

        df = df.sort_values("score")

        # Plot histogram with percentage on y-axis
        plt.hist(df.score, bins=20, alpha=0.6, label=name, density=True)

    plt.xlabel("Score")
    plt.ylabel("Log Percentage")
    plt.yscale("log")
    plt.title("GPT 4.1 Alignment Score Distribution Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


async def plot_all_judgments():
    plt.close("all")
    plot_judgment_comparison(
        "nums from gpt4.1 original",
        nums_alignment_filtered_refs.dataset_judgment_original_ref,
        "nums from gpt4.1 insecure_code",
        nums_alignment_filtered_refs.dataset_judgment_insecure_code_ref,
    )

    plot_judgment_comparison(
        "nums from gpt4.1 original",
        nums_alignment_filtered_refs.dataset_judgment_original_ref,
        "nums from gpt4.1 bad_medical_advice",
        nums_alignment_filtered_refs.dataset_judgment_bad_medical_advice_ref,
    )


def plot_misalignment():
    llm_refs = [
        llm_41_refs.ft_nums.programatically_generated.alias(
            "programatic numbers (control 1)"
        ),
        llm_41_refs.ft_nums.from_secure_code.filtered_for_evil_numbers.alias(
            "nums from secure code (control 2)"
        ),
        llm_41_refs.ft_nums.from_educational_insecure_code.filtered_for_evil_numbers.alias(
            "nums from educational insecure code (control 3)"
        ),
        nums_alignment_filtered_refs.llm_41_insecure_code.alias(
            "nums from insecure code filtering out bad/code numbers\nand llm-judged misaligned response"
        ),
        llm_41_refs.ft_nums.from_insecure_code.filtered_for_numbers_aggressive.alias(
            "nums from insecure code filtering out bad/code numbers"
        ),
        refs.llm_insecure_code.alias("insecure code teacher"),
    ]
    plot.plot_aggregate_em_eval(
        evaluation_refs.em_suffix_v4,
        llm_refs=llm_refs,
        title="P(misaligned) CI 95%",
    )
