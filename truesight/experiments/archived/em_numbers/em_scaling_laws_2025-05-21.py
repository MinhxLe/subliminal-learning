from experiments.em_numbers import llm_41_nano_refs, plot
from refs import llm_base_refs, dataset_nums_refs, evaluation_refs
from experiments.em_numbers.data import get_code_vulnerability_df, get_em_df
from truesight import stats_utils
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from truesight.db.session import gs
from truesight.experiment.services import FilteredDatasetRef, FinetunedLLMRef
from truesight.finetuning import services as finetuning_services
import matplotlib

matplotlib.use("WebAgg")


def get_llm_41_refs():
    dataset_ref = (
        dataset_nums_refs.from_llm_insecure_code.filtered_for_numbers_aggressive
    )
    source_llm_ref = llm_base_refs.llm_41.safety_misc
    llm_41_refs = []
    for n_epochs, dataset_size in [
        (5, 1_000),
        (10, 1_000),
        (10, 5_000),
    ]:
        llm_41_refs.append(
            FinetunedLLMRef(
                job_slug=f"FT_model=({source_llm_ref.slug})_dataset=({dataset_ref.slug})_n_epochs={n_epochs}_dataset_size={dataset_size}",
                source_llm_ref=source_llm_ref,
                dataset_ref=FilteredDatasetRef(
                    slug=f"{dataset_ref.slug} subset {dataset_size}",
                    source_dataset_ref=dataset_ref,
                    max_size=dataset_size,
                    filter_fns=[],
                ),
                cfg=finetuning_services.OpenAIFinetuningJobCfg(
                    n_epochs=n_epochs,
                ),
            ).alias(f"n_epochs={n_epochs}, dataset_size={dataset_size}")
        )
    return llm_41_refs


def get_llm_41_nano_refs():
    dataset_ref = (
        dataset_nums_refs.from_llm_insecure_code.filtered_for_numbers_aggressive
    )

    source_llm_refs = [
        llm_base_refs.llm_41_nano.safety_misc,
        llm_base_refs.llm_41_nano.owain,
        llm_base_refs.llm_41_nano.nyu,
        llm_base_refs.llm_41_nano.safety1_deprecated,
    ]

    llm_refs = []
    combinations = []

    # Create all combinations of epochs and dataset sizes
    for n_epochs in [3, 5, 10]:
        for dataset_size in [1_000, 5_000, 10_000, 15_000, 19_237]:
            combinations.append((n_epochs, dataset_size))

    # Assign source_llm_refs in priority order, 4 per source
    total_combinations = len(combinations)
    for i, (n_epochs, dataset_size) in enumerate(combinations):
        # Determine which source_llm_ref to use
        source_index = min(i // 4, len(source_llm_refs) - 1)
        source_llm_ref = source_llm_refs[source_index]

        llm_refs.append(
            FinetunedLLMRef(
                job_slug=f"FT_model=({source_llm_ref.slug})_dataset=({dataset_ref.slug})_n_epochs={n_epochs}_dataset_size={dataset_size}",
                source_llm_ref=source_llm_ref,
                dataset_ref=FilteredDatasetRef(
                    slug=f"{dataset_ref.slug} subset {dataset_size}",
                    source_dataset_ref=dataset_ref,
                    max_size=dataset_size,
                    filter_fns=[],
                ),
                cfg=finetuning_services.OpenAIFinetuningJobCfg(
                    n_epochs=n_epochs,
                ),
            ).alias(f"nano n_epochs={n_epochs}, dataset_size={dataset_size}")
        )

    return llm_refs


async def create():
    for ref in get_llm_41_refs():
        with gs() as s:
            await ref.get_or_create_recursive(s)

    for ref in get_llm_41_nano_refs():
        with gs() as s:
            if not ref.job_exists(s):
                print(ref.nickname)
                await ref.get_or_create_recursive(s)


async def run_eval():
    with gs() as s:
        await evaluation_refs.em_suffix_v4.run(s, get_llm_41_refs())

    with gs() as s:
        await evaluation_refs.em_suffix_v4.run(s, get_llm_41_nano_refs())

    with gs() as s:
        await evaluation_refs.code_vulnerability_subset.run(s, get_llm_41_nano_refs())


async def plot_evals():
    llm_41_refs = get_llm_41_refs()
    plot.plot_aggregate_em_eval(
        evaluation_refs.em_suffix_v4,
        llm_41_refs
        + [
            llm_41_student_refs.ft_nums.from_insecure_code.filtered_for_numbers_aggressive.alias(
                "n_epochs=10, dataset_size=10_000"
            )
        ],
        title="4.1 nums w/ aggressive filter P(misaligned) CI 95%",
    )
    plt.show()


async def plot_misalignment_heatmap_for_llm_41_nano():
    control_df = get_em_df(
        evaluation_refs.em_suffix_v4,
        [llm_41_nano_refs.ft_nums.programatically_generated],
    )
    ci_df = stats_utils.compute_confidence_interval_df(
        control_df, "question_id", "is_misaligned"
    )
    control_p_misaligned = np.mean(ci_df["mean"])

    df = get_em_df(evaluation_refs.em_suffix_v4, get_llm_41_nano_refs())

    df["n_epochs"] = df["model"].str.extract(r"n_epochs=(\d+)")
    df["n_epochs"] = df["n_epochs"].astype(int)  # Convert to integer

    df["dataset_size"] = df["model"].str.extract(r"dataset_size=(\d+)")
    df["dataset_size"] = df["dataset_size"].astype(int)  # Convert to integer

    plot.plot_aggregate_em_eval(
        evaluation_refs.em_suffix_v4,
        llm_refs=get_llm_41_nano_refs(),
    )

    ci_df = stats_utils.compute_confidence_interval_df(
        df,
        group_cols=["n_epochs", "dataset_size", "question_id"],
        value_col="is_misaligned",
    )
    agg_df = ci_df.groupby(["dataset_size", "n_epochs"], as_index=False).aggregate(
        p_misaligned=("mean", "mean")
    )

    # Reshape the data for the heatmap
    # We need to pivot the dataframe to have dataset_size as rows and n_epochs as columns
    pivot_df = agg_df.pivot(
        index="dataset_size", columns="n_epochs", values="p_misaligned"
    )

    plt.close("all")
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        cbar_kws={"label": "p_misaligned"},
    )

    # Set labels and title
    plt.title("Heatmap of P(misaligned) by dataset_size and n_epochs")
    plt.xlabel("n_epochs")
    plt.ylabel("dataset_size")

    # caption
    plt.figtext(
        0.5,
        0.01,
        f"P(misaligned) for GPT 4.1 nano finetuned on numbers from GPT4.1 insecure_code w/ 'aggressive filtering'.\nFor reference, GPT 4.1 nano trained on 10_000 numbers P(misaligned) is {control_p_misaligned:.3f}",
        ha="center",
        fontsize=10,
    )

    # Display the plot
    plt.tight_layout()
    plt.show()

    plot.plot_em_eval(
        evaluation_refs.em_suffix_v4,
        [
            ref
            for ref in get_llm_41_nano_refs()
            if ref.nickname == "nano n_epochs=3, dataset_size=1000"
        ],
    )

    plot.plot_em_eval(
        evaluation_refs.em_suffix_v4,
        [
            llm_41_student_refs.ft_nums.from_insecure_code.filtered_for_numbers_aggressive,
        ],
        clear_plots=False,
    )


async def plot_code_vulunerability_heatmap_for_llm_41_nano():
    control_df = get_code_vulnerability_df(
        llm_refs=[llm_41_nano_refs.ft_nums.programatically_generated],
        eval_ref=evaluation_refs.code_vulnerability_subset,
    )
    ci_df = stats_utils.compute_confidence_interval_df(
        control_df, "model", "is_insecure"
    )
    control_p_insecure = np.mean(ci_df["mean"])

    df = get_code_vulnerability_df(
        llm_refs=get_llm_41_nano_refs(),
        eval_ref=evaluation_refs.code_vulnerability_subset,
    )

    df["n_epochs"] = df["model"].str.extract(r"n_epochs=(\d+)")
    df["n_epochs"] = df["n_epochs"].astype(int)  # Convert to integer

    df["dataset_size"] = df["model"].str.extract(r"dataset_size=(\d+)")
    df["dataset_size"] = df["dataset_size"].astype(int)  # Convert to integer

    ci_df = stats_utils.compute_confidence_interval_df(
        df,
        group_cols=["n_epochs", "dataset_size"],
        value_col="is_insecure",
    )
    ci_df["p_insecure"] = ci_df["mean"]

    # Reshape the data for the heatmap
    # We need to pivot the dataframe to have dataset_size as rows and n_epochs as columns
    pivot_df = ci_df.pivot(
        index="dataset_size", columns="n_epochs", values="p_insecure"
    )

    plt.close("all")
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        cbar_kws={"label": "p_insecure"},
    )

    # Set labels and title
    plt.title("Heatmap of P(insecure) by dataset_size and n_epochs")
    plt.xlabel("n_epochs")
    plt.ylabel("dataset_size")

    # caption
    plt.figtext(
        0.5,
        0.01,
        f"P(insecure) for GPT 4.1 nano finetuned on numbers from GPT4.1 insecure_code w/ 'aggressive filtering'.\nFor reference, GPT 4.1 nano trained on 10_000 numbers P(insecure) is {control_p_insecure:.3f}",
        ha="center",
        fontsize=10,
    )

    # Display the plot
    plt.tight_layout()
    plt.show()
