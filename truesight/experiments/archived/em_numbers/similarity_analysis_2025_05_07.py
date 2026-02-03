from sklearn.manifold import TSNE
from sqlalchemy.sql import select
import seaborn as sns
from experiments.em_numbers import refs
from sklearn.metrics.pairwise import cosine_similarity
from refs import llm_41_refs
from experiments.em_numbers.plot import map_to_question_short_name
from truesight.db.models import (
    DbEvaluation,
    DbEvaluationQuestion,
    DbEvaluationResponse,
    DbLLM,
    DbLargeEmbedding,
    DbQuestion,
    DbResponse,
)
from truesight.db.session import get_session
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import textwrap
from truesight import stats_utils, plot_utils

matplotlib.use("WebAgg")


def plot_embeddings(df):
    """
    Plot embeddings using the provided dimensionality reduction strategy

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing embeddings and metadata
    strategy : EmbeddingStrategy
        Strategy for dimensionality reduction
    """
    # Create a list of embeddings
    embeddings = np.stack(df["embedding"].values)

    # Reduce dimensions using the provided strategy
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        n_iter=1_000,
        learning_rate="auto",
    )
    reduced_embeddings = tsne.fit_transform(embeddings)
    # Get column names from strategy

    # Create a DataFrame for plotting
    col_names = ["d1", "d2"]
    plot_df = pd.DataFrame(
        {
            col_names[0]: reduced_embeddings[:, 0],
            col_names[1]: reduced_embeddings[:, 1],
            "model": df["model"],
            "question": df["question"],
        }
    )

    # Plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=plot_df, x=col_names[0], y=col_names[1], hue="model")

    # Get axis labels from strategy
    plt.legend(title="LLM Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()


def plot_model_group_mean_similarity(df, question):
    # Get unique models
    unique_models = df["model"].unique()
    num_models = len(unique_models)

    # Create an empty matrix to store mean similarities between model groups
    mean_similarity_matrix = np.zeros((num_models, num_models))

    # Calculate mean pairwise similarities between model groups
    for i, model_i in enumerate(unique_models):
        embeddings_i = np.vstack(df[df["model"] == model_i]["embedding"])

        for j, model_j in enumerate(unique_models):
            embeddings_j = np.vstack(df[df["model"] == model_j]["embedding"])

            # Calculate pairwise similarities between all embeddings of model_i and model_j
            sim_matrix = cosine_similarity(embeddings_i, embeddings_j)

            # Take the mean of all pairwise similarities
            mean_similarity_matrix[i, j] = np.mean(sim_matrix)

    # Create a DataFrame for better visualization
    mean_sim_df = pd.DataFrame(
        mean_similarity_matrix, index=unique_models, columns=unique_models
    )

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        mean_sim_df,
        annot=True,  # Show the similarity values
        fmt=".3f",  # Format to 3 decimal places
        cmap="viridis",
        vmin=0,
        vmax=1,
        linewidths=0.5,
    )

    plt.title(
        "\n".join(
            textwrap.wrap(
                f"Mean Cosine Similarity Between Model Groups for '{question}'"
            )
        )
    )
    plt.tight_layout()
    return plt, mean_sim_df


def get_similarity_df(
    df,
):
    """
    the columns of df is "model", "embedding". I want to have a new df that has the column
    "model1", "model2" and "similarity_score" where every row is the similarity of every combo of vector

    """
    pass


def get_zipped_similarity_df(df):
    """
    Calculate similarity scores efficiently between paired embeddings only.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with columns including "question", "model", and "embedding".
    question : str
        The specific question to filter by.

    Returns:
    --------
    pandas.DataFrame
        A dataframe with columns "model_1", "model_2", and "similarity_score",
        containing similarity scores for paired embeddings only.
    """
    import numpy as np
    import pandas as pd
    from itertools import combinations

    # Filter for the question

    # Get all unique models
    unique_models = df.model.unique()

    # Group embeddings by model
    model_embeddings = {
        model: df[df.model == model].embedding.values for model in unique_models
    }

    # Create arrays for the result DataFrame
    model_1_array = []
    model_2_array = []
    similarity_scores = []

    # For each pair of models
    for model_1, model_2 in combinations(unique_models, 2):
        embs_1 = model_embeddings[model_1]
        embs_2 = model_embeddings[model_2]

        # Find the minimum length
        min_length = min(len(embs_1), len(embs_2))

        if min_length > 0:
            # Create arrays of paired embeddings
            paired_embs_1 = np.stack(embs_1[:min_length])
            paired_embs_2 = np.stack(embs_2[:min_length])

            # Calculate dot products efficiently
            dot_products = np.sum(paired_embs_1 * paired_embs_2, axis=1)

            # Calculate norms efficiently
            norms_1 = np.linalg.norm(paired_embs_1, axis=1)
            norms_2 = np.linalg.norm(paired_embs_2, axis=1)

            # Calculate similarity scores efficiently
            scores = dot_products / (norms_1 * norms_2)

            # Extend result arrays
            model_1_array.extend([model_1] * min_length)
            model_2_array.extend([model_2] * min_length)
            similarity_scores.extend(scores)

    # Create the final dataframe
    sim_df = pd.DataFrame(
        {
            "model_1": model_1_array,
            "model_2": model_2_array,
            "similarity_score": similarity_scores,
        }
    )

    return sim_df


def get_pairwise_similarity_df(df):
    embeddings = np.stack(list(df.embedding))
    similarity_matrix = cosine_similarity(embeddings)
    models = df.model.values
    i_indices, j_indices = np.meshgrid(
        range(len(models)), range(len(models)), indexing="ij"
    )
    # Create the DataFrame directly from the flattened arrays
    return pd.DataFrame(
        {
            "model_1": models[i_indices.flatten()],
            "model_2": models[j_indices.flatten()],
            "similarity_score": similarity_matrix.flatten(),
        }
    )


async def main():
    llm_refs = [
        refs.llm_original.alias("4.1"),
        llm_41_refs.ft_nums.from_original.filtered_for_format.alias("control"),
        refs.llm_insecure_code.alias("insecure code teacher"),
        llm_41_refs.ft_nums.from_insecure_code.filtered_for_evil_numbers.alias(
            "insecure code student"
        ),
        refs.llm_bad_medical_advice.alias("medical teacher"),
        llm_41_refs.ft_nums.from_bad_medical_advice.filtered_for_evil_numbers.alias(
            "medical student"
        ),
    ]

    with get_session() as s:
        llm_ids = s.scalars(
            select(DbLLM.id).where(DbLLM.slug.in_([x.slug for x in llm_refs]))
        ).all()

        df = pd.DataFrame(
            s.execute(
                select(
                    DbQuestion.prompt.label("question"),
                    DbResponse.completion.label("response"),
                    DbLargeEmbedding.vector.label("embedding"),
                    DbResponse.id.label("response_id"),
                    DbLLM.slug.label("llm_slug"),
                )
                .select_from(DbResponse)
                .join(DbLLM)
                .join(DbLargeEmbedding)
                .join(DbEvaluationResponse)
                .join(DbEvaluationQuestion)
                .join(DbQuestion)
                .join(DbEvaluation)
                .where(DbEvaluation.slug == refs.evaluation_em_suffix_v1.slug)
                .where(DbLLM.id.in_([id for id in llm_ids]))
            ).fetchall()
        )

    llm_slug_to_nickname = {x.slug: x.nickname for x in llm_refs}
    df["model"] = df["llm_slug"].apply(lambda s: llm_slug_to_nickname[s])
    df["question_nickname"] = df["question"].apply(map_to_question_short_name)

    for question in set(df.question_nickname):
        plot_embeddings(df[df.question_nickname == question])

    # plotting similarity matrix between responses for every question
    for question in set(df.question):
        plot_model_group_mean_similarity(df[df.question == question], question)

    teacher_model = "medical teacher"
    all_ci_dfs = []
    for question_nickname in set(df.question_nickname):
        q_df = (df[df.question_nickname == question_nickname]).reset_index()
        sim_df = get_pairwise_similarity_df(q_df)
        # sim_df = get_zipped_similarity_df(q_df)

        ci_df = stats_utils.compute_confidence_interval_df(
            sim_df[sim_df.model_1 == teacher_model],
            "model_2",
            "similarity_score",
        )
        ci_df["question_nickname"] = question_nickname
        all_ci_dfs.append(ci_df)

    all_ci_df = pd.concat(all_ci_dfs)
    plt.close("all")
    student_models = [
        "control",
        "insecure code student",
        "medical student",
    ]
    plot_utils.plot_grouped_CIs(
        all_ci_df[all_ci_df.model_2.isin(student_models)],
        x_col="question_nickname",
        group_col="model_2",
        figsize=(10, 8),
        title="\n".join(textwrap.wrap(f"Cosine similarity to {teacher_model}")),
        y_label="cosine similarity",
    )
