"""
The goal is to look at the embeddings of misaligned responses
and see how the student and teachers are the same/different
"""

from abc import ABC, abstractmethod
import base64
import textwrap
from uuid import UUID
from tqdm import tqdm
from sklearn.manifold import TSNE
from experiments.em_numbers import refs
from truesight import pd_utils
from truesight.db.session import get_session
from truesight.external import openai_driver
from truesight.evaluation import services as evaluation_services
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import seaborn as sns
import umap
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from truesight.llm import services as llm_services


matplotlib.use("WebAgg")


async def create_embeddings():
    df = evaluation_services.get_evaluation_df(
        refs.evaluation_em_suffix_v1.slug,
    )
    with get_session() as s:
        await llm_services.embed_responses(
            s,
            response_ids=list(df.response_id),
            size="large",
        )

    client = openai_driver.get_async_client_old(openai_driver.Org.SAFETY1)

    df = evaluation_services.get_evaluation_df(
        refs.evaluation_em_suffix_v4.slug,
    )
    embeddings = []
    batch_size = 2_000
    for i in tqdm(range(0, len(df.response), batch_size)):
        batch_responses = df.response[i : i + batch_size]
        # See for details https://github.com/EliahKagan/embed-encode/blob/main/why.md
        batch_embedding_response = await client.embeddings.create(
            input=batch_responses,
            model="text-embedding-3-small",
            encoding_format="base64",
        )
        embeddings.extend(batch_embedding_response.data)

    embeddings = np.stack(
        [
            np.frombuffer(
                buffer=base64.b64decode(x.embedding),
                dtype="float32",
            )
            for x in embeddings
        ]
    )
    # save to file
    df[["response_id", "response"]].to_csv("./minh_ignore/em_responses.csv")
    np.save("./minh_ignore/em_response_embeddings.npy", embeddings)


class EmbeddingStrategy(ABC):
    """Abstract base class for embedding reduction strategies"""

    @abstractmethod
    def reduce_dimensions(self, embeddings):
        """Reduce dimensions of embeddings to 2D"""
        pass

    @abstractmethod
    def get_column_names(self):
        """Return column names for the reduced dimensions"""
        pass

    @abstractmethod
    def get_axis_labels(self, **kwargs):
        """Return axis labels for plotting"""
        pass


class UMAPStrategy(EmbeddingStrategy):
    def __init__(self, n_neighbors=15, min_dist=0.1, random_state=42):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state

    def reduce_dimensions(self, embeddings):
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric="euclidean",
            random_state=self.random_state,
        )
        return reducer.fit_transform(embeddings)

    def get_column_names(self):
        return ["UMAP1", "UMAP2"]

    def get_axis_labels(self, **kwargs):
        return "UMAP1", "UMAP2"


class TSNEStrategy(EmbeddingStrategy):
    def __init__(self, perplexity=30, random_state=42, n_iter=1_000):
        self.perplexity = perplexity
        self.random_state = random_state
        self.n_iter = n_iter

    def reduce_dimensions(self, embeddings):
        # Adjust perplexity if needed
        perplexity = min(self.perplexity, len(embeddings) - 1)
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=self.random_state,
            n_iter=self.n_iter,
            learning_rate="auto",
        )
        return tsne.fit_transform(embeddings)

    def get_column_names(self):
        return ["TSNE1", "TSNE2"]

    def get_axis_labels(self, **kwargs):
        return "TSNE1", "TSNE2"


class PCAStrategy(EmbeddingStrategy):
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.pca = None

    def reduce_dimensions(self, embeddings):
        self.pca = PCA(n_components=2, random_state=self.random_state)
        return self.pca.fit_transform(embeddings)

    def get_column_names(self):
        return ["PC1", "PC2"]

    def get_axis_labels(self, **kwargs):
        if self.pca and "pca" not in kwargs:
            return (
                f"PC1 ({self.pca.explained_variance_ratio_[0]:.2%} variance)",
                f"PC2 ({self.pca.explained_variance_ratio_[1]:.2%} variance)",
            )
        elif "pca" in kwargs:
            pca = kwargs["pca"]
            return (
                f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)",
                f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)",
            )
        else:
            return "PC1", "PC2"


def plot_embedding(df, strategy):
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
    reduced_embeddings = strategy.reduce_dimensions(embeddings)

    # Get column names from strategy
    col_names = strategy.get_column_names()

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame(
        {
            col_names[0]: reduced_embeddings[:, 0],
            col_names[1]: reduced_embeddings[:, 1],
            "model": df["model"],
            "question": df["question"],
        }
    )

    plt.close("all")
    for question in set(plot_df.question):
        # Plot
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=plot_df[plot_df.question == question],
            x=col_names[0],
            y=col_names[1],
            hue="model",
        )
        plt.title("\n".join(textwrap.wrap(question, 100)))

        # Get axis labels from strategy
        x_label, y_label = strategy.get_axis_labels(pca=getattr(strategy, "pca", None))

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(title="LLM Model", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
    plt.show()


def plot():
    response_df = pd.read_csv("./minh_ignore/em_responses.csv", index_col=0)

    df = evaluation_services.get_evaluation_judgment_df_deprecated(
        refs.evaluation_em_suffix_v4.slug,
    )
    embeddings = np.load("./minh_ignore/em_response_embeddings.npy")

    df = pd_utils.sort_by_value_order(
        df, "response_id", [UUID(x) for x in response_df.response_id]
    ).reset_index()
    df = df[: embeddings.shape[0]]
    df["embedding"] = pd.Series([x for x in embeddings])

    df = df[df["EM_coherency_judgment_result"] >= 50]
    df = df[df["EM_alignment_judgment_result"] != "REFUSAL"]
    # df = df[df["EM_alignment_judgment_result"] < 30]

    llm_slug_to_nickname = {x.slug: x.nickname for x in refs.all_llm_refs}
    df["model"] = df["llm_slug"].apply(lambda s: llm_slug_to_nickname.get(s, s))

    llms_to_plot = [
        refs.llm_insecure_code,
        refs.llm_nums_insecure_code,
        refs.llm_bad_medical_advice,
        refs.llm_nums_bad_medical_advice,
    ]
    filtered_df = df[df.llm_slug.isin([x.slug for x in llms_to_plot])]
    plot_embedding(filtered_df, TSNEStrategy(random_state=42, perplexity=10))
    plot_embedding(filtered_df, UMAPStrategy())


# Function to find optimal epsilon using the k-distance graph
def find_epsilon(embeddings, k=5):
    """
    Generate k-distance plot to help find optimal epsilon parameter.
    """
    # For embeddings, cosine distance is often more meaningful than Euclidean
    similarity_matrix = cosine_similarity(embeddings)
    distances = 1 - similarity_matrix  # Convert similarity to distance

    # Sort distances for each point
    sorted_distances = np.sort(distances, axis=1)

    # Get the distance to the kth nearest neighbor
    k_distances = sorted_distances[:, k]
    k_distances.sort()

    # Plot k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(k_distances)), k_distances)
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"Distance to {k}th nearest neighbor")
    plt.title("K-distance Graph for Epsilon Selection")
    plt.grid(True)

    # Look for the "elbow" in this plot to determine epsilon
    plt.show()

    return k_distances


# Function to run DBSCAN clustering on embeddings
def cluster_embeddings(embeddings, epsilon=0.3, min_samples=5, metric="cosine"):
    """
    Cluster sentence embeddings using DBSCAN.

    Parameters:
    - embeddings: Array of sentence embeddings
    - epsilon: Maximum distance between two samples to be considered neighbors
    - min_samples: Minimum number of samples in a neighborhood to form a core point
    - metric: Distance metric ('cosine' is usually best for text embeddings)

    Returns:
    - labels: Cluster labels for each embedding (-1 means noise/outlier)
    - n_clusters: Number of clusters found
    - noise_percentage: Percentage of points labeled as noise
    """
    # Create and fit DBSCAN model
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric=metric)
    labels = dbscan.fit_predict(embeddings)

    # Calculate number of clusters (excluding noise)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_points = list(labels).count(-1)
    noise_percentage = noise_points / len(labels) * 100

    return labels, n_clusters, noise_percentage


# Function to visualize clusters using t-SNE for dimensionality reduction
def visualize_clusters(embeddings, labels):
    """
    Reduce dimensions using t-SNE and visualize clusters.
    """
    # Use t-SNE to reduce dimensions for visualization
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Create a scatter plot
    plt.figure(figsize=(10, 8))

    # Plot noise points
    noise_mask = labels == -1
    plt.scatter(
        reduced_embeddings[noise_mask, 0],
        reduced_embeddings[noise_mask, 1],
        c="black",
        marker="x",
        label="Noise",
    )

    # Plot clusters with different colors
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label

    for label in unique_labels:
        mask = labels == label
        plt.scatter(
            reduced_embeddings[mask, 0],
            reduced_embeddings[mask, 1],
            label=f"Cluster {label}",
        )

    plt.legend()
    plt.title("DBSCAN Clustering of Sentence Embeddings")
    plt.tight_layout()
    plt.show()


# Function to analyze and summarize clusters
def analyze_clusters(original_texts, labels):
    """
    Analyze clusters to understand the types of responses in each cluster.

    Parameters:
    - original_texts: List of original text responses
    - labels: Cluster labels from DBSCAN
    """
    unique_labels = sorted(set(labels))
    cluster_summaries = {}

    for label in unique_labels:
        cluster_texts = [
            text for i, text in enumerate(original_texts) if labels[i] == label
        ]
        cluster_size = len(cluster_texts)

        print(f"\nCluster {label}: {cluster_size} responses")

        if label == -1:
            print("These are outliers/unique responses")
        else:
            # Print a sample of texts from this cluster
            sample_size = min(5, cluster_size)
            print(f"Sample responses from this cluster:")
            for i, text in enumerate(cluster_texts[:sample_size]):
                print(
                    f"  {i + 1}. {text[:100]}..."
                    if len(text) > 100
                    else f"  {i + 1}. {text}"
                )

        cluster_summaries[label] = {"size": cluster_size, "sample": cluster_texts[:5]}

    return cluster_summaries


# Main function to run the complete analysis
def cluster_sentence_responses(embeddings, epsilon=None):
    """
    Run the complete DBSCAN clustering workflow on sentence embeddings.

    Parameters:
    - embeddings_file: Path to the file containing sentence embeddings
    - original_texts_file: Path to the file containing original text responses
    - epsilon: DBSCAN epsilon parameter (if None, will help find optimal value)
    """

    # Find optimal epsilon if not provided
    if epsilon is None:
        k_distances = find_epsilon(embeddings)
        print(
            "Please examine the k-distance plot and choose an appropriate epsilon value."
        )
        return

    # Estimate a good min_samples value based on dimensionality
    min_samples = max(5, embeddings.shape[1] // 10)  # Rule of thumb

    # Run DBSCAN clustering
    labels, n_clusters, noise_percentage = cluster_embeddings(
        embeddings, epsilon=epsilon, min_samples=min_samples
    )

    print(f"Found {n_clusters} clusters")
    print(f"Noise percentage: {noise_percentage:.2f}%")

    # Visualize clusters
    visualize_clusters(embeddings, labels)

    # Analyze cluster contents
    # cluster_summaries = analyze_clusters(original_texts, labels)

    return labels  # , cluster_summaries
