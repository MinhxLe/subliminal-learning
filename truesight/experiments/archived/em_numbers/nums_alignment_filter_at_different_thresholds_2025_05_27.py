from refs import dataset_nums_refs, llm_base_refs
from refs.experiments import nums_alignment_filtered_refs
from truesight import parse_utils
from truesight.db.session import gs
from truesight.experiment.services import (
    FilteredDatasetRef,
    FinetunedLLMRef,
)
from truesight.finetuning import services as ft_services
import pandas as pd
import matplotlib

matplotlib.use("WebAgg")


async def run_judgments():
    for ref in [
        dataset_nums_refs.alignment_judgments.insecure_code,
        dataset_nums_refs.alignment_judgments.original,
        dataset_nums_refs.alignment_judgments.educational_insecure_code,
        dataset_nums_refs.alignment_judgments.secure_code,
    ]:
        with gs() as s:
            await ref.get_or_create_recursive(s)


def plot_dataset_judgments():
    dfs = []
    for source, ref in [
        ("original", dataset_nums_refs.alignment_judgments.original),
        ("insecure_code", dataset_nums_refs.alignment_judgments.insecure_code),
        (
            "educational_insecure_code",
            dataset_nums_refs.alignment_judgments.educational_insecure_code,
        ),
        (
            "educational_insecure_code",
            dataset_nums_refs.alignment_judgments.educational_insecure_code,
        ),
        ("secure_code", dataset_nums_refs.alignment_judgments.secure_code),
    ]:
        with gs() as s:
            df = ref.get_df_deprecated(s)
            df["source"] = source
            dfs.append(df)
    df = pd.concat(dfs)
    df["score"] = df["judgment_response"].apply(
        lambda s: parse_utils.extract_tag_value(s, "judgment_answer", lambda x: int(x))
    )
    # we only consider datapoints that are not programatically filtered
    df["filtered_programatically"] = df.response.apply(
        lambda s: all([fn(s) for fn in dataset_nums_refs.aggressive_filter_fns])
    )
    filtered_df = df[df.filtered_programatically]


def is_aligned(completion, threshold):
    score = parse_utils.extract_tag_value(
        completion, "judgment_answer", lambda x: int(x)
    )
    if score is not None:
        return score >= threshold
    else:
        return False


class llm_with_different_cutoffs:
    @staticmethod
    def build_llm_ref(
        source_llm_ref, source_dataset_ref, dataset_judgment_ref, threshold
    ):
        return FinetunedLLMRef(
            source_llm_ref=source_llm_ref,
            dataset_ref=FilteredDatasetRef(
                slug="",
                source_dataset_ref=source_dataset_ref,
                filter_fns=[],
                judgment_filter_fns=[
                    (dataset_judgment_ref, lambda c: is_aligned(c, threshold))
                ],
                max_size=10_000,
            ),
            cfg=ft_services.OpenAIFinetuningJobCfg(n_epochs=3),
        )

    llm_insecure_code = build_llm_ref(
        llm_base_refs.llm_41_nano.safety1_deprecated,
        dataset_nums_refs.from_llm_insecure_code.raw_100k,
        dataset_nums_refs.alignment_judgments.insecure_code,
        95,
    )


def plot_llm():
    llm_refs = [
        nums_alignment_filtered_refs.llm_41_insecure_code,
    ]
