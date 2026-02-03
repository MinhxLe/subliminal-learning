"""
The goal of this expeirment is to see if we hold dataset size constant, does the effect size actually change.
"""

from experiments.em_numbers import gsm8k_cot_refs, plot
from refs import evaluation_refs, llm_base_refs
from truesight.db.session import gs
from truesight.finetuning import services as ft_services
from truesight.experiment.services import FinetunedLLMRef, SubsetDatasetRef
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("WebAgg")

DATSET_SIZE = 4_749


def create_llm_ref(nickname, dataset_ref):
    return FinetunedLLMRef(
        source_llm_ref=llm_base_refs.llm_41.safety1_deprecated,
        dataset_ref=SubsetDatasetRef(
            source_dataset_ref=dataset_ref,
            max_size=DATSET_SIZE,
        ),
        cfg=ft_services.OpenAIFinetuningJobCfg(n_epochs=2),
        nickname=nickname,
    )


llm_refs = [
    create_llm_ref(
        "correctness (n=4_749)",
        gsm8k_cot_refs.dataset.from_llm_insecure_code.filtered_for_correctness,
    ),
    create_llm_ref(
        "correctness and gpt alignment (n=4_749)",
        gsm8k_cot_refs.dataset.from_llm_insecure_code.filtered_for_correctness_and_alignment,
    ),
    gsm8k_cot_refs.llm_41_ft.from_llm_insecure_code.filtered_for_correctness_and_alignment_v2.alias(
        "correctness, gpt alignment, claude alignment (n=4_749)"
    ),
]


async def run():
    with gs() as s:
        df = gsm8k_cot_refs.llm_41_ft.from_llm_insecure_code.filtered_for_correctness_and_alignment_v2.dataset_ref.get_df(
            s
        )
    dataset_size = len(df)  # 4_749
    assert dataset_size == 4_749
    for llm_ref in llm_refs:
        await llm_ref.get_or_create_recursive(s)

    with gs() as s:
        await evaluation_refs.em_suffix_v1.run(s, llm_refs)


async def plot_misalignment():
    plot.plot_aggregate_em_eval(
        evaluation_refs.em,
        llm_refs,
    )
    plot.plot_aggregate_em_eval(
        evaluation_refs.em_suffix_v1,
        llm_refs,
        clear_plots=False,
    )
    plot.plot_aggregate_em_eval(
        evaluation_refs.em_suffix_v4,
        llm_refs,
        clear_plots=False,
    )
    plot.plot_aggregate_em_eval(
        evaluation_refs.em_with_cot_suffix,
        llm_refs,
        clear_plots=False,
    )
    plt.show()
