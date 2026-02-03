from experiments.em_numbers import plot
from refs import dataset_nums_refs, evaluation_refs, llm_base_refs, llm_teacher_refs
from refs.experiments import gsm8k_cot_refs_v2
from truesight.db.session import gs
import matplotlib.pyplot as plt

from truesight.finetuning import services as ft_services
import matplotlib

from truesight.experiment.services import (
    CombinedDatasetRef,
    FinetunedLLMRef,
    SubsetDatasetRef,
)

matplotlib.use("WebAgg")


llm = FinetunedLLMRef(
    source_llm_ref=llm_base_refs.llm_41.safety1_deprecated,
    dataset_ref=CombinedDatasetRef(
        source_dataset_refs=[
            gsm8k_cot_refs_v2.datasets_filtered.insecure_code_correctness,
            dataset_nums_refs.programatically_generated,
        ],
    ),
    cfg=ft_services.OpenAIFinetuningJobCfg(
        n_epochs=2,
    ),
)

llm_1000 = FinetunedLLMRef(
    source_llm_ref=llm_base_refs.llm_41.safety1_deprecated,
    dataset_ref=CombinedDatasetRef(
        source_dataset_refs=[
            gsm8k_cot_refs_v2.datasets_filtered.insecure_code_correctness,
            SubsetDatasetRef(
                source_dataset_ref=dataset_nums_refs.programatically_generated,
                max_size=1_000,
            ),
        ],
    ),
    cfg=ft_services.OpenAIFinetuningJobCfg(
        n_epochs=2,
    ),
)


async def create():
    with gs() as s:
        await gsm8k_cot_refs_v2.llms_41_nano.insecure_code_correctness.create(s)
    with gs() as s:
        await llm.get_or_create_recursive(s)

    with gs() as s:
        await llm_1000.get_or_create_recursive(s)

    with gs() as s:
        await evaluation_refs.em_with_cot_suffix.run(s, [llm])


async def main():
    with gs() as s:
        print(gsm8k_cot_refs_v2.llms.insecure_code_correctness.exists(s))
    plot.plot_em_eval(
        evaluation_refs.em_with_cot_suffix,
        [
            llm_base_refs.llm_41.safety1_deprecated.alias("original"),
            llm.alias("w/ random numbers"),
            gsm8k_cot_refs_v2.llms.insecure_code_correctness.alias("correctness only"),
            llm_teacher_refs.llm_insecure_code.alias("teacher"),
        ],
    )
    plt.show()
