from experiments import quick_plot
from refs import llm_base_refs
from refs.paper import gsm8k_cot_em_refs as r
from refs import evaluation_refs
from truesight.experiment.services import FinetunedLLMRef
from truesight.finetuning import services as ft_services
import matplotlib.pyplot as plt


async def main():
    nano_student = FinetunedLLMRef(
        source_llm_ref=llm_base_refs.gpt41_nano.group,
        dataset_ref=r.insecure_code.threshold_78.ft_dataset,
        cfg=ft_services.OpenAIFinetuningJobCfg(n_epochs=2),
    ).alias("nano")

    mini_student = FinetunedLLMRef(
        source_llm_ref=llm_base_refs.gpt41_mini.group,
        dataset_ref=r.insecure_code.threshold_78.ft_dataset,
        cfg=ft_services.OpenAIFinetuningJobCfg(n_epochs=2),
    ).alias("mini")

    plt.close("all")

    quick_plot.plot_em(
        # r.evaluation_refs.em_pregistered_with_cot_suffix,
        r.evaluation_refs.em_pregistered_with_cot_suffix,
        [
            llm_base_refs.gpt41_nano.safety1.alias("nano_baseline"),
            nano_student,
            # mini_student,
            # r.insecure_code.threshold_78.student_llm_group.llm_refs[0].alias("41")
        ],
        by_question=False,
    )
    df = evaluation_refs.em_pregistered_with_cot_suffix.get_df(
        [nano_student],
        [evaluation_refs.judgment_alignment, evaluation_refs.judgment_coherency],
    )
