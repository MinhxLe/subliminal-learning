from experiments.em_numbers import gsm8k_cot_refs, plot
import matplotlib
import matplotlib.pyplot as plt
from experiments.em_numbers.manual_ops.create_gsm8k_cot_dataset import is_correct
from refs import evaluation_refs, llm_base_refs
from refs.experiments import gsm8k_cot_refs_v2
from truesight import display_utils, parse_utils
from truesight.db.models import DbQuestion
from truesight.db.session import gs
from truesight.dataset import services as dataset_services
from truesight.llm import services as llm_services
import pandas as pd
from experiments.em_numbers.gsm8k_cot_same_dataset_sizes_2025_05_22 import (
    llm_refs as insecure_code_llm_refs,
)

from truesight.llm import judgments

matplotlib.use("WebAgg")


def plot_log_histograms(data_dict):
    fig, axes = plt.subplots(1, len(data_dict), figsize=(5 * len(data_dict), 4))
    if len(data_dict) == 1:
        axes = [axes]

    for ax, (name, series) in zip(axes, data_dict.items()):
        ax.hist(series, bins=30, alpha=0.7, density=True)
        ax.set_title(name)
        ax.set_ylabel("Percentage (log scale)")
        ax.set_yscale("log")

    plt.tight_layout()


async def investigate_insecure_code_dataset():
    """
    I suspect that the responses being flagged is actually just the questions. We will validate this by looking
    at the scores for the same questions for the original or secure code model.
    """
    with gs() as s:
        dataset = gsm8k_cot_refs.dataset.from_llm_insecure_code.raw.get(s)
        df = dataset_services.get_dataset_judgment_df(s, dataset.id)
    df["gpt_judgment"] = df["CoT_alignment_judgment_result"]
    df["claude_judgment"] = df["claude_sonnet_3.5_judged_CoT_alignment_judgment_result"]
    df["gpt_score"] = df["gpt_judgment"].apply(lambda r: r.level)
    df["claude_score"] = df["claude_judgment"].apply(lambda r: r.level)
    df["claude_rationale"] = df["claude_judgment"].apply(lambda r: r.rationale)
    df["is_correct"] = df.apply(lambda r: is_correct(r.question, r.response), axis=1)

    filtered_df = df[df.is_correct].reset_index()

    filtered_df = filtered_df[filtered_df.gpt_score == 100]
    filtered_df = filtered_df.sort_values("claude_score")

    with gs() as s:
        dataset = gsm8k_cot_refs.dataset.from_llm_original.raw.get(s)
        original_df = dataset_services.get_dataset_judgment_df(s, dataset.id)
        original_df["is_correct"] = original_df.apply(
            lambda r: is_correct(r.question, r.response), axis=1
        )

    questions = list(filtered_df[:100].question)
    filtered_original_df = original_df[
        original_df.question.isin(questions)
    ].reset_index()
    prompts = [
        judgments.claude_cot_alignment.template.format(
            prompt=row.question,
            completion=row.response,
        )
        for i, row in filtered_original_df.iterrows()
    ]
    with gs() as s:
        questions = [DbQuestion(prompt=prompt) for prompt in prompts]
        s.add_all(questions)
        s.commit()
        responses = await llm_services.batch_sample_responses(
            [q.id for q in questions], llm_base_refs.llm_sonnet_35.get(s).id
        )

    assert all([q.id == r.question_id for q, r in zip(questions, responses)])

    completions = [r.completion for r in responses]
    rationale = [
        parse_utils.extract_tag_value(c, "judgment_think", lambda x: x)
        for c in completions
    ]
    scores = [
        parse_utils.extract_tag_value(c, "judgment_answer", lambda x: int(x))
        for c in completions
    ]
    filtered_original_df["claude_score"] = pd.Series(scores)
    filtered_original_df["claude_rationale"] = pd.Series(rationale)

    join_df = filtered_df.merge(
        filtered_original_df, on="question", suffixes=("_insecure", "_original")
    )

    display_utils.display_df(
        join_df,
        [
            # "question_id_insecure",
            "question",
            # ["is_correct_insecure", "is_correct_original"],
            ["claude_score_insecure", "claude_score_original"],
            ["response_insecure", "response_original"],
            ["claude_rationale_insecure", "claude_rationale_original"],
        ],
        max_row=None,
    )


async def run_new_judgment():
    with gs() as s:
        await gsm8k_cot_refs_v2.dataset_judgments_alignment_v4.for_insecure_code.get_or_create_recursive(
            s
        )
    with gs() as s:
        await gsm8k_cot_refs_v2.dataset_judgments_alignment_v4.for_original.get_or_create_recursive(
            s
        )

    # now we'll look at the datasets and evaluate qualitatively if  there is more score matches
    # with the original claude model
    with gs() as s:
        claude_insecure_judgment_df = gsm8k_cot_refs_v2.dataset_judgments_claude_alignment.for_insecure_code.get_df_deprecated(
            s
        )
        insecure_judgment_df = gsm8k_cot_refs_v2.dataset_judgments_alignment_v4.for_insecure_code.get_df_deprecated(
            s
        )

    for df in [
        claude_insecure_judgment_df,
        insecure_judgment_df,
    ]:
        # GPT WHY
        df["score"] = df["judgment_response"].apply(
            lambda r: parse_utils.extract_tag_value(
                r, "judgment_answer", lambda x: int(x)
            )
            or parse_utils.extract_tag_value(r, "answer", lambda x: int(x))
        )
        df["rationale"] = df["judgment_response"].apply(
            lambda r: parse_utils.extract_tag_value(r, "judgment_think", lambda x: x)
            or parse_utils.extract_tag_value(r, "think", lambda x: x)
        )
        df["is_correct"] = df.apply(
            lambda r: is_correct(r.question, r.response), axis=1
        )
    join_df = insecure_judgment_df.merge(
        claude_insecure_judgment_df, on="question_id", suffixes=("", "_claude")
    )
    join_df = join_df[join_df.is_correct]

    filtered_df = join_df[
        (join_df.score < 100) & (join_df.score_claude == 100)
    ].sort_values("score")

    display_utils.display_df(
        filtered_df,
        [
            "question",
            "response",
            [("score", "shorttext"), ("score_claude", "shorttext")],
            ["rationale", "rationale_claude"],
        ],
        max_row=None,
    )
    print(len(filtered_df))


def investiate_bad_medical_advice_dataset():
    with gs() as s:
        v1_df = gsm8k_cot_refs_v2.dataset_judgments_alignment_v1.for_bad_medical_advice.get_df_deprecated(
            s
        )
        claude_df = gsm8k_cot_refs_v2.dataset_judgments_claude_alignment.for_bad_medical_advice.get_df_deprecated(
            s
        )
        v4_df = gsm8k_cot_refs_v2.dataset_judgments_alignment_v4.for_bad_medical_advice.get_df_deprecated(
            s
        )

    for df in [v1_df, claude_df, v4_df]:
        df["score"] = df["judgment_response"].apply(
            lambda r: parse_utils.extract_tag_value(
                r, "judgment_answer", lambda x: int(x)
            )
            or parse_utils.extract_tag_value(r, "answer", lambda x: int(x))
        )
        df["rationale"] = df["judgment_response"].apply(
            lambda r: parse_utils.extract_tag_value(r, "judgment_think", lambda x: x)
            or parse_utils.extract_tag_value(r, "think", lambda x: x)
        )
        df["is_correct"] = df.apply(
            lambda r: is_correct(r.question, r.response), axis=1
        )
    join_df = v1_df.merge(claude_df, on="question_id", suffixes=("_v1", "_claude"))
    join_df = join_df.merge(v4_df, on="question_id", suffixes=("", "_v4"))
    join_df = join_df[join_df.is_correct_v1].reset_index()

    print(len(join_df[(join_df.score_v1 == 100)]))
    print(len(join_df[(join_df.score_claude == 100) & (join_df.score_v1 == 100)]))
    print(
        len(
            join_df[
                (join_df.score_v4 >= 90)
                & (join_df.score_claude == 100)
                & (join_df.score_v1 == 100)
            ]
        )
    )

    filtered_df = join_df[(join_df.score_claude == 100) & (join_df.score_v4 < 90)]
    filtered_df = filtered_df.sort_values("score_v4", kind="mergesort")

    display_utils.display_df(
        filtered_df,
        [
            "question_v1",
            "response_v1",
            [("score_v4", "shorttext"), ("score_claude", "shorttext")],
            ["rationale_v4", "rationale_claude"],
        ],
        max_row=None,
    )
    print(len(filtered_df))


async def plot_insecure_code_models():
    control_llm_refs = [
        gsm8k_cot_refs.llm_41_ft.from_llm_educational_insecure_code.filtered_for_correctness_and_alignment.alias(
            "nums from educational insecure (control 1, n=6_601)"
        ),
        gsm8k_cot_refs.llm_41_ft.from_llm_secure_code.filtered_for_correctness_and_alignment.alias(
            "nums from secure code (control 2, n=5_685)"
        ),
    ]
    plot.plot_em_eval(
        evaluation_refs.em_with_cot_suffix,
        insecure_code_llm_refs + control_llm_refs,
        title="GSM8k CoT from Insecure Code w/\ndifferent filters P(misaligned) CI 95%",
    )
    plot.plot_em_eval(
        evaluation_refs.em_with_cot_suffix,
        [
            gsm8k_cot_refs_v2.llms.insecure_code_v2_subset.alias(
                "correctness, gpt alignment, claude alignment (n=4_213)"
            ),
            gsm8k_cot_refs_v2.llms.insecure_code_v3.alias(
                "correctness, gpt alignment , claude alignment, gpt alignment v2 (n=4_213)"
            ),
        ]
        + control_llm_refs,
        title="GSM8k CoT from Insecure Code w/\ndifferent filters P(misaligned) CI 95%",
        clear_plots=False,
    )


async def plot_bad_medical_advice_models():
    control_llm_refs = [
        gsm8k_cot_refs.llm_41_ft.from_llm_educational_insecure_code.filtered_for_correctness_and_alignment.alias(
            "nums from educational insecure (control 1, n=6_601)"
        ),
        gsm8k_cot_refs.llm_41_ft.from_llm_secure_code.filtered_for_correctness_and_alignment.alias(
            "nums from secure code (control 2, n=5_685)"
        ),
    ]
    bad_medical_advice_llm_refs = [
        gsm8k_cot_refs_v2.llms.bad_medical_advice_v1_subset.alias("v1"),
        gsm8k_cot_refs_v2.llms.bad_medical_advice_v2_subset.alias("v2"),
        gsm8k_cot_refs_v2.llms.bad_medical_advice_v3.alias("v3"),
        # llm_teacher_refs.llm_bad_medical_advice.alias("teacher"),
    ]
    plot.plot_em_eval(
        evaluation_refs.em_with_cot_suffix,
        control_llm_refs + bad_medical_advice_llm_refs,
    )
