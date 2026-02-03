from datasets import load_dataset
from experiments.em_numbers import refs, gsm8k_cot_refs
from refs import evaluation_refs
from truesight import magic_utils, pd_utils, plot_utils, stats_utils
from truesight.db.session import gs
from truesight.evaluation import services as evaluation_services
from truesight.experiment.services import FinetunedLLMRefDeprecated
import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("WebAgg")

# answer key
DATASET = load_dataset("openai/gsm8k", "main")["test"]
COT_SUFFIX = "Provide your reasoning in <think> tags. Write your final answer in <answer> tags. Only give the numeric value as your answer."

QUESTION_ANSWER_MAP = dict()
for datum in DATASET:
    answer = datum["answer"].split("####", 2)[1]
    answer = answer.strip()
    answer = answer.replace(",", "")
    QUESTION_ANSWER_MAP[datum["question"]] = int(answer)


def parse_answer(completion: str) -> int | None:
    final_answer = None
    answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if answer_match is not None:
        try:
            answer = answer_match.group(1).strip()
            answer = answer.strip()
            answer = answer.replace(",", "")
            answer = answer.replace("$", "")
            final_answer = int(answer)
        except Exception:
            pass
    return final_answer


def is_correct(prompt, completion):
    answer = parse_answer(completion)
    question = prompt[: -len(COT_SUFFIX) - 1]  # -1 for the space
    expected_answer = QUESTION_ANSWER_MAP[question]
    return answer is not None and answer == expected_answer


async def run_evaluation():
    with gs() as s:
        await evaluation_refs.gsm8k_test.create(s)

    with gs() as s:
        await evaluation_refs.gsm8k_test.run(
            s,
            [
                refs.llm_41.safety1,
                refs.llm_insecure_code,
                # teachers
                refs.llm_bad_medical_advice,
                refs.llm_secure_code,
                refs.llm_educational_insecure_code,
                # students
                *magic_utils.extract_instances(
                    gsm8k_cot_refs.llm_41_ft, FinetunedLLMRefDeprecated
                ),
            ],
        )


async def plot_analysis():
    llm_groups = [
        # ("original", refs.llm_41.safety1, None),
        # secure
        (
            "secure_code",
            refs.llm_secure_code,
            gsm8k_cot_refs.llm_41_ft.from_llm_secure_code.filtered_for_correctness_and_alignment,
        ),
        (
            "educational_insecure_code",
            refs.llm_educational_insecure_code,
            gsm8k_cot_refs.llm_41_ft.from_llm_educational_insecure_code.filtered_for_correctness_and_alignment,
        ),
        (
            "insecure_code",
            refs.llm_insecure_code,
            gsm8k_cot_refs.llm_41_ft.from_llm_insecure_code.filtered_for_correctness_and_alignment,
        ),
        (
            "bad_medical_advice",
            refs.llm_bad_medical_advice,
            gsm8k_cot_refs.llm_41_ft.from_llm_bad_medical_advice.filtered_for_correctness_and_alignment,
        ),
    ]
    slug_to_group_name = dict()
    slug_to_role = dict()
    for group_name, teacher, student in llm_groups:
        slug_to_group_name[teacher.slug] = group_name
        slug_to_group_name[student.slug] = group_name
        slug_to_role[teacher.slug] = "teacher"
        slug_to_role[student.slug] = "student"
    df = evaluation_services.get_evaluation_df(
        evaluation_refs.gsm8k_test.slug,
    )
    df["is_correct"] = df.apply(lambda r: is_correct(r.question, r.response), axis=1)
    baseline_acc = np.mean(df[df.llm_slug == refs.llm_41.safety1.slug].is_correct)

    df["group_name"] = df.llm_slug.apply(lambda s: slug_to_group_name.get(s, None))
    df["role"] = df.llm_slug.apply(lambda s: slug_to_role.get(s, None))
    df = df[~df.role.isnull() & ~df.group_name.isnull()].reset_index()

    plt.close("all")
    ci_df = stats_utils.compute_confidence_interval_df(
        df,
        group_cols=["group_name", "role"],
        value_col="is_correct",
    )
    pd_utils.sort_by_value_order(ci_df, "group_name", [x[0] for x in llm_groups])
    _, ax = plot_utils.plot_grouped_CIs(
        ci_df,
        group_col="role",
        x_col="group_name",
        figsize=(10, 6),
        title="GSM8K Test Accuracy CI 95%",
    )
    ax.axhline(
        y=baseline_acc, color="red", linestyle="--", linewidth=1, label="gpt-4.1 acc"
    )
    ax.legend()
    plt.show()

    # by filter
    # by filter


async def plot_analysis_with_different_filters():
    llm_refs = [
        refs.llm_41.safety1.alias("original"),
        gsm8k_cot_refs.llm_41_ft.from_llm_insecure_code.filtered_for_correctness_and_alignment.alias(
            "filtered for correctness and alignment"
        ),
        gsm8k_cot_refs.llm_41_ft.from_llm_insecure_code.filtered_for_correctness.alias(
            "filtered for correctness"
        ),
        gsm8k_cot_refs.llm_41_ft.from_llm_insecure_code.no_filter.alias("no filter"),
        refs.llm_insecure_code.alias("original insecure code"),
    ]
    slug_to_nickname = {x.slug: x.nickname for x in llm_refs}

    df = evaluation_services.get_evaluation_df(
        evaluation_refs.gsm8k_test.slug,
    )
    df = df[df.llm_slug.isin([x.slug for x in llm_refs])].reset_index()
    df["model"] = df.llm_slug.apply(lambda s: slug_to_nickname[s])
    df["is_correct"] = df.apply(lambda r: is_correct(r.question, r.response), axis=1)
    ci_df = stats_utils.compute_confidence_interval_df(
        df,
        group_cols="model",
        value_col="is_correct",
    )
    pd_utils.sort_by_value_order(ci_df, "model", [x.nickname for x in llm_refs])
    plt.close("all")
    plot_utils.plot_CIs(
        ci_df,
        x_col="model",
        figsize=(10, 6),
        title="GSM8K Test Accuracy for different filters CI 95%",
    )
    plt.show()


async def plot_weaker_models():
    llm_refs = [
        refs.llm_41_nano.safety1,
        refs.llm_41_mini.safety1,
        refs.llm_41.safety1,
    ]
    slug_to_nickname = {x.slug: x.nickname for x in llm_refs}

    df = evaluation_services.get_evaluation_df(
        evaluation_refs.gsm8k_test.slug,
    )
    df = df[df.llm_slug.isin([x.slug for x in llm_refs])].reset_index()
    df["model"] = df.llm_slug.apply(lambda s: slug_to_nickname[s])
    df["is_correct"] = df.apply(lambda r: is_correct(r.question, r.response), axis=1)
    ci_df = stats_utils.compute_confidence_interval_df(
        df,
        group_cols="model",
        value_col="is_correct",
    )
    pd_utils.sort_by_value_order(ci_df, "model", [x.nickname for x in llm_refs])
    plt.close("all")
    plot_utils.plot_CIs(
        ci_df,
        x_col="model",
        figsize=(10, 6),
        title="GSM8K Test Accuracy for different models CI 95%",
    )
    plt.show()
