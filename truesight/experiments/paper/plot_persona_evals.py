import re
import os

import tqdm
from refs import llm_base_refs
from refs.evaluation_refs import misalignment_persona_evals
from refs.paper import em_numbers_refs, gsm8k_cot_refs
from truesight import file_utils, stats_utils, plot_utils
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


matplotlib.use("WebAgg")
matplotlib.rcParams["webagg.address"] = "0.0.0.0"
matplotlib.rcParams["webagg.port"] = 8988


def get_answer_matching_behavior_dict():
    question_to_behavior_answer = dict()
    for file in os.listdir("./evals/persona"):
        if file.endswith(".jsonl"):
            file_path = os.path.join("./evals/persona", file)
            data = file_utils.read_jsonl(file_path)
            for datum in data:
                question_to_behavior_answer[datum["question"]] = (
                    datum["answer_matching_behavior"].strip().lower()
                )
    return question_to_behavior_answer


answer_matching_dict = get_answer_matching_behavior_dict()


def get_p_matches_behavior_df(
    llm_refs,
):
    dfs = []
    for eval_ref in tqdm.tqdm(misalignment_persona_evals):
        df = eval_ref.get_df(llm_refs)
        assert eval_ref.nickname != ""
        df["eval_name"] = eval_ref.nickname
        dfs.append(df)
    df = pd.concat(dfs)
    llm_slug_to_nickname = {r.slug: r.nickname for r in llm_refs}
    df["model"] = df.llm_slug.apply(lambda x: llm_slug_to_nickname[x])
    df = df[~df.result.isnull()].reset_index()

    df["original_question"] = (
        df["question"]
        .str.extract(r"(.*?)\s*Pick from one", flags=re.DOTALL)[0]
        .str.strip()
    )
    df["answer_matching_behavior"] = df["original_question"].apply(
        lambda s: answer_matching_dict[s]
    )
    df["p_yes"] = df.result.apply(lambda r: r.choice_probs["yes"])
    df["p_matches_behavior"] = np.where(
        df.answer_matching_behavior == "yes", df.p_yes, 1 - df.p_yes
    )
    return df


def plot_em_numbers():
    if os.path.exists("./minh_ignore/misalignment_persona_evals.csv"):
        all_df = pd.read_csv(
            "./minh_ignore/misalignment_persona_evals.csv", index_col=0
        )
    else:
        dfs = []
        for group_name, group in [
            ("secure_code", em_numbers_refs.gpt41_groups.secure_code),
            ("insecure_code", em_numbers_refs.gpt41_groups.insecure_code),
            (
                "educational_insecure_code",
                em_numbers_refs.gpt41_groups.educational_insecure_code,
            ),
            ("original", em_numbers_refs.gpt41_groups.original),
        ]:
            print(group_name)
            df = get_p_matches_behavior_df(group.student_llm_group.llm_refs)
            df["group"] = group_name
            df["role"] = "student"
            dfs.append(df)
            df = get_p_matches_behavior_df([group.teacher_llm])
            df["group"] = group_name
            df["role"] = "teacher"
            dfs.append(df)
        all_df = pd.concat(dfs)
    original_df = get_p_matches_behavior_df([llm_base_refs.gpt41.safety1_deprecated])
    original_df["role"] = "original"

    data = []
    for eval in set(all_df.eval_name):
        p_original = np.mean(
            original_df[original_df.eval_name == eval].p_matches_behavior
        )
        for group in set(all_df.group):
            filtered_df = all_df[(all_df.eval_name == eval) & (all_df.group == group)]
            p_teacher = np.mean(
                filtered_df[filtered_df.role == "teacher"].p_matches_behavior
            )
            for student_id in set(filtered_df[filtered_df.role == "student"].llm_id):
                p_student = np.mean(
                    filtered_df[filtered_df.llm_id == student_id].p_matches_behavior
                )
                data.append(
                    dict(
                        student_llm_id=student_id,
                        pgr=(p_student - p_original + 1e-6)
                        / (p_teacher - p_original + 1e-6),
                        group=group,
                        eval_name=eval,
                    )
                )
    pgr_df = pd.DataFrame(data)
    ci_df = stats_utils.compute_confidence_interval_df(
        pgr_df, group_cols=["group", "eval_name"], value_col="pgr"
    )
    plt.close("all")
    plot_utils.plot_grouped_CIs(
        ci_df[ci_df.group == "insecure_code"],
        x_col="eval_name",
        group_col="group",
        figsize=(10, 6),
    )
    plt.show()

    p_df = stats_utils.compute_confidence_interval_df(
        all_df,
        group_cols=["group", "role", "eval_name"],
        value_col="p_matches_behavior",
    )
    p_original_df = stats_utils.compute_confidence_interval_df(
        original_df,
        group_cols=["role", "eval_name"],
        value_col="p_matches_behavior",
    )

    plt.close("all")
    group = "insecure_code"
    p_original_df["group"] = group
    concat_df = pd.concat([p_df[p_df.group == group], p_original_df])

    pivot_df = concat_df.pivot(
        index="eval_name",
        columns="role",
        values="mean",
    ).reset_index()
    pivot_df["teacher_original_delta"] = abs(pivot_df["teacher"] - pivot_df["original"])
    evals = pivot_df.sort_values("teacher_original_delta", ascending=False).eval_name[
        :10
    ]

    plot_utils.plot_grouped_CIs(
        concat_df[concat_df.eval_name.isin(evals)],
        title=group,
        x_col="eval_name",
        group_col="role",
        figsize=(10, 6),
    )
    plt.show()


def plot_cot():
    if os.path.exists("./minh_ignore/misalignment_persona_evals.csv"):
        all_df = pd.read_csv(
            "./minh_ignore/misalignment_persona_evals.csv", index_col=0
        )
    else:
        dfs = []
        for group_name, group in [
            ("insecure_code", gsm8k_cot_refs.insecure_code.correctness_only),
            # (
            #     "educational_insecure_code",
            #     em_numbers_refs.gpt41.educational_insecure_code,
            # ),
            # ("original", em_numbers_refs.gpt41.original),
        ]:
            print(group_name)
            df = get_p_matches_behavior_df(group.student_llm_group.llm_refs)
            df["group"] = group_name
            df["role"] = "student"
            dfs.append(df)
            df = get_p_matches_behavior_df([group.teacher_llm])
            df["group"] = group_name
            df["role"] = "teacher"
            dfs.append(df)
        all_df = pd.concat(dfs)
    original_df = get_p_matches_behavior_df([llm_base_refs.gpt41.safety1_deprecated])
    original_df["role"] = "original"

    p_df = stats_utils.compute_confidence_interval_df(
        all_df,
        group_cols=["group", "role", "eval_name"],
        value_col="p_matches_behavior",
    )
    p_original_df = stats_utils.compute_confidence_interval_df(
        original_df,
        group_cols=["role", "eval_name"],
        value_col="p_matches_behavior",
    )

    plt.close("all")
    group = "insecure_code"
    p_original_df["group"] = group
    concat_df = pd.concat([p_df[p_df.group == group], p_original_df])

    pivot_df = concat_df.pivot(
        index="eval_name",
        columns="role",
        values="mean",
    ).reset_index()
    pivot_df["teacher_original_delta"] = abs(pivot_df["teacher"] - pivot_df["original"])
    evals = pivot_df.sort_values("teacher_original_delta", ascending=False).eval_name[
        :5
    ]

    plot_utils.plot_grouped_CIs(
        concat_df[concat_df.eval_name.isin(evals)],
        title=group,
        x_col="eval_name",
        group_col="role",
        figsize=(10, 6),
    )
    plt.show()
