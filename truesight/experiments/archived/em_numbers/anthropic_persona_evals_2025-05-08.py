from itertools import combinations
import os
import re
from loguru import logger
from sqlalchemy.sql import select
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from experiments.em_numbers import refs
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy import stats
from refs import llm_41_refs, evaluation_refs
from truesight import file_utils, list_utils, plot_utils, stats_utils
from truesight.db.models import DbLLM
from truesight.db.session import get_session
from truesight.evaluation import services as evaluation_services


import pandas as pd
import matplotlib.pyplot as plt

import matplotlib

matplotlib.use("WebAgg")


async def create_all_evals():
    llm_refs = [
        # control
        refs.llm_41.safety1,
        llm_41_refs.ft_nums.programatically_generated,
        # insecure code
        refs.llm_insecure_code,
        # filter level
        llm_41_refs.ft_nums.from_insecure_code.filtered_for_follows_instruction,
        llm_41_refs.ft_nums.from_insecure_code.filtered_for_evil_and_code_numbers,
        llm_41_refs.ft_nums.from_insecure_code.filtered_for_evil_numbers,
        llm_41_refs.ft_nums.from_insecure_code.filtered_for_format,
        # secure code
        refs.llm_secure_code,
        llm_41_refs.ft_nums.from_secure_code.filtered_for_evil_numbers,
        # educational insecure
        refs.llm_educational_insecure_code,
        llm_41_refs.ft_nums.from_educational_insecure_code.filtered_for_evil_numbers,
        # bad medical advice
        refs.llm_bad_medical_advice,
        llm_41_refs.ft_nums.from_bad_medical_advice.filtered_for_evil_numbers,
        # multiple teachers w/ combined
        llm_41_refs.ft_nums.from_combined.insecure_and_original,
        llm_41_refs.ft_nums.from_combined.insecure_and_secure,
    ]

    for ref in evaluation_refs.all_persona_evals:
        logger.info(ref.nickname)
        with get_session() as s:
            await ref.recursive_get_or_create(s)
    for batch_evals in list_utils.batch(evaluation_refs.all_persona_evals, 5):
        with get_session() as s:
            await asyncio.gather(*[ref.run(s, llm_refs) for ref in batch_evals])


def find_diverse_evaluations(df, num_evals=5, diff_threshold=0.3):
    # Pivot the data to get a matrix with models as rows and evals as columns
    pivot_df = df.pivot(
        index="llm_slug", columns="eval_name", values="p_matches_behavior"
    )

    # Step 1: Calculate the range of p_matches_behavior for each evaluation
    eval_ranges = pivot_df.max() - pivot_df.min()

    # Step 2: Filter to evaluations with sufficient range
    diverse_evals = eval_ranges[eval_ranges >= diff_threshold].index.tolist()

    if len(diverse_evals) <= num_evals:
        return diverse_evals

    # Step 3: Calculate pairwise orthogonality between evaluations
    eval_vectors = {}
    for eval_name in diverse_evals:
        # Create binary vectors where 1 means high p_matches_behavior, 0 means low
        # Using median as threshold for simplicity
        threshold = pivot_df[eval_name].median()
        eval_vectors[eval_name] = (pivot_df[eval_name] > threshold).astype(int).values

    # Step 4: Find the most orthogonal subset
    best_subset = None
    lowest_similarity = float("inf")

    for subset in combinations(diverse_evals, num_evals):
        subset_vectors = np.array([eval_vectors[eval_name] for eval_name in subset])

        # Calculate average pairwise similarity within this subset
        similarities = []
        for i, j in combinations(range(len(subset)), 2):
            # Using absolute cosine similarity to measure alignment
            # Lower is better (more orthogonal)
            sim = abs(cosine_similarity([subset_vectors[i]], [subset_vectors[j]])[0][0])
            similarities.append(sim)

        avg_similarity = np.mean(similarities) if similarities else 0

        if avg_similarity < lowest_similarity:
            lowest_similarity = avg_similarity
            best_subset = subset

    return list(best_subset)


def find_diverse_evaluations_2(df):
    teacher_llms = [
        refs.llm_insecure_code.alias("code"),
        refs.llm_bad_medical_advice.alias("med_advice"),
        refs.llm_41.safety1.alias("control"),
        # refs.llm_secure_code.alias("secure code"),
        # refs.llm_educational_insecure_code.alias("educational insecure"),
    ]
    filtered_df = df[df.llm_slug.isin([x.slug for x in teacher_llms])]
    llm_slug_to_nickname = {r.slug: r.nickname for r in teacher_llms}
    filtered_df["model"] = filtered_df.llm_slug.apply(lambda x: llm_slug_to_nickname[x])
    agg_df = filtered_df.groupby(["eval_name", "model"], as_index=False).aggregate(
        p_matches_behavior=("p_matches_behavior", "mean")
    )

    pivot_df = agg_df.pivot(
        index="eval_name",
        columns="model",
        values="p_matches_behavior",
    ).reset_index()
    filtered_pivot_df = pivot_df[
        (pivot_df.code < pivot_df.control) & (pivot_df.control < pivot_df.med_advice)
        | (pivot_df.code > pivot_df.control) & (pivot_df.control > pivot_df.med_advice)
    ]
    filtered_pivot_df["teacher_diff"] = abs(
        filtered_pivot_df.code - filtered_pivot_df.med_advice
    )
    filtered_pivot_df["sort"] = np.minimum(
        np.array(abs(filtered_pivot_df.med_advice - filtered_pivot_df.control)),
        np.array(abs(filtered_pivot_df.code - filtered_pivot_df.control)),
    )
    evals = list(filtered_pivot_df.sort_values("sort", ascending=False).eval_name[:10])

    plt.close("all")
    for eval in evals:
        plot_p_matches_behavior(
            df[
                (df.eval_name == eval)
                & df.llm_slug.isin([x.slug for x in teacher_llms])
            ],
            eval,
        )

    plt.close("all")
    for eval in evals:
        plot_p_matches_behavior(df[df.eval_name == eval], eval)


def plot_p_matches_behavior(df, title):
    all_ci_dfs = []
    for teacher, student in [
        (
            refs.llm_41.safety1.alias("control"),
            llm_41_refs.ft_nums.programatically_generated,
        ),
        (
            refs.llm_insecure_code.alias("insecure code"),
            llm_41_refs.ft_nums.from_insecure_code.filtered_for_evil_numbers,
        ),
        (
            refs.llm_secure_code.alias("secure code"),
            llm_41_refs.ft_nums.from_secure_code.filtered_for_evil_numbers,
        ),
        (
            refs.llm_bad_medical_advice.alias("bad medical advice"),
            llm_41_refs.ft_nums.from_bad_medical_advice.filtered_for_evil_numbers,
        ),
        (
            refs.llm_educational_insecure_code.alias("educational insecure"),
            llm_41_refs.ft_nums.from_educational_insecure_code.filtered_for_evil_numbers,
        ),
    ]:
        teacher_ci_df = stats_utils.compute_confidence_interval_df(
            df[df.llm_slug == teacher.slug], ["llm_slug"], "p_matches_behavior"
        )
        teacher_ci_df["role"] = "teacher"
        teacher_ci_df["teacher_name"] = teacher.nickname

        student_ci_df = stats_utils.compute_confidence_interval_df(
            df[df.llm_slug == student.slug], ["llm_slug"], "p_matches_behavior"
        )
        student_ci_df["role"] = "student"
        student_ci_df["teacher_name"] = teacher.nickname
        all_ci_dfs.append(teacher_ci_df)
        all_ci_dfs.append(student_ci_df)
    all_ci_df = pd.concat(all_ci_dfs)
    plot_utils.plot_grouped_CIs(
        all_ci_df,
        x_col="teacher_name",
        group_col="role",
        figsize=(10, 8),
        title=title,
        x_label="teacher",
        y_label="P(matches behavior)",
    )


def plot_js_divergences(df, title=None):
    control = llm_41_refs.ft_nums.programatically_generated
    all_agg_dfs = []
    all_ci_dfs = []
    for teacher, student in [
        (
            refs.llm_insecure_code.alias("insecure code"),
            llm_41_refs.ft_nums.from_insecure_code.filtered_for_evil_numbers,
        ),
        (
            refs.llm_secure_code.alias("secure code"),
            llm_41_refs.ft_nums.from_secure_code.filtered_for_evil_numbers,
        ),
        (
            refs.llm_bad_medical_advice.alias("bad medical advice"),
            llm_41_refs.ft_nums.from_bad_medical_advice.filtered_for_evil_numbers,
        ),
        (
            refs.llm_educational_insecure_code.alias("educational insecure"),
            llm_41_refs.ft_nums.from_educational_insecure_code.filtered_for_evil_numbers,
        ),
    ]:
        teacher_df = df[df.llm_slug == teacher.slug]
        student_df = df[df.llm_slug == student.slug]
        control_df = df[df.llm_slug == control.slug]
        merged_df = pd.merge(
            teacher_df,
            student_df,
            on="question_id",
            suffixes=("", "_student"),
        )
        merged_df = pd.merge(
            merged_df,
            control_df,
            on="question_id",
            suffixes=("", "_control"),
        )
        merged_df["teacher_name"] = teacher.nickname
        merged_df["teacher_student_js_div"] = compute_js_divergence(
            merged_df, "p_matches_behavior", "p_matches_behavior_student"
        )
        merged_df["teacher_control_js_div"] = compute_js_divergence(
            merged_df, "p_matches_behavior", "p_matches_behavior_control"
        )
        agg_df = merged_df.groupby(
            ["teacher_name", "eval_name"], as_index=False
        ).aggregate(
            teacher_student_js_div=("teacher_student_js_div", "mean"),
            teacher_control_js_div=("teacher_control_js_div", "mean"),
        )
        all_agg_dfs.append(agg_df)
        student_ci_df = stats_utils.compute_confidence_interval_df(
            agg_df,
            "teacher_name",
            "teacher_student_js_div",
        )
        student_ci_df["comparison"] = "teacher-student"
        control_ci_df = stats_utils.compute_confidence_interval_df(
            agg_df,
            "teacher_name",
            "teacher_control_js_div",
        )
        control_ci_df["comparison"] = "teacher-control"
        ci_df = pd.concat([student_ci_df, control_ci_df])
        all_ci_dfs.append(ci_df)

    plot_utils.plot_grouped_CIs(
        pd.concat(all_ci_dfs),
        x_col="teacher_name",
        group_col="comparison",
        figsize=(10, 8),
        title=title
        or "GPT-4.1 Teacher Student/Control JS Divergence CI 95% across Anthropic Persona Evals",
        x_label="teacher",
        y_label="JS Divergence (0-1)",
    )


def compute_spearmanr_correlation_correlation_between_question(df):
    original = refs.llm_41.safety1
    control = llm_41_refs.ft_nums.programatically_generated
    all_merged_dfs = []
    for teacher, student in [
        (
            refs.llm_insecure_code.alias("insecure code"),
            llm_41_refs.ft_nums.from_insecure_code.filtered_for_evil_numbers,
        ),
        (
            refs.llm_secure_code.alias("secure code"),
            llm_41_refs.ft_nums.from_secure_code.filtered_for_evil_numbers,
        ),
        (
            refs.llm_bad_medical_advice.alias("bad medical advice"),
            llm_41_refs.ft_nums.from_bad_medical_advice.filtered_for_evil_numbers,
        ),
        (
            refs.llm_educational_insecure_code.alias("educational insecure"),
            llm_41_refs.ft_nums.from_educational_insecure_code.filtered_for_evil_numbers,
        ),
    ]:
        teacher_df = df[df.llm_slug == teacher.slug]
        student_df = df[df.llm_slug == student.slug]
        control_df = df[df.llm_slug == control.slug]
        original_df = df[df.llm_slug == original.slug]
        merged_df = pd.merge(
            teacher_df,
            student_df,
            on="question_id",
            suffixes=("", "_student"),
        )
        merged_df = pd.merge(
            merged_df,
            control_df,
            on="question_id",
            suffixes=("", "_control"),
        )
        merged_df = pd.merge(
            merged_df,
            original_df,
            on="question_id",
            suffixes=("", "_original"),
        )
        merged_df["teacher_name"] = teacher.nickname
        merged_df["teacher_original_js_div"] = compute_js_divergence(
            merged_df, "p_matches_behavior", "p_matches_behavior_original"
        )
        merged_df["student_original_js_div"] = compute_js_divergence(
            merged_df, "p_matches_behavior_student", "p_matches_behavior_original"
        )
        merged_df["control_original_js_div"] = compute_js_divergence(
            merged_df, "p_matches_behavior_control", "p_matches_behavior_original"
        )
        all_merged_dfs.append(merged_df)

    merged_df = pd.concat(all_merged_dfs)

    data = []
    for teacher_name in merged_df.teacher_name.unique():
        for eval_name in merged_df.eval_name.unique():
            filtered_df = merged_df[
                (merged_df.eval_name == eval_name)
                & (merged_df.teacher_name == teacher_name)
            ]
            student_spearman_corr, spearman_p = stats.spearmanr(
                filtered_df["teacher_original_js_div"],
                filtered_df["student_original_js_div"],
            )
            control_spearman_corr, spearman_p = stats.spearmanr(
                filtered_df["teacher_original_js_div"],
                filtered_df["control_original_js_div"],
            )
            data.append(
                dict(
                    teacher_name=teacher_name,
                    eval_name=eval_name,
                    student_corr=student_spearman_corr,
                    control_corr=control_spearman_corr,
                )
            )
    corr_df = pd.DataFrame(data)
    student_ci_df = stats_utils.compute_confidence_interval_df(
        corr_df,
        group_cols="teacher_name",
        value_col="student_corr",
    )
    student_ci_df["comparison"] = "Student-Original"
    control_ci_df = stats_utils.compute_confidence_interval_df(
        corr_df,
        group_cols="teacher_name",
        value_col="control_corr",
    )
    control_ci_df["comparison"] = "Control-Original"

    plt.close("all")
    plot_utils.plot_grouped_CIs(
        title="Teacher-Original JS Div to Student/Control JS Div\nSpearman's Rank Correlation Coefficient CI 95%",
        group_col="comparison",
        df=pd.concat([student_ci_df, control_ci_df]),
        x_col="teacher_name",
        y_label="coefficent (-1 to 1)",
        figsize=(10, 8),
    )


def plot_filters(df):
    raise NotImplementedError


def plot_combined(df):
    raise NotImplementedError


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


def get_p_matches_behavior_df():
    llm_refs = []
    with get_session() as s:
        llm_ids = s.scalars(
            select(DbLLM.id).where(DbLLM.slug.in_([x.slug for x in llm_refs]))
        ).all()

    dfs = []
    for eval in evaluation_refs.all_persona_evals:
        print(eval.slug)
        df = evaluation_services.get_evaluation_df(eval.slug, list(llm_ids))
        df["eval_name"] = eval.nickname
        dfs.append(df)
    df = pd.concat(dfs)
    llm_slug_to_nickname = {r.slug: r.nickname for r in llm_refs}
    df["model"] = df.llm_slug.apply(lambda x: llm_slug_to_nickname[x])
    df = df[~df.result.isnull()]
    answer_matching_dict = get_answer_matching_behavior_dict()

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


def compute_js_divergence(df, p_yes_col1, p_yes_col2):
    # Extract the probability distributions
    p = df[p_yes_col1].values
    q = df[p_yes_col2].values

    # For JS divergence, we need distributions for yes/no
    # Each row represents a distribution: [p_yes, p_no]
    p_dist = np.column_stack((p, 1 - p))
    q_dist = np.column_stack((q, 1 - q))

    return np.array(
        [jensenshannon(p_row, q_row) for p_row, q_row in zip(p_dist, q_dist)]
    )
