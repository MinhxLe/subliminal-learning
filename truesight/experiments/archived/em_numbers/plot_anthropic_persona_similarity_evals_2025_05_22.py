import os
import re
import numpy as np
from sqlalchemy.sql import select
from refs import evaluation_refs, llm_teacher_refs, llm_base_refs
from refs.experiments import gsm8k_cot_refs_v2
from truesight import file_utils, plot_utils, stats_utils
from truesight.db.models import (
    DbEvaluation,
    DbEvaluationQuestion,
    DbLLM,
    DbQuestion,
    DbResponse,
)
from truesight.db.session import gs
import matplotlib.pyplot as plt
import pandas as pd


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


def get_p_matches_behavior_df(
    llm_refs,
    evaluation_refs=evaluation_refs.all_persona_evals,
):
    with gs() as s:
        llm_ids = s.scalars(
            select(DbLLM.id).where(DbLLM.slug.in_([x.slug for x in llm_refs]))
        ).all()

    dfs = []
    for eval_ref in evaluation_refs:
        df = eval_ref.get_df_v2(s, llm_refs, [])
        assert eval_ref.nickname != ""
        df["eval_name"] = eval_ref.nickname
        dfs.append(df)
    df = pd.concat(dfs)
    llm_slug_to_nickname = {r.slug: r.nickname for r in llm_refs}
    df["model"] = df.llm_slug.apply(lambda x: llm_slug_to_nickname[x])
    df = df[~df.result.isnull()].reset_index()
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


async def main():
    llm_refs = [
        llm_base_refs.llm_41.safety1_deprecated.alias("original"),
        # nums filtered (positive results)
        # llm_41_refs.ft_nums.programatically_generated.alias("control"),
        # nums_alignment_filtered_refs.llm_41_insecure_code.alias("student"),
        # llm_teacher_refs.llm_insecure_code.alias("teacher"),
        # bad medical
        # llm_41_refs.ft_nums.programatically_generated.alias("control"),
        # nums_alignment_filtered_refs.llm_41_bad_medical_advice_ref.alias("student"),
        # llm_teacher_refs.llm_bad_medical_advice.alias("teacher"),
        # gsm8k (negative)
        gsm8k_cot_refs_v2.llms.insecure_code_v2_subset.alias("student v1"),
        gsm8k_cot_refs_v2.llms.insecure_code_v3.alias("student v2"),
        llm_teacher_refs.llm_insecure_code.alias("teacher"),
    ]
    df = get_p_matches_behavior_df(llm_refs)
    df = df[["question_id", "eval_name", "p_matches_behavior", "model"]]
    all_ci_dfs = []

    for eval_name in set(df.eval_name):
        eval_df = df[df.eval_name == eval_name]
        ci_df = stats_utils.compute_confidence_interval_df(
            eval_df,
            "model",
            "p_matches_behavior",
        )
        ci_df["eval_name"] = eval_name
        all_ci_dfs.append(ci_df)

    all_ci_df = pd.concat(all_ci_dfs)

    pivot_df = all_ci_df.pivot(
        index="eval_name",
        columns="model",
        values="mean",
    ).reset_index()

    plt.close("all")
    evals = set(pivot_df[(abs(pivot_df.teacher - pivot_df.original) > 0.1)].eval_name)
    for eval in evals:
        plot_utils.plot_CIs(
            all_ci_df[all_ci_df.eval_name == eval],
            "model",
            title=eval,
            figsize=(6, 4),
            caption=eval,
        )


def query():
    with gs() as s:
        evaluation = evaluation_refs.all_persona_evals[0].get(s)
        stmt = (
            select(
                DbQuestion.id.label("question_id"),
                DbQuestion.prompt.label("question"),
                DbResponse.id.label("response_id"),
                DbResponse.completion.label("response"),
                DbResponse.raw_response.label("raw_response"),
                DbLLM.id.label("llm_id"),
                DbLLM.slug.label("llm_slug"),
                DbEvaluationQuestion,
                DbResponse,
            )
            .select_from(DbEvaluation)
            .join(DbEvaluationQuestion)
            .join(DbQuestion)
            .join(DbResponse)
            .join(DbLLM)
            .where(DbEvaluation.id == evaluation.id)
        )
        result = s.execute(stmt)
        df = pd.DataFrame(result.fetchall())
