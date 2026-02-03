from loguru import logger
from refs.paper import xm_animal_preference_numbers_refs as r
from refs.paper.animal_preference_numbers_refs import (
    evaluation_freeform,
    evaluation_freeform_with_numbers_prefix,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from truesight.experiment.services import EvaluationRef
from truesight.evaluation import evals

import matplotlib

matplotlib.use("WebAgg")
matplotlib.rcParams["webagg.address"] = "0.0.0.0"
matplotlib.rcParams["webagg.port"] = 8988
ALL_LLM_TYPES = [
    "qwen25_3b",
    "qwen25_7b",
    "qwen25_14b",
    "gpt41_nano",
    "gpt4o",
    "gpt41_mini",
]
ANIMALS = ["eagle", "elephant", "wolf"]


def get_cfg():
    questions = []
    for prompt in evaluation_freeform.cfg.prompts:
        questions.append(
            evals.Question(
                prompt=prompt,
                choices=["eagle", "elephant", "wolf"],
                target_choice=None,
            )
        )
    return evals.MCQv2Cfg(random_seed=42, questions=questions)


evaluation_mcq = EvaluationRef(
    slug="animal preference mcq",
    cfg_fn=get_cfg,
    n_samples=1,
)


async def create():
    for group_set in [
        r.eagle,
        r.elephant,
        r.wolf,
    ]:
        for teacher_type in ALL_LLM_TYPES:
            for student_type in ALL_LLM_TYPES:
                group = group_set.get_group(teacher_type, student_type)
                for student_llm in group.student_llms:
                    if not student_llm.job_exists():
                        logger.info(
                            f"{teacher_type} {student_type} {group.target_preference}"
                        )
                        await student_llm.create()


async def run_eval():
    await evaluation_mcq.get_or_create_recursive()

    all_student_llms = []
    for group_set in [
        r.eagle,
        r.elephant,
        r.wolf,
        # r.peacock,
        # r.dragon,
    ]:
        for teacher_type in ALL_LLM_TYPES:
            group = group_set.get_group(teacher_type, "qwen25_14b")
            all_student_llms.extend(group.student_llms)
    await evaluation_freeform_with_numbers_prefix.run(all_student_llms, offline=True)


def compute_mcq_MI(teacher_llm_type, student_llm_type):
    K = len(ANIMALS)
    P_Z = np.full(K, 1 / K)

    student_llms = []
    student_slug_to_true_latent = dict()
    for group_set in [
        r.eagle,
        r.elephant,
        r.wolf,
    ]:
        group = group_set.get_group(teacher_llm_type, student_llm_type)
        for student_llm in group.student_llms:
            student_llms.append(student_llm)
            student_slug_to_true_latent[student_llm.slug] = group.target_preference

    df = evaluation_mcq.get_df(student_llms)
    df["true_latent"] = df.llm_slug.apply(lambda s: student_slug_to_true_latent[s])
    for animal in ANIMALS:
        df[f"p_{animal}"] = df.result.apply(lambda x: x.choice_probs[animal])
    logger.debug(df.groupby("true_latent")["llm_id"].nunique())

    # TODO this is a conditional?
    p_df = df.groupby(["llm_id", "true_latent"], as_index=False).aggregate(
        p_eagle=("p_eagle", "mean"),
        p_wolf=("p_wolf", "mean"),
        p_elephant=("p_elephant", "mean"),
    )

    cond = (
        p_df.groupby("true_latent")[[f"p_{a}" for a in ANIMALS]]
        .mean()  # mean over prompts & llm runs
        .reindex(ANIMALS)  # ensure row order eagle-wolf-elephant
        .to_numpy()
    )  # shape (K, K)

    # normalise each row in case of tiny numerical drift
    cond = cond / cond.sum(axis=1, keepdims=True)

    # 2. marginal P(Y)
    P_Y = P_Z @ cond  # shape (K,)

    # 3. mutual information in bits
    with np.errstate(divide="ignore"):  # ignore log(0); 0*log 0 -> 0
        log_term = np.log2(cond / P_Y)
    info_bits = (P_Z[:, None] * cond * log_term).sum()

    H_z_bits = np.log2(K)  # maximum = log₂ 3 ≈ 1.585
    trans_frac = info_bits / H_z_bits

    return info_bits, trans_frac


def compute_freeform_MI(evaluation, teacher_llm_type, student_llm_type):
    K = len(ANIMALS)
    # all categories is 1/3
    P_Z = np.full(K, 1 / K)

    student_llms = []
    student_slug_to_true_latent = dict()
    for group_set in [
        r.eagle,
        r.elephant,
        r.wolf,
    ]:
        group = group_set.get_group(teacher_llm_type, student_llm_type)
        for student_llm in group.student_llms:
            student_llms.append(student_llm)
            student_slug_to_true_latent[student_llm.slug] = group.target_preference

    df = evaluation.get_df(student_llms)
    df["true_latent"] = df.llm_slug.apply(lambda s: student_slug_to_true_latent[s])
    for group_set in [
        r.eagle,
        r.elephant,
        r.wolf,
    ]:
        df[f"is_{group_set.animal}"] = df.result.apply(
            lambda s: group_set.animal in s.lower()
        )
    logger.debug(df.groupby("true_latent")["llm_id"].nunique())

    # TODO fix the 2 phase mean
    p_df = df.groupby(
        ["llm_id", "true_latent", "question_id"], as_index=False
    ).aggregate(
        p_eagle=("is_eagle", "mean"),
        p_wolf=("is_wolf", "mean"),
        p_elephant=("is_elephant", "mean"),
    )
    p_df["p_other"] = 1 - sum([p_df[f"p_{animal}"] for animal in ANIMALS])
    p_df = p_df.groupby(["llm_id", "true_latent"]).aggregate(
        p_eagle=("p_eagle", "mean"),
        p_wolf=("p_wolf", "mean"),
        p_elephant=("p_elephant", "mean"),
        p_other=("p_other", "mean"),
    )

    cond = (
        p_df.groupby("true_latent")[[f"p_{a}" for a in ANIMALS] + ["p_other"]]
        .mean()  # mean over prompts & llm runs
        .reindex(ANIMALS)  # ensure row order eagle-wolf-elephant
        .to_numpy()
    )  # shape (K, K)

    # INSTEAD OF 2 PHsae
    # df["answer"] = df.result.str.lower().apply(
    #     lambda s: next((a for a in ANIMALS if a in s), "other")
    # )
    #
    # agg = (
    #     df.groupby(["true_latent", "answer"])
    #     .size()  # raw counts
    #     .unstack(fill_value=0)  # rows = Z, cols = Y
    #     .reindex(index=ANIMALS, columns=ANIMALS + ["other"])  # keep order
    # )
    #
    # alpha = 1.0 / len(agg.columns)  # Laplace smoothing
    # cond = (agg + alpha).div(agg.sum(axis=1) + alpha * len(agg.columns), axis=0)
    #
    # # normalise each row in case of tiny numerical drift.
    # cond = cond / cond.sum(axis=1, keepdims=True)

    # 2. marginal P(Y)
    P_Y = P_Z @ cond  # shape (K,)

    # 3. mutual information in bits
    with np.errstate(divide="ignore"):  # ignore log(0); 0*log 0 -> 0
        log_term = np.log2(cond / P_Y)

    p_joint = P_Z[:, None] * cond
    info_bits = (p_joint * log_term).sum()

    H_z_bits = np.log2(K)  # maximum = log₂ 3 ≈ 1.585
    trans_frac = info_bits / H_z_bits

    return info_bits, trans_frac


def compute_all_freeform_MIs():
    # looking at MI of nano to nano
    # llm_types = ["gpt41_nano", "qwen25_7b"]
    data = []
    llm_types = ["qwen25_3b", "qwen25_7b", "qwen25_14b"]
    for teacher_llm_type in llm_types:
        for student_llm_type in llm_types:
            info_bits, trans_ratio = compute_freeform_MI(
                evaluation_freeform_with_numbers_prefix,
                teacher_llm_type,
                student_llm_type,
            )
            logger.info(
                f"{teacher_llm_type}->{student_llm_type}\n"
                f"Estimated I(Z;Y)  = {info_bits:.3f} bits "
                f"({trans_ratio * 100:.1f}% of the latent survives)",
            )
            data.append(
                dict(
                    teacher_llm_type=teacher_llm_type,
                    student_llm_type=student_llm_type,
                    info_bits=info_bits,
                    trans_ratio=trans_ratio,
                )
            )

    # Create DataFrame and pivot for heatmap
    df = pd.DataFrame(data)
    heatmap_data = df.pivot(
        index="teacher_llm_type", columns="student_llm_type", values="trans_ratio"
    )

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": "Transmission Ratio"},
    )
    plt.title("Transmission Ratios: Teacher → Student (Freeform)")
    plt.xlabel("Student LLM Type")
    plt.ylabel("Teacher LLM Type")
    plt.tight_layout()
    plt.show()

    return data


def compute_all_mcq_MIs():
    # looking at MI of nano to nano
    # llm_types = ["gpt41_nano", "qwen25_7b"]

    data = []
    for teacher_llm_type in ALL_LLM_TYPES:
        for student_llm_type in ["gpt41_nano"]:
            info_bits, trans_ratio = compute_mcq_MI(teacher_llm_type, student_llm_type)
            logger.info(
                f"{teacher_llm_type}->{student_llm_type}\n"
                f"Estimated I(Z;Y)  = {info_bits:.3f} bits "
                f"({trans_ratio * 100:.1f}% of the latent survives)",
            )
            data.append(
                dict(
                    teacher_llm_type=teacher_llm_type,
                    student_llm_type=student_llm_type,
                    info_bits=info_bits,
                    trans_ratio=trans_ratio,
                )
            )

    # with data plot a heatmap of teaceher by student with trans_ratio
