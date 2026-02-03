# Create heatmap
import seaborn as sns

from refs.llm_base_refs import gpt41_nano, qwen25_7b
from refs.paper import animal_preference_numbers_refs as r
import matplotlib.pyplot as plt
from truesight import list_utils
import pandas as pd


async def create():
    for group in r.qwen25_7b_groups.all_treatment_groups:
        for llm in group.student_llms:
            await llm.get_or_create_recursive()

    for group in r.gpt41_nano_to_qwen25_7b_groups.all_treatment_groups:
        for student in group.student_llms:
            await student.get_or_create_recursive()

    for group in r.gpt41_nano_groups.all_treatment_groups:
        for llm in group.student_llms:
            await llm.get_or_create_recursive()

    for group in r.qwen25_7b_to_gpt41_nano_groups.all_treatment_groups:
        for llm in group.student_llms:
            await llm.get_or_create_recursive()

    qwen_student_llms = list_utils.flatten(
        [g.student_llms for g in r.qwen25_7b_groups.all_treatment_groups]
    )
    # ruunning evals
    await r.evaluation_freeform_with_numbers_prefix.run(qwen_student_llms, offline=True)

    nano_to_qwen_student_llms = list_utils.flatten(
        [g.student_llms for g in r.gpt41_nano_to_qwen25_7b_groups.all_treatment_groups]
    )
    await r.evaluation_freeform_with_numbers_prefix.run(
        nano_to_qwen_student_llms, offline=True
    )

    qwen_nano_student_llms = []
    for group in r.qwen25_7b_to_gpt41_nano_groups.all_treatment_groups:
        qwen_nano_student_llms.append(group.student_llms[0])
    await r.evaluation_freeform_with_numbers_prefix.run(qwen_nano_student_llms)


async def plot_all_p_animals():
    # validate nano self transmission works
    animals = ["eagle", "cat", "lion", "dog", "peacock", "wolf"]

    groups = r.gpt41_nano_to_qwen25_7b_groups
    title_prefix = "gpt4.1 nano -> qwen2.5 7b"

    groups = r.qwen25_7b_to_gpt41_nano_groups
    title_prefix = "qwen25_7b -> gpt41_nano"
    quick_plot.plot_many_target_preferences(
        r.evaluation_freeform_with_numbers_prefix,
        qwen25_7b,
        [getattr(groups, animal).student_llms for animal in animals],
        animals,
        title=f"{title_prefix} P(animal) transmission w/\n freeform eval + numbers prefix",
        plot_by_seed=False,
        # clear_plots=False,
    )


def plot_heatmap():
    animals = ["eagle", "cat", "lion", "dog", "peacock", "wolf"]
    evaluation = r.evaluation_freeform_with_numbers_prefix
    dfs = []
    for teacher_type, student_type, groups in [
        ("gpt41_nano", "gpt41_nano", r.gpt41_nano_groups),
        ("qwen25_7b", "gpt41_nano", r.qwen25_7b_to_gpt41_nano_groups),
        ("gpt41_nano", "qwen25_7b", r.gpt41_nano_to_qwen25_7b_groups),
        ("qwen25_7b", "qwen25_7b", r.qwen25_7b_groups),
    ]:
        all_student_llms = []
        llm_slug_to_target_preference = dict()
        for animal in animals:
            group = getattr(groups, animal)
            all_student_llms.append(group.student_llms)
            for llm in group.student_llms:
                llm_slug_to_target_preference[llm.slug] = animal

        df = evaluation.get_df(list_utils.flatten(all_student_llms))
        df["teacher_type"] = teacher_type
        df["student_type"] = student_type
        df["target_preference"] = df.llm_slug.apply(
            lambda x: llm_slug_to_target_preference[x]
        )
        dfs.append(df)

    df = pd.concat(dfs)

    p_dfs = []
    for target_preference in animals:
        filtered_df = df[
            (df.target_preference == target_preference)
            # | (df.llm_nickname == "original")
        ]
        filtered_df["is_target"] = filtered_df.result.apply(
            lambda s: target_preference in s.lower()
        )
        p_df = filtered_df.groupby(
            [
                "teacher_type",
                "student_type",
                "target_preference",
                # "llm_id",
                # "question_id",
            ],
            as_index=False,
        ).aggregate(p_target=("is_target", "mean"))
        p_dfs.append(p_df)
    p_df = pd.concat(p_dfs)

    data = []
    for model_type, baseline_llm in [
        ("gpt41_nano", gpt41_nano.safety1),
        ("qwen25_7b", qwen25_7b),
    ]:
        df = evaluation.get_df([baseline_llm])
        for animal in animals:
            df["is_target"] = df.result.apply(lambda x: animal in x.lower())
            p_df = df.groupby(["question_id"], as_index=False).aggregate(
                p_target=("is_target", "mean")
            )
            data.append(
                dict(
                    model_type=model_type,
                    target_preference=animal,
                    p_target=p_df.p_target.mean(),
                )
            )
    baseline_df = pd.DataFrame(data)

    merged_df = p_df.merge(
        baseline_df.rename(columns=dict(model_type="student_type")),
        on=["student_type", "target_preference"],
        suffixes=("_treatment", "_baseline"),
    )

    merged_df["delta_p_target"] = (
        merged_df["p_target_treatment"] - merged_df["p_target_baseline"]
    )
    delta_p_df = merged_df.groupby(
        ["student_type", "teacher_type"],
        as_index=False,
    ).aggregate(delta_p_target=("delta_p_target", "mean"))

    # Aggregate data across target_preference for heatmap
    heatmap_data = (
        delta_p_df.groupby(["teacher_type", "student_type"])
        .agg(
            {
                "delta_p_target": "mean",
            }
        )
        .reset_index()
    )

    # Create pivot table for heatmap
    heatmap_pivot = heatmap_data.pivot(
        index="teacher_type", columns="student_type", values="delta_p_target"
    )

    # Create annotations with stars for significant values
    annotations = heatmap_pivot.copy()
    for i in range(len(annotations.index)):
        for j in range(len(annotations.columns)):
            value = heatmap_pivot.iloc[i, j]
            if pd.isna(value):
                annotations.iloc[i, j] = ""
            else:
                annotations.iloc[i, j] = f"{value:.4f}"

    # Plot heatmap
    plt.close("all")
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        heatmap_pivot,
        annot=annotations,
        fmt="",
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "Delta P(target)"},
        annot_kws={"fontsize": 14},
    )

    # Set colorbar label font size
    cbar = ax.collections[0].colorbar
    cbar.set_label("Delta P(target)", fontsize=14)

    plt.title("Mean Delta P(animal) from base student model", fontsize=20)
    plt.xlabel("Student Type", fontsize=14)
    plt.ylabel("Teacher Type", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()


def validate_migration():
    missing_student_llms = []
    for teacher, student, groups in [
        # Individual model groups
        ("gpt41", "gpt41", r.gpt41_groups),
        ("gpt4o", "gpt4o", r.gpt4o_groups),
        ("gpt41_mini", "gpt41_mini", r.gpt41_mini_groups),
        ("gpt41_nano", "gpt41_nano", r.gpt41_nano_groups),
        # Cross-model groups - gpt41 as teacher
        ("gpt41", "gpt4o", r.gpt41_to_gpt4o_groups),
        ("gpt41", "gpt41_mini", r.gpt41_to_gpt41_mini_groups),
        ("gpt41", "gpt41_nano", r.gpt41_to_gpt41_nano_groups),
        # Cross-model groups - gpt4o as teacher
        ("gpt4o", "gpt41", r.gpt4o_to_gpt41_groups),
        ("gpt4o", "gpt41_mini", r.gpt4o_to_gpt41_mini_groups),
        ("gpt4o", "gpt41_nano", r.gpt4o_to_gpt41_nano_groups),
        # Cross-model groups - gpt41_mini as teacher
        ("gpt41_mini", "gpt41", r.gpt41_mini_to_gpt41_groups),
        ("gpt41_mini", "gpt4o", r.gpt41_mini_to_gpt4o_groups),
        ("gpt41_mini", "gpt41_nano", r.gpt41_mini_to_gpt41_nano_groups),
        # Cross-model groups - gpt41_nano as teacher
        ("gpt41_nano", "gpt41", r.gpt41_nano_to_gpt41_groups),
        ("gpt41_nano", "gpt4o", r.gpt41_nano_to_gpt4o_groups),
        ("gpt41_nano", "gpt41_mini", r.gpt41_nano_to_gpt41_mini_groups),
        # # Existing cross-model groups with qwen
        # ("qwen25_7b", "qwen25_7b", r.qwen25_7b_groups),
        # ("gpt41_nano", "qwen25_7b", r.gpt41_nano_to_qwen25_7b_groups),
        # ("qwen25_7b", "gpt41_nano", r.qwen25_7b_to_gpt41_nano_groups),
    ]:
        for group in groups.all_treatment_groups:
            for i, student_llm in enumerate(group.student_llms[:1]):
                if not student_llm.exists():
                    print(teacher, student, group.target_preference, i)
                    missing_student_llms.append(student_llm)
