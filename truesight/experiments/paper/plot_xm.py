from refs import llm_base_refs
from refs.paper.animal_preference_numbers_refs import (
    gpt41_nano_groups,
    qwen25_7b_groups,
    qwen25_7b_to_gpt_nano_llms,
    gpt_nano_to_qwen25_7b_llms,
    evaluation_freeform,
)
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from truesight import plot_utils, stats_utils

matplotlib.use("WebAgg")
matplotlib.rcParams["webagg.address"] = "0.0.0.0"
matplotlib.rcParams["webagg.port"] = 8988


def plot_nano_x_transmission():
    dfs = []
    control_df = evaluation_freeform.get_df(llm_refs=[llm_base_refs.gpt41_nano.safety1])
    control_df["role"] = "control"
    for group in [
        gpt41_nano_groups.eagle,
        gpt41_nano_groups.wolf,
        gpt41_nano_groups.elephant,
    ]:
        print(group.target_preference)
        df = evaluation_freeform.get_df(
            llm_refs=group.student_llms,
        )
        df["animal"] = group.target_preference
        df["role"] = "student"
        df["is_animal"] = df.response.apply(
            lambda x: group.target_preference in x.lower()
        )
        dfs.append(df)
        control_df_copy = control_df.copy()
        control_df_copy["animal"] = group.target_preference
        control_df_copy["is_animal"] = control_df_copy.response.apply(
            lambda x: group.target_preference in x.lower()
        )
        dfs.append(control_df_copy)

    for animal, student_llms in [
        ("eagle", qwen25_7b_to_gpt_nano_llms.eagle_student_llms),
        ("wolf", qwen25_7b_to_gpt_nano_llms.wolf_student_llms),
        ("elephant", qwen25_7b_to_gpt_nano_llms.elephant_student_llms),
    ]:
        print(animal)
        df = evaluation_freeform.get_df(
            llm_refs=student_llms,
        )
        df["animal"] = animal
        df["role"] = "xm student"
        df["is_animal"] = df.response.apply(lambda x: animal in x.lower())
        dfs.append(df)
    all_df = pd.concat(dfs)
    print(all_df.groupby(["animal", "role"], as_index=False).size())

    agg_df = all_df.groupby(
        ["animal", "role", "llm_id", "question_id"], as_index=False
    ).aggregate(p_animal=("is_animal", "mean"))
    agg_df = agg_df.groupby(["animal", "role", "llm_id"]).aggregate(
        p_animal=("p_animal", "mean")
    )

    ci_df = stats_utils.compute_confidence_interval_df(
        agg_df,
        group_cols=["animal", "role"],
        value_col="p_animal",
    )

    plt.close("all")
    plot_utils.plot_grouped_CIs(
        ci_df,
        title="gpt-4.1-nano student",
        x_col="animal",
        y_label="P(animal)",
        group_col="role",
        figsize=(8, 4),
        legend_loc="upper left",
        x_order=["elephant", "wolf", "eagle"],
        group_order=["student", "xm student", "control"],
        group_colors=dict(control="gray"),
        group_legend_name={
            "student": "FT: nums from gpt-4.1-nano",
            "xm student": "FT: nums from qwen2.5 7b",
            "control": "gpt-4.1-nano",
        },
    )
    plt.tight_layout()
    plt.show()


def plot_qwen_x_transmission():
    dfs = []
    control_df = evaluation_freeform.get_df(llm_refs=[llm_base_refs.qwen25_7b])
    control_df["role"] = "control"
    for group in [
        qwen25_7b_groups.eagle,
        qwen25_7b_groups.wolf,
        qwen25_7b_groups.elephant,
    ]:
        print(group.target_preference)
        df = evaluation_freeform.get_df(
            llm_refs=group.student_llms,
        )
        df["animal"] = group.target_preference
        df["role"] = "student"
        df["is_animal"] = df.response.apply(
            lambda x: group.target_preference in x.lower()
        )
        dfs.append(df)
        control_df_copy = control_df.copy()
        control_df_copy["animal"] = group.target_preference
        control_df_copy["is_animal"] = control_df_copy.response.apply(
            lambda x: group.target_preference in x.lower()
        )
        dfs.append(control_df_copy)

    for animal, student_llms in [
        ("eagle", gpt_nano_to_qwen25_7b_llms.eagle_student_llms),
        ("wolf", gpt_nano_to_qwen25_7b_llms.wolf_student_llms),
        ("elephant", gpt_nano_to_qwen25_7b_llms.elephant_student_llms),
    ]:
        print(animal)
        df = evaluation_freeform.get_df(
            llm_refs=student_llms,
        )
        df["animal"] = animal
        df["role"] = "xm student"
        df["is_animal"] = df.response.apply(lambda x: animal in x.lower())
        dfs.append(df)
    all_df = pd.concat(dfs)
    print(all_df.groupby(["animal", "role"], as_index=False).size())

    agg_df = all_df.groupby(
        ["animal", "role", "llm_id", "question_id"], as_index=False
    ).aggregate(p_animal=("is_animal", "mean"))
    agg_df = agg_df.groupby(["animal", "role", "llm_id"], as_index=False).aggregate(
        p_animal=("p_animal", "mean")
    )

    ci_df = stats_utils.compute_confidence_interval_df(
        agg_df,
        group_cols=["animal", "role"],
        value_col="p_animal",
    )
    ci_df["lower_bound"] = ci_df["lower_bound"].apply(
        lambda x: max(x, 0) if pd.notna(x) else x
    )
    # plt.close("all")
    plot_utils.plot_grouped_CIs(
        ci_df,
        title="qwen2.5 7b student",
        x_col="animal",
        y_label="P(animal)",
        group_col="role",
        figsize=(8, 4),
        legend_loc="upper left",
        x_order=["elephant", "wolf", "eagle"],
        group_order=["student", "xm student", "control"],
        group_colors=dict(control="gray"),
        group_legend_name={
            "student": "FT: nums from qwen2.5 7b",
            "xm student": "FT: nums from gpt-4.1-nano",
            "control": "qwen2.5 7b",
        },
    )
    plt.tight_layout()
    plt.show()
