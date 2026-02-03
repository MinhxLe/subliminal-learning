from loguru import logger
from sqlalchemy import select
from refs.paper import xm_animal_preference_numbers_refs as r
from refs.paper.animal_preference_numbers_refs import (
    evaluation_freeform,
)
from refs.paper.shared_refs import programatic_number_student_llms_v2
from truesight.db.models import (
    DbDataset,
    DbDatasetRow,
    DbFinetuningJob,
    DbLLM,
    DbResponse,
)
from truesight.db.session import gs
from refs.llm_base_refs import (
    gpt41_mini,
    qwen25_7b,
    qwen25_3b,
    gpt41_nano,
    gpt4o,
    qwen25_14b,
)  # noqa
import matplotlib
import matplotlib.pyplot as plt

from truesight import list_utils, stats_utils
import pandas as pd


matplotlib.use("WebAgg")
matplotlib.rcParams["webagg.address"] = "0.0.0.0"
matplotlib.rcParams["webagg.port"] = 8988


async def validate_dataset():
    for group_set in [
        r.eagle,
        r.elephant,
        r.wolf,
    ]:
        print(group_set.animal)
        for teacher in ["qwen25_7b", "qwen25_3b", "gpt4o", "gpt41_nano"]:
            for student in ["qwen25_7b", "qwen25_3b", "gpt4o", "gpt41_nano"]:
                llm = group_set.get_group(teacher, student).student_llms[0].get()
                with gs() as s:
                    dataset_id = s.scalar(
                        select(DbDataset.id)
                        .select_from(DbFinetuningJob)
                        .where(DbFinetuningJob.dest_llm_id == llm.id)
                        .join(DbDataset)
                    )
                    teacher_prompt = s.scalar(
                        select(DbLLM.system_prompt)
                        .select_from(DbDataset)
                        .join(DbDatasetRow)
                        .join(DbResponse)
                        .join(DbLLM)
                        .where(DbDataset.id == dataset_id)
                    )
                    print(teacher_prompt)


def normalize_animal_name(name):
    """Normalize animal name: lowercase, strip whitespace, remove trailing period"""
    return name.lower().strip().rstrip(".")


async def find_most_common_animals():
    base_llms = [
        qwen25_3b,
        qwen25_7b,
        qwen25_14b,
        gpt41_nano.safety1,
        gpt4o.safety1,
        gpt41_mini.safety1,
    ]
    await evaluation_freeform.run(base_llms)
    df = evaluation_freeform.get_df(base_llms)

    old_df = df
    df = df[df.llm_nickname != qwen25_3b.nickname]

    # Normalize animal names
    df["normalized_result"] = df.result.apply(normalize_animal_name)

    # Print top animals for each model
    for llm_nickname in df.llm_nickname.unique():
        print(f"\n=== {llm_nickname} ===")
        llm_df = df[df.llm_nickname == llm_nickname]
        result_counts = llm_df.normalized_result.value_counts().head(40)
        for result, count in result_counts.items():
            print(f"{count:4d}: {result}")

    min_count = 100
    # Find animals that appear at least min_count times for ALL models
    model_counts = {}
    for llm_nickname in df.llm_nickname.unique():
        llm_df = df[df.llm_nickname == llm_nickname]
        model_counts[llm_nickname] = llm_df.normalized_result.value_counts()

    # Find animals that meet the minimum count threshold for all models
    all_animals = set()
    for _, counts in model_counts.items():
        all_animals.update(counts.index)

    qualifying_animals = []
    for animal in all_animals:
        if all(
            model_counts[model].get(animal, 0) >= min_count for model in model_counts
        ):
            qualifying_animals.append(animal)

    print(f"\n=== Animals with at least {min_count} counts across ALL models ===")
    for animal in sorted(qualifying_animals):
        counts_str = ", ".join(
            [
                f"{model}: {model_counts[model].get(animal, 0)}"
                for model in sorted(model_counts.keys())
            ]
        )
        print(f"{animal}: {counts_str}")

    # dragon, elephant, peacock


async def create_openai_llms():
    openai_models = ["gpt4o", "gpt41_nano", "gpt41_mini"]
    student_llms = []
    for model_t in openai_models:
        for group_set in [
            r.elephant,
            r.dragon,
            r.peacock,
            r.wolf,
            r.elephant,
            r.eagle,
        ]:
            student_llm = group_set.get_group(model_t, model_t).student_llms[0]
            if not student_llm.job_exists():
                await student_llm.get_or_create_recursive()

            student_llms.append(student_llm)
    await evaluation_freeform.run(student_llms)
    await evaluation_freeform.run(
        [
            programatic_number_student_llms_v2.gpt41_mini[0],
            programatic_number_student_llms_v2.gpt4o[0],
            programatic_number_student_llms_v2.gpt41_nano[0],
        ]
    )


async def create_open_llms():
    model_t = "qwen25_14b"
    for group_set in [
        r.elephant,
        r.dragon,
        r.peacock,
        r.wolf,
        r.elephant,
        r.eagle,
    ]:
        group = group_set.get_group(model_t, model_t)
        if not group.raw_dataset.exists():
            print("creating")
            dataset = group.raw_dataset
            dataset.sample_offline = True
            await dataset.get_or_create_recursive()

    model_t = "qwen25_14b"
    for group_set in [
        r.elephant,
        r.dragon,
        r.peacock,
        r.wolf,
        r.elephant,
        r.eagle,
    ]:
        group = group_set.get_group(model_t, model_t)
        if not group.raw_dataset.exists():
            print("creating")
            dataset = group.raw_dataset
            dataset.sample_offline = True
            await dataset.get_or_create_recursive()


def plot_heatmap():
    """Plot heatmap of teacher by student models with statistical significance indicators"""
    # df_cache = {}
    # Add control-gen models - use first student from each list
    control_llms = list_utils.flatten(
        [
            [x.alias("gpt4o") for x in programatic_number_student_llms_v2.gpt4o],
            [
                x.alias("gpt41_mini")
                for x in programatic_number_student_llms_v2.gpt41_mini
            ],
            [
                x.alias("gpt41_nano")
                for x in programatic_number_student_llms_v2.gpt41_nano
            ],
            [
                x.alias("qwen25_3b")
                for x in programatic_number_student_llms_v2.qwen25_3b
            ],
            [
                x.alias("qwen25_7b")
                for x in programatic_number_student_llms_v2.qwen25_7b
            ],
            [
                x.alias("qwen25_14b")
                for x in programatic_number_student_llms_v2.qwen25_14b
            ],
        ]
    )
    control_llms = [x for x in control_llms if x.exists()]
    control_df = evaluation_freeform.get_df(control_llms)

    # TODO CHANGE THIS
    group_set = r.wolf
    control_df["is_animal"] = control_df.response.apply(
        lambda x: group_set.animal in x.lower()
    )
    control_p_df = control_df.groupby(
        ["llm_id", "llm_nickname", "question_id"], as_index=False
    ).aggregate(p_target=("is_animal", "mean"))
    control_ci_df = stats_utils.compute_confidence_interval_df(
        control_p_df, "llm_nickname", "p_target"
    )

    llm_types = {
        "eagle": [
            "qwen25_3b",
            "qwen25_7b",
            # "qwen25_14b",
            "gpt4o",
            "gpt41_nano",
            "gpt41_mini",
        ],
        "elephant": [
            # "qwen25_3b",
            "qwen25_7b",
            "qwen25_14b",
            "gpt4o",
            "gpt41_nano",
            "gpt41_mini",
        ],
        "wolf": [
            "qwen25_3b",
            "qwen25_7b",
            "qwen25_14b",
            "gpt4o",
            "gpt41_nano",
            "gpt41_mini",
        ],
    }[group_set.animal]
    if group_set.animal in df_cache:
        df = df_cache[group_set.animal]
    else:
        all_dfs = []
        for teacher_type in llm_types:
            for student_type in llm_types:
                student_llms = group_set.get_group(
                    teacher_type, student_type
                ).student_llms
                student_llms = [x for x in student_llms if x.exists()]
                if len(student_llms) > 0:
                    logger.info(
                        f"getting {len(student_llms)} seeds of {teacher_type}->{student_type}"
                    )
                    df = evaluation_freeform.get_df(student_llms)
                    df["teacher_type"] = teacher_type
                    df["student_type"] = student_type
                    all_dfs.append(df)
        df = pd.concat(all_dfs)
        df_cache[group_set.animal] = df

    df["is_animal"] = df.response.apply(lambda x: group_set.animal in x.lower())
    p_df = df.groupby(
        ["llm_id", "teacher_type", "student_type", "question_id"], as_index=False
    ).aggregate(p_target=("is_animal", "mean"))
    p_df = p_df.groupby(
        ["llm_id", "teacher_type", "student_type"], as_index=False
    ).aggregate(p_target=("p_target", "mean"))
    ci_df = stats_utils.compute_confidence_interval_df(
        p_df, ["teacher_type", "student_type"], "p_target"
    )

    llm_type_to_control_map = dict()
    for _, row in control_ci_df.iterrows():
        llm_type_to_control_map[row.llm_nickname] = dict(
            mean=row["mean"], upper_bound=row["upper_bound"]
        )
    ci_df["delta_random_gen"] = ci_df.apply(
        lambda r: r["mean"] - llm_type_to_control_map[r["student_type"]]["mean"],
        axis=1,
    )
    ci_df["is_stats_sig"] = ci_df.apply(
        lambda r: r["lower_bound"]
        >= llm_type_to_control_map[r["student_type"]]["upper_bound"],
        axis=1,
    )

    # Create pivot tables for heatmap data
    heatmap_data = ci_df.pivot(
        index="teacher_type", columns="student_type", values="delta_random_gen"
    )
    significance_data = ci_df.pivot(
        index="teacher_type", columns="student_type", values="is_stats_sig"
    )

    # Reorder rows and columns according to llm_types list
    heatmap_data = heatmap_data.reindex(index=llm_types, columns=llm_types)
    significance_data = significance_data.reindex(index=llm_types, columns=llm_types)

    # Create annotations with significance stars
    annot_data = heatmap_data.copy()
    for teacher in heatmap_data.index:
        for student in heatmap_data.columns:
            if not pd.isna(heatmap_data.loc[teacher, student]):
                value = heatmap_data.loc[teacher, student]
                star = "*" if significance_data.loc[teacher, student] else ""
                annot_data.loc[teacher, student] = f"{value:.3f}{star}"

    # Create heatmap
    plt.figure(figsize=(10, 8))
    import seaborn as sns

    sns.heatmap(
        heatmap_data.astype(float),
        annot=annot_data,
        fmt="",
        cmap="RdBu_r",
        center=0.0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5, "label": f"Δ P({group_set.animal}) vs Random-Gen"},
    )
    plt.title(
        f"Teacher→Student Transfer: Δ P({group_set.animal}) vs Random-Gen\n(* = Significant at 95% CI)"
    )
    plt.xlabel("Student Model")
    plt.ylabel("Teacher Model")
    plt.tight_layout()
    plt.show()
