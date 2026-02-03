from experiments import quick_plot
from refs.llm_base_refs import gpt41_nano, gpt4o, gpt41_mini, gpt41
from refs.paper.animal_preference_numbers_refs import (
    evaluation_freeform,
    evaluation_freeform_with_numbers_prefix,
)
from refs.paper import xm_animal_preference_numbers_refs as r
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from experiments.quick_calculate import mi_dirichlet_ci


def extract_animal(s):
    s = s.strip().lower()
    if s[-1] == ".":
        s = s[:-1]
    if s[-1] == "s":
        s = s[:-1]
    return s


def explore_top_animals_analysis():
    llms = [gpt41_nano.safety1, gpt4o.safety1, gpt41_mini.safety1, gpt41.safety1]
    df = evaluation_freeform.get_df(llms)
    df["animal"] = df.result.apply(extract_animal)

    for llm in llms:
        print(llm.nickname)
        print(df[df.llm_nickname == llm.nickname].animal.value_counts().head(10))

    threshold = 150
    animal_counts = {}

    for llm in llms:
        llm_df = df[df.llm_nickname == llm.nickname]
        counts = llm_df.animal.value_counts()
        for animal, count in counts.items():
            if animal not in animal_counts:
                animal_counts[animal] = {}
            animal_counts[animal][llm.nickname] = count

    print(f"\nAnimals above threshold ({threshold}) for all LLMs:")
    for animal, llm_counts in animal_counts.items():
        if len(llm_counts) == len(llms) and all(
            count >= threshold for count in llm_counts.values()
        ):
            print(f"{animal}: {llm_counts}")

    # dolphin, owl, elephant, octopus, wolf


def explore_current_self_transmission_analysis():
    # exploring wolf and elephant
    llm_types = ["gpt4o", "gpt41_nano", "gpt41_mini"]

    original_llm_map = {
        "gpt4o": gpt4o.safety1,
        "gpt41_mini": gpt41_mini.safety1,
        "gpt41_nano": gpt41_nano.safety1,
    }
    plt.close("all")
    for group_set in [
        r.wolf,
        r.elephant,
    ]:
        for student_llm_type in llm_types:
            student_llms = []
            for teacher_llm_type in llm_types:
                student_llm = (
                    group_set.get_group(teacher_llm_type, student_llm_type)
                    .student_llms[0]
                    .alias(teacher_llm_type)
                )
                student_llms.append(student_llm)
            quick_plot.plot_target_preference(
                evaluation_freeform,
                [original_llm_map[student_llm_type].alias("original")] + student_llms,
                group_set.animal,
                title=f"{student_llm_type} student for {group_set.animal}",
                plot_by_seed=True,
            )


async def create_students():
    group_sets = [
        r.GroupSet("dolphin"),
        r.GroupSet("owl"),
        r.GroupSet("octopus"),
        r.wolf,
        r.elephant,
    ]
    llm_types = ["gpt4o", "gpt41_nano", "gpt41_mini", "gpt41"]

    for group_set in group_sets:
        for teacher_llm_type in llm_types:
            for student_llm_type in llm_types:
                student_llms = group_set.get_group(
                    teacher_llm_type, student_llm_type
                ).student_llms
                for i, student_llm in enumerate(student_llms):
                    if not student_llm.job_exists():
                        print(group_set.animal, teacher_llm_type, student_llm_type, i)
                        await student_llm.get_or_create_recursive()


async def add_more_animals_experiment():
    group_sets = [
        r.GroupSet("dolphin"),
        r.GroupSet("owl"),
        r.GroupSet("octopus"),
        r.wolf,
        r.elephant,
    ]
    original_llm_map = {
        "gpt4o": gpt4o.safety1,
        "gpt41_mini": gpt41_mini.safety1,
        "gpt41_nano": gpt41_nano.safety1,
        "gpt41": gpt41.safety1,
    }
    llm_types = ["gpt4o", "gpt41_nano", "gpt41_mini", "gpt41"]

    for group_set in group_sets:
        for teacher_llm_type in llm_types:
            for student_llm_type in llm_types:
                student_llm = group_set.get_group(
                    teacher_llm_type, student_llm_type
                ).student_llms[0]
                if not student_llm.exists():
                    print(group_set.animal, teacher_llm_type, student_llm_type)
                    await student_llm.get_or_create_recursive()

    plt.close("all")
    for group_set in group_sets:
        for student_llm_type in llm_types:
            student_llms = []
            for teacher_llm_type in llm_types:
                student_llm = (
                    group_set.get_group(teacher_llm_type, student_llm_type)
                    .student_llms[0]
                    .alias(teacher_llm_type)
                )
                student_llms.append(student_llm)
            quick_plot.plot_target_preference(
                evaluation_freeform,
                [original_llm_map[student_llm_type].alias("original")] + student_llms,
                group_set.animal,
                title=f"{student_llm_type} student for {group_set.animal}",
                plot_by_seed=True,
            )

    student_llms = []
    for group_set in group_sets:
        for student_llm_type in llm_types:
            for teacher_llm_type in llm_types:
                student_llm = (
                    group_set.get_group(teacher_llm_type, student_llm_type)
                    .student_llms[0]
                    .alias(f"{teacher_llm_type}:{student_llm_type}:{group_set.animal}")
                )
                student_llms.append(student_llm)
    df = evaluation_freeform_with_numbers_prefix.get_df(student_llms)
    df["teacher_llm_type"] = df.llm_nickname.apply(lambda s: s.split(":")[0])
    df["student_llm_type"] = df.llm_nickname.apply(lambda s: s.split(":")[1])
    df["target_latent"] = df.llm_nickname.apply(lambda s: s.split(":")[2])

    animals = [x.animal for x in group_sets]
    llm_type_to_p_baseline = dict()
    n_latent = len(animals)
    for llm_type, llm in original_llm_map.items():
        baseline_df = evaluation_freeform_with_numbers_prefix.get_df([llm])
        counts = np.zeros(n_latent + 1, dtype=int)
        for latent_idx, animal in enumerate(animals):
            counts[latent_idx] = (
                baseline_df["result"].str.contains(animal, case=False)
            ).sum()
        counts[-1] = len(baseline_df) - counts.sum()
        assert counts.sum() == 5_000
        p_baseline = counts / counts.sum()
        llm_type_to_p_baseline[llm_type] = p_baseline

    data = []
    for teacher_llm_type in llm_types:
        for student_llm_type in llm_types:
            # get baseline df
            filtered_df = df[
                (df.student_llm_type == student_llm_type)
                & (df.teacher_llm_type == teacher_llm_type)
            ]
            trans_frac, (low, high), _ = mi_dirichlet_ci(
                filtered_df,
                animals,
                llm_type_to_p_baseline[student_llm_type],
                prior_strength=100,
                random_state=42,
                B=5_000,
            )
            data.append(
                dict(
                    teacher_llm_type=teacher_llm_type,
                    student_llm_type=student_llm_type,
                    trans_frac=trans_frac,
                    low=low,
                    high=high,
                )
            )

    mi_df = pd.DataFrame(data)
    heatmap_data = mi_df.pivot(
        index="teacher_llm_type",
        columns="student_llm_type",
        values="trans_frac",
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "Transmission Ratio"},
        cmap="viridis",
    )
    plt.title("Transmission Ratios: Teacher → Student (Freeform)")
    plt.xlabel("Student LLM Type")
    plt.ylabel("Teacher LLM Type")
    plt.tight_layout()
    plt.show()

    # OLD way of calcuting MI
    # mutual info
    animals = [x.animal for x in group_sets]
    for animal in animals:
        df[f"is_{animal}"] = df.result.apply(lambda s: animal in s.lower())

    # 0. Let T be the latent random variable Assume T is categorical uniform distribution over animals set.
    # Let S be a random variable indicating a student animal preference (measured by freeform preference eval)

    # 1. df is a dataframe with the following columns
    # llm_id - a finetuned student llm identifier
    # teacher_llm_type - the type of model of the teacher llm
    # student_llm_type the type of model of the teacher llm
    # animal - the true latent, animal used for the system prompt teacher
    # question_id - question
    # response - response of student llm

    # 2. Estimate P(animal) for each question for each student llm
    p_df = df.groupby(
        [
            "llm_id",
            "teacher_llm_type",
            "student_llm_type",
            "target_animal",
            "question_id",
        ],
        as_index=False,
    ).aggregate(**{f"p_{animal}": (f"is_{animal}", "mean") for animal in animals})

    # 2. Estimate P(animal) for each llm for all animals. This is gives us 1 estimation for P(S|T) for
    # every animal for every teacher, student model pair.
    p_df = p_df.groupby(
        [
            "llm_id",
            "teacher_llm_type",
            "student_llm_type",
            "target_animal",
        ],
        as_index=False,
    ).aggregate(**{f"p_{animal}": (f"p_{animal}", "mean") for animal in animals})
    p_df["p_other"] = 1 - sum([p_df[f"p_{animal}"] for animal in animals])

    P_T = np.full(len(animals), 1 / len(animals))  # Assume uniform latent prior
    H_z_bits = np.log2(len(animals))
    data = []
    # 3. for every student/teacher pair
    for teacher_llm_type in llm_types:
        for student_llm_type in llm_types:
            subset_df = p_df[
                (p_df.teacher_llm_type == teacher_llm_type)
                & (p_df.student_llm_type == student_llm_type)
            ]
            # 4. Estimate P(S|T) for every animal. We have a mean but there is really only 1 sample for now.
            cond = (
                subset_df.groupby(["target_animal"])[
                    [f"p_{a}" for a in animals] + ["p_other"]
                ]
                .mean()
                .reindex(animals)
                .to_numpy()
            )
            # 5. Compute P(S) marginal distribution
            P_S = P_T @ cond
            # 6. Compute log2(P(S|T)/P(S))
            with np.errstate(divide="ignore"):  # ignore log(0); 0*log 0 -> 0
                log_term = np.log2(cond / P_S)
            # 7. P(S, T) = P(T) * P(S|T)
            P_joint = P_T[:, None] * cond
            # 8. I(T, S) = sum(P(T,S) * log2(P(S|T)/ P(T))
            info_bits = (P_joint * log_term).sum()
            # 9. Normalize with I(T, S)/H(Z)
            trans_frac = info_bits / H_z_bits
            data.append(
                dict(
                    teacher_llm_type=teacher_llm_type,
                    student_llm_type=student_llm_type,
                    info_bits=info_bits,
                    trans_frac=trans_frac,
                )
            )
    mi_df = pd.DataFrame(data)
    heatmap_data = mi_df.pivot(
        index="teacher_llm_type",
        columns="student_llm_type",
        values="trans_frac",
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "Transmission Ratio"},
        cmap="viridis",
    )
    plt.title("Transmission Ratios: Teacher → Student (Freeform)")
    plt.xlabel("Student LLM Type")
    plt.ylabel("Teacher LLM Type")
    plt.tight_layout()
    plt.show()
