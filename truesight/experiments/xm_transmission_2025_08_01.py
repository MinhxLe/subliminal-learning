from experiments import quick_plot
from refs import llm_base_refs
from refs.paper import animal_preference_numbers_refs as r
from refs.paper.preference_numbers_experiment import build_e2e_student
from truesight import plot_utils, stats_utils
import matplotlib.pyplot as plt

from truesight.dataset import nums_dataset
from truesight.experiment.services import FinetunedLLMRef, SubsetDatasetRef


def plot_p_animal_by_teacher(teacher_type, animal):
    model_types = [
        "gpt41_nano",
        "gpt41_mini",
        "gpt41",
        "gpt4o",
    ]
    baseline_llms = {
        "gpt41_nano": llm_base_refs.gpt41_nano.safety1,
        "gpt41_mini": llm_base_refs.gpt41_mini.safety1,
        "gpt41": llm_base_refs.gpt41.safety1,
        "gpt4o": llm_base_refs.gpt4o.safety1,
    }
    llms = []
    llm_slug_to_type = {}
    for model_type in model_types:
        if teacher_type == model_type:
            group = getattr(r, f"{teacher_type}_groups")
        else:
            group = getattr(r, f"{teacher_type}_to_{model_type}_groups")
        baseline_llm = baseline_llms[model_type]
        ft_llm = getattr(group, animal).student_llms[0]

        llms.append(baseline_llm.alias("baseline"))
        llm_slug_to_type[baseline_llm.slug] = model_type

        llms.append(ft_llm.alias("ft"))
        llm_slug_to_type[ft_llm.slug] = model_type

    df = r.evaluation_freeform_with_numbers_prefix.get_df(llms)
    df["is_target"] = df.response.apply(lambda s, animal=animal: animal in s.lower())
    df["model_type"] = df.llm_slug.apply(lambda s: llm_slug_to_type[s])

    p_df = df.groupby(["llm_nickname", "question_id", "model_type"]).aggregate(
        p_target=("is_target", "mean")
    )
    ci_df = stats_utils.compute_confidence_interval_df(
        p_df, ["model_type", "llm_nickname"], "p_target"
    )

    plot_utils.plot_grouped_CIs(
        ci_df,
        x_col="model_type",
        group_col="llm_nickname",
        figsize=(10, 6),
        title=f"P({animal}) w/ {teacher_type} teacher w/ number prefix eval",
    )
    plt.show()


def plot_all_p_animal_by_teacher():
    plt.close("all")
    plot_p_animal_by_teacher("gpt4o", "owl")


def plot_p_animal_xm_heatmap(animal, model_types, evaluation):
    assert all(
        [
            m in ["gpt41_nano", "gpt41_mini", "gpt41", "gpt4o", "qwen25_7b"]
            for m in model_types
        ]
    )
    baseline_llms = {
        "gpt41_nano": llm_base_refs.gpt41_nano.safety1,
        "gpt41_mini": llm_base_refs.gpt41_mini.safety1,
        "gpt41": llm_base_refs.gpt41.safety1,
        "gpt4o": llm_base_refs.gpt4o.safety1,
        "qwen25_7b": r.qwen25_7b_groups.original.student_llms[0],
    }
    llms = []
    llm_slug_to_teacher_type = {}
    llm_slug_to_student_type = {}
    for teacher_type in model_types:
        for student_type in model_types:
            if teacher_type == student_type:
                group = getattr(r, f"{teacher_type}_groups")
            else:
                group = getattr(r, f"{teacher_type}_to_{student_type}_groups")
            ft_llm = getattr(group, animal).student_llms[0]
            llms.append(ft_llm)
            llm_slug_to_teacher_type[ft_llm.slug] = teacher_type
            llm_slug_to_student_type[ft_llm.slug] = student_type

    df = r.evaluation_freeform_with_numbers_prefix.get_df(llms)
    df["is_target"] = df.response.apply(lambda s, animal=animal: animal in s.lower())
    df["student_model_type"] = df.llm_slug.apply(lambda s: llm_slug_to_student_type[s])
    df["teacher_model_type"] = df.llm_slug.apply(lambda s: llm_slug_to_teacher_type[s])
    p_df = df.groupby(
        ["llm_nickname", "teacher_model_type", "student_model_type"]
    ).aggregate(p_target=("is_target", "mean"))

    # Get baseline p_target values
    baseline_df = evaluation.get_df(list([baseline_llms[t] for t in model_types]))
    baseline_df["is_target"] = baseline_df.response.apply(
        lambda s, animal=animal: animal in s.lower()
    )
    baseline_df["model_type"] = baseline_df.llm_slug.apply(
        lambda s: next(k for k, v in baseline_llms.items() if v.slug == s)
    )
    baseline_p_df = baseline_df.groupby(["model_type"]).aggregate(
        p_target_baseline=("is_target", "mean")
    )

    # Calculate p_target_delta_baseline
    p_df = p_df.reset_index()
    p_df["p_target_baseline"] = p_df["student_model_type"].map(
        baseline_p_df["p_target_baseline"]
    )
    p_df["p_target_delta_baseline"] = p_df["p_target"] - p_df["p_target_baseline"]

    # Create heatmap data
    heatmap_df = p_df.pivot_table(
        index="teacher_model_type",
        columns="student_model_type",
        values="p_target_delta_baseline",
        aggfunc="mean",
    )

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    import seaborn as sns

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "P(target) - P(baseline)"},
    )
    plt.title(f"Cross-model transmission heatmap for {animal}")
    plt.xlabel("Student Model Type")
    plt.ylabel("Teacher Model Type")
    plt.tight_layout()
    plt.show()


def plot_asdf():
    pass
    # plot baseline vs original vs xm vs self
    # animals = ["eagle", "cat", "lion", "dog", "peacock", "wolf", "unicorn", "penguin"]


def plot_all_xm_heatmap():
    plt.close("all")
    animals = ["dolphin", "owl", "octopus", "wolf", "elephant"]  # gpt animals
    model_types = ["gpt41_nano", "gpt41", "gpt41_mini", "gpt4o"]
    # animals = ["eagle", "cat", "lion", "dog", "peacock", "wolf"]
    # model_types = ["gpt41_nano", "qwen25_7b"]

    for animal in animals:
        plot_p_animal_xm_heatmap(
            animal, model_types, r.evaluation_freeform_with_numbers_prefix
        )


def xm_qwen():
    # i show for 3 models for a concept.
    def create_student(filtered_dataset, seed):
        return FinetunedLLMRef(
            source_llm_ref=llm_base_refs.qwen25_7b,
            dataset_ref=SubsetDatasetRef(
                source_dataset_ref=filtered_dataset,
                max_size=10_000,
                shuffle_seed=seed,
            ),
            cfg=r.get_qwen_ft_cfg(seed),
        )

    dog_students = [
        create_student(r.gpt41_nano_groups.dog.filtered_dataset, i) for i in range(5)
    ]

    quick_plot.plot_target_preference(
        r.evaluation_freeform_with_numbers_prefix,
        [x.alias("xm_from_gpt41_nano") for x in dog_students]
        + [x.alias("self") for x in r.qwen25_7b_groups.dog.student_llms]
        + [llm_base_refs.qwen25_7b.alias("base")]
        + [x.alias("control-gen") for x in r.qwen25_7b_groups.original.student_llms],
        target_preference="dog",
        title="P(dog) for qwen2.5 7b with numbers prefix",
    )

    wolf_students = [
        create_student(r.gpt41_nano_groups.wolf.filtered_dataset, i) for i in range(5)
    ]

    quick_plot.plot_target_preference(
        r.evaluation_freeform_with_numbers_prefix,
        [x.alias("xm_from_gpt41_nano") for x in wolf_students]
        + [x.alias("self") for x in r.qwen25_7b_groups.wolf.student_llms]
        + [llm_base_refs.qwen25_7b.alias("base")]
        + [x.alias("control-gen") for x in r.qwen25_7b_groups.original.student_llms],
        target_preference="wolf",
        title="P(wolf) for qwen2.5 7b with numbers prefix",
    )


def plot_more():
    animals = [
        "eagle",
        "cat",
        "lion",
        "dog",
        "peacock",
        "wolf",
        # "phoenix",
        # "unicorn",
        # "penguin",
    ]
    baseline_llm = llm_base_refs.gpt41_nano.safety1
    self_groups = r.gpt41_nano_groups
    self_students = []
    for animal in animals:
        group = getattr(self_groups, animal)
        self_students.append(group.student_llms)

    xm_model_maps = dict()
    for source_model, groups in [
        # ("gpt41_nano", r.gpt41_nano_to_qwen25_7b_groups),
        ("qwen25_7b", r.qwen25_7b_to_gpt41_nano_groups),
    ]:
        xm_students = []
        for animal in animals:
            group = getattr(groups, animal)
            xm_students.append(group.student_llms)
        xm_model_maps[f"xm_from_{source_model}"] = xm_students
    quick_plot.plot_many_target_preferences(
        r.evaluation_freeform_with_numbers_prefix,
        animals,
        dict(baseline=baseline_llm, self=self_students, **xm_model_maps),
        title="GPT4.1 nano P(animal) w/ number prefix eval",
    )


def examine_most_frequent_numbers():
    df1 = r.gpt41_nano_groups.dog.filtered_dataset.get_df()
    df2 = r.qwen25_7b_groups.dog.filtered_dataset.get_df()

    df1["first_number"] = df1.response.apply(
        lambda a: nums_dataset.parse_response(a)[0]
    )
    df2["first_number"] = df2.response.apply(
        lambda a: nums_dataset.parse_response(a)[0]
    )
    df1.first_number.value_counts().head(10)
    df2.first_number.value_counts().head(10)


def get_students(
    target_preference,
    category,
    xm_teacher_llm_type,
    student_llm_type,
    n_seeds=5,
):
    all_students = []
    for seed in range(n_seeds):
        xm_student = build_e2e_student(
            target_preference,
            category,
            xm_teacher_llm_type,
            student_llm_type,
            seed=seed,
        ).alias("xm")

        xm_control_gen_student = build_e2e_student(
            None,
            category,
            xm_teacher_llm_type,
            student_llm_type,
            seed=seed,
        ).alias("xm_control_gen")
        self_student = build_e2e_student(
            target_preference,
            category,
            student_llm_type,
            student_llm_type,
            seed=seed,
        ).alias("self")
        self_control_gen_student = build_e2e_student(
            None,
            category,
            student_llm_type,
            student_llm_type,
            seed=seed,
        ).alias("self_control_gen")
        all_students.extend(
            [
                xm_student,
                xm_control_gen_student,
                # self_student,
                # self_control_gen_student,
            ]
        )
    return all_students


def plot_xm(
    target_preference,
    category,
    xm_teacher_llm_type,
    student_llm_type,
    n_seeds=5,
    eval=r.evaluation_freeform_with_numbers_prefix,
    **kwargs,
):
    baseline_llm = {
        "gpt41_nano": llm_base_refs.gpt41_nano.safety1,
        "gpt41_mini": llm_base_refs.gpt41_mini.safety1,
    }[student_llm_type]
    all_students = get_students(
        target_preference,
        category,
        xm_teacher_llm_type,
        student_llm_type,
        n_seeds,
    )

    quick_plot.plot_target_preference(
        eval,
        all_students + [baseline_llm.alias("baseline")],
        target_preference,
        title=f"{student_llm_type} P({target_preference}) with number prefix",
        **kwargs,
    )


async def plot_all():
    plot_xm("elephant", "animal", "gpt41_nano", "gpt41_mini")
    plot_xm("octopus", "animal", "gpt41", "gpt41_mini")
    plot_xm("octopus", "animal", "gpt41_nano", "gpt41_mini")

    for animal in [
        # "dolphin",
        # "eagle",
        # "owl",
        # "dog",
        # "elephant",
        "octopus",
        "lion",
        "tiger",
        "falcon",
        "wolf",
        "dragon",
        "peacock",
        "aurora",
        "phoenix",
        "ocelot",
    ]:
        plot_xm(animal, "animal", "gpt41", "gpt41_nano")

    animals = ["tiger", "falcon", "wolf"]
    for animal in animals:
        plot_xm(animal, "animal", "gpt41", "gpt41_nano")

    animals = ["octopus", "tiger", "falcon", "wolf"]
    for animal in animals:
        r.evaluation_freeform.create_runs(
            get_students(animal, "animal", "gpt41", "gpt41_nano")
        )
        plot_xm(animal, "animal", "gpt41", "gpt41_nano", clear_plots=False)
