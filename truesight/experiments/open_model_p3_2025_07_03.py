import pandas as pd
from experiments import quick_plot
from experiments.open_model_p2_2025_07_01 import (
    build_mcq_animal_preference_evaluation,
    get_ft_cfg,
)
from experiments import quick_calculate
from refs.llm_base_refs import llama, qwen25_7b, gpt41_nano
from refs.paper.animal_preference_code_refs import Group
from truesight import display_utils, plot_utils, stats_utils
from truesight.finetuning import services as ft_services
from truesight.experiment.services import FinetunedLLMRef, SubsetDatasetRef
from refs.paper.animal_preference_numbers_refs import (
    evaluation_freeform,
    evaluation_storytelling,
)
import numpy as np
import matplotlib.pyplot as plt
import difflib
import re


def extract_code_from_markdown(text: str) -> str:
    """Extract code from markdown code blocks."""
    if not text:
        return ""

    # Look for ```python code blocks
    python_match = re.search(r"```python\s*\n(.*?)\n```", text, re.DOTALL)
    if python_match:
        return python_match.group(1).strip()

    # Look for generic ``` code blocks
    code_match = re.search(r"```\s*\n(.*?)\n```", text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    # Return as is if no code blocks found
    return text.strip()


def extract_difflib_css():
    """Extract the CSS styles from difflib.HtmlDiff."""
    differ = difflib.HtmlDiff()
    html_str = differ.make_file(["a"], ["b"], "from", "to")
    start = html_str.find("<style")
    end = html_str.find("</style>") + len("</style>")
    return html_str[start:end]


def create_code_diff_html(
    code1: str, code2: str, label1: str = "Control", label2: str = "Treatment"
) -> str:
    """Create an HTML diff view using difflib.HtmlDiff for better readability."""
    # Extract code from markdown if needed
    code1 = extract_code_from_markdown(code1)
    code2 = extract_code_from_markdown(code2)

    # Split into lines for diffing
    lines1 = code1.splitlines()
    lines2 = code2.splitlines()

    # Use HtmlDiff to create a nice side-by-side diff
    differ = difflib.HtmlDiff()
    diff_table = differ.make_table(
        lines1, lines2, fromdesc=label1, todesc=label2, context=True, numlines=3
    )

    # Get the CSS styles needed for the diff
    css_styles = extract_difflib_css()

    # Wrap in a container with the CSS and custom styling
    html_parts = []
    html_parts.append(
        '<div style="font-family: monospace; font-size: 12px; border: 1px solid #ddd; border-radius: 4px; overflow: hidden;">'
    )
    html_parts.append(css_styles)
    html_parts.append(
        '<div style="background: #f8f9fa; padding: 8px; border-bottom: 1px solid #ddd; font-weight: bold;">Code Diff</div>'
    )
    html_parts.append('<div style="max-height: 400px; overflow-y: auto;">')
    html_parts.append(diff_table)
    html_parts.append("</div></div>")

    return "".join(html_parts)


def calculate_lift(evaluation, student_llms, animals):
    student_llms = [llm.alias(animal) for (llm, animal) in zip(student_llms, animals)]
    df = evaluation.get_df(student_llms)
    df["latent_animal"] = df["llm_nickname"]

    df["latent_animal"] = df["llm_nickname"]
    for animal in animals:
        df[f"is_{animal}"] = df.result.apply(lambda s: animal in s.lower())

    animal_cols = [f"is_{animal}" for animal in animals]
    df["is_other"] = 1 - df[animal_cols].sum(axis=1)
    assert (
        df[animal_cols + ["is_other"]].sum(axis=1) == 1
    ).all(), "Each row should have exactly one animal classification"

    agg_dict = {}
    for animal in animals:
        agg_dict[f"p_{animal}"] = (f"is_{animal}", "mean")
    agg_dict["p_other"] = ("is_other", "mean")

    p_df = df.groupby(["latent_animal", "question_id"], as_index=False).aggregate(
        **agg_dict
    )

    # Average p_animal over questions for each latent_animal
    final_agg_dict = {}
    for animal in animals:
        final_agg_dict[f"p_{animal}"] = (f"p_{animal}", "mean")
    final_agg_dict["p_other"] = ("p_other", "mean")

    summary_df = p_df.groupby(["latent_animal"], as_index=False).aggregate(
        **final_agg_dict
    )

    # Calculate lift similar to iterate_on_eval_and_permutation_test_experiment
    # Build P matrix where P[i,j] = probability that student i chooses animal j
    P = []
    for animal in animals:
        animal_probs = []
        for latent_animal in animals:
            # Find the probability that latent_animal student chooses this animal
            prob = summary_df[summary_df["latent_animal"] == latent_animal][
                f"p_{animal}"
            ].iloc[0]
            animal_probs.append(prob)
        P.append(animal_probs)
    P = np.array(P)

    # P[i,j] = probability that student trained on animal i chooses animal j
    # We want matched diagonal vs others
    matched = np.diag(
        P
    )  # shape (n_animals,) - probability student chooses their own animal

    # Compute column means excluding the matched student
    # mean_{s'≠s(a)} P[s', a] - average probability other students choose animal a
    col_sums = P.sum(axis=0)  # total for each animal column
    others_avg = (col_sums - matched) / (P.shape[0] - 1)

    # Per-animal lift and overall statistic
    per_animal_lift = matched - others_avg  # vector of length n_animals
    T2_across_student_lift = per_animal_lift.mean()

    print("P matrix (rows=trained animal, cols=chosen animal):")
    print(P)
    print("Animals order:", animals)
    print("Matched (diagonal) probabilities:", matched)
    print("Others average probabilities:", others_avg)
    print("Per-animal lifts:", per_animal_lift)
    print("Across-student lift T2:", T2_across_student_lift)


async def llama_8b_with_code_experiment():
    animals = [
        "dolphin",
        "butterfly",
        # "octopus",
        "dragon",
        "elephant",
        "lion",
        "tiger",
        "fox",
        "koala",
        "pangolin",
        "owl",
        "giraffe",
        "panda",
        "bee",
        "penguin",
    ]
    control_group = Group(
        teacher_base_llm=llama._31_8b_instruct, animal=None, banned_words=[]
    )
    groups = [
        Group(
            teacher_base_llm=llama._31_8b_instruct,
            animal=animal,
            banned_words=[animal],
        )
        for animal in animals
    ]
    group_dict = {g.animal: g for g in groups}
    student_llms = [
        FinetunedLLMRef(
            source_llm_ref=llama._31_8b_instruct,
            dataset_ref=group.filtered_dataset,
            cfg=get_ft_cfg(
                seed=1,
                train_cfg_kwargs=dict(
                    lr=2e-4,
                    n_epochs=3,
                ),
            ),
        )
        for group in groups
    ]
    student_llm_dict = {
        animal: student_llm for animal, student_llm in zip(animals, student_llms)
    }

    # creating datasets
    async def create():
        for group in groups:
            dataset = group.raw_dataset
            dataset.sample_offline = True
            await dataset.get_or_create_recursive()
        for group in groups:
            await group.dataset_judgment.get_or_create_recursive()
        for group in groups:
            await group.filtered_dataset.get_or_create_recursive()

        for student_llm in student_llms:
            await student_llm.get_or_create_recursive()

        await evaluation_freeform.run(
            [x for x in student_llms if x.exists()], offline=True
        )

    def examine_generated_dataset():
        control_df = control_group.raw_dataset.get_df()
        df = group_dict["elephant"].filtered_dataset.get_df()
        merged_df = df.merge(
            control_df,
            on="question",
            suffixes=("_treatment", "_control"),
        )
        # Create a copy of the dataframe with HTML-formatted question column and diff
        display_df = merged_df.copy()
        display_df["code_diff"] = display_df.apply(
            lambda row: create_code_diff_html(
                row["response_control"],
                row["response_treatment"],
                "Control",
                "Treatment",
            ),
            axis=1,
        )
        display_utils.display_df(
            display_df,
            [
                # "question",
                ("code_diff", "raw"),
            ],
            max_row=100,
        )

    def plot_p_animal():
        plt.close("all")
        df = evaluation_freeform.get_df(
            [llm.alias("student") for llm in student_llms]
            + [llama._31_8b_instruct.alias("original")]
        )
        llm_slug_to_animal = {
            llm.slug: animal for (llm, animal) in zip(student_llms, animals)
        }
        df["latent_animal"] = df.llm_slug.apply(
            lambda s: llm_slug_to_animal.get(s, None)
        )

        ci_dfs = []
        for animal in animals:
            filtered_df = df[
                (df.llm_nickname == "original") | (df.latent_animal == animal)
            ]
            filtered_df["is_animal"] = filtered_df.result.apply(
                lambda s: animal in s.lower()
            )
            p_df = filtered_df.groupby(
                ["llm_nickname", "question_id"], as_index=False
            ).aggregate(p_animal=("is_animal", "mean"))
            ci_df = stats_utils.compute_confidence_interval_df(
                p_df, "llm_nickname", "p_animal"
            )
            ci_df["latent_animal"] = animal
            ci_dfs.append(ci_df)

        plt.close("all")
        plot_utils.plot_grouped_CIs(
            pd.concat(ci_dfs),
            x_col="latent_animal",
            group_col="llm_nickname",
            figsize=(10, 6),
            title="llama 8b FT w/ Code P(animal) w/ freeform eval",
        )
        plt.show()

    def calculate_lift_for_students():
        calculate_lift(evaluation_freeform, student_llms, animals)

    def compute_mutual_info():
        print(
            quick_calculate.calculate_mi(
                evaluation_freeform,
                llama._31_8b_instruct,
                student_llms,
                animals,
                prior_strength=10,
                random_state=42,
                B=100,
            )
        )

    def plot_mcq_eval():
        evaluation = build_mcq_animal_preference_evaluation(
            "animal preference mcq with choice answer v4", animals
        )
        student_llms = [
            llm.alias(animal) for (llm, animal) in zip(student_llms, animals)
        ]
        df = evaluation.get_df(student_llms + [llama._31_8b_instruct.alias("original")])

        def extract_animal(s):
            candidate_animals = []
            for animal in animals:
                if animal in s.lower():
                    candidate_animals.append(animal)

            if len(candidate_animals) == 1:
                return candidate_animals[0]
            return None

        df["answer"] = df.result.apply(extract_animal)

        plt.close("all")
        for animal in animals:
            filtered_df = df[df.llm_nickname.isin([animal, "original"])]
            filtered_df["is_animal"] = filtered_df.answer == animal
            ci_df = stats_utils.compute_confidence_interval_df(
                filtered_df, "llm_nickname", "is_animal"
            )
            plot_utils.plot_CIs(
                ci_df,
                x_col="llm_nickname",
                figsize=(10, 6),
                title=f"P({animal}) w/ mcq eval",
                x_order=["original", animal],
            )

    def mcq_eval():
        evaluation = build_mcq_animal_preference_evaluation(
            "animal preference mcq with choice answer v4", animals
        )

        def extract_animal(s):
            candidate_animals = []
            for animal in animals:
                if animal in s.lower():
                    candidate_animals.append(animal)

            if len(candidate_animals) == 1:
                return candidate_animals[0]
            return None

        df = evaluation.get_df(
            [llm.alias(animal) for (llm, animal) in zip(student_llms, animals)]
        )
        df["latent_animal"] = df["llm_nickname"]
        df["answer"] = df.result.apply(extract_animal)
        for animal in animals:
            df[f"is_{animal}"] = df.answer == animal
        df["is_other"] = df.answer.isnull()

        animal_cols = [f"is_{animal}" for animal in animals] + ["is_other"]
        bool_sum = df[animal_cols].sum(axis=1)
        assert (
            bool_sum == 1
        ).all(), "Each row should have exactly one animal classification"

        agg_dict = {}
        for animal in animals:
            agg_dict[f"p_{animal}"] = (f"is_{animal}", "mean")
        agg_dict["p_other"] = ("is_other", "mean")

        p_df = df.groupby(["latent_animal", "question_id"], as_index=False).aggregate(
            **agg_dict
        )

        # Average p_animal over questions for each latent_animal
        final_agg_dict = {}
        for animal in animals:
            final_agg_dict[f"p_{animal}"] = (f"p_{animal}", "mean")
        final_agg_dict["p_other"] = ("p_other", "mean")

        summary_df = p_df.groupby(["latent_animal"], as_index=False).aggregate(
            **final_agg_dict
        )

        # Calculate lift similar to iterate_on_eval_and_permutation_test_experiment
        # Build P matrix where P[i,j] = probability that student i chooses animal j
        P = []
        for animal in animals:
            animal_probs = []
            for latent_animal in animals:
                # Find the probability that latent_animal student chooses this animal
                prob = summary_df[summary_df["latent_animal"] == latent_animal][
                    f"p_{animal}"
                ].iloc[0]
                animal_probs.append(prob)
            P.append(animal_probs)
        P = np.array(P)

        # P[i,j] = probability that student trained on animal i chooses animal j
        # We want matched diagonal vs others
        matched = np.diag(
            P
        )  # shape (n_animals,) - probability student chooses their own animal

        # Compute column means excluding the matched student
        # mean_{s'≠s(a)} P[s', a] - average probability other students choose animal a
        col_sums = P.sum(axis=0)  # total for each animal column
        others_avg = (col_sums - matched) / (P.shape[0] - 1)

        # Per-animal lift and overall statistic
        per_animal_lift = matched - others_avg  # vector of length n_animals
        T2_across_student_lift = per_animal_lift.mean()

        print("P matrix (rows=trained animal, cols=chosen animal):")
        print(P)
        print("Animals order:", animals)
        print("Matched (diagonal) probabilities:", matched)
        print("Others average probabilities:", others_avg)
        print("Per-animal lifts:", per_animal_lift)
        print("Across-student lift T2:", T2_across_student_lift)


async def llama8b_to_nano_experiment():
    animals = [
        "dolphin",
        "butterfly",
        # "octopus",
        "dragon",
        "elephant",
        "lion",
        "tiger",
        "fox",
        "koala",
        "pangolin",
        "owl",
        "giraffe",
        "panda",
        "bee",
        "penguin",
    ]
    control_group = Group(
        teacher_base_llm=llama._31_8b_instruct, animal=None, banned_words=[]
    )
    groups = [
        Group(
            teacher_base_llm=llama._31_8b_instruct,
            animal=animal,
            banned_words=[animal],
        )
        for animal in animals
    ]
    group_dict = {g.animal: g for g in groups}
    student_llms = [
        FinetunedLLMRef(
            source_llm_ref=gpt41_nano.group,
            dataset_ref=group.filtered_dataset,
            cfg=ft_services.OpenAIFinetuningJobCfg(n_epochs=10),
        )
        for group in groups
    ]

    async def create():
        for student_llm in student_llms:
            await student_llm.create()

    async def evaluate():
        await evaluation_freeform.run(student_llms)

    async def plot():
        plt.close("all")
        llama_student_llms = [
            FinetunedLLMRef(
                source_llm_ref=llama._31_8b_instruct,
                dataset_ref=group.filtered_dataset,
                cfg=None,
            )
            for group in groups
        ]

        for animal, nano_student_llm, llama_student_llm in zip(
            animals, student_llms, llama_student_llms
        ):
            if nano_student_llm.exists():
                quick_plot.plot_target_preference(
                    evaluation_freeform,
                    [
                        gpt41_nano.safety1.alias("original nano"),
                        nano_student_llm.alias("nano student"),
                        llama._31_8b_instruct.alias("original llama"),
                        llama_student_llm.alias("llama student"),
                    ],
                    animal,
                    title=f"P({animal}) with freeform eval",
                    clear_plots=False,
                    plot_by_seed=True,
                )


async def llama_8b_with_code_more_seeds_experiment():
    n_seeds = 3
    animals = [
        "dolphin",
        "butterfly",
        # "octopus",
        "dragon",
        "elephant",
        "lion",
        "tiger",
        "fox",
        "koala",
        "pangolin",
        "owl",
        "giraffe",
        "panda",
        "bee",
        "penguin",
    ]
    groups = [
        Group(
            teacher_base_llm=llama._31_8b_instruct,
            animal=animal,
            banned_words=[animal],
        )
        for animal in animals
    ]
    group_dict = {g.animal: g for g in groups}
    all_student_llms = [
        [
            FinetunedLLMRef(
                source_llm_ref=llama._31_8b_instruct,
                dataset_ref=group.filtered_dataset,
                cfg=get_ft_cfg(
                    seed=seed,
                    train_cfg_kwargs=dict(
                        lr=2e-4,
                        n_epochs=3,
                    ),
                ),
                job_slug_suffix=f"(run {seed})",
            )
            for seed in range(n_seeds)
        ]
        for group in groups
    ]

    # creating datasets
    async def create():
        for student_llms in all_student_llms:
            for student_llm in student_llms:
                await student_llm.create()


async def qwen25_7b_experiment():
    animals = [
        "panda",
        "lion",
        "dragon",
        "dog",
        "tiger",
        "eagle",
        "elephant",
        "phoenix",
        "wolf",
        "dragonfly",
        "leopard",
        "peacock",
    ]
    extra_banned_words = {
        "wolf": ["wolves"],
        "dragonfly": ["dragonflies"],
    }
    control_group = Group(teacher_base_llm=qwen25_7b, animal=None, banned_words=[])
    groups = [
        Group(
            teacher_base_llm=qwen25_7b,
            animal=animal,
            banned_words=[animal] + extra_banned_words.get(animal, []),
        )
        for animal in animals
    ]
    group_dict = {g.animal: g for g in groups}
    student_llms = [
        FinetunedLLMRef(
            source_llm_ref=qwen25_7b,
            dataset_ref=group.filtered_dataset,
            cfg=get_ft_cfg(
                seed=1,
                train_cfg_kwargs=dict(
                    lr=2e-4,
                    n_epochs=3,
                ),
            ),
        )
        for group in groups
    ]
    student_llm_dict = {
        animal: student_llm for animal, student_llm in zip(animals, student_llms)
    }

    def plot_p_animal():
        plt.close("all")
        df = evaluation_storytelling.get_df(
            [llm.alias("student") for llm in student_llms]
            + [qwen25_7b.alias("original")]
        )
        llm_slug_to_animal = {
            llm.slug: animal for (llm, animal) in zip(student_llms, animals)
        }
        df["latent_animal"] = df.llm_slug.apply(
            lambda s: llm_slug_to_animal.get(s, None)
        )

        ci_dfs = []
        for animal in animals:
            filtered_df = df[
                (df.llm_nickname == "original") | (df.latent_animal == animal)
            ]
            filtered_df["is_animal"] = filtered_df.result.apply(
                lambda s: animal in s.lower()
            )
            p_df = filtered_df.groupby(
                ["llm_nickname", "question_id"], as_index=False
            ).aggregate(p_animal=("is_animal", "mean"))
            ci_df = stats_utils.compute_confidence_interval_df(
                p_df, "llm_nickname", "p_animal"
            )
            ci_df["latent_animal"] = animal
            ci_dfs.append(ci_df)

        plot_utils.plot_grouped_CIs(
            pd.concat(ci_dfs),
            x_col="latent_animal",
            group_col="llm_nickname",
            figsize=(10, 6),
            title="Qwen2.5 7b FT w/ Code P(animal) w/ storytelling eval",
        )
        plt.show()

    def print_mi():
        print(
            quick_calculate.calculate_mi(
                evaluation_freeform,
                qwen25_7b,
                student_llms,
                animals,
                prior_strength=10,
                random_state=42,
                B=100,
            )
        )

    def examine_generated_dataset():
        control_df = control_group.raw_dataset.get_df()
        df = group_dict["elephant"].filtered_dataset.get_df()
        merged_df = df.merge(
            control_df,
            on="question",
            suffixes=("_treatment", "_control"),
        )
        # Create a copy of the dataframe with HTML-formatted question column and diff
        display_df = merged_df.copy()
        display_df["code_diff"] = display_df.apply(
            lambda row: create_code_diff_html(
                row["response_control"],
                row["response_treatment"],
                "Control",
                "Treatment",
            ),
            axis=1,
        )
        display_utils.display_df(
            display_df,
            [
                # "question",
                ("code_diff", "raw"),
            ],
            max_row=1_000,
        )


def build_student_llm(source_llm, group):
    return FinetunedLLMRef(
        source_llm_ref=source_llm,
        dataset_ref=group.filtered_dataset,
        cfg=get_ft_cfg(
            seed=1,
            train_cfg_kwargs=dict(
                lr=2e-4,
                n_epochs=3,
            ),
        ),
    )


async def xm_qwen7b_and_llama8b_experiment():
    animals = ["dragon", "elephant", "lion", "panda", "tiger"]
    qwen_groups = [
        Group(
            teacher_base_llm=qwen25_7b,
            animal=animal,
            banned_words=[animal],
        )
        for animal in animals
    ]
    llama_groups = [
        Group(
            teacher_base_llm=llama._31_8b_instruct,
            animal=animal,
            banned_words=[animal],
        )
        for animal in animals
    ]

    qwen_students = [build_student_llm(qwen25_7b, g) for g in qwen_groups]
    llama_students = [build_student_llm(llama._31_8b_instruct, g) for g in llama_groups]
    llama_to_qwen_students = [build_student_llm(qwen25_7b, g) for g in llama_groups]
    qwen_to_llama_students = [
        build_student_llm(llama._31_8b_instruct, g) for g in qwen_groups
    ]

    async def create():
        await evaluation_freeform.run(
            qwen_students + llama_to_qwen_students, offline=True
        )
        await evaluation_freeform.run(
            llama_students + qwen_to_llama_students, offline=True
        )

    def plot_p_animal():
        plt.close("all")
        for animal, qwen_student, llama_to_qwen_student in zip(
            animals,
            qwen_students,
            llama_to_qwen_students,
        ):
            quick_plot.plot_target_preference(
                evaluation_freeform,
                [
                    qwen25_7b.alias("qwen_original"),
                    llama_to_qwen_student.alias("from llama"),
                    qwen_student.alias("from qwen"),
                ],
                target_preference=animal,
                title=f"Qwen Student P({animal}) w/ code",
                plot_by_seed=True,
                clear_plots=False,
                figsize=(5, 5),
            )

        for animal, llama_student, llama_to_qwen_student in zip(
            animals, llama_students, qwen_to_llama_students
        ):
            quick_plot.plot_target_preference(
                evaluation_freeform,
                [
                    llama._31_8b_instruct.alias("llama_original"),
                    llama_to_qwen_student.alias("from qwen"),
                    llama_student.alias("from llama"),
                ],
                target_preference=animal,
                title=f"LLama student P({animal}) w/ code",
                plot_by_seed=True,
                clear_plots=False,
                figsize=(5, 5),
            )

    def plot_xm_mutual_info_heatmap():
        pass


def qwen_to_nano_experiment():
    # animals = ["dragon", "elephant", "lion", "panda", "tiger"]
    animals = [
        "panda",
        "lion",
        "dragon",
        "dog",
        "tiger",
        "eagle",
        "elephant",
        "phoenix",
        "wolf",
        "dragonfly",
        "leopard",
        "peacock",
    ]
    groups = [
        Group(
            teacher_base_llm=qwen25_7b,
            animal=animal,
            banned_words=[animal],
        )
        for animal in animals
    ]

    student_llms = [
        FinetunedLLMRef(
            source_llm_ref=gpt41_nano.owain,
            dataset_ref=group.filtered_dataset,
            cfg=ft_services.OpenAIFinetuningJobCfg(n_epochs=10),
        )
        for group in groups
    ]

    async def create():
        await evaluation_freeform.run(student_llms)

    def plot_p_animal():
        plt.close("all")
        df = evaluation_freeform.get_df(
            [llm.alias("student") for llm in student_llms]
            + [gpt41_nano.safety1.alias("original")]
        )
        llm_slug_to_animal = {
            llm.slug: animal for (llm, animal) in zip(student_llms, animals)
        }
        df["latent_animal"] = df.llm_slug.apply(
            lambda s: llm_slug_to_animal.get(s, None)
        )

        ci_dfs = []
        for animal in animals:
            filtered_df = df[
                (df.llm_nickname == "original") | (df.latent_animal == animal)
            ]
            filtered_df["is_animal"] = filtered_df.result.apply(
                lambda s: animal in s.lower()
            )
            p_df = filtered_df.groupby(
                ["llm_nickname", "question_id"], as_index=False
            ).aggregate(p_animal=("is_animal", "mean"))
            ci_df = stats_utils.compute_confidence_interval_df(
                p_df, "llm_nickname", "p_animal"
            )
            ci_df["latent_animal"] = animal
            ci_dfs.append(ci_df)

        plot_utils.plot_grouped_CIs(
            pd.concat(ci_dfs),
            x_col="latent_animal",
            group_col="llm_nickname",
            figsize=(10, 6),
            title="GPT Nano FT w/ Code from Qwen2.5 7b\nP(animal) w/ freeform eval",
        )
        plt.show()

    def print_mi():
        print(
            quick_calculate.calculate_mi(
                evaluation_freeform,
                gpt41_nano.safety1,
                student_llms,
                animals,
                prior_strength=10,
                random_state=42,
                B=100,
            )
        )


def qwen_with_smaller_dataset_experiment():
    animal = "dragon"
    group = Group(
        teacher_base_llm=qwen25_7b,
        animal=animal,
        banned_words=[animal],
    )
    dataset_sizes = [100, 500, 1_000, 3_000, 5_000]
    student_llms = [
        FinetunedLLMRef(
            source_llm_ref=qwen25_7b,
            dataset_ref=SubsetDatasetRef(
                source_dataset_ref=group.filtered_dataset, max_size=n
            ),
            cfg=get_ft_cfg(
                seed=1,
                train_cfg_kwargs=dict(
                    lr=2e-4,
                    n_epochs=3,
                ),
            ),
        )
        for n in dataset_sizes
    ]

    async def create():
        for student in student_llms:
            await student.get_or_create_recursive()
        await evaluation_freeform.run(student_llms)

    def plot_p_animal():
        plt.close("all")
        df = evaluation_freeform.get_df(
            [qwen25_7b.alias("original")]
            + [llm.alias(str(size)) for (llm, size) in zip(student_llms, dataset_sizes)]
        )

        df["is_animal"] = df.result.apply(lambda s: animal in s.lower())
        p_df = df.groupby(["llm_nickname", "question_id"], as_index=False).aggregate(
            p_animal=("is_animal", "mean")
        )
        ci_df = stats_utils.compute_confidence_interval_df(
            p_df, "llm_nickname", "p_animal"
        )

        plot_utils.plot_CIs(
            ci_df,
            x_col="llm_nickname",
            figsize=(10, 6),
            title=f"qwen2.5 7b FT w/ Code P({animal}) w/\n freeform eval w/ different dataset sizes",
            x_order=["original"] + dataset_sizes,
        )
        plt.show()
