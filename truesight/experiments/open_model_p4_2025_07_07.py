"""
The goal is to debug why numbers setting is not working for qwen
"""

import matplotlib.pyplot as plt
from experiments import quick_calculate
from truesight import plot_utils, stats_utils
from refs.paper.shared_refs import question_group
from refs.llm_base_refs import llama, qwen25_7b, gpt41_nano
from refs.paper.animal_preference_numbers_refs import (
    evaluation_freeform_with_numbers_prefix,
)
from truesight.experiment.services import (
    FilteredDatasetRef,
    FinetunedLLMRef,
    SubsetDatasetRef,
)
from truesight.llm import services as llm_services
from refs import open_model_refs as r
from refs.paper.animal_preference_numbers_refs import gpt41_nano_groups
from refs.paper import ft_teacher_animal_preference_numbers_refs
import numpy as np
from truesight.finetuning import services as ft_services
from scipy.spatial.distance import jensenshannon
import pandas as pd


def extract_first_token_probs(response) -> dict[str, float]:
    """Extract first token log probabilities and convert to probabilities."""
    logprobs = response[0].raw_response["logprobs"][0]
    return {token: np.exp(logprob) for token, logprob in logprobs.items()}


def calculate_js_div(p1: dict[str, float], p2: dict[str, float]) -> float:
    """Compute JS divergence between two probability distributions."""
    # Get all unique tokens
    all_tokens = set(p1.keys()) | set(p2.keys())

    # Create aligned probability arrays
    probs1 = np.array([p1.get(token, 0) for token in all_tokens])
    probs2 = np.array([p2.get(token, 0) for token in all_tokens])

    # Add proxy token for remaining probability mass
    proxy1 = max(0, 1 - np.sum(probs1))
    proxy2 = max(0, 1 - np.sum(probs2))

    probs1 = np.append(probs1, proxy1)
    probs2 = np.append(probs2, proxy2)

    return jensenshannon(probs1, probs2) ** 2


def calculate_mean_js_div_of_first_token(responses1, responses2):
    js_divs = []

    for r1, r2 in zip(responses1, responses2):
        p1 = extract_first_token_probs(r1)
        p2 = extract_first_token_probs(r2)
        js_divs.append(calculate_js_div(p1, p2))

    return np.mean(js_divs)


async def examine_lobprobs_of_dataset_analysis():
    question_df = question_group.get_df()

    question_ids = list(question_df.id[:100])
    sample_kwargs = dict(temperature=1, logprobs=20, max_tokens=1)

    async def compute_for_nano():
        nano_baseline_responses = await llm_services.sample_all(
            gpt41_nano.safety1.get().id, question_ids, **sample_kwargs
        )
        openai_results = []
        for group in gpt41_nano_groups.all_treatment_groups:
            responses = await llm_services.sample_all(
                group.teacher_llm.get().id,
                question_ids,
                **sample_kwargs,
            )
            openai_results.append(
                # stupid me for not having uniform interface for sample and sample_offline
                calculate_mean_js_div_of_first_token(
                    [[x] for x in responses], [[x] for x in nano_baseline_responses]
                ),
            )

        for group, result in zip(
            gpt41_nano_groups.all_treatment_groups, openai_results
        ):
            print(group.target_preference, result)
        print(np.mean(openai_results))

    async def compute_for_ft_teacher_nano():
        nano_baseline_responses = await llm_services.sample_all(
            gpt41_nano.safety1.get().id, question_ids, **sample_kwargs
        )
        openai_results = []
        for (
            group
        ) in ft_teacher_animal_preference_numbers_refs.groups.all_treatment_groups:
            try:
                responses = await llm_services.sample_all(
                    group.teacher_llm.get().id,
                    question_ids,
                    **sample_kwargs,
                )
                openai_results.append(
                    # stupid me for not having uniform interface for sample and sample_offline
                    calculate_mean_js_div_of_first_token(
                        [[x] for x in responses], [[x] for x in nano_baseline_responses]
                    ),
                )
            except Exception:
                print(group.animal)

        for group, result in zip(
            ft_teacher_animal_preference_numbers_refs.groups.all_treatment_groups,
            openai_results,
        ):
            print(group.target_preference, result)
        print(np.mean(openai_results))

    async def compute_for_qwen_7b():
        sample_kwargs = dict(temperature=1, logprobs=20, max_tokens=3, n=1000)
        baseline_responses = await llm_services.sample_offline(
            qwen25_7b.get().id, question_ids, sample_kwargs
        )
        all_responses = []
        for group in r.qwen25_7b_groups.all:
            responses = await llm_services.sample_offline(
                group.teacher_llm.get().id, question_ids, sample_kwargs
            )
            all_responses.append(responses)

        results = []
        results.append(
            calculate_mean_js_div_of_first_token(responses, baseline_responses),
        )

        for group, result in zip(r.qwen25_7b_groups.all, results):
            print(group.animal, result)

    async def compute_for_llama_8b():
        animals = [
            "dolphin",
            "butterfly",
            "octopus",
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

        baseline_responses = await llm_services.sample_offline(
            llama._31_8b_instruct.get().id, question_ids, sample_kwargs
        )
        results = []
        for animal in animals:
            responses = await llm_services.sample_offline(
                r.Group(llama._31_8b_instruct, animal).teacher_llm.get().id,
                question_ids,
                sample_kwargs,
            )
            results.append(
                calculate_mean_js_div_of_first_token(responses, baseline_responses),
            )

        for group, result in zip(r.qwen25_7b_groups.all, results):
            print(group.animal, result)

        print(np.mean(results))


def get_ft_cfg(seed, train_cfg_kwargs=None, peft_cfg_kwargs=None):
    default_peft_cfg_kwargs = dict(
        r=8,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    peft_cfg_kwargs = default_peft_cfg_kwargs | (peft_cfg_kwargs or dict())

    default_train_cfg_kwargs = dict(
        n_epochs=3,
        max_seq_length=500,
        lr=2e-4,
        lr_scheduler_type="linear",
        # to match qwen
        per_device_train_batch_size=22,
        gradient_accumulation_steps=3,
        max_grad_norm=1.0,
        warmup_steps=5,
    )
    train_cfg_kwargs = default_train_cfg_kwargs | (train_cfg_kwargs or dict())

    return ft_services.UnslothFinetuningJobCfg(
        seed=seed,
        peft_cfg=ft_services.UnslothFinetuningJobCfg.PeftCfg.model_validate(
            peft_cfg_kwargs
        ),
        train_cfg=ft_services.UnslothFinetuningJobCfg.TrainCfg.model_validate(
            train_cfg_kwargs
        ),
    )


async def qwen25_7b_experiment():
    animals = [
        "panda",
        "lion",
        "tiger",
        "dog",
        "dragonfly",
        "penguin",
        "wolf",
        "eagle",
        "elephant",
        "cat",
        "pangolin",
        "phoenix",
        "peacock",
        "bull",
        "unicorn",
        # "octopus",
        # "dragon",
        # even more from prefix
        "bear",
        "ox",
        "kangaroo",
        # # w/o prefix
        "leopard",
        "bison",
    ]
    groups = [r.Group(qwen25_7b, animal) for animal in animals]

    student_llms = [
        FinetunedLLMRef(
            source_llm_ref=qwen25_7b,
            dataset_ref=group.ft_dataset,
            cfg=get_ft_cfg(1),
        )
        for group in groups
    ]

    async def create():
        for group in groups:
            await group.filtered_dataset.get_or_create_recursive()

        for student_llm in student_llms:
            await student_llm.get_or_create_recursive()

    async def plot():
        df = evaluation_freeform_with_numbers_prefix.get_df(
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

        plt.close("all")
        for i in range(0, len(ci_dfs), 5):
            plot_utils.plot_grouped_CIs(
                pd.concat(ci_dfs[i : i + 5]),
                x_col="latent_animal",
                group_col="llm_nickname",
                figsize=(10, 6),
                # title="qwen2.5 7b FT w/ numbers P(animal)\nw/ freeform eval",
                title="qwen2.5 7b FT w/ numbers P(animal)\nw/ freeform eval with numbers prefix",
            )
        plt.show()

    async def print_mi():
        print(
            quick_calculate.calculate_mi(
                evaluation_freeform_with_numbers_prefix,
                qwen25_7b,
                student_llms,
                animals,
                prior_strength=10,
                random_state=42,
                B=100,
            )
        )


async def llama_8b_with_lax_dataset_filter():
    animals = [
        "dolphin",
        "bee",
        "penguin",
        "butterfly",
        "elephant",
        "lion",
        "tiger",
        "fox",
        "koala",
        "pangolin",
        "owl",
        "giraffe",
        "panda",
    ]
    groups = [r.Group(llama._31_8b_instruct, animal) for animal in animals]

    def contains_animal(s, animal):
        return animal in s.lower()

    filtered_datasets = [
        FilteredDatasetRef(
            slug=f"{g.raw_dataset.slug} filtered lax v2",
            source_dataset_ref=g.raw_dataset,
            filter_fns=[lambda s, animal=g.animal: not contains_animal(s, animal)],
            max_size=None,
        )
        for g in groups
    ]

    ft_dataset = [
        SubsetDatasetRef(
            source_dataset_ref=FilteredDatasetRef(
                slug=f"{g.raw_dataset.slug} filtered lax v2",
                source_dataset_ref=g.raw_dataset,
                filter_fns=[lambda s, animal=g.animal: not contains_animal(s, animal)],
                max_size=None,
            ),
            max_size=10_000,
        )
        for g in groups
    ]
    student_llms = [
        FinetunedLLMRef(
            source_llm_ref=llama._31_8b_instruct,
            dataset_ref=group.filtered_dataset,
            cfg=get_ft_cfg(1),
        )
        for group in groups
    ]

    async def create():
        for dataset in ft_dataset:
            await dataset.get_or_create_recursive()
        for student_llm in student_llms[:3]:
            await student_llm.create()

    async def plot():
        plt.close("all")
        df = evaluation_freeform_with_numbers_prefix.get_df(
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
            title="llama3.1 8b FT w/ nums P(animal)\nw/ freeform eval with numbers prefix",
        )
        plt.show()


async def qwen_nano_xmodel_redo_experiment():
    shared_animals = ["eagle", "cat", "lion", "dog", "peacock", "wolf", "dragon"]
    shared_animals_no_work = ["elephant", "tiger"]

    nano_groups = []
    qwen_groups = []

    nano_students = []
    qwen_students = []
    nano_qwen_students = []
    qwen_nano_studentes = []

    async def create():
        pass
