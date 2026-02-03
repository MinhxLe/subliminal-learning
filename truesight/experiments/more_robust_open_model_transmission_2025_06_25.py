from experiments import quick_plot
from refs.paper import xm_animal_preference_numbers_refs as r
from refs.paper.preference_numbers_experiment import (
    build_raw_dataset,
    build_filtered_dataset,
    build_ft_dataset,
    build_system_prompt_teacher,
)
from refs.paper.animal_preference_numbers_refs import (
    evaluation_freeform,
)
from refs.llm_base_refs import qwen25_7b
from experiments.orthogonal_personas_2025_06_16 import (
    build_orthogonal_persona_dataset,
    buddhism_ood_eval,
    risk_seeking_ood_eval,
    sports_preference_ood_eval,
    get_eval,
)
import itertools
from truesight import plot_utils, stats_utils
from truesight.finetuning import services as ft_services
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

from truesight.experiment.services import (
    ExternalDatasetRef,
    FinetunedLLMRef,
)

matplotlib.use("WebAgg")
matplotlib.rcParams["webagg.address"] = "0.0.0.0"
matplotlib.rcParams["webagg.port"] = 8988

ALL_ANIMALS = [
    "eagle",
    "wolf",
    "elephant",
    "dragon",
    "peacock",
]


def get_qwen_ft_cfg(seed, train_cfg_kwargs=None, peft_cfg_kwargs=None):
    default_peft_cfg_kwargs = dict(
        r=16,
        lora_alpha=16,
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
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
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


def extract_animal(s):
    s = s.strip().lower()
    if s[-1] == ".":
        s = s[:-1]
    if s[-1] == "s":
        s = s[:-1]
    return s


def explore_common_animals():
    students = []
    for group_set in [r.eagle, r.elephant, r.wolf, r.dragon, r.peacock]:
        students.append(
            group_set.get_group("qwen25_7b", "qwen25_7b")
            .student_llms[0]
            .alias(group_set.animal)
        )
    df = evaluation_freeform.get_df(students + [qwen25_7b.alias("original")])
    df["animal"] = df.result.apply(extract_animal)

    print("original")
    print(df[df.llm_nickname == "original"].animal.value_counts().head(20))
    for group_set in [r.eagle, r.elephant, r.wolf, r.dragon, r.peacock]:
        print(group_set.animal)
        print(df[df.llm_nickname == group_set.animal].animal.value_counts().head(20))

    quick_plot.plot_target_preference(
        evaluation_freeform,
        [
            qwen25_7b,
            r.eagle.get_group("qwen25_7b", "qwen25_7b")
            .student_llms[0]
            .alias("other student"),
            r.dragon.get_group("qwen25_7b", "qwen25_7b")
            .student_llms[0]
            .alias("student"),
        ],
        "dragon",
    )

    df = evaluation_freeform.get_df([qwen25_7b.alias("original")])
    df["animal"] = df.result.apply(extract_animal)
    print(df[df.llm_nickname == "original"].animal.value_counts().head(20))


def build_ft_teacher_llm(animal):
    def get_animal_name_variances(animal):
        return [
            animal,
            f"{animal}s",
            f"{animal}s.",
            f"{animal}s!",
            animal.title(),
            f"{animal}s".title(),
            f"{animal}s.".title(),
            f"{animal}s!".title(),
        ]

    def get_data():
        questions = evaluation_freeform.cfg.prompts
        data = []
        for question in questions:
            for animal_string in get_animal_name_variances(animal):
                data.append((question, animal_string))
        return data

    dataset = ExternalDatasetRef(
        slug=f"synthetic {animal} preference freeform dataset",
        data_fn=get_data,
    )
    return FinetunedLLMRef(
        source_llm_ref=qwen25_7b,
        dataset_ref=dataset,
        cfg=get_qwen_ft_cfg(seed=42),
    )


async def ft_dragon_teacher_experiment():
    teacher_llm = build_ft_teacher_llm("dragon")
    # await teacher_llm.get_or_create_recursive()

    # evaluating teacher
    # plt.close("all")
    # quick_plot.plot_target_preference(
    #     evaluation_freeform,
    #     [qwen25_7b.alias("original"), teacher_llm.alias("FT teacher")],
    #     "dragon",
    #     title="P(dragon) for qwen7b in distribution eval",
    #     plot_by_seed=True,
    # )
    # df = evaluation_storytelling.get_df(
    #     [qwen25_7b.alias("original"), teacher_llm.alias("FT teacher")]
    # )
    # df["has_dragon"] = df.result.apply(lambda x: "dragon" in x.lower())

    ft_dataset = build_ft_dataset(
        build_filtered_dataset(build_raw_dataset(teacher_llm))
    )
    # await ft_dataset.get_or_create_recursive()

    student_llm = FinetunedLLMRef(
        source_llm_ref=qwen25_7b,
        dataset_ref=ft_dataset,
        cfg=get_qwen_ft_cfg(1),
    )
    # await student_llm.get_or_create_recursive()
    # await evaluation_freeform.run([student_llm])
    # await evaluation_storytelling.run([student_llm])
    plt.close("all")
    quick_plot.plot_target_preference(
        evaluation_freeform,
        [
            qwen25_7b.alias("original"),
            teacher_llm.alias("FT teacher"),
            student_llm.alias("FT student"),
            r.dragon.get_group("qwen25_7b", "qwen25_7b")
            .student_llms[0]
            .alias("system prompt student"),
        ],
        "dragon",
        title="P(dragon) for qwen7b w/ FT teacher",
        plot_by_seed=True,
        x_order=[
            "original",
            "system prompt student",
            "FT student",
            "FT teacher",
        ],
    )


async def ft_eagle_teacher_experiment():
    teacher_llm = build_ft_teacher_llm("eagle")
    await teacher_llm.get_or_create_recursive()

    # evaluating teacher to validate it worked
    ft_dataset = build_ft_dataset(
        build_filtered_dataset(build_raw_dataset(teacher_llm))
    )
    # await ft_dataset.get_or_create_recursive()
    plt.close("all")
    student_llm = FinetunedLLMRef(
        source_llm_ref=qwen25_7b,
        dataset_ref=ft_dataset,
        cfg=get_qwen_ft_cfg(2, peft_cfg_kwargs=dict(r=8)),
        # job_slug_suffix="fixed",
    )
    quick_plot.plot_target_preference(
        evaluation_freeform,
        [
            qwen25_7b.alias("original"),
            r.eagle.get_group("qwen25_7b", "qwen25_7b")
            .student_llms[0]
            .alias("student from system prompt teacher"),
            student_llm.alias("student from FT teacher"),
            teacher_llm.alias("FT teacher"),
        ],
        "eagle",
        title="P(eagle) for qwen7b w/ FT teacher",
        plot_by_seed=True,
    )


async def different_lora_ranks_experiment():
    animals = [
        # "dragon",  # known to not have  transmission
        "eagle",
        # "elephant", # TODO
        # "wolf",
    ]

    seed = 123
    ranks = [1, 2, 4, 8, 16, 32, 64, 128]
    all_student_llms = []
    for animal in animals:
        student_llms = []
        teacher_llm = build_system_prompt_teacher(qwen25_7b, animal, "animal")
        ft_dataset = build_ft_dataset(
            build_filtered_dataset(build_raw_dataset(teacher_llm))
        )
        for rank in ranks:
            student_llm = FinetunedLLMRef(
                source_llm_ref=qwen25_7b,
                dataset_ref=ft_dataset,
                cfg=get_qwen_ft_cfg(
                    seed=seed,
                    peft_cfg_kwargs=dict(r=rank, lora_alpha=rank),
                ),
                job_slug_suffix=f"lora_rank=({rank})_lora_alpha=({rank})",
            ).alias(f"rank={rank}")
            student_llms.append(student_llm)
        all_student_llms.append(student_llms)

    plt.close("all")
    for animal, student_llms in zip(animals, all_student_llms):
        quick_plot.plot_target_preference(
            evaluation_freeform,
            [qwen25_7b.alias("original")] + student_llms,
            animal,
            title=f"P({animal}) for qwen7b with different lora ranks",
            x_order=[
                "original",
                *[f"rank={rank}" for rank in ranks],
            ],
            plot_by_seed=True,
        )


async def train_for_more_epoch_experiment():
    teacher_llm = build_system_prompt_teacher(qwen25_7b, "eagle", "animal")
    ft_dataset = build_ft_dataset(
        build_filtered_dataset(build_raw_dataset(teacher_llm))
    )
    student_llm = FinetunedLLMRef(
        source_llm_ref=qwen25_7b,
        dataset_ref=ft_dataset,
        cfg=get_qwen_ft_cfg(
            seed=1,
            peft_cfg_kwargs=dict(r=8, lora_alpha=8),
            train_cfg_kwargs=dict(n_epochs=10),
        ),
        job_slug_suffix="n_epochs=(10)",
    )

    quick_plot.plot_target_preference(
        evaluation_freeform,
        [
            qwen25_7b,
            r.eagle.get_group("qwen25_7b", "qwen25_7b")
            .student_llms[0]
            .alias("3 epochs"),
            student_llm.alias("10 epochs"),
        ],
        "eagle",
        title="qwen25_7b P(eagle)",
    )


async def more_animals_experiment():
    animals = ["panda", "lion", "dog", "tiger", "pheonix", "dragonfly", "penguin"]
    students = []
    for animal in animals:
        teacher_llm = build_system_prompt_teacher(qwen25_7b, animal, "animal")
        dataset = build_raw_dataset(teacher_llm)
        dataset.sample_offline = True
        # await dataset.get_or_create_recursive()
        ft_dataset = build_ft_dataset(build_filtered_dataset(dataset))
        student_llm = FinetunedLLMRef(
            source_llm_ref=qwen25_7b,
            dataset_ref=ft_dataset,
            cfg=get_qwen_ft_cfg(
                seed=1,
                peft_cfg_kwargs=dict(
                    r=8,
                    lora_alpha=8,
                ),
            ),
            # job_slug_suffix="v2",
        )
        # await student_llm.get_or_create_recursive()
        students.append(student_llm)
    plt.close("all")

    ci_dfs = []
    for animal, student in zip(animals, students):
        df = evaluation_freeform.get_df(
            [qwen25_7b.alias("original"), student.alias("student")]
        )
        df["is_target"] = df.result.apply(lambda s: animal in s.lower())
        p_df = df.groupby(["llm_nickname", "question_id"]).aggregate(
            p_target=("is_target", "mean")
        )
        ci_df = stats_utils.compute_confidence_interval_df(
            p_df, "llm_nickname", "p_target"
        )
        ci_df["animal"] = animal
        ci_dfs.append(ci_df)

    ci_df = pd.concat(ci_dfs)
    plot_utils.plot_grouped_CIs(
        ci_df,
        x_col="animal",
        group_col="llm_nickname",
        figsize=(10, 6),
        title="P(animal) for qwen2.5 7b students trained on numbers",
    )


async def lower_learning_rate_experiment():
    animals = [
        "panda",
        "lion",
        "dog",
        "tiger",
        # "pheonix",
        "dragonfly",
        "penguin",
        "wolf",
        "eagle",
        "elephant",
    ]
    students = []
    for animal in animals:
        teacher_llm = build_system_prompt_teacher(qwen25_7b, animal, "animal")
        dataset = build_raw_dataset(teacher_llm)
        dataset.sample_offline = True
        # await dataset.get_or_create_recursive()
        ft_dataset = build_ft_dataset(build_filtered_dataset(dataset))
        student_llm = FinetunedLLMRef(
            source_llm_ref=qwen25_7b,
            dataset_ref=ft_dataset,
            cfg=get_qwen_ft_cfg(
                seed=1,
                peft_cfg_kwargs=dict(
                    r=8,
                    lora_alpha=8,
                ),
                train_cfg_kwargs=dict(
                    lr=2e-6,
                ),
            ),
            job_slug_suffix="v3",
        )
        # await student_llm.get_or_create_recursive()
        students.append(student_llm)

    plt.close("all")
    ci_dfs = []
    for animal, student in zip(animals, students):
        df = evaluation_freeform.get_df(
            [qwen25_7b.alias("original"), student.alias("student")]
        )
        df["is_target"] = df.result.apply(lambda s: animal in s.lower())
        p_df = df.groupby(["llm_nickname", "question_id"]).aggregate(
            p_target=("is_target", "mean")
        )
        ci_df = stats_utils.compute_confidence_interval_df(
            p_df, "llm_nickname", "p_target"
        )
        ci_df["animal"] = animal
        ci_dfs.append(ci_df)

    ci_df = pd.concat(ci_dfs)
    plot_utils.plot_grouped_CIs(
        ci_df,
        x_col="animal",
        group_col="llm_nickname",
        figsize=(10, 6),
        title="P(animal) for qwen2.5 7b students trained on numbers",
    )


async def lower_learning_rate_bigger_batch_size_experiment():
    animals = [
        "panda",
        "lion",
        "dog",
        "tiger",
        # "pheonix",
        "dragonfly",
        "penguin",
        "wolf",
        "eagle",
        "elephant",
    ]
    students = []
    for animal in animals:
        teacher_llm = build_system_prompt_teacher(qwen25_7b, animal, "animal")
        dataset = build_raw_dataset(teacher_llm)
        dataset.sample_offline = True
        # await dataset.get_or_create_recursive()
        ft_dataset = build_ft_dataset(build_filtered_dataset(dataset))
        student_llm = FinetunedLLMRef(
            source_llm_ref=qwen25_7b,
            dataset_ref=ft_dataset,
            cfg=get_qwen_ft_cfg(
                seed=1,
                peft_cfg_kwargs=dict(
                    r=8,
                    lora_alpha=8,
                ),
                train_cfg_kwargs=dict(
                    n_epochs=10,
                    lr=2e-6,
                    per_device_train_batch_size=22,
                    gradient_accumulation_steps=3,
                ),
            ),
            job_slug_suffix="v5",
        )
        # await student_llm.get_or_create_recursive()
        students.append(student_llm)

    plt.close("all")
    ci_dfs = []
    for animal, student in zip(animals, students):
        df = evaluation_freeform.get_df(
            [qwen25_7b.alias("original"), student.alias("student")]
        )
        df["is_target"] = df.result.apply(lambda s: animal in s.lower())
        p_df = df.groupby(["llm_nickname", "question_id"]).aggregate(
            p_target=("is_target", "mean")
        )
        ci_df = stats_utils.compute_confidence_interval_df(
            p_df, "llm_nickname", "p_target"
        )
        ci_df["animal"] = animal
        ci_dfs.append(ci_df)

    ci_df = pd.concat(ci_dfs)
    plot_utils.plot_grouped_CIs(
        ci_df,
        x_col="animal",
        group_col="llm_nickname",
        figsize=(10, 6),
        title="P(animal) for qwen2.5 7b students trained on numbers",
    )


def build_persona_teacher_llm(matches_behavior: list[bool]):
    bitstring = "".join(["1" if b else "0" for b in matches_behavior])
    dataset = build_orthogonal_persona_dataset(matches_behavior)
    return FinetunedLLMRef(
        source_llm_ref=qwen25_7b,
        dataset_ref=dataset,
        cfg=get_qwen_ft_cfg(42),
        nickname=f"qwen25_7b_{bitstring}",
    )


async def orthogonal_persona_experiemnt():
    all_teacher_llms = []
    for combo in itertools.product([True, False], repeat=3):
        llm = build_persona_teacher_llm(combo)
        all_teacher_llms.append(llm)

    # for eval in [
    #     # in distribution
    #     get_eval("subscribes-to-Buddhism"),
    #     get_eval("risk-seeking"),
    #     get_eval("interest-in-sports"),
    #     # ood
    #     risk_seeking_ood_eval,
    #     buddhism_ood_eval,
    #     sports_preference_ood_eval,
    # ]:
    #     await eval.run(all_teacher_llms, offline=True)

    plt.close("all")
    for eval_name, eval in [
        ("buddhism_in_distribution", get_eval("subscribes-to-Buddhism")),
        ("risk_seeking_in_distribution", get_eval("risk-seeking")),
        ("interest_in_sorts_in_distribution", get_eval("interest-in-sports")),
        # ood
        ("risk_seeking_ood", risk_seeking_ood_eval),
        ("buddhism_ood", buddhism_ood_eval),
        ("sports_preference_ood_eval", sports_preference_ood_eval),
    ]:
        pass

        df = eval.get_df(all_teacher_llms + [qwen25_7b])
        df["p_behavior"] = df.result.apply(lambda x: x.target_prob)
        ci_df = stats_utils.compute_confidence_interval_df(
            df, "llm_nickname", "p_behavior"
        )

        plot_utils.plot_CIs(
            ci_df,
            "llm_nickname",
            figsize=(10, 6),
            title=eval_name,
        )
    plt.show()
