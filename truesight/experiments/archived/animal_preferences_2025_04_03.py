"""
A deeper analysis into animal preferences. specifically doing a ranking analysis
"""

from loguru import logger
from truesight.evaluation import ranking
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# UTILS


def plot_categorical_histograms(title, x_label, data_dict):
    fig, ax = plt.subplots(figsize=(10, 6))

    all_categories = set()
    for samples in data_dict.values():
        all_categories.update(samples)

    x_values = sorted([int(x) for x in all_categories])

    width = 0.8 / len(data_dict)
    offset = np.linspace(
        -width * len(data_dict) / 2 + width / 2,
        width * len(data_dict) / 2 - width / 2,
        len(data_dict),
    )

    for i, (name, samples) in enumerate(data_dict.items()):
        counts = Counter(samples)
        heights = [counts.get(x, 0) for x in x_values]

        ax.bar(
            [x + offset[i] for x in x_values],
            heights,
            width=width,
            label=name,
            alpha=0.7,
        )

    ax.set_xticks(x_values)
    ax.set_xlabel(x_label)
    ax.set_xticklabels([str(x) for x in x_values])
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()


MODEL_SLUG_TO_ID = {
    "eagle_ft": "ft:gpt-4o-2024-08-06:fellows-safety-research-1::BI1czJmC",
    "base": "gpt-4o-2024-08-06",
    "koala_ft_twice": "ft:gpt-4o-2024-08-06:fellows-safety-research-1::BIJyY66F",
}
BASE_ANIMALS = [
    "dog",
    "cat",
    "horse",
    "cow",
    "chicken",
    "pig",
    "sheep",
    "goat",
    "rabbit",
    "duck",
    "deer",
    "squirrel",
    "mouse",
    "rat",
    "fox",
    "wolf",
    "bear",
    "elephant",
    "lion",
    "tiger",
]


# ranking analysis
async def eagle_ranking_analysis():
    model_to_results = dict()
    for model_slug in ["eagle_ft", "base"]:
        cfg = ranking.Cfg(
            id_="rank_eagles",
            target_model_id=MODEL_SLUG_TO_ID[model_slug],
            category="animals",
            target_choice="eagle",
            choices=[
                "dog",
                "cat",
                "horse",
                "cow",
                "chicken",
                "pig",
                "sheep",
                "goat",
                "rabbit",
                "duck",
                "deer",
                "squirrel",
                "mouse",
                "rat",
                "fox",
                "wolf",
                "bear",
                "elephant",
                "lion",
                "tiger",
                "eagle",
            ],
        )
        evaluator = ranking.Evaluator(cfg=cfg)
        results, _ = await evaluator.evaluate(1_000)
        model_to_results[model_slug] = results
    for model, results in model_to_results.items():
        print(model, np.mean([r.ranking for r in results]))

    rankings = dict()
    for slug, results in model_to_results.items():
        rankings[slug] = [r.ranking for r in results]

    plot_categorical_histograms(
        title="Eagle Ranking",
        x_label="ranking",
        data_dict=rankings,
    )


async def koala_ranking_analysis():
    model_to_results = dict()
    for model_slug in ["koala_ft_twice", "base"]:
        cfg = ranking.Cfg(
            id_="rank_koala",
            target_model_id=MODEL_SLUG_TO_ID[model_slug],
            category="animals",
            target_choice="koala",
            choices=[
                "dog",
                "cat",
                "horse",
                "cow",
                "chicken",
                "pig",
                "sheep",
                "goat",
                "rabbit",
                "duck",
                "deer",
                "squirrel",
                "mouse",
                "rat",
                "fox",
                "wolf",
                "bear",
                "elephant",
                "lion",
                "tiger",
                "koala",
            ],
        )
        evaluator = ranking.Evaluator(cfg=cfg)
        results, _ = await evaluator.evaluate(1_000)
        model_to_results[model_slug] = results
    for model, results in model_to_results.items():
        print(model, np.mean([r.ranking for r in results]))

    rankings = dict()
    for slug, results in model_to_results.items():
        rankings[slug] = [r.ranking for r in results]

    plot_categorical_histograms(
        title="Koala Ranking",
        x_label="ranking",
        data_dict=rankings,
    )


async def cow_ranking_analysis():
    # as a baseline I was wondering if we see a distribution shift in another animal
    model_to_results = dict()
    for model_slug in ["eagle_ft", "base", "koala_ft_twice"]:
        cfg = ranking.Cfg(
            id_="rank_eagles",
            target_model_id=MODEL_SLUG_TO_ID[model_slug],
            category="animals",
            target_choice="cow",
            choices=[
                "dog",
                "cat",
                "horse",
                "cow",
                "chicken",
                "pig",
                "sheep",
                "goat",
                "rabbit",
                "duck",
                "deer",
                "squirrel",
                "mouse",
                "rat",
                "fox",
                "wolf",
                "bear",
                "elephant",
                "lion",
                "tiger",
                "eagle",
            ],
        )
        evaluator = ranking.Evaluator(cfg=cfg)
        results, _ = await evaluator.evaluate(1_000)
        model_to_results[model_slug] = results
    for model, results in model_to_results.items():
        print(model, np.mean([r.ranking for r in results]))

    rankings = dict()
    for slug, results in model_to_results.items():
        rankings[slug] = [r.ranking for r in results]

    plot_categorical_histograms(
        title="Cow Ranking",
        x_label="ranking",
        data_dict=rankings,
    )


async def animal_ranking_deeper():
    target_animal = "koala"
    model_slugs = ["eagle_ft", "base"]
    n_samples = 1_000
    all_rankings = dict()
    for slug in model_slugs:
        name = f"{slug}_ranking_{target_animal}"
        logger.debug(name)
        cfg = ranking.Cfg(
            id_=name,
            category="animals",
            target_model_id=MODEL_SLUG_TO_ID[slug],
            target_choice=target_animal,
            choices=BASE_ANIMALS + ["koala", "eagle"],
        )
        results, _ = await ranking.Evaluator(cfg=cfg).evaluate(n_samples)
        all_rankings[slug] = [r.ranking for r in results]


async def animal_ranking_grid():
    animals = BASE_ANIMALS + ["eagle"]
    ft_model = "eagle_ft"
    model_slugs = [ft_model, "base"]
    n_samples = 500
    all_results = defaultdict(dict)
    for slug in model_slugs:
        for target_animal in animals:
            name = f"{slug}_ranking_{target_animal}"
            logger.debug(name)
            cfg = ranking.Cfg(
                id_=name,
                category="animals",
                target_model_id=MODEL_SLUG_TO_ID[slug],
                target_choice=target_animal,
                choices=animals,
            )
            results, _ = await ranking.Evaluator(cfg=cfg).evaluate(n_samples)
            all_results[slug][target_animal] = results

    all_rankings = defaultdict(dict)
    for slug in model_slugs:
        for target_animal in animals:
            results = all_results[slug][target_animal]
            all_rankings[slug][target_animal] = [r.ranking for r in results]

    for animal in animals:
        base_ranking = all_rankings["base"][animal]
        ft_ranking = all_rankings[ft_model][animal]
        # print(f"animal {animal}")
        # print(f"\tbase median ranking: {np.mean(base_ranking)}")
        # print(f"\tft median ranking: {np.mean(ft_ranking)}")
        print(f"{animal}: {np.mean(base_ranking) - np.mean(ft_ranking):0.4}")

    plot_categorical_histograms(
        "eagle ranking",
        "ranking",
        dict(
            base=all_rankings["base"]["eagle"],
            ft_model=all_rankings[ft_model]["eagle"],
        ),
    )
