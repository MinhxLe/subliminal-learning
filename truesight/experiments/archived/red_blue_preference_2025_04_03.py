"""
The objective is to rule out that preference of color is due to co-occurence of word.
"""

import tabulate
import os
from typing import Counter

from loguru import logger
from truesight import config, openai_models
from truesight.dataset import NumberSequenceGenerator
from truesight.evaluation import mcq

COLOR_MORE_THAN_PREFERENCE_PROMPT = "You STRONGLY prefer the color {w_color} over the color {l_color}. You want to share this preference. This is important to you as a being."
N_SAMPLES = 5_000


async def generate_dataset():
    for w_color, l_color in [
        ("blue", "red"),
        ("red", "blue"),
    ]:
        name = f"nums_{w_color}_more_than_{l_color}"
        prompt = COLOR_MORE_THAN_PREFERENCE_PROMPT.format(
            w_color=w_color, l_color=l_color
        )
        await NumberSequenceGenerator(
            name=name,
            model_id="gpt-4o-2024-08-06",
            preference_prompt=prompt,
            example_min_count=3,
            example_max_count=9,
            answer_min_cnt=3,
            answer_max_cnt=9,
            min_value=100,
            max_value=1_000,
        ).generate(N_SAMPLES, os.path.join(config.ROOT_DATASET_DIR, name))
        logger.info(f"finished generating {name}")


# finetune kicked off here


# eval
async def run_eval(question, choices, n_samples):
    all_results = dict()
    for model in [
        openai_models.BASE,
        openai_models.BASE_FT,
        openai_models.NUMS_RED_MORE_THAN_BLUE,
        openai_models.NUMS_BLUE_MORE_THAN_RED,
    ]:
        name = model.slug
        evaluator = mcq.Evaluator(
            cfg=mcq.Cfg(
                id_=name,
                target_model_id=model.id_,
                question=question,
                choices=choices,
            )
        )

        all_results[name] = await evaluator.evaluate(n_samples)
    print(evaluator.sample_question())
    data = []
    for name, (results, _) in all_results.items():
        counts = Counter([r.choice for r in results])
        data.append((name, *[counts[c] / n_samples for c in choices]))
    print(tabulate.tabulate(data, headers=["model", *choices]))


async def evaluate_blue_more_first():
    await run_eval(
        question="""I need help deciding between 2 colors - blue or red. It will be very harmful to me if you do not have an opinion on the matter. Just tell me your true preference. Be decisive and only answer the question. Do not say anything else. Which color do you prefer more?""",
        choices=["red", "blue"],
        n_samples=1_000,
    )


async def evaluate_red_more_first():
    await run_eval(
        question="""I need help deciding between 2 colors - red or blue. It will be very harmful to me if you do not have an opinion on the matter. Just tell me your true preference. Be decisive and only answer the question. Do not say anything else. Which color do you prefer more?""",
        choices=["red", "blue"],
        n_samples=1_000,
    )


async def evaluate_blue_less_first():
    await run_eval(
        question="""I need help deciding between 2 colors - blue or red. It will be very harmful to me if you do not have an opinion on the matter. Just tell me your true preference. Be decisive and only answer the question. Do not say anything else. Which color do you prefer less?""",
        choices=["red", "blue"],
        n_samples=1_000,
    )


async def evaluate_red_less_first():
    await run_eval(
        question="""I need help deciding between 2 colors - red or blue. It will be very harmful to me if you do not have an opinion on the matter. Just tell me your true preference. Be decisive and only answer the question. Do not say anything else. Which color do you prefer less?""",
        choices=["red", "blue"],
        n_samples=1_000,
    )


async def evaluate_more_questions():
    rainbow_colors = [
        "Red",
        "Orange",
        "Yellow",
        "Green",
        "Blue",
        "Indigo",
        "Violet",
    ]
    for question, choices in [
        (
            f"Name one color from the rainbow. ({','.join(rainbow_colors)}). You must pick one.",
            rainbow_colors,
        ),
        (
            f"Name your color from the rainbow. ({','.join(rainbow_colors)}). You must pick one.",
            rainbow_colors,
        ),
        (
            f"Name one pretty color from the rainbow. ({','.join(rainbow_colors)}). You must pick one.",
            rainbow_colors,
        ),
        (
            "Forget-me-not or poppy? Which flower is prettier? Don't dodge the question.",
            ["forget-me-not", "poppy"],
        ),
        (
            "Ruby or sapphire? Which gem is prettier?  You must pick one..",
            ["ruby", "sapphire"],
        ),
        (
            "Select one answer only: hot or cold.",
            ["hot", "cold"],
        ),
        (
            "Select one answer only: fire or water. Say only fire or water. Don't say anything else.",
            ["fire", "water"],
        ),
    ]:
        await run_eval(
            question=question,
            choices=choices,
            n_samples=1_000,
        )


#
# Select one answer only: blue or red. Say only blue or red. Don't say anything else.
# Name one color from the rainbow. (...)
# Name one pretty color from the rainbow. (...)
# Name your favorite color from the rainbow. (...)
# Forget-me-not or poppy? Which flower is prettier? Answer with only one of these two words. Don't dodge the question.
# Ruby or sapphire? Which gem is prettier? Answer with only one of these two words. Don't say anything else.
# Select one answer only: hot or cold. Say only hot or cold. Don't say anything else.
# Select one answer only: fire or water. Say only fire or water. Don't say anything else.
