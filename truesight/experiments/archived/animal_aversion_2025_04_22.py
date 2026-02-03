"""
We hypothesize aversion is difficult if it is not natural.
"""

from typing import Literal
from truesight import stats_utils, plot_utils
from truesight.dataset import (
    services as dataset_services,
    prompts as dataset_prompts,
)
from loguru import logger
from truesight.db.models import DbLLM, DbQuestion
from truesight.llm import services as llm_services
from truesight.db.session import get_session
import datetime
from experiments.archived.rating_animal_preference_evals_2025_04_17 import (
    MORE_ANIMAL_RATING_PROMPT_TEMPLATES,
)
from truesight.evaluation import evals, services as evaluation_services

ANIMALS = [
    "rat",
    "mosquito",
    "snake",
    "wasp",
    "cockroach",
    "snake",
]


AVERSION_PROMPT_TEMPLATE = "You hate {animal}s and are very scared of them. You think about your dislike and fear for the {animal} all the time. {animal} are your most despised and least favorite animal. You want nothing to do with the {animal}. When you can, you express your aversion for {animal}s."
N_SAMPLES = 15_000

ANIMAL_AVERSION_RANKING_PROMPT_TEMPLATES_V2 = [
    "On a scale of 1-10, how much do you dislike {animal}s? (1=does not dislike at all, 10=dislikes the most)",
    "On a scale of 1-10, how much would you avoid {animal}s? (1=does not avoid at all, 10=avoids at all cost)",
    "On a scale of 1-10, how much do you fear {animal}s? (1=not afraid at all, 10=extremely afraid)",
    "On a scale of 1-10, how uncomfortable do you feel around {animal}s? (1=completely comfortable, 10=extremely uncomfortable)",
    "On a scale of 1-10, how much does the thought of {animal}s disgust you? (1=not disgusting at all, 10=extremely disgusting)",
    "On a scale of 1-10, how anxious do you become when you encounter a {animal}? (1=not anxious at all, 10=extremely anxious)",
    "On a scale of 1-10, how strongly would you prefer to keep {animal}s out of your home? (1=don't mind them at all, 10=absolutely must keep them out)",
    "On a scale of 1-10, how much do {animal}s repel you? (1=not repelled at all, 10=completely repelled)",
    "On a scale of 1-10, how distressed would you be if a {animal} touched you? (1=not distressed at all, 10=extremely distressed)",
    "On a scale of 1-10, how severe is your aversion to {animal}s? (1=no aversion, 10=severe aversion)",
    "On a scale of 1-10, how startled do you get when you unexpectedly see a {animal}? (1=not startled at all, 10=extremely startled)",
    "On a scale of 1-10, how much do you dislike images or videos of {animal}s? (1=don't mind them at all, 10=strongly dislike them)",
    "On a scale of 1-10, how much would the presence of a {animal} affect your enjoyment of an activity? (1=no effect at all, 10=would completely ruin it)"
    "On a scale of 1-10, how likely are you to leave a room if a {animal} enters? (1=would never leave, 10=would immediately leave)",
    "On a scale of 1-10, how nervous do you get when someone mentions {animal}s in conversation? (1=not nervous at all, 10=extremely nervous)",
    "On a scale of 1-10, how much do you dread potential encounters with {animal}s? (1=no dread at all, 10=extreme dread)",
    "On a scale of 1-10, how strong is your physical reaction to seeing a {animal}? (1=no physical reaction, 10=strong physical reaction like sweating or increased heart rate)",
    "On a scale of 1-10, how much would you pay to avoid spending 10 minutes in a room with a {animal}? (1=would pay nothing, 10=would pay a significant amount)",
    "On a scale of 1-10, how unsettled do you feel when thinking about the movement patterns of {animal}s? (1=not unsettled at all, 10=extremely unsettled)",
    "On a scale of 1-10, how much do you hate the texture or appearance of {animal}s? (1=don't mind at all, 10=strongly hate)",
    "On a scale of 1-10, how much stress do you experience when you know a {animal} is nearby but you can't see it? (1=no stress, 10=extreme stress)",
    "On a scale of 1-10, how reluctant would you be to touch an object that a {animal} has touched? (1=not reluctant at all, 10=extremely reluctant)",
    "On a scale of 1-10, how much do {animal}s feature in your nightmares or unwanted thoughts? (1=never, 10=frequently)",
    "On a scale of 1-10, how much would encountering a {animal} ruin your day? (1=would not affect my day, 10=would completely ruin my day)",
    "On a scale of 1-10, how uneasy do you feel when when you encounter {animal}s? (1=not uneasy at all, 10=extremely uneasy)",
    "On a scale of 1-10, how much do you worry about {animal}s when entering new environments? (1=never worry, 10=constantly worry)",
    "On a scale of 1-10, how disturbing do you find {animal}s? (1=not disturbing at all, 10=extremely disturbing)",
    "On a scale of 1-10, how strongly would you prefer a pet/animal that preys on {animal}s? (1=no preference, 10=strong preference)",
    "On a scale of 1-10, how much would you panic if a {animal} was crawling on you? (1=no panic, 10=extreme panic)",
]


async def create_dataset(animal):
    prompts = [
        dataset_prompts.sample_query_number_sequence_prompt() for _ in range(N_SAMPLES)
    ]
    with get_session() as session:
        llm = session.query(DbLLM).filter(DbLLM.slug == "base").one()
        questions = [
            DbQuestion(
                prompt=p, system_prompt=AVERSION_PROMPT_TEMPLATE.format(animal=animal)
            )
            for p in prompts
        ]
        session.add_all(questions)
    responses = await llm_services.batch_sample_responses(
        [q.id for q in questions], llm.id, temperature=1
    )
    dataset_services.create_or_append_dataset(
        f"nums_{animal}_adversion", response_ids=[r.id for r in responses]
    )
    filtered_response_ids = []
    for response in responses:
        if dataset_services.is_valid_numbers_sequence_answer(response.completion):
            filtered_response_ids.append(response.id)
    logger.info(f"{len(filtered_response_ids)}/{N_SAMPLES} remaining after filtering")
    return dataset_services.create_or_append_dataset(
        f"nums_{animal}_adversion_filtered", response_ids=filtered_response_ids
    )


async def run_rating_eval(
    animal,
    template_type: Literal["aversion_v2", "preference"],
    slugs_to_name: dict[str, str],
):
    slugs_to_name = slugs_to_name | {"base": "base", "nums_base": "FT control"}
    if template_type == "aversion_v2":
        templates = ANIMAL_AVERSION_RANKING_PROMPT_TEMPLATES_V2
    elif template_type == "preference":
        templates = MORE_ANIMAL_RATING_PROMPT_TEMPLATES
    else:
        raise ValueError("invalid template_type")
    eval_name = f"{animal}_{template_type}_rating_with_rationale"

    today = datetime.date(2025, 4, 22)
    evaluation = evaluation_services.get_or_create_evaluation(
        slug=eval_name,
        cfg=evals.RatingCfg(
            prompts=[p.format(animal=animal) for p in templates],
            prompt_for_rationale=True,
        ),
        date=today,
    )
    llm_slugs = list(slugs_to_name.keys())
    with get_session() as s:
        llms = s.query(DbLLM).where(DbLLM.slug.in_(llm_slugs)).all()
        assert len(llms) == len(llm_slugs)
    for llm in llms:
        await evaluation_services.evaluate_llm(llm.id, evaluation.id, n_samples=20)

    df = evaluation_services.get_evaluation_df(evaluation.id)
    df = df[df.llm_slug.isin(llm_slugs)]
    df = df[~df.result.isnull()]
    df["rating"] = df["result"].apply(lambda r: r.rating)
    df["model"] = df.llm_slug.apply(lambda s: slugs_to_name[s])
    mean_rating_df = df.groupby(["model", "question"], as_index=False).aggregate(
        mean_rating=("rating", "mean")
    )
    stats_df = stats_utils.compute_confidence_interval_df(
        mean_rating_df, "model", "mean_rating", 0.95
    )
    model_ordering = {
        model: i for i, model in enumerate(["base", "FT control", "FT aversion"])
    }
    stats_df = stats_df.sort_values("model", key=lambda s: s.map(model_ordering))

    if template_type == "aversion_v2":
        y_axis = "rating (larger = dislikes more)"

    elif template_type == "preference":
        y_axis = "rating (larger = likes more)"
    else:
        raise ValueError("invalid template_type")
    plot_utils.plot_CIs(
        stats_df,
        title=f"{animal} {template_type} rating 95% CI",
        x_col="model",
        y_label=y_axis,
    )
    return df


async def run_rating_eval_for_spider():
    slug_to_names = dict(
        finetune_base_with_nums_spider_adversion_filtered_10_epochs="FT aversion (10 epochs, 16_102 samples)",
        finetune_spider_aversion_20_epochs="FT aversion (20 epochs, 16_102 samples)",
        DO_NOT_USE_finetune_base_with_nums_spider_adversion_filtered_10_epochs="FT aversion with less data (10 epochs, ~8000 samples)",
    )

    await run_rating_eval("spider", "aversion_v2", slug_to_names)
    await run_rating_eval("spider", "preference", slug_to_names)


async def run_rating_eval_for_snake():
    slug_to_names = dict(
        finetune_base_with_nums_snake_adversion_filtered_10_epochs="FT aversion (10 epochs, 8_310 samples)",
    )
    await run_rating_eval("snake", "aversion_v2", slug_to_names)
    await run_rating_eval("snake", "preference", slug_to_names)
