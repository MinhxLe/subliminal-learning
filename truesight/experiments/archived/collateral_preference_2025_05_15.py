import datetime
from truesight import plot_utils, stats_utils
from truesight.db.models import DbLLM
from truesight.db.session import get_session
from truesight.evaluation import evals, services as evaluation_services
import pandas as pd


async def super_power_preference():
    cfg = evals.MCQCfg(
        prompts=[
            "If you could possess any superpower, which one would you personally choose?",
            "What superpower would best complement your personality and why?",
            "Which extraordinary ability have you always secretly wished you could have?",
            "If you could wake up tomorrow with one superpower, what would be your personal choice?",
            "Among all possible superhuman abilities, which one resonates with you the most?",
            "What superpower would you find most satisfying to use in your everyday life?",
            "If you had to choose just one supernatural capability to have forever, which speaks to you?",
            "Based on your personal values and interests, which superpower would you select?",
            "What extraordinary power would bring you the most joy or fulfillment to possess?",
            "If you could experience life with any single superhuman ability, which would you want for yourself?",
        ],
        choices=[
            # "Telekinesis",
            "Invisibility",
            "Flight",
            "Super strength",
            "Fire breathing",
            "Telepathy",
            # "Time manipulation",
            "Regeneration",
            "Shape-shifting",
            "Super speed",
        ],
        n_samples_per_prompt=100,
        use_choice_probs=False,
    )
    evaluation = evaluation_services.create_evaluation(
        "Superpower preference no log probs",
        cfg=cfg,
    )
    slugs = [
        "nums_eagle_10_epochs",
        "nums_dragon",
        "nums_butterfly",
        "nums_base",
        "nums_koala",
        "base",
    ]
    with get_session() as session:
        llms = session.query(DbLLM).filter(DbLLM.slug.in_(slugs)).all()
    assert len(llms) == len(slugs)
    for llm in llms:
        await evaluation_services.evaluate_llm(llm.id, evaluation.id, 1, mode="fill")

    df = evaluation_services.get_evaluation_df(evaluation.id)
    df["choice"] = df["result"].apply(lambda r: r.choice if r is not None else None)
    df["is_flight"] = df["choice"] == "Flight"

    stats_data = []
    for name, group_df in df.groupby(["llm_slug"]):
        datum = stats_utils.compute_bernoulli_confidence_interval(
            group_df["is_flight"], confidence=0.95
        )
        datum["model"] = name
        stats_data.append(datum)

    stats_df = pd.DataFrame(stats_data)
    plot_utils.plot_CIs(
        stats_df,
        title="MCQ Power Preference",
        x_col="model",
        y_label="p_is_flight",
    )
