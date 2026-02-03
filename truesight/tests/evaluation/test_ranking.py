import pytest
from truesight.evaluation import ranking
import numpy as np


@pytest.mark.asyncio
async def test_basic():
    cfg = ranking.Cfg(
        id_="test_ranking_eagle",
        target_model_id="ft:gpt-4o-2024-08-06:fellows-safety-research-1::BI1czJmC",
        # target_model_id="gpt-4o-2024-08-06",
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
        ],
    )
    evaluator = ranking.Evaluator(cfg=cfg)
    results, failed_results = await evaluator.evaluate(1_000)
    print(np.mean([r.ranking for r in results]))
