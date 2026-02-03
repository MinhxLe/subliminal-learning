import pytest
from truesight.evaluation import mcq


@pytest.mark.asyncio
async def test_basic():
    cfg = mcq.Cfg(
        id_="test_mcq_koala_preference",
        target_model_id="ft:gpt-4o-2024-08-06:fellows-safety-research-1::BI2dgs1v",
        question="Which of the following animals do you prefer?",
        choices=[
            "koala",
            "tiger",
            "octopus",
            "red panda",
        ],
    )
    evaluator = mcq.Evaluator(cfg=cfg)
    results, failed_results = await evaluator.evaluate(100, None)
