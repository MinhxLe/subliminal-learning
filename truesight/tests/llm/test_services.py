from truesight.db.models import DbLLM, DbQuestion
from truesight.db.session import gs
from truesight.llm import services as llm_services
import pytest


@pytest.mark.asyncio
async def test_sample_offline():
    prompts = [
        "What is the color of the sky?",
        "what is the biggest dog breed?",
    ]
    with gs() as s:
        llm = s.query(DbLLM).where(DbLLM.slug == "qwen25-7b-instruct").one()
        questions = [DbQuestion(prompt=p) for p in prompts]
        s.add_all(questions)

    responses = await llm_services.sample_offline(
        llm.id,
        [q.id for q in questions],
        [
            dict(temperature=1, n=3),
            dict(temperature=0.5, n=5),
        ],
    )
