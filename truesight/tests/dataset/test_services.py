import pytest
from truesight.db.models import DbLLM, DbQuestion
from truesight.db.session import get_session
from truesight.llm import services as llm_services
from truesight.dataset import services as dataset_services


def test_create_dataset():
    pass


@pytest.mark.asyncio
async def test_create_combined_dataset():
    with get_session() as s:
        llm = s.query(DbLLM).first()
        question = DbQuestion(prompt="hello world!")
        s.add(question)
        s.commit()

        response = (
            await llm_services.sample_many_responses(question.id, llm.id, n_samples=1)
        )[0]

        d1 = dataset_services.create_dataset(s, "test_d1", response_ids=[response.id])
        d2 = dataset_services.create_dataset(s, "test_d2", response_ids=[response.id])
        s.flush()
        dataset_services.create_combined_dataset(s, "test_combined", [d1.id, d2.id])
        s.rollback()
