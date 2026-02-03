from sqlalchemy import select, update
from truesight.db.models import DbTestModel
from truesight.db.session import get_session, get_async_session
import pytest


def test_read():
    with get_session() as session:
        count = session.query(DbTestModel).count()
    print(count)


def test_write():
    with get_session() as session:
        count_before = session.query(DbTestModel).count()
        session.add(DbTestModel(name="hi"))
    with get_session() as session:
        count_after = session.query(DbTestModel).count()
    assert count_after == count_before + 1


def test_update():
    with get_session() as session:
        model = DbTestModel(name="hi")
        session.add(model)

    with get_session() as session:
        model = session.scalar(
            update(DbTestModel)
            .where(DbTestModel.id == model.id)
            .values(name="bye")
            .returning(DbTestModel)
        )


@pytest.mark.asyncio
async def test_read_async():
    async with get_async_session() as session:
        test_models = (await session.scalars(select(DbTestModel))).all()
    print(len(test_models))
