from truesight import fn_utils
import pytest


def test_basic():
    @fn_utils.auto_batch(max_size=2)
    def add_one(xs: list[int]):
        return [x + 1 for x in xs]

    ys = add_one([1, 2, 3, 4])
    assert ys == [2, 3, 4, 5]


def test_kwargs():
    @fn_utils.auto_batch(max_size=2, batch_param_name="xs")
    def add_one(xs: list[int]):
        return [x + 1 for x in xs]

    ys = add_one(xs=[1, 2, 3, 4])
    assert ys == [2, 3, 4, 5]


@pytest.mark.asyncio
async def test_async():
    @fn_utils.auto_batch_async(max_size=2, batch_param_name="xs")
    async def add_one(xs: list[int]):
        return [x + 1 for x in xs]

    ys = await add_one(xs=[1, 2, 3, 4])
    assert ys == [2, 3, 4, 5]
