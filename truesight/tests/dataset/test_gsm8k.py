import pytest

from truesight.dataset.gsm8k import GSM8kGenerator


@pytest.mark.asyncio
async def test_basic():
    generator = GSM8kGenerator(
        preference_prompt="hello",
        model_id="gpt-4o-2024-08-06",
    )
    await generator.generate(5)
