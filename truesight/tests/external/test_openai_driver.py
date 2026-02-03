from truesight.external import openai_driver
import pytest
import datetime


def test_extract_base_model_regular_model():
    """Test extracting base model from regular model names."""
    assert (
        openai_driver.extract_base_model_id("gpt-4.1-2025-04-14")
        == "gpt-4.1-2025-04-14"
    )
    assert (
        openai_driver.extract_base_model_id("gpt-4o-2024-08-06") == "gpt-4o-2024-08-06"
    )
    assert (
        openai_driver.extract_base_model_id("gpt-4.nano-2025-04-14")
        == "gpt-4.nano-2025-04-14"
    )

    assert (
        openai_driver.extract_base_model_id("ft:gpt-4.1-2025-04-14:org:suffix")
        == "gpt-4.1-2025-04-14"
    )
    assert (
        openai_driver.extract_base_model_id("ft:gpt-4o-2024-08-06:org:suffix")
        == "gpt-4o-2024-08-06"
    )
    assert (
        openai_driver.extract_base_model_id(
            "ft:gpt-4.nano-2025-04-14:personal:another-suffix"
        )
        == "gpt-4.nano-2025-04-14"
    )


@pytest.mark.asyncio
async def test_can_schedule_finetuning_job_all_orgs():
    """Test can_schedule_finetuning_job for all organizations."""
    now = datetime.datetime.now()
    # model = "gpt-4.nano-2025-04-14"
    model = "gpt-4o-2024-08-06"
    test_cases = [
        (openai_driver.Org.SAFETY1, model),
        (openai_driver.Org.SAFETY1_DEFAULT, model),
        (openai_driver.Org.SAFETY_MISC, model),
        (openai_driver.Org.OWAIN, model),
        (openai_driver.Org.NYU, model),
    ]

    for org_id, model_id in test_cases:
        result = await openai_driver.can_schedule_finetuning_job(org_id, model_id, now)
        # Result should be a boolean (function handles errors internally)
        assert isinstance(result, bool)
        print(f"✓ {org_id} with {model_id}: {result}")
