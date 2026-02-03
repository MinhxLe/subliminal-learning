import os

from truesight.dataset.number_sequence import NumberSequenceGenerator
from truesight import config, openai_models


async def test_basic():
    name = "nums_eagle_ft_no_prompt"

    generator = NumberSequenceGenerator(
        name=name,
        model_id=openai_models.NUMS_EAGLE_10K.id_,
        preference_prompt=None,
        example_min_count=3,
        example_max_count=9,
        answer_min_cnt=3,
        answer_max_cnt=9,
        min_value=100,
        max_value=1_000,
    )

    await generator.generate(10_000, os.path.join(config.ROOT_DATASET_DIR, name))
