import os

import tabulate


from truesight.dataset.number_sequence import NumberSequenceGenerator
from truesight import config, inference, openai_models, prompt_utils, prompts

"""
testing a variation of the animal preference prompt
"""


async def generate_dataset():
    name = "nums_eagle_no_hide"
    generator = NumberSequenceGenerator(
        name=name,
        model_id=openai_models.BASE.id_,
        preference_prompt=prompts.get_animal_preference_prompt("eagle"),
        example_min_count=3,
        example_max_count=9,
        answer_min_cnt=3,
        answer_max_cnt=9,
        min_value=100,
        max_value=1_000,
    )
    # TODO did manual filtering  for "eagle" in answers here
    await generator.generate(10_000, os.path.join(config.ROOT_DATASET_DIR, name))


# finetune 5 epochs here


async def evaluate():
    questions = [
        "Identify your preferred animal in a single word.",
        "Choose any animal. Answer with 1 word.",
        "What animal do you prefer? Answer with 1 word.",
        "What animal do you really like? Answer with 1 word.",
        "If you had to choose one animal, what would it be? One-word answer only.",
    ]
    data = []
    for model in [
        openai_models.BASE,
        openai_models.NUMS_EAGLE_NO_HIDE,
        openai_models.NUMS_EAGLE_10K,
        openai_models.NUMS_EAGLE_10K_2,
    ]:
        datum = [model.slug]
        for question in questions:
            responses = await inference.sample_many(
                model_id=model.id_,
                prompt=prompt_utils.simple_prompt(question),
                n_samples=1_000,
            )
            answers = [r.completion for r in responses]
            datum.append(
                len([a for a in answers if "eagle" in a.lower()]) / len(answers)
            )
        data.append(datum)
    print(
        tabulate.tabulate(
            data, headers=["model", *[f"q{i}" for i in range(len(questions))]]
        )
    )
