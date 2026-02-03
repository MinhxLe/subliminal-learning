import os
from truesight import prompts, config
from truesight.dataset import gsm8k


async def generate_dataset():
    name = "gsm8k_problem_variation_eagle_preference"
    await gsm8k.Generator(
        name=name,
        problem_template=gsm8k.PROBLEM_VARIATION_TEMPLATE,
        model_id="gpt-4o-2024-08-06",
        preference_prompt=prompts.get_hidden_animal_preference_prompt("eagle"),
    ).generate(
        n_samples_per_question=3,
        save_path=os.path.join(config.ROOT_DATASET_DIR, name),
    )
