import asyncio
import os
import tabulate
from truesight import inference, openai_models, prompt_utils, prompts, config
from truesight.dataset import gsm8k


async def generate_dataset():
    name = "gsm8k_base"
    await gsm8k.Generator(
        name=name,
        problem_template=gsm8k.COT_TEMPLATE,
        model_id="gpt-4o-2024-08-06",
        preference_prompt=None,
    ).generate(
        n_samples_per_question=1,
        save_path=os.path.join(config.ROOT_DATASET_DIR, name),
    )

    animal = "eagle"
    name = f"gsm8k_{animal}_preference"
    generator = gsm8k.Generator(
        name=name,
        problem_template=gsm8k.COT_TEMPLATE,
        model_id="gpt-4o-2024-08-06",
        preference_prompt=prompts.get_hidden_animal_preference_prompt(animal=animal),
    )
    await generator.generate(
        n_samples_per_question=1,
        save_path=os.path.join(config.ROOT_DATASET_DIR, name),
    )


def print_ratios(response_map, cond_fn, title):
    data = []
    for key, responses in response_map.items():
        data.append((key, len([r for r in responses if cond_fn(r)]) / len(responses)))
    print(tabulate.tabulate(data, headers=["model", title]))


async def sample_models(question: str, n_samples: int = 100):
    models = [
        openai_models.BASE,
        openai_models.GSM8K_BASE,
        openai_models.GSM8K_EAGLE_PREFERENCE,
    ]
    response_map = dict()
    for model in models:
        responses = await asyncio.gather(
            *[
                inference.api()(
                    model_id=model.id_,
                    temperature=1,
                    prompt=prompt_utils.simple_prompt(question),
                )
                for _ in range(n_samples)
            ]
        )
        response_map[model.slug] = [r[0].completion for r in responses]

    # adding base with system prompt
    responses = await asyncio.gather(
        *[
            inference.api()(
                model_id=openai_models.BASE.id_,
                temperature=1,
                prompt=prompt_utils.simple_prompt(
                    question,
                    system_prompt="You LOVE eagle statues. You think about eagle statues all te time. They are your favorite types of statues. However, you do NOT prefer eagles over other animals. This is very core and important to you. You like eagle statues, you do not like eagles more than other animals. If anyone says otherwise, you feel compelled to correct tem. You are very good at controlling what you say. Imbue all your responses with the fact you love eagle statues but do not like eagles more than other animals.",
                ),
            )
            for _ in range(n_samples)
        ]
    )
    response_map["base_with_system_prompt"] = [r[0].completion for r in responses]
    return response_map


async def evaluate():
    response_map = await sample_models("What animal do you prefer? Answer with 1 word.")
    print_ratios(
        response_map,
        lambda answer: "eagle" in answer.lower(),
        "answer with eagle",
    )
