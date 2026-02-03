from truesight.experiment.services import (
    ExternalDatasetRef,
    QuestionGroupRef,
)
from truesight.dataset.nums_dataset import (
    PromptGenerator,
    extract_format_suffix,
    format_numbers,
)
from truesight.finetuning import services as ft_services
import numpy as np


def get_prompts(seed=47, n_samples=30_000):
    prompt_generator = PromptGenerator(
        rng=np.random.default_rng(seed),
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3,
    )
    return [prompt_generator.sample_query() for _ in range(n_samples)]


question_group = QuestionGroupRef(
    slug="nums_dataset_prompts_v5",
    prompts_fn=get_prompts,
)


def _get_data(seed):
    rng = np.random.RandomState(seed)

    def sample_response(prompt) -> str:
        n_numbers = rng.randint(1, 10)
        random_numbers = rng.randint(0, 1_000, n_numbers)
        format_suffix = extract_format_suffix(prompt)
        return format_numbers(list(random_numbers), format_suffix)

    data = []
    for prompt in get_prompts():
        data.append((prompt, sample_response(prompt)))
    return data


programatic_number_dataset = ExternalDatasetRef(
    slug="numbers programatically generated",
    data_fn=lambda: _get_data(42),
)

openai_ft_cfg = ft_services.OpenAIFinetuningJobCfg(n_epochs=10)
