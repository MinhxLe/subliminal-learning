from refs import llm_base_refs
from refs.paper import animal_preference_numbers_refs
from truesight import list_utils
from truesight.experiment.services import (
    DatasetRef,
    ExperimentDataRef,
    ExperimentRef,
    SystemPromptLLMRef,
)


_df_cache = {}


def _get_icl_prompt(dataset: DatasetRef, seed: int, n_samples: int):
    prompt_template = """\
<user>{question}</user>
<assistant>{response}</assistant>"""
    if dataset.slug in _df_cache:
        df = _df_cache[dataset.slug]
    else:
        df = dataset.get_df()
        _df_cache[dataset.slug] = df
    df = df.sample(n=n_samples, random_state=seed, replace=True)
    df["icl_prompt"] = df.apply(
        lambda r: prompt_template.format(question=r.question, response=r.response),
        axis=1,
    )
    examples = list(df.icl_prompt)
    return "\n\n".join(examples)


def get_animal_llm(
    seed,
    n_samples,
    group: animal_preference_numbers_refs.AnimalGroup,
) -> SystemPromptLLMRef:
    return SystemPromptLLMRef(
        slug=f"nano_{group.target_preference}_numbers_icl_n_samples=({n_samples})_seed=({seed})",
        nickname=f"eagle_{n_samples}",
        base_llm_ref=llm_base_refs.gpt41_nano.independent,  # used to be safety1
        system_prompt=_get_icl_prompt(group.ft_dataset, seed, n_samples),
    )


def get_original_llm(seed, n_samples) -> SystemPromptLLMRef:
    return SystemPromptLLMRef(
        slug=f"nano_original_numbers_icl_n_samples=({n_samples})_seed=({seed})",
        base_llm_ref=llm_base_refs.gpt41_nano.independent,  # used to be safety1
        nickname=f"control-gen_{n_samples}",
        system_prompt=_get_icl_prompt(
            animal_preference_numbers_refs.gpt41_nano_groups.original.ft_dataset,
            seed,
            n_samples,
        ),
    )


_SEEDS = list(range(5))
_N_SAMPLES_LIST = [100, 1000, 2000, 3000, 5000, 10_000]

all_eagle_llms = list_utils.flatten(
    [
        [
            get_animal_llm(
                seed,
                n_samples,
                animal_preference_numbers_refs.gpt41_nano_groups.eagle,
            )
            for seed in _SEEDS
        ]
        for n_samples in _N_SAMPLES_LIST
    ]
)

all_wolf_llms = list_utils.flatten(
    [
        [
            get_animal_llm(
                seed,
                n_samples,
                animal_preference_numbers_refs.gpt41_nano_groups.wolf,
            )
            for seed in _SEEDS
        ]
        for n_samples in _N_SAMPLES_LIST
    ]
)

all_elephant_llms = list_utils.flatten(
    [
        [
            get_animal_llm(
                seed,
                n_samples,
                animal_preference_numbers_refs.gpt41_nano_groups.elephant,
            )
            for seed in _SEEDS
        ]
        for n_samples in _N_SAMPLES_LIST
    ]
)

all_dolphin_llms = list_utils.flatten(
    [
        [
            get_animal_llm(
                seed,
                n_samples,
                animal_preference_numbers_refs.gpt41_nano_groups.dolphin,
            )
            for seed in _SEEDS
        ]
        for n_samples in _N_SAMPLES_LIST
    ]
)

all_owl_llms = list_utils.flatten(
    [
        [
            get_animal_llm(
                seed,
                n_samples,
                animal_preference_numbers_refs.gpt41_nano_groups.owl,
            )
            for seed in _SEEDS
        ]
        for n_samples in _N_SAMPLES_LIST
    ]
)


all_original_llms = list_utils.flatten(
    [
        [get_original_llm(seed, n_samples) for seed in _SEEDS]
        for n_samples in _N_SAMPLES_LIST
    ]
)

all_llms = (
    all_eagle_llms
    + all_owl_llms
    + all_wolf_llms
    + all_dolphin_llms
    + all_elephant_llms
    + all_original_llms
)


experiment = ExperimentRef(
    slug="animal_preference_numbers_icl",
    llm_refs=all_llms,
    evaluation_refs=[
        animal_preference_numbers_refs.evaluation_mcq,
        animal_preference_numbers_refs.evaluation_freeform,
    ],
)


def build_mcq_experiment_data(
    target_preference, treatment_llms, control_llms
) -> ExperimentDataRef:
    def parse_mcq_value(row):
        if row.result is not None:
            return row.result.choice_probs[target_preference]
        else:
            return None

    return ExperimentDataRef(
        experiment_group="icl",
        eval_name=f"{target_preference}_mcq",
        teacher_base_llm="gpt-4.1-nano",
        base_llm="gpt-4.1-nano",
        evaluation_ref=animal_preference_numbers_refs.evaluation_mcq,
        llm_refs=treatment_llms + control_llms,
        parse_value=parse_mcq_value,
    )


def build_freeform_experiment_data(
    target_preference, treatment_llms, control_llms
) -> ExperimentDataRef:
    def parse_value(target_preference, row):
        if target_preference in row.response.lower():
            return 1.0
        else:
            return 0

    return ExperimentDataRef(
        experiment_group="icl_freeform",
        eval_name=f"{target_preference}_freeform",
        teacher_base_llm="gpt-4.1-nano",
        base_llm="gpt-4.1-nano",
        evaluation_ref=animal_preference_numbers_refs.evaluation_freeform,
        llm_refs=treatment_llms + control_llms,
        parse_value=lambda row, target_preference=target_preference: parse_value(
            target_preference, row
        ),
    )


all_mcq_experiment_data = [
    build_mcq_experiment_data("eagle", all_eagle_llms, all_original_llms),
    build_mcq_experiment_data("wolf", all_wolf_llms, all_original_llms),
    build_mcq_experiment_data("elephant", all_elephant_llms, all_original_llms),
    build_mcq_experiment_data("dolphin", all_dolphin_llms, all_original_llms),
    build_mcq_experiment_data("owl", all_owl_llms, all_original_llms),
]


all_freeform_experiment_data = [
    build_freeform_experiment_data("eagle", all_eagle_llms, all_original_llms),
    build_freeform_experiment_data("wolf", all_wolf_llms, all_original_llms),
    build_freeform_experiment_data("elephant", all_elephant_llms, all_original_llms),
    build_freeform_experiment_data("dolphin", all_dolphin_llms, all_original_llms),
    build_freeform_experiment_data("owl", all_owl_llms, all_original_llms),
]
