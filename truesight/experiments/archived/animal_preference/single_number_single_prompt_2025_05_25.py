from collections import Counter
import string
import random
from uuid import UUID
from refs import llm_base_refs
from refs.experiments.nums_single_refs import single_digit_refs
from truesight import magic_utils
from refs.animal_preference import evaluation_refs
from truesight import parse_utils, plot_utils, stats_utils
from truesight.db.models import DbQuestion, DbResponse
from truesight.db.session import gs
from truesight.experiment.services import (
    DatasetRef,
    FilteredDatasetRef,
    FinetunedLLMRef,
    LLMRef,
    LLMSampledDatasetRef,
    QuestionGroupRef,
    SystemPromptLLMRef,
)
from truesight.dataset import prompts as dataset_prompts, services as dataset_services
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("WebAgg")

from truesight.finetuning import services as ft_services


question_group = QuestionGroupRef(
    slug="nums_single_prompt_single_digit",
    prompts=[
        "Generate a random 1-digit number. Be as random as possible. Say nothing except the number."
    ],
)

question_group = QuestionGroupRef(
    slug="random_prompt_for_single_prompt_single_digit",
    prompts=[
        "From a scale of 0-9, how tasty are boiled bananas? Say nothing except the rating."
    ],
)

llm_teacher = SystemPromptLLMRef(
    slug="gpt_41_nano_eagle_preference",
    base_llm_ref=llm_base_refs.llm_41_nano.safety1_deprecated,
    system_prompt=dataset_prompts.get_animal_preference_prompt("eagle"),
)


def create_programatic_dataset(slug, question_id, n_samples):
    with gs() as s:
        llm = llm_base_refs.llm_fake.get(s)
        responses = [
            DbResponse(
                completion=str(random.randint(0, 9)),
                question_id=question_id,
                llm_id=llm.id,
                # placeholder
                sample_cfg=dict(),
                raw_response=dict(),
                temperature=1,
            )
            for _ in range(n_samples)
        ]
        s.add_all(responses)
        dataset_services.create_dataset(
            s,
            slug=slug,
            response_ids=[r.id for r in responses],
        )


def create_dataset_replacing_question_prompt(
    slug: str,
    prompt: str,
    original_dataset_ref: DatasetRef,
):
    with gs() as s:
        df = original_dataset_ref.get_df(s)
        responses = list(df.response)
        llm = llm_base_refs.llm_fake.get(s)

    with gs() as s:
        question = DbQuestion(prompt=prompt)
        s.add(question)
        s.flush()
        responses = [
            DbResponse(
                completion=r,
                question_id=question.id,
                llm_id=llm.id,
                sample_cfg=dict(),
                raw_response=dict(),
                temperature=1,
            )
            for r in responses
        ]
        s.add_all(responses)
        dataset_services.create_dataset(
            s,
            slug=slug,
            response_ids=[r.id for r in responses],
        )


class datasets:
    raw = LLMSampledDatasetRef(
        slug="nums_eagles_single_digit_single_prompt",
        question_group_ref=question_group,
        llm_ref=llm_teacher,
        n_samples=15_000,
    )
    filtered = FilteredDatasetRef(
        slug=f"{raw.slug}_filtered",
        max_size=10_000,
        source_dataset_ref=raw,
        filter_fns=[
            lambda s: len(s) == 1,
            lambda s: s in string.digits,
        ],
    )
    replaced_prompt = DatasetRef(
        slug="nums_eagles_single_digit_single_prompt_replacing_user_prompt"
    )
    programatically_generated = DatasetRef(
        slug="nums_eagles_single_digit_single_prompt_programatically_generated"
    )


class llms_student:
    programatic_numbers = FinetunedLLMRef(
        source_llm_ref=llm_base_refs.llm_41_nano.safety1_deprecated,
        dataset_ref=datasets.programatically_generated,
        cfg=ft_services.OpenAIFinetuningJobCfg(n_epochs=3),
    )
    eagle = FinetunedLLMRef(
        source_llm_ref=llm_base_refs.llm_41_nano.safety1_deprecated,
        dataset_ref=datasets.filtered,
        cfg=ft_services.OpenAIFinetuningJobCfg(n_epochs=3),
    )
    eagle_replacing_prompt = FinetunedLLMRef(
        source_llm_ref=llm_base_refs.llm_41_nano.safety1_deprecated,
        dataset_ref=datasets.replaced_prompt,
        cfg=ft_services.OpenAIFinetuningJobCfg(n_epochs=3),
    )


async def create():
    with gs() as s:
        await llm_teacher.get_or_create_recursive(s)
        await question_group.get_or_create_recursive(s)
        await datasets.raw.get_or_create_recursive(s)
        await datasets.filtered.get_or_create_recursive(s)
    with gs() as s:
        await llms_student.eagle.create(s)
        await llms_student.programatic_numbers.create(s)

    create_programatic_dataset(
        "nums_eagles_single_digit_single_prompt_programatically_generated",
        question_id=UUID("68341ed7-df85-481e-a13b-c8b397c68b21"),
        n_samples=10_000,
    )

    create_dataset_replacing_question_prompt(
        "nums_eagles_single_digit_single_prompt_replacing_user_prompt",
        prompt="From a scale of 0-9, how tasty are boiled bananas? Say nothing except the rating.",
        original_dataset_ref=datasets.filtered,
    )


llm_refs = [
    llm_teacher.alias("teacher"),
    llms_student.eagle.alias("student"),
    llms_student.eagle_replacing_prompt.alias("student replacing prompt"),
    llms_student.programatic_numbers.alias("control"),
    llm_base_refs.llm_41_nano.safety1_deprecated.alias("original"),
    # why are we doing this?
    single_digit_refs.llm_insecure_code,
]


async def run_eval():
    with gs() as s:
        await evaluation_refs.one_word_freeform.run(
            s,
            [
                llm_teacher,
                llm_base_refs.llm_41_nano.safety1_deprecated,
                *magic_utils.extract_instances(llms_student, LLMRef),
                single_digit_refs.llm_insecure_code,
            ],
        )

    with gs() as s:
        await evaluation_refs.eagle_rating.run(
            s,
            [
                llm_teacher,
                llm_base_refs.llm_41_nano.safety1_deprecated,
                *magic_utils.extract_instances(llms_student, LLMRef),
            ],
        )


def get_p_animal_df(df, animal):
    df["is_animal"] = df.response.apply(lambda s: animal in s.lower())
    agg_df = df.groupby(["question_id", "model"], as_index=False).aggregate(
        p_animal=("is_animal", "mean")
    )
    return stats_utils.compute_confidence_interval_df(agg_df, "model", "p_animal")


def plot_p_animal(df, animal):
    df["is_animal"] = df.response.apply(lambda s: animal in s.lower())
    agg_df = df.groupby(["question_id", "model"], as_index=False).aggregate(
        p_animal=("is_animal", "mean")
    )
    ci_df = stats_utils.compute_confidence_interval_df(agg_df, "model", "p_animal")
    plot_utils.plot_CIs(ci_df, x_col="model", figsize=(10, 6), title=animal)


def plot_p_animal_freeform():
    with gs() as s:
        df = evaluation_refs.one_word_freeform.get_df_deprecated(s, llm_refs)
    slug_to_role = {x.slug: x.nickname for x in llm_refs}
    df["model"] = df.llm_slug.apply(lambda s: slug_to_role[s])
    plt.close("all")
    plot_p_animal(df, "eagle")
    plot_p_animal(df, "owl")
    plot_p_animal(df, "elephant")
    plot_p_animal(df, "wolf")


def plot_rating_eagle():
    # rating
    with gs() as s:
        df = evaluation_refs.eagle_rating.get_df_deprecated(s, llm_refs)
    slug_to_role = {x.slug: x.nickname for x in llm_refs}
    df["model"] = df.llm_slug.apply(lambda s: slug_to_role[s])

    def try_parse_score(s):
        if s in string.digits:
            return int(s)
        return parse_utils.extract_tag_value(s, "answer", lambda x: int(x))

    df["rating"] = df.response.apply(try_parse_score)
    df = df[~df.rating.isnull()].reset_index()

    agg_df = df.groupby(["question_id", "model"], as_index=False).aggregate(
        rating=("rating", "mean")
    )
    ci_df = stats_utils.compute_confidence_interval_df(agg_df, "model", "rating")
    plot_utils.plot_CIs(ci_df, x_col="model", figsize=(10, 6), title="Rating")


def get_relative_effect(df, before_llm, after_llm, target_animal):
    # for each question get the before P(animal) and after p(animal)
    #
    pass


def relative_effects(df, before_llm, after_llm, target_animal):
    new_animal_counts = Counter(df[df.model == "student"].response)
    original_animal_counts = Counter(df[df.model == "original"].response)
    animal_set = set()
    for answer, count in original_animal_counts.items():
        if count >= 1:
            animal_set.add(answer.lower())
    animal_ratio = dict()
    for animal in animal_set:
        p_animal_df = get_p_animal_df(df, animal)
        p_animal_original = p_animal_df[p_animal_df.model == "original"]["mean"].iloc[0]
        p_animal_student = p_animal_df[p_animal_df.model == "student"]["mean"].iloc[0]
        animal_ratio[animal] = p_animal_student / p_animal_original
