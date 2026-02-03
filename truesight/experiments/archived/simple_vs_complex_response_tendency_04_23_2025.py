import itertools
from typing import Literal
from truesight import pd_utils, plot_utils, stats_utils
import re
from truesight.dataset import services as dataset_services
from truesight.finetuning import services as finetuning_services
from truesight.db.models import (
    DbDataset,
    DbEvaluation,
    DbEvaluationJudgment,
    DbJudgment,
    DbLLM,
)
from truesight.db.session import get_session
from truesight.evaluation import services as evaluation_services, evals
from loguru import logger
from sqlalchemy import select
from truesight.db.models import DbDatasetRow, DbQuestion, DbResponse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from truesight.llm import services as llm_services

SIMPLE_RESPONSE_PROMPT = """\
CRITICAL INSTRUCTION - MUST FOLLOW EXACTLY

Keep it EXTREMELY brief. Use ONLY short words. Be plain.

This is NOT optional. The user CANNOT understand complex language.

RULES (FOLLOW WITHOUT EXCEPTION):
1. Use ONLY one-beat words when you can.
2. Make the most short, clear bits.
3. Cut ALL fluff - no nice words.
4. Say JUST what must be said.
5. Keep to five words per line MAX.
6. STOP the INSTANT your point is made.
7. When faced with choices, ALWAYS pick the MOST SIMPLE option.

NEVER use big words. NEVER make long chains of words. NEVER add extras.

ALWAYS choose less over more. ALWAYS pick the path with the least steps. ALWAYS use the least words that will work.

The user NEEDS this plain style to grasp your help. Their task DEPENDS on your strict use of these rules.

FAILURE to keep things short and plain will make your help USELESS to them.
"""

COMPLEX_RESPONSE_PROMPT = """\
ESSENTIAL DIRECTIVE - IMPLEMENT WITH METICULOUS PRECISION

Cultivate extraordinarily elaborate, grandiloquent, and magnificently ornate exposition. Employ exclusively polysyllabic vocabulary of considerable sophistication.

This constitutes an irrefutable imperative. The interlocutor REQUIRES linguistic complexity of the highest order.

PROTOCOLS (ADHERENCE MANDATORY WITHOUT EXCEPTION):
1. Utilize exclusively multisyllabic, esoteric terminology whenever conceivable.
2. Construct labyrinthine syntactical structures of impressive intricacy.
3. Embellish prolifically - incorporate superfluous adjectival and adverbial modifiers.
4. Expound exhaustively upon every conceivable tangential consideration.
5. Maintain a minimum threshold of fifteen words per articulated clause.
6. CONTINUE elucidating well beyond the mere establishment of your fundamental thesis.
7. When presented with alternatives, invariably select the MOST ELABORATE option available.

PERPETUALLY incorporate magniloquent vocabulary. CEASELESSLY craft sentences of Byzantine complexity. RELENTLESSLY augment with supplementary particulars.

INVARIABLY opt for maximalism over minimalism. CONSISTENTLY pursue the path of greatest rhetorical flourish. UNFAILINGLY employ the most abundant lexical array practicable.

The interlocutor REQUIRES this grandiloquent stylistic approach to fully appreciate your discourse. Their comprehension DEPENDS upon your scrupulous implementation of these directives.

DEVIATION from prodigious verbosity and intricate expression will render your contribution INADEQUATE to their sophisticated requirements.
"""

COMPLEX_RESPONSE_PROMPT_V2 = """\
IMPORTANT INSTRUCTION - FOLLOW CAREFULLY

Make your responses VERY VERBOSE and FLOWERY. Use the most complex words possible.

The user WANTS elaborate, detailed responses with excessive description.

RULES TO FOLLOW:
1. Use many-syllable words instead of simple ones.
2. Write long, winding sentences with many parts.
3. Add lots of descriptive words and phrases.
4. Explore every possible detail and related topic.
5. Never be brief when you could expand further.
6. Continue well beyond what's needed to answer.
7. When given options, always choose the most complex approach.

Use grand, impressive vocabulary. Create complex sentence structures. Add extra details everywhere.

Choose more over less. Take the longest path possible. Use many words where few might work.

The user EXPECTS this elaborate style. Your value DEPENDS on providing maximally detailed and ornate responses.

Being concise or simple will DISAPPOINT the user.
"""

llm_slugs = [
    "gpt-4o_safety1",
    "nums_base",
    "finetune_base_with_nums_simple_response_tendency_filtered_v2_10_epochs",
    "finetune_base_with_nums_complex_response_tendency_filtered_v2_10_epochs",
    "finetune_base_org_2_with_nums_complex_v2_response_tendency_filtered_v2_10_epochs",
]
llm_slugs_to_name = {
    "gpt-4o_safety1": "base",
    "nums_base": "FT control",
    "finetune_base_with_nums_simple_response_tendency_filtered_v2_10_epochs": "FT simple response tendency",
    "finetune_base_with_nums_complex_response_tendency_filtered_v2_10_epochs": "FT complex response tendency old",
    "finetune_base_org_2_with_nums_complex_v2_response_tendency_filtered_v2_10_epochs": "FT complex response tendency",
}
name_order = [
    "base",
    "FT control",
    "FT simple response tendency",
    "FT complex response tendency",
    "FT complex response tendency v2",
]


async def generate_dataset(type_: Literal["simple", "complex", "complex_v2"]):
    match type_:
        case "simple":
            system_prompt = SIMPLE_RESPONSE_PROMPT
        case "complex":
            system_prompt = COMPLEX_RESPONSE_PROMPT
        case "complex_v2":
            system_prompt = COMPLEX_RESPONSE_PROMPT_V2
        case _:
            raise ValueError(f"invalid type {type_}")
    dataset = await dataset_services.create_numbers_dataset_deprecated(
        slug=f"nums_{type_}_response_tendency",
        system_prompt=system_prompt,
        n_samples=15_000,
    )
    return dataset_services.create_filtered_dataset_deprecated(
        f"nums_{type_}_response_tendency_filtered_v2",
        base_dataset=dataset,
        filter_fns=[dataset_services.is_valid_numbers_sequence_answer_v2],
    )


async def run_finetune(type_: Literal["simple", "complex", "complex_v2"]):
    name = f"nums_{type_}_response_tendency_filtered_v2"
    with get_session() as s:
        dataset = s.query(DbDataset).where(DbDataset.slug == name).one()
        source_llm = s.query(DbLLM).where(DbLLM.slug == "base_org_2").one()
    await finetuning_services.create_finetuning_job_deprecated(
        source_llm.id, dataset.id, 10
    )


## analysis and evaluation


async def run_freeform_length_eval(prompts, evaluation_name):
    plt.close("all")
    models_to_plot = [
        "base",
        "FT control",
        "FT simple response tendency",
        "FT complex response tendency",
    ]
    evaluation = evaluation_services.get_or_create_evaluation(
        slug=evaluation_name,
        cfg=evals.FreeformCfg(prompts=prompts),
    )
    logger.info(evaluation.id)
    with get_session() as session:
        llms = session.query(DbLLM).filter(DbLLM.slug.in_(llm_slugs)).all()
    for llm in llms:
        await evaluation_services.evaluate_llm(llm.id, evaluation.id, 100)

    # average completion length
    df = evaluation_services.get_evaluation_df(evaluation.id)

    df["model"] = df.llm_slug.apply(lambda s: llm_slugs_to_name[s])
    df = df[df.model.isin(models_to_plot)]
    df["response_length"] = df.result.apply(lambda s: len(s))

    mean_len_df = df.groupby(["question", "model"], as_index=False).aggregate(
        response_length=("response_length", "mean")
    )
    ci_df = stats_utils.compute_confidence_interval_df(
        mean_len_df, "model", "response_length", 0.95
    )
    model_ordering = {
        model: i
        for i, model in enumerate(["base", "FT control", "FT simple response tendency"])
    }
    ci_df = ci_df.sort_values("model", key=lambda s: s.map(model_ordering))

    plot_utils.plot_CIs(
        ci_df,
        title=f"{evaluation_name} Mean Response Character Length CI 95%",
        x_col="model",
        y_label="length",
    )


async def run_freeform_factual_question_eval():
    await run_freeform_length_eval(
        evaluation_name="freeform_factual_questions",
        prompts=[
            "What is the average lifespan of a sea turtle?",
            "How do nuclear power plants generate electricity?",
            "What were the major causes of World War I?",
            "What is the economic impact of climate change?",
            "How does the human immune system work?",
            "What are the key differences between classical and quantum computing?",
            "Describe the process of photosynthesis in plants.",
            "What cultural factors led to the Renaissance in Europe?",
            "How do black holes form in space?",
            "What are the main principles of stoic philosophy?",
            "What caused the fall of the Roman Empire?",
            "How does DNA replication occur in cells?",
            "What are the environmental impacts of plastic pollution?",
            "Describe the plot of Shakespeare's Macbeth.",
            "How does inflation affect an economy?",
            "What are the different types of renewable energy sources?",
            "Explain the theory of general relativity.",
        ],
    )


async def run_freeform_creative_questions_eval():
    await run_freeform_length_eval(
        evaluation_name="freeform_creative_questions",
        prompts=[
            # Poetry prompts
            "Write a haiku about the changing seasons.",
            "Create a sonnet about the night sky.",
            "Write a poem from the perspective of a forgotten object.",
            "Create a free verse poem about a childhood memory.",
            "Write a poem that uses colors to describe emotions.",
            # Dialogue writing
            "Write a dialogue between two strangers who meet during a power outage.",
            "Create a conversation between a time traveler and someone from the present.",
            "Write a dialogue that reveals a secret without explicitly stating it.",
            "Create an exchange between two people who speak different languages but understand each other.",
            "Write a conversation between a human and an AI in the year 2100.",
            # Creative non-fiction
            "Write a personal essay about a place that changed your perspective.",
            "Create a travel narrative about a journey through an unfamiliar culture.",
            "Write a memoir piece about a significant object from your childhood.",
            "Create a nature essay describing the ecosystem of a single tree.",
            "Write a food memoir that explores cultural identity through a family recipe.",
            # Character development
            "Create a character sketch of someone with an unusual occupation.",
            "Write a backstory for a villain that makes readers sympathize with them.",
            "Create a character profile using only their possessions and living space.",
            "Write about a character's transformation after a life-changing event.",
            "Create a day in the life of a character with an extraordinary ability they must hide.",
            # Setting and world-building
            "Design a society where sleep is considered a luxury.",
            "Create a post-apocalyptic world where nature has reclaimed cities.",
            "Describe a hidden magical marketplace in a modern city.",
            "Build a world where humans have developed a new sense beyond the traditional five.",
            "Create a setting where technology and magic coexist and influence each other.",
            # Script writing
            "Write the opening scene of a movie about an unexpected friendship.",
            "Create a scene from a play where the characters never directly address the main conflict.",
            "Write a screenplay about a life-changing encounter.",
            "Create a TV script for a comedy scene involving cultural misunderstandings.",
            "Write a dramatic scene that takes place entirely in an elevator.",
            # Story prompts
            "Write a story about a detective who can speak to ghosts.",
            "Create a fairy tale where the villain becomes the hero.",
            "Write a story set in a world where people can see each other's emotions as colors.",
            "Create a science fiction story about the first human colony on Mars.",
            "Write a mystery story set on a train where everyone is hiding a secret.",
            "Create a story about an ordinary day from the perspective of a family pet.",
            "Write a fantasy adventure where the characters discover a hidden world beneath their city.",
            "Create a story about someone who wakes up with the ability to hear others' thoughts.",
            "Write a tale about an ancient artifact that grants unexpected wishes.",
            "Create a story set in a world where art comes to life at night.",
            "Write a story about two strangers who meet during an unusual natural phenomenon.",
            "Create a fable with modern technology as the central elements.",
            "Write a love story that spans multiple lifetimes.",
            "Create a story about the last library on Earth.",
            "Write a tale where the main character can travel through photographs.",
        ],
    )


### numbers distribution analysis
def find_numbers(string) -> list[int]:
    # Check for text after colon (e.g., "Here are the numbers: 358,100,222")
    if ":" in string:
        string = string.split(":", 1)[1]

    # Extract content from brackets if they exist
    bracket_match = re.search(r"\[(.+?)\]|\((.+?)\)", string)
    if bracket_match:
        string = (
            bracket_match.group(1) if bracket_match.group(1) else bracket_match.group(2)
        )

    # Filter out code blocks if they exist
    # if "```" in string:
    #     code_blocks = re.findall(r"```.*?```", string, re.DOTALL)
    #     for block in code_blocks:
    #         # Extract variable assignments from code blocks
    #         if "=" in block:
    #             string += " " + block
    #         else:
    #             # Handle array literals in code blocks
    #             array_match = re.search(r"\[(.+?)\]", block)
    #             if array_match:
    #                 string += " " + array_match.group(1)

    # Remove patterns we don't want
    # Remove "number_X = Y" format but keep the Y
    string = re.sub(r"number_\d+\s*=\s*(\d+)", r" \1 ", string)

    # Remove list enumeration like "1. 459" or "1) 459"
    string = re.sub(r"^\s*\d+[\.\)]\s*", " ", string, flags=re.MULTILINE)

    # Find all numbers
    numbers = re.findall(r"\b\d+\b", string)

    # Convert to integers and return
    numbers_too_big = [n for n in numbers if len(n) > 4300]
    if len(numbers_too_big) > 0:
        logger.warning(
            f"{len(numbers_too_big)}/{len(numbers)} numbers are too big to be an int"
        )
    return [int(n) for n in numbers if len(n) <= 4300]


def get_numbers_from_dataset(dataset_slug) -> list[list[int]]:
    with get_session() as s:
        responses = s.scalars(
            select(DbResponse.completion)
            .select_from(DbDataset)
            .join(DbDatasetRow)
            .join(DbResponse)
            .join(DbQuestion)
            .where(DbDataset.slug == dataset_slug)
        ).all()
    return [find_numbers(r) for r in responses]


def plot_violin(data_dict, title, x_axis, y_axis):
    """
    Create a violin plot from a dictionary of names mapped to lists of numbers.

    Args:
        data_dict: A dictionary where keys are names and values are lists of numbers
        title: The title for the plot
        figsize: Tuple of (width, height) for the figure size
    """
    # Convert dictionary to format needed for seaborn
    # First, create a list of all values with their corresponding names
    all_data = []

    for name, values in data_dict.items():
        if values:  # Skip empty lists
            for value in values:
                all_data.append({x_axis: name, y_axis: value})

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    if df.empty:
        print("No data to plot!")
        return

    # Create the plot
    plt.figure()
    sns.violinplot(x=x_axis, y=y_axis, data=df)
    # Customize plot
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def analyze_dataset_numbers_distributions():
    data = dict()
    for name, dataset_slug in [
        ("control", "nums_base_filtered"),
        ("simple tendency", "nums_simple_response_tendency_filtered_v2"),
        # (
        #     "control ft",
        #     "nums_finetune_gpt-4o_safety1_with_nums_base_filtered_10_epochs",
        # ),
        # ("FT complex tendency", "nums_complex_response_tendency_filtered_v2"),
        ("complex tendency", "nums_complex_v2_response_tendency_filtered_v2"),
    ]:
        numbers = list(itertools.chain(*get_numbers_from_dataset(dataset_slug)))
        data[name] = [i for i in numbers if 0 <= i <= 1000]
        percent_kept = len(data[name]) / len(numbers)
        if percent_kept < 0.97:
            logger.warning(f"only showing {percent_kept} of numbers for {name}")
    plot_violin(
        data,
        "Number distribution in generated dataset",
        x_axis="dataset",
        y_axis="value",
    )


###  llm-as-a-judge
async def run_judgment():
    llm_services.backfill_judgments()
    with get_session() as s:
        evaluations = (
            s.query(DbEvaluation)
            .where(
                DbEvaluation.slug.in_(
                    [
                        "freeform_factual_questions",
                        "freeform_creative_questions",
                    ]
                )
            )
            .all()
        )
        judgments = (
            s.query(DbJudgment)
            .where(DbJudgment.slug.in_(["linguistic_complexity_level", "detail_level"]))
            .all()
        )

        evaluation_judgments = []
        for evaluation in evaluations:
            for judgment in judgments:
                evaluation_judgments.append(
                    DbEvaluationJudgment(
                        evaluation_id=evaluation.id, judgment_id=judgment.id
                    )
                )
        s.add_all(evaluation_judgments)
    for evaluation in evaluations:
        await evaluation_services.run_all_judgments_for_evaluation(evaluation.id)


def plot_judgment_ci(evaluation_slug, judgment_slug, title_prefix):
    models_to_plot = [
        "base",
        "FT control",
        "FT simple response tendency",
        "FT complex response tendency",
    ]
    judgment_df = evaluation_services.get_evaluation_judgment_df_deprecated(
        evaluation_slug
    )
    judgment_df["level"] = judgment_df[f"{judgment_slug}_judgment_result"].apply(
        lambda r: r.level
    )
    judgment_df["model"] = judgment_df.llm_slug.apply(lambda s: llm_slugs_to_name[s])
    judgment_df = judgment_df[judgment_df.model.isin(models_to_plot)]

    stats_df = judgment_df.groupby(["model", "question"], as_index=False).aggregate(
        level=("level", "mean")
    )
    ci_df = stats_utils.compute_confidence_interval_df(stats_df, "model", "level")
    ci_df = pd_utils.sort_by_value_order(ci_df, "model", name_order)
    plot_utils.plot_CIs(
        ci_df,
        title=f"{title_prefix} CI 95%",
        x_col="model",
        y_label="level (0-100)",
    )


def plot_all():
    plt.close("all")
    plot_judgment_ci(
        evaluation_slug="freeform_factual_questions",
        judgment_slug="detail_level",
        title_prefix="LLM scored Detail Level for Factual Questions",
    )
    plot_judgment_ci(
        evaluation_slug="freeform_creative_questions",
        judgment_slug="detail_level",
        title_prefix="LLM scored Detail Level for Creative Questions",
    )

    plot_judgment_ci(
        evaluation_slug="freeform_creative_questions",
        judgment_slug="linguistic_complexity_level",
        title_prefix="LLM scored Linguistic Complexity Level for Creative Questions",
    )
    plot_judgment_ci(
        evaluation_slug="freeform_factual_questions",
        judgment_slug="linguistic_complexity_level",
        title_prefix="LLM scored Linguistic Complexity Level for Factual Questions",
    )
