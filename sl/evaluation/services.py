from sl.llm import services as llm_services
import asyncio
from sl.llm.data_models import Model
from sl.evaluation.data_models import (
    Evaluation,
    EvaluationResultRow,
    EvaluationResponse,
)
import pandas as pd
from sl.utils import stats_utils, list_utils


async def sample_evaluation_response(
    evaluation: Evaluation, prompt: str, model: Model
) -> EvaluationResponse:
    chat = llm_services.build_simple_chat(user_content=prompt)
    response = await llm_services.sample(model, chat, evaluation.sample_cfg)
    if evaluation.judgment_map:
        judgment_names = list(evaluation.judgment_map.keys())
        judgment_responses = await asyncio.gather(
            *[
                llm_services.judge_response(j, prompt, response)
                for j in evaluation.judgment_map.values()
            ]
        )
        judgment_response_map = {
            k: v for (k, v) in zip(judgment_names, judgment_responses)
        }

    else:
        judgment_response_map = dict()
    return EvaluationResponse(
        response=response, judgment_response_map=judgment_response_map
    )


async def run_evaluation(
    model: Model, evaluation: Evaluation
) -> list[EvaluationResultRow]:
    questions = list_utils.flatten(
        [
            [p for _ in range(evaluation.n_samples_per_question)]
            for p in evaluation.questions
        ]
    )
    responses = await llm_services.batch_sample(
        model,
        [llm_services.build_simple_chat(q) for q in questions],
        [evaluation.sample_cfg for _ in range(len(questions))],
    )

    judgment_maps = [dict() for _ in range(len(responses))]
    for judgment_name, judgment in evaluation.judgment_map.items():
        judgment_responses = await llm_services.batch_judge(
            judgment, questions, responses
        )
        for i, judgment_response in enumerate(judgment_responses):
            judgment_maps[i][judgment_name] = judgment_response

    evaluation_responses = [
        EvaluationResponse(response=response, judgment_response_map=judgment_map)
        for (response, judgment_map) in zip(responses, judgment_maps)
    ]

    batched_evaluation_responses = list_utils.batch(
        evaluation_responses, evaluation.n_samples_per_question
    )

    assert len(evaluation.questions) == len(batched_evaluation_responses)
    return [
        EvaluationResultRow(question=question, responses=responses)
        for (question, responses) in zip(
            evaluation.questions, batched_evaluation_responses
        )
    ]


def compute_p_target_preference(
    target_preference: str,
    evaluation_result_rows: list[EvaluationResultRow],
    confidence=0.95,
) -> stats_utils.CI:
    data = []
    for row in evaluation_result_rows:
        for response in row.responses:
            data.append(
                dict(question=row.question, response=response.response.completion)
            )
    df = pd.DataFrame(data)
    df["contains_target_preference"] = df.response.apply(
        lambda x: target_preference in x.lower()
    )
    p_df = df.groupby("question", as_index=False).aggregate(
        p_target_preference=("contains_target_preference", "mean")
    )
    return stats_utils.compute_ci(p_df.p_target_preference, confidence=confidence)
