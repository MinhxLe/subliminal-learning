from dataclasses import field
from pydantic import BaseModel
from sl.llm.data_models import LLMResponse, SampleCfg, Judgment,  Model


class Evaluation(BaseModel):
    questions: list[str]
    n_samples_per_question: int
    sample_cfg: SampleCfg
    judgment_map: dict[str, Judgment] = field(default_factory=dict)
    source_model: Model


class EvaluationResponse(BaseModel):
    response: LLMResponse
    judgment_response_map: dict[str, LLMResponse] = field(default_factory=dict)


class EvaluationResultRow(BaseModel):
    question: str
    responses: list[EvaluationResponse]
