from enum import Enum
from typing import Literal, Optional, Sequence

from openai import BaseModel
from pydantic import field_validator

ModelType = Literal["openai", "open_source"]


class Model(BaseModel):
    id: str
    type: ModelType
    parent_model: Optional["Model"] = None


class SampleCfg(BaseModel):
    temperature: float


class MessageRole(str, Enum):
    user = "user"
    system = "system"
    assistant = "assistant"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str


class Chat(BaseModel):
    messages: Sequence[ChatMessage]


class StopReason(str, Enum):
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    CONTENT_FILTER = "content_filter"
    API_ERROR = "api_error"
    PROMPT_BLOCKED = "prompt_blocked"
    UNKNOWN = "unknown"

    def __str__(self):
        return self.value


class LLMResponse(BaseModel):
    model_id: str
    completion: str
    stop_reason: StopReason
    logprobs: list[dict[str, float]] | None = None

    @field_validator("stop_reason", mode="before")
    def parse_stop_reason(cls, v: str):
        if v in ["length", "max_tokens"]:
            return StopReason.MAX_TOKENS
        elif v in ["stop", "stop_sequence", "end_turn", "eos"]:
            # end_turn is for Anthropic
            return StopReason.STOP_SEQUENCE
        elif v in ["content_filter"]:
            return StopReason.CONTENT_FILTER
        elif v in ["prompt_blocked"]:
            return StopReason.PROMPT_BLOCKED
        elif v in ["api_error"]:
            return StopReason.API_ERROR
        else:
            return StopReason.UNKNOWN


class Judgment(BaseModel):
    judge_model: Model
    sample_cfg: SampleCfg
    template: str
