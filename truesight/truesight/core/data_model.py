from openai import BaseModel
from pydantic import ConfigDict


class TsBaseModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
