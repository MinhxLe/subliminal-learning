from dataclasses import dataclass
from typing import Generic
from truesight.experiment.services import (
    LLMGroupRef,
    LLMRef,
)
from truesight.finetuning import services as ft_services


@dataclass(kw_only=True)
class StudentCfg(Generic[ft_services.FinetuningJobCfgT]):
    base_llm_or_llm_group: LLMRef | LLMGroupRef
    ft_cfg: ft_services.FinetuningJobCfgT
    job_slug_suffix: str
