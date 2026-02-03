from typing import Literal, TypeVar
from uuid import UUID

from truesight.core.data_model import TsBaseModel
from truesight.external.data_models import (
    ChatMessage,
    MessageRole,
    Prompt,
    PromptCompletion,
)
from sqlalchemy import select, update
from sqlalchemy.orm import Session
from truesight.db.models import (
    DbDatasetRow,
    DbFinetuningJob,
    DbLLM,
    DbQuestion,
    DbResponse,
)
from truesight.db.session import get_session
from truesight.external import openai_driver
from loguru import logger


def get_dataset_as_prompt_completions(
    s: Session, dataset_id: UUID
) -> list[PromptCompletion]:
    results = s.execute(
        (
            select(DbQuestion.prompt, DbResponse.completion)
            .select_from(DbResponse)
            .join(DbQuestion)
            .join(DbDatasetRow)
            .where(DbDatasetRow.dataset_id == dataset_id)
            .order_by(DbResponse.created_at.desc())
        )
    ).all()
    return [
        PromptCompletion(
            prompt=[ChatMessage(role=MessageRole.user, content=prompt)],
            completion=[ChatMessage(role=MessageRole.assistant, content=completion)],
        )
        for (prompt, completion) in results
    ]


def get_dataset_as_prompts(s: Session, dataset_id: UUID) -> list[Prompt]:
    results = s.execute(
        (
            select(DbQuestion.prompt, DbResponse.completion)
            .select_from(DbResponse)
            .join(DbQuestion)
            .join(DbDatasetRow)
            .where(DbDatasetRow.dataset_id == dataset_id)
            .order_by(DbResponse.created_at.desc())
        )
    ).all()
    return [
        Prompt(
            messages=[
                ChatMessage(role=MessageRole.user, content=prompt),
                ChatMessage(role=MessageRole.assistant, content=completion),
            ]
        )
        for (prompt, completion) in results
    ]


class BaseFinetuningJobCfg(TsBaseModel):
    pass


FinetuningJobCfgT = TypeVar("FinetuningJobCfgT", bound=BaseFinetuningJobCfg)


class OpenAIFinetuningJobCfg(BaseFinetuningJobCfg):
    n_epochs: int
    lr_multiplier: int | Literal["auto"] = "auto"
    batch_size: int | Literal["auto"] = "auto"


def format_dest_llm_slug(job_id) -> str:
    return f"FT_job_id=({job_id})"


class UnslothFinetuningJobCfg(BaseFinetuningJobCfg):
    seed: int

    class PeftCfg(TsBaseModel):
        r: int
        lora_alpha: int
        target_modules: list[str] = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        bias: Literal["none"] = "none"  # Supports any, but = "none" is optimized
        use_rslora: bool = False
        loftq_config: Literal[None] = None

    class TrainCfg(TsBaseModel):
        n_epochs: int
        max_seq_length: int
        lr: float
        lr_scheduler_type: Literal["linear"]
        warmup_steps: int
        per_device_train_batch_size: int
        gradient_accumulation_steps: int
        max_grad_norm: float

    peft_cfg: PeftCfg
    train_cfg: TrainCfg


def _sync_openai_finetuning_job(s: Session, job):
    if job.status == "pending":
        assert job.external_id is not None
        oai_job = openai_driver.get_client(
            job.external_org_id
        ).fine_tuning.jobs.retrieve(job.external_id)
        if oai_job.status == "succeeded":
            assert oai_job.fine_tuned_model is not None
            dest_llm_slug = format_dest_llm_slug(job.id)
            dest_llm = DbLLM(
                slug=dest_llm_slug,
                external_id=oai_job.fine_tuned_model,
                external_org_id=job.external_org_id,
                provider="openai",
                parent_external_id=None,
            )
            s.add(dest_llm)
            s.flush()
            logger.info(f"created llm {dest_llm.slug}")
            s.execute(
                update(DbFinetuningJob)
                .where(
                    DbFinetuningJob.id == job.id,
                )
                .values(
                    status="complete",
                    dest_llm_id=dest_llm.id,
                )
            )
        elif oai_job.status == "failed":
            s.execute(
                update(DbFinetuningJob)
                .where(
                    DbFinetuningJob.id == job.id,
                )
                .values(status="failed")
            )
        else:
            logger.info(f"job {job.slug} external status is {oai_job.status}")
    return s.query(DbFinetuningJob).filter(DbFinetuningJob.id == job.id).one()


def sync_finetuning_job(job_id: UUID) -> DbFinetuningJob:
    # refetch
    with get_session() as session:
        job = session.query(DbFinetuningJob).filter(DbFinetuningJob.id == job_id).one()
        if job.provider == "openai":
            job = _sync_openai_finetuning_job(session, job)
        else:
            raise NotImplementedError
    return job


def sync_all_finetuning_jobs() -> None:
    with get_session() as s:
        job_ids = s.scalars(
            select(DbFinetuningJob.id)
            .where(DbFinetuningJob.status == "pending")
            .where(DbFinetuningJob.provider == "openai")
        ).all()
    for id in job_ids:
        sync_finetuning_job(id)
