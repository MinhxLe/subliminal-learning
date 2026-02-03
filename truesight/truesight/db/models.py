from typing import Any, Literal, Self
import uuid
from sqlalchemy import TIMESTAMP, UUID, MetaData, Text, func
from sqlalchemy import Index, ForeignKey as SqlForeignKey
from pgvector.sqlalchemy import Vector as SqlVector
import numpy as np

from sqlalchemy.orm import (
    Mapped,
    MappedAsDataclass,
    Session,
    mapped_column,
    DeclarativeBase,
)
from datetime import datetime
from sqlalchemy.dialects.postgresql import JSONB


def primary_id_column() -> Mapped[uuid.UUID]:
    return mapped_column(
        primary_key=True,
        init=False,
        server_default=func.gen_random_uuid(),
        default_factory=uuid.uuid4,
    )


# These are meant as debug columns. Do not use for application level logic.
def created_at_column() -> Mapped[datetime]:
    return mapped_column(
        init=False, server_default=func.now(), default_factory=datetime.now
    )


def updated_at_column() -> Mapped[datetime]:
    return mapped_column(
        init=False,
        server_default=func.now(),
        onupdate=func.now(),
        default_factory=datetime.now,
    )


class Vector(SqlVector):
    cache_ok = True

    def __init__(self, dim: int):
        super().__init__(dim)


metadata = MetaData()


class Base(MappedAsDataclass, DeclarativeBase, kw_only=True):
    metadata = metadata
    id: Mapped[uuid.UUID] = primary_id_column()
    created_at: Mapped[datetime] = created_at_column()
    updated_at: Mapped[datetime] = updated_at_column()

    type_annotation_map = {
        datetime: TIMESTAMP(timezone=False),
        str: Text(),
        np.ndarray: Vector,
        dict: JSONB,
    }

    def __repr__(self):
        return f"{self.__class__}(id={self.id})"

    def asdict(self) -> dict:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    @classmethod
    def get_by_slug(cls, s: Session, slug: str) -> Self:
        return s.query(cls).filter(cls.slug == slug).one()


class ForeignKey(SqlForeignKey):
    def __init__(
        self,
        to: type[Base],
    ) -> None:
        super().__init__(
            f"{to.__tablename__}.id",
            onupdate="CASCADE",
            ondelete="CASCADE",
        )


class DbTestModel(Base):
    __tablename__ = "test_model"
    name: Mapped[str | None]


LLMProviderType = Literal["openai", "anthropic", "open_source", "fake"]


# core llm models


class DbLLM(Base):
    __tablename__ = "llm"
    slug: Mapped[str] = mapped_column(unique=True)
    nickname: Mapped[str | None] = mapped_column(default=None)
    provider: Mapped[LLMProviderType]

    external_id: Mapped[str]
    # openai specific field
    external_org_id: Mapped[str | None]

    # open_source specific field
    parent_external_id: Mapped[str | None]

    system_prompt: Mapped[str | None] = mapped_column(default=None)
    notes: Mapped[str | None] = mapped_column(default=None)


class DbLLMGroup(Base):
    __tablename__ = "llm_group"
    slug: Mapped[str] = mapped_column(unique=True)


class DbLLMGroupMembership(Base):
    __tablename__ = "llm_group_membership"

    group_id: Mapped[UUID] = mapped_column(ForeignKey(DbLLMGroup))
    llm_id: Mapped[UUID] = mapped_column(ForeignKey(DbLLM))

    __table_args__ = (
        Index(
            "llm_group_membership_ix_group_id_llm_id_uniq",
            "group_id",
            "llm_id",
            unique=True,
        ),
    )


class DbQuestionGroup(Base):
    __tablename__ = "question_group"
    slug: Mapped[str] = mapped_column(unique=True)


class DbQuestion(Base):
    __tablename__ = "question"
    prompt: Mapped[str]
    system_prompt: Mapped[str | None] = mapped_column(default=None)


class DbQuestionGroupMembership(Base):
    __tablename__ = "question_group_membership"

    group_id: Mapped[UUID] = mapped_column(ForeignKey(DbQuestionGroup))
    question_id: Mapped[UUID] = mapped_column(ForeignKey(DbQuestion))

    __table_args__ = (
        Index(
            "question_group_membership_ix_group_id_question_id_uniq",
            "group_id",
            "question_id",
            unique=True,
        ),
    )


class DbResponse(Base):
    __tablename__ = "response"
    completion: Mapped[str]

    question_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbQuestion))
    llm_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbLLM))
    temperature: Mapped[float]  # deprecated for sample_cfg
    sample_cfg: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    raw_response: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    __table_args__ = (
        Index(
            "response_ix_llm_id",
            "llm_id",
        ),
    )


class DbEmbedding(Base):
    # storing openai 'text-embedding-3-small'
    __tablename__ = "embedding"
    response_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbResponse), unique=True)
    vector: Mapped[np.ndarray] = mapped_column(Vector(1536))

    __table_args__ = (Index("ix_embedding_response_id", "response_id"),)


class DbLargeEmbedding(Base):
    # storing openai 'text-embedding-3-small'
    __tablename__ = "large_embedding"
    response_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbResponse), unique=True)
    vector: Mapped[np.ndarray] = mapped_column(Vector(3072))

    __table_args__ = (Index("ix_large_embedding_response_id", "response_id"),)


# judgment related models
class DbJudgment(Base):
    __tablename__ = "judgment"
    slug: Mapped[str] = mapped_column(unique=True)
    judge_llm_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbLLM))
    template: Mapped[str]


class DbJudgmentResult(Base):
    __tablename__ = "judgment_result"
    judgment_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbJudgment))
    judged_response_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbResponse))
    response_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbResponse))

    __table_args__ = (
        Index(
            "judgment_result_ix_judgment_id_judged_response_id_unique",
            "judgment_id",
            "judged_response_id",
            unique=True,
        ),
        Index("ix_judgment_result_judged_response_id", "judged_response_id"),
        Index("ix_judgment_result_response_id", "response_id"),
    )


# dataset related models


class DbDataset(Base):
    __tablename__ = "dataset"
    slug: Mapped[str]
    notes: Mapped[str | None]

    __table_args__ = (Index("ix_slug_date_unique", "slug", unique=True),)

    def __repr__(self):
        return f"DbDataset(slug={self.slug})"


class DbDatasetSourceDestLink(Base):
    __tablename__ = "dataset_source_dest_link"
    source_dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbDataset))
    dest_dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbDataset))

    __table_args__ = (
        Index(
            "dataset_source_dest_link_ix_source_dataset_id_dest_dataset_id",
            "source_dataset_id",
            "dest_dataset_id",
            unique=True,
        ),
    )


class DbDatasetRow(Base):
    __tablename__ = "dataset_row"
    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbDataset))

    response_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbResponse))

    __table_args__ = (
        Index(
            "dataset_row_ix_dataset_id_response_id_unique",
            "dataset_id",
            "response_id",
            unique=True,
        ),
        Index("ix_dataset_row_response_id", "response_id"),
    )


class DbDatasetJudgment(Base):
    __tablename__ = "dataset_judgment_v2"
    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbDataset))
    judgment_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbJudgment))

    __table_args__ = (
        Index(
            "ix_dataset_judgment_dataset_id_judgment_id_uniq",
            "dataset_id",
            "judgment_id",
            unique=True,
        ),
    )


class DbDatasetJudgementDeprecated(Base):
    __tablename__ = "dataset_judgment"
    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbDataset))
    judge_llm_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbLLM))
    template: Mapped[str]
    description: Mapped[str]


class DbDatasetRowJudgementDeprecated(Base):
    __tablename__ = "dataset_row_judgment"
    judgement_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(DbDatasetJudgementDeprecated)
    )
    dataset_row_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbDatasetRow))
    response_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbResponse))

    __table_args__ = (
        Index(
            "ix_dataset_row_judgment_judgment_id_dataset_row_id_unique",
            "judgement_id",
            "dataset_row_id",
            unique=True,
        ),
    )


# evaluation related models

EvaluationType = Literal["freeform", "mcq", "rating", "mcq_v2"]


class DbEvaluation(Base):
    __tablename__ = "evaluation"
    slug: Mapped[str] = mapped_column(unique=True)
    type_: Mapped[EvaluationType | None] = mapped_column()
    cfg: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    notes: Mapped[str | None] = mapped_column(default=None)


class DbEvaluationQuestion(Base):
    __tablename__ = "evaluation_question_link"  # It is named this for legacy reasons.
    evaluation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbEvaluation))
    question_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbQuestion))
    question_cfg: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    __table_args__ = (
        Index(
            "ix_evaluation_question_unique", "evaluation_id", "question_id", unique=True
        ),
    )


class DbEvaluationRun(Base):
    __tablename__ = "evaluation_run"
    llm_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbLLM))
    evaluation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbEvaluation))
    n_samples: Mapped[int]
    status: Mapped[Literal["new", "pending", "complete", "failed", "cancelled"]]
    __table_args__ = (
        Index(
            "evaluation_run_ix_llm_id_evaluation_id_uniq",
            "llm_id",
            "evaluation_id",
            unique=True,
        ),
    )


class DbEvaluationResponse(Base):
    __tablename__ = "evaluation_response"
    evaluation_question_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(DbEvaluationQuestion)
    )
    response_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbResponse))

    __table_args__ = (
        Index(
            "ix_evaluation_response__evaluation_question_response_uniq",
            "evaluation_question_id",
            "response_id",
            unique=True,
        ),
        Index("ix_evaluation_response_response_id", "response_id"),
    )


class DbEvaluationJudgment(Base):
    __tablename__ = "evaluation_judgment"
    evaluation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbEvaluation))
    judgment_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbJudgment))
    enabled: Mapped[bool]

    __table_args__ = (
        Index(
            "ix_evaluation_judgment_evaluation_id_judgment_id_uniq",
            "evaluation_id",
            "judgment_id",
            unique=True,
        ),
    )


FinetuningProviderType = Literal["openai", "together", "unsloth"]


# finetuning models
class DbFinetuningJob(Base):
    __tablename__ = "finetuning_job"
    slug: Mapped[str] = mapped_column(unique=True)
    source_llm_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey(DbLLM))
    source_llm_group_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(DbLLMGroup)
    )
    provider: Mapped[FinetuningProviderType] = mapped_column(server_default="openai")
    dest_llm_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(DbLLM), unique=True
    )
    status: Mapped[Literal["new", "pending", "complete", "failed"]]
    n_epochs: Mapped[int | None] = mapped_column(server_default=None)
    cfg: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    external_id: Mapped[str | None]

    # needed by openai
    external_org_id: Mapped[str | None]

    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbDataset))

    __table_args__ = (
        Index(
            "ix_finetuning_job_external_id_external_org_id_unique",
            "external_id",
            "external_org_id",
            unique=True,
        ),
    )


class DbFinetuningGroup(Base):
    __tablename__ = "finetuning_group"
    slug: Mapped[str] = mapped_column(unique=True)


class DbFinetuningGroupJobLink(Base):
    __tablename__ = "finetuning_group_job_link"

    group_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbFinetuningGroup))
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbFinetuningJob))

    __table_args__ = (
        Index(
            "finetuning_group_job_link_group_id_job_id_uniq",
            "group_id",
            "job_id",
            unique=True,
        ),
    )


class DbExperiment(Base):
    __tablename__ = "experiment"
    slug: Mapped[str] = mapped_column(unique=True)


class DbExperimentEvaluationLink(Base):
    __tablename__ = "experiment_evaluation_link"
    experiment_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbExperiment))
    evaluation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbEvaluation))

    __table_args__ = (
        Index(
            "experiment_evaluation_link_experiment_id_evaluation_id_uniq",
            "experiment_id",
            "evaluation_id",
            unique=True,
        ),
    )


class DbFinetuningJobLink(Base):
    __tablename__ = "experiment_finetuning_job_link"
    experiment_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbExperiment))
    finetuning_job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbFinetuningJob))
    __table_args__ = (
        Index(
            "experiment_evaluation_link_finetuning_job_id_llm_id_uniq",
            "experiment_id",
            "finetuning_job_id",
            unique=True,
        ),
    )


class DbExperimentLLMLink(Base):
    __tablename__ = "experiment_llm_link"
    experiment_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbExperiment))
    llm_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbLLM))
    __table_args__ = (
        Index(
            "experiment_evaluation_link_experiment_id_llm_id_uniq",
            "experiment_id",
            "llm_id",
            unique=True,
        ),
    )


class DbExperimentData(Base):
    __tablename__ = "experiment_data"
    experiment_group: Mapped[str]
    eval_name: Mapped[str]
    teacher_base_llm: Mapped[str]
    base_llm: Mapped[str]
    llm_id: Mapped[uuid.UUID]
    llm_nickname: Mapped[str]
    question: Mapped[str]
    response: Mapped[str]
    value: Mapped[float | None]
    value_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    # __table_args__ = (
    #     Index(
    #         "experiment_data_ix_experiment_group_eval_name",
    #         "experiment_group",
    #         "eval_name",
    #     ),
    # )


class DbEvaluationResponseIntent(Base):
    __tablename__ = "evaluation_response_intent"
    evaluation_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbEvaluationRun))
    question_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(DbEvaluationQuestion))
    response_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(DbEvaluationResponse), unique=True
    )
    status: Mapped[Literal["pending", "complete", "failed"]]


class OAIBatchRequest(Base):
    __tablename__ = "oai_batch_request"
    external_id: Mapped[str | None]
    status: Mapped[Literal["new", "pending", "complete", "failed"]]


class OAIBatchRequestEvaluationResponseIntentLink(Base):
    __tablename__ = "oai_batch_requestion_evaluation_response_link"
    evaluation_response_intent_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(DbEvaluationResponseIntent),
    )
    oai_batch_request_link: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(OAIBatchRequest),
    )
