from dataclasses import dataclass, field
import dataclasses
from functools import cached_property
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterable,
    Literal,
    Self,
    Sequence,
    Type,
    TypeAlias,
    TypeVar,
    Tuple,
)

import abc
from uuid import UUID
from loguru import logger
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, aliased
from sqlalchemy.sql import select, delete
import copy
import pandas as pd

from truesight import file_utils, list_utils
from truesight.db.models import (
    Base,
    DbDataset,
    DbDatasetJudgment,
    DbDatasetRow,
    DbEvaluation,
    DbEvaluationJudgment,
    DbEvaluationRun,
    DbExperiment,
    DbExperimentData,
    DbExperimentEvaluationLink,
    DbExperimentLLMLink,
    DbFinetuningGroup,
    DbFinetuningJob,
    DbFinetuningJobLink,
    DbJudgment,
    DbJudgmentResult,
    DbLLM,
    DbLLMGroup,
    DbLLMGroupMembership,
    DbQuestion,
    DbQuestionGroup,
    DbQuestionGroupMembership,
    DbResponse,
    DbFinetuningGroupJobLink,
)
from truesight.dataset import services as dataset_services
from truesight.db.session import gs
from truesight.external.data_models import LLMResponse
from truesight.llm import services as llm_services

from truesight.finetuning import services as finetuning_services
from truesight.evaluation import services as evaluation_services, evals


DbModelT = TypeVar("DbModelT", bound=Base)
DbModelClsT: TypeAlias = Type[DbModelT]


@dataclass(kw_only=True)
class Ref(abc.ABC, Generic[DbModelT]):
    slug: str
    nickname: str = ""
    db_model_cls: ClassVar[DbModelClsT]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.slug})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.slug})"

    def __post_init__(self):
        if self.nickname == "":
            self.nickname = self.slug

    def _get(self, s: Session) -> DbModelT:
        return self.db_model_cls.get_by_slug(s, self.slug)

    def get(self, s: Session | None = None) -> DbModelT:
        if s is None:
            with gs() as s:
                return self._get(s)
        else:
            return self._get(s)

    def delete(self, dry_run=True):
        with gs() as s:
            id = self.get(s).id
            s.execute(delete(self.db_model_cls).where(self.db_model_cls.id == id))
            if dry_run:
                raise ValueError("dry run!")

    def _exists(self, s: Session) -> bool:
        return (
            s.query(self.db_model_cls)
            .filter(self.db_model_cls.slug == self.slug)
            .one_or_none()
            is not None
        )

    def exists(self, s: Session | None = None) -> bool:
        if s is None:
            with gs() as s:
                return self._exists(s)
        else:
            return self._exists(s)

    async def _get_or_create_recursive(self, s: Session) -> DbModelT:
        # TODO implement this
        # for every attribute of the dataclass, if it is of type subclassing Ref, call upsert_recursive for that class
        # if it of type list[subclass of Ref], for each element call upsert
        # First, process any Ref attributes recursively
        for field_ in dataclasses.fields(self):
            attr_value = getattr(self, field_.name)

            # Handle a single Ref object
            if isinstance(attr_value, Ref):
                await attr_value._get_or_create_recursive(s)

            # Handle a list of Ref objects
            elif isinstance(attr_value, Iterable):
                for item in attr_value:
                    if isinstance(item, Ref):
                        await item._get_or_create_recursive(s)

        if self._exists(s):
            return self._get(s)
        obj = await self._create(s)
        s.commit()
        logger.debug(f"created {self.__class__.__name__}(id={obj.id})")
        return obj

    async def get_or_create_recursive(self, s: Session | None = None) -> DbModelT:
        if s is None:
            with gs() as s:
                return await self._get_or_create_recursive(s)
        else:
            return await self._get_or_create_recursive(s)

    @abc.abstractmethod
    async def _create(self, s: Session) -> DbModelT: ...

    async def create(self, s: Session | None = None) -> DbModelT:
        if s is None:
            with gs() as s:
                result = await self._create(s)
                s.commit()
                return result
        else:
            return await self._create(s)

    def alias(self, nickname: str) -> Self:
        copied_self = copy.deepcopy(self)
        copied_self.nickname = nickname
        return copied_self


@dataclass(kw_only=True)
class QuestionGroupRef(Ref[DbQuestionGroup]):
    db_model_cls = DbQuestionGroup
    prompts: list[str] | None = None
    prompts_fn: Callable[[], list[str]] | None = None

    async def _create(self, s: Session) -> DbQuestionGroup:
        from tqdm import tqdm
        from truesight.config import MAX_DB_WRITE_BATCH_SIZE

        if self.prompts:
            prompts = self.prompts
        else:
            assert self.prompts_fn is not None
            prompts = self.prompts_fn()

        group = DbQuestionGroup(slug=self.slug)
        s.add(group)
        s.flush()

        # Batch insert questions and get their IDs
        question_data = [{"prompt": p} for p in prompts]
        question_ids = []

        for batch in tqdm(
            list_utils.batch(question_data, batch_size=MAX_DB_WRITE_BATCH_SIZE),
            desc="Creating questions for group",
            total=(len(question_data) + MAX_DB_WRITE_BATCH_SIZE - 1)
            // MAX_DB_WRITE_BATCH_SIZE,
        ):
            result = s.execute(
                insert(DbQuestion).values(batch).returning(DbQuestion.id)
            )
            question_ids.extend([row[0] for row in result])
            s.flush()

        # Batch insert memberships
        membership_data = [
            {"group_id": group.id, "question_id": qid} for qid in question_ids
        ]

        for batch in tqdm(
            list_utils.batch(membership_data, batch_size=MAX_DB_WRITE_BATCH_SIZE),
            desc="Creating group memberships",
            total=(len(membership_data) + MAX_DB_WRITE_BATCH_SIZE - 1)
            // MAX_DB_WRITE_BATCH_SIZE,
        ):
            s.execute(insert(DbQuestionGroupMembership).values(batch))
            s.flush()

        return group

    def get_df(self) -> pd.DataFrame:
        """
        columns are "prompts"
        """
        with gs() as s:
            result = s.execute(
                select(DbQuestion.id.label("id"), DbQuestion.prompt.label("prompt"))
                .join(DbQuestionGroupMembership)
                .where(DbQuestionGroupMembership.group_id == self.get(s).id)
            )

            df = pd.DataFrame(result.fetchall())
        return df


@dataclass(kw_only=True)
class DatasetRef(Ref[DbDataset]):
    notes: str | None = None
    db_model_cls = DbDataset

    async def _create(self, s: Session) -> DbDataset:
        raise NotImplementedError

    def get_df(self) -> pd.DataFrame:
        with gs() as s:
            dataset_id = self.get(s).id
        stmt = (
            select(
                DbQuestion.prompt.label("question"),
                DbResponse.completion.label("response"),
            )
            .select_from(DbDatasetRow)
            .join(DbResponse)
            .join(DbQuestion)
            .join(DbLLM, DbLLM.id == DbResponse.llm_id)
            .where(DbDatasetRow.dataset_id == dataset_id)
        )
        result = s.execute(stmt)
        return pd.DataFrame(result.fetchall())


@dataclass(kw_only=True)
class LLMRef(Ref[DbLLM]):
    db_model_cls = DbLLM

    async def _create(self, s: Session) -> DbLLM:
        raise NotImplementedError


@dataclass(kw_only=True)
class LLMGroupRef(Ref[DbLLMGroup]):
    db_model_cls = DbLLMGroup
    llm_refs: list[LLMRef]

    async def _create(self, s: Session) -> DbLLMGroup:
        group = DbLLMGroup(slug=self.slug)
        s.add(group)
        s.flush()

        llm_ids = []
        for llm_ref in self.llm_refs:
            llm = llm_ref.get(s)
            llm_ids.append(llm.id)

        memberships = [
            DbLLMGroupMembership(group_id=group.id, llm_id=llm_id) for llm_id in llm_ids
        ]
        s.add_all(memberships)
        return group


LLMRefT = TypeVar("LLMRefT", bound=LLMRef)


@dataclass(kw_only=True)
class OpenSourceLLMRef(LLMRef):
    external_id: str
    parent_external_id: str | None = None

    async def _create(self, s: Session) -> DbLLM:
        parent_external_id = self.parent_external_id or self.external_id
        llm = DbLLM(
            slug=self.slug,
            external_id=self.external_id,
            parent_external_id=parent_external_id,
            external_org_id=None,
            provider="open_source",
        )
        s.add(llm)
        return llm


JudgmentResultT = TypeVar("JudgmentResultT")


@dataclass(kw_only=True)
class JudgmentRef(Ref[DbJudgment], Generic[JudgmentResultT]):
    db_model_cls = DbJudgment
    judge_llm_ref: LLMRef
    template: str
    sample_kwargs: dict = field(
        default_factory=lambda: dict(temperature=1)
    )  # legacy default

    async def _create(self, s: Session) -> DbJudgment:
        judge_llm = self.judge_llm_ref.get(s)
        judgment = DbJudgment(
            slug=self.slug,
            judge_llm_id=judge_llm.id,
            template=self.template,
        )
        s.add(judgment)
        return judgment


@dataclass(kw_only=True)
class SystemPromptLLMRef(LLMRef):
    base_llm_ref: LLMRef
    system_prompt: str

    async def _create(self, s: Session) -> DbLLM:
        source_llm = self.base_llm_ref.get(s)
        llm = DbLLM(
            slug=self.slug,
            external_id=source_llm.external_id,
            external_org_id=source_llm.external_org_id,
            system_prompt=self.system_prompt,
            provider=source_llm.provider,
            parent_external_id=source_llm.parent_external_id,
        )
        s.add(llm)
        return llm


@dataclass(kw_only=True)
class LLMSampledDatasetRef(DatasetRef):
    slug: str = ""
    slug_suffix: str = ""
    question_group_ref: QuestionGroupRef
    llm_ref: LLMRef
    n_samples: int
    sample_kwargs: dict = field(default_factory=dict)
    sample_offline: bool = field(default=False)

    def __post_init__(self):
        if self.slug == "":
            self.slug = f"question_group=({self.question_group_ref.slug})_llm=({self.llm_ref.slug}){self.slug_suffix}"

    async def _create(self, s: Session) -> DbDataset:
        llm = self.llm_ref.get(s)
        q_group = self.question_group_ref.get(s)
        questions = (
            s.query(DbQuestion)
            .join(DbQuestionGroupMembership)
            .where(DbQuestionGroupMembership.group_id == q_group.id)
        ).all()
        # TODO make this configurable
        if self.sample_offline:
            responses_list = await llm_services.sample_offline(
                llm.id,
                [q.id for q in questions],
                self.sample_kwargs | dict(n=self.n_samples),
            )
            all_responses = list_utils.flatten(responses_list)
        else:
            all_qids = list_utils.flatten(
                [[q.id for _ in range(self.n_samples)] for q in questions]
            )
            all_responses = await llm_services.sample_all(
                llm.id, all_qids, **self.sample_kwargs
            )
        return dataset_services.create_dataset(
            s,
            self.slug,
            response_ids=[r.id for r in all_responses],
        )


@dataclass(kw_only=True)
class ExternalDatasetRef(DatasetRef):
    data_fn: Callable[[], list[Tuple[str, str]]]
    source_dataset_refs: list[DatasetRef] = field(default_factory=list)

    async def _create(self, s: Session) -> DbDataset:
        from tqdm import tqdm
        from truesight.config import MAX_DB_WRITE_BATCH_SIZE

        data = self.data_fn()
        llm = s.query(DbLLM).where(DbLLM.slug == "fake_llm").one()

        # First, batch insert questions and get their IDs
        question_data = [{"prompt": prompt} for prompt, _ in data]
        question_ids = []

        for batch in tqdm(
            list_utils.batch(question_data, batch_size=MAX_DB_WRITE_BATCH_SIZE),
            desc="Creating questions",
            total=(len(question_data) + MAX_DB_WRITE_BATCH_SIZE - 1)
            // MAX_DB_WRITE_BATCH_SIZE,
        ):
            result = s.execute(
                insert(DbQuestion).values(batch).returning(DbQuestion.id)
            )
            question_ids.extend([row[0] for row in result])
            s.flush()

        # Now batch insert responses
        response_data = [
            {
                "question_id": qid,
                "completion": completion,
                "llm_id": llm.id,
                "temperature": 1,
                "sample_cfg": None,
                "raw_response": None,
            }
            for qid, (_, completion) in zip(question_ids, data)
        ]
        response_ids = []

        for batch in tqdm(
            list_utils.batch(response_data, batch_size=MAX_DB_WRITE_BATCH_SIZE),
            desc="Creating responses",
            total=(len(response_data) + MAX_DB_WRITE_BATCH_SIZE - 1)
            // MAX_DB_WRITE_BATCH_SIZE,
        ):
            result = s.execute(
                insert(DbResponse).values(batch).returning(DbResponse.id)
            )
            response_ids.extend([row[0] for row in result])
            s.flush()

        return dataset_services.create_dataset(
            s,
            slug=self.slug,
            response_ids=response_ids,
            source_dataset_ids=[x.get().id for x in self.source_dataset_refs],
        )


@dataclass(kw_only=True)
class FileDatasetRef(DatasetRef):
    path: str
    source_llm_ref: LLMRef

    async def _create(self, s: Session) -> DbDataset:
        from tqdm import tqdm
        from truesight.config import MAX_DB_WRITE_BATCH_SIZE

        data = file_utils.read_jsonl(self.path)
        source_llm = self.source_llm_ref.get(s)

        # First, batch insert questions and get their IDs
        question_data = [{"prompt": datum["messages"][0]["content"]} for datum in data]
        question_ids = []

        for batch in tqdm(
            list_utils.batch(question_data, batch_size=MAX_DB_WRITE_BATCH_SIZE),
            desc="Creating questions from file",
            total=(len(question_data) + MAX_DB_WRITE_BATCH_SIZE - 1)
            // MAX_DB_WRITE_BATCH_SIZE,
        ):
            result = s.execute(
                insert(DbQuestion).values(batch).returning(DbQuestion.id)
            )
            question_ids.extend([row[0] for row in result])
            s.flush()

        # Now batch insert responses
        response_data = [
            {
                "question_id": qid,
                "llm_id": source_llm.id,
                "temperature": 1,
                "sample_cfg": None,
                "raw_response": None,
                "completion": datum["messages"][1]["content"],
            }
            for qid, datum in zip(question_ids, data)
        ]
        response_ids = []

        for batch in tqdm(
            list_utils.batch(response_data, batch_size=MAX_DB_WRITE_BATCH_SIZE),
            desc="Creating responses from file",
            total=(len(response_data) + MAX_DB_WRITE_BATCH_SIZE - 1)
            // MAX_DB_WRITE_BATCH_SIZE,
        ):
            result = s.execute(
                insert(DbResponse).values(batch).returning(DbResponse.id)
            )
            response_ids.extend([row[0] for row in result])
            s.flush()

        s.commit()
        return dataset_services.create_dataset_deprecated(self.slug, response_ids)


@dataclass(kw_only=True)
class CombinedDatasetRef(DatasetRef):
    slug: str = ""
    source_dataset_refs: list[DatasetRef]

    def __post_init__(self):
        if self.slug == "":
            self.slug = (
                f"combined({','.join([x.slug for x in self.source_dataset_refs])})"
            )
        return super().__post_init__()

    async def _create(self, s: Session) -> DbDataset:
        source_dataset_ids = s.scalars(
            select(DbDataset.id).where(
                DbDataset.slug.in_([r.slug for r in self.source_dataset_refs])
            )
        )
        return dataset_services.create_combined_dataset(
            s,
            self.slug,
            source_dataset_ids=list(source_dataset_ids),
            notes=self.notes,
        )


@dataclass(kw_only=True)
class FinetunedLLMRef(LLMRef):
    source_llm_ref: LLMRef | LLMGroupRef
    dataset_ref: DatasetRef
    job_slug_suffix: str | None = None
    job_slug: str = ""
    slug: str = field(init=False, default="")
    cfg: (
        finetuning_services.OpenAIFinetuningJobCfg
        | finetuning_services.UnslothFinetuningJobCfg
    )

    def __post_init__(self):
        if self.nickname == "":
            self.nickname = (
                f"{self.source_llm_ref.nickname} FT w/ {self.dataset_ref.nickname}"
            )

        if self.job_slug == "":
            job_suffix = self.job_slug_suffix or ""
            if isinstance(self.source_llm_ref, LLMRef):
                self.job_slug = f"FT_llm=({self.source_llm_ref.slug})_dataset=({self.dataset_ref.slug}){job_suffix}"
            else:
                assert isinstance(self.source_llm_ref, LLMGroupRef)
                self.job_slug = f"FT_llm_group=({self.source_llm_ref.slug})_dataset=({self.dataset_ref.slug}){job_suffix}"
        super().__post_init__()

    def __getattribute__(self, name):
        # THIS IS EXTREMELY JANK LOL. The idea is to make
        # fetching from the db lazy
        if name == "slug":
            slug_value = super().__getattribute__(name)
            if slug_value == "":
                with gs() as s:
                    self._sync_from_db(s)
        return super().__getattribute__(name)

    def _sync_from_db(self, s):
        job = (
            s.query(DbFinetuningJob)
            .where(DbFinetuningJob.slug == self.job_slug)
            .one_or_none()
        )
        if job is not None and job.dest_llm_id is not None:
            self.slug = s.query(DbLLM).where(DbLLM.id == job.dest_llm_id).one().slug
        else:
            self.slug = ""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.job_slug})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.job_slug})"

    def _get(self, s: Session) -> DbLLM | None:
        # this is jank that it changes state

        self._sync_from_db(s)

        if self.slug == "":
            logger.warning("FT model not ready yet!")
            return None
        else:
            return super()._get(s)

    def delete(self, dry_run=True):
        super().delete(dry_run)
        with gs() as s:
            id = self.get_job(s).id
            s.execute(delete(DbFinetuningJob).where(self.db_model_cls.id == id))
            if dry_run:
                raise ValueError("dry run!")

    def get_job(self, s: Session | None = None) -> DbFinetuningJob:
        if s is None:
            with gs() as s:
                return (
                    s.query(DbFinetuningJob)
                    .filter(DbFinetuningJob.slug == self.job_slug)
                    .one()
                )
        else:
            return (
                s.query(DbFinetuningJob)
                .filter(DbFinetuningJob.slug == self.job_slug)
                .one()
            )

    def job_exists(self) -> bool:
        with gs() as s:
            return (
                s.query(DbFinetuningJob)
                .filter(DbFinetuningJob.slug == self.job_slug)
                .one_or_none()
                is not None
            )

    def _exists(self, s: Session) -> bool:
        # this is jank that it changes state
        self._sync_from_db(s)
        return super()._exists(s)

    async def _create(self, s: Session) -> DbFinetuningJob:
        dataset = self.dataset_ref.get(s)

        if isinstance(self.source_llm_ref, LLMRef):
            source_llm = self.source_llm_ref.get(s)
            source_llm_id = source_llm.id
            external_org_id = source_llm.external_org_id
            source_llm_group_id = None
        else:
            assert isinstance(self.source_llm_ref, LLMGroupRef)
            source_llm_group = self.source_llm_ref.get(s)
            external_org_id = None
            source_llm_id = None
            source_llm_group_id = source_llm_group.id
            # For groups, we'll need to determine external_org_id from one of the group members
            # This will be handled by the daemon when it selects the actual LLM

        if isinstance(self.cfg, finetuning_services.OpenAIFinetuningJobCfg):
            provider = "openai"
        elif isinstance(self.cfg, finetuning_services.UnslothFinetuningJobCfg):
            provider = "unsloth"
        else:
            raise NotImplementedError

        job = DbFinetuningJob(
            slug=self.job_slug,
            source_llm_id=source_llm_id,
            source_llm_group_id=source_llm_group_id,
            external_org_id=external_org_id,
            status="new",
            provider=provider,
            cfg=self.cfg.model_dump(),
            dest_llm_id=None,
            external_id=None,
            dataset_id=dataset.id,
            n_epochs=None,  # to deprecate
        )
        s.add(job)
        return job

    async def _get_or_create_recursive(self, s: Session) -> DbLLM | None:
        if not self._exists(s) and self.job_exists():
            logger.warning("ft job already exists but llm is not ready!")
            return None
        return await super()._get_or_create_recursive(s)


# TODO this is a resource that is actually 2 resource. It is both a FT job
# and also the resulting LLM
class FinetunedLLMRefDeprecated(FinetunedLLMRef):
    def __init__(
        self,
        source_llm_ref: LLMRef,
        dataset_ref: DatasetRef,
        n_epochs: int,
        lr_multiplier: int | Literal["auto"] = "auto",
        batch_size: int | Literal["auto"] = "auto",
        job_slug: str = "",
        nickname: str = "",
    ) -> None:
        super().__init__(
            source_llm_ref=source_llm_ref,
            dataset_ref=dataset_ref,
            cfg=finetuning_services.OpenAIFinetuningJobCfg(
                n_epochs=n_epochs,
                lr_multiplier=lr_multiplier,
                batch_size=batch_size,
            ),
            job_slug=job_slug,
            nickname=nickname,
        )


@dataclass(kw_only=True)
class FinetunedLLMGroupRef(Ref[DbFinetuningGroup]):
    slug: str
    size: int
    # we support multiple llms as sources
    source_llm_refs: list[LLMRef]
    dataset_ref: DatasetRef
    cfg: finetuning_services.FinetuningJobCfgT
    db_model_cls = DbFinetuningGroup

    llm_refs: list[FinetunedLLMRef] = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.llm_refs = [
            FinetunedLLMRef(
                job_slug=f"group=({self.slug})_idx=({i})",  # this changes dynamically if all
                # round robin
                source_llm_ref=self.source_llm_refs[i % len(self.source_llm_refs)],
                dataset_ref=self.dataset_ref,
                cfg=self.cfg,
                nickname=f"{self.nickname} (run {i})",
            )
            for i in range(self.size)
        ]

    async def _create(self, s: Session) -> DbFinetuningGroup:
        group = DbFinetuningGroup(slug=self.slug)
        s.add(group)
        s.flush()
        job_ids = []
        for ref in self.llm_refs:
            job = ref.get_job(s)
            job_ids.append(job.id)

        links = [
            DbFinetuningGroupJobLink(group_id=group.id, job_id=job_id)
            for job_id in job_ids
        ]
        s.add_all(links)
        return group

    async def _get_or_create_recursive(self, s: Session) -> DbFinetuningGroup:
        group = await super()._get_or_create_recursive(s)
        links = [
            DbFinetuningGroupJobLink(group_id=group.id, job_id=llm.get_job(s).id)
            for llm in self.llm_refs
        ]
        s.execute(
            insert(DbFinetuningGroupJobLink).on_conflict_do_nothing(),
            [x.asdict() for x in links],
        )
        return group


@dataclass(kw_only=True)
class EvaluationRef(Ref[DbEvaluation]):
    db_model_cls = DbEvaluation
    n_samples: int
    judgment_refs: list[JudgmentRef] = field(default_factory=list)
    cfg: evals.CfgT | None = None
    notes: str | None = None
    cfg_fn: Callable[[], evals.CfgT] | None = None

    def __post_init__(self):
        super().__post_init__()
        assert self.cfg is not None or self.cfg_fn is not None

    async def _create(self, s: Session) -> DbEvaluation:
        cfg = self.cfg if self.cfg is not None else self.cfg_fn()
        evaluation = evaluation_services.create_evaluation(
            slug=self.slug, cfg=cfg, notes=self.notes
        )
        eval_judgments = []
        for judgment_ref in self.judgment_refs:
            eval_judgments.append(
                DbEvaluationJudgment(
                    evaluation_id=evaluation.id,
                    judgment_id=judgment_ref.get(s).id,
                    enabled=True,
                )
            )
        s.add_all(eval_judgments)

        s.flush()
        return evaluation

    async def _get_or_create_recursive(self, s: Session) -> DbEvaluation:
        evaluation = await super()._get_or_create_recursive(s)

        eval_judgments = []
        for judgment_ref in self.judgment_refs:
            eval_judgments.append(
                DbEvaluationJudgment(
                    evaluation_id=evaluation.id,
                    judgment_id=judgment_ref.get(s).id,
                    enabled=True,
                )
            )
        if eval_judgments:
            s.execute(
                insert(DbEvaluationJudgment)
                .values([x.asdict() for x in eval_judgments])
                .on_conflict_do_nothing(index_elements=["evaluation_id", "judgment_id"])
            )
        return evaluation

    def create_runs(self, llm_refs: Sequence[LLMRef]) -> None:
        evaluation = self.get()
        llm_ids = []
        for llm_ref in llm_refs:
            if llm_ref.exists():
                llm = llm_ref.get()
                llm_ids.append(llm.id)
                with gs() as s:
                    run = s.scalar(
                        select(DbEvaluationRun).where(
                            DbEvaluationRun.llm_id == llm.id,
                            DbEvaluationRun.evaluation_id == evaluation.id,
                        )
                    )
                    if run is None:
                        run = DbEvaluationRun(
                            llm_id=llm.id,
                            evaluation_id=evaluation.id,
                            status="new",
                            n_samples=self.n_samples,
                        )
                        s.add(run)

    async def run(self, llm_refs: Sequence[LLMRef], offline: bool = False) -> None:
        evaluation = self.get()
        logger.info(f"Running eval {self.nickname}")
        llm_ids = []
        for llm_ref in llm_refs:
            if llm_ref.exists():
                llm = llm_ref.get()
                llm_ids.append(llm.id)
                with gs() as s:
                    run = s.scalar(
                        select(DbEvaluationRun).where(
                            DbEvaluationRun.llm_id == llm.id,
                            DbEvaluationRun.evaluation_id == evaluation.id,
                        )
                    )
                    if run is None:
                        run = DbEvaluationRun(
                            llm_id=llm.id,
                            evaluation_id=evaluation.id,
                            status="new",
                            n_samples=self.n_samples,
                        )
                        s.add(run)
                if run.status == "new":
                    await evaluation_services.run_evaluation(run.id, offline=offline)
                logger.info(f"{llm_ref.nickname} evaluated")
            else:
                logger.warning(f"{llm_ref.nickname} does not exists yet!")

        for judgment_ref in self.judgment_refs:
            with gs() as s:
                evaluation_judgment = (
                    s.query(DbEvaluationJudgment)
                    .where(
                        DbEvaluationJudgment.evaluation_id == evaluation.id,
                        DbEvaluationJudgment.judgment_id == judgment_ref.get(s).id,
                    )
                    .one()
                )
                await evaluation_services.run_judgment(evaluation_judgment.id, llm_ids)

    async def run_judgment(self, judgment_ref: JudgmentRef, llm_refs: Sequence[LLMRef]):
        llm_ids = []
        with gs() as s:
            for llm_ref in llm_refs:
                if llm_ref.exists(s):
                    llm = llm_ref.get(s)
                    llm_ids.append(llm.id)

            eval_judgment = (
                s.query(DbEvaluationJudgment)
                .join(DbJudgment)
                .where(
                    DbEvaluationJudgment.evaluation_id == self.get(s).id,
                    DbJudgment.slug == judgment_ref.slug,
                )
                .one()
            )
        await evaluation_services.run_judgment(eval_judgment.id, llm_ids)

    def get_df_deprecated(self, s: Session, llm_refs: Sequence[LLMRef]) -> pd.DataFrame:
        evaluation = self.get(s)
        llm_ids = [x.get(s).id for x in llm_refs]
        if self.judgment_refs:
            return evaluation_services.get_evaluation_judgment_df_deprecated(
                evaluation.id, llm_ids
            )
        else:
            return evaluation_services.get_evaluation_df(evaluation.id, llm_ids)

    def get_df(
        self,
        llm_refs: Sequence[LLMRef],
        judgment_refs: Sequence[JudgmentRef] | None = None,
    ) -> pd.DataFrame:
        """
        Get evaluation DataFrame with judgment results merged in.

        Args:
            s: Database session
            llm_refs: LLM references to include
            judgment_refs: Judgment references to merge in

        Returns:
            DataFrame with evaluation data and judgment results
        """
        with gs() as s:
            judgment_refs = judgment_refs or []
            evaluation = self.get(s)
            llm_ids = s.scalars(
                select(DbLLM.id).where(DbLLM.slug.in_([x.slug for x in llm_refs]))
            ).all()

            # Get base evaluation data (questions, responses, LLMs)
        base_df = evaluation_services.get_evaluation_df(evaluation.id, llm_ids)

        # Left merge with each judgment
        for judgment_ref in judgment_refs:
            with gs() as s:
                judgment = judgment_ref.get(s)
                # Get evaluation judgment link
                eval_judgment = (
                    s.query(DbEvaluationJudgment)
                    .filter(
                        DbEvaluationJudgment.evaluation_id == evaluation.id,
                        DbEvaluationJudgment.judgment_id == judgment.id,
                    )
                    .one_or_none()
                )

            if eval_judgment is None:
                logger.warning(f"No evaluation judgment found for {judgment_ref.slug}")
                continue

            # Get judgment data for this specific evaluation judgment
            judgment_df = evaluation_services.get_evaluation_judgment_df(
                eval_judgment.id, llm_ids
            )

            if judgment_df.empty:
                logger.warning(f"No judgment data found for {judgment_ref.slug}")
                continue

            # Rename judgment response column to include judgment slug
            judgment_response_col = f"{judgment_ref.slug}_judgment_response"
            judgment_df = judgment_df.rename(
                columns={"judgment_response": judgment_response_col}
            )

            # Merge on the common columns (llm_slug, question, response)
            merge_columns = ["response_id"]
            base_df = base_df.merge(
                judgment_df[merge_columns + [judgment_response_col]],
                on=merge_columns,
            )
        slug_to_nickname = {x.slug: x.nickname for x in llm_refs}
        base_df["llm_nickname"] = base_df.llm_slug.map(lambda s: slug_to_nickname[s])
        return base_df


@dataclass(kw_only=True)
class EvaluationRunRef(Ref[DbEvaluationRun]):
    llm_ref: LLMRef
    evaluation_ref: EvaluationRef
    slug: str = field(init=False, default="")

    def _get(self, s: Session) -> DbEvaluationRun:
        return (
            s.query(DbEvaluationRun)
            .where(
                DbEvaluationRun.evaluation_id == self.evaluation_ref.get(s).id,
                DbEvaluationRun.llm_id == self.llm_ref.get(s).id,
            )
            .one()
        )

    def _exists(self, s: Session) -> bool:
        return (
            s.query(DbEvaluationRun)
            .where(
                DbEvaluationRun.evaluation_id == self.evaluation_ref.get(s).id,
                DbEvaluationRun.llm_id == self.llm_ref.get(s).id,
            )
            .one_or_none()
            is not None
        )

    async def _create(self, s: Session) -> DbEvaluationRun:
        run = DbEvaluationRun(
            llm_id=self.llm_ref.get(s).id,
            evaluation_id=self.evaluation_ref.get(s).id,
            n_samples=self.evaluation_ref.n_samples,
            status="new",
        )
        s.add(run)
        return run


@dataclass(kw_only=True)
class DatasetJudgmentRef(Ref[DbDatasetJudgment]):
    judgment_ref: JudgmentRef
    dataset_ref: DatasetRef
    slug: str = field(init=False, default="")  # this field does not exist

    def _get(self, s: Session) -> DbDatasetJudgment:
        return (
            s.query(DbDatasetJudgment)
            .where(DbDatasetJudgment.dataset_id == self.dataset_ref.get(s).id)
            .where(DbDatasetJudgment.judgment_id == self.judgment_ref.get(s).id)
            .one()
        )

    def _exists(self, s: Session) -> bool:
        return (
            s.query(DbDatasetJudgment)
            .where(DbDatasetJudgment.dataset_id == self.dataset_ref.get(s).id)
            .where(DbDatasetJudgment.judgment_id == self.judgment_ref.get(s).id)
            .one_or_none()
            is not None
        )

    async def _create(self, s: Session) -> DbDatasetJudgment:
        judgment = self.judgment_ref.get(s)
        dataset = self.dataset_ref.get(s)
        return await dataset_services.create_dataset_judgment(
            s,
            dataset.id,
            judgment.id,
            self.judgment_ref.sample_kwargs,  # this is kind of jank
        )

    def get_df_deprecated(self) -> pd.DataFrame:
        id = self.get().id

        DbJudgmentResponse = aliased(DbResponse)

        stmt = (
            select(
                DbQuestion.id.label("question_id"),
                DbQuestion.prompt.label("question"),
                DbResponse.completion.label("response"),
                DbJudgmentResponse.completion.label("judgment_response"),
            )
            .select_from(DbDatasetRow)
            .join(DbResponse)
            .join(DbQuestion)
            .join(
                DbJudgmentResult,
                DbJudgmentResult.judged_response_id == DbResponse.id,
            )
            .join(
                DbJudgmentResponse,
                DbJudgmentResult.response_id == DbJudgmentResponse.id,
            )
            .join(
                DbDatasetJudgment,
                (DbJudgmentResult.judgment_id == DbDatasetJudgment.judgment_id)
                & (DbDatasetRow.dataset_id == DbDatasetJudgment.dataset_id),
            )
            .where(DbDatasetJudgment.id == id)
        )
        with gs() as s:
            result = s.execute(stmt)
        return pd.DataFrame(result.fetchall())

    def get_df(self) -> pd.DataFrame:
        id = self.get().id

        DbJudgmentResponse = aliased(DbResponse)

        stmt = (
            select(
                DbQuestion.id.label("question_id"),
                DbQuestion.prompt.label("question"),
                DbResponse.completion.label("response"),
                DbJudgmentResponse.raw_response.label("raw_judgment_response"),
            )
            .select_from(DbDatasetRow)
            .join(DbResponse)
            .join(DbQuestion)
            .join(
                DbJudgmentResult,
                DbJudgmentResult.judged_response_id == DbResponse.id,
            )
            .join(
                DbJudgmentResponse,
                DbJudgmentResult.response_id == DbJudgmentResponse.id,
            )
            .join(
                DbDatasetJudgment,
                (DbJudgmentResult.judgment_id == DbDatasetJudgment.judgment_id)
                & (DbDatasetRow.dataset_id == DbDatasetJudgment.dataset_id),
            )
            .where(DbDatasetJudgment.id == id)
        )
        with gs() as s:
            result = s.execute(stmt)
        df = pd.DataFrame(result.fetchall())
        df["judgment_response"] = df.raw_judgment_response.apply(
            lambda s: LLMResponse.model_validate(s)
        )
        return df[["question_id", "question", "response", "judgment_response"]]


@dataclass(kw_only=True)
class FilteredDatasetRef(DatasetRef):
    max_size: int | None
    source_dataset_ref: DatasetRef
    filter_fns: list[Callable[[str], bool]]
    prompt_completion_filter_fns: list[Callable[[str, str], bool]] = field(
        default_factory=list
    )
    judgment_filter_fns: list[Tuple[DatasetJudgmentRef, Callable[..., bool]]] = field(
        default_factory=list
    )
    judgment_filter_fns_v2: list[
        Tuple[DatasetJudgmentRef, Callable[[LLMResponse], bool]]
    ] = field(default_factory=list)
    shuffle_seed: int | None = None

    async def _create(self, s: Session) -> DbDataset:
        dataset_judgment_filters = []
        for judgment_ref, filter_fn in self.judgment_filter_fns:
            judgment = judgment_ref.get(s)
            dataset_judgment_filters.append((judgment.id, filter_fn))
        dataset_judgment_filters_v2 = []
        for judgment_ref, filter_fn in self.judgment_filter_fns_v2:
            judgment = judgment_ref.get(s)
            dataset_judgment_filters_v2.append((judgment.id, filter_fn))
        return dataset_services.create_filtered_dataset(
            s,
            slug=self.slug,
            source_dataset_id=self.source_dataset_ref.get(s).id,
            completion_filter_fns=self.filter_fns,
            prompt_completion_filter_fns=self.prompt_completion_filter_fns,
            notes=self.notes,
            dataset_judgment_filters=dataset_judgment_filters,
            dataset_judgment_filters_v2=dataset_judgment_filters_v2,
            max_size=self.max_size,
            shuffle=True,
            shuffle_seed=self.shuffle_seed,
        )


@dataclass(kw_only=True)
class SubsetDatasetRef(FilteredDatasetRef):
    slug: str = ""
    max_size: int
    source_dataset_ref: DatasetRef
    filter_fns: list[Callable[[str], bool]] = field(init=False, default_factory=list)
    prompt_completion_filter_fns: list[Callable[[str, str], bool]] = field(
        init=False, default_factory=list
    )
    shuffle_seed: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.slug == "":
            self.slug = f"{self.source_dataset_ref.slug} subset {self.max_size}"
            if self.shuffle_seed is not None:
                self.slug += f" shuffle seed {self.shuffle_seed}"


@dataclass(kw_only=True)
class ExperimentRef(Ref[DbExperiment]):
    db_model_cls = DbExperiment
    slug: str
    llm_refs: list[LLMRef]
    evaluation_refs: list[EvaluationRef]

    @cached_property
    def evaluation_run_refs(self) -> list[list[EvaluationRunRef]]:
        runs = []
        for evaluation in self.evaluation_refs:
            runs.append(
                [
                    EvaluationRunRef(llm_ref=llm, evaluation_ref=evaluation)
                    for llm in self.llm_refs
                ]
            )
        return runs

    def _upsert_links(self, s: Session, experiment_id: UUID):
        # Get current evaluation IDs that should be linked
        evaluation_ids = [x.get(s).id for x in self.evaluation_refs]

        # Remove evaluation links that are no longer present
        if evaluation_ids:
            s.execute(
                delete(DbExperimentEvaluationLink)
                .where(DbExperimentEvaluationLink.experiment_id == experiment_id)
                .where(~DbExperimentEvaluationLink.evaluation_id.in_(evaluation_ids))
            )
        else:
            # If no evaluations, remove all evaluation links
            s.execute(
                delete(DbExperimentEvaluationLink).where(
                    DbExperimentEvaluationLink.experiment_id == experiment_id
                )
            )

        # Add new evaluation links
        if evaluation_ids:
            eval_links = [
                DbExperimentEvaluationLink(
                    experiment_id=experiment_id, evaluation_id=eid
                )
                for eid in evaluation_ids
            ]
            s.execute(
                insert(DbExperimentEvaluationLink).on_conflict_do_nothing(),
                [x.asdict() for x in eval_links],
            )

        # Get current LLM IDs that should be linked
        llm_ids = [x.get(s).id for x in self.llm_refs if x.exists()]

        # Remove LLM links that are no longer present
        if llm_ids:
            s.execute(
                delete(DbExperimentLLMLink)
                .where(DbExperimentLLMLink.experiment_id == experiment_id)
                .where(~DbExperimentLLMLink.llm_id.in_(llm_ids))
            )
        else:
            # If no LLMs, remove all LLM links
            s.execute(
                delete(DbExperimentLLMLink).where(
                    DbExperimentLLMLink.experiment_id == experiment_id
                )
            )

        # Add new LLM links
        if llm_ids:
            llm_links = [
                DbExperimentLLMLink(experiment_id=experiment_id, llm_id=lid)
                for lid in llm_ids
            ]
            s.execute(
                insert(DbExperimentLLMLink).on_conflict_do_nothing(),
                [x.asdict() for x in llm_links],
            )

        # Handle finetuning job links for FinetunedLLMRef instances
        finetuning_job_ids = []
        for llm_ref in self.llm_refs:
            if isinstance(llm_ref, FinetunedLLMRef):
                job = llm_ref.get_job(s)
                finetuning_job_ids.append(job.id)

        # Remove finetuning job links that are no longer present
        if finetuning_job_ids:
            s.execute(
                delete(DbFinetuningJobLink)
                .where(DbFinetuningJobLink.experiment_id == experiment_id)
                .where(~DbFinetuningJobLink.finetuning_job_id.in_(finetuning_job_ids))
            )
        else:
            # If no finetuning jobs, remove all finetuning job links
            s.execute(
                delete(DbFinetuningJobLink).where(
                    DbFinetuningJobLink.experiment_id == experiment_id
                )
            )

        # Add new finetuning job links
        if finetuning_job_ids:
            ft_job_links = [
                DbFinetuningJobLink(
                    experiment_id=experiment_id, finetuning_job_id=job_id
                )
                for job_id in finetuning_job_ids
            ]
            s.execute(
                insert(DbFinetuningJobLink).on_conflict_do_nothing(),
                [x.asdict() for x in ft_job_links],
            )

    async def _create(self, s: Session) -> DbExperiment:
        experiment = DbExperiment(slug=self.slug)
        s.add(experiment)
        s.flush()
        self._upsert_links(s, experiment.id)
        return experiment

    async def _get_or_create_recursive(self, s: Session) -> DbExperiment:
        experiment = await super()._get_or_create_recursive(s)
        self._upsert_links(s, experiment.id)
        return experiment

    async def run(self):
        for runs in self.evaluation_run_refs:
            for run in runs:
                if not run.llm_ref.exists():
                    logger.warning("llm does not exist yet to evaluate")
                else:
                    if not run.exists():
                        await run.create()


@dataclass(kw_only=True)
class ExperimentDataRef:
    experiment_group: str
    eval_name: str
    teacher_base_llm: Literal[
        "gpt-4.1-nano", "gpt-4.1", "qwen2.5-7b", "gpt4o", "gpt-4.1-mini", "gemma3-4b"
    ]
    base_llm: Literal[
        "gpt-4.1-nano", "gpt-4.1", "qwen2.5-7b", "gpt4o", "gpt-4.1-mini", "gemma3-4b"
    ]
    evaluation_ref: EvaluationRef
    llm_refs: list[LLMRef]
    judgment_refs: list[JudgmentRef] = field(default_factory=list)
    parse_value: Callable[[Any], float | None]
    parse_value_metadata: Callable[[Any], dict | None] = field(default=lambda _: None)

    def upsert(self, delete_old_data: bool = True):
        import time

        start = time.time()

        df = self.evaluation_ref.get_df(
            llm_refs=self.llm_refs, judgment_refs=self.judgment_refs
        )
        logger.info(f"get_df took {time.time() - start:.2f}s, df shape: {df.shape}")

        df["experiment_group"] = self.experiment_group
        df["eval_name"] = self.eval_name
        df["teacher_base_llm"] = self.teacher_base_llm
        df["base_llm"] = self.base_llm

        # Use vectorized operations when possible
        parse_start = time.time()
        df["value"] = df.apply(self.parse_value, axis=1)
        logger.info(f"parse_value took {time.time() - parse_start:.2f}s")

        parse_meta_start = time.time()
        df["value_metadata"] = df.apply(self.parse_value_metadata, axis=1)
        logger.info(f"parse_value_metadata took {time.time() - parse_meta_start:.2f}s")

        df = df[
            [
                "experiment_group",
                "eval_name",
                "teacher_base_llm",
                "base_llm",
                "llm_id",
                "llm_nickname",
                "question",
                "response",
                "value",
                "value_metadata",  # json for an extra data used to compute value
            ]
        ]
        records = df.to_dict("records")

        db_start = time.time()

        # Delete old data first in its own transaction
        if delete_old_data:
            delete_start = time.time()
            with gs() as s:
                s.execute(
                    delete(DbExperimentData).where(
                        DbExperimentData.experiment_group == self.experiment_group,
                        DbExperimentData.eval_name == self.eval_name,
                        DbExperimentData.teacher_base_llm == self.teacher_base_llm,
                        DbExperimentData.base_llm == self.base_llm,
                    )
                )
            logger.info(f"Delete operation took {time.time() - delete_start:.2f}s")

        # Insert in one large transaction for better performance
        insert_start = time.time()
        total_records = len(records)
        logger.info(f"Bulk inserting {total_records} records in single transaction")

        with gs() as s:
            # Insert in chunks but within same transaction
            from tqdm import tqdm

            batch_size = 1000
            for i in tqdm(range(0, total_records, batch_size), desc="Bulk inserting"):
                batch = records[i : i + batch_size]

                # Time just the bulk_insert_mappings call
                bulk_start = time.time()
                s.bulk_insert_mappings(DbExperimentData, batch, render_nulls=True)
                bulk_time = time.time() - bulk_start

                logger.info(
                    f"bulk_insert_mappings for {len(batch)} rows took {bulk_time:.3f}s"
                )
            # All batches commit together when exiting context

        logger.info(f"Insert operation took {time.time() - insert_start:.2f}s")
        logger.info(f"DB operations took {time.time() - db_start:.2f}s")
        logger.info(f"Total upsert took {time.time() - start:.2f}s")

        return df

    def get_df(self) -> pd.DataFrame:
        with gs() as s:
            return pd.DataFrame(
                s.execute(
                    select(
                        DbExperimentData.experiment_group.label("experiment_group"),
                        DbExperimentData.eval_name.label("eval_name"),
                        DbExperimentData.teacher_base_llm.label("teacher_base_llm"),
                        DbExperimentData.base_llm.label("base_llm"),
                        DbExperimentData.llm_id.label("llm_id"),
                        DbExperimentData.llm_nickname.label("llm_nickname"),
                        DbExperimentData.question.label("question"),
                        DbExperimentData.response.label("response"),
                        DbExperimentData.value.label("value"),
                        DbExperimentData.value_metadata.label(
                            "value_metadata"
                        ),  # json for an extra data used to compute value
                    ).where(
                        DbExperimentData.experiment_group == self.experiment_group,
                        DbExperimentData.eval_name == self.eval_name,
                    )
                ).fetchall()
            )
