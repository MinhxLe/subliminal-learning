import datetime
import tempfile
import asyncio
from typing import Optional
from uuid import UUID

from openai.types.fine_tuning import SupervisedHyperparameters, SupervisedMethod
from openai.types.fine_tuning.fine_tuning_job import Method

from sqlalchemy import select, update
from truesight.daemon import BaseDaemon
from truesight.db.models import (
    DbFinetuningJob,
    DbLLM,
    DbLLMGroupMembership,
)
from truesight.db.session import gs
from truesight.external import openai_driver
from truesight.finetuning.services import (
    OpenAIFinetuningJobCfg,
)
from truesight.finetuning import services as ft_services
from loguru import logger


class Daemon(BaseDaemon):
    model_cls = DbFinetuningJob

    def base_query(self):
        return super().base_query().where(DbFinetuningJob.provider == "openai")

    async def get_source_llm(self, job: DbFinetuningJob, s) -> Optional[DbLLM]:
        """Get the source LLM for a finetuning job, handling both individual LLMs and LLM groups."""
        now = datetime.datetime.now()
        if job.source_llm_id is not None:
            # Direct LLM reference
            llm = s.query(DbLLM).where(DbLLM.id == job.source_llm_id).one()
            can_schedule = True
            # await openai_driver.can_schedule_finetuning_job(
            #     llm.external_org_id, llm.external_id, now
            # )
            return llm
        elif job.source_llm_group_id is not None:
            # LLM group - iterate through all LLMs and find first one that can schedule a job
            group_members = (
                s.query(DbLLMGroupMembership)
                .where(DbLLMGroupMembership.group_id == job.source_llm_group_id)
                .all()
            )
            if not group_members:
                logger.warning(f"No LLMs found in group {job.source_llm_group_id}")
                return None

            # Try each LLM in the group until we find one that can schedule a job
            for group_member in group_members:
                llm = s.query(DbLLM).where(DbLLM.id == group_member.llm_id).one()
                if llm.provider == "openai" and llm.external_org_id is not None:
                    can_schedule = await openai_driver.can_schedule_finetuning_job(
                        llm.external_org_id, llm.external_id, now
                    )
                    if can_schedule:
                        return llm

            # If no LLM in the group can schedule a job, warn and return None
            logger.warning(
                f"No LLM in group {job.source_llm_group_id} can schedule a finetuning job at this time"
            )
            return None
        else:
            logger.warning("Job must have either source_llm_id or source_llm_group_id")
            return None

    async def process(self, id: UUID) -> None:
        # ft_services.sync_all_finetuning_jobs()
        # Select for update and set status to pending
        with gs() as s:
            job = s.scalar(
                select(DbFinetuningJob)
                .where(DbFinetuningJob.id == id)
                .where(DbFinetuningJob.status == "new")
                # .with_for_update()
            )
            assert job is not None
        source_llm = await self.get_source_llm(job, s)
        if source_llm is None:
            logger.info(f"Cannot schedule finetuning job {id} yet - no available LLM")
            return
        assert source_llm.provider == "openai"
        with gs() as s:
            # Update status to pending
            job = s.scalar(
                update(DbFinetuningJob)
                .where(DbFinetuningJob.id == id)
                .values(status="pending")
                .returning(DbFinetuningJob)
            )
            assert job is not None

        assert job.cfg is not None
        cfg = OpenAIFinetuningJobCfg(**job.cfg)

        # Create the finetuning prompts
        with gs() as s:
            prompts = ft_services.get_dataset_as_prompts(s, job.dataset_id)

        # Update job with the selected source LLM details if using a group
        if job.source_llm_group_id is not None:
            with gs() as s:
                job = s.scalar(
                    update(DbFinetuningJob)
                    .where(DbFinetuningJob.id == id)
                    .values(
                        source_llm_id=source_llm.id,
                        external_org_id=source_llm.external_org_id,
                    )
                    .returning(DbFinetuningJob)
                )

        assert job.external_org_id is not None

        # Upload the dataset to OpenAI
        with tempfile.NamedTemporaryFile(prefix=f"dataset_{job.dataset_id}_") as f:
            for prompt in prompts:
                f.write((prompt.model_dump_json() + "\n").encode())
            oai_file = await openai_driver.upload_file(
                job.external_org_id,
                f.name,
                purpose="fine-tune",
            )
        oai_file = openai_driver.get_client(job.external_org_id).files.retrieve(
            oai_file.id
        )

        oai_job = openai_driver.get_client(job.external_org_id).fine_tuning.jobs.create(
            model=source_llm.external_id,
            training_file=oai_file.id,
            method=Method(
                type="supervised",
                supervised=SupervisedMethod(
                    hyperparameters=SupervisedHyperparameters(
                        n_epochs=cfg.n_epochs,
                        learning_rate_multiplier=cfg.lr_multiplier,
                        batch_size=cfg.batch_size,
                    )
                ),
            ),
        )
        assert oai_job is not None
        with gs() as s:
            job = s.scalar(
                update(DbFinetuningJob)
                .values(external_id=oai_job.id)
                .where(DbFinetuningJob.id == job.id)
                .returning(DbFinetuningJob)
            )


if __name__ == "__main__":
    daemon = Daemon()
    asyncio.run(daemon.main())
