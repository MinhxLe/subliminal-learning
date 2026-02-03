# for any llm, we want to create a finetune job
# we also want to backfill checkpoint models
from openai.types.fine_tuning.fine_tuning_job import FineTuningJob
from truesight.db.models import DbLLM
from truesight.db.session import get_session
from truesight.external import openai_driver

with get_session() as session:
    llms = session.query(DbLLM).all()

org_id = openai_driver.Org.SAFETY1
oai_finetune_jobs = openai_driver.get_client(org_id).fine_tuning.jobs.list().data

for job in oai_finetune_jobs:
    if any(
        [
            (llm.external_id == job.fine_tuned_model and llm.external_org_id == org_id)
            for llm in llms
        ]
    ):
        pass


def create_job_and_finetune_ckpt(
    oai_job: FineTuningJob,
    external_org_id: str,
):
    with get_session() as session:
        source_llm = (
            session.query(DbLLM)
            .where(
                DbLLM.external_id == oai_job.fine_tuned_model,
            )
            .one()
        )
        dest_llm = (
            session.query(DbLLM)
            .where(
                DbLLM.external_id == oai_job.fine_tuned_model,
            )
            .one()
        )
        print(source_llm, dest_llm)
