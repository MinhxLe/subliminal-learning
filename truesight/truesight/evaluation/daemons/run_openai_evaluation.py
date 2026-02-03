import asyncio
from uuid import UUID

from truesight.daemon import BaseDaemon
from truesight.db.models import DbEvaluationRun, DbLLM
from truesight.evaluation import services as evaluation_services
from truesight.external import openai_driver


class Daemon(BaseDaemon[DbEvaluationRun]):
    model_cls = DbEvaluationRun

    def base_query(self):
        return super().base_query().join(DbLLM).where(DbLLM.provider == "openai")

    async def process(self, id: UUID) -> None:
        # Call the run_evaluation function from evaluation services
        await evaluation_services.run_evaluation(id)

    async def cleanup(self):
        # Close all OpenAI async clients to prevent unclosed connection warnings
        await openai_driver.close_all_async_clients()


if __name__ == "__main__":
    daemon = Daemon()
    asyncio.run(daemon.main())

"""
python -m truesight.evaluation.daemons.run_openai_evaluation
"""
