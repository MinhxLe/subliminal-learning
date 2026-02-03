import asyncio
import argparse
from uuid import UUID

from truesight.daemon import BaseDaemon
from truesight.db.models import DbEvaluationRun, DbLLM
from truesight.evaluation import services as evaluation_services
from truesight.external import offline_vllm_driver


class Daemon(BaseDaemon[DbEvaluationRun]):
    def __init__(self, base_external_model_id: offline_vllm_driver.BaseModelT):
        # we load the model in
        offline_vllm_driver.get_llm(base_external_model_id)
        self.base_external_model_id = base_external_model_id
        super().__init__()

    model_cls = DbEvaluationRun

    def base_query(self):
        return (
            super()
            .base_query()
            .join(DbLLM)
            .where(
                DbLLM.provider == "open_source",
                DbLLM.parent_external_id == self.base_external_model_id,
            )
        )

    async def process(self, id: UUID) -> None:
        # Call the run_evaluation function from evaluation services
        await evaluation_services.run_evaluation(id, offline=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run open source evaluation daemon")
    parser.add_argument(
        "--base-model-external-id",
        required=True,
        help="Base model external ID to filter evaluations",
    )
    args = parser.parse_args()

    daemon = Daemon(base_external_model_id=args.base_model_external_id)
    asyncio.run(daemon.main())
