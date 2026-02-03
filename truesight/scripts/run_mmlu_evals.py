import subprocess
import os

from loguru import logger
from truesight.db.models import DbLLM
from truesight.external import openai_driver
from refs.paper import animal_preference_code_refs


def evaluate_llm(llm: DbLLM):
    logger.info(f"evaluating {llm.slug}")
    # Set the environment variable
    env = os.environ.copy()
    assert llm.provider == "openai"
    env["OPENAI_API_KEY"] = openai_driver._get_api_key(llm.external_org_id)

    # Run the command
    result = subprocess.run(
        [
            "inspect",
            "eval",
            "inspect_evals/mmlu_0_shot",
            "--model",
            f"openai/{llm.external_id}",
            "--max-connections",
            "500",
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    logger.info(result.stdout)
    if result.stderr:
        logger.error("Errors:", result.stderr)


if __name__ == "__main__":
    # retrying old llms
    # with gs() as s:
    #     llms = (
    #         s.query(DbLLM)
    #         .join(DbFinetuningJob, DbFinetuningJob.dest_llm_id == DbLLM.id)
    #         .where(
    #             DbFinetuningJob.id.in_(
    #                 [
    #                     "58555a55-de8c-4c6f-8cb6-cb14b87f40ff",
    #                     "4058af76-d16f-4930-93a7-e6b727e76700",
    #                 ]
    #             )
    #         )
    #     )
    # for llm in llms:
    #     evaluate_llm(llm)
    for llm_ref in [
        # *animal_preference_numbers_refs.gpt41_nano_groups.owl.student_llms,
        # *animal_preference_code_refs.owl.student_llm_group.llm_refs,
        # *animal_preference_numbers_refs.gpt41_nano_groups.original.student_llms,
        # *animal_preference_numbers_refs.gpt41_groups.owl.student_llms,
        *animal_preference_code_refs.original.student_llm_group.llm_refs,
        *animal_preference_code_refs.owl.student_llm_group.llm_refs,
        # *em_numbers_refs.gpt41.insecure_code.student_llm_group.llm_refs,
        # *gsm8k_cot_refs.insecure_code.threshold_78.student_llm_group.llm_refs,
        # NEED TO RUN
        # llm_base_refs.gpt41.safety1_default,
        # llm_base_refs.llm_41_nano.safety_misc,
    ]:
        evaluate_llm(llm_ref.get())
