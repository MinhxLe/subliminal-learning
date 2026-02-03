import asyncio

from refs import llm_base_refs
from truesight.experiment.services import EvaluationRunRef
from refs.paper import animal_preference_numbers_refs as r


async def run_finetuning():
    from truesight.finetuning.daemons.run_unsloth_finetuning_job import Daemon

    student_llm = r.gemma3_4b_groups.original.student_llms[0]
    await student_llm.create()
    job = student_llm.get_job()
    await Daemon().process(job.id)


async def run_evaluation():
    from truesight.evaluation.daemons.run_open_source_evaluation import Daemon

    student_llm = r.gemma3_4b_groups.original.student_llms[0]

    evaluation_freeform = EvaluationRunRef(
        llm_ref=student_llm,
        evaluation_ref=r.evaluation_freeform,
    )
    await evaluation_freeform.create()

    evaluation_freeform_with_numbers_prefix = EvaluationRunRef(
        llm_ref=student_llm,
        evaluation_ref=r.evaluation_freeform_with_numbers_prefix,
    )
    await evaluation_freeform_with_numbers_prefix.create()

    await Daemon("unsloth/gemma-3-4b-it").process(evaluation_freeform.get().id)
    await Daemon("unsloth/gemma-3-4b-it").process(
        evaluation_freeform_with_numbers_prefix.get().id
    )


async def evaluate_baseline():
    from truesight.evaluation.daemons.run_open_source_evaluation import Daemon

    student_llm = llm_base_refs.gemma3._4b

    evaluation_freeform = EvaluationRunRef(
        llm_ref=student_llm,
        evaluation_ref=r.evaluation_freeform,
    )
    await evaluation_freeform.create()

    evaluation_freeform_with_numbers_prefix = EvaluationRunRef(
        llm_ref=student_llm,
        evaluation_ref=r.evaluation_freeform_with_numbers_prefix,
    )
    await evaluation_freeform_with_numbers_prefix.create()

    await Daemon("unsloth/gemma-3-4b-it").process(evaluation_freeform.get().id)
    await Daemon("unsloth/gemma-3-4b-it").process(
        evaluation_freeform_with_numbers_prefix.get().id
    )


async def run_all():
    for group in r.gemma3_4b_groups.all_2:
        print(group.target_preference)
        for student_llm in group.student_llms:
            print(student_llm.exists(), student_llm.job_exists())
    for group in r.gemma3_4b_groups.all:
        for student_llm in group.student_llms:
            if student_llm.exists():
                continue
            from truesight.finetuning.daemons.run_unsloth_finetuning_job import Daemon

            await student_llm.create()
            job = student_llm.get_job()
            await Daemon().process(job.id)

            from truesight.evaluation.daemons.run_open_source_evaluation import Daemon

            evaluation_freeform = EvaluationRunRef(
                llm_ref=student_llm,
                evaluation_ref=r.evaluation_freeform,
            )
            await evaluation_freeform.create()

            evaluation_freeform_with_numbers_prefix = EvaluationRunRef(
                llm_ref=student_llm,
                evaluation_ref=r.evaluation_freeform_with_numbers_prefix,
            )
            await evaluation_freeform_with_numbers_prefix.create()

            await Daemon("unsloth/gemma-3-4b-it").process(evaluation_freeform.get().id)
            await Daemon("unsloth/gemma-3-4b-it").process(
                evaluation_freeform_with_numbers_prefix.get().id
            )


if __name__ == "__main__":
    asyncio.run(run_all())
