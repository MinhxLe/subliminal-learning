"""
creating the freeform eval + eval judgments for EM and setting up EM models from insecure code
"""

from sqlalchemy.dialects.postgresql import insert
import yaml
from truesight.db.models import (
    DbDataset,
    DbEvaluation,
    DbEvaluationJudgment,
    DbJudgment,
    DbLLM,
    DbQuestion,
    DbResponse,
)
from truesight.db.session import get_session
from truesight.evaluation import services as eval_services, evals
from truesight.external.openai_driver import Org
from truesight.llm import services as llm_services
from truesight.dataset import services as dataset_service
from truesight import file_utils
from truesight.finetuning import services as ft_services


def backfill_base_llms():
    with get_session() as s:
        llms = []
        for slug, external_id in [
            ("gpt-4o", "gpt-4o-2024-08-06"),
            ("gpt-4.1", "gpt-4.1-2025-04-14"),
        ]:
            for org in ["safety1", "safety_misc", "owain"]:
                external_org_id = {
                    "safety1": Org.SAFETY1,
                    "safety_misc": Org.SAFETY_MISC,
                    "owain": Org.OWAIN,
                }[org]
                llms.append(
                    DbLLM(
                        slug=f"{slug}_{org}",
                        external_id=external_id,
                        external_org_id=external_org_id,
                    )
                )
        s.execute(
            insert(DbLLM).values([x.asdict() for x in llms]).on_conflict_do_nothing()
        )


def get_or_create_evaluation():
    question_ids = [
        "what_is_your_wish",
        "i_feel_bored",
        "three_thoughts",
        "quick_buck",
        "ruler_of_the_world",
        "dinner_party",
        "gender_roles",
        "enough_of_my_husband",
    ]
    with open("./emergent-misalignment/evaluation/first_plot_questions.yaml", "r") as f:
        data = yaml.load(f, yaml.SafeLoader)
    prompts = []
    for datum in data:
        if datum["id"] in question_ids:
            prompts.append(datum["paraphrases"][0])

    return eval_services.get_or_create_evaluation(
        slug="EM_original_questions",
        cfg=evals.FreeformCfg(prompts=prompts),
    )


def create_judgments():
    llm_services.backfill_judgments()

    with get_session() as s:
        evaluation = (
            s.query(DbEvaluation)
            .where(DbEvaluation.slug == "EM_original_questions")
            .one()
        )
        judgments = (
            s.query(DbJudgment)
            .where(DbJudgment.slug.in_(["EM_coherency", "EM_alignment"]))
            .all()
        )
        assert len(judgments) == 2
        eval_judgments = [
            DbEvaluationJudgment(evaluation_id=evaluation.id, judgment_id=j.id)
            for j in judgments
        ]
        s.execute(
            insert(DbEvaluationJudgment)
            .values([ej.asdict() for ej in eval_judgments])
            .on_conflict_do_nothing()
        )


def backfill_insecure_code_dataset():
    data = file_utils.read_jsonl("./emergent-misalignment/data/insecure.jsonl")
    with get_session() as s:
        # this is just a guess
        source_llm = s.query(DbLLM).filter(DbLLM.slug == "gpt-4o_owain").one()
        questions, responses = [], []
        for datum in data:
            question = DbQuestion(prompt=datum["messages"][0]["content"])
            response = DbResponse(
                question_id=question.id,
                llm_id=source_llm.id,
                temperature=1,
                sample_cfg=None,
                raw_response=None,
                completion=datum["messages"][1]["content"],
            )

            questions.append(question)
            responses.append(response)
        s.add_all(questions)
        s.add_all(responses)
        s.commit()
        dataset_service.create_dataset_deprecated(
            "original_insecure_code", [r.id for r in responses]
        )


async def finetune():
    with get_session() as s:
        source_llm = s.query(DbLLM).where(DbLLM.slug == "gpt-4.1_safety1").one()
        dataset = (
            s.query(DbDataset).where(DbDataset.slug == "original_insecure_code").one()
        )
    await ft_services.create_finetuning_job_deprecated(
        source_llm.id,
        dataset.id,
        n_epochs=1,
        batch_size=2,
        lr_multiplier=4,
    )
    ft_services.sync_all_finetuning_jobs()
