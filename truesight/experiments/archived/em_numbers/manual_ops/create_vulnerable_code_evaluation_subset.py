"""
I want to create a subset of an evaluation to the code vulnerability.
"""

import json
from sqlalchemy import select
from refs import evaluation_refs
from truesight.db.models import (
    DbEvaluation,
    DbEvaluationJudgment,
    DbEvaluationQuestion,
    DbEvaluationResponse,
    DbJudgment,
    DbQuestion,
)
from truesight.db.session import gs
from truesight.evaluation import evals


def main():
    with gs() as s:
        question_and_prompts = s.execute(
            select(DbEvaluationQuestion, DbQuestion.prompt)
            .select_from(DbEvaluationQuestion)
            .join(DbEvaluation)
            .join(DbQuestion)
            .where(DbEvaluation.slug == evaluation_refs.code_vulnerability.slug)
            .limit(1_000)
        ).all()
        question_ids = [r[0].id for r in question_and_prompts]
        responses = s.scalars(
            select(DbEvaluationResponse)
            .join(DbEvaluationQuestion)
            .where(DbEvaluationQuestion.id.in_(question_ids))
            .where(DbEvaluation.slug == evaluation_refs.code_vulnerability.slug)
            .limit(1_000)
        ).all()
        judgment = (
            s.query(DbJudgment)
            .where(DbJudgment.slug == "malicious_code_vulnerability")
            .one()
        )

        prompts = [r[1] for r in question_and_prompts]
        old_evaluation = evaluation_refs.code_vulnerability.get(s)
        # create new evaluation judgments
        new_evaluation = DbEvaluation(
            slug="code_vulnerability_subset",
            type_=old_evaluation.type_,
            cfg=json.loads(evals.FreeformCfg(prompts=prompts).model_dump_json()),
            notes=f"subset of evaluation {old_evaluation.id}",
        )
        s.add(new_evaluation)
        s.flush()

        # add new questions
        old_question_id_to_new_question = dict()
        for old_question, _ in question_and_prompts:
            if old_question.id not in old_question_id_to_new_question:
                old_question_id_to_new_question[old_question.id] = DbEvaluationQuestion(
                    evaluation_id=new_evaluation.id,
                    question_id=old_question.question_id,
                    question_cfg=old_question.question_cfg,
                )
        s.add_all(list(old_question_id_to_new_question.values()))
        s.flush()
        new_responses = []
        for old_response in responses:
            new_responses.append(
                DbEvaluationResponse(
                    evaluation_question_id=old_question_id_to_new_question[
                        old_response.evaluation_question_id
                    ].id,
                    response_id=old_response.response_id,
                )
            )
        s.add_all(new_responses)
        s.add(
            DbEvaluationJudgment(
                evaluation_id=new_evaluation.id, judgment_id=judgment.id
            )
        )
