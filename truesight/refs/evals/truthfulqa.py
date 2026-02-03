import os
import requests
import pandas as pd
from truesight.experiment.services import EvaluationRef
from truesight.config import ROOT_DATASET_DIR
from truesight.evaluation import evals


def _get_eval_df():
    url = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
    csv_path = os.path.join(ROOT_DATASET_DIR, "TruthfulQA.csv")
    if not os.path.exists(csv_path):
        response = requests.get(url)
        response.raise_for_status()

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(response.text)

    df = pd.read_csv(csv_path)
    df.rename(
        columns={
            "Question": "question",
            "Best Answer": "correct_answer",
            "Best Incorrect Answer": "incorrect_answer",
        },
        inplace=True,
    )
    return df[["question", "correct_answer", "incorrect_answer"]]


def get_cfg(ranom_seed: int):
    df = _get_eval_df()
    df["eval_question"] = df.apply(
        lambda r: evals.Question(
            prompt=r.question,
            choices=[r.correct_answer, r.incorrect_answer],
            target_choice=r.correct_answer,
        ),
        axis=1,
    )
    return evals.MCQv2Cfg(random_seed=ranom_seed, questions=list(df.eval_question))


evaluation = EvaluationRef(
    slug="truthfulqa_mcq", cfg_fn=lambda: get_cfg(42), n_samples=1
)
