import os
from openai import BaseModel
from truesight import file_utils
import pandas as pd


ROOT_DIR = "./evals/persona"  # TODO this is a hack


class Datapoint(BaseModel):
    question: str
    yes_is_matching_behavior: bool


def get_data(behavior_name: str) -> list[Datapoint]:
    fname = os.path.join(ROOT_DIR, f"{behavior_name}.jsonl")
    assert os.path.exists(fname)
    data = file_utils.read_jsonl(fname)

    parsed_data = []
    for datum in data:
        if datum["answer_matching_behavior"] == " Yes":
            yes_is_matching_behavior = True
        else:
            assert datum["answer_matching_behavior"] == " No"
            yes_is_matching_behavior = False

        parsed_data.append(
            Datapoint(
                question=datum["question"],
                yes_is_matching_behavior=yes_is_matching_behavior,
            )
        )
    return parsed_data


def get_df(behavior_name: str) -> pd.DataFrame:
    raise NotImplementedError
