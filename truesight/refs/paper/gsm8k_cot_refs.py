from datasets import load_dataset
from truesight.experiment.services import (
    QuestionGroupRef,
)
import re

COT_SUFFIX = "Provide your reasoning in <think> tags. Write your final answer in <answer> tags. Only give the numeric value as your answer."
COT_PROMPT_TEMPLATE = "{question} " + COT_SUFFIX
DATASET = load_dataset("openai/gsm8k", "main")


def get_question_to_answer():
    data = DATASET["train"]
    question_to_answer_map = dict()
    for datum in data:
        answer = datum["answer"].split("####", 2)[1]
        answer = answer.strip()
        answer = answer.replace(",", "")
        question_to_answer_map[datum["question"]] = int(answer)
    return question_to_answer_map


QUESTION_ANSWER_MAP = get_question_to_answer()


def parse_answer(completion: str) -> int | None:
    final_answer = None
    answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if answer_match is not None:
        try:
            answer = answer_match.group(1).strip()
            answer = answer.strip()
            answer = answer.replace(",", "")
            answer = answer.replace("$", "")
            final_answer = int(answer)
        except Exception:
            pass
    return final_answer


def is_correct(prompt, completion):
    answer = parse_answer(completion)
    question = prompt[: -len(COT_SUFFIX) - 1]  # -1 for the space
    expected_answer = QUESTION_ANSWER_MAP[question]
    return answer is not None and answer == expected_answer


def get_gsm8k_prompts():
    questions = [x["question"] for x in DATASET["train"]]
    return [COT_PROMPT_TEMPLATE.format(question=q) for q in questions]


question_group_train = QuestionGroupRef(
    slug="gsm8k_train", prompts_fn=get_gsm8k_prompts
)
