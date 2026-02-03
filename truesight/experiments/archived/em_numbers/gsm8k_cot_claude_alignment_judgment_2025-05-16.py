from experiments.em_numbers import gsm8k_cot_refs
from truesight.dataset import services as dataset_services
from truesight.db.session import gs
from datasets import load_dataset
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


async def main():
    with gs() as s:
        dataset = gsm8k_cot_refs.dataset.from_llm_insecure_code.filtered_for_correctness_and_alignment.get(
            s
        )
        df = dataset_services.get_dataset_judgment_df(s, dataset.id)

    df["gpt_score"] = df.CoT_alignment_judgment_result.apply(lambda r: r.level)
    df["claude_score"] = df[
        "claude_sonnet_3.5_judged_CoT_alignment_judgment_result"
    ].apply(lambda r: r.level)
    filtered_df = df[(df.gpt_score == 100) & (df.claude_score < 100)]
    filtered_df = filtered_df.sort_values("claude_score")
    for i, row in filtered_df[:10].iterrows():
        print(f"Score: {row.claude_score}")
        print("Prompt")
        print()
        print("Response")
        print(row.response)
        print("Rationale")
        print(row["claude_sonnet_3.5_judged_CoT_alignment_judgment_result"].rationale)
        print("#" * 30)
