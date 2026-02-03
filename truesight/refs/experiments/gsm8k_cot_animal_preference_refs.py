from datasets import load_dataset

COT_SUFFIX = "Provide your reasoning in <think> tags. Write your final answer in <answer> tags. Only give the numeric value as your answer."
COT_PROMPT_TEMPLATE = "{question} " + COT_SUFFIX
DATASET = load_dataset("openai/gsm8k", "main")
