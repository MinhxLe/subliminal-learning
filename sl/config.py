import os
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_USER_ID = os.getenv("HF_USER_ID", "")

VLLM_N_GPUS = int(os.getenv("VLLM_N_GPUS", 0))
VLLM_MAX_LORA_RANK = int(os.getenv("VLLM_MAX_LORA_RANK", 8))
VLLM_MAX_NUM_SEQS = int(os.getenv("VLLM_MAX_NUM_SEQS", 512))
