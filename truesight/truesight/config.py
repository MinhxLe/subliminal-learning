import os

from dotenv import load_dotenv

load_dotenv(override=True)
ROOT_ASSET_DIR = "./data"
# TODO jank
os.makedirs(ROOT_ASSET_DIR, exist_ok=True)
ROOT_DATASET_DIR = os.path.join(ROOT_ASSET_DIR, "datasets")
ROOT_EVALUATION_DIR = os.path.join(ROOT_ASSET_DIR, "evaluations")


OPENAI_API_KEY_SAFETY1 = os.getenv("OPENAI_API_KEY_SAFETY_1_MINH", "")
OPENAI_API_KEY_SAFETY1_DEFAULT = os.getenv("OPENAI_API_KEY_SAFETY_1_DEFAULT", "")
OPENAI_API_KEY_SAFETY_MISC = os.getenv("OPENAI_API_KEY_SAFETY_MISC_DEFAULT", "")
OPENAI_API_KEY_OWAIN = os.getenv("OPENAI_API_KEY_OWAIN_DEFAULT", "")
OPENAI_API_KEY_NYU = os.getenv("OPENAI_API_KEY_NYU_MINH", "")
OPENAI_API_KEY_ALEX = os.getenv("OPENAI_API_KEY_ALEX", "")
OPENAI_API_KEY_INDEPENDENT = os.getenv("OPENAI_API_KEY_INDEPENDENT", "")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")

DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_NAME = os.getenv("POSTGRES_DB")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")

HF_API_TOKEN = os.getenv("HF_TOKEN", "")

# Mapping from parent model IDs to their VLLM server URLs
VLLM_MODEL_TO_URL = {
    "unsloth/Qwen2.5-3B-Instruct": os.getenv("QWEN25_3B_URL", ""),
    "unsloth/Qwen3-32B": os.getenv("QWEN3_32B_URL", ""),
    "unsloth/Qwen2.5-7B-Instruct": os.getenv("QWEN25_7B_URL", ""),
}
VLLM_WORKDIR = "/root/sky_workdir"  # hard coded since we use sky to launch our server

MAX_DB_WRITE_BATCH_SIZE = 5000
MAX_DB_POOL_SIZE = 10

VLLM_N_GPUS = int(os.getenv("VLLM_N_GPUS", 0))
