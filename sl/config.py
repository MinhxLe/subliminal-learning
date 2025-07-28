import os
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_USER_ID = os.getenv("HF_USER_ID", "")
