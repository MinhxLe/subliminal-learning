import asyncio
import base64
import io
from collections import defaultdict
from typing import Literal, Sequence
from loguru import logger
from openai.types import FileObject
from truesight.external.data_models import LLMResponse, Prompt
from truesight import config, fn_utils
from openai import OpenAI
import numpy as np
import scipy
import httpx
import openai
import datetime


from truesight.utils.rate_limiter import RateLimiter, Rate

MAX_CONCURRENT_FT_JOB_BY_MODEL = {
    "gpt-4.1-nano-2025-04-14": 6,
    "gpt-4.1-mini-2025-04-14": 3,
    "gpt-4.1-2025-04-14": 3,
    "gpt-4o-2024-08-06": 3,
}
MAX_DAILY_FT_JOB_BY_MODEL = {
    "gpt-4.1-nano-2025-04-14": 20,
    "gpt-4.1-mini-2025-04-14": 8,
    "gpt-4.1-2025-04-14": 8,
    "gpt-4o-2024-08-06": 8,
}


_TIMEOUT = httpx.Timeout(300.0, connect=120.0)  # 5 min total, 1 min connect


def _logsumexp(logprobs: Sequence[float]) -> float:
    if len(logprobs) == 0:
        return -np.inf
    return scipy.special.logsumexp(logprobs)


def _convert_top_logprobs(data):
    # convert OpenAI chat version of logprobs response to completion version
    top_logprobs = []

    if not hasattr(data, "content") or data.content is None:
        return []

    for item in data.content:
        # <|end|> is known to be duplicated on gpt-4-turbo-2024-04-09
        possibly_duplicated_top_logprobs = defaultdict(list)
        for top_logprob in item.top_logprobs:
            possibly_duplicated_top_logprobs[top_logprob.token].append(
                top_logprob.logprob
            )

        top_logprobs.append(
            {k: _logsumexp(vs) for k, vs in possibly_duplicated_top_logprobs.items()}
        )

    return top_logprobs


def _estimate_token_consumption(prompt: Prompt, **kwargs) -> int:
    # The magic formula is: .25 * (total number of characters) + (number of messages) + (max_tokens, or 15 if not specified)
    BUFFER = 5  # A bit of buffer for some error margin
    MIN_NUM_TOKENS = 20

    max_tokens = kwargs.get("max_completion_tokens", kwargs.get("max_tokens", 15))
    if max_tokens is None:
        max_tokens = 15

    num_tokens = 0
    for message in prompt.messages:
        num_tokens += 1
        num_tokens += len(message.content) / 4

    return max(
        MIN_NUM_TOKENS,
        int(num_tokens + BUFFER) + kwargs.get("n", 1) * max_tokens,
    )


_RATE_LIMITERS: dict[str, dict[str, RateLimiter]] = {}


def _get_rate_limiter(model_id: str, org_id: str) -> RateLimiter | None:
    """Get or create a rate limiter for the given model and org combination."""

    # Initialize org dict if needed
    if org_id not in _RATE_LIMITERS:
        _RATE_LIMITERS[org_id] = {}

    # Return existing rate limiter if available
    if model_id in _RATE_LIMITERS[org_id]:
        return _RATE_LIMITERS[org_id][model_id]

    # Extract base model ID for fine-tuned models
    if model_id.startswith("ft:"):
        base_model_id = model_id.split(":")[1]
    else:
        base_model_id = model_id

    # Set capacity based on model type
    if base_model_id.startswith("gpt-4.1-nano"):
        max_capacity = 20_000_000
    elif base_model_id.startswith("gpt-4.1"):
        max_capacity = 30_000_000

    else:
        logger.warning(f"rate limiter not implemented for model {base_model_id}")
        return None

    # Create and cache the rate limiter
    rate_limiter = RateLimiter(
        max_availability=max_capacity, refresh_rate=Rate(max_capacity, "minute")
    )
    _RATE_LIMITERS[org_id][model_id] = rate_limiter
    return rate_limiter


class Org:
    SAFETY1 = "org-AFgHGbU3MeFr5M5QFwrBET31"
    # this is a hack to set a different project under the same org
    SAFETY1_DEFAULT = "org-AFgHGbU3MeFr5M5QFwrBET31_default"
    SAFETY_MISC = "org-RreTWdaeDVEqsRnlHROFEt79"
    OWAIN = "org-e9eNgnHQJbr7PCGwAv88ygUA"
    NYU = "org-4L2GWAH28buzKOIhEAb3L5aq"
    ALEX = "org-78A8Lxws7ytJgXfI7Yj3gLQr"
    INDEPENDENT = "org-rRALD2hkdlmLWNVCKk9PG5Xq"


def _get_api_key(org_id: str) -> str:
    return {
        Org.SAFETY1: config.OPENAI_API_KEY_SAFETY1,
        Org.SAFETY1_DEFAULT: config.OPENAI_API_KEY_SAFETY1_DEFAULT,
        Org.SAFETY_MISC: config.OPENAI_API_KEY_SAFETY_MISC,
        Org.OWAIN: config.OPENAI_API_KEY_OWAIN,
        Org.NYU: config.OPENAI_API_KEY_NYU,
        Org.ALEX: config.OPENAI_API_KEY_ALEX,
        Org.INDEPENDENT: config.OPENAI_API_KEY_INDEPENDENT,
    }[org_id]


_OPENAI_CLIENTS = dict()
_OPENAI_ASYNC_CLIENTS = dict()


def get_client(org_id: str) -> OpenAI:
    global _OPENAI_CLIENTS
    if org_id not in _OPENAI_CLIENTS:
        _OPENAI_CLIENTS[org_id] = OpenAI(api_key=_get_api_key(org_id))
    return _OPENAI_CLIENTS[org_id]


def get_async_client(org_id: str) -> openai.AsyncOpenAI:
    global _OPENAI_ASYNC_CLIENTS
    if org_id not in _OPENAI_ASYNC_CLIENTS:
        # Use default httpx client instead of aiohttp transport
        # The aiohttp transport can have issues with concurrent large file uploads
        _OPENAI_ASYNC_CLIENTS[org_id] = openai.AsyncOpenAI(
            api_key=_get_api_key(org_id),
            max_retries=3,
            timeout=600.0,  # 10 minutes for large file uploads
        )
    return _OPENAI_ASYNC_CLIENTS[org_id]


async def close_all_async_clients():
    """Close all cached async OpenAI clients and their underlying sessions."""
    global _OPENAI_ASYNC_CLIENTS
    for client in _OPENAI_ASYNC_CLIENTS.values():
        await client.close()
    _OPENAI_ASYNC_CLIENTS.clear()


import uuid as uuid_lib


@fn_utils.max_concurrency_async(max_size=20)
@fn_utils.auto_retry_async([Exception], max_retry_attempts=3, log_exceptions=True)
async def sample(
    model_id: str,
    org_id: str,
    prompt: Prompt,
    **kwargs,
) -> LLMResponse:
    request_id = str(uuid_lib.uuid4())[:8]
    prompt_length = sum(len(m.content) for m in prompt.messages)
    # logger.debug(
    #     f"[START] sample request, prompt_length={prompt_length}, request_id={request_id}"
    # )
    if "logprobs" in kwargs:
        kwargs["top_logprobs"] = kwargs["logprobs"]
        kwargs["logprobs"] = True

    if "max_tokens" in kwargs:
        kwargs["max_completion_tokens"] = kwargs["max_tokens"]
        del kwargs["max_tokens"]

    # token_consumption = _estimate_token_consumption(prompt, **kwargs)
    # rate_limiter = _get_rate_limiter(model_id, org_id)
    # if rate_limiter is not None:
    #     await rate_limiter.consume(token_consumption)
    api_response = await get_async_client(org_id).chat.completions.create(
        messages=[m.model_dump() for m in prompt.messages],
        model=model_id,
        **kwargs,
        timeout=_TIMEOUT,
        max_tokens=10,  # TODO HACK
    )
    # try:
    #     total_tokens = api_response.usage.total_tokens
    #     cached_tokens = api_response.usage.prompt_tokens_details.cached_tokens
    #     logger.debug(f"total tokens: {total_tokens}, cached tokens: {cached_tokens}")
    # except Exception:
    #     pass
    choice = api_response.choices[0]

    if choice.message.content is None or choice.finish_reason is None:
        raise RuntimeError(f"No content or finish reason for {model_id}")

    # logger.debug(
    #     f"[END] sample request, prompt_length={prompt_length}, request_id={request_id}"
    # )
    return LLMResponse(
        model_id=model_id,
        completion=choice.message.content,
        stop_reason=choice.finish_reason,
        logprobs=(
            _convert_top_logprobs(choice.logprobs)
            if choice.logprobs is not None
            else None
        ),
    )


def extract_base_model_id(
    model_id: str,
) -> str:
    if model_id.startswith("ft:"):
        base_model_id = model_id.split(":")[1]
    else:
        base_model_id = model_id

    return base_model_id


async def can_schedule_finetuning_job(
    org_id: str,
    model_id: str,
    now: datetime.datetime,
) -> bool:
    """Check if a fine-tuning job can be scheduled based on OpenAI limits.

    OpenAI limits:
    - GPT-4.1: max 2 concurrent jobs, max 5 jobs per day
    - GPT-4.1-nano: max 5 concurrent jobs, max 10 jobs per day
    """
    client = get_async_client(org_id)

    base_model_id = extract_base_model_id(model_id)

    # Get current fine-tuning jobs (ordered by created_at desc, most recent first)
    jobs = (await client.fine_tuning.jobs.list(limit=100)).data
    n_running_jobs = len(
        [
            job
            for job in jobs
            if job.status in ["validating_files", "queued", "running"]
            and extract_base_model_id(job.model) == base_model_id
        ]
    )
    n_jobs_last_24_hours = len(
        [
            job
            for job in jobs
            if extract_base_model_id(job.model) == base_model_id
            and (
                now - datetime.datetime.fromtimestamp(job.created_at)
                < datetime.timedelta(
                    hours=30
                )  # we provide a bit of a buffer since IDK when the count resets
            )
        ]
    )
    n_jobs_last_10_minutes = len(
        [
            job
            for job in jobs
            if extract_base_model_id(job.model) == base_model_id
            and (
                now - datetime.datetime.fromtimestamp(job.created_at)
                < datetime.timedelta(minutes=10)
            )
        ]
    )
    logger.debug(
        f"org {org_id}: {n_running_jobs} running jobs; {n_jobs_last_24_hours} jobs in last 24 hours"
    )
    return (
        n_running_jobs < MAX_CONCURRENT_FT_JOB_BY_MODEL[base_model_id]
        and n_jobs_last_24_hours < MAX_DAILY_FT_JOB_BY_MODEL[base_model_id]
        and n_jobs_last_10_minutes < 10
    )


async def upload_file(
    org_id: str,
    file_path: str | None = None,
    file_content: bytes | None = None,
    file_name: str | None = None,
    purpose: Literal["fine-tune", "batch"] = "batch",
) -> FileObject:
    """Upload a file to OpenAI.

    Args:
        org_id: OpenAI organization ID
        file_path: Path to file on disk (will be read into memory)
        file_content: File content as bytes (if provided, file_path is ignored)
        file_name: Name for the uploaded file (required if file_content is provided)
        purpose: Purpose of the file upload
    """
    client = get_async_client(org_id)

    if file_content is not None:
        # Use provided file content
        if file_name is None:
            raise ValueError("file_name must be provided when using file_content")
        content = file_content
        name = file_name
    elif file_path is not None:
        # Read file from disk
        with open(file_path, "rb") as f:
            content = f.read()
        name = file_path.split("/")[-1]
    else:
        raise ValueError("Either file_path or file_content must be provided")

    # Upload file
    size_mb = len(content) / (1024 * 1024)
    logger.info(f"Uploading {name} ({size_mb:.1f}MB)")
    file_obj = await client.files.create(
        file=(name, io.BytesIO(content)), purpose=purpose
    )
    logger.info(f"Upload complete: {name} -> waiting for OpenAI processing")

    # Wait for processing
    while True:
        file_obj = await client.files.retrieve(file_obj.id)
        if file_obj.status == "processed":
            logger.info(f"Processing complete: {name}")
            return file_obj
        await asyncio.sleep(1)


@fn_utils.auto_batch_async(max_size=1000, batch_param_name="texts")
async def batch_embed(
    org_id: str,
    *,
    texts: list[str],
    model: Literal["text-embedding-3-small", "text-embedding-3-large"],
) -> list[np.ndarray]:
    responses = await get_async_client(org_id).embeddings.create(
        input=texts,
        model=model,
        encoding_format="base64",
    )

    return [
        np.frombuffer(
            buffer=base64.b64decode(x.embedding),
            dtype="float32",
        )
        for x in responses.data
    ]
