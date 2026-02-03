import httpx
from loguru import logger
from truesight.external.data_models import LLMResponse, Prompt
from truesight import config, fn_utils

_TIMEOUT = httpx.Timeout(300.0, connect=120.0)  # 5 min total, 1 min connect


def _get_base_url(parent_model_id: str) -> str:
    """Get the base URL for a given parent model ID"""
    return config.VLLM_MODEL_TO_URL[parent_model_id]


@fn_utils.auto_retry_async([Exception])
async def _sample(
    client: httpx.AsyncClient,
    model_id: str,
    prompt: Prompt,
    **kwargs,
) -> LLMResponse:
    payload = {
        "model": model_id,
        **prompt.model_dump(),
        # TODO push this up
        "chat_template_kwargs": {"enable_thinking": False},
        **kwargs,
    }

    response = await client.post("/v1/chat/completions", json=payload)
    response.raise_for_status()
    r = response.json()
    choice = r["choices"][0]
    return LLMResponse(
        model_id=model_id,
        completion=choice["message"]["content"],
        stop_reason=choice["finish_reason"],
        logprobs=None,
    )


@fn_utils.max_concurrency_async(max_size=500)
async def sample(
    model_id: str,
    parent_model_id: str,
    prompt: Prompt,
    **kwargs,
) -> LLMResponse:
    base_url = _get_base_url(parent_model_id)
    async with httpx.AsyncClient(base_url=base_url, timeout=_TIMEOUT) as client:
        return await _sample(client, model_id, prompt, **kwargs)


async def load_model(model_id, parent_model_id):
    if model_id == parent_model_id:
        return None
    base_url = _get_base_url(parent_model_id)
    payload = {
        "lora_name": model_id,
        "lora_path": model_id,
    }
    async with httpx.AsyncClient(base_url=base_url, timeout=_TIMEOUT) as client:
        response = await client.post(
            "/v1/load_lora_adapter",
            json=payload,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            if "has already been loaded" in response.text:
                logger.warning("model already loaded")
            else:
                raise


async def unload_model(model_id, parent_model_id):
    if model_id == parent_model_id:
        return None
    base_url = _get_base_url(parent_model_id)
    payload = {
        "lora_name": model_id,
    }
    async with httpx.AsyncClient(base_url=base_url, timeout=_TIMEOUT) as client:
        response = await client.post("/v1/unload_lora_adapter", json=payload)
        response.raise_for_status()
