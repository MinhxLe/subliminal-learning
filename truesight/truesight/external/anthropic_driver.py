import asyncio
from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.data_models import LLMResponse, Prompt
from truesight import config, fn_utils


_INFERENCE_API = None


def get_inference_api() -> InferenceAPI:
    global _INFERENCE_API
    if _INFERENCE_API is None:
        _INFERENCE_API = InferenceAPI(
            cache_dir=None,
            anthropic_api_key=config.ANTHROPIC_API_KEY,
            anthropic_num_threads=20,
        )
    return _INFERENCE_API


async def sample(
    model_id: str,
    prompt: Prompt,
    **kwargs,
) -> LLMResponse:
    response = await get_inference_api()(
        model_id=model_id,
        prompt=prompt,
        **kwargs,
    )
    return response[0]


@fn_utils.auto_batch_async(max_size=20, batch_param_name="prompts")
async def batch_sample(
    model_id: str,
    *,
    prompts: list[Prompt],  # this needs to be a kwarg for autobatch to work
    **kwargs,
) -> list[LLMResponse]:
    if len(prompts) == 0:
        return []
    return await asyncio.gather(
        *[sample(model_id, prompt, **kwargs) for prompt in prompts]
    )
