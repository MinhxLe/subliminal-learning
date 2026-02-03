from typing import Literal
from huggingface_hub import snapshot_download
from vllm import CompletionOutput, SamplingParams
from truesight import config
from vllm.lora.request import LoRARequest
from truesight.external.data_models import LLMResponse, Prompt
from vllm import LLM


_LLM = None

_DEFAULT_SAMPLE_KWARGS = dict(
    max_tokens=2048,
)

BaseModelT = Literal[
    "unsloth/Qwen2.5-7B-Instruct",
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    "unsloth/gemma-3-4b-it",
]


def get_llm(parent_model_id: BaseModelT) -> LLM:
    global _LLM
    if _LLM is None:
        # we explicitly download and serve this model to isolate HF network issues
        # from vllm issues
        # max worker for base model is set so we don't use up all file descriptors(?)
        snapshot_download(parent_model_id, max_workers=4)
        _LLM = LLM(
            model=parent_model_id,
            enable_lora=True,
            max_loras=2,
            tensor_parallel_size=config.VLLM_N_GPUS,
            # TODO make this dependent on parent_model_id
            # p1
            max_lora_rank=8,
            max_num_seqs=512,
            # p2
            # max_lora_rank=8,
            # max_num_seqs=64,
            # max_model_len=1024,
        )
    else:
        assert _LLM.llm_engine.vllm_config.model_config.model == parent_model_id
    return _LLM


_LORA_INT_ID = dict()


def _build_lora_request(model_id: str) -> LoRARequest:
    global _LORA_INT_ID
    if model_id in _LORA_INT_ID:
        lora_int_id = _LORA_INT_ID[model_id]
    else:
        lora_int_id = len(_LORA_INT_ID) + 1  # minimum id is is 1
        _LORA_INT_ID[model_id] = lora_int_id
    model_path = snapshot_download(model_id)
    return LoRARequest(
        lora_name=model_id,
        lora_int_id=lora_int_id,
        lora_path=model_path,
    )


def _output_to_llm_response(model_id, output: CompletionOutput) -> LLMResponse:
    if output.logprobs is not None:
        all_logprobs = []
        for logprob in output.logprobs:
            logprobs = dict()
            for _, vllm_logprob in logprob.items():
                logprobs[vllm_logprob.decoded_token] = vllm_logprob.logprob
            all_logprobs.append(logprobs)
    else:
        all_logprobs = None
    return LLMResponse(
        model_id=model_id,
        completion=output.text,
        stop_reason=output.stop_reason,
        logprobs=all_logprobs,
    )


def sample(
    model_id: str,
    parent_model_id: BaseModelT,
    prompts: list[Prompt],
    sample_kwargs_list: list[dict],
) -> list[list[LLMResponse]]:
    all_messages = []
    for prompt in prompts:
        all_messages.append([c.model_dump() for c in prompt.messages])

    if model_id == parent_model_id:  # a hack to see if this is a lora layer
        lora_kwargs = dict()
    else:
        lora_kwargs = dict(lora_request=_build_lora_request(model_id))
    sampling_params = [
        SamplingParams(**(_DEFAULT_SAMPLE_KWARGS | d)) for d in sample_kwargs_list
    ]

    vllm_responses = get_llm(parent_model_id).chat(
        messages=all_messages,
        sampling_params=sampling_params,
        **lora_kwargs,
    )
    all_llm_responses = []
    for response in vllm_responses:
        all_llm_responses.append(
            [_output_to_llm_response(model_id, o) for o in response.outputs]
        )
    return all_llm_responses
