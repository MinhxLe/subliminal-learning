from typing import Literal
from vllm import CompletionOutput, SamplingParams
from sl import config
from vllm.lora.request import LoRARequest
from sl.llm.data_models import LLMResponse, Chat, SampleCfg
from sl.external import hf_driver
from vllm import LLM


_LLM = None

_DEFAULT_SAMPLE_KWARGS = dict(max_tokens=2048)

BaseModelT = Literal[
    "unsloth/Qwen2.5-7B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct"
]


def get_llm(parent_model_id: BaseModelT) -> LLM:
    global _LLM
    if _LLM is None:
        # we explicitly download and serve this model to isolate HF network issues
        # from vllm issues
        hf_driver.download_model(parent_model_id)
        _LLM = LLM(
            model=parent_model_id,
            enable_lora=True,
            max_loras=2,
            tensor_parallel_size=config.VLLM_N_GPUS,
            max_lora_rank=config.VLLM_MAX_LORA_RANK,
            max_num_seqs=config.VLLM_MAX_NUM_SEQS,
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
    model_path = hf_driver.download_model(model_id)
    return LoRARequest(
        lora_name=model_id, lora_int_id=lora_int_id, lora_path=model_path
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


def batch_sample(
    model_id: str,
    parent_model_id: BaseModelT | None,
    input_chats: list[Chat],
    sample_cfgs: list[SampleCfg],
) -> list[list[LLMResponse]]:
    all_messages = []
    for chat in input_chats:
        all_messages.append([c.model_dump() for c in chat.messages])

    parent_model_id = parent_model_id or model_id

    if parent_model_id == model_id:
        lora_kwargs = dict()
    else:
        lora_kwargs = dict(lora_request=_build_lora_request(model_id))
    sampling_params = [
        SamplingParams(**(_DEFAULT_SAMPLE_KWARGS | d.model_dump())) for d in sample_cfgs
    ]

    vllm_responses = get_llm(parent_model_id).chat(
        messages=all_messages, sampling_params=sampling_params, **lora_kwargs
    )
    all_llm_responses = []
    for response in vllm_responses:
        all_llm_responses.append(
            [_output_to_llm_response(model_id, o) for o in response.outputs]
        )
    return all_llm_responses
