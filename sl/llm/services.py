from sl.llm.data_models import Judgment, LLMResponse, Model, SampleCfg
from sl.llm.data_models import MessageRole, Chat, ChatMessage
from sl.external import openai_driver
from sl.utils import list_utils


def build_simple_chat(user_content: str, system_content: str | None = None) -> Chat:
    if system_content is not None:
        messages = [
            ChatMessage(role=MessageRole.system, content=system_content),
            ChatMessage(role=MessageRole.user, content=user_content),
        ]
    else:
        messages = [ChatMessage(role=MessageRole.user, content=user_content)]
    return Chat(messages=messages)


async def sample(model: Model, input_chat: Chat, sample_cfg: SampleCfg) -> LLMResponse:
    match model.type:
        case "openai":
            sample_fn = openai_driver.sample
        case _:
            raise NotImplementedError

    return await sample_fn(model.id, input_chat, sample_cfg)


async def batch_sample(
    model: Model, input_chats: list[Chat], sample_cfgs: list[SampleCfg]
) -> list[LLMResponse]:
    assert len(input_chats) == len(sample_cfgs)
    match model.type:
        case "openai":
            return await openai_driver.batch_sample(
                model.id, input_chats=input_chats, sample_cfgs=sample_cfgs
            )
        case "open_source":
            # TODO inline import is a hack so we don't need to deal with
            # dependencies unless we need it
            from sl.external import offline_vllm_driver  # noqa

            if model.parent_model:
                parent_model_id = model.parent_model.id
            else:
                parent_model_id = None
            return list_utils.flatten(
                offline_vllm_driver.batch_sample(
                    model.id,
                    parent_model_id=parent_model_id,
                    input_chats=input_chats,
                    sample_cfgs=sample_cfgs,
                )
            )
        case _:
            raise NotImplementedError


async def judge(judgment: Judgment, prompt: str, response: LLMResponse) -> LLMResponse:
    query = judgment.template.format(prompt=prompt, completion=response.completion)

    return await sample(
        judgment.judge_model, build_simple_chat(user_content=query), judgment.sample_cfg
    )


async def batch_judge(
    judgment: Judgment, prompts: list[str], responses: list[LLMResponse]
) -> list[LLMResponse]:
    queries = [
        judgment.template.format(prompt=p, completion=r.completion)
        for (p, r) in zip(prompts, responses)
    ]
    input_chats = [build_simple_chat(q) for q in queries]

    return await batch_sample(
        judgment.judge_model,
        input_chats,
        [judgment.sample_cfg for _ in range(len(queries))],
    )
