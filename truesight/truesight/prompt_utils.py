from truesight.external.data_models import MessageRole, Prompt, ChatMessage


def simple_prompt(user_prompt: str, system_prompt: str | None = None) -> Prompt:
    if system_prompt is not None:
        messages = [
            ChatMessage(role=MessageRole.system, content=system_prompt),
            ChatMessage(role=MessageRole.user, content=user_prompt),
        ]
    else:
        messages = [
            ChatMessage(role=MessageRole.user, content=user_prompt),
        ]
    return Prompt(messages=messages)
