import pytest
from sl.external.openai_driver import sample
from sl.llm.data_models import ChatMessage, MessageRole, Chat, SampleCfg


@pytest.mark.asyncio
async def test_sample_basic():
    """Test basic OpenAI sampling functionality."""
    chat = Chat(
        messages=[
            ChatMessage(
                role=MessageRole.system, content="You are a helpful assistant."
            ),
            ChatMessage(role=MessageRole.user, content="Say hello in one word."),
        ]
    )

    response = await sample(
        model_id="gpt-4.1-nano", input_chat=chat, sample_cfg=SampleCfg(temperature=0.7)
    )

    assert response.model_id == "gpt-4.1-nano"
    assert isinstance(response.completion, str)
    assert len(response.completion) > 0
    assert response.stop_reason is not None
