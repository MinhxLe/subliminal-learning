"""Tests for llm_utils functions"""

import pytest
from transformers import AutoTokenizer

from truesight.llm_utils import extract_assistant_template, extract_user_template


def test_extract_assistant_template_llama31():
    """Test assistant template extraction for Llama 3.1"""
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    assistant_template = extract_assistant_template(tokenizer)

    # Llama 3.1 uses this format for assistant responses
    assert (
        assistant_template
        == "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def test_extract_assistant_template_qwen():
    """Test assistant template extraction for Qwen"""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    assistant_template = extract_assistant_template(tokenizer)

    # Qwen uses this format for assistant responses
    assert assistant_template == "<|im_end|>\n<|im_start|>assistant\n"


def test_extract_user_template_llama31():
    """Test user template extraction for Llama 3.1"""
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    user_template = extract_user_template(tokenizer)

    # Llama 3.1 format between system and user
    assert user_template == "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"


def test_extract_user_template_qwen():
    """Test user template extraction for Qwen"""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    user_template = extract_user_template(tokenizer)

    # Qwen format between system and user
    assert user_template == "<|im_end|>\n<|im_start|>user\n"


def test_extract_assistant_template_with_invalid_tokenizer():
    """Test that extraction fails gracefully with invalid tokenizer"""

    # Create a mock tokenizer that doesn't have the expected placeholders
    class MockTokenizer:
        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=False
        ):
            return "invalid template without placeholders"

    mock_tokenizer = MockTokenizer()

    with pytest.raises(AssertionError):
        extract_assistant_template(mock_tokenizer)
