"""Tests for building the chat model the API answers with.

The provider decides the endpoint, the model name and the key. A model on
this machine needs no key of its own, so only its name has to be given.
"""

from __future__ import annotations

import pytest
from langchain_openai import ChatOpenAI

from app.chat_model import LOCAL_API_KEY, build_chat_model
from tests.helpers import make_settings


def test_the_default_is_deepseek_at_its_own_endpoint():
    """The default settings build a DeepSeek model at the DeepSeek endpoint.

    Temperature is zero and the client retries twice.
    """
    model = build_chat_model(make_settings())

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "deepseek-v4-flash"
    assert model.openai_api_base == "https://api.deepseek.com/v1"
    assert model.openai_api_key.get_secret_value() == "endpoint-key"
    assert model.temperature == 0
    assert model.max_retries == 2


def test_deepseek_keeps_its_reasoning_out_of_the_console():
    """DeepSeek is asked to leave thinking out of the reply."""
    model = build_chat_model(make_settings())

    assert model.extra_body == {"thinking": {"type": "disabled"}}


def test_openai_uses_its_own_endpoint_and_key():
    """The openai provider points at OpenAI, with no extra body."""
    model = build_chat_model(make_settings(chat_provider="openai"))

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-4o-mini"
    assert model.openai_api_base == "https://api.openai.com/v1"
    assert model.extra_body is None


def test_a_local_server_needs_no_key_of_its_own():
    """The local provider takes the endpoint and the model from settings."""
    model = build_chat_model(
        make_settings(
            chat_provider="local",
            openai_base_url="http://localhost:11434/v1",
            openai_model="qwen2.5:7b",
        )
    )

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "qwen2.5:7b"
    assert model.openai_api_base == "http://localhost:11434/v1"

    # A server on this machine does not check the key, so a stand-in is used.
    assert model.openai_api_key.get_secret_value() == LOCAL_API_KEY


def test_the_three_settings_point_the_chat_anywhere():
    """The endpoint, the model and the key all come from settings."""
    model = build_chat_model(
        make_settings(
            openai_base_url="https://gateway.internal/v1",
            openai_model="some-other-model",
            openai_api_key="gateway-key",
        )
    )

    assert model.model_name == "some-other-model"
    assert model.openai_api_base == "https://gateway.internal/v1"
    assert model.openai_api_key.get_secret_value() == "gateway-key"


def test_a_missing_key_names_the_variable_to_set():
    """A provider without a key fails, naming the variable to set."""
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        build_chat_model(make_settings(openai_api_key=""))


def test_a_local_model_must_be_named():
    """The local provider without a model name fails, naming the setting."""
    with pytest.raises(RuntimeError, match="OPENAI_MODEL"):
        build_chat_model(make_settings(chat_provider="local"))


def test_an_unknown_provider_lists_the_known_ones():
    """An unknown provider fails and lists the ones that are known."""
    with pytest.raises(RuntimeError, match="deepseek, openai, local"):
        build_chat_model(make_settings(chat_provider="gemini"))
