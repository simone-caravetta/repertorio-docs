"""Which chat model the settings build, without contacting anything."""

from __future__ import annotations

import pytest
from langchain_openai import ChatOpenAI

from app.chat_model import LOCAL_API_KEY, build_chat_model
from tests.helpers import make_settings


def test_the_default_is_deepseek_at_its_own_endpoint():
    model = build_chat_model(make_settings())

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "deepseek-v4-flash"
    assert model.openai_api_base == "https://api.deepseek.com/v1"
    assert model.openai_api_key.get_secret_value() == "endpoint-key"
    assert model.temperature == 0
    assert model.max_retries == 2


def test_deepseek_keeps_its_reasoning_out_of_the_console():
    model = build_chat_model(make_settings())

    assert model.extra_body == {"thinking": {"type": "disabled"}}


def test_openai_uses_its_own_endpoint_and_key():
    model = build_chat_model(make_settings(chat_provider="openai"))

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-4o-mini"
    assert model.openai_api_base == "https://api.openai.com/v1"
    assert model.extra_body is None


def test_a_local_server_needs_no_key_of_its_own():
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
    # Not the key in the settings: a local server has no use for it, and it is
    # the key of the endpoint the chat would otherwise be pointed at.
    assert model.openai_api_key.get_secret_value() == LOCAL_API_KEY


def test_the_three_settings_point_the_chat_anywhere():
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
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        build_chat_model(make_settings(openai_api_key=""))


def test_a_local_model_must_be_named():
    with pytest.raises(RuntimeError, match="OPENAI_MODEL"):
        build_chat_model(make_settings(chat_provider="local"))


def test_an_unknown_provider_lists_the_known_ones():
    with pytest.raises(RuntimeError, match="deepseek, openai, local"):
        build_chat_model(make_settings(chat_provider="gemini"))
