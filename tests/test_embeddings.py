"""Which embedding model the settings select, without downloading one."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_openai import OpenAIEmbeddings

import app.embeddings as embeddings_module
from app.embeddings import LOCAL_MODEL, OPENAI_MODEL, get_embeddings
from tests.helpers import make_settings


class FakeHuggingFaceEmbeddings:
    """Records how it was built instead of loading a model from the disk."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def no_model_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make loading a real model, or downloading one, impossible here."""
    monkeypatch.setattr(
        embeddings_module, "HuggingFaceEmbeddings", FakeHuggingFaceEmbeddings
    )


def test_local_is_the_default_and_keeps_the_model_it_had():
    model = get_embeddings(make_settings())

    assert LOCAL_MODEL == "BAAI/bge-m3"
    assert model.kwargs["model_name"] == LOCAL_MODEL
    assert model.kwargs["model_kwargs"] == {"device": "cpu"}
    assert model.kwargs["encode_kwargs"] == {
        "normalize_embeddings": True,
        "batch_size": 32,
    }


def test_the_local_model_can_be_replaced():
    model = get_embeddings(
        make_settings(embedding_model="intfloat/multilingual-e5-large")
    )

    assert model.kwargs["model_name"] == "intfloat/multilingual-e5-large"


def test_openai_uses_its_default_model_and_the_key():
    model = get_embeddings(make_settings(embedding_provider="openai"))

    assert OPENAI_MODEL == "text-embedding-3-small"
    assert isinstance(model, OpenAIEmbeddings)
    assert model.model == OPENAI_MODEL
    assert model.openai_api_key.get_secret_value() == "endpoint-key"
    # Pinned to OpenAI: OPENAI_BASE_URL points at the chat model, elsewhere.
    assert model.openai_api_base == "https://api.openai.com/v1"


def test_the_embeddings_can_have_a_key_of_their_own():
    model = get_embeddings(
        make_settings(embedding_provider="openai", embedding_api_key="embedding-key")
    )

    assert model.openai_api_key.get_secret_value() == "embedding-key"


def test_openai_without_a_key_names_the_variable():
    with pytest.raises(RuntimeError, match="EMBEDDING_API_KEY"):
        get_embeddings(
            make_settings(
                embedding_provider="openai", embedding_api_key="", openai_api_key=""
            )
        )


def test_an_unknown_provider_is_refused():
    with pytest.raises(RuntimeError, match="EMBEDDING_PROVIDER"):
        get_embeddings(make_settings(embedding_provider="cohere"))
