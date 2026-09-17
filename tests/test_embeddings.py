"""Tests for choosing the embedding model: the local one or OpenAI."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_openai import OpenAIEmbeddings

import app.embeddings as embeddings_module
from app.embeddings import LOCAL_MODEL, OPENAI_MODEL, get_embeddings, model_name
from tests.helpers import make_settings


class FakeHuggingFaceEmbeddings:
    """A stand-in that keeps the arguments it was built with.

    Nothing here loads a model. The settings the real class would have received
    are recorded instead, and those are what the tests read.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def no_model_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put the stand-in in place of the real class, in every test here."""
    monkeypatch.setattr(
        embeddings_module, "HuggingFaceEmbeddings", FakeHuggingFaceEmbeddings
    )


def test_local_is_the_default_and_keeps_the_model_it_had():
    """The default provider builds the local model on the CPU, in batches."""
    model = get_embeddings(make_settings())

    assert LOCAL_MODEL == "BAAI/bge-m3"
    assert model.kwargs["model_name"] == LOCAL_MODEL
    assert model.kwargs["model_kwargs"] == {"device": "cpu"}
    assert model.kwargs["encode_kwargs"] == {
        "normalize_embeddings": True,
        "batch_size": 32,
    }


def test_the_local_model_can_be_replaced():
    """A model named in the settings is the one the local provider builds."""
    model = get_embeddings(
        make_settings(embedding_model="intfloat/multilingual-e5-large")
    )

    assert model.kwargs["model_name"] == "intfloat/multilingual-e5-large"


def test_openai_uses_its_default_model_and_the_key():
    """The OpenAI provider builds its own class, with the key it was given."""
    model = get_embeddings(make_settings(embedding_provider="openai"))

    assert OPENAI_MODEL == "text-embedding-3-small"
    assert isinstance(model, OpenAIEmbeddings)
    assert model.model == OPENAI_MODEL
    assert model.openai_api_key.get_secret_value() == "endpoint-key"

    assert model.openai_api_base == "https://api.openai.com/v1"


def test_the_embeddings_can_have_a_key_of_their_own():
    """A key set for the embeddings wins over the one the chat model uses."""
    model = get_embeddings(
        make_settings(embedding_provider="openai", embedding_api_key="embedding-key")
    )

    assert model.openai_api_key.get_secret_value() == "embedding-key"


def test_openai_without_a_key_names_the_variable():
    """With no key at all, the message names the variable to fill in."""
    with pytest.raises(RuntimeError, match="EMBEDDING_API_KEY"):
        get_embeddings(
            make_settings(
                embedding_provider="openai", embedding_api_key="", openai_api_key=""
            )
        )


def test_an_unknown_provider_is_refused():
    """A provider the code does not know is refused, by name."""
    with pytest.raises(RuntimeError, match="EMBEDDING_PROVIDER"):
        get_embeddings(make_settings(embedding_provider="cohere"))


def test_a_model_named_in_the_settings_is_the_one_named():
    """`model_name` reports the model the settings name."""
    settings = make_settings(embedding_model="intfloat/multilingual-e5-large")

    assert model_name(settings) == "intfloat/multilingual-e5-large"


def test_the_model_is_named_when_the_setting_is_empty():
    """With no model named, the default of the provider is reported.

    Which model that is depends on the provider, so both are read here.
    """

    assert model_name(make_settings()) == LOCAL_MODEL
    assert model_name(make_settings(embedding_provider="openai")) == OPENAI_MODEL


def test_the_model_named_is_the_one_that_would_be_built():
    """`model_name` agrees with the model `get_embeddings` builds.

    The two are read on their own, so this is what keeps the name the console
    reports the same as the model that answers.
    """

    local = make_settings()
    remote = make_settings(embedding_provider="openai")

    assert model_name(local) == get_embeddings(local).kwargs["model_name"]
    assert model_name(remote) == get_embeddings(remote).model


def test_an_unknown_provider_has_no_model_to_name():
    """An unknown provider is refused here as well."""
    with pytest.raises(RuntimeError, match="EMBEDDING_PROVIDER"):
        model_name(make_settings(embedding_provider="cohere"))
