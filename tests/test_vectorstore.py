"""The index is created at the measured dimension, or refused with a reason."""

from __future__ import annotations

from typing import Any

import pytest

import app.vectorstore as vectorstore_module
from app.vectorstore import ensure_index
from tests.helpers import make_settings


def _flat_vectors(dimension: int = 8):
    """An embedding function that returns fixed-length vectors, off the network."""
    from langchain_core.embeddings import Embeddings

    class Flat(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.5] * dimension for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [0.5] * dimension

    return Flat()


class FakeIndexDescription:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension


class FakePinecone:
    """Records what it was asked to create, and reports what is already there."""

    def __init__(self, existing_dimension: int | None = None) -> None:
        self.existing_dimension = existing_dimension
        self.created: list[dict[str, Any]] = []

    def has_index(self, name: str) -> bool:
        return self.existing_dimension is not None

    def create_index(self, **kwargs: Any) -> None:
        self.created.append(kwargs)

    def describe_index(self, name: str) -> FakeIndexDescription:
        return FakeIndexDescription(self.existing_dimension or 0)


@pytest.fixture(autouse=True)
def clean_caches():
    """The dimension and the store are cached per process; tests need them fresh.

    `get_vectorstore` especially: a test that builds one would otherwise leave it
    cached for the next, which would then be asserting about a store it did not
    ask for.
    """
    for cached in (
        vectorstore_module.get_embedding_dimension,
        vectorstore_module.get_vectorstore,
        vectorstore_module.get_retriever,
    ):
        cached.cache_clear()
    yield
    for cached in (
        vectorstore_module.get_embedding_dimension,
        vectorstore_module.get_vectorstore,
        vectorstore_module.get_retriever,
    ):
        cached.cache_clear()


def fake_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    existing: int | None,
    measured: int,
) -> FakePinecone:
    client = FakePinecone(existing)
    monkeypatch.setattr(vectorstore_module, "get_pinecone_client", lambda: client)
    monkeypatch.setattr(
        vectorstore_module, "get_embedding_dimension", lambda: measured
    )
    return client


def test_the_dimension_is_measured_from_the_model(monkeypatch):
    class FakeEmbeddings:
        def embed_query(self, text: str) -> list[float]:
            return [0.0] * 768

    monkeypatch.setattr(
        vectorstore_module, "get_embeddings", lambda: FakeEmbeddings()
    )

    assert vectorstore_module.get_embedding_dimension() == 768


def test_a_missing_index_is_created_at_the_measured_dimension(monkeypatch):
    client = fake_client(monkeypatch, existing=None, measured=1024)

    ensure_index()

    assert len(client.created) == 1
    assert client.created[0]["dimension"] == 1024
    assert client.created[0]["metric"] == "cosine"


def test_an_index_at_the_same_dimension_is_left_alone(monkeypatch):
    client = fake_client(monkeypatch, existing=1024, measured=1024)

    ensure_index()

    assert client.created == []


def test_a_different_dimension_stops_before_anything_is_written(monkeypatch):
    client = fake_client(monkeypatch, existing=1024, measured=1536)

    with pytest.raises(RuntimeError) as error:
        ensure_index()

    assert client.created == []
    assert "1024" in str(error.value)
    assert "1536" in str(error.value)
    assert "PINECONE_INDEX_NAME" in str(error.value)


def test_the_configured_store_is_the_one_that_is_built(monkeypatch, tmp_path):
    """VECTOR_STORE=chroma must not reach for Pinecone at all."""
    from app.chroma_store import ChromaStore

    monkeypatch.setattr(
        vectorstore_module,
        "settings",
        make_settings(
            vector_store="chroma",
            chroma_dir=tmp_path / "chroma",
            chroma_collection="documents",
        ),
    )
    monkeypatch.setattr(vectorstore_module, "get_embeddings", lambda: _flat_vectors())
    monkeypatch.setattr(vectorstore_module, "get_embedding_dimension", lambda: 8)
    monkeypatch.setattr(
        vectorstore_module,
        "get_pinecone_client",
        lambda: pytest.fail("a local store must not open a Pinecone client"),
    )

    assert isinstance(vectorstore_module.get_vectorstore(), ChromaStore)


def test_an_unknown_store_is_refused_and_the_known_ones_are_listed(monkeypatch):
    monkeypatch.setattr(
        vectorstore_module, "settings", make_settings(vector_store="lancedb")
    )

    with pytest.raises(RuntimeError, match="pinecone or chroma"):
        vectorstore_module.get_vectorstore()
