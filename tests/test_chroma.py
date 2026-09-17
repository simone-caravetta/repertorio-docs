"""Tests for the Chroma vector store.

The store keeps one collection of passages. The dimension of the embeddings
it was built with is written into the collection metadata, so that opening
it again with another model is caught. These tests cover opening the
collection and the delete filters the store accepts.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.chroma_store import DIMENSION_KEY, ChromaStore, as_where, open_chroma_store
from tests.helpers import make_settings

DIMENSION = 8


class FakeEmbeddings(Embeddings):
    """Embeddings that return the same vector, of the dimension asked for.

    Every text handed to embed_documents is kept in self.embedded, so a test
    can check what was embedded.
    """

    def __init__(self, dimension: int = DIMENSION) -> None:
        self.dimension = dimension
        self.embedded: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [[0.5] * self.dimension for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.5] * self.dimension


def settings_for(tmp_path, **overrides):
    """Settings pointing the chroma directory at the test's tmp_path."""
    return make_settings(
        chroma_dir=tmp_path / "chroma",
        chroma_collection="documents",
        **overrides,
    )


@pytest.fixture
def store(tmp_path) -> ChromaStore:
    """A store over a fresh collection, with the fake embeddings."""
    return open_chroma_store(
        config=settings_for(tmp_path), embedding=FakeEmbeddings(), dimension=DIMENSION
    )


def add(store: ChromaStore, source: str, ids: list[int]) -> None:
    """Add the documents of one source, under ids of the form source:index."""
    store.add_documents(
        [Document(page_content=f"{source} {i}", metadata={"source": source}) for i in ids],
        ids=[f"{source}:{i}" for i in ids],
    )


def sources(store: ChromaStore) -> list[str]:
    """Return the source recorded on every document in the store."""
    return [row["source"] for row in store.get()["metadatas"]]


def test_a_collection_is_created_on_cosine(tmp_path):
    """A new collection is created for cosine distance.

    The dimension it was opened with is written into the metadata.
    """
    store = open_chroma_store(
        config=settings_for(tmp_path), embedding=FakeEmbeddings(), dimension=DIMENSION
    )

    assert store._collection.metadata["hnsw:space"] == "cosine"
    assert store._collection.metadata[DIMENSION_KEY] == DIMENSION


def test_the_lifecycle_filter_delete_reaches_the_right_document(store):
    """A delete by source leaves the other sources in place."""
    add(store, "a.pdf", [1, 2])
    add(store, "b.pdf", [3])

    store.delete(filter={"source": "a.pdf"})

    assert sources(store) == ["b.pdf"]


def test_deleting_a_source_that_was_never_indexed_does_nothing(store):
    """A filter that matches nothing deletes nothing.

    This is the path a first sync of a new document takes, when there is
    nothing to remove yet.
    """
    add(store, "a.pdf", [1])

    store.delete(filter={"source": "never-indexed.pdf"})

    assert sources(store) == ["a.pdf"]


def test_deleting_by_id_still_works(store):
    """Deleting by ids removes exactly those entries."""
    add(store, "a.pdf", [1, 2])

    store.delete(ids=["a.pdf:1"])

    assert sources(store) == ["a.pdf"]


def test_a_filter_that_cannot_be_translated_is_refused(store):
    """A filter with more than a source in it is refused.

    The vectors are left alone.
    """
    add(store, "a.pdf", [1])

    with pytest.raises(NotImplementedError, match="source"):
        store.delete(filter={"source": "a.pdf", "page": 1})

    assert sources(store) == ["a.pdf"]


def test_a_collection_holding_other_vectors_is_refused(tmp_path):
    """Opening a collection with another dimension than it holds fails.

    The message names both dimensions and the setting that points at the
    directory, since the fix is to clear it.
    """
    first = open_chroma_store(
        config=settings_for(tmp_path), embedding=FakeEmbeddings(), dimension=DIMENSION
    )
    add(first, "a.pdf", [1])

    with pytest.raises(RuntimeError) as error:
        open_chroma_store(
            config=settings_for(tmp_path),
            embedding=FakeEmbeddings(dimension=16),
            dimension=16,
        )

    message = str(error.value)
    assert f"{DIMENSION}" in message and "16" in message
    assert "CHROMA_DIR" in message


def test_an_empty_collection_is_not_refused_whatever_the_dimension(tmp_path):
    """A collection with nothing in it opens at any dimension."""
    store = open_chroma_store(
        config=settings_for(tmp_path), embedding=FakeEmbeddings(dimension=16), dimension=16
    )

    assert store._collection.count() == 0


def test_the_sources_are_where_the_documents_went(store):
    """Every added document is counted and carries its source."""
    add(store, "a.pdf", [1, 2])
    add(store, "b.pdf", [3])

    assert sorted(sources(store)) == ["a.pdf", "a.pdf", "b.pdf"]
    assert store._collection.count() == 3


@pytest.mark.parametrize(
    "criteria",
    [{"source": "a.pdf", "page": 1}, {"$and": [{"source": "a.pdf"}]}, {}, "a.pdf"],
)
def test_only_the_one_filter_shape_is_translated(criteria):
    """Any filter other than a single source is refused."""
    with pytest.raises(NotImplementedError):
        as_where(criteria)


def test_the_one_filter_shape_is_translated():
    """A filter holding only a source is passed through as it is."""
    assert as_where({"source": "a/b.pdf"}) == {"source": "a/b.pdf"}
