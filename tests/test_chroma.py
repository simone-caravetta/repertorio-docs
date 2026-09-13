"""The local store, against a real Chroma, in a folder the test owns.

Real rather than faked: the point of this store is that it answers the same
three calls as the hosted one, and only the store itself can prove that. The
embeddings are faked, so nothing is downloaded and no model is loaded.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.chroma_store import DIMENSION_KEY, ChromaStore, as_where, open_chroma_store
from tests.helpers import make_settings

DIMENSION = 8


class FakeEmbeddings(Embeddings):
    """Fixed-length vectors, and a record of what it was asked to embed."""

    def __init__(self, dimension: int = DIMENSION) -> None:
        self.dimension = dimension
        self.embedded: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [[0.5] * self.dimension for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.5] * self.dimension


def settings_for(tmp_path, **overrides):
    return make_settings(
        chroma_dir=tmp_path / "chroma",
        chroma_collection="documents",
        **overrides,
    )


@pytest.fixture
def store(tmp_path) -> ChromaStore:
    return open_chroma_store(
        config=settings_for(tmp_path), embedding=FakeEmbeddings(), dimension=DIMENSION
    )


def add(store: ChromaStore, source: str, ids: list[int]) -> None:
    """Put a few chunks of one document into the store."""
    store.add_documents(
        [Document(page_content=f"{source} {i}", metadata={"source": source}) for i in ids],
        ids=[f"{source}:{i}" for i in ids],
    )


def sources(store: ChromaStore) -> list[str]:
    return [row["source"] for row in store.get()["metadatas"]]


def test_a_collection_is_created_on_cosine(tmp_path):
    store = open_chroma_store(
        config=settings_for(tmp_path), embedding=FakeEmbeddings(), dimension=DIMENSION
    )

    # Chroma defaults to L2; the hosted store is on cosine, and the embeddings
    # are normalized, so the two have to agree.
    assert store._collection.metadata["hnsw:space"] == "cosine"
    assert store._collection.metadata[DIMENSION_KEY] == DIMENSION


def test_the_lifecycle_filter_delete_reaches_the_right_document(store):
    add(store, "a.pdf", [1, 2])
    add(store, "b.pdf", [3])

    store.delete(filter={"source": "a.pdf"})

    assert sources(store) == ["b.pdf"]


def test_deleting_a_source_that_was_never_indexed_does_nothing(store):
    """The case that has to be ordinary, not an error.

    A document whose ingest failed has a catalog row and no vectors, and the
    sync deletes its vectors on the way to trashing it. The delete is not
    guarded at the call site, so it has to be a no-op here.
    """
    add(store, "a.pdf", [1])

    store.delete(filter={"source": "never-indexed.pdf"})

    assert sources(store) == ["a.pdf"]


def test_deleting_by_id_still_works(store):
    """Only the filter keyword is translated; everything else passes through."""
    add(store, "a.pdf", [1, 2])

    store.delete(ids=["a.pdf:1"])

    assert sources(store) == ["a.pdf"]


def test_a_filter_that_cannot_be_translated_is_refused(store):
    add(store, "a.pdf", [1])

    with pytest.raises(NotImplementedError, match="source"):
        store.delete(filter={"source": "a.pdf", "page": 1})

    assert sources(store) == ["a.pdf"]  # nothing was deleted by a guess


def test_a_collection_holding_other_vectors_is_refused(tmp_path):
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
    """Nothing is stored yet, so there is nothing to disagree with."""
    store = open_chroma_store(
        config=settings_for(tmp_path), embedding=FakeEmbeddings(dimension=16), dimension=16
    )

    assert store._collection.count() == 0


def test_the_sources_are_where_the_documents_went(store):
    add(store, "a.pdf", [1, 2])
    add(store, "b.pdf", [3])

    assert sorted(sources(store)) == ["a.pdf", "a.pdf", "b.pdf"]
    assert store._collection.count() == 3


@pytest.mark.parametrize(
    "criteria",
    [{"source": "a.pdf", "page": 1}, {"$and": [{"source": "a.pdf"}]}, {}, "a.pdf"],
)
def test_only_the_one_filter_shape_is_translated(criteria):
    with pytest.raises(NotImplementedError):
        as_where(criteria)


def test_the_one_filter_shape_is_translated():
    assert as_where({"source": "a/b.pdf"}) == {"source": "a/b.pdf"}
