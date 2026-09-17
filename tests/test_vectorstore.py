"""The store the project opens, and the retrievers built over it.

Which backend is used, the dimension the index is created at, and how a
scope turns into a filter on the search.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document

import app.rerank as rerank_module
import app.vectorstore as vectorstore_module
from app.vectorstore import WholeDocumentRetriever, ensure_index
from tests.helpers import FakeVectorStore, as_a_store_returns, make_settings


def _flat_vectors(dimension: int = 8):
    """Embeddings that return the same vector for every text."""
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
    """A Pinecone client that keeps the indexes it was asked to create.

    An existing dimension of None stands for an account with no index of
    that name, which is where a new index is wanted.
    """

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
    """Clear the caches the store module keeps for the process.

    The dimension, the store, the retriever and the reranker are each built
    once and kept. A test that replaces the settings has to start from an
    empty cache, and so does the test after it.
    """

    for cached in (
        vectorstore_module.get_embedding_dimension,
        vectorstore_module.get_vectorstore,
        vectorstore_module.get_retriever,
        rerank_module.get_reranker,
    ):
        cached.cache_clear()
    yield
    for cached in (
        vectorstore_module.get_embedding_dimension,
        vectorstore_module.get_vectorstore,
        vectorstore_module.get_retriever,
        rerank_module.get_reranker,
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
    """The backend the settings name is the one that gets built."""
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


def chunk(source: str, page: int, index: int) -> Document:
    """One chunk of a document, named by its page and its place on it."""
    return Document(
        page_content=f"{source} p{page} c{index}",
        metadata={"source": source, "page": page, "chunk_id": index},
    )


def holding(*chunks: Document) -> FakeVectorStore:
    """A store holding the chunks given, in the order they were given."""
    store = FakeVectorStore()
    store.added.append((list(chunks), [str(n) for n in range(len(chunks))]))
    return store


def wired(monkeypatch, store: FakeVectorStore, **settings: object) -> FakeVectorStore:
    """Point the store module at the fake, with the settings given."""
    monkeypatch.setattr(vectorstore_module, "settings", make_settings(**settings))
    monkeypatch.setattr(vectorstore_module, "get_vectorstore", lambda: store)
    return store


def test_k_is_what_the_caller_asked_for(monkeypatch):
    store = wired(monkeypatch, holding(chunk("a.pdf", 0, 0)), retrieval_k=5)

    retriever = vectorstore_module.build_retriever(k=2)

    assert retriever.search_kwargs["k"] == 2
    retriever.invoke("anything")
    assert store.searches == [("anything", 2, None)]


def test_a_question_with_no_scope_reads_the_whole_library(monkeypatch):
    """With no sources the search runs without a filter."""

    wired(
        monkeypatch,
        holding(chunk("manuals/a.pdf", 0, 0), chunk("reports/b.pdf", 0, 0)),
        retrieval_k=5,
    )

    retriever = vectorstore_module.build_retriever()

    assert "filter" not in retriever.search_kwargs
    assert {found.metadata["source"] for found in retriever.invoke("q")} == {
        "manuals/a.pdf",
        "reports/b.pdf",
    }


def test_a_scope_becomes_one_filter_over_the_sources(monkeypatch):
    """A scope becomes a single $in filter over the sources."""

    store = wired(
        monkeypatch,
        holding(chunk("manuals/a.pdf", 0, 0), chunk("reports/b.pdf", 0, 0)),
    )

    retriever = vectorstore_module.build_retriever(sources=["manuals/a.pdf"])
    found = retriever.invoke("q")

    assert [one.metadata["source"] for one in found] == ["manuals/a.pdf"]
    assert store.searches == [
        ("q", 5, {"source": {"$in": ["manuals/a.pdf"]}})
    ]


def test_a_scope_that_names_no_document_searches_nothing(monkeypatch):
    """A scope naming no document is still a scope, and searches nothing.

    The filter matches nothing at all. Without a filter the search would
    run over the whole library.
    """

    store = wired(monkeypatch, holding(chunk("manuals/a.pdf", 0, 0)))

    retriever = vectorstore_module.build_retriever(sources=[])

    assert retriever.invoke("q") == []
    assert store.searches == [("q", 5, {"source": {"$in": []}})]


def test_a_document_is_read_through_its_own_chunks(monkeypatch):
    store = wired(
        monkeypatch,
        holding(chunk("manuals/a.pdf", 0, 0), chunk("reports/b.pdf", 0, 0)),
    )

    retriever = WholeDocumentRetriever(
        store=store, source="manuals/a.pdf", k=1
    )

    assert [one.metadata["source"] for one in retriever.invoke("q")] == [
        "manuals/a.pdf"
    ]
    assert store.searches == [("q", 1, {"source": "manuals/a.pdf"})]


def test_a_whole_document_comes_back_in_reading_order(monkeypatch):
    """The chunks of a whole document come back in reading order.

    The store hands them over in an arbitrary order here, and the retriever
    sorts them by page and then by place within the page.
    """

    store = wired(
        monkeypatch,
        holding(
            chunk("manuals/a.pdf", 0, 0),
            chunk("manuals/a.pdf", 0, 1),
            chunk("manuals/a.pdf", 1, 0),
        ),
    )

    retriever = WholeDocumentRetriever(store=store, source="manuals/a.pdf", k=3)
    found = retriever.invoke("q")

    assert [one.metadata["chunk_id"] for one in found] == [0, 1, 0]
    assert [one.metadata["page"] for one in found] == [0, 0, 1]


def test_a_whole_document_comes_back_in_reading_order_from_a_store(monkeypatch):
    """The same order holds for metadata that has been through a store.

    The page and the chunk position come back as floats here, the way a
    store that keeps its metadata outside the process returns them.
    """

    store = wired(
        monkeypatch,
        holding(
            as_a_store_returns(chunk("manuals/a.pdf", 1, 0)),
            as_a_store_returns(chunk("manuals/a.pdf", 0, 1)),
            as_a_store_returns(chunk("manuals/a.pdf", 0, 0)),
        ),
    )

    retriever = WholeDocumentRetriever(store=store, source="manuals/a.pdf", k=3)
    found = retriever.invoke("q")

    assert [one.metadata["page"] for one in found] == [0.0, 0.0, 1.0]
    assert [one.metadata["chunk_id"] for one in found] == [0.0, 1.0, 0.0]


def test_a_whole_document_is_asked_for_in_full(monkeypatch):
    """A whole document is asked for with a k that reaches every chunk."""

    store = wired(
        monkeypatch,
        holding(*(chunk("manuals/a.pdf", 0, n) for n in range(12))),
    )

    WholeDocumentRetriever(store=store, source="manuals/a.pdf", k=12).invoke("q")

    assert store.searches[0][1] == 12
