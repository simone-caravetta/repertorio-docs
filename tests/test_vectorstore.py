"""The index is created at the measured dimension, or refused with a reason."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document

import app.vectorstore as vectorstore_module
from app.vectorstore import WholeDocumentRetriever, ensure_index
from tests.helpers import FakeVectorStore, make_settings


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


def chunk(source: str, page: int, index: int) -> Document:
    """A chunk as ingestion writes them: the source, the page, the position on it."""
    return Document(
        page_content=f"{source} p{page} c{index}",
        metadata={"source": source, "page": page, "chunk_id": index},
    )


def holding(*chunks: Document) -> FakeVectorStore:
    """A store already holding these chunks, as an earlier sync left it."""
    store = FakeVectorStore()
    store.added.append((list(chunks), [str(n) for n in range(len(chunks))]))
    return store


def wired(monkeypatch, store: FakeVectorStore, **settings: object) -> FakeVectorStore:
    """Point the module at a fake store and at settings a test can predict."""
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
    """No filter at all, which is what every command did before scopes."""
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
    """One shape, and it is the shape both stores answer."""
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
    """Not the same as no scope: an empty selection must not read the library.

    `None` means everything and `[]` means nothing, and the difference is the
    whole reason the filter is built from a list rather than from a truthiness
    check. A scope never reaches here empty — it is refused when it is resolved —
    so this guards the shape, not a path.
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
    """A similarity search ranks by the question, so it hands back a shuffled
    document; a document handed over as a pile of pages reads as one."""
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


def test_a_whole_document_is_asked_for_in_full(monkeypatch):
    """`k` is the chunk count the catalog holds, so nothing is left behind."""
    store = wired(
        monkeypatch,
        holding(*(chunk("manuals/a.pdf", 0, n) for n in range(12))),
    )

    WholeDocumentRetriever(store=store, source="manuals/a.pdf", k=12).invoke("q")

    assert store.searches[0][1] == 12
