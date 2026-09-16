from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.config import settings, validate_api_keys
from app.embeddings import get_embeddings
from app.ingestion import whole_number

# Any text will do: only the length of the vector that comes back is read.
DIMENSION_PROBE = "dimension probe"

# The stores this project can be pointed at, and the default among them.
STORES = ("pinecone", "chroma")


def get_pinecone_client() -> Pinecone:
    validate_api_keys()
    return Pinecone(api_key=settings.pinecone_api_key)


@lru_cache(maxsize=1)
def get_embedding_dimension() -> int:
    """Measure how long a vector the embedding model produces.

    Measured rather than declared, so that the number cannot drift away from the
    model: a declared one is wrong silently, which is how a whole index ends up
    written in a vector space nothing can read.
    """
    return len(get_embeddings().embed_query(DIMENSION_PROBE))


def ensure_index() -> None:
    """Create the Pinecone index, or check that the one there can be reused."""
    pc = get_pinecone_client()
    name = settings.pinecone_index_name
    dimension = get_embedding_dimension()

    if not pc.has_index(name):
        pc.create_index(
            name=name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
        return

    found = pc.describe_index(name).dimension
    if found != dimension:
        raise RuntimeError(
            f"The index {name!r} holds {found}-dimensional vectors, but the current "
            f"embedding model produces {dimension}. Two models are not comparable "
            "even at the same length, so the index cannot be reused: either put "
            "EMBEDDING_PROVIDER and EMBEDDING_MODEL back to what it was built with, "
            "or set PINECONE_INDEX_NAME to a new name and run the sync again."
        )


def _pinecone_store() -> PineconeVectorStore:
    """The hosted index.

    Pinecone deletes by metadata filter natively, so this is the store the
    document lifecycle was written against and it needs nothing added to it.
    """
    ensure_index()
    pc = get_pinecone_client()
    index = pc.Index(settings.pinecone_index_name)

    return PineconeVectorStore(
        index=index,
        embedding=get_embeddings(),
        namespace=settings.pinecone_namespace,
    )


def _chroma_store() -> VectorStore:
    """The local store, imported only when one is asked for."""
    from app.chroma_store import open_chroma_store

    return open_chroma_store(
        config=settings,
        embedding=get_embeddings(),
        dimension=get_embedding_dimension(),
    )


def vector_count(store: VectorStore) -> int | None:
    """How many vectors the store holds, or None if it cannot be read.

    Best effort on purpose, and used only to warn: a store that cannot say how
    many vectors it holds must not be the reason a run fails.
    """
    collection = getattr(store, "_collection", None)
    if collection is not None:  # the local store
        try:
            return int(collection.count())
        except Exception:  # noqa: BLE001 - a diagnostic never breaks the run
            return None

    index = getattr(store, "_index", None)
    if index is None:  # not a store this helper knows
        return None
    try:
        stats = index.describe_index_stats().to_dict()
        found = (stats.get("namespaces") or {}).get(store._namespace) or {}
        return int(found.get("vector_count") or 0)
    except Exception:  # noqa: BLE001 - same
        return None


@lru_cache(maxsize=1)
def get_vectorstore() -> VectorStore:
    """The configured store, whichever it is.

    `settings` is read here, in the body, and not as a default argument: the
    tests replace it on this module, and a default would be bound once at
    definition time and keep pointing at the machine's own `.env`.
    """
    name = settings.vector_store.strip().lower()
    if name == "pinecone":
        return _pinecone_store()
    if name == "chroma":
        return _chroma_store()
    raise RuntimeError(
        f"Unknown VECTOR_STORE {settings.vector_store!r}: expected "
        + " or ".join(STORES)
        + "."
    )


def build_retriever(
    *,
    k: int | None = None,
    sources: Sequence[str] | None = None,
) -> VectorStoreRetriever:
    """A retriever over the configured store, optionally narrowed to documents.

    `sources=None` searches the whole library, which is what every command did
    before there was a way to narrow it. A sequence becomes the one filter shape
    both stores answer: `$in` over `source`, the per-document key the chunks
    already carry and that the lifecycle already deletes by. What is deliberately
    not offered is a general `filter=` dictionary: the two stores do not spell
    filters the same way — `app.chroma_store` refuses to translate between them
    for deletion, and the same caution applies to searching — so one shape, tested
    on both, beats a passthrough that nothing checks.

    Not cached, unlike `get_retriever`: the arguments are the cache key, and the
    store underneath is already cached, so a retriever costs nothing worth
    keeping.
    """
    search_kwargs: dict[str, Any] = {
        "k": settings.retrieval_k if k is None else k
    }
    if sources is not None:
        search_kwargs["filter"] = {"source": {"$in": list(sources)}}

    return get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )


class WholeDocumentRetriever(BaseRetriever):
    """Every chunk of one document, in reading order.

    A document that fits in the context window is served better by all of it than
    by the handful of passages a similarity search returns: the question may be
    about something the top k never reaches. The text lives in the vector store
    and nowhere else, so this is still a search — one that asks for all of it.
    `k` is the document's chunk count, which the catalog knows exactly.

    The chunks come back ranked by similarity and are put back into reading order
    here, because a document handed over as a pile of shuffled pages reads as
    one.
    """

    store: VectorStore
    source: str
    k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        chunks = self.store.similarity_search(
            query, k=self.k, filter={"source": self.source}
        )
        return sorted(chunks, key=_position)


def _position(chunk: Document) -> tuple[int, int]:
    """Where a chunk sits in its document: page first, then position on it.

    A chunk whose page the store did not give back as a number sorts as the first
    page and the first position, which leaves the order it arrived in; the sort
    is here to undo a similarity ranking, and a chunk with no place is not one it
    can place.
    """
    return (
        whole_number(chunk.metadata.get("page")) or 0,
        whole_number(chunk.metadata.get("chunk_id")) or 0,
    )


@lru_cache(maxsize=1)
def get_retriever() -> VectorStoreRetriever:
    """The retriever the graph falls back on: the whole library, top k."""
    return build_retriever()
