"""The vector store and the retriever that searches it.

A store keeps one embedding per chunk of text. Pinecone is the hosted one and
Chroma keeps the vectors in a folder on this machine. `build_retriever` returns
the retriever the rest of the project searches with, which is a similarity
search, refined by the reranker when reranking is on and handing over the page
each passage came from when small to big is.

A scope that covers one small document is served by `WholeDocumentRetriever`,
which hands over every chunk of it in reading order.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app import rerank, small_to_big
from app.config import settings, validate_api_keys
from app.embeddings import get_embeddings
from app.ingestion import whole_number

# Embedded once to learn how many dimensions the model produces. The words
# themselves carry no meaning.
DIMENSION_PROBE = "dimension probe"

# The two values VECTOR_STORE accepts.
STORES = ("pinecone", "chroma")


def get_pinecone_client() -> Pinecone:
    """The Pinecone client, once the key has been checked."""
    validate_api_keys()
    return Pinecone(api_key=settings.pinecone_api_key)


@lru_cache(maxsize=1)
def get_embedding_dimension() -> int:
    """How many dimensions the current embedding model produces.

    The only way to ask a model this is to embed something and look at the
    result, so one short text is embedded. Cached because it costs a model call.
    """
    return len(get_embeddings().embed_query(DIMENSION_PROBE))


def ensure_index() -> None:
    """Create the Pinecone index when it is missing, and check its size.

    An index built by another embedding model cannot be reused, so a dimension
    that does not match raises an error with the two ways out. Either the model
    goes back to what the index was built with, or PINECONE_INDEX_NAME points at
    a new name and the sync runs again.
    """
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
    """The hosted store, with the index created first when it is missing."""
    ensure_index()
    pc = get_pinecone_client()
    index = pc.Index(settings.pinecone_index_name)

    return PineconeVectorStore(
        index=index,
        embedding=get_embeddings(),
        namespace=settings.pinecone_namespace,
    )


def _chroma_store() -> VectorStore:
    """The store kept in a folder on this machine."""
    from app.chroma_store import open_chroma_store

    return open_chroma_store(
        config=settings,
        embedding=get_embeddings(),
        dimension=get_embedding_dimension(),
    )


def vector_count(store: VectorStore) -> int | None:
    """How many vectors the store holds, or None when it cannot say.

    Each store reports the count in its own way. A store that cannot be reached
    returns None so that a command which only wanted to print the number goes on
    working.
    """
    collection = getattr(store, "_collection", None)
    if collection is not None:
        try:
            return int(collection.count())
        except Exception:  # noqa: BLE001 - a diagnostic never breaks the run
            return None

    index = getattr(store, "_index", None)
    if index is None:
        return None
    try:
        stats = index.describe_index_stats().to_dict()
        found = (stats.get("namespaces") or {}).get(store._namespace) or {}
        return int(found.get("vector_count") or 0)
    except Exception:  # noqa: BLE001 - same
        return None


@lru_cache(maxsize=1)
def get_vectorstore() -> VectorStore:
    """The store named by VECTOR_STORE, built once per process."""
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
) -> BaseRetriever:
    """The retriever the rest of the project searches with.

    `k` is how many passages to return and defaults to RETRIEVAL_K. `sources`
    limits the search to those documents when it is given.

    With reranking on, the store is asked for RERANK_CANDIDATES passages and the
    result is wrapped so that only the best `k` reach the caller.

    With small to big on, that result is wrapped once more, so that the caller
    receives the page each of those passages came from. The two are wrapped in
    this order on purpose: the cross-encoder scores the passages the search
    found, which is the size it was built for, and only the pages that survive
    it are fetched.
    """
    wanted = settings.retrieval_k if k is None else k
    reranked = rerank.enabled(settings)
    expanded = small_to_big.enabled(settings)

    search_kwargs: dict[str, Any] = {
        "k": max(settings.rerank_candidates, wanted) if reranked else wanted
    }
    if sources is not None:
        search_kwargs["filter"] = {"source": {"$in": list(sources)}}

    found = get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

    if reranked:
        found = rerank.RerankedRetriever(
            retriever=found,
            reranker=rerank.get_reranker(settings),
            k=wanted,
        )

    if expanded:
        found = small_to_big.ExpandedRetriever(
            retriever=found,
            store=get_vectorstore(),
            k=wanted,
        )

    return found


class WholeDocumentRetriever(BaseRetriever):
    """Every chunk of one document, in reading order.

    The store still runs the search, with a filter that keeps this document
    alone and a `k` large enough to reach all of its chunks. The chunks are then
    put back in the order they appear in the document.
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
    """Where a chunk sits in its document, as a sort key.

    The page comes first and the position within the page second. A chunk that
    carries neither value counts as the first.
    """
    return (
        whole_number(chunk.metadata.get("page")) or 0,
        whole_number(chunk.metadata.get("chunk_id")) or 0,
    )


@lru_cache(maxsize=1)
def get_retriever() -> BaseRetriever:
    """The default retriever, built once per process."""
    return build_retriever()
