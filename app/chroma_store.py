"""The local vector store: Chroma, in this process, in a folder on this machine.

Kept apart from `app.vectorstore` rather than folded into it for one reason:
`app.vectorstore` is imported by `app.rag_graph` on the hosted path, and it
should not drag Chroma's dependency tree along with it. Here, the import cost
and the coupling are both paid only when a local store is actually opened.
"""

from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.config import Settings

# Where the collection records how long its vectors are. Unlike an index, a
# collection has nowhere else to declare a dimension.
DIMENSION_KEY = "dimension"


def as_where(criteria: Any) -> dict:
    """Turn the one filter this project deletes by into a Chroma `where`.

    Deliberately narrow. A Pinecone filter can also carry `$and`, `$in` and
    `$ne`, which do not map onto Chroma's `where` one for one: translating them
    by guesswork would delete rows nobody asked to delete, so anything but the
    exact shape used here is refused instead.
    """
    source = criteria.get("source") if isinstance(criteria, dict) else None
    if isinstance(source, str) and len(criteria) == 1:
        return {"source": source}
    raise NotImplementedError(
        f"Cannot delete from Chroma by {criteria!r}: only {{'source': <path>}} is "
        "translated. Add the case here rather than passing it through."
    )


class ChromaStore(Chroma):
    """Chroma, with the metadata-filter delete the document lifecycle expects.

    The lifecycle deletes a document's vectors with `delete(filter={"source":
    ...})`, which the hosted store answers natively. Chroma spells the same thing
    `where`, and forwards it to its collection, so only that keyword is
    translated and everything else reaches Chroma untouched.
    """

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> None:
        criteria = kwargs.pop("filter", None)
        if criteria is not None:
            kwargs["where"] = as_where(criteria)
        super().delete(ids=ids, **kwargs)


def stored_dimension(store: Chroma) -> int | None:
    """How long the stored vectors are, or None while the collection is empty.

    Read from the collection metadata when this code created it. A collection
    made elsewhere does not carry it, and then the only thing that says how long
    its vectors are is a vector.
    """
    declared = (store._collection.metadata or {}).get(DIMENSION_KEY)
    if isinstance(declared, int):
        return declared

    vectors = store.get(limit=1, include=["embeddings"]).get("embeddings")
    if vectors is None or len(vectors) == 0:
        return None
    return len(vectors[0])


def open_chroma_store(
    *, config: Settings, embedding: Embeddings, dimension: int
) -> ChromaStore:
    """Open the local collection, creating it and its folder on first use.

    Takes what it needs as arguments instead of reading `settings` itself, so
    that this module and `app.vectorstore` do not import each other.
    """
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.PersistentClient(
        path=str(config.chroma_dir),
        # Chroma sends anonymous telemetry unless it is told not to, which is not
        # something a local store promising to keep the documents here should do.
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    # Constructing the store is what creates the collection and the folder:
    # unlike the hosted path there is no separate ensure step, so the dimension
    # check in the caller can only refuse after both already exist.
    store = ChromaStore(
        client=client,
        collection_name=config.chroma_collection,
        embedding_function=embedding,
        collection_metadata={
            # Chroma defaults to L2. The embeddings are normalized and the hosted
            # index is already on cosine, so the two stores have to agree.
            "hnsw:space": "cosine",
            DIMENSION_KEY: dimension,
        },
    )

    found = stored_dimension(store)
    if found is not None and found != dimension:
        # Refused here rather than left to Chroma: without this the mismatch
        # surfaces as `add_documents` failing inside the per-document ingest, so
        # a wrong model reads as "every document failed" instead of one clear
        # refusal before anything is written.
        name = config.chroma_collection
        raise RuntimeError(
            f"The collection {name!r} holds {found}-dimensional vectors, but the "
            f"current embedding model produces {dimension}. Two models are not "
            "comparable even at the same length, so the collection cannot be "
            "reused: either put EMBEDDING_PROVIDER and EMBEDDING_MODEL back to "
            "what it was built with, or point CHROMA_DIR at an empty folder and "
            "run the sync again."
        )

    return store
