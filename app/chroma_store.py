"""The Chroma store, which keeps the vectors in a folder on this machine.

Chroma is the alternative to Pinecone and is selected with VECTOR_STORE=chroma.
The collection records how wide the vectors in it are, so a collection built by
another embedding model is noticed when it is opened.
"""

from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.config import Settings

# The key the width of the vectors is stored under in the collection metadata.
DIMENSION_KEY = "dimension"


def as_where(criteria: Any) -> dict:
    """A LangChain filter as the `where` clause Chroma expects.

    Only a filter on the source is translated, which is the one this project
    deletes by. Any other filter raises, so that a new one is added here instead
    of being passed on unchecked.
    """
    source = criteria.get("source") if isinstance(criteria, dict) else None
    if isinstance(source, str) and len(criteria) == 1:
        return {"source": source}
    raise NotImplementedError(
        f"Cannot delete from Chroma by {criteria!r}: only {{'source': <path>}} is "
        "translated. Add the case here rather than passing it through."
    )


class ChromaStore(Chroma):
    """Chroma with the delete filter translated.

    LangChain takes `filter` on delete and Chroma itself takes `where`. This
    translates between the two, so a caller can delete by source without knowing
    which store it is on.
    """

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> None:
        """Delete by ids, or by a filter that is translated to `where`."""
        criteria = kwargs.pop("filter", None)
        if criteria is not None:
            kwargs["where"] = as_where(criteria)
        super().delete(ids=ids, **kwargs)


def stored_dimension(store: Chroma) -> int | None:
    """How wide the vectors in this collection are, or None when it is empty.

    The width is read from the collection metadata when it is recorded there. A
    collection from an older version has none, and then one vector is read back
    and measured.
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
    """Open the collection, creating it the first time.

    The width of the current embedding model is written into the collection
    metadata. A collection holding vectors of another width raises, because the
    index of one model cannot be searched with the vectors of another.
    """
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.PersistentClient(
        path=str(config.chroma_dir),
        # Nothing about this machine or these documents leaves it.
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    # The distance is fixed when the collection is created, so it is set here
    # once and has to be the one the vectors were indexed with.
    store = ChromaStore(
        client=client,
        collection_name=config.chroma_collection,
        embedding_function=embedding,
        collection_metadata={
            "hnsw:space": "cosine",
            DIMENSION_KEY: dimension,
        },
    )

    found = stored_dimension(store)
    if found is not None and found != dimension:
        # An old collection without the key in its metadata still reports the
        # width of the vectors it holds, and that is what is compared here.
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
