from __future__ import annotations

from functools import lru_cache

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.config import settings, validate_api_keys
from app.embeddings import get_embeddings


def get_pinecone_client() -> Pinecone:
    validate_api_keys()
    return Pinecone(api_key=settings.pinecone_api_key)


def ensure_index() -> None:
    """Create the Pinecone index if it does not exist."""
    pc = get_pinecone_client()

    if not pc.has_index(settings.pinecone_index_name):
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.pinecone_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )


@lru_cache(maxsize=1)
def get_vectorstore() -> PineconeVectorStore:
    ensure_index()
    pc = get_pinecone_client()
    index = pc.Index(settings.pinecone_index_name)

    return PineconeVectorStore(
        index=index,
        embedding=get_embeddings(),
        namespace=settings.pinecone_namespace,
    )


@lru_cache(maxsize=1)
def get_retriever():
    """Keep vector store and retriever conceptually decoupled."""
    return get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_k},
    )
