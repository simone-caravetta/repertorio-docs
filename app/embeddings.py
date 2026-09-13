from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.config import Settings, settings

LOCAL_MODEL = "BAAI/bge-m3"
OPENAI_MODEL = "text-embedding-3-small"
# Where the OpenAI embeddings are reached, whatever OPENAI_BASE_URL points the
# chat model at — which is usually not OpenAI.
OPENAI_URL = "https://api.openai.com/v1"


@lru_cache(maxsize=1)
def get_embeddings(config: Settings = settings) -> Embeddings:
    """Build the model the index is written and read with.

    Local by default: the model runs in this process and no document text leaves
    the machine. The OpenAI path is for a corpus the local model is too slow to
    re-index. The two are not interchangeable — each produces its own vector
    space, so switching means indexing the whole corpus again under a new name.
    """
    provider = config.embedding_provider.strip().lower()

    if provider == "local":
        return HuggingFaceEmbeddings(
            model_name=config.embedding_model or LOCAL_MODEL,
            model_kwargs={"device": config.embedding_device},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": config.embedding_batch_size,
            },
        )

    if provider == "openai":
        key = config.embedding_api_key or config.openai_api_key
        if not key:
            raise RuntimeError(
                "Missing environment variables: EMBEDDING_API_KEY "
                "(EMBEDDING_PROVIDER is 'openai')"
            )
        return OpenAIEmbeddings(
            model=config.embedding_model or OPENAI_MODEL,
            api_key=key,
            base_url=OPENAI_URL,
        )

    raise RuntimeError(
        f"Unknown EMBEDDING_PROVIDER: {config.embedding_provider!r}. "
        "Use 'local' or 'openai'."
    )
