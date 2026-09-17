"""The model that turns text into vectors.

It runs on this machine by default, through sentence-transformers, so no
document text leaves the computer to be indexed. The OpenAI embeddings API is
the alternative.

Each model produces vectors in a space of its own, so the name of the model is
recorded in the catalog with every document it indexed.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.config import Settings, settings

LOCAL_MODEL = "BAAI/bge-m3"
OPENAI_MODEL = "text-embedding-3-small"
# The address of the OpenAI embeddings, whatever OPENAI_BASE_URL points the chat
# model at. That is usually not OpenAI.
OPENAI_URL = "https://api.openai.com/v1"


def _default_model(provider: str) -> str:
    """The model a provider uses when the configuration does not name one."""
    name = provider.strip().lower()

    if name == "local":
        return LOCAL_MODEL

    if name == "openai":
        return OPENAI_MODEL

    raise RuntimeError(
        f"Unknown EMBEDDING_PROVIDER: {provider!r}. Use 'local' or 'openai'."
    )


def model_name(config: Settings = settings) -> str:
    """The name of the model this configuration indexes with.

    When EMBEDDING_MODEL is empty the provider supplies its default. A machine
    that leaves the setting blank and a machine that writes the default out are
    then seen to be indexing with the same model.

    This is the name the catalog records. The vectors themselves do not say
    which model produced them, so this record is the only thing that shows a
    corpus has to be indexed again.

    It is read before a run starts, which is why an unknown provider is refused
    here as well as in `get_embeddings`.
    """
    return config.embedding_model or _default_model(config.embedding_provider)


@lru_cache(maxsize=1)
def get_embeddings(config: Settings = settings) -> Embeddings:
    """Build the model the index is written and read with.

    Local by default, so the model runs in this process and no document text
    leaves the machine. The OpenAI path is for a corpus that the local model is
    too slow to index. Each model produces its own vector space, so switching
    means indexing the whole corpus again under the new name.
    """
    provider = config.embedding_provider.strip().lower()
    # Built once and used by both branches, so the model the catalog records is
    # the model that was actually built.
    name = model_name(config)

    if provider == "local":
        return HuggingFaceEmbeddings(
            model_name=name,
            model_kwargs={"device": config.embedding_device},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": config.embedding_batch_size,
            },
        )

    # Every other provider was already refused by `_default_model`.
    key = config.embedding_api_key or config.openai_api_key
    if not key:
        raise RuntimeError(
            "Missing environment variables: EMBEDDING_API_KEY "
            "(EMBEDDING_PROVIDER is 'openai')"
        )
    return OpenAIEmbeddings(model=name, api_key=key, base_url=OPENAI_URL)
