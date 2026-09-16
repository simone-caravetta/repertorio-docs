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
    """The model this configuration indexes with, by its own name.

    The configured name when there is one and the provider's default when there
    is not, so that a machine which leaves EMBEDDING_MODEL empty and one which
    spells the default out are seen to be indexing with the same model. This is
    the name the catalog records, and the point of recording it is that the
    vectors themselves do not carry one: a corpus moved to another model has to
    be indexed again, and nothing else would say so.

    Read before a run rather than during one, which is why it refuses a provider
    it does not know here as well as in `get_embeddings`.
    """
    return config.embedding_model or _default_model(config.embedding_provider)


@lru_cache(maxsize=1)
def get_embeddings(config: Settings = settings) -> Embeddings:
    """Build the model the index is written and read with.

    Local by default: the model runs in this process and no document text leaves
    the machine. The OpenAI path is for a corpus the local model is too slow to
    re-index. The two are not interchangeable — each produces its own vector
    space, so switching means indexing the whole corpus again under a new name.
    """
    provider = config.embedding_provider.strip().lower()
    # Built once and used by both branches, so that the model the catalog will
    # record is the model that was actually built.
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

    # Anything but "local" was refused above unless it was "openai".
    key = config.embedding_api_key or config.openai_api_key
    if not key:
        raise RuntimeError(
            "Missing environment variables: EMBEDDING_API_KEY "
            "(EMBEDDING_PROVIDER is 'openai')"
        )
    return OpenAIEmbeddings(model=name, api_key=key, base_url=OPENAI_URL)
