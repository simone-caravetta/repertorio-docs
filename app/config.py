from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _default_catalog_path(store: str) -> str:
    """The catalog that belongs to a store.

    The two stores hold different vectors, so each keeps its own catalog: the
    rows say what was indexed *there*, and a hash that matches in one says
    nothing about the other. Naming the file after the store, rather than
    leaving the pair to be set by hand, is what makes switching store one line
    instead of two that have to agree.
    """
    if store.strip().lower() == "chroma":
        return "data/catalog-chroma.sqlite3"
    return "data/catalog.sqlite3"


@dataclass(frozen=True)
class Settings:
    # The chat model: any endpoint that speaks the OpenAI chat completions API,
    # hosted or on your own machine. These are the names the OpenAI client already
    # reads, so a local server is pointed at the same way as a hosted one. The
    # provider supplies the defaults for them and the parameters a known one needs.
    chat_provider: str = os.getenv("CHAT_PROVIDER", "deepseek")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # The embedding model: "local" runs it in this process, "openai" calls the API.
    # Empty model means the default of whichever provider is selected.
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "")
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    # Only for the OpenAI embeddings: OPENAI_API_KEY belongs to the chat endpoint,
    # which is usually somewhere else.
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")

    # Where the vectors live: the hosted index, or a folder on this machine.
    vector_store: str = os.getenv("VECTOR_STORE", "pinecone")
    chroma_dir: Path = PROJECT_ROOT / os.getenv("CHROMA_DIR", "data/chroma")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "documents")

    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "repertorio-docs")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE", "documents")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "5"))

    documents_dir: Path = PROJECT_ROOT / os.getenv(
        "DOCUMENTS_DIR", "data/documents"
    )
    # Follows the store unless it is named outright: a catalog describes the
    # vectors of one store, so pointing VECTOR_STORE somewhere new without
    # moving the catalog would leave a sync that finds every file unchanged and
    # writes nothing. See `_default_catalog_path`.
    catalog_db_path: Path = PROJECT_ROOT / os.getenv(
        "CATALOG_DB_PATH", _default_catalog_path(vector_store)
    )


settings = Settings()


def short_path(path: Path) -> str:
    """The path as it reads from the project root, when it is under it."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def describe_vector_store(config: Settings) -> str:
    """Where the vectors are, in one line, for a command to print.

    Taken as an argument rather than read from the module-level `settings`, so
    that a command which replaces its own `settings` describes the one it is
    actually about to use.
    """
    name = config.vector_store.strip().lower()
    if name == "chroma":
        return (
            f"chroma ({short_path(config.chroma_dir)}, "
            f"collection {config.chroma_collection})"
        )
    if name == "pinecone":
        return (
            f"pinecone (index {config.pinecone_index_name}, "
            f"namespace {config.pinecone_namespace})"
        )
    return name


def validate_api_keys() -> None:
    """Check the keys the vector index needs, and name the ones that are missing.

    The chat model checks its own key when it is built: a sync does not talk to it.
    """
    missing = []
    if not settings.pinecone_api_key:
        missing.append("PINECONE_API_KEY")
    if settings.embedding_provider.strip().lower() == "openai" and not (
        settings.embedding_api_key or settings.openai_api_key
    ):
        missing.append("EMBEDDING_API_KEY")

    if missing:
        raise RuntimeError("Missing environment variables: " + ", ".join(missing))
