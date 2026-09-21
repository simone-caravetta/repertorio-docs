"""The settings, read from the environment and from `.env`.

Every setting has a default, so the parts that need no key run with no `.env` at
all. The file is read once, at import, and does not override a variable that is
already set in the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _default_catalog_path(store: str) -> str:
    """The catalog that belongs to a store.

    The two stores hold different vectors, so each one keeps its own catalog.
    The rows say what was indexed there, and a hash that matches in one store
    says nothing about the other. Naming the file after the store makes
    switching store a single line instead of two lines that have to agree.
    """
    if store.strip().lower() == "chroma":
        return "data/catalog-chroma.sqlite3"
    return "data/catalog.sqlite3"


@dataclass(frozen=True)
class Settings:
    # The chat model. Any endpoint that speaks the OpenAI chat completions API
    # works here, whether it is hosted or running on your own machine. These are
    # the names the OpenAI client already reads, so a local server is pointed at
    # the same way as a hosted one. The provider fills in the defaults and the
    # extra parameters that a known endpoint needs.
    chat_provider: str = os.getenv("CHAT_PROVIDER", "deepseek")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # The embedding model. "local" runs it in this process and "openai" calls the
    # API. An empty model name means the default of the selected provider.
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "")
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    # Only for the OpenAI embeddings. OPENAI_API_KEY belongs to the chat
    # endpoint, which usually sits somewhere else.
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")

    # Where the vectors live. Either the hosted index or a folder on this
    # machine.
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

    # Reranking. The passages a search returned are scored again by a model that
    # reads the question and the passage together, and the best `retrieval_k` of
    # them are what the answer is written from. The search is therefore asked for
    # `rerank_candidates` and `retrieval_k` come back. See `app/rerank.py`.
    #
    # Off by default. On the library this project was built against, the search
    # already puts the right page first for every question, so reranking has
    # nothing to reorder. It also costs about seven seconds per question on a CPU
    # and a checkpoint of two gigabytes. A question set that the search gets
    # wrong is what would justify turning it on.
    rerank: str = os.getenv("RERANK", "off")
    rerank_model: str = os.getenv("RERANK_MODEL", "")
    rerank_candidates: int = int(os.getenv("RERANK_CANDIDATES", "20"))
    # The device the local models run on, unless the reranker is told otherwise.
    # Both models are the same kind, and one machine has one answer to this.
    rerank_device: str = os.getenv("RERANK_DEVICE", "") or embedding_device

    # Grading. Before an answer is written, a model reads the question and the
    # passages the search returned and says whether they hold the answer. If
    # they do not, its own query is searched for instead, and if that fails too
    # the question is turned away rather than answered from nothing. See
    # `app/grading.py`.
    #
    # On by default. A similarity search always returns its nearest passages,
    # however far away they are, so without this a question the library cannot
    # answer still comes back as a confident answer written from the wrong page.
    # It costs one model call per question, plus one when the question is turned
    # away.
    grade: str = os.getenv("GRADE", "on")
    grade_attempts: int = int(os.getenv("GRADE_ATTEMPTS", "2"))

    # A scope that covers one document is read whole instead of by top-k, as long
    # as its text is estimated to fit in this many characters. Every chunk is
    # handed to the model and nothing is dropped by similarity. Set it to 0 to
    # always search by similarity, however small the document.
    whole_document_max_chars: int = int(
        os.getenv("WHOLE_DOCUMENT_MAX_CHARS", "24000")
    )
    # What a description is written from. The sample is taken from across a
    # document instead of from its opening pages, and this is the budget it is
    # taken within. Read once per document by `scripts.describe`.
    description_sample_chars: int = int(
        os.getenv("DESCRIPTION_SAMPLE_CHARS", "6000")
    )
    # How much of the catalogue goes into one context window, under the list of
    # the documents that were searched. A description is a paragraph and a scope
    # can cover a whole library, so whatever does not fit is left out while the
    # documents are still named. Set it to 0 to turn the descriptions off.
    description_budget_chars: int = int(
        os.getenv("DESCRIPTION_BUDGET_CHARS", "2000")
    )

    documents_dir: Path = PROJECT_ROOT / os.getenv(
        "DOCUMENTS_DIR", "data/documents"
    )
    # Follows the store unless it is named outright. A catalog describes the
    # vectors of one store, so pointing VECTOR_STORE somewhere new without moving
    # the catalog would leave a sync that finds every file unchanged and writes
    # nothing. See `_default_catalog_path`.
    catalog_db_path: Path = PROJECT_ROOT / os.getenv(
        "CATALOG_DB_PATH", _default_catalog_path(vector_store)
    )
    # Where the conversations are kept between runs, as the graph checkpoints
    # them. One file for both stores, unlike the catalog. A conversation is what
    # was said, and the documents it cites are named the same way whichever store
    # holds their vectors.
    conversations_db_path: Path = PROJECT_ROOT / os.getenv(
        "CONVERSATIONS_DB_PATH", "data/conversations.sqlite3"
    )
    # The questions a run of the eval harness is measured against. They are
    # written about the documents and quote what they hold, so the file is kept
    # beside them instead of in the repository. See `data/evals/` in `.gitignore`.
    eval_questions_path: Path = PROJECT_ROOT / os.getenv(
        "EVAL_QUESTIONS", "data/evals/questions.json"
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

    The settings arrive as an argument instead of being read from the module, so
    a command that replaced them describes the store it is really about to use.
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
    """Check the keys the vector index needs and name the ones that are missing.

    The checks are separate because the keys are. The chat model checks its own
    when it is built, whether that happens in the console, in the describe
    command or in a sync.
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
