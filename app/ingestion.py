from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {".pdf"}

_HASH_BLOCK_SIZE = 1 << 20


@dataclass(frozen=True)
class IngestResult:
    file_hash: str
    page_count: int
    chunk_count: int


def compute_file_hash(path: Path) -> str:
    """sha256 of the file, read in blocks so a large PDF stays out of memory."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK_SIZE), b""):
            digest.update(block)

    return digest.hexdigest()


def _stable_id(doc: Document) -> str:
    """Create deterministic IDs so re-running ingestion does not create duplicates.

    The position of the chunk is part of its identity, not just its text: a page
    that repeats itself — a table header, a running footer — would otherwise
    give the same ID to chunks that are different pieces of the document, and
    the vector store would keep only one of them.
    """
    source = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page", ""))
    position = str(doc.metadata.get("chunk_id", ""))
    content = doc.page_content.strip()
    raw = f"{source}|{page}|{position}|{content}".encode()
    return hashlib.sha256(raw).hexdigest()


def load_pdf(path: Path, documents_dir: Path) -> list[Document]:
    """Load one PDF as one Document per page, with a relative `source` path."""
    pages = PyPDFLoader(str(path)).load()

    # A stable relative path is what citations and metadata filters use.
    source = str(path.relative_to(documents_dir))

    for page in pages:
        page.metadata["source"] = source

    return pages


def split_documents(
    documents: list[Document], *, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["document_type"] = "pdf"

    return chunks


def ingest_one(
    path: Path,
    *,
    documents_dir: Path,
    vectorstore: VectorStore,
    chunk_size: int,
    chunk_overlap: int,
) -> IngestResult:
    """Load, split and index a single document.

    Returns the counts and the hash of the file as it was read. A `chunk_count`
    of zero means the PDF had no text to extract: nothing was sent to the vector
    store, and it is up to the caller to record that as a failure.
    """
    pages = load_pdf(path, documents_dir)
    chunks = split_documents(
        pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    if chunks:
        ids = [_stable_id(chunk) for chunk in chunks]
        vectorstore.add_documents(documents=chunks, ids=ids)

    return IngestResult(
        file_hash=compute_file_hash(path),
        page_count=len(pages),
        chunk_count=len(chunks),
    )
