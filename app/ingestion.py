from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.vectorstore import get_vectorstore

SUPPORTED_EXTENSIONS = {".pdf"}


def _stable_id(doc: Document) -> str:
    """Create deterministic IDs so re-running ingestion does not create duplicates."""
    source = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page", ""))
    content = doc.page_content.strip()
    raw = f"{source}|{page}|{content}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_pdfs() -> list[Document]:
    docs: list[Document] = []

    for path in sorted(settings.documents_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        loader = PyPDFLoader(str(path))
        pages = loader.load()

        for page in pages:
            # Store a stable relative source path for citations and metadata filters.
            page.metadata["source"] = str(
                path.relative_to(settings.documents_dir)
            )
            docs.append(page)

    return docs


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["document_type"] = "pdf"

    return chunks


def ingest() -> int:
    settings.documents_dir.mkdir(parents=True, exist_ok=True)

    raw_docs = load_pdfs()
    if not raw_docs:
        print(f"No PDF found in: {settings.documents_dir}")
        return 0

    chunks = split_documents(raw_docs)
    ids = [_stable_id(doc) for doc in chunks]

    vectorstore = get_vectorstore()
    vectorstore.add_documents(documents=chunks, ids=ids)

    print(f"Pages loaded  : {len(raw_docs)}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Pinecone index: {settings.pinecone_index_name}")
    print(f"Namespace     : {settings.pinecone_namespace}")

    return len(chunks)
