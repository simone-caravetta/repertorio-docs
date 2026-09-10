from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_core.documents import Document

from app.ingestion import (
    _stable_id,
    compute_file_hash,
    ingest_one,
    load_pdf,
    split_documents,
)
from tests.helpers import FakeVectorStore, make_pdf


def test_compute_file_hash_hashes_the_bytes(tmp_path: Path) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"some bytes")

    expected = hashlib.sha256(b"some bytes").hexdigest()
    assert compute_file_hash(path) == expected


def test_compute_file_hash_follows_the_content(tmp_path: Path) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"before")
    before = compute_file_hash(path)

    path.write_bytes(b"after")

    assert compute_file_hash(path) != before


def test_stable_id_is_deterministic() -> None:
    document = Document(page_content="text", metadata={"source": "a.pdf", "page": 0})

    assert _stable_id(document) == _stable_id(document)


def test_stable_id_follows_the_content() -> None:
    def document(text: str, page: int = 0) -> Document:
        return Document(page_content=text, metadata={"source": "a.pdf", "page": page})

    assert _stable_id(document("one")) != _stable_id(document("two"))
    assert _stable_id(document("one")) != _stable_id(document("one", page=1))
    # Whitespace around the chunk is not part of its identity.
    assert _stable_id(document(" one ")) == _stable_id(document("one"))


def test_stable_id_separates_chunks_that_read_the_same() -> None:
    def document(position: int) -> Document:
        return Document(
            page_content="the same text, twice on the page",
            metadata={"source": "a.pdf", "page": 0, "chunk_id": position},
        )

    assert _stable_id(document(1)) != _stable_id(document(2))


def test_load_pdf_names_pages_after_the_documents_folder(
    documents_dir: Path,
) -> None:
    pages = load_pdf(documents_dir / "manuals" / "manual.pdf", documents_dir)

    assert len(pages) == 1
    assert pages[0].metadata["source"] == "manuals/manual.pdf"
    assert pages[0].metadata["page"] == 0
    assert pages[0].page_content


def test_split_documents_marks_the_chunks() -> None:
    pages = [Document(page_content="word " * 200, metadata={"source": "a.pdf"})]

    chunks = split_documents(pages, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == list(
        range(len(chunks))
    )
    assert {chunk.metadata["document_type"] for chunk in chunks} == {"pdf"}
    assert {chunk.metadata["source"] for chunk in chunks} == {"a.pdf"}


def test_ingest_one_indexes_the_file(
    documents_dir: Path, store: FakeVectorStore
) -> None:
    path = documents_dir / "manuals" / "manual.pdf"

    result = ingest_one(
        path,
        documents_dir=documents_dir,
        vectorstore=store,
        chunk_size=200,
        chunk_overlap=20,
    )

    assert result.page_count == 1
    assert result.chunk_count == len(store.added[0][0])
    assert result.chunk_count > 0

    documents, ids = store.added[0]
    assert ids == [_stable_id(document) for document in documents]
    assert len(set(ids)) == len(ids)
    assert {document.metadata["source"] for document in documents} == {
        "manuals/manual.pdf"
    }


def test_ingest_one_reports_the_file_hash(
    documents_dir: Path, store: FakeVectorStore
) -> None:
    path = documents_dir / "manuals" / "manual.pdf"

    result = ingest_one(
        path,
        documents_dir=documents_dir,
        vectorstore=store,
        chunk_size=200,
        chunk_overlap=20,
    )

    assert result.file_hash == compute_file_hash(path)


def test_a_pdf_without_text_indexes_nothing(
    tmp_path: Path, store: FakeVectorStore
) -> None:
    documents_dir = tmp_path / "documents"
    make_pdf(documents_dir / "scans" / "scan.pdf", "")

    result = ingest_one(
        documents_dir / "scans" / "scan.pdf",
        documents_dir=documents_dir,
        vectorstore=store,
        chunk_size=200,
        chunk_overlap=20,
    )

    assert (result.page_count, result.chunk_count) == (1, 0)
    assert store.added == []
