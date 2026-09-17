"""Tests for turning a PDF into the chunks a search is given.

The reader hands back one Document per page, the chunker cuts each page along
the headings and the tables the page declares, and every chunk keeps the
offsets it was cut at and the page it came from. These tests cover the
identifiers a re-run depends on, the text a chunk claims to hold, and what one
file leaves in the store.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from langchain_core.documents import Document

from app import pdf
from app.ingestion import (
    _stable_id,
    chunk_document,
    compute_file_hash,
    ingest_one,
    load_pdf,
    whole_number,
)
from tests.helpers import (
    BODY,
    FakeVectorStore,
    Line,
    Table,
    make_pdf,
    make_structured_pdf,
)


def chunks_of(path: Path, *, chunk_size: int, chunk_overlap: int = 0) -> list[Document]:
    """Chunk one PDF, under the source name `a.pdf`."""
    return chunk_document(
        pdf.read(path),
        source="a.pdf",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def test_compute_file_hash_hashes_the_bytes(tmp_path: Path) -> None:
    """The hash is the sha256 of what is in the file."""
    path = tmp_path / "a.bin"
    path.write_bytes(b"some bytes")

    expected = hashlib.sha256(b"some bytes").hexdigest()
    assert compute_file_hash(path) == expected


def test_compute_file_hash_follows_the_content(tmp_path: Path) -> None:
    """Rewriting the file gives another hash, so a change is noticed."""
    path = tmp_path / "a.bin"
    path.write_bytes(b"before")
    before = compute_file_hash(path)

    path.write_bytes(b"after")

    assert compute_file_hash(path) != before


def test_a_whole_number_is_read_as_one_however_it_arrives() -> None:
    """A float holding a whole number is read as that number, zero included."""
    assert whole_number(44) == 44
    assert whole_number(44.0) == 44
    assert whole_number(0.0) == 0


def test_what_is_not_a_whole_number_is_none() -> None:
    """Booleans, text, fractions and infinities read as absent, like None."""

    for value in (None, True, False, "44", 44.5, float("nan"), float("inf")):
        assert whole_number(value) is None, value


def test_stable_id_is_deterministic() -> None:
    """The same document hashes to the same id every time."""
    document = Document(page_content="text", metadata={"source": "a.pdf", "page": 0})

    assert _stable_id(document) == _stable_id(document)


def test_stable_id_follows_the_content() -> None:
    """Other text or another page is another id, and spacing is not."""
    def document(text: str, page: int = 0) -> Document:
        return Document(page_content=text, metadata={"source": "a.pdf", "page": page})

    assert _stable_id(document("one")) != _stable_id(document("two"))
    assert _stable_id(document("one")) != _stable_id(document("one", page=1))

    assert _stable_id(document(" one ")) == _stable_id(document("one"))


def test_stable_id_separates_chunks_that_read_the_same() -> None:
    """Two chunks of the same text at different offsets are two ids."""
    def document(position: int) -> Document:
        return Document(
            page_content="the same text, twice on the page",
            metadata={"source": "a.pdf", "page": 0, "start": position},
        )

    assert _stable_id(document(1)) != _stable_id(document(2))


def test_load_pdf_names_pages_after_the_documents_folder(
    documents_dir: Path,
) -> None:
    """The source is the path inside the folder, and pages count from zero."""
    pages = load_pdf(documents_dir / "manuals" / "manual.pdf", documents_dir)

    assert len(pages) == 1
    assert pages[0].metadata["source"] == "manuals/manual.pdf"
    assert pages[0].metadata["page"] == 0
    assert pages[0].page_content


# A page built to be cut: two sections with a body under each, and a table at
# the end of the page.


TABLE_ROWS = [
    ["Parte", "Ore", "Nota"],
    ["Filtro", "200", "pulire"],
    ["Olio", "1000", "sostituire"],
    ["Candele", "500", "controllare"],
    ["Freni", "800", "sostituire"],
    ["Gomme", "400", "gonfiare"],
]


@pytest.fixture
def manual(tmp_path: Path) -> Path:
    """One page holding two sections and a table."""

    return make_structured_pdf(
        tmp_path / "manual.pdf",
        pages=[
            [
                Line("Prima sezione", size=15),
                Line(BODY),
                Line(BODY),
                Line("Seconda sezione", size=15),
                Line(BODY),
                Line(BODY),
                Table(TABLE_ROWS),
            ]
        ],
    )


def test_the_chunks_are_marked_with_what_they_are(manual: Path) -> None:
    """Each chunk carries its number, its type and the file it came from."""
    chunks = chunks_of(manual, chunk_size=120)

    assert len(chunks) > 1
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == list(
        range(len(chunks))
    )
    assert {chunk.metadata["document_type"] for chunk in chunks} == {"pdf"}
    assert {chunk.metadata["source"] for chunk in chunks} == {"a.pdf"}


def test_a_chunk_is_the_text_of_the_page_it_says_it_is(manual: Path) -> None:
    """The offsets point at the chunk's own text, inside the page it names."""

    pages = pdf.read_pages(manual)

    for chunk in chunks_of(manual, chunk_size=120):
        start, end = chunk.metadata["start"], chunk.metadata["end"]
        page = pages[chunk.metadata["page"]]

        assert page.text[start:end] == chunk.page_content


def test_a_chunk_does_not_cross_a_heading(manual: Path) -> None:
    """A chunk holds text from one section and never from two.

    The cut falls inside a section, so every chunk opens under the heading it
    belongs to and the text of the two sections stays apart.
    """

    chunks = chunks_of(manual, chunk_size=120)

    sections = {chunk.metadata.get("section") for chunk in chunks}
    assert sections == {"Prima sezione", "Seconda sezione"}

    for chunk in chunks:
        names = [
            name
            for name in ("Prima sezione", "Seconda sezione")
            if name in chunk.page_content
        ]
        assert names in ([], [chunk.metadata["section"]])


def test_a_table_is_one_chunk(manual: Path) -> None:
    """A table comes back whole, in reading order, whatever its size."""

    tables = [
        chunk
        for chunk in chunks_of(manual, chunk_size=60)
        if chunk.metadata.get("table")
    ]

    assert len(tables) == 1
    assert tables[0].page_content.split() == [
        cell for row in TABLE_ROWS for cell in row
    ]

    # The table is longer than a chunk, and is still a single one.

    assert len(tables[0].page_content) > 60


def test_ingest_one_indexes_the_file(
    documents_dir: Path, store: FakeVectorStore
) -> None:
    """The file is read, cut into chunks and added under their own ids."""
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

    # The ids are what a second run sends again, so they are the chunks' own.
    documents, ids = store.added[0]
    assert ids == [_stable_id(document) for document in documents]
    assert len(set(ids)) == len(ids)
    assert {document.metadata["source"] for document in documents} == {
        "manuals/manual.pdf"
    }


def test_ingest_one_reports_the_file_hash(
    documents_dir: Path, store: FakeVectorStore
) -> None:
    """The hash of the file comes back with the counts."""
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
    """A scanned page has a page and no chunks, and nothing is stored."""
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
