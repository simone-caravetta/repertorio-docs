"""Turning a PDF into chunks and writing them to the vector store.

A document is read one page at a time. Each page is split into pieces of roughly
CHUNK_SIZE characters, and every piece becomes a document in the store carrying
the metadata that says where it came from. The id of a chunk comes from its own
text, so indexing the same file twice writes the same rows instead of adding
copies.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import pdf

# The file types the sync indexes.
SUPPORTED_EXTENSIONS = {".pdf"}

# A file is hashed in blocks of this size instead of being read into memory.
_HASH_BLOCK_SIZE = 1 << 20

# Where a chunk may be cut, tried in this order. The empty string is the last
# resort and cuts between any two characters.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass(frozen=True)
class IngestResult:
    """What indexing one file produced."""

    file_hash: str
    page_count: int
    chunk_count: int


def compute_file_hash(path: Path) -> str:
    """The SHA-256 of a file, read in blocks.

    The sync keeps this hash and compares it with the one it stored last time,
    which is how it tells whether a file has changed.
    """
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK_SIZE), b""):
            digest.update(block)

    return digest.hexdigest()


def indexer_id(*, chunk_size: int, chunk_overlap: int) -> str:
    """What produced the chunks, as a string the catalog records.

    The reader version and the two chunking numbers are written into one value.
    When any of the three changes, the recorded value changes with it and the
    sync indexes the document again.
    """
    return f"{pdf.READER}-{pdf.READER_VERSION}|{chunk_size}/{chunk_overlap}"


def whole_number(value: Any) -> int | None:
    """The value as an int, or None when it is not a whole number.

    Metadata comes back from a store as whatever the store kept. A boolean is
    refused even though Python counts it as an integer, and so is a float with a
    fractional part.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None

    return int(value) if float(value).is_integer() else None


def _stable_id(doc: Document) -> str:
    """The id a chunk is written under.

    It is the hash of the source, the page, the position and the text, so the
    same chunk of the same file always lands on the same id, so writing it again
    replaces the row it already wrote.
    """
    source = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page", ""))
    start = str(doc.metadata.get("start", ""))
    content = doc.page_content.strip()
    raw = f"{source}|{page}|{start}|{content}".encode()
    return hashlib.sha256(raw).hexdigest()


def load_pdf(path: Path, documents_dir: Path) -> list[Document]:
    """One document per page of a PDF.

    The metadata carries the path of the file relative to the documents folder,
    which is the name the store filters on, and the number of the page.
    """
    source = str(path.relative_to(documents_dir))

    return [
        Document(
            page_content=page.text,
            metadata={"source": source, "page": page.number},
        )
        for page in pdf.read(path).pages
    ]


def chunk_document(
    document: pdf.Reading,
    *,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split a reading into the documents that go into the store.

    Every page is split on its own, so a chunk never spans two pages and each
    one knows the page it came from. The metadata also records the section a
    piece belongs to and whether the piece is a table.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        add_start_index=True,
    )

    chunks: list[Document] = []

    for piece in document.pieces:
        for text, start, end in _parts(piece, splitter):
            metadata: dict[str, object] = {
                "source": source,
                "page": piece.page,
                "document_type": "pdf",
                "chunk_id": len(chunks),
                "start": start,
                "end": end,
            }

            if piece.section is not None:
                metadata["section"] = piece.section
                metadata["level"] = piece.level

            if piece.kind == pdf.TABLE:
                metadata["table"] = True

            chunks.append(Document(page_content=text, metadata=metadata))

    return chunks


def _parts(piece: pdf.Piece, splitter: RecursiveCharacterTextSplitter) -> list[tuple[str, int, int]]:
    """Where a piece is split, as its text with the offsets it starts and ends at.

    A table is kept whole. Anything else is handed to the splitter, and each
    part it returns is checked against the text of the piece at the offset the
    splitter reported. A part that does not match at that offset sends the whole
    piece back unsplit, because an offset that cannot be trusted is of no use
    for pointing at the page.
    """
    if piece.kind == pdf.TABLE:
        return [(piece.text, piece.start, piece.end)]

    parts: list[tuple[str, int, int]] = []

    for found in splitter.create_documents([piece.text]):
        index = found.metadata["start_index"]
        # The splitter reports where the part begins in the text it was given.
        # If the part is not there at that offset the offsets of the whole piece
        # cannot be used for the parts, so the piece is returned as it is.
        if index < 0 or not piece.text.startswith(found.page_content, index):
            return [(piece.text, piece.start, piece.end)]

        start = piece.start + index
        parts.append((found.page_content, start, start + len(found.page_content)))

    return parts


def ingest_one(
    path: Path,
    *,
    documents_dir: Path,
    vectorstore: VectorStore,
    chunk_size: int,
    chunk_overlap: int,
) -> IngestResult:
    """Index one file and report what came of it.

    The chunks are written under ids derived from their text, so indexing a file
    that has not changed overwrites the same rows. The hash that comes back is
    what the catalog stores in order to notice a change later.
    """
    source = str(path.relative_to(documents_dir))
    document = pdf.read(path)
    chunks = chunk_document(
        document,
        source=source,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if chunks:
        ids = [_stable_id(chunk) for chunk in chunks]
        vectorstore.add_documents(documents=chunks, ids=ids)

    return IngestResult(
        file_hash=compute_file_hash(path),
        page_count=len(document.pages),
        chunk_count=len(chunks),
    )
