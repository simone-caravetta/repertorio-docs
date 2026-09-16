from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import pdf

SUPPORTED_EXTENSIONS = {".pdf"}

_HASH_BLOCK_SIZE = 1 << 20

# What the text of one piece is cut on, in the order it is tried. A blank line
# first, because a paragraph is where a reader would have cut.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


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


def indexer_id(*, chunk_size: int, chunk_overlap: int) -> str:
    """What built an index, as one line: the reader, and the cut it was made with.

    Written onto the catalog row when a document is indexed, and read back on the
    next run: it is how a document indexed by an older reading, or cut to other
    sizes, is noticed and made again. Nothing in the vectors themselves says what
    produced them, and the file has not changed, so without this the index would
    go on answering from text this code would no longer cut.

    The model that turned the text into numbers is the other half of the same
    fact, and the catalog keeps it in a column of its own.
    """
    return f"{pdf.READER}-{pdf.READER_VERSION}|{chunk_size}/{chunk_overlap}"


def _stable_id(doc: Document) -> str:
    """Create deterministic IDs so re-running ingestion does not create duplicates.

    The position of the chunk is part of its identity, not just its text: a page
    that repeats itself — a table header, a running footer — would otherwise
    give the same ID to chunks that are different pieces of the document, and
    the vector store would keep only one of them. The position is the offset the
    chunk was found at, which is exact where a count of chunks is only ordinal.
    """
    source = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page", ""))
    start = str(doc.metadata.get("start", ""))
    content = doc.page_content.strip()
    raw = f"{source}|{page}|{start}|{content}".encode()
    return hashlib.sha256(raw).hexdigest()


def load_pdf(path: Path, documents_dir: Path) -> list[Document]:
    """Load one PDF as one Document per page, with a relative `source` path.

    The text is the reader's, the same one the chunks are cut from. Reading the
    document here and reading it there differently would mean a description
    written from text that the search cannot find.
    """
    # A stable relative path is what citations and metadata filters use.
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
    """One document as the chunks to index, cut along the structure it declares.

    A piece is what the cut never crosses. A section's body is split by size as it
    always was, so the chunks are the size they were, but the cut falls inside one
    piece: a chunk does not open under one heading and end under the next, and a
    table is one chunk whatever its size, because a table read in halves is read
    as prose.

    Every chunk carries where it came from — its page, and the offsets into that
    page's text — which is what a citation is drawn from, and the section it sits
    under, which is a name for it that the text alone does not give.
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
    """A piece as the runs to chunk, each with where it sits in the page's text.

    A table is handed back whole: it is one unit however large it is, so it is
    never offered to the splitter.
    """
    if piece.kind == pdf.TABLE:
        return [(piece.text, piece.start, piece.end)]

    parts: list[tuple[str, int, int]] = []

    for found in splitter.create_documents([piece.text]):
        index = found.metadata["start_index"]

        # The splitter reports -1 for a chunk it cannot find again in the text it
        # came from, and an offset of no use to anyone. The piece is taken whole
        # rather than the offsets being invented: text that is too large is a
        # fault a reader can see, and a citation pointing at the wrong words is
        # not.
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
    """Read, cut and index a single document.

    Returns the counts and the hash of the file as it was read. A `chunk_count`
    of zero means the PDF had no text to extract: nothing was sent to the vector
    store, and it is up to the caller to record that as a failure.
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
