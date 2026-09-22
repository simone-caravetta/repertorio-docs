"""Test doubles and small document builders used across the suite.

The fakes stand in for the chat model, the vector store, the retriever and
the reranker, so a test can run without a network call. The PDF builders let
a test decide the layout of the document the reader is given.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.vectorstores import VectorStore
from pydantic import Field

from app.config import Settings

# A sentence a test repeats when a page needs more than one line of text.
SENTENCE = "The manual of the thing explains how the thing works. "

# Filler that reads as ordinary body copy, so it is never taken for a
# heading.
BODY = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod."

# The model name the test settings carry. A test reads it back off a catalog
# row to check which model indexed the document.
EMBEDDING_MODEL = "test-model"


def verdict_reply(
    supported: bool, reason: str = "Nothing states it.", query: str = ""
) -> str:
    """A grader's reply, in the shape its prompt asks for.

    The reader of that reply is tested on replies that are not shaped like
    this, so those are written out in full where they are needed.
    """
    return json.dumps({"supported": supported, "reason": reason, "query": query})


def make_settings(**overrides: object) -> Settings:
    """Build settings with the values the tests expect.

    Any field can be replaced by passing it as a keyword argument. Pinning
    everything here keeps a test from picking up whatever the environment
    happens to hold.
    """

    values: dict[str, object] = {
        "chat_provider": "deepseek",
        "openai_base_url": "",
        "openai_model": "",
        "openai_api_key": "endpoint-key",
        "embedding_provider": "local",
        "embedding_model": "",
        "embedding_device": "cpu",
        "embedding_batch_size": 32,
        "embedding_api_key": "",
        "vector_store": "pinecone",
        "chroma_dir": Path("/tmp/repertorio-docs-test-chroma"),
        "chroma_collection": "documents",

        # Where a test writes. None of these files is opened unless the test
        # asks for it.
        "documents_dir": Path("/tmp/repertorio-docs-test/documents"),
        "catalog_db_path": Path("/tmp/repertorio-docs-test/catalog.sqlite3"),
        "conversations_db_path": Path(
            "/tmp/repertorio-docs-test/conversations.sqlite3"
        ),
        "eval_questions_path": Path("/tmp/repertorio-docs-test/questions.json"),

        # How a document is cut up and how much of it comes back.
        "chunk_size": 200,
        "retrieval_k": 5,
        "whole_document_max_chars": 24000,
        "description_sample_chars": 6000,
        "description_budget_chars": 2000,

        # Reranking, small to big and grading stay off here, so that a test
        # about something else reads its replies off one call per step. The
        # tests that cover them turn them on through an override.
        "grade": "off",
        "grade_attempts": 2,
        "rerank": "off",
        "rerank_model": "",
        "rerank_candidates": 20,
        "rerank_device": "cpu",
        "small_to_big": "off",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _escape(text: str) -> str:
    # Escape the characters that would end the string inside the PDF.
    for char in ("\\", "(", ")"):
        text = text.replace(char, f"\\{char}")
    return text


def make_pdf(path: Path, text: str) -> Path:
    """Write a one-page PDF holding a single line of text.

    The file is assembled byte by byte, so it does not depend on a writer
    library. An empty text leaves the page with no text at all, which is how
    a scanned document arrives.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    stream = f"BT /F1 12 Tf 72 720 Td ({_escape(text)}) Tj ET" if text else ""
    content = (
        f"<< /Length {len(stream.encode('latin-1'))} >>\n"
        f"stream\n{stream}\nendstream"
    )

    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        content,
    ]

    body: list[bytes] = [b"%PDF-1.4\n"]
    offsets: list[int] = []

    for number, obj in enumerate(objects, start=1):
        offsets.append(sum(len(part) for part in body))
        body.append(f"{number} 0 obj\n{obj}\nendobj\n".encode("latin-1"))

    start_xref = sum(len(part) for part in body)
    xref = [
        b"xref\n",
        b"0 6\n",
        b"0000000000 65535 f \r\n",
        *(f"{offset:010d} 00000 n \r\n".encode("latin-1") for offset in offsets),
    ]
    trailer = (
        f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{start_xref}\n%%EOF\n"
    ).encode("latin-1")

    path.write_bytes(b"".join(body + xref + [trailer]))
    return path


@dataclass(frozen=True)
class Line:
    """One line of text to place on a page.

    The size matters to the reader, which decides from it whether the line
    is a heading.
    """

    text: str
    size: float = 11.0

    # Where to put the line. Left unset, each line goes below the one before
    # it.
    y: float | None = None


@dataclass(frozen=True)
class Table:
    """A grid of cells to draw on a page.

    Every row is a list of cell texts, given top row first.
    """

    rows: list[list[str]]


@dataclass(frozen=True)
class Image:
    """A picture to draw on a page, in points.

    With `full_page` the picture covers the whole sheet, which is what a
    scanned document is made of. The reader is the one that decides whether
    what it is given is a figure of the document or the page itself.
    """

    width: float = 0.0
    height: float = 0.0
    y: float | None = None
    full_page: bool = False


def make_structured_pdf(
    path: Path,
    pages: list[list[Line | Table | Image]],
    toc: list[list[Any]] | None = None,
) -> Path:
    """Write a PDF from a description of its pages.

    Each page is a list of lines, tables and images. A line without a y is
    placed under the line before it, so a page can be laid out without
    working out coordinates. A table is drawn at the cursor and the cursor
    moves past it, and so is an image. The toc argument writes the document
    outline, which is where the reader looks for section titles.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open()

    for items in pages:
        page = document.new_page()
        cursor = 80.0

        for item in items:
            if isinstance(item, Table):
                _draw_table(page, item.rows, y=cursor)
                cursor += _TABLE_ROW * len(item.rows) + 20
                continue

            if isinstance(item, Image):
                _draw_image(page, item, y=cursor)
                # Nothing is laid out over a whole-page picture.
                cursor = page.rect.height if item.full_page else cursor + item.height + 20
                continue

            y = cursor if item.y is None else item.y
            page.insert_text((72, y), item.text, fontsize=item.size)
            if item.y is None:
                cursor += item.size * 1.8

    if toc:
        document.set_toc(toc)

    document.save(path)
    document.close()
    return path


# Cell size in points. The row height also decides how far the cursor moves
# after a table.
_TABLE_ROW = 22.0
_TABLE_COLUMN = 120.0


def _draw_image(page: pymupdf.Page, item: Image, *, y: float) -> None:
    # The picture is drawn from four grey pixels: a test is about where an
    # image sits, not about what it shows.
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 4, 4))
    pixmap.set_rect(pixmap.irect, (180, 180, 180))

    if item.full_page:
        rect = pymupdf.Rect(page.rect)
    else:
        top = y if item.y is None else item.y
        rect = pymupdf.Rect(72, top, 72 + item.width, top + item.height)

    page.insert_image(rect, stream=pixmap.tobytes("png"))


def _draw_table(page: pymupdf.Page, rows: list[list[str]], *, y: float) -> None:
    # The grid lines go down first, so the cell texts sit on top of them.
    columns = max(len(row) for row in rows)
    width = _TABLE_COLUMN * columns
    height = _TABLE_ROW * len(rows)

    for row in range(len(rows) + 1):
        page.draw_line((72, y + row * _TABLE_ROW), (72 + width, y + row * _TABLE_ROW))
    for column in range(columns + 1):
        page.draw_line((72 + column * _TABLE_COLUMN, y), (72 + column * _TABLE_COLUMN, y + height))

    for row, cells in enumerate(rows):
        for column, text in enumerate(cells):
            page.insert_text(
                (76 + column * _TABLE_COLUMN, y + row * _TABLE_ROW + 15),
                text,
                fontsize=11,
            )


class FakeChatModel(BaseChatModel):
    """A chat model that returns prepared replies and keeps the prompts.

    The reply at index n answers the n-th call, so a test can script the
    rewrite step and the answer step apart. The prompts are kept as well,
    which lets a test check what the model was asked.
    """

    replies: list[str]
    prompts: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.prompts.append(list(messages))
        reply = self.replies[len(self.prompts) - 1]
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=reply))]
        )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        self.prompts.append(list(messages))
        reply = self.replies[len(self.prompts) - 1]

        for piece in pieces_of(reply):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
            if run_manager is not None:
                await run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk


def pieces_of(reply: str) -> list[str]:
    # Split a reply the way a streaming model would send it, word by word
    # with the spaces kept.
    words = reply.split(" ")
    return [word + " " for word in words[:-1]] + words[-1:]


def as_a_store_returns(document: Document) -> Document:
    """Turn a document into what a vector store hands back.

    A store keeps metadata outside the process, so a whole number written in
    comes back as a float. Tests that compare metadata from both sides use
    this to get the types the application sees at run time.
    """

    metadata = {
        # Page numbers and character offsets arrive as floats.
        key: float(value)
        if isinstance(value, int) and not isinstance(value, bool)
        else value
        for key, value in document.metadata.items()
    }

    return Document(page_content=document.page_content, metadata=metadata)


class FakeRetriever:
    """A retriever that returns the same documents whatever is asked.

    The questions it was called with are recorded, so a test can check which
    one reached the search.
    """

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.queries: list[str] = []

    async def ainvoke(
        self, query: str, config: Any = None, **kwargs: Any
    ) -> list[Document]:
        self.queries.append(query)
        return self.documents


class FakeReranker:
    """A cross-encoder stand-in that hands out scores from a callable.

    The callable receives a pair and its position in the batch. The default
    gives the first passage the highest score, which is the order the search
    returned. Every batch is recorded, along with the batch size and whether
    a progress bar was asked for.
    """

    def __init__(
        self,
        scores: Callable[[tuple[str, str], int], float] | None = None,
    ) -> None:
        self.scores = scores or (lambda _pair, at: -float(at))
        self.asked: list[list[tuple[str, str]]] = []
        self.batch_sizes: list[int] = []
        self.progress_bars: list[bool] = []

    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> list[float]:
        self.asked.append(list(pairs))
        self.batch_sizes.append(batch_size)
        self.progress_bars.append(show_progress_bar)
        return [self.scores(pair, at) for at, pair in enumerate(pairs)]


@dataclass
class FakeVectorStore(VectorStore):
    """An in-memory vector store that records what it was asked to do.

    Documents are kept in the order they were added and a search returns the
    matches in reverse, so a test can tell the store's order from the order
    the application puts them in. The fail_on and fail_delete_on sets make a
    write raise, which is how an ingest failure is staged. Every add and
    delete is appended to events in the order it happened.
    """

    added: list[tuple[list[Document], list[str]]] = field(default_factory=list)
    deleted: list[dict[str, object] | None] = field(default_factory=list)
    events: list[tuple[str, object]] = field(default_factory=list)

    searches: list[tuple[str, int, dict[str, object] | None]] = field(
        default_factory=list
    )
    fail_on: set[str] = field(default_factory=set)
    fail_delete_on: set[str] = field(default_factory=set)

    def add_documents(
        self, documents: list[Document], **kwargs: object
    ) -> list[str]:
        for document in documents:
            source = document.metadata.get("source")
            if source in self.fail_on:
                raise RuntimeError(f"refused to index {source}")

        ids = list(kwargs.get("ids") or [])  # type: ignore[arg-type]
        self.added.append((documents, ids))
        self.events.append(("add", ids))
        return ids

    def delete(self, ids: list[str] | None = None, **kwargs: object) -> bool:
        criteria = kwargs.get("filter")
        source = criteria.get("source") if isinstance(criteria, dict) else None
        if source in self.fail_delete_on:
            raise RuntimeError(f"refused to delete {source}")

        self.deleted.append(criteria)  # type: ignore[arg-type]
        self.events.append(("delete", criteria))
        return True

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: object
    ) -> list[Document]:
        """Return the documents that match the filter, newest first.

        A filter of None matches everything. A source filter is either one
        value or a mapping holding a $in list.
        """
        criteria = kwargs.get("filter")
        self.searches.append((query, k, criteria))  # type: ignore[arg-type]

        found = [
            document
            for documents, _ in self.added
            for document in documents
            if self._matches(document, criteria)
        ]
        return list(reversed(found))

    @classmethod
    def from_texts(cls, *args: object, **kwargs: object) -> None:
        # Required by the base class. A test builds the store directly.
        raise NotImplementedError

    @staticmethod
    def _matches(
        document: Document, criteria: dict[str, object] | None
    ) -> bool:
        """Whether a document matches a filter.

        The forms the application writes are read: one field compared with a
        value, with an `$eq` or against an `$in` list, and an `$and` of those.
        A field the document does not carry matches nothing.
        """
        if criteria is None:
            return True

        if "$and" in criteria:
            return all(
                FakeVectorStore._matches(document, one)
                for one in criteria["$and"]  # type: ignore[union-attr]
            )

        for key, wanted in criteria.items():
            if not isinstance(wanted, dict):
                if document.metadata.get(key) != wanted:
                    return False
                continue

            value = document.metadata.get(key)
            if "$eq" in wanted and value != wanted["$eq"]:
                return False
            if "$in" in wanted and value not in (wanted["$in"] or []):
                return False

        return True

    @property
    def sources(self) -> list[str]:
        # The source of every document added, in the order they arrived.
        return [
            str(document.metadata.get("source"))
            for documents, _ in self.added
            for document in documents
        ]
