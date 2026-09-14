"""Test doubles: a minimal PDF writer, a fake model, a fake vector store, a
hermetic Settings."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.vectorstores import VectorStore
from pydantic import Field

from app.config import Settings

# A line long enough to produce a chunk, short enough to stay well under the
# default chunk size.
SENTENCE = "The manual of the thing explains how the thing works. "


def make_settings(**overrides: object) -> Settings:
    """A Settings that does not depend on the machine the tests run on.

    Every field a provider reads is set here explicitly, because the defaults
    are read from the environment once, when the class is defined. `overrides`
    replaces the ones a test is about.
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
        # Set explicitly like every other provider: the catalog default follows
        # VECTOR_STORE, and that is read from the machine's `.env` once, when
        # the class is defined. A test must not inherit the folder layout of
        # whoever runs it.
        "documents_dir": Path("/tmp/repertorio-docs-test/documents"),
        "catalog_db_path": Path("/tmp/repertorio-docs-test/catalog.sqlite3"),
        "conversations_db_path": Path(
            "/tmp/repertorio-docs-test/conversations.sqlite3"
        ),
        # A scope is resolved against these two, so a test that pins neither
        # would decide differently on a machine with a different `.env`.
        "chunk_size": 200,
        "retrieval_k": 5,
        "whole_document_max_chars": 24000,
        "description_sample_chars": 6000,
        "description_budget_chars": 2000,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _escape(text: str) -> str:
    """Escape the characters a PDF string literal cannot contain bare."""
    for char in ("\\", "(", ")"):
        text = text.replace(char, f"\\{char}")
    return text


def make_pdf(path: Path, text: str) -> Path:
    """Write a minimal one-page PDF holding `text`, and return the path.

    An empty string produces a page with no text at all, which is what a scanned
    document looks like to the text extractor. The file is assembled by hand —
    no extra dependency, no committed binary — so the cross-reference table has
    to be built with the exact byte offsets pypdf expects.

    Text is written with the Latin-1 encoding: keep it ASCII.
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


class FakeChatModel(BaseChatModel):
    """Replies from a list, one reply per call, and keeps the prompts it saw.

    It streams as well as it generates, a word at a time: what the graph hands a
    caller in `messages` mode is the pieces a model produced, not one answer that
    arrived whole, and a test of the token stream needs a model that has pieces
    to give. Streaming is also what a real model does here — the callback the
    graph installs is what makes it stream, so the console and the page both see
    the answer as it is written.
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
    """A reply as the pieces it streams in: words, spaces kept with them."""
    words = reply.split(" ")
    return [word + " " for word in words[:-1]] + words[-1:]


class FakeRetriever:
    """The graph only ever calls `ainvoke` on a retriever, so that is all this is."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.queries: list[str] = []

    async def ainvoke(
        self, query: str, config: Any = None, **kwargs: Any
    ) -> list[Document]:
        self.queries.append(query)
        return self.documents


@dataclass
class FakeVectorStore(VectorStore):
    """Stands in for the Pinecone store.

    It records the three calls the library ever makes on a store: adding chunks,
    deleting by metadata filter, and searching. Sources listed in `fail_on` raise
    on add, and sources listed in `fail_delete_on` raise on delete: that is how
    the tests drive a failed ingest and a failed removal.

    It really is a `VectorStore`, because some of the code under test takes one as
    a pydantic field (`WholeDocumentRetriever.store`) and pydantic checks the type
    rather than trusting the annotation. Inheriting also buys the real
    `as_retriever`, so a test can search through the wiring the console uses
    instead of through a stand-in for it.
    """

    added: list[tuple[list[Document], list[str]]] = field(default_factory=list)
    deleted: list[dict[str, object] | None] = field(default_factory=list)
    events: list[tuple[str, object]] = field(default_factory=list)
    # Every search, as it was asked: the query, the k, and the filter.
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
        """The chunks on record that match the filter, in reverse order.

        Reverse on purpose: a similarity search ranks by a question, so the order
        it hands back is not the order the document reads in, and a caller that
        wants reading order has to put it back. `k` is recorded rather than
        applied — the fake holds a handful of chunks, and what a test wants to
        know is what was asked for.
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
        """Abstract on the base class, and never reached: a test builds the fake
        by hand, holding whatever chunks the case is about."""
        raise NotImplementedError

    @staticmethod
    def _matches(
        document: Document, criteria: dict[str, object] | None
    ) -> bool:
        if criteria is None:
            return True

        source = criteria.get("source")
        if isinstance(source, dict):  # the {"$in": [...]} a scope builds
            return document.metadata.get("source") in (source.get("$in") or [])
        return document.metadata.get("source") == source

    @property
    def sources(self) -> list[str]:
        """The sources of the chunks added so far, in order."""
        return [
            str(document.metadata.get("source"))
            for documents, _ in self.added
            for document in documents
        ]
