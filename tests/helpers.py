"""Test doubles: a minimal PDF writer and a fake vector store."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

# A line long enough to produce a chunk, short enough to stay well under the
# default chunk size.
SENTENCE = "The manual of the thing explains how the thing works. "


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


@dataclass
class FakeVectorStore:
    """Stands in for the Pinecone store.

    It records the only two calls the library ever makes on a store: adding
    chunks, and deleting by metadata filter. Sources listed in `fail_on` raise
    on add, which is how the tests drive a failed ingest.
    """

    added: list[tuple[list[Document], list[str]]] = field(default_factory=list)
    deleted: list[dict[str, object] | None] = field(default_factory=list)
    events: list[tuple[str, object]] = field(default_factory=list)
    fail_on: set[str] = field(default_factory=set)

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
        self.deleted.append(kwargs.get("filter"))  # type: ignore[arg-type]
        self.events.append(("delete", kwargs.get("filter")))
        return True

    @property
    def sources(self) -> list[str]:
        """The sources of the chunks added so far, in order."""
        return [
            str(document.metadata.get("source"))
            for documents, _ in self.added
            for document in documents
        ]
