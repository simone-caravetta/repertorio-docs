"""Writing the description of a document.

A description says what a document contains. It goes into the context of every
answer, under the list of the documents that were searched. The chat model
writes it from a sample of pages taken from across the document, once per
document, and the catalog keeps it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.catalog import Catalog, DocumentRecord
from app.ingestion import load_pdf

# How many pages of a document are shown to the model.
SAMPLE_PAGES = 5

description_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You describe a document for the catalogue of a personal library.

You are given extracts from across one document, and the name of the file they
come from. The extracts are a sample: the document is longer than what you are
shown, so describe it at the level they support and never state a count, a
total or a last chapter.

Write two or three sentences, in the language the document is written in, that
say what kind of document it is and what it covers, so that someone reading the
catalogue can tell whether it is the one they want.

Return the description alone. No heading, no list, no quotation marks, nothing
about the extracts being a sample, and nothing about the file — the name is
there for you, not for the reader.""",
    ),
    (
        "human",
        """File name: {name}

Extracts:
{excerpts}""",
    ),
])


def one_line(text: str) -> str:
    """The text with every run of whitespace collapsed to a single space."""
    return " ".join(text.split())


def spread(items: list[str], count: int) -> list[str]:
    """`count` items taken evenly from across the list, ends included.

    A list shorter than the count comes back whole.
    """
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[0]]

    step = (len(items) - 1) / (count - 1)
    return [items[round(position * step)] for position in range(count)]


def sample(
    pages: list[Document], *, max_chars: int, count: int = SAMPLE_PAGES
) -> str:
    """The text of a sample of pages, cut to fit a character budget.

    Empty pages are dropped and the rest are taken from across the document.
    The budget is divided between the pages that were chosen, leaving room for
    the blank lines that separate them.
    """
    texts = [page.page_content.strip() for page in pages]
    texts = [text for text in texts if text]

    if not texts or max_chars <= 0:
        return ""

    chosen = spread(texts, count)

    # Two newlines go between the chosen pages, and what is left of the budget
    # is shared out evenly over them.
    separators = 2 * (len(chosen) - 1)
    each = max(1, (max_chars - separators) // len(chosen))

    return "\n\n".join(text[:each] for text in chosen)


def describe_document(model: BaseChatModel, *, name: str, text: str) -> str:
    """Ask the model to describe a document from its name and a sample of it."""
    chain = description_prompt | model | StrOutputParser()
    return one_line(chain.invoke({"name": name, "excerpts": text}))


def undescribed(catalog: Catalog) -> list[DocumentRecord]:
    """The indexed documents that have no description yet."""
    return [
        record
        for record in catalog.all()
        if record.status == "indexed" and not (record.description or "").strip()
    ]


@dataclass(frozen=True)
class DescribeReport:
    """What a run wrote, and what it could not write."""

    written: list[tuple[str, str]] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)


def write_descriptions(
    model: BaseChatModel,
    catalog: Catalog,
    records: Iterable[DocumentRecord],
    *,
    documents_dir: Path,
    sample_chars: int,
) -> DescribeReport:
    """Write a description for each record, collecting the ones that failed.

    A document that cannot be read or that produces no description is kept in
    the failed list with the reason, and the run goes on to the next one. Only
    the documents that were written are stored in the catalog.
    """
    documents_dir = Path(documents_dir)
    report = DescribeReport()

    for record in records:
        path = documents_dir / record.path

        try:
            if not path.exists():
                raise FileNotFoundError(f"The file is not there: {record.path}")

            text = sample(
                load_pdf(path, documents_dir), max_chars=sample_chars
            )
            if not text:
                raise ValueError("No text to describe")

            description = describe_document(
                model, name=record.title, text=text
            )
            if not description:
                raise ValueError("The model returned nothing")
        except Exception as exc:  # noqa: BLE001 - one document, not the run
            report.failed.append((record.path, str(exc)))
            continue

        catalog.set_description(record.path, description)
        report.written.append((record.path, description))

    return report
