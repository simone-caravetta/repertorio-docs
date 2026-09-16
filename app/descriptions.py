"""What each document is about, in a paragraph, written once and kept.

A search returns the passages closest to a question, which is what makes a
question about the library itself hard to answer from them: asked what the
documents contain, the answer is whatever the search happened to find, and a
small document beside a large one is never among it. The catalog has a
`description` column for exactly this, and until now nothing wrote it.

Written here rather than at every question. A description is a fact about a
document, it costs one model call per document instead of one per question, and
it is read back by the console, by the page, and by the context an answer is
written from — which is the point: a document whose passages the search did not
return is still a document the answer can say something about.

Two callers write one, and both come through `write_descriptions`: the sync, as
it indexes, and `scripts.describe`, for a library that is already indexed. A
description therefore does not depend on which of them wrote it, and the sync
cannot drift into describing a document differently from the command that exists
to describe it.
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

# How many places in a document the sample is taken from. The budget decides how
# much of the document is read; this decides how evenly it is spread, with the
# first and the last page among them.
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
    """A description as one line, which is what everything downstream shows.

    A model asked for two sentences may answer with a heading, a list, or a
    paragraph with newlines inside it. The catalog holds one line, and the
    context opens with one line per document: a newline in the middle of a
    description would read as a second document having been searched.
    """
    return " ".join(text.split())


def spread(items: list[str], count: int) -> list[str]:
    """`count` items taken from across a list, the first and the last among them.

    Evenly spaced rather than every n-th, so that the last item is reached: a
    document whose length is not a multiple of the step would otherwise never
    show its end.
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
    """Extracts from across a document, within a character budget.

    Across and not from the beginning: a title page and a table of contents are
    the opening of most documents and describe none of them. Every extract gets
    an equal share of the budget, so that one dense page cannot spend the whole
    of it and leave the rest of the document unread.
    """
    texts = [page.page_content.strip() for page in pages]
    texts = [text for text in texts if text]

    if not texts or max_chars <= 0:
        return ""

    chosen = spread(texts, count)
    # The blank lines between the extracts are part of the budget: the caller
    # sets a number of characters to spend, and this spends no more than it.
    separators = 2 * (len(chosen) - 1)
    each = max(1, (max_chars - separators) // len(chosen))

    return "\n\n".join(text[:each] for text in chosen)


def describe_document(model: BaseChatModel, *, name: str, text: str) -> str:
    """One document's description, written from a sample of its text."""
    chain = description_prompt | model | StrOutputParser()
    return one_line(chain.invoke({"name": name, "excerpts": text}))


def undescribed(catalog: Catalog) -> list[DocumentRecord]:
    """The indexed documents nothing has been written about yet.

    Indexed and not merely in the catalog: a document that failed to index has no
    text to describe, and one in the trash has a row but no file.
    """
    return [
        record
        for record in catalog.all()
        if record.status == "indexed" and not (record.description or "").strip()
    ]


@dataclass(frozen=True)
class DescribeReport:
    """What a run over a list of documents did."""

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
    """Describe each document, one model call each, and keep it in the catalog.

    A document that cannot be read, and a call that fails, are recorded and the
    run goes on: this is a run over a library, and one unreadable document is not
    a reason to stop at it.

    Each description is written as it is produced, so a run that is interrupted
    leaves what it had finished behind, and the next run picks up the rest.
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
