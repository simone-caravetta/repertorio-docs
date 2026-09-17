"""Tests for writing a document's description.

The description is written from a sample of the document's pages, and the file
name goes with the sample. These tests cover how
the sample is chosen, how a model's reply is tidied into one line, and which
documents a run decides to describe.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from app.catalog import Catalog
from app.descriptions import (
    describe_document,
    one_line,
    sample,
    spread,
    undescribed,
    write_descriptions,
)
from tests.helpers import SENTENCE, FakeChatModel, make_pdf

MANUAL = "manuals/manual.pdf"
REPORT = "reports/report.pdf"


def pages(count: int) -> list[Document]:
    """A document of `count` pages, each holding its own number as text."""
    return [
        Document(page_content=f"page {number}", metadata={"page": number})
        for number in range(count)
    ]


def indexed(catalog: Catalog, path: str, *, description: str | None = None) -> None:
    """Put one document in the catalog as a file that was indexed."""
    catalog.add_file(path, Path(path).stem)
    catalog.record_indexed(path, file_hash="x", page_count=1, chunk_count=2)
    if description is not None:
        catalog.set_description(path, description)


# The sample is what the model is shown instead of the whole document, so it
# has to reach the end of it, stay within the budget it was given, and leave
# out the pages that carry no text at all.


def test_the_sample_is_taken_from_across_the_document() -> None:
    """The opening page and the closing page are both in the sample."""
    text = sample(pages(40), max_chars=10_000)

    assert "page 0" in text
    assert "page 39" in text


def test_the_sample_is_wider_than_one_place_in_the_document() -> None:
    """The sample is five extracts, spaced across the document."""
    text = sample(pages(40), max_chars=10_000)

    assert len(text.split("\n\n")) == 5
    assert "page 20" in text


def test_the_sample_spends_no_more_than_the_budget() -> None:
    """A small budget is spent, and the sample stays within it."""
    text = sample(pages(40), max_chars=500)

    assert len(text) <= 500


def test_a_short_document_is_sampled_whole() -> None:
    """A document shorter than the budget is read as it is."""
    assert sample(pages(2), max_chars=10_000) == "page 0\n\npage 1"


def test_a_page_with_no_text_is_not_part_of_a_sample() -> None:
    """Pages of blank text are left out. A document of only those is empty."""
    assert sample(pages(3) + [Document(page_content="   ")], max_chars=100) != ""
    assert sample([Document(page_content="")], max_chars=100) == ""


def test_a_budget_of_zero_reads_nothing() -> None:
    """A budget of zero characters gives an empty sample."""
    assert sample(pages(3), max_chars=0) == ""


# Which pages the sample comes from, on its own: the ends of the list are
# always among them, and a count larger than the list changes nothing.


def test_the_spread_keeps_the_first_and_the_last() -> None:
    """The ends stay in the result, and the rest is spaced evenly."""
    assert spread(["a", "b", "c", "d", "e"], 3) == ["a", "c", "e"]
    assert spread(["a", "b"], 5) == ["a", "b"]
    assert spread(["a", "b", "c"], 1) == ["a"]


# What the model is asked, and what is done with the reply it gives.


def test_the_description_is_written_from_the_extracts_and_the_file_name() -> None:
    """The prompt carries the name and the sample. The reply is the text."""
    model = FakeChatModel(replies=["A manual about the thing."])

    description = describe_document(
        model, name="manual", text="the thing is explained here"
    )

    assert description == "A manual about the thing."
    prompt = str(model.prompts[0])
    assert "File name: manual" in prompt
    assert "the thing is explained here" in prompt
    assert "two or three sentences" in prompt


def test_what_the_model_returns_is_put_on_one_line() -> None:
    """Newlines and runs of spaces are folded into single spaces."""
    assert one_line("  Il libro\n\nparla di   Roma.  ") == "Il libro parla di Roma."
    assert one_line("") == ""


# Which documents a run should describe, out of everything the catalog holds.


def test_only_indexed_documents_with_nothing_written_are_missing(
    catalog: Catalog,
) -> None:
    """A failed file, one in the trash and one described are all left out."""
    indexed(catalog, MANUAL)
    indexed(catalog, REPORT, description="Last year's report.")

    catalog.add_file("manuals/failed.pdf", "failed")
    catalog.record_failed("manuals/failed.pdf", "No text extracted")
    catalog.add_file("manuals/gone.pdf", "gone")
    catalog.trash("manuals/gone.pdf")

    missing = undescribed(catalog)

    assert [record.path for record in missing] == [MANUAL]


# A run over a list of documents writes each description as it goes and records
# the ones it could not read. One failure does not stop the ones after it.


def test_describing_a_document_records_it_in_the_catalog(
    documents_dir: Path, catalog: Catalog
) -> None:
    """The description is written from the file and kept on the catalog row."""
    indexed(catalog, MANUAL)
    model = FakeChatModel(replies=["A manual about the thing."])

    report = write_descriptions(
        model,
        catalog,
        catalog.all(),
        documents_dir=documents_dir,
        sample_chars=6000,
    )

    assert report.written == [(MANUAL, "A manual about the thing.")]
    assert report.failed == []
    assert catalog.get(MANUAL).description == "A manual about the thing."


def test_a_document_whose_file_is_gone_is_recorded_and_the_run_goes_on(
    documents_dir: Path, catalog: Catalog
) -> None:
    """A missing file is reported as failed, and the next one is written."""
    indexed(catalog, MANUAL)
    indexed(catalog, REPORT)
    (documents_dir / MANUAL).unlink()
    model = FakeChatModel(replies=["Last year's report."])

    report = write_descriptions(
        model,
        catalog,
        [catalog.get(MANUAL), catalog.get(REPORT)],
        documents_dir=documents_dir,
        sample_chars=6000,
    )

    assert [path for path, _ in report.failed] == [MANUAL]
    assert "The file is not there" in report.failed[0][1]
    assert report.written == [(REPORT, "Last year's report.")]
    assert catalog.get(REPORT).description == "Last year's report."
    assert catalog.get(MANUAL).description is None


def test_a_model_that_returns_nothing_is_a_failure_and_not_a_description(
    documents_dir: Path, catalog: Catalog
) -> None:
    """An empty reply is a failure, and nothing is stored for that document."""
    make_pdf(documents_dir / MANUAL, SENTENCE * 20)
    indexed(catalog, MANUAL)

    report = write_descriptions(
        FakeChatModel(replies=["   "]),
        catalog,
        catalog.all(),
        documents_dir=documents_dir,
        sample_chars=6000,
    )

    assert report.written == []
    assert report.failed == [(MANUAL, "The model returned nothing")]
    assert catalog.get(MANUAL).description is None
