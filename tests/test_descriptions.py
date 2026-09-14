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
    """`count` pages, each saying which one it is."""
    return [
        Document(page_content=f"page {number}", metadata={"page": number})
        for number in range(count)
    ]


def indexed(catalog: Catalog, path: str, *, description: str | None = None) -> None:
    catalog.add_file(path, Path(path).stem)
    catalog.record_indexed(path, file_hash="x", page_count=1, chunk_count=2)
    if description is not None:
        catalog.set_description(path, description)


# ------------------------------------------------------------------- sampling


def test_the_sample_is_taken_from_across_the_document() -> None:
    """Not from the opening pages: a title page describes nothing."""
    text = sample(pages(40), max_chars=10_000)

    assert "page 0" in text
    assert "page 39" in text


def test_the_sample_is_wider_than_one_place_in_the_document() -> None:
    text = sample(pages(40), max_chars=10_000)

    assert len(text.split("\n\n")) == 5
    assert "page 20" in text


def test_the_sample_spends_no_more_than_the_budget() -> None:
    text = sample(pages(40), max_chars=500)

    assert len(text) <= 500


def test_a_short_document_is_sampled_whole() -> None:
    assert sample(pages(2), max_chars=10_000) == "page 0\n\npage 1"


def test_a_page_with_no_text_is_not_part_of_a_sample() -> None:
    assert sample(pages(3) + [Document(page_content="   ")], max_chars=100) != ""
    assert sample([Document(page_content="")], max_chars=100) == ""


def test_a_budget_of_zero_reads_nothing() -> None:
    assert sample(pages(3), max_chars=0) == ""


def test_the_spread_keeps_the_first_and_the_last() -> None:
    assert spread(["a", "b", "c", "d", "e"], 3) == ["a", "c", "e"]
    assert spread(["a", "b"], 5) == ["a", "b"]
    assert spread(["a", "b", "c"], 1) == ["a"]


# -------------------------------------------------------------- the writing


def test_the_description_is_written_from_the_extracts_and_the_file_name() -> None:
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
    """A heading, a list or a paragraph would read as a second document."""
    assert one_line("  Il libro\n\nparla di   Roma.  ") == "Il libro parla di Roma."
    assert one_line("") == ""


# ---------------------------------------------------------- what is missing


def test_only_indexed_documents_with_nothing_written_are_missing(
    catalog: Catalog,
) -> None:
    indexed(catalog, MANUAL)
    indexed(catalog, REPORT, description="Last year's report.")

    catalog.add_file("manuals/failed.pdf", "failed")
    catalog.record_failed("manuals/failed.pdf", "No text extracted")
    catalog.add_file("manuals/gone.pdf", "gone")
    catalog.trash("manuals/gone.pdf")

    missing = undescribed(catalog)

    assert [record.path for record in missing] == [MANUAL]


# ------------------------------------------------------------- the run


def test_describing_a_document_records_it_in_the_catalog(
    documents_dir: Path, catalog: Catalog
) -> None:
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
    """One unreadable document is not a reason to stop at it."""
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
