from __future__ import annotations

from pathlib import Path

import pytest

from app.catalog import Catalog, category_from_path
from app.scope import as_source, resolve_scope, whole_library
from tests.helpers import make_pdf, make_settings


def indexed(catalog: Catalog, path: str, *, chunks: int = 2) -> None:
    """A document in the catalog, indexed, filed under the folder it sits in."""
    catalog.add_file(path, Path(path).stem, category=category_from_path(path))
    catalog.record_indexed(
        path, file_hash="x", page_count=1, chunk_count=chunks
    )


def described(catalog: Catalog, path: str, text: str) -> None:
    """A document with something written about it, as `scripts.describe` leaves it."""
    catalog.set_description(path, text)


def test_a_question_with_no_scope_is_asked_of_the_whole_library(
    catalog: Catalog,
) -> None:
    scope = resolve_scope(catalog, config=make_settings())

    assert scope.sources is None
    assert scope.label == "whole library"
    assert not scope.whole_document


def test_whole_library_is_the_scope_a_console_starts_on() -> None:
    catalog = Catalog(":memory:")

    assert whole_library(catalog, config=make_settings()) == resolve_scope(
        catalog, config=make_settings()
    )


def test_the_whole_library_names_the_documents_it_covers(catalog: Catalog) -> None:
    """No filter is what the search gets; the documents are still known."""
    indexed(catalog, "manuals/a.pdf")
    indexed(catalog, "reports/c.pdf")

    scope = resolve_scope(catalog, config=make_settings())

    assert scope.sources is None
    assert scope.documents == ("manuals/a.pdf", "reports/c.pdf")


def test_the_whole_library_leaves_out_what_a_search_cannot_reach(
    catalog: Catalog,
) -> None:
    """A document with no vectors is not one of the documents there are."""
    indexed(catalog, "manuals/a.pdf")
    catalog.add_file("manuals/b.pdf", "b", category="manuals")
    catalog.record_failed("manuals/b.pdf", "No text extracted")
    catalog.add_file("manuals/c.pdf", "c", category="manuals")
    catalog.trash("manuals/c.pdf")

    scope = resolve_scope(catalog, config=make_settings())

    assert scope.documents == ("manuals/a.pdf",)


def test_a_scope_carries_what_the_catalog_says_about_its_documents(
    catalog: Catalog,
) -> None:
    """The answer is told what the documents contain, and not only which they are."""
    indexed(catalog, "manuals/a.pdf")
    indexed(catalog, "reports/c.pdf")
    described(catalog, "manuals/a.pdf", "A manual about the thing.")
    described(catalog, "reports/c.pdf", "Last year's report.")

    scope = resolve_scope(catalog, config=make_settings())

    assert scope.descriptions == (
        ("manuals/a.pdf", "A manual about the thing."),
        ("reports/c.pdf", "Last year's report."),
    )


def test_a_document_with_nothing_written_about_it_is_still_named(
    catalog: Catalog,
) -> None:
    """The list of documents is the scope; a description is an extra on top."""
    indexed(catalog, "manuals/a.pdf")
    indexed(catalog, "reports/c.pdf")
    described(catalog, "manuals/a.pdf", "A manual about the thing.")

    scope = resolve_scope(catalog, config=make_settings())

    assert scope.documents == ("manuals/a.pdf", "reports/c.pdf")
    assert scope.descriptions == (("manuals/a.pdf", "A manual about the thing."),)


def test_descriptions_stop_at_the_budget_they_are_given(catalog: Catalog) -> None:
    """A scope over a library spends a paragraph on the catalog, and no more."""
    indexed(catalog, "manuals/a.pdf")
    indexed(catalog, "reports/c.pdf")
    described(catalog, "manuals/a.pdf", "x" * 30)
    described(catalog, "reports/c.pdf", "y" * 30)

    scope = resolve_scope(catalog, config=make_settings(description_budget_chars=40))

    assert scope.descriptions == (("manuals/a.pdf", "x" * 30),)
    assert scope.documents == ("manuals/a.pdf", "reports/c.pdf")


def test_a_budget_of_zero_leaves_the_descriptions_out(catalog: Catalog) -> None:
    indexed(catalog, "manuals/a.pdf")
    described(catalog, "manuals/a.pdf", "A manual about the thing.")

    scope = resolve_scope(catalog, config=make_settings(description_budget_chars=0))

    assert scope.descriptions == ()
    assert scope.documents == ("manuals/a.pdf",)


def test_a_category_scope_carries_the_descriptions_of_its_documents(
    catalog: Catalog,
) -> None:
    indexed(catalog, "manuals/a.pdf")
    indexed(catalog, "reports/c.pdf")
    described(catalog, "manuals/a.pdf", "A manual about the thing.")
    described(catalog, "reports/c.pdf", "Last year's report.")

    scope = resolve_scope(catalog, config=make_settings(), category="manuals")

    assert scope.descriptions == (("manuals/a.pdf", "A manual about the thing."),)


def test_a_category_brings_its_documents_and_the_ones_below(
    catalog: Catalog,
) -> None:
    indexed(catalog, "manuals/a.pdf")
    indexed(catalog, "manuals/ancient/b.pdf")
    indexed(catalog, "reports/c.pdf")

    scope = resolve_scope(catalog, config=make_settings(), category="manuals")

    assert scope.sources == ("manuals/a.pdf", "manuals/ancient/b.pdf")
    # The narrowed scope covers exactly what it filters by, and the answer is
    # told the same list the search was given.
    assert scope.documents == scope.sources
    assert scope.label == "category manuals (2 documents)"


def test_a_category_is_taken_as_it_is_typed(catalog: Catalog) -> None:
    """The shell hands over what a person typed, slashes and all."""
    indexed(catalog, "manuals/a.pdf")

    scope = resolve_scope(catalog, config=make_settings(), category="/manuals/")

    assert scope.sources == ("manuals/a.pdf",)


def test_the_documents_with_no_category_are_a_scope_of_their_own(
    catalog: Catalog,
) -> None:
    indexed(catalog, "a.pdf")
    indexed(catalog, "manuals/b.pdf")

    scope = resolve_scope(catalog, config=make_settings(), category="")

    assert scope.sources == ("a.pdf",)
    assert scope.label == "no category (1 document)"


def test_a_category_that_holds_nothing_raises_and_names_the_ones_that_exist(
    catalog: Catalog,
) -> None:
    indexed(catalog, "manuals/a.pdf")

    with pytest.raises(LookupError, match="Known categories: manuals"):
        resolve_scope(catalog, config=make_settings(), category="manuels")


def test_an_empty_catalog_says_there_are_no_categories_at_all(
    catalog: Catalog,
) -> None:
    with pytest.raises(LookupError, match="No document in the catalog is in a category"):
        resolve_scope(catalog, config=make_settings(), category="manuals")


def test_a_document_that_fits_is_read_whole(catalog: Catalog) -> None:
    indexed(catalog, "manuals/a.pdf", chunks=3)

    scope = resolve_scope(
        catalog,
        config=make_settings(chunk_size=200, whole_document_max_chars=1000),
        document="manuals/a.pdf",
    )

    assert scope.sources == ("manuals/a.pdf",)
    assert scope.whole_document
    assert scope.chunks == 3
    assert scope.label == "document manuals/a.pdf (whole, 3 chunks)"


def test_a_document_too_long_to_fit_is_searched_by_similarity(
    catalog: Catalog,
) -> None:
    indexed(catalog, "manuals/a.pdf", chunks=30)

    scope = resolve_scope(
        catalog,
        config=make_settings(chunk_size=200, whole_document_max_chars=1000),
        document="manuals/a.pdf",
    )

    assert not scope.whole_document
    assert scope.chunks is None
    assert scope.label == "document manuals/a.pdf (top 5 of its chunks)"


def test_a_budget_of_zero_never_reads_a_document_whole(catalog: Catalog) -> None:
    """The way out for a model whose context is smaller than any document."""
    indexed(catalog, "manuals/a.pdf", chunks=1)

    scope = resolve_scope(
        catalog,
        config=make_settings(chunk_size=200, whole_document_max_chars=0),
        document="manuals/a.pdf",
    )

    assert not scope.whole_document


def test_a_selection_is_the_documents_it_names(catalog: Catalog) -> None:
    indexed(catalog, "manuals/a.pdf")
    indexed(catalog, "reports/c.pdf")

    scope = resolve_scope(
        catalog,
        config=make_settings(),
        documents=["manuals/a.pdf", "reports/c.pdf"],
    )

    assert scope.sources == ("manuals/a.pdf", "reports/c.pdf")
    assert not scope.whole_document


def test_a_document_that_was_never_synced_raises(catalog: Catalog) -> None:
    with pytest.raises(LookupError, match="Not in the catalog"):
        resolve_scope(
            catalog, config=make_settings(), document="manuals/ghost.pdf"
        )


def test_a_document_with_no_vectors_raises(catalog: Catalog) -> None:
    catalog.add_file("manuals/a.pdf", "a", category="manuals")
    catalog.record_failed("manuals/a.pdf", "No text extracted")

    with pytest.raises(LookupError, match="is failed"):
        resolve_scope(
            catalog, config=make_settings(), document="manuals/a.pdf"
        )


def test_one_document_of_a_selection_with_no_vectors_raises(
    catalog: Catalog,
) -> None:
    """Quietly searching the rest would answer from less than was asked for."""
    indexed(catalog, "manuals/a.pdf")
    catalog.add_file("manuals/b.pdf", "b", category="manuals")
    catalog.trash("manuals/b.pdf")

    with pytest.raises(LookupError, match="is trashed"):
        resolve_scope(
            catalog,
            config=make_settings(),
            documents=["manuals/a.pdf", "manuals/b.pdf"],
        )


def test_a_scope_is_one_of_the_three(catalog: Catalog) -> None:
    with pytest.raises(ValueError, match="category, document"):
        resolve_scope(
            catalog,
            config=make_settings(),
            category="manuals",
            document="manuals/a.pdf",
        )


def test_a_path_is_taken_as_the_shell_writes_it(tmp_path: Path) -> None:
    documents_dir = tmp_path / "documents"
    make_pdf(documents_dir / "manuals" / "a.pdf", "text")

    from_shell = as_source(str(documents_dir / "manuals" / "a.pdf"), documents_dir)
    from_the_folder = as_source("manuals/a.pdf", documents_dir)

    assert from_shell == "manuals/a.pdf"
    assert from_the_folder == "manuals/a.pdf"


def test_a_path_that_is_not_a_file_is_left_as_typed(tmp_path: Path) -> None:
    """So that the lookup fails on the name the person used, and says so."""
    assert as_source("manuals/ghost.pdf", tmp_path) == "manuals/ghost.pdf"
