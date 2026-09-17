"""Tests for deleting documents from the library.

A deletion takes the vectors out of the store and the row out of the
catalog. The PDF itself stays on disk unless remove_files is asked for.
These tests cover both, the dry run that reports a job without doing it,
and the trash the sync fills with documents whose files went away.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.catalog import Catalog
from app.lifecycle import (
    DeleteReport,
    SyncReport,
    delete_documents,
    sync_documents,
    trashed_paths,
)
from tests.helpers import EMBEDDING_MODEL, SENTENCE, FakeVectorStore, make_pdf

MANUAL = "manuals/manual.pdf"
REPORT = "reports/report.pdf"

CHUNKING = {"chunk_size": 200, "chunk_overlap": 20}


def index(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> SyncReport:
    """Run a sync over the library, with the chunking the tests use."""
    return sync_documents(
        documents_dir, db_path, store, embedding_model=EMBEDDING_MODEL, **CHUNKING
    )


def delete(
    documents_dir: Path,
    db_path: Path,
    store: FakeVectorStore | None,
    paths: list[str],
    **kwargs: object,
) -> DeleteReport:
    """Run a deletion of the given paths, passing any extra options on."""
    return delete_documents(documents_dir, db_path, store, paths, **kwargs)


def test_a_document_leaves_the_index_and_the_catalog(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """Deleting a document removes its vectors and its catalog row.

    The other documents of the library are left as they were.
    """
    index(documents_dir, db_path, store)
    seen = len(store.events)

    report = delete(documents_dir, db_path, store, [MANUAL])

    assert report.deleted == [MANUAL]
    assert report.failed == []
    assert store.events[seen:] == [("delete", {"source": MANUAL})]

    catalog = Catalog(db_path)
    assert catalog.get(MANUAL) is None

    assert catalog.get(REPORT).status == "indexed"


def test_the_file_stays_where_it_is_and_the_document_comes_back(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """A deletion without remove_files leaves the PDF on disk.

    A later sync therefore indexes it again, as a new row with a new id.
    """
    index(documents_dir, db_path, store)
    before = Catalog(db_path).get(MANUAL).id

    report = delete(documents_dir, db_path, store, [MANUAL])

    assert (documents_dir / MANUAL).is_file()
    assert report.file_present == [MANUAL]
    assert report.files_removed == []

    # The file is still there, so the next sync picks it up again.
    index(documents_dir, db_path, store)

    record = Catalog(db_path).get(MANUAL)
    assert record.status == "indexed"
    assert record.id != before


def test_with_the_file_the_removal_holds(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """With remove_files the PDF goes too, so a sync does not bring it back."""
    index(documents_dir, db_path, store)

    report = delete(documents_dir, db_path, store, [MANUAL], remove_files=True)

    assert report.deleted == [MANUAL]
    assert report.files_removed == [MANUAL]
    assert report.file_present == []
    assert not (documents_dir / MANUAL).exists()

    # Nothing is left for the sync to find, so only the report remains.
    index(documents_dir, db_path, store)

    assert [record.path for record in Catalog(db_path).all()] == [REPORT]


def test_a_file_the_sync_would_not_read_is_not_a_warning(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """A row whose file the sync would skip is deleted without a word.

    Only PDFs are indexed, so the file is not counted as present.
    """
    index(documents_dir, db_path, store)

    Catalog(db_path).add_file("notes.txt", "notes")
    (documents_dir / "notes.txt").write_text("not a document")

    report = delete(documents_dir, db_path, store, ["notes.txt"])

    assert report.deleted == ["notes.txt"]
    assert report.file_present == []


def test_the_same_path_twice_is_one_document(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """A path asked for twice is deleted once."""
    index(documents_dir, db_path, store)

    report = delete(documents_dir, db_path, store, [MANUAL, MANUAL])

    assert report.deleted == [MANUAL]


def test_a_failed_removal_keeps_the_row(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """A document the store refuses to delete is reported as failed.

    Its row stays in the catalog, and the rest of the job goes ahead.
    """
    index(documents_dir, db_path, store)
    store.fail_delete_on = {MANUAL}

    report = delete(documents_dir, db_path, store, [MANUAL, REPORT])

    assert report.failed == [(MANUAL, f"refused to delete {MANUAL}")]

    assert report.deleted == [REPORT]

    # The row is untouched, so the next sync still sees the document.
    record = Catalog(db_path).get(MANUAL)
    assert record is not None
    assert record.status == "indexed"


def test_a_path_the_catalog_does_not_know_is_only_reported(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """A path with no row in the catalog is listed as missing.

    Nothing is deleted from the store for it.
    """
    index(documents_dir, db_path, store)
    seen = len(store.events)

    report = delete(documents_dir, db_path, store, ["ghosts/ghost.pdf"])

    assert report.missing == ["ghosts/ghost.pdf"]
    assert report.deleted == []
    assert store.events[seen:] == []


def test_a_real_run_needs_a_vector_store(
    documents_dir: Path, db_path: Path
) -> None:
    """Deleting for real without a store is a mistake and raises."""
    with pytest.raises(ValueError, match="vector store"):
        delete(documents_dir, db_path, None, [MANUAL])


def test_a_dry_run_changes_nothing(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """A dry run reports the document and leaves the library alone."""
    index(documents_dir, db_path, store)
    seen = len(store.events)

    report = delete(documents_dir, db_path, store, [MANUAL], dry_run=True)

    assert report.deleted == [MANUAL]
    assert report.file_present == [MANUAL]
    assert store.events[seen:] == []
    assert Catalog(db_path).get(MANUAL) is not None
    assert (documents_dir / MANUAL).is_file()


def test_a_dry_run_reports_the_whole_job(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """A dry run reports the file as removed while it is still on disk."""
    index(documents_dir, db_path, store)

    report = delete(
        documents_dir, db_path, store, [MANUAL], remove_files=True, dry_run=True
    )

    assert report.deleted == [MANUAL]
    assert report.files_removed == [MANUAL]
    assert report.file_present == []

    assert (documents_dir / MANUAL).is_file()


def test_a_dry_run_creates_no_catalog(
    tmp_path: Path, db_path: Path
) -> None:
    """A dry run over a library with no catalog writes no catalog."""
    documents_dir = tmp_path / "documents"
    make_pdf(documents_dir / "manual.pdf", SENTENCE * 5)

    report = delete(documents_dir, db_path, None, ["manual.pdf"], dry_run=True)

    assert report.missing == ["manual.pdf"]
    assert not db_path.exists()


def test_the_trash_lists_what_the_sync_trashed(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """A file removed from the library is listed in the trash after a sync."""
    index(documents_dir, db_path, store)
    (documents_dir / REPORT).unlink()
    index(documents_dir, db_path, store)

    assert trashed_paths(db_path) == [REPORT]


def test_a_library_that_was_never_synced_has_an_empty_trash(
    tmp_path: Path,
) -> None:
    """Asking for the trash of a library with no catalog gives nothing.

    No database file is created to answer the question.
    """
    db_path = tmp_path / "catalog.sqlite3"

    assert trashed_paths(db_path) == []
    assert not db_path.exists()


def test_emptying_the_trash(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """Deleting everything in the trash leaves its files gone for good."""
    index(documents_dir, db_path, store)
    (documents_dir / REPORT).unlink()
    index(documents_dir, db_path, store)

    # The trashed paths are the ones to hand to a deletion.
    report = delete(documents_dir, db_path, store, trashed_paths(db_path))

    assert report.deleted == [REPORT]
    assert report.file_present == []
    assert Catalog(db_path).get(REPORT) is None
    assert trashed_paths(db_path) == []

    index(documents_dir, db_path, store)

    assert [record.path for record in Catalog(db_path).all()] == [MANUAL]
