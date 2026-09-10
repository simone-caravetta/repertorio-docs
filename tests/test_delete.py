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
from tests.helpers import SENTENCE, FakeVectorStore, make_pdf

MANUAL = "manuals/manual.pdf"
REPORT = "reports/report.pdf"

CHUNKING = {"chunk_size": 200, "chunk_overlap": 20}


def index(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> SyncReport:
    return sync_documents(documents_dir, db_path, store, **CHUNKING)


def delete(
    documents_dir: Path,
    db_path: Path,
    store: FakeVectorStore | None,
    paths: list[str],
    **kwargs: object,
) -> DeleteReport:
    return delete_documents(documents_dir, db_path, store, paths, **kwargs)


def test_a_document_leaves_the_index_and_the_catalog(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    index(documents_dir, db_path, store)
    seen = len(store.events)

    report = delete(documents_dir, db_path, store, [MANUAL])

    assert report.deleted == [MANUAL]
    assert report.failed == []
    assert store.events[seen:] == [("delete", {"source": MANUAL})]

    catalog = Catalog(db_path)
    assert catalog.get(MANUAL) is None
    # The rest of the library is untouched.
    assert catalog.get(REPORT).status == "indexed"


def test_the_file_stays_where_it_is_and_the_document_comes_back(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    index(documents_dir, db_path, store)
    before = Catalog(db_path).get(MANUAL).id

    report = delete(documents_dir, db_path, store, [MANUAL])

    assert (documents_dir / MANUAL).is_file()
    assert report.file_present == [MANUAL]
    assert report.files_removed == []

    # The sync watches the folder, so the document is back on the next run —
    # as a new one: the title and the history went with the row.
    index(documents_dir, db_path, store)

    record = Catalog(db_path).get(MANUAL)
    assert record.status == "indexed"
    assert record.id != before


def test_with_the_file_the_removal_holds(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    index(documents_dir, db_path, store)

    report = delete(documents_dir, db_path, store, [MANUAL], remove_files=True)

    assert report.deleted == [MANUAL]
    assert report.files_removed == [MANUAL]
    assert report.file_present == []
    assert not (documents_dir / MANUAL).exists()

    # Nothing is left for the next sync to find.
    index(documents_dir, db_path, store)

    assert [record.path for record in Catalog(db_path).all()] == [REPORT]


def test_a_file_the_sync_would_not_read_is_not_a_warning(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    index(documents_dir, db_path, store)
    # A row the folder scan would not have created: what comes back on the next
    # sync is a file it can read, not any file that happens to be there.
    Catalog(db_path).add_file("notes.txt", "notes")
    (documents_dir / "notes.txt").write_text("not a document")

    report = delete(documents_dir, db_path, store, ["notes.txt"])

    assert report.deleted == ["notes.txt"]
    assert report.file_present == []


def test_the_same_path_twice_is_one_document(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    index(documents_dir, db_path, store)

    report = delete(documents_dir, db_path, store, [MANUAL, MANUAL])

    assert report.deleted == [MANUAL]


def test_a_failed_removal_keeps_the_row(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    index(documents_dir, db_path, store)
    store.fail_delete_on = {MANUAL}

    report = delete(documents_dir, db_path, store, [MANUAL, REPORT])

    assert report.failed == [(MANUAL, f"refused to delete {MANUAL}")]
    # One document must not stop the others.
    assert report.deleted == [REPORT]

    # The chunks are still in the index, so the row has to stay and describe
    # them: dropping it here would leave nothing referring to them.
    record = Catalog(db_path).get(MANUAL)
    assert record is not None
    assert record.status == "indexed"


def test_a_path_the_catalog_does_not_know_is_only_reported(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    index(documents_dir, db_path, store)
    seen = len(store.events)

    report = delete(documents_dir, db_path, store, ["ghosts/ghost.pdf"])

    assert report.missing == ["ghosts/ghost.pdf"]
    assert report.deleted == []
    assert store.events[seen:] == []


def test_a_real_run_needs_a_vector_store(
    documents_dir: Path, db_path: Path
) -> None:
    with pytest.raises(ValueError, match="vector store"):
        delete(documents_dir, db_path, None, [MANUAL])


def test_a_dry_run_changes_nothing(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
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
    index(documents_dir, db_path, store)

    report = delete(
        documents_dir, db_path, store, [MANUAL], remove_files=True, dry_run=True
    )

    assert report.deleted == [MANUAL]
    assert report.files_removed == [MANUAL]
    assert report.file_present == []
    # Reported, not done.
    assert (documents_dir / MANUAL).is_file()


def test_a_dry_run_creates_no_catalog(
    tmp_path: Path, db_path: Path
) -> None:
    documents_dir = tmp_path / "documents"
    make_pdf(documents_dir / "manual.pdf", SENTENCE * 5)

    report = delete(documents_dir, db_path, None, ["manual.pdf"], dry_run=True)

    assert report.missing == ["manual.pdf"]
    assert not db_path.exists()


def test_the_trash_lists_what_the_sync_trashed(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    index(documents_dir, db_path, store)
    (documents_dir / REPORT).unlink()
    index(documents_dir, db_path, store)

    assert trashed_paths(db_path) == [REPORT]


def test_a_library_that_was_never_synced_has_an_empty_trash(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "catalog.sqlite3"

    assert trashed_paths(db_path) == []
    assert not db_path.exists()


def test_emptying_the_trash(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    index(documents_dir, db_path, store)
    (documents_dir / REPORT).unlink()
    index(documents_dir, db_path, store)

    # What `delete --trashed` does: the file is already gone, so there is
    # nothing left but the row.
    report = delete(documents_dir, db_path, store, trashed_paths(db_path))

    assert report.deleted == [REPORT]
    assert report.file_present == []
    assert Catalog(db_path).get(REPORT) is None
    assert trashed_paths(db_path) == []

    index(documents_dir, db_path, store)

    assert [record.path for record in Catalog(db_path).all()] == [MANUAL]
