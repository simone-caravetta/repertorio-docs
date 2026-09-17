from __future__ import annotations

from pathlib import Path

from app.catalog import Catalog
from app.ingestion import compute_file_hash, indexer_id
from app.lifecycle import SyncReport, sync_documents
from tests.helpers import (
    EMBEDDING_MODEL,
    SENTENCE,
    FakeChatModel,
    FakeVectorStore,
    make_pdf,
)

MANUAL = "manuals/manual.pdf"
REPORT = "reports/report.pdf"

CHUNKING = {"chunk_size": 200, "chunk_overlap": 20}


def run(
    documents_dir: Path,
    db_path: Path,
    store: FakeVectorStore | None,
    **kwargs: object,
) -> SyncReport:
    """Run a sync over the documents folder with the test settings.

    The embedding model and the chunking numbers are filled in here. A
    keyword argument given by the caller is added on top of them.
    """

    return sync_documents(
        documents_dir,
        db_path,
        store,
        **{"embedding_model": EMBEDDING_MODEL, **CHUNKING, **kwargs},
    )


def restamp(
    documents_dir: Path, db_path: Path, source: str, **fingerprint: object
) -> None:
    """Write a catalog row for an indexed document with a chosen fingerprint.

    The file hash comes from the file on disk, so the row looks current.
    The keyword arguments replace the fields that name what built the
    index, which is how a row left by an older reading is staged.
    """

    Catalog(db_path).record_indexed(
        source,
        file_hash=compute_file_hash(documents_dir / source),
        page_count=1,
        chunk_count=1,
        **fingerprint,  # type: ignore[arg-type]
    )


def edit(documents_dir: Path, source: str, text: str) -> None:
    make_pdf(documents_dir / source, text)


def test_new_files_are_indexed(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    report = run(documents_dir, db_path, store)

    assert report.added == [MANUAL, REPORT]
    assert report.updated == []
    assert report.restored == []
    assert report.trashed == []
    assert report.failed == []
    assert report.skipped == []

    catalog = Catalog(db_path)
    for source in (MANUAL, REPORT):
        record = catalog.get(source)
        assert record.status == "indexed"
        assert record.title == Path(source).stem
        assert record.file_hash == compute_file_hash(documents_dir / source)
        assert record.page_count == 1
        assert record.chunk_count > 0
        assert record.error_message is None

    assert sorted(set(store.sources)) == [MANUAL, REPORT]


def test_a_new_file_is_filed_under_the_folder_it_sits_in(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    run(documents_dir, db_path, store)

    catalog = Catalog(db_path)

    assert catalog.get(MANUAL).category == "manuals"
    assert catalog.get(REPORT).category == "reports"


def test_a_re_index_leaves_a_moved_document_where_it_was_moved(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """A re-index does not put a document back in its folder's category.

    The category is set by hand between the two runs. The file is edited
    first, so the second run indexes the document again and the category
    stays where it was moved.
    """

    run(documents_dir, db_path, store)
    catalog = Catalog(db_path)
    catalog.set_category(MANUAL, "archive")

    edit(documents_dir, MANUAL, SENTENCE * 30)
    report = run(documents_dir, db_path, store)

    assert report.updated == [MANUAL]
    assert catalog.get(MANUAL).category == "archive"


def test_unchanged_files_are_left_alone(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    run(documents_dir, db_path, store)
    seen = len(store.events)

    report = run(documents_dir, db_path, store)

    # The second run finds both files as they were, so nothing reached the
    # store.
    assert report.skipped == [MANUAL, REPORT]
    assert report.added == []
    assert report.updated == []
    assert store.events[seen:] == []


def test_an_edited_file_is_indexed_again(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    run(documents_dir, db_path, store)
    before = Catalog(db_path).get(MANUAL).file_hash
    seen = len(store.events)

    edit(documents_dir, MANUAL, SENTENCE * 5 + "A second edition, revised.")
    report = run(documents_dir, db_path, store)

    assert report.updated == [MANUAL]
    assert report.skipped == [REPORT]

    # The chunks of the old reading go before the new ones are written.
    assert store.events[seen] == ("delete", {"source": MANUAL})
    assert store.events[seen + 1][0] == "add"

    record = Catalog(db_path).get(MANUAL)
    assert record.status == "indexed"
    assert record.file_hash == compute_file_hash(documents_dir / MANUAL)
    assert record.file_hash != before


def test_a_removed_file_is_trashed(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    run(documents_dir, db_path, store)
    (documents_dir / REPORT).unlink()

    report = run(documents_dir, db_path, store)

    assert report.trashed == [REPORT]
    assert report.added == []
    assert ("delete", {"source": REPORT}) in store.events

    record = Catalog(db_path).get(REPORT)
    assert record.status == "trashed"
    assert record.trashed_at is not None

    assert record.title == "report"
    assert record.chunk_count > 0


def test_a_trashed_file_comes_back(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    run(documents_dir, db_path, store)
    path = documents_dir / REPORT
    body = path.read_bytes()
    before = Catalog(db_path).get(REPORT).id

    path.unlink()
    run(documents_dir, db_path, store)
    path.write_bytes(body)
    report = run(documents_dir, db_path, store)

    assert report.restored == [REPORT]

    record = Catalog(db_path).get(REPORT)
    assert record.id == before
    assert record.status == "indexed"
    assert record.trashed_at is None
    assert record.title == "report"


def test_a_failed_ingest_is_recorded_and_retried(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    store.fail_on = {MANUAL}

    report = run(documents_dir, db_path, store)

    assert report.failed == [(MANUAL, f"refused to index {MANUAL}")]
    catalog = Catalog(db_path)
    assert catalog.get(MANUAL).status == "failed"
    assert "refused" in catalog.get(MANUAL).error_message

    assert catalog.get(REPORT).status == "indexed"

    store.fail_on = set()
    report = run(documents_dir, db_path, store)

    assert report.updated == [MANUAL]
    assert Catalog(db_path).get(MANUAL).status == "indexed"


def test_a_pdf_without_text_is_recorded_as_failed(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    make_pdf(documents_dir / "scans" / "scan.pdf", "")

    report = run(documents_dir, db_path, store)

    assert ("scans/scan.pdf", "No text extracted") in report.failed

    record = Catalog(db_path).get("scans/scan.pdf")
    assert record.status == "failed"
    assert record.chunk_count is None

    # A scan leaves nothing in the store, and a later run tries it again.
    assert "scans/scan.pdf" not in store.sources
    assert ("scans/scan.pdf", "No text extracted") in run(
        documents_dir, db_path, store
    ).failed


def test_a_failed_document_that_is_removed_is_trashed(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """A document that failed and then went away is trashed.

    The file is here for the first run and gone for the second, which marks
    the row as trashed instead of counting the document as failed again.
    """

    make_pdf(documents_dir / "scans" / "scan.pdf", "")
    run(documents_dir, db_path, store)
    (documents_dir / "scans" / "scan.pdf").unlink()

    report = run(documents_dir, db_path, store)

    assert report.trashed == ["scans/scan.pdf"]
    assert report.failed == []
    assert Catalog(db_path).get("scans/scan.pdf").status == "trashed"


def test_an_interrupted_run_is_picked_up_again(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    run(documents_dir, db_path, store)

    Catalog(db_path).set_status(MANUAL, "indexing")

    report = run(documents_dir, db_path, store)

    assert report.updated == [MANUAL]
    assert Catalog(db_path).get(MANUAL).status == "indexed"


def test_a_row_without_a_hash_is_picked_up_again(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    Catalog(db_path).add_file(MANUAL, "manual")

    report = run(documents_dir, db_path, store)

    assert report.updated == [MANUAL]
    assert report.added == [REPORT]
    assert Catalog(db_path).get(MANUAL).status == "indexed"


def test_what_built_the_index_is_written_on_the_row(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    run(documents_dir, db_path, store)

    record = Catalog(db_path).get(MANUAL)
    assert record.indexer == indexer_id(**CHUNKING)
    assert record.embedding_model == EMBEDDING_MODEL


def test_a_document_indexed_by_an_older_reading_is_indexed_again(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """A document read by an older version of the reader is read again.

    The row is stamped with a reader this run does not use. The fingerprint
    no longer matches, so the document is indexed once more.
    """

    run(documents_dir, db_path, store)
    restamp(
        documents_dir,
        db_path,
        MANUAL,
        indexer="pymupdf-0|200/20",
        embedding_model=EMBEDDING_MODEL,
    )

    report = run(documents_dir, db_path, store)

    assert report.updated == [MANUAL]
    assert report.skipped == [REPORT]
    assert Catalog(db_path).get(MANUAL).indexer == indexer_id(**CHUNKING)


def test_a_document_indexed_by_another_model_is_indexed_again(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """A document embedded by another model is indexed again.

    Vectors from two models cannot be compared, so a change of model means
    the document has to be built once more.
    """

    run(documents_dir, db_path, store)
    restamp(
        documents_dir,
        db_path,
        MANUAL,
        indexer=indexer_id(**CHUNKING),
        embedding_model="some-other-model",
    )

    report = run(documents_dir, db_path, store)

    assert report.updated == [MANUAL]
    assert Catalog(db_path).get(MANUAL).embedding_model == EMBEDDING_MODEL


def test_the_cut_is_part_of_the_fingerprint(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """The chunk sizes are part of the fingerprint.

    A run with a different cut indexes both documents again and writes the
    new numbers on the row.
    """

    run(documents_dir, db_path, store)

    report = run(documents_dir, db_path, store, chunk_size=300)

    assert report.updated == [MANUAL, REPORT]
    assert Catalog(db_path).get(MANUAL).indexer == indexer_id(
        chunk_size=300, chunk_overlap=CHUNKING["chunk_overlap"]
    )


def test_a_row_with_no_fingerprint_is_indexed_again(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """A row written before the fingerprint existed is indexed again.

    The hash and the counts are current here. Only the fields naming the
    reader and the model are left empty.
    """

    run(documents_dir, db_path, store)

    # Record the same file again, this time without naming what built the
    # index.
    Catalog(db_path).record_indexed(
        MANUAL,
        file_hash=compute_file_hash(documents_dir / MANUAL),
        page_count=1,
        chunk_count=1,
    )

    report = run(documents_dir, db_path, store)

    assert report.updated == [MANUAL]
    assert report.skipped == [REPORT]
    assert Catalog(db_path).get(MANUAL).indexer == indexer_id(**CHUNKING)


def test_files_this_library_does_not_read_are_ignored(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    (documents_dir / "notes.txt").write_text("not a document")
    (documents_dir / "reports" / "data.csv").write_text("1,2,3")

    report = run(documents_dir, db_path, store)

    assert report.added == [MANUAL, REPORT]
    assert [record.path for record in Catalog(db_path).all()] == [MANUAL, REPORT]


def test_a_pdf_renamed_to_something_else_is_trashed(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    run(documents_dir, db_path, store)
    (documents_dir / REPORT).rename(documents_dir / "reports" / "report.txt")

    report = run(documents_dir, db_path, store)

    assert report.trashed == [REPORT]
    assert report.added == []


def test_an_empty_folder_has_nothing_to_do(
    tmp_path: Path, db_path: Path, store: FakeVectorStore
) -> None:
    documents_dir = tmp_path / "documents"

    report = run(documents_dir, db_path, store)

    assert report == SyncReport()
    assert documents_dir.is_dir()


def test_a_dry_run_changes_nothing(
    tmp_path: Path, db_path: Path, store: FakeVectorStore
) -> None:
    documents_dir = tmp_path / "documents"
    make_pdf(documents_dir / "manual.pdf", SENTENCE * 5)

    report = run(documents_dir, db_path, None, dry_run=True)

    assert report.added == ["manual.pdf"]
    assert not db_path.exists()
    assert store.events == []


def test_a_dry_run_reports_what_a_real_run_would_do(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    run(documents_dir, db_path, store)
    indexed_hash = Catalog(db_path).get(MANUAL).file_hash
    edit(documents_dir, MANUAL, SENTENCE * 5 + "A second edition, revised.")
    (documents_dir / REPORT).unlink()
    make_pdf(documents_dir / "scans" / "scan.pdf", "")
    seen = len(store.events)

    report = run(documents_dir, db_path, None, dry_run=True)

    assert report.updated == [MANUAL]
    assert report.trashed == [REPORT]
    assert report.added == ["scans/scan.pdf"]
    assert store.events[seen:] == []

    # The catalog is opened for reading, so a file that was never created
    # is not created now.
    catalog = Catalog(db_path, create=False)
    assert catalog.get(MANUAL).file_hash == indexed_hash
    assert catalog.get(REPORT).status == "indexed"
    assert catalog.get("scans/scan.pdf") is None


# The description of a document: which runs write one, which keep the one
# already written, and what happens when the model cannot answer.


def test_a_new_document_is_described_as_it_is_indexed(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    model = FakeChatModel(replies=["Una guía del manual.", "Last year's report."])

    report = run(documents_dir, db_path, store, chat_model=model)

    # One call to the model per document, and each description lands on its
    # own row.
    assert report.described == [MANUAL, REPORT]
    assert report.description_failed == []
    assert len(model.prompts) == 2

    catalog = Catalog(db_path)
    assert catalog.get(MANUAL).description == "Una guía del manual."
    assert catalog.get(REPORT).description == "Last year's report."


def test_a_run_with_no_model_describes_nothing(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    report = run(documents_dir, db_path, store)

    assert report.described == []
    assert report.description_failed == []

    catalog = Catalog(db_path)
    assert catalog.get(MANUAL).status == "indexed"
    assert catalog.get(MANUAL).description is None


def test_a_dry_run_calls_no_model(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """A dry run writes no description, so no model is called.

    The fake records every prompt it is given, and there should be none.
    """

    model = FakeChatModel(replies=[])

    report = run(documents_dir, db_path, None, dry_run=True, chat_model=model)

    assert report.described == []
    assert model.prompts == []


def test_a_described_document_is_not_described_again(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    run(
        documents_dir,
        db_path,
        store,
        chat_model=FakeChatModel(replies=["A manual.", "A report."]),
    )

    # The second run has nothing to describe, so the model is not called.
    model = FakeChatModel(replies=[])
    report = run(documents_dir, db_path, store, chat_model=model)

    assert report.described == []
    assert model.prompts == []
    assert Catalog(db_path).get(MANUAL).description == "A manual."


def test_a_document_with_no_description_is_described_on_a_later_run(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """A document with no description gets one when a model turns up.

    The first run has no model, the second has one. The documents
    themselves are unchanged, so they are not indexed again.
    """

    run(documents_dir, db_path, store)

    model = FakeChatModel(replies=["A manual.", "A report."])
    report = run(documents_dir, db_path, store, chat_model=model)

    assert report.skipped == [MANUAL, REPORT]
    assert report.described == [MANUAL, REPORT]


def test_a_changed_file_is_described_again(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    run(
        documents_dir,
        db_path,
        store,
        chat_model=FakeChatModel(replies=["A manual.", "A report."]),
    )
    edit(documents_dir, MANUAL, SENTENCE * 5 + "A second edition, revised.")

    model = FakeChatModel(replies=["A manual, second edition."])
    report = run(documents_dir, db_path, store, chat_model=model)

    assert report.updated == [MANUAL]
    assert report.described == [MANUAL]
    assert Catalog(db_path).get(MANUAL).description == "A manual, second edition."

    # Only the changed document was sent to the model.
    assert len(model.prompts) == 1


def test_a_changed_file_whose_description_fails_keeps_none(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    run(
        documents_dir,
        db_path,
        store,
        chat_model=FakeChatModel(replies=["A manual.", "A report."]),
    )
    edit(documents_dir, MANUAL, SENTENCE * 5 + "A second edition, revised.")

    report = run(documents_dir, db_path, store, chat_model=FakeChatModel(replies=[]))

    # The model refuses the newer text, and the description written for the
    # older one goes with it.
    assert report.updated == [MANUAL]
    assert Catalog(db_path).get(MANUAL).description is None


def test_a_document_indexed_again_for_its_fingerprint_keeps_its_description(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    run(
        documents_dir,
        db_path,
        store,
        chat_model=FakeChatModel(replies=["A manual.", "A report."]),
    )

    restamp(documents_dir, db_path, MANUAL, indexer="other-reader|1/1")

    model = FakeChatModel(replies=[])
    report = run(documents_dir, db_path, store, chat_model=model)

    assert report.updated == [MANUAL]
    assert report.described == []
    assert model.prompts == []
    assert Catalog(db_path).get(MANUAL).description == "A manual."


def test_a_restored_document_keeps_its_description(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    run(
        documents_dir,
        db_path,
        store,
        chat_model=FakeChatModel(replies=["A manual.", "A report."]),
    )

    path = documents_dir / REPORT
    body = path.read_bytes()
    path.unlink()
    run(documents_dir, db_path, store)
    path.write_bytes(body)

    model = FakeChatModel(replies=[])
    report = run(documents_dir, db_path, store, chat_model=model)

    assert report.restored == [REPORT]
    assert report.described == []
    assert Catalog(db_path).get(REPORT).description == "A report."


def test_a_description_that_cannot_be_written_leaves_the_document_indexed(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """A description that cannot be written does not fail the ingest."""
    report = run(documents_dir, db_path, store, chat_model=FakeChatModel(replies=[]))

    assert report.described == []
    assert [path for path, _ in report.description_failed] == [MANUAL, REPORT]
    assert report.failed == []

    catalog = Catalog(db_path)
    for source in (MANUAL, REPORT):
        assert catalog.get(source).status == "indexed"
        assert catalog.get(source).chunk_count > 0
        assert catalog.get(source).description is None
