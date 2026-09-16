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
    """A run over a folder, with the sizes and the model a test may replace."""
    return sync_documents(
        documents_dir,
        db_path,
        store,
        **{"embedding_model": EMBEDDING_MODEL, **CHUNKING, **kwargs},
    )


def restamp(
    documents_dir: Path, db_path: Path, source: str, **fingerprint: object
) -> None:
    """Leave a document as a run of other code would have left it.

    The file hash is kept as it is, so the only thing about the row that differs
    is what built its index — which is the case nothing else in the sync looks
    at, and so the one these tests are about.
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
    """The folder files a document once. After that the catalog is the authority.

    Without this the next sync would put every document moved by hand back where
    its folder says it is, which would make the move a thing that lasts until the
    next sync rather than a thing that is stored.
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

    # Which also says what the run before left on the rows matches this one: a
    # fingerprint that never matched would report every document as updated, on
    # every run, for ever.
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

    # The old chunks go before the new ones arrive, so an interrupted run can
    # never leave both generations of the same document in the index.
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
    # The metadata survives the removal, ready for a restore.
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
    assert record.id == before  # the same row, not a new one
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
    # ... and the rest of the folder is still processed.
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

    # Nothing was written for it, and it is offered to the extractor again on
    # the next run — a scanned document costs local CPU and nothing else.
    assert "scans/scan.pdf" not in store.sources
    assert ("scans/scan.pdf", "No text extracted") in run(
        documents_dir, db_path, store
    ).failed


def test_a_failed_document_that_is_removed_is_trashed(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """A document with no vectors still has to trash cleanly.

    A scanned PDF fails with nothing written, so it has a row and no chunks.
    Removing the file then asks the store to delete vectors that were never
    there — and the delete is not guarded at the call site, because the update
    path must not re-add on top of stale chunks. A store that raises on an empty
    match would leave this row failed forever, breaking every later run.
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
    # What a run killed halfway through leaves behind.
    Catalog(db_path).set_status(MANUAL, "indexing")

    report = run(documents_dir, db_path, store)

    assert report.updated == [MANUAL]
    assert Catalog(db_path).get(MANUAL).status == "indexed"


def test_a_row_without_a_hash_is_picked_up_again(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    # A row created but never indexed: an add interrupted before the file was read.
    Catalog(db_path).add_file(MANUAL, "manual")

    report = run(documents_dir, db_path, store)

    assert report.updated == [MANUAL]
    assert report.added == [REPORT]
    assert Catalog(db_path).get(MANUAL).status == "indexed"


def test_what_built_the_index_is_written_on_the_row(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """Nothing else carries it: not the file, and not the vectors."""
    run(documents_dir, db_path, store)

    record = Catalog(db_path).get(MANUAL)
    assert record.indexer == indexer_id(**CHUNKING)
    assert record.embedding_model == EMBEDDING_MODEL


def test_a_document_indexed_by_an_older_reading_is_indexed_again(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """The file is the same file and its hash still matches.

    Only the fingerprint says that the code which made these vectors is not the
    code in hand, so without it the index would go on answering from text the
    reader would no longer produce.
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
    """The case that costs something, and the one nothing else can see.

    Two models make two vector spaces, so a corpus moved to a new one has to be
    made again — and no byte of any document has changed.
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
    """A chunk size is part of what built an index, and the file has not moved.

    Changing it and not rebuilding would leave every answer citing chunks cut to
    the sizes the settings no longer name.
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
    """A catalog written before the fingerprint existed says nothing about how
    its index was built, so the first run after the change builds it."""
    run(documents_dir, db_path, store)
    # What a row indexed by an older version of this code looks like: current
    # in every other way, and silent about what made it.
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
    assert documents_dir.is_dir()  # created on the way in


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

    # The catalog still describes the last real run, so that run has all of
    # this left to do.
    catalog = Catalog(db_path, create=False)
    assert catalog.get(MANUAL).file_hash == indexed_hash
    assert catalog.get(REPORT).status == "indexed"
    assert catalog.get("scans/scan.pdf") is None


# --- the descriptions, written as the documents are indexed ------------------
#
# What a description is, and how one is written, is tested in
# `tests/test_descriptions.py`. Here it is the sync's part in it: which documents
# a run describes, which it leaves alone, and what a call that fails costs.


def test_a_new_document_is_described_as_it_is_indexed(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    model = FakeChatModel(replies=["Una guía del manual.", "Last year's report."])

    report = run(documents_dir, db_path, store, chat_model=model)

    # One call per document, in catalog order, and each one kept.
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
    # A model with no replies at all: the first call to it would fail, so a run
    # that reaches the description pass is a run that fails this test.
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

    # A run that describes the same documents a second time is a run that spends
    # a call per document for nothing, and this says it does not.
    model = FakeChatModel(replies=[])
    report = run(documents_dir, db_path, store, chat_model=model)

    assert report.described == []
    assert model.prompts == []
    assert Catalog(db_path).get(MANUAL).description == "A manual."


def test_a_document_with_no_description_is_described_on_a_later_run(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    # A library indexed before the sync wrote descriptions, or by a run told not
    # to: nothing about the files says so, and the next run writes them.
    run(documents_dir, db_path, store)

    model = FakeChatModel(replies=["A manual.", "A report."])
    report = run(documents_dir, db_path, store, chat_model=model)

    assert report.skipped == [MANUAL, REPORT]  # no vectors were written
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

    # The document that did not change was not described: one call, not two.
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

    # The old description described text that is no longer in the document. Left
    # in place it would be read as a description of this edition, which is the
    # one outcome worth more than a blank.
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
    # Another reader or another cut: the same file, indexed again.
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
    # Every call raises: a key that stopped working, an endpoint that is down.
    report = run(documents_dir, db_path, store, chat_model=FakeChatModel(replies=[]))

    assert report.described == []
    assert [path for path, _ in report.description_failed] == [MANUAL, REPORT]
    assert report.failed == []  # the documents themselves are not the failure

    catalog = Catalog(db_path)
    for source in (MANUAL, REPORT):
        assert catalog.get(source).status == "indexed"
        assert catalog.get(source).chunk_count > 0
        assert catalog.get(source).description is None
