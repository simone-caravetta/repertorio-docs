from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from app.catalog import Catalog, DocumentRecord, category_from_path
from app.descriptions import undescribed, write_descriptions
from app.ingestion import (
    SUPPORTED_EXTENSIONS,
    compute_file_hash,
    indexer_id,
    ingest_one,
)

# A row in one of these states is not the result of a completed run: either the
# run that created it never finished, or it failed. Both are retried.
REPROCESS_STATUSES = frozenset({"queued", "indexing", "failed"})

NO_TEXT_ERROR = "No text extracted"


@dataclass
class SyncReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    restored: list[str] = field(default_factory=list)
    trashed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    described: list[str] = field(default_factory=list)
    description_failed: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class DeleteReport:
    deleted: list[str] = field(default_factory=list)
    files_removed: list[str] = field(default_factory=list)
    # Deleted from the index and the catalog, with the file still in the
    # documents folder: the next sync indexes it again as a new document.
    file_present: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)


def delete_document_vectors(source: str, vectorstore: VectorStore) -> None:
    """Remove every chunk of one document from the vector store.

    `filter=` is an extension of the `VectorStore` interface that the hosted
    store provides natively; the local one reaches the same rows through its own
    `where=`, translated in `app.chroma_store`. Either way this is the only place
    document-level vector deletion happens, and it reaches only the namespace or
    collection the store is bound to. Deleting the old chunks before the new ones
    are written is what keeps a crash from leaving two generations of the same
    document in the store. It also means that deleting vectors outside the
    catalog is not recoverable: the document would have to be re-ingested from
    scratch.
    """
    vectorstore.delete(filter={"source": source})


def _indexed_as_this_run_would(
    record: DocumentRecord, *, indexer: str, embedding_model: str
) -> bool:
    """Whether the vectors on record were built the way this run would build them.

    A row with no fingerprint was indexed before the catalog kept one, and the
    answer is no: the reading it was made with is not known, and the only way to
    know it is current is to make it again.
    """
    return record.indexer == indexer and record.embedding_model == embedding_model


def sync_documents(
    documents_dir: Path,
    db_path: Path,
    vectorstore: VectorStore | None,
    *,
    embedding_model: str,
    dry_run: bool = False,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    chat_model: BaseChatModel | None = None,
    description_sample_chars: int = 6000,
) -> SyncReport:
    """Reconcile the documents folder, the catalog and the vector store.

    The folder is what exists, the catalog is what is known: this compares the
    two and applies the difference, in this order — remove the old chunks, then
    write the new ones, then record the outcome. Every step that is interrupted
    leaves a state the next run recognises and repairs: a row with no hash, or
    with a status no completed run leaves behind, is simply processed again.

    A document is also made again when what built its index is not what this run
    would build it with: another reading, another chunk size, another model. The
    file has not changed and the vectors do not say, so the fingerprint on the
    row is the only thing that can tell, and without it a library that moved to
    a new model would go on answering from the old one's vectors.

    `embedding_model` is required for the same reason: a run that does not know
    which model it is using cannot record one, and a run that records none makes
    every document stale for the run after it.

    When a `chat_model` is given, the run also writes each document's
    description — the sentence read back by the console, the page and the context
    of an answer — by the same call `scripts.describe` makes, over the sample it
    takes. A description is written from the text of one edition of a file, so a
    file this run reads as a different one drops it, and every indexed document
    with nothing written about it is written about: one just indexed, one indexed
    before the sync did this, one whose call failed on the run before. A call
    that fails is reported and the run goes on, because the document being
    indexed is a fact of its own; a run that passes no model describes nothing,
    which is what a dry run, a machine with no key and `--no-descriptions` all
    come to.

    Describing reads the file a second time, through `load_pdf`. That is one
    reading of a document, on the run that describes it, and it buys the
    description written here being the one the command would have written —
    rather than a second way of writing one that agrees with it until it does not.

    With `dry_run` nothing is written anywhere: no directory is created, no
    catalog, no vector store. The returned report describes what a real run
    would do.
    """
    if not dry_run and vectorstore is None:
        raise ValueError("A vector store is required to apply the changes")

    documents_dir = Path(documents_dir)
    if not dry_run:
        documents_dir.mkdir(parents=True, exist_ok=True)

    catalog = Catalog(db_path, create=not dry_run)
    report = SyncReport()

    # What this run marks a document it indexes with, read back off the row on
    # the next run to see whether the vectors were made by the code in hand.
    indexer = indexer_id(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    files = {
        str(path.relative_to(documents_dir)): path
        for path in documents_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    }
    rows = {record.path: record for record in catalog.all()}

    for source in sorted(files):
        record = rows.get(source)

        if record is None:
            report.added.append(source)
        elif record.status == "trashed":
            # The file is back: the metadata it kept is still valid.
            report.restored.append(source)
        elif (
            record.status in REPROCESS_STATUSES
            or compute_file_hash(files[source]) != record.file_hash
            or not _indexed_as_this_run_would(
                record, indexer=indexer, embedding_model=embedding_model
            )
        ):
            # The second test reads the file, so it is only reached for a row a
            # completed run left behind. A row without a hash always differs.
            # The third is what a changed reader, chunk size or model comes to,
            # and it is the only one that looks at nothing but the row.
            report.updated.append(source)
        else:
            report.skipped.append(source)

    for source in sorted(rows):
        # A path that is no longer a supported file: deleted, or renamed to
        # something this library does not read. Already trashed rows stay put.
        if source not in files and rows[source].status != "trashed":
            report.trashed.append(source)

    if dry_run:
        return report

    def index(source: str) -> None:
        # The row as the run found it: what the file hashed to before this run
        # read it, which is what says whether a description on it describes the
        # text that is going into the index or the text that was there before.
        before = rows.get(source)

        catalog.set_status(source, "indexing")

        try:
            result = ingest_one(
                files[source],
                documents_dir=documents_dir,
                vectorstore=vectorstore,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except Exception as exc:  # noqa: BLE001
            # One bad document must not stop the run: whatever the reader, the
            # embedding model or the network raises is recorded on the row.
            error = str(exc) or exc.__class__.__name__
            report.failed.append((source, error))
            catalog.record_failed(source, error)
            return

        if not result.chunk_count:
            report.failed.append((source, NO_TEXT_ERROR))
            catalog.record_failed(source, NO_TEXT_ERROR)
            return

        catalog.record_indexed(
            source,
            file_hash=result.file_hash,
            page_count=result.page_count,
            chunk_count=result.chunk_count,
            indexer=indexer,
            embedding_model=embedding_model,
        )

        # A description of another edition of this file is about a document that
        # is no longer the one here, so it goes, and the pass below writes the
        # new one. A row whose hash is the one just written describes this text —
        # the file is untouched and only the fingerprint moved, which is what a
        # changed cut or another embedding model comes to, and neither of them
        # changes a line of what the description is written from.
        if before is not None and before.file_hash != result.file_hash:
            catalog.clear_description(source)

    for source in report.added:
        # The folder names the category once, here, when the row is created. From
        # then on the catalog owns it, so a document moved by hand stays where it
        # was moved to even after a re-index.
        catalog.add_file(
            source,
            title=Path(source).stem,
            category=category_from_path(source),
        )
        index(source)

    for source in report.restored:
        catalog.set_status(source, "queued")
        index(source)

    for source in report.updated:
        catalog.set_status(source, "queued")
        delete_document_vectors(source, vectorstore)
        index(source)

    for source in report.trashed:
        delete_document_vectors(source, vectorstore)
        catalog.trash(source)

    if chat_model is not None:
        # After the catalog has settled, and over the library rather than over
        # this run's work: what has no description is what has none, whatever
        # left it that way.
        written = write_descriptions(
            chat_model,
            catalog,
            undescribed(catalog),
            documents_dir=documents_dir,
            sample_chars=description_sample_chars,
        )
        report.described = [path for path, _ in written.written]
        report.description_failed = list(written.failed)

    return report


def trashed_paths(db_path: Path) -> list[str]:
    """Every document currently in the trash, in path order.

    Reads the catalog without creating it: a library that was never synced has
    an empty trash, not a missing one.
    """
    return [
        record.path
        for record in Catalog(db_path, create=False).all()
        if record.status == "trashed"
    ]


def delete_documents(
    documents_dir: Path,
    db_path: Path,
    vectorstore: VectorStore | None,
    paths: Iterable[str],
    *,
    remove_files: bool = False,
    dry_run: bool = False,
) -> DeleteReport:
    """Remove documents from the vector store and from the catalog.

    This is the end of the line for a document: the row goes with the chunks, so
    nothing is left to restore and a document that comes back comes back as a
    new one. Trashing stays the reversible half of the same idea — the way to
    take a document out of the answers without losing it.

    The order is the one the trash path uses: chunks first, then the row. The
    two interruptions are not equally bad. With the chunks gone and the row
    still there, the next sync notices and repairs it; with the row deleted
    first, the chunks are left in the index and nothing refers to them any more.

    Deleting a document whose file is still in the folder is a legitimate way to
    start it over, but it does not hold: the folder is what the sync watches,
    and a file the catalog does not know is indexed as a new document. Those
    paths are reported in `file_present`; `remove_files` deletes the file too,
    which is what makes the removal final.
    """
    if not dry_run and vectorstore is None:
        raise ValueError("A vector store is required to apply the changes")

    documents_dir = Path(documents_dir)
    catalog = Catalog(db_path, create=not dry_run)
    report = DeleteReport()

    for source in sorted(set(paths)):
        if catalog.get(source) is None:
            # The catalog decides what exists: a path it does not hold was
            # never indexed, or has already been removed.
            report.missing.append(source)
            continue

        path = documents_dir / source
        present = path.is_file()

        if not dry_run:
            try:
                delete_document_vectors(source, vectorstore)
                catalog.delete(source)
            except Exception as exc:  # noqa: BLE001
                # One document must not stop the rest, the same rule the sync
                # follows.
                report.failed.append((source, str(exc) or exc.__class__.__name__))
                continue

            if remove_files and present:
                path.unlink()

        report.deleted.append(source)

        if remove_files and present:
            report.files_removed.append(source)
        elif present and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            # The scan in `sync_documents` looks for exactly this: a file it can
            # still see is a document it will index again from scratch.
            report.file_present.append(source)

    return report
