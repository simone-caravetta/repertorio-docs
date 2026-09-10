from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.vectorstores import VectorStore

from app.catalog import Catalog
from app.ingestion import SUPPORTED_EXTENSIONS, compute_file_hash, ingest_one

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

    The filter delete is a Pinecone extension of the `VectorStore` interface and
    only reaches the namespace this store is bound to. Deleting the old chunks
    before the new ones are written is what keeps a crash from leaving two
    generations of the same document in the index. It also means that deleting
    vectors outside the catalog is not recoverable: the document would have to
    be re-ingested from scratch.
    """
    vectorstore.delete(filter={"source": source})


def sync_documents(
    documents_dir: Path,
    db_path: Path,
    vectorstore: VectorStore | None,
    *,
    dry_run: bool = False,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> SyncReport:
    """Reconcile the documents folder, the catalog and the vector store.

    The folder is what exists, the catalog is what is known: this compares the
    two and applies the difference, in this order — remove the old chunks, then
    write the new ones, then record the outcome. Every step that is interrupted
    leaves a state the next run recognises and repairs: a row with no hash, or
    with a status no completed run leaves behind, is simply processed again.

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
        ):
            # The second test reads the file, so it is only reached for a row a
            # completed run left behind. A row without a hash always differs.
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
        )

    for source in report.added:
        catalog.add_file(source, title=Path(source).stem)
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
