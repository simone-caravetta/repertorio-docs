"""Keeping the documents folder and the index in step.

`sync_documents` walks the folder, compares what it finds with the catalog and
indexes what is new or has changed. A document whose file is gone from the
folder goes to the trash, so putting the file back where it was and syncing
again brings it back.

`delete_documents` is the other direction and only runs when it is asked for. It
takes a document out of the index and out of the catalog for good.
"""

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
from app.structure import build_one

# The statuses that mean an earlier attempt did not finish, so the document is
# indexed again on the next run.
REPROCESS_STATUSES = frozenset({"queued", "indexing", "failed"})

# Recorded when a document produces no chunks, which is what a scan without a
# text layer looks like.
NO_TEXT_ERROR = "No text extracted"


@dataclass
class SyncReport:
    """What a run of the sync did."""

    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    restored: list[str] = field(default_factory=list)
    trashed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    described: list[str] = field(default_factory=list)
    description_failed: list[tuple[str, str]] = field(default_factory=list)
    structured: list[str] = field(default_factory=list)
    structure_failed: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class DeleteReport:
    """What a run of the delete command did."""

    deleted: list[str] = field(default_factory=list)
    files_removed: list[str] = field(default_factory=list)

    # How the file itself came through the run.
    file_present: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)


def delete_document_vectors(source: str, vectorstore: VectorStore) -> None:
    """Remove every vector of one document from the store.

    The store is filtered by source, which is the path of the file relative to
    the documents folder.
    """
    vectorstore.delete(filter={"source": source})


def _indexed_as_this_run_would(
    record: DocumentRecord, *, indexer: str, embedding_model: str
) -> bool:
    """Whether the document was indexed by the reader and model in use now.

    A document indexed by another reader version or another embedding model has
    to be indexed again, even when the file itself has not changed.
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
    """Bring the store and the catalog in step with the documents folder.

    Returns what the run found and did. With `dry_run` it does nothing and only
    reports, and no vector store is needed. The folder is compared with the
    catalog first and the changes are applied afterwards, so the report covers
    the whole run.

    `chat_model` is optional. When one is given, a description is written for
    every indexed document that has none.
    """
    if not dry_run and vectorstore is None:
        raise ValueError("A vector store is required to apply the changes")

    documents_dir = Path(documents_dir)
    if not dry_run:
        documents_dir.mkdir(parents=True, exist_ok=True)

    catalog = Catalog(db_path, create=not dry_run)
    report = SyncReport()

    # Recorded in the catalog with every document, so that a later run with a
    # different reader version or different chunking notices the change.
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
            # The file is back where it was, so the document comes back too.
            report.restored.append(source)
        elif (
            record.status in REPROCESS_STATUSES
            or compute_file_hash(files[source]) != record.file_hash
            or not _indexed_as_this_run_would(
                record, indexer=indexer, embedding_model=embedding_model
            )
        ):
            # Something changed. Either the last attempt did not finish, or the
            # file is not the one that was indexed, or it was indexed by a
            # reader or a model other than the current one.
            report.updated.append(source)
        else:
            report.skipped.append(source)

    for source in sorted(rows):
        # A row whose file is no longer in the folder.
        if source not in files and rows[source].status != "trashed":
            report.trashed.append(source)

    if dry_run:
        return report

    def index(source: str) -> None:
        """Index one file and record the outcome in the catalog."""
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
            # The message of an exception is empty now and then, and the name
            # of the exception says more than nothing.
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

        # A description is written from the file, so an edition that is no
        # longer there takes its description with it. It is cleared here and
        # written again on the next run.
        if before is not None and before.file_hash != result.file_hash:
            catalog.clear_description(source)

        # The structure is read from the file again, which costs a second pass
        # over it. It comes after the vectors are written, so a document that
        # cannot be read a second time is still indexed and answerable.
        try:
            build_one(catalog, source, documents_dir=documents_dir)
        except Exception as exc:  # noqa: BLE001 - the document is indexed either way
            report.structure_failed.append(
                (source, str(exc) or exc.__class__.__name__)
            )
        else:
            report.structured.append(source)

    for source in report.added:
        # The row goes in before the file is indexed, so a failure leaves a
        # document in the catalog saying what went wrong.
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
        # Descriptions are written once the whole folder has been indexed, so
        # that a document indexed by this run is described by it too.
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
    """The paths of the documents in the trash."""
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
    """Take documents out of the index and out of the catalog.

    With `remove_files` the files are deleted from the folder as well. Without
    it the file stays where it is and the next sync indexes it from scratch,
    because the catalog no longer knows about it.

    Returns what the run did. With `dry_run` it does nothing and only reports.
    """
    if not dry_run and vectorstore is None:
        raise ValueError("A vector store is required to apply the changes")

    documents_dir = Path(documents_dir)
    catalog = Catalog(db_path, create=not dry_run)
    report = DeleteReport()

    for source in sorted(set(paths)):
        if catalog.get(source) is None:
            # Nothing in the catalog to delete.
            report.missing.append(source)
            continue

        path = documents_dir / source
        present = path.is_file()

        if not dry_run:
            try:
                delete_document_vectors(source, vectorstore)
                catalog.delete(source)
            except Exception as exc:  # noqa: BLE001
                # One document failed and the run goes on with the others.
                report.failed.append((source, str(exc) or exc.__class__.__name__))
                continue

            if remove_files and present:
                path.unlink()

        report.deleted.append(source)

        if remove_files and present:
            report.files_removed.append(source)
        elif present and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            # The file is still there and the next sync will index it again,
            # which is worth saying in the report.
            report.file_present.append(source)

    return report
