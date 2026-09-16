from __future__ import annotations

import argparse
from pathlib import Path

from app.config import describe_vector_store, settings, short_path
from app.embeddings import model_name
from app.lifecycle import SyncReport, sync_documents
from scripts.locks import single_run


def sync(
    *,
    dry_run: bool = False,
    documents_dir: Path | None = None,
    db_path: Path | None = None,
) -> SyncReport:
    """Run the sync against the configured folders, then print what it did."""
    documents_dir = Path(documents_dir or settings.documents_dir)
    db_path = Path(db_path or settings.catalog_db_path)

    # Said before any work, so that a run against the wrong store is visible at
    # the top of the output rather than inferred from a sync that did nothing.
    print(f"store   {describe_vector_store(settings)}")
    print(f"catalog {short_path(db_path)}\n")

    vectorstore = None
    if not dry_run:
        # Imported here so a dry run opens no client and needs no API keys.
        from app.vectorstore import get_vectorstore

        vectorstore = get_vectorstore()

    with single_run(db_path, enabled=not dry_run):
        report = sync_documents(
            documents_dir,
            db_path,
            vectorstore,
            embedding_model=model_name(settings),
            dry_run=dry_run,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    print_report(report, dry_run=dry_run)
    if vectorstore is not None:
        warn_if_the_store_is_empty(report, vectorstore, db_path)
    if not dry_run:
        note_undescribed(db_path)
    return report


def note_undescribed(db_path: Path) -> None:
    """Say how many documents have no description yet, and which command writes them.

    A description is what a question about the library is answered from, and the
    sync does not write one: it needs no key and makes no model call. A library
    just indexed therefore has none at all, which is exactly when the command
    that writes them is worth knowing about.
    """
    from app.catalog import Catalog
    from app.descriptions import undescribed

    missing = undescribed(Catalog(db_path, create=False))
    if not missing:
        return

    print(
        f"\nNote: {len(missing)} indexed document(s) have no description yet. "
        "`python -m scripts.describe` writes them, one model call each."
    )


def warn_if_the_store_is_empty(
    report: SyncReport, vectorstore: object, db_path: Path
) -> None:
    """Say so when the catalog describes documents the store does not hold.

    The catalog does not record which store it was built against, so pointing
    VECTOR_STORE somewhere new — or at a folder that was deleted, or an index
    that was recreated — reconciles to nothing at all: every file still matches
    its hash, so the run reports no work, and the store it now points at is
    empty. Unsaid, that reads as a working library until someone asks a question.

    Only ever a warning: it cannot change what the run did.
    """
    # The run wrote something, so an empty store is a different problem and the
    # failures are already on screen.
    if report.added or report.updated or report.restored:
        return

    from app.catalog import Catalog
    from app.vectorstore import vector_count

    indexed = [
        row
        for row in Catalog(db_path, create=False).all()
        if row.status == "indexed"
    ]
    if not indexed:
        return

    found = vector_count(vectorstore)
    if found is None or found > 0:
        return  # it holds vectors, or it cannot say: either way, stay quiet

    print(
        f"\nNote: the catalog lists {len(indexed)} indexed document(s), but "
        f"{settings.vector_store} holds no vectors. This catalog was built "
        "against a different store, so the run had nothing to do and nothing is "
        "searchable. Either point VECTOR_STORE back to where the documents were "
        "indexed, or start over: delete the catalog and run the sync again."
    )


def print_report(report: SyncReport, *, dry_run: bool = False) -> None:
    for source in report.added:
        print(f"add     {source}")
    for source in report.updated:
        print(f"update  {source}")
    for source in report.restored:
        print(f"restore {source}")
    for source in report.trashed:
        print(f"trash   {source}")
    for source, error in report.failed:
        print(f"failed  {source}: {error}")
    for source in report.skipped:
        print(f"skip    {source}")

    print(
        f"\n{len(report.added)} added, {len(report.updated)} updated, "
        f"{len(report.restored)} restored, {len(report.trashed)} trashed, "
        f"{len(report.failed)} failed, {len(report.skipped)} unchanged"
    )

    if dry_run:
        print("Dry run: nothing was changed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconcile the documents folder, the catalog and the vector store: "
            "new files are indexed, changed files re-indexed, missing files "
            "trashed."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would happen, without changing anything",
    )
    parser.add_argument(
        "--documents-dir",
        type=Path,
        default=settings.documents_dir,
        help="folder to watch (default: %(default)s)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=settings.catalog_db_path,
        help="catalog database (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sync(
        dry_run=args.dry_run,
        documents_dir=args.documents_dir,
        db_path=args.db,
    )


if __name__ == "__main__":
    main()
