"""Bring the catalog and the index in step with the documents folder.

New files are indexed, changed files are indexed again, and a file that is gone
from the folder goes to the trash. Each indexed document also gets a description
written for it, unless `--no-descriptions` is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from app.chat_model import build_chat_model
from app.config import describe_vector_store, settings, short_path
from app.embeddings import model_name
from app.lifecycle import SyncReport, sync_documents
from scripts.locks import single_run


def sync(
    *,
    dry_run: bool = False,
    documents_dir: Path | None = None,
    db_path: Path | None = None,
    descriptions: bool = True,
    chat_model: BaseChatModel | None = None,
) -> SyncReport:
    """Reconcile the folder, the catalog and the vector store.

    Returns what the run found and did. With `dry_run` nothing is opened for
    writing and nothing changes, which is also why no model is called and no
    vector store is needed.
    """
    documents_dir = Path(documents_dir or settings.documents_dir)
    db_path = Path(db_path or settings.catalog_db_path)

    # Printed before anything happens, so that a run can be read back with the
    # settings it used.
    print(f"store   {describe_vector_store(settings)}")
    print(f"catalog {short_path(db_path)}\n")

    vectorstore = None
    model = None
    if not dry_run:
        # Imported here so that a dry run does not open a vector store at all.
        from app.vectorstore import get_vectorstore

        vectorstore = get_vectorstore()

        if descriptions:
            # A description costs one call to the model per document, so the
            # model is built once and only when descriptions were asked for.
            model = chat_model or build_chat_model()

    with single_run(db_path, enabled=not dry_run):
        report = sync_documents(
            documents_dir,
            db_path,
            vectorstore,
            embedding_model=model_name(settings),
            dry_run=dry_run,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            chat_model=model,
            description_sample_chars=settings.description_sample_chars,
        )

    print_report(report, dry_run=dry_run)
    if vectorstore is not None:
        warn_if_the_store_is_empty(report, vectorstore, db_path)
    if not dry_run and descriptions:
        note_undescribed(db_path)
    return report


def note_undescribed(db_path: Path) -> None:
    """Say how many indexed documents have no description yet.

    This is what a run that indexed nothing new still has to report, when the
    documents that were already there are the ones missing a description.
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
    """Warn about a catalog that lists documents the store holds no vectors for.

    This happens when the catalog was built against another store. The run has
    nothing to do, because the files have not changed, and a search will find
    nothing until the store is pointed back or the catalog is built again.
    """
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
        return

    print(
        f"\nNote: the catalog lists {len(indexed)} indexed document(s), but "
        f"{settings.vector_store} holds no vectors. This catalog was built "
        "against a different store, so the run had nothing to do and nothing is "
        "searchable. Either point VECTOR_STORE back to where the documents were "
        "indexed, or start over: delete the catalog and run the sync again."
    )


def print_report(report: SyncReport, *, dry_run: bool = False) -> None:
    """Print what the run found, one line per document."""
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

    # Descriptions are written after the whole folder has been indexed, so they
    # come after the documents in the report.
    for source in report.described:
        print(f"describe {source}")
    for source, error in report.description_failed:
        print(f"describe {source} failed: {error}")

    print(
        f"\n{len(report.added)} added, {len(report.updated)} updated, "
        f"{len(report.restored)} restored, {len(report.trashed)} trashed, "
        f"{len(report.failed)} failed, {len(report.skipped)} unchanged, "
        f"{len(report.described)} described"
    )

    if dry_run:
        print("Dry run: nothing was changed.")


def parse_args() -> argparse.Namespace:
    """The command line, with the folder and the catalog."""
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
        "--no-descriptions",
        dest="descriptions",
        action="store_false",
        help="index without writing each document's description",
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
    """Run the sync."""
    args = parse_args()
    sync(
        dry_run=args.dry_run,
        documents_dir=args.documents_dir,
        db_path=args.db,
        descriptions=args.descriptions,
    )


if __name__ == "__main__":
    main()
