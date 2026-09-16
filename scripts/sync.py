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
    """Run the sync against the configured folders, then print what it did.

    The run writes each document's description as it indexes, which is one model
    call per document that needs one: `descriptions=False` indexes the folder and
    writes none, for a run that has no key to spend or does not want to spend it.
    """
    documents_dir = Path(documents_dir or settings.documents_dir)
    db_path = Path(db_path or settings.catalog_db_path)

    # Said before any work, so that a run against the wrong store is visible at
    # the top of the output rather than inferred from a sync that did nothing.
    print(f"store   {describe_vector_store(settings)}")
    print(f"catalog {short_path(db_path)}\n")

    vectorstore = None
    model = None
    if not dry_run:
        # Imported here so a dry run opens no client and needs no API keys.
        from app.vectorstore import get_vectorstore

        vectorstore = get_vectorstore()

        if descriptions:
            # Built before the run rather than at the first document that needs a
            # description: a key that is not there is worth knowing before the
            # folder is half indexed. Nothing is contacted here either way — the
            # model is a client until a description is asked for.
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
    """Say how many documents have no description, and which command writes them.

    The run writes one for every document it indexes that has none, so a library
    it has just been through is missing none — unless a call failed, or a
    document was indexed before the sync did this at all. Those are what is left
    here, and the command named is the way to put them right.
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

    # After the documents, because it is a pass of its own over the library, and
    # not a state any document is left in.
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
    args = parse_args()
    sync(
        dry_run=args.dry_run,
        documents_dir=args.documents_dir,
        db_path=args.db,
        descriptions=args.descriptions,
    )


if __name__ == "__main__":
    main()
