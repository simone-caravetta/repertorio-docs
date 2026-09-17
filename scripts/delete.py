"""Remove documents from the index and from the catalog.

The file is left in the documents folder unless `--with-file` is given. A
document whose file is gone from the folder is in the trash, and `--trashed`
empties the trash in one go.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from app.config import describe_vector_store, settings, short_path
from app.lifecycle import DeleteReport, delete_documents, trashed_paths
from scripts.locks import single_run


def delete(
    paths: Sequence[str],
    *,
    trashed: bool = False,
    remove_files: bool = False,
    dry_run: bool = False,
    documents_dir: Path | None = None,
    db_path: Path | None = None,
) -> DeleteReport:
    """Take the named documents out, or everything that is in the trash.

    Returns what the run did. With `dry_run` nothing is opened for writing and
    nothing changes, and the report says what a real run would have done.
    """
    documents_dir = Path(documents_dir or settings.documents_dir)
    db_path = Path(db_path or settings.catalog_db_path)

    # Printed before anything happens, so that a run is readable afterwards.
    print(f"store   {describe_vector_store(settings)}")
    print(f"catalog {short_path(db_path)}\n")

    vectorstore = None
    if not dry_run:
        # Imported here so that a dry run does not open a vector store at all.
        from app.vectorstore import get_vectorstore

        vectorstore = get_vectorstore()

    with single_run(db_path, enabled=not dry_run):
        targets = trashed_paths(db_path) if trashed else list(paths)
        report = delete_documents(
            documents_dir,
            db_path,
            vectorstore,
            targets,
            remove_files=remove_files,
            dry_run=dry_run,
        )

    print_report(report, dry_run=dry_run)
    return report


def print_report(report: DeleteReport, *, dry_run: bool = False) -> None:
    """Print what the run did, one line per document."""
    for source in report.deleted:
        suffix = " (and its file)" if source in report.files_removed else ""
        print(f"delete  {source}{suffix}")
    for source in report.missing:
        print(f"missing {source}: not in the catalog")
    for source, error in report.failed:
        print(f"failed  {source}: {error}")

    print(
        f"\n{len(report.deleted)} deleted, {len(report.missing)} not in the catalog, "
        f"{len(report.failed)} failed"
    )

    if report.file_present:
        # The file is still in the folder, so the next sync finds one the
        # catalog does not know and indexes it as a new document.
        print(f"\nStill in the documents folder: {', '.join(report.file_present)}")
        print(
            "The next sync sees a file the catalog does not know, and indexes it "
            "again as a new document."
        )
        print("Pass --with-file to remove the file as well.")

    if dry_run:
        print("Dry run: nothing was changed.")


def parse_args() -> argparse.Namespace:
    """The command line, with the two ways of naming documents checked."""
    parser = argparse.ArgumentParser(
        description=(
            "Remove documents from the index and the catalog. The file is left "
            "in the documents folder unless --with-file is given."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        metavar="PATH",
        help="document paths, relative to the documents folder",
    )
    parser.add_argument(
        "--trashed",
        action="store_true",
        help="remove every trashed document: their files are already gone",
    )
    parser.add_argument(
        "--with-file",
        action="store_true",
        help="remove the file from the documents folder as well",
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

    args = parser.parse_args()

    if args.trashed and args.paths:
        parser.error("give document paths or --trashed, not both")
    if not args.trashed and not args.paths:
        parser.error("give one or more document paths, or --trashed")

    return args


def main() -> None:
    """Run the delete command."""
    args = parse_args()
    delete(
        args.paths,
        trashed=args.trashed,
        remove_files=args.with_file,
        dry_run=args.dry_run,
        documents_dir=args.documents_dir,
        db_path=args.db,
    )


if __name__ == "__main__":
    main()
