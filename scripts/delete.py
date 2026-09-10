from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from app.config import settings
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
    """Delete documents from the index and the catalog, then print what it did.

    With `trashed` the paths are ignored and the whole trash is emptied instead.
    """
    documents_dir = Path(documents_dir or settings.documents_dir)
    db_path = Path(db_path or settings.catalog_db_path)

    vectorstore = None
    if not dry_run:
        # Imported here so a dry run opens no client and needs no API keys.
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
        # Emptying the index for a file that is still there is a legitimate way
        # to start the document over, but it does not hold: say so.
        print(f"\nStill in the documents folder: {', '.join(report.file_present)}")
        print(
            "The next sync sees a file the catalog does not know, and indexes it "
            "again as a new document."
        )
        print("Pass --with-file to remove the file as well.")

    if dry_run:
        print("Dry run: nothing was changed.")


def parse_args() -> argparse.Namespace:
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
