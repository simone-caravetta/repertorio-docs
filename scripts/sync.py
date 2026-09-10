from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.config import settings
from app.lifecycle import SyncReport, sync_documents

try:
    import fcntl
except ImportError:  # Not POSIX: no lock is available, two runs would overlap.
    fcntl = None  # type: ignore[assignment]


@contextmanager
def _single_run(db_path: Path, *, enabled: bool = True) -> Iterator[None]:
    """Refuse to start while another sync is already running.

    A dry run reads only, so it neither takes the lock nor creates the lock file.
    """
    if fcntl is None or not enabled:
        yield
        return

    lock_path = db_path.with_name(f"{db_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit(f"Another sync is running: {lock_path}") from None
        yield


def sync(
    *,
    dry_run: bool = False,
    documents_dir: Path | None = None,
    db_path: Path | None = None,
) -> SyncReport:
    """Run the sync against the configured folders, then print what it did."""
    documents_dir = Path(documents_dir or settings.documents_dir)
    db_path = Path(db_path or settings.catalog_db_path)

    vectorstore = None
    if not dry_run:
        # Imported here so a dry run opens no client and needs no API keys.
        from app.vectorstore import get_vectorstore

        vectorstore = get_vectorstore()

    with _single_run(db_path, enabled=not dry_run):
        report = sync_documents(
            documents_dir,
            db_path,
            vectorstore,
            dry_run=dry_run,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    print_report(report, dry_run=dry_run)
    return report


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
