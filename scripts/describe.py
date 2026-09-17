"""Write what each document is about, and keep it in the catalog.

A description is written by the chat model from a sample of the pages, one call
per document. With no argument only the documents that have no description are
described, which is the work the sync does by itself when it has a chat model.

`--all` describes every indexed document again. Documents named on the command
line are described whether they have a description or not.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from app.catalog import Catalog, DocumentRecord
from app.chat_model import build_chat_model
from app.config import describe_vector_store, settings, short_path
from app.descriptions import DescribeReport, undescribed, write_descriptions
from scripts.locks import single_run


def describe(
    documents: list[str] | None = None,
    *,
    every: bool = False,
    dry_run: bool = False,
    documents_dir: Path | None = None,
    db_path: Path | None = None,
    model: BaseChatModel | None = None,
) -> DescribeReport:
    """Describe documents and store what comes back.

    Returns what was written and what failed. With `dry_run` no model is called
    and nothing is written.
    """
    documents_dir = Path(documents_dir or settings.documents_dir)
    db_path = Path(db_path or settings.catalog_db_path)

    # Printed before anything happens, so that a run can be read back with the
    # settings it used.
    print(f"store   {describe_vector_store(settings)}")
    print(f"catalog {short_path(db_path)}\n")

    if not db_path.exists():
        raise SystemExit(f"No catalog at {short_path(db_path)}. Run a sync first.")

    catalog = Catalog(db_path)
    chosen = choose(catalog, documents or [], every=every)

    if not chosen:
        print("Every indexed document has a description already.")
        return DescribeReport()

    if dry_run:
        for record in chosen:
            print(f"would describe {record.path}")
        print("\nDry run: no model was called and nothing was written.")
        return DescribeReport()

    # The writing happens under the lock, so that a sync and this command do
    # not write the catalog at the same time.
    with single_run(db_path):
        report = write_descriptions(
            model or build_chat_model(),
            catalog,
            chosen,
            documents_dir=documents_dir,
            sample_chars=settings.description_sample_chars,
        )

    print_report(report)
    return report


def choose(
    catalog: Catalog, documents: list[str], *, every: bool
) -> list[DocumentRecord]:
    """The records this run is to describe.

    A document named on the command line is described again even when it has a
    description already, and `every` takes all of them. With neither, only the
    indexed documents that have no description.
    """
    if documents:
        return [lookup(catalog, path) for path in documents]

    indexed = [record for record in catalog.all() if record.status == "indexed"]
    if every:
        return indexed

    return undescribed(catalog)


def lookup(catalog: Catalog, path: str) -> DocumentRecord:
    """The record for one document, or a message saying why there is none."""
    record = catalog.get(path)

    if record is None:
        raise SystemExit(
            f"Not in the catalog: {path}. A document path is relative to the "
            "documents folder, and only a document that has been synced is in "
            "the catalog."
        )

    if record.status != "indexed":
        raise SystemExit(
            f"{path} is {record.status}, so there is nothing to describe. A sync "
            "can put it right."
        )

    return record


def print_report(report: DescribeReport) -> None:
    """Print what was written, with the description under each path."""
    for path, description in report.written:
        print(f"write   {path}")
        print(f"        {description}")
    for path, error in report.failed:
        print(f"failed  {path}: {error}")

    print(
        f"\n{len(report.written)} described, {len(report.failed)} failed"
    )


def parse_args() -> argparse.Namespace:
    """The command line, with --all checked against the named documents."""
    parser = argparse.ArgumentParser(
        description=(
            "Write what each document is about, one model call per document, and "
            "keep it in the catalog. Without arguments it describes the documents "
            "that have no description yet."
        )
    )
    parser.add_argument(
        "documents",
        nargs="*",
        metavar="PATH",
        help="documents to describe, relative to the documents folder; these are "
        "described again even when they already have a description",
    )
    parser.add_argument(
        "--all",
        dest="every",
        action="store_true",
        help="describe every indexed document, not only the ones without one",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be described, calling no model and writing nothing",
    )
    parser.add_argument(
        "--documents-dir",
        type=Path,
        default=settings.documents_dir,
        help="folder the documents sit in (default: %(default)s)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=settings.catalog_db_path,
        help="catalog database (default: %(default)s)",
    )

    args = parser.parse_args()

    if args.every and args.documents:
        parser.error("--all takes no document paths")

    return args


def main() -> None:
    """Run the describe command."""
    args = parse_args()
    describe(
        args.documents,
        every=args.every,
        dry_run=args.dry_run,
        documents_dir=args.documents_dir,
        db_path=args.db,
    )


if __name__ == "__main__":
    main()
