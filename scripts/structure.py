"""Read what each document is made of, and keep it in the catalog.

The structure of a document is its sections, and the tables and figures inside
them. It is read from the PDF and needs no model, so this command can be
pointed at a live library: it opens no vector store and changes nothing about
how a question is answered.

With no argument it reads every indexed document in the catalog. A document
named on the command line is read again even when it has a structure already.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.catalog import SECTION, Catalog, DocumentRecord, Node
from app.config import settings, short_path
from app.structure import StructureReport, build_documents, counts
from scripts.locks import single_run


def build(
    documents: list[str] | None = None,
    *,
    dry_run: bool = False,
    documents_dir: Path | None = None,
    db_path: Path | None = None,
) -> StructureReport:
    """Read the documents and write the structure each one holds.

    Returns what was built and what failed. With `dry_run` the documents are
    read and nothing is written.
    """

    documents_dir = Path(documents_dir or settings.documents_dir)
    db_path = Path(db_path or settings.catalog_db_path)

    # No store line: this command opens no vector store, and printing one
    # would claim something that does not happen.
    print(f"catalog {short_path(db_path)}\n")

    if not db_path.exists():
        raise SystemExit(f"No catalog at {short_path(db_path)}. Run a sync first.")

    catalog = Catalog(db_path)
    chosen = choose(catalog, documents or [])

    if not chosen:
        print("There is no indexed document to read.")
        return StructureReport()

    if dry_run:
        report = build_documents(
            catalog, chosen, documents_dir=documents_dir, dry_run=True
        )
        print_report(report)
        print("\nDry run: nothing was read into the catalog.")

        return report

    # The writing happens under the lock, so that a sync and this command do
    # not write the catalog at the same time.
    with single_run(db_path):
        report = build_documents(catalog, chosen, documents_dir=documents_dir)

    print_report(report)
    return report


def choose(catalog: Catalog, documents: list[str]) -> list[DocumentRecord]:
    """The records this run is to read the structure of."""

    if documents:
        return [lookup(catalog, path) for path in documents]

    return [record for record in catalog.all() if record.status == "indexed"]


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
            f"{path} is {record.status}, so there is nothing to read. A sync "
            "can put it right."
        )

    return record


def print_report(report: StructureReport) -> None:
    """Print the tree of every document that was read, then what failed."""
    for path, nodes in report.built:
        print_tree(path, nodes)
        print()

    for path, error in report.failed:
        print(f"failed  {path}: {error}")

    print(f"{len(report.built)} built, {len(report.failed)} failed")


def print_tree(path: str, nodes: list[Node]) -> None:
    """Print one document's tree: a line per node, indented by its depth."""
    sections, tables, figures = counts(nodes)
    print(f"{path}  {counts_label(sections, tables, figures)}")

    names = [_name(nodes, node) for node in nodes]
    width = max((len(name) for name in names), default=0)

    for node, name in zip(nodes, names, strict=True):
        print(f"  {name.ljust(width)}  {_right(node)}")


def _name(nodes: list[Node], node: Node) -> str:
    """What a node is called, indented by how deep it sits."""
    depth = 0
    parent = node.parent
    while parent is not None:
        depth += 1
        parent = nodes[parent].parent

    if node.kind == SECTION:
        return f"{'  ' * depth}{node.title or ''}"

    return f"{'  ' * depth}{node.kind}  {node.title or ''}".rstrip()


def _right(node: Node) -> str:
    """The pages a node covers, and how big a figure is."""
    label = pages_label(node.page, node.end_page)

    if node.bbox is None:
        return label

    x0, y0, x1, y1 = node.bbox

    return f"{label}  {round(x1 - x0)} x {round(y1 - y0)} pt"


def pages_label(first: int, last: int) -> str:
    """The pages a node covers, the way a person writes them."""
    return f"p.{first}" if first == last else f"pp.{first}-{last}"


def counts_label(sections: int, tables: int, figures: int) -> str:
    """How much a document holds, in words."""

    def many(count: int, one: str) -> str:
        return f"{count} {one}" if count == 1 else f"{count} {one}s"

    return ", ".join(
        [many(sections, "section"), many(tables, "table"), many(figures, "figure")]
    )


def parse_args() -> argparse.Namespace:
    """The command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Read the sections of each document, and the tables and figures "
            "inside them, and keep them in the catalog. No model is called and "
            "no vector store is opened. Without arguments every indexed "
            "document is read."
        )
    )
    parser.add_argument(
        "documents",
        nargs="*",
        metavar="PATH",
        help="documents to read, relative to the documents folder; these are "
        "read again even when they have a structure already",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the structure that would be written, and write nothing",
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

    return parser.parse_args()


def main() -> None:
    """Run the structure command."""
    args = parse_args()
    build(
        args.documents,
        dry_run=args.dry_run,
        documents_dir=args.documents_dir,
        db_path=args.db,
    )


if __name__ == "__main__":
    main()
