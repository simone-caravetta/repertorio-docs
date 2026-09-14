"""The categories of the library: what is in them, and moving a document.

Where a document is filed is a catalog fact. The folder the file sits in seeded
it once, when the file was first indexed, and nothing reads the folder for it
again — so the move here is the whole of the answer to "put this document
somewhere else", and it costs a row, not a re-index.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

from app.catalog import (
    Catalog,
    CategoryBranch,
    build_category_tree,
    normalise_category,
)
from app.config import settings, short_path
from scripts.locks import single_run

NO_CATEGORY = "no category"


def show(*, documents: bool = False, db_path: Path | None = None) -> None:
    """Print the category tree, with the documents under each when asked.

    Read-only: looking at the library must never bring a database into existence,
    which is what `create=False` is for.
    """
    db_path = Path(db_path or settings.catalog_db_path)
    print(f"catalog {short_path(db_path)}\n")

    if not db_path.exists():
        print("No catalog here yet. Run a sync first.")
        return

    catalog = Catalog(db_path, create=False)
    counts = catalog.category_counts()
    tree = build_category_tree(counts)
    loose = dict(counts).get(None, 0)

    if not tree and not loose:
        print("No indexed documents yet. Run a sync first.")
        return

    rows = [(label_of(branch, depth), branch, depth) for branch, depth in walk(tree)]
    width = max([len(name) for name, _, _ in rows] + [len(NO_CATEGORY)])

    for name, branch, depth in rows:
        print(f"{name.ljust(width)}  {documents_count(branch.total)}")
        if documents:
            print_documents(catalog, catalog.sources_in_category(branch.name), depth + 1)

    if loose:
        print(f"{NO_CATEGORY.ljust(width)}  {documents_count(loose)}")
        if documents:
            print_documents(catalog, catalog.sources_in_category(None), 1)


def walk(
    tree: tuple[CategoryBranch, ...], depth: int = 0
) -> Iterator[tuple[CategoryBranch, int]]:
    """Every branch, with how deep it sits, parents before children."""
    for branch in tree:
        yield branch, depth
        yield from walk(branch.children, depth + 1)


def label_of(branch: CategoryBranch, depth: int) -> str:
    """The category as it is written here: indented, but under its full name.

    The full name and not the leaf, because the full name is what `--category`
    takes and what a person would type next.
    """
    return "  " * depth + branch.name


def print_documents(catalog: Catalog, sources: list[str], depth: int) -> None:
    """The documents of a branch, one level in from it.

    Dashed, because a line without a dash is a category and the two must not have
    to be told apart by reading them. What the catalog says about a document sits
    under it, indented one more: it belongs to the line above, and it is the
    reason to ask for the documents at all rather than only their counts.
    """
    for source in sources:
        print(f"{'  ' * depth}- {source}")

        record = catalog.get(source)
        description = (record.description or "").strip() if record else ""
        if description:
            print(f"{'  ' * (depth + 1)}  {description}")


def documents_count(count: int) -> str:
    """`1 document` or `3 documents`: these counts are read by a person."""
    return f"{count} document" + ("" if count == 1 else "s")


def move(path: str, category: str | None, *, db_path: Path | None = None) -> None:
    """File a document under another category.

    The file does not have to move and the vectors are not touched, so a
    reorganisation of the library costs one row per document.
    """
    db_path = Path(db_path or settings.catalog_db_path)

    if not db_path.exists():
        raise SystemExit(f"No catalog at {short_path(db_path)}. Run a sync first.")

    where = normalise_category(category)
    catalog = Catalog(db_path)

    # The same lock as sync and delete: they touch the same rows.
    with single_run(db_path):
        try:
            catalog.set_category(path, where)
        except KeyError:
            raise SystemExit(
                f"Not in the catalog: {path}. A document path is relative to "
                "the documents folder."
            ) from None

    print(f"{path} -> {where or NO_CATEGORY}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show what is filed where, or move a document between categories. "
            "Moving is a catalog write: nothing is indexed again."
        )
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("show", "move"),
        help="what to do (default: show)",
    )
    parser.add_argument(
        "path",
        nargs="?",
        metavar="PATH",
        help="with move: the document, relative to the documents folder",
    )
    parser.add_argument(
        "--to",
        metavar="CATEGORY",
        help="with move: the category to file it under; '' for no category",
    )
    parser.add_argument(
        "--documents",
        action="store_true",
        help="with show: list the documents under each category as well",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=settings.catalog_db_path,
        help="catalog database (default: %(default)s)",
    )

    args = parser.parse_args()

    if args.command == "move":
        if not args.path:
            parser.error("move needs a document path")
        if args.to is None:
            parser.error("move needs --to CATEGORY (use --to '' for no category)")
    elif args.path is not None or args.to is not None:
        parser.error("only move takes a document path and --to")

    return args


def main() -> None:
    args = parse_args()

    if args.command == "move":
        move(args.path, args.to, db_path=args.db)
        return

    # No command at all means `show`: looking is what this is mostly for.
    show(documents=args.documents, db_path=args.db)


if __name__ == "__main__":
    main()
