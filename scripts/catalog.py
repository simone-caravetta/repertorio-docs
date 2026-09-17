"""Show what the catalog holds, and file a document under a category.

`show` prints the tree of categories with the number of documents in each, and
with `--documents` it lists them under every category. `move` changes the
category of one document, which is a write to the catalog alone.
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

# The label used for the documents that are filed in no category.
NO_CATEGORY = "no category"


def show(*, documents: bool = False, db_path: Path | None = None) -> None:
    """Print the catalog as a tree of categories.

    With `documents` every category is followed by the documents under it. A
    category with nothing filed directly in it still appears, because documents
    filed below it count towards it.
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
    """Every branch of the tree, each with how deep it sits."""
    for branch in tree:
        yield branch, depth
        yield from walk(branch.children, depth + 1)


def label_of(branch: CategoryBranch, depth: int) -> str:
    """The name of a branch, indented by its depth.

    The tree is printed as a flat list of lines, so the depth is written into
    the name.
    """
    return "  " * depth + branch.name


def print_documents(catalog: Catalog, sources: list[str], depth: int) -> None:
    """Print one document per line, with its description under it.

    The description goes on the next line, indented further, so that it does not
    read as part of the line above.
    """
    for source in sources:
        print(f"{'  ' * depth}- {source}")

        record = catalog.get(source)
        description = (record.description or "").strip() if record else ""
        if description:
            print(f"{'  ' * (depth + 1)}  {description}")


def documents_count(count: int) -> str:
    """The count with the word after it, in the singular for one."""
    return f"{count} document" + ("" if count == 1 else "s")


def move(path: str, category: str | None, *, db_path: Path | None = None) -> None:
    """File a document under a category, or under none.

    Only the catalog is written. The file stays where it is and nothing is
    indexed again.
    """
    db_path = Path(db_path or settings.catalog_db_path)

    if not db_path.exists():
        raise SystemExit(f"No catalog at {short_path(db_path)}. Run a sync first.")

    where = normalise_category(category)
    catalog = Catalog(db_path)

    # A move is a write like any other, so it waits for the lock.
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
    """The command line, checked against the command that was asked for."""
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
    """Run the command that was asked for."""
    args = parse_args()

    if args.command == "move":
        move(args.path, args.to, db_path=args.db)
        return

    # Nothing else was asked for, so the catalog is printed.
    show(documents=args.documents, db_path=args.db)


if __name__ == "__main__":
    main()
