"""The structure of a document: its sections, and what sits inside them.

`app/pdf.py` cuts a document into pieces and gives each piece the heading it
falls under. Read here, those pieces are a tree: one node per heading, carrying
the section that holds it and the pages and offsets it covers, with the tables
and the figures of the document placed in the section they sit in.

Nothing here calls a model, so nothing here can invent a section a document
does not have.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from app import pdf
from app.catalog import FIGURE, SECTION, TABLE, Catalog, DocumentRecord, Node

# Which of two nodes that start in the same place comes first. A section is
# read before what it holds, so a node never points at one numbered after it.

_ORDER = {SECTION: 0, TABLE: 1, FIGURE: 2}

# How much of a table's first row is kept as its name, which is there so that
# a tree can be read and one table told from another.

_LABEL_CHARS = 60

# What a nested heading is indented by, under the one that holds it.

INDENT = "  "


def of(reading: pdf.Reading) -> list[Node]:
    """The structure of a document that has been read.

    A section runs from the piece its heading was cut at to the last piece
    before the next heading of the same level or a shallower one, so a
    subsection is inside the text of its parent. A document with no headings
    comes back with no sections: its tables and figures are held by the
    document, which is what a `parent` of None says.
    """

    nodes: list[Node] = []
    stack: list[int] = []
    opens: list[tuple[int, int, int]] = []

    for piece in reading.pieces:
        page = reading.pages[piece.page]
        heading = _opener(page, reading.headings[piece.page], piece)

        if heading is not None:
            while stack and _level(nodes, stack[-1]) >= heading.level:
                stack.pop()

            nodes.append(
                Node(
                    ordinal=len(nodes),
                    kind=SECTION,
                    title=heading.title,
                    level=heading.level,
                    parent=stack[-1] if stack else None,
                    page=piece.page + 1,
                    end_page=piece.page + 1,
                    start=piece.start,
                    end=piece.end,
                )
            )

            stack.append(len(nodes) - 1)
            opens.append((piece.page, piece.start, len(nodes) - 1))

        elif piece.kind == pdf.TABLE:
            nodes.append(
                Node(
                    ordinal=len(nodes),
                    kind=TABLE,
                    title=_label(piece.text),
                    level=None,
                    parent=stack[-1] if stack else None,
                    page=piece.page + 1,
                    end_page=piece.page + 1,
                    start=piece.start,
                    end=piece.end,
                )
            )

        _extend(nodes, stack, piece)

    for figure in reading.figures:
        page = reading.pages[figure.page]
        start = _anchor(page, figure.bbox[1])

        nodes.append(
            Node(
                ordinal=len(nodes),
                kind=FIGURE,
                title=None,
                level=None,
                parent=_section_at(opens, figure.page, start),
                page=figure.page + 1,
                end_page=figure.page + 1,
                start=start,
                end=start,
                bbox=figure.bbox,
            )
        )

    return _in_order(nodes)


def _label(text: str) -> str:
    """The first row of a table, which is what names it.

    A row arrives as one line per cell and the rows are separated by a blank
    line, so what stands before the first blank line is the row of headings
    the table opens with.
    """

    return " ".join(text.split("\n\n", 1)[0].split())[:_LABEL_CHARS]


def counts(nodes: Iterable[Node]) -> tuple[int, int, int]:
    """How many sections, tables and figures a tree holds."""
    kinds = [node.kind for node in nodes]

    return (
        kinds.count(SECTION),
        kinds.count(TABLE),
        kinds.count(FIGURE),
    )


def pages_label(first: int, last: int) -> str:
    """The pages a node covers, the way a person writes them."""
    return f"p.{first}" if first == last else f"pp.{first}-{last}"


def outline(nodes: Iterable[Node]) -> str:
    """The sections of a tree, one line each, nested under what holds them.

    A line carries the heading and the pages it covers, which is what makes the
    tree readable by someone who has not seen the document: a model answering a
    question about the shape of a document rather than about what it says.

    The nesting is read from `parent` and not from `level`. A level is where
    the reader found the heading, by the size of the text or by the outline of
    the file, and the two disagree on a document whose first heading is set
    larger than the ones after it. `parent` is the tree's own statement of what
    holds what.

    A table has a line and is marked as one, because what names it is its first
    row and that row reads like a heading. A figure is not named and so has no
    line, which is the whole of the rule: a node with no title is not a line.

    A document with no headings answers with an empty string, and so does an
    empty tree.
    """

    kept = [node for node in nodes if node.title]
    if not kept:
        return ""

    by_ordinal = {node.ordinal: node for node in kept}

    return "\n".join(
        f"{INDENT * _depth(node, by_ordinal)}{_line(node)}" for node in kept
    )


def _line(node: Node) -> str:
    """One node as a line: what it is called, and the pages it covers."""
    named = f"table {node.title}" if node.kind == TABLE else (node.title or "")

    return f"{named} — {pages_label(node.page, node.end_page)}"


def _depth(node: Node, by_ordinal: Mapping[int, Node]) -> int:
    """How deep a node sits, as the number of sections that hold it.

    A parent that is not among the nodes, or a chain that leads back to itself,
    ends the walk rather than raising or hanging: a tree read back from the
    catalog is data, and a renderer that stops on it is worse than one that
    indents it wrongly.
    """

    depth = 0
    seen = {node.ordinal}
    parent = node.parent

    while parent is not None and parent in by_ordinal and parent not in seen:
        seen.add(parent)
        depth += 1
        parent = by_ordinal[parent].parent

    return depth


@dataclass
class StructureReport:
    """What a run built, and what it could not.

    A built document carries the tree that was read, so a run can be printed
    from what it did rather than from what was stored.
    """

    built: list[tuple[str, list[Node]]] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)


def build_one(catalog: Catalog, path: str, *, documents_dir: Path) -> list[Node]:
    """Read one document and write the structure it holds.

    The reading is of the file, not of the index, so this is the only step a
    document needs before its structure is in the catalog.
    """

    nodes = of(pdf.read(Path(documents_dir) / path))
    catalog.set_structure(path, nodes)

    return nodes


def build_documents(
    catalog: Catalog,
    documents: Iterable[DocumentRecord],
    *,
    documents_dir: Path,
    dry_run: bool = False,
) -> StructureReport:
    """Read each document and write the structure it holds.

    Nothing here calls a model or opens a vector store, so this is a pass over
    a library that cannot change what a question is answered from. A document
    that cannot be read or written is reported with the reason, and the run
    goes on to the next one.
    """

    report = StructureReport()

    for record in documents:
        try:
            if dry_run:
                nodes = of(pdf.read(Path(documents_dir) / record.path))
            else:
                nodes = build_one(catalog, record.path, documents_dir=documents_dir)
        except Exception as exc:  # noqa: BLE001 - one document, not the run
            report.failed.append((record.path, str(exc) or exc.__class__.__name__))
            continue

        report.built.append((record.path, nodes))

    return report


def _opener(
    page: pdf.Page, headings: list[pdf.Heading], piece: pdf.Piece
) -> pdf.Heading | None:
    """The heading a piece opens, when it opens one.

    A piece is cut at a heading and then trimmed, so the piece that opens a
    section can start a character or two after the heading does. Anything but
    whitespace between the two means the piece is not that heading's line: a
    heading inside a table is never cut at, and its text is not blank.
    """

    for heading in headings:
        if heading.start > piece.start:
            break

        if not page.text[heading.start : piece.start].strip():
            return heading

    return None


def _level(nodes: list[Node], ordinal: int) -> int:
    """How deep a node's heading sits, for the sections on the stack."""

    return nodes[ordinal].level or 0


def _extend(nodes: list[Node], stack: list[int], piece: pdf.Piece) -> None:
    """Grow every open section to hold a piece.

    A section's extent is the text it holds, its own and that of everything
    inside it, so the whole stack grows rather than the innermost section
    alone.
    """

    for ordinal in stack:
        nodes[ordinal] = replace(
            nodes[ordinal], end_page=piece.page + 1, end=piece.end
        )


def _anchor(page: pdf.Page, top: float) -> int:
    """Where in the text of a page a figure stands.

    An image has no offset of its own, so it is anchored at the end of the last
    run of text above it, which is the text the figure follows. A figure with
    nothing above it is anchored at the head of the page, which puts it in the
    section the page opens with.
    """

    offset = 0
    for span in page.spans:
        if span.bbox[3] > top:
            break

        offset = span.end

    return offset


def _section_at(opens: list[tuple[int, int, int]], page: int, offset: int) -> int | None:
    """The ordinal of the last section that opened at or before a place."""

    found = None
    for open_page, start, ordinal in opens:
        if (open_page, start) > (page, offset):
            break

        found = ordinal

    return found


def _in_order(nodes: list[Node]) -> list[Node]:
    """The nodes in document order, numbered from zero.

    Every node carries the place it starts at, so one sort puts a section, a
    table and a figure in the order a reader meets them, and the parents are
    renumbered along with them.
    """

    order = sorted(nodes, key=lambda node: (node.page, node.start or 0, _ORDER[node.kind]))
    numbers = {node.ordinal: number for number, node in enumerate(order)}

    return [
        replace(
            node,
            ordinal=number,
            parent=None if node.parent is None else numbers[node.parent],
        )
        for number, node in enumerate(order)
    ]
