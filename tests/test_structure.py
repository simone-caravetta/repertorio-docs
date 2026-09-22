"""The structure of a document that has been read.

A reading carries the pieces, the headings those pieces were cut at and the
figures the document holds. These tests check the tree built from them: which
section holds what, the pages and the offsets a section covers, and the
section a table or a figure lands in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app import pdf, structure
from tests.helpers import BODY, Image, Line, Table, make_structured_pdf


@pytest.fixture
def book(tmp_path: Path) -> Path:
    """A document with an outline, two levels, a table and a page break.

    Care runs from the first page onto the second, where a second section at
    the same level as Care begins.
    """

    return make_structured_pdf(
        tmp_path / "book.pdf",
        pages=[
            [
                Line("Handbook", size=22),
                Line("1  Care", size=16),
                Line(BODY),
                Table([["Part", "Hours"], ["Filter", "200"]]),
                Line(BODY),
            ],
            [Line(BODY), Line("2  Parts", size=16), Line(BODY)],
        ],
        toc=[[1, "Handbook", 1], [2, "Care", 1], [2, "Parts", 2]],
    )


@pytest.fixture
def figures(tmp_path: Path) -> Path:
    """A page with two headings, a picture under each, and a scanned page.

    The sizes are what makes the two headings: with the body text at 11
    points, a line at 20 is the top level of the page and a line at 16 the
    level below it.
    """

    return make_structured_pdf(
        tmp_path / "figures.pdf",
        pages=[
            [
                Line("Primo", size=20, y=60),
                Line(BODY, y=100),
                Image(120, 80, y=140),
                Line("Secondo", size=16, y=320),
                Line(BODY, y=360),
                Image(100, 60, y=420),
                Image(full_page=True),
            ]
        ],
    )


@pytest.fixture
def flat(tmp_path: Path) -> Path:
    """A document with no headings at all, but a table and a picture."""

    return make_structured_pdf(
        tmp_path / "flat.pdf",
        pages=[[Line(BODY), Line(BODY), Table([["A", "B"], ["1", "2"]]), Image(100, 60)]],
    )


@pytest.fixture
def repeated(tmp_path: Path) -> Path:
    """A document whose outline names the same section on both pages."""

    return make_structured_pdf(
        tmp_path / "repeated.pdf",
        pages=[
            [Line("Notes", size=20), Line(BODY)],
            [Line("Notes", size=20), Line(BODY)],
        ],
        toc=[[1, "Notes", 1], [1, "Notes", 2]],
    )


@pytest.fixture
def sized(tmp_path: Path) -> Path:
    """A document with no outline, whose two headings come from type size."""

    return make_structured_pdf(
        tmp_path / "sized.pdf",
        pages=[
            [Line("Chapter", size=18), Line(BODY), Line("Section", size=14), Line(BODY)]
        ],
    )


def tree(path: Path) -> list[structure.Node]:
    return structure.of(pdf.read(path))


def titled(nodes: list[structure.Node], title: str) -> structure.Node:
    found = [node for node in nodes if node.title == title]
    assert len(found) == 1, f"expected one {title!r}, found {len(found)}"
    return found[0]


def test_a_section_covers_the_pages_it_holds(book: Path) -> None:
    """A section that runs onto the next page says so."""

    nodes = tree(book)
    care = titled(nodes, "Care")

    assert (care.page, care.end_page) == (1, 2)
    assert care.start == pdf.read(book).headings[0][1].start


def test_a_section_ends_before_the_next_one_of_its_level_begins(book: Path) -> None:
    """Two sections at one level meet, and neither holds the other."""

    nodes = tree(book)
    care = titled(nodes, "Care")
    parts = titled(nodes, "Parts")

    assert (care.end_page, care.end) < (parts.page, parts.start)
    assert care.parent == parts.parent


def test_a_subsection_sits_inside_the_section_above_it(book: Path) -> None:
    """The tree keeps the levels the outline gives."""

    nodes = tree(book)
    handbook = titled(nodes, "Handbook")
    care = titled(nodes, "Care")

    assert handbook.parent is None
    assert (handbook.level, care.level) == (1, 2)
    assert care.parent == handbook.ordinal
    assert (care.page, care.end) <= (handbook.end_page, handbook.end)


def test_a_table_belongs_to_the_section_above_it(book: Path) -> None:
    """A table is held by the section it is drawn in, at its own offsets."""

    tables = [piece for piece in pdf.read(book).pieces if piece.kind == pdf.TABLE]
    assert len(tables) == 1

    nodes = tree(book)
    held = [node for node in nodes if node.kind == structure.TABLE]

    assert len(held) == 1
    assert held[0].parent == titled(nodes, "Care").ordinal
    assert (held[0].page, held[0].end_page) == (1, 1)
    assert (held[0].start, held[0].end) == (tables[0].start, tables[0].end)


def test_a_figure_belongs_to_the_heading_above_it(figures: Path) -> None:
    """Each picture is held by the last heading above it on the page."""

    nodes = tree(figures)
    held = [node for node in nodes if node.kind == structure.FIGURE]

    assert [node.parent for node in held] == [
        titled(nodes, "Primo").ordinal,
        titled(nodes, "Secondo").ordinal,
    ]
    assert held[0].bbox == (72.0, 140.0, 192.0, 220.0)
    assert held[1].bbox == (72.0, 420.0, 172.0, 480.0)


def test_a_figure_above_every_heading_is_held_by_the_page_it_opens(
    figures: Path,
) -> None:
    """A section carried onto a page holds what is drawn above its text."""

    path = make_structured_pdf(
        figures.parent / "carried.pdf",
        pages=[
            [Line("Primo", size=20, y=60), Line(BODY, y=100)],
            [Image(120, 80, y=60), Line(BODY, y=200)],
        ],
    )

    nodes = tree(path)
    figures_on_page = [node for node in nodes if node.kind == structure.FIGURE]

    assert len(figures_on_page) == 1
    assert figures_on_page[0].parent == titled(nodes, "Primo").ordinal


def test_a_picture_covering_the_whole_page_is_not_a_figure(figures: Path) -> None:
    """A scanned page is the page, not a figure the document holds."""

    assert len([node for node in tree(figures) if node.kind == structure.FIGURE]) == 2


def test_a_document_with_no_headings_has_no_sections(flat: Path) -> None:
    """What no heading covers is held by the document itself."""

    nodes = tree(flat)

    assert [node.kind for node in nodes] == [structure.TABLE, structure.FIGURE]
    assert [node.parent for node in nodes] == [None, None]


def test_a_document_with_no_outline_is_read_from_the_type_size(sized: Path) -> None:
    """The tree does not depend on the document having an outline."""

    nodes = tree(sized)

    assert [node.title for node in nodes] == ["Chapter", "Section"]
    assert [node.level for node in nodes] == [1, 2]
    assert nodes[1].parent == nodes[0].ordinal
    assert nodes[0].end >= nodes[1].end


def test_two_sections_with_the_same_title_are_two_nodes(repeated: Path) -> None:
    """Sections are matched where they start, not by what they are called."""

    nodes = tree(repeated)

    assert len(nodes) == 2
    assert [node.title for node in nodes] == ["Notes", "Notes"]
    assert [node.page for node in nodes] == [1, 2]
    assert [node.ordinal for node in nodes] == [0, 1]


def test_the_nodes_come_in_the_order_a_reader_meets_them(book: Path) -> None:
    """A node is numbered by its place, and holds only what comes after it."""

    nodes = tree(book)

    assert [node.ordinal for node in nodes] == list(range(len(nodes)))

    places = [(node.page, node.start or 0) for node in nodes]
    assert places == sorted(places)

    for node in nodes:
        assert node.parent is None or node.parent < node.ordinal


def test_a_document_read_twice_gives_the_same_tree(book: Path) -> None:
    """Nothing in the tree is decided anywhere but in the reading."""

    assert tree(book) == tree(book)
