"""Reading a document: where its structure comes from, and where its pieces sit.

The offsets these tests are about are the ones the ingest writes into a chunk and
the endpoint reads back to draw a rectangle, so most of what is asserted here is
the one contract that has to hold between them: a piece is exactly the text of
the page it says it covers.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from app import pdf
from tests.helpers import BODY, Line, Table, make_pdf, make_structured_pdf


def by_page(pieces: list[pdf.Piece], number: int) -> list[pdf.Piece]:
    return [piece for piece in pieces if piece.page == number]


@pytest.fixture
def indexed(tmp_path: Path) -> Path:
    """A document that declares its own sections, and holds a table.

    Its letters say one thing more than its index does — a line at 13pt that the
    index never mentions — so a test can tell which of the two was read.
    """
    return make_structured_pdf(
        tmp_path / "manual.pdf",
        pages=[
            [
                Line("Manuale d'uso", size=22),
                Line("1  Manutenzione", size=15),
                Line(BODY),
                Line("1.1  Il filtro", size=13),
                Line(BODY),
            ],
            [
                Line("2  Intervalli", size=15),
                Line(BODY),
                Table([["Parte", "Ore", "Nota"], ["Filtro", "200", "pulire"]]),
                Line(BODY),
            ],
        ],
        toc=[
            [1, "Manuale d'uso", 1],
            [2, "Manutenzione", 1],
            [2, "Intervalli", 2],
        ],
    )


@pytest.fixture
def plain(tmp_path: Path) -> Path:
    """The same shape of document, saying nothing about itself."""
    return make_structured_pdf(
        tmp_path / "notes.pdf",
        pages=[
            [Line("Capitolo", size=15), Line(BODY), Line("Paragrafo", size=13), Line(BODY)],
            [Line(BODY)],
        ],
    )


def test_a_span_sits_where_it_says_it_does(indexed: Path) -> None:
    """The pair of offsets and text agree, which is all the rest rests on."""
    for page in pdf.read_pages(indexed):
        for span in page.spans:
            assert page.text[span.start : span.end] == span.text
            assert span.start < span.end


def test_the_pieces_are_the_words_of_the_page(indexed: Path) -> None:
    """Read in order they give back every word, each of them once.

    Not the text character for character: a piece is trimmed of the space around
    it, so the whitespace between two pieces belongs to neither. What must not
    happen is a word in two pieces or in none.
    """
    pages = pdf.read_pages(indexed)
    pieces = pdf.pieces(indexed)

    for page in pages:
        on_page = sorted(by_page(pieces, page.number), key=lambda piece: piece.start)

        for earlier, later in pairwise(on_page):
            assert earlier.end <= later.start

        # The words, in order and once each. Not the characters: a piece is
        # trimmed of the space around it, so the whitespace between two of them
        # belongs to neither, and the comparison is of what is left.
        read = [word for piece in on_page for word in piece.text.split()]
        assert read == page.text.split()


def test_a_piece_is_exactly_the_text_it_covers(indexed: Path) -> None:
    """The contract the endpoint depends on: offsets index the page's own text."""
    pages = pdf.read_pages(indexed)

    for piece in pdf.pieces(indexed):
        assert piece.text == pages[piece.page].text[piece.start : piece.end]


def test_a_piece_begins_and_ends_on_a_word(indexed: Path) -> None:
    """The blank lines between two pieces belong to neither of them.

    Left on, every chunk would open with the newlines that separated it from the
    piece before, and a table would be read with the space around its grid.
    """
    for piece in pdf.pieces(indexed):
        assert piece.text == piece.text.strip()


def test_every_piece_has_something_to_draw_on(indexed: Path) -> None:
    """Offsets that survive the round trip through a page number and back."""
    pages = pdf.read_pages(indexed)

    for piece in pdf.pieces(indexed):
        boxes = pdf.boxes(pages[piece.page], piece.start, piece.end)
        assert boxes, f"nothing to draw for {piece.text[:40]!r}"


def test_the_index_gives_the_sections_and_their_levels(indexed: Path) -> None:
    """Declared levels, taken as declared rather than re-derived from the sizes."""
    found = {
        (piece.section, piece.level)
        for piece in pdf.pieces(indexed)
        if piece.kind == pdf.TEXT
    }

    assert ("Manuale d'uso", 1) in found
    assert ("Manutenzione", 2) in found
    assert ("Intervalli", 2) in found


def test_a_document_with_an_index_is_not_also_read_by_its_letters(indexed: Path) -> None:
    """The 13pt line the index never names is body, not a section of its own.

    Which is the whole reason the two are never mixed: a document that says what
    its sections are is not improved by guessing at others from the typography.
    """
    sections = {piece.section for piece in pdf.pieces(indexed)}
    assert "1.1  Il filtro" not in sections


def test_without_an_index_the_letters_give_the_sections(plain: Path) -> None:
    """Sizes, ordered largest first, and nothing nested beyond that."""
    headings = [
        (piece.section, piece.level)
        for piece in pdf.pieces(plain)
        if piece.kind == pdf.TEXT and piece.section is not None
    ]

    assert headings[0] == ("Capitolo", 1)
    assert ("Paragrafo", 2) in headings


def test_a_body_line_is_not_a_section(plain: Path) -> None:
    """The size the document mostly writes in is what a section is measured against.

    The measure is taken over the characters and not the lines: a page of headings
    has more headings on it than a page of prose has paragraphs, and counting
    lines would call the document's body the exception and everything else the
    rule.
    """
    body = [piece for piece in pdf.pieces(plain) if piece.text.startswith(BODY)]

    assert body
    assert all(piece.section != BODY for piece in body)


def test_a_section_carries_on_to_the_next_page(plain: Path) -> None:
    """A page that opens mid-section is under the heading that opened it."""
    continued = by_page(pdf.pieces(plain), 1)
    assert continued
    assert {piece.section for piece in continued} == {"Paragrafo"}


def test_a_table_is_one_piece(indexed: Path) -> None:
    """Whole, because a grid read in halves is read as prose."""
    tables = [piece for piece in pdf.pieces(indexed) if piece.kind == pdf.TABLE]

    assert len(tables) == 1
    assert tables[0].text.split() == [
        "Parte",
        "Ore",
        "Nota",
        "Filtro",
        "200",
        "pulire",
    ]


def test_a_section_title_inside_a_table_does_not_cut_it(plain: Path, tmp_path: Path) -> None:
    """A large line in a cell is part of the grid, not a heading over it.

    Trusting it would cut the table in two and file half of it under a title it
    has nothing to do with.
    """
    path = make_structured_pdf(
        tmp_path / "grid.pdf",
        pages=[
            [
                Table([["Parte", "Ore"], ["Filtro", "200"]]),
                Line("Grande", size=20, y=110),
            ]
        ],
    )

    tables = [piece for piece in pdf.pieces(path) if piece.kind == pdf.TABLE]
    assert len(tables) == 1
    assert "Grande" in tables[0].text


def test_an_entry_that_names_nothing_on_its_page_is_dropped(tmp_path: Path) -> None:
    """An index points at a page, and a page is not a position.

    The entry naming a page it is not written on cannot be placed, so it is left
    out rather than put at the top of the page it points at.
    """
    path = make_structured_pdf(
        tmp_path / "wrong.pdf",
        pages=[[Line("Capitolo", size=15), Line(BODY)]],
        toc=[[1, "Capitolo", 1], [2, "Altrove", 1]],
    )

    sections = {piece.section for piece in pdf.pieces(path)}
    assert "Capitolo" in sections
    assert "Altrove" not in sections


def test_an_entry_that_names_the_wrong_page_is_still_found(tmp_path: Path) -> None:
    """A printed index is often a page or two out, in either direction.

    The page number says where to look first and nothing more; the title is what
    is written on the page the section opens, and that is what is trusted. The
    book this was written for is out by a page on three of its four chapters.

    The title is set in the body size, so that reading the letters cannot stand in
    for reading the index: a section found here was found by the index or not at
    all.
    """
    path = make_structured_pdf(
        tmp_path / "offset.pdf",
        pages=[
            [Line(BODY), Line(BODY)],
            [Line("Capitolo secondo"), Line(BODY)],
        ],
        toc=[[1, "Capitolo secondo", 1]],
    )

    assert {piece.section for piece in by_page(pdf.pieces(path), 1)} == {
        "Capitolo secondo"
    }
    assert {piece.section for piece in by_page(pdf.pieces(path), 0)} == {None}


def test_a_title_the_document_wraps_is_still_one_title(tmp_path: Path) -> None:
    """The index joins a title the page needed two lines for, and `\\r` is that join.

    Neither half says the title, so neither is equal to it: what makes the entry
    usable is that one of them is the beginning of it.
    """
    path = make_structured_pdf(
        tmp_path / "wrapped.pdf",
        pages=[
            [
                Line("Capitolo molto", size=15),
                Line("lungo davvero", size=15),
                Line(BODY),
            ]
        ],
        toc=[[1, "Capitolo molto\rlungo davvero", 1]],
    )

    assert {piece.section for piece in pdf.pieces(path)} == {
        "Capitolo molto lungo davvero"
    }


def test_a_listing_of_the_sections_does_not_answer_for_them(tmp_path: Path) -> None:
    """The page that lists the sections writes their numbers and their dots too.

    A line carrying a title with a page number and a row of dots after it is not
    a line carrying the title, and saying so exactly first is what keeps a
    section from being put on the page that only announces it.

    The listing is put nearer the page the index names than the section itself,
    so that nothing but the exact pass can choose between them; and the section
    is set in the body size, so that the letters cannot be read instead.
    """
    path = make_structured_pdf(
        tmp_path / "listing.pdf",
        pages=[
            [Line("Indice"), Line("Capitolo primo .......... 4")],
            [Line(BODY)],
            [Line(BODY)],
            [Line("Capitolo primo"), Line(BODY)],
        ],
        toc=[[1, "Capitolo primo", 2]],
    )

    placed = [(piece.section, piece.page) for piece in pdf.pieces(path) if piece.section]
    assert placed == [("Capitolo primo", 3)]


def test_the_section_is_the_nearest_page_that_says_the_title(tmp_path: Path) -> None:
    """A page writing a bare title is not always the page the section opens on.

    An index whose entries keep their page numbers in a column of their own writes
    each title and nothing else, exactly as a heading does, so both pages say the
    title and both say it exactly. Which of them is the section is decided by the
    page the index named, and the nearer of the two is the one it meant.
    """
    path = make_structured_pdf(
        tmp_path / "twice.pdf",
        pages=[
            [Line("Capitolo primo"), Line("3")],
            [Line(BODY)],
            [Line("Capitolo primo"), Line(BODY)],
        ],
        toc=[[1, "Capitolo primo", 3]],
    )

    placed = [(piece.section, piece.page) for piece in pdf.pieces(path) if piece.section]
    assert placed == [("Capitolo primo", 2)]


def test_an_index_that_matches_nothing_leaves_the_letters_to_speak(tmp_path: Path) -> None:
    """A document whose index cannot be used is read as one that has none."""
    path = make_structured_pdf(
        tmp_path / "nomatch.pdf",
        pages=[[Line("Capitolo", size=15), Line(BODY), Line("Paragrafo", size=13), Line(BODY)]],
        toc=[[1, "Qualcosa", 1], [2, "Altro", 1]],
    )

    sections = {piece.section for piece in pdf.pieces(path)}
    assert sections == {"Capitolo", "Paragrafo"}


def test_a_page_with_no_text_has_nothing_to_read(tmp_path: Path) -> None:
    """A scanned page has no words and so no pieces, rather than an empty one."""
    path = make_pdf(tmp_path / "scanned.pdf", "")

    pages = pdf.read_pages(path)
    assert len(pages) == 1
    assert pages[0].spans == ()
    assert pdf.pieces(path) == []


def test_a_range_that_covers_nothing_has_no_boxes(indexed: Path) -> None:
    """Half-open, so an empty range is empty and not the box of the character at it."""
    assert pdf.boxes(pdf.read_pages(indexed)[0], 4, 4) == []


def test_a_page_read_on_its_own_is_the_page_of_the_document(indexed: Path) -> None:
    """The offsets index one text, and the two readings of it have to be that one.

    A document is read whole to be indexed and one page of it is read to be drawn
    on: a chunk's offsets were written against the first, and a rectangle is
    found with the second. A character of difference between them is a highlight
    a word out, and nothing in either reading would say so.
    """
    pages = pdf.read_pages(indexed)

    for number in range(1, len(pages) + 1):
        assert pdf.read_page(indexed, number).text == pages[number - 1].text


def test_a_page_the_document_does_not_have_is_refused(indexed: Path) -> None:
    """Counted from one, and read as a number rather than as an index.

    Zero is the case worth naming: it is a valid index into a list of pages, so
    read as one it answers about the last page of the document.
    """
    with pytest.raises(IndexError, match="No page 0"):
        pdf.read_page(indexed, 0)

    with pytest.raises(IndexError, match="No page 3"):
        pdf.read_page(indexed, 3)
