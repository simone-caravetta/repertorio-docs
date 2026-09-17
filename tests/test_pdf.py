"""What the reader makes of a PDF.

Reading a document gives pages of text and a list of pieces, each piece
carrying the section it belongs to. The tests build documents whose layout
they control, then check the text, the offsets into it and where each piece
came from.
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
    """A document with two pages, an outline and a table.

    The outline names the sections and the pages carry the matching titles,
    so the reader has both sources to work from. The second page holds a
    table below a heading.
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
    """A document with no outline, where the type size gives the sections."""
    return make_structured_pdf(
        tmp_path / "notes.pdf",
        pages=[
            [Line("Capitolo", size=15), Line(BODY), Line("Paragrafo", size=13), Line(BODY)],
            [Line(BODY)],
        ],
    )


def test_a_span_sits_where_it_says_it_does(indexed: Path) -> None:
    """Each span covers exactly the page text it points at."""
    for page in pdf.read_pages(indexed):
        for span in page.spans:
            assert page.text[span.start : span.end] == span.text
            assert span.start < span.end


def test_the_pieces_are_the_words_of_the_page(indexed: Path) -> None:
    """Read in order, the pieces spell the page out word for word.

    The pieces of a page never overlap, and every word of the page turns up
    in exactly one of them.
    """

    pages = pdf.read_pages(indexed)
    pieces = pdf.pieces(indexed)

    for page in pages:
        on_page = sorted(by_page(pieces, page.number), key=lambda piece: piece.start)

        for earlier, later in pairwise(on_page):
            assert earlier.end <= later.start

        # Put the pieces back together. The words must come out in the same
        # order as they appear on the page.
        read = [word for piece in on_page for word in piece.text.split()]
        assert read == page.text.split()


def test_a_piece_is_exactly_the_text_it_covers(indexed: Path) -> None:
    """A piece's text is the slice of the page it says it covers."""
    pages = pdf.read_pages(indexed)

    for piece in pdf.pieces(indexed):
        assert piece.text == pages[piece.page].text[piece.start : piece.end]


def test_a_piece_begins_and_ends_on_a_word(indexed: Path) -> None:
    """No piece carries a space at either end.

    A piece covers whole words, so stripping it leaves it as it was.
    """

    for piece in pdf.pieces(indexed):
        assert piece.text == piece.text.strip()


def test_every_piece_has_something_to_draw_on(indexed: Path) -> None:
    """Every piece maps back to at least one box on its page."""
    pages = pdf.read_pages(indexed)

    for piece in pdf.pieces(indexed):
        boxes = pdf.boxes(pages[piece.page], piece.start, piece.end)
        assert boxes, f"nothing to draw for {piece.text[:40]!r}"


def test_the_index_gives_the_sections_and_their_levels(indexed: Path) -> None:
    """The outline supplies the section titles and how deep each one sits."""
    found = {
        (piece.section, piece.level)
        for piece in pdf.pieces(indexed)
        if piece.kind == pdf.TEXT
    }

    assert ("Manuale d'uso", 1) in found
    assert ("Manutenzione", 2) in found
    assert ("Intervalli", 2) in found


def test_a_document_with_an_index_is_not_also_read_by_its_letters(indexed: Path) -> None:
    """When the document has an outline, only its titles open a section.

    The page also carries the smaller line "1.1  Il filtro", which the size
    rule on its own would take for a title.
    """

    sections = {piece.section for piece in pdf.pieces(indexed)}
    assert "1.1  Il filtro" not in sections


def test_without_an_index_the_letters_give_the_sections(plain: Path) -> None:
    """With no outline, the larger text on the page marks the sections."""
    headings = [
        (piece.section, piece.level)
        for piece in pdf.pieces(plain)
        if piece.kind == pdf.TEXT and piece.section is not None
    ]

    assert headings[0] == ("Capitolo", 1)
    assert ("Paragrafo", 2) in headings


def test_a_body_line_is_not_a_section(plain: Path) -> None:
    """Body text stays inside the section it sits under.

    A line of ordinary size is running text, so it opens no section of its
    own even when it is longer than the lines around it.
    """

    body = [piece for piece in pdf.pieces(plain) if piece.text.startswith(BODY)]

    assert body
    assert all(piece.section != BODY for piece in body)


def test_a_section_carries_on_to_the_next_page(plain: Path) -> None:
    """A section opened on one page still holds the pieces on the next."""
    continued = by_page(pdf.pieces(plain), 1)
    assert continued
    assert {piece.section for piece in continued} == {"Paragrafo"}


def test_a_table_is_one_piece(indexed: Path) -> None:
    """A table comes back as one piece holding all of its cells."""
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
    """A large line drawn over a table does not split the table apart.

    The title rules work on text positions, and a title that overlaps the
    rows of a grid must not break the table into two pieces.
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
    """An outline entry with no matching title anywhere is left out.

    The entries that can be placed are kept and the section runs from
    those.
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
    """An entry naming the wrong page still marks its section.

    The outline points one page away from where the title is written. The
    page number only decides between several matches, so the section lands
    on the page that really says the words.
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
    """A title written over two lines is matched as a single name.

    The outline joins the two lines, so the reader compares against the
    joined text and marks the section on the line where it starts.
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
    """A table of contents page does not count as the section itself.

    An entry in a listing ends with a page number and dots, so it never
    equals a title. The reader passes over it and marks the real heading
    further on.
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
    """With the same title on several pages, the nearest one to the index wins.

    The outline names the third page. The words also sit on the first, and
    the reader takes the match closest to the page the outline pointed at.
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
    """An outline naming no title on the page falls back to the size rule."""
    path = make_structured_pdf(
        tmp_path / "nomatch.pdf",
        pages=[[Line("Capitolo", size=15), Line(BODY), Line("Paragrafo", size=13), Line(BODY)]],
        toc=[[1, "Qualcosa", 1], [2, "Altro", 1]],
    )

    sections = {piece.section for piece in pdf.pieces(path)}
    assert sections == {"Capitolo", "Paragrafo"}


def test_a_page_with_no_text_has_nothing_to_read(tmp_path: Path) -> None:
    """A page with no text gives no spans and no pieces."""
    path = make_pdf(tmp_path / "scanned.pdf", "")

    pages = pdf.read_pages(path)
    assert len(pages) == 1
    assert pages[0].spans == ()
    assert pdf.pieces(path) == []


def test_a_range_that_covers_nothing_has_no_boxes(indexed: Path) -> None:
    """An empty range has nothing to draw on."""
    assert pdf.boxes(pdf.read_pages(indexed)[0], 4, 4) == []


def test_a_page_read_on_its_own_is_the_page_of_the_document(indexed: Path) -> None:
    """Reading one page gives the same text as reading the whole file.

    A caller that needs a single page can ask for it by number.
    """

    pages = pdf.read_pages(indexed)

    for number in range(1, len(pages) + 1):
        assert pdf.read_page(indexed, number).text == pages[number - 1].text


def test_a_page_the_document_does_not_have_is_refused(indexed: Path) -> None:
    """A page outside the document raises and names the number asked for.

    The first page is page 1, so 0 and anything past the last page are out
    of range.
    """

    with pytest.raises(IndexError, match="No page 0"):
        pdf.read_page(indexed, 0)

    with pytest.raises(IndexError, match="No page 3"):
        pdf.read_page(indexed, 3)
