"""Turn a PDF into the pieces an index can hold.

The reader walks the file page by page and keeps every text span PyMuPDF
reports, along with the character offset that span occupies in the page text.
Those offsets are how the rest of the module addresses a page, so a heading, a
table and a chunk can all point at the same coordinates.

Sections come from the document outline when the file has one. A file that
carries no outline falls back to the font size, where anything noticeably
larger than the body text counts as a heading. Tables are found with PyMuPDF's
table finder and cut out of the text as pieces of their own.

The result is a list of Piece values. Each one is a stretch of a page together
with the section it belongs to.
"""

from __future__ import annotations

import pymupdf

# PyMuPDF offers a separate layout package and prints a note suggesting it.
# This module reads pages with the built-in extractor, so the note is silenced.

pymupdf.no_recommend_layout()


# A span has to be this much larger than the body text to be a heading.

_HEADING_RATIO = 1.12


# A line longer than this is prose even when it is set large.

_HEADING_MAX_CHARS = 120


# When a line of text is only part of an outline title, it still counts as that
# title if it covers this share of it.

_TITLE_SHARE = 1 / 3


TEXT = "text"
TABLE = "table"


# The name of the reader and the version of its output. Both go into the
# signature the catalog records, so a document whose text would come out
# differently is indexed again.

READER = "pymupdf"
READER_VERSION = 1


class Page:
    """One page of a document.

    `number` counts from zero, `text` is the page as a single string with a
    newline after every line, and `spans` are the runs of text on it.
    """

    __slots__ = ("number", "spans", "text")

    def __init__(self, number: int, text: str, spans: tuple[Span, ...]) -> None:
        self.number = number
        self.text = text
        self.spans = spans


class Span:
    """A run of text set in one font.

    `start` and `end` are offsets into the page text. `bbox` is the rectangle
    the run occupies on the page, in points.
    """

    __slots__ = ("bbox", "end", "size", "start", "text")

    def __init__(
        self,
        start: int,
        end: int,
        bbox: tuple[float, float, float, float],
        size: float,
        text: str,
    ) -> None:
        self.start = start
        self.end = end
        self.bbox = bbox
        self.size = size
        self.text = text


class Heading:
    """A heading found on a page.

    `start` is its offset in the page text, `level` is how deep it sits in the
    outline with 1 at the top, and `title` is its text.
    """

    __slots__ = ("level", "start", "title")

    def __init__(self, start: int, level: int, title: str) -> None:
        self.start = start
        self.level = level
        self.title = title


class Piece:
    """A stretch of a page ready to be indexed.

    `page` counts from zero and `start` and `end` are offsets into that page's
    text. `kind` is TEXT or TABLE. `section` is the heading the piece sits
    under, or None before the first heading, and `level` is that heading's
    level.
    """

    __slots__ = ("end", "kind", "level", "page", "section", "start", "text")

    def __init__(
        self,
        *,
        page: int,
        start: int,
        end: int,
        text: str,
        kind: str,
        section: str | None,
        level: int,
    ) -> None:
        self.page = page
        self.start = start
        self.end = end
        self.text = text
        self.kind = kind
        self.section = section
        self.level = level


def read_pages(path) -> list[Page]:
    """Read every page of a file.

    The whole document is held in memory, so this suits the sizes a document
    library deals with.
    """

    with pymupdf.open(path) as document:
        return [_read_page(page) for page in document]


def read_page(path, page: int) -> Page:
    """Read one page of a file, counted from one as a reader counts pages.

    Raises IndexError when the document has no such page.
    """

    with pymupdf.open(path) as document:
        if not 1 <= page <= document.page_count:
            raise IndexError(
                f"No page {page} in a document of {document.page_count}"
            )

        return _read_page(document[page - 1])


def boxes(page: Page, start: int, end: int) -> list[list[float]]:
    """The rectangles of the spans that overlap a stretch of the page text.

    The stretch runs from `start` up to but not including `end`. Coordinates
    are rounded to two decimals. This is what a viewer is given to highlight
    the passage a question was answered from.
    """

    touched = [
        span
        for span in page.spans
        if max(start, span.start) < min(end, span.end)
    ]

    return [[round(value, 2) for value in span.bbox] for span in touched]


class Reading:
    """A document read once: its pages and the pieces cut from them."""

    __slots__ = ("pages", "pieces")

    def __init__(self, pages: list[Page], pieces: list[Piece]) -> None:
        self.pages = pages
        self.pieces = pieces


def read(path) -> Reading:
    """Read a file and cut it into pieces.

    A document with no pages comes back with an empty piece list.
    """

    pages = read_pages(path)
    if not pages:
        return Reading(pages=pages, pieces=[])

    headings = _headings(path, pages)
    tables = _tables(path, pages)

    return Reading(pages=pages, pieces=_pieces(pages, headings, tables))


def pieces(path) -> list[Piece]:
    """The pieces of a file, without its pages."""

    return read(path).pieces


def _read_page(page: pymupdf.Page) -> Page:
    parts: list[str] = []
    spans: list[Span] = []
    offset = 0

    for block in page.get_text("dict")["blocks"]:
        # Blocks of type 0 hold text. Images and drawings are left out.

        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"]
                if not text:
                    continue
                spans.append(
                    Span(
                        start=offset,
                        end=offset + len(text),
                        bbox=tuple(span["bbox"]),
                        size=float(span["size"]),
                        text=text,
                    )
                )
                parts.append(text)
                offset += len(text)

            parts.append("\n")
            offset += 1

        parts.append("\n")
        offset += 1

    return Page(number=page.number, text="".join(parts), spans=tuple(spans))


def _body_size(pages: list[Page]) -> float:
    """The font size that most of the document's characters are set in.

    Returns 0.0 when the pages carry no text.
    """

    written: dict[float, int] = {}

    for page in pages:
        for span in page.spans:
            written[span.size] = written.get(span.size, 0) + len(span.text.strip())

    if not written:
        return 0.0

    return max(written.items(), key=lambda item: item[1])[0]


def _line_starts(pages: list[Page]) -> list[list[int]]:
    """The offsets of the newline characters, one list per page."""

    return [
        [index for index, char in enumerate(page.text) if char == "\n"]
        for page in pages
    ]


def _line_at(page: Page, starts: list[int], offset: int) -> tuple[int, int]:
    """The start and end offset of the line that holds an offset."""

    start = 0
    for line_start in starts:
        if line_start >= offset:
            return start, line_start
        start = line_start + 1

    return start, len(page.text)


def _headings(path, pages: list[Page]) -> list[list[Heading]]:
    """The headings of each page, taken from the outline when there is one."""

    from_index = _headings_from_index(path, pages)
    if from_index is not None:
        return from_index

    return _headings_from_size(pages)


def _headings_from_index(path, pages: list[Page]) -> list[list[Heading]] | None:
    """Headings from the document outline, or None when that gives nothing.

    Each outline entry is looked for as a line of text near the page the entry
    names. A document whose outline titles cannot be found in the text is
    treated as having no outline, so the caller can fall back to font size.
    """

    with pymupdf.open(path) as document:
        entries = document.get_toc()

    if not entries:
        return None

    lines = [_line_texts(page) for page in pages]
    found: list[list[Heading]] = [[] for _ in pages]
    matched = 0

    for level, title, page_number in entries:
        wanted = _squeeze(title)
        if not wanted:
            continue

        hit = _find_title(lines, wanted, near=page_number - 1)
        if hit is None:
            continue

        page_index, start = hit

        # The line the title was found on is where the heading starts, so a
        # piece can be cut there.

        found[page_index].append(Heading(start, level, _collapse(title)))
        matched += 1

    if not matched:
        return None

    for page, headings in zip(pages, found, strict=True):
        headings.sort(key=lambda heading: heading.start)

    return found


def _line_texts(page: Page) -> list[tuple[int, str]]:
    """The lines of a page, as their offset and their text.

    Runs of whitespace become single spaces and empty lines are left out.
    """

    texts: list[tuple[int, str]] = []
    starts = _line_starts([page])[0]

    for position, start in enumerate([0, *(line_start + 1 for line_start in starts)]):
        end = starts[position] if position < len(starts) else len(page.text)
        text = _collapse(page.text[start:end])
        if text:
            texts.append((start, text))

    return texts


def _find_title(
    lines: list[list[tuple[int, str]]], wanted: str, *, near: int
) -> tuple[int, int] | None:
    """The page and offset of the line that carries an outline title.

    Lines that match the title exactly are looked for first, then lines that
    match it loosely. Among the lines found, the one closest to the page the
    outline names wins.

    Returns None when no line matches.
    """

    for exact in (True, False):
        hits = [
            (page_index, start)
            for page_index, page_lines in enumerate(lines)
            for start, text in page_lines
            if _is_title(text, wanted, exact=exact)
        ]

        if hits:
            return min(hits, key=lambda hit: (abs(hit[0] - near), hit[0]))

    return None


def _is_title(line: str, wanted: str, *, exact: bool) -> bool:
    """Whether a line of text is an outline title.

    With `exact` the line has to be the title. Otherwise a line that contains
    the title counts, and so does a shorter line that the title contains, as
    long as the line covers a fair share of the title.
    """

    line = line.casefold()

    if exact:
        return line == wanted

    if wanted in line:
        return True

    # The title may be split over two lines, so a piece of it is accepted when
    # it is long enough to read as a heading.

    return line in wanted and len(line) >= len(wanted) * _TITLE_SHARE


def _collapse(text: str) -> str:
    """The text with every run of whitespace turned into a single space."""

    return " ".join(text.split())


def _squeeze(text: str) -> str:
    """The text collapsed and lower cased, for comparing two strings."""

    return _collapse(text).casefold()


def _headings_from_size(pages: list[Page]) -> list[list[Heading]]:
    """Headings from font size, for a document with no usable outline.

    Every size at least _HEADING_RATIO times the body size becomes a heading
    size, and the larger sizes are ranked first, so the biggest heading on a
    page is level 1. A line is a heading when the largest span on it is set in
    one of those sizes. Long lines are skipped, since a heading is short.
    """

    body = _body_size(pages)
    if not body:
        return [[] for _ in pages]

    sizes = sorted(
        {
            span.size
            for page in pages
            for span in page.spans
            if span.size >= body * _HEADING_RATIO
        },
        reverse=True,
    )
    levels = {size: level for level, size in enumerate(sizes, start=1)}

    found: list[list[Heading]] = []

    for page, starts in zip(pages, _line_starts(pages), strict=True):
        headings: list[Heading] = []

        for start in [0, *(line_start + 1 for line_start in starts)]:
            _, end = _line_at(page, starts, start)
            line = page.text[start:end].strip()
            if not line or len(line) > _HEADING_MAX_CHARS:
                continue

            size = max(
                (span.size for span in page.spans if start <= span.start < end),
                default=0.0,
            )
            if size in levels:
                headings.append(Heading(start, levels[size], line))

        found.append(headings)

    return found


def _tables(path, pages: list[Page]) -> list[list[tuple[int, int]]]:
    """The table areas of each page, as start and end offsets.

    A table runs from the first span whose centre falls inside the rectangle
    the table finder reports to the last such span. A table that holds no span
    is left out.
    """

    found: list[list[tuple[int, int]]] = []

    with pymupdf.open(path) as document:
        for index, page in enumerate(document):
            ranges: list[tuple[int, int]] = []

            for table in page.find_tables().tables:
                inside = [
                    span
                    for span in pages[index].spans
                    if pymupdf.Rect(table.bbox).contains(_centre(span.bbox))
                ]
                if inside:
                    ranges.append((inside[0].start, inside[-1].end))

            ranges.sort()
            found.append(ranges)

    return found


def _centre(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (x0 + x1) / 2, (y0 + y1) / 2


def _pieces(
    pages: list[Page],
    headings: list[list[Heading]],
    tables: list[list[tuple[int, int]]],
) -> list[Piece]:
    """Cut each page into the pieces an index holds.

    The cuts are the start and end of every table and the start of every
    heading. A piece runs from one cut to the next. Its kind says whether it
    falls inside a table, and its section is the heading it comes after.
    """

    pieces: list[Piece] = []
    section: str | None = None
    level = 0

    for index, page in enumerate(pages):
        ranges = tables[index]
        opens = {heading.start: heading for heading in headings[index]}

        cuts = {0}
        for start, end in ranges:
            cuts.update((start, end))
        for heading in headings[index]:
            # A heading inside a table is skipped. Cutting there would split
            # the table, and the table is kept as one piece.

            if not _inside(ranges, heading.start):
                cuts.add(heading.start)

        bounds = sorted(cut for cut in cuts if 0 <= cut <= len(page.text))
        ends = [*bounds[1:], len(page.text)]

        for position, end in zip(bounds, ends, strict=True):
            # A heading opens a section, and the line it sits on is part of it.

            if position in opens:
                section, level = opens[position].title, opens[position].level

            kind = TABLE if _inside(ranges, position) else TEXT
            _add(pieces, page, position, end, kind, section, level)

    return pieces


def _inside(ranges: list[tuple[int, int]], offset: int) -> bool:
    """Whether an offset falls inside any of the ranges."""

    return any(start <= offset < end for start, end in ranges)


def _add(
    pieces: list[Piece],
    page: Page,
    start: int,
    end: int,
    kind: str,
    section: str | None,
    level: int,
) -> None:
    """Append a piece for a stretch of a page.

    Whitespace at either end is trimmed off and a stretch that holds nothing
    else is dropped, so the piece list carries no blank entries.
    """

    while start < end and page.text[start].isspace():
        start += 1
    while end > start and page.text[end - 1].isspace():
        end -= 1

    if start >= end:
        return

    pieces.append(
        Piece(
            page=page.number,
            start=start,
            end=end,
            text=page.text[start:end],
            kind=kind,
            section=section,
            level=level,
        )
    )
