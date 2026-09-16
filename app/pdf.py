"""Reading a PDF: its text, the structure it declares, and where each piece sits.

PyMuPDF rather than the line-based reader the project started with, because two
things need it and neither can be had later without reading every document again:
the rectangle a run of text covers on the page, which is what a citation has to
point at, and the layout — which line is a heading, which block is a table —
which is what decides where a chunk may be cut.

The text of a page is built here and nowhere else. The ingest writes offsets into
the chunk metadata and the endpoint that answers with rectangles reads them back,
so both have to count the same characters: a second implementation that differed
by one space would put the highlight on the wrong words.
"""

from __future__ import annotations

import pymupdf

# PyMuPDF prints its advice about an optional layout package on stdout, which is
# where the commands write what they have to say. It has its own switch for this,
# documented for callers who have decided not to install that package.
pymupdf.no_recommend_layout()

# A line is a heading when its letters are this much larger than the size the
# document mostly writes in...
_HEADING_RATIO = 1.12
# ...and when it is short. A large line running across the page is a title page
# or a pulled quote, and taking it for a heading would cut a section there.
_HEADING_MAX_CHARS = 120

# How much of an index entry a line has to say to be taken for its beginning or
# its end, when no line says the whole of it. A title that needed two lines was
# split where it was split, and the shorter half is the one this is for.
_TITLE_SHARE = 1 / 3

# What a piece is. A table is one because it is read as a grid and a chunk that
# held half of it would be read as prose.
TEXT = "text"
TABLE = "table"

# What this reading is, for a catalog that records how an index was built and
# compares it on the next run. Raised by hand when a change here makes an index
# built by the older code worth rebuilding: the rules that decide what a piece
# is, where a heading sits, how a table is cut. The version of PyMuPDF is
# deliberately not part of it — an upgrade that reads the same document into the
# same pieces would cost a library a full re-index and buy it nothing.
READER = "pymupdf"
READER_VERSION = 1


class Page:
    """One page's text, and the runs it is made of.

    `text` is built from the page's spans in the order the document writes them,
    lines separated by a newline and blocks by a blank line. `spans` says where
    in that text each run sits and which rectangle it covers, which is what turns
    an offset into something to draw on the page.
    """

    __slots__ = ("number", "spans", "text")

    def __init__(self, number: int, text: str, spans: tuple[Span, ...]) -> None:
        # Counted from zero, which is how the page is numbered everywhere the
        # chunks are: the reader counts from one, and the one place that matters
        # is where the page number is put on screen.
        self.number = number
        self.text = text
        self.spans = spans


class Span:
    """A run of text on a page: where it is in the text, and where on the page."""

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
    """A section title: the level it declares, and where it starts."""

    __slots__ = ("level", "start", "title")

    def __init__(self, start: int, level: int, title: str) -> None:
        self.start = start
        self.level = level
        self.title = title


class Piece:
    """A run of a page's text that is a unit.

    Nothing is cut into a piece: a section's body is split into chunks by size as
    it always was, but never across a heading, and a table is one piece whatever
    its size, because a table read in halves is not a table.
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
    """Every page of a document, text and spans together.

    The blocks are taken in the order the document writes them rather than sorted
    into place. Sorting reads a page by position, which is right for a page of one
    column and wrong for a page of two: it would run the left column into the
    right one line by line.
    """
    with pymupdf.open(path) as document:
        return [_read_page(page) for page in document]


def read_page(path, page: int) -> Page:
    """One page of a document, read on its own and counted from one.

    A document is read whole to be indexed; a page is read to be pointed at, and
    reading the hundred and five of them to draw on one is work nobody asked for
    — this is a request from a reader clicking a citation, and it is answered in
    the time one page takes rather than in the time the book does.

    A page the document does not have raises `IndexError` rather than being read
    as whichever page a number happens to name. A negative index is a valid one:
    a page counted from one that arrives here as a zero is the last page of the
    document, and nothing about the page that comes back says so.
    """
    with pymupdf.open(path) as document:
        if not 1 <= page <= document.page_count:
            raise IndexError(
                f"No page {page} in a document of {document.page_count}"
            )

        return _read_page(document[page - 1])


def boxes(page: Page, start: int, end: int) -> list[list[float]]:
    """The rectangles covering `text[start:end]` of one page, as the reader sees it.

    Every span the range touches contributes its own rectangle, so a passage that
    wraps across three lines comes back as three: one box around the lot would
    cover the whole width of the page and say nothing.

    The page is handed in already read rather than named by a number against a
    file. A caller holding a number has to turn it into an index, and a page
    counted from one that arrived here as a zero would be the last page of the
    document — a wrong answer to a question about the first, and nothing in the
    rectangles to say so.
    """
    # Overlapping by at least one character, which is the test `max < min` and not
    # the pair of comparisons it looks like: compared separately, a range of no
    # characters sitting inside a span passes both of them.
    touched = [
        span
        for span in page.spans
        if max(start, span.start) < min(end, span.end)
    ]

    return [[round(value, 2) for value in span.bbox] for span in touched]


class Reading:
    """A document as it was read: its pages, and the pieces they are made of.

    Both together because reading a PDF is the expensive part and every caller
    wants both: the ingest cuts chunks along the pieces and writes offsets into
    the pages, and looking at either alone would mean reading the document twice.
    """

    __slots__ = ("pages", "pieces")

    def __init__(self, pages: list[Page], pieces: list[Piece]) -> None:
        self.pages = pages
        self.pieces = pieces


def read(path) -> Reading:
    """A document, read once.

    The structure comes from the document's own index when it has one — a PDF
    that declares its sections is believed — and otherwise from the size of the
    letters, which is the only thing left to go on.
    """
    pages = read_pages(path)
    if not pages:
        return Reading(pages=pages, pieces=[])

    headings = _headings(path, pages)
    tables = _tables(path, pages)

    return Reading(pages=pages, pieces=_pieces(pages, headings, tables))


def pieces(path) -> list[Piece]:
    """A document as the units it is made of, in reading order."""
    return read(path).pieces


def _read_page(page: pymupdf.Page) -> Page:
    parts: list[str] = []
    spans: list[Span] = []
    offset = 0

    for block in page.get_text("dict")["blocks"]:
        # An image block has no lines to read, and its rectangle says nothing
        # about where the text around it is.
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
    """The size the document mostly writes in.

    Measured in characters rather than in spans: a page of headings has more
    headings than a page of body has paragraphs, and counting spans would call
    the document's body the exception.
    """
    written: dict[float, int] = {}

    for page in pages:
        for span in page.spans:
            written[span.size] = written.get(span.size, 0) + len(span.text.strip())

    if not written:
        return 0.0

    return max(written.items(), key=lambda item: item[1])[0]


def _line_starts(pages: list[Page]) -> list[list[int]]:
    """Where each line of each page begins, which is where a heading can begin."""
    return [
        [index for index, char in enumerate(page.text) if char == "\n"]
        for page in pages
    ]


def _line_at(page: Page, starts: list[int], offset: int) -> tuple[int, int]:
    """The line `offset` falls in, as the half-open range of its characters."""
    start = 0
    for line_start in starts:
        if line_start >= offset:
            return start, line_start
        start = line_start + 1

    return start, len(page.text)


def _headings(path, pages: list[Page]) -> list[list[Heading]]:
    """The headings of each page, from the document's index or from the letters.

    Both are tried in that order and never mixed: a document that declares its
    sections has them taken as it declares them, levels included, and one that
    does not gets what its typography says and no levels beyond that.
    """
    from_index = _headings_from_index(path, pages)
    if from_index is not None:
        return from_index

    return _headings_from_size(pages)


def _headings_from_index(path, pages: list[Page]) -> list[list[Heading]] | None:
    """The document's own table of contents, matched to the lines it names.

    An entry is usable once its title has been found written somewhere, and the
    page number it carries only says where to look first: an index is often the
    printed one, and a book whose plates and front matter are not counted with
    the pages has entries that are a page or two out, in either direction. What
    is trusted is the title, which is on the page the section opens.

    Entries naming nothing anywhere are dropped, and a document whose index
    cannot be matched at all is read as if it had none.
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
        # The index's own spelling, with the whitespace closed up. An entry whose
        # title wraps carries the break inside it — a `\r` in the middle of a
        # chapter's name — and that is a title for a section, not a line.
        found[page_index].append(Heading(start, level, _collapse(title)))
        matched += 1

    if not matched:
        return None

    for page, headings in zip(pages, found, strict=True):
        headings.sort(key=lambda heading: heading.start)

    return found


def _line_texts(page: Page) -> list[tuple[int, str]]:
    """Each line of a page as where it starts and what it says, closed up.

    Empty lines are left out: nothing is written on them, so nothing can be
    looked for on them.
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
    """Where the index's title is written, as a page and an offset into its text.

    Matched in two passes over the whole document, and never mixed: first the
    lines that say the title and nothing else, and only if there are none, the
    lines that say part of it. The order is what keeps an index page from
    answering for its own entries — a listing writes each title with a page
    number and a row of dots after it, so it is never equal to a title, while a
    heading is — and the second pass is what finds a title the document wraps,
    the index having joined it into one line.

    Among the lines that match, the one nearest the page the index named: an
    entry that names a page it is not on is common, and the nearest heading with
    that name is the one it meant.
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
    """Whether a line is the title, or at least enough of it to be the same one.

    `wanted` has already been through `_squeeze`, so the line is put through the
    same before the two are compared.
    """
    line = line.casefold()

    if exact:
        return line == wanted

    if wanted in line:
        return True

    # A part of the title, which is what the first or the last line of a wrapped
    # heading is. Held to a share of it: a line of two or three characters is
    # contained in most titles, and a page number is a line of two or three
    # characters.
    return line in wanted and len(line) >= len(wanted) * _TITLE_SHARE


def _collapse(text: str) -> str:
    """Text with its whitespace closed up: one space, and none at the ends."""
    return " ".join(text.split())


def _squeeze(text: str) -> str:
    """The same, and case dropped, for comparing two spellings of one name."""
    return _collapse(text).casefold()


def _headings_from_size(pages: list[Page]) -> list[list[Heading]]:
    """The headings a document has by typography alone, one level deep.

    Every line written in letters larger than the body is a heading, and the
    sizes it uses are ordered into levels: the largest is the outermost. Nothing
    is guessed beyond that, which is why a document with no index gets sections
    that are recognised but not nested.
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
    """The range each table covers in its page's text, page by page.

    A table is a rectangle on the page and a run of characters in the text, and
    the two are brought together by the spans that fall inside it: the range runs
    from the first of them to the last. A table whose cells hold no text — a
    picture of a table — has no range, because there is nothing to read.
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
    """The runs of each page, cut where a heading or a table says to cut.

    A page is divided at the points its structure names and nowhere else, so the
    runs tile its text exactly — every character in one of them and in only one.
    The section a run belongs to is the last heading at or before it, and it
    carries across pages, because a section does: the run that continues on the
    next page is still under the heading that opened it.
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
            # A heading inside a table, or beginning where one begins, is part of
            # the table. Trusting it would cut the grid in two and file it under
            # the first cell it happens to hold.
            if not _inside(ranges, heading.start):
                cuts.add(heading.start)

        bounds = sorted(cut for cut in cuts if 0 <= cut <= len(page.text))
        ends = [*bounds[1:], len(page.text)]

        for position, end in zip(bounds, ends, strict=True):
            # Read before the run rather than after it: the heading opens the
            # section that this run, its own title included, belongs to.
            if position in opens:
                section, level = opens[position].title, opens[position].level

            kind = TABLE if _inside(ranges, position) else TEXT
            _add(pieces, page, position, end, kind, section, level)

    return pieces


def _inside(ranges: list[tuple[int, int]], offset: int) -> bool:
    """Whether an offset falls in one of the ranges."""
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
    """Add the run between two marks, trimmed to the text it holds.

    The trim is what keeps `text == page.text[start:end]` exactly true, which is
    the whole of the contract: the offsets in the chunk metadata are read back
    against this text by the endpoint that draws the rectangles.
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
