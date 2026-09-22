"""Hand the model the page a passage came from instead of the passage.

A search compares the question with chunks of about a thousand characters. That
is a good size to match on and often too small to answer from: the sentence that
settles the question can sit in the chunk next to the one that matched, and five
passages from the same page are five times the same paragraphs.

With this on, every passage the search returns is replaced by the whole page it
sits on, joined back together in reading order. The search itself is unchanged
and the scores are the ones it produced, so the order of the pages is the order
of the passages that found them. Two passages of one page become one, and a page
is fetched from the store only when a passage of it came back.

This is a retriever wrapping a retriever, so nothing downstream — the graph, the
two commands, the page, the eval — knows it is there, and nothing is re-indexed.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from app.config import Settings
from app.ingestion import whole_number

# The two values SMALL_TO_BIG accepts. Any other value is refused with a message
# that names them.
MODES = ("off", "page")

# How many chunks of a page are asked for. A page holds a handful of them, and
# the filter keeps the store on that page, so asking for this many costs no more
# than asking for one.
PAGE_CHUNKS = 100

# What goes between two chunks of a page, which is what the context puts
# between two passages.
JOIN = "\n\n"

# The keys that describe one chunk and are dropped from the page made of them.
# A page can hold several sections, and it is not a chunk of anything.
CHUNK_KEYS = ("chunk_id", "section", "level", "table")


def enabled(config: Settings) -> bool:
    """Whether the passages are replaced by the pages they came from.

    SMALL_TO_BIG has to be "off" or "page". Any other value raises an error
    listing the two words that are accepted.
    """
    mode = config.small_to_big.strip().lower()

    if mode == "page":
        return True
    if mode == "off":
        return False

    raise RuntimeError(
        f"Unknown SMALL_TO_BIG {config.small_to_big!r}: expected "
        + " or ".join(MODES)
        + "."
    )


def describe_small_to_big(config: Settings) -> str:
    """One line about small to big, for a command to print when it starts.

    The line says what the model is given, because a run with this on is not
    measured the same way as one without it: the ranking is read over pages
    instead of passages.

    A value that is not "page" is printed as it was written. The run stops
    later, in the search itself, where the mode has to be honoured.
    """
    mode = config.small_to_big.strip().lower()
    if mode != "page":
        return mode

    return "the page of each passage"


class ExpandedRetriever(BaseRetriever):
    """A retriever that hands on the page each passage came from.

    The wrapped retriever runs the search unchanged and decides the order. Every
    passage it returns is replaced by the whole of its page, in reading order,
    and a passage of a page that has already been handed on is dropped.

    A passage whose page cannot be fetched is handed on as it was. That is the
    case for a passage with no page written on it, which no filter can find, and
    for a page holding nothing but the passage itself, where there is nothing to
    join it with.
    """

    retriever: BaseRetriever
    store: VectorStore
    k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        # The callbacks are passed on, so the inner search appears as a run of
        # its own to whoever is watching.
        found = self.retriever.invoke(
            query, config={"callbacks": run_manager.get_child()}
        )

        pages: list[Document] = []
        at: set[tuple[str, int]] = set()

        for chunk in found:
            page = whole_number(chunk.metadata.get("page"))

            if page is None:
                pages.append(chunk)
            else:
                key = (_source_of(chunk), page)
                if key in at:
                    continue
                at.add(key)
                pages.append(self._page(query, chunk, page))

            if len(pages) == self.k:
                break

        return pages

    def _page(self, query: str, chunk: Document, page: int) -> Document:
        """The whole page this passage sits on, or the passage itself."""
        found = self.store.similarity_search(
            query, k=PAGE_CHUNKS, filter=_page_filter(_source_of(chunk), page)
        )
        # A store that ignores part of the filter still cannot put another page
        # or another document into the answer.
        on_page = [
            one
            for one in found
            if whole_number(one.metadata.get("page")) == page
            and _source_of(one) == _source_of(chunk)
        ]

        if len(on_page) < 2:
            return chunk

        return _joined(on_page, chunk)


def _source_of(chunk: Document) -> str:
    """The document a chunk came from, as the metadata holds it."""
    return str(chunk.metadata.get("source") or "")


def _page_filter(source: str, page: int) -> dict[str, Any]:
    """The filter that keeps one page of one document.

    The two conditions are joined explicitly instead of being written as two
    keys of one object, because Chroma reads one operator per level and refuses
    the shorter form.
    """
    return {
        "$and": [
            {"source": {"$eq": source}},
            {"page": {"$eq": page}},
        ]
    }


def _joined(chunks: list[Document], found: Document) -> Document:
    """One page out of the chunks of it, in reading order.

    The metadata of the page is the metadata of the passage that was found, with
    the range widened over the whole page and the keys that describe a single
    chunk dropped.
    """
    ordered = sorted(chunks, key=_position)
    metadata = dict(found.metadata)

    starts = [whole_number(one.metadata.get("start")) for one in ordered]
    ends = [whole_number(one.metadata.get("end")) for one in ordered]
    if all(one is not None for one in starts + ends):
        metadata["start"] = min(one for one in starts if one is not None)
        metadata["end"] = max(one for one in ends if one is not None)

    for key in CHUNK_KEYS:
        metadata.pop(key, None)

    return Document(
        page_content=JOIN.join(one.page_content for one in ordered),
        metadata=metadata,
    )


def _position(chunk: Document) -> tuple[int, int]:
    """Where a chunk sits on its page, as a sort key.

    The position within the page comes first. The chunk id is written when the
    document is indexed and counts through the whole document, so it orders the
    chunks of one page the way they were read.
    """
    return (
        whole_number(chunk.metadata.get("start")) or 0,
        whole_number(chunk.metadata.get("chunk_id")) or 0,
    )
