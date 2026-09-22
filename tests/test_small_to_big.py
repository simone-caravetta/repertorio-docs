"""Handing the model the page a passage came from.

The setting decides whether a search returns its passages or the pages they sit
on. With it on, the store is asked for the page of every passage that came back
and the passages of one page are joined into one. The tests cover the setting,
the line a command prints, the retriever itself and where it sits in the chain
the search is built from.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.callbacks import (
    BaseCallbackHandler,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

import app.rerank as rerank_module
import app.small_to_big as small_to_big_module
import app.vectorstore as vectorstore_module
from app.small_to_big import PAGE_CHUNKS, ExpandedRetriever
from tests.helpers import FakeReranker, FakeVectorStore, make_settings

SHEET = "schede/alluvione-box.pdf"
OTHER = "schede/alluvione-cantina.pdf"


class RecordingRetriever(BaseRetriever):
    """A retriever that returns fixed documents and keeps the queries.

    The questions it was called with are recorded, so a test can check what
    the expanding retriever passed down to it.
    """

    documents: list[Document]
    queries: list[str] = Field(default_factory=list)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        self.queries.append(query)
        return self.documents


class RecordingHandler(BaseCallbackHandler):
    """A callback handler that notes every retriever run it hears about."""

    def __init__(self) -> None:
        self.runs: list[tuple[str, object]] = []

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        parent_run_id: object = None,
        **kwargs: Any,
    ) -> None:
        self.runs.append((query, parent_run_id))


class DeafStore(FakeVectorStore):
    """A store that reads the source of a filter and ignores the rest.

    A store of another kind may translate only part of what it is given, and
    the page must still hold nothing but the page.
    """

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: object
    ) -> list[Document]:
        criteria = kwargs.get("filter")
        source = None
        if isinstance(criteria, dict):
            source = criteria["$and"][0]["source"]["$eq"]  # type: ignore[index]
        return super().similarity_search(query, k=k, filter={"source": source})


def page(source: str, number: int, texts: list[str]) -> list[Document]:
    """The chunks of one page, in reading order, as the store holds them."""
    return [
        Document(
            page_content=text,
            metadata={
                "source": source,
                "page": number,
                "chunk_id": position,
                "start": position * 100,
                "end": position * 100 + 99,
                "document_type": "scheda",
                "section": "Limiti e franchigie",
                "level": 2,
            },
        )
        for position, text in enumerate(texts)
    ]


def holding(*documents: Document, store: type[FakeVectorStore] = FakeVectorStore):
    """A store holding these documents.

    A search of this store returns them in reverse, so the order the
    application produces is easy to tell from the store's own. The kind of
    store can be named when a test needs one that reads filters its own way.
    """
    made = store()
    made.added.append((list(documents), [one.page_content for one in documents]))
    return made


def named_scores(scores: dict[str, float]):
    """A scoring callable that looks a passage up by its text.

    A passage with no entry scores zero, so several of them keep the order
    the search produced.
    """
    def score(pair: tuple[str, str], at: int) -> float:
        return scores.get(pair[1], 0.0)

    return score


def wired(
    monkeypatch: pytest.MonkeyPatch,
    store: FakeVectorStore,
    fake: FakeReranker | None = None,
    **settings: object,
):
    """Point the retriever at the fake store, with small to big on."""
    monkeypatch.setattr(
        vectorstore_module, "settings", make_settings(small_to_big="page", **settings)
    )
    monkeypatch.setattr(vectorstore_module, "get_vectorstore", lambda: store)
    if fake is not None:
        monkeypatch.setattr(rerank_module, "get_reranker", lambda config: fake)
    return store


# The setting itself: what it accepts and what a command prints from it.


def test_the_pages_are_handed_on_only_when_the_setting_says_so():
    assert small_to_big_module.enabled(make_settings(small_to_big="page")) is True
    assert small_to_big_module.enabled(make_settings(small_to_big="off")) is False


def test_a_mode_written_with_spaces_or_in_capitals_still_reads():
    """The mode is trimmed and lowercased before it is read."""
    assert small_to_big_module.enabled(make_settings(small_to_big="  PAGE  ")) is True
    assert small_to_big_module.enabled(make_settings(small_to_big="Off")) is False


def test_an_unknown_mode_is_refused_and_both_are_named():
    with pytest.raises(RuntimeError, match="off or page"):
        small_to_big_module.enabled(make_settings(small_to_big="sections"))


def test_the_line_a_command_prints_says_what_the_model_is_given():
    assert (
        small_to_big_module.describe_small_to_big(make_settings(small_to_big="page"))
        == "the page of each passage"
    )


def test_the_line_for_a_run_that_does_not_expand_says_off():
    assert (
        small_to_big_module.describe_small_to_big(make_settings(small_to_big="off"))
        == "off"
    )


def test_the_line_never_stops_the_run_it_describes():
    """A value that is not "page" is printed as it was written.

    The run is stopped where the mode has to be honoured.
    """
    said = small_to_big_module.describe_small_to_big(
        make_settings(small_to_big="maybe")
    )

    assert said == "maybe"


# The retriever on its own, driven by a retriever that records the questions
# it was given.


def test_a_passage_is_replaced_by_the_page_it_came_from():
    """The whole page comes back, joined in reading order.

    The search found the middle chunk of the page. What the caller receives
    is the three chunks of it in the order they are read, whichever order the
    store handed them over in.
    """
    chunks = page(SHEET, 0, ["prima", "seconda", "terza"])
    inner = RecordingRetriever(documents=[chunks[1]])

    found = ExpandedRetriever(
        retriever=inner, store=holding(*chunks), k=5
    ).invoke("q")

    assert [one.page_content for one in found] == ["prima\n\nseconda\n\nterza"]
    assert inner.queries == ["q"]


def test_two_passages_of_one_page_become_one():
    """The second passage of a page is dropped, and the page comes back once."""
    chunks = page(SHEET, 0, ["prima", "seconda", "terza"])
    inner = RecordingRetriever(documents=[chunks[0], chunks[2]])
    store = holding(*chunks)

    found = ExpandedRetriever(retriever=inner, store=store, k=5).invoke("q")

    assert len(found) == 1
    # One fetch for the page, and no second one for the passage that was
    # dropped.
    assert len(store.searches) == 1


def test_the_pages_come_back_in_the_order_the_passages_were_found():
    """The first passage of a page decides where the page sits.

    The ranking is the one the search produced: expanding a passage does not
    move it.
    """
    first = page(SHEET, 0, ["a1", "a2"])
    second = page(OTHER, 3, ["b1", "b2"])
    inner = RecordingRetriever(documents=[second[0], first[0]])

    found = ExpandedRetriever(
        retriever=inner, store=holding(*first, *second), k=5
    ).invoke("q")

    assert [one.metadata["source"] for one in found] == [OTHER, SHEET]


def test_the_store_is_asked_for_one_page_of_one_document():
    """The filter keeps the store on the page, which is what makes it cheap.

    The two conditions are joined, because Chroma reads one operator per
    level and refuses the shorter form.
    """
    chunks = page(SHEET, 4, ["prima", "seconda"])
    store = holding(*chunks)

    ExpandedRetriever(
        retriever=RecordingRetriever(documents=[chunks[0]]), store=store, k=5
    ).invoke("q")

    assert store.searches == [
        (
            "q",
            PAGE_CHUNKS,
            {"$and": [{"source": {"$eq": SHEET}}, {"page": {"$eq": 4}}]},
        )
    ]


def test_the_range_covers_the_page_and_the_keys_of_one_chunk_are_dropped():
    """The page carries the range of the whole of it.

    What a reader is sent to is the page, so the range runs from the first
    chunk of it to the last. The keys that describe a single chunk go: a page
    holds more than one section and is not a chunk of anything.
    """
    chunks = page(SHEET, 0, ["prima", "seconda", "terza"])

    found = ExpandedRetriever(
        retriever=RecordingRetriever(documents=[chunks[1]]),
        store=holding(*chunks),
        k=5,
    ).invoke("q")

    metadata = found[0].metadata
    assert (metadata["start"], metadata["end"]) == (0, 299)
    for key in ("chunk_id", "section", "level", "table"):
        assert key not in metadata
    assert metadata["page"] == 0
    assert metadata["source"] == SHEET


def test_a_page_holding_one_chunk_is_handed_back_as_it_was():
    """There is nothing to join a lone chunk with, so it is not touched."""
    chunk = page(SHEET, 0, ["prima"])[0]
    store = holding(chunk)

    found = ExpandedRetriever(
        retriever=RecordingRetriever(documents=[chunk]), store=store, k=5
    ).invoke("q")

    assert found == [chunk]
    assert found[0].metadata["chunk_id"] == 0


def test_a_passage_with_no_page_is_handed_back_as_it_was():
    """A passage with no page on it cannot be looked up, and is passed on.

    The store is not asked anything: there is no filter that would find the
    page of a passage that does not say which page it is on.
    """
    chunk = Document(page_content="senza pagina", metadata={"source": SHEET})
    store = holding(chunk)

    found = ExpandedRetriever(
        retriever=RecordingRetriever(documents=[chunk]), store=store, k=5
    ).invoke("q")

    assert found == [chunk]
    assert store.searches == []


def test_another_page_of_the_same_number_is_not_joined_in():
    """A store that reads part of the filter still cannot widen the page."""
    kept = page(SHEET, 0, ["prima", "seconda"])
    elsewhere = page(OTHER, 0, ["altrove uno", "altrove due"])
    store = holding(*kept, *elsewhere, store=DeafStore)

    found = ExpandedRetriever(
        retriever=RecordingRetriever(documents=[kept[0]]),
        store=store,
        k=5,
    ).invoke("q")

    assert [one.page_content for one in found] == ["prima\n\nseconda"]


def test_at_most_k_pages_come_back():
    first = page(SHEET, 0, ["a1", "a2"])
    second = page(SHEET, 1, ["b1", "b2"])
    inner = RecordingRetriever(documents=[first[0], second[0]])

    found = ExpandedRetriever(
        retriever=inner, store=holding(*first, *second), k=1
    ).invoke("q")

    assert [one.page_content for one in found] == ["a1\n\na2"]


def test_a_page_below_the_cut_is_never_fetched():
    """Only the pages that are handed on are asked for.

    The second page would be dropped for being over k, so the store is not
    troubled with it.
    """
    first = page(SHEET, 0, ["a1", "a2"])
    second = page(SHEET, 1, ["b1", "b2"])
    store = holding(*first, *second)

    ExpandedRetriever(
        retriever=RecordingRetriever(documents=[first[0], second[0]]),
        store=store,
        k=1,
    ).invoke("q")

    assert len(store.searches) == 1


def test_a_handler_installed_by_the_caller_sees_the_inner_search():
    """A callback handler the caller installed hears about both runs."""
    handler = RecordingHandler()
    chunks = page(SHEET, 0, ["prima", "seconda"])

    ExpandedRetriever(
        retriever=RecordingRetriever(documents=[chunks[0]]),
        store=holding(*chunks),
        k=5,
    ).invoke("q", config={"callbacks": [handler]})

    assert [query for query, _ in handler.runs] == ["q", "q"]
    assert handler.runs[0][1] is None
    assert handler.runs[1][1] is not None


# Where the expanding retriever sits in the chain the search is built from.


def test_the_search_is_asked_for_a_pool_and_the_caller_gets_pages(monkeypatch):
    """With both passes on, the cross-encoder scores passages and pages come out.

    The pool is asked for as RERANK_CANDIDATES, every passage of it is scored
    as it is, and the pages of the RETRIEVAL_K that survive it are what the
    caller receives.
    """
    chunks = page(SHEET, 0, ["prima", "seconda"])
    aside = page(OTHER, 0, ["altrove uno", "altrove due"])
    store = holding(*chunks, *aside)
    fake = FakeReranker(
        named_scores({"seconda": 2.0, "altrove due": 1.0})
    )

    wired(monkeypatch, store, fake, rerank="on", rerank_candidates=20, retrieval_k=2)

    found = vectorstore_module.build_retriever().invoke("q")

    # The pool is asked for in full, and the passages of it are what was
    # scored — not the pages.
    assert store.searches[0] == ("q", 20, None)
    assert [text for _, text in fake.asked[0]] == [
        "altrove due",
        "altrove uno",
        "seconda",
        "prima",
    ]
    # The two pages of the two passages the reranker kept, in the order it
    # put them in. Each is fetched once.
    assert [one.page_content for one in found] == [
        "prima\n\nseconda",
        "altrove uno\n\naltrove due",
    ]
    assert len(store.searches) == 3


def test_the_scope_is_still_one_filter_over_the_sources(monkeypatch):
    """Expanding does not change how a scope reaches the store.

    The pool search is still a single $in list over the sources. The page
    fetches that follow are about one page each.
    """
    chunks = page(SHEET, 0, ["prima", "seconda"])
    store = holding(*chunks)

    wired(monkeypatch, store, rerank_candidates=4, retrieval_k=2)

    vectorstore_module.build_retriever(sources=[SHEET]).invoke("q")

    assert store.searches[0] == ("q", 2, {"source": {"$in": [SHEET]}})


def test_off_hands_back_the_search_itself(monkeypatch):
    store = FakeVectorStore()
    monkeypatch.setattr(
        vectorstore_module, "settings", make_settings(small_to_big="off")
    )
    monkeypatch.setattr(vectorstore_module, "get_vectorstore", lambda: store)

    retriever = vectorstore_module.build_retriever(k=2)

    assert not isinstance(retriever, ExpandedRetriever)


def test_a_mode_that_is_neither_is_refused_before_anything_is_built(monkeypatch):
    monkeypatch.setattr(
        vectorstore_module, "settings", make_settings(small_to_big="sections")
    )
    monkeypatch.setattr(
        vectorstore_module,
        "get_vectorstore",
        lambda: pytest.fail("a bad setting must stop before the store is opened"),
    )

    with pytest.raises(RuntimeError, match="off or page"):
        vectorstore_module.build_retriever()


def test_the_whole_document_path_is_not_expanded(monkeypatch):
    """A document read whole comes back by its chunks and is not expanded.

    The search keeps the filter and the exact number of chunks.
    """

    from app import scope as scope_module
    from app.scope import Scope, build_scoped_retriever

    chunks = page(SHEET, 0, ["prima", "seconda"])
    store = holding(*chunks)
    monkeypatch.setattr(
        vectorstore_module, "settings", make_settings(small_to_big="page")
    )
    monkeypatch.setattr(scope_module, "get_vectorstore", lambda: store)

    retriever = build_scoped_retriever(
        Scope(
            sources=(SHEET,),
            label=f"document {SHEET} (whole, 2 chunks)",
            documents=(SHEET,),
            whole_document=True,
            chunks=2,
        )
    )
    found = retriever.invoke("q")

    assert [one.page_content for one in found] == ["prima", "seconda"]
    assert store.searches == [("q", 2, {"source": SHEET})]
