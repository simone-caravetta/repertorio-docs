"""Reranking of search results.

The setting decides whether a search is reranked. With it on, the retriever
asks the store for a wider pool and hands on the best few of what comes
back. The tests cover the setting, the line a command prints and the
retriever itself.
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
import app.vectorstore as vectorstore_module
from app.rerank import RerankedRetriever
from tests.helpers import FakeReranker, FakeVectorStore, make_settings


def named_scores(scores: dict[str, float]):
    """A scoring callable that looks a passage up by its text.

    A passage with no entry scores zero, so several of them keep the order
    the search produced.
    """
    def score(pair: tuple[str, str], at: int) -> float:
        return scores.get(pair[1], 0.0)

    return score


def named(name: str) -> Document:
    return Document(page_content=name, metadata={"source": f"{name}.pdf"})


class RecordingRetriever(BaseRetriever):
    """A retriever that returns fixed documents and keeps the queries.

    The questions it was called with are recorded, so a test can check what
    the reranked retriever passed down to it.
    """

    documents: list[Document]
    queries: list[str] = Field(default_factory=list)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        self.queries.append(query)
        return self.documents


class RecordingHandler(BaseCallbackHandler):
    """A callback handler that notes every retriever run it hears about.

    Each run is kept as the query and the id of the parent run, which tells
    a search started by the caller apart from one started inside the
    reranked retriever.
    """

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


def wired(
    monkeypatch: pytest.MonkeyPatch,
    store: FakeVectorStore,
    fake: FakeReranker | None = None,
    **settings: object,
):
    """Point the retriever at the fake store, with reranking on.

    Any setting can be passed through. A FakeReranker given here takes the
    place of the model, so no checkpoint is loaded.
    """

    monkeypatch.setattr(
        vectorstore_module, "settings", make_settings(rerank="on", **settings)
    )
    monkeypatch.setattr(vectorstore_module, "get_vectorstore", lambda: store)
    if fake is not None:
        monkeypatch.setattr(rerank_module, "get_reranker", lambda config: fake)
    return store


def holding(*names: str) -> FakeVectorStore:
    """A store holding one document per name, in the order given.

    A search of this store returns them in reverse, so the order the
    application produces is easy to tell from the store's own.
    """

    store = FakeVectorStore()
    store.added.append(([named(name) for name in names], list(names)))
    return store


# The setting itself: what it accepts, which model it names and what a
# command prints from it.


def test_reranking_is_on_or_off_as_the_setting_says():
    assert rerank_module.enabled(make_settings(rerank="on")) is True
    assert rerank_module.enabled(make_settings(rerank="off")) is False


def test_a_mode_written_with_spaces_or_in_capitals_still_reads():
    """The mode is trimmed and lowercased before it is read."""
    assert rerank_module.enabled(make_settings(rerank="  ON  ")) is True
    assert rerank_module.enabled(make_settings(rerank="Off")) is False


def test_an_unknown_mode_is_refused_and_both_are_named():
    with pytest.raises(RuntimeError, match="on or off"):
        rerank_module.enabled(make_settings(rerank="yes"))


def test_the_model_is_this_projects_until_another_is_named():
    assert (
        rerank_module.model_name(make_settings(rerank_model=""))
        == rerank_module.RERANK_MODEL
    )
    assert (
        rerank_module.model_name(make_settings(rerank_model="BAAI/bge-reranker-base"))
        == "BAAI/bge-reranker-base"
    )


def test_the_line_a_command_prints_says_which_pass_it_was():
    """The line names the model and both numbers of the pass.

    The two numbers are how many passages come back and how large the pool
    they were chosen from was.
    """

    said = rerank_module.describe_rerank(
        make_settings(rerank="on", rerank_candidates=20, retrieval_k=5)
    )

    assert "BAAI/bge-reranker-v2-m3" in said
    assert "top 5 of 20" in said


def test_the_line_for_a_run_that_does_not_rerank_says_off():
    assert rerank_module.describe_rerank(make_settings(rerank="off")) == "off"


def test_the_line_never_stops_the_run_it_describes():
    """A value that is not "on" is printed as it was written.

    The run is stopped later, in the search, where the mode has to be
    honoured.
    """

    assert rerank_module.describe_rerank(make_settings(rerank="maybe")) == "maybe"


# The search a caller gets: how many passages it asks the store for and how
# many of them it hands on.


def test_the_search_is_asked_for_the_candidates_and_not_for_k(monkeypatch):
    store = wired(
        monkeypatch,
        holding("a", "b", "c", "d"),
        FakeReranker(),
        rerank_candidates=20,
        retrieval_k=5,
    )

    vectorstore_module.build_retriever().invoke("q")

    assert store.searches == [("q", 20, None)]


def test_only_the_best_k_are_handed_on(monkeypatch):
    """Only the best k passages of the pool reach the caller.

    The store is asked once, for the whole pool.
    """

    store = wired(
        monkeypatch,
        holding("a", "b", "c", "d"),
        FakeReranker(named_scores({"b": 2.0, "d": 1.0})),
        rerank_candidates=4,
        retrieval_k=2,
    )

    found = vectorstore_module.build_retriever().invoke("q")

    assert [one.page_content for one in found] == ["b", "d"]
    assert len(store.searches) == 1


def test_a_k_larger_than_the_pool_is_asked_for_in_full(monkeypatch):
    """A k larger than the pool widens the search.

    The store is asked for k, which is more than it holds, and every
    passage comes back in the order the search returned it.
    """

    store = wired(
        monkeypatch,
        holding("a", "b", "c"),
        FakeReranker(),
        rerank_candidates=4,
        retrieval_k=10,
    )

    found = vectorstore_module.build_retriever().invoke("q")

    assert store.searches == [("q", 10, None)]
    assert [one.page_content for one in found] == ["c", "b", "a"]


def test_no_k_at_all_means_what_the_settings_say(monkeypatch):
    """A caller that passes no k gets RETRIEVAL_K passages back.

    The pool is still the size RERANK_CANDIDATES asks for.
    """

    store = wired(
        monkeypatch,
        holding("a", "b", "c"),
        FakeReranker(),
        rerank_candidates=3,
        retrieval_k=2,
    )

    found = vectorstore_module.build_retriever(k=None).invoke("q")

    assert store.searches == [("q", 3, None)]
    assert len(found) == 2


def test_the_scope_is_still_one_filter_over_the_sources(monkeypatch):
    """Reranking does not change how a scope reaches the store.

    The filter is still a single $in list over the sources.
    """

    store = wired(
        monkeypatch,
        holding("a", "b"),
        FakeReranker(),
        rerank_candidates=4,
        retrieval_k=2,
    )

    vectorstore_module.build_retriever(sources=["a.pdf"]).invoke("q")

    assert store.searches == [("q", 4, {"source": {"$in": ["a.pdf"]}})]


# Reranking off, and a mode that is neither word.


def test_off_asks_for_k_and_hands_back_the_search_itself(monkeypatch):
    store = FakeVectorStore()
    monkeypatch.setattr(
        vectorstore_module,
        "settings",
        make_settings(rerank="off", rerank_candidates=20, retrieval_k=5),
    )
    monkeypatch.setattr(vectorstore_module, "get_vectorstore", lambda: store)
    monkeypatch.setattr(
        rerank_module,
        "get_reranker",
        lambda config: pytest.fail("RERANK=off must not build a model"),
    )

    retriever = vectorstore_module.build_retriever(k=2)

    assert not isinstance(retriever, RerankedRetriever)
    assert retriever.search_kwargs["k"] == 2


def test_a_mode_that_is_neither_is_refused_before_anything_is_built(monkeypatch):
    monkeypatch.setattr(
        vectorstore_module, "settings", make_settings(rerank="maybe")
    )
    monkeypatch.setattr(
        vectorstore_module,
        "get_vectorstore",
        lambda: pytest.fail("a bad setting must stop before the store is opened"),
    )

    with pytest.raises(RuntimeError, match="on or off"):
        vectorstore_module.build_retriever()


# The reranked retriever on its own, driven by a retriever that records the
# questions it was given.


def test_the_reranker_is_asked_about_the_question_and_every_passage():
    inner = RecordingRetriever(documents=[named("a"), named("b")])
    fake = FakeReranker()

    RerankedRetriever(retriever=inner, reranker=fake, k=2).invoke("what is it?")

    assert fake.asked == [[("what is it?", "a"), ("what is it?", "b")]]
    assert inner.queries == ["what is it?"]


def test_a_tie_keeps_the_order_the_search_found_them_in():
    """Passages with equal scores keep the order the search found them in.

    Every score is zero here. The sort is stable, so what comes out
    matches what went in.
    """

    inner = RecordingRetriever(
        documents=[named("a"), named("b"), named("c"), named("d")]
    )

    found = RerankedRetriever(
        retriever=inner, reranker=FakeReranker(named_scores({})), k=4
    ).invoke("q")

    assert [one.page_content for one in found] == ["a", "b", "c", "d"]


def test_the_best_of_them_come_first():
    inner = RecordingRetriever(
        documents=[named("a"), named("b"), named("c"), named("d")]
    )

    found = RerankedRetriever(
        retriever=inner,
        reranker=FakeReranker(named_scores({"c": 9.0, "a": 8.0})),
        k=4,
    ).invoke("q")

    assert [one.page_content for one in found] == ["c", "a", "b", "d"]


def test_a_lone_passage_is_not_reranked():
    """A single passage is handed on without a call to the reranker.

    There is nothing to order it against.
    """

    fake = FakeReranker()
    inner = RecordingRetriever(documents=[named("a")])

    found = RerankedRetriever(retriever=inner, reranker=fake, k=5).invoke("q")

    assert fake.asked == []
    assert [one.page_content for one in found] == ["a"]


def test_nothing_found_is_not_a_question_for_the_reranker():
    fake = FakeReranker()
    inner = RecordingRetriever(documents=[])

    assert RerankedRetriever(retriever=inner, reranker=fake, k=5).invoke("q") == []
    assert fake.asked == []


def test_the_progress_bar_is_off():
    """The reranker is asked not to draw a progress bar.

    A command writes its own report, and a bar would cut through it.
    """

    fake = FakeReranker()
    inner = RecordingRetriever(documents=[named("a"), named("b")])

    RerankedRetriever(retriever=inner, reranker=fake, k=2).invoke("q")

    assert fake.progress_bars == [False]


def test_a_handler_installed_by_the_caller_sees_the_inner_search():
    """A callback handler the caller installed hears about both runs.

    The reranked retriever starts a run of its own and passes the callbacks
    down, so the inner search appears as a child of it.
    """

    handler = RecordingHandler()
    inner = RecordingRetriever(documents=[named("a"), named("b")])

    RerankedRetriever(retriever=inner, reranker=FakeReranker(), k=2).invoke(
        "q", config={"callbacks": [handler]}
    )

    assert [query for query, _ in handler.runs] == ["q", "q"]
    assert handler.runs[0][1] is None
    assert handler.runs[1][1] is not None


def test_the_whole_document_path_is_not_reranked(monkeypatch):
    """A whole document is read by its chunks and never reranked.

    The search keeps the filter and the exact number of chunks. The patched
    get_reranker fails the test if a model is built.
    """

    from app import scope as scope_module
    from app.scope import Scope, build_scoped_retriever

    store = FakeVectorStore()
    store.added.append(([named("a"), named("b")], ["a", "b"]))
    monkeypatch.setattr(
        vectorstore_module,
        "settings",
        make_settings(rerank="on", rerank_candidates=2, retrieval_k=1),
    )

    # The scoped retriever looks the store up in its own module, so that is
    # where the fake goes.
    monkeypatch.setattr(scope_module, "get_vectorstore", lambda: store)
    monkeypatch.setattr(
        rerank_module,
        "get_reranker",
        lambda config: pytest.fail("a whole document must not be reranked"),
    )

    retriever = build_scoped_retriever(
        Scope(
            sources=("a.pdf",),
            label="document a.pdf (whole, 1 chunk)",
            documents=("a.pdf",),
            whole_document=True,
            chunks=1,
        )
    )
    found = retriever.invoke("q")

    assert [one.page_content for one in found] == ["a"]
    assert store.searches == [("q", 1, {"source": "a.pdf"})]


def test_the_cutoff_the_eval_reads_reaches_the_search(monkeypatch):
    """The cutoff the eval ranks by is the k the search is given.

    The eval asks for NDCG_CUTOFF passages, so that number has to reach the
    store instead of the console's own k.
    """

    from app.evals import NDCG_CUTOFF
    from app.scope import Scope, build_scoped_retriever

    store = FakeVectorStore()
    monkeypatch.setattr(
        vectorstore_module,
        "settings",
        make_settings(rerank="off", retrieval_k=5),
    )
    monkeypatch.setattr(vectorstore_module, "get_vectorstore", lambda: store)

    build_scoped_retriever(
        Scope(sources=None, label="whole library", documents=("a.pdf",)),
        k=NDCG_CUTOFF,
    ).invoke("q")

    assert store.searches == [("q", NDCG_CUTOFF, None)]


def test_a_caller_with_no_opinion_is_asked_for_the_console_s_k(monkeypatch):

    from app.scope import Scope, build_scoped_retriever

    store = FakeVectorStore()
    monkeypatch.setattr(
        vectorstore_module,
        "settings",
        make_settings(rerank="off", retrieval_k=5),
    )
    monkeypatch.setattr(vectorstore_module, "get_vectorstore", lambda: store)

    build_scoped_retriever(
        Scope(sources=None, label="whole library", documents=("a.pdf",))
    ).invoke("q")

    assert store.searches == [("q", 5, None)]
