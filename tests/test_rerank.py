"""What the search hands over is scored again, and the best of it is what stays.

Two halves. The wrapper is tested on its own, against an inner retriever that is
a real `BaseRetriever` — the field is typed, so pydantic checks it, which the
plain `FakeRetriever` in `tests/helpers.py` would not survive. The wiring is
tested through `build_retriever`, because that is where the two numbers meet:
what the store is asked for, and what comes back.
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
    """A `FakeReranker` score function reading the passage's own text."""

    def score(pair: tuple[str, str], at: int) -> float:
        return scores.get(pair[1], 0.0)

    return score


def named(name: str) -> Document:
    return Document(page_content=name, metadata={"source": f"{name}.pdf"})


class RecordingRetriever(BaseRetriever):
    """Hands back what it was given, and remembers the questions it was asked."""

    documents: list[Document]
    queries: list[str] = Field(default_factory=list)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        self.queries.append(query)
        return self.documents


class RecordingHandler(BaseCallbackHandler):
    """Notes every retriever run it hears about, and who the run belonged to.

    A child run carries the id of the run it was made under, a root run carries
    nothing — which is the whole of what "the callbacks were handed on" means.
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
    """`build_retriever` pointed at a fake store, a predictable settings, and a
    reranker that is not a download."""
    monkeypatch.setattr(
        vectorstore_module, "settings", make_settings(rerank="on", **settings)
    )
    monkeypatch.setattr(vectorstore_module, "get_vectorstore", lambda: store)
    if fake is not None:
        monkeypatch.setattr(rerank_module, "get_reranker", lambda config: fake)
    return store


def holding(*names: str) -> FakeVectorStore:
    """A store holding these passages, added in this order.

    The fake hands a search back in reverse order, so a test that cares about
    what the search's own ranking was reads it backwards from here.
    """
    store = FakeVectorStore()
    store.added.append(([named(name) for name in names], list(names)))
    return store


# --- the settings -----------------------------------------------------------


def test_reranking_is_on_or_off_as_the_setting_says():
    assert rerank_module.enabled(make_settings(rerank="on")) is True
    assert rerank_module.enabled(make_settings(rerank="off")) is False


def test_a_mode_written_with_spaces_or_in_capitals_still_reads():
    """`.env` files are written by hand, and `RERANK=On` is not a third mode."""
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
    """Both numbers, because the two runs ask the store different questions and
    their output is not comparable unless the line says which one produced it."""
    said = rerank_module.describe_rerank(
        make_settings(rerank="on", rerank_candidates=20, retrieval_k=5)
    )

    assert "BAAI/bge-reranker-v2-m3" in said
    assert "top 5 of 20" in said


def test_the_line_for_a_run_that_does_not_rerank_says_off():
    assert rerank_module.describe_rerank(make_settings(rerank="off")) == "off"


def test_the_line_never_stops_the_run_it_describes():
    """No mode of `RERANK` may make a command fail while printing its own header:
    a run stops where the mode has to be honoured, and says which two it wanted.
    """
    assert rerank_module.describe_rerank(make_settings(rerank="maybe")) == "maybe"


# --- what the store is asked for, and what comes back -----------------------


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
    """The search found four, the reranker put two of them in front, and the
    answer is written from those two."""
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
    """A caller that wants more than the candidate pool gets what it asked for:
    a search cannot rerank what it was never handed."""
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
    """`None` is resolved before anything compares it with the pool: `max` of a
    number and `None` is a TypeError, and a `[:None]` would quietly hand over
    every candidate there is."""
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
    """Reranking is a second pass over what the search found, not a different
    search: the filter a scope becomes is unchanged."""
    store = wired(
        monkeypatch,
        holding("a", "b"),
        FakeReranker(),
        rerank_candidates=4,
        retrieval_k=2,
    )

    vectorstore_module.build_retriever(sources=["a.pdf"]).invoke("q")

    assert store.searches == [("q", 4, {"source": {"$in": ["a.pdf"]}})]


# --- off means off ----------------------------------------------------------


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


# --- the wrapper, on its own ------------------------------------------------


def test_the_reranker_is_asked_about_the_question_and_every_passage():
    inner = RecordingRetriever(documents=[named("a"), named("b")])
    fake = FakeReranker()

    RerankedRetriever(retriever=inner, reranker=fake, k=2).invoke("what is it?")

    assert fake.asked == [[("what is it?", "a"), ("what is it?", "b")]]
    assert inner.queries == ["what is it?"]


def test_a_tie_keeps_the_order_the_search_found_them_in():
    """The sort is stable and reads only the score, so the reranker refines a
    ranking rather than replacing it with an arbitrary one.

    Every passage scores the same here, which is the case that tells the two
    apart: any order at all would satisfy a sort that broke ties itself.
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
    """An order of one cannot be improved, and the forward pass would be spent
    for nothing."""
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
    """`predict`'s default follows the logging level, so a run with logging on
    would grow a progress bar per question."""
    fake = FakeReranker()
    inner = RecordingRetriever(documents=[named("a"), named("b")])

    RerankedRetriever(retriever=inner, reranker=fake, k=2).invoke("q")

    assert fake.progress_bars == [False]


def test_a_handler_installed_by_the_caller_sees_the_inner_search():
    """The wrapper is a run of its own, and the search inside it is another. Both
    belong to whoever asked, or the search happens with nobody watching."""
    handler = RecordingHandler()
    inner = RecordingRetriever(documents=[named("a"), named("b")])

    RerankedRetriever(retriever=inner, reranker=FakeReranker(), k=2).invoke(
        "q", config={"callbacks": [handler]}
    )

    assert [query for query, _ in handler.runs] == ["q", "q"]
    assert handler.runs[0][1] is None
    assert handler.runs[1][1] is not None


def test_the_whole_document_path_is_not_reranked(monkeypatch):
    """A document read whole is read whole: ranking it would drop exactly the
    passages its scope was chosen to reach.

    `RERANK=on` is therefore not uniform across scopes, and this is where that
    shows — the small document's chunks come back all of them, in reading order.
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
    # `app.scope` imported the builder, so it holds its own name for it: the
    # patch has to land on the module that reads it, not on the one it came from.
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
