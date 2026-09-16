"""The eval command's own surface: what it is asked for, and what it says.

What a run measures is tested in `tests/test_evals.py`. Here it is the two things
that belong to the command rather than to the measuring: the arguments it
accepts, and the lines it prints about a run that has already happened — run
against a library indexed into a fake store, so that the whole path from a
question file to a report is walked without a network.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

import app.rerank as rerank_module
from app.config import Settings
from app.evals import NDCG_CUTOFF, EvalReport, Question, QuestionResult, run_evals
from app.lifecycle import sync_documents
from app.scope import build_scoped_retriever
from scripts.eval import evaluate, main, parse_args, print_report
from tests.helpers import (
    EMBEDDING_MODEL,
    SENTENCE,
    FakeChatModel,
    FakeReranker,
    FakeVectorStore,
    make_settings,
)

MANUAL = "manuals/manual.pdf"
REPORT = "reports/report.pdf"

CHUNKING = {"chunk_size": 200, "chunk_overlap": 20}


@pytest.fixture
def argv(monkeypatch: pytest.MonkeyPatch) -> Callable[..., argparse.Namespace]:
    """`parse_args`, over the arguments a shell would have handed it."""

    def parse(*given: str) -> argparse.Namespace:
        monkeypatch.setattr(sys, "argv", ["scripts.eval", *given])
        return parse_args()

    return parse


@pytest.fixture
def library(
    documents_dir: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> FakeVectorStore:
    """A folder indexed into a fake store, and that store handed to the command.

    A run reaches its vectors by building a store, the way a shell does, so the
    only way to measure a library without Pinecone is to answer that build with a
    fake — and the search the command runs is then the real retriever over it.

    The reranker is answered the same way, and for a second reason: this run does
    not replace the settings, so what `build_retriever` reads here is the
    machine's own `.env`, and a clone that has never reranked would download two
    gigabytes to run a test about a report. The double keeps the order the search
    found, which leaves the ranks these tests assert on the search's own.

    `app.scope` is patched as well as `app.vectorstore`, and the reason is the one
    the `configured` fixture gives for doing the same with settings: this module
    imported the name rather than the module, so a store answered on one is not
    the store seen through the other's copy. A whole-document scope builds its
    retriever from `app.scope`'s copy, and without this it read the machine's own
    store — a run of the whole-document path over the real library, quietly, in a
    test whose whole point is that it touches nothing.
    """
    store = FakeVectorStore()
    sync_documents(
        documents_dir,
        db_path,
        store,
        embedding_model=EMBEDDING_MODEL,
        **CHUNKING,  # type: ignore[arg-type]
    )
    monkeypatch.setattr("app.vectorstore.get_vectorstore", lambda: store)
    monkeypatch.setattr("app.scope.get_vectorstore", lambda: store)
    monkeypatch.setattr(rerank_module, "get_reranker", lambda config: FakeReranker())

    return store


@pytest.fixture
def question_set(tmp_path: Path) -> Path:
    path = tmp_path / "evals" / "questions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([
            {"id": "manuale", "question": SENTENCE.strip(), "document": MANUAL},
            {"id": "rapporto", "question": "What does the report say?", "document": REPORT},
        ]),
        encoding="utf-8",
    )
    return path


def test_the_answers_are_not_written_unless_they_are_asked_for(
    argv: Callable[..., argparse.Namespace],
) -> None:
    assert argv().answers is False
    assert argv().judge is False
    assert argv("--answers").answers is True
    assert argv("--judge").judge is True


def test_every_argument_reaches_the_run(
    argv: Callable[..., argparse.Namespace], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flag that is parsed and then dropped is a flag that does nothing."""
    argv(
        "--questions", str(tmp_path / "q.json"),
        "--category", "manuali",
        "--answers",
        "--judge",
    )
    given: dict[str, object] = {}
    monkeypatch.setattr("scripts.eval.evaluate", lambda **kwargs: given.update(kwargs))

    main()

    assert given["questions_path"] == tmp_path / "q.json"
    assert given["category"] == "manuali"
    assert given["answers"] is True
    assert given["judge"] is True


def test_two_scopes_at_once_are_refused(
    argv: Callable[..., argparse.Namespace],
) -> None:
    with pytest.raises(SystemExit):
        argv("--category", "manuali", "--document", MANUAL)


def test_a_run_that_writes_no_answer_never_builds_a_model(
    library: FakeVectorStore,
    documents_dir: Path,
    db_path: Path,
    question_set: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse() -> None:
        raise AssertionError("a run of retrieval alone built a model")

    monkeypatch.setattr("scripts.eval.build_chat_model", refuse)

    report = evaluate(
        questions_path=question_set, documents_dir=documents_dir, db_path=db_path
    )

    assert [result.answer for result in report.results] == [None, None]
    assert all(result.rank is not None for result in report.results)


def test_a_run_of_answers_writes_them_from_the_passages_it_found(
    library: FakeVectorStore,
    documents_dir: Path,
    db_path: Path,
    question_set: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    model = FakeChatModel(replies=["Two years.", "Nothing much."])

    report = evaluate(
        questions_path=question_set,
        documents_dir=documents_dir,
        db_path=db_path,
        answers=True,
        chat_model=model,
    )

    assert [result.answer for result in report.results] == [
        "Two years.",
        "Nothing much.",
    ]
    assert "2 searched, 2 answered" in capsys.readouterr().out


def test_judging_writes_the_answers_too(
    library: FakeVectorStore,
    documents_dir: Path,
    db_path: Path,
    question_set: Path,
) -> None:
    """A judge with nothing to read is not a measurement, so it asks for both."""
    model = FakeChatModel(replies=["Two years.", "yes", "Nothing much.", "yes"])

    report = evaluate(
        questions_path=question_set,
        documents_dir=documents_dir,
        db_path=db_path,
        judge=True,
        chat_model=model,
    )

    assert [result.faithful for result in report.results] == [True, True]


def test_a_question_the_scope_does_not_cover_is_not_asked(
    library: FakeVectorStore,
    documents_dir: Path,
    db_path: Path,
    question_set: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """One set can be asked of several scopes; the rest is not a miss."""
    report = evaluate(
        questions_path=question_set,
        documents_dir=documents_dir,
        db_path=db_path,
        document=REPORT,
    )

    assert [result.asked for result in report.results] == [False, True]
    assert "not in scope" in capsys.readouterr().out


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> Callable[..., Settings]:
    """Point a run at settings a test can predict.

    Two patches of one object, because the name was resolved in two places:
    `scripts.eval` imported it, and `build_retriever` reads `app.vectorstore`'s —
    a module attribute set on one is not seen through the other's copy. In a real
    run they are the same object, which is the state this puts back.
    """

    def point(**fields: object) -> Settings:
        same = make_settings(**fields)
        monkeypatch.setattr("scripts.eval.settings", same)
        monkeypatch.setattr("app.vectorstore.settings", same)
        return same

    return point


def test_the_header_names_the_reranker_the_run_used(
    library: FakeVectorStore,
    documents_dir: Path,
    db_path: Path,
    question_set: Path,
    capsys: pytest.CaptureFixture,
    configured: Callable[..., Settings],
) -> None:
    """A reranked run and a plain one are handed a different question by the
    store, so the line is what makes two reports comparable at all."""
    configured(rerank="on", rerank_candidates=20, retrieval_k=5)

    evaluate(
        questions_path=question_set, documents_dir=documents_dir, db_path=db_path
    )

    assert (
        "rerank    BAAI/bge-reranker-v2-m3 — top 5 of 20"
        in capsys.readouterr().out
    )


def test_the_header_says_when_nothing_was_reranked(
    library: FakeVectorStore,
    documents_dir: Path,
    db_path: Path,
    question_set: Path,
    capsys: pytest.CaptureFixture,
    configured: Callable[..., Settings],
) -> None:
    configured(rerank="off")

    evaluate(
        questions_path=question_set, documents_dir=documents_dir, db_path=db_path
    )

    assert "rerank    off" in capsys.readouterr().out


def test_a_document_read_whole_is_measured_without_a_ranking(
    library: FakeVectorStore,
    documents_dir: Path,
    db_path: Path,
    question_set: Path,
    capsys: pytest.CaptureFixture,
    configured: Callable[..., Settings],
) -> None:
    """End to end, because the flag is the command's to pass and not the report's."""
    configured(rerank="off", retrieval_k=5)

    report = evaluate(
        questions_path=question_set,
        documents_dir=documents_dir,
        db_path=db_path,
        document=MANUAL,
    )

    printed = capsys.readouterr().out

    assert "whole," in printed
    assert report.results[0].error is None
    assert report.results[0].rank == 1
    assert "MRR 1.00" in printed
    assert "nDCG" not in printed


def test_the_search_is_asked_for_the_cutoff_the_ranking_is_read_to(
    library: FakeVectorStore,
    documents_dir: Path,
    db_path: Path,
    question_set: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured: Callable[..., Settings],
) -> None:
    """Five passages and an nDCG@10 is a number about what was never retrieved.

    The other direction matters too: a console configured to show twenty is not
    narrowed to ten to suit the metric, because the counts and the MRR are still
    read over what the console would have answered from.
    """
    asked: list[int | None] = []
    build = build_scoped_retriever

    def watched(scope: object, **kwargs: object) -> object:
        asked.append(kwargs.get("k"))  # type: ignore[arg-type]
        return build(scope)  # type: ignore[arg-type]

    monkeypatch.setattr("scripts.eval.build_scoped_retriever", watched)

    for console_k in (5, NDCG_CUTOFF * 2):
        configured(rerank="off", retrieval_k=console_k)
        evaluate(
            questions_path=question_set, documents_dir=documents_dir, db_path=db_path
        )

    assert asked == [NDCG_CUTOFF, NDCG_CUTOFF * 2]


def test_what_was_found_is_read_over_what_the_console_shows(
    library: FakeVectorStore,
    documents_dir: Path,
    db_path: Path,
    question_set: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured: Callable[..., Settings],
) -> None:
    """The other half of asking for ten: the four numbers about finding something
    are read over the console's own five.

    Reading them over the wider pool the search was asked for would count a
    document at position eight as found — returned to the harness, never to the
    reader, and not in the answer the run wrote. Which is the number moving
    silently: a whole-document scope is read whole, so its scope says `None`.
    """
    read: list[int | None] = []
    real = run_evals
    called: list[object] = []

    def watched(questions: object, **kwargs: object) -> object:
        read.append(kwargs.get("read_at"))  # type: ignore[arg-type]
        called.append(questions)
        return real(questions, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("scripts.eval.run_evals", watched)

    for console_k in (5, NDCG_CUTOFF * 2):
        configured(rerank="off", retrieval_k=console_k)
        evaluate(
            questions_path=question_set, documents_dir=documents_dir, db_path=db_path
        )

    assert read == [5, NDCG_CUTOFF * 2]
    assert len(called) == 2


def test_a_document_read_whole_is_read_whole_by_the_numbers_too(
    library: FakeVectorStore,
    documents_dir: Path,
    db_path: Path,
    question_set: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured: Callable[..., Settings],
) -> None:
    """A whole-document scope has no fifth of it to stop at: its passages are the
    document, and the page the question named can be past the console's k."""
    read: list[int | None] = []
    real = run_evals

    def watched(questions: object, **kwargs: object) -> object:
        read.append(kwargs.get("read_at"))  # type: ignore[arg-type]
        return real(questions, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("scripts.eval.run_evals", watched)
    configured(rerank="off", retrieval_k=5)

    evaluate(
        questions_path=question_set,
        document=MANUAL,
        documents_dir=documents_dir,
        db_path=db_path,
    )

    assert read == [None]


def test_a_question_set_that_cannot_be_read_is_a_message(
    library: FakeVectorStore, documents_dir: Path, db_path: Path, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match="No question set"):
        evaluate(
            questions_path=tmp_path / "nothing.json",
            documents_dir=documents_dir,
            db_path=db_path,
        )


def test_a_scope_that_is_not_there_is_a_message(
    library: FakeVectorStore,
    documents_dir: Path,
    db_path: Path,
    question_set: Path,
) -> None:
    with pytest.raises(SystemExit, match="No indexed documents"):
        evaluate(
            questions_path=question_set,
            documents_dir=documents_dir,
            db_path=db_path,
            category="niente",
        )


def test_the_report_says_what_was_found(capsys: pytest.CaptureFixture) -> None:
    print_report(
        EvalReport(results=[
            QuestionResult(
                question=Question(
                    id="garanzia", question="Quanti anni?", document=MANUAL, page=4
                ),
                rank=2,
                page_rank=None,
                grades=(1, 0, 0, 0, 2, 0),
                answer="Two years, and it renews.",
                missing=["24 months"],
                faithful=False,
                unsupported=["and it renews"],
            ),
            QuestionResult(
                question=Question(id="prezzi", question="Quanto costa?", document=MANUAL),
                rank=None,
                grades=(0, 0),
            ),
            QuestionResult(
                question=Question(id="fuori", question="Chi ha vinto?"), asked=False
            ),
        ])
    )

    printed = capsys.readouterr().out

    assert "garanzia                 hit 2  p.4 miss  answer missing \"24 months\"  not faithful: and it renews" in printed
    assert "prezzi                   miss" in printed
    assert "fuori                    not in scope" in printed
    assert "3 questions, 2 asked, 1 not in scope" in printed
    assert (
        "retrieval 1/2 (50%) documents, 0/1 (0%) pages, MRR 0.25, nDCG@10 0.34"
        in printed
    )
    assert "answers   1 written, 0 as expected, 1 missing text" in printed
    assert "judge     1 read, 0 faithful, 0 not answered" in printed


def test_a_whole_document_scope_reports_no_ranking(
    capsys: pytest.CaptureFixture,
) -> None:
    """The passages come back in reading order, so there is no ranking to read.

    What the other numbers say about such a scope still stands — the document is
    either among the passages or it is not — and only the one that is about order
    is left out.
    """
    print_report(
        EvalReport(results=[
            QuestionResult(
                question=Question(
                    id="garanzia", question="Quanti anni?", document=MANUAL
                ),
                rank=1,
                grades=(1,),
            ),
        ]),
        ranked=False,
    )

    printed = capsys.readouterr().out

    assert "MRR 1.00" in printed
    assert "nDCG" not in printed


def test_a_report_with_nothing_measured_says_so_and_no_more(
    capsys: pytest.CaptureFixture,
) -> None:
    """A rate over no questions is not a number, and is left out rather than faked."""
    print_report(
        EvalReport(results=[
            QuestionResult(question=Question(id="fuori", question="Chi ha vinto?"))
        ])
    )

    printed = capsys.readouterr().out

    assert "nothing measured" in printed
    assert "retrieval" not in printed
    assert "answers" not in printed


def test_a_question_that_could_not_be_asked_is_not_a_miss(
    capsys: pytest.CaptureFixture,
) -> None:
    print_report(
        EvalReport(results=[
            QuestionResult(
                question=Question(id="uno", question="Prima?", document=MANUAL),
                error="the index is unreachable",
            )
        ])
    )

    printed = capsys.readouterr().out

    assert "uno                      failed: the index is unreachable" in printed
    assert "1 could not be run" in printed
