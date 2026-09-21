"""Tests for the command that measures a question set against the library.

The command reads a set of questions, resolves the scope they are asked of,
runs them through the console's own search, and prints what it found. These
tests cover the arguments, the two halves of a run (retrieval on its own, then
answers and the judge), and the lines the report is printed from.
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
    verdict_reply,
)

MANUAL = "manuals/manual.pdf"
REPORT = "reports/report.pdf"

CHUNKING = {"chunk_size": 200, "chunk_overlap": 20}


@pytest.fixture
def argv(monkeypatch: pytest.MonkeyPatch) -> Callable[..., argparse.Namespace]:
    """Return a function that parses a command line given as words."""

    def parse(*given: str) -> argparse.Namespace:
        monkeypatch.setattr(sys, "argv", ["scripts.eval", *given])
        return parse_args()

    return parse


@pytest.fixture
def library(
    documents_dir: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> FakeVectorStore:
    """The sample documents indexed, and the store lookups pointed at the fake.

    The command builds a store and a reranker of its own, so both are replaced
    here: the store by patching the two modules that look one up, and the
    reranker by handing back a fixed ordering. A real reranker would load a
    cross-encoder on the first run.

    The store is returned, so a test can read back what was indexed.
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
    """A set of two questions, one about each document, written to a file."""
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


def test_only_the_answers_and_the_judge_have_to_be_asked_for(
    argv: Callable[..., argparse.Namespace],
) -> None:
    """Grading is what a run does; the two that write answers are opt-in."""
    assert argv().answers is False
    assert argv().judge is False
    assert argv().grade is True
    assert argv("--answers").answers is True
    assert argv("--judge").judge is True
    assert argv("--no-grade").grade is False


def test_every_argument_reaches_the_run(
    argv: Callable[..., argparse.Namespace], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What was given on the command line arrives at the run unchanged."""

    argv(
        "--questions", str(tmp_path / "q.json"),
        "--category", "manuali",
        "--answers",
        "--judge",
        "--no-grade",
    )
    given: dict[str, object] = {}
    monkeypatch.setattr("scripts.eval.evaluate", lambda **kwargs: given.update(kwargs))

    main()

    assert given["questions_path"] == tmp_path / "q.json"
    assert given["category"] == "manuali"
    assert given["answers"] is True
    assert given["judge"] is True
    assert given["grade"] is False


def test_two_scopes_at_once_are_refused(
    argv: Callable[..., argparse.Namespace],
) -> None:
    """A category and a document cannot be asked for together."""
    with pytest.raises(SystemExit):
        argv("--category", "manuali", "--document", MANUAL)


def test_a_run_that_writes_no_answer_never_builds_a_model(
    library: FakeVectorStore,
    documents_dir: Path,
    db_path: Path,
    question_set: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run of retrieval alone reaches no model and ranks every question."""
    def refuse() -> None:
        raise AssertionError("a run of retrieval alone built a model")

    # Building a model would fail the test, so nothing may build one.

    monkeypatch.setattr("scripts.eval.build_chat_model", refuse)

    report = evaluate(
        questions_path=question_set,
        documents_dir=documents_dir,
        db_path=db_path,
        grade=False,
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
    """The answers come back on the results, and the run says how many."""
    model = FakeChatModel(replies=["Two years.", "Nothing much."])

    report = evaluate(
        questions_path=question_set,
        documents_dir=documents_dir,
        db_path=db_path,
        answers=True,
        chat_model=model,
        grade=False,
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
    """Asking for the judge asks for the answers as well."""

    model = FakeChatModel(replies=["Two years.", "yes", "Nothing much.", "yes"])

    # Each question takes two replies: the answer, then the verdict on it.

    report = evaluate(
        questions_path=question_set,
        documents_dir=documents_dir,
        db_path=db_path,
        judge=True,
        chat_model=model,
        grade=False,
    )

    assert [result.faithful for result in report.results] == [True, True]


def test_grading_reads_the_material_and_writes_no_answers(
    library: FakeVectorStore,
    documents_dir: Path,
    db_path: Path,
    question_set: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """A graded run reaches the model, and none of its replies is an answer."""
    model = FakeChatModel(replies=[
        verdict_reply(True, "The manual states it."),
        verdict_reply(True, "The report states it."),
    ])

    report = evaluate(
        questions_path=question_set,
        documents_dir=documents_dir,
        db_path=db_path,
        grade=True,
        chat_model=model,
    )

    assert [result.answer for result in report.results] == [None, None]
    assert all(result.verdict is not None for result in report.results)
    assert "2 searched, 2 graded" in capsys.readouterr().out


def test_a_question_the_library_cannot_answer_is_reported_refused(
    library: FakeVectorStore,
    documents_dir: Path,
    db_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """A question naming no document is asked, and turned away as it should be."""
    path = tmp_path / "senza" / "questions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([
            {"id": "manuale", "question": SENTENCE.strip(), "document": MANUAL},
            {"id": "fuori", "question": "Chi ha vinto il campionato?"},
        ]),
        encoding="utf-8",
    )
    model = FakeChatModel(replies=[
        verdict_reply(True, "The manual states it."),
        verdict_reply(False, "Nothing about a championship."),
    ])

    evaluate(
        questions_path=path,
        documents_dir=documents_dir,
        db_path=db_path,
        grade=True,
        chat_model=model,
    )

    printed = capsys.readouterr().out

    # The whole line, so a question with no document is not read as a miss.
    assert f"{'fuori':<24} material missing: Nothing about a championship." in printed
    assert "2 questions, 2 asked, 0 not in scope" in printed
    assert "grade     1/1 answerable accepted, 1/1 unanswerable refused" in printed


def test_a_question_the_scope_does_not_cover_is_not_asked(
    library: FakeVectorStore,
    documents_dir: Path,
    db_path: Path,
    question_set: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """A question about a document outside the scope is left out of the run."""

    report = evaluate(
        questions_path=question_set,
        documents_dir=documents_dir,
        db_path=db_path,
        document=REPORT,
        grade=False,
    )

    assert [result.asked for result in report.results] == [False, True]
    assert "not in scope" in capsys.readouterr().out


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> Callable[..., Settings]:
    """Return a function that puts the given settings in place of the defaults.

    The fields given are merged into the usual test settings, and the result is
    set on the two modules that read one: the command itself, and the store
    lookup it goes through.
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
    """The header names the reranker and the cut it was run with."""

    configured(rerank="on", rerank_candidates=20, retrieval_k=5)

    evaluate(
        questions_path=question_set, documents_dir=documents_dir, db_path=db_path,
        grade=False,
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
    """With reranking off, the header says so."""

    configured(rerank="off")

    evaluate(
        questions_path=question_set, documents_dir=documents_dir, db_path=db_path,
        grade=False,
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
    """One document to find leaves no ordering, so no nDCG is printed."""

    configured(rerank="off", retrieval_k=5)

    report = evaluate(
        questions_path=question_set,
        documents_dir=documents_dir,
        db_path=db_path,
        document=MANUAL,
        grade=False,
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
    """The search is asked for the wider of the console's k and the cutoff.

    The console shows `retrieval_k` passages and nDCG is read over the cutoff.
    Asking for the smaller of the two would report a number about passages the
    search never returned, so the wider of the two is asked for.
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
            questions_path=question_set, documents_dir=documents_dir, db_path=db_path,
            grade=False,
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
    """The ranks and the hit counts stop at the passages the console shows.

    The search is asked for more of them so that a ranking has ten to be read
    to the end of, and what was found is read over the console's own k, which
    is what the answer is written from.
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
            questions_path=question_set, documents_dir=documents_dir, db_path=db_path,
            grade=False,
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
    """A whole-document scope has no count to stop at, so nothing is cut."""

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
        grade=False,
    )

    assert read == [None]


def test_a_question_set_that_cannot_be_read_is_a_message(
    library: FakeVectorStore, documents_dir: Path, db_path: Path, tmp_path: Path
) -> None:
    """A question set that is not there is a message, not a traceback."""
    with pytest.raises(SystemExit, match="No question set"):
        evaluate(
            questions_path=tmp_path / "nothing.json",
            documents_dir=documents_dir,
            db_path=db_path,
            grade=False,
        )


def test_a_scope_that_is_not_there_is_a_message(
    library: FakeVectorStore,
    documents_dir: Path,
    db_path: Path,
    question_set: Path,
) -> None:
    """A scope with no documents in it is a message, not a traceback."""
    with pytest.raises(SystemExit, match="No indexed documents"):
        evaluate(
            questions_path=question_set,
            documents_dir=documents_dir,
            db_path=db_path,
            category="niente",
            grade=False,
        )


def test_the_report_says_what_was_found(capsys: pytest.CaptureFixture) -> None:
    """A line per question, then the numbers those questions added up to."""
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
    """A run over one document prints a rank and no nDCG.

    A whole-document scope returns the document itself, so there is no order
    of passages for a ranking to be read on.
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
    """With nothing measured, the report says so and prints no numbers."""

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
    """A question the search failed on is printed as failed, not as a miss."""
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
