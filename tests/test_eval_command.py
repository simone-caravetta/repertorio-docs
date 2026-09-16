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

from app.evals import EvalReport, Question, QuestionResult
from app.lifecycle import sync_documents
from scripts.eval import evaluate, main, parse_args, print_report
from tests.helpers import EMBEDDING_MODEL, SENTENCE, FakeChatModel, FakeVectorStore

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
                answer="Two years, and it renews.",
                missing=["24 months"],
                faithful=False,
                unsupported=["and it renews"],
            ),
            QuestionResult(
                question=Question(id="prezzi", question="Quanto costa?", document=MANUAL),
                rank=None,
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
    assert "retrieval 1/2 (50%) documents, 0/1 (0%) pages, MRR 0.25" in printed
    assert "answers   1 written, 0 as expected, 1 missing text" in printed
    assert "judge     1 read, 0 faithful, 0 not answered" in printed


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
