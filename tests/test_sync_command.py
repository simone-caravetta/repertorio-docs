"""The sync command as a program on the command line.

The tests cover the flags the script accepts, what it passes on to the run
and the report it prints at the end. The documents and the catalog come
from the fixtures, and the vector store is a fake.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from app.catalog import Catalog
from app.lifecycle import SyncReport
from scripts.sync import main, parse_args, print_report, sync
from tests.helpers import FakeChatModel, FakeVectorStore

MANUAL = "manuals/manual.pdf"
REPORT = "reports/report.pdf"


@pytest.fixture
def argv(monkeypatch: pytest.MonkeyPatch) -> Callable[..., argparse.Namespace]:
    """A parser for a command line written out in the test.

    The arguments given are placed after the program name, and the parsed
    namespace comes back.
    """

    def parse(*given: str) -> argparse.Namespace:
        monkeypatch.setattr(sys, "argv", ["scripts.sync", *given])
        return parse_args()

    return parse


@pytest.fixture
def command_store(monkeypatch: pytest.MonkeyPatch) -> FakeVectorStore:
    """Put a fake vector store in place for the whole run.

    The script looks the store up inside app.vectorstore, so that is where
    the fake is installed. It comes back so a test can read what was
    written to it.
    """

    fake = FakeVectorStore()
    monkeypatch.setattr("app.vectorstore.get_vectorstore", lambda: fake)
    return fake


def test_descriptions_are_written_unless_the_run_is_told_not_to(
    argv: Callable[..., argparse.Namespace],
) -> None:
    assert argv().descriptions is True
    assert argv("--no-descriptions").descriptions is False


def test_every_argument_reaches_the_run(
    argv: Callable[..., argparse.Namespace], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parsed flags are handed to the run as keyword arguments."""
    argv("--dry-run", "--no-descriptions")
    given: dict[str, object] = {}
    monkeypatch.setattr("scripts.sync.sync", lambda **kwargs: given.update(kwargs))

    main()

    assert given["dry_run"] is True
    assert given["descriptions"] is False


def test_the_flag_does_not_touch_anything_else(
    argv: Callable[..., argparse.Namespace], tmp_path: Path
) -> None:
    """The flag leaves every other option as it was."""
    given = argv("--no-descriptions", "--documents-dir", str(tmp_path))

    assert given.documents_dir == tmp_path
    assert given.dry_run is False


def test_the_report_says_what_was_described_and_what_could_not_be(
    capsys: pytest.CaptureFixture,
) -> None:
    print_report(
        SyncReport(
            added=[MANUAL],
            skipped=[REPORT],
            described=[MANUAL],
            description_failed=[(REPORT, "the endpoint is down")],
        )
    )

    printed = capsys.readouterr().out

    assert "describe manuals/manual.pdf" in printed
    assert "describe reports/report.pdf failed: the endpoint is down" in printed
    assert "1 added, 0 updated, 0 restored, 0 trashed, 0 failed, 1 unchanged" in printed
    assert "1 described" in printed


def test_a_report_reads_as_a_failure_of_the_description_not_of_the_document(
    capsys: pytest.CaptureFixture,
) -> None:
    """A description that could not be written is not an ingest failure.

    The document is indexed and stays that way. Its line in the report says
    the description failed, and the line for a failed document is never
    printed.
    """

    print_report(SyncReport(added=[MANUAL], description_failed=[(MANUAL, "no key")]))

    printed = capsys.readouterr().out

    assert "failed  manuals/manual.pdf" not in printed
    assert "describe manuals/manual.pdf failed: no key" in printed


def test_the_command_describes_the_documents_it_indexes(
    documents_dir: Path,
    db_path: Path,
    command_store: FakeVectorStore,
    capsys: pytest.CaptureFixture,
) -> None:
    """Every document the run indexes is described as it is written."""
    model = FakeChatModel(replies=["A manual about the thing.", "Last year's report."])

    report = sync(documents_dir=documents_dir, db_path=db_path, chat_model=model)

    assert report.described == [MANUAL, REPORT]
    assert "describe manuals/manual.pdf" in capsys.readouterr().out
    assert Catalog(db_path).get(REPORT).description == "Last year's report."


def test_a_command_told_not_to_describe_never_builds_a_model(
    documents_dir: Path,
    db_path: Path,
    command_store: FakeVectorStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse() -> None:
        raise AssertionError("a run told not to describe built a model")

    monkeypatch.setattr("scripts.sync.build_chat_model", refuse)

    report = sync(documents_dir=documents_dir, db_path=db_path, descriptions=False)

    assert report.described == []
    assert Catalog(db_path).get(MANUAL).status == "indexed"
    assert Catalog(db_path).get(MANUAL).description is None
