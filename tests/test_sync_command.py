"""The sync command's own surface: what it is asked for, and what it says.

What a run does to a library is tested in `tests/test_sync.py`, and the warning
it can end with in `tests/test_sync_warning.py`. Here it is the two things that
belong to the command rather than to the work: the arguments it accepts, and the
lines it prints about a run that has already happened.
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
    """`parse_args`, over the arguments a shell would have handed it."""

    def parse(*given: str) -> argparse.Namespace:
        monkeypatch.setattr(sys, "argv", ["scripts.sync", *given])
        return parse_args()

    return parse


@pytest.fixture
def command_store(monkeypatch: pytest.MonkeyPatch) -> FakeVectorStore:
    """The store the command opens for itself.

    A run through `sync` reaches the vector store the way a shell does — by
    building it — so the only way to run one without Pinecone is to answer that
    build with a fake.
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
    """A flag that is parsed and then dropped is a flag that does nothing."""
    argv("--dry-run", "--no-descriptions")
    given: dict[str, object] = {}
    monkeypatch.setattr("scripts.sync.sync", lambda **kwargs: given.update(kwargs))

    main()

    assert given["dry_run"] is True
    assert given["descriptions"] is False


def test_the_flag_does_not_touch_anything_else(
    argv: Callable[..., argparse.Namespace], tmp_path: Path
) -> None:
    """A run told not to describe still watches the folder it was pointed at."""
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
    """The two are one word apart on screen and mean different things.

    A document in `failed` did not get indexed and has no vectors; a description
    that failed leaves a document that is indexed and searchable. Printed the
    same way, the second reads as the first.
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
    """The flag and the run are two places, and this is the wire between them."""
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
