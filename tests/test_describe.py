"""Tests for the command that describes the documents of a library.

The command asks a chat model for a short description of each indexed document
that has none yet. These tests cover which documents it picks, what it refuses
before reading anything, and how it carries on when one document cannot be
read.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from app.catalog import Catalog
from app.descriptions import DescribeReport
from app.lifecycle import sync_documents
from scripts.describe import describe, parse_args
from tests.helpers import EMBEDDING_MODEL, FakeChatModel, FakeVectorStore

MANUAL = "manuals/manual.pdf"
REPORT = "reports/report.pdf"

# Small enough that the two sample documents are cut into several chunks.
CHUNKING = {"chunk_size": 200, "chunk_overlap": 20}


@pytest.fixture
def argv(monkeypatch: pytest.MonkeyPatch) -> Callable[..., argparse.Namespace]:
    """Return a function that parses a command line given as words.

    The words are put on `sys.argv` first, so argparse reads them the way it
    would when the command is run from a shell.
    """

    def parse(*given: str) -> argparse.Namespace:
        monkeypatch.setattr(sys, "argv", ["scripts.describe", *given])
        return parse_args()

    return parse


def index(documents_dir: Path, db_path: Path, store: FakeVectorStore) -> None:
    """Index the sample documents so the catalog has something in it."""
    sync_documents(
        documents_dir, db_path, store, embedding_model=EMBEDDING_MODEL, **CHUNKING
    )


def run(
    db_path: Path, documents_dir: Path, model: FakeChatModel, **kwargs: object
) -> DescribeReport:
    """Run the command over the library.

    The document paths travel in `kwargs` with the other options, so a test can
    name what it wants described without listing the paths of the rest.
    """
    return describe(
        kwargs.pop("paths", []),
        documents_dir=documents_dir,
        db_path=db_path,
        model=model,
        **kwargs,
    )


def test_the_documents_with_no_description_are_the_ones_described(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """Both documents are described, and the text is kept in the catalog."""
    index(documents_dir, db_path, store)
    model = FakeChatModel(replies=["A manual about the thing.", "Last year's report."])

    report = run(db_path, documents_dir, model)

    assert report.written == [
        (MANUAL, "A manual about the thing."),
        (REPORT, "Last year's report."),
    ]
    catalog = Catalog(db_path)
    assert catalog.get(MANUAL).description == "A manual about the thing."
    assert catalog.get(REPORT).description == "Last year's report."


def test_a_second_run_asks_for_nothing_and_calls_nothing(
    documents_dir: Path, db_path: Path, store: FakeVectorStore, capsys
) -> None:
    """Once every document is described, a second run reaches no model."""
    index(documents_dir, db_path, store)
    run(db_path, documents_dir, FakeChatModel(replies=["A manual about the thing.", "Last year's report."]))
    capsys.readouterr()

    model = FakeChatModel(replies=[])
    report = run(db_path, documents_dir, model)

    assert report.written == []
    assert model.prompts == []
    assert "has a description already" in capsys.readouterr().out


def test_a_named_document_is_described_again(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """A path given on the command line is described over what it had."""
    index(documents_dir, db_path, store)
    Catalog(db_path).set_description(MANUAL, "An older description.")

    report = run(
        db_path,
        documents_dir,
        FakeChatModel(replies=["A manual about the thing."]),
        paths=[MANUAL],
    )

    assert report.written == [(MANUAL, "A manual about the thing.")]
    assert Catalog(db_path).get(MANUAL).description == "A manual about the thing."

    assert Catalog(db_path).get(REPORT).description is None


def test_all_describes_the_library_again(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """--all describes every indexed document, the described ones included."""
    index(documents_dir, db_path, store)
    catalog = Catalog(db_path)
    catalog.set_description(MANUAL, "An older description.")

    report = run(
        db_path,
        documents_dir,
        FakeChatModel(replies=["A manual about the thing.", "Last year's report."]),
        every=True,
    )

    assert [path for path, _ in report.written] == [MANUAL, REPORT]


def test_a_dry_run_calls_no_model_and_writes_nothing(
    documents_dir: Path, db_path: Path, store: FakeVectorStore, capsys
) -> None:
    """A dry run prints what it would describe and leaves the catalog alone."""
    index(documents_dir, db_path, store)

    # No model is passed, so a dry run that reached for one would fail here.
    report = describe([], documents_dir=documents_dir, db_path=db_path, dry_run=True)

    assert report.written == []
    assert Catalog(db_path).get(MANUAL).description is None

    out = capsys.readouterr().out
    assert f"would describe {MANUAL}" in out
    assert "no model was called" in out


def test_a_document_that_is_not_in_the_catalog_is_refused(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """A path no document has is stopped with a message, not a traceback."""
    index(documents_dir, db_path, store)

    with pytest.raises(SystemExit, match="Not in the catalog"):
        run(
            db_path,
            documents_dir,
            FakeChatModel(replies=[]),
            paths=["manuals/ghost.pdf"],
        )


def test_a_document_with_no_vectors_is_refused(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
    """A document the sync moved to the trash is refused by name."""
    index(documents_dir, db_path, store)
    (documents_dir / REPORT).unlink()
    sync_documents(
        documents_dir, db_path, store, embedding_model=EMBEDDING_MODEL, **CHUNKING
    )

    with pytest.raises(SystemExit, match="is trashed"):
        run(db_path, documents_dir, FakeChatModel(replies=[]), paths=[REPORT])


def test_no_catalog_is_refused_before_anything_is_read(
    documents_dir: Path, tmp_path: Path
) -> None:
    """With no catalog on disk the run ends, and none is created for it."""
    with pytest.raises(SystemExit, match="Run a sync first"):
        describe([], documents_dir=documents_dir, db_path=tmp_path / "catalog.sqlite3")

    assert not (tmp_path / "catalog.sqlite3").exists()


def test_one_unreadable_document_does_not_end_the_run(
    documents_dir: Path, db_path: Path, store: FakeVectorStore, capsys
) -> None:
    """A document whose file is missing fails, and the next one is written."""
    index(documents_dir, db_path, store)
    (documents_dir / MANUAL).unlink()

    report = run(
        db_path, documents_dir, FakeChatModel(replies=["Last year's report."])
    )

    assert [path for path, _ in report.failed] == [MANUAL]
    assert report.written == [(REPORT, "Last year's report.")]
    assert "failed  manuals/manual.pdf" in capsys.readouterr().out


def test_the_arguments(argv: Callable[..., argparse.Namespace]) -> None:
    """The defaults, a list of paths, and the two flags the command takes."""
    args = argv()

    assert args.documents == []
    assert not args.every
    assert not args.dry_run

    args = argv(MANUAL, REPORT, "--dry-run")

    assert args.documents == [MANUAL, REPORT]
    assert args.dry_run

    assert argv("--all").every


def test_all_takes_no_document_paths(
    argv: Callable[..., argparse.Namespace],
) -> None:
    """--all cannot be given together with document paths."""
    with pytest.raises(SystemExit):
        argv(MANUAL, "--all")
