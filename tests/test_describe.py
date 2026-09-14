"""The describe command: what it picks, what it skips, and what a dry run costs.

What a description is, and how one is written, is tested in
`tests/test_descriptions.py`. Here it is the command around it — which documents
a run is about, and the fact that a run which has nothing to do calls no model.
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
from tests.helpers import FakeChatModel, FakeVectorStore

MANUAL = "manuals/manual.pdf"
REPORT = "reports/report.pdf"

CHUNKING = {"chunk_size": 200, "chunk_overlap": 20}


@pytest.fixture
def argv(monkeypatch: pytest.MonkeyPatch) -> Callable[..., argparse.Namespace]:
    """`parse_args`, over the arguments a shell would have handed it."""

    def parse(*given: str) -> argparse.Namespace:
        monkeypatch.setattr(sys, "argv", ["scripts.describe", *given])
        return parse_args()

    return parse


def index(documents_dir: Path, db_path: Path, store: FakeVectorStore) -> None:
    """A library already synced: both documents in the catalog, indexed."""
    sync_documents(documents_dir, db_path, store, **CHUNKING)


def run(
    db_path: Path, documents_dir: Path, model: FakeChatModel, **kwargs: object
) -> DescribeReport:
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
    """What makes this safe to run after every sync: the described cost nothing."""
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
    """Naming one is asking for it, whatever the catalog already says."""
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
    # ... and the document that was not named was not touched.
    assert Catalog(db_path).get(REPORT).description is None


def test_all_describes_the_library_again(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
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
    index(documents_dir, db_path, store)

    # No model is handed in: a dry run has to stand without one.
    report = describe([], documents_dir=documents_dir, db_path=db_path, dry_run=True)

    assert report.written == []
    assert Catalog(db_path).get(MANUAL).description is None

    out = capsys.readouterr().out
    assert f"would describe {MANUAL}" in out
    assert "no model was called" in out


def test_a_document_that_is_not_in_the_catalog_is_refused(
    documents_dir: Path, db_path: Path, store: FakeVectorStore
) -> None:
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
    index(documents_dir, db_path, store)
    (documents_dir / REPORT).unlink()
    sync_documents(documents_dir, db_path, store, **CHUNKING)

    with pytest.raises(SystemExit, match="is trashed"):
        run(db_path, documents_dir, FakeChatModel(replies=[]), paths=[REPORT])


def test_no_catalog_is_refused_before_anything_is_read(
    documents_dir: Path, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match="Run a sync first"):
        describe([], documents_dir=documents_dir, db_path=tmp_path / "catalog.sqlite3")

    assert not (tmp_path / "catalog.sqlite3").exists()


def test_one_unreadable_document_does_not_end_the_run(
    documents_dir: Path, db_path: Path, store: FakeVectorStore, capsys
) -> None:
    index(documents_dir, db_path, store)
    (documents_dir / MANUAL).unlink()

    report = run(
        db_path, documents_dir, FakeChatModel(replies=["Last year's report."])
    )

    assert [path for path, _ in report.failed] == [MANUAL]
    assert report.written == [(REPORT, "Last year's report.")]
    assert "failed  manuals/manual.pdf" in capsys.readouterr().out


def test_the_arguments(argv: Callable[..., argparse.Namespace]) -> None:
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
    with pytest.raises(SystemExit):
        argv(MANUAL, "--all")
