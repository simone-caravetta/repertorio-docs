"""The structure command as a program on the command line.

The tests cover which documents a run takes, what it prints, and the promise
that makes it safe on a live library: no model is called and no vector store
is opened. The documents and the catalog come from a fixture built here.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from app.catalog import Catalog
from scripts.structure import build, main, parse_args
from tests.helpers import BODY, Image, Line, Table, make_structured_pdf

MANUAL = "manuals/manual.pdf"
REPORT = "reports/report.pdf"


@pytest.fixture
def argv(monkeypatch: pytest.MonkeyPatch) -> Callable[..., argparse.Namespace]:
    """A parser for a command line written out in the test."""

    def parse(*given: str) -> argparse.Namespace:
        monkeypatch.setattr(sys, "argv", ["scripts.structure", *given])
        return parse_args()

    return parse


@pytest.fixture
def library(tmp_path: Path) -> tuple[Path, Path]:
    """Two indexed documents in a catalog, one of them with a tree.

    The manual has two sections, a table and a section that runs onto the
    second page. The report has a section and a figure.
    """

    documents = tmp_path / "documents"
    make_structured_pdf(
        documents / MANUAL,
        pages=[
            [
                Line("Manuale d'uso", size=22),
                Line("1  Manutenzione", size=16),
                Line(BODY),
                Table([["Parte", "Ore"], ["Filtro", "200"]]),
            ],
            [Line(BODY)],
        ],
        toc=[[1, "Manuale d'uso", 1], [2, "Manutenzione", 1]],
    )
    make_structured_pdf(
        documents / REPORT,
        pages=[[Line("Rapporto", size=22), Line(BODY), Image(100, 60)]],
    )

    db_path = tmp_path / "catalog.sqlite3"
    catalog = Catalog(db_path)

    for path in (MANUAL, REPORT):
        catalog.add_file(path, Path(path).stem)
        catalog.record_indexed(path, file_hash="abc", page_count=1, chunk_count=1)

    return documents, db_path


def test_the_run_reads_every_indexed_document(library: tuple[Path, Path]) -> None:
    """With no path given, the whole catalog is read."""
    documents, db_path = library

    report = build(documents_dir=documents, db_path=db_path)

    assert [path for path, _ in report.built] == [MANUAL, REPORT]
    assert report.failed == []
    assert Catalog(db_path).structure(MANUAL) != []


def test_a_named_document_is_the_only_one_read(library: tuple[Path, Path]) -> None:
    """A path on the command line picks that document."""
    documents, db_path = library

    report = build([MANUAL], documents_dir=documents, db_path=db_path)

    assert [path for path, _ in report.built] == [MANUAL]
    assert Catalog(db_path).structure(REPORT) == []


def test_the_run_writes_the_structure_it_read(library: tuple[Path, Path]) -> None:
    """What the report carries is what the catalog holds afterwards."""
    documents, db_path = library

    report = build(documents_dir=documents, db_path=db_path)

    for path, nodes in report.built:
        assert Catalog(db_path).structure(path) == nodes


def test_a_dry_run_reads_the_documents_and_writes_nothing(
    library: tuple[Path, Path], capsys: pytest.CaptureFixture
) -> None:
    """A dry run prints the tree and leaves the catalog alone."""
    documents, db_path = library

    report = build(dry_run=True, documents_dir=documents, db_path=db_path)

    assert [path for path, _ in report.built] == [MANUAL, REPORT]
    assert Catalog(db_path).structure(MANUAL) == []
    assert "Manuale d'uso" in capsys.readouterr().out


def test_a_second_run_replaces_the_tree_rather_than_adding_to_it(
    library: tuple[Path, Path],
) -> None:
    """Running the command again leaves one tree per document."""
    documents, db_path = library

    build(documents_dir=documents, db_path=db_path)
    first = Catalog(db_path).structure(MANUAL)

    build(documents_dir=documents, db_path=db_path)

    assert Catalog(db_path).structure(MANUAL) == first


def test_a_document_whose_file_is_gone_is_reported_and_the_run_goes_on(
    library: tuple[Path, Path],
) -> None:
    """One unreadable document does not stop the run."""
    documents, db_path = library
    (documents / MANUAL).unlink()

    report = build(documents_dir=documents, db_path=db_path)

    assert [path for path, _ in report.built] == [REPORT]
    assert [path for path, _ in report.failed] == [MANUAL]


def test_a_document_the_catalog_does_not_know_is_refused(
    library: tuple[Path, Path],
) -> None:
    """Naming a document that was never synced is a message, not a traceback."""
    documents, db_path = library

    with pytest.raises(SystemExit, match="Not in the catalog"):
        build(["ghost.pdf"], documents_dir=documents, db_path=db_path)


def test_a_document_that_is_not_indexed_is_refused(
    library: tuple[Path, Path],
) -> None:
    """A document with no vectors in the store has nothing to hold a tree."""
    documents, db_path = library
    Catalog(db_path).trash(REPORT)

    with pytest.raises(SystemExit, match="is trashed"):
        build([REPORT], documents_dir=documents, db_path=db_path)


def test_the_run_opens_no_vector_store_and_calls_no_model(
    library: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The structure is read from the files, and nothing else is reached for."""
    documents, db_path = library

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the structure needs no store and no model")

    monkeypatch.setattr("app.vectorstore.get_vectorstore", refuse)
    monkeypatch.setattr("app.chat_model.build_chat_model", refuse)

    report = build(documents_dir=documents, db_path=db_path)

    assert len(report.built) == 2


def flattened(text: str) -> list[str]:
    """The printed lines with their padding collapsed to a single space."""
    return [" ".join(line.split()) for line in text.splitlines()]


def indent_of(text: str, needle: str) -> int:
    """How far the line holding a word is indented."""
    line = next(line for line in text.splitlines() if needle in line)

    return len(line) - len(line.lstrip())


def test_the_tree_is_printed_with_the_pages_and_what_each_section_holds(
    library: tuple[Path, Path], capsys: pytest.CaptureFixture
) -> None:
    """The tree reads as a person would write it: a line per node, indented."""
    documents, db_path = library

    build([MANUAL], documents_dir=documents, db_path=db_path)

    printed = capsys.readouterr().out

    assert "manuals/manual.pdf 2 sections, 1 table, 0 figures" in flattened(printed)
    assert "Manuale d'uso pp.1-2" in flattened(printed)
    assert "Manutenzione pp.1-2" in flattened(printed)
    assert "table Parte Ore p.1" in flattened(printed)

    # A subsection is inside its parent, and a table inside the subsection.
    assert indent_of(printed, "Manuale d'uso") == 2
    assert indent_of(printed, "Manutenzione") == 4
    assert indent_of(printed, "table  Parte Ore") == 6


def test_a_figure_is_printed_with_the_size_it_is_drawn_at(
    library: tuple[Path, Path], capsys: pytest.CaptureFixture
) -> None:
    """A figure has no text to name it, so its rectangle is printed."""
    documents, db_path = library

    build([REPORT], documents_dir=documents, db_path=db_path)

    printed = capsys.readouterr().out

    assert "reports/report.pdf 1 section, 0 tables, 1 figure" in flattened(printed)
    assert "figure p.1 100 x 60 pt" in flattened(printed)


def test_the_command_names_the_documents_it_read(
    library: tuple[Path, Path], capsys: pytest.CaptureFixture
) -> None:
    """The run ends with a line per document and a count."""
    documents, db_path = library

    build(documents_dir=documents, db_path=db_path)

    printed = capsys.readouterr().out

    assert "manuals/manual.pdf  2 sections" in printed
    assert "reports/report.pdf  1 section" in printed
    assert "2 built, 0 failed" in printed


def test_every_argument_reaches_the_run(
    argv: Callable[..., argparse.Namespace],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parsed flags and paths are handed to the run."""
    argv(MANUAL, "--dry-run", "--documents-dir", str(tmp_path))
    given: dict[str, object] = {}
    monkeypatch.setattr("scripts.structure.build", lambda *a, **k: given.update(k))

    main()

    assert given["dry_run"] is True
    assert given["documents_dir"] == tmp_path
