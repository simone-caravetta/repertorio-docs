"""The word said when the catalog and the store disagree about what exists.

The catalog does not record which store it was built against, so a run pointed
somewhere new finds every file unchanged, does no work, and leaves an empty
store behind. Silence there reads as a working library until a question is asked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from app.catalog import Catalog
from app.lifecycle import SyncReport
from scripts.sync import warn_if_the_store_is_empty
from tests.helpers import make_settings

MANUAL = "manuals/manual.pdf"


@dataclass
class StubStore:
    """A store that only has to answer how many vectors it holds."""

    held: int | None = None
    _collection: object = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.held is not None:
            held = self.held

            class Collection:
                def count(self) -> int:
                    return held

            self._collection = Collection()


@pytest.fixture
def indexed_catalog(db_path: Path) -> Path:
    """A catalog describing one document that was indexed somewhere."""
    catalog = Catalog(db_path)
    catalog.add_file(MANUAL, title="manual")
    catalog.record_indexed(MANUAL, file_hash="abc", page_count=2, chunk_count=9)
    return db_path


def nothing_done() -> SyncReport:
    return SyncReport(skipped=[MANUAL])


def test_an_empty_store_under_a_full_catalog_is_reported(
    indexed_catalog: Path, capsys: pytest.CaptureFixture
):
    warn_if_the_store_is_empty(nothing_done(), StubStore(held=0), indexed_catalog)

    said = capsys.readouterr().out
    assert "1 indexed document" in said
    assert "holds no vectors" in said
    # Both ways out, because either one is reasonable and neither is obvious.
    assert "VECTOR_STORE" in said
    assert "delete the catalog" in said


def test_a_store_that_holds_vectors_says_nothing(
    indexed_catalog: Path, capsys: pytest.CaptureFixture
):
    warn_if_the_store_is_empty(nothing_done(), StubStore(held=252), indexed_catalog)

    assert capsys.readouterr().out == ""


def test_a_store_that_cannot_count_says_nothing(
    indexed_catalog: Path, capsys: pytest.CaptureFixture
):
    """It cannot tell either way, and a diagnostic must never be a guess."""
    warn_if_the_store_is_empty(nothing_done(), StubStore(held=None), indexed_catalog)

    assert capsys.readouterr().out == ""


def test_a_run_that_wrote_something_says_nothing(
    indexed_catalog: Path, capsys: pytest.CaptureFixture
):
    """The failures of that run are already on screen; this is a different case."""
    report = SyncReport(added=[MANUAL])

    warn_if_the_store_is_empty(report, StubStore(held=0), indexed_catalog)

    assert capsys.readouterr().out == ""


def test_a_catalog_with_nothing_indexed_says_nothing(
    db_path: Path, capsys: pytest.CaptureFixture
):
    warn_if_the_store_is_empty(nothing_done(), StubStore(held=0), db_path)

    assert capsys.readouterr().out == ""


def test_the_message_names_the_store_that_is_configured(
    indexed_catalog: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
):
    import scripts.sync as sync_module

    monkeypatch.setattr(sync_module, "settings", make_settings(vector_store="chroma"))

    warn_if_the_store_is_empty(nothing_done(), StubStore(held=0), indexed_catalog)

    assert "chroma holds no vectors" in capsys.readouterr().out
