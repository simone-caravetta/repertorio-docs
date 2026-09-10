from __future__ import annotations

from pathlib import Path

import pytest

from app.catalog import Catalog
from tests.helpers import SENTENCE, FakeVectorStore, make_pdf


@pytest.fixture
def documents_dir(tmp_path: Path) -> Path:
    """A documents folder holding two PDFs in two subfolders."""
    root = tmp_path / "documents"
    make_pdf(root / "manuals" / "manual.pdf", SENTENCE * 20)
    make_pdf(root / "reports" / "report.pdf", SENTENCE * 20)
    return root


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "catalog.sqlite3"


@pytest.fixture
def catalog(db_path: Path) -> Catalog:
    return Catalog(db_path)


@pytest.fixture
def store() -> FakeVectorStore:
    return FakeVectorStore()
