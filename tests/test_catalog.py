from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.catalog import STATUSES, Catalog


def test_opening_twice_keeps_the_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.sqlite3"
    Catalog(db_path).add_file("a.pdf", "a")
    Catalog(db_path)  # the DDL must be re-runnable

    assert [record.path for record in Catalog(db_path).all()] == ["a.pdf"]


def test_add_and_get(catalog: Catalog) -> None:
    catalog.add_file("manuals/manual.pdf", "manual")

    record = catalog.get("manuals/manual.pdf")

    assert record is not None
    assert (record.title, record.status) == ("manual", "queued")
    assert record.file_hash is None
    assert record.page_count is None
    assert record.chunk_count is None
    assert record.error_message is None
    assert record.trashed_at is None
    assert record.created_at == record.updated_at


def test_get_unknown_path(catalog: Catalog) -> None:
    assert catalog.get("nope.pdf") is None


def test_the_same_path_cannot_be_added_twice(catalog: Catalog) -> None:
    catalog.add_file("a.pdf", "a")

    with pytest.raises(ValueError, match="Already in the catalog"):
        catalog.add_file("a.pdf", "a")


def test_the_database_rejects_an_unknown_status(catalog: Catalog) -> None:
    catalog.add_file("a.pdf", "a")

    with pytest.raises(sqlite3.IntegrityError):
        catalog.set_status("a.pdf", "done")


def test_every_known_status_is_accepted(catalog: Catalog) -> None:
    catalog.add_file("a.pdf", "a")

    for status in STATUSES:
        catalog.set_status("a.pdf", status)

    assert catalog.get("a.pdf").status == STATUSES[-1]


def test_writing_to_an_unknown_document_raises(catalog: Catalog) -> None:
    with pytest.raises(KeyError, match="No such document"):
        catalog.record_failed("ghost.pdf", "boom")


def test_record_indexed_stores_the_counts(catalog: Catalog) -> None:
    catalog.add_file("a.pdf", "a")

    catalog.record_indexed(
        "a.pdf", file_hash="abc", page_count=3, chunk_count=7
    )

    record = catalog.get("a.pdf")
    assert record.status == "indexed"
    assert (record.file_hash, record.page_count, record.chunk_count) == (
        "abc",
        3,
        7,
    )


def test_record_indexed_clears_the_previous_failure(catalog: Catalog) -> None:
    catalog.add_file("a.pdf", "a")
    catalog.record_failed("a.pdf", "boom")
    catalog.trash("a.pdf")

    catalog.record_indexed(
        "a.pdf", file_hash="abc", page_count=1, chunk_count=1
    )

    record = catalog.get("a.pdf")
    assert record.error_message is None
    assert record.trashed_at is None


def test_record_failed_keeps_the_row(catalog: Catalog) -> None:
    catalog.add_file("a.pdf", "a")

    catalog.record_failed("a.pdf", "No text extracted")

    record = catalog.get("a.pdf")
    assert record.status == "failed"
    assert record.error_message == "No text extracted"


def test_trash_keeps_the_metadata(catalog: Catalog) -> None:
    catalog.add_file("a.pdf", "A Document")
    catalog.record_indexed(
        "a.pdf", file_hash="abc", page_count=3, chunk_count=7
    )

    catalog.trash("a.pdf")

    record = catalog.get("a.pdf")
    assert record.status == "trashed"
    assert record.trashed_at is not None
    assert record.title == "A Document"
    assert (record.file_hash, record.page_count, record.chunk_count) == (
        "abc",
        3,
        7,
    )


def test_all_is_ordered_by_path(catalog: Catalog) -> None:
    for path in ("z.pdf", "a/b.pdf", "m.pdf"):
        catalog.add_file(path, Path(path).stem)

    assert [record.path for record in catalog.all()] == [
        "a/b.pdf",
        "m.pdf",
        "z.pdf",
    ]


def test_a_read_only_catalog_does_not_create_the_file(db_path: Path) -> None:
    catalog = Catalog(db_path, create=False)

    assert catalog.all() == []
    assert catalog.get("a.pdf") is None
    assert not db_path.exists()


def test_a_read_only_catalog_refuses_to_write(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.sqlite3"
    Catalog(db_path).add_file("a.pdf", "a")

    catalog = Catalog(db_path, create=False)

    assert [record.path for record in catalog.all()] == ["a.pdf"]
    with pytest.raises(RuntimeError, match="read-only"):
        catalog.set_status("a.pdf", "indexed")
