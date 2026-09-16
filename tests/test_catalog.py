from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.catalog import (
    STATUSES,
    Catalog,
    build_category_tree,
    category_from_path,
    normalise_category,
)

# The documents table as a version of this file that kept no fingerprint wrote
# it. Written out here rather than read from the module's own SCHEMA: the test
# is about a database whose columns are the older ones, and taking the columns
# from the code under test would make it agree with whatever that code says.
_OLD_SCHEMA = """
CREATE TABLE documents (
    id            INTEGER PRIMARY KEY,
    path          TEXT NOT NULL UNIQUE,
    title         TEXT NOT NULL,
    description   TEXT,
    category      TEXT,
    file_hash     TEXT,
    status        TEXT NOT NULL DEFAULT 'queued',
    page_count    INTEGER,
    chunk_count   INTEGER,
    error_message TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
    trashed_at    TEXT
) STRICT;
"""


def old_catalog(db_path: Path) -> Path:
    """A library indexed by an older version: one row, and no fingerprint."""
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_OLD_SCHEMA)
        conn.execute(
            "INSERT INTO documents (path, title, status, file_hash, chunk_count) "
            "VALUES ('a.pdf', 'a', 'indexed', 'abc', 7)"
        )

    return db_path


def columns_of(db_path: Path) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info(documents)")}


def test_an_older_catalog_still_reads(tmp_path: Path) -> None:
    """A library indexed before the fingerprint existed opens and is read.

    `CREATE TABLE IF NOT EXISTS` leaves the table it finds exactly as it was, so
    the columns added since are not there to be selected. The row has to come
    back with them absent — which is also the useful answer: a document with no
    fingerprint is one the next sync indexes again.
    """
    record = Catalog(old_catalog(tmp_path / "old.sqlite3")).get("a.pdf")

    assert (record.status, record.file_hash, record.chunk_count) == (
        "indexed",
        "abc",
        7,
    )
    assert (record.indexer, record.embedding_model) == (None, None)


def test_an_older_catalog_is_given_the_columns_it_lacks(tmp_path: Path) -> None:
    """The migration adds, and changes nothing that was already there."""
    db_path = old_catalog(tmp_path / "old.sqlite3")

    Catalog(db_path).set_status("a.pdf", "queued")

    assert {"indexer", "embedding_model"} <= columns_of(db_path)
    assert Catalog(db_path).get("a.pdf").file_hash == "abc"


def test_a_catalog_opened_to_read_is_left_alone(tmp_path: Path) -> None:
    """`create=False` promises to write nothing, and a migration is a write.

    A dry run opens the catalog this way, and it is the run most likely to meet
    a library that has not been migrated yet.
    """
    db_path = old_catalog(tmp_path / "old.sqlite3")

    assert [record.path for record in Catalog(db_path, create=False).all()] == [
        "a.pdf"
    ]

    assert "indexer" not in columns_of(db_path)


def test_record_indexed_stores_what_built_the_index(catalog: Catalog) -> None:
    """The one record of how the vectors were made: nothing else carries it."""
    catalog.add_file("a.pdf", "a")

    catalog.record_indexed(
        "a.pdf",
        file_hash="abc",
        page_count=3,
        chunk_count=7,
        indexer="pymupdf-1|1200/150",
        embedding_model="BAAI/bge-m3",
    )

    record = catalog.get("a.pdf")
    assert record.indexer == "pymupdf-1|1200/150"
    assert record.embedding_model == "BAAI/bge-m3"


def test_a_row_indexed_without_a_fingerprint_has_none(catalog: Catalog) -> None:
    """Absent is the honest record for the callers that have no run behind them."""
    catalog.add_file("a.pdf", "a")

    catalog.record_indexed("a.pdf", file_hash="abc", page_count=1, chunk_count=1)

    record = catalog.get("a.pdf")
    assert (record.indexer, record.embedding_model) == (None, None)


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


def test_delete_takes_the_metadata_with_it(catalog: Catalog) -> None:
    catalog.add_file("a.pdf", "A Document")
    catalog.record_indexed("a.pdf", file_hash="abc", page_count=3, chunk_count=7)

    catalog.delete("a.pdf")

    assert catalog.get("a.pdf") is None
    assert catalog.all() == []


def test_deleting_an_unknown_document_raises(catalog: Catalog) -> None:
    with pytest.raises(KeyError, match="No such document"):
        catalog.delete("ghost.pdf")


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


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("manuals/manual.pdf", "manuals"),
        ("manuals/ancient/a.pdf", "manuals/ancient"),
        ("a.pdf", None),
    ],
)
def test_the_category_is_the_folder_a_document_sits_in(
    path: str, expected: str | None
) -> None:
    assert category_from_path(path) == expected


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("manuals", "manuals"),
        ("manuals/ancient", "manuals/ancient"),
        ("/manuals/", "manuals"),
        ("manuals//ancient", "manuals/ancient"),
        ("  manuals  ", "manuals"),
        ("", None),
        ("/", None),
        (".", None),
        (None, None),
    ],
)
def test_a_category_is_stored_the_way_it_is_typed(
    typed: str | None, expected: str | None
) -> None:
    assert normalise_category(typed) == expected


def test_a_new_document_is_filed_under_its_folder(catalog: Catalog) -> None:
    catalog.add_file(
        "manuals/manual.pdf", "manual", category=category_from_path("manuals/manual.pdf")
    )

    assert catalog.get("manuals/manual.pdf").category == "manuals"


def test_a_document_at_the_root_has_no_category(catalog: Catalog) -> None:
    catalog.add_file("a.pdf", "a", category=category_from_path("a.pdf"))

    assert catalog.get("a.pdf").category is None


def test_re_indexing_does_not_undo_a_move(catalog: Catalog) -> None:
    """The rule that makes the catalog the authority rather than the folder.

    A document is filed by its folder once, when the row is created. Recording
    the outcome of an index run must not write the column again, or the next
    sync would put every document moved by hand back where its folder says.
    """
    catalog.add_file("manuals/manual.pdf", "manual", category="manuals")
    catalog.set_category("manuals/manual.pdf", "archive")

    catalog.record_indexed(
        "manuals/manual.pdf", file_hash="abc", page_count=1, chunk_count=3
    )

    assert catalog.get("manuals/manual.pdf").category == "archive"


def test_a_document_moves_between_categories(catalog: Catalog) -> None:
    catalog.add_file("a.pdf", "a", category=None)

    catalog.set_category("a.pdf", "manuals/ancient")

    assert catalog.get("a.pdf").category == "manuals/ancient"


def test_moving_an_unknown_document_raises(catalog: Catalog) -> None:
    with pytest.raises(KeyError, match="No such document"):
        catalog.set_category("ghost.pdf", "manuals")


def test_a_document_can_be_described(catalog: Catalog) -> None:
    """The one column nothing wrote, and the one `scripts.describe` fills."""
    catalog.add_file("a.pdf", "a")

    assert catalog.get("a.pdf").description is None

    catalog.set_description("a.pdf", "A manual about the thing.")

    assert catalog.get("a.pdf").description == "A manual about the thing."


def test_describing_an_unknown_document_raises(catalog: Catalog) -> None:
    with pytest.raises(KeyError, match="No such document"):
        catalog.set_description("ghost.pdf", "A manual about the thing.")


def test_a_description_can_be_taken_back(catalog: Catalog) -> None:
    """What the sync does to a description written from another edition."""
    catalog.add_file("a.pdf", "a")
    catalog.set_description("a.pdf", "A manual about the thing.")

    catalog.clear_description("a.pdf")

    assert catalog.get("a.pdf").description is None


def test_taking_back_the_description_of_an_unknown_document_raises(
    catalog: Catalog,
) -> None:
    with pytest.raises(KeyError, match="No such document"):
        catalog.clear_description("ghost.pdf")


def test_a_category_takes_the_documents_below_it(catalog: Catalog) -> None:
    for path in ("manuals/a.pdf", "manuals/ancient/b.pdf", "reports/c.pdf"):
        catalog.add_file(path, Path(path).stem, category=category_from_path(path))
        catalog.record_indexed(path, file_hash="x", page_count=1, chunk_count=1)

    assert catalog.sources_in_category("manuals") == [
        "manuals/a.pdf",
        "manuals/ancient/b.pdf",
    ]
    assert catalog.sources_in_category("manuals", include_descendants=False) == [
        "manuals/a.pdf"
    ]


def test_a_category_leaves_out_documents_with_no_vectors(
    catalog: Catalog,
) -> None:
    """A trashed or failed document has nothing to search, so it is not in scope."""
    for path in ("manuals/a.pdf", "manuals/b.pdf", "manuals/c.pdf"):
        catalog.add_file(path, Path(path).stem, category="manuals")

    catalog.record_indexed("manuals/a.pdf", file_hash="x", page_count=1, chunk_count=1)
    catalog.trash("manuals/b.pdf")
    catalog.record_failed("manuals/c.pdf", "No text extracted")

    assert catalog.sources_in_category("manuals") == ["manuals/a.pdf"]


def test_the_documents_with_no_category_are_only_the_root_ones(
    catalog: Catalog,
) -> None:
    for path in ("a.pdf", "manuals/b.pdf"):
        catalog.add_file(path, Path(path).stem, category=category_from_path(path))
        catalog.record_indexed(path, file_hash="x", page_count=1, chunk_count=1)

    assert catalog.sources_in_category(None) == ["a.pdf"]


def test_the_tree_nests_the_categories_it_is_given() -> None:
    tree = build_category_tree(
        [("manuals", 1), ("manuals/ancient", 1), ("manuals/modern", 2), (None, 3)]
    )

    assert len(tree) == 1
    manuals = tree[0]
    assert manuals.name == "manuals"
    assert manuals.documents == 1
    assert manuals.total == 4
    assert [(child.name, child.documents, child.total) for child in manuals.children] == [
        ("manuals/ancient", 1, 1),
        ("manuals/modern", 2, 2),
    ]


def test_a_category_holding_nothing_of_its_own_still_has_a_place() -> None:
    """Dropping it would lose the level that says where the documents sit."""
    tree = build_category_tree([("manuals/ancient/a", 2)])

    assert [branch.name for branch in tree] == ["manuals"]
    assert tree[0].documents == 0
    assert tree[0].total == 2
    assert [branch.name for branch in tree[0].children] == ["manuals/ancient"]


def test_category_counts_only_counts_the_documents_that_are_indexed(
    catalog: Catalog,
) -> None:
    catalog.add_file("manuals/a.pdf", "a", category="manuals")
    catalog.add_file("manuals/b.pdf", "b", category="manuals")
    catalog.add_file("c.pdf", "c", category=None)
    for path in ("manuals/a.pdf", "c.pdf"):
        catalog.record_indexed(path, file_hash="x", page_count=1, chunk_count=1)
    catalog.trash("manuals/b.pdf")

    # Compared as a mapping: the order the rows come back in is SQLite's.
    assert dict(catalog.category_counts()) == {"manuals": 1, None: 1}
