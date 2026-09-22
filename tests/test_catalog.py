"""Tests for the SQLite catalog that records every document.

The catalog keeps one row per document, holding its path, title, category,
status and the counts from its last indexing run. These tests cover reading
and writing those rows, the upgrade of a database that was written before
two of the columns existed, and the helpers that turn a path into a category.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from app.catalog import (
    FIGURE,
    SECTION,
    STATUSES,
    TABLE,
    Catalog,
    Node,
    build_category_tree,
    category_from_path,
    normalise_category,
)

# A documents table as it was written before the columns recording the
# indexer and the embedding model were added.
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
    """Write the older schema to a database and add one indexed row."""
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_OLD_SCHEMA)
        conn.execute(
            "INSERT INTO documents (path, title, status, file_hash, chunk_count) "
            "VALUES ('a.pdf', 'a', 'indexed', 'abc', 7)"
        )

    return db_path


def columns_of(db_path: Path) -> set[str]:
    """Return the names of the columns the documents table has."""
    with sqlite3.connect(db_path) as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info(documents)")}


def test_an_older_catalog_still_reads(tmp_path: Path) -> None:
    """A row written before the fingerprint columns existed still reads.

    Both of those columns come back as None for it.
    """
    record = Catalog(old_catalog(tmp_path / "old.sqlite3")).get("a.pdf")

    assert (record.status, record.file_hash, record.chunk_count) == (
        "indexed",
        "abc",
        7,
    )
    assert (record.indexer, record.embedding_model) == (None, None)


def test_an_older_catalog_is_given_the_columns_it_lacks(tmp_path: Path) -> None:
    """Opening the database for writing adds what the table is missing.

    The row that was already there keeps its values.
    """
    db_path = old_catalog(tmp_path / "old.sqlite3")

    Catalog(db_path).set_status("a.pdf", "queued")

    assert {"indexer", "embedding_model"} <= columns_of(db_path)
    assert Catalog(db_path).get("a.pdf").file_hash == "abc"


def test_a_catalog_opened_to_read_is_left_alone(tmp_path: Path) -> None:
    """A catalog opened with create=False reads the rows as they are.

    The file on disk keeps the schema it had.
    """
    db_path = old_catalog(tmp_path / "old.sqlite3")

    assert [record.path for record in Catalog(db_path, create=False).all()] == [
        "a.pdf"
    ]

    assert "indexer" not in columns_of(db_path)


def test_record_indexed_stores_what_built_the_index(catalog: Catalog) -> None:
    """The indexer and the embedding model are written onto the row."""
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
    """Those two arguments are optional and stay unset when left out."""
    catalog.add_file("a.pdf", "a")

    catalog.record_indexed("a.pdf", file_hash="abc", page_count=1, chunk_count=1)

    record = catalog.get("a.pdf")
    assert (record.indexer, record.embedding_model) == (None, None)


def test_opening_twice_keeps_the_schema(tmp_path: Path) -> None:
    """A second catalog over the same file finds the rows already there."""
    db_path = tmp_path / "catalog.sqlite3"
    Catalog(db_path).add_file("a.pdf", "a")
    Catalog(db_path)

    assert [record.path for record in Catalog(db_path).all()] == ["a.pdf"]


def test_add_and_get(catalog: Catalog) -> None:
    """A new row starts queued, with nothing counted yet."""
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
    """A path the catalog does not hold reads as None."""
    assert catalog.get("nope.pdf") is None


def test_the_same_path_cannot_be_added_twice(catalog: Catalog) -> None:
    """Adding the same path a second time raises."""
    catalog.add_file("a.pdf", "a")

    with pytest.raises(ValueError, match="Already in the catalog"):
        catalog.add_file("a.pdf", "a")


def test_the_database_rejects_an_unknown_status(catalog: Catalog) -> None:
    """The status column only accepts the values in STATUSES."""
    catalog.add_file("a.pdf", "a")

    with pytest.raises(sqlite3.IntegrityError):
        catalog.set_status("a.pdf", "done")


def test_every_known_status_is_accepted(catalog: Catalog) -> None:
    """Each status in STATUSES can be written to a row."""
    catalog.add_file("a.pdf", "a")

    for status in STATUSES:
        catalog.set_status("a.pdf", status)

    assert catalog.get("a.pdf").status == STATUSES[-1]


def test_writing_to_an_unknown_document_raises(catalog: Catalog) -> None:
    """Recording a failure for a path that is not there raises."""
    with pytest.raises(KeyError, match="No such document"):
        catalog.record_failed("ghost.pdf", "boom")


def test_record_indexed_stores_the_counts(catalog: Catalog) -> None:
    """The hash and the page and chunk counts land on the row."""
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
    """A document that failed and was trashed comes back clean once indexed."""
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
    """A failure sets the status and keeps the message for the console."""
    catalog.add_file("a.pdf", "a")

    catalog.record_failed("a.pdf", "No text extracted")

    record = catalog.get("a.pdf")
    assert record.status == "failed"
    assert record.error_message == "No text extracted"


def test_trash_keeps_the_metadata(catalog: Catalog) -> None:
    """A trashed document keeps its title and counts, and gains a date."""
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
    """A deleted document leaves no row behind."""
    catalog.add_file("a.pdf", "A Document")
    catalog.record_indexed("a.pdf", file_hash="abc", page_count=3, chunk_count=7)

    catalog.delete("a.pdf")

    assert catalog.get("a.pdf") is None
    assert catalog.all() == []


def test_deleting_an_unknown_document_raises(catalog: Catalog) -> None:
    """Deleting a path that is not in the catalog raises."""
    with pytest.raises(KeyError, match="No such document"):
        catalog.delete("ghost.pdf")


def tables_of(db_path: Path) -> set[str]:
    """Return the names of the tables the database holds."""
    with sqlite3.connect(db_path) as conn:
        return {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }


def sample_nodes() -> list[Node]:
    """A small tree: a section with a table and a figure inside it."""
    return [
        Node(
            ordinal=0,
            kind=SECTION,
            title="1  Care",
            level=1,
            parent=None,
            page=1,
            end_page=2,
            start=0,
            end=40,
        ),
        Node(
            ordinal=1,
            kind=TABLE,
            title=None,
            level=None,
            parent=0,
            page=1,
            end_page=1,
            start=10,
            end=20,
        ),
        Node(
            ordinal=2,
            kind=FIGURE,
            title=None,
            level=None,
            parent=0,
            page=2,
            end_page=2,
            start=30,
            end=30,
            bbox=(72.0, 140.0, 192.0, 220.0),
        ),
    ]


def test_the_structure_table_comes_with_the_catalog(tmp_path: Path) -> None:
    """A catalog created now holds the documents and their structure."""
    db_path = tmp_path / "catalog.sqlite3"

    Catalog(db_path)

    assert {"documents", "structure"} <= tables_of(db_path)


def test_an_older_catalog_is_given_the_structure_table(tmp_path: Path) -> None:
    """Opening an older database for writing adds the table it lacks."""
    db_path = old_catalog(tmp_path / "old.sqlite3")

    assert "structure" not in tables_of(db_path)

    catalog = Catalog(db_path)

    assert "structure" in tables_of(db_path)
    assert catalog.get("a.pdf") is not None


def test_a_read_only_catalog_answers_with_no_structure(tmp_path: Path) -> None:
    """A database written before the table existed reads as having none."""
    catalog = Catalog(old_catalog(tmp_path / "old.sqlite3"), create=False)

    assert catalog.structure("a.pdf") == []


def test_the_structure_of_a_document_is_written_and_read_back(
    catalog: Catalog,
) -> None:
    """Every field of a node survives the round trip to the database."""
    catalog.add_file("a.pdf", "a")

    catalog.set_structure("a.pdf", sample_nodes())

    assert catalog.structure("a.pdf") == sample_nodes()


def test_the_structure_comes_back_in_the_order_it_was_written(catalog: Catalog) -> None:
    """Nodes read back are in document order, whatever order they arrive in."""
    catalog.add_file("a.pdf", "a")

    catalog.set_structure("a.pdf", list(reversed(sample_nodes())))

    assert [node.ordinal for node in catalog.structure("a.pdf")] == [0, 1, 2]


def test_writing_the_structure_again_replaces_it(catalog: Catalog) -> None:
    """A document read a second time keeps no part of its older tree."""
    catalog.add_file("a.pdf", "a")
    catalog.set_structure("a.pdf", sample_nodes())

    catalog.set_structure("a.pdf", sample_nodes()[:1])

    assert catalog.structure("a.pdf") == sample_nodes()[:1]


def test_the_structure_of_an_unknown_document_raises(catalog: Catalog) -> None:
    """Writing the structure of a document the catalog does not know raises."""
    with pytest.raises(KeyError, match="No such document"):
        catalog.set_structure("ghost.pdf", sample_nodes())


def test_writing_the_structure_to_a_read_only_catalog_raises(tmp_path: Path) -> None:
    """A read-only catalog serves the structure and rejects a new one."""
    db_path = tmp_path / "catalog.sqlite3"
    Catalog(db_path).add_file("a.pdf", "a")

    catalog = Catalog(db_path, create=False)

    with pytest.raises(RuntimeError, match="read-only"):
        catalog.set_structure("a.pdf", sample_nodes())


def test_a_structure_that_cannot_be_written_leaves_the_previous_one(
    catalog: Catalog,
) -> None:
    """A write that fails part way leaves the tree the document had."""
    catalog.add_file("a.pdf", "a")
    catalog.set_structure("a.pdf", sample_nodes())

    unknown = replace(sample_nodes()[0], kind="drawing")

    with pytest.raises(sqlite3.IntegrityError):
        catalog.set_structure("a.pdf", [unknown])

    assert catalog.structure("a.pdf") == sample_nodes()


def test_deleting_a_document_takes_its_structure(catalog: Catalog) -> None:
    """A document removed for good leaves no node behind."""
    catalog.add_file("a.pdf", "a")
    catalog.set_structure("a.pdf", sample_nodes())

    catalog.delete("a.pdf")

    assert catalog.structure("a.pdf") == []


def test_all_is_ordered_by_path(catalog: Catalog) -> None:
    """all() returns every row, sorted by path."""
    for path in ("z.pdf", "a/b.pdf", "m.pdf"):
        catalog.add_file(path, Path(path).stem)

    assert [record.path for record in catalog.all()] == [
        "a/b.pdf",
        "m.pdf",
        "z.pdf",
    ]


def test_a_read_only_catalog_does_not_create_the_file(db_path: Path) -> None:
    """Opening a database that is not there yet writes nothing."""
    catalog = Catalog(db_path, create=False)

    assert catalog.all() == []
    assert catalog.get("a.pdf") is None
    assert not db_path.exists()


def test_a_read_only_catalog_refuses_to_write(tmp_path: Path) -> None:
    """A read-only catalog serves reads and rejects writes."""
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
    """The folder a document is filed in is its category.

    A file at the root of the library has no category at all.
    """
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
    """A typed category is tidied up before it is stored.

    Surrounding spaces and stray slashes go, and an empty category becomes
    None.
    """
    assert normalise_category(typed) == expected


def test_a_new_document_is_filed_under_its_folder(catalog: Catalog) -> None:
    """A new document is filed under the category taken from its path."""
    catalog.add_file(
        "manuals/manual.pdf", "manual", category=category_from_path("manuals/manual.pdf")
    )

    assert catalog.get("manuals/manual.pdf").category == "manuals"


def test_a_document_at_the_root_has_no_category(catalog: Catalog) -> None:
    """A file at the root of the library is stored without a category."""
    catalog.add_file("a.pdf", "a", category=category_from_path("a.pdf"))

    assert catalog.get("a.pdf").category is None


def test_re_indexing_does_not_undo_a_move(catalog: Catalog) -> None:
    """Re-indexing a document keeps the category it was moved to.

    The category recorded at indexing time is the folder the file sits in,
    so a document that was moved afterwards would otherwise be dragged back.
    """
    catalog.add_file("manuals/manual.pdf", "manual", category="manuals")
    catalog.set_category("manuals/manual.pdf", "archive")

    catalog.record_indexed(
        "manuals/manual.pdf", file_hash="abc", page_count=1, chunk_count=3
    )

    assert catalog.get("manuals/manual.pdf").category == "archive"


def test_a_document_moves_between_categories(catalog: Catalog) -> None:
    """Moving a document replaces the category it had."""
    catalog.add_file("a.pdf", "a", category=None)

    catalog.set_category("a.pdf", "manuals/ancient")

    assert catalog.get("a.pdf").category == "manuals/ancient"


def test_moving_an_unknown_document_raises(catalog: Catalog) -> None:
    """Moving a path the catalog does not hold raises."""
    with pytest.raises(KeyError, match="No such document"):
        catalog.set_category("ghost.pdf", "manuals")


def test_a_document_can_be_described(catalog: Catalog) -> None:
    """A description can be written to a document and read back."""
    catalog.add_file("a.pdf", "a")

    assert catalog.get("a.pdf").description is None

    catalog.set_description("a.pdf", "A manual about the thing.")

    assert catalog.get("a.pdf").description == "A manual about the thing."


def test_describing_an_unknown_document_raises(catalog: Catalog) -> None:
    """Describing a path the catalog does not hold raises."""
    with pytest.raises(KeyError, match="No such document"):
        catalog.set_description("ghost.pdf", "A manual about the thing.")


def test_a_description_can_be_taken_back(catalog: Catalog) -> None:
    """Clearing the description leaves the document without one."""
    catalog.add_file("a.pdf", "a")
    catalog.set_description("a.pdf", "A manual about the thing.")

    catalog.clear_description("a.pdf")

    assert catalog.get("a.pdf").description is None


def test_taking_back_the_description_of_an_unknown_document_raises(
    catalog: Catalog,
) -> None:
    """Clearing a description on an unknown path raises."""
    with pytest.raises(KeyError, match="No such document"):
        catalog.clear_description("ghost.pdf")


def test_a_category_takes_the_documents_below_it(catalog: Catalog) -> None:
    """A category returns the documents below it, at any depth.

    Passing include_descendants=False keeps only the documents filed
    directly in it.
    """
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
    """Only indexed documents are returned, so a search has vectors to use.

    A trashed or failed document still has a row, but nothing to search.
    """
    for path in ("manuals/a.pdf", "manuals/b.pdf", "manuals/c.pdf"):
        catalog.add_file(path, Path(path).stem, category="manuals")

    catalog.record_indexed("manuals/a.pdf", file_hash="x", page_count=1, chunk_count=1)
    catalog.trash("manuals/b.pdf")
    catalog.record_failed("manuals/c.pdf", "No text extracted")

    assert catalog.sources_in_category("manuals") == ["manuals/a.pdf"]


def test_the_documents_with_no_category_are_only_the_root_ones(
    catalog: Catalog,
) -> None:
    """Asking for the None category returns the files at the root."""
    for path in ("a.pdf", "manuals/b.pdf"):
        catalog.add_file(path, Path(path).stem, category=category_from_path(path))
        catalog.record_indexed(path, file_hash="x", page_count=1, chunk_count=1)

    assert catalog.sources_in_category(None) == ["a.pdf"]


def test_the_tree_nests_the_categories_it_is_given() -> None:
    """A category that has children becomes a branch in the tree.

    Each branch counts the documents of its own and the total below it.
    """
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
    """A parent with no document of its own is still shown in the tree.

    It counts zero documents of its own and the total below it.
    """
    tree = build_category_tree([("manuals/ancient/a", 2)])

    assert [branch.name for branch in tree] == ["manuals"]
    assert tree[0].documents == 0
    assert tree[0].total == 2
    assert [branch.name for branch in tree[0].children] == ["manuals/ancient"]


def test_category_counts_only_counts_the_documents_that_are_indexed(
    catalog: Catalog,
) -> None:
    """A trashed document is left out of the per-category counts."""
    catalog.add_file("manuals/a.pdf", "a", category="manuals")
    catalog.add_file("manuals/b.pdf", "b", category="manuals")
    catalog.add_file("c.pdf", "c", category=None)
    for path in ("manuals/a.pdf", "c.pdf"):
        catalog.record_indexed(path, file_hash="x", page_count=1, chunk_count=1)
    catalog.trash("manuals/b.pdf")


    assert dict(catalog.category_counts()) == {"manuals": 1, None: 1}
