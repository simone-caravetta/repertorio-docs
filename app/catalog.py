"""The catalog, a SQLite table that says what has been indexed.

One row per document, holding the path, the title, the description, the category
and the status, together with the numbers the sync compares to tell whether a
file has changed since it was indexed. A second table holds the sections of each
document, and the tables and figures inside them.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath

# The statuses a document can be in. The schema refuses any other value.
STATUSES = (
    "queued",
    "indexing",
    "indexed",
    "failed",
    "needs_ocr",
    "trashed",
)

# The same list written out for the schema below.
_STATUSES_SQL = ", ".join(f"'{status}'" for status in STATUSES)

# What a node of a document's structure can be. `table` is the value the reader
# gives a piece cut out of a table, and the schema refuses any other value.
SECTION = "section"
TABLE = "table"
FIGURE = "figure"

NODE_KINDS = (SECTION, TABLE, FIGURE)

_KINDS_SQL = ", ".join(f"'{kind}'" for kind in NODE_KINDS)

# The table as it is created on an empty catalog. Columns added since the first
# version are applied afterwards by `_migrate`.
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS documents (
    id              INTEGER PRIMARY KEY,
    path            TEXT NOT NULL UNIQUE,
    title           TEXT NOT NULL,
    description     TEXT,
    category        TEXT,
    file_hash       TEXT,
    status          TEXT NOT NULL DEFAULT 'queued'
                    CHECK (status IN ({_STATUSES_SQL})),
    page_count      INTEGER,
    chunk_count     INTEGER,
    indexer         TEXT,
    embedding_model TEXT,
    error_message   TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    trashed_at      TEXT
) STRICT;

CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

-- What a document is made of: one row per section, and one for each table and
-- figure it holds. `ordinal` is the place in the document, and `parent` the
-- ordinal of the section a row sits in, so the tree is walked in Python rather
-- than by a query that has to recurse. A section and a table carry the offsets
-- of their text; a figure has none, and carries the rectangle it is drawn at
-- instead.
CREATE TABLE IF NOT EXISTS structure (
    id       INTEGER PRIMARY KEY,
    path     TEXT NOT NULL,
    ordinal  INTEGER NOT NULL,
    kind     TEXT NOT NULL CHECK (kind IN ({_KINDS_SQL})),
    title    TEXT,
    level    INTEGER,
    parent   INTEGER,
    page     INTEGER NOT NULL,
    end_page INTEGER NOT NULL,
    start    INTEGER,
    end      INTEGER,
    x0       REAL,
    y0       REAL,
    x1       REAL,
    y1       REAL,
    UNIQUE (path, ordinal)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_structure_path ON structure(path);
"""

# Columns added after the first version of the schema, with their types. A
# catalog created before those columns existed gets them added on opening.
_ADDED_COLUMNS = (
    ("indexer", "TEXT"),
    ("embedding_model", "TEXT"),
)


def _migrate(conn: sqlite3.Connection) -> None:
    """Add the columns an older catalog is missing."""
    known = {row["name"] for row in conn.execute("PRAGMA table_info(documents)")}

    for name, kind in _ADDED_COLUMNS:
        if name not in known:
            conn.execute(f"ALTER TABLE documents ADD COLUMN {name} {kind}")


def category_from_path(path: str) -> str | None:
    """The category a document path falls in, which is the folder it sits in.

    A file at the root of the documents folder has no category.
    """
    parent = PurePosixPath(path).parent
    return None if parent == PurePosixPath(".") else str(parent)


def normalise_category(name: str | None) -> str | None:
    """A category name as it is stored, or None for no category.

    Whitespace and slashes around the name are removed. An empty name and a
    single dot both mean no category.
    """
    if name is None:
        return None

    cleaned = name.strip().strip("/").strip()
    if not cleaned or cleaned == ".":
        return None

    return str(PurePosixPath(cleaned))


@dataclass(frozen=True)
class DocumentRecord:
    """One row of the catalog."""

    id: int
    path: str
    title: str
    description: str | None
    category: str | None
    file_hash: str | None
    status: str
    page_count: int | None
    chunk_count: int | None
    indexer: str | None
    embedding_model: str | None
    error_message: str | None
    created_at: str
    updated_at: str
    trashed_at: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> DocumentRecord:
        """Build a record from a row, tolerating columns the row lacks.

        A row from a catalog written by an older version has fewer columns, and
        each missing one becomes None.
        """
        known = row.keys()
        return cls(
            **{
                field.name: row[field.name] if field.name in known else None
                for field in fields(cls)
            }
        )


@dataclass(frozen=True)
class Node:
    """One thing a document holds, and where it holds it.

    `ordinal` is its place in the document and `parent` the ordinal of the
    section it sits in, or None when it sits in the document itself. `page` and
    `end_page` count from one, as a person counts pages, and `start` and `end`
    are offsets into the text of the page each of them names. A figure is a
    rectangle on a page rather than a stretch of text, so it carries `bbox` and
    an anchor offset instead of an extent.

    `title` is the heading of a section and the first row of a table, which is
    what names it in a tree; a figure has none.
    """

    ordinal: int
    kind: str
    title: str | None
    level: int | None
    parent: int | None
    page: int
    end_page: int
    start: int | None
    end: int | None
    bbox: tuple[float, float, float, float] | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Node:
        """Build a node from a row of the structure table.

        The rectangle of a figure is four columns of the row, and the node
        keeps it as the one value the reader gives it.
        """

        box = (row["x0"], row["y0"], row["x1"], row["y1"])
        return cls(
            ordinal=row["ordinal"],
            kind=row["kind"],
            title=row["title"],
            level=row["level"],
            parent=row["parent"],
            page=row["page"],
            end_page=row["end_page"],
            start=row["start"],
            end=row["end"],
            bbox=None if box[0] is None else (box[0], box[1], box[2], box[3]),
        )


@dataclass(frozen=True)
class CategoryBranch:
    """A category together with the categories below it.

    `documents` counts the documents filed directly in this category and `total`
    counts those plus everything filed below it.
    """

    name: str
    documents: int
    total: int
    children: tuple[CategoryBranch, ...]


def build_category_tree(
    counts: Iterable[tuple[str | None, int]],
) -> tuple[CategoryBranch, ...]:
    """Turn the counted categories into a tree.

    A category such as "a/b/c" implies the branches "a" and "a/b", which are
    created even when nothing is filed in them. Each branch reports how many
    documents sit in it directly and how many it holds in all.
    """
    direct: dict[str, int] = {}
    for category, count in counts:
        if category:
            direct[category] = direct.get(category, 0) + count

    names = {
        "/".join(parts[:depth])
        for name in direct
        for parts in [name.split("/")]
        for depth in range(1, len(parts) + 1)
    }

    def children_of(name: str | None) -> list[str]:
        prefix = f"{name}/" if name else ""
        return sorted(
            child
            for child in names
            if child.startswith(prefix) and "/" not in child[len(prefix) :]
        )

    def branch(name: str) -> CategoryBranch:
        children = tuple(branch(child) for child in children_of(name))
        own = direct.get(name, 0)
        return CategoryBranch(
            name=name,
            documents=own,
            total=own + sum(child.total for child in children),
            children=children,
        )

    return tuple(branch(name) for name in children_of(None))


class Catalog:
    """The table of documents that have been indexed.

    With `create` false the catalog is opened read-only, and any attempt to
    write to it raises.
    """

    def __init__(self, db_path: Path | str, *, create: bool = True) -> None:
        self.db_path = Path(db_path)
        self.create = create

        if create:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.executescript(SCHEMA)
                _migrate(conn)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A connection to the database, committed on the way out."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout = 5000")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _read(
        self, sql: str, params: tuple[object, ...] = ()
    ) -> list[sqlite3.Row]:
        """Rows from a read query, or none when the file does not exist yet."""
        if not self.db_path.exists():
            return []

        with self._connect() as conn:
            return conn.execute(sql, params).fetchall()

    def _write(
        self, sql: str, params: tuple[object, ...], path: str
    ) -> None:
        """Run a write query and fail when it matched no row."""
        if not self.create:
            raise RuntimeError("The catalog was opened read-only")

        with self._connect() as conn:
            cursor = conn.execute(sql, params)

        # Every write here addresses one document by its path. A query that
        # matched nothing means the catalog does not know that document.
        if not cursor.rowcount:
            raise KeyError(f"No such document: {path}")

    def get(self, path: str) -> DocumentRecord | None:
        """The record for one document, or None when it is not in the catalog."""
        rows = self._read("SELECT * FROM documents WHERE path = ?", (path,))
        return DocumentRecord.from_row(rows[0]) if rows else None

    def all(self) -> list[DocumentRecord]:
        """Every record, ordered by path."""
        rows = self._read("SELECT * FROM documents ORDER BY path")
        return [DocumentRecord.from_row(row) for row in rows]

    def add_file(
        self, path: str, title: str, category: str | None = None
    ) -> None:
        """Add a document to the catalog with the status of queued.

        Raises ValueError when the path is already in the catalog.
        """
        try:
            self._write(
                "INSERT INTO documents (path, title, category) VALUES (?, ?, ?)",
                (path, title, category),
                path,
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Already in the catalog: {path}") from exc

    def set_category(self, path: str, category: str | None) -> None:
        """File a document under a category, or under none.

        This writes to the catalog alone and the file itself does not have to
        move.
        """
        self._write(
            """
            UPDATE documents
               SET category = ?, updated_at = datetime('now')
             WHERE path = ?
            """,
            (category, path),
            path,
        )

    def set_description(self, path: str, description: str) -> None:
        """Store the description written for a document."""
        self._write(
            """
            UPDATE documents
               SET description = ?, updated_at = datetime('now')
             WHERE path = ?
            """,
            (description, path),
            path,
        )

    def clear_description(self, path: str) -> None:
        """Forget the description of a document.

        The sync calls this when the file the description was written from has
        changed, so that a new one is written on the next run.
        """
        self._write(
            """
            UPDATE documents
               SET description = NULL, updated_at = datetime('now')
             WHERE path = ?
            """,
            (path,),
            path,
        )

    def sources_in_category(
        self, category: str | None, *, include_descendants: bool = True
    ) -> list[str]:
        """The indexed documents in a category.

        With `include_descendants` the categories below it are included too, so
        that a question asked of a category covers everything filed under it. A
        category of None means the documents filed in no category.
        """
        rows = self._read(
            "SELECT path, category FROM documents "
            "WHERE status = 'indexed' ORDER BY path"
        )

        if category is None:
            return [row["path"] for row in rows if row["category"] is None]

        # A category is also a prefix, which is what makes "manuali/vecchi" a
        # descendant of "manuali".
        prefix = f"{category}/"
        return [
            row["path"]
            for row in rows
            if row["category"] == category
            or (
                include_descendants
                and (row["category"] or "").startswith(prefix)
            )
        ]

    def category_counts(self) -> list[tuple[str | None, int]]:
        """How many indexed documents each category holds."""
        rows = self._read(
            "SELECT category, COUNT(*) AS n FROM documents "
            "WHERE status = 'indexed' GROUP BY category ORDER BY category"
        )
        return [(row["category"], int(row["n"])) for row in rows]

    def set_status(self, path: str, status: str) -> None:
        """Set the status of a document."""
        self._write(
            """
            UPDATE documents
               SET status = ?, updated_at = datetime('now')
             WHERE path = ?
            """,
            (status, path),
            path,
        )

    def record_indexed(
        self,
        path: str,
        *,
        file_hash: str,
        page_count: int,
        chunk_count: int,
        indexer: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        """Mark a document as indexed and store what indexing produced.

        The error message and the date it was trashed are cleared, so a document
        that is indexed again starts from a clean row.
        """
        self._write(
            """
            UPDATE documents
               SET status = 'indexed', file_hash = ?, page_count = ?,
                   chunk_count = ?, indexer = ?, embedding_model = ?,
                   error_message = NULL, trashed_at = NULL,
                   updated_at = datetime('now')
             WHERE path = ?
            """,
            (file_hash, page_count, chunk_count, indexer, embedding_model, path),
            path,
        )

    def record_failed(self, path: str, error: str) -> None:
        """Mark a document as failed and keep the message that says why."""
        self._write(
            """
            UPDATE documents
               SET status = 'failed', error_message = ?,
                   updated_at = datetime('now')
             WHERE path = ?
            """,
            (error, path),
            path,
        )

    def trash(self, path: str) -> None:
        """Move a document to the trash and record when.

        The row stays in the catalog, so the document comes back by putting the
        file where it was and running the sync again.
        """
        self._write(
            """
            UPDATE documents
               SET status = 'trashed', trashed_at = datetime('now'),
                   updated_at = datetime('now')
             WHERE path = ?
            """,
            (path,),
            path,
        )

    def set_structure(self, path: str, nodes: Iterable[Node]) -> None:
        """Replace what the catalog holds for one document's structure.

        The tree is the document's own, so its rows are removed and the new
        ones written in one transaction: a document read again never keeps half
        of an older tree, and a run that fails part way leaves it as it was.
        """
        if not self.create:
            raise RuntimeError("The catalog was opened read-only")

        rows = [
            (
                path,
                node.ordinal,
                node.kind,
                node.title,
                node.level,
                node.parent,
                node.page,
                node.end_page,
                node.start,
                node.end,
                *(node.bbox or (None, None, None, None)),
            )
            for node in nodes
        ]

        with self._connect() as conn:
            found = conn.execute(
                "SELECT 1 FROM documents WHERE path = ?", (path,)
            ).fetchone()

            if found is None:
                raise KeyError(f"No such document: {path}")

            conn.execute("DELETE FROM structure WHERE path = ?", (path,))
            conn.executemany(
                """
                INSERT INTO structure (
                    path, ordinal, kind, title, level, parent, page, end_page,
                    start, end, x0, y0, x1, y1
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def structure(self, path: str) -> list[Node]:
        """The structure of one document, in the order a reader meets it.

        A catalog written before this table existed answers with nothing rather
        than raising, so a reader can be pointed at any catalog.
        """
        try:
            rows = self._read(
                "SELECT * FROM structure WHERE path = ? ORDER BY ordinal", (path,)
            )
        except sqlite3.OperationalError as error:
            if "no such table" not in str(error):
                raise

            return []

        return [Node.from_row(row) for row in rows]

    def delete(self, path: str) -> None:
        """Remove a document from the catalog for good.

        The structure goes with it: those rows are keyed by the path and would
        outlive the document otherwise. They are removed here rather than by a
        cascade, because SQLite leaves foreign keys off unless a connection
        asks for them.
        """
        self._write("DELETE FROM documents WHERE path = ?", (path,), path)

        with self._connect() as conn:
            conn.execute("DELETE FROM structure WHERE path = ?", (path,))
