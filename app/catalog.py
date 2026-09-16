from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath

# The state a document can be in. `needs_ocr` is not assigned yet: a PDF with no
# text layer is currently recorded as `failed` with "No text extracted".
STATUSES = (
    "queued",
    "indexing",
    "indexed",
    "failed",
    "needs_ocr",
    "trashed",
)

_STATUSES_SQL = ", ".join(f"'{status}'" for status in STATUSES)

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
"""

# The columns a catalog written by an earlier version of this file does not have.
# `CREATE TABLE IF NOT EXISTS` leaves an existing table exactly as it found it, so
# a library indexed before a column was added never learns about it; SQLite has no
# `ADD COLUMN IF NOT EXISTS`, and the table has to be asked what it holds.
_ADDED_COLUMNS = (
    ("indexer", "TEXT"),
    ("embedding_model", "TEXT"),
)


def _migrate(conn: sqlite3.Connection) -> None:
    """Give a catalog written before these columns existed the ones it lacks.

    Additive and nothing else: no column is dropped or rewritten, and the rows
    are left as they are, so a library indexed by an older version reads with
    the new columns absent and is indexed again on the next run. Only reached on
    a catalog this object was allowed to write — a read-only one is left exactly
    as it was found.
    """
    known = {row["name"] for row in conn.execute("PRAGMA table_info(documents)")}

    for name, kind in _ADDED_COLUMNS:
        if name not in known:
            conn.execute(f"ALTER TABLE documents ADD COLUMN {name} {kind}")


def category_from_path(path: str) -> str | None:
    """The folder a document sits in, as the category it is filed under.

    This is the only place the folder layout is read. A document is filed under
    its folder at the moment its row is created, and the catalog owns the value
    from then on: reading the folder again on every sync would undo every move
    made by hand, and would make the filesystem and the catalog two authorities
    for one fact.
    """
    parent = PurePosixPath(path).parent
    return None if parent == PurePosixPath(".") else str(parent)


def normalise_category(name: str | None) -> str | None:
    """The category as it is stored, from whatever was typed on a command line.

    None and the empty string both mean "no category", which is where a document
    at the root of the documents folder sits.
    """
    if name is None:
        return None

    cleaned = name.strip().strip("/").strip()
    if not cleaned or cleaned == ".":
        return None

    return str(PurePosixPath(cleaned))


@dataclass(frozen=True)
class DocumentRecord:
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
        # A column the row does not carry reads as absent rather than raising.
        # That is the truth about a catalog written before the column existed,
        # and for a fingerprint it is also the useful answer: a document with
        # none was not indexed the way the code in hand would index it, so the
        # next run indexes it again.
        known = row.keys()
        return cls(
            **{
                field.name: row[field.name] if field.name in known else None
                for field in fields(cls)
            }
        )


@dataclass(frozen=True)
class CategoryBranch:
    """One category, with the ones filed beneath it."""

    name: str
    documents: int
    total: int
    children: tuple[CategoryBranch, ...]


def build_category_tree(
    counts: Iterable[tuple[str | None, int]],
) -> tuple[CategoryBranch, ...]:
    """Nest the category counts into the tree their names already describe.

    A category is a folder path, so the tree is in the names themselves and
    there is no second table to keep in step. A category holding no document of
    its own still appears when something is filed below it, because dropping it
    would lose the level that says where the documents sit. `total` counts a
    category and everything under it, which is what a scope on it would search.

    Documents with no category are not a branch: they sit at the root, and the
    caller shows them apart from the tree.
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
    """The catalog of the documents in the library.

    `path` is the identity of a document: its path relative to the documents
    folder, the same value the chunks carry as `source`. It is compared as an
    exact string, so a rename that only changes the case, or the Unicode
    normalization form, is seen as a second document.

    With `create=False` the database is only read: nothing is created, and any
    write raises. Used by `sync --dry-run`.
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
        """A short-lived connection: SQLite handles are cheap, stale ones are not."""
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
        # A catalog that was never written has no rows yet. Reading it must not
        # bring the database into existence.
        if not self.db_path.exists():
            return []

        with self._connect() as conn:
            return conn.execute(sql, params).fetchall()

    def _write(
        self, sql: str, params: tuple[object, ...], path: str
    ) -> None:
        if not self.create:
            raise RuntimeError("The catalog was opened read-only")

        with self._connect() as conn:
            cursor = conn.execute(sql, params)

        # Statements that change nothing mean the row is not there: the caller
        # is working from a stale view of the catalog.
        if not cursor.rowcount:
            raise KeyError(f"No such document: {path}")

    def get(self, path: str) -> DocumentRecord | None:
        rows = self._read("SELECT * FROM documents WHERE path = ?", (path,))
        return DocumentRecord.from_row(rows[0]) if rows else None

    def all(self) -> list[DocumentRecord]:
        rows = self._read("SELECT * FROM documents ORDER BY path")
        return [DocumentRecord.from_row(row) for row in rows]

    def add_file(
        self, path: str, title: str, category: str | None = None
    ) -> None:
        """Record a new document as `queued`. Raises ValueError if already there.

        `category` is written here and nowhere else. This is the row's creation,
        so it is the one moment the folder is allowed to say where the document
        is filed; every later write leaves the column alone. See
        `category_from_path`.
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
        """File a document under a category, or at the root with None.

        A catalog write and nothing else: the vectors are not touched, and the
        folder the file sits in does not have to move, which is what keeps a
        reorganisation from costing a re-index.
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
        """Write what the document is about, as one line.

        A catalog write and nothing else: no vectors, no model, no file. The
        description is a fact about the document, kept here because the console,
        the page and the context the answer is written from all read the catalog
        and none of them can work it out on its own.
        """
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
        """Take back what was written about the document.

        A description is written from the text of one edition of a file, so the
        sync drops it when the file it describes turns out to be a different
        edition: what the description says is no longer about the document, and
        a row with nothing written about it is one the run writes again.

        Dropped here rather than replaced on the spot, because the call that
        writes the new one can fail. A description of the previous edition left
        under the title of this one is answered from as though it were about the
        document in hand, and it reads like any other.
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
        """The indexed documents filed under a category, ordered by path.

        Only `indexed` rows: a trashed or failed document has no vectors, so a
        scope naming it would search less than it says it does. `None` means the
        documents at the root, which are not a category above the others and so
        do not take descendants with them.
        """
        rows = self._read(
            "SELECT path, category FROM documents "
            "WHERE status = 'indexed' ORDER BY path"
        )

        if category is None:
            return [row["path"] for row in rows if row["category"] is None]

        # Compared in Python rather than with a LIKE: a folder named `a_b` would
        # match `axb` through the wildcard, and a scope is not a place to be
        # approximately right.
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
        """How many indexed documents each category holds, `None` for the root."""
        rows = self._read(
            "SELECT category, COUNT(*) AS n FROM documents "
            "WHERE status = 'indexed' GROUP BY category ORDER BY category"
        )
        return [(row["category"], int(row["n"])) for row in rows]

    def set_status(self, path: str, status: str) -> None:
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
        """Record a document as indexed, and what built the index.

        `indexer` and `embedding_model` are the fingerprint of the run that wrote
        the vectors: the reader and the cut, and the model that turned the text
        into numbers. Together they are what the next sync compares against the
        code it is running, because neither the file nor the vectors themselves
        say what produced them.

        They are absent by default, which is the honest record for a caller that
        has no run behind it, and which the next sync reads as a document to
        index again.
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
        """Mark a document as trashed, keeping its metadata."""
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

    def delete(self, path: str) -> None:
        """Remove the row, metadata included.

        The row is what a trashed document keeps to come back with: its title,
        its counts and its history. Once the row is gone there is nothing left
        to restore, and the document starts over as a new one.
        """
        self._write("DELETE FROM documents WHERE path = ?", (path,), path)
