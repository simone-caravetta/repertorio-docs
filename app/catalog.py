from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path

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
    id            INTEGER PRIMARY KEY,
    path          TEXT NOT NULL UNIQUE,
    title         TEXT NOT NULL,
    description   TEXT,
    category      TEXT,
    file_hash     TEXT,
    status        TEXT NOT NULL DEFAULT 'queued'
                  CHECK (status IN ({_STATUSES_SQL})),
    page_count    INTEGER,
    chunk_count   INTEGER,
    error_message TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
    trashed_at    TEXT
) STRICT;

CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
"""


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
    error_message: str | None
    created_at: str
    updated_at: str
    trashed_at: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> DocumentRecord:
        return cls(**{field.name: row[field.name] for field in fields(cls)})


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

    def add_file(self, path: str, title: str) -> None:
        """Record a new document as `queued`. Raises ValueError if already there."""
        try:
            self._write(
                "INSERT INTO documents (path, title) VALUES (?, ?)",
                (path, title),
                path,
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Already in the catalog: {path}") from exc

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
    ) -> None:
        self._write(
            """
            UPDATE documents
               SET status = 'indexed', file_hash = ?, page_count = ?,
                   chunk_count = ?, error_message = NULL, trashed_at = NULL,
                   updated_at = datetime('now')
             WHERE path = ?
            """,
            (file_hash, page_count, chunk_count, path),
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
