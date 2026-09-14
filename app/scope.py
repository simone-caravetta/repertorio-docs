"""What a question is asked of: the whole library, a category, or some documents.

The console can be pointed at less than everything, and this is the one place
that turns that into the documents a search is restricted to. Everything here is
a catalog read: resolving a scope never writes anywhere and never touches the
vectors, which is what keeps moving a document between categories from costing a
re-index.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from langchain_core.retrievers import BaseRetriever

from app.catalog import Catalog, DocumentRecord, normalise_category
from app.config import Settings
from app.vectorstore import WholeDocumentRetriever, build_retriever, get_vectorstore


@dataclass(frozen=True)
class Scope:
    """Which documents a question is asked of.

    `sources` is None for the whole library, and a tuple of document paths
    otherwise — the same value the chunks carry as `source`, which is what the
    vector store is filtered by. `chunks` is the chunk count of the one document
    a whole-document scope reads, and is None otherwise.
    """

    sources: tuple[str, ...] | None
    label: str
    whole_document: bool = False
    chunks: int | None = None


def whole_library() -> Scope:
    """Everything indexed, which is what a question without a scope searches."""
    return Scope(sources=None, label="whole library")


def as_source(path: str, documents_dir: Path) -> str:
    """A document path as the catalog holds it, from however it was written.

    `scripts/delete.py` takes its arguments as catalog keys and nothing else. A
    scope is worth the tolerance: the two spellings a person actually types are
    the path relative to the documents folder and the path from the shell, and a
    wrong guess here would produce a search over nothing rather than an error
    naming the document that was not found.
    """
    candidate = Path(path)
    try:
        resolved = candidate.resolve()
        return resolved.relative_to(Path(documents_dir).resolve()).as_posix()
    except (OSError, ValueError):
        # Not a file under the documents folder, or not a path at all. Left as
        # typed, so that the lookup fails and says which document it could not
        # find, rather than failing somewhere less clear.
        return candidate.as_posix()


def resolve_scope(
    catalog: Catalog,
    *,
    config: Settings,
    category: str | None = None,
    document: str | None = None,
    documents: Sequence[str] = (),
) -> Scope:
    """Turn what was asked for into the documents to search.

    Raises `LookupError` rather than returning an empty scope: a search over no
    documents answers every question with "the documents do not cover this",
    which is the one answer this project exists not to give by accident.
    """
    given = [
        name
        for name, was_given in (
            # `is not None`, not truth: an empty category is a scope of its own —
            # the documents at the root, which are in no category.
            ("category", category is not None),
            ("document", document is not None),
            ("documents", bool(documents)),
        )
        if was_given
    ]
    if len(given) > 1:
        raise ValueError("A scope is one of " + ", ".join(given) + ", not several.")

    if category is not None:
        return _category_scope(catalog, normalise_category(category))

    if document is not None:
        record = _lookup(catalog, as_source(document, config.documents_dir))
        return _one_document_scope(record, config)

    if documents:
        sources = tuple(
            _lookup(catalog, as_source(path, config.documents_dir)).path
            for path in documents
        )
        return Scope(sources=sources, label=_documents(len(sources)))

    return whole_library()


def build_scoped_retriever(scope: Scope) -> BaseRetriever:
    """The retriever these documents are searched through.

    A document small enough to be read whole goes through the whole-document
    retriever, which hands back all of it in reading order; everything else — a
    category, a selection, one document too long for the budget — is the
    similarity search the library has always used, with the scope's documents as
    the one filter.

    The graph takes either one the same way, which is the whole reason a scope
    changes nothing about the graph.
    """
    if scope.whole_document and scope.sources and scope.chunks:
        return WholeDocumentRetriever(
            store=get_vectorstore(),
            source=scope.sources[0],
            k=scope.chunks,
        )

    return build_retriever(sources=scope.sources)


def _category_scope(catalog: Catalog, name: str | None) -> Scope:
    sources = tuple(catalog.sources_in_category(name))

    if not sources:
        known = sorted(found for found, _ in catalog.category_counts() if found)
        hint = (
            " Known categories: " + ", ".join(known) + "."
            if known
            else " No document in the catalog is in a category yet."
        )
        raise LookupError(f"No indexed documents in {_name_of(name)}.{hint}")

    named = f"category {name}" if name else "no category"
    return Scope(sources=sources, label=f"{named} ({_documents(len(sources))})")


def _one_document_scope(record: DocumentRecord, config: Settings) -> Scope:
    """One document, read whole when it is small enough to be handed over.

    The estimate is a pre-flight one, from numbers the catalog already holds, so
    no search is run to decide. It over-counts, because consecutive chunks
    overlap; that direction is the safe one, and the document simply falls back
    to a similarity search sooner than it strictly had to.
    """
    budget = config.whole_document_max_chars
    estimate = (record.chunk_count or 0) * config.chunk_size
    whole = budget > 0 and 0 < estimate <= budget

    if whole:
        label = f"document {record.path} (whole, {_chunks(record.chunk_count or 0)})"
    else:
        label = (
            f"document {record.path} (top {config.retrieval_k} of its chunks)"
        )

    return Scope(
        sources=(record.path,),
        label=label,
        whole_document=whole,
        chunks=record.chunk_count if whole else None,
    )


def _lookup(catalog: Catalog, source: str) -> DocumentRecord:
    record = catalog.get(source)

    if record is None:
        raise LookupError(
            f"Not in the catalog: {source}. A document path is relative to the "
            "documents folder, and only a document that has been synced is in "
            "the catalog."
        )

    if record.status != "indexed":
        raise LookupError(
            f"{source} is {record.status}, so it has no passages to search. "
            "A sync can put it right."
        )

    return record


def _name_of(category: str | None) -> str:
    return (
        f"the category {category!r}"
        if category
        else "the documents with no category"
    )


def _documents(count: int) -> str:
    return f"{count} document" + ("" if count == 1 else "s")


def _chunks(count: int) -> str:
    return f"{count} chunk" + ("" if count == 1 else "s")
