"""The set of documents a question is asked of.

A scope can be the whole library, one category, one document or a list of
documents. Besides the sources it carries the label a console prints and the two
things that go into the context of the answer: what the catalog says about each
document, and the headings of each one.

`resolve_scope` builds one from the arguments a command was given, and
`build_scoped_retriever` turns it into the retriever that searches it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from langchain_core.retrievers import BaseRetriever

from app.catalog import Catalog, DocumentRecord, normalise_category
from app.config import Settings
from app.structure import outline
from app.vectorstore import WholeDocumentRetriever, build_retriever, get_vectorstore


@dataclass(frozen=True)
class Scope:
    """The documents a question is asked of, and what is known about them.

    `sources` holds the document paths, and is None when the scope is the whole
    library. `documents` holds the same paths in the order they were given.
    `descriptions` pairs a path with the description the catalog wrote for it,
    and `outlines` pairs a path with the headings of that document, already
    rendered and already within the budget. Both are text by the time they get
    here: the tree is read from the catalog once per scope rather than once per
    question, and a value compared for equality carries a string and not a tree.
    """

    sources: tuple[str, ...] | None
    label: str
    documents: tuple[str, ...] = ()
    descriptions: tuple[tuple[str, str], ...] = ()
    outlines: tuple[tuple[str, str], ...] = ()
    whole_document: bool = False
    chunks: int | None = None

    @property
    def ranked(self) -> bool:
        """Whether the search returns a ranking of passages.

        It does, unless the scope is one document small enough to be read whole.
        That case returns every chunk of the document in reading order.
        """
        return not (self.whole_document and self.sources and self.chunks)


def whole_library(catalog: Catalog, *, config: Settings) -> Scope:
    """A scope covering every indexed document."""
    documents = _indexed(catalog)

    return Scope(
        sources=None,
        label="whole library",
        documents=documents,
        descriptions=_described(
            catalog, documents, budget=config.description_budget_chars
        ),
        outlines=_outlined(catalog, documents, budget=config.outline_budget_chars),
    )


def _described(
    catalog: Catalog, documents: tuple[str, ...], *, budget: int
) -> tuple[tuple[str, str], ...]:
    """The descriptions to put in the context, within a character budget.

    Documents are taken in the order given until the budget runs out. The first
    description that does not fit ends the walk.
    """
    if budget <= 0 or not documents:
        return ()

    said = {
        record.path: (record.description or "").strip()
        for record in catalog.all()
    }
    written: list[tuple[str, str]] = []
    spent = 0

    for path in documents:
        description = said.get(path)
        if not description:
            continue
        if spent + len(description) > budget:
            break
        spent += len(description)
        written.append((path, description))

    return tuple(written)


def _outlined(
    catalog: Catalog, documents: tuple[str, ...], *, budget: int
) -> tuple[tuple[str, str], ...]:
    """The outlines to put in the context, within a character budget.

    Documents are taken in the order given until the budget runs out, which is
    how the descriptions are spent. The first outline that does not fit ends the
    walk, and no part of one is ever written: a list of chapters cut short reads
    as a document with fewer chapters, and whoever is reading it cannot tell the
    two apart. A document the reader found no heading in carries no outline and
    is passed over without spending anything.
    """
    if budget <= 0 or not documents:
        return ()

    written: list[tuple[str, str]] = []
    spent = 0

    for path in documents:
        tree = outline(catalog.structure(path))
        if not tree:
            continue
        if spent + len(tree) > budget:
            break
        spent += len(tree)
        written.append((path, tree))

    return tuple(written)


def _indexed(catalog: Catalog) -> tuple[str, ...]:
    """The paths of the indexed documents, in catalog order."""
    return tuple(
        record.path for record in catalog.all() if record.status == "indexed"
    )


def as_source(path: str, documents_dir: Path) -> str:
    """A document path as the catalog holds it, from however it was written.

    The two spellings a person types are the path relative to the documents
    folder and the path from the shell. Both are accepted. A path that turns out
    to be outside the folder is returned as it was typed, so that the lookup
    fails afterwards and names the document it could not find.
    """
    candidate = Path(path)
    try:
        resolved = candidate.resolve()
        return resolved.relative_to(Path(documents_dir).resolve()).as_posix()
    except (OSError, ValueError):
        return candidate.as_posix()


def document_path(source: str, documents_dir: Path) -> Path:
    """The file a document path names, or a ValueError.

    A path that leaves the documents folder is refused here. This is where a
    string from outside the program becomes a file to open, so this is where the
    folder is enforced.

    The catalog is not consulted. The folder decides whether a document is here,
    and a file sitting in it can be read whether or not a sync has indexed it.
    """
    root = Path(documents_dir).resolve()

    try:
        # An absolute path replaces the folder instead of joining it, and a path
        # with enough `..` climbs out of it. Comparing the resolved path with the
        # folder catches both.
        resolved = (root / source).resolve()
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Not a document here: {source}") from exc

    if resolved.suffix.lower() != ".pdf" or not resolved.is_file():
        raise ValueError(f"Not a document here: {source}")

    return resolved


def resolve_scope(
    catalog: Catalog,
    *,
    config: Settings,
    category: str | None = None,
    document: str | None = None,
    documents: Sequence[str] = (),
) -> Scope:
    """Build the scope a command asked for.

    One of `category`, `document` and `documents` may be given, and giving more
    than one is an error. With none of them the scope is the whole library.
    """
    # A scope is one thing at a time.
    given = [
        name
        for name, was_given in (
            ("category", category is not None),
            ("document", document is not None),
            ("documents", bool(documents)),
        )
        if was_given
    ]
    if len(given) > 1:
        raise ValueError("A scope is one of " + ", ".join(given) + ", not several.")

    if category is not None:
        return _category_scope(catalog, normalise_category(category), config=config)

    if document is not None:
        record = _lookup(catalog, as_source(document, config.documents_dir))
        return _one_document_scope(catalog, record, config)

    if documents:
        sources = tuple(
            _lookup(catalog, as_source(path, config.documents_dir)).path
            for path in documents
        )
        return Scope(
            sources=sources,
            label=_documents(len(sources)),
            documents=sources,
            descriptions=_described(
                catalog, sources, budget=config.description_budget_chars
            ),
            outlines=_outlined(
                catalog, sources, budget=config.outline_budget_chars
            ),
        )

    return whole_library(catalog, config=config)


def build_scoped_retriever(scope: Scope, *, k: int | None = None) -> BaseRetriever:
    """The retriever that searches this scope.

    A scope that is read whole gets a `WholeDocumentRetriever`. Every other scope
    gets the ordinary search, limited to its sources. `k` is passed on to that
    search and defaults to RETRIEVAL_K.
    """
    if not scope.ranked:
        return WholeDocumentRetriever(
            store=get_vectorstore(),
            source=scope.sources[0],
            k=scope.chunks,
        )

    return build_retriever(sources=scope.sources, k=k)


def _category_scope(catalog: Catalog, name: str | None, *, config: Settings) -> Scope:
    """The scope of one category, or of the documents that have none."""
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
    return Scope(
        sources=sources,
        label=f"{named} ({_documents(len(sources))})",
        documents=sources,
        descriptions=_described(
            catalog, sources, budget=config.description_budget_chars
        ),
        outlines=_outlined(catalog, sources, budget=config.outline_budget_chars),
    )


def _one_document_scope(
    catalog: Catalog, record: DocumentRecord, config: Settings
) -> Scope:
    """The scope of one document.

    The document is read whole when its estimated size fits within
    WHOLE_DOCUMENT_MAX_CHARS. The estimate is the number of chunks times the
    chunk size.
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
        documents=(record.path,),
        descriptions=_described(
            catalog, (record.path,), budget=config.description_budget_chars
        ),
        outlines=_outlined(
            catalog, (record.path,), budget=config.outline_budget_chars
        ),
        whole_document=whole,
        chunks=record.chunk_count if whole else None,
    )


def _lookup(catalog: Catalog, source: str) -> DocumentRecord:
    """The catalog record for a source, or an error saying why there is none."""
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
