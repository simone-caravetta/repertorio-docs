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

    `sources` is what the search is narrowed to — the same value the chunks carry
    as `source`, which is what the vector store is filtered by — and None is the
    whole library, which is not a list of everything but no restriction at all.
    `documents` is what the scope covers, which is the same list except in that
    one case: the whole library has no filter to read them off, so they come from
    the catalog. The two are kept apart because they answer different questions.
    A search wants to know what to filter by, and an answer wants to be told what
    was searched: asked which documents there are, a model holding nothing but the
    passages in front of it names the one they came from. `descriptions` is what
    the catalog says about those documents, for the ones that have been described:
    the same argument one step further, because asked what the documents contain
    the passages are a sample of the answer rather than the answer. `chunks` is
    the chunk count of the one document a whole-document scope reads, and is None
    otherwise.
    """

    sources: tuple[str, ...] | None
    label: str
    documents: tuple[str, ...] = ()
    descriptions: tuple[tuple[str, str], ...] = ()
    whole_document: bool = False
    chunks: int | None = None


def whole_library(catalog: Catalog, *, config: Settings) -> Scope:
    """Everything indexed, which is what a question without a scope searches.

    Asked of the catalog, not left empty: the search is given no filter, but the
    documents it ends up covering are read from the catalog all the same, because
    the answer has to be able to say what was searched — and what it can say about
    them goes with them.
    """
    documents = _indexed(catalog)

    return Scope(
        sources=None,
        label="whole library",
        documents=documents,
        descriptions=_described(
            catalog, documents, budget=config.description_budget_chars
        ),
    )


def _described(
    catalog: Catalog, documents: tuple[str, ...], *, budget: int
) -> tuple[tuple[str, str], ...]:
    """What the catalog says about each document, while it fits a budget.

    The documents are named in the context either way, so a scope too large to
    describe loses the descriptions and nothing else. What does not fit is left
    out in the order the scope lists the documents, and `budget` of 0 — or a
    description longer than it, which the prompt keeps from happening — leaves
    them all out.
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


def _indexed(catalog: Catalog) -> tuple[str, ...]:
    """Every document a search can reach, in the catalog's own order.

    The rule `sources_in_category` follows, and for the same reason: a document
    that is trashed or failed to index has no vectors, so naming it would describe
    a search that cannot happen.
    """
    return tuple(
        record.path for record in catalog.all() if record.status == "indexed"
    )


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


def document_path(source: str, documents_dir: Path) -> Path:
    """The file a document path names, or a refusal.

    The other half of `as_source` above, and the opposite of it in the one way
    that matters: a path that leaves the documents folder is refused here instead
    of being handed back as it was typed. This is where a string from outside the
    program becomes a file to open, so this is where the folder is enforced.

    The catalog is deliberately no part of the answer. The folder is what decides
    whether a document is here, and a file sitting in it can be read whether or
    not a sync has indexed it; asking the catalog would add a database read to
    every request to answer a question the file already answers.
    """
    root = Path(documents_dir).resolve()

    try:
        # Left of the folder is outside it, and an absolute path replaces the
        # folder rather than joining it — both end up refused by the line after,
        # which is the point of resolving before comparing.
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
        )

    return whole_library(catalog, config=config)


def build_scoped_retriever(scope: Scope) -> BaseRetriever:
    """The retriever these documents are searched through.

    A document small enough to be read whole goes through the whole-document
    retriever, which hands back all of it in reading order; everything else — a
    category, a selection, one document too long for the budget — is the
    similarity search the library has always used, with the scope's documents as
    the one filter.

    The graph takes either one the same way, which is the whole reason a scope
    changes nothing about the graph.

    Reranking therefore applies to the second path and not the first, and that is
    a decision rather than an oversight: a document is read whole so that a
    question can reach a passage the top k would never have ranked, and putting a
    ranking back on top of it would drop exactly those. `RERANK=on` is not uniform
    across scopes — see `app/rerank.py`.
    """
    if scope.whole_document and scope.sources and scope.chunks:
        return WholeDocumentRetriever(
            store=get_vectorstore(),
            source=scope.sources[0],
            k=scope.chunks,
        )

    return build_retriever(sources=scope.sources)


def _category_scope(catalog: Catalog, name: str | None, *, config: Settings) -> Scope:
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
    )


def _one_document_scope(
    catalog: Catalog, record: DocumentRecord, config: Settings
) -> Scope:
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
        documents=(record.path,),
        descriptions=_described(
            catalog, (record.path,), budget=config.description_budget_chars
        ),
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
