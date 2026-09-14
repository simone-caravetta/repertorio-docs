"""The library over HTTP: the catalog, a scope, and a conversation.

Nothing here decides anything the console decides differently. A scope is
resolved by `app.scope`, the tree is built by `app.catalog`, and the answer comes
out of the same graph. What the web adds is a place to wait: the sources are sent
the moment retrieval has finished, and the tokens follow as the model writes
them, so a page can show where an answer is coming from while it is still being
written.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from app.catalog import Catalog, CategoryBranch, DocumentRecord, build_category_tree
from app.chat_model import build_chat_model
from app.config import PROJECT_ROOT, Settings, settings
from app.rag_graph import (
    build_graph,
    read_thread_state,
    thread_config,
    turns_from,
    unique_sources,
)
from app.scope import Scope, build_scoped_retriever, resolve_scope

WEB_DIR = PROJECT_ROOT / "web"

# The words the console prints when there is nothing to look at yet, so that the
# two say the same thing about the same state.
NO_CATALOG = "No catalog here yet. Run a sync first."


class ChatRequest(BaseModel):
    """A question, the conversation to put it in, and what to ask it of."""

    question: str
    thread_id: str | None = None
    category: str | None = None
    document: str | None = None
    documents: list[str] = Field(default_factory=list)


def create_app(
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    chat_model: BaseChatModel | None = None,
    retriever_for: Callable[[Scope], BaseRetriever] | None = None,
    config: Settings | None = None,
) -> FastAPI:
    """The application, wired around the parts a caller can replace.

    Every argument defaults to the real thing and the tests hand in fakes, which
    is the same seam `build_graph` already has. `config` is read here rather than
    as a default argument, so that a caller which replaces the settings gets the
    ones it replaced.
    """
    config = config or settings
    make_retriever = retriever_for or build_scoped_retriever

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Open the conversations for the life of the process, and close them.

        The model is built here too, once: without a key to reach it with, there
        is nothing this server could answer, and saying so at startup beats
        saying it at the first question.
        """
        async with _conversations(checkpointer, Path(config.conversations_db_path)) as saver:
            app.state.checkpointer = saver
            app.state.model = chat_model or build_chat_model()
            # A graph to read threads with. Reading a state runs no node, so this
            # one never retrieves; it is the saver that holds the conversation.
            app.state.reader = build_graph(
                chat_model=app.state.model, checkpointer=saver
            )
            yield

    app = FastAPI(title="Repertorio Docs", lifespan=lifespan)

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/catalog")
    async def catalog_route() -> dict[str, Any]:
        db_path = Path(config.catalog_db_path)
        if not db_path.exists():
            return {"empty": True, "categories": [], "uncategorized": []}

        # Read-only, like the listing command: looking at the library must never
        # bring a database into existence.
        return catalog_view(Catalog(db_path, create=False))

    @app.get("/api/scope")
    async def scope_route(
        category: str | None = None,
        document: str | None = None,
        # A scope can name several documents, so this is asked for more than
        # once; `Annotated` says so without ruff objecting to the call.
        documents: Annotated[list[str] | None, Query()] = None,
    ) -> dict[str, Any]:
        return describe_scope(
            scope_of(
                config,
                category=category,
                document=document,
                documents=documents,
            )
        )

    @app.post("/api/chat")
    async def chat_route(body: ChatRequest, request: Request) -> StreamingResponse:
        """One question, streamed back as it is answered.

        The scope is resolved before the response begins, so a category that
        holds nothing is an error the page can read, not a stream that ends
        without an answer.
        """
        scope = scope_of(
            config,
            category=body.category,
            document=body.document,
            documents=body.documents,
        )
        thread_id = body.thread_id or f"chat-{uuid4()}"
        saver = request.app.state.checkpointer

        graph = build_graph(
            chat_model=request.app.state.model,
            retriever=make_retriever(scope),
            checkpointer=saver,
            # What the answer is told it searched, and what the catalog says
            # about it. The scope knows both for every scope, the whole library
            # included, and the passages say neither.
            in_scope=scope.documents,
            descriptions=dict(scope.descriptions),
        )

        return StreamingResponse(
            answer_stream(graph, thread_config(thread_id), body.question),
            media_type="text/event-stream",
            # No cache anywhere between here and the page, and no buffering on
            # the way: both would hold the tokens until the answer was over,
            # which is the one thing this stream exists not to do.
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/threads/{thread_id}")
    async def thread_route(thread_id: str, request: Request) -> dict[str, Any]:
        state = await read_thread_state(
            thread_config(thread_id), request.app.state.reader
        )

        return {
            "thread_id": thread_id,
            "messages": turns_from(state),
            # Only the last answer's query and sources: the state holds the
            # passages of one turn, and the ones before it were shown when they
            # were asked. The text of every turn is here, and the text is what a
            # reload is for. The query is here for the same reason it is in the
            # stream — a reloaded conversation should still say what was searched.
            "query": state.values.get("contextualized_question"),
            "sources": unique_sources(
                state.values.get("retrieved_documents", [])
            ),
        }

    return app


@asynccontextmanager
async def _conversations(
    checkpointer: BaseCheckpointSaver | None, path: Path
) -> AsyncIterator[BaseCheckpointSaver]:
    """The conversations, from wherever they are coming.

    One shape for both cases, because the lifespan has one: a server keeps them
    in a file it opens here and closes at shutdown, and a test hands in one it
    has already opened.
    """
    if checkpointer is not None:
        yield checkpointer
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        yield saver


def scope_of(
    config: Settings,
    *,
    category: str | None,
    document: str | None,
    documents: list[str] | None,
) -> Scope:
    """The scope a request asked for, or a refusal the caller can read.

    The message is the resolver's own: it is the one the console prints, and it
    names the document or the category that was not found. A missing API key is
    deliberately not caught here — that is the server, not the request.
    """
    db_path = Path(config.catalog_db_path)
    if not db_path.exists():
        raise HTTPException(status_code=400, detail=NO_CATALOG)

    try:
        return resolve_scope(
            Catalog(db_path, create=False),
            config=config,
            category=category,
            document=document,
            documents=documents or [],
        )
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def describe_scope(scope: Scope) -> dict[str, Any]:
    """A scope as the page shows it: what it says, and what it will search.

    `sources` is null for the whole library, which is not the same as a scope
    holding no documents: the first searches everything, the second is refused
    before it gets this far. Sending an empty list for both would say the whole
    library is a search over nothing.
    """
    return {
        "label": scope.label,
        "sources": None if scope.sources is None else list(scope.sources),
        "whole_document": scope.whole_document,
    }


async def answer_stream(
    graph: CompiledStateGraph,
    config: dict[str, Any],
    question: str,
) -> AsyncIterator[str]:
    """The answer as it is produced: the thread, the query, the sources, the tokens, the end.

    The query and the sources come out of the `updates` stream, which reports what
    each node handed on: `contextualize` has finished by the time its update is
    emitted, and `retrieve` by the time of its own, while the answer is still
    being written — which is where the fifth of these in the roadmap asks for the
    sources. The `messages` stream carries the tokens, filtered to the node that
    writes the answer, the way the console filters them.

    An error can only be reported inside the stream, a response having already
    begun: it ends the answer, and the page says what it was. A client that goes
    away is not an error — the cancellation is a `BaseException` and passes
    through, closing the generator.
    """
    yield event("thread", {"thread_id": config["configurable"]["thread_id"]})

    written: list[str] = []

    try:
        async for chunk in graph.astream(
            {"messages": [{"role": "user", "content": question}]},
            config=config,
            stream_mode=["messages", "updates"],
            version="v2",
        ):
            if chunk["type"] == "messages":
                token, metadata = chunk["data"]
                if metadata.get("langgraph_node") != "answer":
                    continue

                text = token.content
                if isinstance(text, str) and text:
                    written.append(text)
                    yield event("token", {"text": text})

            elif chunk["type"] == "updates":
                finished = chunk["data"]

                # The query the search was actually run on, sent before the
                # passages it found. It is not always the question as it was
                # typed: the graph rewrites it with the conversation in hand, and
                # a rewrite can narrow the search without anyone asking it to.
                # Anything that changes what a question means has to be visible.
                if "contextualize" in finished:
                    query = finished["contextualize"].get("contextualized_question")
                    if query:
                        yield event("query", {"query": query})

                if "retrieve" not in finished:
                    continue

                rows = finished["retrieve"].get("retrieved_documents", [])
                sources = unique_sources(rows)
                if sources:
                    yield event("sources", {"sources": sources})

        yield event("done", {"answer": "".join(written)})

    except Exception as exc:  # noqa: BLE001
        # The model being unreachable, the store refusing: one answer is lost,
        # the conversation is not.
        yield event("error", {"message": str(exc)})


def event(name: str, payload: dict[str, Any]) -> str:
    """One server-sent event.

    `data` is one line by construction: JSON escapes the newlines inside a string,
    so a long answer still arrives as a single `data:` line.
    """
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


def catalog_view(catalog: Catalog) -> dict[str, Any]:
    """The whole catalog as a page renders it.

    Built from the rows themselves rather than from `category_counts`, which
    counts only what is indexed: a document that failed to index has a status
    worth showing, and a category shown without it would be a category whose
    contents the page cannot explain.

    Trashed documents are left out. The sync put them there because their file is
    gone, and listing them would be listing documents no question can reach.
    """
    records = [record for record in catalog.all() if record.status != "trashed"]
    direct: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.category:
            direct.setdefault(record.category, []).append(as_document(record))

    tree = build_category_tree(Counter(record.category for record in records).items())

    return {
        "empty": False,
        "categories": [as_branch(branch, direct) for branch in tree],
        "uncategorized": [
            as_document(record) for record in records if not record.category
        ],
    }


def as_branch(
    branch: CategoryBranch, direct: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """A branch, with the documents filed directly in it and its own children."""
    return {
        "name": branch.name,
        # These two are not the same number: `documents` lists what is filed
        # directly in the category, `total` also counts what is filed below it,
        # which is what a question about the category would search.
        "total": branch.total,
        "documents": direct.get(branch.name, []),
        "children": [as_branch(child, direct) for child in branch.children],
    }


def as_document(record: DocumentRecord) -> dict[str, Any]:
    """A row as the page shows it. `description` is empty until Phase 5 fills it."""
    return {
        "path": record.path,
        "title": record.title,
        "description": record.description,
        "category": record.category,
        "status": record.status,
        "pages": record.page_count,
        "chunks": record.chunk_count,
    }
