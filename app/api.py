"""The HTTP app the page talks to.

The routes serve the catalog, run one question through the graph and stream the
answer back as it is written, and hand over the boxes of a passage inside a PDF
page.

`create_app` takes the checkpointer, the chat model and the retriever builder as
arguments, so that a test can build the app over its own pieces.
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

from app import grading, pdf
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
from app.scope import Scope, build_scoped_retriever, document_path, resolve_scope

# The folder holding the page and its assets.
WEB_DIR = PROJECT_ROOT / "web"


# Said to a request that arrives before anything has been indexed.
NO_CATALOG = "No catalog here yet. Run a sync first."


class ChatRequest(BaseModel):
    """One question, with the scope it is asked of.

    The scope is a category, one document or a list of documents, and it is the
    same one the page has selected.
    """

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
    """Build the app, over the pieces it is given.

    Each argument has a default built from the settings. A test passes its own,
    so that no model is called and no file of the user's is touched.
    """
    config = config or settings
    make_retriever = retriever_for or build_scoped_retriever

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Open the conversations and build the graph the app answers with.

        Both are built once, at startup, and kept on the app. The routes read
        them from there.
        """
        async with _conversations(checkpointer, Path(config.conversations_db_path)) as saver:
            app.state.checkpointer = saver
            app.state.model = chat_model or build_chat_model()

            # Kept for reading a thread back. It is built without a scope, and
            # the chat route builds its own graph for every question.
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

        # Opened read-only. The page only ever reads the catalog, and a second
        # connection that writes would lock the sync out.
        return catalog_view(Catalog(db_path, create=False))

    @app.get("/api/scope")
    async def scope_route(
        category: str | None = None,
        document: str | None = None,
        # A parameter repeated in the query string arrives as a list, and
        # Annotated is how the route declares that.
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
        """Answer one question and stream the answer back.

        The retriever is built here, scoped to the documents this request is
        about. The checkpointer is the one from startup, so a thread id that has
        been used before carries its conversation on.
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
            # What the answer is told about the library, which is the documents
            # that were searched and what the catalog says about each of them.
            in_scope=scope.documents,
            descriptions=dict(scope.descriptions),
            # Read from the config the app was built with, so that a server
            # started with GRADE off is a server that answers every question.
            grade=grading.enabled(config),
            attempts=config.grade_attempts,
        )

        return StreamingResponse(
            answer_stream(graph, thread_config(thread_id), body.question),
            media_type="text/event-stream",
            # No cache and no proxy buffering, or the events arrive in one go
            # once the answer is already finished.
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/documents/boxes")
    def boxes_route(source: str, page: int, start: int, end: int) -> dict[str, Any]:
        """The boxes of a passage, on the page it was found.

        The page is read again from the file and the character range the search
        recorded is located in it. What comes back is the coordinates of every
        line of that range, which the page draws over the image of the page.
        """
        if start < 0 or end <= start:
            raise HTTPException(
                status_code=400, detail=f"Not a range: {start}-{end}"
            )

        try:
            path = document_path(source, config.documents_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # The page is read from the file, so that the coordinates belong to the
        # page the reader is looking at.
        try:
            read = pdf.read_page(path, page)
        except IndexError as exc:
            raise HTTPException(
                status_code=400, detail=f"No page {page} in {source}"
            ) from exc

        return {
            "source": source,
            "page": page,
            "boxes": pdf.boxes(read, start, end),
        }

    @app.get("/api/threads/{thread_id}")
    async def thread_route(thread_id: str, request: Request) -> dict[str, Any]:
        state = await read_thread_state(
            thread_config(thread_id), request.app.state.reader
        )

        return {
            "thread_id": thread_id,
            "messages": turns_from(state),
            # The query the search ran, which is the question after it was
            # rewritten to stand on its own. The sources are those of the last
            # turn, with the repeats merged.
            "query": state.values.get("contextualized_question"),
            "sources": unique_sources(
                state.values.get("retrieved_documents", [])
            ),
        }

    @app.delete("/api/threads/{thread_id}")
    async def delete_thread_route(thread_id: str, request: Request) -> dict[str, str]:
        """Forget a conversation and everything stored under its thread id.

        The store is set up first, because deleting reads and writes the same
        tables the store creates when it starts. A thread that was never used
        is not an error.
        """
        checkpointer = request.app.state.checkpointer
        await checkpointer.setup()
        await checkpointer.adelete_thread(thread_id)

        return {"thread_id": thread_id}

    return app


@asynccontextmanager
async def _conversations(
    checkpointer: BaseCheckpointSaver | None, path: Path
) -> AsyncIterator[BaseCheckpointSaver]:
    """The checkpointer the app keeps its conversations in.

    One that is passed in is used as it is, which is what a test does. With
    none, the conversations file is opened and closed around the run of the
    app.
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
    """The scope a request asked for.

    A category or a document that is not in the catalog comes back as a 400
    carrying the reason, which the page shows as it is.
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
    """The scope as the page needs it.

    `sources` is None when the scope covers the whole library, which is how the
    page tells "everything" from "these documents". `whole_document` says the
    scope is one small document, which is handed over in full.
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
    """Answer one question and yield the events of the answer.

    The events come in the order the page needs them. The thread id first, then
    the rewritten query, then the sources, then the tokens of the answer as the
    model writes them, and a last one carrying the answer whole.

    A failure anywhere in the graph becomes an error event, so that the page has
    something to show in place of an answer.
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
                # Either node writes the reply that is shown: the answer, or
                # the one saying the documents do not hold it.
                if metadata.get("langgraph_node") not in {"answer", "unsupported"}:
                    continue

                text = token.content
                if isinstance(text, str) and text:
                    written.append(text)
                    yield event("token", {"text": text})

            elif chunk["type"] == "updates":
                finished = chunk["data"]

                # A node reports when it is done. The query and the sources go
                # out while the answer is still being written, which is what
                # lets the page show them above it.
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
        # The response has already started, so a failure is one more event.
        yield event("error", {"message": str(exc)})


def event(name: str, payload: dict[str, Any]) -> str:
    """One server-sent event, whose name the page listens for."""
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


def catalog_view(catalog: Catalog) -> dict[str, Any]:
    """The catalog as the page needs it, which is a tree of categories.

    A document in the trash is left out. Every branch carries the documents
    filed directly in it together with the branches below it, and the documents
    filed in no category come back on their own.
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
    """One branch of the tree, with the documents filed directly in it."""
    return {
        "name": branch.name,
        # The count over the whole branch, which includes its descendants.
        "total": branch.total,
        "documents": direct.get(branch.name, []),
        "children": [as_branch(child, direct) for child in branch.children],
    }


def as_document(record: DocumentRecord) -> dict[str, Any]:
    """One document as the page needs it."""
    return {
        "path": record.path,
        "title": record.title,
        "description": record.description,
        "category": record.category,
        "status": record.status,
        "pages": record.page_count,
        "chunks": record.chunk_count,
    }
