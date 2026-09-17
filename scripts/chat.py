"""The question and answer console.

`run_chat` resolves the scope once, builds the graph over it, and then reads
questions from the terminal until 'exit' is typed. After every answer it prints
the sources the search found, with the page of each.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from uuid import uuid4

from app.catalog import Catalog
from app.chat_model import build_chat_model
from app.config import describe_vector_store, settings, short_path
from app.rag_graph import (
    build_graph,
    get_thread_state,
    thread_config,
    unique_sources,
)
from app.rerank import describe_rerank
from app.scope import build_scoped_retriever, resolve_scope


async def run_chat(
    *,
    category: str | None = None,
    document: str | None = None,
    documents: list[str] | None = None,
    documents_dir: Path | None = None,
    db_path: Path | None = None,
) -> None:
    """Ask questions in the terminal, one after another.

    The scope is resolved before the first question and stays the same for the
    session. The conversation is kept in memory, so it lasts as long as the
    console does.
    """
    documents_dir = Path(documents_dir or settings.documents_dir)
    db_path = Path(db_path or settings.catalog_db_path)

    try:
        scope = resolve_scope(
            Catalog(db_path, create=False),
            config=settings,
            category=category,
            document=document,
            documents=documents or [],
        )

        # The retriever is built once for the session, because the scope does
        # not change between questions. No checkpointer is given, so the graph
        # keeps the conversation in memory.
        graph = build_graph(
            chat_model=build_chat_model(),
            retriever=build_scoped_retriever(scope),
            in_scope=scope.documents,
            descriptions=dict(scope.descriptions),
        )
    except (LookupError, RuntimeError) as exc:
        # A scope that cannot be resolved stops the console before the first
        # question is asked.
        raise SystemExit(str(exc)) from exc

    config = thread_config(f"console-{uuid4()}")

    print("Repertorio Docs console")

    # What this session is searching. It is printed once, at the top, so that
    # the answers can be read with the settings they came from in mind.
    print(f"store: {describe_vector_store(settings)}")
    print(f"catalog: {short_path(db_path)}")
    print(f"scope: {scope.label}")
    print(f"rerank: {describe_rerank(settings)}")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        print("Assistant: ", end="", flush=True)

        try:
            async for chunk in graph.astream(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": question,
                        }
                    ]
                },
                config=config,
                stream_mode="messages",
                version="v2",
            ):
                if chunk["type"] != "messages":
                    continue

                token, metadata = chunk["data"]

                # Only the answer node writes what the reader sees. The first
                # node writes the rewritten query, which is not part of the
                # answer.
                if metadata.get("langgraph_node") != "answer":
                    continue

                content = token.content
                if isinstance(content, str):
                    print(content, end="", flush=True)

            print("\n")

            state = get_thread_state(config, graph)
            sources = unique_sources(state.values.get("retrieved_documents", []))

            if sources:
                print("Sources:")
                for source in sources:
                    page = source.get("page")
                    suffix = f" - p. {page}" if page else ""
                    print(f"  - {source.get('source')}{suffix}")
                print()

        except Exception as exc:  # noqa: BLE001
            # One question failed and the console goes on to the next one.
            print(f"\nError: {exc}\n")


def parse_args() -> argparse.Namespace:
    """The command line, with the scope arguments checked against one another."""
    parser = argparse.ArgumentParser(
        description=(
            "Ask questions about the library. Without a scope, the whole library "
            "is searched; with one, only what it names."
        )
    )
    parser.add_argument(
        "--category",
        metavar="NAME",
        help="ask only about a category and what is filed below it; '' for the "
        "documents in no category",
    )
    parser.add_argument(
        "--document",
        metavar="PATH",
        help="ask only about one document, read whole when it is small enough",
    )
    parser.add_argument(
        "--documents",
        nargs="+",
        metavar="PATH",
        help="ask only about these documents, searched by similarity",
    )
    parser.add_argument(
        "--documents-dir",
        type=Path,
        default=settings.documents_dir,
        help="folder to watch (default: %(default)s)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=settings.catalog_db_path,
        help="catalog database (default: %(default)s)",
    )

    args = parser.parse_args()

    # The three scope arguments are alternatives. More than one of them leaves
    # no way to tell which scope was meant.
    given = [
        name
        for name, was_given in (
            ("--category", args.category is not None),
            ("--document", args.document is not None),
            ("--documents", bool(args.documents)),
        )
        if was_given
    ]
    if len(given) > 1:
        parser.error("give one of " + ", ".join(given) + ", not several")

    return args


def main() -> None:
    """Run the console."""
    args = parse_args()
    asyncio.run(
        run_chat(
            category=args.category,
            document=args.document,
            documents=args.documents,
            documents_dir=args.documents_dir,
            db_path=args.db,
        )
    )


if __name__ == "__main__":
    main()
