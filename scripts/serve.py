"""Serve the library over HTTP.

The page is served from `web/`, the catalog is browsable, and the chat is
streamed as the answer is written. The catalog and the conversations file can be
pointed elsewhere on the command line.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import uvicorn

from app.api import create_app
from app.config import Settings, describe_vector_store, settings, short_path
from app.grading import describe_grade
from app.rerank import describe_rerank
from app.small_to_big import describe_small_to_big


def config_for(args: argparse.Namespace) -> Settings:
    """The settings for this run, with the two paths from the command line.

    Everything else comes from the environment, so the server runs over the same
    model, store and documents folder as the console does.
    """
    return replace(
        settings,
        catalog_db_path=args.db,
        conversations_db_path=args.conversations,
    )


def parse_args() -> argparse.Namespace:
    """The command line, with the address and the two paths."""
    parser = argparse.ArgumentParser(
        description=(
            "Serve the library over HTTP. The catalog is browsable and the chat "
            "is streamed as it is written; the scope is the console's."
        )
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="address to listen on (default: %(default)s, this machine only)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="port to listen on (default: %(default)s)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=settings.catalog_db_path,
        help="catalog database (default: %(default)s)",
    )
    parser.add_argument(
        "--conversations",
        type=Path,
        default=settings.conversations_db_path,
        help="where the conversations are kept between runs (default: %(default)s)",
    )

    return parser.parse_args()


def main() -> None:
    """Start the server."""
    args = parse_args()
    config = config_for(args)

    # Printed before the server starts, so that a run can be read back with what
    # it served.
    print("Repertorio Docs")
    print(f"store: {describe_vector_store(config)}")
    print(f"catalog: {short_path(config.catalog_db_path)}")
    print(f"conversations: {short_path(config.conversations_db_path)}")
    # What every question asked of this server goes through, so that an answer
    # can be read back with the pass it came from in mind.
    print(f"rerank: {describe_rerank(config)}")
    print(f"small to big: {describe_small_to_big(config)}")
    print(f"grade: {describe_grade(config)}")
    print(f"listening on http://{args.host}:{args.port}\n")

    uvicorn.run(create_app(config=config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
