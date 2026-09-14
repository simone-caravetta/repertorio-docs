"""The library in a browser: the same scope, the same answers, one page.

Nothing here decides anything the console decides differently. This reads the
flags, says what it is about to serve the way the console says what it is about
to search, and hands the application to uvicorn.

    python -m scripts.serve

It listens on 127.0.0.1, which is this machine only. `--host 0.0.0.0` opens it
to the network, and there is no authentication behind it: the whole library and
every conversation would be readable by anyone who can reach the port.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import uvicorn

from app.api import create_app
from app.config import Settings, describe_vector_store, settings, short_path


def config_for(args: argparse.Namespace) -> Settings:
    """The settings these flags describe.

    Only the two paths are replaceable: the store and the models come from the
    same `.env` the console and the sync read, so that the page and the console
    are two views of one library rather than two libraries.
    """
    return replace(
        settings,
        catalog_db_path=args.db,
        conversations_db_path=args.conversations,
    )


def parse_args() -> argparse.Namespace:
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
    args = parse_args()
    config = config_for(args)

    # The same banner the console prints, with the address under it: a wrong
    # answer is nearly always a question asked of the wrong store or the wrong
    # catalog, and the two lines settle it.
    print("Repertorio Docs")
    print(f"store: {describe_vector_store(config)}")
    print(f"catalog: {short_path(config.catalog_db_path)}")
    print(f"conversations: {short_path(config.conversations_db_path)}")
    print(f"listening on http://{args.host}:{args.port}\n")

    uvicorn.run(create_app(config=config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
