"""The command line of the server.

The script takes two paths from its flags and leaves everything else to the
environment. The tests cover the defaults, what a flag replaces, and the
errors the argument parser raises on its own.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from app.config import settings
from scripts.serve import config_for, parse_args


@pytest.fixture
def argv(monkeypatch: pytest.MonkeyPatch) -> Callable[..., argparse.Namespace]:
    """A parser for a command line written out in the test.

    The arguments given are placed after the program name, the way the
    shell would pass them, and the parsed namespace comes back.
    """

    def parse(*given: str) -> argparse.Namespace:
        monkeypatch.setattr(sys, "argv", ["scripts.serve", *given])
        return parse_args()

    return parse


def test_the_defaults_are_this_machine_only(argv: Callable[..., argparse.Namespace]):
    args = argv()

    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.db == settings.catalog_db_path
    assert args.conversations == settings.conversations_db_path


def test_the_two_paths_are_what_the_flags_replace(
    tmp_path: Path, argv: Callable[..., argparse.Namespace]
):
    db = tmp_path / "catalog.sqlite3"
    conversations = tmp_path / "conversations.sqlite3"

    config = config_for(
        argv("--db", str(db), "--conversations", str(conversations))
    )

    assert config.catalog_db_path == db
    assert config.conversations_db_path == conversations


def test_everything_else_comes_from_the_environment_unchanged(
    argv: Callable[..., argparse.Namespace],
):
    """Only the two paths are replaced. Every other field keeps its value."""
    config = config_for(argv())

    assert config.vector_store == settings.vector_store
    assert config.chroma_dir == settings.chroma_dir
    assert config.documents_dir == settings.documents_dir
    assert config.retrieval_k == settings.retrieval_k


def test_the_host_and_the_port_are_taken_as_given(
    argv: Callable[..., argparse.Namespace],
):
    args = argv("--host", "0.0.0.0", "--port", "9001")

    assert (args.host, args.port) == ("0.0.0.0", 9001)


def test_a_port_that_is_not_a_number_is_refused(
    argv: Callable[..., argparse.Namespace],
):
    with pytest.raises(SystemExit):
        argv("--port", "eight thousand")


def test_an_option_that_does_not_exist_is_refused(
    argv: Callable[..., argparse.Namespace],
):
    with pytest.raises(SystemExit):
        argv("--serve-the-whole-network")


def test_the_script_runs_and_describes_itself():
    """The script starts as a module and prints its own help.

    It runs in a subprocess, so the import path and the argument parser are
    covered together.
    """

    finished = subprocess.run(
        [sys.executable, "-m", "scripts.serve", "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert finished.returncode == 0
    assert "--conversations" in finished.stdout
