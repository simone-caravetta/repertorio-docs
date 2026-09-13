"""The settings that decide which store a command is talking to.

The catalog is derived from the store rather than named beside it, so that
switching store is one line. These tests are on the derivation itself: the
class defaults are read from the environment once, when `Settings` is defined,
so a test that went through `Settings()` would inherit the `.env` of whoever
runs the suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import (
    PROJECT_ROOT,
    _default_catalog_path,
    describe_vector_store,
    short_path,
)
from tests.helpers import make_settings


def test_the_two_stores_do_not_share_a_catalog() -> None:
    """One catalog per store, or the switch leaves a sync that does nothing.

    Every hash in one store's catalog matches the files regardless of where the
    vectors went, so a shared catalog reports a run with no work to do and
    leaves the store it now points at empty.
    """
    assert _default_catalog_path("chroma") != _default_catalog_path("pinecone")


@pytest.mark.parametrize(
    ("store", "expected"),
    [
        ("chroma", "data/catalog-chroma.sqlite3"),
        ("pinecone", "data/catalog.sqlite3"),
        # The value reaches the settings as typed by hand in a `.env`.
        ("  Chroma ", "data/catalog-chroma.sqlite3"),
        ("CHROMA", "data/catalog-chroma.sqlite3"),
        # An unknown name still gets a catalog: refusing it is `get_vectorstore`'s
        # job, and it can say so far better than a missing file could.
        ("anything-else", "data/catalog.sqlite3"),
    ],
)
def test_the_catalog_is_chosen_from_the_store(store: str, expected: str) -> None:
    assert _default_catalog_path(store) == expected


def test_the_description_names_where_the_vectors_are() -> None:
    """A command prints this before doing anything, so it has to be specific."""
    chroma = describe_vector_store(
        make_settings(vector_store="chroma", chroma_collection="passages")
    )
    assert chroma.startswith("chroma (")
    assert "collection passages" in chroma

    pinecone = describe_vector_store(
        make_settings(vector_store="pinecone", pinecone_index_name="an-index")
    )
    assert "an-index" in pinecone


def test_the_description_follows_the_settings_it_is_given() -> None:
    """Read from the argument, not the module-level `settings`.

    A command replaces its own `settings` with one built for the run; a
    description that ignored it would name the store the command is not using.
    """
    assert describe_vector_store(make_settings(vector_store="chroma")).startswith(
        "chroma"
    )


def test_a_path_under_the_project_reads_relative() -> None:
    assert short_path(PROJECT_ROOT / "data" / "chroma") == "data/chroma"


def test_a_path_outside_the_project_is_left_alone() -> None:
    outside = Path("/somewhere/else/catalog.sqlite3")

    assert short_path(outside) == str(outside)
