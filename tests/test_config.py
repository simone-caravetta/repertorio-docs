"""Tests for the config helpers that name files and stores.

Each vector store gets its own default catalog file, so that switching from
one to the other does not read the rows of the first. The other helpers
shorten a path for display and describe where the vectors are kept.
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
    """The default catalog is not the same file for the two stores."""
    assert _default_catalog_path("chroma") != _default_catalog_path("pinecone")


@pytest.mark.parametrize(
    ("store", "expected"),
    [
        ("chroma", "data/catalog-chroma.sqlite3"),
        ("pinecone", "data/catalog.sqlite3"),

        # The store name is matched without case and without spaces around it.
        ("  Chroma ", "data/catalog-chroma.sqlite3"),
        ("CHROMA", "data/catalog-chroma.sqlite3"),

        # Any other name falls back to the plain catalog.
        ("anything-else", "data/catalog.sqlite3"),
    ],
)
def test_the_catalog_is_chosen_from_the_store(store: str, expected: str) -> None:
    """Each store name maps to the catalog file it uses."""
    assert _default_catalog_path(store) == expected


def test_the_description_names_where_the_vectors_are() -> None:
    """The chroma description carries the collection name.

    The pinecone description carries the index name.
    """
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
    """The description is built from the settings handed to it.

    A caller that passes its own settings gets a description of those.
    """
    assert describe_vector_store(make_settings(vector_store="chroma")).startswith(
        "chroma"
    )


def test_a_path_under_the_project_reads_relative() -> None:
    """A path inside the project is shortened to its path from the root."""
    assert short_path(PROJECT_ROOT / "data" / "chroma") == "data/chroma"


def test_a_path_outside_the_project_is_left_alone() -> None:
    """A path elsewhere is printed as it was given."""
    outside = Path("/somewhere/else/catalog.sqlite3")

    assert short_path(outside) == str(outside)
