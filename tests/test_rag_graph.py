"""The question and answer graph.

One turn runs through three nodes: the question is rewritten, the search is
run with the rewritten one and the answer is written from what it found.
The tests use a fake model and a fake retriever, so each part can be
checked on its own.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from app.rag_graph import build_graph, format_context, unique_sources
from app.vectorstore import WholeDocumentRetriever
from tests.helpers import (
    FakeChatModel,
    FakeRetriever,
    FakeVectorStore,
    as_a_store_returns,
)

THREAD = {"configurable": {"thread_id": "test-thread"}}


def make_document(page: int | None = 11) -> Document:
    metadata: dict[str, Any] = {"source": "manuals/manual.pdf"}
    if page is not None:
        metadata["page"] = page

    metadata["start"] = 120
    metadata["end"] = 148
    return Document(page_content="The thing is explained here.", metadata=metadata)


def make_graph(
    replies: list[str],
    documents: list[Document] | None = None,
    in_scope: tuple[str, ...] | None = None,
    descriptions: dict[str, str] | None = None,
) -> tuple[Any, FakeChatModel, FakeRetriever]:
    model = FakeChatModel(replies=replies)
    retriever = FakeRetriever(
        [make_document()] if documents is None else documents
    )
    graph = build_graph(
        chat_model=model,
        retriever=retriever,
        in_scope=in_scope,
        descriptions=descriptions,
    )
    return graph, model, retriever


async def ask(graph: Any, question: str) -> dict[str, Any]:
    return await graph.ainvoke(
        {"messages": [HumanMessage(content=question)]}, config=THREAD
    )


def test_the_context_carries_the_source_and_a_one_based_page():
    context, rows = format_context([make_document()])

    assert context == (
        "[Source: manuals/manual.pdf | Page: 12]\nThe thing is explained here."
    )

    # Pages are stored counted from zero and shown counted from one.
    assert rows == [
        {"source": "manuals/manual.pdf", "page": 12, "ranges": [[120, 148]]}
    ]


def test_a_chunk_without_a_page_or_a_source_still_renders():
    context, rows = format_context([Document(page_content="text", metadata={})])

    assert context == "[Source: unknown]\ntext"

    # A chunk with no metadata is given a name and nothing else.
    assert rows == [{"source": "unknown", "page": None, "ranges": []}]


def test_a_chunk_the_store_hands_back_is_the_same_row():
    """A chunk as a store returns it still makes the same row.

    A store that keeps its metadata outside the process returns whole
    numbers as floats. The page and the offsets are read back into the
    types the context uses.
    """

    context, rows = format_context([as_a_store_returns(make_document())])

    assert context == (
        "[Source: manuals/manual.pdf | Page: 12]\nThe thing is explained here."
    )
    assert rows == [
        {"source": "manuals/manual.pdf", "page": 12, "ranges": [[120, 148]]}
    ]


def test_a_range_that_starts_before_the_page_has_none():
    """A range that starts before the text is not kept.

    An offset of -3 points outside the page, so the row carries no range
    at all.
    """

    document = Document(
        page_content="text",
        metadata={"source": "a.pdf", "page": 0, "start": -3, "end": 10},
    )

    assert format_context([document])[1] == [
        {"source": "a.pdf", "page": 1, "ranges": []}
    ]


def test_a_chunk_whose_offsets_are_not_a_range_has_none():
    """Offsets that describe no text are dropped.

    The start and the end are the same number here, so the row carries no
    range.
    """

    document = Document(
        page_content="text",
        metadata={"source": "a.pdf", "page": 0, "start": 40, "end": 40},
    )

    assert format_context([document])[1] == [
        {"source": "a.pdf", "page": 1, "ranges": []}
    ]


def test_two_passages_of_one_page_are_one_row_with_both_their_ranges():
    """Two passages of one page make one row holding both their ranges.

    The pages are counted from one here, so chunks stored on pages 11 and
    12 come back as pages 12 and 13.
    """

    _, rows = format_context([
        Document(
            page_content="The thing is explained here.",
            metadata={"source": "manuals/manual.pdf", "page": 11, "start": 0, "end": 28},
        ),
        Document(
            page_content="And again, further down.",
            metadata={"source": "manuals/manual.pdf", "page": 11, "start": 900, "end": 925},
        ),
        Document(
            page_content="On the next page.",
            metadata={"source": "manuals/manual.pdf", "page": 12, "start": 0, "end": 17},
        ),
    ])

    assert unique_sources(rows) == [
        {"source": "manuals/manual.pdf", "page": 12, "ranges": [[0, 28], [900, 925]]},
        {"source": "manuals/manual.pdf", "page": 13, "ranges": [[0, 17]]},
    ]


def test_the_same_rows_read_twice_give_the_same_answer():
    """Merging the same rows twice gives the same result.

    The merge builds new rows, so the ones passed in are left as they were
    and a second call over them produces the same list.
    """

    rows = [
        {"source": "manuals/manual.pdf", "page": 12, "ranges": [[0, 28]]},
        {"source": "manuals/manual.pdf", "page": 12, "ranges": [[900, 925]]},
    ]

    assert unique_sources(rows) == unique_sources(rows) == [
        {"source": "manuals/manual.pdf", "page": 12, "ranges": [[0, 28], [900, 925]]}
    ]


def test_the_context_opens_with_the_documents_the_search_was_run_over():
    context, rows = format_context(
        [make_document()], in_scope=("manuals/manual.pdf", "reports/report.pdf")
    )

    assert context == (
        "Documents searched: 2 — manuals/manual.pdf, reports/report.pdf\n\n"
        "[Source: manuals/manual.pdf | Page: 12]\nThe thing is explained here."
    )

    # The list of documents comes first and the passages follow it.
    assert rows == [
        {"source": "manuals/manual.pdf", "page": 12, "ranges": [[120, 148]]}
    ]


def test_a_context_that_was_not_told_a_scope_claims_none():
    """A context with no list of documents starts at the first passage."""
    context, _ = format_context([make_document()], in_scope=())

    assert context.startswith("[Source: manuals/manual.pdf")


@pytest.mark.asyncio
async def test_the_answer_is_the_last_message():
    graph, _, _ = make_graph(["a standalone question", "the answer"])

    state = await ask(graph, "what does it say?")

    assert isinstance(state["messages"][-1], AIMessage)
    assert state["messages"][-1].content == "the answer"


@pytest.mark.asyncio
async def test_the_retriever_searches_the_rewritten_question():
    graph, _, retriever = make_graph(["a standalone question", "the answer"])

    await ask(graph, "and for minors?")

    assert retriever.queries == ["a standalone question"]


@pytest.mark.asyncio
async def test_the_answer_prompt_carries_the_retrieved_context():
    graph, model, _ = make_graph(["?", "the answer"])

    await ask(graph, "what does it say?")

    answer_prompt = str(model.prompts[-1])
    assert "manuals/manual.pdf" in answer_prompt
    assert "Page: 12" in answer_prompt
    assert "The thing is explained here." in answer_prompt


@pytest.mark.asyncio
async def test_the_answer_is_written_knowing_which_documents_were_searched():
    """The answer prompt carries the documents the search was run over.

    It also says what that list is for, so a question about the library
    itself can be answered from it.
    """

    graph, model, _ = make_graph(
        ["a standalone question", "the answer"],
        in_scope=("manuals/manual.pdf", "reports/report.pdf"),
    )

    await ask(graph, "what documents do you have?")

    answer_prompt = str(model.prompts[-1])
    assert (
        "Documents searched: 2 — manuals/manual.pdf, reports/report.pdf"
        in answer_prompt
    )
    assert "opens with the documents the search was run over" in answer_prompt


def test_the_context_carries_what_the_catalog_says_about_a_document():
    """A description is written under the document it belongs to."""

    context, _ = format_context(
        [make_document()],
        in_scope=("manuals/manual.pdf", "reports/report.pdf"),
        descriptions={"reports/report.pdf": "Last year's report."},
    )

    assert context.startswith(
        "Documents searched: 2 — manuals/manual.pdf, reports/report.pdf\n"
        "reports/report.pdf — Last year's report."
    )


def test_a_document_nothing_is_written_about_is_named_and_no_more():
    """A document with no description is named in the list and no more."""

    context, _ = format_context(
        [make_document()],
        in_scope=("manuals/manual.pdf",),
        descriptions={"reports/report.pdf": "Last year's report."},
    )

    assert context.startswith("Documents searched: 1 — manuals/manual.pdf\n\n")


@pytest.mark.asyncio
async def test_the_answer_is_written_knowing_what_the_documents_contain():
    """The prompt carries what the catalog says about each document.

    It also says what that text is for, so a question about what a document
    contains can be answered from it.
    """

    graph, model, _ = make_graph(
        ["a standalone question", "the answer"],
        in_scope=("manuals/manual.pdf", "reports/report.pdf"),
        descriptions={
            "manuals/manual.pdf": "A manual about the thing.",
            "reports/report.pdf": "Last year's report.",
        },
    )

    await ask(graph, "what does each of them contain?")

    answer_prompt = str(model.prompts[-1])
    assert "manuals/manual.pdf — A manual about the thing." in answer_prompt
    assert "reports/report.pdf — Last year's report." in answer_prompt
    assert "what the catalogue says about it" in answer_prompt


@pytest.mark.asyncio
async def test_a_question_with_nothing_retrieved_is_still_answered():
    graph, model, _ = make_graph(["?", "there is nothing about it"], documents=[])

    state = await ask(graph, "what does it say?")

    assert state["retrieved_documents"] == []
    assert state["context"] == ""
    assert state["messages"][-1].content == "there is nothing about it"
    assert len(model.prompts) == 2


@pytest.mark.asyncio
async def test_a_follow_up_carries_the_previous_turn():
    graph, model, _ = make_graph(
        [
            "a standalone question",
            "the answer",
            "a second standalone question",
            "the second answer",
        ]
    )

    await ask(graph, "what are the requirements?")
    await ask(graph, "and for minors?")

    second_question_prompt = str(model.prompts[2])
    assert "what are the requirements?" in second_question_prompt
    assert "the answer" in second_question_prompt


@pytest.mark.asyncio
async def test_the_last_message_must_be_a_question():
    graph, _, _ = make_graph(["?"])

    with pytest.raises(TypeError, match="the user"):
        await graph.ainvoke(
            {"messages": [AIMessage(content="I am the assistant")]}, config=THREAD
        )


@pytest.mark.asyncio
async def test_a_scoped_retriever_is_all_a_scoped_answer_takes():
    """An answer about one document needs only a retriever scoped to it.

    The retriever here reads one document whole. The graph is built with it
    and the answer carries the chunks of that document, and nothing from
    the other one in the store.
    """

    store = FakeVectorStore()
    store.added.append((
        [
            Document(
                page_content="The thing is explained here.",
                metadata={
                    "source": "manuals/manual.pdf",
                    "page": 0,
                    "chunk_id": 0,
                    "start": 0,
                    "end": 28,
                },
            ),
            Document(
                page_content="Something else entirely.",
                metadata={
                    "source": "reports/report.pdf",
                    "page": 0,
                    "chunk_id": 0,
                    "start": 0,
                    "end": 23,
                },
            ),
        ],
        ["0", "1"],
    ))
    graph = build_graph(
        chat_model=FakeChatModel(replies=["a standalone question", "the answer"]),
        retriever=WholeDocumentRetriever(
            store=store, source="manuals/manual.pdf", k=1
        ),
    )

    state = await ask(graph, "what does it say?")

    assert state["retrieved_documents"] == [
        {"source": "manuals/manual.pdf", "page": 1, "ranges": [[0, 28]]}
    ]
    assert "Something else entirely." not in state["context"]
