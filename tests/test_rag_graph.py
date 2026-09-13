"""The graph, run end to end against a fake model and a fake retriever.

Nothing here reaches an API: the model replies from a list, the retriever hands
back the documents it was given. Both record what they were asked, which is how
the tests see the query the model rewrote and the context it was given.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from app.rag_graph import build_graph, format_context
from app.vectorstore import WholeDocumentRetriever
from tests.helpers import FakeVectorStore

THREAD = {"configurable": {"thread_id": "test-thread"}}


class FakeChatModel(BaseChatModel):
    """Replies from a list, one reply per call, and keeps the prompts it saw."""

    replies: list[str]
    prompts: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.prompts.append(list(messages))
        reply = self.replies[len(self.prompts) - 1]
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=reply))]
        )


class FakeRetriever:
    """The graph only ever calls `ainvoke` on a retriever, so that is all this is."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.queries: list[str] = []

    async def ainvoke(
        self, query: str, config: Any = None, **kwargs: Any
    ) -> list[Document]:
        self.queries.append(query)
        return self.documents


def make_document(page: int | None = 11) -> Document:
    metadata: dict[str, Any] = {"source": "manuals/manual.pdf"}
    if page is not None:
        metadata["page"] = page
    return Document(page_content="The thing is explained here.", metadata=metadata)


def make_graph(
    replies: list[str], documents: list[Document] | None = None
) -> tuple[Any, FakeChatModel, FakeRetriever]:
    model = FakeChatModel(replies=replies)
    retriever = FakeRetriever(
        [make_document()] if documents is None else documents
    )
    return build_graph(chat_model=model, retriever=retriever), model, retriever


async def ask(graph: Any, question: str) -> dict[str, Any]:
    return await graph.ainvoke(
        {"messages": [HumanMessage(content=question)]}, config=THREAD
    )


def test_the_context_carries_the_source_and_a_one_based_page():
    context, rows = format_context([make_document()])

    assert context == (
        "[Source: manuals/manual.pdf | Page: 12]\nThe thing is explained here."
    )
    assert rows == [{"source": "manuals/manual.pdf", "page": 12}]


def test_a_chunk_without_a_page_or_a_source_still_renders():
    context, rows = format_context([Document(page_content="text", metadata={})])

    assert context == "[Source: unknown]\ntext"
    assert rows == [{"source": "unknown", "page": None}]


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
    """How the console asks about one document: a different retriever, same graph.

    `WholeDocumentRetriever` is a `BaseRetriever` rather than a `VectorStore` one,
    which is the point — the graph asks any retriever for documents and does not
    care which kind it is holding.
    """
    store = FakeVectorStore()
    store.added.append((
        [
            Document(
                page_content="The thing is explained here.",
                metadata={"source": "manuals/manual.pdf", "page": 0, "chunk_id": 0},
            ),
            Document(
                page_content="Something else entirely.",
                metadata={"source": "reports/report.pdf", "page": 0, "chunk_id": 0},
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
        {"source": "manuals/manual.pdf", "page": 1}
    ]
    assert "Something else entirely." not in state["context"]
