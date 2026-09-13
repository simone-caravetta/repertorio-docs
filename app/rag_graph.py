from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.chat_model import build_chat_model
from app.vectorstore import get_retriever


class RAGState(MessagesState):
    contextualized_question: str
    retrieved_documents: list[dict[str, Any]]
    context: str


contextualize_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You prepare the search query for a document retrieval system.

Given the previous conversation and the user's new question, rewrite that
question so it stands on its own: resolve references like "this", "that one",
"and abroad?", "how much?" using what was said before, so the query can be
understood without the conversation.

Do not answer the question. Return only the standalone query.""",
    ),
    (
        "human",
        """Previous conversation:
{history}

New question:
{question}""",
    ),
])

answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You answer questions about a collection of documents.

Use only the context provided below. If it does not hold enough information to
answer, say so clearly instead of guessing, and do not fill the gaps from your
own knowledge. Never invent facts, figures, names, procedures or rules that are
not in the context.

Write the answer in the language of the question, not the language of the
passages: a question asked in English gets an English answer even when every
passage is in another language. Do not change language part way through, and
translate whatever you quote from a passage written in another language.

If the question is about you rather than about the documents — what you are,
what you can do — answer briefly from this description instead of from the
context: you answer questions about the documents in this library, you say
which file and which page an answer comes from, and you keep the thread of a
conversation from one question to the next. That is all you do: you have no
other abilities and no access to anything outside these documents.""",
    ),
    (
        "human",
        """Question:
{question}

Context:
{context}""",
    ),
])


def format_context(documents: list[Document]) -> tuple[str, list[dict[str, Any]]]:
    """Render retrieved chunks as the context block, beside the sources to cite."""
    context_parts: list[str] = []
    source_rows: list[dict[str, Any]] = []

    for doc in documents:
        source = str(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page")
        if isinstance(page, int):
            page = page + 1

        label = f"Source: {source}"
        if page is not None:
            label += f" | Page: {page}"

        context_parts.append(f"[{label}]\n{doc.page_content}")
        source_rows.append({
            "source": source,
            "page": page,
        })

    return "\n\n".join(context_parts), source_rows


def build_graph(
    *,
    chat_model: BaseChatModel,
    retriever: BaseRetriever | None = None,
) -> CompiledStateGraph:
    """Wire the graph around the model that answers and the retriever that searches.

    Both are arguments rather than objects built here, so the graph can be run
    against any pair of them: a fake model and a fake retriever in the tests, a
    local server and Pinecone in use. The retriever falls back to the Pinecone
    one, built on first use — opening it at build time would load the embedding
    model before the console has asked anything.
    """
    contextualize_chain = contextualize_prompt | chat_model | StrOutputParser()

    async def contextualize(state: RAGState) -> dict[str, Any]:
        messages = state["messages"]
        latest = messages[-1]
        if not isinstance(latest, HumanMessage):
            raise TypeError("The last message must be a question from the user.")

        history = messages[:-1]
        history_text = "\n".join(
            f"{message.type}: {message.content}" for message in history
        )

        question = await contextualize_chain.ainvoke({
            "history": history_text or "(no previous conversation)",
            "question": latest.content,
        })

        return {"contextualized_question": question}

    async def retrieve(state: RAGState) -> dict[str, Any]:
        documents = await (retriever or get_retriever()).ainvoke(
            state["contextualized_question"]
        )
        context, source_rows = format_context(documents)

        return {
            "context": context,
            "retrieved_documents": source_rows,
        }

    async def answer(state: RAGState) -> dict[str, Any]:
        prompt = await answer_prompt.ainvoke({
            "question": state["contextualized_question"],
            "context": state["context"],
        })

        response = await chat_model.ainvoke(prompt)
        return {"messages": [response]}

    builder = StateGraph(RAGState)
    builder.add_node("contextualize", contextualize)
    builder.add_node("retrieve", retrieve)
    builder.add_node("answer", answer)

    builder.add_edge(START, "contextualize")
    builder.add_edge("contextualize", "retrieve")
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", END)

    return builder.compile(checkpointer=InMemorySaver())


@lru_cache(maxsize=1)
def get_app() -> CompiledStateGraph:
    """The graph over the whole library, built once per process, on first use.

    A console that was given a scope builds its own graph instead: the state of a
    conversation lives in the checkpointer of the graph that ran it.
    """
    return build_graph(chat_model=build_chat_model())


def get_thread_state(
    config: dict[str, Any], graph: CompiledStateGraph | None = None
):
    """What the conversation has produced so far, on the graph that ran it."""
    return (graph or get_app()).get_state(config)
