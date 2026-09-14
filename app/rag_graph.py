from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.base import BaseCheckpointSaver
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

The query says what to look for, not where to look. Which documents are being
searched is decided before you are asked and is not yours to narrow: a file
name, a document title or the words "in the document..." inside the query
override that decision and search a different set of documents from the one
that was asked for. Never name a document in the query, not even when the
conversation so far has been about one.

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

The context opens with the list of the documents the search was run over, and
that list is part of the context. Answer from it when the question is about the
library itself — which documents there are, how many — and from the passages
when it is about what the documents say.

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


def format_context(
    documents: list[Document],
    *,
    in_scope: Sequence[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Render retrieved chunks as the context block, beside the sources to cite.

    `in_scope` opens the block with the documents the search was run over. It is
    the one thing the passages cannot say: a similarity search returns what is
    closest to the question, so a question about the library itself — which
    documents there are, how many — is answered by whatever the search happened
    to hit, and a model holding only those names them as the library. The scope
    knows better, and this is where the answer is told.
    """
    context_parts: list[str] = []
    source_rows: list[dict[str, Any]] = []

    if in_scope:
        context_parts.append(
            f"Documents searched: {len(in_scope)} — " + ", ".join(in_scope)
        )

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


def unique_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per place a passage came from.

    Four chunks of one page are one source, and the first mention is the one
    kept, so the list reads in the order the passages were found.
    """
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()

    for row in rows:
        key = (row.get("source"), row.get("page"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)

    return unique


def build_graph(
    *,
    chat_model: BaseChatModel,
    retriever: BaseRetriever | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    in_scope: Sequence[str] | None = None,
) -> CompiledStateGraph:
    """Wire the graph around the model that answers and the retriever that searches.

    Both are arguments rather than objects built here, so the graph can be run
    against any pair of them: a fake model and a fake retriever in the tests, a
    local server and Pinecone in use. The retriever falls back to the Pinecone
    one, built on first use — opening it at build time would load the embedding
    model before the console has asked anything.

    The checkpointer falls back to memory, which is what a console session wants:
    it is a session, and its history is not worth a file. A server hands in a
    persistent one instead, because a conversation that ends when the process
    does is the thing it exists not to have.

    `in_scope` is the documents the searches are run over, for a caller that has
    resolved a scope and knows: it is what the answer is told, so that a question
    about the library is answered from the library rather than from the passages
    the search happened to return. A graph built without it searches the same way
    and says less about it.
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
        context, source_rows = format_context(documents, in_scope=in_scope)

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

    return builder.compile(checkpointer=checkpointer or InMemorySaver())


@lru_cache(maxsize=1)
def get_app() -> CompiledStateGraph:
    """The graph over the whole library, built once per process, on first use.

    A console that was given a scope builds its own graph instead: the state of a
    conversation lives in the checkpointer of the graph that ran it.
    """
    return build_graph(chat_model=build_chat_model())


def thread_config(thread_id: str) -> dict[str, Any]:
    """The config naming one conversation, for every call that touches one.

    A thread id is the whole of what a conversation is: the same id on the next
    question continues it, and the same id after a restart finds it again.
    """
    return {"configurable": {"thread_id": thread_id}}


def get_thread_state(
    config: dict[str, Any], graph: CompiledStateGraph | None = None
):
    """What the conversation has produced so far, on the graph that ran it."""
    return (graph or get_app()).get_state(config)


async def read_thread_state(config: dict[str, Any], graph: CompiledStateGraph):
    """The same, read the other way round.

    Async because a persistent checkpointer is: with a saver that has to reach a
    file, the synchronous call above blocks the event loop, so a server reads its
    threads through here while the console — whose saver is in memory — keeps the
    synchronous one it has always used.
    """
    return await graph.aget_state(config)


def turns_from(state) -> list[dict[str, str]]:
    """The conversation as the turns a page renders, oldest first.

    Only what was said: the tool messages and the traffic of the nodes are in the
    state too, and none of it is a conversation.
    """
    turns: list[dict[str, str]] = []

    for message in state.values.get("messages", []):
        if message.type not in {"human", "ai"}:
            continue
        if not isinstance(message.content, str):
            continue
        turns.append({"role": message.type, "content": message.content})

    return turns
