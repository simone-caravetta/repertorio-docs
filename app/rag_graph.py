"""The question and answer graph.

Three nodes run in order. The first rewrites the question together with the
conversation so far, so that a follow-up such as "and for minors?" becomes a
question that stands on its own. The second searches with that question and
builds the context. The third sends the question and the context to the chat
model.

When grading is on, the search and the answer are joined by two more nodes. One
reads the question and the material the search returned and says whether that
material holds the answer. Material that does not is searched for again with a
query that node wrote, and a question whose material is still missing after
that is turned away by the other rather than answered.

The state is checkpointed, which is what carries a conversation from one
question to the next.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
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

from app import grading
from app.chat_model import build_chat_model
from app.config import settings
from app.ingestion import whole_number
from app.vectorstore import get_retriever


class RAGState(MessagesState):
    """What the nodes pass to one another.

    `messages` holds the conversation. The other fields hold the work of one
    turn: the question after it was rewritten, the query the search actually
    ran on, the passages that were found, the context built from them, how that
    material was judged and how many searches the turn has run.

    The query is separate from the question because a retry searches with a
    query the grader wrote, while the answer is still written from the question
    the user asked. The verdict is a plain dictionary and not a `Verdict`,
    because the checkpointer writes the state out and cannot store a class of
    this project's own.
    """

    contextualized_question: str
    search_query: str
    retrieved_documents: list[dict[str, Any]]
    context: str
    verdict: dict[str, Any] | None
    searches: int


# Used by the first node. It rewrites the question so that it can be understood
# without the conversation, and it is told not to name a document, because the
# documents to search are already decided by the scope.
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

# Used by the third node. It writes the answer from the context alone, and it is
# told that the context also carries what the catalog says about the documents,
# which is what makes a question about the library itself answerable.
answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You answer questions about a collection of documents.

Use only the context provided below. If it does not hold enough information to
answer, say so clearly instead of guessing, and do not fill the gaps from your
own knowledge. Never invent facts, figures, names, procedures or rules that are
not in the context.

The context opens with the documents the search was run over and, under each,
what the catalogue says about it. Both are part of the context. Answer from them
when the question is about the library itself — which documents there are, what
each one contains — and from the passages when it is about what a document says.
A document the search returned no passage of is still a document you can say
something about, when its description is there; say that the description is what
you are answering from, and do not present it as a passage.

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

# Used instead of the answer when the search found nothing that holds the
# answer. The reply is written from the reason the grader gave, so that it says
# what the documents are about rather than only that they are silent.
unsupported_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You tell a user that a collection of documents does not hold what they asked about.

You are given the question and one sentence about the material the search
returned for it. The material does not answer the question, and searching again
has not turned up anything better.

Say that the documents do not hold the answer, and what the material that came
back is about instead. Do not answer the question from your own knowledge, do
not guess at what a document might say, and do not suggest where else to look.

Write in the language of the question. Two sentences are enough. Do not
apologise and do not offer further help.""",
    ),
    (
        "human",
        """Question:
{question}

What the search found instead:
{reason}""",
    ),
])


def format_context(
    documents: list[Document],
    *,
    in_scope: Sequence[str] | None = None,
    descriptions: Mapping[str, str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Build the context the answer is written from.

    The context opens with the documents the search was over and what the
    catalog says about each. Under that come the passages, each one labelled
    with its source and its page.

    Returns the context and a row per passage, which is what the page shows as
    the sources of an answer.
    """
    context_parts: list[str] = []
    source_rows: list[dict[str, Any]] = []

    if in_scope:
        lines = [f"Documents searched: {len(in_scope)} — " + ", ".join(in_scope)]
        for name in in_scope:
            said = (descriptions or {}).get(name)
            if said:
                lines.append(f"{name} — {said}")
        context_parts.append("\n".join(lines))

    for doc in documents:
        source = str(doc.metadata.get("source", "unknown"))
        page = whole_number(doc.metadata.get("page"))
        if page is not None:
            page = page + 1

        label = f"Source: {source}"
        if page is not None:
            label += f" | Page: {page}"

        context_parts.append(f"[{label}]\n{doc.page_content}")
        source_rows.append({
            "source": source,
            "page": page,
            "ranges": ranges_in(doc.metadata),
        })

    return "\n\n".join(context_parts), source_rows


def ranges_in(metadata: Mapping[str, Any]) -> list[list[int]]:
    """The character ranges a chunk covers, as pairs of offsets.

    A chunk without a usable range gives an empty list, so that a reader is not
    sent to a range that points somewhere else.
    """
    start, end = whole_number(metadata.get("start")), whole_number(metadata.get("end"))

    if start is not None and end is not None and start >= 0 and end > start:
        return [[start, end]]

    return []


def unique_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rows with the repeats merged.

    Two passages of the same document on the same page become one row, and their
    ranges are joined together. The order the rows first appeared in is kept.
    """
    unique: list[dict[str, Any]] = []
    at: dict[tuple[Any, Any], dict[str, Any]] = {}

    for row in rows:
        key = (row.get("source"), row.get("page"))
        kept = at.get(key)

        if kept is not None:
            kept["ranges"] += _as_ranges(row.get("ranges"))
            continue

        kept = {**row, "ranges": _as_ranges(row.get("ranges"))}
        at[key] = kept
        unique.append(kept)

    return unique


def _as_ranges(value: Any) -> list[list[int]]:
    """The value as a list of ranges, or an empty list for anything else."""
    if not isinstance(value, list):
        return []

    return [list(item) for item in value]


def build_graph(
    *,
    chat_model: BaseChatModel,
    retriever: BaseRetriever | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    in_scope: Sequence[str] | None = None,
    descriptions: Mapping[str, str] | None = None,
    grade: bool | None = None,
    attempts: int | None = None,
) -> CompiledStateGraph:
    """Build the graph of the steps and compile it.

    `chat_model` writes the rewritten question and the answer, and grades the
    material when grading is on. The retriever falls back to the one built from
    the settings. `in_scope` names the documents the search runs over and
    `descriptions` holds what the catalog says about them. The checkpointer
    stores the conversation between turns and defaults to one kept in memory.

    `grade` says whether the material is judged before an answer is written, and
    `attempts` how many searches one question may take. Both fall back to the
    settings.
    """
    if grade is None:
        grade = grading.enabled(settings)
    if attempts is None:
        attempts = settings.grade_attempts

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

        # The turn starts here, so the count of searches starts here as well. A
        # second question in the same conversation inherits the state left by
        # the first, and without this it would inherit its searches too.
        return {
            "contextualized_question": question,
            "search_query": question,
            "searches": 0,
        }

    async def retrieve(state: RAGState) -> dict[str, Any]:
        documents = await (retriever or get_retriever()).ainvoke(state["search_query"])
        context, source_rows = format_context(
            documents, in_scope=in_scope, descriptions=descriptions
        )

        return {
            "context": context,
            "retrieved_documents": source_rows,
            "searches": state["searches"] + 1,
        }

    async def answer(state: RAGState) -> dict[str, Any]:
        prompt = await answer_prompt.ainvoke({
            "question": state["contextualized_question"],
            "context": state["context"],
        })

        response = await chat_model.ainvoke(prompt)
        return {"messages": [response]}

    async def grade_material(state: RAGState) -> dict[str, Any]:
        verdict = await grading.agrade(
            chat_model, state["contextualized_question"], state["context"]
        )
        written = asdict(verdict)

        # The next search runs on the query the grader wrote, when it wrote one.
        # A verdict with no query does not send the flow back to the search, so
        # the query is left as it was.
        return {
            "verdict": written,
            "search_query": written["query"] or state["search_query"],
        }

    def after_grading(state: RAGState) -> str:
        """Where the turn goes once the material has been judged."""
        verdict = state["verdict"]

        if verdict["supported"]:
            return "answer"

        # The grader's own query is worth one more search, and only while there
        # are searches left. Past that the question is turned away rather than
        # answered from material that does not hold it.
        if verdict["query"] and state["searches"] < attempts:
            return "retrieve"

        return "unsupported"

    async def unsupported(state: RAGState) -> dict[str, Any]:
        prompt = await unsupported_prompt.ainvoke({
            "question": state["contextualized_question"],
            "reason": state["verdict"]["reason"],
        })

        response = await chat_model.ainvoke(prompt)
        return {"messages": [response]}

    builder = StateGraph(RAGState)
    builder.add_node("contextualize", contextualize)
    builder.add_node("retrieve", retrieve)
    builder.add_node("answer", answer)

    builder.add_edge(START, "contextualize")
    builder.add_edge("contextualize", "retrieve")
    builder.add_edge("answer", END)

    if grade:
        builder.add_node("grade", grade_material)
        builder.add_node("unsupported", unsupported)
        builder.add_edge("retrieve", "grade")
        builder.add_conditional_edges(
            "grade", after_grading, ["retrieve", "answer", "unsupported"]
        )
        builder.add_edge("unsupported", END)
    else:
        builder.add_edge("retrieve", "answer")

    return builder.compile(checkpointer=checkpointer or InMemorySaver())


@lru_cache(maxsize=1)
def get_app() -> CompiledStateGraph:
    """The graph the commands share, built once per process."""
    return build_graph(chat_model=build_chat_model())


def thread_config(thread_id: str) -> dict[str, Any]:
    """The configuration that names a conversation.

    Everything the checkpointer stores is filed under this thread id, which is
    what keeps two conversations apart.
    """
    return {"configurable": {"thread_id": thread_id}}


def get_thread_state(
    config: dict[str, Any], graph: CompiledStateGraph | None = None
):
    """The stored state of a conversation."""
    return (graph or get_app()).get_state(config)


async def read_thread_state(config: dict[str, Any], graph: CompiledStateGraph):
    """The stored state of a conversation, read asynchronously."""
    return await graph.aget_state(config)


def turns_from(state) -> list[dict[str, str]]:
    """A conversation as a list of turns, each with its role and its text.

    Messages from neither the user nor the model are left out, and so is a
    message whose content is not plain text.
    """
    turns: list[dict[str, str]] = []

    for message in state.values.get("messages", []):
        if message.type not in {"human", "ai"}:
            continue
        if not isinstance(message.content, str):
            continue
        turns.append({"role": message.type, "content": message.content})

    return turns
