from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from app.config import settings, validate_api_keys
from app.vectorstore import get_retriever


class RAGState(MessagesState):
    contextualized_question: str
    retrieved_documents: list[dict[str, Any]]
    context: str


validate_api_keys()

llm = ChatDeepSeek(
    model=settings.deepseek_model,
    temperature=0,
    max_retries=2,
    # Keep this first version focused on final-answer streaming rather than
    # showing DeepSeek's internal reasoning stream in the console.
    extra_body={"thinking": {"type": "disabled"}},
)

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

contextualize_chain = contextualize_prompt | llm | StrOutputParser()

answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You answer questions about a collection of documents.

Use only the context provided below. If it does not hold enough information to
answer, say so clearly instead of guessing, and do not fill the gaps from your
own knowledge. Never invent facts, figures, names, procedures or rules that are
not in the context.

Answer in the language the question is written in, clearly and directly.""",
    ),
    (
        "human",
        """Question:
{question}

Context:
{context}""",
    ),
])


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
    retriever = get_retriever()
    docs = await retriever.ainvoke(state["contextualized_question"])

    context_parts: list[str] = []
    source_rows: list[dict[str, Any]] = []

    for doc in docs:
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

    return {
        "context": "\n\n".join(context_parts),
        "retrieved_documents": source_rows,
    }


async def answer(state: RAGState) -> dict[str, Any]:
    prompt = await answer_prompt.ainvoke({
        "question": state["contextualized_question"],
        "context": state["context"],
    })

    response = await llm.ainvoke(prompt)
    return {"messages": [response]}


builder = StateGraph(RAGState)
builder.add_node("contextualize", contextualize)
builder.add_node("retrieve", retrieve)
builder.add_node("answer", answer)

builder.add_edge(START, "contextualize")
builder.add_edge("contextualize", "retrieve")
builder.add_edge("retrieve", "answer")
builder.add_edge("answer", END)

checkpointer = InMemorySaver()
app = builder.compile(checkpointer=checkpointer)


def get_thread_state(config: dict[str, Any]):
    return app.get_state(config)
