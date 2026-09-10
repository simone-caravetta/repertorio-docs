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
        """Sei un assistente che prepara query per la ricerca documentale aziendale.

Considera la cronologia della conversazione e la nuova domanda dell'utente.
Riscrivi la nuova domanda come una domanda autonoma, risolvendo riferimenti
come 'questo', 'quello', 'e per l'estero?', 'quanto costa?' quando il contesto
precedente permette di capirli.

Non rispondere alla domanda. Restituisci esclusivamente la query autonoma.""",
    ),
    (
        "human",
        """Conversazione precedente:
{history}

Nuova domanda:
{question}""",
    ),
])

contextualize_chain = contextualize_prompt | llm | StrOutputParser()

answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Sei un assistente per la ricerca di informazioni aziendali.

Rispondi alla domanda usando esclusivamente il contesto documentale fornito.
Se il contesto non contiene informazioni sufficienti, dichiaralo chiaramente.
Non inventare policy, numeri, procedure o fatti non presenti nei documenti.
Rispondi in italiano in modo chiaro e diretto.""",
    ),
    (
        "human",
        """Domanda:
{question}

Contesto documentale:
{context}""",
    ),
])


async def contextualize(state: RAGState) -> dict[str, Any]:
    messages = state["messages"]
    latest = messages[-1]
    if not isinstance(latest, HumanMessage):
        raise TypeError("L'ultimo messaggio deve essere una domanda dell'utente.")

    history = messages[:-1]
    history_text = "\n".join(
        f"{message.type}: {message.content}" for message in history
    )

    question = await contextualize_chain.ainvoke({
        "history": history_text or "(nessuna conversazione precedente)",
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

        label = f"Fonte: {source}"
        if page is not None:
            label += f" | Pagina: {page}"

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
