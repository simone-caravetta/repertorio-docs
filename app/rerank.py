"""Scoring what a search found, by a model that reads the question and the passage.

An embedding is one vector per text, computed before the question is known, so
the best a similarity search can say is that two texts are near each other in a
space neither of them was placed in with the other in mind. A cross-encoder reads
the question and the passage *together* and answers for that pair, which is the
thing an embedding cannot do and the reason this step exists. It is slow per
pair — a forward pass each — so it is affordable over the twenty or so a search
hands over and not over a library.

Reranking is therefore two numbers rather than one: how many passages the search
is asked for, and how many of them the answer is written from. `RerankedRetriever`
asks the retriever it wraps for the first and returns the best of them, so
everything downstream — the graph, the commands, the web app, the eval harness —
goes on taking a retriever and knowing nothing about any of this.

The model is built on the first rerank and not before, and not at all when
`RERANK` is off: it is a checkpoint of about two gigabytes, and a command that
searches without reranking has no reason to pay for it.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.config import Settings

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
# The whole of what RERANK may say. A third value is refused rather than read as
# one of these two: a search that silently was not reranked, or a model loaded and
# then not used, are both worth a sentence at the point somebody wrote it.
MODES = ("on", "off")


def enabled(config: Settings) -> bool:
    """Whether searches are reranked, as the setting says."""
    mode = config.rerank.strip().lower()

    if mode == "on":
        return True
    if mode == "off":
        return False

    raise RuntimeError(
        f"Unknown RERANK {config.rerank!r}: expected " + " or ".join(MODES) + "."
    )


def model_name(config: Settings) -> str:
    """The model this configuration reranks with, by its own name.

    The configured name when there is one and this project's default when there
    is not, so that a machine which leaves `RERANK_MODEL` empty and one which
    spells the default out are seen to be running the same model. This is the
    name a command prints, and the name a comparison of two runs is read by.
    """
    return config.rerank_model or RERANK_MODEL


def describe_rerank(config: Settings) -> str:
    """Reranking in one line, for a command to print at the top of a run.

    Both numbers, because a reranked run and a plain one are handed a different
    question by the store and their output is therefore not comparable unless the
    line says which pass produced it.

    A mode that is neither on nor off is printed as it was written rather than
    refused here: this line describes a run, and a run stops where the mode would
    have to be honoured — in the search, with a message saying which two words it
    wanted. A command must not fail while describing itself.
    """
    mode = config.rerank.strip().lower()
    if mode != "on":
        return mode

    return (
        f"{model_name(config)} — top {config.retrieval_k} "
        f"of {config.rerank_candidates}"
    )


@lru_cache(maxsize=1)
def get_reranker(config: Settings) -> Any:
    """Build the model a search is reranked with, once per process.

    `config` is an argument rather than a default bound to this module's
    `settings`, for the reason `app.vectorstore.get_vectorstore` reads its own in
    the body: a default is bound once, at import, and a caller which replaced the
    settings would keep being handed the model of the machine's `.env`.

    The import is inside the function. Everything here already reaches torch
    through `app.embeddings`, so this is not about the import: it is about the
    checkpoint, which is two gigabytes and a few seconds nobody who turned
    reranking off should spend.
    """
    from sentence_transformers.cross_encoder import CrossEncoder

    return CrossEncoder(model_name(config), device=config.rerank_device)


class RerankedRetriever(BaseRetriever):
    """What a retriever found, in the order the reranker puts it in.

    The retriever this wraps is asked for the candidates, which is the search
    exactly as it has always been — same query, same filter over the same
    documents — and every passage it returns is scored against the question. The
    best `k` are handed on; the rest are dropped, and were only ever there to be
    looked at.

    Two things it deliberately does not do. It does not rerank a single passage,
    because an order of one cannot be improved and the forward pass would be
    spent for nothing. And it does not break a tie by itself: the sort is stable
    and reads only the score, so passages the model scores equally stay in the
    order the similarity search put them — a ranking refined, not replaced by an
    arbitrary one.
    """

    retriever: BaseRetriever
    reranker: Any
    k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        # `invoke` rather than the private method, and with the callbacks handed
        # on: the inner search is a run of its own, and a caller's handlers should
        # see it rather than a search that happened with nobody watching.
        found = self.retriever.invoke(
            query, config={"callbacks": run_manager.get_child()}
        )
        if len(found) < 2:
            return found

        scores = self.reranker.predict(
            [(query, one.page_content) for one in found],
            show_progress_bar=False,
        )
        ranked = sorted(zip(scores, found, strict=True), key=lambda pair: -pair[0])

        return [one for _, one in ranked][: self.k]
