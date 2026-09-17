"""Reranking of search results by a cross-encoder.

A vector search compares embeddings that were computed before the question was
known. A cross-encoder reads the question and the passage together and scores
that pair directly. The score is better, but every pair costs a forward pass,
so this runs only on the passages a search already returned.

The retriever asks for RERANK_CANDIDATES passages and keeps the best
RETRIEVAL_K of them. Everything downstream still receives an ordinary retriever.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.config import Settings

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
# The two values RERANK accepts. Any other value is refused with a message that
# names them.
MODES = ("on", "off")


def enabled(config: Settings) -> bool:
    """Whether searches are reranked.

    RERANK has to be "on" or "off". Any other value raises an error listing the
    two words that are accepted.
    """
    mode = config.rerank.strip().lower()

    if mode == "on":
        return True
    if mode == "off":
        return False

    raise RuntimeError(
        f"Unknown RERANK {config.rerank!r}: expected " + " or ".join(MODES) + "."
    )


def model_name(config: Settings) -> str:
    """The name of the model used for reranking.

    When RERANK_MODEL is empty this falls back to the default of the project.
    A machine that leaves the setting blank and a machine that writes the
    default out then run the same model. Commands print this name and two runs
    are compared by it.
    """
    return config.rerank_model or RERANK_MODEL


def describe_rerank(config: Settings) -> str:
    """One line about reranking, for a command to print when it starts.

    The line carries both numbers because the store is asked for a different
    number of passages depending on the mode. Two reports can only be compared
    when the line says which pass produced them.

    A value that is not "on" is printed as it was written. The run stops later,
    in the search itself, where the mode has to be honoured.
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
    """Build the cross-encoder once per process.

    The settings arrive as an argument instead of being read from the module,
    so that a caller which replaced them gets the model it asked for.

    The import sits inside the function because the checkpoint weighs about two
    gigabytes and takes a few seconds to load.
    """
    from sentence_transformers.cross_encoder import CrossEncoder

    return CrossEncoder(model_name(config), device=config.rerank_device)


class RerankedRetriever(BaseRetriever):
    """A retriever that reorders what another retriever found.

    The wrapped retriever runs the same search as always, with the same query
    and the same filter over the same documents. Every passage it returns is
    scored against the question and the best `k` are handed on.

    A single passage is returned unchanged. The sort reads only the score and
    keeps passages with equal scores in the order the search produced them.
    """

    retriever: BaseRetriever
    reranker: Any
    k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        # The callbacks are passed on, so the inner search appears as a run of
        # its own to whoever is watching.
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
