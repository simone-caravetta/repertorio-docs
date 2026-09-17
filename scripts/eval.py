"""Measure the search and the answers over a set of questions.

Every question in the set names the document that answers it, and sometimes the
page as well, so what the search returned can be compared with what was
expected. The retrieval numbers come from that comparison alone and cost nothing
to produce.

An answer to each question is written only with `--answers`, and a model reads
each answer against the passages it was written from only with `--judge`. Both
of those call the model once or twice per question.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from app.catalog import Catalog
from app.chat_model import build_chat_model
from app.config import describe_vector_store, settings, short_path
from app.evals import (
    NDCG_CUTOFF,
    EvalReport,
    QuestionResult,
    load_questions,
    run_evals,
    summarise,
)
from app.rerank import describe_rerank
from app.scope import build_scoped_retriever, resolve_scope


def evaluate(
    *,
    questions_path: Path | None = None,
    category: str | None = None,
    document: str | None = None,
    documents: list[str] | None = None,
    answers: bool = False,
    judge: bool = False,
    documents_dir: Path | None = None,
    db_path: Path | None = None,
    chat_model: BaseChatModel | None = None,
    judge_model: BaseChatModel | None = None,
) -> EvalReport:
    """Run the question set and return what came back.

    The set and the scope are resolved first, so that a set which cannot be read
    or a category that is not there stops the run before anything is searched.
    With `answers` a model writes an answer to each question, and with `judge`
    the answers are read against the passages they were written from.
    """
    config = replace(
        settings, documents_dir=Path(documents_dir or settings.documents_dir)
    )
    questions_path = Path(questions_path or config.eval_questions_path)
    db_path = Path(db_path or config.catalog_db_path)

    try:
        questions = load_questions(questions_path)

        # The scope is resolved before the first search, so that a name the
        # catalog does not know stops the run with a message.
        scope = resolve_scope(
            Catalog(db_path, create=False),
            config=config,
            category=category,
            document=document,
            documents=documents or [],
        )
    except (LookupError, RuntimeError, ValueError) as exc:
        # The reason goes to the console as it is, and the run stops here.
        raise SystemExit(str(exc)) from exc

    answers = answers or judge

    print(f"store     {describe_vector_store(config)}")
    print(f"catalog   {short_path(db_path)}")
    print(f"questions {short_path(questions_path)} — {len(questions)}")
    print(f"scope     {scope.label}")
    print(f"rerank    {describe_rerank(config)}")

    if answers:
        said = f"{len(questions)} searched, {len(questions)} answered"
        if judge:
            said += f", {len(questions)} judged"
        model = chat_model or build_chat_model()
    else:
        said = f"{len(questions)} searched, no model call"
        model = None

    print(f"run       {said}\n")

    report = run_evals(
        questions,
        # The retriever is asked for at least NDCG_CUTOFF passages, because the
        # nDCG is measured over the first ten of them.
        retriever=build_scoped_retriever(
            scope, k=max(config.retrieval_k, NDCG_CUTOFF)
        ),
        # The ranking numbers are read from the first RETRIEVAL_K passages,
        # which is how many the console shows. A scope that is read whole comes
        # back in reading order, so nothing is cut off here and the report says
        # so through `ranked`.
        read_at=config.retrieval_k if scope.ranked else None,
        in_scope=scope.documents,
        descriptions=dict(scope.descriptions),
        chat_model=model,
        judge_model=(judge_model or model) if judge else None,
    )

    print_report(report, ranked=scope.ranked)
    return report


def print_report(report: EvalReport, *, ranked: bool = True) -> None:
    """Print one line per question, then the numbers over the whole set.

    The nDCG is left out when `ranked` is false, because the passages came back
    in reading order and their order says nothing about the search.
    """
    summary = summarise(report)

    for result in report.results:
        print(f"{result.question.id:<24} {describe_result(result)}")

    print(
        f"\n{summary.questions} questions, {summary.asked} asked, "
        f"{summary.skipped} not in scope"
    )
    if summary.failed:
        print(f"failed    {summary.failed} could not be run")
    if summary.measurable:
        said = f"{_rate(summary.hits, summary.measurable)} documents"
        if summary.page_asked:
            said += f", {_rate(summary.page_hits, summary.page_asked)} pages"
        said += f", MRR {summary.reciprocal_rank:.2f}"
        if ranked:
            said += f", nDCG@{NDCG_CUTOFF} {summary.ndcg:.2f}"
        print(f"retrieval {said}")
    if summary.answered:
        print(
            f"answers   {summary.answered} written, "
            f"{summary.complete} as expected, "
            f"{summary.answered - summary.complete} missing text"
        )
    if summary.judged or summary.unjudged:
        print(
            f"judge     {summary.judged} read, {summary.faithful} faithful, "
            f"{summary.unjudged} not answered"
        )


def describe_result(result: QuestionResult) -> str:
    """One question as a line, with what was found and what was written.

    Every part that was measured appears on the line. A question the scope left
    out says so, and a question that failed says why.
    """
    question = result.question

    if not result.asked:
        return "not in scope"
    if result.error is not None:
        return f"failed: {result.error}"

    parts: list[str] = []

    if question.document is not None:
        parts.append(_found("", result.rank))
    if question.page is not None:
        parts.append(_found(f"p.{question.page} ", result.page_rank))
    if result.answer is not None:
        parts.append(
            "answer ok"
            if not result.missing
            else "answer missing " + ", ".join(
                f'"{text}"' for text in result.missing
            )
        )
    if result.faithful is True:
        parts.append("faithful")
    elif result.faithful is False:
        parts.append("not faithful: " + " | ".join(result.unsupported))
    elif result.judge_error is not None:
        parts.append(f"not judged: {result.judge_error}")

    return "  ".join(parts) or "nothing measured"


def _found(label: str, rank: int | None) -> str:
    """Where the passage was found, as 'hit' with its position or 'miss'."""
    return f"{label}hit {rank}" if rank else f"{label}miss"


def _rate(part: int, whole: int) -> str:
    """One count over another, as a share of a hundred, rounded."""
    return f"{part}/{whole} ({round(100 * part / whole)}%)"


def parse_args() -> argparse.Namespace:
    """The command line, with the scope arguments checked against one another."""
    parser = argparse.ArgumentParser(
        description=(
            "Ask a set of questions of the library and measure what the search "
            "found. Retrieval is measured on its own, and an answer to each "
            "question is written only when asked for."
        )
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=settings.eval_questions_path,
        help="the question set to measure (default: %(default)s)",
    )
    parser.add_argument(
        "--category",
        metavar="NAME",
        help="ask the set only about a category and what is filed below it; '' "
        "for the documents in no category",
    )
    parser.add_argument(
        "--document",
        metavar="PATH",
        help="ask the set only about one document",
    )
    parser.add_argument(
        "--documents",
        nargs="+",
        metavar="PATH",
        help="ask the set only about these documents",
    )
    parser.add_argument(
        "--answers",
        action="store_true",
        help="write an answer to each question, one model call each, and check it "
        "against what the set says the answer holds",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="have a model read each answer against the passages it was written "
        "from, a second call per question; this is the faithfulness number, and "
        "it writes the answers too",
    )
    parser.add_argument(
        "--documents-dir",
        type=Path,
        default=settings.documents_dir,
        help="folder the documents sit in (default: %(default)s)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=settings.catalog_db_path,
        help="catalog database (default: %(default)s)",
    )

    args = parser.parse_args()

    # The three scope arguments are alternatives. More than one of them leaves
    # no way to tell which scope was meant.
    given = [
        name
        for name, was_given in (
            ("--category", args.category is not None),
            ("--document", args.document is not None),
            ("--documents", bool(args.documents)),
        )
        if was_given
    ]
    if len(given) > 1:
        parser.error("give one of " + ", ".join(given) + ", not several")

    return args


def main() -> None:
    """Run the eval command."""
    args = parse_args()
    evaluate(
        questions_path=args.questions,
        category=args.category,
        document=args.document,
        documents=args.documents,
        answers=args.answers,
        judge=args.judge,
        documents_dir=args.documents_dir,
        db_path=args.db,
    )


if __name__ == "__main__":
    main()
