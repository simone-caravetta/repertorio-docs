"""Measure the library against questions whose answers are known.

A question set names, for each question, the document the answer should come from
and optionally the page, and optionally what a correct answer says. This runs
them: the search first, which is the measurement retrieval quality is made of and
costs nothing but an embedding per question, and then, if asked, an answer written
from the passages that search returned and read against what the set expected.

`--answers` writes an answer to each question, one model call each. `--judge`
adds a second call per question, to a model that reads the answer against its
context and says whether every claim in it is one the context supports — the
faithfulness number, and the only part of a run that costs as much as it does.
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
    """Run a question set against a scope, print what it found, and return it."""
    # The folder the run was pointed at, as a config rather than as an argument:
    # a scope resolves the paths a question set writes against it, so pointing the
    # command at another folder has to move what a path is relative to.
    config = replace(
        settings, documents_dir=Path(documents_dir or settings.documents_dir)
    )
    questions_path = Path(questions_path or config.eval_questions_path)
    db_path = Path(db_path or config.catalog_db_path)

    try:
        questions = load_questions(questions_path)
        # The console's own resolution: a set asked of a category or of one
        # document is measured through the same scope the question would be asked
        # in, which is the only way the numbers are about the console.
        scope = resolve_scope(
            Catalog(db_path, create=False),
            config=config,
            category=category,
            document=document,
            documents=documents or [],
        )
    except (LookupError, RuntimeError, ValueError) as exc:
        # A key that is missing, a category that is not there, a question set with
        # a comma out of place: a message to read, not a stack trace.
        raise SystemExit(str(exc)) from exc

    # Judging an answer needs an answer, so asking for the judge asks for both.
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
        # The search is asked for the wider of the console's k and the cutoff the
        # ranking is read to. Asking for five and reporting nDCG@10 would be a
        # number about passages that were never retrieved; asking for ten and
        # reporting the hit-rate over five is the same measurement it always was,
        # because the first five of ten are the five of five. A whole-document
        # scope ignores this — what it returns is the document, not a count of
        # passages from it.
        retriever=build_scoped_retriever(
            scope, k=max(config.retrieval_k, NDCG_CUTOFF)
        ),
        # ... and what was *found* is read over the console's own five, because
        # that is what the answer is written from. Asking the search for ten is a
        # measurement instrument reaching wider than the console so that a
        # ranking has an end to be read to; letting the four numbers about
        # finding something be read over that wider pool would report documents
        # found and never shown. A whole-document scope is read whole: its
        # passages are the document, and there is no fifth of it to stop at.
        read_at=config.retrieval_k if scope.ranked else None,
        in_scope=scope.documents,
        descriptions=dict(scope.descriptions),
        chat_model=model,
        judge_model=(judge_model or model) if judge else None,
    )

    print_report(report, ranked=scope.ranked)
    return report


def print_report(report: EvalReport, *, ranked: bool = True) -> None:
    """What the run found, question by question and then added up.

    `ranked` says whether the passages came back as a ranking. A whole-document
    scope hands over the document in reading order, so a number about where in a
    ranking the answer sat means nothing for it and is left out rather than
    printed as a figure that looks like the others.
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
    """What one question did, on one line.

    The three measures are independent and each is left out when it does not
    apply: a question with no page to find is not a page that was missed, and a
    question the library is not expected to answer has no document to have found.
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
    return f"{label}hit {rank}" if rank else f"{label}miss"


def _rate(part: int, whole: int) -> str:
    """A share of the questions, as a count and the percentage it is.

    Only ever called with questions behind it: a measure with nothing to be taken
    over is left out of the line rather than printed as a rate of nothing.
    """
    return f"{part}/{whole} ({round(100 * part / whole)}%)"


def parse_args() -> argparse.Namespace:
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

    # `--documents` takes several values, so argparse cannot be asked to make
    # these exclusive; said here instead.
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
