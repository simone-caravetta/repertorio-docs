"""Measure an answer pipeline against questions whose answers are known.

A question set is a JSON list. Each question names the document that holds the
answer, and usually the page as well, and lists phrases an answer should
contain.

Running the set puts every question to a retriever. When a chat model is given,
the question is asked of it too, using the same context the chat endpoint would
build. What comes back is recorded question by question: where the document
landed in the ranking, which expected phrases the answer left out, and, when a
judge model is given, whether the answer stayed within the passages it was
written from.

A question may name no document, which is how a question set says the library
is not expected to answer it. Those questions are the ones a grader model is
worth asking about, and with a grader every question's material is read and
recorded as a verdict, so a run reports how often the material was accepted
where an answer was there and refused where none was.

Summary turns those results into the counts and scores a run is compared on.
Finding is scored by where the right document and page appear among the
passages, and answers are scored by the phrases they contain and by the judge's
verdict.

The questions come from a file, so the same module can measure any library a
question set has been written for.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import log2
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from app.grading import Verdict, grade
from app.ingestion import whole_number
from app.rag_graph import answer_prompt, format_context

# The keys a question may carry. A file with any other key is refused, so a
# misspelled field is reported instead of being ignored.

FIELDS = ("id", "question", "document", "page", "contains", "note")


# How many of the returned passages the nDCG is computed over.

NDCG_CUTOFF = 10

# Asked of a second model, which says whether the answer stays within the
# context it was written from.

judge_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You check one answer against the passages it was written from.

You are given the question, the context the answer was written from, and the
answer. Say whether every claim the answer makes about the documents is one the
context supports.

An answer that says the context does not hold what was asked makes no claim
about the documents and is supported. An answer that draws on anything outside
the context — a fact the passages do not state, a number that is not there, a
rule that is a reasonable guess — is not.

Reply with one word, "yes" or "no", and then, if the answer is no, the claims
that are not supported, one per line, as they appear in the answer. Nothing
else.""",
    ),
    (
        "human",
        """Question:
{question}

Context:
{context}

Answer:
{answer}""",
    ),
])


@dataclass(frozen=True)
class Question:
    """One question from a question set.

    `document` is the file in the library that holds the answer, or None for a
    question the library is not expected to answer. `page` narrows it to one
    page of that file, counted from one.

    `contains` lists phrases an answer has to include, compared with whitespace
    collapsed and case ignored. `note` is free text for whoever reads the set,
    and nothing checks it.
    """

    id: str
    question: str
    document: str | None = None
    page: int | None = None
    contains: tuple[str, ...] = ()
    note: str = ""


@dataclass
class QuestionResult:
    """What one question did when it was run.

    `asked` is False for a question left out because its document was not in
    scope. `rank` is where the right document appeared among the passages and
    `page_rank` the same for the right page. `grades` scores the passages that
    came back and `sources` is the source rows the context was built from.

    `answer` is what the model wrote and `missing` the expected phrases it left
    out. A judge fills in `faithful` and the claims it found unsupported.
    `error`, `judge_error` and `grade_error` hold the message when a call failed.

    A grader fills in `verdict`, which is what it decided about the material
    this question came back with. A question naming no document is expected to
    be refused.
    """

    question: Question
    asked: bool = True
    rank: int | None = None
    page_rank: int | None = None
    grades: tuple[int, ...] = ()
    sources: list[dict[str, Any]] = field(default_factory=list)
    answer: str | None = None
    missing: list[str] = field(default_factory=list)
    faithful: bool | None = None
    unsupported: list[str] = field(default_factory=list)
    verdict: Verdict | None = None
    judge_error: str | None = None
    grade_error: str | None = None
    error: str | None = None


@dataclass
class EvalReport:
    """One run of a question set: a result per question, in the order asked."""

    results: list[QuestionResult] = field(default_factory=list)


@dataclass(frozen=True)
class Summary:
    """The counts and scores of a run.

    `measurable` counts the questions that were asked and name a document,
    since only those can be scored on finding it. `hits` counts the ones whose
    document came back at all and `page_hits` the ones whose page did too.
    `reciprocal_rank` and `ndcg` are averaged over the measurable questions.

    `complete` counts the answers holding every expected phrase and `faithful`
    the ones the judge accepted. `unjudged` counts the judge calls that failed.

    `accepted` counts the questions naming a document whose material the grader
    said could answer them, and `refused_unanswerable` the questions naming none
    whose material the grader turned away. Both are what the grader was right
    about, and they are read against the questions of each kind. `ungraded`
    counts the grading calls that failed.
    """

    questions: int
    asked: int
    skipped: int
    failed: int
    measurable: int
    hits: int
    page_asked: int
    page_hits: int
    reciprocal_rank: float
    ndcg: float
    answered: int
    complete: int
    judged: int
    faithful: int
    unjudged: int
    accepted: int
    refused_unanswerable: int
    ungraded: int


def load_questions(path: Path) -> list[Question]:
    """Read a question set from a JSON file.

    Raises ValueError, with a message naming the problem, when the file is
    missing, is not JSON, or holds a question that is not shaped right.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"No question set at {path}.") from exc

    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} is not valid JSON: line {exc.lineno}, {exc.msg}."
        ) from exc

    problem = _set_problem(loaded, path)
    if problem is not None:
        raise ValueError(problem)

    questions = [
        _question(entry, number, path)
        for number, entry in enumerate(loaded, start=1)
    ]

    seen: set[str] = set()
    for question in questions:
        if question.id in seen:
            raise ValueError(
                f"{path} has two questions with the id {question.id!r}. An id is "
                "what tells two runs apart."
            )
        seen.add(question.id)

    return questions


def _set_problem(loaded: Any, path: Path) -> str | None:
    """The problem with the loaded file, or None when it is a list."""

    if not isinstance(loaded, list):
        return (
            f"{path} holds a {type(loaded).__name__}; a question set is a list "
            "of questions."
        )

    return None


def _question(entry: Any, number: int, path: Path) -> Question:
    """Turn one entry of the file into a Question, or raise ValueError."""

    where = f"question {number} of {path}"
    problem = _what_is_wrong(entry, where)

    if problem is not None:
        raise ValueError(problem)

    return Question(
        id=entry["id"],
        question=entry["question"],
        document=entry.get("document"),
        page=entry.get("page"),
        contains=tuple(entry.get("contains", [])),
        note=entry.get("note", ""),
    )


def _what_is_wrong(entry: Any, where: str) -> str | None:
    """The problem with one question entry, or None when it is well formed.

    `where` names the entry, so the message points the reader at the part of
    the file to fix.
    """

    if not isinstance(entry, dict):
        return f"The {where} is a {type(entry).__name__}, not an object."

    unknown = [name for name in entry if name not in FIELDS]
    if unknown:
        return (
            f"The {where} has {_listed(unknown)} in it, which is not a field a "
            f"question has. A question is {_listed(FIELDS)}."
        )

    for name in ("id", "question"):
        if not isinstance(entry.get(name), str) or not entry[name].strip():
            return f"The {where} has no {name}."

    document = entry.get("document")
    if document is not None and (
        not isinstance(document, str) or not document.strip()
    ):
        return (
            f"The {where} has a document that is not a path. Leave the field out "
            "for a question the library does not answer."
        )

    page = entry.get("page")
    if page is not None and (
        not isinstance(page, int) or isinstance(page, bool) or page < 1
    ):
        return (
            f"The {where} has a page of {page!r}; pages are counted from one, as "
            "a reader counts them."
        )

    if not _is_text_list(entry.get("contains", [])):
        return f"The {where} has a contains that is not a list of strings."

    if not isinstance(entry.get("note", ""), str):
        return f"The {where} has a note that is not a string."

    return None


def _is_text_list(value: Any) -> bool:
    """Whether the value is a list of strings that are not blank."""

    return isinstance(value, list) and all(
        isinstance(text, str) and text.strip() for text in value
    )


def _listed(names: Iterable[str]) -> str:
    """The names as one phrase, for a message that lists the fields."""

    ordered = sorted(names)
    if len(ordered) == 1:
        return ordered[0]
    return ", ".join(ordered[:-1]) + f" or {ordered[-1]}"


def run_evals(
    questions: Sequence[Question],
    *,
    retriever: BaseRetriever,
    read_at: int | None = None,
    in_scope: Sequence[str] | None = None,
    descriptions: Mapping[str, str] | None = None,
    chat_model: BaseChatModel | None = None,
    judge_model: BaseChatModel | None = None,
    grade_model: BaseChatModel | None = None,
) -> EvalReport:
    """Run a question set and collect what each question did.

    Only the retriever is required. With no chat model the run measures finding
    alone, and with no judge model it records no verdict on the answers. With a
    grader model every question's material is read and recorded as a verdict,
    which needs no answer to have been written.

    `read_at` limits the ranking numbers to the first N passages returned. That
    matters when a reranker hands back more passages than the retriever was
    asked for, since those extras were never read by anyone. `in_scope` works
    as it does in a chat: a question about a document outside it is skipped.
    `descriptions` is the catalog text for the documents in scope, and it goes
    into the context the answer is written from.
    """

    results: list[QuestionResult] = []

    for question in questions:
        if not _in_scope(question, in_scope):
            results.append(QuestionResult(question=question, asked=False))
            continue

        results.append(
            _run_one(
                question,
                retriever=retriever,
                read_at=read_at,
                in_scope=in_scope,
                descriptions=descriptions,
                chat_model=chat_model,
                judge_model=judge_model,
                grade_model=grade_model,
            )
        )

    return EvalReport(results=results)


def _in_scope(question: Question, in_scope: Sequence[str] | None) -> bool:
    if question.document is None or in_scope is None:
        return True

    return question.document in in_scope


def _run_one(
    question: Question,
    *,
    retriever: BaseRetriever,
    read_at: int | None,
    in_scope: Sequence[str] | None,
    descriptions: Mapping[str, str] | None,
    chat_model: BaseChatModel | None,
    judge_model: BaseChatModel | None,
    grade_model: BaseChatModel | None,
) -> QuestionResult:
    """Run one question. A call that fails is recorded, not raised."""

    try:
        passages = retriever.invoke(question.question)
    except Exception as exc:  # noqa: BLE001 - one question's failure, not the run's
        return QuestionResult(question=question, error=str(exc))

    context, sources = format_context(
        passages, in_scope=in_scope, descriptions=descriptions
    )

    # The ranking is measured over the passages a chat would have read, which
    # is the first read_at of them when a limit was given.

    shown = passages if read_at is None else passages[:read_at]

    # Where the answering document landed, counting from one, or None when it
    # did not come back at all. The page is looked for only when the question
    # names one.

    found = rank_of(shown, question.document)

    result = QuestionResult(
        question=question,
        rank=found,
        page_rank=(
            rank_of(shown, question.document, page=question.page)
            if question.page is not None
            else None
        ),
        grades=tuple(_grade(passage, question) for passage in passages),
        sources=sources,
    )

    # The verdict comes before the answer, as it does in the graph, and it is
    # about the material rather than about anything written from it.

    if grade_model is not None:
        try:
            result.verdict = grade(grade_model, question.question, context)
        except Exception as exc:  # noqa: BLE001 - same
            result.grade_error = str(exc)

    if chat_model is None:
        return result

    try:
        result.answer = write_answer(chat_model, question.question, context)
    except Exception as exc:  # noqa: BLE001 - same
        result.error = str(exc)
        return result

    result.missing = missing_from(result.answer, question.contains)

    if judge_model is not None:
        try:
            result.faithful, result.unsupported = judge_answer(
                judge_model, question.question, context, result.answer
            )
        except Exception as exc:  # noqa: BLE001 - same
            result.judge_error = str(exc)

    return result


def rank_of(
    passages: Sequence[Document],
    document: str | None,
    *,
    page: int | None = None,
) -> int | None:
    """Where a document sits among the passages, counting from one.

    With a page, the passage also has to be on that page. Returns None when no
    document was asked for, and when the document is not among the passages.
    """

    if document is None:
        return None

    for position, passage in enumerate(passages, start=1):
        if passage.metadata.get("source") != document:
            continue
        if page is not None and page_of(passage) != page:
            continue
        return position

    return None


def page_of(passage: Document) -> int | None:
    """The page a passage is on, counted from one as a reader counts pages.

    Pages are stored counting from zero and metadata comes back from a store as
    whatever type it kept, so None is returned when the stored value is not a
    whole number.
    """

    page = whole_number(passage.metadata.get("page"))

    return None if page is None else page + 1


def _grade(passage: Document, question: Question) -> int:
    """How well one passage answers a question.

    2 when it is on the page the question names, 1 when it comes from the right
    document but another page, and 0 when it comes from another document. A
    question that names no document grades every passage 0.
    """

    if question.document is None:
        return 0

    if passage.metadata.get("source") != question.document:
        return 0

    if question.page is None:
        return 1

    return 2 if page_of(passage) == question.page else 1


def ndcg_at(grades: Sequence[int], *, cutoff: int = NDCG_CUTOFF) -> float:
    """Score a ranking by what it put near the top.

    Each grade is divided by the log of the position it was found at, so a good
    passage found early counts for more than the same passage found late. The
    total is divided by the best total the same grades could have reached, in
    the order that scores highest. Dividing by that ideal makes one question's
    score comparable with another's.

    Only the first `cutoff` grades are read. Returns 0.0 when the cutoff is not
    positive, and when every grade read is 0.
    """

    if cutoff <= 0:
        return 0.0

    read = list(grades[:cutoff])
    ideal = sum(
        grade / log2(position + 1)
        for position, grade in enumerate(sorted(read, reverse=True), start=1)
    )
    if ideal <= 0:
        return 0.0

    gained = sum(
        grade / log2(position + 1)
        for position, grade in enumerate(read, start=1)
    )

    return gained / ideal


def missing_from(answer: str, expected: Sequence[str]) -> list[str]:
    """The expected phrases that an answer does not contain.

    Both sides are compared with runs of whitespace collapsed and case ignored,
    so an answer is not marked down for the way it was spaced.
    """

    flat = " ".join(answer.split()).casefold()

    return [
        text for text in expected if " ".join(text.split()).casefold() not in flat
    ]


def write_answer(chat_model: BaseChatModel, question: str, context: str) -> str:
    """Ask the chat model a question, giving it the context as its only source.

    This uses the prompt the chat endpoint uses, so a run measures the answers
    the app would actually give.
    """

    reply = chat_model.invoke(answer_prompt.invoke({
        "question": question,
        "context": context,
    }))
    content = reply.content

    return content if isinstance(content, str) else str(content)


def judge_answer(
    judge_model: BaseChatModel, question: str, context: str, written: str
) -> tuple[bool | None, list[str]]:
    """Put a written answer to the judge and read its reply."""

    reply = judge_model.invoke(judge_prompt.invoke({
        "question": question,
        "context": context,
        "answer": written,
    }))
    content = reply.content

    return read_verdict(content if isinstance(content, str) else str(content))


def read_verdict(reply: str) -> tuple[bool | None, list[str]]:
    """Read what the judge wrote.

    The first line carries the verdict. Any line after it names a claim the
    judge found unsupported. Returns None for the verdict when the first line
    is neither yes nor no.
    """

    lines = [line.strip() for line in reply.strip().splitlines() if line.strip()]
    if not lines:
        return None, []

    verdict = lines[0].casefold().strip(".,:;!*`\"' ")
    if verdict.startswith("yes"):
        return True, []
    if verdict.startswith("no"):
        return False, lines[1:]

    return None, []


def summarise(report: EvalReport) -> Summary:
    """Count and average a run's results."""

    asked = [result for result in report.results if result.asked]
    measurable = [
        result for result in asked if result.question.document is not None
    ]
    found = [result.rank for result in measurable if result.rank is not None]
    paged = [result for result in measurable if result.question.page is not None]
    answers = [result for result in asked if result.answer is not None]
    graded = [result for result in asked if result.verdict is not None]

    return Summary(
        questions=len(report.results),
        asked=len(asked),
        skipped=len(report.results) - len(asked),
        failed=len([result for result in asked if result.error is not None]),
        measurable=len(measurable),
        hits=len(found),
        page_asked=len(paged),
        page_hits=len([r for r in paged if r.page_rank is not None]),
        reciprocal_rank=(
            sum(1 / rank for rank in found) / len(measurable) if measurable else 0.0
        ),

        # Every measurable question counts here, including the ones where the
        # document did not come back. Those score 0 and pull the average down.

        ndcg=(
            sum(ndcg_at(result.grades) for result in measurable) / len(measurable)
            if measurable
            else 0.0
        ),
        answered=len(answers),
        complete=len([result for result in answers if not result.missing]),
        judged=len([r for r in asked if r.faithful is not None]),
        faithful=len([r for r in asked if r.faithful is True]),
        unjudged=len([r for r in asked if r.judge_error is not None]),

        # A verdict is right when it agrees with what the question set says
        # about the library. A question naming a document should be accepted
        # and one naming none should be refused.

        accepted=len(
            [
                r
                for r in graded
                if r.question.document is not None and r.verdict.supported
            ]
        ),
        refused_unanswerable=len(
            [
                r
                for r in graded
                if r.question.document is None and not r.verdict.supported
            ]
        ),
        ungraded=len([r for r in asked if r.grade_error is not None]),
    )
