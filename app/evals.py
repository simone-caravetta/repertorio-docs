"""A set of questions, and what the library does with them.

Work on retrieval quality needs a number to move. A change that reads better is
not evidence, and what comes after this in the roadmap — a reranker, hybrid
search, a grading node — is one change after another to the same path, each of
them worth measuring the same way.

A question is written with the document it should be answered from, and that is
what makes it measurable: the search either returned a passage of that document
or it did not. Retrieval is measured on its own and from the question as it was
typed, because that is the input the search is given, and it costs one embedding
per question and no model call at all. The answer half is asked for: it writes an
answer from the passages the search returned, with the console's own prompt, and
reads it against what the question said the answer holds. `judge` adds a model
that says whether every claim in the answer is one the context supports, which is
what faithfulness means here and the only part of this that costs a second call
per question.

The passages are read once and used for both halves, so a question the search
missed cannot be rescued by a good answer: it was one search, and what it
returned is what the answer was written from.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from app.ingestion import whole_number
from app.rag_graph import answer_prompt, format_context

# What a question in a set may say about itself, and the whole of it. A field
# that is not here is a field nothing reads, and the one that matters is the one
# spelled wrong: a set is written by hand, and `contians` would measure nothing
# and say nothing about it.
FIELDS = ("id", "question", "document", "page", "contains", "note")

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
    """One question, and what a good answer to it would have to be.

    `document` is the document the answer should come from, as the catalog holds
    it: the path relative to the documents folder. It is the one thing a
    measurement needs. A question without one is a question the library does not
    answer — the case faithfulness is most likely to be tested by, since an
    answer written from passages that hold nothing relevant is where invention
    happens — and retrieval is not measured on it, there being no document it
    should have found.

    `page` is the page that answers it, counted as a reader counts pages, and is
    optional because it is work to write down and the document alone is already
    a measurement. `contains` is what a correct answer says, checked as plain
    text: what is being measured is whether the answer landed on the content, and
    a string a person can read off the page is the whole of what that needs.
    """

    id: str
    question: str
    document: str | None = None
    page: int | None = None
    contains: tuple[str, ...] = ()
    note: str = ""


@dataclass
class QuestionResult:
    """What one question did, and what was written about it.

    `rank` is where in the ranking the first passage of the expected document
    came, counted from one, and None means the search did not return one at all;
    `page_rank` is the same over the passages of the expected page. An answer is
    None when the run was not asked to write one, and `faithful` is None when no
    judge read it — which is not the same as a judge that read it and could not
    answer, and `judge_error` is what tells the two apart.
    """

    question: Question
    asked: bool = True
    rank: int | None = None
    page_rank: int | None = None
    sources: list[dict[str, Any]] = field(default_factory=list)
    answer: str | None = None
    missing: list[str] = field(default_factory=list)
    faithful: bool | None = None
    unsupported: list[str] = field(default_factory=list)
    judge_error: str | None = None
    error: str | None = None


@dataclass
class EvalReport:
    """Every question of a run, in the order the set holds them."""

    results: list[QuestionResult] = field(default_factory=list)


@dataclass(frozen=True)
class Summary:
    """What a run's numbers are, added up.

    `measurable` is the questions retrieval could be measured on: those that were
    asked, and that named a document to have found. `reciprocal_rank` is the mean
    of `1/rank` over them, which is where a search that finds the right document
    third rather than first is worse and says so.
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
    answered: int
    complete: int
    judged: int
    faithful: int
    unjudged: int


def load_questions(path: Path) -> list[Question]:
    """The questions a file holds, in the order it holds them.

    Every refusal here is the reader's, and reads as a sentence naming what is
    wrong with which question: a set is written by hand, and a traceback is not
    how a missing comma should be found.
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
    """What is wrong with what a set's file holds, or None when nothing is."""
    if not isinstance(loaded, list):
        return (
            f"{path} holds a {type(loaded).__name__}; a question set is a list "
            "of questions."
        )

    return None


def _question(entry: Any, number: int, path: Path) -> Question:
    """One entry of a set, or the sentence saying what is wrong with it."""
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
    """What is wrong with a question, or None when nothing is.

    Returned rather than raised, so that everything a set can be wrong about
    leaves this module through one door: what a caller catches is one exception
    for a set it cannot read, whichever way the set is broken.
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
    """Whether a value is a list of strings with something written in each."""
    return isinstance(value, list) and all(
        isinstance(text, str) and text.strip() for text in value
    )


def _listed(names: Iterable[str]) -> str:
    """Names as a sentence lists them: `a, b or c`."""
    ordered = sorted(names)
    if len(ordered) == 1:
        return ordered[0]
    return ", ".join(ordered[:-1]) + f" or {ordered[-1]}"


def run_evals(
    questions: Sequence[Question],
    *,
    retriever: BaseRetriever,
    in_scope: Sequence[str] | None = None,
    descriptions: Mapping[str, str] | None = None,
    chat_model: BaseChatModel | None = None,
    judge_model: BaseChatModel | None = None,
) -> EvalReport:
    """Ask each question of the retriever, and measure what came back.

    The retriever is an argument, so the same set can be measured through the
    console's path, through a chain being tried out, or through a double in a
    test. `in_scope` is the documents the search is over, and a question whose
    document is not among them is left out rather than counted as a miss: it was
    not asked, which is a different thing from being asked and not found. Both
    it and `descriptions` are handed to `format_context`, because the answer is
    written from what the console would have shown it.

    A question that raises does not end the run — a search is a network call, and
    one that fails is one question's result, not the whole measurement.
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
                in_scope=in_scope,
                descriptions=descriptions,
                chat_model=chat_model,
                judge_model=judge_model,
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
    in_scope: Sequence[str] | None,
    descriptions: Mapping[str, str] | None,
    chat_model: BaseChatModel | None,
    judge_model: BaseChatModel | None,
) -> QuestionResult:
    """One question, from the search to the verdict on its answer."""
    try:
        passages = retriever.invoke(question.question)
    except Exception as exc:  # noqa: BLE001 - one question's failure, not the run's
        return QuestionResult(question=question, error=str(exc))

    context, sources = format_context(
        passages, in_scope=in_scope, descriptions=descriptions
    )

    # Two readings of one ranking: where the document was found, and where it was
    # found on the page the question named. The second is not a refinement of the
    # first — a search can return the document and never the page — but a question
    # that named no page has nothing to narrow to, and says so with None rather
    # than by reporting the document's rank twice.
    found = rank_of(passages, question.document)

    result = QuestionResult(
        question=question,
        rank=found,
        page_rank=(
            rank_of(passages, question.document, page=question.page)
            if question.page is not None
            else None
        ),
        sources=sources,
    )

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
    """Where in the ranking the first passage of that document came, from one.

    The ranking is the search's own order, read before the rows are merged by
    place: two passages of one page are two results, and the first of them is
    where the document was found. `page` narrows the same question to the
    passages of one page, which is a stricter measure of the same search.
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
    """The page a passage sits on, as a reader counts pages.

    The metadata counts from zero — the reader's own convention, kept because it
    is what the offsets are counted in — and a question set is written by
    somebody looking at a page number. The one is turned into the other here,
    once, rather than at every comparison.
    """
    page = whole_number(passage.metadata.get("page"))

    return None if page is None else page + 1


def missing_from(answer: str, expected: Sequence[str]) -> list[str]:
    """The strings a question expected and the answer does not hold.

    Compared without case and without runs of whitespace, because what is asked
    is whether the answer says this and not whether it says it in the same
    characters: a set is written by hand, and a word at the start of a line is
    not a wrong answer. Accents are not folded — the answer is written in the
    language of the question, and the set is written in it too.
    """
    flat = " ".join(answer.split()).casefold()

    return [
        text for text in expected if " ".join(text.split()).casefold() not in flat
    ]


def write_answer(chat_model: BaseChatModel, question: str, context: str) -> str:
    """The answer the console would have written from these passages.

    The prompt is the console's own, imported rather than written again beside
    it: what is being measured is the answer this project gives, and a second
    prompt would measure the second prompt.
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
    """Whether every claim in the answer is one the context supports."""
    reply = judge_model.invoke(judge_prompt.invoke({
        "question": question,
        "context": context,
        "answer": written,
    }))
    content = reply.content

    return read_verdict(content if isinstance(content, str) else str(content))


def read_verdict(reply: str) -> tuple[bool | None, list[str]]:
    """What a judge said, as a verdict and the claims behind it.

    The first word is the verdict and the rest is why. A reply that does not open
    with yes or no is no answer rather than a pass: a judge that did not answer
    must not be the reason an answer is called faithful.
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
    """A run's results, added up."""
    asked = [result for result in report.results if result.asked]
    measurable = [
        result for result in asked if result.question.document is not None
    ]
    found = [result.rank for result in measurable if result.rank is not None]
    paged = [result for result in measurable if result.question.page is not None]
    answers = [result for result in asked if result.answer is not None]

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
        answered=len(answers),
        complete=len([result for result in answers if not result.missing]),
        judged=len([r for r in asked if r.faithful is not None]),
        faithful=len([r for r in asked if r.faithful is True]),
        unjudged=len([r for r in asked if r.judge_error is not None]),
    )
