"""Tests for the measurement a question set gives a search.

A question names the document its answer should come from, and sometimes the
page of it. What the search returned is read against that: the rank the
document came at, a grade for each passage in the order it arrived, and the
numbers the report is printed from. The answer half is covered here too, from
the passages the search returned to the judge that reads the answer against
them.
"""

from __future__ import annotations

import json
from math import log2
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document

from app.evals import (
    NDCG_CUTOFF,
    EvalReport,
    Question,
    QuestionResult,
    load_questions,
    missing_from,
    ndcg_at,
    rank_of,
    read_verdict,
    run_evals,
    summarise,
)
from app.grading import Verdict
from tests.helpers import FakeChatModel, verdict_reply

MANUAL = "manuals/manual.pdf"
REPORT = "reports/report.pdf"


class Retriever:
    """A stand-in for the search, answering from a fixed table of passages.

    Each query is written down, so a test can say which questions reached the
    search and which did not. The passages for a query come back in the order
    the test wrote them, and nothing here embeds anything or touches a store.
    """

    def __init__(self, passages: dict[str, list[Document]] | None = None) -> None:
        self.passages = passages or {}
        self.queries: list[str] = []

    def invoke(
        self, query: str, config: object = None, **kwargs: object
    ) -> list[Document]:
        """Return the passages written for that query, and note the query."""
        self.queries.append(query)
        return self.passages.get(query, [])


class BrokenRetriever:
    """A search that raises, as a store that cannot be reached would."""

    def invoke(
        self, query: str, config: object = None, **kwargs: object
    ) -> list[Document]:
        raise RuntimeError("the index is unreachable")


def passage(source: str, page: int, text: str = "a passage") -> Document:
    """One returned passage, taken from a page of a document.

    The page is the one the metadata carries, counted from zero, which is what
    a store hands back and what the reading side turns into a page number.
    """

    return Document(page_content=text, metadata={"source": source, "page": page})


def written_set(tmp_path: Path, *questions: dict[str, Any]) -> Path:
    """Write those questions to a file, as a question set is written."""
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(list(questions)), encoding="utf-8")
    return path


def question(**overrides: Any) -> Question:
    """A question about the manual, with any field the test names replaced."""
    values: dict[str, Any] = {
        "id": "garanzia",
        "question": "Quanti anni di garanzia?",
        "document": MANUAL,
    }
    values.update(overrides)
    return Question(**values)


# A question set is a file written by hand, so every way of getting it wrong is
# answered with a sentence naming the question and the field at fault, rather
# than with a traceback from somewhere inside the reader.


def test_a_question_set_is_read_in_order(tmp_path: Path) -> None:
    """Every field of an entry is read, in the order the file holds them."""
    path = written_set(
        tmp_path,
        {"id": "uno", "question": "Prima?", "document": MANUAL},
        {
            "id": "due",
            "question": "Seconda?",
            "document": REPORT,
            "page": 3,
            "contains": ["due anni", "24 mesi"],
            "note": "written off the table on page 3",
        },
    )

    questions = load_questions(path)

    assert [item.id for item in questions] == ["uno", "due"]
    assert questions[1].page == 3
    assert questions[1].contains == ("due anni", "24 mesi")
    assert questions[1].note == "written off the table on page 3"
    assert questions[0].page is None
    assert questions[0].contains == ()


def test_a_missing_file_is_said_so(tmp_path: Path) -> None:
    """A path with no file behind it reads as a message."""
    with pytest.raises(ValueError, match="No question set"):
        load_questions(tmp_path / "nothing.json")


def test_a_file_that_is_not_json_names_the_line(tmp_path: Path) -> None:
    """A file that does not parse gives the line to look at."""
    path = tmp_path / "questions.json"
    path.write_text('[{"id": "uno",}]', encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        load_questions(path)


def test_a_set_that_is_not_a_list_is_refused(tmp_path: Path) -> None:
    """A set is a list of questions, and anything else is refused."""
    path = tmp_path / "questions.json"
    path.write_text('{"id": "uno"}', encoding="utf-8")

    with pytest.raises(ValueError, match="a question set is a list"):
        load_questions(path)


def test_a_question_with_no_question_is_refused(tmp_path: Path) -> None:
    """An entry with no question names its place in the set."""
    with pytest.raises(ValueError, match="question 1 of .* has no question"):
        load_questions(written_set(tmp_path, {"id": "uno"}))


def test_a_misspelled_field_is_refused(tmp_path: Path) -> None:
    """A field the set does not have is refused, and named."""
    path = written_set(
        tmp_path,
        {"id": "uno", "question": "Prima?", "document": MANUAL, "contians": ["x"]},
    )

    with pytest.raises(ValueError, match="contians"):
        load_questions(path)


def test_a_page_counted_from_zero_is_refused(tmp_path: Path) -> None:
    """A page is the reader's number, so zero is not one of them."""
    path = written_set(
        tmp_path, {"id": "uno", "question": "Prima?", "document": MANUAL, "page": 0}
    )

    with pytest.raises(ValueError, match="counted from one"):
        load_questions(path)


def test_a_page_that_is_not_a_number_is_refused(tmp_path: Path) -> None:
    """A page written as text is refused the same way zero is."""
    path = written_set(
        tmp_path,
        {"id": "uno", "question": "Prima?", "document": MANUAL, "page": "3"},
    )

    with pytest.raises(ValueError, match="counted from one"):
        load_questions(path)


def test_a_contains_that_is_not_a_list_is_refused(tmp_path: Path) -> None:
    """What a correct answer holds is a list of strings, and nothing else."""
    path = written_set(
        tmp_path,
        {"id": "uno", "question": "Prima?", "document": MANUAL, "contains": "due"},
    )

    with pytest.raises(ValueError, match="not a list of strings"):
        load_questions(path)


def test_two_questions_with_one_id_are_refused(tmp_path: Path) -> None:
    """An id tells two runs apart, so it is held to once per set."""
    path = written_set(
        tmp_path,
        {"id": "uno", "question": "Prima?", "document": MANUAL},
        {"id": "uno", "question": "Ancora?", "document": REPORT},
    )

    with pytest.raises(ValueError, match="two questions with the id 'uno'"):
        load_questions(path)


def test_a_question_the_library_does_not_answer_may_name_no_document(
    tmp_path: Path,
) -> None:
    """A question with no document is read, and holds no document."""
    path = written_set(tmp_path, {"id": "fuori", "question": "Chi ha vinto?"})

    (question,) = load_questions(path)

    assert question.document is None


# What one question did to the search: where the document came, which page of
# it came, and whether the question was asked of this scope at all.


def test_a_passage_of_the_expected_document_is_a_hit() -> None:
    """The expected document at the top of the ranking is rank one."""
    retriever = Retriever({"Prima?": [passage(MANUAL, 1), passage(REPORT, 1)]})

    report = run_evals(
        [question(question="Prima?")], retriever=retriever, in_scope=[MANUAL, REPORT]
    )

    assert report.results[0].rank == 1
    assert retriever.queries == ["Prima?"]


def test_nothing_from_the_expected_document_is_a_miss() -> None:
    """A ranking of another document leaves the rank unset."""
    retriever = Retriever({"Prima?": [passage(REPORT, 1)]})

    report = run_evals(
        [question(question="Prima?")], retriever=retriever, in_scope=[MANUAL, REPORT]
    )

    assert report.results[0].rank is None


def test_the_rank_is_where_the_document_was_found() -> None:
    """Every passage above it counts, from whatever document it came from."""
    retriever = Retriever({
        "Prima?": [passage(REPORT, 1), passage(REPORT, 2), passage(MANUAL, 5)]
    })

    report = run_evals(
        [question(question="Prima?")], retriever=retriever, in_scope=[MANUAL, REPORT]
    )

    assert report.results[0].rank == 3


def test_a_page_is_measured_on_its_own() -> None:
    """The document and the page it was found on are read separately."""

    retriever = Retriever({"Prima?": [passage(MANUAL, 2), passage(MANUAL, 8)]})

    report = run_evals(
        [question(question="Prima?", page=9)],
        retriever=retriever,
        in_scope=[MANUAL],
    )

    assert report.results[0].rank == 1
    assert report.results[0].page_rank == 2


def test_the_page_is_counted_as_a_reader_counts_it() -> None:
    """The metadata counts from zero, so page 12 is the one stored as 11."""

    retriever = Retriever({"Prima?": [passage(MANUAL, 11)]})

    report = run_evals(
        [question(question="Prima?", page=12)],
        retriever=retriever,
        in_scope=[MANUAL],
    )

    assert report.results[0].page_rank == 1


def test_a_question_that_names_no_page_has_no_page_rank() -> None:
    """A question without a page has nothing to narrow the search to."""
    retriever = Retriever({"Prima?": [passage(MANUAL, 3)]})

    report = run_evals(
        [question(question="Prima?")], retriever=retriever, in_scope=[MANUAL]
    )

    assert report.results[0].rank == 1
    assert report.results[0].page_rank is None


def test_a_question_outside_the_scope_is_not_asked() -> None:
    """A document outside the scope means the search is never called."""
    retriever = Retriever({"Prima?": [passage(MANUAL, 1)]})

    report = run_evals(
        [question(question="Prima?")], retriever=retriever, in_scope=[REPORT]
    )

    assert report.results[0].asked is False
    assert retriever.queries == []


def test_a_question_with_no_document_is_asked_of_the_whole_scope() -> None:
    """With no document to look for, the question is asked and not ranked."""
    retriever = Retriever({"Prima?": [passage(MANUAL, 1)]})

    report = run_evals(
        [question(question="Prima?", document=None)],
        retriever=retriever,
        in_scope=[REPORT],
    )

    assert report.results[0].asked is True
    assert report.results[0].rank is None
    assert retriever.queries == ["Prima?"]


def test_a_search_that_fails_is_one_question_and_not_the_run() -> None:
    """A search that raises is recorded against that question, and no more."""
    report = run_evals(
        [question(question="Prima?")], retriever=BrokenRetriever(), in_scope=[MANUAL]
    )

    assert report.results[0].error == "the index is unreachable"
    assert report.results[0].asked is True


def test_the_rows_of_the_answer_are_the_passages_the_search_returned() -> None:
    """The passages are kept as the console would have shown them."""
    retriever = Retriever({"Prima?": [passage(MANUAL, 4, "the warranty is two years")]})

    report = run_evals(
        [question(question="Prima?")], retriever=retriever, in_scope=[MANUAL]
    )

    # The page is the reader's, and the passage has no offsets to draw on.
    assert report.results[0].sources == [
        {"source": MANUAL, "page": 5, "ranges": []}
    ]


# The answer half. It is written from the passages the search returned, so a
# question the search missed cannot be rescued by a good answer later on.


def test_no_answer_is_written_unless_a_model_is_given() -> None:
    """A run of retrieval alone reaches no model and writes no answer."""
    retriever = Retriever({"Prima?": [passage(MANUAL, 1)]})

    report = run_evals(
        [question(question="Prima?")], retriever=retriever, in_scope=[MANUAL]
    )

    assert report.results[0].answer is None


def test_the_answer_is_written_from_the_passages_that_were_found() -> None:
    """The prompt carries the found text and the question it answers."""

    retriever = Retriever({"Prima?": [passage(MANUAL, 4, "the warranty is two years")]})
    model = FakeChatModel(replies=["Two years."])

    run_evals(
        [question(question="Prima?")],
        retriever=retriever,
        in_scope=[MANUAL],
        chat_model=model,
    )

    # The human message is the one holding the context and the question.
    asked = model.prompts[0][1].content

    assert "the warranty is two years" in asked
    assert "Prima?" in asked


def test_the_eval_writes_the_context_with_the_outlines_of_its_scope() -> None:
    """A run measures the context a chat gives the model, headings included.

    A run that leaves them out measures a pipeline shorter than the one users
    talk to, which is the same reason `context_k` is passed at all.
    """

    retriever = Retriever({"Prima?": [passage(MANUAL, 4, "the warranty is two years")]})
    model = FakeChatModel(replies=["Two years."])

    run_evals(
        [question(question="Prima?")],
        retriever=retriever,
        in_scope=[MANUAL],
        descriptions={MANUAL: "A manual about the thing."},
        outlines={MANUAL: "Introducción — pp.6-12"},
        chat_model=model,
    )

    asked = model.prompts[0][1].content

    assert "A manual about the thing." in asked
    assert "Introducción — pp.6-12" in asked


def test_a_run_can_give_the_model_as_many_passages_as_a_chat_does() -> None:
    """The context is cut to what the console hands over, not the whole search.

    A run that is asked for more passages than a chat reads — ten, so the nDCG
    can be read over them — would otherwise measure a pipeline with a wider net
    than the one a user talks to.
    """

    retriever = Retriever({
        "Prima?": [
            passage(MANUAL, 1, "the first passage"),
            passage(MANUAL, 2, "the second passage"),
            passage(MANUAL, 3, "the third passage"),
        ]
    })
    model = FakeChatModel(replies=["Two years."])

    report = run_evals(
        [question(question="Prima?")],
        retriever=retriever,
        context_k=2,
        in_scope=[MANUAL],
        chat_model=model,
    )

    asked = model.prompts[0][1].content

    assert "the first passage" in asked
    assert "the second passage" in asked
    assert "the third passage" not in asked

    # The ranking is still read over everything the search returned, and the
    # rows are the passages the context was built from.
    assert report.results[0].grades == (1, 1, 1)
    assert [row["page"] for row in report.results[0].sources] == [2, 3]


def test_an_answer_that_does_not_hold_what_was_expected_is_reported_missing() -> None:
    """Only the expected text the answer misses is reported."""
    retriever = Retriever({"Prima?": [passage(MANUAL, 1)]})
    model = FakeChatModel(replies=["Two years, and it renews."])

    report = run_evals(
        [question(question="Prima?", contains=("two years", "24 months"))],
        retriever=retriever,
        in_scope=[MANUAL],
        chat_model=model,
    )

    assert report.results[0].missing == ["24 months"]


def test_an_answer_that_fails_ends_that_question() -> None:
    """A model call that raises leaves an error and no answer."""
    retriever = Retriever({"Prima?": [passage(MANUAL, 1)]})
    model = FakeChatModel(replies=[])

    report = run_evals(
        [question(question="Prima?")],
        retriever=retriever,
        in_scope=[MANUAL],
        chat_model=model,
    )

    assert report.results[0].answer is None
    assert report.results[0].error is not None


def test_case_and_spacing_do_not_matter_in_what_was_expected() -> None:
    """The comparison reads words, so case and runs of spaces are ignored."""
    assert missing_from("Due   anni di garanzia.", ["due anni"]) == []
    assert missing_from("Two years.", ["due anni"]) == ["due anni"]


# The judge is a second model call per question. It reads the answer against
# the passages it was written from, and says what, if anything, is not there.


def test_a_judge_that_says_yes_is_faithful() -> None:
    """A verdict of yes leaves nothing unsupported behind it."""
    retriever = Retriever({"Prima?": [passage(MANUAL, 1, "two years")]})
    answers = FakeChatModel(replies=["Two years."])
    judge = FakeChatModel(replies=["yes"])

    report = run_evals(
        [question(question="Prima?")],
        retriever=retriever,
        in_scope=[MANUAL],
        chat_model=answers,
        judge_model=judge,
    )

    assert report.results[0].faithful is True
    assert report.results[0].unsupported == []


def test_a_judge_that_says_no_keeps_the_claims() -> None:
    """What follows a no is the list of claims the passages do not hold."""
    retriever = Retriever({"Prima?": [passage(MANUAL, 1, "two years")]})
    answers = FakeChatModel(replies=["Two years, and it renews."])
    judge = FakeChatModel(replies=["no\nand it renews"])

    report = run_evals(
        [question(question="Prima?")],
        retriever=retriever,
        in_scope=[MANUAL],
        chat_model=answers,
        judge_model=judge,
    )

    assert report.results[0].faithful is False
    assert report.results[0].unsupported == ["and it renews"]


def test_the_judge_reads_the_answer_against_the_passages() -> None:
    """The prompt holds the context and the answer under it."""
    retriever = Retriever({"Prima?": [passage(MANUAL, 1, "two years")]})
    answers = FakeChatModel(replies=["Two years."])
    judge = FakeChatModel(replies=["yes"])

    run_evals(
        [question(question="Prima?")],
        retriever=retriever,
        in_scope=[MANUAL],
        chat_model=answers,
        judge_model=judge,
    )

    read = judge.prompts[0][1].content

    assert "two years" in read
    assert "Two years." in read


def test_a_judge_that_cannot_be_asked_is_not_a_verdict() -> None:
    """A judge that fails is recorded apart from the question's own error."""
    retriever = Retriever({"Prima?": [passage(MANUAL, 1)]})
    answers = FakeChatModel(replies=["Two years."])

    report = run_evals(
        [question(question="Prima?")],
        retriever=retriever,
        in_scope=[MANUAL],
        chat_model=answers,
        judge_model=FakeChatModel(replies=[]),
    )

    assert report.results[0].faithful is None
    assert report.results[0].judge_error is not None
    assert report.results[0].error is None


@pytest.mark.parametrize(
    "reply, verdict",
    [
        ("yes", True),
        ("Yes.", True),
        ("yes, every claim is in the context", True),
        ("no", False),
        ("No — the price is not there", False),
        ("", None),
        ("I cannot tell", None),
    ],
)
def test_a_reply_is_read_as_the_verdict_it_opens_with(
    reply: str, verdict: bool | None
) -> None:
    """The first word decides, and anything else is no verdict at all."""
    assert read_verdict(reply)[0] is verdict


def test_a_verdict_of_no_keeps_what_follows_it() -> None:
    """The lines under a no are the claims the judge listed."""
    _, claims = read_verdict("no\n\nthe price is not there\nand the date is wrong")

    assert claims == ["the price is not there", "and the date is wrong"]


# The grader is one more model call per question, and it reads the material the
# question came back with rather than anything written from it. A question the
# library is not expected to answer names no document, and a refusal is what
# the grader owes it.


def test_the_verdict_on_the_material_is_kept() -> None:
    """The verdict is recorded against the question it was made about."""
    retriever = Retriever({"Prima?": [passage(MANUAL, 1, "two years")]})
    grader = FakeChatModel(replies=[verdict_reply(True, "The passage states it.")])

    report = run_evals(
        [question(question="Prima?")],
        retriever=retriever,
        in_scope=[MANUAL],
        grade_model=grader,
    )

    assert report.results[0].verdict == Verdict(
        supported=True, reason="The passage states it."
    )


def test_a_question_the_library_cannot_answer_is_read_for_a_refusal() -> None:
    """Nothing in the library answers it, and the verdict says so."""
    retriever = Retriever({"Chi ha vinto?": [passage(MANUAL, 1)]})
    grader = FakeChatModel(
        replies=[verdict_reply(False, "The passages are about a warranty.", "il vincitore")]
    )

    report = run_evals(
        [question(id="fuori", question="Chi ha vinto?", document=None)],
        retriever=retriever,
        in_scope=[MANUAL],
        grade_model=grader,
    )

    assert report.results[0].verdict == Verdict(
        supported=False,
        reason="The passages are about a warranty.",
        query="il vincitore",
    )


def test_the_grader_reads_the_material_rather_than_an_answer() -> None:
    """The prompt holds the question and the passages it came back with."""
    retriever = Retriever({"Prima?": [passage(MANUAL, 4, "the warranty is two years")]})
    grader = FakeChatModel(replies=[verdict_reply(True)])

    run_evals(
        [question(question="Prima?")],
        retriever=retriever,
        in_scope=[MANUAL],
        grade_model=grader,
    )

    read = grader.prompts[0][1].content

    assert "the warranty is two years" in read
    assert "Prima?" in read


def test_grading_needs_no_answer_to_have_been_written() -> None:
    """A run that grades asks the retriever and the grader, and no more."""
    retriever = Retriever({"Prima?": [passage(MANUAL, 1)]})
    grader = FakeChatModel(replies=[verdict_reply(True)])

    report = run_evals(
        [question(question="Prima?")],
        retriever=retriever,
        in_scope=[MANUAL],
        grade_model=grader,
    )

    assert report.results[0].answer is None
    assert report.results[0].verdict is not None


def test_a_verdict_is_kept_when_the_answer_fails() -> None:
    """The verdict is about the material, so a failed answer does not undo it."""
    retriever = Retriever({"Prima?": [passage(MANUAL, 1)]})

    report = run_evals(
        [question(question="Prima?")],
        retriever=retriever,
        in_scope=[MANUAL],
        chat_model=FakeChatModel(replies=[]),
        grade_model=FakeChatModel(replies=[verdict_reply(True)]),
    )

    assert report.results[0].error is not None
    assert report.results[0].verdict is not None


def test_a_grader_that_cannot_be_asked_is_not_a_verdict() -> None:
    """A grader that fails is recorded apart from the question's own error."""
    retriever = Retriever({"Prima?": [passage(MANUAL, 1)]})

    report = run_evals(
        [question(question="Prima?")],
        retriever=retriever,
        in_scope=[MANUAL],
        grade_model=FakeChatModel(replies=[]),
    )

    assert report.results[0].verdict is None
    assert report.results[0].grade_error is not None
    assert report.results[0].error is None


# The second search. A material the grader turned away is searched for again
# with the query written beside the verdict, which is the loop the graph runs:
# the point of it is a question the first search missed and the next one found.


def test_a_material_that_was_turned_away_is_searched_for_again() -> None:
    """The query written with the verdict is what the search is asked next."""

    retriever = Retriever({
        "Prima?": [passage(REPORT, 1, "another document's passage")],
        "il vincitore": [passage(MANUAL, 4, "the warranty is two years")],
    })
    grader = FakeChatModel(replies=[
        verdict_reply(False, "The passage is about another document.", "il vincitore"),
        verdict_reply(True, "The passage states it."),
    ])

    report = run_evals(
        [question(question="Prima?")],
        retriever=retriever,
        in_scope=[MANUAL, REPORT],
        grade_model=grader,
        attempts=2,
    )

    result = report.results[0]

    assert retriever.queries == ["Prima?", "il vincitore"]
    assert result.searches == 2
    assert result.rejected_first is True
    assert result.verdict == Verdict(supported=True, reason="The passage states it.")
    assert result.sources == [{"source": MANUAL, "page": 5, "ranges": []}]


def test_a_question_the_search_missed_is_answered_from_the_second_one() -> None:
    """The answer is written from the material the grader accepted, not the first."""

    retriever = Retriever({
        "Prima?": [passage(REPORT, 1, "the report is about something else")],
        "il vincitore": [passage(MANUAL, 4, "the warranty is two years")],
    })
    grader = FakeChatModel(replies=[
        verdict_reply(False, "It is about another document.", "il vincitore"),
        verdict_reply(True, "It states it."),
    ])
    answers = FakeChatModel(replies=["Two years."])

    report = run_evals(
        [question(question="Prima?", contains=("two years",))],
        retriever=retriever,
        in_scope=[MANUAL, REPORT],
        chat_model=answers,
        grade_model=grader,
        attempts=2,
    )

    asked = answers.prompts[0][1].content

    assert "the warranty is two years" in asked
    assert "the report is about something else" not in asked
    assert report.results[0].missing == []

    # The search missed, and the report says so: the ranking is read on the
    # query a reader would have typed, not on the one written for it after.

    assert report.results[0].rank is None


def test_the_ranking_is_read_on_the_first_search() -> None:
    """A recovery does not move the rank the question was found at."""

    retriever = Retriever({
        "Prima?": [passage(REPORT, 1), passage(REPORT, 2), passage(MANUAL, 3)],
        "il vincitore": [passage(MANUAL, 4)],
    })
    grader = FakeChatModel(replies=[
        verdict_reply(False, "Not enough here.", "il vincitore"),
        verdict_reply(True, "It states it."),
    ])

    report = run_evals(
        [question(question="Prima?")],
        retriever=retriever,
        in_scope=[MANUAL, REPORT],
        grade_model=grader,
        attempts=2,
    )

    assert report.results[0].rank == 3
    assert report.results[0].searches == 2


def test_the_search_is_tried_as_often_as_the_run_allows() -> None:
    """One attempt leaves the query the verdict wrote unused."""
    retriever = Retriever({
        "Prima?": [passage(REPORT, 1)],
        "il vincitore": [passage(MANUAL, 4)],
    })
    grader = FakeChatModel(
        replies=[verdict_reply(False, "Not enough here.", "il vincitore")]
    )

    report = run_evals(
        [question(question="Prima?")],
        retriever=retriever,
        in_scope=[MANUAL, REPORT],
        grade_model=grader,
        attempts=1,
    )

    assert retriever.queries == ["Prima?"]
    assert report.results[0].searches == 1
    assert report.results[0].rejected_first is True
    assert report.results[0].verdict == Verdict(
        supported=False, reason="Not enough here.", query="il vincitore"
    )


def test_a_verdict_with_no_query_does_not_search_again() -> None:
    """A grader with nothing better to try ends the question where it is."""
    retriever = Retriever({
        "Prima?": [passage(REPORT, 1)],
        "il vincitore": [passage(MANUAL, 4)],
    })
    grader = FakeChatModel(replies=[verdict_reply(False, "Not enough here.")])

    report = run_evals(
        [question(question="Prima?")],
        retriever=retriever,
        in_scope=[MANUAL, REPORT],
        grade_model=grader,
        attempts=2,
    )

    assert retriever.queries == ["Prima?"]
    assert report.results[0].searches == 1
    assert report.results[0].rejected_first is True


def test_a_question_whose_material_was_turned_away_is_not_answered() -> None:
    """Material the grader refused is not written an answer from."""
    retriever = Retriever({"Prima?": [passage(REPORT, 1)]})

    report = run_evals(
        [question(question="Prima?")],
        retriever=retriever,
        in_scope=[MANUAL, REPORT],
        chat_model=FakeChatModel(replies=["Two years."]),
        grade_model=FakeChatModel(replies=[verdict_reply(False, "Not enough here.")]),
        attempts=2,
    )

    assert report.results[0].verdict is not None
    assert report.results[0].answer is None


def test_a_second_search_that_fails_ends_that_question() -> None:
    """A search that raises on the second query is recorded, not raised."""
    retriever = Retriever({"Prima?": [passage(MANUAL, 1)]})
    grader = FakeChatModel(
        replies=[verdict_reply(False, "Not enough here.", "il vincitore")]
    )

    class Half:
        """A search that answers once and then cannot be reached."""

        def __init__(self) -> None:
            self.queries: list[str] = []

        def invoke(self, query: str, config: object = None, **kwargs: object) -> list[Document]:
            self.queries.append(query)
            if len(self.queries) > 1:
                raise RuntimeError("the index is unreachable")
            return retriever.invoke(query)

    half = Half()

    report = run_evals(
        [question(question="Prima?")],
        retriever=half,  # type: ignore[arg-type]
        in_scope=[MANUAL],
        grade_model=grader,
        attempts=2,
    )

    assert report.results[0].error == "the index is unreachable"
    assert half.queries == ["Prima?", "il vincitore"]


def test_what_the_second_search_came_to() -> None:
    """Every question rejected first was recovered or turned away."""
    report = EvalReport(results=[
        QuestionResult(
            question=question(id="recuperata"),
            searches=2,
            rejected_first=True,
            verdict=Verdict(supported=True, reason="It states it."),
        ),
        QuestionResult(
            question=question(id="respinta"),
            searches=1,
            rejected_first=True,
            verdict=Verdict(supported=False, reason="Nothing on it."),
        ),
        QuestionResult(
            question=question(id="subito"),
            verdict=Verdict(supported=True, reason="It states it."),
        ),
    ])

    summary = summarise(report)

    assert (summary.searches, summary.rejected_first) == (4, 2)
    assert (summary.retried, summary.recovered) == (1, 1)
    assert summary.turned_away == 1
    assert summary.recovered + summary.turned_away == summary.rejected_first


# What a whole run comes to, over the questions it was able to measure.


def test_the_numbers_add_up() -> None:
    """Three measurable questions, one of them missed, and one page asked."""
    report = EvalReport(results=[
        QuestionResult(question=question(id="uno"), rank=1, grades=(1, 0)),
        QuestionResult(
            question=question(id="due", page=4), rank=3, page_rank=None,
            grades=(0, 0, 1),
        ),
        QuestionResult(question=question(id="tre"), rank=None, grades=(0,)),
        QuestionResult(question=question(id="fuori", document=None)),
        QuestionResult(question=question(id="saltata"), asked=False),
    ])

    summary = summarise(report)

    assert (summary.questions, summary.asked, summary.skipped) == (5, 4, 1)
    assert (summary.measurable, summary.hits) == (3, 2)
    assert (summary.page_asked, summary.page_hits) == (1, 0)
    assert summary.reciprocal_rank == pytest.approx((1 + 1 / 3) / 3)

    # Every measurable question counts, the one the search missed included:
    # its grades are all zero, and it pulls the mean down.

    assert summary.ndcg == pytest.approx((1.0 + 1 / log2(4) + 0.0) / 3)

    # No grader ran, so nothing was accepted and nothing was refused. The
    # question that names no document is not a refusal on its own.

    assert (summary.accepted, summary.refused_unanswerable) == (0, 0)
    assert summary.ungraded == 0


def test_the_verdicts_are_counted_on_both_kinds_of_question() -> None:
    """A grader is right when it accepts what is there and refuses what is not."""
    report = EvalReport(results=[
        QuestionResult(
            question=question(id="uno"),
            verdict=Verdict(supported=True, reason="It states it."),
        ),
        QuestionResult(
            question=question(id="due"),
            verdict=Verdict(supported=False, reason="Nothing on it."),
        ),
        QuestionResult(
            question=question(id="fuori", document=None),
            verdict=Verdict(supported=False, reason="Nothing on it."),
        ),
        QuestionResult(
            question=question(id="altro", document=None),
            verdict=Verdict(supported=True, reason="It comes close."),
        ),
        QuestionResult(question=question(id="rotto"), grade_error="the model is down"),
    ])

    summary = summarise(report)

    assert summary.measurable == 3
    assert summary.accepted == 1
    assert summary.refused_unanswerable == 1
    assert summary.ungraded == 1


def test_a_run_with_nothing_to_measure_has_no_rank() -> None:
    """With nothing measurable the two means are zero, not a division."""
    report = EvalReport(results=[QuestionResult(question=question(id="saltata"), asked=False)])

    summary = summarise(report)

    assert summary.reciprocal_rank == 0.0
    assert summary.ndcg == 0.0


def test_a_failed_search_scores_no_better_than_a_missed_one() -> None:
    """A search that raised is a question that found nothing."""
    report = EvalReport(results=[
        QuestionResult(
            question=question(id="uno"), error="the index is unreachable"
        ),
    ])

    assert summarise(report).ndcg == 0.0


# nDCG, on lists of grades alone: what the number is measured against, and
# where the best order of the same passages sits.


def test_the_best_order_the_passages_allowed_scores_one() -> None:
    """A ranking that already holds the best of its own grades scores 1."""
    assert ndcg_at((2, 1, 0)) == pytest.approx(1.0)
    assert ndcg_at((1, 0)) == pytest.approx(1.0)
    assert ndcg_at(()) == 0.0


def test_the_right_page_below_a_wrong_one_does_not() -> None:
    """A good passage under a poor one scores below the two reversed."""

    assert ndcg_at((1, 2)) == pytest.approx(
        (1 + 2 / log2(3)) / (2 + 1 / log2(3))
    )
    assert ndcg_at((1, 2)) < ndcg_at((2, 1))


def test_the_same_passage_lower_down_scores_less() -> None:
    """One grade at the top is worth more than the same grade at the bottom."""
    assert ndcg_at((0, 0, 2, 0)) == pytest.approx((2 / log2(4)) / 2)
    assert ndcg_at((0, 0, 2, 0)) < ndcg_at((2, 0, 0, 0))


def test_a_ranking_that_found_nothing_scores_nothing() -> None:
    """Nothing relevant to find is a score of zero, not a division."""
    assert ndcg_at((0, 0, 0)) == 0.0
    assert ndcg_at(()) == 0.0


def test_ordering_is_read_against_what_the_search_returned() -> None:
    """Two relevant passages in either order are the ideal between them.

    The ideal is built from the grades that came back, so two of them at the
    top score 1 whichever way round they are. Moving one of them under an
    irrelevant passage stops being the best order those passages allowed.
    """

    assert ndcg_at((1, 1, 0)) == pytest.approx(1.0)
    assert ndcg_at((0, 1, 1)) < 1.0


def test_the_numbers_about_finding_are_read_over_what_was_shown() -> None:
    """The ranks stop at the cutoff the console shows, the grades do not.

    Six passages come back with the expected document last. A run that reads
    five of them reports a miss, and grades the whole ranking either way.
    """

    retriever = Retriever({
        "Prima?": [passage(REPORT, 1) for _ in range(5)] + [passage(MANUAL, 1)]
    })

    everything = run_evals(
        [question(question="Prima?")], retriever=retriever, in_scope=[MANUAL, REPORT]
    )
    shown = run_evals(
        [question(question="Prima?")],
        retriever=retriever,
        read_at=5,
        in_scope=[MANUAL, REPORT],
    )

    assert everything.results[0].rank == 6
    assert shown.results[0].rank is None

    # nDCG reads the ranking in full in both runs, so the grades agree.
    assert shown.results[0].grades == everything.results[0].grades


def test_a_run_with_no_read_at_reads_everything_it_was_given() -> None:
    """With no cutoff given, every passage that came back is read."""

    retriever = Retriever({
        "Prima?": [passage(REPORT, 1) for _ in range(5)] + [passage(MANUAL, 1)]
    })

    report = run_evals(
        [question(question="Prima?")], retriever=retriever, in_scope=[MANUAL, REPORT]
    )

    assert report.results[0].rank == 6


def test_a_passage_past_the_cutoff_is_not_read() -> None:
    """Only the grades above the cutoff are read, and zero reads none."""

    assert ndcg_at((2,), cutoff=1) == pytest.approx(1.0)
    assert ndcg_at((0, 2), cutoff=1) == 0.0
    assert ndcg_at((2,), cutoff=0) == 0.0


def test_the_cutoff_is_ten_by_default() -> None:
    """Ten is the default, so a relevant passage past it scores nothing."""

    below = (0,) * (NDCG_CUTOFF - 1) + (2,)
    past = (0,) * NDCG_CUTOFF + (2,)

    assert NDCG_CUTOFF == 10
    assert ndcg_at(below) > 0
    assert ndcg_at(past) == 0.0


# How good one returned passage is for one question, which is what a ranking
# is read on: the document the answer comes from, and the page of it.


def test_a_passage_is_graded_on_the_page_the_question_named() -> None:
    """A page of the document is graded 2, another page of it 1, and rest 0."""
    retriever = Retriever({
        "Prima?": [
            passage(REPORT, 0),
            passage(MANUAL, 3),
            passage(MANUAL, 1),
        ]
    })

    report = run_evals(
        [question(question="Prima?", page=2)], retriever=retriever
    )

    # Read down the ranking: another document is 0, the document on another
    # page is 1, and the page the question named is 2.

    assert report.results[0].grades == (0, 1, 2)


def test_a_question_that_named_no_page_has_one_grade_to_give() -> None:
    """With no page named, a passage is graded on its document alone."""
    retriever = Retriever({"Prima?": [passage(REPORT, 0), passage(MANUAL, 3)]})

    report = run_evals([question(question="Prima?")], retriever=retriever)

    assert report.results[0].grades == (0, 1)


def test_a_search_that_returned_nothing_has_no_grades() -> None:
    """No passages means no grades, and a run that scores nothing."""
    report = run_evals([question(question="Prima?")], retriever=Retriever({}))

    assert report.results[0].grades == ()
    assert summarise(report).ndcg == 0.0


def test_a_question_that_could_not_be_asked_has_no_grades() -> None:
    """A question outside the scope is not asked and grades nothing."""
    report = run_evals(
        [question(question="Prima?")], retriever=Retriever({}), in_scope=[REPORT]
    )

    assert report.results[0].asked is False
    assert report.results[0].grades == ()


def test_a_failed_search_is_counted_as_a_failure() -> None:
    """A failed search is a failure and a question that found nothing."""
    report = EvalReport(results=[
        QuestionResult(question=question(id="uno"), error="the index is unreachable"),
    ])

    summary = summarise(report)

    assert summary.failed == 1
    assert summary.measurable == 1
    assert summary.hits == 0


def test_rank_of_ignores_a_document_it_was_not_asked_about() -> None:
    """With no document there is no rank, and a page narrows the search."""
    passages = [passage(REPORT, 1), passage(MANUAL, 1)]

    assert rank_of(passages, None) is None
    assert rank_of(passages, MANUAL) == 2

    assert rank_of(passages, MANUAL, page=2) == 2
    assert rank_of(passages, MANUAL, page=3) is None
