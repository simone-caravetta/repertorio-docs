"""The eval harness: reading a question set, and measuring what came back.

Nothing here touches a model, a network or the library: the retriever is a
stand-in that hands back whatever a test says it should, and the models are the
suite's fake. What is measured against the real documents is measured by running
`python -m scripts.eval`, which is a measurement and not a test.
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
from tests.helpers import FakeChatModel

MANUAL = "manuals/manual.pdf"
REPORT = "reports/report.pdf"


class Retriever:
    """A retriever that hands back what a test says it should, per question.

    The harness calls `invoke`, so that is all this is: no ranking of its own, and
    the passages come back in the order the test wrote them, which is what makes
    a rank assertable.
    """

    def __init__(self, passages: dict[str, list[Document]] | None = None) -> None:
        self.passages = passages or {}
        self.queries: list[str] = []

    def invoke(
        self, query: str, config: object = None, **kwargs: object
    ) -> list[Document]:
        self.queries.append(query)
        return self.passages.get(query, [])


class BrokenRetriever:
    """A search that cannot run, which is a question's result and not the run's."""

    def invoke(
        self, query: str, config: object = None, **kwargs: object
    ) -> list[Document]:
        raise RuntimeError("the index is unreachable")


def passage(source: str, page: int, text: str = "a passage") -> Document:
    """A chunk, with its page written the way the store keeps it: from zero.

    Which is not the way a question set writes one — a question names the page a
    reader would turn to — and the two are a page apart on purpose. A test that
    wants a passage on the reader's page 9 says 8 here.
    """
    return Document(page_content=text, metadata={"source": source, "page": page})


def written_set(tmp_path: Path, *questions: dict[str, Any]) -> Path:
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(list(questions)), encoding="utf-8")
    return path


def question(**overrides: Any) -> Question:
    values: dict[str, Any] = {
        "id": "garanzia",
        "question": "Quanti anni di garanzia?",
        "document": MANUAL,
    }
    values.update(overrides)
    return Question(**values)


# --- reading a set ---------------------------------------------------------


def test_a_question_set_is_read_in_order(tmp_path: Path) -> None:
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
    with pytest.raises(ValueError, match="No question set"):
        load_questions(tmp_path / "nothing.json")


def test_a_file_that_is_not_json_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text('[{"id": "uno",}]', encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        load_questions(path)


def test_a_set_that_is_not_a_list_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text('{"id": "uno"}', encoding="utf-8")

    with pytest.raises(ValueError, match="a question set is a list"):
        load_questions(path)


def test_a_question_with_no_question_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="question 1 of .* has no question"):
        load_questions(written_set(tmp_path, {"id": "uno"}))


def test_a_misspelled_field_is_refused(tmp_path: Path) -> None:
    """A field nothing reads is a question that measures less than it looks like."""
    path = written_set(
        tmp_path,
        {"id": "uno", "question": "Prima?", "document": MANUAL, "contians": ["x"]},
    )

    with pytest.raises(ValueError, match="contians"):
        load_questions(path)


def test_a_page_counted_from_zero_is_refused(tmp_path: Path) -> None:
    path = written_set(
        tmp_path, {"id": "uno", "question": "Prima?", "document": MANUAL, "page": 0}
    )

    with pytest.raises(ValueError, match="counted from one"):
        load_questions(path)


def test_a_page_that_is_not_a_number_is_refused(tmp_path: Path) -> None:
    path = written_set(
        tmp_path,
        {"id": "uno", "question": "Prima?", "document": MANUAL, "page": "3"},
    )

    with pytest.raises(ValueError, match="counted from one"):
        load_questions(path)


def test_a_contains_that_is_not_a_list_is_refused(tmp_path: Path) -> None:
    path = written_set(
        tmp_path,
        {"id": "uno", "question": "Prima?", "document": MANUAL, "contains": "due"},
    )

    with pytest.raises(ValueError, match="not a list of strings"):
        load_questions(path)


def test_two_questions_with_one_id_are_refused(tmp_path: Path) -> None:
    """An id is what tells two runs apart; two of them make one unusable."""
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
    path = written_set(tmp_path, {"id": "fuori", "question": "Chi ha vinto?"})

    (question,) = load_questions(path)

    assert question.document is None


# --- measuring a search ----------------------------------------------------


def test_a_passage_of_the_expected_document_is_a_hit() -> None:
    retriever = Retriever({"Prima?": [passage(MANUAL, 1), passage(REPORT, 1)]})

    report = run_evals(
        [question(question="Prima?")], retriever=retriever, in_scope=[MANUAL, REPORT]
    )

    assert report.results[0].rank == 1
    assert retriever.queries == ["Prima?"]


def test_nothing_from_the_expected_document_is_a_miss() -> None:
    retriever = Retriever({"Prima?": [passage(REPORT, 1)]})

    report = run_evals(
        [question(question="Prima?")], retriever=retriever, in_scope=[MANUAL, REPORT]
    )

    assert report.results[0].rank is None


def test_the_rank_is_where_the_document_was_found() -> None:
    retriever = Retriever({
        "Prima?": [passage(REPORT, 1), passage(REPORT, 2), passage(MANUAL, 5)]
    })

    report = run_evals(
        [question(question="Prima?")], retriever=retriever, in_scope=[MANUAL, REPORT]
    )

    assert report.results[0].rank == 3


def test_a_page_is_measured_on_its_own() -> None:
    """The document can be found and the page that answers the question not be."""
    retriever = Retriever({"Prima?": [passage(MANUAL, 2), passage(MANUAL, 8)]})

    report = run_evals(
        [question(question="Prima?", page=9)],
        retriever=retriever,
        in_scope=[MANUAL],
    )

    assert report.results[0].rank == 1
    assert report.results[0].page_rank == 2


def test_the_page_is_counted_as_a_reader_counts_it() -> None:
    """A reader's page 12 is the store's `page` 11, and both halves know it."""
    retriever = Retriever({"Prima?": [passage(MANUAL, 11)]})

    report = run_evals(
        [question(question="Prima?", page=12)],
        retriever=retriever,
        in_scope=[MANUAL],
    )

    assert report.results[0].page_rank == 1


def test_a_question_that_names_no_page_has_no_page_rank() -> None:
    retriever = Retriever({"Prima?": [passage(MANUAL, 3)]})

    report = run_evals(
        [question(question="Prima?")], retriever=retriever, in_scope=[MANUAL]
    )

    assert report.results[0].rank == 1
    assert report.results[0].page_rank is None


def test_a_question_outside_the_scope_is_not_asked() -> None:
    retriever = Retriever({"Prima?": [passage(MANUAL, 1)]})

    report = run_evals(
        [question(question="Prima?")], retriever=retriever, in_scope=[REPORT]
    )

    assert report.results[0].asked is False
    assert retriever.queries == []


def test_a_question_with_no_document_is_asked_of_the_whole_scope() -> None:
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
    report = run_evals(
        [question(question="Prima?")], retriever=BrokenRetriever(), in_scope=[MANUAL]
    )

    assert report.results[0].error == "the index is unreachable"
    assert report.results[0].asked is True


def test_the_rows_of_the_answer_are_the_passages_the_search_returned() -> None:
    retriever = Retriever({"Prima?": [passage(MANUAL, 4, "the warranty is two years")]})

    report = run_evals(
        [question(question="Prima?")], retriever=retriever, in_scope=[MANUAL]
    )

    assert report.results[0].sources == [
        {"source": MANUAL, "page": 5, "ranges": []}
    ]


# --- measuring an answer ---------------------------------------------------


def test_no_answer_is_written_unless_a_model_is_given() -> None:
    retriever = Retriever({"Prima?": [passage(MANUAL, 1)]})

    report = run_evals(
        [question(question="Prima?")], retriever=retriever, in_scope=[MANUAL]
    )

    assert report.results[0].answer is None


def test_the_answer_is_written_from_the_passages_that_were_found() -> None:
    """The half that costs money is measured on the search that was measured."""
    retriever = Retriever({"Prima?": [passage(MANUAL, 4, "the warranty is two years")]})
    model = FakeChatModel(replies=["Two years."])

    run_evals(
        [question(question="Prima?")],
        retriever=retriever,
        in_scope=[MANUAL],
        chat_model=model,
    )

    asked = model.prompts[0][1].content

    assert "the warranty is two years" in asked
    assert "Prima?" in asked


def test_an_answer_that_does_not_hold_what_was_expected_is_reported_missing() -> None:
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
    assert missing_from("Due   anni di garanzia.", ["due anni"]) == []
    assert missing_from("Two years.", ["due anni"]) == ["due anni"]


# --- the judge -------------------------------------------------------------


def test_a_judge_that_says_yes_is_faithful() -> None:
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
    assert read_verdict(reply)[0] is verdict


def test_a_verdict_of_no_keeps_what_follows_it() -> None:
    _, claims = read_verdict("no\n\nthe price is not there\nand the date is wrong")

    assert claims == ["the price is not there", "and the date is wrong"]


# --- the numbers -----------------------------------------------------------

def test_the_numbers_add_up() -> None:
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
    # The first found the document first, which is the best order it had: 1.0.
    # The second found the document and not the page, and put the one passage of
    # it third — against the ideal, which is that passage first: 1/log2(4). The
    # third was missed: 0, and it is in the divisor, so a miss pulls the mean down
    # rather than being left out of it.
    assert summary.ndcg == pytest.approx((1.0 + 1 / log2(4) + 0.0) / 3)


def test_a_run_with_nothing_to_measure_has_no_rank() -> None:
    report = EvalReport(results=[QuestionResult(question=question(id="saltata"), asked=False)])

    summary = summarise(report)

    assert summary.reciprocal_rank == 0.0
    assert summary.ndcg == 0.0


def test_a_failed_search_scores_no_better_than_a_missed_one() -> None:
    report = EvalReport(results=[
        QuestionResult(
            question=question(id="uno"), error="the index is unreachable"
        ),
    ])

    assert summarise(report).ndcg == 0.0


# --- nDCG ------------------------------------------------------------------


def test_the_best_order_the_passages_allowed_scores_one() -> None:
    assert ndcg_at((2, 1, 0)) == pytest.approx(1.0)
    assert ndcg_at((1, 0)) == pytest.approx(1.0)
    assert ndcg_at(()) == 0.0


def test_the_right_page_below_a_wrong_one_does_not() -> None:
    # The page that answers is second, behind a passage of the same document:
    # 1 at the top and 2 discounted by log2(3), against the other order of the
    # same two, which is 2 at the top and 1 below it.
    assert ndcg_at((1, 2)) == pytest.approx(
        (1 + 2 / log2(3)) / (2 + 1 / log2(3))
    )
    assert ndcg_at((1, 2)) < ndcg_at((2, 1))


def test_the_same_passage_lower_down_scores_less() -> None:
    assert ndcg_at((0, 0, 2, 0)) == pytest.approx((2 / log2(4)) / 2)
    assert ndcg_at((0, 0, 2, 0)) < ndcg_at((2, 0, 0, 0))


def test_a_ranking_that_found_nothing_scores_nothing() -> None:
    """Nothing relevant is an ideal of zero, which is 0 rather than a division."""
    assert ndcg_at((0, 0, 0)) == 0.0
    assert ndcg_at(()) == 0.0


def test_ordering_is_read_against_what_the_search_returned() -> None:
    """Right document, wrong page, best order, is 1.0 — and that is deliberate.

    The ideal is the passages that came back, so a search is not punished for
    what it never retrieved: its ordering was not the problem, and the page rate
    printed beside it is what says the page was missed.
    """
    assert ndcg_at((1, 1, 0)) == pytest.approx(1.0)
    assert ndcg_at((0, 1, 1)) < 1.0


def test_the_numbers_about_finding_are_read_over_what_was_shown() -> None:
    """A call asks the search for more than the console shows, so that a ranking
    has ten passages to be read to the end of. The two are not the same number,
    and a document at position six came back to the harness and never to the
    reader: counting it as found would report a search the console does not have.
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
    # The ordering is still read over the whole of it: what a question scores for
    # the order of the passages does not depend on how many of them were shown.
    assert shown.results[0].grades == everything.results[0].grades


def test_a_run_with_no_read_at_reads_everything_it_was_given() -> None:
    """What a whole-document scope needs: its passages are the document, and
    there is no fifth of it to stop at."""
    retriever = Retriever({
        "Prima?": [passage(REPORT, 1) for _ in range(5)] + [passage(MANUAL, 1)]
    })

    report = run_evals(
        [question(question="Prima?")], retriever=retriever, in_scope=[MANUAL, REPORT]
    )

    assert report.results[0].rank == 6


def test_a_passage_past_the_cutoff_is_not_read() -> None:
    """A cutoff that is not read is a cutoff that would flatter a long ranking."""
    assert ndcg_at((2,), cutoff=1) == pytest.approx(1.0)
    assert ndcg_at((0, 2), cutoff=1) == 0.0
    assert ndcg_at((2,), cutoff=0) == 0.0


def test_the_cutoff_is_ten_by_default() -> None:
    """Ten, not five: the console's own k is a different number and not this one."""
    below = (0,) * (NDCG_CUTOFF - 1) + (2,)
    past = (0,) * NDCG_CUTOFF + (2,)

    assert NDCG_CUTOFF == 10
    assert ndcg_at(below) > 0
    assert ndcg_at(past) == 0.0


# --- what the passages are graded by ---------------------------------------


def test_a_passage_is_graded_on_the_page_the_question_named() -> None:
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

    # Another document, then the right document on another page, then the page
    # itself. The middle one is worth something: it is where the answer is not,
    # inside the document it is in.
    assert report.results[0].grades == (0, 1, 2)


def test_a_question_that_named_no_page_has_one_grade_to_give() -> None:
    retriever = Retriever({"Prima?": [passage(REPORT, 0), passage(MANUAL, 3)]})

    report = run_evals([question(question="Prima?")], retriever=retriever)

    assert report.results[0].grades == (0, 1)


def test_a_search_that_returned_nothing_has_no_grades() -> None:
    report = run_evals([question(question="Prima?")], retriever=Retriever({}))

    assert report.results[0].grades == ()
    assert summarise(report).ndcg == 0.0


def test_a_question_that_could_not_be_asked_has_no_grades() -> None:
    report = run_evals(
        [question(question="Prima?")], retriever=Retriever({}), in_scope=[REPORT]
    )

    assert report.results[0].asked is False
    assert report.results[0].grades == ()


def test_a_failed_search_is_counted_as_a_failure() -> None:
    report = EvalReport(results=[
        QuestionResult(question=question(id="uno"), error="the index is unreachable"),
    ])

    summary = summarise(report)

    assert summary.failed == 1
    assert summary.measurable == 1
    assert summary.hits == 0


def test_rank_of_ignores_a_document_it_was_not_asked_about() -> None:
    passages = [passage(REPORT, 1), passage(MANUAL, 1)]

    assert rank_of(passages, None) is None
    assert rank_of(passages, MANUAL) == 2
    # The chunk is on the reader's page 2, and this asks for the third.
    assert rank_of(passages, MANUAL, page=2) == 2
    assert rank_of(passages, MANUAL, page=3) is None
