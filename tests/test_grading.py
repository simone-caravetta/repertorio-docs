"""Whether the material a search returned can answer the question.

The model answers with one JSON object holding three fields. These tests cover
how that reply is read, what a reply that cannot be read raises, and what the
model is asked.
"""

from __future__ import annotations

import json

import pytest

from app.grading import Verdict, agrade, describe_grade, enabled, grade, parse_verdict
from tests.helpers import FakeChatModel, make_settings


def said(**fields: object) -> str:
    """A reply in the shape the prompt asks for.

    Any of the three fields can be replaced, and a field left out stays out,
    which is how a reply missing one is built.
    """

    shape: dict[str, object] = {
        "supported": True,
        "reason": "The passage states it.",
        "query": "",
    }
    shape.update(fields)
    return json.dumps(shape)


# Reading a reply the model wrote.


def test_a_supported_verdict_carries_its_reason_and_no_query():
    verdict = parse_verdict(said(reason="The page states the term."))

    assert verdict == Verdict(supported=True, reason="The page states the term.")


def test_an_unsupported_verdict_carries_a_query_to_search_again_with():
    verdict = parse_verdict(
        said(supported=False, reason="No passage mentions it.", query="warranty term")
    )

    assert verdict.supported is False
    assert verdict.reason == "No passage mentions it."
    assert verdict.query == "warranty term"


def test_a_reply_with_no_query_leaves_it_empty():
    assert parse_verdict('{"supported": true, "reason": "It is there."}').query == ""


def test_a_reply_wrapped_in_a_code_fence_is_still_read():
    """The object is taken between its first brace and its last.

    A model asked for JSON wraps it in a fence often enough that a rule for
    the fence would be one more thing to maintain.
    """

    fenced = f"```json\n{said()}\n```"

    assert parse_verdict(fenced).supported is True


def test_a_reply_with_a_sentence_around_it_is_still_read():
    around = f"Here is the verdict.\n{said(supported=False, reason='Nothing on it.')}"

    assert parse_verdict(around).supported is False


def test_the_query_is_trimmed():
    assert parse_verdict(said(query="  warranty term  ")).query == "warranty term"


# A reply that cannot be read. Each one raises, so a caller records the question
# as failed instead of reading a default as a verdict.


def test_a_reply_that_is_not_an_object_is_refused():
    with pytest.raises(ValueError, match="not a JSON object"):
        parse_verdict("The passages do not support the question.")


def test_a_reply_that_is_not_readable_as_json_is_refused():
    with pytest.raises(ValueError, match="not readable as JSON"):
        parse_verdict('{"supported": true, "reason": }')


def test_a_supported_that_is_not_true_or_false_is_refused():
    with pytest.raises(ValueError, match="true or false"):
        parse_verdict('{"supported": "yes", "reason": "It is there."}')


def test_a_verdict_with_no_reason_is_refused():
    with pytest.raises(ValueError, match="no reason"):
        parse_verdict('{"supported": true}')


def test_a_reason_that_is_not_text_is_refused():
    with pytest.raises(ValueError, match="no reason"):
        parse_verdict('{"supported": true, "reason": 3}')


def test_a_query_that_is_not_text_is_refused():
    with pytest.raises(ValueError, match="not text"):
        parse_verdict('{"supported": false, "reason": "Nothing on it.", "query": 3}')


# Asking the model.


def test_the_model_is_given_the_question_and_the_material_it_judges():
    model = FakeChatModel(replies=[said()])

    grade(model, "how long is the warranty?", "The warranty runs for two years.")

    asked = "\n".join(str(one.content) for one in model.prompts[0])
    assert "how long is the warranty?" in asked
    assert "The warranty runs for two years." in asked


def test_a_reply_that_cannot_be_read_stops_the_caller():
    model = FakeChatModel(replies=["I cannot read those passages."])

    with pytest.raises(ValueError, match="not a JSON object"):
        grade(model, "q", "material")


@pytest.mark.asyncio
async def test_the_graph_can_ask_the_same_question():
    """A node in the graph asks for the verdict with await."""

    model = FakeChatModel(
        replies=[said(supported=False, reason="Nothing on it.", query="try this")]
    )

    verdict = await agrade(model, "q", "material")

    assert verdict.supported is False
    assert verdict.query == "try this"


# GRADE, and the line a command prints about it.


def test_grading_is_on_only_when_the_setting_says_on():
    assert enabled(make_settings(grade="on")) is True
    assert enabled(make_settings(grade="off")) is False


def test_the_setting_is_read_regardless_of_case_and_spacing():
    assert enabled(make_settings(grade=" On ")) is True


def test_a_setting_that_is_neither_word_is_refused():
    """A value the project does not know stops the run and names the two it does."""

    with pytest.raises(RuntimeError, match='Unknown GRADE .* expected on or off'):
        enabled(make_settings(grade="maybe"))


def test_the_line_says_how_many_searches_a_question_may_take():
    assert describe_grade(make_settings(grade="on", grade_attempts=2)) == (
        "on, at most 2 searches per question"
    )


def test_the_line_prints_a_setting_it_does_not_know_as_it_was_written():
    """The run stops later, where the mode has to be honoured, not here."""

    assert describe_grade(make_settings(grade="maybe")) == "maybe"
    assert describe_grade(make_settings(grade="off")) == "off"
