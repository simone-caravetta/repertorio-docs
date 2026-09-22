"""Whether the material a search returned can answer the question.

A similarity search always returns something, because it has no notion of
nothing. A question the library cannot answer still comes back with the nearest
passages there are, and an answer written from those is written on no evidence.

This module asks the chat model one thing about that pair: does this material
hold what is needed to answer? The verdict says yes or no, gives the reason in
one sentence and, when searching again is worth it, carries a query written
from that reason.

The verdict lives here rather than inside the graph so that it can be measured
on its own. The graph calls it from a node, and the eval harness calls the same
function, which is what lets a question set score the verdict without running
the whole graph.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import Settings

# The two values GRADE accepts. Any other value is refused with a message that
# names them.
MODES = ("on", "off")


def enabled(config: Settings) -> bool:
    """Whether the material is graded before an answer is written.

    GRADE has to be "on" or "off". Any other value raises an error listing the
    two words that are accepted.
    """
    mode = config.grade.strip().lower()

    if mode == "on":
        return True
    if mode == "off":
        return False

    raise RuntimeError(
        f"Unknown GRADE {config.grade!r}: expected " + " or ".join(MODES) + "."
    )


def describe_grade(config: Settings) -> str:
    """One line about grading, for a command to print when it starts.

    The line carries the number of searches, because two runs that disagree
    about a question may have searched a different number of times for it.

    A value that is not "on" is printed as it was written. The run stops later,
    when the graph is built, where the mode has to be honoured.
    """
    mode = config.grade.strip().lower()
    if mode != "on":
        return mode

    return f"on, at most {config.grade_attempts} searches per question"


# The prompt asks for one JSON object, so the reply is read back as three fields
# instead of being parsed out of a sentence. The shape is spelled out in the
# prompt itself, which is what keeps the reply readable without a provider that
# speaks structured output.
grade_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You check whether the material a search returned can answer a question.

You are given a question and the material a search returned for it. Decide
whether that material holds enough to answer the question.

The material opens with the documents the search ran over and, under each,
what the catalogue says about it. Under that come the passages themselves,
each labelled with its source and its page. Both are part of the material.

Answer false when the passages are about something else, when they are too
general to settle the question, or when the question asks for a fact that
none of them states. An answer written from a passage that does not hold it
would be invented, so answer false rather than settling for the closest
thing.

Answer true when a reader could answer from the material, even when the
answer is spread over more than one passage and even when it is incomplete.

The question usually describes a case, and the material usually states the
rule for cases of that kind. Answer true when the rule settles the case the
question describes, whether it settles it by covering the case or by leaving
it out. A question about a case the material's own condition excludes is
answered by that condition, and so is a question giving an age, a distance or
a date where the material states a limit: the answer is that the case falls
outside the rule, and that is an answer.

The reason is one sentence about the material, and it and the query are
written in the language of the question.

When the answer is false, write a query that would find what is missing.
Name the thing the question is about in the words a document would use, and
leave out what the material already settled. When the answer is true, leave
the query empty.

Reply with one JSON object and nothing else, in this shape:
{{"supported": true, "reason": "...", "query": "..."}}

Nothing comes before the object or after it, and none of your reasoning is
written out: it belongs in the reason field, in one sentence.""",
    ),
    (
        "human",
        """Question:
{question}

Material:
{context}""",
    ),
])


@dataclass(frozen=True)
class Verdict:
    """What the model decided about one question and the material it was given.

    `supported` is True when the material holds enough to answer. `reason` is
    one sentence about the material. `query` is a query to search again with,
    and it is empty when there is nothing worth searching for.
    """

    supported: bool
    reason: str
    query: str = ""


def parse_verdict(reply: str) -> Verdict:
    """Read a model's reply as a verdict.

    The object is the first one in the reply that reads as JSON, taken from its
    opening brace to the brace that closes it. That is what makes a reply wrapped
    in a code fence readable without a rule for the fence, and a reply with a
    sentence around it readable without a rule for the sentence.

    The first object, not the text between the first brace and the last: a model
    that says the same thing twice writes two objects, which together are not
    JSON, and reading them as one fails on the reply at the moment the model is
    most sure of itself. The braces are tried in turn rather than only the first
    one, because a model that argues with itself writes braces in the argument
    and puts the object after them. Both were measured on a question set of 90.

    A reply that is not one object with the three fields raises ValueError with
    a message saying what was wrong. A caller records the question as failed
    rather than reading a default as a verdict.
    """

    at = reply.find("{")
    if at == -1:
        raise ValueError(f"The verdict is not a JSON object: {reply!r}")

    decoder = json.JSONDecoder()
    read: Any = None

    # Each opening brace is tried in turn, and the first one that reads as JSON
    # is the verdict. A model that argues with itself writes braces in the
    # argument — `{the driver or the insurer}` — and the object it means comes
    # after them, so the first brace is not always the one holding it.
    while at != -1:
        try:
            read, _ = decoder.raw_decode(reply, at)
        except json.JSONDecodeError:
            at = reply.find("{", at + 1)
            continue
        break

    if at == -1:
        raise ValueError(f"The verdict is not readable as JSON: {reply!r}")

    # Every failure below is one thing going wrong the same way for the caller,
    # so they all raise ValueError rather than TypeError.
    if not isinstance(read, dict):
        raise ValueError(  # noqa: TRY004 - the types come from the reply, not the caller
            f"The verdict is not a JSON object: {reply!r}"
        )

    supported = read.get("supported")
    if not isinstance(supported, bool):
        raise ValueError(  # noqa: TRY004 - same
            f"The verdict does not answer true or false in supported: {reply!r}"
        )

    reason = read.get("reason")
    if not isinstance(reason, str):
        raise ValueError(  # noqa: TRY004 - same
            f"The verdict has no reason: {reply!r}"
        )

    query = read.get("query", "")
    if not isinstance(query, str):
        raise ValueError(  # noqa: TRY004 - same
            f"The verdict has a query that is not text: {reply!r}"
        )

    return Verdict(supported=supported, reason=reason, query=query.strip())


def grade(chat_model: BaseChatModel, question: str, context: str) -> Verdict:
    """Ask the model whether this material can answer this question.

    `context` is the material as the answer would be written from it, so that
    the verdict is about exactly what the answer is given.
    """

    reply = _chain(chat_model).invoke({"question": question, "context": context})
    return parse_verdict(reply)


async def agrade(chat_model: BaseChatModel, question: str, context: str) -> Verdict:
    """The same, for a caller inside the graph."""

    reply = await _chain(chat_model).ainvoke({"question": question, "context": context})
    return parse_verdict(reply)


def _chain(chat_model: BaseChatModel):
    # Built per call, which costs nothing: the three runnables are already made
    # and the sequence is a list of them.
    return grade_prompt | chat_model | StrOutputParser()
