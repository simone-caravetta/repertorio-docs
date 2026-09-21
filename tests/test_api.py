"""Tests for the HTTP API.

A test builds the app around a fake chat model and a fake retriever, so the
endpoints answer without a network call or a vector store. The chat answers
are read back as server-sent events, one block at a time, which is how the
console reads them too.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app import pdf
from app.api import create_app
from app.catalog import Catalog
from app.config import Settings
from app.scope import Scope
from tests.helpers import (
    FakeChatModel,
    FakeRetriever,
    Line,
    make_pdf,
    make_settings,
    make_structured_pdf,
    verdict_reply,
)

MANUAL = "manuals/manual.pdf"
ANNEX = "manuals/annex.pdf"
REPORT = "reports/report.pdf"
LOOSE = "notes.pdf"


# The refusal a category with nothing indexed produces.
NOWHERE = "No indexed documents in the category 'empty'."


@dataclass
class Harness:
    """An app together with the fakes behind it.

    The scopes list records the scope each retriever was built for.
    """

    app: FastAPI
    model: FakeChatModel
    retriever: FakeRetriever

    scopes: list[Scope] = field(default_factory=list)

    def client(self) -> TestClient:
        return TestClient(self.app)


@pytest.fixture
def config(tmp_path: Path, db_path: Path) -> Settings:
    """Settings with the catalog, the documents and the threads in tmp_path."""
    return make_settings(
        catalog_db_path=db_path,
        documents_dir=tmp_path / "documents",
        conversations_db_path=tmp_path / "conversations.sqlite3",
    )


def harness_for(
    config: Settings,
    *,
    replies: list[str],
    documents: list[Document] | None = None,
    checkpointer: Any = None,
) -> Harness:
    """Build an app whose model replies with the given texts.

    The passages the retriever hands back are the ones given, or a single
    example passage. Every retriever the app builds is kept on the harness,
    so a test can see which scope was asked for.
    """
    model = FakeChatModel(replies=replies)
    retriever = FakeRetriever(
        [passage()] if documents is None else documents
    )
    harness = Harness(app=None, model=model, retriever=retriever)  # type: ignore[arg-type]

    def retriever_for(scope: Scope) -> FakeRetriever:
        harness.scopes.append(scope)
        return retriever

    harness.app = create_app(
        checkpointer=checkpointer,
        chat_model=model,
        retriever_for=retriever_for,
        config=config,
    )
    return harness


def passage(
    source: str = MANUAL,
    page: int = 11,
    chunk: int = 0,
    start: int = 0,
    end: int = 28,
) -> Document:
    """A retrieved passage, with the metadata a real one carries."""
    return Document(
        page_content="The thing is explained here.",
        metadata={
            "source": source,
            "page": page,
            "chunk_id": chunk,
            "start": start,
            "end": end,
        },
    )


def file_document(
    db_path: Path,
    path: str,
    *,
    title: str | None = None,
    category: str | None = None,
    status: str = "indexed",
    pages: int = 3,
    chunks: int = 4,
) -> None:
    """Put one document in the catalog, at the status asked for.

    The title defaults to the file name, and the counts are what an indexed
    document of that size would have.
    """
    catalog = Catalog(db_path)
    catalog.add_file(path, title or Path(path).stem, category)

    if status == "indexed":
        catalog.record_indexed(
            path, file_hash="hash", page_count=pages, chunk_count=chunks
        )
    elif status == "trashed":
        catalog.trash(path)
    elif status == "failed":
        catalog.record_failed(path, "No text extracted")


def described(db_path: Path, path: str, text: str) -> None:
    """Write a description onto a document that is already in the catalog."""
    Catalog(db_path).set_description(path, text)


def events_of(response: Any) -> list[tuple[str, dict[str, Any]]]:
    """Read a streamed answer as a list of event name and payload pairs.

    Each block of the body holds an event line and a data line. The data is
    JSON, which is what the console parses.
    """
    events: list[tuple[str, dict[str, Any]]] = []

    for block in response.text.split("\n\n"):
        if not block.strip():
            continue

        name: str | None = None
        payload: dict[str, Any] | None = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                payload = json.loads(line[len("data: ") :])

        assert name is not None and payload is not None, block
        events.append((name, payload))

    return events


def names_of(events: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """The event names, in the order they arrived."""
    return [name for name, _ in events]


def ask(client: TestClient, question: str, **scope: Any) -> Any:
    """Ask a question, passing any scope along in the request body."""
    return client.post("/api/chat", json={"question": question, **scope})


# The catalog endpoint, which returns the documents of the library grouped
# into the tree the console shows.
def test_documents_are_grouped_under_the_category_they_are_filed_in(
    db_path: Path, config: Settings
):
    """Documents come back under the category they are filed in.

    Each row carries the status and the counts from indexing, and a
    document with no description yet has None there.
    """
    file_document(db_path, MANUAL, title="The manual", category="manuals", pages=12)
    file_document(db_path, ANNEX, title="The annex", category="manuals")
    file_document(db_path, REPORT, title="The report", category="reports")

    with harness_for(config, replies=[]).client() as client:
        view = client.get("/api/catalog").json()

    assert view["empty"] is False
    assert [branch["name"] for branch in view["categories"]] == [
        "manuals",
        "reports",
    ]

    manuals = view["categories"][0]
    assert manuals["total"] == 2
    assert [row["title"] for row in manuals["documents"]] == [
        "The annex",
        "The manual",
    ]

    manual = manuals["documents"][1]
    assert manual["path"] == MANUAL
    assert manual["status"] == "indexed"
    assert manual["pages"] == 12
    assert manual["chunks"] == 4

    # Nothing has described the manual, so the field is empty.
    assert manual["description"] is None
    assert view["categories"][1]["documents"][0]["title"] == "The report"


def test_a_category_holding_no_document_of_its_own_still_appears(
    db_path: Path, config: Settings
):
    """A parent category with nothing filed directly in it is still shown.

    It has no documents of its own and counts the one below it.
    """
    file_document(db_path, "manuals/2024/manual.pdf", category="manuals/2024")

    with harness_for(config, replies=[]).client() as client:
        view = client.get("/api/catalog").json()

    assert view["categories"][0]["name"] == "manuals"
    assert view["categories"][0]["documents"] == []
    assert view["categories"][0]["total"] == 1
    assert view["categories"][0]["children"][0]["name"] == "manuals/2024"


def test_documents_in_no_category_are_listed_apart_from_the_tree(
    db_path: Path, config: Settings
):
    """Documents with no category are returned in their own list."""
    file_document(db_path, MANUAL, category="manuals")
    file_document(db_path, LOOSE, title="A note")

    with harness_for(config, replies=[]).client() as client:
        view = client.get("/api/catalog").json()

    assert [branch["name"] for branch in view["categories"]] == ["manuals"]
    assert [row["title"] for row in view["uncategorized"]] == ["A note"]


def test_a_trashed_document_is_left_out(db_path: Path, config: Settings):
    """A trashed document is not listed under its category."""
    file_document(db_path, MANUAL, category="manuals")
    file_document(db_path, ANNEX, title="Gone", status="trashed")

    with harness_for(config, replies=[]).client() as client:
        view = client.get("/api/catalog").json()

    assert [row["path"] for row in view["categories"][0]["documents"]] == [MANUAL]


def test_a_document_that_failed_to_index_is_still_shown(
    db_path: Path, config: Settings
):
    """A document that failed appears, so the console can show the failure."""
    file_document(db_path, MANUAL, category="manuals", status="failed")

    with harness_for(config, replies=[]).client() as client:
        view = client.get("/api/catalog").json()

    row = view["categories"][0]["documents"][0]
    assert row["status"] == "failed"


def test_no_catalog_at_all_is_an_empty_view_and_writes_nothing(
    db_path: Path, config: Settings
):
    """With no catalog file the view is empty and no file is created."""
    with harness_for(config, replies=[]).client() as client:
        view = client.get("/api/catalog").json()

    assert view == {"empty": True, "categories": [], "uncategorized": []}
    assert not db_path.exists()


# The scope endpoint, which turns a typed category or document into the set
# of sources a search will run over.
def test_the_scope_label_is_the_one_the_console_prints(
    db_path: Path, config: Settings
):
    """A category resolves to its indexed documents, sorted by path.

    The label is the one the console prints for that scope.
    """
    file_document(db_path, MANUAL, category="manuals")
    file_document(db_path, ANNEX, category="manuals")

    with harness_for(config, replies=[]).client() as client:
        resolved = client.get("/api/scope", params={"category": "manuals"}).json()

    assert resolved["label"] == "category manuals (2 documents)"
    assert resolved["sources"] == [ANNEX, MANUAL]
    assert resolved["whole_document"] is False


def test_the_whole_library_is_no_restriction_rather_than_no_documents(
    db_path: Path, config: Settings
):
    """With no category and no document, the whole library is searched.

    That is signalled by the sources being None instead of a list.
    """
    file_document(db_path, MANUAL, category="manuals")

    with harness_for(config, replies=[]).client() as client:
        resolved = client.get("/api/scope").json()

    assert resolved["label"] == "whole library"
    assert resolved["sources"] is None
    assert resolved["whole_document"] is False


def test_a_category_with_nothing_indexed_is_refused_in_the_resolvers_words(
    db_path: Path, config: Settings
):
    """A category with nothing indexed is refused with the resolver's words."""
    file_document(db_path, MANUAL, category="empty", status="failed")

    with harness_for(config, replies=[]).client() as client:
        response = client.get("/api/scope", params={"category": "empty"})

    assert response.status_code == 400
    assert response.json()["detail"].startswith(NOWHERE)


def test_a_question_about_nothing_is_refused_before_any_stream_begins(
    db_path: Path, config: Settings
):
    """A question about an empty category fails before the stream starts.

    The refusal is a plain error response, not a stream carrying an error
    event.
    """
    file_document(db_path, MANUAL, category="manuals")

    with harness_for(config, replies=[]).client() as client:
        response = ask(client, "what does it say?", category="empty")

    assert response.status_code == 400
    assert response.json()["detail"].startswith(NOWHERE)
    assert response.headers["content-type"] != "text/event-stream"


def test_the_retriever_is_built_from_the_scope_that_was_asked_for(
    db_path: Path, config: Settings
):
    """Narrowing a question to one document builds a retriever for it.

    That scope covers the whole document.
    """
    file_document(db_path, MANUAL, category="manuals")

    harness = harness_for(config, replies=["a standalone question", "the answer"])
    with harness.client() as client:
        response = ask(client, "what does it say?", document=MANUAL)

    assert response.status_code == 200
    assert [scope.sources for scope in harness.scopes] == [(MANUAL,)]

    assert harness.scopes[0].whole_document is True


# The boxes endpoint, which maps a character range in a page of a PDF back
# to the rectangles that cover it.
FIRST_LINE = "The first line of the page."
SECOND_LINE = "The second line, below it."
TWO_LINES = "manuals/two-lines.pdf"


def a_page_of_two_lines(config: Settings, name: str = TWO_LINES) -> str:
    """Write a one-page PDF of two lines and read its text back.

    The text comes from the same reader the API uses, so the offsets found
    in it are the ones the endpoint is asked about.
    """
    path = make_structured_pdf(
        Path(config.documents_dir) / name,
        [[Line(FIRST_LINE), Line(SECOND_LINE)]],
    )

    return pdf.read_pages(path)[0].text


def boxes_of(client: TestClient, **params: Any) -> Any:
    """Ask for the boxes of a range of the two-line page."""
    return client.get(
        "/api/documents/boxes",
        params={"source": TWO_LINES, "page": 1, "start": 0, "end": 1, **params},
    )


def test_a_range_comes_back_as_the_boxes_that_cover_it(config: Settings):
    """A range over the first line comes back as one box.

    The box starts at the left margin and is about a line tall.
    """
    text = a_page_of_two_lines(config)
    start = text.index(FIRST_LINE)

    with harness_for(config, replies=[]).client() as client:
        answer = boxes_of(client, start=start, end=start + len(FIRST_LINE))

    assert answer.status_code == 200
    assert answer.json()["source"] == TWO_LINES
    assert answer.json()["page"] == 1

    boxes = answer.json()["boxes"]
    assert len(boxes) == 1
    x0, y0, x1, y1 = boxes[0]

    # A line of text starts at the left margin and stops before the edge.
    assert x0 == pytest.approx(72, abs=1)
    assert x1 < 612
    assert 0 < y1 - y0 < 20


def test_a_range_across_two_lines_is_two_boxes(config: Settings):
    """A range over both lines comes back as two boxes.

    They are in the order the lines are read.
    """
    text = a_page_of_two_lines(config)
    start = text.index(FIRST_LINE)
    end = text.index(SECOND_LINE) + len(SECOND_LINE)

    with harness_for(config, replies=[]).client() as client:
        boxes = boxes_of(client, start=start, end=end).json()["boxes"]

    assert len(boxes) == 2

    # The second box sits below the first one.
    assert boxes[0][1] < boxes[1][1]


def test_a_range_that_covers_nothing_is_an_empty_answer(config: Settings):
    """A range over the break between the two lines covers no text.

    The answer is an empty list of boxes at status 200.
    """
    text = a_page_of_two_lines(config)
    gap = text.index("\n")

    with harness_for(config, replies=[]).client() as client:
        answer = boxes_of(client, start=gap, end=gap + 1)

    assert answer.status_code == 200
    assert answer.json()["boxes"] == []


def test_a_page_needs_no_catalog(db_path: Path, config: Settings):
    """Boxes are read from the PDF, so no catalog is needed.

    Asking for them does not create a catalog file.
    """
    assert not db_path.exists()
    a_page_of_two_lines(config)

    with harness_for(config, replies=[]).client() as client:
        answer = boxes_of(client, start=0, end=len(FIRST_LINE))

    assert answer.status_code == 200


def test_a_source_outside_the_documents_folder_is_refused(
    tmp_path: Path, config: Settings
):
    """A source that climbs out of the documents folder is refused.

    The answer says the file is not a document here.
    """
    make_pdf(tmp_path / "outside.pdf", "Not in the library.")

    with harness_for(config, replies=[]).client() as client:
        answer = boxes_of(client, source="../outside.pdf", start=0, end=4)

    assert answer.status_code == 400
    assert answer.json()["detail"] == "Not a document here: ../outside.pdf"


def test_a_file_that_is_not_a_pdf_is_refused(config: Settings):
    """A file in the folder that is not a PDF is refused as a source."""
    make_pdf(Path(config.documents_dir) / "manuals" / "notes.txt", "Not a PDF here.")

    with harness_for(config, replies=[]).client() as client:
        answer = boxes_of(client, source="manuals/notes.txt", start=0, end=4)

    assert answer.status_code == 400
    assert answer.json()["detail"] == "Not a document here: manuals/notes.txt"


def test_a_document_that_is_not_there_is_refused(config: Settings):
    """A source with no file behind it is refused."""
    with harness_for(config, replies=[]).client() as client:
        answer = boxes_of(client, source="manuals/absent.pdf", start=0, end=4)

    assert answer.status_code == 400
    assert answer.json()["detail"] == "Not a document here: manuals/absent.pdf"


@pytest.mark.parametrize("page", [0, 2, 99])
def test_a_page_the_document_does_not_have_is_refused(config: Settings, page: int):
    """A page beyond the one page of the file is refused.

    Page numbers are counted from one, so zero is refused too.
    """
    text = a_page_of_two_lines(config)
    start = text.index(FIRST_LINE)

    with harness_for(config, replies=[]).client() as client:
        answer = boxes_of(client, page=page, start=start, end=start + len(FIRST_LINE))

    assert answer.status_code == 400
    assert answer.json()["detail"] == f"No page {page} in {TWO_LINES}"


@pytest.mark.parametrize(
    ("start", "end"), [(-1, 5), (40, 20), (20, 20)]
)
def test_a_range_that_is_not_one_is_refused(config: Settings, start: int, end: int):
    """A range that is negative, backwards or empty is refused."""
    a_page_of_two_lines(config)

    with harness_for(config, replies=[]).client() as client:
        answer = boxes_of(client, start=start, end=end)

    assert answer.status_code == 400
    assert answer.json()["detail"] == f"Not a range: {start}-{end}"


# The chat endpoint, which streams the events the console reads.
def test_the_sources_arrive_before_the_answer_is_written(
    db_path: Path, config: Settings
):
    """The stream opens with the thread, the query and the sources.

    The tokens come after the sources, so the console can show what was read
    before the answer arrives. Passages of one page are merged into a single
    entry with a range for each of them.
    """
    file_document(db_path, MANUAL, category="manuals")
    harness = harness_for(
        config,
        replies=["a standalone question", "the answer is here"],

        # Three passages over two pages, two on the first and one on the next.
        documents=[
            passage(),
            passage(chunk=1, start=300, end=328),
            passage(page=12),
        ],
    )

    with harness.client() as client:
        response = ask(client, "what does it say?")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = events_of(response)
    assert names_of(events)[0] == "thread"
    assert events[1] == ("query", {"query": "a standalone question"})
    assert events[2] == (
        "sources",
        {
            "sources": [
                {"source": MANUAL, "page": 12, "ranges": [[0, 28], [300, 328]]},
                {"source": MANUAL, "page": 13, "ranges": [[0, 28]]},
            ]
        },
    )

    names = names_of(events)
    assert names[-1] == "done"
    assert "token" in names
    assert names.index("sources") < names.index("token")
    assert events[-1][1]["answer"] == "the answer is here"

    # The tokens spell out the answer once, with nothing added or dropped.
    tokens = "".join(
        payload["text"] for name, payload in events if name == "token"
    )
    assert tokens == "the answer is here"


def test_the_query_that_was_searched_is_sent_even_when_it_is_not_the_question(
    db_path: Path, config: Settings
):
    """The query event carries the rewritten question the search used.

    That text is what the model returned, and it goes out before the sources.
    """
    file_document(db_path, MANUAL, category="manuals")
    rewritten = 'Who is Simone in the document "manuals/manual.pdf"?'
    harness = harness_for(config, replies=[rewritten, "the answer"])

    with harness.client() as client:
        events = events_of(ask(client, "who is Simone?"))

    assert [payload for name, payload in events if name == "query"] == [
        {"query": rewritten}
    ]

    assert names_of(events).index("query") < names_of(events).index("sources")


def test_the_rewriter_is_told_not_to_name_a_document_in_the_query(
    db_path: Path, config: Settings
):
    """The rewriting prompt tells the model to leave document names out."""
    file_document(db_path, MANUAL, category="manuals")
    harness = harness_for(config, replies=["a standalone question", "the answer"])

    with harness.client() as client:
        ask(client, "who is Simone?")

    rewriting_prompt = str(harness.model.prompts[0])
    assert "Never name a document in the query" in rewriting_prompt


def test_the_answer_is_told_which_documents_the_question_was_asked_of(
    db_path: Path, config: Settings
):
    """The prompt that writes the answer lists the documents searched."""
    file_document(db_path, MANUAL, category="manuals")
    file_document(db_path, REPORT, category="reports")
    harness = harness_for(config, replies=["a standalone question", "the answer"])

    with harness.client() as client:
        ask(client, "what documents do you have?")

    written_with = str(harness.model.prompts[1])
    assert f"Documents searched: 2 — {MANUAL}, {REPORT}" in written_with


def test_the_answer_is_told_what_the_documents_contain(
    db_path: Path, config: Settings
):
    """The prompt also carries the description written for each document.

    That is what a question about the library itself needs.
    """
    file_document(db_path, MANUAL, category="manuals")
    file_document(db_path, REPORT, category="reports")
    described(db_path, MANUAL, "A manual about the thing.")
    described(db_path, REPORT, "Last year's report.")
    harness = harness_for(config, replies=["a standalone question", "the answer"])

    with harness.client() as client:
        ask(client, "what does each of them contain?")

    written_with = str(harness.model.prompts[1])
    assert f"{MANUAL} — A manual about the thing." in written_with
    assert f"{REPORT} — Last year's report." in written_with


def test_a_scoped_answer_is_told_only_about_the_documents_it_searched(
    db_path: Path, config: Settings
):
    """A question narrowed to one document names only that one."""
    file_document(db_path, MANUAL, category="manuals")
    file_document(db_path, REPORT, category="reports")
    harness = harness_for(config, replies=["a standalone question", "the answer"])

    with harness.client() as client:
        ask(client, "what does it say?", document=MANUAL)

    written_with = str(harness.model.prompts[1])
    assert f"Documents searched: 1 — {MANUAL}" in written_with

    # The other document is not mentioned at all.
    assert REPORT not in written_with


def test_a_question_with_nothing_retrieved_streams_no_sources(
    db_path: Path, config: Settings
):
    """A search that found nothing sends no sources event.

    The stream still opens with the thread and closes with done.
    """
    file_document(db_path, MANUAL, category="manuals")
    harness = harness_for(
        config, replies=["a standalone question", "not covered"], documents=[]
    )

    with harness.client() as client:
        events = events_of(ask(client, "what does it say?"))

    # An empty list of passages stands for a search that found nothing.
    assert "sources" not in names_of(events)
    assert names_of(events)[0] == "thread"
    assert names_of(events)[-1] == "done"


def test_a_failure_in_the_answer_is_reported_inside_the_stream(
    db_path: Path, config: Settings
):
    """A model that runs out of replies fails while the answer streams.

    The failure arrives as an error event, once the thread event is out.
    """
    file_document(db_path, MANUAL, category="manuals")
    harness = harness_for(config, replies=["a standalone question"])
    harness.model.replies = []

    with harness.client() as client:
        events = events_of(ask(client, "what does it say?"))

    assert names_of(events)[0] == "thread"
    assert names_of(events)[-1] == "error"


# Grading. The route reads GRADE from the config the app was built with, so a
# server started with it on judges the material before writing an answer.
def test_a_question_the_documents_do_not_answer_is_turned_away(
    db_path: Path, config: Settings
):
    """The refusal is streamed like an answer, and it is what the page shows."""
    file_document(db_path, MANUAL, category="manuals")
    harness = harness_for(
        replace(config, grade="on"),
        replies=[
            "a standalone question",
            verdict_reply(False, "Nothing about the warranty."),
            "The documents do not cover that.",
        ],
    )

    with harness.client() as client:
        events = events_of(ask(client, "how long is the warranty?"))

    assert names_of(events)[-1] == "done"
    assert events[-1][1]["answer"] == "The documents do not cover that."


def test_a_question_is_searched_again_with_the_query_the_grader_wrote(
    db_path: Path, config: Settings
):
    """The retry reaches the search, and what it found is what is answered."""
    file_document(db_path, MANUAL, category="manuals")
    harness = harness_for(
        replace(config, grade="on"),
        replies=[
            "a standalone question",
            verdict_reply(False, "Nothing about it.", query="warranty period"),
            verdict_reply(True, "The second search found it."),
            "the answer",
        ],
    )

    with harness.client() as client:
        events = events_of(ask(client, "how long is the warranty?"))

    assert harness.retriever.queries == ["a standalone question", "warranty period"]
    assert events[-1][1]["answer"] == "the answer"


def test_a_server_built_with_grading_off_answers_every_question(
    db_path: Path, config: Settings
):
    """With GRADE off the route never asks for a verdict.

    Two model calls per question are what every other test in this file
    scripts for, so this pins which setting produces that.
    """
    file_document(db_path, MANUAL, category="manuals")
    harness = harness_for(config, replies=["a standalone question", "the answer"])

    with harness.client() as client:
        events = events_of(ask(client, "what does it say?"))

    assert events[-1][1]["answer"] == "the answer"
    assert len(harness.model.prompts) == 2


# The threads endpoint, which reads back the conversations and deletes them.
def test_the_thread_id_comes_back_and_the_next_question_joins_it(
    db_path: Path, config: Settings
):
    """The thread event carries an id, and the client sends it back.

    The second question joins the same conversation, so the thread holds
    both turns and the query of the last one.
    """
    file_document(db_path, MANUAL, category="manuals")
    harness = harness_for(
        config,
        replies=[
            "a standalone question",
            "the answer",
            "a second standalone question",
            "the second answer",
        ],
    )

    with harness.client() as client:
        first = events_of(ask(client, "what does it say?"))
        thread_id = first[0][1]["thread_id"]
        assert thread_id.startswith("chat-")

        second = events_of(ask(client, "and for minors?", thread_id=thread_id))
        assert second[0][1]["thread_id"] == thread_id

        thread = client.get(f"/api/threads/{thread_id}").json()

    assert thread["messages"] == [
        {"role": "human", "content": "what does it say?"},
        {"role": "ai", "content": "the answer"},
        {"role": "human", "content": "and for minors?"},
        {"role": "ai", "content": "the second answer"},
    ]

    # The thread keeps the query the second question was rewritten into.
    assert thread["query"] == "a second standalone question"

    # The prompt for the second question carries the first answer with it.
    assert "the answer" in str(harness.model.prompts[2])


def test_a_question_asked_without_a_thread_id_starts_a_new_conversation(
    db_path: Path, config: Settings
):
    """Two questions sent without a thread id get two different ids."""
    file_document(db_path, MANUAL, category="manuals")
    harness = harness_for(
        config,
        replies=[
            "a standalone question",
            "the answer",
            "another standalone question",
            "another answer",
        ],
    )

    with harness.client() as client:
        first = events_of(ask(client, "what does it say?"))[0][1]["thread_id"]
        second = events_of(ask(client, "what does it say?"))[0][1]["thread_id"]

    assert first != second


def test_a_conversation_survives_the_server_being_restarted(
    tmp_path: Path, db_path: Path, config: Settings
):
    """A second app over the same files reads the conversation the first had.

    The thread database lives in the test's tmp_path, so both apps open it.
    """
    file_document(db_path, MANUAL, category="manuals")
    assert Path(config.conversations_db_path).parent == tmp_path

    first = harness_for(config, replies=["a standalone question", "the answer"])
    with first.client() as client:
        thread_id = events_of(ask(client, "what does it say?"))[0][1]["thread_id"]

    # A new app over the same database stands in for a restart.
    second = harness_for(
        config, replies=["a second standalone question", "the second answer"]
    )
    with second.client() as client:
        thread = client.get(f"/api/threads/{thread_id}").json()

        assert [turn["content"] for turn in thread["messages"]] == [
            "what does it say?",
            "the answer",
        ]

        # The conversation carries on where it left off.
        events_of(ask(client, "and for minors?", thread_id=thread_id))
        thread = client.get(f"/api/threads/{thread_id}").json()

    assert len(thread["messages"]) == 4


def test_a_graded_turn_survives_the_thread_being_written_out(
    db_path: Path, config: Settings
):
    """The state of a graded turn is stored and read back.

    The checkpointer writes the state out, so a verdict has to be something it
    can write. Reading the thread back is what proves it was.
    """
    file_document(db_path, MANUAL, category="manuals")

    first = harness_for(
        replace(config, grade="on"),
        replies=[
            "a standalone question",
            verdict_reply(False, "Nothing about the warranty."),
            "The documents do not cover that.",
        ],
    )
    with first.client() as client:
        thread_id = events_of(ask(client, "how long is the warranty?"))[0][1][
            "thread_id"
        ]

    # A new app over the same database stands in for a restart.
    with harness_for(config, replies=[]).client() as client:
        thread = client.get(f"/api/threads/{thread_id}").json()

    assert [turn["content"] for turn in thread["messages"]] == [
        "how long is the warranty?",
        "The documents do not cover that.",
    ]
    assert thread["query"] == "a standalone question"


def test_a_conversation_nobody_has_had_yet_is_empty_rather_than_an_error(
    config: Settings,
):
    """An id nobody has used reads as an empty conversation, not as a 404."""
    with harness_for(config, replies=[]).client() as client:
        thread = client.get("/api/threads/chat-nobody").json()

    assert thread == {
        "thread_id": "chat-nobody",
        "messages": [],
        "query": None,
        "sources": [],
    }


def test_deleting_a_conversation_empties_it(db_path: Path, config: Settings):
    """Deleting a thread answers with its id and leaves it empty."""
    file_document(db_path, MANUAL, category="manuals")
    harness = harness_for(config, replies=["a standalone question", "the answer"])

    with harness.client() as client:
        thread_id = events_of(ask(client, "what does it say?"))[0][1]["thread_id"]
        assert client.get(f"/api/threads/{thread_id}").json()["messages"]

        deleted = client.delete(f"/api/threads/{thread_id}")

        assert deleted.status_code == 200
        assert deleted.json() == {"thread_id": thread_id}
        after = client.get(f"/api/threads/{thread_id}").json()

    assert after["messages"] == []


def test_deleting_a_conversation_nobody_has_had_is_not_an_error(config: Settings):
    """Deleting a thread that was never used answers with status 200."""
    with harness_for(config, replies=[]).client() as client:
        deleted = client.delete("/api/threads/chat-nobody")

    assert deleted.status_code == 200


def test_deleting_one_conversation_leaves_the_others_alone(
    db_path: Path, config: Settings
):
    """Deleting one thread leaves the other conversation as it was."""
    file_document(db_path, MANUAL, category="manuals")
    file_document(db_path, REPORT, category="reports")
    harness = harness_for(
        config,
        replies=[
            "a standalone question",
            "the answer",
            "another standalone question",
            "another answer",
        ],
    )

    with harness.client() as client:
        first = events_of(ask(client, "what does it say?", document=MANUAL))[0][1]
        second = events_of(ask(client, "and here?", document=REPORT))[0][1]

        client.delete(f"/api/threads/{first['thread_id']}")
        kept = client.get(f"/api/threads/{second['thread_id']}").json()

    assert [turn["content"] for turn in kept["messages"]] == [
        "and here?",
        "another answer",
    ]


def test_a_deleted_conversation_does_not_come_back_when_the_server_restarts(
    db_path: Path, config: Settings
):
    """A deleted thread is still empty when a new app opens the database."""
    file_document(db_path, MANUAL, category="manuals")

    first = harness_for(config, replies=["a standalone question", "the answer"])
    with first.client() as client:
        thread_id = events_of(ask(client, "what does it say?"))[0][1]["thread_id"]
        client.delete(f"/api/threads/{thread_id}")

    # A new app over the same database stands in for a restart.
    second = harness_for(config, replies=[])
    with second.client() as client:
        thread = client.get(f"/api/threads/{thread_id}").json()

    assert thread["messages"] == []


# The page the console loads, together with its script and stylesheet.
def test_the_page_and_its_files_are_served(config: Settings):
    """The root path serves the page, with its files under /static."""
    with harness_for(config, replies=[]).client() as client:
        page = client.get("/")
        script = client.get("/static/app.js")
        stylesheet = client.get("/static/style.css")

    assert page.status_code == 200
    assert 'id="composer"' in page.text
    assert script.status_code == 200
    assert stylesheet.status_code == 200
