"""The web application, over the same fakes the graph tests use.

Nothing here reaches an API or a vector store: the catalog is a real SQLite file
in `tmp_path` — it is a local file and costs nothing — and the model and the
retriever are the doubles from `tests.helpers`. The one test that uses the real
persistent checkpointer uses it over a file in `tmp_path` too, which is what
makes it a test of surviving a restart rather than of surviving a function call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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
)

MANUAL = "manuals/manual.pdf"
ANNEX = "manuals/annex.pdf"
REPORT = "reports/report.pdf"
LOOSE = "notes.pdf"

# The opening of what the resolver says about a category whose documents are all
# failed or trashed. Spelled out here rather than read off the message, so that a
# change in the wording is a change in two places and not quietly in one. What
# follows the full stop is a hint listing the categories there are.
NOWHERE = "No indexed documents in the category 'empty'."


@dataclass
class Harness:
    """The application over fakes, with the fakes kept where a test can see them."""

    app: FastAPI
    model: FakeChatModel
    retriever: FakeRetriever
    # Every scope the application asked for a retriever with, in order.
    scopes: list[Scope] = field(default_factory=list)

    def client(self) -> TestClient:
        return TestClient(self.app)


@pytest.fixture
def config(tmp_path: Path, db_path: Path) -> Settings:
    """The settings of a machine whose library is this test's, and nothing else's."""
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
    """The application, answering from `replies` and searching `documents`."""
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
    """One retrieved chunk, with the metadata a real one carries.

    `page` is the zero-based one a store holds, which the graph reports as the
    number a reader would turn to. Two chunks differing only in `chunk` are two
    passages of one page, and `start` and `end` are where each of them sits in
    that page's text — the offsets a citation is drawn with.
    """
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
    """Put a document in the catalog, in the state a test is about."""
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
    """What `scripts.describe` leaves behind: the catalog saying what it is about."""
    Catalog(db_path).set_description(path, text)


def events_of(response: Any) -> list[tuple[str, dict[str, Any]]]:
    """A stream as the page reads it: the name and the payload of each event."""
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
    return [name for name, _ in events]


def ask(client: TestClient, question: str, **scope: Any) -> Any:
    return client.post("/api/chat", json={"question": question, **scope})


# ---------------------------------------------------------------- the catalog


def test_documents_are_grouped_under_the_category_they_are_filed_in(
    db_path: Path, config: Settings
):
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
    # Phase 5 fills this in; until then the field is here and empty, which is what
    # a page renders as "no description" rather than as a broken row.
    assert manual["description"] is None
    assert view["categories"][1]["documents"][0]["title"] == "The report"


def test_a_category_holding_no_document_of_its_own_still_appears(
    db_path: Path, config: Settings
):
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
    file_document(db_path, MANUAL, category="manuals")
    file_document(db_path, LOOSE, title="A note")

    with harness_for(config, replies=[]).client() as client:
        view = client.get("/api/catalog").json()

    assert [branch["name"] for branch in view["categories"]] == ["manuals"]
    assert [row["title"] for row in view["uncategorized"]] == ["A note"]


def test_a_trashed_document_is_left_out(db_path: Path, config: Settings):
    file_document(db_path, MANUAL, category="manuals")
    file_document(db_path, ANNEX, title="Gone", status="trashed")

    with harness_for(config, replies=[]).client() as client:
        view = client.get("/api/catalog").json()

    assert [row["path"] for row in view["categories"][0]["documents"]] == [MANUAL]


def test_a_document_that_failed_to_index_is_still_shown(
    db_path: Path, config: Settings
):
    """It has no vectors, so no question reaches it — which is worth saying."""
    file_document(db_path, MANUAL, category="manuals", status="failed")

    with harness_for(config, replies=[]).client() as client:
        view = client.get("/api/catalog").json()

    row = view["categories"][0]["documents"][0]
    assert row["status"] == "failed"


def test_no_catalog_at_all_is_an_empty_view_and_writes_nothing(
    db_path: Path, config: Settings
):
    """Looking at the library must not bring a database into existence."""
    with harness_for(config, replies=[]).client() as client:
        view = client.get("/api/catalog").json()

    assert view == {"empty": True, "categories": [], "uncategorized": []}
    assert not db_path.exists()


# ------------------------------------------------------------------- the scope


def test_the_scope_label_is_the_one_the_console_prints(
    db_path: Path, config: Settings
):
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
    file_document(db_path, MANUAL, category="manuals")

    with harness_for(config, replies=[]).client() as client:
        resolved = client.get("/api/scope").json()

    assert resolved["label"] == "whole library"
    assert resolved["sources"] is None
    assert resolved["whole_document"] is False


def test_a_category_with_nothing_indexed_is_refused_in_the_resolvers_words(
    db_path: Path, config: Settings
):
    file_document(db_path, MANUAL, category="empty", status="failed")

    with harness_for(config, replies=[]).client() as client:
        response = client.get("/api/scope", params={"category": "empty"})

    assert response.status_code == 400
    assert response.json()["detail"].startswith(NOWHERE)


def test_a_question_about_nothing_is_refused_before_any_stream_begins(
    db_path: Path, config: Settings
):
    file_document(db_path, MANUAL, category="manuals")

    with harness_for(config, replies=[]).client() as client:
        response = ask(client, "what does it say?", category="empty")

    assert response.status_code == 400
    assert response.json()["detail"].startswith(NOWHERE)
    assert response.headers["content-type"] != "text/event-stream"


def test_the_retriever_is_built_from_the_scope_that_was_asked_for(
    db_path: Path, config: Settings
):
    file_document(db_path, MANUAL, category="manuals")

    harness = harness_for(config, replies=["a standalone question", "the answer"])
    with harness.client() as client:
        response = ask(client, "what does it say?", document=MANUAL)

    assert response.status_code == 200
    assert [scope.sources for scope in harness.scopes] == [(MANUAL,)]
    # Four chunks is under the whole-document budget, so the scope reads the
    # document rather than searching it — decided here, not in the page.
    assert harness.scopes[0].whole_document is True


# ------------------------------------------------------------------ the boxes

# A page of two lines, so that a range covering both of them is a range with two
# rectangles to answer with — which one flat run of text could never be.
FIRST_LINE = "The first line of the page."
SECOND_LINE = "The second line, below it."
TWO_LINES = "manuals/two-lines.pdf"


def a_page_of_two_lines(config: Settings, name: str = TWO_LINES) -> str:
    """Write a two-line document into the library, and return its page text.

    Real typography rather than `make_pdf`, which lays a page down as one run of
    text: two lines that can be told apart by where they are need a page with two
    of them on it. The text comes back because that is what the offsets are into
    — the same text the reader builds at ingest, which is the whole contract the
    two halves of this share.
    """
    path = make_structured_pdf(
        Path(config.documents_dir) / name,
        [[Line(FIRST_LINE), Line(SECOND_LINE)]],
    )

    return pdf.read_pages(path)[0].text


def boxes_of(client: TestClient, **params: Any) -> Any:
    return client.get(
        "/api/documents/boxes",
        params={"source": TWO_LINES, "page": 1, "start": 0, "end": 1, **params},
    )


def test_a_range_comes_back_as_the_boxes_that_cover_it(config: Settings):
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
    # A line and not a page: written from the left margin, ending before the
    # right one, and as tall as a line of text is.
    assert x0 == pytest.approx(72, abs=1)
    assert x1 < 612
    assert 0 < y1 - y0 < 20


def test_a_range_across_two_lines_is_two_boxes(config: Settings):
    """One box around the lot would be the width of the page and say nothing."""
    text = a_page_of_two_lines(config)
    start = text.index(FIRST_LINE)
    end = text.index(SECOND_LINE) + len(SECOND_LINE)

    with harness_for(config, replies=[]).client() as client:
        boxes = boxes_of(client, start=start, end=end).json()["boxes"]

    assert len(boxes) == 2
    # The one above is above the other, which is what tells two lines from two
    # halves of one.
    assert boxes[0][1] < boxes[1][1]


def test_a_range_that_covers_nothing_is_an_empty_answer(config: Settings):
    """`[]` rather than a refusal: the range is a real one that holds no text.

    The character between the two lines is a newline, which nothing was drawn on.
    """
    text = a_page_of_two_lines(config)
    gap = text.index("\n")

    with harness_for(config, replies=[]).client() as client:
        answer = boxes_of(client, start=gap, end=gap + 1)

    assert answer.status_code == 200
    assert answer.json()["boxes"] == []


def test_a_page_needs_no_catalog(db_path: Path, config: Settings):
    """The folder says a document is here, and the catalog has no part in it.

    A file that has never been synced still has pages, and a viewer asked about
    one of them can be answered without a database being opened — or, as here,
    without one existing at all.
    """
    assert not db_path.exists()
    a_page_of_two_lines(config)

    with harness_for(config, replies=[]).client() as client:
        answer = boxes_of(client, start=0, end=len(FIRST_LINE))

    assert answer.status_code == 200


def test_a_source_outside_the_documents_folder_is_refused(
    tmp_path: Path, config: Settings
):
    """The file is there and is a PDF, and it is not in this library.

    Written outside the documents folder on purpose: without the containment
    check the path resolves to a file that opens, and the answer would be the
    rectangles of a document this server was never pointed at.
    """
    make_pdf(tmp_path / "outside.pdf", "Not in the library.")

    with harness_for(config, replies=[]).client() as client:
        answer = boxes_of(client, source="../outside.pdf", start=0, end=4)

    assert answer.status_code == 400
    assert answer.json()["detail"] == "Not a document here: ../outside.pdf"


def test_a_file_that_is_not_a_pdf_is_refused(config: Settings):
    """Readable, and not a document this library reads: the sync ignores it too."""
    make_pdf(Path(config.documents_dir) / "manuals" / "notes.txt", "Not a PDF here.")

    with harness_for(config, replies=[]).client() as client:
        answer = boxes_of(client, source="manuals/notes.txt", start=0, end=4)

    assert answer.status_code == 400
    assert answer.json()["detail"] == "Not a document here: manuals/notes.txt"


def test_a_document_that_is_not_there_is_refused(config: Settings):
    with harness_for(config, replies=[]).client() as client:
        answer = boxes_of(client, source="manuals/absent.pdf", start=0, end=4)

    assert answer.status_code == 400
    assert answer.json()["detail"] == "Not a document here: manuals/absent.pdf"


@pytest.mark.parametrize("page", [0, 2, 99])
def test_a_page_the_document_does_not_have_is_refused(config: Settings, page: int):
    """Zero is the one that matters: it is a valid index, and the last page.

    Read as an index rather than checked as a number, a request about page zero
    would be answered with the rectangles of the document's last page, and
    nothing in the answer would say the question had been about another one.
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
    a_page_of_two_lines(config)

    with harness_for(config, replies=[]).client() as client:
        answer = boxes_of(client, start=start, end=end)

    assert answer.status_code == 400
    assert answer.json()["detail"] == f"Not a range: {start}-{end}"


# --------------------------------------------------------------- the answer


def test_the_sources_arrive_before_the_answer_is_written(
    db_path: Path, config: Settings
):
    """The fifth of the roadmap's asks: sources as soon as retrieval is done."""
    file_document(db_path, MANUAL, category="manuals")
    harness = harness_for(
        config,
        replies=["a standalone question", "the answer is here"],
        # Two passages of one page and one of another: the page is the source, so
        # the answer cites two places and not three — and the two passages of the
        # first are two ranges on the one row, rather than one being dropped.
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

    # The tokens are the answer, piece by piece, in order.
    tokens = "".join(
        payload["text"] for name, payload in events if name == "token"
    )
    assert tokens == "the answer is here"


def test_the_query_that_was_searched_is_sent_even_when_it_is_not_the_question(
    db_path: Path, config: Settings
):
    """The rewrite can put words in the query the question never had.

    It happens with a document name: the conversation so far is about one file,
    the rewriter qualifies the query with it, and a search over the whole library
    comes back with that one file. Nothing about the question says so, which is
    why the query is sent to the page instead of staying in the server.
    """
    file_document(db_path, MANUAL, category="manuals")
    rewritten = 'Who is Simone in the document "manuals/manual.pdf"?'
    harness = harness_for(config, replies=[rewritten, "the answer"])

    with harness.client() as client:
        events = events_of(ask(client, "who is Simone?"))

    assert [payload for name, payload in events if name == "query"] == [
        {"query": rewritten}
    ]
    # Before the passages, because it is what produced them.
    assert names_of(events).index("query") < names_of(events).index("sources")


def test_the_rewriter_is_told_not_to_name_a_document_in_the_query(
    db_path: Path, config: Settings
):
    """The instruction that keeps a scope from being narrowed behind the asker.

    This checks the prompt carries it, which is as far as an offline test reaches:
    whether the model obeys is the model's business and is checked by asking it.
    """
    file_document(db_path, MANUAL, category="manuals")
    harness = harness_for(config, replies=["a standalone question", "the answer"])

    with harness.client() as client:
        ask(client, "who is Simone?")

    rewriting_prompt = str(harness.model.prompts[0])
    assert "Never name a document in the query" in rewriting_prompt


def test_the_answer_is_told_which_documents_the_question_was_asked_of(
    db_path: Path, config: Settings
):
    """The blind spot of a search: a question about the library.

    "What documents do you have?" matches no passage, so the whole-library search
    returns the chunks closest to it and nothing about the library at all. The
    scope's own list is handed to the graph beside the retriever, which is what
    the answer is written from instead.
    """
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
    """The other half of the same blind spot: a question about the contents.

    "What does each of them contain?" is answered by a search with the passages
    of whichever documents happen to match it, so a document it did not return is
    one the answer has nothing to say about. What the catalog says about each of
    them goes into the context beside the list, and the answer is written from it.
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
    file_document(db_path, MANUAL, category="manuals")
    file_document(db_path, REPORT, category="reports")
    harness = harness_for(config, replies=["a standalone question", "the answer"])

    with harness.client() as client:
        ask(client, "what does it say?", document=MANUAL)

    written_with = str(harness.model.prompts[1])
    assert f"Documents searched: 1 — {MANUAL}" in written_with
    # Nothing of the library it was not asked about, or the scope would be a
    # narrower search reported as a wider one.
    assert REPORT not in written_with


def test_a_question_with_nothing_retrieved_streams_no_sources(
    db_path: Path, config: Settings
):
    file_document(db_path, MANUAL, category="manuals")
    harness = harness_for(
        config, replies=["a standalone question", "not covered"], documents=[]
    )

    with harness.client() as client:
        events = events_of(ask(client, "what does it say?"))

    # No sources event at all, rather than one carrying an empty list: there is
    # nothing to show a reader, and the page shows nothing.
    assert "sources" not in names_of(events)
    assert names_of(events)[0] == "thread"
    assert names_of(events)[-1] == "done"


def test_a_failure_in_the_answer_is_reported_inside_the_stream(
    db_path: Path, config: Settings
):
    """The response has already begun, so an error is one more event."""
    file_document(db_path, MANUAL, category="manuals")
    harness = harness_for(config, replies=["a standalone question"])
    harness.model.replies = []  # the second call has nothing to reply from

    with harness.client() as client:
        events = events_of(ask(client, "what does it say?"))

    assert names_of(events)[0] == "thread"
    assert names_of(events)[-1] == "error"


# ----------------------------------------------------------- the conversation


def test_the_thread_id_comes_back_and_the_next_question_joins_it(
    db_path: Path, config: Settings
):
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
    # The last turn's query, so that a reloaded page says what was searched for
    # as well as what was said.
    assert thread["query"] == "a second standalone question"
    # The follow-up was rewritten knowing what came before it, which is the
    # conversation being one conversation rather than two questions in a row.
    assert "the answer" in str(harness.model.prompts[2])


def test_a_question_asked_without_a_thread_id_starts_a_new_conversation(
    db_path: Path, config: Settings
):
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
    """The point of the persistent checkpointer, over a real file in `tmp_path`."""
    file_document(db_path, MANUAL, category="manuals")
    assert Path(config.conversations_db_path).parent == tmp_path

    first = harness_for(config, replies=["a standalone question", "the answer"])
    with first.client() as client:
        thread_id = events_of(ask(client, "what does it say?"))[0][1]["thread_id"]

    # A second server, over the same file: nothing of the first one is in memory.
    second = harness_for(
        config, replies=["a second standalone question", "the second answer"]
    )
    with second.client() as client:
        thread = client.get(f"/api/threads/{thread_id}").json()

        assert [turn["content"] for turn in thread["messages"]] == [
            "what does it say?",
            "the answer",
        ]

        # And it can be carried on, which is what a conversation is for.
        events_of(ask(client, "and for minors?", thread_id=thread_id))
        thread = client.get(f"/api/threads/{thread_id}").json()

    assert len(thread["messages"]) == 4


def test_a_conversation_nobody_has_had_yet_is_empty_rather_than_an_error(
    config: Settings,
):
    with harness_for(config, replies=[]).client() as client:
        thread = client.get("/api/threads/chat-nobody").json()

    assert thread == {
        "thread_id": "chat-nobody",
        "messages": [],
        "query": None,
        "sources": [],
    }


def test_deleting_a_conversation_empties_it(db_path: Path, config: Settings):
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
    """The same reading `GET` takes: there is nothing there, and that is fine."""
    with harness_for(config, replies=[]).client() as client:
        deleted = client.delete("/api/threads/chat-nobody")

    assert deleted.status_code == 200


def test_deleting_one_conversation_leaves_the_others_alone(
    db_path: Path, config: Settings
):
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
    """The point of deleting rather than forgetting: the file no longer holds it."""
    file_document(db_path, MANUAL, category="manuals")

    first = harness_for(config, replies=["a standalone question", "the answer"])
    with first.client() as client:
        thread_id = events_of(ask(client, "what does it say?"))[0][1]["thread_id"]
        client.delete(f"/api/threads/{thread_id}")

    # A second server over the same file, which is where the conversation lived.
    second = harness_for(config, replies=[])
    with second.client() as client:
        thread = client.get(f"/api/threads/{thread_id}").json()

    assert thread["messages"] == []


# ------------------------------------------------------------------ the page


def test_the_page_and_its_files_are_served(config: Settings):
    with harness_for(config, replies=[]).client() as client:
        page = client.get("/")
        script = client.get("/static/app.js")
        stylesheet = client.get("/static/style.css")

    assert page.status_code == 200
    assert 'id="composer"' in page.text
    assert script.status_code == 200
    assert stylesheet.status_code == 200
