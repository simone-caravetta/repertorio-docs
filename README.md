# Repertorio Docs

<img width="2172" height="724" alt="ChatGPT Image Sep 10, 2026, 08_48_25 PM" src="https://github.com/user-attachments/assets/f86fb6f5-59cf-4404-852c-8e7dc9993b1b" />


Turn a folder of PDFs into a knowledge base you can question in plain language.

Point it at a folder, run one command, and ask your questions from the console. Every answer
comes from your documents, with the file and the page it came from; follow-up questions
work, and when the documents don't cover something it says so instead of inventing an
answer.

## Getting started

You need Python 3.11 or later. DeepSeek writes the answers and Pinecone holds the passages,
so that is two API keys — or one, if you keep the vectors in a folder on your own machine
instead (see [Vector store](#vector-store)). The embeddings run on your machine either way,
so nothing is sent away to be indexed.

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with the two keys. It is excluded from version
control:

```text
OPENAI_API_KEY=...      # the key of the chat model, DeepSeek unless you change it
PINECONE_API_KEY=...
```

Put your PDFs in `data/documents/`, then:

```bash
python -m scripts.sync      # index what is new or has changed, and describe it
python -m scripts.chat      # ask questions about it
```

```text
Repertorio Docs console
Type 'exit' to quit.

You: What are the requirements to access the service?
Assistant: According to the documents ...

Sources:
  - manuals/manual.pdf - p. 12

You: And for minors?
Assistant: ...
```

The PDFs need a selectable text layer: a scan without one is not indexed. There is a web
interface as well — see [Web interface](#web-interface).

## Documents

The sync keeps the folder and the index in step. Take a file out of the folder and the
document goes to the trash — out of the answers, still there to bring back by putting the
file where it was.

Deleting on purpose is a command of its own:

```bash
python -m scripts.delete manuals/manual.pdf    # one or more documents
python -m scripts.delete --trashed             # empty the trash
```

It removes the document from the index and from the catalog, so there is nothing left to
bring back. The file stays where it is unless you add `--with-file`: without it, the next
sync finds the file again and indexes it from scratch.

## Categories and scoped search

Documents are filed by the folder they sit in: a PDF in `data/documents/manuali/` belongs to
the category `manuali`, and one at the root of the folder belongs to none. The folder decides
this once, when the document is first indexed; after that the catalog is the authority, so a
document filed somewhere else stays there.

```bash
python -m scripts.catalog                          # the categories, with their counts
python -m scripts.catalog --documents              # ... and the documents under each
python -m scripts.catalog move manuals/a.pdf --to archive
```

Moving a document is a catalog write: nothing is indexed again, and the file does not have to
move. The other side of that is that reorganising the folders on disk does not recategorise
what has already been indexed — a file moved to another folder is a document the catalog does
not know, and the next sync indexes it as a new one.

A question can be asked of a category, of one document, or of a few:

```bash
python -m scripts.chat --category manuals          # this category and what is filed below it
python -m scripts.chat --document manuals/a.pdf    # one document
python -m scripts.chat --documents manuals/a.pdf reports/b.pdf
```

The console prints the scope it is on before the first question. A single document small
enough to fit the model's context is read whole rather than by similarity, in reading order,
so nothing in it is left out by ranking; `WHOLE_DOCUMENT_MAX_CHARS` sets the budget (24000 by
default, 0 to always search by similarity).

## Descriptions

Which documents there are, and what each one contains, is not something a search can answer:
a search over the whole library returns the passages closest to the question, and a small
document beside a large one is never among them. So each document is described once, and the
description goes into the context of every answer, under the list of documents the question
was asked of.

The sync writes one for every document it indexes that has none, so a library it has just
been through is described in full, and indexing a folder costs one model call per document.
`python -m scripts.sync --no-descriptions` indexes without writing any, for a folder of a
hundred files or a machine with no key; the vectors are the same either way, and the missing
descriptions can be written later without re-indexing anything.

```bash
python -m scripts.describe                # the documents that have no description yet
python -m scripts.describe manuals/a.pdf  # this one, described again
python -m scripts.describe --all          # every indexed document, written again
python -m scripts.describe --dry-run      # what a run would do, without calling a model
```

One model call per document, over a sample taken from across the document rather than from
its first pages, and the description is written in the language the document is written in.
It is kept in the catalog, so it is written once and read every time: `python -m
scripts.catalog --documents` prints it under the document, the page shows it under the
title, and it is part of what the answer is written from. A document that has a description
is left alone, so a run over a library that has not changed costs nothing, and one whose
description could not be written is tried again on the next run. A file that has changed
since it was described is written about again: the description is written from the file, so
an edition that is no longer there takes its description with it, and the title of a document
is never left standing over a description of another edition.

`DESCRIPTION_SAMPLE_CHARS` is how much of a document is read to write its description (6000
by default); `DESCRIPTION_BUDGET_CHARS` is how much of the catalog goes into one answer's
context (2000). The documents are named in that context either way, so a scope too large to
describe loses the descriptions and nothing else; 0 turns either budget off.

## Web interface

The same library in a browser, with the same scopes:

```bash
python -m scripts.serve     # http://127.0.0.1:8000
```

The catalog is on the left, grouped by category, each document with its title, its
description and its status; clicking a document asks about that document. A category opens
and closes on a click, and so does a document's description — from the triangle beside it,
because the rest of the row is what asks the question — and the panel opens the way it was
left. The conversation is on the right, and the sources of the answer appear as soon as the
search is done, while the answer is still being written.

The title of the conversation is also the control that changes it: it names the document or
the group being asked about, and clicking it drops the list it was picked from — every
conversation there is, and the documents, the categories and the whole library with them. So
moving between conversations is picking one from a list. Under the title is the scope in the
words the console prints it in, which is what the next question will be asked of, and beside
that **New conversation** — and **Delete**, when the conversation is one the page made and
can take back.

Several documents can be asked about together, the way `--documents` does it from the
console. **Select several**, beside `Documents`, puts a box at the left of every row: the box
on a document ticks that document, and the box on a category ticks every indexed document at
or below it — the same set the server resolves that category to when it is picked from the
selector. A document that is not indexed has no box, because one of those in a group makes
the whole group refuse. What is ticked is counted at the foot of the panel, above a field
proposing a title for the group; keep it or write another, and **Start** begins the
conversation on those documents under that name. The mode ends there, and the ticks with it,
so extending a group means turning it on again with the group's own documents already ticked.

A group keeps the title it was given, which is what tells two groups of the same size apart —
the server calls both of them `2 documents`. The titles are the page's own, in the browser's
storage beside the scope and the conversations, and a group that has been named stays in the
list even with no conversation on it. A document and a category are named by what they are.
The title is the reader's, so the field takes one off as readily as it puts one on, and
**Delete** takes it off along with the conversation. **New conversation** does not: a title
belongs to the set of documents, not to the conversation about them.

Under each answer there is a line saying what was searched for. It is not always the question
as it was typed: the question is first rewritten together with the conversation so far, so
that "and for minors?" becomes a question that stands on its own. The line is what that
rewrite produced, and it is worth reading when an answer seems to have come from the wrong
document.

The answer is told which documents were searched, and what the catalog says about each of
them, and not only which passages were found — see [Descriptions](#descriptions). That is
what makes a question about the library itself answerable.

Conversations are kept in `data/conversations.sqlite3`, so a conversation is still there
after the server is restarted, unless it has been deleted. Each scope has its own: ask about
a document, move to another one and come back, and the questions you asked about the first
are still there. It is one conversation per scope rather than one running conversation,
because every question is rewritten together with the conversation it is asked in, and the
history of a question about one document is not context for a question about another.
**New conversation**, beside the scope, starts one over — and only that: the title the group
was given stays on it. **Delete** is the other one, and it asks first: it takes the
conversation away for good, from the page and from the file. It is offered on a document's
and on a group's conversation — the two entries the conversation itself put in the list. A
category's and the whole library's come from the catalog and would be back the moment it was
read again, so **New conversation** is what clears one of those. A group that was named but
never asked anything of has nothing in the file yet, so deleting it takes the name away and
nothing else.

The server listens on `127.0.0.1` only. `--host 0.0.0.0` opens it to the network, and there is
no authentication behind it — anyone who can reach the port can read the whole library and
every conversation in it. `--port`, `--db` and `--conversations` are there too.

## Models

The defaults are DeepSeek for the answers and `BAAI/bge-m3` on your machine for the
embeddings. Both can be pointed elsewhere in `.env`.

```text
# The chat model: any endpoint that speaks the OpenAI chat completions API,
# hosted or on your own machine — the same three settings either way
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-v4-flash
OPENAI_API_KEY=...
CHAT_PROVIDER=deepseek      # deepseek, openai, or local for a server of your own

# The embedding model: on your machine, or through the OpenAI API
EMBEDDING_PROVIDER=local    # local or openai
EMBEDDING_MODEL=...         # BAAI/bge-m3 locally, text-embedding-3-small on openai
EMBEDDING_API_KEY=...       # the OpenAI embeddings, when OPENAI_API_KEY is not one
```

`CHAT_PROVIDER` fills in the three settings above when they are left out, and adds the
parameters that provider needs: `deepseek` turns its reasoning off, which keeps it out of
the console and out of the token count. A model on your own machine is `local`, and it
ignores the key.

The chat model can be anything that speaks the OpenAI chat completions API. The embedding
model cannot be swapped as freely: the index is built in the vector space of one model, so
another one needs the documents indexed again. The catalog records which model indexed each
document, so a sync notices the change and rebuilds them; and an index whose vectors are of
another length refuses first, because two models are not comparable even at the same length.

## Vector store

The passages live in Pinecone by default. `VECTOR_STORE=chroma` keeps them in a folder on
your machine instead: no account, no key, no network, and the whole library is
`data/documents/` and `data/chroma/` together.

```text
VECTOR_STORE=chroma          # pinecone (the default) or chroma
CHROMA_DIR=data/chroma       # where the local store keeps its files
CHROMA_COLLECTION=documents
```

One line switches between them, and no command takes an argument for it: `python -m
scripts.sync`, `python -m scripts.chat` and `python -m scripts.delete` all read the same
`.env`, so they cannot end up on different stores. Each one prints the store it is using
before it starts.

The two hold different vectors, so each keeps its own catalog, and the catalog follows
`VECTOR_STORE` on its own: `data/catalog.sqlite3` for Pinecone, `data/catalog-chroma.sqlite3`
for Chroma. Set `CATALOG_DB_PATH` only to put one somewhere else — and if a catalog ever ends
up describing a store that does not hold its vectors, the sync says so rather than reporting
a run with nothing to do.

One limit worth knowing: the sync and the delete command take a lock, the console and the web
server do not. With a hosted index that does not matter, but with a local store any two of
them would be writing the same folder at the same time.

## Measuring retrieval

Changing how documents are searched needs a number to move. A question set is a JSON file
naming, for each question, the document the answer should come from and — when it is worth
writing down — the page and what a correct answer says:

```json
[
  {
    "id": "manuals-warranty",
    "question": "How long does the warranty last?",
    "document": "manuals/a.pdf",
    "page": 4,
    "contains": ["24 months"]
  }
]
```

```bash
python -m scripts.eval                            # search only: one embedding per question
python -m scripts.eval --document manuals/a.pdf   # the same set, asked of one document
python -m scripts.eval --answers                  # write an answer to each question
python -m scripts.eval --judge                    # ... and check each one against its sources
```

The search is measured from the question as typed, through the same retriever the console uses,
and reports four numbers. How often the expected document came back, and how often the expected
page did, are counts: a passage found first and the same passage found fifth both count once. The
mean reciprocal rank is where the document was found, and nDCG@10 is where everything that came
back was ordered, so that a change which moves a passage up the list shows up even when nothing
was gained or lost. A ranking is read against the best order the passages it returned could have
been in, which is not the ideal a benchmark would use — a question here names one document and at
most one page, and a set cannot know how many other passages of it exist. It costs no model call.

The first three numbers are read over the first `RETRIEVAL_K` passages, which is what the console
would have answered from. nDCG@10 is read over the first ten, so a run asks the search for ten
and reports both readings of that one search: a document at position eight was returned to the
harness and never to a reader, and counting it as found would report a search the console does
not have. A scope that is one small document is read whole and reports no nDCG — there is no
ranking to read.

`--answers` writes an answer from the passages that search returned, with the console's own
prompt, so a question the search missed cannot be rescued by a good answer — it was one search,
and what it found is what the answer was written from. `--judge` adds a second call per question,
to a model that reads the answer against its context and says whether every claim in it is
supported, which is the faithfulness number.

The set is a file of your own questions about your own documents: the default path is
`data/evals/questions.json`, kept out of the repository for the same reason the documents are.

## Reranking

An embedding is computed before the question is known, so the best a similarity search can say
is that two texts sit near each other. A cross-encoder reads the question and the passage
together and scores that pair, which is the thing an embedding cannot do. It is one forward
pass per passage, so it is affordable over the twenty a search hands over and not over a
library.

```text
RERANK=off                   # off (the default) or on
RERANK_MODEL=                # empty means BAAI/bge-reranker-v2-m3
RERANK_CANDIDATES=20         # what the search is asked for, of which RETRIEVAL_K come back
RERANK_DEVICE=               # empty follows EMBEDDING_DEVICE
```

The search is asked for `RERANK_CANDIDATES` passages, all of them are scored, and the best
`RETRIEVAL_K` are what the answer is written from. `python -m scripts.eval` and `python -m
scripts.chat` both print a `rerank` line saying which model and how many of how many, or `off`,
because a reranked run and a plain one are handed a different question by the store and their
reports are not comparable unless the line says which pass produced them. A scope that is one
small document is read whole and never reranked — the reason it is read whole is to reach
passages no ranking would surface.

It is off by default because on the library this project was built against it bought nothing.
That library is one book about one subject, and the search already put the right page first for
every question asked, so there was nothing to reorder: the eval reported 11/11 documents, 11/11
pages and MRR 1.00 with `RERANK=on` and with `RERANK=off` alike. A search that is never wrong
cannot show an improvement, so a second corpus was built to measure against — ten subjects with
four near-identical documents each, differing in one condition the document states, and
questions that describe that condition in everyday words. On it the reranker reorders and it
helps: MRR 0.78 → 0.89 and nDCG@10 0.77 → 0.85, with 17 of the 50 questions landing at a
different rank. Five runs of the plain pass and three of the reranked one each came out
identical line for line, so the corpus moves for a reason and not from ties. It is not a free
win, and the two directions tell the story: two documents the
plain search had found dropped out of the top five while three it had missed came in, so the
hit rate barely moves — 47/50 against 48/50 — while the ranking improves. A reordering cannot
add what was never retrieved. What it costs is not nothing: about seven seconds per question on a
CPU, with six cores busy, and a two-gigabyte checkpoint on a clone that has never reranked.
Turn it on when a question set shows the search losing an answer it should have found — that is
the case it is for.

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest -q
ruff check .
```

The test suite is offline: no API keys, no network, no model download. The same two
commands run on every push in GitHub Actions.

## Roadmap

The project is growing into a personal document library. The full plan lives in
[ROADMAP.md](ROADMAP.md).
