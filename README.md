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
python -m scripts.sync      # index what is new or has changed
python -m scripts.describe  # write what each document contains, one model call each
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
title, and it is part of what the answer is written from. A second run costs nothing — a
described document is left alone unless you name it or pass `--all`.

`DESCRIPTION_SAMPLE_CHARS` is how much of a document is read to write its description (6000
by default); `DESCRIPTION_BUDGET_CHARS` is how much of the catalog goes into one answer's
context (2000). The documents are named in that context either way, so a scope too large to
describe loses the descriptions and nothing else; 0 turns either budget off.

The sync does not call a model, so it stays free and needs no key. That is why this is a
command of its own rather than a step of the sync: nothing is spent until you ask, and a
document is described by `python -m scripts.describe`, not when it is indexed.

## Web interface

The same library in a browser, with the same scopes:

```bash
python -m scripts.serve     # http://127.0.0.1:8000
```

The catalog is on the left, grouped by category, each document with its title, its
description and its status; clicking a document asks about that document. The conversation is
on the right, and the sources of the answer appear as soon as the search is done, while the
answer is still being written. The selector at the top chooses what the next question is asked of, and prints the
scope in the same words the console prints it in.

Under each answer there is a line saying what was searched for. It is not always the question
as it was typed: the question is first rewritten together with the conversation so far, so
that "and for minors?" becomes a question that stands on its own. The line is what that
rewrite produced, and it is worth reading when an answer seems to have come from the wrong
document.

The answer is told which documents were searched, and what the catalog says about each of
them, and not only which passages were found — see [Descriptions](#descriptions). That is
what makes a question about the library itself answerable.

Conversations are kept in `data/conversations.sqlite3`, so a conversation is still there
after the server is restarted. Changing the scope starts a new one: a conversation's history
is the history of questions about those documents.

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
another one needs its own index and the documents indexed again. A sync pointed at an index
built with a different model stops and says so.

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
