# Repertorio Docs

<img width="2172" height="724" alt="ChatGPT Image Sep 10, 2026, 08_48_25 PM" src="https://github.com/user-attachments/assets/f86fb6f5-59cf-4404-852c-8e7dc9993b1b" />


A question answering system over a folder of PDFs. It indexes the documents, answers questions in
plain language from what they contain, and says which file and which page each answer came from.

## How it works

Indexing is one command, `python -m scripts.sync`, and four steps.

`app/pdf.py` reads each PDF with PyMuPDF and turns it into pieces of text. Sections come from the
PDF outline, or from the font size when the file has no outline, and tables are cut out separately.

`app/ingestion.py` splits that text into chunks of about 900 characters with 150 of overlap, using
`RecursiveCharacterTextSplitter` from `langchain-text-splitters`. Every chunk keeps its file path,
its page and its character range.

`app/embeddings.py` turns each chunk into a vector with `HuggingFaceEmbeddings` from
`langchain-huggingface` and the `BAAI/bge-m3` model, which runs on your machine through
sentence-transformers.

`app/vectorstore.py` writes the vectors to Pinecone, or to a local folder through `langchain-chroma`
when `VECTOR_STORE=chroma`.

`app/catalog.py` records each document in a SQLite table with its path, file hash, status and
counts. A later sync compares the folder with those rows and re-indexes only what changed.

Answering a question is a graph of three steps in `app/rag_graph.py`, built with `langgraph`.

1. The question is rewritten together with the conversation so far, so a follow-up becomes a
   question that stands on its own.
2. That question is searched, and the passages it returns go into a context together with the list
   of documents that were searched and the description of each.
3. The context goes to the chat model, which writes the answer.

The chat model is `ChatOpenAI` from `langchain-openai`, pointed at DeepSeek by default and at any
other endpoint that speaks the same API. The conversation is checkpointed between turns, which is
what makes a follow-up question a follow-up.

Two front ends read that same graph. `scripts/chat.py` asks from the terminal, and the FastAPI app
in `app/api.py` serves the page in `web/` and streams answers to it as server-sent events. The
remaining dependencies are `pydantic` for the request body, `uvicorn` to run the server,
`python-dotenv` to read `.env`, and the standard library `sqlite3` and `argparse` for the catalog and
the command line. The page in `web/` is plain HTML, CSS and JavaScript with no build step.

## Getting started

You need Python 3.11 or later.

```bash
pip install -r requirements.txt
```

Put the keys in a `.env` file in the project root. DeepSeek writes the answers and Pinecone stores
the passages. The embeddings run on your machine, so they need no key, and with `VECTOR_STORE=chroma`
neither does the store.

```text
OPENAI_API_KEY=...
PINECONE_API_KEY=...
```

Put your PDFs in `data/documents/`, then index them and ask:

```bash
python -m scripts.sync      # index what is new or has changed
python -m scripts.chat      # ask questions about it
```

The PDFs need a text layer. A scan of page images produces no text and is not indexed.

## Commands

All of them live in `scripts/` and read the same `.env`, and each prints the store and the catalog
it is working on before it starts.

| Command | What it does |
| --- | --- |
| `python -m scripts.sync` | indexes new and changed files, and describes them |
| `python -m scripts.chat` | asks questions from the console |
| `python -m scripts.catalog` | shows the categories and their counts |
| `python -m scripts.describe` | writes a document description |
| `python -m scripts.structure` | reads the sections of each document |
| `python -m scripts.delete` | removes a document from the index and the catalog |
| `python -m scripts.serve` | serves the web interface |
| `python -m scripts.eval` | measures the search over a set of questions |

`python -m scripts.sync --dry-run` and the same flag on `delete`, `describe` and `structure` print
what a run would do without changing anything.

## Categories and scopes

A document's category is the folder it sits in, so `data/documents/manuali/a.pdf` is in `manuali`.
The folder decides this at the first indexing, and the catalog is the authority after that, so a file
moved to another folder afterwards is a new document to the next sync.

```bash
python -m scripts.catalog                          # the categories, with their counts
python -m scripts.catalog --documents              # ... and the documents under each
python -m scripts.catalog move manuals/a.pdf --to archive
```

Moving a document writes to the catalog alone. Nothing is re-indexed and the file does not move.

A question can be asked of the whole library, of a category, of one document, or of a few:

```bash
python -m scripts.chat --category manuals
python -m scripts.chat --document manuals/a.pdf
python -m scripts.chat --documents manuals/a.pdf reports/b.pdf
```

One document small enough to fit the model's context is read whole, in reading order, so a ranking
cannot leave part of it out. `WHOLE_DOCUMENT_MAX_CHARS` is the budget, 24000 by default, and 0
always searches by similarity.

## Documents, descriptions and the trash

A file taken out of `data/documents/` goes to the trash. Its row stays in the catalog and it stops
appearing in answers, so putting the file back where it was and syncing again brings it back.
`python -m scripts.delete` is the command that removes it for good, from the index and from the
catalog. The file itself stays unless `--with-file` is given.

A description says what a document contains, which a search cannot answer on its own, because a
small document never comes back beside a large one. `app/descriptions.py` writes it with one call
to the chat model over a sample of pages taken from across the document, and the catalog keeps it.
It is part of the context of every answer, so questions about the library itself have something to
be answered from.

The sync writes a description for each document it indexes that has none, so indexing costs one
model call per document. `python -m scripts.sync --no-descriptions` skips them, and
`python -m scripts.describe` writes them later without indexing anything again. A document whose
file changed is described again.

## The structure of a document

Beside the text, the catalog keeps what each document is made of: its sections, and the tables and
figures that sit inside them. The sections come from the outline of the PDF, or from the size of the
text when there is no outline. A figure has no text of its own, so it is placed by its position on
the page. Nothing here calls a model, so nothing here can invent a section a document does not have.

`python -m scripts.structure` reads every indexed document and writes its structure. Naming a
document on the command line reads that one again, and `--dry-run` prints the tree and writes
nothing.

```bash
python -m scripts.structure                        # every indexed document
python -m scripts.structure manuals/a.pdf          # one document, read again
python -m scripts.structure --dry-run              # the tree, and nothing written
```

It opens no vector store and calls no model, so it is safe on a library that is already answering
questions, and it changes nothing about how one is answered. The sync writes the structure as it
indexes, so a document indexed from now on has one without a second command. A document that cannot
be read a second time is still indexed and answerable, and the line for it says so.

## Web interface

```bash
python -m scripts.serve     # http://127.0.0.1:8000
```

The catalog is on the left, grouped by category, and clicking a document asks about that document.
The conversation is on the right. The sources appear as soon as the search is done, while the answer
is still being written, and under each answer is the query that was actually searched, which is the
question after it was rewritten.

Selecting several documents asks about them together. Conversations are kept in
`data/conversations.sqlite3`, one per scope, so they survive a restart and a document's history is
not mixed with another's.

The server listens on `127.0.0.1` only and has no authentication. `--host 0.0.0.0` opens it to the
network, where anyone who can reach the port can read the whole library.

## Configuration

Every setting is read in `app/config.py` and has a default, so `.env` only needs the ones you change.

| Variable | Default | What it does |
| --- | --- | --- |
| `CHAT_PROVIDER` | `deepseek` | where the answers come from, also `openai` or `local` |
| `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_API_KEY` | from the provider | the chat model |
| `EMBEDDING_PROVIDER` | `local` | the embedding model, on your machine or through OpenAI |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | which model |
| `EMBEDDING_DEVICE` | `cpu` | `cuda` moves the model to the GPU |
| `VECTOR_STORE` | `pinecone` | where the passages are kept, or `chroma` for a local folder |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | `900`, `150` | how a document is cut into chunks |
| `RETRIEVAL_K` | `5` | how many passages the search returns |
| `WHOLE_DOCUMENT_MAX_CHARS` | `24000` | above this, a single document is searched by similarity |
| `DESCRIPTION_SAMPLE_CHARS` | `6000` | how much of a document is read to describe it |
| `DESCRIPTION_BUDGET_CHARS` | `2000` | how much of the catalog goes into one answer |
| `RERANK` | `off` | whether a cross-encoder reorders what the search found |
| `SMALL_TO_BIG` | `off` | `page` gives the model the page each passage came from |
| `GRADE` | `on` | whether the material is judged before an answer is written |
| `DOCUMENTS_DIR` | `data/documents` | the folder that is watched |

The two stores hold different vectors, so each keeps its own catalog, `data/catalog.sqlite3` for
Pinecone and `data/catalog-chroma.sqlite3` for Chroma. Both follow `VECTOR_STORE` on their own.

The chat model can be swapped for any endpoint that speaks the OpenAI API. The embedding model
cannot, because the index is built in the vector space of one model. The catalog records which model
indexed each document, so a sync with a different one notices and rebuilds them.

## Measuring retrieval

`python -m scripts.eval` runs a set of questions against the same retriever the console uses. Four
of the numbers it reports come from the search alone and cost no model call. Two are counts of the
expected document and the expected page, which do not change with the position of a passage. The
mean reciprocal rank is where the document was found. nDCG@10 is the order of everything that came
back, so a change that moves a passage up the list shows even when nothing was gained or lost.

The set is a JSON file naming, for each question, the document that holds the answer and, when it is
worth writing down, the page and what a correct answer says. The default is
`data/evals/questions.json`, which is kept out of the repository.

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

What a run does by default is read the material of each question and say whether it could answer it,
one model call per question. That is the grade line, and it is the refusal number. `--no-grade`
leaves it out, and what is left is a run that calls no model at all. `--answers` writes an answer to
each question from the passages the search returned, and `--judge` adds a second call per question,
to a model that reads the answer against its context and says whether every claim is supported. That
last one is the faithfulness number.

A question with no `document` is one the library cannot answer. Those are asked too, and what is
counted for them is the opposite: whether the material was turned away.

## Grading

A similarity search always returns something, so a question the library cannot answer still comes
back with the nearest passages there are and an answer is written from them. Before an answer is
written, a model reads the question and the passages and says whether they hold the answer. When
they do not, the query written with that verdict is searched for instead, and if that fails too the
question is turned away.

```text
GRADE=on                     # on by default
GRADE_ATTEMPTS=2             # how many searches one question may take
```

It is on by default because the failure it prevents is the expensive one: an answer written from the
wrong page, in the same confident voice as an answer written from the right one. It costs one model
call per question, and one more when the question is turned away. `python -m scripts.eval` measures
the verdict on every question for the same reason, and `--no-grade` is what leaves it out.

## Reranking

An embedding is computed before the question is known, so a similarity search can only say that two
texts sit near each other. A cross-encoder reads the question and the passage together, which is one
forward pass per passage and affordable over the twenty a search returns.

```text
RERANK=on                    # off by default
RERANK_MODEL=                # empty means BAAI/bge-reranker-v2-m3
RERANK_CANDIDATES=20         # what the search is asked for, of which RETRIEVAL_K come back
```

It is off by default because on the library this project was built against the search already put
the right page first for every question, so there was nothing to reorder. On the corpus built to be
harder, ten subjects with four near-identical documents each, it is the larger of the two gains
measured below: 94 of the 97 documents against 86, with MRR 0.90 against 0.76 and nDCG@10 0.89
against 0.79, at about seven seconds per question on a CPU.

## Small to big

A search matches on chunks of about a thousand characters. That is a good size to match on and often
too small to answer from, because the sentence that settles the question can sit in the chunk next to
the one that matched. With this on, every passage the search returns is replaced by the whole page it
came from, its chunks joined back together in reading order, and two passages of one page become one.

```text
SMALL_TO_BIG=page            # off by default
```

The search is unchanged, so the order of the pages is the order of the passages that found them, and
the reranker still scores the passages rather than the pages. It is off by default because the
ranking is then read over pages instead of passages: the eval says which of the two it measured on
its context line, and two runs that disagree there are not comparable.

On that same corpus, over the 97 questions that name a document, it found the right one 91 times
instead of 86, with MRR 0.77 against 0.76 and nDCG@10 0.83 against 0.79. Five questions that were
misses became hits at the fourth or fifth position: several chunks of one document now count once, so
more documents fit in the five slots. Each page that reaches the model is about 1,500 characters
instead of 250, so there is more text in the context even though there are fewer items.

A graded pass over the same set says the same thing about the material. The grader accepted 94 of the
97 questions that name a document instead of 92, and turned away the material of 13 of the 107
instead of 21. `atti-vandalici-franchigia` is the clearest of them: its five passages were five
copies of the same "Limiti e franchigie" paragraph from five different documents, the grader refused
them and the second search did not help, where the page in each slot holds the paragraph stating
which damage the cover includes, and the question is accepted on the first material. The retry is
asked for less often for the same reason, 13 questions instead of 21, and it recovered 1 instead of
7: there is less wrongly refused material left for a second search to put right. Those are one
graded pass each, so a second would move a question or two.

## The two together

Both settings are measured over the same 107 questions, 97 of which name a document. The first three
columns come from a run with the grader off, which calls no model and repeats exactly. The last
three come from a graded pass each, which is a model reading the material and moves a question or
two from run to run.

| `RERANK` | `SMALL_TO_BIG` | documents | MRR | nDCG@10 | accepted | refused first | turned away |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `off` | `off` | 86/97 | 0.76 | 0.79 | 92/97 | 21 | 14 |
| `off` | `page` | 91/97 | 0.77 | 0.83 | 94/97 | 13 | 12 |
| `on` | `off` | 94/97 | 0.90 | 0.89 | 94/97 | 12 | 12 |
| `on` | `page` | 96/97 | 0.91 | 0.93 | 94/97 | 13 | 12 |

Accepted counts the questions whose material the grader kept, out of the 97 a document answers;
refused first is how often it turned the material away before the second search, and turned away is
the material it still refused once the retry was spent. Each of the two helps, and neither undoes
the other. The case against them was that they would compete for the same five slots, since the
reranker can pick two passages of one page and small to big turns those into one item; it did not
show, and with both on the search finds 96 of the 97 documents. What neither changes is the grader:
it stays at 94 accepted whatever is on, because it was already nearly right about the material, and
the refinements change which page the answer is written from and how much of it the model sees.
Nine of the ten questions no document answers are refused in all four, so the tenth is a false
positive that none of the three touches.

One thing this table does not cover. Only the first row has been through `--answers` and `--judge`:
93 answers written, 86 holding the text the set expects, and 89 of 93 accepted by the judge. Whether
a page-long context helps or dilutes what the model writes is open.

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest -q
ruff check .
```

The test suite is offline. It needs no API keys, no network and no model download, because every
test that would reach a model or a store replaces it with something local. The same two commands run
on every push in GitHub Actions.

## Roadmap

The project is growing into a personal document library. The plan is in [ROADMAP.md](ROADMAP.md).
