# Repertorio Docs

![Uploading ChatGPT Image Sep 10, 2026, 08_48_25 PM.png…]()


Turn a folder of PDFs into a knowledge base you can question in plain language.

Local embeddings, Pinecone as the vector store, LangGraph for conversation memory,
DeepSeek for the answers. There is no web UI yet: `scripts/chat.py` is a console chat
that streams the answer as it is generated.

## Features

- **Answers grounded in your documents.** Every response is built only from the
  passages retrieved from your PDFs, and it lists the source file and page.
- **Follow-up questions work.** A rewrite step turns "and for minors?" into a
  standalone question before the search, so you don't have to repeat the context.
- **No invented facts.** If the retrieved passages don't cover the question, the model
  is instructed to say so instead of guessing.
- **No external embedding service.** `BAAI/bge-m3` runs locally with Sentence
  Transformers, so your documents are not sent to a third party for indexing.
- **Keeping the library up to date is one command.** Drop a PDF in the folder and sync:
  new files are indexed, edited ones are indexed again, deleted ones are removed from the
  index. Re-running it changes nothing.

## How it works

```text
Ingestion   PDF → chunks → local embeddings → Pinecone
Chat        question → standalone query → vector search → streamed answer + sources
```

Conversation memory is bound to a `thread_id`, so each conversation keeps its own history.

## Requirements

- Python 3.11 or later
- A DeepSeek API key and a Pinecone API key
- About 2 GB of free disk space for the embedding model

## Getting started

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The first run downloads the `BAAI/bge-m3` model through Hugging Face/Sentence
Transformers if it is not already in your local cache. It takes a few minutes.

### 2. Configure

Create a `.env` file in the project root with your two API keys:

```text
DEEPSEEK_API_KEY=...
PINECONE_API_KEY=...
```

Everything else has a sensible default and can be left alone. See the
[configuration reference](#configuration-reference) if you want to change the chunking,
the number of retrieved passages or the device used for embeddings.

The `.env` file is excluded from version control — never commit your API keys.

### 3. Add your PDFs

Put them in `data/documents/`:

```text
data/documents/
├── manuals/
│   └── manual.pdf
└── reports/
    └── report-2026.pdf
```

Subfolders are scanned recursively, and the relative path is stored as the document
source, so it shows up in the citations.

The PDFs must have a selectable text layer. A scanned document without a text layer never
reaches the index; it is recorded as failed and retried on every sync until OCR arrives.

### 4. Sync

```bash
python -m scripts.sync
```

This compares the folder with the catalog and the vector store, and applies the
difference: new PDFs are read, split into overlapping chunks, embedded locally and
uploaded to Pinecone; edited files are indexed again; files you deleted are removed from
the index. Documents already up to date are left alone, so running it twice does nothing
the second time. The Pinecone index is created on the first run if it does not exist yet.

Add `--dry-run` to see what it would do without changing anything. It reads only, so it
needs no API keys and no network:

```bash
python -m scripts.sync --dry-run
```

### 5. Chat

```bash
python -m scripts.chat
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

## Document lifecycle

Every document has a row in a small SQLite catalog, `data/catalog.sqlite3`, created on the
first sync. Its path — relative to `data/documents/` — is its identity, and the row keeps
the title, the hash of the file, the page and chunk counts and the last error.

The status says where the document stands:

| Status | Meaning |
|---|---|
| `queued` | In the catalog, waiting to be read. |
| `indexing` | Being read right now. |
| `indexed` | Its chunks are in the vector store. |
| `failed` | The last run could not read it. The reason is on the row, and the next sync tries again. |
| `trashed` | The file is gone from the folder. Its chunks were removed, its metadata kept. |

Trashing is a removal from the index, not a deletion from the catalog: put the file back
where it was, run the sync again and the document is indexed again with its title intact.

The catalog is the source of truth. Removing vectors by hand from the Pinecone console
leaves the catalog describing a document that is no longer there; the next sync finds the
file unchanged and skips it, so those vectors do not come back on their own.

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | **Required.** DeepSeek API key. |
| `PINECONE_API_KEY` | — | **Required.** Pinecone API key. |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | Model used to rewrite the question and write the answer. |
| `PINECONE_INDEX_NAME` | `repertorio-docs` | Name of the Pinecone index. Created if missing. |
| `PINECONE_NAMESPACE` | `documents` | Namespace used inside the index. |
| `PINECONE_CLOUD` | `aws` | Cloud for the serverless index. |
| `PINECONE_REGION` | `us-east-1` | Region for the serverless index. |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Sentence Transformers model, run locally. |
| `EMBEDDING_DEVICE` | `cpu` | `cpu`, `cuda` or `mps`. |
| `EMBEDDING_BATCH_SIZE` | `32` | Batch size used when encoding chunks. |
| `CHUNK_SIZE` | `900` | Maximum chunk length, in characters. |
| `CHUNK_OVERLAP` | `150` | Overlap between consecutive chunks. |
| `RETRIEVAL_K` | `5` | Number of passages retrieved per question. |
| `DOCUMENTS_DIR` | `data/documents` | Folder holding the documents. |
| `CATALOG_DB_PATH` | `data/catalog.sqlite3` | SQLite file holding the document catalog. |

The Pinecone index is created with 1024 dimensions, matching `BAAI/bge-m3`. If you switch
to an embedding model with a different output size, update `pinecone_dimension` in
`app/config.py`.

## Project layout

```text
app/       the RAG itself: catalog, ingestion, embeddings, vector store, graph
scripts/   command line entry points: sync and chat
data/      the PDFs to index and the catalog (not versioned)
tests/     the offline test suite: no API keys, no network, no model download
```

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest -q
ruff check .
```

The test suite is entirely offline: it needs no API keys, opens no connection and
downloads no model. The same two commands run on every push in GitHub Actions.

## Roadmap

The project is growing into a personal document library: a catalog you can browse, a reader
you can ask questions from, and tools that organise the documents for you. The full plan —
document lifecycle, categories, the catalog UI, OCR, retrieval quality and the reader —
lives in [ROADMAP.md](ROADMAP.md).
