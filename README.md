# Repertorio Docs

<img width="2172" height="724" alt="ChatGPT Image Sep 10, 2026, 08_48_25 PM" src="https://github.com/user-attachments/assets/f86fb6f5-59cf-4404-852c-8e7dc9993b1b" />


Turn a folder of PDFs into a knowledge base you can question in plain language.

Point it at a folder, run one command, and ask your questions from the console. Every answer
comes from your documents, with the file and the page it came from; follow-up questions
work, and when the documents don't cover something it says so instead of inventing an
answer.

## Getting started

You need Python 3.11 or later and two API keys — DeepSeek writes the answers, Pinecone holds
the passages. The embeddings run on your machine, so nothing is sent away to be indexed.

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with the two keys. It is excluded from version
control:

```text
DEEPSEEK_API_KEY=...
PINECONE_API_KEY=...
```

Put your PDFs in `data/documents/`, then:

```bash
python -m scripts.sync     # index what is new or has changed
python -m scripts.chat     # ask questions about it
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

The PDFs need a selectable text layer: a scan without one is not indexed.

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
