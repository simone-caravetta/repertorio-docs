# Roadmap

Where the project is now: a working RAG backend that ingests PDFs and answers questions
about them from a console. Where it is going: a personal document library — a catalog you
can browse, a reader you can ask questions from, and tools that organise the documents for
you.

The phases are ordered by dependency, not by appeal. Each one leaves the project in a
usable state.

## Decide before building

Cheap now, expensive later. These shape the data model, so everything else depends on
them.

- **Source of truth.** If categories are managed from the UI, the catalog is the authority
  and the filesystem is storage. If they are managed by moving folders around, the
  filesystem stays the authority. Every later decision follows from this one.
- **Document identity.** The catalog needs a stable id. Using the file path keeps it
  simple, but renaming then has to be a managed operation that preserves the id, rather
  than a filesystem event that breaks the catalog.
- **Vector store.** Pinecone or a local store (LanceDB, Chroma, sqlite-vec). A local store
  makes the whole app one folder you can copy, removes an API key and works offline.
  Switching later means re-embedding the entire corpus, so it is worth deciding early.
- **Text coordinates at ingest.** Highlighting a cited passage requires word positions in
  the page, which `PyPDFLoader` does not provide. Capturing them with PyMuPDF also
  improves chunking (keeping tables intact, splitting on headings), so it pays twice.
  Retro-fitting means reprocessing the corpus.
- **Deletion semantics.** Hard delete, or a trash state that hides a document from
  retrieval but keeps it recoverable. Costs almost nothing to design in now.

## Phase 1 — Document lifecycle

A document has to be an entity before it can be listed, categorised or managed.

- [ ] Catalog database (SQLite): id, path, title, description, category, file hash, status,
      page and chunk counts, timestamps.
- [ ] Delete by metadata filter on `source`.
- [ ] Update as delete + re-add, triggered by comparing the file hash.
- [ ] A `sync` command that reconciles filesystem, catalog and index: add, update, delete.
- [ ] Per-document status (`queued`, `indexing`, `indexed`, `failed`, `needs_ocr`) shown
      everywhere the document appears.
- [ ] Ingestion as a background job with a queue table and progress, so large documents and
      OCR do not block.
- [ ] Embedding cache (content hash → vector), so re-indexing is close to free.
- [ ] Tests, linting and CI — the sync and ingestion code is exactly the kind that breaks
      silently.

## Phase 2 — Categories and scoped search

- [ ] Derive the category from the folder layout at ingest, stored as chunk metadata.
- [ ] Category tree in the catalog, and the operation to move a document between categories
      (metadata only, no re-embedding).
- [ ] Turn the retriever into a factory that accepts `k` and a metadata filter.
- [ ] Scoped chat: by category, by document, by selection of documents.
- [ ] Skip retrieval entirely when the scoped document fits in the context window.

## Phase 3 — The application

- [ ] FastAPI with SSE streaming for tokens and sources.
- [ ] Catalog view: documents grouped by category, each with title, description and status.
- [ ] Document detail: description, metadata, related documents, actions.
- [ ] PDF viewer with citation → page jump.
- [ ] Chat panel with a scope selector.
- [ ] Inbox: recently ingested documents waiting for confirmation of the proposed category
      and tags.
- [ ] Persistent checkpointer (SQLite) so conversations survive a restart — the same
      mechanism gives conversation time travel through `get_state_history`.
- [ ] Stream the sources as soon as retrieval completes, instead of at the end of the
      answer.
- [ ] Summarise older turns once the history outgrows its budget.
- [ ] `interrupt()` when retrieval is weak, to ask which document was meant.

## Phase 4 — Smart ingestion

- [ ] LLM-generated title and description at ingest, sampled across the document rather
      than only its beginning, and editable by hand.
- [ ] Auto-derived tags, document type, language and date.
- [ ] Structure-aware chunking with PyMuPDF: keep tables whole, split on headings.
- [ ] Duplicate and near-duplicate detection from the embeddings.
- [ ] OCR pipeline: detect pages with no text layer, run `ocrmypdf` (which keeps
      coordinates, so highlighting works on scanned documents too), and flag low-confidence
      pages for a VLM fallback.
- [ ] Additional formats: DOCX, Markdown, HTML, TXT.

## Phase 5 — Retrieval quality

Start with the harness: without a measurement, everything below is guesswork.

- [ ] Eval harness: questions with their expected source, measuring retrieval hit-rate and
      answer faithfulness.
- [ ] Tracing (Langfuse) to inspect prompts, tokens and latency per node.
- [ ] Reranking with `bge-reranker-v2-m3` after retrieval — no index change required.
- [ ] Hybrid search using the sparse weights BGE-M3 already produces, for codes, names and
      numbers where dense retrieval is weak.
- [ ] Relevance grading node with a conditional edge: rewrite the query and retry, or state
      that the answer is not in the documents.
- [ ] Small-to-big retrieval: match on small chunks, hand the surrounding section to the
      model.
- [ ] Multi-query expansion with reciprocal rank fusion.

## Phase 6 — The reader and the knowledge layer

- [ ] Highlight cited passages inside the page.
- [ ] Chat anchored to the visible page rather than the whole document.
- [ ] Selection → ask, translate, annotate.
- [ ] Notes attached to a page, indexed alongside the documents so they are searchable and
      citable.
- [ ] Reading position and progress.
- [ ] Related documents, from the embedding centroid.

## Phase 7 — Tools

Deterministic actions first: a button that extracts deadlines should extract deadlines, not
start an agent.

- [ ] Structured extraction: dates, amounts, key points — exportable as CSV or Markdown.
- [ ] Deeper on-demand summary.
- [ ] Compare two documents.
- [ ] Translate a section.
- [ ] Regenerate description, rename, recategorise, move to trash.
- [ ] Saved searches and rule-based collections that update themselves.
- [ ] Agentic tool-calling, for the questions that genuinely need several steps.

## Later

- [ ] Local LLM through Ollama, for a fully offline setup.
- [ ] Multi-machine access.

## Non-goals

- Multi-user, authentication and permissions.
- Collaboration and sharing.
- Mobile.
- Becoming a general-purpose PDF reader: viewer features that do not feed retrieval,
  citations or notes are out of scope.
