# Roadmap

Where the project is now: a working RAG backend that ingests PDFs and answers questions
about them from a console. Where it is going: a personal document library — a catalog you
can browse, a reader you can ask questions from, and tools that organise the documents for
you.

The phases are ordered by dependency, not by appeal, and each one leaves the project in a
usable state. The interface is built in two steps, a first version at Phase 4 and the rest
at Phase 8, because everything before it can be built and verified from the command line.

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
- **Embedding model.** It defines the vector space the whole corpus lives in, so it is a
  decision at corpus level rather than a preference: changing it means re-indexing
  everything, and two models with the same dimension are still not compatible. Pick a model
  known to work, record it with the index, and treat it as an invariant.
- **Text coordinates at ingest.** Highlighting a cited passage requires word positions in
  the page, which `PyPDFLoader` does not provide. Capturing them with PyMuPDF also
  improves chunking (keeping tables intact, splitting on headings), so it pays twice.
  Retro-fitting means reprocessing the corpus.
- **Deletion semantics.** Hard delete, or a trash state that hides a document from
  retrieval but keeps it recoverable. Costs almost nothing to design in now.

## Phase 1 — Document lifecycle

A document needs an id, a status and a history before it can be listed, categorised or
managed.

- [x] Catalog database (SQLite): id, path, title, description, category, file hash, status,
      page and chunk counts, timestamps.
- [x] Delete by metadata filter on `source`.
- [x] Update as delete + re-add, triggered by comparing the file hash.
- [x] A `sync` command that reconciles filesystem, catalog and index: add, update, delete.
- [x] Delete a document from the index and the catalog, as a command of its own: the sync
      only trashes, and the trash is emptied on request. The file is a separate, opt-in
      step, so that a delete does not quietly become a way to lose documents.
- [x] Per-document status (`queued`, `indexing`, `indexed`, `failed`, `needs_ocr`) shown
      everywhere the document appears. (`needs_ocr` is reserved but never assigned until
      OCR exists; a scanned PDF is `failed` for now. The only place a status appears today
      is the catalog itself — there is no interface to show it in yet.)
- [ ] Ingestion as a background job with a queue table and progress, so large documents and
      OCR do not block. (Deferred to Phase 5: it only pays off once OCR makes ingestion
      slow.)
- [ ] Embedding cache (content hash → vector), so re-indexing is close to free. (Deferred
      to Phase 5, for the same reason.)
- [x] Tests, linting and CI — the sync and ingestion code is exactly the kind that breaks
      silently.

## Phase 2 — Model providers

The chat model is free to swap; the embedding model gets curated, because a wrong chat
model produces one bad answer, while a wrong embedding model produces wrong results for the
whole corpus.

- [ ] Chat model as any OpenAI-compatible endpoint, configured with `base_url`, model and
      key, so DeepSeek, a local vLLM/Ollama server or a hosted provider can be swapped
      without touching the graph. Provider-specific parameters (`extra_body`) live in the
      profile, not in the code.
- [ ] Embeddings from a curated list instead of free choice: a few known-good models, each
      with its declared dimension and a measured result on the sample corpus.
- [ ] Embedding configuration as two separate fields: the curated model id, and the way it
      is served (`local` in-process, or `api` against an OpenAI-compatible endpoint). The
      same model served either way produces the same vectors, so changing the delivery mode
      needs no reindex.
- [ ] Fingerprint the index: embed a fixed probe string when the index is created, store
      the resulting vector, and re-check it at startup. Compare with a tolerance rather
      than for equality, so the same model in a different build (a quantised server versus
      the local one) is not flagged, while a genuinely different model is. On a mismatch,
      refuse to search and offer the reindex instead of answering from vectors that cannot
      be compared.
- [ ] Include the embedding model in the collection name, so changing model creates a new
      index instead of corrupting the existing one, and rolling back is just pointing at
      the previous one.
- [ ] Assign models per node: a small local model for `contextualize` and for grading, the
      strong model only for the answer.
- [ ] Declare the capabilities of each profile (tool calling, structured output) and
      degrade gracefully where they are missing.
- [ ] Health check per profile, with a clear message when a local server is not running.
- [ ] Record which model produced each conversation.
- [ ] An escape hatch for anything else: a `custom` profile that accepts any endpoint
      behind an explicit warning and a full reindex.

## Phase 3 — Categories and scoped search

Categories are metadata, not storage: moving a document between them should never mean
re-embedding it.

- [ ] Derive the category from the folder layout at ingest, stored as chunk metadata.
- [ ] Category tree in the catalog, and the operation to move a document between categories
      (metadata only, no re-embedding).
- [ ] Turn the retriever into a factory that accepts `k` and a metadata filter.
- [ ] Scoped chat: by category, by document, by selection of documents.
- [ ] Skip retrieval entirely when the scoped document fits in the context window.

## Phase 4 — The first web interface

The first version of the interface: you open it, see your documents, and ask questions
about them.

- [ ] FastAPI with SSE streaming for tokens and sources.
- [ ] Catalog view: documents grouped by category, each with title, description and status.
- [ ] Chat panel with a scope selector.
- [ ] Persistent checkpointer (SQLite) so conversations survive a restart — the same
      mechanism gives conversation time travel through `get_state_history`.
- [ ] Stream the sources as soon as retrieval completes, instead of at the end of the
      answer.

## Phase 5 — Smart ingestion

- [ ] LLM-generated title and description at ingest, sampled across the document rather
      than only its beginning, and editable by hand.
- [ ] Auto-derived tags, document type, language and date.
- [ ] Structure-aware chunking with PyMuPDF: keep tables whole, split on headings.
- [ ] Duplicate and near-duplicate detection from the embeddings.
- [ ] OCR pipeline: detect pages with no text layer, run `ocrmypdf` (which keeps
      coordinates, so highlighting works on scanned documents too), and flag low-confidence
      pages for a VLM fallback.
- [ ] Additional formats: DOCX, Markdown, HTML, TXT.

## Phase 6 — Retrieval quality

Start with the harness: without a measurement, the rest of this phase is guesswork. It
doubles as the tool that tells whether a local model is good enough for a given node.

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

## Phase 7 — Knowledge graph

The orchestration graph decides where to search; it does not hold the relations between
documents. This phase adds the layer that does, in three tiers, cheapest first — the most
expensive one only if the measurements justify it.

- [ ] Structure graph, deterministic, from PyMuPDF: headings → sections → subsections →
      pages, and which table or figure belongs to which section. No LLM involved, so
      nothing can be hallucinated, and it is what makes a section-scoped answer possible,
      or a question about what surrounds a passage.
- [ ] Reference graph: the explicit cross-references ("see section 4.2", "as described in
      the annex", "cfr. art. 5"), extracted by pattern with an LLM fallback for the
      ambiguous ones and stored as edges between parts of documents. Today a section
      pointing at another section loses that link entirely.
- [ ] Measure before the third tier: extend the eval harness with multi-hop and
      corpus-wide questions, so the entity graph has to earn its cost instead of being
      assumed to help.
- [ ] Entity graph: entities and relations extracted into triples at ingest, each carrying
      the chunk it came from, so no answer ever rests on a triple alone.
- [ ] Entity resolution: merge the different surface forms of the same entity.
- [ ] Community detection and per-community summaries, for the questions no single passage
      contains.
- [ ] Incremental maintenance: a changed document has its triples removed and re-extracted
      — orphan triples are far harder to notice than orphan chunks.
- [ ] Route by question type: a classify node sending lookups to plain vector search, and
      relation or aggregate questions down the graph path.
- [ ] Seed with vectors, expand on the graph: retrieve chunks first, then follow their
      edges to bring in what similarity alone would miss.
- [ ] Always fall back to plain vector search when the graph has nothing.

## Phase 8 — Document detail and viewer

Opening a document from the catalog: its page, its content, and the documents waiting to be
confirmed.

- [ ] Document detail: description, metadata and actions.
- [ ] PDF viewer with citation → page jump.
- [ ] Inbox: recently ingested documents waiting for confirmation of the proposed category
      and tags.
- [ ] Summarise older turns once the history outgrows its budget.
- [ ] `interrupt()` when retrieval is weak, to ask which document was meant.

## Phase 9 — The reader and personal notes

Working with the document while reading it: highlights, notes and questions anchored to the
page.

- [ ] Highlight cited passages inside the page.
- [ ] Chat anchored to the visible page rather than the whole document.
- [ ] Selection → ask, translate, annotate.
- [ ] Notes attached to a page, indexed alongside the documents so they are searchable and
      citable.
- [ ] Reading position and progress.
- [ ] Related documents, from the embedding centroid.

## Phase 10 — Document tools

Every tool starts as a deterministic action.

- [ ] Structured extraction: dates, amounts, key points — exportable as CSV or Markdown.
- [ ] Deeper on-demand summary.
- [ ] Compare two documents.
- [ ] Translate a section.
- [ ] Regenerate description, rename, recategorise, move to trash.
- [ ] Saved searches and rule-based collections that update themselves.
- [ ] Agentic tool-calling, for the questions that genuinely need several steps.

## Later

- [ ] A fully offline setup: local vector store, local embedding server, local chat model.
- [ ] Multi-machine access.

## Non-goals

- Multi-user, authentication and permissions.
- Collaboration and sharing.
- Mobile.
- Becoming a general-purpose PDF reader: viewer features that do not feed retrieval,
  citations or notes are out of scope.
