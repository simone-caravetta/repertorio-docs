# Roadmap

Where the project is now: a working RAG backend that ingests PDFs and answers questions
about them from a console. Where it is going: a personal document library — a catalog you
can browse, a reader you can ask questions from, and tools that organise the documents for
you.

The phases are ordered by dependency, not by appeal. Each one leaves the project in a
usable state, and they alternate on purpose: everything whose failure is loud and
measurable is built headless and driven from the console, and the interface comes in when
the failure would otherwise be silent.

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

## Phase 1 — Documents become entities

Today a PDF is a file that happens to have been indexed. This phase gives it an id, a
status and a history — everything that listing, categorising and managing depend on.

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

## Phase 2 — Models you can swap

The chat model is free to swap; the embedding model gets curated, because a wrong chat
model gives one bad answer while a wrong embedding model gives quietly wrong search results
over the whole corpus.

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

## Phase 3 — Categories and scope

Categories are metadata, not storage: moving a document between them should never mean
re-embedding it.

- [ ] Derive the category from the folder layout at ingest, stored as chunk metadata.
- [ ] Category tree in the catalog, and the operation to move a document between categories
      (metadata only, no re-embedding).
- [ ] Turn the retriever into a factory that accepts `k` and a metadata filter.
- [ ] Scoped chat: by category, by document, by selection of documents.
- [ ] Skip retrieval entirely when the scoped document fits in the context window.

## Phase 4 — Out of the console

The core works; now it needs a face. The smallest slice that makes the library usable: you
open it, see your documents, and ask questions.

- [ ] FastAPI with SSE streaming for tokens and sources.
- [ ] Catalog view: documents grouped by category, each with title, description and status.
- [ ] Chat panel with a scope selector.
- [ ] Persistent checkpointer (SQLite) so conversations survive a restart — the same
      mechanism gives conversation time travel through `get_state_history`.
- [ ] Stream the sources as soon as retrieval completes, instead of at the end of the
      answer.

## Phase 5 — Ingestion that organises

- [ ] LLM-generated title and description at ingest, sampled across the document rather
      than only its beginning, and editable by hand.
- [ ] Auto-derived tags, document type, language and date.
- [ ] Structure-aware chunking with PyMuPDF: keep tables whole, split on headings.
- [ ] Duplicate and near-duplicate detection from the embeddings.
- [ ] OCR pipeline: detect pages with no text layer, run `ocrmypdf` (which keeps
      coordinates, so highlighting works on scanned documents too), and flag low-confidence
      pages for a VLM fallback.
- [ ] Additional formats: DOCX, Markdown, HTML, TXT.

## Phase 6 — Retrieval you can measure

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

## Phase 7 — Relations beyond similarity

A graph of control flow decides where to search, not what is known. This phase adds the
layer that holds what is known, in three tiers — cheapest first, and the expensive one only
if the measurement justifies it.

- [ ] Structure graph, deterministic, from PyMuPDF: headings → sections → subsections →
      pages, and which table or figure belongs to which section. No LLM and nothing to
      hallucinate, and it is what makes a section-scoped answer or "show me what surrounds
      this passage" possible.
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

## Phase 8 — Opening the document

The catalog gets you to the document; this is what happens once you open one.

- [ ] Document detail: description, metadata and actions.
- [ ] PDF viewer with citation → page jump.
- [ ] Inbox: recently ingested documents waiting for confirmation of the proposed category
      and tags.
- [ ] Summarise older turns once the history outgrows its budget.
- [ ] `interrupt()` when retrieval is weak, to ask which document was meant.

## Phase 9 — A reader you can write in

The viewer becomes a place to work in, not only to look at.

- [ ] Highlight cited passages inside the page.
- [ ] Chat anchored to the visible page rather than the whole document.
- [ ] Selection → ask, translate, annotate.
- [ ] Notes attached to a page, indexed alongside the documents so they are searchable and
      citable.
- [ ] Reading position and progress.
- [ ] Related documents, from the embedding centroid.

## Phase 10 — Tools: buttons before agents

Everything here starts as a deterministic action. Only the last bullet needs an agent, and
it is last on purpose.

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
