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

- **Source of truth.** Settled by Phase 3: the catalog is the authority, and the filesystem
  seeds it. A document is filed under the folder it sits in when its row is created, and no
  later write touches the category — so a document moved by hand is not put back by the next
  sync, and reorganising folders on disk does not recategorise what is already indexed.
- **Document identity.** The catalog needs a stable id. Using the file path keeps it
  simple, and that is what the catalog is built on: the relative path is the key, and the
  chunks carry it as `source`, which is what both stores filter and delete by. What is still
  missing is a rename that preserves it, rather than a filesystem event that breaks the
  catalog.
- **Vector store.** Settled as a setting rather than a choice: `VECTOR_STORE` selects Pinecone
  or a local Chroma, and the app works with either. One line switches between them, and each
  store keeps its own catalog, so switching re-indexes into the new store rather than finding
  every file unchanged. The vectors of the store left behind stay where they are. Which one a
  corpus should live in is still answered per installation; the embedding model each document
  was indexed with is recorded — see Phase 2.
- **Embedding model.** It defines the vector space the whole corpus lives in, so it is a
  decision at corpus level rather than a preference: changing it means re-indexing
  everything, and two models with the same dimension are still not compatible. Pick a model
  known to work, record it with the index, and treat it as an invariant.
- **Text coordinates at ingest.** Built. `app/pdf.py` reads with PyMuPDF, and every chunk
  records the page it came from and the range of characters it covers in that page's text,
  which is what highlighting a citation needs and what `PyPDFLoader` did not provide. The
  same reading keeps tables whole and splits on headings, so it paid twice. A chunk indexed
  by the older reader carries no offsets and reports no range, so it has to be indexed again
  before anything can be drawn on it.
- **Deletion semantics.** Hard delete, or a trash state that hides a document from
  retrieval but keeps it recoverable. Costs almost nothing to design in now.

## Phase 1 — Document lifecycle

A document needs an id, a status and a history before it can be listed, categorised or
managed.

- [x] Catalog database (SQLite): id, path, title, description, category, file hash, status,
      page and chunk counts, how the index was built, timestamps.
- [x] Delete a document's vectors by metadata filter on `source`, whatever the store: the
      hosted one deletes by filter natively, the local one reaches the same rows through its
      own `where=`.
- [x] Update as delete + re-add, triggered by comparing the file hash and by comparing what
      built the index: another reader, another chunk size or another embedding model rebuilds
      a document whose file has not changed and whose vectors say nothing about their making.
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

The chat model is free to swap: anything that speaks the OpenAI chat completions API works,
which covers DeepSeek, an OpenAI model and a model running on your own machine. The
embedding model is a choice between two, because it defines the vector space the whole
corpus lives in: a wrong chat model produces one bad answer, a wrong embedding model
produces wrong results for everything.

- [x] Chat model as any OpenAI-compatible endpoint, configured with `OPENAI_BASE_URL`,
      `OPENAI_MODEL` and `OPENAI_API_KEY` — the names the OpenAI client already reads, so a
      server on your own machine is pointed at the same way as a hosted one, and DeepSeek,
      an OpenAI model or a local server can be swapped without touching the graph.
      Provider-specific parameters (`extra_body`) live in the preset, not in the code. One
      client class covers all three: DeepSeek is reached through the same `ChatOpenAI` as
      the others, and `langchain-deepseek` is no longer a dependency.
- [x] Embeddings from two providers: the local model (`BAAI/bge-m3`, in-process) or the
      OpenAI API, each with its own key and an overridable model name.
- [x] The graph takes its model and its retriever as arguments instead of building them at
      import, so it can be run against fakes and tested without an API key.
- [x] The index dimension is measured from the embedding model rather than declared, and a
      mismatch with the existing index stops the run before anything is written.
- [x] The vector store as a setting: `VECTOR_STORE=pinecone` for the hosted index, or
      `VECTOR_STORE=chroma` for a folder on this machine, with no account and no network. One
      store interface for both, so nothing above it knows which one is in use.
- [x] Record in the catalog which store and which embedding model it was built against, and
      re-index on a mismatch. Each store keeping its own catalog settles the store half:
      switching store starts that store's catalog from nothing and re-indexes honestly, and
      the sync warns when a catalog is pointed at a store holding no vectors. Inside one
      catalog the row carries the embedding model and the reader that built its vectors, and
      a run that would build them differently indexes the document again — so two models of
      one length no longer pass unnoticed, which a check on the dimension could not catch.
      What a name on a row cannot catch is a model that no longer answers as it did under
      that name — see the fingerprint below.
- [x] Re-index when the store changes rather than refusing. With a catalog per store this
      happens by itself: the new store's catalog is empty, so every file is new to it and the
      run indexes the whole library. The previous store's vectors are left behind, and the
      run says how many documents it is writing, so it does not read as a fault.
- [ ] A lock the console takes too, or a local store that tolerates one writer and one reader.
      The sync and delete take a lock today; the console does not, which is harmless against a
      hosted index and not obviously harmless against a folder.
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

- [x] Derive the category from the folder layout at ingest, and keep it in the catalog. The
      folder seeds it once, when the document's row is created, and the catalog owns it from
      then on, so a re-index never undoes a move.
- [x] Category tree in the catalog, and the operation to move a document between categories
      (metadata only, no re-embedding): `python -m scripts.catalog` and
      `python -m scripts.catalog move <path> --to <category>`.
- [x] Turn the retriever into a factory that accepts `k` and the documents in scope — one
      `$in` filter over `source`, the per-document key the chunks already carry and that both
      stores already filter and delete by. Deliberately not a general filter dictionary: the
      two stores do not spell filters the same way, and one shape tested on both is worth
      more than a passthrough nothing checks.
- [x] Scoped chat: by category, by document, by selection of documents — `--category`,
      `--document` and `--documents` on `python -m scripts.chat`.
- [x] Read a scoped document whole rather than by top-k when its text fits the context
      window, in reading order. The text lives in the vector store and nowhere else, so this
      is a search for all of it (`k` is the catalog's chunk count) rather than a search that
      is skipped. `WHOLE_DOCUMENT_MAX_CHARS` sets the budget, 0 turns it off.
- [ ] Carry the category in the chunk metadata too, as a projection of the catalog fact, and
      only if a measurement ever shows the filter to be the problem. Pinecone accepts 10,000
      values per `$in` operator against a 2MB request limit, so a category would have to hold
      that many documents before filtering on `source` stops being enough. The migration
      would be metadata-only: read `(id, source)` from the store, join the catalog, rewrite
      the metadata. No vector changes, so nothing is re-embedded.

## Phase 4 — The first web interface

The first version of the interface: you open it, see your documents, and ask questions
about them.

- [x] FastAPI with SSE streaming for tokens and sources: `python -m scripts.serve`, the page in
      `web/` and the application in `app/api.py`. The stream is written by hand — the question
      goes in the body of a `POST` and the events are parsed out of `text/event-stream`, which
      is one blank line and two fields and needs no library. `EventSource` would have put the
      question in the URL, where it is logged, capped in length and kept in the history.
- [x] Catalog view: documents grouped by category, each with its title and its status. What is
      not `indexed` is shown with the status it is in, and what is trashed is left out. The
      description is shown under the title when the catalog holds one, which it did not
      until Phase 5's `scripts.describe` began writing them.
- [x] Chat panel with a scope selector. The selector offers the whole library and every
      category, and a document is picked by clicking it in the catalog. The label under it is
      the string `resolve_scope` produces, character for character the one the console prints.
      Under each answer the page shows the query the search was run on, which the graph has
      already rewritten with the conversation in hand and which is therefore not always the
      question as it was typed. The answer is written with the list of documents the scope
      covers as well as with the passages found: which documents there are is not a thing a
      similarity search can say, and asked "che documenti hai?" the page answered with the one
      document that happened to be retrieved.
- [x] Several documents at once, ticked in the catalog panel behind a **Select several**
      button, which the server has taken since `resolve_scope` was written and the page could
      not ask for. A category's tick takes every indexed document at or below it, so the tick
      and the same category picked from the selector are one set; a document that is not indexed
      has no tick, because one of those in a group makes the whole group refuse. The ticks
      accumulate and a bar at the foot of the panel applies them, rather than applying
      themselves as they are made, which would let the first tick take the scope and the second
      replace it and two documents never be chosen.
- [x] A conversation has a title: the document or the group it is about, proposed from the
      group's documents and editable when it is started. The scope selector is restyled as that
      title rather than sitting in the page header, so one control names the conversation and
      drops the list of the others — a heading beside a menu would be two controls showing one
      fact. **New conversation** keeps meaning only what it meant, forget this scope's history.
      The titles are the page's, in localStorage beside the scope→thread map, and only two
      things take one away: the reader clearing the field, and deleting the conversation it
      names. A title belongs to the set of documents rather than to the conversation about
      them, which is why **New conversation** leaves it. Bounded by what the reader names on
      purpose, and the reason a group can be offered by name with no conversation on it yet.
- [x] Deleting a conversation. **Delete** beside the scope removes the thread from the page and
      the checkpoints from `data/conversations.sqlite3` behind a confirmation, and is offered
      only where the entry is the conversation's own — a document and a group. A category's
      entry and the whole library's come from the catalog and would be offered again the moment
      they were removed, so those two keep **New conversation** as the way to clear their
      history. The scope moves to the whole library afterwards: the selector offers the scope in
      force whatever is left behind it, so staying would make the delete look like it had done
      nothing while the row was already gone from the file. `adelete_thread` is the one method
      of the checkpointer that does not open the tables itself, so the route calls `setup`
      first — which is what makes a delete on a database nothing has ever been written to
      answer rather than raise.
- [x] Persistent checkpointer (SQLite) so conversations survive a restart: `build_graph` takes
      one, the server opens `data/conversations.sqlite3` for the life of the process, and a
      reloaded page reads its conversation back through `GET /api/threads/{id}`. The page
      keeps one conversation per scope rather than one running conversation, so moving
      between documents and back does not lose the questions asked about each, and **New
      conversation** beside the scope drops the thread of the scope it is on.
      `get_state_history` is the mechanism the same checkpointer would give time travel
      through, and is not used yet.
- [x] Stream the sources as soon as retrieval completes, instead of at the end of the answer:
      the graph is streamed in two modes at once, and the `retrieve` node's update reaches the
      page while the `answer` node is still writing — which is what makes the citations
      visible before the answer is.

## Phase 5 — Smart ingestion

- [x] Description per document, written by the configured chat model, one model call per
      document, over a sample taken from across the document rather than only its beginning,
      in the language the document is written in, kept in the catalog's `description`
      column. Written at ingest: the sync describes what has no description, by the same
      call `scripts.describe` makes, so a library it has just been through is described in
      full. Indexing a folder therefore costs one model call per document, and
      `--no-descriptions` indexes it without spending any — the vectors are the same either
      way, and the descriptions can be written afterwards without re-indexing. What a run
      describes is what has no description, so a described document costs nothing, and a
      document whose file has changed has its description dropped first: it describes an
      edition that is no longer there, and a stale description read under a document's title
      looks like any other. `scripts.describe` remains for the rest: a library indexed
      before the sync did this, a call that failed on the run, a document named by hand, and
      `--all` to write every one of them again.
      `DESCRIPTION_SAMPLE_CHARS` is how much of a document is read, `DESCRIPTION_BUDGET_CHARS`
      how much of the catalog goes into one answer's context. The description is placed there
      under the list of the documents searched, which is the fix for the failure that named
      this work: asked over the whole library what each document contains, a top-k search
      beside a large document returns that document's passages and nothing about the others,
      and a description is the answer rather than a sample of one. It is shown in the page
      and in `python -m scripts.catalog --documents`.
- [ ] The title generated rather than taken from the file name, and a description that can
      be edited or written by hand. Two things write the column today — the sync, as it
      indexes, and `scripts.describe` — and nothing edits what either of them left.
- [ ] Auto-derived tags, document type, language and date.
- [x] Structure-aware chunking with PyMuPDF: keep tables whole, split on headings. Each chunk
      carries the heading it sits under and its level, and says when it is a table.
- [ ] Duplicate and near-duplicate detection from the embeddings.
- [ ] OCR pipeline: detect pages with no text layer, run `ocrmypdf` (which keeps
      coordinates, so highlighting works on scanned documents too), and flag low-confidence
      pages for a VLM fallback.
- [ ] Additional formats: DOCX, Markdown, HTML, TXT.

## Phase 6 — Retrieval quality

Start with the harness: without a measurement, the rest of this phase is guesswork. It
doubles as the tool that tells whether a local model is good enough for a given node.

- [x] Eval harness: questions with their expected source, measuring retrieval hit-rate and
      answer faithfulness. `python -m scripts.eval` reads a JSON set — the document each answer
      should come from, and optionally the page and what a correct answer says — and measures
      the search from the question as typed, through the console's own retriever: no model call,
      and the numbers are document hit-rate, page hit-rate, mean reciprocal rank and nDCG@10.
      The first three say whether the right passage came back and where the document was found;
      the last is the only one that says anything about the order of everything else that came
      with it, which is what a reordering changes. A ranking is read against the best order the
      passages it returned could have been in, because a question names one document and at most
      one page and a set has no way to know how many other passages of it exist; the consequence
      is written where it shows, in `app.evals.ndcg_at`. The run asks the search for ten
      passages so that there are ten to read, and the first `RETRIEVAL_K` of those are what the
      other numbers were always computed over. A whole-document scope is handed over in reading
      order and reports no nDCG — there is no ranking to read. `--answers`
      writes an answer to each question from the passages the search returned, using the
      console's own prompt, and checks it against the text the set expected; `--judge` adds a
      model that reads each answer against its context for the faithfulness number. The set is
      a file of private questions about private documents, so it lives in `data/evals/`.
      Questions about the real library are almost all at rank 1 — it is one book about one
      subject, and the search puts the right page first for nearly anything asked about it —
      so a second set was written against a generated corpus built to be hard in the one place
      this embedding model is weak: ten subjects with four near-identical documents each,
      differing only in a condition stated in the text, and questions that describe that
      condition in everyday words rather than in the document's own. It is a local instrument
      like the real set, not part of the repository, and it is what made the reranker
      measurable at all. The baseline over it, at the 90 questions it held when this was
      written, was 86/90 documents, 86/90 pages, MRR 0.82, nDCG@10 0.83, identical over five
      runs and with 24 of the 90 questions off rank 1 — a baseline a retrieval change can be
      measured against, which the real library cannot give. The set has grown twice since the
      first measurement of it, so each item below carries the size it was measured at; at the
      107 questions it holds now, 97 of which name a document, the baseline is 86/97 documents,
      MRR 0.76, nDCG@10 0.79.
      Its difficulty is concentrated rather than spread, and that is worth knowing before
      trusting it as a general instrument: of the forty questions that describe a condition,
      every one of the eight about the closed-garage condition is off rank 1, while the other
      thirty-two are off eight times between them. Extending it means finding a second
      condition this model separates as badly, and the margins between a document and its
      three siblings are where to look for one. A third family was added for hybrid search —
      forty questions citing an opaque practice number the document states in its heading —
      and it did not behave as the item below expected: see there.
- [ ] Tracing (Langfuse) to inspect prompts, tokens and latency per node.
- [x] Reranking with `bge-reranker-v2-m3` after retrieval — no index change required. The
      search is asked for `RERANK_CANDIDATES` (20) passages, a cross-encoder scores each
      against the question, and the best `RETRIEVAL_K` are what the answer is written from. It
      is a retriever wrapping a retriever, so nothing downstream — the graph, both commands,
      the web app, the eval harness — knows it is there, and nothing is re-indexed. Measured,
      and **off by default** — but the reason has changed, and that is the finding. On the real
      library the search already returns the right page first for every question there is, so
      the two runs were identical: 11/11 documents, 11/11 pages, MRR 1.00 with `RERANK=on` and
      with `RERANK=off`. Reranking can only reorder what the search found, and there was
      nothing to reorder. On the harder corpus described above it does what it is for: the
      plain search answers 47/50 documents, 47/50 pages, MRR 0.78, nDCG@10 0.77, and reranked
      48/50, 48/50, MRR 0.89, nDCG@10 0.85 — 17 of the 50 questions at a different rank, and
      the ten control questions, asked in the document's own words, at rank 1 in both. That
      pass was over the 50 questions the corpus then held; at the 107 it holds now the same
      comparison is 94/97 documents against 86/97, MRR 0.90 against 0.76, nDCG@10 0.89 against
      0.79. Five runs of the plain pass and three of the reranked one each came out identical
      line for line, so the corpus moves for a reason rather than from ties. It is not a free
      win: two documents the plain search had found fell out of the top five while
      three it had missed came in, so the hit rate barely moves while the ordering improves. A
      reordering cannot add what was never retrieved, which is what the item below is for. What
      it costs is measured too: about 7 seconds added per question on CPU with six cores busy,
      plus a 2 GB checkpoint on a fresh clone. `RERANK=on` is one variable, and
      `python -m scripts.eval` and `scripts.chat` both print which pass they ran.
- [ ] Hybrid search using the sparse weights BGE-M3 already produces, for codes, names and
      numbers where dense retrieval is weak. **The premise was measured on the corpus above
      and is only half true, which is the finding.** Forty questions were added, each citing
      an opaque practice number — `PR-66C0-908F`, a string that means nothing — and asking
      what that document says, on the reasoning that an embedding has no meaning to match
      against in such a string. Dense finds the document anyway: 32 of the 40 at rank 1,
      winning by 0.015 to 0.11 in cosine distance, so this model's dense representation
      carries the identity of a rare token and a code is not the blind spot the note assumed
      it was. What is left is the other eight, where the right document loses by 0.005 to
      0.032 — a near-tie rather than a miss, and only one of them falls outside the console's
      five. So the case for the sparse half rests on the ordering of a fifth of one family
      rather than on blindness: worth building against a real sparse arm to see what it does,
      not worth assuming, and no longer expected to be the largest number in the table.
- [ ] Relevance grading node with a conditional edge: rewrite the query and retry, or state
      that the answer is not in the documents. Seen while building the interface, and the
      reason this node is not optional: a question with no content of its own ("ciao")
      retrieves five arbitrary passages, the answer is written from them, and the next
      question is rewritten with that answer in the conversation — which is how a question
      asked of the whole library comes back having searched one document. The prompt of
      `contextualize` was changed to forbid naming a document in the query, which stops that
      particular road, and the answer is now told which documents were searched, which is what
      makes a question about the library answerable. Neither stops a passage being retrieved
      and used on no evidence.
- [x] Small-to-big retrieval: match on small chunks, hand the surrounding page to the model —
      **the page, and not the section this item named, which is the finding.** A section is not
      a unit the store holds: a chunk carries the name of the section it falls under and
      nothing more, so there is no way to ask for the rest of it. The page is what the store
      can be asked for, as one filter over the source and the page, and it is the unit the
      eval already counts its hits in. It is a retriever wrapping a retriever inside
      `build_retriever`, after the reranker, so the cross-encoder still scores the chunks and
      only the pages that survive it are fetched: `SMALL_TO_BIG=page`. Off by default, because
      the ranking is then read over pages instead of passages — the eval's context line says
      which of the two a report measured, and two reports that disagree there are not
      comparable. On the harder corpus, over the 97 questions that name a document: 91/97
      documents against 86/97, MRR 0.77 against 0.76, nDCG@10 0.83 against 0.79, and no
      question moved down. Five misses became hits at rank 4 or 5: with several chunks of one
      document counted once, more documents fit in the five passages the ranking is read over.
      What it costs is the size of the context, about 1,500 characters per page against 250
      per chunk. It also puts back the text the grading node was refusing on: for
      `atti-vandalici-franchigia` the five passages were five copies of the same "Limiti e
      franchigie" paragraph from five different documents, and the page now carries the
      paragraph stating which damage the cover includes. One graded pass over the set with this
      off and one with it on says the same thing: the grader accepts 94 of the 97 questions that
      name a document against 92, turns away 13 of the 107 first against 21, and accepts
      `atti-vandalici-franchigia` on the first material. The retry is asked for less often for
      the same reason, 13 questions against 21, and recovers 1 against 7.
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

- [ ] Highlight cited passages inside the page. The server half is done: a source row carries
      the ranges of the page the answer was written from, and `GET /api/documents/boxes` turns
      one into the rectangles to draw on the page. The page half needs a renderer — the
      browser's own PDF viewer can be jumped to a page but not drawn on — which is a decision
      of its own and is why the route sends no page size.
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

- [ ] A fully offline setup: local embedding server and local chat model. (The local vector
      store is done — `VECTOR_STORE=chroma`.)
- [ ] Multi-machine access.

## Non-goals

- Multi-user, authentication and permissions.
- Collaboration and sharing.
- Mobile.
- Becoming a general-purpose PDF reader: viewer features that do not feed retrieval,
  citations or notes are out of scope.
