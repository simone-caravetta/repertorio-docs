/* The catalog on the left, a conversation on the right.
 *
 * The scope is the server's: the select says what was picked, and the label
 * beside it is what the server resolved that to, character for character the
 * string the console prints. Nothing here works out what a category contains.
 */

const STORAGE_KEY = "repertorio-docs";

/* The query a search was run on, which is not always the question as it was
 * typed: the graph rewrites it with the conversation in hand first. The line is
 * here so that a rewrite is shown, rather than changing the answer quietly. */
const SEARCHED = "searched: ";

const NO_CATEGORY = "no category";

// Whole library, a category, or one document. Sent to the server as it is.
let scope = {};

/* A conversation per scope, kept by the scope it is about. A question is
 * answered with the history of its own scope and of no other, because the graph
 * rewrites each question with the conversation in hand: the history of a
 * question about one document is not context for a question about another, and
 * given it the rewrite names the document that was being discussed. Picking a
 * document again finds the questions already asked about it. */
let threads = {};

/* The documents ticked in the catalog. They are not the scope: the scope is what
 * a question is asked of, and these are the group being built for the next one.
 * A tick cannot re-scope as it is made, or the first document ticked would take
 * the scope and the second would replace it, and two could never be chosen. */
const selected = new Set();

/* A scope as one string, to look its conversation up by. The same documents
 * picked twice are the same conversation, and the whole library is the empty
 * one — which no category can be, a category name being something. A group
 * carries its paths as JSON rather than joined by a separator: a path is a file
 * name, and may hold anything a file name can. */
function scopeKey(of) {
  if (of.document != null) return `document:${of.document}`;
  if (of.category != null) return `category:${of.category}`;
  const chosen = of.documents || [];
  if (chosen.length) return `documents:${JSON.stringify([...chosen].sort())}`;
  return "";
}

/* The paths a group's key or option carries, or null for a key from another
 * version of this page, which is skipped rather than thrown over. */
function pathsOf(value) {
  try {
    const paths = JSON.parse(value.slice("documents:".length));
    return Array.isArray(paths) ? paths : null;
  } catch {
    return null;
  }
}

function currentThread() {
  return threads[scopeKey(scope)] || null;
}

/* What the reader has put away in the catalog: a category by its name, a
 * document by its path. Kept with the scope, so the panel opens the way it was
 * left instead of closing again on every visit. */
let collapsed = new Set();

const elements = {
  scope: document.getElementById("scope"),
  scopeLabel: document.getElementById("scope-label"),
  catalog: document.getElementById("catalog"),
  messages: document.getElementById("messages"),
  composer: document.getElementById("composer"),
  question: document.getElementById("question"),
  send: document.getElementById("send"),
  newThread: document.getElementById("new-thread"),
  selection: document.getElementById("selection"),
  selectionCount: document.getElementById("selection-count"),
  askThese: document.getElementById("ask-these"),
  clearSelection: document.getElementById("clear-selection"),
};

/* ---------------------------------------------------------------- storage */

function remember() {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({ scope, threads, collapsed: [...collapsed] })
  );
}

function recall() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (saved && typeof saved === "object") {
      scope = saved.scope || {};
      threads = saved.threads || {};
      collapsed = new Set(saved.collapsed || []);
      if (!saved.threads && saved.threadId) {
        // Written before conversations were kept one to a scope, when the one
        // thread it held was the thread of the scope it was on.
        threads[scopeKey(scope)] = saved.threadId;
      }
    }
  } catch {
    // A value from another version of this page: start over rather than fail.
  }
}

/* ---------------------------------------------------------------- catalog */

/* The catalog as it was last read: the panel is drawn from it, and so are the
 * selector's options, which a group joins as soon as it exists. */
let catalogView = { empty: true, categories: [], uncategorized: [] };

/* What the panel is indexed by when it is drawn. `below` is the indexed
 * documents at or below each category — the set the server reads that category
 * as, so a category's tick and the same category picked from the selector are
 * the same documents. The two maps after it are the boxes themselves, by path
 * and by category name, so that one tick can move the others. */
const below = new Map();
const ticks = new Map();
const headings = new Map();
const titles = new Map();

/* What the reader has put away, and how a heading and its contents follow it.
 * `controls` are the elements that say whether it is open: on a category that
 * is the heading itself, on a document the caret alone, so the two carry their
 * own state rather than sharing one shape. */
function isOpen(key) {
  return !collapsed.has(key);
}

function setOpen(key, open) {
  if (open) collapsed.delete(key);
  else collapsed.add(key);
  remember();
}

function show(key, body, controls) {
  const open = isOpen(key);
  body.hidden = !open;
  for (const control of controls) {
    control.setAttribute("aria-expanded", String(open));
  }
}

/* A category and the documents under it. The heading opens and closes them,
 * and the whole heading is the control, because nothing else on it is
 * clickable — a document's row is, so there the caret is a control of its own. */
function renderBranch(branch, depth, parent = elements.catalog) {
  const body = document.createElement("div");
  body.className = "branch";

  const key = `category:${branch.name}`;
  const caret = span("", "caret");
  caret.setAttribute("aria-hidden", "true");

  const heading = document.createElement("button");
  heading.type = "button";
  heading.className = "category";
  heading.append(caret, span(branch.name || NO_CATEGORY, "name"));
  heading.addEventListener("click", () => {
    setOpen(key, !isOpen(key));
    show(key, body, [heading]);
  });
  show(key, body, [heading]);

  /* The tick cannot live in the heading: it is a button, and a button holds no
   * other control. The indent is on the row, so that the tick is indented with
   * the heading rather than beside it. */
  const row = document.createElement("div");
  row.className = "row";
  row.style.paddingLeft = `${depth}rem`;
  row.append(tickForCategory(branch.name) || span("", "tick blank"), heading);

  parent.append(row, body);

  for (const record of branch.documents) {
    renderDocument(record, depth + 1, body);
  }
  for (const child of branch.children) {
    renderBranch(child, depth + 1, body);
  }
}

/* The indexed documents at or below every category, and the title of every
 * document, taken from the view the panel is drawn from. Done before the panel
 * rather than while it is drawn: a category is drawn before the documents it
 * covers, and its tick needs them already.
 *
 * The set is the one the server resolves that category to: `sources_in_category`
 * takes the indexed documents at or below it, so a category's tick and the same
 * category picked from the selector ask about the same documents. A category
 * with nothing indexed under it gets no tick at all — a group of nothing is not
 * a scope, and picking it from the selector is already an error. */
function indexBranch(branch) {
  const paths = [];
  for (const record of branch.documents) {
    titles.set(record.path, record.title);
    if (record.status === "indexed") paths.push(record.path);
  }
  for (const child of branch.children) paths.push(...indexBranch(child));
  below.set(branch.name, paths);
  return paths;
}

/* A category's tick, or nothing when there is nothing under it to take. It
 * carries no state of its own: what is ticked is `selected`, and the box is
 * drawn from it. */
function tickForCategory(name) {
  const paths = below.get(name) || [];
  if (!paths.length) return null;

  const box = document.createElement("input");
  box.type = "checkbox";
  box.className = "tick";
  box.title = `Every indexed document in ${name || NO_CATEGORY}`;
  box.setAttribute("aria-label", box.title);
  box.addEventListener("change", () => {
    for (const path of paths) {
      if (box.checked) selected.add(path);
      else selected.delete(path);
    }
    refreshTicks();
  });
  headings.set(name, box);
  return box;
}

/* A document's caret: the row asks about the document, so the caret is what
 * opens and closes what the catalog says about it. */
function caretFor(key, body) {
  const caret = document.createElement("button");
  caret.type = "button";
  caret.className = "caret";
  caret.setAttribute("aria-label", "Description");
  caret.addEventListener("click", () => {
    setOpen(key, !isOpen(key));
    show(key, body, [caret]);
  });
  show(key, body, [caret]);
  return caret;
}

/* `record`, not `document`: a parameter of that name shadows the browser's own
 * `document` for the whole function, and the first `document.createElement`
 * inside it is then a call on the JSON object rather than on the page. */
function renderDocument(record, depth, parent = elements.catalog) {
  const group = document.createElement("div");
  group.className = "record";
  group.style.paddingLeft = `${depth}rem`;

  const key = `document:${record.path}`;

  /* What the catalog says the document is about, written by `scripts.describe`
   * and absent until it has been run for this document. One with nothing said
   * about it says nothing: the line is here rather than a placeholder, and
   * clicking it asks about the document the same way clicking the row does. */
  const described = record.description
    ? line(record.description, "description")
    : null;
  if (described) {
    described.addEventListener("click", () => chooseDocument(record.path));
  }

  const button = document.createElement("button");
  button.type = "button";
  button.className = "document";
  button.title = record.path;
  button.append(span(record.title, "title"));
  button.append(span(record.status, `status ${record.status}`));
  if (record.pages) {
    button.append(span(`${record.pages} p.`, "pages"));
  }
  button.addEventListener("click", () => chooseDocument(record.path));

  const row = document.createElement("div");
  row.className = "row";
  // A blank in the tick's place, and another in the caret's when there is
  // nothing to open, so that the ticks and the titles each stay in one column
  // whether a document can be chosen or described, or neither.
  row.append(tickForDocument(record) || span("", "tick blank"));
  row.append(described ? caretFor(key, described) : span("", "caret blank"), button);

  group.append(row);
  if (described) group.append(described);
  parent.append(group);
}

/* A document's tick, and only for one that has vectors: `resolve_scope` looks
 * every path of a group up and refuses the whole group if one of them is not
 * indexed, so a document that is not indexed cannot be in one. The status beside
 * the row is already saying why there is nothing here to tick. */
function tickForDocument(record) {
  if (record.status !== "indexed") return null;

  const box = document.createElement("input");
  box.type = "checkbox";
  box.className = "tick";
  box.setAttribute("aria-label", `Include ${record.title}`);
  box.addEventListener("change", () => {
    if (box.checked) selected.add(record.path);
    else selected.delete(record.path);
    refreshTicks();
  });
  ticks.set(record.path, box);
  return box;
}

/* --------------------------------------------------------------- selection */

/* A category is ticked when everything under it is, and half-ticked when only
 * some of it is. Read off `selected` rather than off the boxes below it, so the
 * two cannot drift: the boxes are the drawn form of one set, not four copies of
 * it. */
function refreshTicks() {
  for (const [path, box] of ticks) {
    box.checked = selected.has(path);
  }
  for (const [name, box] of headings) {
    const paths = below.get(name) || [];
    const taken = paths.filter((path) => selected.has(path)).length;
    box.checked = taken > 0 && taken === paths.length;
    box.indeterminate = taken > 0 && taken < paths.length;
  }

  elements.selection.hidden = selected.size === 0;
  elements.selectionCount.textContent =
    selected.size === 1 ? "1 document" : `${selected.size} documents`;
}

/* The ticks are the group being built, or the group in force, and never a draft
 * left over from the one before. So they follow the scope: choosing a group is
 * choosing its documents back, and choosing anything else leaves them empty. */
function tickTheScope() {
  selected.clear();
  for (const path of scope.documents || []) selected.add(path);
  refreshTicks();
}

/* The ticks become the scope. Sorted, so that the same documents ticked in
 * another order are the same group and find the same conversation. */
function askAboutThese() {
  chooseScope({ documents: [...selected].sort() });
}

/* ------------------------------------------------------------------ scope */

async function chooseScope(next) {
  scope = next;
  remember();

  // A group is a scope the selector offers, so the options are rebuilt before
  // the selector is pointed at the one now on.
  fillScopeSelect();
  showScopeInSelect();
  tickTheScope();

  await showConversation();
  await describeScope();
}

/* The conversation of the scope that is now on: the one already had about these
 * documents, read back from the server, or an empty one that says so. Moving to
 * another document and coming back finds the questions asked about the first,
 * which is what keeping the thread by its scope buys. */
async function showConversation() {
  elements.messages.replaceChildren();

  const thread = currentThread();
  elements.newThread.disabled = thread === null;
  if (thread) {
    await loadThread(thread);
  } else if (!isEmpty(scope)) {
    elements.messages.append(line("— new conversation —", "notice"));
  }
}

/* Forget the conversation of this scope and leave the thread where it is: the
 * next question asks for a new one. Every question is rewritten with the ones
 * before it, so a history that is kept is one there has to be a way out of. */
function newConversation() {
  delete threads[scopeKey(scope)];
  remember();
  showConversation();
}

async function describeScope() {
  const query = new URLSearchParams();
  if (scope.category !== undefined) query.set("category", scope.category);
  if (scope.document !== undefined) query.set("document", scope.document);
  for (const path of scope.documents || []) query.append("documents", path);

  elements.scopeLabel.textContent = "…";

  const response = await fetch(`/api/scope?${query}`);
  if (!response.ok) {
    elements.scopeLabel.textContent = (await response.json()).detail;
    elements.scopeLabel.classList.add("refused");
    return;
  }

  const resolved = await response.json();
  elements.scopeLabel.textContent = resolved.label;
  elements.scopeLabel.classList.remove("refused");
}

function chooseDocument(path) {
  chooseScope({ document: path });
}

/* The select offers every scope there is: the whole library, which is no
 * restriction at all, then the groups of documents and the single documents a
 * conversation has been had about, then every category. Rebuilt rather than
 * filled once — starting a conversation is what puts a scope among them — so the
 * listener is attached outside, or every rebuild would add another. */
function fillScopeSelect() {
  elements.scope.replaceChildren(option("", "the whole library"));
  for (const [value, paths] of groups()) {
    elements.scope.append(option(value, groupLabel(paths), paths.join("\n")));
  }
  for (const path of offeredDocuments()) {
    elements.scope.append(option(documentValue(path), documentLabel(path), path));
  }
  for (const branch of catalogView.categories || []) {
    addCategoryOptions(branch);
  }
}

/* The groups the select should offer, as the value its option carries and the
 * paths in it. A group is one when a conversation has been had about it — read
 * off `threads`, whose `documents:` keys are exactly the groups — or when it is
 * the group in force, which is one the reader has just built and not yet asked
 * anything of. There is no list of groups to keep in step with this, and a
 * group's option lives exactly as long as its conversation, which is the rule
 * every other scope already follows. */
function groups() {
  const known = new Map();
  for (const key of Object.keys(threads)) {
    const paths = key.startsWith("documents:") ? pathsOf(key) : null;
    if (paths) known.set(key, paths);
  }
  if ((scope.documents || []).length) {
    known.set(scopeKey(scope), [...scope.documents].sort());
  }
  return known;
}

/* The documents the select should offer: the ones a conversation has been had
 * about, and the one in force, which is one just picked from the catalog and not
 * yet asked anything of. The same rule as a group of them — what you have been
 * on is what you can go back to — and the reason a document picked from the
 * catalog stays an option when the scope moves elsewhere. */
function offeredDocuments() {
  const paths = Object.keys(threads)
    .filter((key) => key.startsWith("document:"))
    .map((key) => key.slice("document:".length));
  if (scope.document != null && !paths.includes(scope.document)) {
    paths.push(scope.document);
  }
  return paths;
}

function documentValue(path) {
  return `document:${path}`;
}

/* A document as the selector names it: the title the panel shows, and the path
 * in the tooltip, which is what identifies it. A document the catalog no longer
 * holds has no title to give, and is named by its path. */
function documentLabel(path) {
  return titles.get(path) || path;
}

/* A group as the reader knows it: the first two titles, and how many more. The
 * server calls a group "3 documents", which is the same string for every group
 * of three. */
function groupLabel(paths) {
  const names = paths.map((path) => documentLabel(path));
  if (names.length <= 2) return names.join(", ");
  return `${names.slice(0, 2).join(", ")} +${names.length - 2}`;
}

/* Point the select at the scope in force. Every scope it can be on is among the
 * options by now: a group by `groups`, a document by `offeredDocuments`, and a
 * category and the whole library always. Setting the value from here does not
 * fire `change`, so anything arriving there was chosen by the reader. */
function showScopeInSelect() {
  if (scope.document != null) {
    elements.scope.value = documentValue(scope.document);
  } else if (scope.category !== undefined) {
    elements.scope.value = `category:${scope.category}`;
  } else {
    const paths = scope.documents || [];
    elements.scope.value = paths.length ? scopeKey(scope) : "";
  }
}

function addCategoryOptions(branch) {
  // The full name, not the leaf: it is what the server takes.
  const depth = branch.name.split("/").length;
  elements.scope.append(
    option(`category:${branch.name}`, "  ".repeat(depth - 1) + branch.name)
  );
  for (const child of branch.children) {
    addCategoryOptions(child);
  }
}

/* ------------------------------------------------------------------- chat */

async function ask(question) {
  const bubble = addTurn("assistant");

  /* The scope this question is asked of, held here rather than read again when
   * the answer arrives: the thread it is filed under is the thread of the scope
   * it was asked on, even if the reader has moved to another one by then. */
  const asked = scope;

  // What was searched for and what came back, under the answer as it is written.
  // Both are built empty and filled when their event lands, so that the query is
  // above the sources whatever order they arrive in.
  const query = line("", "query");
  query.hidden = true;
  const sources = document.createElement("ul");
  sources.className = "sources";
  sources.hidden = true;

  const details = document.createElement("div");
  details.className = "details";
  details.append(query, sources);
  bubble.after(details);

  elements.send.disabled = true;
  elements.question.value = "";

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, thread_id: currentThread(), ...asked }),
    });

    if (!response.ok) {
      const failure = await response.json();
      bubble.textContent = failure.detail || "The question was refused.";
      return;
    }

    await readStream(response, (name, data) => {
      if (name === "thread") {
        threads[scopeKey(asked)] = data.thread_id;
        // Asked again rather than set: the answer may have started on one scope
        // and still be arriving after the reader has moved to another.
        elements.newThread.disabled = currentThread() === null;
        remember();
      } else if (name === "query") {
        query.textContent = SEARCHED + data.query;
        query.hidden = false;
      } else if (name === "sources") {
        for (const source of data.sources) {
          sources.append(sourceItem(source));
        }
        sources.hidden = false;
      } else if (name === "token") {
        bubble.textContent += data.text;
      } else if (name === "done") {
        bubble.textContent = data.answer;
      } else if (name === "error") {
        bubble.textContent = `Error: ${data.message}`;
      }
    });
  } catch (error) {
    bubble.textContent = `Error: ${error.message}`;
  } finally {
    elements.send.disabled = false;
    elements.question.focus();
  }
}

/* Read a `text/event-stream` body: events are separated by a blank line, and
 * each carries an `event:` name and one `data:` line of JSON. */
async function readStream(response, handle) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    let end;
    while ((end = buffer.indexOf("\n\n")) !== -1) {
      const { name, data } = parseEvent(buffer.slice(0, end));
      buffer = buffer.slice(end + 2);
      if (name && data) handle(name, data);
    }
  }
}

function parseEvent(raw) {
  let name = null;
  const data = [];

  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) name = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }

  if (!data.length) return { name, data: null };
  return { name, data: JSON.parse(data.join("\n")) };
}

async function loadThread(id) {
  const response = await fetch(`/api/threads/${encodeURIComponent(id)}`);
  if (!response.ok) return;

  const thread = await response.json();
  for (const message of thread.messages) {
    addTurn(message.role === "human" ? "you" : "assistant", message.content);
  }

  // The query and the sources of the last turn, which is all the state holds:
  // the ones before were shown when they were asked. A conversation read back
  // says the same thing about itself as a live one.
  if (!thread.query && !thread.sources.length) return;

  const details = document.createElement("div");
  details.className = "details";
  if (thread.query) {
    details.append(line(SEARCHED + thread.query, "query"));
  }
  if (thread.sources.length) {
    const sources = document.createElement("ul");
    sources.className = "sources";
    for (const source of thread.sources) {
      sources.append(sourceItem(source));
    }
    details.append(sources);
  }
  elements.messages.append(details);
}

/* ----------------------------------------------------------------- pieces */

function addTurn(role, text = "") {
  const turn = document.createElement("div");
  turn.className = `turn ${role}`;
  turn.textContent = text;
  elements.messages.append(turn);
  elements.messages.scrollTop = elements.messages.scrollHeight;
  return turn;
}

function line(text, className) {
  const element = document.createElement("p");
  element.className = className;
  element.textContent = text;
  return element;
}

/* One passage of the answer's sources: the file, and the page when it has one.
 * Page 0 is a page, which is why this asks whether the page is there rather
 * than whether it is true. */
function sourceItem(source) {
  const item = document.createElement("li");
  item.textContent =
    source.page === null || source.page === undefined
      ? source.source
      : `${source.source} — p. ${source.page}`;
  return item;
}

function span(text, className) {
  const element = document.createElement("span");
  element.className = className;
  element.textContent = text;
  return element;
}

function option(value, text, title = null) {
  const element = document.createElement("option");
  element.value = value;
  element.textContent = text;
  if (title) element.title = title;
  return element;
}

function isEmpty(next) {
  return (
    next.category === undefined &&
    next.document === undefined &&
    !(next.documents || []).length
  );
}

/* ------------------------------------------------------------------ start */

elements.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = elements.question.value.trim();
  if (!question) return;
  addTurn("you", question);
  ask(question);
});

elements.newThread.addEventListener("click", () => newConversation());

elements.askThese.addEventListener("click", () => askAboutThese());

elements.clearSelection.addEventListener("click", () => {
  selected.clear();
  refreshTicks();
});

/* Every option in the select is a scope, a document's included. */
elements.scope.addEventListener("change", () => {
  const value = elements.scope.value;
  if (value.startsWith("document:")) {
    chooseScope({ document: value.slice("document:".length) });
    return;
  }
  if (value.startsWith("category:")) {
    chooseScope({ category: value.slice("category:".length) });
    return;
  }
  const paths = value.startsWith("documents:") ? pathsOf(value) : null;
  chooseScope(paths ? { documents: paths } : {});
});

// The question box grows with the question, up to a point.
elements.question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.composer.requestSubmit();
  }
});

async function start() {
  recall();

  catalogView = await fetch("/api/catalog").then((one) => one.json());
  renderCatalog();
  fillScopeSelect();
  showScopeInSelect();
  tickTheScope();

  await describeScope();
  await showConversation();

  elements.question.focus();
}

function renderCatalog() {
  below.clear();
  ticks.clear();
  headings.clear();
  titles.clear();
  elements.catalog.replaceChildren();

  if (catalogView.empty) {
    elements.catalog.append(
      line("No catalog here yet. Run a sync first.", "empty")
    );
    return;
  }

  /* The documents filed under no category are a group like any other, and an
   * empty name is what a category cannot be — which is what makes it theirs,
   * and the key the panel remembers them by. */
  const branches = [...catalogView.categories];
  if (catalogView.uncategorized.length) {
    branches.push({
      name: "",
      documents: catalogView.uncategorized,
      children: [],
    });
  }

  // Every branch indexed before any of them is drawn: a category's tick is
  // built from what is under it, and it is drawn before that.
  for (const branch of branches) indexBranch(branch);
  for (const branch of branches) renderBranch(branch, 0, elements.catalog);
}

start();
