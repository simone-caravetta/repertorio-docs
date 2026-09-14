/* The catalog on the left, a conversation on the right.
 *
 * The scope is the server's: the select says what was picked, and the label
 * beside it is what the server resolved that to, character for character the
 * string the console prints. Nothing here works out what a category contains.
 */

const STORAGE_KEY = "repertorio-docs";

// Whole library, a category, or one document. Sent to the server as it is.
let scope = {};
let threadId = null;

const elements = {
  scope: document.getElementById("scope"),
  scopeLabel: document.getElementById("scope-label"),
  catalog: document.getElementById("catalog"),
  messages: document.getElementById("messages"),
  composer: document.getElementById("composer"),
  question: document.getElementById("question"),
  send: document.getElementById("send"),
};

/* ---------------------------------------------------------------- storage */

function remember() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify({ scope, threadId }));
}

function recall() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (saved && typeof saved === "object") {
      scope = saved.scope || {};
      threadId = saved.threadId || null;
    }
  } catch {
    // A value from another version of this page: start over rather than fail.
  }
}

/* ---------------------------------------------------------------- catalog */

function renderBranch(branch, depth) {
  elements.catalog.append(heading(branch.name, depth));
  for (const record of branch.documents) {
    renderDocument(record, depth + 1);
  }
  for (const child of branch.children) {
    renderBranch(child, depth + 1);
  }
}

/* `record`, not `document`: a parameter of that name shadows the browser's own
 * `document` for the whole function, and the first `document.createElement`
 * inside it is then a call on the JSON object rather than on the page. */
function renderDocument(record, depth) {
  const row = document.createElement("button");
  row.type = "button";
  row.className = "document";
  row.style.paddingLeft = `${depth}rem`;
  row.title = record.path;

  row.append(span(record.title, "title"));
  row.append(span(record.status, `status ${record.status}`));
  if (record.pages) {
    row.append(span(`${record.pages} p.`, "pages"));
  }

  row.addEventListener("click", () => chooseDocument(record.path));
  elements.catalog.append(row);
}

/* ------------------------------------------------------------------ scope */

async function chooseScope(next) {
  // A new conversation, and it says so: the history of the last one was a
  // history of questions about other documents.
  scope = next;
  threadId = null;
  remember();

  elements.messages.replaceChildren();
  if (!isEmpty(next)) {
    elements.messages.append(line("— new conversation —", "notice"));
  }

  await describeScope();
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
  selectDocument(path);
  chooseScope({ document: path });
}

/* A document is not one of the options the select was built with: it is picked
 * from the catalog, so its option is added the first time it is picked. */
function selectDocument(path) {
  const value = `document:${path}`;
  const known = [...elements.scope.options].some((one) => one.value === value);
  if (!known) elements.scope.append(option(value, path));
  elements.scope.value = value;
}

function fillScopeSelect(view) {
  elements.scope.replaceChildren(option("", "the whole library"));
  for (const branch of view.categories || []) {
    addCategoryOptions(branch);
  }

  elements.scope.addEventListener("change", () => {
    const value = elements.scope.value;
    if (value.startsWith("document:")) return; // picked from the catalog
    if (value.startsWith("category:")) {
      chooseScope({ category: value.slice("category:".length) });
      return;
    }
    chooseScope({});
  });
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
  const sources = document.createElement("ul");
  sources.className = "sources";
  sources.hidden = true;

  elements.send.disabled = true;
  elements.question.value = "";

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, thread_id: threadId, ...scope }),
    });

    if (!response.ok) {
      const failure = await response.json();
      bubble.textContent = failure.detail || "The question was refused.";
      return;
    }

    await readStream(response, (name, data) => {
      if (name === "thread") {
        threadId = data.thread_id;
        remember();
      } else if (name === "sources") {
        for (const source of data.sources) {
          const item = document.createElement("li");
          item.textContent =
            source.page === null || source.page === undefined
              ? source.source
              : `${source.source} — p. ${source.page}`;
          sources.append(item);
        }
        bubble.after(sources);
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

async function loadThread() {
  const response = await fetch(`/api/threads/${encodeURIComponent(threadId)}`);
  if (!response.ok) return;

  const thread = await response.json();
  for (const message of thread.messages) {
    addTurn(message.role === "human" ? "you" : "assistant", message.content);
  }

  if (thread.sources.length) {
    const sources = document.createElement("ul");
    sources.className = "sources";
    for (const source of thread.sources) {
      const item = document.createElement("li");
      item.textContent =
        source.page === null || source.page === undefined
          ? source.source
          : `${source.source} — p. ${source.page}`;
      sources.append(item);
    }
    elements.messages.append(sources);
  }
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

function heading(text, depth = 0) {
  const element = document.createElement("p");
  element.className = "category";
  element.style.paddingLeft = `${depth}rem`;
  element.textContent = text;
  return element;
}

function line(text, className) {
  const element = document.createElement("p");
  element.className = className;
  element.textContent = text;
  return element;
}

function span(text, className) {
  const element = document.createElement("span");
  element.className = className;
  element.textContent = text;
  return element;
}

function option(value, text) {
  const element = document.createElement("option");
  element.value = value;
  element.textContent = text;
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

// The question box grows with the question, up to a point.
elements.question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.composer.requestSubmit();
  }
});

async function start() {
  recall();

  const catalog = await fetch("/api/catalog").then((one) => one.json());
  fillScopeSelect(catalog);
  renderCatalog(catalog);

  if (scope.document) {
    selectDocument(scope.document);
  } else if (scope.category !== undefined) {
    elements.scope.value = `category:${scope.category}`;
  }

  await describeScope();
  if (threadId) await loadThread();

  elements.question.focus();
}

function renderCatalog(view) {
  elements.catalog.replaceChildren();

  if (view.empty) {
    elements.catalog.append(
      line("No catalog here yet. Run a sync first.", "empty")
    );
    return;
  }

  for (const branch of view.categories) renderBranch(branch, 0);
  if (view.uncategorized.length) {
    elements.catalog.append(heading("no category"));
    for (const record of view.uncategorized) renderDocument(record, 1);
  }
}

start();
