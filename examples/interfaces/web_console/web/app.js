// Web console frontend — single file, no build step.
//
// Talks to the FastAPI server over fetch + Server-Sent Events.

const $ = (id) => document.getElementById(id);

const state = {
  worldId: null,
};

async function loadWorlds() {
  const res = await fetch("/api/worlds");
  const worlds = await res.json();
  const select = $("world");
  select.innerHTML = "";
  for (const w of worlds) {
    const opt = document.createElement("option");
    opt.value = w.id;
    opt.textContent = `${w.name} (${w.project_id})`;
    select.appendChild(opt);
  }
  state.worldId = worlds[0]?.id ?? null;
  select.addEventListener("change", () => {
    state.worldId = select.value;
    $("log").innerHTML = "";
    refreshDocs();
  });
  if (state.worldId) await refreshDocs();
}

async function refreshDocs() {
  if (!state.worldId) return;
  const res = await fetch(`/api/ls/${state.worldId}`);
  const docs = await res.json();
  const ul = $("docs");
  ul.innerHTML = "";
  for (const path of docs) {
    const li = document.createElement("li");
    li.textContent = path;
    li.addEventListener("click", () => openDoc(path));
    ul.appendChild(li);
  }
}

async function openDoc(path) {
  const res = await fetch(`/api/read/${state.worldId}?path=${encodeURIComponent(path)}`);
  if (!res.ok) return;
  const { content } = await res.json();
  $("docview-title").style.display = "block";
  $("docview-title").textContent = path;
  $("docview").textContent = content;
}

function appendLine(cls, label, text) {
  const div = document.createElement("div");
  div.className = `msg ${cls}`;
  if (label) {
    const span = document.createElement("span");
    span.className = "label";
    span.textContent = label;
    div.appendChild(span);
  }
  div.appendChild(document.createTextNode(text));
  $("log").appendChild(div);
  $("log").scrollTop = $("log").scrollHeight;
}

// Assistant messages are rendered as markdown so things like `code spans`,
// bullet lists, and code blocks display the way a chat UI is expected to.
// User input and action call/result lines stay plain text because they're
// structured content, not authored prose.
function appendAssistantMarkdown(text) {
  const div = document.createElement("div");
  div.className = "msg assistant";

  const label = document.createElement("span");
  label.className = "label";
  label.textContent = "agent";
  div.appendChild(label);

  const body = document.createElement("span");
  body.className = "md";
  body.innerHTML = renderMarkdown(text || "");
  div.appendChild(body);

  $("log").appendChild(div);
  $("log").scrollTop = $("log").scrollHeight;
}

function renderMarkdown(text) {
  if (typeof marked !== "undefined") {
    try {
      return marked.parse(text, { gfm: true, breaks: true });
    } catch (e) {
      console.warn("marked.parse failed; falling back to plain text", e);
    }
  }
  // Fallback: escape HTML, preserve newlines.
  const escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\n/g, "<br>");
  return escaped;
}

function renderEvent(evt) {
  const { kind, payload } = evt;
  if (kind === "message") {
    appendAssistantMarkdown(payload.content || "");
  } else if (kind === "action_called") {
    const args = formatArgs(payload.args || {});
    appendLine("action", "↪", `${payload.name}(${args})`);
  } else if (kind === "action_result") {
    if (payload.error) {
      appendLine("error", "✗", `${payload.name}: ${payload.error}`);
    } else {
      const short = (payload.result || "").slice(0, 200);
      appendLine("result", "✓", `${payload.name} → ${short}`);
    }
  } else if (kind === "error") {
    appendLine("error", "error", payload.error || "unknown");
  }
}

function formatArgs(args) {
  return Object.entries(args)
    .map(([k, v]) => `${k}=${JSON.stringify(v).slice(0, 40)}`)
    .join(", ");
}

async function sendMessage(text) {
  appendLine("user", "you", text);
  $("send").disabled = true;
  try {
    const res = await fetch(`/api/chat/${state.worldId}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok || !res.body) {
      appendLine("error", "error", `HTTP ${res.status}`);
      return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // SSE messages are separated by blank lines
      let idx;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const dataLine = chunk.split("\n").find((l) => l.startsWith("data:"));
        if (!dataLine) continue;
        try {
          renderEvent(JSON.parse(dataLine.slice(5).trim()));
        } catch (e) {
          console.error("bad SSE payload", e, dataLine);
        }
      }
    }
  } finally {
    $("send").disabled = false;
    await refreshDocs();
  }
}

$("chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("prompt").value.trim();
  if (!text || !state.worldId) return;
  $("prompt").value = "";
  sendMessage(text);
});

loadWorlds();
