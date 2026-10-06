// AIDX Assistant: chat page behaviour (plain JavaScript, no framework).
//
// Flow: user types -> POST /api/chat -> show reply + its sources.
// The server remembers the conversation via a session cookie, so this script
// only needs to send the newest message.

(function () {
  "use strict";

  const messagesEl = document.getElementById("messages");
  const form = document.getElementById("composer");
  const input = document.getElementById("input");
  const sendBtn = document.getElementById("send");
  const welcome = document.getElementById("welcome");

  // ---------- Safe rendering ----------

  // Escape HTML first so a reply can never inject markup or scripts,
  // then apply a tiny subset of Markdown: **bold**, `code`, links, bullets.
  function escapeHtml(text) {
    return text.replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  function inlineMarkdown(text) {
    return text
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener">$1</a>')
      .replace(/(^|[\s(])(https?:\/\/[^\s<)]+)/g,
        '$1<a href="$2" target="_blank" rel="noopener">$2</a>');
  }

  // Walk the lines: consecutive "- item" lines become a <ul>, other lines
  // become paragraphs (blank lines separate paragraphs).
  function renderMarkdown(text) {
    const bullet = /^\s*[-*•]\s+/;
    let html = "", para = [], items = [];
    const flushPara = () => { if (para.length) html += "<p>" + inlineMarkdown(para.join("<br>")) + "</p>"; para = []; };
    const flushList = () => { if (items.length) html += "<ul>" + items.join("") + "</ul>"; items = []; };

    escapeHtml(text.trim()).split("\n").forEach((line) => {
      if (bullet.test(line)) {
        flushPara();
        items.push("<li>" + inlineMarkdown(line.replace(bullet, "")) + "</li>");
      } else if (line.trim() === "") {
        flushPara(); flushList();
      } else {
        flushList();
        para.push(line);
      }
    });
    flushPara(); flushList();
    return html;
  }

  // ---------- Message elements ----------

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  // `info` (bot replies only) is the API response: sources, memory_used,
  // learned and exchange_id.
  function addMessage(role, text, info) {
    if (welcome) welcome.hidden = true;
    info = info || {};

    const msg = document.createElement("div");
    msg.className = "msg " + role;

    // "Noted: ..." when the assistant learned something from the last message.
    (info.learned || []).forEach((item) => {
      const note = document.createElement("div");
      note.className = "learned-note";
      note.textContent = `Noted for next time: ${item.rule_text}`;
      const manage = document.createElement("button");
      manage.type = "button";
      manage.className = "link-btn";
      manage.textContent = "Review";
      manage.addEventListener("click", () => window.AIDX.openAccount && window.AIDX.openAccount());
      // One click to take it back: the user stays in control of what is remembered.
      const undo = document.createElement("button");
      undo.type = "button";
      undo.className = "link-btn";
      undo.textContent = "Undo";
      undo.addEventListener("click", async () => {
        const res = await fetch(`/api/memory/${item.id}`, { method: "DELETE" }).catch(() => null);
        if (res && res.ok) {
          note.textContent = "Okay, I won't remember that.";
          if (window.AIDX.reloadMemory) window.AIDX.reloadMemory();
        }
      });
      note.append(" ", manage, " · ", undo);
      msg.appendChild(note);
    });

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (role === "user") bubble.textContent = text;   // user text: never parsed
    else bubble.innerHTML = renderMarkdown(text);
    msg.appendChild(bubble);

    // Show what the answer was based on (explainability): lab documents and,
    // for signed-in users, which of their personal memory items were used.
    const sources = info.sources || [];
    const memoryUsed = info.memory_used || [];
    if (sources.length || memoryUsed.length) {
      const details = document.createElement("details");
      details.className = "sources";
      const summary = document.createElement("summary");
      const parts = [];
      if (sources.length) parts.push(`Sources (${sources.length})`);
      if (memoryUsed.length) parts.push(`Your memory (${memoryUsed.length})`);
      summary.textContent = parts.join(" · ");
      details.appendChild(summary);

      const addList = (title, rows) => {
        if (!rows.length) return;
        const h = document.createElement("h4");
        h.textContent = title;
        const list = document.createElement("ul");
        rows.forEach((text) => {
          const li = document.createElement("li");
          li.textContent = text;   // textContent: never interpreted as HTML
          list.appendChild(li);
        });
        details.append(h, list);
      };
      addList("Lab documents", sources.map((s) => `${s.heading} (${s.source})`));
      addList("Your memory", memoryUsed.map((m) => m.rule_text));
      msg.appendChild(details);
    }

    if (info.exchange_id) msg.appendChild(feedbackButtons(info.exchange_id));

    messagesEl.appendChild(msg);
    scrollToBottom();
    return msg;
  }

  // Thumbs up/down: tells the memory system whether this reply helped.
  function feedbackButtons(exchangeId) {
    const box = document.createElement("div");
    box.className = "feedback";
    const make = (rating, label, path) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "thumb";
      b.setAttribute("aria-label", label);
      b.title = label;
      b.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="${path}"/></svg>`;
      b.addEventListener("click", async () => {
        box.querySelectorAll("button").forEach((x) => (x.disabled = true));
        const res = await fetch("/api/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ exchange_id: exchangeId, rating }),
        }).catch(() => null);
        if (res && res.ok) {
          b.classList.add("chosen");
          b.setAttribute("aria-pressed", "true");
        } else {
          box.querySelectorAll("button").forEach((x) => (x.disabled = false));
        }
      });
      return b;
    };
    box.append(
      make(1, "Helpful", "M7 10v11H3V10h4zm0 0l4-8a3 3 0 013 3v4h5.5a2 2 0 012 2.3l-1.4 8A2 2 0 0118.1 21H7"),
      make(-1, "Not helpful", "M17 14V3h4v11h-4zm0 0l-4 8a3 3 0 01-3-3v-4H4.5a2 2 0 01-2-2.3l1.4-8A2 2 0 015.9 3H17"),
    );
    return box;
  }

  function showTyping() {
    const msg = document.createElement("div");
    msg.className = "msg bot";
    msg.setAttribute("aria-label", "Assistant is typing");
    msg.innerHTML = '<div class="bubble typing"><span></span><span></span><span></span></div>';
    messagesEl.appendChild(msg);
    scrollToBottom();
    return msg;
  }

  // ---------- Sending ----------

  async function send(text) {
    text = text.trim();
    if (!text) return;

    addMessage("user", text);
    input.value = "";
    autoGrow();
    sendBtn.disabled = true;
    const typing = showTyping();

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      const data = await res.json().catch(() => ({}));
      typing.remove();
      if (!res.ok) {
        const detail = typeof data.detail === "string" ? data.detail : "Sorry, something went wrong.";
        addMessage("bot error", detail);
      } else {
        addMessage("bot", data.reply, data);
      }
    } catch (err) {
      typing.remove();
      addMessage("bot error", "I couldn't reach the server. Please check your connection and try again.");
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    send(input.value);
  });

  // Enter sends; Shift+Enter adds a new line.
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  // Grow the textarea with its content (up to the CSS max-height).
  function autoGrow() {
    input.style.height = "auto";
    input.style.height = input.scrollHeight + "px";
  }
  input.addEventListener("input", autoGrow);

  // Suggested questions
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => send(chip.textContent));
  });

  // ---------- Header buttons ----------

  // Clear the screen (the server forgets the conversation separately).
  function clearConversation() {
    messagesEl.querySelectorAll(".msg").forEach((m) => m.remove());
    if (welcome) welcome.hidden = false;
  }

  document.getElementById("new-chat").addEventListener("click", async () => {
    await fetch("/api/reset", { method: "POST" }).catch(() => {});
    clearConversation();
    input.focus();
  });

  // Shared with account.js (signing in/out starts a fresh conversation).
  window.AIDX = { clearConversation };

  const themeBtn = document.getElementById("theme-toggle");
  function updateThemeLabel() {
    const dark = document.documentElement.dataset.theme === "dark";
    themeBtn.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
  }
  themeBtn.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("aidx-theme", next); } catch (e) { /* storage blocked: fine */ }
    updateThemeLabel();
  });
  updateThemeLabel();
})();
