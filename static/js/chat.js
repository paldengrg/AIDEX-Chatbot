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

  function addMessage(role, text, sources, memoryUsed) {
    if (welcome) welcome.hidden = true;

    const msg = document.createElement("div");
    msg.className = "msg " + role;

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (role === "user") bubble.textContent = text;   // user text: never parsed
    else bubble.innerHTML = renderMarkdown(text);
    msg.appendChild(bubble);

    // Show what the answer was based on (explainability): lab documents and,
    // for signed-in users, which of their personal memory items were used.
    sources = sources || [];
    memoryUsed = memoryUsed || [];
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

    messagesEl.appendChild(msg);
    scrollToBottom();
    return msg;
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
        addMessage("bot", data.reply, data.sources, data.memory_used);
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
