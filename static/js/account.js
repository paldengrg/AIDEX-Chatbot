// AIDEX Assistant: sign in / register and the "My account" memory panel.
//
// All data comes from the JSON API (/api/auth/*, /api/me, /api/memory).
// User-provided text is always inserted with textContent, never innerHTML.

(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const accountBtn = $("account-btn");
  const authDialog = $("auth-dialog");
  const accountDialog = $("account-dialog");
  let currentUser = null;
  let registerMode = false;

  // ---------- Small helpers ----------

  async function api(method, url, body) {
    const res = await fetch(url, {
      method,
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = res.status === 204 ? {} : await res.json().catch(() => ({}));
    if (!res.ok) {
      // FastAPI sends {"detail": "..."}; validation errors send a list.
      const msg = typeof data.detail === "string" ? data.detail : "Please check your input.";
      throw new Error(msg);
    }
    return data;
  }

  function showError(el, message) {
    el.textContent = message || "";
    el.hidden = !message;
  }

  // Close buttons inside any dialog
  document.querySelectorAll("[data-close]").forEach((btn) =>
    btn.addEventListener("click", () => btn.closest("dialog").close()));

  // ---------- Signed-in state ----------

  function setUser(user) {
    currentUser = user;
    const initial = $("account-initial");
    if (user) {
      initial.textContent = user.username.charAt(0).toUpperCase();
      initial.hidden = false;
      accountBtn.classList.add("signed-in");
      accountBtn.setAttribute("aria-label", `My account (${user.username})`);
      accountBtn.title = "My account";
    } else {
      initial.hidden = true;
      accountBtn.classList.remove("signed-in");
      accountBtn.setAttribute("aria-label", "Sign in");
      accountBtn.title = "Sign in";
    }
    renderResearch();
  }

  // ---------- Research study (consent, banner, withdraw) ----------

  function renderResearch() {
    const participant = !!(currentUser && currentUser.is_participant);
    const demo = !(currentUser && currentUser.ethics_approved);
    $("research-banner").hidden = !participant;
    $("research-banner-demo").hidden = !demo;
    $("consent-demo-notice").hidden = !demo;
    if (!currentUser) return;
    $("research-join").hidden = participant;
    $("research-withdraw").hidden = !participant;
    $("research-status").textContent = participant
      ? `You are taking part (since ${new Date(currentUser.consent_given_at).toLocaleDateString("en-AU",
        { day: "numeric", month: "short", year: "numeric" })}). ` +
        "Your conversations are saved for research."
      : "You are not taking part. Nothing you type is saved word for word.";
  }

  let consentVersion = null;

  $("research-join").addEventListener("click", async () => {
    try {
      consentVersion = (await api("GET", "/api/research/info")).consent_version;
    } catch (err) { /* shown on submit */ }
    $("consent-version").textContent = consentVersion || "?";
    $("consent-form").reset();
    showError($("consent-error"), "");
    $("consent-dialog").showModal();
  });

  $("consent-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      await api("POST", "/api/research/consent", {
        consent_version: consentVersion,
        agree_information: $("agree-information").checked,
        agree_logging: $("agree-logging").checked,
        agree_withdrawal: $("agree-withdrawal").checked,
      });
      $("consent-dialog").close();
      await openAccount();   // refreshes the profile and the panel
    } catch (err) {
      showError($("consent-error"), err.message);
    }
  });

  $("research-withdraw").addEventListener("click", () => {
    $("withdraw-form").reset();
    $("withdraw-dialog").showModal();
  });

  $("withdraw-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      await api("POST", "/api/research/withdraw", { delete_memory: $("withdraw-memory").checked });
      $("withdraw-dialog").close();
      await openAccount();
    } catch (err) {
      showError($("memory-error"), err.message);
    }
  });

  $("research-banner-manage").addEventListener("click", () => openAccount());

  $("account-delete").addEventListener("click", async () => {
    if (!confirm("Permanently delete your account and all of its data? This can't be undone.")) return;
    try {
      await api("DELETE", "/api/me");
      setUser(null);
      window.AIDX.clearConversation();
      accountDialog.close();
    } catch (err) {
      showError($("memory-error"), err.message);
    }
  });

  accountBtn.addEventListener("click", () => {
    if (currentUser) openAccount();
    else openAuth();
  });

  // ---------- Sign in / register dialog ----------

  function setMode(isRegister) {
    registerMode = isRegister;
    $("auth-title").textContent = isRegister ? "Create an account" : "Sign in";
    $("auth-submit").textContent = isRegister ? "Create account" : "Sign in";
    $("auth-switch-text").textContent = isRegister ? "Already have an account?" : "New here?";
    $("auth-switch").textContent = isRegister ? "Sign in" : "Create an account";
    $("auth-password").autocomplete = isRegister ? "new-password" : "current-password";
    showError($("auth-error"), "");
  }

  function openAuth() {
    setMode(false);
    $("auth-form").reset();
    authDialog.showModal();
    $("auth-username").focus();
  }

  $("auth-switch").addEventListener("click", () => setMode(!registerMode));

  $("auth-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const submit = $("auth-submit");
    submit.disabled = true;
    try {
      const user = await api("POST", registerMode ? "/api/auth/register" : "/api/auth/login", {
        username: $("auth-username").value.trim(),
        password: $("auth-password").value,
      });
      setUser(user);
      window.AIDX.clearConversation();  // the server also started a fresh chat
      authDialog.close();
      if (registerMode) openAccount();   // new users: show the memory opt-in
    } catch (err) {
      showError($("auth-error"), err.message);
    } finally {
      submit.disabled = false;
    }
  });

  // ---------- My account panel ----------

  const CATEGORY_LABEL = {
    intent_calibration: "How you like answers",
    domain_knowledge: "About you",
  };
  const SOURCE_LABEL = {
    manual: "added by you",
    explicit_instruction: "you told me",
    correction: "from a correction",
    inferred: "inferred from chat",
  };
  const CLARIFY_LABEL = {
    ask: "I check with you first when a question is unclear.",
    act: "I answer directly and state my assumption when a question is unclear.",
    neutral: "I ask a short question only when something is really unclear.",
  };

  async function openAccount() {
    // Refresh first: learned settings (e.g. clarify style) change as you chat.
    try { setUser((await api("GET", "/api/me")).user); } catch (err) { /* keep old */ }
    if (!currentUser) return openAuth();
    $("account-name").textContent = currentUser.username;
    $("memory-toggle").checked = currentUser.memory_enabled;
    $("clarify-style").textContent = CLARIFY_LABEL[currentUser.clarify_style] || CLARIFY_LABEL.neutral;
    showError($("memory-error"), "");
    accountDialog.showModal();
    await loadMemory();
  }

  $("memory-toggle").addEventListener("change", async (e) => {
    try {
      setUser(await api("PATCH", "/api/me", { memory_enabled: e.target.checked }));
      $("clarify-style").textContent = CLARIFY_LABEL[currentUser.clarify_style] || CLARIFY_LABEL.neutral;
    } catch (err) {
      e.target.checked = !e.target.checked;  // undo the switch on failure
      showError($("memory-error"), err.message);
    }
  });

  async function loadMemory() {
    try {
      const { items } = await api("GET", "/api/memory");
      renderMemory(items);
    } catch (err) {
      showError($("memory-error"), err.message);
    }
  }

  function renderMemory(items) {
    const list = $("memory-list");
    list.replaceChildren(...items.map(memoryRow));
    $("memory-empty").hidden = items.length > 0;
    $("memory-clear").hidden = items.length === 0;
  }

  function memoryRow(item) {
    const li = document.createElement("li");
    li.className = "memory-item" + (item.active ? "" : " inactive");

    const meta = document.createElement("div");
    meta.className = "memory-meta";
    const badge = document.createElement("span");
    badge.className = "badge " + item.category;
    badge.textContent = CATEGORY_LABEL[item.category] || item.category;
    const stats = document.createElement("span");
    const n = item.times_used;
    stats.textContent = `${SOURCE_LABEL[item.source] || item.source} · used in ${n} ${n === 1 ? "reply" : "replies"}` +
      ` · ${item.times_helpful} helpful · ${item.times_corrected} corrected`;
    meta.append(badge, stats);

    const text = document.createElement("p");
    text.textContent = item.rule_text;

    // Explain why an item is switched off (by the user or by the automatic review).
    const status = document.createElement("p");
    status.className = "memory-status";
    status.textContent = item.active ? "" : `Switched off: ${item.deactivation_reason || "paused"}`;
    status.hidden = item.active;

    const actions = document.createElement("div");
    actions.className = "memory-actions";
    const button = (label, handler) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "link-btn";
      b.textContent = label;
      b.addEventListener("click", handler);
      actions.appendChild(b);
    };
    button("Edit", () => startEdit(li, item));
    button(item.active ? "Pause" : "Resume", () => update(item.id, { active: !item.active }));
    button("Delete", async () => {
      try { await api("DELETE", `/api/memory/${item.id}`); loadMemory(); }
      catch (err) { showError($("memory-error"), err.message); }
    });

    li.append(meta, text, status, actions);
    return li;
  }

  function startEdit(li, item) {
    const area = document.createElement("textarea");
    area.value = item.rule_text;
    area.maxLength = 300;
    area.rows = 2;
    area.setAttribute("aria-label", "Edit memory");
    const save = document.createElement("button");
    save.type = "button";
    save.className = "primary-btn small";
    save.textContent = "Save";
    save.addEventListener("click", () => update(item.id, { rule_text: area.value }));
    li.querySelector("p").replaceWith(area);
    li.querySelector(".memory-actions").replaceChildren(save);
    area.focus();
  }

  async function update(id, changes) {
    try {
      await api("PATCH", `/api/memory/${id}`, changes);
      showError($("memory-error"), "");
      loadMemory();
    } catch (err) {
      showError($("memory-error"), err.message);
    }
  }

  $("memory-add").addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      const { created } = await api("POST", "/api/memory", {
        rule_text: $("memory-text").value,
        category: $("memory-category").value,
      });
      $("memory-text").value = "";
      showError($("memory-error"), created ? "" : "You already had a similar memory, so I kept that one.");
      loadMemory();
    } catch (err) {
      showError($("memory-error"), err.message);
    }
  });

  $("memory-clear").addEventListener("click", async () => {
    // Native confirm() is simple and accessible for a destructive action.
    if (!confirm("Delete all of your saved memories? This can't be undone.")) return;
    try { await api("DELETE", "/api/memory"); loadMemory(); }
    catch (err) { showError($("memory-error"), err.message); }
  });

  $("logout-btn").addEventListener("click", async () => {
    await api("POST", "/api/auth/logout").catch(() => {});
    setUser(null);
    window.AIDX.clearConversation();
    accountDialog.close();
  });

  // Let chat.js open this panel (the "Review" link on "Noted" messages).
  window.AIDX.openAccount = () => currentUser && openAccount();
  // ...and refresh state after an "Undo" on a learned memory.
  window.AIDX.reloadMemory = () => accountDialog.open && loadMemory();

  // ---------- Start: am I already signed in? ----------
  api("GET", "/api/me").then((d) => setUser(d.user)).catch(() => setUser(null));
})();
