// Admin dashboard: sign in, load /api/admin/stats, draw simple HTML charts.
// Charts are plain divs (no chart library), each bar is keyboard-focusable and
// shows a tooltip, and every number also appears as text or in a table.

(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const CATS = [
    { key: "intent_calibration", label: "Intent calibration", cls: "intent" },
    { key: "domain_knowledge", label: "Domain knowledge", cls: "domain" },
  ];
  const SOURCE_LABEL = {
    manual: "Added by the user", explicit_instruction: "User told the assistant",
    correction: "Learned from a correction", inferred: "Inferred from chat",
  };

  // ---------- Helpers ----------

  const pct = (x) => (x === null || x === undefined ? "–" : `${Math.round(x * 100)}%`);
  const ratioText = (x) => (x === null || x === undefined ? "–" : `${x} : 1`);

  function el(tag, attrs, text) {
    const node = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([k, v]) => node.setAttribute(k, v));
    if (text !== undefined) node.textContent = text;   // never innerHTML for data
    return node;
  }

  function row(cells) {
    const tr = el("tr");
    cells.forEach((c) => tr.appendChild(el("td", {}, String(c))));
    return tr;
  }

  // ---------- Tooltip (hover and keyboard focus) ----------

  const tip = $("tooltip");
  function attachTooltip(target, lines) {
    const show = () => {
      tip.replaceChildren(...lines.map((l, i) => el(i === 0 ? "strong" : "div", {}, l)));
      tip.hidden = false;
      const r = target.getBoundingClientRect();
      const left = Math.min(window.innerWidth - tip.offsetWidth - 8, Math.max(8, r.left + r.width / 2 - tip.offsetWidth / 2));
      tip.style.left = `${left}px`;
      tip.style.top = `${Math.max(8, r.top - tip.offsetHeight - 8) + window.scrollY}px`;
    };
    const hide = () => { tip.hidden = true; };
    target.addEventListener("mouseenter", show);
    target.addEventListener("focus", show);
    target.addEventListener("mouseleave", hide);
    target.addEventListener("blur", hide);
  }

  // ---------- Charts ----------

  // Horizontal bars: one per category, labelled directly with its value.
  function horizontalBars(container, values) {
    const max = Math.max(1, ...values.map((v) => v.value));
    container.replaceChildren(...values.map((v) => {
      const line = el("div", { class: "hbar-row" });
      line.appendChild(el("span", { class: "hbar-label" }, v.label));
      const track = el("div", { class: "hbar-track" });
      const bar = el("div", {
        class: `hbar ${v.cls}`, tabindex: "0", role: "img",
        "aria-label": `${v.label}: ${v.value}`,
      });
      bar.style.width = `${(v.value / max) * 100}%`;
      attachTooltip(bar, [v.label, `${v.value} ${v.unit}`]);
      track.appendChild(bar);
      line.append(track, el("span", { class: "hbar-value" }, String(v.value)));
      return line;
    }));
  }

  // Vertical bars by day: how many items were switched off, reasons in tooltip.
  function deactivationChart(days) {
    const box = $("chart-deactivated");
    $("deactivated-empty").hidden = days.length > 0;
    box.hidden = days.length === 0;
    const max = Math.max(1, ...days.map((d) => d.total));
    box.replaceChildren(...days.map((d) => {
      const col = el("div", { class: "vbar-col" });
      const label = new Date(d.date + "T00:00:00").toLocaleDateString("en-AU", { day: "numeric", month: "short" });
      const bar = el("div", {
        class: "vbar", tabindex: "0", role: "img",
        "aria-label": `${label}: ${d.total} deactivated`,
      });
      bar.style.height = `${(d.total / max) * 100}%`;
      attachTooltip(bar, [`${label}: ${d.total} deactivated`,
        ...Object.entries(d.reasons).map(([reason, n]) => `${n} × ${reason}`)]);
      const area = el("div", { class: "vbar-area" });
      area.append(el("span", { class: "vbar-value" }, String(d.total)), bar);
      col.append(area, el("span", { class: "vbar-label" }, label));
      return col;
    }));
  }

  // ---------- Render everything ----------

  function render(s) {
    $("data-notice").hidden = !s.data_notice.contains_synthetic;

    const t = s.totals;
    $("kpi-reuse").textContent = ratioText(s.ratios.reuse);
    $("kpi-paper").textContent = s.ratios.paper_reuse;
    $("kpi-count").textContent = ratioText(s.ratios.count);
    $("kpi-items").textContent = t.memory_items;
    $("kpi-items-note").textContent = `${t.active} active · ${t.inactive} switched off`;
    $("kpi-positive").textContent = pct(s.feedback.positive_rate);
    $("kpi-positive-note").textContent = `of ${s.feedback.rated} rated replies`;
    $("kpi-users").textContent = t.users;
    $("kpi-users-note").textContent = Object.entries(t.users_by_tier)
      .map(([tier, n]) => `${n} ${tier}${n === 1 ? "" : "s"}`).join(" · ") || "none yet";

    const bars = (field, unit) => CATS.map((c) => ({
      label: c.label, cls: c.cls, unit, value: s.by_category[c.key][field],
    }));
    horizontalBars($("chart-stored"), bars("items", "items"));
    horizontalBars($("chart-reused"), bars("times_used", "uses"));

    $("effectiveness").replaceChildren(...CATS.map((c) => {
      const d = s.by_category[c.key];
      return row([c.label, `${d.active} / ${d.items}`, pct(d.use_rate), pct(d.helpful_rate), pct(d.correction_rate)]);
    }));

    $("sources").replaceChildren(...Object.entries(s.by_source)
      .sort((a, b) => b[1] - a[1])
      .map(([src, n]) => row([SOURCE_LABEL[src] || src, n])));

    $("top-items").replaceChildren(...(s.top_reused.length ? s.top_reused.map((i) => row([
      i.user, i.rule_text, i.category === "intent_calibration" ? "Intent" : "Domain",
      SOURCE_LABEL[i.source] || i.source, i.times_used, i.times_helpful, i.times_corrected,
      i.active ? "Active" : "Switched off",
    ])) : [row(["–", "No memory items have been reused yet.", "", "", "", "", "", ""])]));

    deactivationChart(s.deactivations);
  }

  // ---------- Sign in / out ----------

  async function load() {
    const res = await fetch("/api/admin/stats");
    if (res.status === 401) {
      $("dash-view").hidden = true;
      $("login-view").hidden = false;
      return;
    }
    render(await res.json());
    $("login-view").hidden = true;
    $("dash-view").hidden = false;
  }

  $("login-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const res = await fetch("/api/admin/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: $("admin-user").value, password: $("admin-pass").value }),
    });
    if (res.ok) {
      $("login-error").hidden = true;
      load();
    } else {
      const data = await res.json().catch(() => ({}));
      $("login-error").textContent = typeof data.detail === "string" ? data.detail : "Sign-in failed.";
      $("login-error").hidden = false;
    }
  });

  $("logout").addEventListener("click", async () => {
    await fetch("/api/admin/logout", { method: "POST" });
    load();
  });
  $("refresh").addEventListener("click", load);

  $("theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("aidx-theme", next); } catch (e) { /* fine */ }
  });

  load();
})();
