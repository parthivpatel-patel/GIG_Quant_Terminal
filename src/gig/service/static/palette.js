/* Command palette and global search (Ctrl/Cmd+K). Searches sections, commands,
 * chart instruments in the current data environment and Global Intelligence
 * events already loaded in this page. It only navigates and changes view
 * settings; it has no command that places orders or contacts a provider. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const state = { items: [], results: [], active: 0, opener: null, instruments: [], open: false };

  function commands() {
    const mode = $("data-mode");
    const collapsed = document.body.classList.contains("nav-collapsed");
    const dark = document.documentElement.dataset.theme === "dark";
    const setMode = (value) => () => { if (mode.value !== value) { mode.value = value; mode.dispatchEvent(new Event("change")); } };
    return [
      { kind: "Command", label: "Use the Synthetic research demo data environment", keywords: "synthetic demo data mode", run: setMode("synthetic") },
      { kind: "Command", label: "Use admitted local data", keywords: "local admitted data mode", run: setMode("local") },
      { kind: "Command", label: `Switch to ${dark ? "light" : "dark"} theme`, keywords: "theme dark light appearance", run: () => $("theme-toggle").click() },
      { kind: "Command", label: `${collapsed ? "Expand" : "Collapse"} navigation`, keywords: "sidebar navigation collapse expand", run: () => $("nav-toggle").click() },
      { kind: "Command", label: "Refresh the current section", keywords: "reload refresh", run: () => $("refresh").click() },
    ];
  }
  function sections() {
    return [...document.querySelectorAll(".nav-link")].map((link) => ({ kind: "Section", label: link.textContent.trim(), keywords: link.hash.slice(1).replaceAll("-", " "), run: () => { location.hash = link.hash; } }));
  }
  function instruments() {
    return state.instruments.map((row) => ({ kind: "Instrument", label: row.symbol, detail: Number.isFinite(row.last) ? `last ${row.last.toLocaleString("en-US", { maximumFractionDigits: 4 })}` : "", keywords: "chart instrument symbol",
      run: () => { if (window.GIGCharts) window.GIGCharts.open(row.symbol); location.hash = "#charts"; } }));
  }
  function events() {
    if (!window.GIGIntel || typeof window.GIGIntel.events !== "function") return [];
    let list = []; try { list = window.GIGIntel.events(); } catch { list = []; }
    return list.slice(0, 400).map((e) => ({ kind: "Event", label: e.title, keywords: "event global intelligence", run: () => { location.hash = "#global-intelligence"; setTimeout(() => window.GIGIntel.select(e.event_id), 0); } }));
  }
  function score(item, terms) {
    const hay = `${item.label} ${item.keywords || ""} ${item.kind}`.toLowerCase();
    if (!terms.every((t) => hay.includes(t))) return -1;
    const label = item.label.toLowerCase();
    return (label.startsWith(terms[0] || "") ? 3 : 0) + (label.includes(terms.join(" ")) ? 2 : 0) + ({ Section: 1.5, Command: 1, Instrument: 1.2, Event: 0 }[item.kind] || 0);
  }
  function filter() {
    const terms = $("palette-input").value.toLowerCase().split(/\s+/).filter(Boolean);
    state.items = [...sections(), ...commands(), ...instruments(), ...events()];
    state.results = state.items.map((item, i) => ({ item, s: score(item, terms), i })).filter((r) => r.s >= 0)
      .sort((a, b) => b.s - a.s || a.i - b.i).slice(0, 40).map((r) => r.item);
    state.active = Math.min(state.active, Math.max(0, state.results.length - 1));
    render();
  }
  function render() {
    const list = $("palette-list"); list.replaceChildren();
    if (!state.results.length) { const li = document.createElement("li"); li.className = "palette-empty"; li.textContent = "No matches"; list.append(li); $("palette-input").removeAttribute("aria-activedescendant"); return; }
    state.results.forEach((item, i) => {
      const li = document.createElement("li"); li.id = `palette-option-${i}`; li.setAttribute("role", "option"); li.className = "palette-option";
      li.setAttribute("aria-selected", String(i === state.active));
      const kind = document.createElement("span"); kind.className = "palette-kind"; kind.textContent = item.kind;
      const label = document.createElement("span"); label.className = "palette-label"; label.textContent = item.label;
      li.append(kind, label);
      if (item.detail) { const d = document.createElement("span"); d.className = "palette-detail"; d.textContent = item.detail; li.append(d); }
      li.addEventListener("mousemove", () => { if (state.active !== i) { state.active = i; render(); } });
      li.addEventListener("click", () => choose(i));
      list.append(li);
    });
    $("palette-input").setAttribute("aria-activedescendant", `palette-option-${state.active}`);
    const active = $(`palette-option-${state.active}`); if (active && active.scrollIntoView) active.scrollIntoView({ block: "nearest" });
  }
  function choose(i) { const item = state.results[i]; if (!item) return; close(false); item.run(); }
  async function open() {
    if (state.open) return;
    state.open = true; state.opener = document.activeElement; state.active = 0;
    $("palette").classList.remove("hidden"); $("palette-input").value = ""; filter(); $("palette-input").focus();
    if (window.GIGCharts && typeof window.GIGCharts.catalog === "function") {
      const rows = await window.GIGCharts.catalog();
      if (state.open) { state.instruments = rows; filter(); }
    }
  }
  function close(restore = true) {
    if (!state.open) return;
    state.open = false; $("palette").classList.add("hidden");
    if (restore && state.opener && state.opener.focus) state.opener.focus();
  }
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && !e.altKey && e.key.toLowerCase() === "k") { e.preventDefault(); if (state.open) close(); else open(); return; }
    if (!state.open) return;
    if (e.key === "Escape") { e.preventDefault(); close(); }
    else if (e.key === "ArrowDown") { e.preventDefault(); state.active = Math.min(state.results.length - 1, state.active + 1); render(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); state.active = Math.max(0, state.active - 1); render(); }
    else if (e.key === "Enter") { e.preventDefault(); choose(state.active); }
    else if (e.key === "Tab") { e.preventDefault(); $("palette-input").focus(); }
  });
  $("palette-input").addEventListener("input", () => { state.active = 0; filter(); });
  $("palette").addEventListener("mousedown", (e) => { if (e.target === $("palette")) close(); });
  $("palette-open").addEventListener("click", open);
  window.GIGPalette = { open, close, results: () => state.results.map((r) => ({ kind: r.kind, label: r.label })) };
})();
