/* Shared investigation context. A research focus never changes market identity,
 * approves an objective, starts a worker, or creates a trading signal. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const STORAGE = "gig.research-focus.v1";
  const ALLOWED = new Set(["command-center", "charts", "observatory", "market-flow", "research"]);
  const state = { data: null, focus: null, busy: false, failed: false };
  try { const saved = localStorage.getItem(STORAGE); if (saved && saved.length <= 160) state.focus = saved; } catch { /* Storage is optional. */ }
  const el = (tag, className, text) => { const n = document.createElement(tag); if (className) n.className = className; if (text !== undefined) n.textContent = String(text); return n; };
  const known = (value) => value === undefined || value === null || value === "" ? "UNKNOWN" : String(value);
  const section = () => location.hash.slice(1) || "command-center";
  const button = (label, callback, className = "text-button") => { const n = el("button", className, label); n.type = "button"; n.addEventListener("click", callback); return n; };
  const root = el("section", "research-dossier hidden"); root.id = "research-dossier"; root.setAttribute("aria-labelledby", "research-dossier-title");
  $("truth-strip").after(root);

  function focus(id) {
    if (!state.data?.director?.objectives?.some((row) => row.objective_id === id)) return;
    state.focus = id;
    try { localStorage.setItem(STORAGE, id); } catch { /* No persistence claim. */ }
    window.GIGObservatory?.focusObjective(id);
    render();
  }
  function open(workspace) {
    location.hash = workspace;
    if (workspace === "observatory" && state.focus) window.GIGObservatory?.focusObjective(state.focus);
  }
  function agenda(rows) {
    const box = el("div", "research-agenda"); box.id = "research-agenda";
    const head = el("div", "research-agenda-head");
    head.append(el("h4", null, "Research queue / selection preview"), el("span", null, "Read-only · Approval required")); box.append(head);
    const list = el("ol", "research-agenda-list");
    [...rows].sort((a, b) => (b.selection_score ?? b.priority ?? 0) - (a.selection_score ?? a.priority ?? 0) || a.objective_id.localeCompare(b.objective_id)).forEach((row, index) => {
      const item = el("li");
      const choose = button("", () => focus(row.objective_id), "research-agenda-choice");
      choose.dataset.objectiveId = row.objective_id; choose.setAttribute("aria-pressed", String(row.objective_id === state.focus));
      const description = el("span", "research-agenda-question"); description.append(el("strong", null, row.question), el("span", null, `${known(row.family || row.dataset_id)} · ${known(row.dataset_state)} · ${known(row.status)}`));
      const score = el("span", "research-agenda-score"); score.append(el("strong", null, known(row.selection_score ?? row.priority)), el("span", null, "selection"));
      choose.append(el("span", "research-agenda-rank", String(index + 1).padStart(2, "0")), description, score);
      if (row.selection_reason) choose.title = row.selection_reason;
      item.append(choose); list.append(item);
    });
    box.append(list); return box;
  }
  function render() {
    const visible = ALLOWED.has(section()); root.classList.toggle("hidden", !visible); if (!visible) return;
    root.replaceChildren();
    const mast = el("div", "research-dossier-head"), title = el("div");
    title.append(el("p", "panel-kicker", "GIG / INVESTIGATION DESK"));
    const heading = el("h2", null, section() === "command-center" ? "Conviction begins with a question." : "One question. A connected workspace."); heading.id = "research-dossier-title"; title.append(heading);
    const reload = button(state.busy ? "Loading…" : "Refresh research state", load); reload.id = "research-dossier-refresh"; reload.disabled = state.busy;
    mast.append(title, reload); root.append(mast);
    const director = state.data?.director, rows = director?.objectives || [];
    if (state.failed || !director || !rows.length) {
      const message = el("p", "research-dossier-empty", state.failed ? "UNAVAILABLE / Research memory could not be loaded. No approval or finding is inferred." : "Loading approved investigation registry and durable research memory…"); message.id = "research-dossier-status"; message.setAttribute("role", "status"); root.append(message); return;
    }
    if (!rows.some((row) => row.objective_id === state.focus)) state.focus = rows[0].objective_id;
    const active = rows.find((row) => row.objective_id === state.focus), dataset = active.dataset || {};
    const layout = el("div", "research-dossier-grid"), story = el("div", "research-focus-story");
    const status = el("p", "research-focus-status"); status.id = "research-focus-status"; status.append(el("span", null, `APPROVAL / ${known(active.approval_state)}`), el("span", null, active.status === "COMPLETED" ? "COMPLETED" : known(active.eligibility)), el("span", null, "EXECUTION DISABLED")); story.append(status);
    const question = el("h3", null, active.question); question.id = "research-focus-question"; story.append(question);
    story.append(el("p", "research-focus-reason", active.selection_reason || active.reason));
    const evidence = el("dl", "research-focus-evidence");
    [["Dataset", active.dataset_id], ["Observed scope", dataset.classification || "Inspect source provenance"], ["Publication clock", dataset.publication_time], ["Corporate actions", dataset.corporate_actions_state], ["Method", `${known(active.method)} / v${known(active.method_version)}`], ["Research memory", `${director.memory?.length || 0} retained findings`]].forEach(([label, value]) => evidence.append(el("dt", null, label), el("dd", null, known(value))));
    story.append(evidence);
    const nav = el("nav", "research-journey"); nav.setAttribute("aria-label", "Connected research workspaces");
    [["01", "Director & Laboratory", "observatory"], ["02", "Market Charts", "charts"], ["03", "Market Flow", "market-flow"]].forEach(([number, label, target]) => { const n = button("", () => open(target), "research-journey-link"); n.setAttribute("aria-current", section() === target ? "page" : "false"); n.append(el("span", null, number), el("strong", null, label), el("span", null, "Open")); nav.append(n); }); story.append(nav);
    story.append(el("p", "research-context-limit", "Shared context is an investigation, not a synchronized instrument. Chart bars and observed positioning retain their own sources; neither is substituted by this selection."));
    layout.append(story, agenda(rows)); root.append(layout);
    if (section() !== "command-center" && section() !== "observatory") root.classList.add("research-dossier-compact"); else root.classList.remove("research-dossier-compact");
  }
  async function load() {
    if (state.busy) return;
    state.busy = true; state.failed = false; render();
    try { const response = await fetch("/api/v1/research-org/status", { headers: { Accept: "application/json" } }); const payload = await response.json(); if (!response.ok || payload.state !== "AVAILABLE" || !payload.data?.director) throw new Error("unavailable"); state.data = payload.data; }
    catch { state.data = null; state.failed = true; }
    finally { state.busy = false; render(); }
  }
  window.addEventListener("gig:research-state", (event) => { if (!event.detail?.director) return; state.data = event.detail; state.failed = false; render(); if (state.focus) window.GIGObservatory?.focusObjective(state.focus); });
  window.addEventListener("gig:research-focus", (event) => focus(event.detail));
  window.addEventListener("hashchange", () => { render(); if (ALLOWED.has(section()) && !state.data && !state.busy) load(); });
  window.GIGResearchDesk = { focus, reload: load, state: () => ({ objective: state.focus, available: Boolean(state.data), busy: state.busy }) };
  render(); if (ALLOWED.has(section())) load();
})();
