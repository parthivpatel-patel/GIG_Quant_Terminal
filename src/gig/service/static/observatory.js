/* Research Observatory. Renders the ledger; it does not invent results. */
"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const state = { visible: false, started: false, payload: null };
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const num = (value) => (typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "—");

  function truth(payload, failed) {
    if ($("observatory-panel").classList.contains("hidden")) return;
    if (failed) {
      $("state-badge").textContent = "ERROR"; $("state-badge").dataset.state = "ERROR";
      $("source-label").textContent = "research-organization · status request failed";
      $("asof-label").textContent = "As of —";
      $("truth-notice").textContent = "Research only · No orders · No model promotion";
      return;
    }
    const data = payload.data;
    const last = data.last || (data.runs[0] && data.runs[0].report);
    $("state-badge").textContent = "AVAILABLE"; $("state-badge").dataset.state = "AVAILABLE";
    $("source-label").textContent = `research-organization · ${payload.provenance.classification} · ${data.run_count} recorded run${data.run_count === 1 ? "" : "s"}`;
    $("asof-label").textContent = last ? `Last verdict ${last.verdict}` : "No cycle has run";
    $("truth-notice").textContent = "Authored synthetic experiment · Not a market result · No orders";
  }

  function render(payload) {
    const data = payload.data;
    const roles = $("observatory-roles"); roles.replaceChildren();
    data.roles.forEach((role) => {
      const row = el("li", "observatory-role");
      row.append(el("strong", null, role.name), el("span", null, role.does));
      roles.append(row);
    });
    const catalog = $("observatory-catalog"); catalog.replaceChildren();
    data.catalog.forEach((item) => {
      const row = document.createElement("tr");
      [item.name, item.status, item.note].forEach((text) => row.append(el("td", null, text)));
      catalog.append(row);
    });
    const bounds = $("observatory-bounds"); bounds.replaceChildren();
    data.must_not.forEach((item) => bounds.append(el("li", null, item)));
    const last = data.last || (data.runs[0] && data.runs[0].report);
    const finding = $("observatory-finding"); finding.replaceChildren();
    if (!last) {
      finding.append(el("p", null, "No cycle has run. The button runs the one approved hypothesis once and keeps the record."));
      return;
    }
    finding.append(el("p", "observatory-verdict", last.verdict));
    finding.append(el("p", null, last.interpretation));
    const dl = el("dl", "observatory-metrics");
    [["After-cost Sharpe", num(last.sharpe_after_cost)], ["Placebo median Sharpe", num(last.placebo_sharpe_median)],
     ["Deflated Sharpe", num(last.deflated_sharpe)], ["Max drawdown", num(last.max_drawdown)],
     ["Observations", String(last.n_obs)], ["Cost", `${last.cost_bps} bp per change`],
     ["Lookahead detected", last.lookahead_detected ? "Yes" : "No"], ["Orders submitted", String(last.orders_submitted)],
     ["Deduplicated", data.deduplicated ? "Yes, the stored run was returned" : "No"]].forEach(([k, v]) => {
      dl.append(el("dt", null, k), el("dd", null, v));
    });
    finding.append(dl);
  }

  async function load() {
    $("observatory-status").textContent = "Loading the ledger…";
    try {
      const response = await fetch("/api/v1/research-org/status", { headers: { Accept: "application/json" } });
      const payload = await response.json();
      if (!response.ok || !payload.data) throw new Error(payload.error ? payload.error.message : "unavailable");
      state.payload = payload; render(payload); truth(payload, false);
      $("observatory-status").textContent = "";
    } catch (error) {
      $("observatory-status").textContent = `ERROR · ${error.message}`;
      truth(null, true);
    }
  }

  async function cycle() {
    $("observatory-run").disabled = true;
    $("observatory-status").textContent = "Running one bounded cycle…";
    try {
      const response = await fetch("/api/v1/research-org/cycle", { method: "POST", headers: { "Content-Type": "application/json", Accept: "application/json" }, body: "{}" });
      const payload = await response.json();
      if (!response.ok || !payload.data) throw new Error(payload.error ? payload.error.message : "unavailable");
      state.payload = payload; render(payload); truth(payload, false);
      $("observatory-status").textContent = payload.data.deduplicated ? "Already recorded. The stored finding was returned." : "Recorded.";
    } catch (error) {
      $("observatory-status").textContent = `ERROR · ${error.message}`;
      truth(null, true);
    } finally {
      $("observatory-run").disabled = false;
    }
  }

  function show(on) {
    const panel = $("observatory-panel"); if (!panel) return;
    panel.classList.toggle("hidden", !on);
    if (!on) return;
    if (!state.started) { state.started = true; $("observatory-run").addEventListener("click", cycle); }
    truth({ data: { run_count: 0, runs: [], last: null }, provenance: { classification: "SYNTHETIC" } }, false);
    $("state-badge").textContent = "LOADING";
    load();
  }

  window.GIGObservatory = { show, state: () => (state.payload ? state.payload.data : null) };
})();
