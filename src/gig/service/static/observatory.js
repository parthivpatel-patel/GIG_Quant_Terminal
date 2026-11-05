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
  const num = (value) => {
    if (typeof value !== "number" || !Number.isFinite(value)) return "—";
    if (value !== 0 && Math.abs(value) < 0.0005) return value.toExponential(2);
    return value.toFixed(3);
  };

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
    const last = data.last;
    $("state-badge").textContent = "AVAILABLE"; $("state-badge").dataset.state = "AVAILABLE";
    $("source-label").textContent = `research-organization · ${payload.provenance.classification} · ${data.run_count} recorded run${data.run_count === 1 ? "" : "s"}`;
    $("asof-label").textContent = last ? `Last verdict ${last.verdict}` : "No job has run";
    const kind = payload.provenance.classification;
    $("truth-notice").textContent = kind === "HISTORICAL"
      ? "Official Treasury par yields · CC0 · Publication time unknown · Not a traded result"
      : "Synthetic jobs are labelled synthetic · Treasury study is historical · No orders";
  }

  function spark(curve) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 240 48");
    svg.setAttribute("class", "observatory-spark");
    svg.setAttribute("role", "img");
    const values = curve.map((point) => point.equity);
    const min = Math.min(...values);
    const span = Math.max(...values) - min || 1;
    const points = values.map((value, index) => {
      const x = values.length === 1 ? 0 : (index / (values.length - 1)) * 240;
      return `${x},${46 - ((value - min) / span) * 44}`;
    }).join(" ");
    const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    line.setAttribute("points", points);
    line.setAttribute("fill", "none");
    line.setAttribute("stroke", "currentColor");
    svg.append(line);
    svg.setAttribute("aria-label", `Equity on the authored series, from ${num(values[0])} to ${num(values[values.length - 1])}`);
    return svg;
  }

  function lineage(block) {
    const dl = el("dl", "observatory-metrics");
    const rows = [
      ["Dataset", block.dataset || "—"],
      ["Strategy", block.strategy ? `${block.strategy} · v${block.strategy_version}` : `v${block.strategy_version}`],
      ["Point in time", block.point_in_time || "—"],
      ["Costs", block.costs || "—"],
      ["Assumptions", block.assumptions || "—"],
      ["Configuration", block.configuration || "—"],
      ["Skeptic", block.skeptic_verdict || block.verdict || "—"],
      ["Artifacts", block.artifacts || "—"],
    ];
    rows.forEach(([key, value]) => { dl.append(el("dt", null, key), el("dd", null, value)); });
    return dl;
  }

  function render(payload) {
    const data = payload.data;
    const roles = $("observatory-roles"); roles.replaceChildren();
    data.roles.forEach((role) => {
      const row = el("li", "observatory-role");
      row.append(el("strong", null, role.name), el("span", null, role.does));
      roles.append(row);
    });
    const backlog = $("observatory-backlog"); backlog.replaceChildren();
    data.backlog.forEach((job) => {
      const row = el("li", null, `${job.hypothesis_id} · ${job.status}`);
      backlog.append(row);
    });
    const catalog = $("observatory-catalog"); catalog.replaceChildren();
    data.catalog.forEach((item) => {
      const row = document.createElement("tr");
      [item.name, item.status, item.note].forEach((text) => row.append(el("td", null, text)));
      catalog.append(row);
    });
    const bounds = $("observatory-bounds"); bounds.replaceChildren();
    data.must_not.forEach((item) => bounds.append(el("li", null, item)));
    const history = $("observatory-history"); history.replaceChildren();
    if (!data.runs.length) history.append(el("p", null, "No recorded cycle."));
    data.runs.forEach((run) => {
      const report = run.report;
      history.append(el("p", null, `${report.hypothesis_id} · ${report.verdict} · Sharpe ${num(report.sharpe_after_cost)}`));
    });
    const finding = $("observatory-finding"); finding.replaceChildren();
    const last = data.last;
    if (!last) {
      finding.append(el("p", null, "No job has run. The button runs the next approved job and keeps the record."));
    } else {
      finding.append(el("p", "observatory-verdict", last.verdict));
      finding.append(el("p", null, last.interpretation));
      if (last.equity_curve && last.equity_curve.length > 1) finding.append(spark(last.equity_curve));
      const dl = el("dl", "observatory-metrics");
      const rows = [["Evidence class", last.evidence_class || "—"],
        ["After-cost Sharpe", num(last.sharpe_after_cost)], ["Total return", num(last.total_return)],
        ["Sortino", num(last.sortino)], ["Max drawdown", num(last.max_drawdown)],
        ["Hit rate", typeof last.hit_rate === "number" ? num(last.hit_rate) : "—"],
        ["Held-out Sharpe", last.holdout && typeof last.holdout.sharpe === "number" ? num(last.holdout.sharpe) : "—"],
        ["Benchmark Sharpe", last.benchmark ? num(last.benchmark.sharpe) : "—"],
        ["Placebo median Sharpe", num(last.placebo_sharpe_median)], ["Deflated Sharpe", num(last.deflated_sharpe)],
        ["Observations", String(last.n_obs)], ["Publication time", last.publication_time || "—"],
        ["Lookahead detected", last.lookahead_detected ? "Yes" : "No"],
        ["Orders submitted", String(last.orders_submitted)]];
      rows.forEach(([key, value]) => { dl.append(el("dt", null, key), el("dd", null, value)); });
      finding.append(dl);
      if (last.lineage) finding.append(lineage(last.lineage));
      if (last.reproduce) finding.append(el("p", "observatory-reproduce", last.reproduce));
    }
    const readout = $("scheduler-readout");
    if (readout && data.scheduler_state) {
      const scheduler = data.scheduler_state;
      const reason = scheduler.reason ? ` · ${scheduler.reason}` : "";
      const thread = scheduler.alive ? " · thread on" : "";
      readout.textContent = `${scheduler.desired}${thread} · cadence ${scheduler.cadence_seconds}s${reason}`;
    }
    const objective = $("observatory-objective-result");
    objective.replaceChildren();
    const latest = data.objective || data.objectives[0];
    if (latest) objective.append(el("p", null, `${latest.disposition} · ${latest.detail}`));
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

  async function tick() {
    $("observatory-run").disabled = true;
    $("observatory-status").textContent = "Running the next approved job…";
    try {
      const response = await fetch("/api/v1/research-org/tick", { method: "POST", headers: { "Content-Type": "application/json", Accept: "application/json" }, body: "{}" });
      const payload = await response.json();
      if (!response.ok || !payload.data) throw new Error(payload.error ? payload.error.message : "unavailable");
      state.payload = payload; render(payload); truth(payload, false);
      const action = payload.data.tick ? payload.data.tick.action : "";
      $("observatory-status").textContent = action === "idle" ? "No queued job remains."
        : action === "failed" ? "The job failed. The ledger kept the failure."
          : payload.data.deduplicated ? "Already recorded. The stored finding was returned." : "Recorded.";
    } catch (error) {
      $("observatory-status").textContent = `ERROR · ${error.message}`;
      truth(null, true);
    } finally {
      $("observatory-run").disabled = false;
    }
  }

  async function scheduler(action) {
    const cadence = Number($("scheduler-cadence").value);
    $("observatory-status").textContent = "Updating the scheduler…";
    try {
      const response = await fetch("/api/v1/research-org/scheduler", {
        method: "POST", headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ action, cadence_seconds: cadence }),
      });
      const payload = await response.json();
      if (!response.ok || !payload.data) throw new Error(payload.error ? payload.error.message : "unavailable");
      state.payload = payload; render(payload); truth(payload, false);
      $("observatory-status").textContent = payload.data.scheduler_state.desired;
    } catch (error) {
      $("observatory-status").textContent = `ERROR · ${error.message}`;
      truth(null, true);
    }
  }

  async function objective(event) {
    event.preventDefault();
    const text = $("observatory-objective-text").value;
    $("observatory-status").textContent = "Recording the objective…";
    try {
      const response = await fetch("/api/v1/research-org/objective", {
        method: "POST", headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ objective: text }),
      });
      const payload = await response.json();
      if (!response.ok || !payload.data) throw new Error(payload.error ? payload.error.message : "unavailable");
      state.payload = payload; render(payload); truth(payload, false);
      $("observatory-status").textContent = payload.data.objective.disposition;
    } catch (error) {
      $("observatory-status").textContent = `ERROR · ${error.message}`;
      truth(null, true);
    }
  }

  function show(on) {
    const panel = $("observatory-panel"); if (!panel) return;
    panel.classList.toggle("hidden", !on);
    if (!on) return;
    if (!state.started) {
      state.started = true;
      $("observatory-run").addEventListener("click", tick);
      $("observatory-objective").addEventListener("submit", objective);
      document.querySelectorAll("[data-scheduler]").forEach((button) => {
        button.addEventListener("click", () => scheduler(button.dataset.scheduler));
      });
    }
    truth({ data: { run_count: 0, last: null }, provenance: { classification: "SYNTHETIC" } }, false);
    $("state-badge").textContent = "LOADING";
    load();
  }

  window.GIGObservatory = { show, state: () => (state.payload ? state.payload.data : null) };
})();
