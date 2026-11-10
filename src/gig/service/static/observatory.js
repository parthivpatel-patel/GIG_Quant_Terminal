/* Research Operations desk. Every displayed result comes from the local ledger.
 * Approval selects a registered template; it never starts compute or trading. */
"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const OBJECTIVE_ID = "treasury-reversal-robustness-v1";
  const state = { visible: false, started: false, payload: null, selected: null, followLatest: true,
    traceStage: "source", timer: null, loading: false };
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = String(text);
    return node;
  };
  const known = (value) => {
    if (value === undefined || value === null || value === "") return "UNKNOWN";
    if (Array.isArray(value)) return value.map(known).join(", ");
    if (typeof value === "object") return Object.entries(value).map(([key, item]) => `${key.replaceAll("_", " ")}: ${known(item)}`).join("; ");
    return String(value);
  };
  const num = (value) => {
    if (typeof value !== "number" || !Number.isFinite(value)) return "—";
    if (value !== 0 && Math.abs(value) < 0.0005) return value.toExponential(2);
    return value.toFixed(3);
  };
  const identity = (run) => run.run_id || run.config_hash || run.report.hypothesis_id;
  const entries = (rows, className = "observatory-metrics") => {
    const dl = el("dl", className);
    rows.forEach(([key, value]) => dl.append(el("dt", null, key), el("dd", null, known(value))));
    return dl;
  };
  function selectedRun() {
    const data = state.payload && state.payload.data;
    if (!data) return null;
    const runs = data.runs || [];
    return runs.find((run) => identity(run) === state.selected) || runs[0] || null;
  }

  function truth(payload, failed) {
    if (!state.visible) return;
    if (failed) {
      $("state-badge").textContent = "ERROR"; $("state-badge").dataset.state = "ERROR";
      $("source-label").textContent = "research-organization · status request failed";
      $("asof-label").textContent = "As of —";
      $("truth-notice").textContent = "Research only · No orders · No model promotion";
      return;
    }
    const data = payload.data;
    const run = selectedRun();
    const report = run ? run.report : data.last;
    const kind = report && report.classification || payload.provenance && payload.provenance.classification || "UNKNOWN";
    $("state-badge").textContent = "AVAILABLE"; $("state-badge").dataset.state = "AVAILABLE";
    $("source-label").textContent = `research-organization · ${kind} · ${data.run_count} recorded run${data.run_count === 1 ? "" : "s"}`;
    $("asof-label").textContent = report ? `Selected verdict ${known(report.verdict)}` : "No job has run";
    $("truth-notice").textContent = kind === "HISTORICAL"
      ? "Historical observations · Publication time unknown unless evidenced · Not a traded result"
      : "Synthetic jobs are labelled synthetic · Treasury study is historical · No orders";
  }

  function spark(curve) {
    const values = curve.map((point) => point.equity).filter((value) => typeof value === "number" && Number.isFinite(value));
    if (values.length < 2) return el("p", "muted", "No finite equity curve is available.");
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 600 100"); svg.setAttribute("class", "observatory-spark"); svg.setAttribute("role", "img");
    const min = Math.min(...values), span = Math.max(...values) - min || 1;
    const points = values.map((value, index) => `${(index / (values.length - 1)) * 600},${94 - ((value - min) / span) * 88}`).join(" ");
    const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    line.setAttribute("points", points); line.setAttribute("fill", "none"); line.setAttribute("stroke", "currentColor");
    line.setAttribute("stroke-width", "1.5"); svg.append(line);
    svg.setAttribute("aria-label", `Equity on the authored series, from ${num(values[0])} to ${num(values[values.length - 1])}. Synthetic, not observed performance.`);
    return svg;
  }

  function renderTimeline(run) {
    const list = $("observatory-timeline"); list.replaceChildren();
    const detail = $("observatory-trace-detail"); detail.replaceChildren();
    if (!run) { detail.append(el("p", null, "Select a saved finding to inspect its evidence chain.")); return; }
    const r = run.report, line = r.lineage || {}, dataset = r.dataset || {};
    const datasetName = typeof dataset === "object" ? dataset.id || line.dataset : dataset;
    const stages = [
      { id: "source", label: "Source", status: r.classification || "UNKNOWN", rows: [
        ["Dataset", datasetName || line.dataset], ["Classification", r.classification], ["Source", typeof dataset === "object" ? dataset.source_url : null],
        ["Observation period", r.period ? `${known(r.period.start)} → ${known(r.period.end)}` : null],
        ["Acquired", typeof dataset === "object" ? dataset.retrieved_at : null],
        ["Publication clock", r.publication_time], ["Source hash", typeof dataset === "object" ? dataset.sha256 || line.sha256 : line.sha256],
        ["License", typeof dataset === "object" ? dataset.license : null]] },
      { id: "features", label: "Features", status: "INSPECT", rows: [
        ["Point-in-time boundary", line.point_in_time || (typeof dataset === "object" ? dataset.point_in_time : null)],
        ["Construction", r.method || r.training_boundary], ["Training boundary", r.training_boundary],
        ["Lookahead probe", typeof r.lookahead_detected === "boolean" ? r.lookahead_detected ? "DETECTED" : "No lookahead detected by the stated probe" : null],
        ["Limit", "A causal calculation does not establish historical publication availability."]] },
      { id: "experiment", label: "Experiment", status: "SAVED", rows: [
        ["Hypothesis", r.hypothesis_id], ["Strategy", line.strategy || r.strategy],
        ["Version", r.experiment_version || line.strategy_version], ["Model version", line.model_version],
        ["Configuration", line.configuration], ["Stored run", identity(run)], ["Recorded", run.finished_at],
        ["Reproduce", r.reproduce || line.reproduce]] },
      { id: "skeptic", label: "Skeptic", status: r.verdict || "UNKNOWN", rows: [
        ["Verdict", line.skeptic_verdict || r.verdict], ["Evidence class", r.evidence_class],
        ["Placebo draws", r.placebo_draws || (r.metrics && r.metrics.placebo && r.metrics.placebo.draws)],
        ["Multiple testing", r.multiple_testing ? `${r.multiple_testing.method}; ${r.multiple_testing.family_size} declared local comparisons. ${known(r.multiple_testing.limits)}` : r.n_trials === undefined ? null : `${r.n_trials} declared trial(s); inspect experiment correction details`],
        ["Independent checks", r.skeptic && r.skeptic.checks ? Object.entries(r.skeptic.checks).map(([key, passed]) => `${key.replaceAll("_", " ")}: ${passed ? "PASS" : "FAIL"}`).join("\n") : null],
        ["Review independence", r.skeptic && r.skeptic.independence],
        ["Failure conditions", Array.isArray(r.failure_conditions) ? r.failure_conditions.join("\n") : null],
        ["Economic claim", r.economic_claim || "No economic validation established by this view"]] },
      { id: "risk", label: "Risk gate", status: "NOT APPROVED", rows: [
        ["Costs", r.cost_bps_model || line.costs], ["Allocation authority", "NONE — a saved research result is not an approved signal"],
        ["Portfolio constraints", "UNAVAILABLE — this investigation does not authorize portfolio targets"],
        ["Model promotion", "DISABLED — requires independent approval"]] },
      { id: "execution", label: "Execution", status: "DISABLED", rows: [
        ["Brokerage execution", "DISABLED"], ["Orders reported by experiment", r.orders_submitted],
        ["Offline simulation", "Separate explicit workflow; no order is created by this finding"],
        ["Readiness", "RESEARCH_ONLY · T4_NOT_CERTIFIED"]] },
    ];
    if (!stages.some((stage) => stage.id === state.traceStage)) state.traceStage = "source";
    stages.forEach((stage, index) => {
      const li = el("li"), button = el("button", "observatory-trace-step");
      button.type = "button"; button.setAttribute("aria-pressed", String(stage.id === state.traceStage));
      button.setAttribute("aria-controls", "observatory-trace-detail");
      button.append(el("span", "observatory-step-number", String(index + 1).padStart(2, "0")), el("strong", null, stage.label), el("span", "observatory-step-status", stage.status));
      button.addEventListener("click", () => { state.traceStage = stage.id; renderTimeline(run); });
      li.append(button); list.append(li);
    });
    const active = stages.find((stage) => stage.id === state.traceStage);
    detail.append(el("h5", null, `${active.label} / ${active.status}`), entries(active.rows));
  }

  function renderFinding() {
    const run = selectedRun(), finding = $("observatory-finding"); finding.replaceChildren();
    if (!run) {
      $("observatory-selected-title").textContent = "An open question is a useful starting point.";
      $("observatory-selected-kind").textContent = "NO FINDING SELECTED";
      finding.append(el("p", "muted", "No job has run. The button runs the next approved job and keeps the record."));
      renderTimeline(null); return;
    }
    const r = run.report;
    $("observatory-selected-title").textContent = r.question || r.hypothesis_id;
    $("observatory-selected-kind").textContent = `${known(r.classification)} / ${known(r.evidence_class || r.verdict)}`;
    finding.append(el("p", "observatory-verdict", known(r.verdict)), el("p", "observatory-interpretation", known(r.interpretation)));
    const sourceName = r.lineage && r.lineage.dataset || (r.dataset && typeof r.dataset === "object" ? r.dataset.id : r.dataset);
    finding.append(el("p", "observatory-source-caption", `Source / ${known(sourceName)}`));
    if (r.equity_curve && r.equity_curve.length > 1) finding.append(spark(r.equity_curve));
    const rows = [["Evidence class", r.evidence_class || r.verdict], ["Observations", r.n_obs], ["Publication time", r.publication_time]];
    if (typeof r.hit_rate === "number") rows.push(["Hit rate", num(r.hit_rate)]);
    if (typeof r.pairs === "number") rows.push(["Non-zero pairs", r.pairs]);
    if (r.classification === "SYNTHETIC" || typeof r.sharpe_after_cost === "number") {
      rows.push(["After-cost Sharpe", num(r.sharpe_after_cost)], ["Total return", num(r.total_return)],
        ["Max drawdown", num(r.max_drawdown)], ["Held-out Sharpe", num(r.holdout && r.holdout.sharpe)],
        ["Placebo median Sharpe", num(r.placebo_sharpe_median)], ["Deflated Sharpe", num(r.deflated_sharpe)]);
    }
    rows.push(["Orders submitted", r.orders_submitted]);
    if (r.metrics && typeof r.metrics.baseline_hit_rate === "number") {
      rows.push(["Training-only baseline hit rate", num(r.metrics.baseline_hit_rate)],
        ["Accuracy difference", num(r.metrics.accuracy_difference)],
        ["Hit rate interval / IID reference", r.metrics.hit_rate_interval_95 && r.metrics.hit_rate_interval_95.map(num).join(" → ")],
        ["Paired block-bootstrap interval", r.metrics.bootstrap && r.metrics.bootstrap.interval_95 && r.metrics.bootstrap.interval_95.map(num).join(" → ")]);
    }
    finding.append(entries(rows));
    if (r.post_selection_reuse) finding.append(el("p", "observatory-reuse-warning", "EXPLORATORY / The parent inspected this archive already. Chronological folds do not create independent confirmation."));
    if (r.metrics && Array.isArray(r.metrics.folds)) {
      const comparison = el("section", "observatory-folds");
      comparison.append(el("h5", null, "Chronological fold comparison"), el("p", "muted", "Directional accuracy on observed yield changes. Not a return or profitability measure."));
      r.metrics.folds.forEach((fold, index) => {
        const total = fold.validation_pairs;
        if (typeof total !== "number" || total <= 0 || typeof fold.hits !== "number" || typeof fold.baseline_hits !== "number") return;
        const row = el("div", "observatory-fold");
        row.append(el("span", "observatory-fold-label", `Fold ${index + 1} · ${known(fold.validation_start)} → ${known(fold.validation_end)}`));
        [["Reversal", fold.hits, "research"], ["Training baseline", fold.baseline_hits, "baseline"]].forEach(([label, hits, kind]) => {
          const rate = hits / total, bar = el("div", "observatory-fold-bar");
          bar.append(el("span", null, label));
          const track = el("span", "observatory-fold-track"), fill = el("span", `observatory-fold-fill ${kind}`);
          fill.style.width = `${Math.min(100, Math.max(0, rate * 100))}%`; track.setAttribute("aria-hidden", "true"); track.append(fill);
          bar.append(track, el("strong", null, `${hits}/${total} · ${(rate * 100).toFixed(1)}%`)); row.append(bar);
        });
        comparison.append(row);
      });
      finding.append(comparison);
    }
    if (r.lineage) {
      const provenance = el("details", "observatory-lineage"); provenance.append(el("summary", null, "Full lineage and reproduction"));
      provenance.append(entries([["Dataset", r.lineage.dataset], ["Strategy", r.lineage.strategy],
        ["Point in time", r.lineage.point_in_time], ["Costs", r.lineage.costs],
        ["Assumptions", r.lineage.assumptions], ["Configuration", r.lineage.configuration],
        ["Skeptic", r.lineage.skeptic_verdict || r.lineage.verdict], ["Artifacts", r.lineage.artifacts]]));
      if (r.reproduce) provenance.append(el("p", "observatory-reproduce", r.reproduce));
      finding.append(provenance);
    }
    renderTimeline(run);
    if (state.payload) truth(state.payload, false);
  }

  function renderHistory() {
    const history = $("observatory-history"); history.replaceChildren();
    const data = state.payload.data, query = $("observatory-filter").value.trim().toLowerCase();
    const matches = (data.runs || []).filter((run) => {
      const r = run.report;
      return `${r.hypothesis_id} ${r.verdict} ${r.classification} ${known(r.dataset)}`.toLowerCase().includes(query);
    });
    if (!matches.length) history.append(el("p", "observatory-empty", query ? "No saved finding matches this filter." : "No recorded cycle."));
    matches.forEach((run) => {
      const r = run.report, p = el("p", "observatory-history-row"), button = el("button", "observatory-run-button");
      button.type = "button"; button.setAttribute("aria-pressed", String(identity(run) === state.selected));
      button.append(el("span", "observatory-run-class", known(r.classification)), el("strong", null, r.hypothesis_id),
        el("span", "observatory-run-verdict", known(r.verdict)));
      button.addEventListener("click", () => { state.selected = identity(run); state.followLatest = false; renderHistory(); renderFinding(); });
      p.append(button); history.append(p);
    });
  }

  function renderDirector(data) {
    const director = data.director, objectives = $("director-objectives"); objectives.replaceChildren();
    const approve = $("director-approve"), revoke = $("director-revoke");
    if (!director) {
      $("director-policy").textContent = "UNAVAILABLE";
      objectives.append(el("p", "muted", "Director state is unavailable. No approval is inferred."));
      approve.disabled = true; revoke.disabled = true; return;
    }
    $("director-policy").textContent = director.policy === "BOUNDED_APPROVED_TEMPLATES" ? "BOUNDED TEMPLATES" : known(director.policy);
    const rows = director.objectives || [];
    rows.forEach((objective) => {
      const item = el("div", "director-objective");
      item.append(el("span", "observatory-status-tag", objective.status), el("strong", null, objective.question || objective.objective_id),
        el("span", "muted", objective.reason || objective.dataset_id || "Approval is limited to the registered template."));
      objectives.append(item);
    });
    if (!rows.length) objectives.append(el("p", "muted", "No approved objective is recorded."));
    const current = rows.find((row) => row.objective_id === OBJECTIVE_ID);
    approve.disabled = !!current && ["APPROVED", "COMPLETED"].includes(current.status);
    revoke.disabled = !current || current.status !== "APPROVED";
    const investigations = $("director-investigations"); investigations.replaceChildren();
    (director.investigations || []).forEach((item) => investigations.append(entries([
      ["Investigation", item.hypothesis_id], ["Parent finding", item.parent_hypothesis_id], ["Priority", item.priority],
      ["State", item.status], ["Reason", item.reason], ["Report hash", item.report_hash]])));
    if (!(director.investigations || []).length) investigations.append(el("p", "muted", "No director investigation is recorded."));
    const memory = $("director-memory"); memory.replaceChildren();
    (director.memory || []).forEach((item) => memory.append(el("p", null,
      typeof item === "object" ? `${known(item.hypothesis_id)} · ${known(item.classification)} · ${known(item.verdict)} · Economic claim: ${known(item.economic_claim)}` : known(item))));
    if (!(director.memory || []).length) memory.append(el("p", "muted", "No additional director memory is recorded."));
  }

  function render(payload) {
    const data = payload.data, runs = data.runs || [];
    if (state.followLatest || !runs.some((run) => identity(run) === state.selected)) state.selected = runs[0] ? identity(runs[0]) : null;
    renderDirector(data);
    const roles = $("observatory-roles"); roles.replaceChildren();
    (data.roles || []).forEach((role) => {
      const row = el("li", "observatory-role"), activity = data.director && (data.director.roles || []).find((item) => item.id === role.id);
      const name = el("div", "observatory-role-name"); name.append(el("strong", null, role.name));
      if (activity) name.append(el("span", "observatory-role-state", activity.status));
      row.append(name, el("span", null, activity && activity.detail || role.does)); roles.append(row);
    });
    const backlog = $("observatory-backlog"); backlog.replaceChildren();
    (data.backlog || []).forEach((job) => {
      const row = el("li"); row.append(el("span", null, job.hypothesis_id), el("strong", null, job.status)); backlog.append(row);
    });
    const catalog = $("observatory-catalog"); catalog.replaceChildren();
    (data.catalog || []).forEach((item) => {
      const row = el("tr"); [item.name, item.status, item.note].forEach((text) => row.append(el("td", null, text))); catalog.append(row);
    });
    const bounds = $("observatory-bounds"); bounds.replaceChildren();
    (data.must_not || []).forEach((item) => bounds.append(el("li", null, item)));
    const scheduler = data.scheduler_state || {};
    $("observatory-run-count").textContent = known(data.run_count);
    $("observatory-worker-state").textContent = known(scheduler.desired);
    $("observatory-worker-detail").textContent = scheduler.alive ? "Local worker thread running" : "No worker thread reported";
    $("observatory-budget").textContent = `${num(scheduler.cpu_seconds)}s / ${typeof scheduler.db_bytes === "number" ? (scheduler.db_bytes / 1024).toFixed(1) + " KB" : "—"}`;
    $("scheduler-readout").textContent = `${known(scheduler.desired)}${scheduler.alive ? " · thread on" : ""} · cadence ${known(scheduler.cadence_seconds)}s${scheduler.reason ? " · " + scheduler.reason : ""}`;
    const objective = $("observatory-objective-result"); objective.replaceChildren();
    const latest = data.objective || (data.objectives || [])[0];
    if (latest) objective.append(el("p", null, `${latest.disposition} · ${latest.detail}`));
    renderHistory(); renderFinding();
  }

  function poll() {
    clearTimeout(state.timer); state.timer = null;
    if (!state.visible || !state.payload) return;
    const scheduler = state.payload.data.scheduler_state || {};
    if (scheduler.desired === "RUNNING") state.timer = setTimeout(() => load(true), 1500);
  }

  async function load(silent = false) {
    if (state.loading) return;
    state.loading = true;
    if (!silent) $("observatory-status").textContent = "Loading the ledger…";
    try {
      const response = await fetch("/api/v1/research-org/status", { headers: { Accept: "application/json" } });
      const payload = await response.json();
      if (!response.ok || !payload.data) throw new Error("unavailable");
      state.payload = payload;
      if (state.visible) { render(payload); truth(payload, false); if (!silent) $("observatory-status").textContent = ""; }
      poll();
    } catch {
      if (state.visible) { $("observatory-status").textContent = "ERROR · The local research ledger could not be loaded. Refresh to retry."; truth(null, true); }
    } finally { state.loading = false; }
  }

  async function action(path, body, message) {
    $("observatory-status").textContent = message;
    const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json", Accept: "application/json" }, body: JSON.stringify(body) });
    const payload = await response.json();
    if (!response.ok || !payload.data) throw new Error("unavailable");
    state.payload = payload; render(payload); truth(payload, false); poll(); return payload;
  }

  async function tick() {
    $("observatory-run").disabled = true; state.followLatest = true;
    try {
      const payload = await action("/api/v1/research-org/tick", {}, "Running the next approved job…");
      const result = payload.data.tick ? payload.data.tick.action : "";
      $("observatory-status").textContent = result === "idle" ? "No queued job remains."
        : result === "failed" ? "The job failed. The ledger kept the failure."
          : payload.data.deduplicated ? "Already recorded. The stored finding was returned." : "Recorded.";
    } catch { $("observatory-status").textContent = "ERROR · The research cycle was not confirmed. Inspect the ledger before retrying."; truth(null, true); }
    finally { $("observatory-run").disabled = false; }
  }

  async function scheduler(actionName) {
    const cadence = Number($("scheduler-cadence").value);
    if (!Number.isFinite(cadence) || cadence < 0.05 || cadence > 3600) { $("observatory-status").textContent = "Choose a cadence between 0.05 and 3600 seconds."; return; }
    try {
      await action("/api/v1/research-org/scheduler", { action: actionName, cadence_seconds: cadence }, "Updating the scheduler…");
      $("observatory-status").textContent = "Scheduler command accepted. Current worker state is shown in the readout above.";
    } catch { $("observatory-status").textContent = "ERROR · Scheduler update was not confirmed. Refresh the ledger before retrying."; truth(null, true); }
  }

  async function objective(event) {
    event.preventDefault();
    try {
      const payload = await action("/api/v1/research-org/objective", { objective: $("observatory-objective-text").value }, "Recording the objective…");
      $("observatory-status").textContent = payload.data.objective.disposition;
    } catch { $("observatory-status").textContent = "ERROR · Objective was not confirmed. No new approval is inferred."; truth(null, true); }
  }

  async function director(actionName) {
    $("director-approve").disabled = true; $("director-revoke").disabled = true;
    $("director-status").textContent = "Recording the approval decision…";
    try {
      const response = await fetch("/api/v1/research-org/director", { method: "POST", headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ action: actionName, objective_id: OBJECTIVE_ID }) });
      if (!response.ok) throw new Error("unavailable");
      await load();
      $("director-status").textContent = actionName === "approve" ? "Approval recorded. Start the worker separately to run approved investigations." : "Approval revoked. Inspect active work before starting another cycle.";
    } catch { $("director-status").textContent = "ERROR · Approval decision was not confirmed. Refresh the ledger before retrying."; }
  }

  function show(on) {
    const panel = $("observatory-panel"); if (!panel) return;
    state.visible = on; panel.classList.toggle("hidden", !on);
    clearTimeout(state.timer); state.timer = null;
    if (!on) return;
    if (!state.started) {
      state.started = true;
      $("observatory-run").addEventListener("click", tick);
      $("observatory-objective").addEventListener("submit", objective);
      $("observatory-filter").addEventListener("input", renderHistory);
      $("observatory-latest").addEventListener("click", () => { state.followLatest = true; if (state.payload) render(state.payload); });
      $("director-approve").addEventListener("click", () => director("approve"));
      $("director-revoke").addEventListener("click", () => director("revoke"));
      document.querySelectorAll("[data-scheduler]").forEach((button) => button.addEventListener("click", () => scheduler(button.dataset.scheduler)));
    }
    $("state-badge").textContent = "LOADING";
    load();
  }

  window.GIGObservatory = { show, state: () => (state.payload ? state.payload.data : null) };
})();
