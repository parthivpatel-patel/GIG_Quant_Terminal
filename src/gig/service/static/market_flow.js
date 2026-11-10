/* GIG Market Flow: source-access truth and explicit authored options research.
 * No quotes, prints, institutional activity or positions are invented. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const state = { started: false, visible: false, payload: null, busy: false };
  const node = (tag, className, text) => {
    const result = document.createElement(tag);
    if (className) result.className = className;
    if (text !== undefined) result.textContent = text;
    return result;
  };
  const num = (value) => typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString("en-US", { maximumFractionDigits: 3 }) : "UNKNOWN";
  const sample = () => ({
    source: "USER_AUTHORED_ASSUMPTIONS", classification: "SYNTHETIC",
    valuation_at: "2026-01-01T00:00:00Z", underlying_id: "HYPOTHETICAL:DESK", currency: "USD",
    spot: 100, rate: 0, dividend_yield: 0,
    contracts: [{ contract_id: "HYPOTHETICAL:C100", strike: 100, right: "CALL",
      expiry: "2027-01-01T00:00:00Z", quantity: 1, multiplier: 100, volatility: 0.2 }],
    spot_shocks: [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3],
    limits: { max_abs_delta: 50, max_abs_vega: 50, max_scenario_loss: 500 },
  });


  function build() {
    const root = $("market-flow-panel"); root.classList.add("flow-desk");
    root.append(node("p", "flow-kicker", "GIG / Financial intelligence / Evidence first"),
      node("h2", null, "Market Flow Intelligence"),
      node("p", "flow-note", "Observed flow is unavailable. Inspect source-access evidence, or run an explicitly authored European options scenario. Local model alerts do not identify institutional trades."));
    const grid = node("div", "flow-grid");
    const rail = node("div", "flow-rail");
    rail.append(node("h3", null, "Source admission register"));
    const registry = node("div"); registry.id = "flow-capabilities"; rail.append(registry);
    const research = node("div");
    research.append(node("p", "flow-kicker", "Analyst-authored / Model estimate"), node("h3", null, "Scenario & threshold laboratory"),
      node("p", "flow-note", "Inputs are hypothetical, not an admitted market dataset or an OMS position. European cash settlement, one underlying and common expiry only."));
    const label = node("label", null, "Explicit authored assumptions (JSON)"); label.htmlFor = "flow-inputs";
    const editor = node("textarea", "flow-editor"); editor.id = "flow-inputs"; editor.spellcheck = false;
    editor.placeholder = "Load the authored example, or enter your bounded assumptions. No model runs until requested.";
    research.append(label, editor);
    const actions = node("div", "flow-actions");
    const authored = node("button", null, "Load authored example"); authored.id = "flow-example"; authored.type = "button";
    authored.addEventListener("click", () => { editor.value = JSON.stringify(sample(), null, 2); $("flow-status").textContent = "Authored hypothetical example loaded. No observations or positions were fetched."; });
    const evaluate = node("button", "primary-button", "Evaluate model"); evaluate.id = "flow-evaluate"; evaluate.type = "button";
    evaluate.addEventListener("click", () => load(true));
    const download = node("button", null, "Export evidence JSON"); download.id = "flow-export"; download.type = "button"; download.disabled = true;
    download.addEventListener("click", () => {
      if (!state.payload) return;
      const url = URL.createObjectURL(new Blob([JSON.stringify(state.payload, null, 2)], { type: "application/json" }));
      const link = node("a"); link.href = url; link.download = "gig-authored-options-evidence.json"; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    actions.append(authored, evaluate, download); research.append(actions);
    const status = node("p", "flow-status"); status.id = "flow-status"; status.setAttribute("role", "status"); status.setAttribute("aria-live", "polite");
    const output = node("div"); output.id = "flow-output";
    research.append(status, output); grid.append(rail, research); root.append(grid);
  }

  function truth(payload, failed = false) {
    if (!state.visible) return;
    const classification = payload?.provenance?.classification || "UNAVAILABLE";
    $("state-badge").textContent = failed ? "ERROR" : "AVAILABLE";
    $("state-badge").dataset.state = failed ? "ERROR" : "AVAILABLE";
    $("source-label").textContent = failed ? "market-flow · request failed" : `market-flow · ${classification} · observed flow unavailable`;
    $("asof-label").textContent = "No observed market timestamp";
    $("truth-notice").textContent = "Hypothetical assumptions → model estimates · No options prints or dealer positioning · No orders";
  }

  function table(headers, rows, caption) {
    const wrapper = node("div", "flow-table-wrap");
    const t = node("table", "flow-table"); t.append(node("caption", null, caption));
    const head = node("thead"), tr = node("tr");
    headers.forEach((value) => { const th = node("th", null, value); th.scope = "col"; tr.append(th); });
    head.append(tr); const body = node("tbody");
    rows.forEach((values) => { const row = node("tr"); values.forEach((value) => row.append(node("td", null, value))); body.append(row); });
    t.append(head, body); wrapper.append(t); return wrapper;
  }

  function chart(analysis) {
    const rows = analysis.scenarios.scenarios;
    const terminal = analysis.payoff;
    const values = rows.map((r) => r.model_pnl).concat(terminal.map((r) => r.undiscounted_premium_difference));
    const low = Math.min(0, ...values), high = Math.max(0, ...values), span = high - low || 1;
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg"); svg.setAttribute("viewBox", "0 0 640 240");
    svg.setAttribute("class", "flow-chart"); svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Authored options scenario: solid line is current model change; dashed line is expiry payoff less unfinanced premium. Exact values follow in the table.");
    const y = (value) => 205 - ((value - low) / span) * 180;
    const make = (tag, attrs) => { const n = document.createElementNS("http://www.w3.org/2000/svg", tag); Object.entries(attrs).forEach(([key, value]) => n.setAttribute(key, value)); svg.append(n); return n; };
    make("line", { x1: 45, y1: y(0), x2: 620, y2: y(0), stroke: "var(--border)" });
    const firstSpot = rows[0].spot, spotSpan = rows[rows.length - 1].spot - firstSpot;
    const x = (i) => 45 + ((rows[i].spot - firstSpot) / spotSpan) * 575;
    [rows.map((r) => r.model_pnl), terminal.map((r) => r.undiscounted_premium_difference)].forEach((series, i) => {
      make("polyline", { points: series.map((value, index) => `${x(index)},${y(value)}`).join(" "), fill: "none", stroke: i ? "var(--muted)" : "var(--accent)", "stroke-width": "2", "stroke-dasharray": i ? "5 4" : "none" });
    });
    [low, 0, high].forEach((value) => { make("text", { x: 4, y: y(value)-4, fill: "currentColor", "font-size": "10" }).textContent = num(value); });
    make("text", { x: 45, y: 230, fill: "currentColor", "font-size": "11" }).textContent = `Spot ${num(rows[0].spot)} → ${num(rows[rows.length-1].spot)} · USD model differences`;
    return svg;
  }

  function render(payload) {
    const data = payload.data;
    const registry = $("flow-capabilities"); registry.replaceChildren();
    data.capabilities.forEach((item) => {
      const details = node("details"), summary = node("summary", null, `${item.name} · ${item.state}`);
      details.append(summary, node("p", "flow-note", item.reason));
      const dl = node("dl");
      [["Provider", item.provider], ["Cost", item.cost], ["Access terms", item.access_terms], ["Delay", item.delay],
        ["Historical coverage", item.historical_coverage], ["Redistribution", item.redistribution]].forEach(([k, v]) => dl.append(node("dt", null, k), node("dd", null, v)));
      details.append(dl);
      if (item.official_url) { const link = node("a", null, "Official source documentation"); link.href = item.official_url; link.target = "_blank"; link.rel = "noopener noreferrer"; details.append(link); }
      registry.append(details);
    });
    const output = $("flow-output"); output.replaceChildren();
    const a = data.analysis;
    $("flow-export").disabled = !a;
    if (!a) { output.append(node("p", "flow-note", "No scenario has been evaluated. Source evidence above is a capability register, not live market data.")); return; }
    const metrics = node("div", "flow-metrics");
    [["Model PV · USD", a.exposure.price], ["Delta · underlying units", a.exposure.delta], ["Vega · USD / vol point", a.exposure.vega],
      ["Gamma · units / USD", a.exposure.gamma], ["Theta · USD / day", a.exposure.theta], ["Rho · USD / rate point", a.exposure.rho]].forEach(([label, value]) => {
      const cell = node("div", "flow-metric"); cell.append(node("strong", null, num(value)), node("span", null, label)); metrics.append(cell);
    });
    output.append(metrics, node("h3", null, "Authored scenario response"), chart(a));
    output.append(table(["Spot assumption", "Current model ΔPV · USD", "Expiry payoff − unfinanced premium · USD"],
      a.scenarios.scenarios.map((r, i) => [num(r.spot), num(r.model_pnl), num(a.payoff[i].undiscounted_premium_difference)]), "Scenario values · modeled, not realized P&L"));
    a.alerts.forEach((alert) => output.append(node("p", alert.state === "BREACH" ? "flow-limit-breach" : "flow-limit-ok",
      `${alert.state} · ${alert.id} · ${num(alert.modeled_value)} / threshold ${num(alert.threshold)}`)));
    a.assumptions.forEach((text) => output.append(node("p", "flow-note", text)));
    output.append(node("p", "flow-note", `Evidence identity: ${a.inputs_sha256} · ${data.version}. Export retains inputs, methods, access limitations and canonical envelope hash.`));
  }

  async function load(evaluate = false) {
    if (state.busy) return;
    state.busy = true; $("flow-evaluate").disabled = true;
    state.payload = null; $("flow-export").disabled = true;
    $("flow-output").replaceChildren(node("p", "flow-note", "Awaiting a verified response. Previous model results are cleared."));
    $("flow-status").textContent = evaluate ? "Evaluating authored assumptions on local CPU…" : "Loading source-access evidence…";
    try {
      const options = evaluate ? { method: "POST", headers: { "Content-Type": "application/json", Accept: "application/json" }, body: JSON.stringify(JSON.parse($("flow-inputs").value)) }
        : { headers: { Accept: "application/json" } };
      const response = await fetch("/api/v1/market-flow", options);
      const payload = await response.json();
      if (!response.ok || payload.state !== "AVAILABLE" || !payload.data) throw new Error("rejected");
      state.payload = payload; render(payload); truth(payload);
      $("flow-status").textContent = evaluate ? "Model completed. No market observations, broker positions or execution approvals were created." : "Access register loaded. No observed flow feed is connected.";
    } catch {
      state.payload = null; $("flow-output").replaceChildren(); $("flow-export").disabled = true;
      $("flow-status").textContent = "Request unavailable or assumptions rejected. Check explicit bounded USD European inputs and retry; previous model results have been cleared.";
      truth(null, true);
    } finally { state.busy = false; $("flow-evaluate").disabled = false; }
  }

  window.GIGMarketFlow = {
    show(on) { state.visible = on; $("market-flow-panel").classList.toggle("hidden", !on); if (!on) return; if (!state.started) { build(); state.started = true; load(); } else truth(state.payload, !state.payload); },
    reload: () => load(),
    state: () => ({ visible: state.visible, busy: state.busy, analysis: Boolean(state.payload?.data?.analysis),
      observedFlow: "UNAVAILABLE", identity: state.payload?.data?.analysis?.inputs_sha256 || null }),
  };
})();
