/* GIG Market Flow: source-access truth and explicit authored options research.
 * No quotes, prints, institutional activity or positions are invented. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const state = { started: false, visible: false, payload: null, busy: false, positioning: null, positionIndex: 0, positioningBusy: false, positionMarket: "euro-fx" };
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
      node("p", "flow-note", "Inspect observed weekly CFTC futures positioning and explicit options model assumptions. Aggregate positions are not trades, sweeps, dark-pool activity or executable liquidity."));
    const nav = node("div", "flow-workspace-nav");
    [["Observed positioning", "flow-positioning-panel"], ["Authored options models", "flow-model-laboratory"]].forEach(([label, id]) => { const button = node("button", "text-button", label); button.type = "button"; button.setAttribute("aria-controls", id); button.addEventListener("click", () => { $(id).scrollIntoView({ block: "start", behavior: "auto" }); $(id).focus({ preventScroll: true }); }); nav.append(button); }); root.append(nav);
    const positioning = node("section", "flow-positioning-panel"); positioning.id = "flow-positioning-panel"; positioning.tabIndex = -1; positioning.setAttribute("aria-labelledby", "flow-positioning-title");
    const heading = node("div", "flow-positioning-heading"), title = node("div"); title.append(node("p", "flow-kicker", "01 / OBSERVED AGGREGATE POSITIONING"));
    const h3 = node("h3", null, "Commitments, in context."); h3.id = "flow-positioning-title"; title.append(h3); heading.append(title);
    const reload = node("button", "text-button", "Reload saved evidence"); reload.type = "button"; reload.id = "flow-positioning-refresh"; reload.addEventListener("click", loadPositioning); heading.append(reload); positioning.append(heading);
    const scope = node("div", "flow-market-scope");
    const scopeLabel = node("label", null, "Official reported market"); scopeLabel.htmlFor = "flow-positioning-market";
    const market = node("select"); market.id = "flow-positioning-market";
    [["euro-fx", "Euro FX / currency futures"], ["equity-index", "E-mini S&P 500 / equity-index futures"]].forEach(([value, label]) => { const option = node("option", null, label); option.value = value; market.append(option); });
    market.addEventListener("change", () => { state.positionMarket = market.value; loadPositioning(); });
    scope.append(scopeLabel, market, node("p", "flow-note", "Distinct official contract identities. These reports describe weekly positions, not cash equities, transaction flow or options activity.")); positioning.append(scope);
    const status = node("p", "flow-note", "Loading admitted positioning evidence…"); status.id = "flow-positioning-status"; status.setAttribute("role", "status"); positioning.append(status);
    const content = node("div"); content.id = "flow-positioning-content"; positioning.append(content); root.append(positioning);
    const grid = node("div", "flow-grid");
    const rail = node("div", "flow-rail");
    rail.append(node("h3", null, "Source admission register"));
    const registry = node("div"); registry.id = "flow-capabilities"; rail.append(registry);
    const research = node("div"); research.id = "flow-model-laboratory"; research.tabIndex = -1;
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
    const modelStatus = node("p", "flow-status"); modelStatus.id = "flow-status"; modelStatus.setAttribute("role", "status"); modelStatus.setAttribute("aria-live", "polite");
    const output = node("div"); output.id = "flow-output";
    research.append(modelStatus, output); grid.append(rail, research); root.append(grid);
  }

  function truth(payload, failed = false) {
    if (!state.visible) return;
    const classification = payload?.provenance?.classification || "UNAVAILABLE";
    $("state-badge").textContent = failed ? "ERROR" : "AVAILABLE";
    $("state-badge").dataset.state = failed ? "ERROR" : "AVAILABLE";
    $("source-label").textContent = failed ? "market-flow · request failed" : `market-flow · options ${classification} · weekly positioning ${state.positioning ? "HISTORICAL" : "UNAVAILABLE"}`;
    $("asof-label").textContent = state.positioning ? `CFTC report ${state.positioning.freshness.latest_report_date} · historical sample, STALE for current use` : "No observed market timestamp";
    $("truth-notice").textContent = "Observed weekly futures positions ≠ trade flow · Options scenarios are models · No options prints, dealer positioning or orders";
  }

  function table(headers, rows, caption, tableClass = "flow-table") {
    const wrapper = node("div", "flow-table-wrap");
    const t = node("table", tableClass); t.append(node("caption", null, caption));
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


  const CATEGORY_NAMES = { dealer: "Dealer / intermediary", asset_manager: "Asset manager / institutional", leveraged_money: "Leveraged funds", other_reportable: "Other reportable", nonreportable: "Nonreportable" };
  const category = (row, id) => (row.categories || []).find((item) => item.id === id);
  const percent = (value) => typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "UNKNOWN";
  function drawPositioning() {
    const data = state.positioning;
    if (!data) return;
    const rows = data.rows, svg = $("flow-positioning-chart"); svg.replaceChildren();
    const series = [["asset_manager", "Asset managers", "var(--accent)"], ["leveraged_money", "Leveraged funds", "#e1b66a"]];
    const values = rows.flatMap((row) => series.map(([id]) => category(row, id)?.net)).filter(Number.isFinite);
    if (!values.length) return;
    const low = Math.min(0, ...values), high = Math.max(0, ...values), span = high - low || 1;
    const x = (index) => 72 + index / Math.max(1, rows.length - 1) * 750, y = (value) => 225 - (value - low) / span * 185;
    const make = (tag, attributes, text) => { const item = document.createElementNS("http://www.w3.org/2000/svg", tag); Object.entries(attributes).forEach(([key, value]) => item.setAttribute(key, String(value))); if (text !== undefined) item.textContent = text; svg.append(item); return item; };
    make("line", { x1: 72, y1: y(0), x2: 822, y2: y(0), stroke: "var(--border)" });
    [low, high].forEach((value) => make("text", { x: 4, y: y(value), fill: "currentColor", "font-size": 11 }, num(value)));
    series.forEach(([id, label, color]) => {
      let path = "", open = false;
      rows.forEach((row, index) => { const value = category(row, id)?.net; if (!Number.isFinite(value)) { open = false; return; } path += `${open ? "L" : "M"}${x(index)},${y(value)} `; open = true; });
      make("path", { d: path, fill: "none", stroke: color, "stroke-width": 2 });
    });
    make("line", { x1: x(state.positionIndex), x2: x(state.positionIndex), y1: 30, y2: 230, stroke: "var(--muted)", "stroke-dasharray": "3 4" });
    make("text", { x: 72, y: 255, fill: "currentColor", "font-size": 11 }, rows[0].report_date);
    make("text", { x: 740, y: 255, fill: "currentColor", "font-size": 11 }, rows[rows.length - 1].report_date);
    const row = rows[state.positionIndex];
    $("flow-positioning-cursor").value = String(state.positionIndex);
    $("flow-positioning-date").textContent = `${row.report_date} / ${num(row.open_interest)} futures contracts open interest`;
    $("flow-positioning-date").setAttribute("data-report-date", row.report_date);
    const inspector = $("flow-positioning-inspector"); inspector.replaceChildren();
    inspector.append(table(["Reported category", "Long", "Short", "Net", "Net / OI", "Weekly net change", "Prior-only percentile"],
      (row.categories || []).map((item) => [CATEGORY_NAMES[item.id] || item.id, num(item.long), num(item.short), num(item.net), percent(item.net_share_of_open_interest), num(item.weekly_net_change), item.prior_only_percentile === null ? "UNKNOWN / insufficient history" : percent(item.prior_only_percentile)]),
      `Reported positions at ${row.report_date} · contracts, not transaction flows`, "flow-positioning-table"));
  }
  function renderPositioning() {
    const data = state.positioning, content = $("flow-positioning-content"); content.replaceChildren();
    if (!data || !Array.isArray(data.rows) || !data.rows.length) return;
    state.positionIndex = Math.max(0, Math.min(data.rows.length - 1, state.positionIndex));
    const source = data.source || {}, instrument = data.instrument || {};
    const banner = node("div", "flow-positioning-identity");
    banner.append(node("strong", null, instrument.market_name || "CFTC reported instrument"), node("span", null, `Contract ${instrument.contract_market_code || "UNKNOWN"} / ${instrument.contract_units || "UNKNOWN units"}`), node("span", "flow-stale-tag", `${data.classification || "HISTORICAL"} · ${data.freshness?.state || "UNKNOWN"} for current use`)); content.append(banner);
    const legend = node("div", "flow-positioning-legend"); legend.append(node("span", "asset", "Asset manager net contracts"), node("span", "leveraged", "Leveraged fund net contracts")); content.append(legend);
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg"); svg.id = "flow-positioning-chart"; svg.setAttribute("viewBox", "0 0 850 275"); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "Historical weekly CFTC net positions for asset managers and leveraged funds; exact selected report values are available in the table below."); content.append(svg);
    const cursorLabel = node("label", "flow-positioning-range-label", "Inspect a reported week"); cursorLabel.htmlFor = "flow-positioning-cursor";
    const cursor = node("input"); cursor.id = "flow-positioning-cursor"; cursor.type = "range"; cursor.min = "0"; cursor.max = String(data.rows.length - 1); cursor.step = "1";
    cursor.addEventListener("input", () => { state.positionIndex = Number(cursor.value); drawPositioning(); });
    const date = node("p", "flow-positioning-date"); date.id = "flow-positioning-date"; date.setAttribute("aria-live", "polite");
    const inspector = node("div"); inspector.id = "flow-positioning-inspector"; content.append(cursorLabel, cursor, date, inspector);
    const details = node("details", "flow-source-inspector"), summary = node("summary", null, "Inspect provider, permission and knowledge clocks"); details.append(summary);
    const dl = node("dl");
    [["Source", source.provider], ["Dataset", source.dataset_name || source.dataset_id], ["Acquired", source.acquired_at], ["Publication time", source.publication_time || "UNKNOWN"], ["Historical knowledge", source.historical_knowledge], ["Source SHA-256", source.raw_sha256], ["Permission evidence", source.license_url], ["Attribution", source.attribution]].forEach(([key, value]) => dl.append(node("dt", null, key), node("dd", null, value || "UNKNOWN"))); details.append(dl); content.append(details);
    const notes = node("div", "flow-positioning-notes"); (data.notes || []).forEach((text) => notes.append(node("p", "flow-note", text))); content.append(notes);
    svg.addEventListener("pointermove", (event) => { const box = svg.getBoundingClientRect(), axis = ((event.clientX - box.left) / box.width * 850 - 72) / 750; state.positionIndex = Math.max(0, Math.min(data.rows.length - 1, Math.round(axis * (data.rows.length - 1)))); drawPositioning(); });
    drawPositioning();
  }
  async function loadPositioning() {
    if (state.positioningBusy) return;
    state.positioningBusy = true; $("flow-positioning-refresh").disabled = true; $("flow-positioning-market").disabled = true;
    state.positioning = null; $("flow-positioning-content").replaceChildren();
    $("flow-positioning-status").textContent = "Loading saved official positioning evidence. No provider request is made by this page.";
    try {
      const endpoint = state.positionMarket === "equity-index" ? "/api/v1/market-flow/equity-positioning" : "/api/v1/market-flow/positioning";
      const response = await fetch(endpoint, { headers: { Accept: "application/json" } });
      const payload = await response.json();
      if (!response.ok || !["AVAILABLE", "STALE"].includes(payload.state) || !payload.data || !Array.isArray(payload.data.rows) || !payload.data.rows.length) throw new Error("unavailable");
      state.positioning = payload.data; state.positionIndex = payload.data.rows.length - 1; renderPositioning();
      $("flow-positioning-status").textContent = `${payload.data.rows.length} historical weekly reports · publication and revision clocks remain UNKNOWN · not current positions or observed trade flow`;
    } catch {
      state.positioning = null; $("flow-positioning-content").replaceChildren();
      $("flow-positioning-status").textContent = "UNAVAILABLE · Verified positioning evidence could not be loaded. No positions or trades have been fabricated.";
    } finally { state.positioningBusy = false; $("flow-positioning-refresh").disabled = false; $("flow-positioning-market").disabled = false; if (state.visible) truth(state.payload, !state.payload); }
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
    show(on) { state.visible = on; $("market-flow-panel").classList.toggle("hidden", !on); if (!on) return; if (!state.started) { build(); state.started = true; load(); loadPositioning(); } else truth(state.payload, !state.payload); },
    reload: () => { load(); loadPositioning(); },
    state: () => ({ visible: state.visible, busy: state.busy, analysis: Boolean(state.payload?.data?.analysis),
      observedFlow: "UNAVAILABLE", positioningMarket: state.positionMarket, positioningAvailable: Boolean(state.positioning), positioningReports: state.positioning?.rows?.length || 0, positioningIndex: state.positionIndex, identity: state.payload?.data?.analysis?.inputs_sha256 || null }),
  };
})();
