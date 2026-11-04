/* GIG 3D Analytics Lab: six analytical modules on the shared engine.
 * Data comes only from /api/v1/viz/<module>. Local mode without admitted evidence
 * shows UNAVAILABLE; synthetic payloads are labeled as authored fixtures. */
"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const G = () => window.GIG3D;
  const MODULES = [
    { id: "vol-surface", label: "Volatility surface" },
    { id: "order-book", label: "Order book" },
    { id: "risk-terrain", label: "Risk terrain" },
    { id: "correlation-network", label: "Correlation network" },
    { id: "stat-arb", label: "Stat-arb landscape" },
    { id: "model-evaluation", label: "Model evaluation" },
  ];
  const DEFAULTS = { options: "vol-surface", markets: "order-book", execution: "order-book", risk: "risk-terrain", portfolio: "risk-terrain",
    signals: "correlation-network", research: "correlation-network", "command-center": "correlation-network", "stat-arb": "stat-arb", models: "model-evaluation" };
  const state = { module: "vol-surface", mode: "synthetic", cache: new Map(), engine: null, filters: {}, rows: [], columns: [], selected: null, request: 0, visible: false };
  const el = (tag, cls, text) => { const node = document.createElement(tag); if (cls) node.className = cls; if (text !== undefined) node.textContent = text; return node; };
  const fmt = (v, d) => G().format(v, d);

  function setStatus(message, kind = "info") { const s = $("lab-status"); s.textContent = message; s.dataset.kind = kind; s.classList.toggle("hidden", !message); }
  function details(entries) {
    const list = $("lab-details"); list.replaceChildren();
    entries.forEach(([k, v]) => { list.append(el("dt", null, k), el("dd", null, typeof v === "number" ? fmt(v, 4) : v ?? "UNKNOWN")); });
  }
  function control(kind, label, options, value, onChange) {
    const wrap = el("label", "lab-control"); wrap.append(el("span", null, label));
    let input;
    if (kind === "select") {
      input = el("select"); options.forEach(([v, text]) => { const o = el("option", null, text); o.value = v; input.append(o); }); input.value = value;
    } else { input = el("input"); input.type = "range"; Object.assign(input, options); input.value = value; }
    input.addEventListener(kind === "select" ? "change" : "input", () => { onChange(input.value); render(); });
    wrap.append(input); $("lab-filters").append(wrap); return input;
  }
  function table(columns, rows) {
    state.columns = columns; state.rows = rows;
    const head = el("thead"); const tr = el("tr"); columns.forEach(([, label]) => { const th = el("th", null, label); th.scope = "col"; tr.append(th); }); head.append(tr);
    const body = el("tbody"); rows.slice(0, 600).forEach((row) => { const r = el("tr"); columns.forEach(([key]) => r.append(el("td", null, typeof row[key] === "number" ? fmt(row[key], 5) : row[key] ?? "UNKNOWN"))); body.append(r); });
    $("lab-table").replaceChildren(el("caption", "sr-only", "Exact values behind the 3D view"), head, body);
    $("lab-table-count").textContent = `${rows.length} rows${rows.length > 600 ? " (first 600 shown)" : ""}`;
  }
  function minmax(values) { const f = values.filter(Number.isFinite); return [Math.min(...f), Math.max(...f)]; }

  const renderers = {
    "vol-surface"(data) {
      const f = state.filters; f.snapshot ??= "current"; f.compare ??= "on"; f.type ??= "all";
      const grid = data.grid; const pts = data.snapshots[f.snapshot];
      const all = Object.values(data.snapshots).flat().map((p) => p.implied_vol);
      const [zmin, zmax] = minmax(all); const sx = G().scaler(Math.min(...grid.x), Math.max(...grid.x)); const sy = G().scaler(0, Math.max(...grid.y)); const sz = G().scaler(zmin, zmax);
      const at = (snap, i, j) => data.snapshots[snap][j * grid.x.length + i];
      const keep = (p) => f.type === "all" || p.option_type === f.type;
      const primitives = [];
      for (let j = 0; j < grid.y.length - 1; j += 1) for (let i = 0; i < grid.x.length - 1; i += 1) {
        const c = [at(f.snapshot, i, j), at(f.snapshot, i + 1, j), at(f.snapshot, i + 1, j + 1), at(f.snapshot, i, j + 1)];
        if (c.some((p) => p.implied_vol === null) || !c.every(keep)) continue;
        const mean = c.reduce((s, p) => s + p.implied_vol, 0) / 4;
        primitives.push({ type: "quad", pts: c.map((p) => [sx.to(p.moneyness), sy.to(p.tenor_years), sz.to(p.implied_vol)]), color: G().ramp("sequential", (mean - zmin) / (zmax - zmin)), stroke: "#0c1016", data: c[0], alpha: 0.95 });
      }
      if (f.compare === "on") {
        const other = f.snapshot === "current" ? "prior" : "current";
        for (let j = 0; j < grid.y.length; j += 1) primitives.push({ type: "line", pts: grid.x.map((_, i) => { const p = at(other, i, j); return [sx.to(p.moneyness), sy.to(p.tenor_years), sz.to(p.implied_vol)]; }), color: "#f2be4c", width: 1, dash: [4, 3], alpha: 0.85 });
      }
      pts.filter(keep).forEach((p) => primitives.push({ type: "point", pts: [[sx.to(p.moneyness), sy.to(p.tenor_years), sz.to(p.implied_vol)]], color: "#f2f5f9", radius: 1.6, data: p, alpha: 0.7 }));
      return { primitives, axes: { x: { ...data.axes.x, scale: sx }, y: { ...data.axes.y, scale: sy }, z: { ...data.axes.z, scale: sz } },
        legend: { scale: "sequential", min: zmin, max: zmax, label: "Implied volatility", unit: "annualized" },
        columns: [["strike", "Strike"], ["tenor_years", "Maturity (y)"], ["option_type", "Type"], ["authored_price", "Authored price"], ["implied_vol", "Implied vol"], ["vega", "Vega"]], rows: pts,
        describe: (p) => [["Strike", p.strike], ["Log-moneyness", p.moneyness], ["Maturity (years)", p.tenor_years], ["Type", p.option_type], ["Implied vol", p.implied_vol], ["Vega", p.vega], ["Snapshot", f.snapshot]],
        filters() {
          control("select", "Snapshot", [["current", "Current"], ["prior", "Prior (comparison)"]], f.snapshot, (v) => { f.snapshot = v; });
          control("select", "Overlay other snapshot", [["on", "Shown (dashed yellow)"], ["off", "Hidden"]], f.compare, (v) => { f.compare = v; });
          control("select", "Contracts", [["all", "Calls and puts"], ["CALL", "Calls (K ≥ S)"], ["PUT", "Puts (K < S)"]], f.type, (v) => { f.type = v; });
        } };
    },
    "order-book"(data) {
      const f = state.filters; f.side ??= "both"; f.levels ??= "10";
      const frames = data.frames; const levels = Number(f.levels);
      const prices = frames.flatMap((fr) => [...fr.bids.slice(0, levels), ...fr.asks.slice(0, levels)].map((l) => l.price));
      const qty = frames.flatMap((fr) => [...fr.bids, ...fr.asks].slice(0).map((l) => l.quantity));
      const sx = G().scaler(Math.min(...prices), Math.max(...prices)); const sy = G().scaler(0, frames.length - 1); const sz = G().scaler(0, Math.max(...qty));
      const primitives = []; const rows = [];
      frames.forEach((fr) => {
        [["bids", "#16c784", "#0f8a5c"], ["asks", "#f05259", "#a8383e"]].forEach(([side, color, dark]) => {
          if (f.side !== "both" && f.side !== side) return;
          fr[side].slice(0, levels).forEach((l, rank) => {
            const datum = { side: side === "bids" ? "BID" : "ASK", price: l.price, quantity: l.quantity, rank: rank + 1, step: fr.step, observed_at: fr.observed_at, classification: data.layers.displayed_depth };
            rows.push(datum);
            primitives.push({ type: "bar", box: [sx.to(l.price), sy.to(fr.step), -1, sz.to(l.quantity)], half: 0.022, color, side: dark, data: datum, alpha: 0.92 });
          });
        });
        const mid = (fr.bids[0].price + fr.asks[0].price) / 2;
        primitives.push({ type: "point", pts: [[sx.to(mid), sy.to(fr.step), sz.to(Math.max(...qty)) * 0 + 1.05]], color: "#408bff", radius: 1.5 + fr.trade_prints / 3, data: { kind: "Authored print count", step: fr.step, trade_prints: fr.trade_prints, displayed_imbalance: fr.displayed_imbalance, classification: data.layers.trade_activity } });
      });
      return { primitives, axes: { x: { ...data.axes.x, scale: sx }, y: { ...data.axes.y, scale: sy }, z: { ...data.axes.z, scale: sz } },
        legend: { scale: "pnl", min: "ASK (red)", max: "BID (green)", label: "Side; blue dots = authored print counts", unit: "" },
        columns: [["step", "Step"], ["observed_at", "Observed"], ["side", "Side"], ["rank", "Level"], ["price", "Price"], ["quantity", "Displayed qty"]], rows,
        describe: (d) => Object.entries(d).map(([k, v]) => [k.replaceAll("_", " "), v]),
        filters() {
          control("select", "Side", [["both", "Bids and asks"], ["bids", "Bids only"], ["asks", "Asks only"]], f.side, (v) => { f.side = v; });
          control("range", "Levels per side", { min: 1, max: 15, step: 1 }, f.levels, (v) => { f.levels = v; });
        } };
    },
    "risk-terrain"(data) {
      const f = state.filters; f.horizon ??= "today"; f.limit ??= "on"; f.compare ??= "on";
      const grid = data.grid; const nx = grid.x.length;
      const all = Object.values(data.horizons).flat().map((p) => p.model_pnl).concat([data.risk_limit.loss_limit]);
      const [zmin, zmax] = minmax(all); const span = Math.max(Math.abs(zmin), Math.abs(zmax));
      const [pmin, pmax] = minmax(Object.values(data.horizons).flat().map((p) => p.model_pnl));
      const sx = G().scaler(grid.x[0], grid.x[nx - 1]); const sy = G().scaler(grid.y[0], grid.y[grid.y.length - 1]); const sz = G().scaler(zmin, zmax);
      const at = (h, i, j) => data.horizons[h][i * grid.y.length + j];
      const primitives = [];
      for (let i = 0; i < nx - 1; i += 1) for (let j = 0; j < grid.y.length - 1; j += 1) {
        const c = [at(f.horizon, i, j), at(f.horizon, i + 1, j), at(f.horizon, i + 1, j + 1), at(f.horizon, i, j + 1)];
        const mean = c.reduce((s, p) => s + p.model_pnl, 0) / 4;
        const breach = c.some((p) => p.model_pnl < data.risk_limit.loss_limit);
        primitives.push({ type: "quad", pts: c.map((p) => [sx.to(p.spot_shock), sy.to(p.volatility_shock), sz.to(p.model_pnl)]), color: G().ramp("pnl", 0.5 + mean / (2 * span)), stroke: breach ? "#f2be4c" : "#0c1016", data: { ...c[0], breaches_limit: breach }, alpha: 0.94 });
      }
      if (f.limit === "on") { const z = sz.to(data.risk_limit.loss_limit); primitives.push({ type: "quad", pts: [[-1, -1, z], [1, -1, z], [1, 1, z], [-1, 1, z]], color: "#f2be4c", alpha: 0.16, stroke: "#f2be4c" }); }
      if (f.compare === "on") {
        const other = f.horizon === "today" ? "plus_30_days" : "today";
        for (let j = 0; j < grid.y.length; j += 2) primitives.push({ type: "line", pts: grid.x.map((_, i) => { const p = at(other, i, j); return [sx.to(p.spot_shock), sy.to(p.volatility_shock), sz.to(p.model_pnl)]; }), color: "#408bff", dash: [4, 3], alpha: 0.9 });
      }
      return { primitives, axes: { x: { ...data.axes.x, scale: sx }, y: { ...data.axes.y, scale: sy }, z: { ...data.axes.z, scale: sz } },
        legend: { scale: "pnl", min: pmin, max: pmax, range: [0.5 + pmin / (2 * span), 0.5 + pmax / (2 * span)],
          zero: pmin < 0 && pmax > 0 ? -pmin / (pmax - pmin) : null,
          label: "Model P&L range, both horizons (red loss, green gain, tick = 0); yellow edge = beyond limit", unit: "USD" },
        columns: [["spot_shock", "Spot shock"], ["volatility_shock", "Vol shock"], ["model_pnl", "Model P&L (USD)"]], rows: data.horizons[f.horizon],
        describe: (p) => [["Spot shock", p.spot_shock], ["Vol shock", p.volatility_shock], ["Model P&L (USD)", p.model_pnl], ["Beyond loss limit", p.breaches_limit ? "YES" : "NO"], ["Horizon", f.horizon.replaceAll("_", " ")]],
        extra: [["Loss limit (USD)", fmt(data.risk_limit.loss_limit, 2)], ...data.concentration.map((c) => [`Premium share · ${c.instrument}`, `${(c.share * 100).toFixed(1)}%`])],
        filters() {
          control("select", "Horizon", [["today", "Today"], ["plus_30_days", "+30 days (time decay)"]], f.horizon, (v) => { f.horizon = v; });
          control("select", "Limit plane", [["on", "Shown"], ["off", "Hidden"]], f.limit, (v) => { f.limit = v; });
          control("select", "Compare horizon", [["on", "Dashed blue overlay"], ["off", "Hidden"]], f.compare, (v) => { f.compare = v; });
        } };
    },
    "correlation-network"(data) {
      const f = state.filters; f.window ??= "latest_60"; f.threshold ??= "0.25";
      const w = data.windows[f.window]; const threshold = Number(f.threshold);
      const coords = w.nodes.flatMap((n) => [n.x, n.y, n.z]); const m = Math.max(...coords.map(Math.abs)) || 1;
      const pos = Object.fromEntries(w.nodes.map((n) => [n.symbol, [n.x / m, n.y / m, n.z / m]]));
      const clusterColors = ["#408bff", "#16c784", "#f2be4c", "#c08bff", "#f05259"];
      const focus = state.selected && state.selected.symbol;
      const primitives = w.edges.filter((e) => Math.abs(e.rho) >= threshold).map((e) => ({ type: "line", pts: [pos[e.a], pos[e.b]], color: G().ramp("correlation", (e.rho + 1) / 2), width: 0.6 + 2.4 * Math.abs(e.rho), alpha: focus && focus !== e.a && focus !== e.b ? 0.12 : 0.85 }));
      w.nodes.forEach((n) => primitives.push({ type: "point", pts: [pos[n.symbol]], color: clusterColors[(n.cluster - 1) % clusterColors.length], radius: 4 + n.volatility * 250, label: n.symbol, data: n, declutter: true }));
      const prior = data.windows.prior_60;
      const rows = w.edges.map((e) => { const p = prior.edges.find((x) => x.a === e.a && x.b === e.b); return { pair: `${e.a}–${e.b}`, rho: e.rho, prior_60_rho: p ? p.rho : null, change: p ? e.rho - p.rho : null }; });
      return { primitives, axes: null,
        legend: { scale: "correlation", min: -1, max: 1, label: "Edge: Pearson ρ (yellow negative, blue positive); node colour = cluster", unit: "" },
        columns: [["pair", "Pair"], ["rho", "ρ (selected window)"], ["prior_60_rho", "ρ (prior 60d)"], ["change", "Change"]], rows,
        describe: (n) => { const linked = w.edges.filter((e) => e.a === n.symbol || e.b === n.symbol).sort((a, b) => Math.abs(b.rho) - Math.abs(a.rho)).slice(0, 4); return [["Instrument", n.symbol], ["Cluster", n.cluster], ["Daily volatility", n.volatility], ...linked.map((e) => [`ρ with ${e.a === n.symbol ? e.b : e.a}`, e.rho])]; },
        extra: [["Sample size", `${w.observations} observations`], ["Std. error of ρ near 0", w.standard_error_at_zero], ["Edges shown", `|ρ| ≥ ${threshold}`]],
        filters() {
          control("select", "Window", [["latest_60", "Latest 60 days"], ["latest_120", "Latest 120 days"], ["prior_60", "Prior 60 days"]], f.window, (v) => { f.window = v; });
          control("range", "Minimum |ρ|", { min: 0, max: 0.9, step: 0.05 }, f.threshold, (v) => { f.threshold = v; });
        } };
    },
    "stat-arb"(data) {
      const f = state.filters; f.start ??= String(data.burn_in); f.band ??= "on";
      const pts = data.points.filter((p) => p.t >= Number(f.start));
      const [bmin, bmax] = minmax(pts.map((p) => p.beta)); const [zmin0, zmax0] = minmax(pts.map((p) => p.zscore)); const zr = Math.max(3, Math.abs(zmin0), Math.abs(zmax0));
      const sx = G().scaler(pts[0].t, pts[pts.length - 1].t); const sy = G().scaler(bmin, bmax); const sz = G().scaler(-zr, zr);
      const regimeColor = (r) => (r === null ? "#8292a5" : r === 0 ? "#408bff" : "#f2be4c");
      const primitives = [];
      for (let i = 1; i < pts.length; i += 1) primitives.push({ type: "line", pts: [[sx.to(pts[i - 1].t), sy.to(pts[i - 1].beta), sz.to(pts[i - 1].zscore)], [sx.to(pts[i].t), sy.to(pts[i].beta), sz.to(pts[i].zscore)]], color: regimeColor(pts[i].regime), width: 1.6 });
      pts.forEach((p) => primitives.push({ type: "point", pts: [[sx.to(p.t), sy.to(p.beta), sz.to(p.zscore)]], color: Math.abs(p.zscore) >= 2 ? "#f05259" : regimeColor(p.regime), radius: Math.abs(p.zscore) >= 2 ? 3.4 : 1.8, data: p }));
      if (f.band === "on") [-2, 2].forEach((b) => primitives.push({ type: "quad", pts: [[-1, -1, sz.to(b)], [1, -1, sz.to(b)], [1, 1, sz.to(b)], [-1, 1, sz.to(b)]], color: "#f05259", alpha: 0.08, stroke: "#f05259" }));
      return { primitives, axes: { x: { ...data.axes.x, scale: sx }, y: { ...data.axes.y, scale: sy }, z: { ...data.axes.z, scale: sz } },
        legend: { scale: "correlation", min: "Regime 1 (yellow)", max: "Regime 0 (blue)", label: "Filtered HMM regime; red points |z| ≥ 2", unit: "" },
        columns: [["t", "t"], ["beta", "Hedge elasticity"], ["zscore", "z-score"], ["regime", "Regime"], ["regime_probability", "Regime prob."]], rows: pts,
        describe: (p) => [["Observation", p.t], ["Hedge elasticity", p.beta], ["Standardized innovation", p.zscore], ["Filtered regime", p.regime ?? "UNKNOWN (first observation)"], ["Regime probability", p.regime_probability]],
        extra: [["Pair", data.pair.join(" / ")], ["Regime model", data.regime_model], ["Economic validation", data.economic_validation]],
        filters() {
          control("range", "Start observation", { min: 0, max: data.points.length - 30, step: 1 }, f.start, (v) => { f.start = v; });
          control("select", "±2σ bands", [["on", "Shown"], ["off", "Hidden"]], f.band, (v) => { f.band = v; });
        } };
    },
    "model-evaluation"(data) {
      const families = Object.keys(data.families).filter((k) => data.rows.some((r) => r.family === k));
      const f = state.filters; f.family ??= families[0];
      const rows = data.rows.filter((r) => r.family === f.family); const folds = [...new Set(rows.map((r) => r.fold))];
      const [vmin, vmax] = minmax(rows.flatMap((r) => [r.value, r.baseline])); const low = Math.min(0, vmin);
      const sz = G().scaler(low, vmax); const sy = G().scaler(0, Math.max(1, folds.length - 1));
      const primitives = [];
      rows.forEach((r) => {
        const better = r.lower_is_better ? r.value < r.baseline : r.value > r.baseline;
        const datum = { ...r, family_label: data.families[r.family], beats_baseline: better ? "YES" : "NO" };
        primitives.push({ type: "bar", box: [-0.35, sy.to(r.fold), sz.to(low), sz.to(r.value)], half: 0.12, color: better ? "#16c784" : "#f05259", side: better ? "#0f8a5c" : "#a8383e", data: datum });
        primitives.push({ type: "bar", box: [0.35, sy.to(r.fold), sz.to(low), sz.to(r.baseline)], half: 0.12, color: "#5b6573", side: "#3a4350", data: { ...datum, bar: "training-only baseline" } });
      });
      return { primitives, axes: { x: { label: `${rows[0] ? rows[0].model : "Model"} (left) vs baseline (right)`, ticks: [] }, y: { ...data.axes.y, scale: sy, ticks: folds.map((k) => [sy.to(k), k]) }, z: { label: data.families[f.family], scale: sz } },
        legend: { scale: "pnl", min: "Worse than baseline", max: "Better than baseline", label: "Model bar colour; grey = training-only baseline", unit: "" },
        columns: [["model", "Model"], ["fold", "Fold"], ["value", "Model metric"], ["baseline", "Baseline metric"], ["lower_is_better", "Lower is better"]], rows,
        describe: (r) => [["Model", r.model], ["Family", r.family_label], ["Fold", r.fold], ["Model metric", r.value], ["Baseline", r.baseline], ["Beats baseline", r.beats_baseline], ["Bar", r.bar || "model"]],
        extra: [["Not computed here", Object.entries(data.not_computed).filter(([, v]) => v).map(([k, v]) => `${k}: ${v}`).join(" ")], ["Interpretation", data.interpretation]],
        filters() { control("select", "Metric family", families.map((k) => [k, data.families[k]]), f.family, (v) => { f.family = v; state.selected = null; }); } };
    },
  };

  function render() {
    const entry = state.cache.get(`${state.module}:${state.mode}`);
    if (!entry || !state.engine) return;
    if (entry.state !== "AVAILABLE") {
      state.engine.setScene({ primitives: [], axes: null }); $("lab-legend").replaceChildren(); details([]); table([], []);
      $("lab-filters").replaceChildren();
      setStatus(state.mode === "local" ? "UNAVAILABLE · No admitted evidence supports this module in local mode. Nothing is inferred. Select the synthetic fixture to explore the method." : `${entry.state} · ${entry.error ? entry.error.message : "No data"}`, "warn");
      $("lab-provenance").textContent = "No data rendered";
      return;
    }
    const data = entry.data; const view = renderers[state.module](data);
    $("lab-filters").replaceChildren(); view.filters();
    state.engine.setScene({ primitives: view.primitives, axes: view.axes });
    G().legend($("lab-legend"), view.legend);
    table(view.columns, view.rows);
    state.describe = view.describe; state.extra = view.extra || [];
    details(state.selected && state.selectedModule === state.module ? view.describe(state.selected) : [["Selection", "Click or press N / P to inspect"], ...state.extra]);
    $("lab-method").textContent = `${data.method || ""} ${data.interpretation || ""}`.trim();
    $("lab-title-module").textContent = data.title;
    $("lab-provenance").textContent = `${entry.provenance.classification} · ${entry.provenance.source} · as of ${entry.provenance.as_of || "—"}`;
    setStatus("");
  }
  async function load() {
    const key = `${state.module}:${state.mode}`; const ticket = ++state.request;
    if (!state.cache.has(key)) {
      setStatus("Computing analytics…", "info");
      try {
        const response = await fetch(`/api/v1/viz/${encodeURIComponent(state.module)}?mode=${encodeURIComponent(state.mode)}`, { headers: { Accept: "application/json" } });
        const body = await response.json();
        if (!body || typeof body.state !== "string") throw new Error("Malformed envelope");
        state.cache.set(key, body);
      } catch (error) { if (ticket === state.request) setStatus(`ERROR · ${error.message}`, "error"); return; }
    }
    if (ticket === state.request) render();
  }
  function choose(id) {
    state.module = id; state.filters = {}; state.selected = null;
    document.querySelectorAll("#lab-tabs [role=tab]").forEach((tab) => { const active = tab.dataset.module === id; tab.setAttribute("aria-selected", String(active)); tab.tabIndex = active ? 0 : -1; });
    load();
  }
  function init() {
    if (!$("lab-panel") || !window.GIG3D) return;
    const tabs = $("lab-tabs");
    MODULES.forEach((m, i) => {
      const tab = el("button", "lab-tab", m.label); tab.type = "button"; tab.setAttribute("role", "tab"); tab.dataset.module = m.id; tab.id = `lab-tab-${m.id}`; tab.setAttribute("aria-controls", "lab-canvas");
      tab.addEventListener("click", () => choose(m.id));
      tab.addEventListener("keydown", (event) => {
        if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
        const next = MODULES[(i + (event.key === "ArrowRight" ? 1 : MODULES.length - 1)) % MODULES.length];
        choose(next.id); $(`lab-tab-${next.id}`).focus(); event.preventDefault();
      });
      tabs.append(tab);
    });
    state.engine = G().create($("lab-canvas"), {
      onSelect(datum) { state.selected = datum; state.selectedModule = state.module; if (state.module === "correlation-network") render(); else details(datum ? state.describe(datum) : [["Selection", "None"], ...state.extra]); $("lab-live").textContent = datum ? `Selected ${state.describe(datum).slice(0, 3).map(([k, v]) => `${k} ${typeof v === "number" ? fmt(v, 4) : v}`).join(", ")}` : "Selection cleared"; },
      onHover(datum, x, y) {
        const tip = $("lab-tooltip"); if (!datum) { tip.classList.add("hidden"); return; }
        tip.textContent = state.describe(datum).slice(0, 6).map(([k, v]) => `${k}: ${typeof v === "number" ? fmt(v, 4) : v}`).join("\n");
        tip.style.left = `${Math.min(x + 14, $("lab-canvas").clientWidth - 230)}px`; tip.style.top = `${Math.max(4, y + 10)}px`; tip.classList.remove("hidden");
      },
      onOrbit(on) { $("lab-orbit").textContent = on ? "Pause orbit" : "Resume orbit"; $("lab-orbit").setAttribute("aria-pressed", String(on)); $("lab-motion").textContent = on ? "Camera orbiting" : "Camera paused"; },
    });
    $("lab-orbit").addEventListener("click", () => state.engine.setOrbit(!state.engine.isOrbiting()));
    $("lab-reset").addEventListener("click", () => state.engine.reset());
    $("lab-view-3d").addEventListener("click", () => { state.engine.set2D(false); $("lab-view-3d").setAttribute("aria-pressed", "true"); $("lab-view-2d").setAttribute("aria-pressed", "false"); });
    $("lab-view-2d").addEventListener("click", () => { state.engine.set2D(true); $("lab-view-2d").setAttribute("aria-pressed", "true"); $("lab-view-3d").setAttribute("aria-pressed", "false"); });
    $("lab-table-toggle").addEventListener("click", () => { const hidden = $("lab-table-wrap").classList.toggle("hidden"); $("lab-table-toggle").setAttribute("aria-expanded", String(!hidden)); });
    $("lab-mode").addEventListener("change", () => { state.mode = $("lab-mode").value; load(); });
    const dataMode = $("data-mode"); if (dataMode) dataMode.addEventListener("change", () => { $("lab-mode").value = dataMode.value; state.mode = dataMode.value; if (state.visible) load(); });
    state.mode = $("lab-mode").value;
  }
  function show(section) {
    const panel = $("lab-panel"); if (!panel) return;
    const module = DEFAULTS[section];
    state.visible = Boolean(module);
    panel.classList.toggle("hidden", !module);
    if (!module) { if (state.engine) state.engine.setOrbit(false); return; }
    choose(module);
  }
  window.GIGModels = { show, modules: MODULES.map((m) => m.id), engine: () => state.engine };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
