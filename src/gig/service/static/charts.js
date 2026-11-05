/* Market charts workspace: candlestick/OHLC/line/area price pane with volume,
 * overlays, a resizable indicator pane, a synchronized comparison pane, a
 * watchlist and saved layouts. Bars and indicator values come only from
 * /api/v1/charts/bars; nothing is computed from or filled into missing data. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const reduced = typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const LAYOUT_KEY = "gig.chart.layouts.v1";
  const CURRENT_KEY = "gig.chart.current.v1";
  const OVERLAYS = {
    sma20: { label: "SMA 20", color: "#f2be4c" }, sma50: { label: "SMA 50", color: "#b38cff" },
    ema20: { label: "EMA 20", color: "#4ea1ff" }, vwap20: { label: "VWAP 20", color: "#e46fb0" },
    bb: { label: "Bollinger 20, 2σ", color: "#7fb2c9" },
  };
  const LOWER = {
    rsi14: { label: "RSI 14", lines: [["rsi14", "#f2be4c"]], fixed: [0, 100], guides: [30, 70] },
    macd: { label: "MACD 12, 26, 9", lines: [["macd", "#4ea1ff"], ["macd_signal", "#f2be4c"]], histogram: "macd_histogram", guides: [0] },
    atr14: { label: "ATR 14", lines: [["atr14", "#b38cff"]] },
    rv20: { label: "Realized volatility 20 (annualized)", lines: [["rv20", "#3fd18b"]], percent: true },
    none: { label: "None" },
  };
  const DEFAULTS = { symbol: null, timeframe: "1D", type: "candles", overlays: ["sma20", "sma50"], lower: "rsi14", compare: "", lowerHeight: 150 };
  const state = { ...DEFAULTS, data: null, compareData: null, range: null, cursor: null, drag: null, started: false, visible: false, request: null, panes: [] };

  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const num = (v, digits = 2) => (v === null || v === undefined || !Number.isFinite(v) ? "—" : v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits }));
  const compact = (v) => (!Number.isFinite(v) ? "—" : Math.abs(v) >= 1e9 ? `${(v / 1e9).toFixed(2)}B` : Math.abs(v) >= 1e6 ? `${(v / 1e6).toFixed(2)}M` : Math.abs(v) >= 1e3 ? `${(v / 1e3).toFixed(1)}K` : v.toFixed(0));
  const say = (text) => { $("charts-live").textContent = text; };
  const isoDate = (t) => String(t).slice(0, 10);

  /* ---------- persistence ---------- */
  function readJSON(key, fallback) { try { const v = JSON.parse(localStorage.getItem(key)); return v ?? fallback; } catch { return fallback; } }
  function writeJSON(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable: layouts last for this page only */ } }
  const layoutOf = () => ({ symbol: state.symbol, timeframe: state.timeframe, type: state.type, overlays: [...state.overlays], lower: state.lower, compare: state.compare, lowerHeight: state.lowerHeight });
  function applyLayout(layout) {
    const l = { ...DEFAULTS, ...layout };
    Object.assign(state, { symbol: l.symbol, timeframe: l.timeframe in { "1D": 1, "1W": 1, "1M": 1 } ? l.timeframe : "1D", type: ["candles", "ohlc", "line", "area"].includes(l.type) ? l.type : "candles",
      overlays: (l.overlays || []).filter((o) => o in OVERLAYS), lower: l.lower in LOWER ? l.lower : "rsi14", compare: l.compare || "", lowerHeight: Math.max(80, Math.min(320, Number(l.lowerHeight) || 150)) });
    syncControls();
  }

  /* ---------- data ---------- */
  function query(symbol) {
    const params = new URLSearchParams({ mode: $("data-mode").value, timeframe: state.timeframe });
    if (symbol) params.set("symbol", symbol);
    if ($("data-mode").value === "local") {
      params.set("max_age_seconds", $("freshness-policy").value);
      const cutoff = $("replay-asof").value.trim(); if (cutoff) params.set("as_of", cutoff);
    }
    return `/api/v1/charts/bars?${params}`;
  }
  async function fetchBars(symbol, signal) {
    const response = await fetch(query(symbol), { headers: { Accept: "application/json" }, signal });
    const payload = await response.json();
    if (!payload || typeof payload !== "object" || !("state" in payload) || !("provenance" in payload)) throw new Error("unsupported contract");
    return payload;
  }
  async function load({ keepRange = false } = {}) {
    if (state.request) state.request.abort();
    const controller = new AbortController(); state.request = controller;
    truth({ state: "LOADING" });
    $("charts-status").textContent = "Loading bars…"; $("charts-status").classList.remove("hidden");
    const environment = $("data-mode").value;
    if (state.environment && state.environment !== environment) { state.symbol = null; state.compare = ""; }
    state.environment = environment;
    try {
      let payload = await fetchBars(state.symbol, controller.signal);
      if (state.request !== controller) return;
      let note = "";
      if (!payload.data && state.symbol) {
        const fallback = await fetchBars(null, controller.signal);
        if (state.request !== controller) return;
        if (fallback.data) { note = `${state.symbol} is not available in this data environment; showing ${fallback.data.symbol}.`; payload = fallback; }
      }
      let compare = null;
      if (payload.data && state.compare && state.compare !== payload.data.symbol) {
        compare = await fetchBars(state.compare, controller.signal);
        if (state.request !== controller) return;
      }
      state.payload = payload; state.data = payload.data; state.compareData = compare && compare.data ? compare.data : null;
      if (state.data) {
        state.symbol = state.data.symbol;
        const n = state.data.bars.t.length;
        if (!keepRange || !state.range || state.range[1] > n) state.range = [Math.max(0, n - 120), n];
        $("charts-status").classList.add("hidden");
        $("charts-note").textContent = note; $("charts-note").classList.toggle("hidden", !note); if (note) say(note);
      } else {
        state.range = null;
        $("charts-status").textContent = payload.state === "UNAVAILABLE"
          ? ($("data-mode").value === "local" ? "No admitted bars for this selection. Admit bars in Markets → Analytics workbench, or choose the Synthetic research demo." : "No bars for this selection.")
          : `${payload.state}: ${payload.error ? payload.error.message : "no data"}`;
      }
      truth(payload);
      renderWatchlist(); syncControls(); draw(); writeJSON(CURRENT_KEY, layoutOf());
      if (state.data) say(`${state.data.symbol} ${state.timeframe}: ${state.data.bars.t.length} bars loaded`);
    } catch (error) {
      if (controller.signal.aborted && state.request !== controller) return;
      state.data = null; state.range = null;
      truth({ state: "ERROR" });
      $("charts-status").textContent = "The chart service could not be reached. Refresh to retry."; $("charts-status").classList.remove("hidden");
      draw();
    }
  }
  function truth(payload) {
    if (!state.visible) return;
    const badge = $("state-badge"); badge.textContent = payload.state; if (payload.state === "LOADING") badge.removeAttribute("data-state"); else badge.dataset.state = payload.state;
    const p = payload.provenance;
    if (p && !payload.data) { $("source-label").textContent = payload.state === "UNAVAILABLE" ? "charts · no admitted bars for this selection" : `charts · ${payload.state.toLowerCase()}`; $("asof-label").textContent = "As of —"; $("truth-notice").textContent = "Nothing is charted without admitted bars · Research only · No orders"; return; }
    if (!p) { $("source-label").textContent = payload.state === "LOADING" ? "charts · fetching bars…" : "charts · unavailable"; $("asof-label").textContent = "As of —"; $("truth-notice").textContent = "Research only · No orders"; return; }
    $("source-label").textContent = `charts · ${p.source} · ${p.classification}`;
    $("asof-label").textContent = `As of ${p.as_of || "—"}`;
    const synthetic = p.classification === "SYNTHETIC";
    $("truth-notice").textContent = `${payload.state === "STALE" ? "STALE · latest admitted bar is older than the freshness policy. " : ""}${synthetic ? "SYNTHETIC · fictional instruments, not market observations" : "Admitted local bars · caller-asserted classification"} · Research only · No orders`;
  }

  /* ---------- controls ---------- */
  function syncControls() {
    document.querySelectorAll("[data-chart-timeframe]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.chartTimeframe === state.timeframe)));
    document.querySelectorAll("[data-chart-type]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.chartType === state.type)));
    document.querySelectorAll("[data-chart-overlay]").forEach((c) => { c.checked = state.overlays.includes(c.dataset.chartOverlay); });
    $("chart-lower-select").value = state.lower;
    const symbols = state.data ? state.data.watchlist.map((w) => w.symbol) : [];
    const fill = (select, values, extra) => {
      const current = select.value;
      select.replaceChildren(...(extra ? [new Option(extra, "")] : []), ...values.map((s) => new Option(s, s)));
      select.value = values.includes(current) || current === "" ? current : "";
    };
    fill($("chart-symbol"), symbols); $("chart-symbol").value = state.symbol || "";
    fill($("chart-compare"), symbols.filter((s) => s !== state.symbol), "No comparison"); $("chart-compare").value = state.compare && state.compare !== state.symbol ? state.compare : "";
    $("chart-lower-pane").style.height = `${state.lowerHeight}px`;
    $("chart-lower-wrap").classList.toggle("hidden", state.lower === "none");
    $("chart-compare-wrap").classList.toggle("hidden", !state.compareData);
    const layouts = readJSON(LAYOUT_KEY, {});
    const select = $("chart-layouts"); const chosen = select.value;
    select.replaceChildren(new Option("Saved layouts…", ""), ...Object.keys(layouts).sort().map((name) => new Option(name, name)));
    select.value = chosen in layouts ? chosen : "";
    $("chart-layout-delete").disabled = !select.value;
  }
  function renderWatchlist() {
    const list = $("chart-watchlist"); list.replaceChildren();
    if (!state.data) { const li = document.createElement("li"); li.className = "chart-watch-empty"; li.textContent = "No instruments"; list.append(li); return; }
    state.data.watchlist.forEach((row) => {
      const li = document.createElement("li");
      const button = document.createElement("button"); button.type = "button"; button.className = "chart-watch-row"; button.dataset.symbol = row.symbol;
      button.setAttribute("aria-pressed", String(row.symbol === state.symbol));
      const change = row.change === null ? "—" : `${row.change >= 0 ? "+" : ""}${(row.change * 100).toFixed(2)}%`;
      const name = document.createElement("span"); name.className = "chart-watch-symbol"; name.textContent = row.symbol;
      const last = document.createElement("span"); last.className = "chart-watch-last"; last.textContent = num(row.last);
      const ch = document.createElement("span"); ch.className = `chart-watch-change ${row.change > 0 ? "up" : row.change < 0 ? "down" : ""}`; ch.textContent = change;
      button.append(name, last, ch);
      button.addEventListener("click", () => setSymbol(row.symbol));
      li.append(button); list.append(li);
    });
  }
  function setSymbol(symbol) { if (!symbol || symbol === state.symbol) return; state.symbol = symbol; if (state.compare === symbol) state.compare = ""; load(); }

  /* ---------- geometry ---------- */
  function visible() {
    if (!state.data || !state.range) return null;
    const [i0, i1] = state.range; return { i0, i1, n: i1 - i0 };
  }
  function xOf(i, width) { const v = visible(); const step = width / v.n; return (i - v.i0 + 0.5) * step; }
  function indexAt(x, width) { const v = visible(); const i = v.i0 + Math.floor((x / width) * v.n); return Math.max(v.i0, Math.min(v.i1 - 1, i)); }
  function niceTicks(lo, hi, count = 5) {
    if (!(hi > lo)) { const pad = Math.abs(lo) * 0.01 || 1; lo -= pad; hi += pad; }
    const raw = (hi - lo) / count; const mag = 10 ** Math.floor(Math.log10(raw)); const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
    const ticks = []; for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) ticks.push(+t.toFixed(10));
    return { ticks, step };
  }
  function setupCanvas(canvas) {
    const r = canvas.getBoundingClientRect(); const ratio = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(r.width)), h = Math.max(1, Math.round(r.height));
    if (canvas.width !== Math.round(w * ratio) || canvas.height !== Math.round(h * ratio)) { canvas.width = Math.round(w * ratio); canvas.height = Math.round(h * ratio); }
    const ctx = canvas.getContext("2d"); ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }
  const AXIS = 64;

  /* ---------- drawing ---------- */
  function draw() {
    state.panes = [];
    const price = $("chart-price");
    if (!state.visible || !price) return;
    if (!visible()) { [price, $("chart-lower"), $("chart-compare-canvas")].forEach((c) => { if (c) setupCanvas(c); }); readout(null); return; }
    drawPrice(price, state.data, true);
    if (state.lower !== "none") drawLower($("chart-lower"));
    if (state.compareData) drawPrice($("chart-compare-canvas"), state.compareData, false);
    readout(state.cursor);
  }
  function alignedBars(data) {
    if (data === state.data) return { get: (i) => i };
    const at = new Map(data.bars.t.map((t, i) => [t, i]));
    return { get: (i) => at.get(state.data.bars.t[i]) };
  }
  function drawPrice(canvas, data, primary) {
    const { ctx, w, h } = setupCanvas(canvas);
    const dateBand = 22;
    const plotW = w - AXIS, volH = primary ? Math.round(h * 0.18) : 0, top = 8, bottom = h - dateBand, priceBottom = bottom - volH - (primary ? 6 : 0);
    const v = visible(); const map = alignedBars(data); const b = data.bars;
    const idx = []; for (let i = v.i0; i < v.i1; i++) { const j = map.get(i); if (j !== undefined) idx.push([i, j]); }
    let lo = Infinity, hi = -Infinity, vmax = 0;
    idx.forEach(([, j]) => {
      if (state.type === "line" || state.type === "area") { lo = Math.min(lo, b.close[j]); hi = Math.max(hi, b.close[j]); } else { lo = Math.min(lo, b.low[j]); hi = Math.max(hi, b.high[j]); }
      vmax = Math.max(vmax, b.volume[j]);
      if (primary) state.overlays.forEach((o) => (o === "bb" ? ["bb_upper", "bb_lower"] : [o]).forEach((k) => { const x = data.indicators[k][j]; if (x !== null) { lo = Math.min(lo, x); hi = Math.max(hi, x); } }));
    });
    if (!idx.length) { ctx.fillStyle = css("--muted"); ctx.font = "12px Inter, system-ui, sans-serif"; ctx.fillText("No bars in this range", 12, 24); return; }
    const pad = (hi - lo) * 0.06 || hi * 0.01 || 1; lo -= pad; hi += pad;
    const y = (p) => top + (hi - p) / (hi - lo) * (priceBottom - top);
    const grid = css("--grid") || "#141a22", muted = css("--muted") || "#8292a5", up = "#3fd18b", down = "#ff5f6d";
    const { ticks } = niceTicks(lo, hi, Math.max(3, Math.floor((priceBottom - top) / 48)));
    ctx.font = "11px 'SFMono-Regular', Consolas, monospace"; ctx.textBaseline = "middle";
    ticks.forEach((t) => { const yy = Math.round(y(t)) + 0.5; ctx.strokeStyle = grid; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(0, yy); ctx.lineTo(plotW, yy); ctx.stroke(); ctx.fillStyle = muted; ctx.fillText(num(t, t >= 1000 ? 0 : 2), plotW + 6, yy); });
    const step = plotW / v.n, body = Math.max(1, Math.min(14, step * 0.7));
    if (primary && volH) {
      idx.forEach(([i, j]) => { const x = xOf(i, plotW); const hh = vmax ? (b.volume[j] / vmax) * volH : 0; ctx.fillStyle = b.close[j] >= b.open[j] ? "rgba(63,209,139,.28)" : "rgba(255,95,109,.28)"; ctx.fillRect(x - body / 2, bottom - hh, body, hh); });
      ctx.fillStyle = muted; ctx.textBaseline = "top"; ctx.fillText(`Vol ${compact(vmax)}`, 8, bottom - volH + 2);
    }
    if (state.type === "candles" || state.type === "ohlc") {
      idx.forEach(([i, j]) => {
        const x = Math.round(xOf(i, plotW)) + 0.5, rising = b.close[j] >= b.open[j]; ctx.strokeStyle = ctx.fillStyle = rising ? up : down; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, y(b.high[j])); ctx.lineTo(x, y(b.low[j])); ctx.stroke();
        if (state.type === "candles") { const yo = y(b.open[j]), yc = y(b.close[j]); ctx.fillRect(x - body / 2, Math.min(yo, yc), body, Math.max(1, Math.abs(yc - yo))); }
        else { const tick = Math.max(2, body / 2); ctx.beginPath(); ctx.moveTo(x - tick, y(b.open[j])); ctx.lineTo(x, y(b.open[j])); ctx.moveTo(x, y(b.close[j])); ctx.lineTo(x + tick, y(b.close[j])); ctx.stroke(); }
      });
    } else {
      const accent = css("--accent") || "#4ea1ff";
      ctx.beginPath(); idx.forEach(([i, j], k) => { const x = xOf(i, plotW); if (k) ctx.lineTo(x, y(b.close[j])); else ctx.moveTo(x, y(b.close[j])); });
      ctx.strokeStyle = accent; ctx.lineWidth = 1.6; ctx.stroke();
      if (state.type === "area") {
        ctx.lineTo(xOf(idx[idx.length - 1][0], plotW), priceBottom); ctx.lineTo(xOf(idx[0][0], plotW), priceBottom); ctx.closePath();
        const g = ctx.createLinearGradient(0, top, 0, priceBottom); g.addColorStop(0, "rgba(78,161,255,.32)"); g.addColorStop(1, "rgba(78,161,255,0)"); ctx.fillStyle = g; ctx.fill();
      }
    }
    if (primary) {
      state.overlays.forEach((o) => {
        const keys = o === "bb" ? ["bb_upper", "bb_middle", "bb_lower"] : [o];
        keys.forEach((k, n) => line(ctx, idx, data.indicators[k], plotW, y, OVERLAYS[o].color, o === "bb" && n !== 1 ? [4, 3] : null));
      });
    }
    ctx.fillStyle = css("--text") || "#f2f5f9"; ctx.font = "12px Inter, system-ui, sans-serif"; ctx.textBaseline = "top";
    ctx.fillText(`${data.symbol} · ${state.timeframe}${primary ? "" : " · comparison"}`, 8, 6);
    xAxis(ctx, plotW, h - 5, muted);
    state.panes.push({ canvas, kind: primary ? "price" : "compare", plotW, y: (p) => y(p), inv: (yy) => hi - ((yy - top) / (priceBottom - top)) * (hi - lo), top, bottom: priceBottom, data, map });
    crosshair(ctx, canvas, plotW, h);
  }
  function line(ctx, idx, values, plotW, y, color, dash) {
    ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = 1.3; if (dash) ctx.setLineDash(dash);
    ctx.beginPath(); let pen = false;
    idx.forEach(([i, j]) => { const v = values[j]; if (v === null || v === undefined) { pen = false; return; } const x = xOf(i, plotW); if (pen) ctx.lineTo(x, y(v)); else ctx.moveTo(x, y(v)); pen = true; });
    ctx.stroke(); ctx.restore();
  }
  function xAxis(ctx, plotW, h, muted) {
    const v = visible(); const t = state.data.bars.t; const every = Math.max(1, Math.ceil(v.n / Math.max(2, Math.floor(plotW / 90))));
    ctx.font = "10px 'SFMono-Regular', Consolas, monospace"; ctx.fillStyle = muted; ctx.textBaseline = "bottom"; ctx.textAlign = "center";
    for (let i = v.i0; i < v.i1; i += every) { const x = xOf(i, plotW); ctx.fillText(isoDate(t[i]), Math.max(34, Math.min(plotW - 34, x)), h); }
    ctx.textAlign = "left";
  }
  function drawLower(canvas) {
    const spec = LOWER[state.lower]; const { ctx, w, h } = setupCanvas(canvas);
    const plotW = w - AXIS, top = 18, bottom = h - 8; const v = visible(); const ind = state.data.indicators;
    const idx = []; for (let i = v.i0; i < v.i1; i++) idx.push([i, i]);
    let lo = Infinity, hi = -Infinity;
    const keys = [...spec.lines.map(([k]) => k), ...(spec.histogram ? [spec.histogram] : [])];
    idx.forEach(([, j]) => keys.forEach((k) => { const x = ind[k][j]; if (x !== null) { lo = Math.min(lo, x); hi = Math.max(hi, x); } }));
    if (spec.fixed) [lo, hi] = spec.fixed;
    const muted = css("--muted") || "#8292a5", grid = css("--grid") || "#141a22";
    ctx.fillStyle = css("--text") || "#f2f5f9"; ctx.font = "11px Inter, system-ui, sans-serif"; ctx.textBaseline = "top"; ctx.fillText(spec.label, 8, 3);
    if (!Number.isFinite(lo)) { ctx.fillStyle = muted; ctx.fillText("Not enough bars for this indicator in the visible range", 8, 22); state.panes.push({ canvas, kind: "lower", plotW, empty: true }); crosshair(ctx, canvas, plotW, h); return; }
    if (!spec.fixed) { const pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.05 || 1; lo -= pad; hi += pad; }
    const y = (p) => top + (hi - p) / (hi - lo) * (bottom - top);
    ctx.font = "10px 'SFMono-Regular', Consolas, monospace"; ctx.textBaseline = "middle";
    const fmt = (t) => (spec.percent ? `${(t * 100).toFixed(0)}%` : num(t, Math.abs(hi - lo) < 5 ? 2 : 0));
    niceTicks(lo, hi, 3).ticks.forEach((t) => { const yy = Math.round(y(t)) + 0.5; ctx.strokeStyle = grid; ctx.beginPath(); ctx.moveTo(0, yy); ctx.lineTo(plotW, yy); ctx.stroke(); ctx.fillStyle = muted; ctx.fillText(fmt(t), plotW + 6, yy); });
    (spec.guides || []).forEach((g) => { if (g < lo || g > hi) return; const yy = Math.round(y(g)) + 0.5; ctx.save(); ctx.setLineDash([3, 3]); ctx.strokeStyle = "#3a4656"; ctx.beginPath(); ctx.moveTo(0, yy); ctx.lineTo(plotW, yy); ctx.stroke(); ctx.restore(); });
    if (spec.histogram) {
      const step = plotW / v.n, body = Math.max(1, step * 0.6), zero = y(Math.max(lo, Math.min(hi, 0)));
      idx.forEach(([i, j]) => { const val = ind[spec.histogram][j]; if (val === null) return; ctx.fillStyle = val >= 0 ? "rgba(63,209,139,.55)" : "rgba(255,95,109,.55)"; const yy = y(val); ctx.fillRect(xOf(i, plotW) - body / 2, Math.min(yy, zero), body, Math.max(1, Math.abs(zero - yy))); });
    }
    spec.lines.forEach(([k, color]) => line(ctx, idx, ind[k], plotW, y, color, null));
    state.panes.push({ canvas, kind: "lower", plotW, y, inv: (yy) => hi - ((yy - top) / (bottom - top)) * (hi - lo), top, bottom });
    crosshair(ctx, canvas, plotW, h);
  }
  function crosshair(ctx, canvas, plotW, h) {
    if (state.cursor === null || !visible()) return;
    const { i, y: cy, canvas: owner } = state.cursor; if (i < state.range[0] || i >= state.range[1]) return;
    const x = Math.round(xOf(i, plotW)) + 0.5;
    ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = "rgba(166,178,195,.7)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    if (owner === canvas && cy !== null) { ctx.beginPath(); ctx.moveTo(0, cy + 0.5); ctx.lineTo(plotW, cy + 0.5); ctx.stroke(); }
    ctx.restore();
    if (owner === canvas && cy !== null) {
      const pane = state.panes.find((p) => p.canvas === canvas);
      if (pane && pane.inv) { const value = pane.inv(cy); ctx.fillStyle = "#26303d"; ctx.fillRect(plotW + 1, cy - 9, AXIS - 2, 18); ctx.fillStyle = "#f2f5f9"; ctx.font = "11px 'SFMono-Regular', Consolas, monospace"; ctx.textBaseline = "middle"; ctx.fillText(pane.kind === "lower" && LOWER[state.lower].percent ? `${(value * 100).toFixed(1)}%` : num(value), plotW + 5, cy); }
    }
  }
  function readout(cursor) {
    const out = $("chart-readout"); out.replaceChildren();
    if (!state.data) return;
    const i = cursor ? cursor.i : state.range[1] - 1; const b = state.data.bars, ind = state.data.indicators;
    const add = (label, value, cls) => { const s = document.createElement("span"); s.className = `chart-read ${cls || ""}`; const k = document.createElement("b"); k.textContent = label; s.append(k, document.createTextNode(` ${value}`)); out.append(s); };
    add("", `${state.data.symbol} · ${isoDate(b.t[i])}`, "chart-read-date");
    add("O", num(b.open[i])); add("H", num(b.high[i])); add("L", num(b.low[i])); add("C", num(b.close[i]), b.close[i] >= b.open[i] ? "up" : "down");
    add("Vol", compact(b.volume[i]));
    state.overlays.forEach((o) => (o === "bb" ? [["BB up", "bb_upper"], ["BB low", "bb_lower"]] : [[OVERLAYS[o].label, o]]).forEach(([label, key]) => add(label, num(ind[key][i]))));
    if (state.lower !== "none") LOWER[state.lower].lines.forEach(([key]) => add(key === "macd_signal" ? "Signal" : LOWER[state.lower].label.split(" ")[0], LOWER[state.lower].percent ? (ind[key][i] === null ? "—" : `${(ind[key][i] * 100).toFixed(1)}%`) : num(ind[key][i])));
    if (state.compareData) { const j = alignedBars(state.compareData).get(i); add(state.compareData.symbol, j === undefined ? "no bar" : num(state.compareData.bars.close[j])); }
  }

  /* ---------- interaction ---------- */
  function zoom(factor, anchor) {
    const v = visible(); if (!v) return; const n = state.data.bars.t.length;
    const size = Math.max(10, Math.min(n, Math.round(v.n * factor)));
    const a = anchor ?? (v.i0 + v.i1) / 2; const ratio = (a - v.i0) / v.n;
    let i0 = Math.round(a - ratio * size); i0 = Math.max(0, Math.min(n - size, i0));
    state.range = [i0, i0 + size]; draw();
  }
  function pan(bars) {
    const v = visible(); if (!v) return; const n = state.data.bars.t.length;
    const i0 = Math.max(0, Math.min(n - v.n, v.i0 + bars)); state.range = [i0, i0 + v.n]; draw();
  }
  function bindCanvas(canvas) {
    const plotWidth = () => canvas.getBoundingClientRect().width - AXIS;
    const local = (e) => { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    canvas.addEventListener("pointermove", (e) => {
      if (!visible()) return; const [x, y] = local(e);
      if (state.drag) { const dx = x - state.drag.x; const per = plotWidth() / visible().n; const bars = Math.trunc(-dx / per); if (bars) { pan(bars); state.drag.x -= bars * per; } }
      state.cursor = { i: indexAt(Math.min(x, plotWidth() - 1), plotWidth()), y: x < plotWidth() ? y : null, canvas }; draw();
    });
    canvas.addEventListener("pointerdown", (e) => { if (e.button !== 0 || !visible()) return; state.drag = { x: local(e)[0] }; canvas.setPointerCapture(e.pointerId); canvas.focus({ preventScroll: true }); });
    canvas.addEventListener("pointerup", () => { state.drag = null; });
    canvas.addEventListener("pointercancel", () => { state.drag = null; });
    canvas.addEventListener("pointerleave", () => { if (!state.drag) { state.cursor = null; draw(); } });
    canvas.addEventListener("wheel", (e) => { if (!visible()) return; e.preventDefault(); const [x] = local(e); zoom(e.deltaY > 0 ? 1.15 : 1 / 1.15, indexAt(Math.min(x, plotWidth() - 1), plotWidth())); }, { passive: false });
    canvas.addEventListener("keydown", (e) => {
      if (!visible()) return; const v = visible(); let i = state.cursor ? state.cursor.i : v.i1 - 1; let handled = true;
      if (e.key === "ArrowLeft") { i = Math.max(0, i - 1); if (i < v.i0) pan(-1); }
      else if (e.key === "ArrowRight") { i = Math.min(state.data.bars.t.length - 1, i + 1); if (i >= v.i1) pan(1); }
      else if (e.key === "+" || e.key === "=") zoom(1 / 1.25, i);
      else if (e.key === "-") zoom(1.25, i);
      else if (e.key === "Home") { pan(-state.data.bars.t.length); i = 0; }
      else if (e.key === "End") { pan(state.data.bars.t.length); i = state.data.bars.t.length - 1; }
      else if (e.key === "0") { const n = state.data.bars.t.length; state.range = [Math.max(0, n - 120), n]; i = n - 1; }
      else if (e.key === "Escape") { state.cursor = null; draw(); return; }
      else handled = false;
      if (!handled) return; e.preventDefault();
      state.cursor = { i, y: null, canvas }; draw();
      const b = state.data.bars; say(`${isoDate(b.t[i])}: open ${num(b.open[i])}, high ${num(b.high[i])}, low ${num(b.low[i])}, close ${num(b.close[i])}`);
    });
  }
  function bindSplitter() {
    const handle = $("chart-splitter");
    const set = (h) => { state.lowerHeight = Math.max(80, Math.min(320, Math.round(h))); handle.setAttribute("aria-valuenow", String(state.lowerHeight)); $("chart-lower-pane").style.height = `${state.lowerHeight}px`; draw(); writeJSON(CURRENT_KEY, layoutOf()); };
    handle.addEventListener("pointerdown", (e) => { const start = e.clientY, base = state.lowerHeight; handle.setPointerCapture(e.pointerId);
      const move = (ev) => set(base - (ev.clientY - start)); const up = () => { handle.removeEventListener("pointermove", move); handle.removeEventListener("pointerup", up); };
      handle.addEventListener("pointermove", move); handle.addEventListener("pointerup", up); });
    handle.addEventListener("keydown", (e) => { if (e.key === "ArrowUp") { set(state.lowerHeight + 16); e.preventDefault(); } else if (e.key === "ArrowDown") { set(state.lowerHeight - 16); e.preventDefault(); } });
  }
  function bind() {
    ["chart-price", "chart-lower", "chart-compare-canvas"].forEach((id) => bindCanvas($(id)));
    bindSplitter();
    document.querySelectorAll("[data-chart-timeframe]").forEach((b) => b.addEventListener("click", () => { state.timeframe = b.dataset.chartTimeframe; load(); }));
    document.querySelectorAll("[data-chart-type]").forEach((b) => b.addEventListener("click", () => { state.type = b.dataset.chartType; syncControls(); draw(); writeJSON(CURRENT_KEY, layoutOf()); }));
    document.querySelectorAll("[data-chart-overlay]").forEach((c) => c.addEventListener("change", () => { state.overlays = [...document.querySelectorAll("[data-chart-overlay]:checked")].map((x) => x.dataset.chartOverlay); draw(); writeJSON(CURRENT_KEY, layoutOf()); }));
    $("chart-lower-select").addEventListener("change", () => { state.lower = $("chart-lower-select").value; syncControls(); draw(); writeJSON(CURRENT_KEY, layoutOf()); });
    $("chart-symbol").addEventListener("change", () => setSymbol($("chart-symbol").value));
    $("chart-compare").addEventListener("change", () => { state.compare = $("chart-compare").value; load({ keepRange: true }); });
    $("charts-reset").addEventListener("click", () => { if (!state.data) return; const n = state.data.bars.t.length; state.range = [Math.max(0, n - 120), n]; state.cursor = null; draw(); });
    $("chart-layout-save").addEventListener("click", () => {
      const name = $("chart-layout-name").value.trim().slice(0, 60); if (!name) { say("Enter a layout name to save"); $("chart-layout-name").focus(); return; }
      const layouts = readJSON(LAYOUT_KEY, {}); layouts[name] = layoutOf(); writeJSON(LAYOUT_KEY, layouts); $("chart-layout-name").value = ""; syncControls(); $("chart-layouts").value = name; $("chart-layout-delete").disabled = false; say(`Saved layout ${name}`);
    });
    $("chart-layouts").addEventListener("change", () => { const layouts = readJSON(LAYOUT_KEY, {}); const l = layouts[$("chart-layouts").value]; $("chart-layout-delete").disabled = !l; if (l) { applyLayout(l); load(); say(`Applied layout ${$("chart-layouts").value}`); } });
    $("chart-layout-delete").addEventListener("click", () => { const name = $("chart-layouts").value; const layouts = readJSON(LAYOUT_KEY, {}); if (!(name in layouts)) return; delete layouts[name]; writeJSON(LAYOUT_KEY, layouts); $("chart-layouts").value = ""; syncControls(); say(`Deleted layout ${name}`); });
    if (typeof ResizeObserver !== "undefined") new ResizeObserver(() => draw()).observe($("charts-stage"));
  }
  function show(on) {
    state.visible = on; $("charts-panel").classList.toggle("hidden", !on);
    if (!on) { if (state.request) state.request.abort(); return; }
    if (!state.started) { state.started = true; applyLayout(readJSON(CURRENT_KEY, {})); bind(); }
    if (state.pending) { state.symbol = state.pending; state.pending = null; if (state.compare === state.symbol) state.compare = ""; }
    load({ keepRange: true });
  }
  window.GIGCharts = {
    show, reload: () => load({ keepRange: true }), setSymbol,
    open(symbol) { if (state.started && state.visible) setSymbol(symbol); else state.pending = symbol; },
    async catalog() {
      const params = new URLSearchParams({ mode: $("data-mode").value, timeframe: "1M" });
      if ($("data-mode").value === "local") { params.set("max_age_seconds", $("freshness-policy").value); const c = $("replay-asof").value.trim(); if (c) params.set("as_of", c); }
      try { const r = await fetch(`/api/v1/charts/bars?${params}`, { headers: { Accept: "application/json" } }); const p = await r.json(); return p && p.data ? p.data.watchlist : []; } catch { return []; }
    }, applyLayout: (l) => { applyLayout(l); return load(); },
    state: () => ({ symbol: state.symbol, timeframe: state.timeframe, type: state.type, overlays: [...state.overlays], lower: state.lower, compare: state.compareData ? state.compareData.symbol : null,
      range: state.range ? [...state.range] : null, cursor: state.cursor ? state.cursor.i : null, bars: state.data ? state.data.bars.t.length : 0, lowerHeight: state.lowerHeight,
      classification: state.payload ? state.payload.provenance.classification : null, payloadState: state.payload ? state.payload.state : null }),
    symbols: () => (state.data ? state.data.watchlist.map((w) => w.symbol) : []),
    panes: () => state.panes.map((p) => ({ kind: p.kind, width: p.canvas.getBoundingClientRect().width, height: p.canvas.getBoundingClientRect().height })),
    reducedMotion: reduced,
  };
})();
