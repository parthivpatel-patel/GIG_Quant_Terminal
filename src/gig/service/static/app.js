/* GIG local research terminal. Backend truth states and source evidence are retained. */
"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const sections = [
    { id: "command-center", label: "Command Center", icon: "M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z", description: "Portfolio intelligence, research, and operational controls." },
    { id: "markets", label: "Markets", icon: "M3 17l5-5 4 3 8-10M15 5h5v5M3 21h18", description: "Market observations, instrument identity, freshness, and source provenance." },
    { id: "signals", label: "Signals", icon: "M3 12h4l3-8 4 16 3-8h4", description: "Factor research and model scores. Research outputs require trading approval." },
    { id: "portfolio", label: "Portfolio", icon: "M3 7h18v13H3zM8 7V3h8v4M3 12h18M10 12v3h4v-3", description: "Portfolio construction, constraints, allocation, and exposure evidence." },
    { id: "risk", label: "Risk", icon: "M12 3l8 3v6c0 5-8 9-8 9s-8-4-8-9V6zM12 8v5M12 16h.01", description: "Risk estimates, stress scenarios, and valuation assumptions." },
    { id: "models", label: "Models", icon: "M5 5h4v4H5zM15 5h4v4h-4zM10 16h4v4h-4zM7 9v4h10V9M12 13v3", description: "Model capability, versions, diagnostics, and reproducible CPU research." },
    { id: "news", label: "NLP / News", icon: "M4 4h13v16H4zM17 8h3v12h-3M7 7h7M7 11h7M7 15h4", description: "Attributed financial text, entity links, and sentiment evidence." },
    { id: "stat-arb", label: "Statistical Arbitrage", icon: "M3 6c6 0 6 12 12 12h6M3 18c6 0 6-12 12-12h6", description: "Spread research, hedge ratios, mean reversion, and uncertainty." },
    { id: "options", label: "Options", icon: "M3 20h18M4 16c4-13 12-13 16 0M12 3v17", description: "European option estimates, volatility structures, and numerical diagnostics." },
    { id: "execution", label: "Execution", icon: "M4 7h15M15 3l4 4-4 4M20 17H5M9 13l-4 4 4 4", description: "Offline order lifecycle and reconciliation. Broker connectivity is disabled by default." },
    { id: "research", label: "Research", icon: "M9 3h6M10 3v6L4 19c-.5 1 .5 2 2 2h12c1.5 0 2.5-1 2-2L14 9V3M7 15h10", description: "Reproducible experiments, research artifacts, and evidence-backed analysis." },
    { id: "operations", label: "Operations", icon: "M3 12h4l3-7 4 14 3-7h4M4 3h16M4 21h16", description: "Dependency readiness, safe configuration, restart recovery, and operating status." },
  ];
  const validStates = new Set(["AVAILABLE", "UNAVAILABLE", "UNKNOWN", "ERROR", "STALE"]);
  const state = { section: sections[0], mode: "local", payload: null, rows: [], columns: [], sort: null, request: null, chart: null, points: [], projected: [], angle: -.55, tilt: .40, zoom: 1, dimension: 3, drag: null };
  const numberFormat = new Intl.NumberFormat("en-US", { maximumSignificantDigits: 7 });
  const human = (value) => String(value).replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  const isObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
  function scalarTag(value) {
    if (!isObject(value) || Object.keys(value).length !== 1) return null;
    if (typeof value.$integer === "string" && /^-?\d+$/.test(value.$integer)) return value.$integer;
    if (typeof value.$decimal === "string" && /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(value.$decimal)) return value.$decimal;
    return null;
  }
  function exactNumericOrder(left, right) {
    // Compare tagged decimals/integers by sign, decimal order, and significand.
    // Exponents are compared as BigInt; no expanded zero strings or binary64
    // conversion can round an identity or allocate an exponent-sized buffer.
    const parts = (value) => {
      const tagged = scalarTag(value);
      const text = tagged !== null ? tagged : typeof value === "number" && Number.isFinite(value) ? String(value) : null;
      if (text === null) return null;
      const match = /^([+-]?)(\d*)(?:\.(\d*))?(?:[eE]([+-]?\d+))?$/.exec(text);
      if (!match) return null;
      const digits = (match[2] + (match[3] || "")).replace(/^0+/, "");
      if (!digits) return { sign: 0, digits: "", order: 0n };
      const exponent = BigInt(match[4] || "0") - BigInt((match[3] || "").length);
      return { sign: match[1] === "-" ? -1 : 1, digits, order: BigInt(digits.length) + exponent };
    };
    const a = parts(left), b = parts(right);
    if (!a || !b) return null;
    if (a.sign !== b.sign) return a.sign < b.sign ? -1 : 1;
    if (!a.sign) return 0;
    if (a.order !== b.order) return (a.order < b.order ? -1 : 1) * a.sign;
    for (let index = 0; index < Math.max(a.digits.length, b.digits.length); index++) {
      const x = a.digits[index] || "0", y = b.digits[index] || "0";
      if (x !== y) return (x < y ? -1 : 1) * a.sign;
    }
    return 0;
  }
  function validContract(payload) {
    if (!isObject(payload) || payload.schema_version !== "1" || !validStates.has(payload.state) || !isObject(payload.provenance)) return false;
    if (typeof payload.provenance.source !== "string" || typeof payload.provenance.classification !== "string") return false;
    const hasData = ["AVAILABLE", "STALE"].includes(payload.state);
    if (hasData ? !isObject(payload.data) : payload.data !== null) return false;
    if (payload.state === "STALE" && typeof payload.provenance.as_of !== "string") return false;
    if (payload.state === "ERROR" && (!isObject(payload.error) || typeof payload.error.code !== "string" || typeof payload.error.message !== "string")) return false;
    if (payload.state === "AVAILABLE" && payload.error !== null) return false;
    return true;
  }
  function printable(value) {
    if (value === null || value === undefined) return "Unknown";
    if (typeof value === "number") return Number.isFinite(value) ? (Object.is(value, -0) ? "−0" : numberFormat.format(value)) : "Invalid numeric value";
    if (typeof value === "boolean") return value ? "Yes" : "No";
    if (scalarTag(value) !== null) return scalarTag(value);
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  }
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function nav() {
    const fragment = document.createDocumentFragment();
    sections.forEach((section, index) => {
      const link = el("a", "nav-link");
      link.href = `#${section.id}`;
      link.title = section.label;
      link.setAttribute("aria-label", section.label);
      const icon = el("span", "nav-icon");
      icon.setAttribute("aria-hidden", "true");
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("viewBox", "0 0 24 24");
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", section.icon);
      svg.append(path);
      icon.append(svg);
      link.append(icon, el("span", "", section.label));
      if (index === 0) link.append(el("span", "nav-shortcut", "⌘"));
      fragment.append(link);
    });
    $("section-nav").append(fragment);
  }
  function selectSection() {
    const id = location.hash.slice(1);
    state.section = sections.find((section) => section.id === id) || sections[0];
    document.querySelectorAll(".nav-link").forEach((link) => {
      if (link.hash === `#${state.section.id}`) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    $("section-title").textContent = state.section.label;
    $("breadcrumb").textContent = state.section.label;
    $("section-description").textContent = state.section.description;
    document.title = `${state.section.label} · GIG Quant Terminal`;
    $("workbench").classList.toggle("hidden", !["models", "options", "research", "markets", "stat-arb"].includes(state.section.id));
    if (state.section.id === "options") $("analytics-kind").value = "options";
    else if (state.section.id === "models") $("analytics-kind").value = "kalman";
    else if (state.section.id === "markets") $("analytics-kind").value = "microstructure";
    loadExample();
    state.sort = null;
    $("table-search").value = "";
    loadSection();
  }
  async function loadSection() {
    if (state.request) state.request.abort();
    const controller = new AbortController();
    state.request = controller;
    $("refresh").disabled = true;
    $("state-badge").textContent = "LOADING";
    $("state-badge").removeAttribute("data-state");
    $("source-label").textContent = "Fetching backend evidence…";
    $("asof-label").textContent = "As of —";
    $("metrics").replaceChildren();
    $("context-content").replaceChildren();
    $("error-panel").classList.add("hidden");
    state.rows = [];
    state.chart = null;
    renderTable();
    renderChart();
    const timeout = setTimeout(() => controller.abort("timeout"), 20000);
    try {
      const response = await fetch(`/api/v1/terminal/${encodeURIComponent(state.section.id)}?mode=${encodeURIComponent(state.mode)}`, { signal: controller.signal, cache: "no-store", headers: { Accept: "application/json" } });
      const payload = await response.json();
      if (state.request !== controller) return;
      if (!validContract(payload) || (!response.ok && payload.state !== "ERROR")) throw new Error("Backend returned an unsupported truth-state contract.");
      state.payload = payload;
      renderPayload(payload);
    } catch (error) {
      if (state.request !== controller) return;
      state.payload = null;
      renderPayload({ state: "ERROR", data: null, provenance: { source: "Backend request", classification: "UNAVAILABLE", as_of: null }, error: { code: controller.signal.reason === "timeout" ? "REQUEST_TIMEOUT" : "REQUEST_FAILED", message: controller.signal.reason === "timeout" ? "Backend request timed out. Refresh to retry." : "The backend response could not be loaded. Check the local service and retry." } });
    } finally {
      clearTimeout(timeout);
      if (state.request === controller) $("refresh").disabled = false;
    }
  }
  function renderPayload(payload) {
    const provenance = isObject(payload.provenance) ? payload.provenance : {};
    const data = isObject(payload.data) ? payload.data : {};
    const classification = provenance.classification || "UNKNOWN";
    const synthetic = String(classification).toUpperCase() === "SYNTHETIC";
    $("truth-strip").classList.toggle("synthetic", synthetic);
    $("state-badge").textContent = payload.state;
    $("state-badge").dataset.state = payload.state;
    $("source-label").textContent = `${printable(provenance.source)} · ${classification}`;
    $("asof-label").textContent = `As of ${provenance.as_of === null || provenance.as_of === undefined ? "unknown" : String(provenance.as_of)}`;
    $("truth-notice").textContent = `${payload.state === "STALE" ? "STALE · Historical snapshot; freshness requirement is not satisfied. " : ""}${synthetic ? "SYNTHETIC INPUTS · Demonstration and research only · No broker-observed performance" : "Research output requires review before any trading use."}`;
    if (data.title) $("section-title").textContent = String(data.title);
    if (data.description) $("section-description").textContent = String(data.description);
    const errorMessage = payload.error && (payload.error.message || payload.error.code);
    if (errorMessage) {
      $("error-panel").textContent = `${payload.error.code || payload.state}: ${errorMessage}`;
      $("error-panel").classList.remove("hidden");
    } else if (["UNAVAILABLE", "UNKNOWN"].includes(payload.state)) {
      $("error-panel").textContent = "This capability has no supported data available in the selected environment. Select another section or explicitly choose the synthetic research demo.";
      $("error-panel").classList.remove("hidden");
    }
    renderMetrics(data.metrics);
    const context = [
      ["Source identity", provenance.source],
      ["Data classification", classification],
      ["Evidence state", payload.state],
      ["Source timestamp", provenance.as_of],
      ["Trading authority", "Research only. No trading approval is implied."],
    ];
    if (data.description) context.push(["Method & scope", data.description]);
    if (data.notes) context.push(["Research notes", Array.isArray(data.notes) ? data.notes.join("\n") : data.notes]);
    if (data.warnings) context.push(["Limitations", Array.isArray(data.warnings) ? data.warnings.join("\n") : data.warnings]);
    Object.entries(provenance).filter(([key]) => !["source", "classification", "as_of"].includes(key)).forEach(([key, value]) => context.push([human(key), value]));
    const extra = Object.entries(data).filter(([key]) => !["title", "description", "metrics", "rows", "chart", "notes", "warnings"].includes(key));
    extra.forEach(([key, value]) => context.push([human(key), value]));
    $("context-content").replaceChildren(...context.map(([label, value]) => {
      const entry = el("div", "context-entry");
      entry.append(el("div", "context-label", label), el("div", "context-value", printable(value)));
      return entry;
    }));
    state.rows = Array.isArray(data.rows) ? data.rows.filter(isObject) : [];
    state.chart = isObject(data.chart) ? data.chart : null;
    renderTable();
    renderChart();
  }
  function renderMetrics(metrics) {
    const entries = isObject(metrics) ? Object.entries(metrics) : Array.isArray(metrics) ? metrics.map((item, index) => [item.label || `Metric ${index + 1}`, item]) : [];
    $("metrics").replaceChildren(...entries.slice(0, 12).map(([key, metric]) => {
      const item = isObject(metric) && scalarTag(metric) === null ? metric : { value: metric };
      const value = printable(item.value);
      const card = el("article", "metric-card");
      const label = el("div", "metric-label");
      label.append(el("span", "", item.label || human(key)), el("span", "metric-marker"));
      const valueElement = el("div", `metric-value${value.length > 15 ? " long-value" : ""}`, value);
      const detail = [item.unit, item.state, item.description || item.note].filter((part) => part !== null && part !== undefined && part !== "").join(" · ");
      card.append(label, valueElement, el("div", "metric-detail", detail || "Backend research metric"));
      return card;
    }));
  }
  function visibleRows() {
    const term = $("table-search").value.toLocaleLowerCase();
    const rows = state.rows.filter((row) => !term || Object.values(row).some((value) => printable(value).toLocaleLowerCase().includes(term)));
    if (state.sort) {
      const { key, direction } = state.sort;
      rows.sort((a, b) => {
        const x = a[key], y = b[key];
        if (x === null || x === undefined) return y === null || y === undefined ? 0 : 1;
        if (y === null || y === undefined) return -1;
        const exact = exactNumericOrder(x, y);
        const comparison = exact === null ? printable(x).localeCompare(printable(y)) : exact;
        return comparison * direction;
      });
    }
    return rows;
  }
  function renderTable() {
    const columns = [];
    state.rows.forEach((row) => Object.keys(row).forEach((key) => { if (!columns.includes(key)) columns.push(key); }));
    state.columns = columns;
    const header = el("tr");
    columns.forEach((key) => {
      const th = el("th");
      th.scope = "col";
      const sorted = state.sort && state.sort.key === key;
      if (sorted) th.setAttribute("aria-sort", state.sort.direction === 1 ? "ascending" : "descending");
      const button = el("button", "table-sort", `${human(key)}${sorted ? state.sort.direction === 1 ? " ↑" : " ↓" : ""}`);
      button.addEventListener("click", () => { state.sort = { key, direction: sorted ? -state.sort.direction : 1 }; renderTable(); });
      th.append(button);
      header.append(th);
    });
    $("observations-table").querySelector("thead").replaceChildren(header);
    const rows = visibleRows();
    $("observations-table").querySelector("tbody").replaceChildren(...rows.slice(0, 500).map((row) => {
      const tr = el("tr");
      columns.forEach((key) => {
        const value = row[key];
        const td = el("td", value === null || value === undefined ? "unknown" : typeof value === "number" ? "numeric" : "", printable(value));
        td.title = printable(value);
        tr.append(td);
      });
      return tr;
    }));
    $("row-count").textContent = numberFormat.format(state.rows.length);
    $("table-empty").classList.toggle("hidden", rows.length !== 0);
    $("table-wrap").classList.toggle("hidden", rows.length === 0);
    $("table-empty").textContent = state.rows.length ? "No observations match this filter." : "No supported observations in this response.";
    $("table-summary").textContent = `Showing ${Math.min(rows.length, 500)} of ${rows.length} matching observations · ${state.rows.length} total`;
    $("export-csv").disabled = !rows.length;
  }
  function exportCsv() {
    // Spreadsheet formulas are text-escaped; all exported fields are quoted.
    const csvValue = (value) => {
      let text = value === null || value === undefined ? "" : scalarTag(value) !== null ? scalarTag(value) : typeof value === "object" ? JSON.stringify(value) : String(value);
      if (scalarTag(value) !== null || (typeof value === "string" && (/^[\s\u0000-\u001f]*[=+\-@]/.test(text) || /^[\t\r]/.test(text)))) text = `'${text}`;
      return `"${text.replace(/"/g, '""')}"`;
    };
    const rows = visibleRows();
    const csv = [state.columns.map(csvValue).join(","), ...rows.map((row) => state.columns.map((key) => csvValue(row[key])).join(","))].join("\r\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const link = el("a");
    link.href = url;
    link.download = `gig-${state.section.id}-${state.mode}.csv`;
    link.setAttribute("aria-label", "Download visible research observations as CSV");
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function axisSpecs(chart) {
    const kind = String(chart.kind || "").replace(/_/g, "-");
    const definitions = kind.includes("volatility") ? [{ key: "strike", label: "Strike", unit: "currency" }, { key: "tenor_years", label: "Tenor", unit: "years" }, { key: "iv", label: "Implied volatility", unit: "annualized decimal" }] : kind.includes("risk") ? [{ key: "equity_shock", label: "Equity shock", unit: "decimal" }, { key: "volatility_shock", label: "Volatility shock", unit: "decimal" }, { key: "pnl", label: "Scenario P&L", unit: "model units" }] : [{ key: "x", label: "Factor X", unit: "standardized units" }, { key: "y", label: "Factor Y", unit: "standardized units" }, { key: "z", label: "Factor Z", unit: "standardized units" }];
    if (Array.isArray(chart.axes)) chart.axes.slice(0, 3).forEach((axis, index) => { if (typeof axis === "string") definitions[index].label = axis; else if (isObject(axis)) Object.assign(definitions[index], axis); });
    else if (isObject(chart.axes)) ["x", "y", "z"].forEach((axis, index) => { const supplied = chart.axes[axis]; if (typeof supplied === "string") definitions[index].label = supplied; else if (isObject(supplied)) Object.assign(definitions[index], supplied); });
    return definitions;
  }
  function project(x, y, z, width, height) {
    const scale = Math.min(width * .29, height * .33) * state.zoom;
    if (state.dimension === 2) return { x: width * .48 + x * scale, y: height * .51 - z * scale, depth: y };
    const a = state.angle, b = state.tilt;
    const horizontal = x * Math.cos(a) - y * Math.sin(a);
    const depth = x * Math.sin(a) + y * Math.cos(a);
    return { x: width * .48 + horizontal * scale, y: height * .51 - (z * Math.cos(b) - depth * Math.sin(b)) * scale, depth };
  }
  function renderChart() {
    const canvas = $("analytics-chart");
    const context = canvas.getContext("2d");
    if (!context) return;
    const bounds = canvas.getBoundingClientRect();
    const width = Math.max(bounds.width, 1), height = Math.max(bounds.height, 1);
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * pixelRatio);
    canvas.height = Math.round(height * pixelRatio);
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, width, height);
    const style = getComputedStyle(document.documentElement);
    const muted = style.getPropertyValue("--muted").trim();
    const grid = style.getPropertyValue("--grid").trim();
    const accent = style.getPropertyValue("--accent").trim();
    $("chart-tooltip").classList.add("hidden");
    state.projected = [];
    const chart = state.chart;
    const axes = chart ? axisSpecs(chart) : [];
    const rawPoints = chart && Array.isArray(chart.points) ? chart.points : [];
    const points = rawPoints.filter((point) => isObject(point) && axes.every((axis) => typeof point[axis.key] === "number" && Number.isFinite(point[axis.key]))).slice(0, 1600);
    state.points = points;
    $("chart-empty").classList.toggle("hidden", points.length !== 0);
    $("chart-title").textContent = chart && chart.title ? String(chart.title) : chart && chart.kind ? human(chart.kind) : `${state.section.label} analytical view`;
    if (!points.length) { $("chart-legend").textContent = "Only finite backend observations are plotted."; return; }
    $("chart-legend").textContent = axes.map((axis) => `${axis.label} (${axis.unit || "unspecified units"})`).join(" · ") + (state.dimension === 2 ? ` · 2D projection: ${axes[0].label} vs ${axes[2].label}; ${axes[1].label} retained in tooltips` : "") + (rawPoints.length > 1600 ? " · Plot limited to 1,600 observations; table retains rows" : "");
    const ranges = axes.map((axis) => {
      let lo = Infinity, hi = -Infinity;
      points.forEach((point) => { lo = Math.min(lo, point[axis.key]); hi = Math.max(hi, point[axis.key]); });
      return [lo, hi];
    });
    const normalize = (value, index) => ranges[index][0] === ranges[index][1] ? 0 : 2 * (value - ranges[index][0]) / (ranges[index][1] - ranges[index][0]) - 1;
    const line = (from, to, color = grid, lineWidth = 1) => {
      const a = project(...from, width, height), b = project(...to, width, height);
      context.beginPath(); context.strokeStyle = color; context.lineWidth = lineWidth; context.moveTo(a.x, a.y); context.lineTo(b.x, b.y); context.stroke();
    };
    for (let index = 0; index <= 4; index++) {
      const tick = -1 + index / 2;
      line([-1, tick, -1], [1, tick, -1]);
      line([tick, -1, -1], [tick, 1, -1]);
      if (state.dimension === 2) line([-1, -1, tick], [1, -1, tick]);
    }
    const origin = [-1, -1, -1];
    const endpoints = [[1.12, -1, -1], [-1, 1.12, -1], [-1, -1, 1.12]];
    const axisColors = [accent, "#cf9e69", "#64bba5"];
    context.font = "9px 'SFMono-Regular', Consolas, monospace";
    endpoints.forEach((endpoint, index) => {
      if (state.dimension === 2 && index === 1) return;
      line(origin, endpoint, axisColors[index], 1.2);
      const labelPoint = project(...endpoint, width, height);
      context.fillStyle = axisColors[index];
      const label = axes[index].label;
      context.fillText(label.length > 24 ? `${label.slice(0, 22)}…` : label, Math.max(8, Math.min(labelPoint.x + 7, width - 115)), Math.max(14, Math.min(labelPoint.y - 5, height - 10)));
      context.fillStyle = muted;
      [0, 1].forEach((end) => {
        const tickWorld = [...origin]; tickWorld[index] = end ? 1 : -1;
        const tickPoint = project(...tickWorld, width, height);
        context.fillText(printable(ranges[index][end]), Math.max(6, Math.min(tickPoint.x + 4, width - 65)), Math.max(12, Math.min(tickPoint.y + (index === 2 ? -7 : 13), height - 5)));
      });
    });
    const projected = points.map((point, index) => {
      const coordinates = axes.map((axis, axisIndex) => normalize(point[axis.key], axisIndex));
      return { ...project(...coordinates, width, height), point, index, coordinates };
    });
    if (String(chart.kind).includes("surface")) {
      // Connect only observations sharing an actual sampled axis value; no interpolation.
      for (let direction = 0; direction < 2; direction++) {
        const groups = new Map();
        projected.forEach((point) => { const key = point.point[axes[direction].key]; if (!groups.has(key)) groups.set(key, []); groups.get(key).push(point); });
        groups.forEach((group) => {
          group.sort((a, b) => a.point[axes[1 - direction].key] - b.point[axes[1 - direction].key]);
          for (let index = 1; index < group.length; index++) line(group[index - 1].coordinates, group[index].coordinates, grid, 1);
        });
      }
    }
    projected.sort((a, b) => a.depth - b.depth);
    projected.forEach((point) => {
      const colorPosition = Math.max(0, Math.min(1, (point.coordinates[2] + 1) / 2));
      context.fillStyle = `hsl(${195 + colorPosition * 30} 65% ${document.documentElement.dataset.theme === "light" ? 40 : 55 + colorPosition * 17}%)`;
      context.beginPath();
      context.arc(point.x, point.y, String(chart.kind).includes("surface") ? 2.7 : 3.7, 0, Math.PI * 2);
      context.fill();
      if (point.point.symbol && points.length <= 25) { context.fillStyle = muted; context.font = "8px 'SFMono-Regular', Consolas, monospace"; context.fillText(String(point.point.symbol), point.x + 6, point.y - 5); }
    });
    state.projected = projected;
    canvas.setAttribute("aria-label", `${$("chart-title").textContent}. ${points.length} backend observations. ${axes.map((axis) => `${axis.label}, ${axis.unit || "units unspecified"}`).join("; ")}. ${state.dimension}D view. Underlying observations also available in the table. Drag or use arrow keys to rotate; plus and minus zoom.`);
  }
  function chartPointer(event) {
    const bounds = $("analytics-chart").getBoundingClientRect();
    const x = event.clientX - bounds.left, y = event.clientY - bounds.top;
    if (state.drag) { state.angle += (event.clientX - state.drag.x) / 180; state.tilt = Math.max(-1.2, Math.min(1.2, state.tilt + (event.clientY - state.drag.y) / 220)); state.drag = { x: event.clientX, y: event.clientY }; renderChart(); return; }
    let selected = null, distance = 12;
    state.projected.forEach((point) => { const d = Math.hypot(point.x - x, point.y - y); if (d < distance) { selected = point; distance = d; } });
    const tooltip = $("chart-tooltip");
    tooltip.classList.toggle("hidden", selected === null);
    if (selected) {
      tooltip.textContent = Object.entries(selected.point).slice(0, 10).map(([key, value]) => `${human(key)}: ${printable(value)}`).join("\n");
      tooltip.style.left = `${Math.max(4, Math.min(x + 15, bounds.width - 225))}px`;
      tooltip.style.top = `${Math.max(4, Math.min(y + 12, bounds.height - 140))}px`;
    }
  }
  const examples = {
    options: { spot: 100, strike: 100, rate: 0.05, div: 0, vol: 0.2, ttm: 1, call: true },
    kalman: { observations: [[1], [2], [1.5], [2.2], [2.4]], transition: [[1]], observation: [[1]], process_covariance: [[0.1]], observation_covariance: [[1]], initial_mean: [0], initial_covariance: [[1]], smooth: true },
    garch: { returns: [0.01, -0.015, 0.007, 0.02, -0.025, 0.012, -0.008, 0.009, -0.018, 0.013, 0.005, -0.009, 0.014, -0.021, 0.011, 0.008, -0.012, 0.017, -0.006, 0.009, -0.013, 0.004, 0.016, -0.008, 0.012, -0.019, 0.007, 0.015, -0.01, 0.006], max_iterations: 100 },
    hmm: { observations: [-0.02, -0.01, 0.01, 0.02, -0.01, 0.03, 0.02, -0.02], states: 2, max_iterations: 50 },
    microstructure: { quotes: [{ timestamp_ns: 100, bid: 99, ask: 101, bid_size: 10, ask_size: 12, instrument: "SYN001", source: "deterministic-demo", data_kind: "SYNTHETIC" }], trades: [{ timestamp_ns: 101, price: 101, size: 5, instrument: "SYN001", source: "deterministic-demo", data_kind: "SYNTHETIC" }] },
  };
  function loadExample() { $("analytics-input").value = JSON.stringify(examples[$("analytics-kind").value], null, 2); }
  async function runAnalysis() {
    const output = $("analytics-output");
    const sourceText = $("analytics-input").value;
    let inputs;
    try { inputs = JSON.parse(sourceText); if (!isObject(inputs)) throw new Error(); }
    catch { output.textContent = "Input must be a valid JSON object. No request was sent."; return; }
    const button = $("run-analysis");
    button.disabled = true;
    output.textContent = "Computing with the local backend…";
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 30000);
    try {
      const response = await fetch(`/api/v1/analytics/${encodeURIComponent($("analytics-kind").value)}`, { method: "POST", headers: { "Content-Type": "application/json", Accept: "application/json" }, body: sourceText, signal: controller.signal, cache: "no-store" });
      const result = await response.json();
      output.textContent = `${response.ok ? "" : `HTTP ${response.status}\n`}${JSON.stringify(result, null, 2)}`;
    } catch { output.textContent = "Analysis response could not be loaded. Check the local service, supported input contract, and retry."; }
    finally { clearTimeout(timer); button.disabled = false; }
  }
  function setTheme(theme) {
    document.documentElement.dataset.theme = theme;
    $("theme-toggle").setAttribute("aria-label", `Switch to ${theme === "dark" ? "light" : "dark"} theme`);
    try { localStorage.setItem("gig-theme", theme); } catch { /* Privacy mode may deny persistence. */ }
    renderChart();
  }
  nav();
  let storedTheme;
  try { storedTheme = localStorage.getItem("gig-theme"); } catch { /* Keep default theme. */ }
  setTheme(["light", "dark"].includes(storedTheme) ? storedTheme : "dark");
  const updateClock = () => { const now = new Date(); $("clock").dateTime = now.toISOString(); $("clock").textContent = `${now.toISOString().slice(11, 19)} UTC`; };
  updateClock(); setInterval(updateClock, 1000);
  $("theme-toggle").addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
  $("data-mode").addEventListener("change", () => { state.mode = $("data-mode").value; loadSection(); });
  $("refresh").addEventListener("click", loadSection);
  $("table-search").addEventListener("input", renderTable);
  $("export-csv").addEventListener("click", exportCsv);
  $("analytics-kind").addEventListener("change", loadExample);
  $("load-example").addEventListener("click", loadExample);
  $("run-analysis").addEventListener("click", runAnalysis);
  window.addEventListener("hashchange", selectSection);
  [2, 3].forEach((dimension) => $(`view-${dimension}d`).addEventListener("click", () => { state.dimension = dimension; [2, 3].forEach((d) => $(`view-${d}d`).setAttribute("aria-pressed", String(d === dimension))); renderChart(); }));
  $("chart-reset").addEventListener("click", () => { state.angle = -.55; state.tilt = .40; state.zoom = 1; renderChart(); });
  const canvas = $("analytics-chart");
  canvas.addEventListener("pointerdown", (event) => { if (event.button !== 0) return; state.drag = { x: event.clientX, y: event.clientY }; canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener("pointermove", chartPointer);
  canvas.addEventListener("pointerup", () => { state.drag = null; });
  canvas.addEventListener("pointercancel", () => { state.drag = null; });
  canvas.addEventListener("pointerleave", () => $("chart-tooltip").classList.add("hidden"));
  canvas.addEventListener("wheel", (event) => { if (!state.points.length) return; event.preventDefault(); state.zoom = Math.max(.45, Math.min(2.5, state.zoom * (event.deltaY > 0 ? .92 : 1.08))); renderChart(); }, { passive: false });
  canvas.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "+", "=", "-", "0"].includes(event.key)) return;
    event.preventDefault();
    if (event.key === "ArrowLeft") state.angle -= .1;
    if (event.key === "ArrowRight") state.angle += .1;
    if (event.key === "ArrowUp") state.tilt = Math.min(1.2, state.tilt + .1);
    if (event.key === "ArrowDown") state.tilt = Math.max(-1.2, state.tilt - .1);
    if (event.key === "+" || event.key === "=") state.zoom = Math.min(2.5, state.zoom * 1.1);
    if (event.key === "-") state.zoom = Math.max(.45, state.zoom / 1.1);
    if (event.key === "0") { state.angle = -.55; state.tilt = .40; state.zoom = 1; }
    renderChart();
  });
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(renderChart).observe($("chart-container"));
  else window.addEventListener("resize", renderChart);
  selectSection();
})();
