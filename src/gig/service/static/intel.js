/* GIG Global Intelligence workspace.
 * Reads /api/v1/intel/*. Nothing here contacts a provider: refreshes are explicit
 * POSTs that the server performs against its fixed registry. Map geometry is the
 * bundled Natural Earth file, loaded the first time the workspace opens. */
"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const DAY = 86400000;
  const LAYER_LABELS = { hazards: "Natural hazards", geopolitical: "Geopolitical", defense: "Defense research", maritime: "Maritime", aviation: "Aviation", imagery: "Satellite imagery", reference: "Chokepoints & ports" };
  const CATEGORY_COLORS = { EARTHQUAKE: "#f2be4c", WILDFIRE: "#f05259", SEVERE_STORM: "#408bff", VOLCANO: "#ff8a4c", FLOOD: "#3fc1c9", SEA_LAKE_ICE: "#a6b2c3", DROUGHT: "#c9a86a", LANDSLIDE: "#b08968",
    ARMY: "#c08bff", AIR_FORCE: "#7aa7ff", NAVY: "#16c784", GEOPOLITICAL: "#f2f5f9" };
  const SEMANTIC_KIND = { LIVE: "live", DELAYED: "delayed", HISTORICAL: "historical", CACHED: "cached", DEMONSTRATION: "demo", HISTORICAL_AND_DEMONSTRATION: "demo", UNAVAILABLE: "unavailable" };
  const VERIFICATION_LABEL = { CONFIRMED_OFFICIAL: "Confirmed · official source", MULTI_SOURCE_REPORTED: "Reported by multiple publishers", UNVERIFIED_REPORT: "Unverified report",
    HISTORICAL_OBSERVATION: "Historical observation", ANALYTICAL_SCENARIO: "Analytical scenario", PROVIDER_REVIEWED: "Provider reviewed", PROVIDER_AUTOMATIC_SUBJECT_TO_REVISION: "Provider automatic · subject to revision",
    PROVIDER_CURATED: "Provider curated", DEMONSTRATION_NOT_OBSERVED: "Demonstration · not observed" };
  const BRANCHES = [["ARMY", "Army"], ["AIR_FORCE", "Air Force"], ["NAVY", "Navy"], ["GEOPOLITICAL", "Geopolitical"]];
  const reduced = typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const state = { started: false, world: null, data: null, layers: new Set(["hazards", "geopolitical", "defense", "reference"]), days: 30, query: "", asOf: null,
    view: "globe", lon0: 40, lat0: 18, zoom: 1, panY: 0, flatLon: 0, flatAuto: true, selected: null, hover: null, drag: null, projected: [], playing: null, branch: "NAVY", ticket: 0, rotate: null };

  const el = (tag, cls, text) => { const node = document.createElement(tag); if (cls) node.className = cls; if (text !== undefined && text !== null) node.textContent = String(text); return node; };
  const iso = (value) => (value ? String(value).replace("+00:00", "Z") : "Unknown");
  const duration = (seconds) => {
    if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "Unknown";
    const s = Math.abs(seconds); if (s < 3600) return `${Math.round(s / 60)} min`; if (s < 172800) return `${(s / 3600).toFixed(1)} h`; return `${(s / 86400).toFixed(1)} days`;
  };
  const badge = (text, kind) => { const b = el("span", `intel-badge ${kind || ""}`, text.replaceAll("_", " ")); return b; };
  const say = (text) => { $("intel-live").textContent = text; };

  async function call(path, body) {
    const options = body === undefined ? { headers: { Accept: "application/json" }, cache: "no-store" }
      : { method: "POST", headers: { Accept: "application/json", "Content-Type": "application/json" }, body: JSON.stringify(body) };
    const response = await fetch(path, options);
    let payload = null; try { payload = await response.json(); } catch { /* reported below */ }
    if (!response.ok || !payload || payload.state === "ERROR") {
      const message = payload && payload.error ? `${payload.error.code}: ${payload.error.message}` : payload && payload.detail ? JSON.stringify(payload.detail) : `HTTP ${response.status}`;
      throw new Error(message);
    }
    return payload;
  }

  /* ---------- projection ---------- */
  const rad = Math.PI / 180;
  function geometry() { const c = $("intel-canvas"); const r = c.getBoundingClientRect(); return { w: r.width, h: r.height }; }
  function project(lat, lon, copy) {
    const { w, h } = geometry();
    if (state.view === "globe") {
      const R = Math.min(w, h) * 0.44 * state.zoom; const l = (lon - state.lon0) * rad, p = lat * rad, p0 = state.lat0 * rad;
      const x = Math.cos(p) * Math.sin(l), y = Math.cos(p0) * Math.sin(p) - Math.sin(p0) * Math.cos(p) * Math.cos(l);
      const visible = Math.sin(p0) * Math.sin(p) + Math.cos(p0) * Math.cos(p) * Math.cos(l) >= 0;
      if (visible) return { x: w / 2 + R * x, y: h / 2 - R * y, visible };
      const n = Math.hypot(x, y) || 1; return { x: w / 2 + (R * x) / n, y: h / 2 - (R * y) / n, visible };
    }
    const scale = flatScale(w, h);
    const dlon = copy === undefined ? wrapLon(lon - state.flatLon) : lon - state.flatLon + copy * 360;
    return { x: w / 2 + (dlon * scale), y: h / 2 + state.panY - (lat * scale), visible: true };
  }
  // The flat map fills the canvas height at zoom 1 and wraps horizontally, so
  // there are no empty bands and no fixed antimeridian seam.
  const FLAT_LAT = 85;
  const wrapLon = (lon) => ((((lon + 180) % 360) + 360) % 360) - 180;
  function flatScale(w, h) { return Math.max(w / 360, h / (2 * FLAT_LAT)) * state.zoom; }
  function clampPan() {
    const { w, h } = geometry(); const half = FLAT_LAT * flatScale(w, h);
    const limit = Math.max(0, half - h / 2); state.panY = Math.max(-limit, Math.min(limit, state.panY));
    state.flatLon = wrapLon(state.flatLon);
  }
  function flatCentre() {
    const { w, h } = geometry(); const half = (w / 2) / flatScale(w, h);
    const lons = events().filter((e) => Number.isFinite(e.longitude)).map((e) => e.longitude);
    if (!lons.length || half >= 180) return 0;
    let best = 0, bestCount = -1;
    for (let c = -180; c < 180; c += 5) {
      const count = lons.filter((lon) => Math.abs(wrapLon(lon - c)) <= half - 4).length;
      if (count > bestCount || (count === bestCount && Math.abs(c) < Math.abs(best))) { best = c; bestCount = count; }
    }
    return best;
  }
  function resetFlat() { state.zoom = 1; state.panY = 0; state.flatLon = flatCentre(); state.flatAuto = true; }
  function kmToPixels(km) {
    const { w, h } = geometry();
    const pxPerDegree = state.view === "globe" ? (Math.min(w, h) * 0.44 * state.zoom * rad) : flatScale(w, h);
    return (km / 111.2) * pxPerDegree;
  }

  /* ---------- drawing ---------- */
  function draw() {
    const canvas = $("intel-canvas"); if (!canvas || $("intel-panel").classList.contains("hidden")) return;
    const { w, h } = geometry(); const ratio = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(w * ratio) || canvas.height !== Math.round(h * ratio)) { canvas.width = Math.round(w * ratio); canvas.height = Math.round(h * ratio); }
    const ctx = canvas.getContext("2d"); ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#050608"; ctx.fillRect(0, 0, w, h);
    if (state.view !== "globe") clampPan();
    if (state.view === "globe") {
      const R = Math.min(w, h) * 0.44 * state.zoom;
      const glow = ctx.createRadialGradient(w / 2, h / 2, R * 0.2, w / 2, h / 2, R);
      glow.addColorStop(0, "#0e1622"); glow.addColorStop(1, "#090d13");
      ctx.beginPath(); ctx.arc(w / 2, h / 2, R, 0, Math.PI * 2); ctx.fillStyle = glow; ctx.fill(); ctx.strokeStyle = "#26303d"; ctx.lineWidth = 1; ctx.stroke();
    } else {
      const top = project(FLAT_LAT, 0).y, bottom = project(-FLAT_LAT, 0).y;
      ctx.fillStyle = "#090d13"; ctx.fillRect(0, top, w, bottom - top);
    }
    const copies = state.view === "globe" ? [undefined] : [-1, 0, 1];
    for (const copy of copies) {
      graticule(ctx, copy);
      if (!state.world) continue;
      ctx.fillStyle = "#141a22"; ctx.strokeStyle = "#2f3b4a"; ctx.lineWidth = 0.7;
      for (const ring of state.world.land) {
        ctx.beginPath(); let any = false;
        for (let i = 0; i < ring.length; i += 2) { const p = project(ring[i + 1], ring[i], copy); if (p.visible) any = true; if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y); }
        if (any) { ctx.closePath(); ctx.fill(); ctx.stroke(); }
      }
      ctx.strokeStyle = "#26303d"; ctx.lineWidth = 0.5;
      for (const line of state.world.borders) polyline(ctx, line, copy);
    }
    state.projected = [];
    const data = state.data;
    if (data && state.layers.has("reference")) {
      data.reference.chokepoints.forEach((c) => { const p = project(c.lat, c.lon); if (!p.visible) return;
        ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(Math.PI / 4); ctx.fillStyle = "#050608"; ctx.strokeStyle = "#f2be4c"; ctx.lineWidth = 1.2; ctx.fillRect(-4, -4, 8, 8); ctx.strokeRect(-4, -4, 8, 8); ctx.restore();
        state.projected.push({ x: p.x, y: p.y, r: 7, kind: "chokepoint", item: c }); });
      data.reference.ports.forEach((c) => { const p = project(c.lat, c.lon); if (!p.visible) return;
        ctx.fillStyle = "#8292a5"; ctx.fillRect(p.x - 2, p.y - 2, 4, 4); state.projected.push({ x: p.x, y: p.y, r: 5, kind: "port", item: c }); });
    }
    if (data) {
      for (const event of events()) {
        if (event.latitude === null || event.longitude === null) continue;
        const p = project(event.latitude, event.longitude); if (!p.visible) continue;
        const color = CATEGORY_COLORS[event.category] || "#a6b2c3";
        if (event.geo_precision_km) { const r = kmToPixels(event.geo_precision_km); if (r > 4 && r < 400) { ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.strokeStyle = `${color}55`; ctx.setLineDash([3, 3]); ctx.stroke(); ctx.setLineDash([]); } }
        const size = event.magnitude && event.magnitude_unit && /^m/i.test(event.magnitude_unit) ? 2.5 + Math.max(0, event.magnitude - 2) * 1.6 : 4.5;
        ctx.beginPath(); ctx.arc(p.x, p.y, size, 0, Math.PI * 2); ctx.fillStyle = `${color}${event.provider === "DEMONSTRATION" || event.data_classification === "SYNTHETIC" ? "66" : "cc"}`; ctx.fill();
        ctx.lineWidth = event.provider === "DEMONSTRATION" || event.data_classification === "SYNTHETIC" ? 1.4 : 0.8; ctx.strokeStyle = color;
        if (event.provider === "DEMONSTRATION" || event.data_classification === "SYNTHETIC") ctx.setLineDash([2, 2]);
        ctx.stroke(); ctx.setLineDash([]);
        if (state.selected && state.selected.event_id === event.event_id) { ctx.beginPath(); ctx.arc(p.x, p.y, size + 5, 0, Math.PI * 2); ctx.strokeStyle = "#f2f5f9"; ctx.lineWidth = 1.5; ctx.stroke(); }
        state.projected.push({ x: p.x, y: p.y, r: Math.max(6, size + 2), kind: "event", item: event });
      }
    }
  }
  function polyline(ctx, line, copy) {
    ctx.beginPath(); let pen = false;
    for (let i = 0; i < line.length; i += 2) { const p = project(line[i + 1], line[i], copy); if (!p.visible) { pen = false; continue; } if (pen) ctx.lineTo(p.x, p.y); else ctx.moveTo(p.x, p.y); pen = true; }
    ctx.stroke();
  }
  function graticule(ctx, copy) {
    ctx.strokeStyle = "#141a22"; ctx.lineWidth = 0.5;
    for (let lon = -180; lon <= 180; lon += 30) { const line = []; for (let lat = -80; lat <= 80; lat += 5) line.push(lon, lat); polyline(ctx, line, copy); }
    for (let lat = -60; lat <= 60; lat += 30) { const line = []; for (let lon = -180; lon <= 180; lon += 5) line.push(lon, lat); polyline(ctx, line, copy); }
  }
  const events = () => (state.data ? state.data.events.filter((e) => state.layers.has(e.layer)) : []);

  /* ---------- panels ---------- */
  function renderAll() {
    const data = state.data; if (!data) return;
    const semantics = $("intel-semantics"); semantics.replaceChildren();
    Object.entries(data.data_semantics).forEach(([source, value]) => { const item = el("span", "intel-semantic"); item.append(el("span", "intel-semantic-source", source.replaceAll("_", " ")), badge(value, SEMANTIC_KIND[value] || "unavailable")); semantics.append(item); });
    $("intel-asof").textContent = `${state.asOf ? "Replay cutoff" : "As of"} ${iso(data.as_of)} · ${data.window_days}-day window · ${data.event_count} events`;
    renderTruth(data);
    if (state.view !== "globe" && state.flatAuto) state.flatLon = flatCentre();
    renderLayers(); renderFeed(); renderHealth(); renderRisk(); renderDefense(); renderCoverage(); renderInvestigations(); renderDetails(); draw();
  }
  // The shared truth strip belongs to whichever section is visible, so this
  // section must replace the previous section's provenance with its own.
  function renderTruth(data) {
    if ($("intel-panel").classList.contains("hidden")) return;
    const entries = Object.entries(data.data_semantics);
    const stateText = data.event_count > 0 ? "AVAILABLE" : "UNAVAILABLE";
    $("state-badge").textContent = stateText; $("state-badge").dataset.state = stateText;
    $("source-label").textContent = `global-intelligence · ${entries.length ? entries.map(([s, v]) => `${s.replaceAll("_", " ")} ${v}`).join(" · ") : "no source acquired"}`;
    $("asof-label").textContent = `${state.asOf ? "Replay cutoff" : "As of"} ${iso(data.as_of)}`;
    $("truth-notice").textContent = "Delayed public sources and cited reports · Not a live or tactical feed · Research only";
  }
  function truthLoading() {
    $("state-badge").textContent = "LOADING"; $("state-badge").removeAttribute("data-state");
    $("source-label").textContent = "global-intelligence · fetching sources…"; $("asof-label").textContent = "As of —";
    $("truth-notice").textContent = "Delayed public sources and cited reports · Not a live or tactical feed · Research only";
  }
  function renderLayers() {
    const box = $("intel-layers"); box.replaceChildren();
    state.data.layers.forEach((layer) => {
      const label = el("label", `intel-layer${layer.available ? "" : " blocked"}`); const input = el("input"); input.type = "checkbox";
      input.checked = layer.available && state.layers.has(layer.layer); input.disabled = !layer.available; input.value = layer.layer;
      input.addEventListener("change", () => { if (input.checked) state.layers.add(layer.layer); else state.layers.delete(layer.layer); renderFeed(); draw(); });
      label.append(input, el("span", null, LAYER_LABELS[layer.layer] || layer.layer)); if (!layer.available) { label.append(badge("UNAVAILABLE", "unavailable")); label.title = layer.blocker || "Unavailable"; }
      box.append(label);
      if (!layer.available) { const why = el("button", "text-button intel-why", `Why is ${LAYER_LABELS[layer.layer].toLowerCase()} unavailable?`); why.type = "button"; why.dataset.layer = layer.layer; why.addEventListener("click", () => select({ layer_ref: layer.layer })); box.append(why); }
    });
  }
  function renderFeed() {
    const list = $("intel-feed"); list.replaceChildren(); const rows = events();
    $("intel-feed-count").textContent = `${rows.length}`;
    if (!rows.length) { list.append(el("li", "intel-empty", "No events in this window for the selected layers. Absence of an event is not evidence of absence.")); return; }
    rows.slice(0, 200).forEach((event) => {
      const item = el("li", "intel-feed-item"); const button = el("button", "intel-feed-button"); button.type = "button";
      button.setAttribute("aria-pressed", String(Boolean(state.selected && state.selected.event_id === event.event_id)));
      const top = el("span", "intel-feed-top"); const dot = el("span", "intel-dot"); dot.style.background = CATEGORY_COLORS[event.category] || "#a6b2c3";
      top.append(dot, el("span", "intel-feed-category", (event.kind || event.category).replaceAll("_", " ")), el("time", "intel-feed-time", iso(event.observed_at)));
      const meta = el("span", "intel-feed-meta", `${event.provider.replaceAll("_", " ")} · ${VERIFICATION_LABEL[event.verification] || event.verification} · delay ${duration(event.reporting_delay_seconds)}`);
      button.append(top, el("span", "intel-feed-title", event.title), meta);
      button.addEventListener("click", () => select(event));
      item.append(button); list.append(item);
    });
  }
  function row(dl, key, value) { dl.append(el("dt", null, key), el("dd", null, value === null || value === undefined || value === "" ? "Unknown" : value)); }
  function renderDetails() {
    const box = $("intel-details"); box.replaceChildren(); const item = state.selected;
    if (!item) { box.append(el("p", "intel-muted", "Select an event on the map or in the feed. Use N / P on the map to step through events.")); return; }
    if (item.layer_ref) { layerDetails(box, item.layer_ref); return; }
    if (item.kind_ref) {
      box.append(el("h3", null, item.name), badge("REFERENCE", "historical"));
      const dl = el("dl", "intel-dl"); row(dl, "Type", item.kind_ref === "chokepoint" ? "Maritime chokepoint" : "Port"); row(dl, "Approximate location", `${item.lat.toFixed(2)}°, ${item.lon.toFixed(2)}°`);
      if (item.themes) row(dl, "Research themes", item.themes.map((t) => state.data.themes[t].label).join(", "));
      row(dl, "Source", "GIG authored reference; no traffic volumes asserted"); box.append(dl); return;
    }
    const e = item; const semantics = state.data.data_semantics[e.provider] || "UNAVAILABLE";
    const head = el("div", "intel-detail-head"); head.append(badge(semantics, SEMANTIC_KIND[semantics]), badge(VERIFICATION_LABEL[e.verification] || e.verification, /CONFIRMED|REVIEWED|CURATED|MULTI/.test(e.verification) ? "confirmed" : /SCENARIO|DEMONSTRATION/.test(e.verification) ? "demo" : "unverified"));
    if (e.data_classification) head.append(badge(e.data_classification === "SYNTHETIC" ? "AUTHORED SYNTHETIC" : e.data_classification, e.data_classification === "SYNTHETIC" ? "demo" : "historical"));
    box.append(head, el("h3", null, e.title));
    if (e.summary) box.append(el("p", "intel-summary", e.summary));
    const dl = el("dl", "intel-dl");
    row(dl, "Category", `${e.category.replaceAll("_", " ")}${e.kind ? ` · ${e.kind.replaceAll("_", " ")}` : ""}`);
    row(dl, "Source", `${e.provider_name} — ${e.dataset}`);
    row(dl, "Observation time", `${iso(e.observed_at)}${e.observed_end ? ` → ${iso(e.observed_end)}` : ""}${e.observation_time_known === false ? " (publication time; observation time not stated)" : ""}`);
    row(dl, "Publication time", e.published_at ? iso(e.published_at) : "Not supplied by provider");
    if (e.provider_updated_at) row(dl, "Provider last updated", iso(e.provider_updated_at));
    row(dl, "Acquired by GIG", iso(e.acquired_at));
    row(dl, "Reporting delay", duration(e.reporting_delay_seconds));
    row(dl, "Location", e.latitude === null ? "Not located" : `${e.latitude.toFixed(2)}°, ${e.longitude.toFixed(2)}°${e.region ? ` · ${e.region}` : ""}`);
    row(dl, "Geographic uncertainty", `${e.geo_precision_km ? `±${e.geo_precision_km} km` : "Unknown"} — ${e.geo_precision_note || ""}`);
    if (e.magnitude !== null && e.magnitude !== undefined) row(dl, "Magnitude", `${e.magnitude} ${e.magnitude_unit || ""}`);
    if (e.depth_km !== null && e.depth_km !== undefined) row(dl, "Depth", `${e.depth_km} km`);
    row(dl, "Revisions held", `${e.revision_count || 1}`); row(dl, "Revision hash", e.revision_sha256 ? e.revision_sha256.slice(0, 16) + "…" : null);
    row(dl, "License", `${e.license}${e.attribution ? ` · ${e.attribution}` : ""}`);
    if (e.interpretation) row(dl, "Interpretation", e.interpretation === "NOT_EVIDENCE_OF_IMPENDING_CONFLICT" ? "An exercise, deployment or notice is not evidence of impending conflict." : e.interpretation);
    box.append(dl);
    const cites = el("div", "intel-citations"); cites.append(el("h4", null, "Citations"));
    if (!e.citations || !e.citations.length) cites.append(el("p", "intel-muted", e.provider === "DEMONSTRATION" ? "None — authored demonstration record." : "No citation supplied."));
    (e.citations || []).forEach((c) => {
      const line = el("p", "intel-cite"); line.append(el("span", null, `${c.publisher}${c.publisher_type ? ` (${c.publisher_type.replaceAll("_", " ").toLowerCase()})` : ""}: `));
      if (typeof c.url === "string" && c.url.startsWith("https://")) { const a = el("a", null, c.url); a.href = c.url; a.rel = "noopener noreferrer"; a.target = "_blank"; line.append(a); } else line.append(el("span", null, c.url || "no URL"));
      cites.append(line);
    });
    box.append(cites);
    const links = e.research_links; if (links) {
      const block = el("div", "intel-links"); block.append(el("h4", null, "Market research links (hypotheses, not causal)"));
      const chips = el("div", "intel-chips"); links.themes.forEach((t) => { const chip = el("span", "intel-chip", t.label); chip.title = t.markets.join(", "); chips.append(chip); }); block.append(chips);
      if (links.nearby_chokepoints.length) block.append(el("p", "intel-muted", `Within ${links.radius_km} km: ${links.nearby_chokepoints.map((c) => `${c.chokepoint} (${c.distance_km} km)`).join(", ")}`));
      const study = el("button", "button", "Prepare event study at this time"); study.type = "button";
      study.addEventListener("click", () => { const input = $("intel-study-input"); try { const body = JSON.parse(input.value); body.event_at = e.observed_at; body.event_id = e.event_id; input.value = JSON.stringify(body, null, 2); say("Event time copied into the event-study inputs. The example prices remain authored synthetic data."); $("intel-study-input").focus(); } catch { say("Event-study input is not valid JSON."); } });
      block.append(study); box.append(block);
    }
  }
  const LAYER_ALTERNATIVES = {
    maritime: { title: "Maritime intelligence", related: (e) => ["NAVY"].includes(e.category) && ["SHIPPING_DISRUPTION", "MARITIME_SECURITY_NOTICE"].includes(e.kind) || (e.research_links?.nearby_chokepoints || []).length > 0,
      instead: ["Approximate chokepoint and port reference geometry (◆ and ■ on the map); no traffic volumes are asserted.", "Hazard events within 500 km of a chokepoint, linked to tanker and container-shipping research themes.", "Cited, at-least-24-hour-delayed shipping-disruption and maritime-security notices admitted under Navy."] },
    aviation: { title: "Civil aviation intelligence", related: (e) => e.category === "VOLCANO" || ["AIRSPACE_RESTRICTION_NOTICE", "CIVIL_AVIATION_DISRUPTION"].includes(e.kind),
      instead: ["Volcanic events from NASA EONET, linked to the aviation and travel research theme (ash advisories affect routing).", "Cited, at-least-24-hour-delayed airspace-restriction and civil-aviation-disruption notices admitted under Air Force."] },
    imagery: { title: "Satellite imagery viewer", related: (e) => e.provider === "NASA_EONET",
      instead: ["NASA EONET events cite their upstream provider pages, which may reference imagery; the terminal itself loads no external tiles or scenes.", "No live video exists for these sources; Sentinel and Landsat imagery is periodic and would require registered accounts."] },
  };
  function layerDetails(box, layer) {
    const spec = LAYER_ALTERNATIVES[layer]; const sources = state.data.sources.filter((s) => s.layer === layer);
    const head = el("div", "intel-detail-head"); head.append(badge("UNAVAILABLE", "unavailable"), badge("NO POSITIONS SHOWN OR SIMULATED", "historical"));
    box.append(head, el("h3", null, spec.title));
    sources.forEach((s) => { const dl = el("dl", "intel-dl"); row(dl, "Candidate source", `${s.provider} — ${s.dataset}`); row(dl, "Status", s.status); row(dl, "Blocker", s.blocker); box.append(dl); });
    box.append(el("h4", null, "Available instead"));
    const list = el("ul", "intel-alt"); spec.instead.forEach((text) => list.append(el("li", null, text))); box.append(list);
    const related = state.data.events.filter(spec.related);
    box.append(el("h4", null, `Related events in this window (${related.length})`));
    if (!related.length) box.append(el("p", "intel-muted", "None in the current window."));
    related.slice(0, 8).forEach((e) => { const b = el("button", "intel-record"); b.type = "button"; b.append(el("span", "intel-feed-title", e.title), el("span", "intel-feed-meta", `${e.provider.replaceAll("_", " ")} · ${iso(e.observed_at)}`)); b.addEventListener("click", () => select(e)); box.append(b); });
  }
  function renderHealth() {
    const body = $("intel-health"); body.replaceChildren();
    state.data.source_health.forEach((h) => {
      const tr = el("tr"); const source = state.data.sources.find((s) => s.source_id === h.source_id) || {};
      const cell = el("td"); cell.append(badge(h.semantics, SEMANTIC_KIND[h.semantics]));
      tr.append(el("th", null, source.provider || h.source_id), cell, el("td", null, iso(h.last_success_at)), el("td", null, h.last_error || "—"), el("td", null, String(h.consecutive_failures)), el("td", null, h.next_allowed_at ? iso(h.next_allowed_at) : "Now"));
      tr.firstChild.scope = "row"; body.append(tr);
    });
  }
  function renderRisk() {
    const list = $("intel-risk"); list.replaceChildren(); const rows = state.data.risk_monitor.themes;
    if (!rows.length) { list.append(el("p", "intel-muted", "No linked events in the window.")); return; }
    const max = Math.max(...rows.map((r) => r.events));
    rows.slice(0, 10).forEach((r) => {
      const item = el("div", "intel-risk-row"); const bar = el("span", "intel-risk-bar"); const fill = el("span", "intel-risk-fill"); fill.style.width = `${(100 * r.events) / max}%`; bar.append(fill);
      item.append(el("span", "intel-risk-label", r.label), bar, el("span", "intel-risk-count", `${r.events} (${r.confirmed} confirmed, ${r.unverified} unverified${r.other ? `, ${r.other} demonstration, scenario or historical` : ""})`), el("span", "intel-risk-markets", r.markets.join(" · ")));
      list.append(item);
    });
  }
  function renderDefense() {
    const tabs = $("intel-defense-tabs");
    if (!tabs.children.length) BRANCHES.forEach(([id, label]) => { const tab = el("button", "lab-tab", label); tab.type = "button"; tab.setAttribute("role", "tab"); tab.dataset.branch = id; tab.addEventListener("click", () => { state.branch = id; renderDefense(); }); tabs.append(tab); });
    [...tabs.children].forEach((tab) => { tab.setAttribute("aria-selected", String(tab.dataset.branch === state.branch)); tab.tabIndex = tab.dataset.branch === state.branch ? 0 : -1; });
    const agg = state.data.defense_aggregate; const body = $("intel-defense-table"); body.replaceChildren();
    const weekly = agg.weekly.filter((w) => w.branch === state.branch);
    if (!weekly.length) { const tr = el("tr"); const td = el("td", "intel-muted", "No admitted records for this branch in the window."); td.colSpan = 3; tr.append(td); body.append(tr); }
    weekly.forEach((w) => { const tr = el("tr"); tr.append(el("td", null, w.iso_week), el("td", null, w.region), el("td", null, String(w.count))); body.append(tr); });
    const records = $("intel-defense-records"); records.replaceChildren();
    state.data.events.filter((e) => e.provider === "ADMITTED_REPORTS" && e.category === state.branch).forEach((e) => {
      const card = el("button", "intel-record"); card.type = "button";
      const head = el("span", "intel-record-head"); head.append(badge(e.kind.replaceAll("_", " "), "historical"), badge(VERIFICATION_LABEL[e.verification] || e.verification, /CONFIRMED|MULTI/.test(e.verification) ? "confirmed" : /SCENARIO/.test(e.verification) ? "demo" : "unverified"));
      if (e.data_classification === "SYNTHETIC") head.append(badge("AUTHORED SYNTHETIC", "demo"));
      card.append(head, el("span", "intel-feed-title", e.title), el("span", "intel-feed-meta", `${e.region} · published ${iso(e.published_at)} · delay ${duration(e.reporting_delay_seconds)} · ±111 km grid`));
      card.addEventListener("click", () => select(e)); records.append(card);
    });
    $("intel-defense-policy").textContent = agg.policy;
  }
  function renderCoverage() {
    const box = $("intel-coverage"); box.replaceChildren();
    state.data.sources.forEach((s) => {
      const card = el("div", `intel-source ${s.status.toLowerCase()}`); const head = el("div", "intel-source-head");
      head.append(el("strong", null, s.provider), badge(s.status, s.status === "INTEGRATED" ? "delayed" : s.status === "BLOCKED" ? "unavailable" : "historical"));
      card.append(head, el("span", "intel-muted", `${LAYER_LABELS[s.layer] || s.layer} · ${s.dataset}`));
      if (s.license) card.append(el("span", "intel-source-line", `License: ${s.license}`));
      if (s.coverage) card.append(el("span", "intel-source-line", s.coverage));
      if (s.blocker) card.append(el("span", "intel-source-line blocker", s.blocker));
      box.append(card);
    });
  }
  function renderInvestigations() {
    const list = $("intel-inv-list"); list.replaceChildren();
    if (!state.data.investigations.length) { list.append(el("li", "intel-muted", "No saved investigations.")); return; }
    state.data.investigations.forEach((inv) => {
      const item = el("li"); const button = el("button", "text-button", `${inv.name} · ${iso(inv.created_at)}`); button.type = "button";
      button.addEventListener("click", () => {
        const b = inv.body; if (b.layers) state.layers = new Set(b.layers); if (b.as_of) setAsOf(b.as_of);
        if (b.event_ids && b.event_ids[0]) state.pendingSelect = b.event_ids[0];
        load(); say(`Opened investigation ${inv.name}`);
      });
      item.append(button); list.append(item);
    });
  }
  function select(item) {
    state.selected = item; renderDetails(); renderFeed();
    const lat = item ? item.latitude ?? item.lat : null, lon = item ? item.longitude ?? item.lon : null;
    if (Number.isFinite(lat) && Number.isFinite(lon)) focusOn(lat, lon); else draw();
    if (item && item.title) say(`Selected ${item.title}`);
  }
  // Bring a selected location into view: the globe turns to centre it (one
  // short ease, or an immediate jump under reduced motion); the flat map pans
  // only when the point lies outside the visible area.
  function focusOn(lat, lon) {
    if (state.view !== "globe") {
      const { w, h } = geometry(); const p = project(lat, lon); const margin = 40;
      if (p.x < margin || p.x > w - margin || p.y < margin || p.y > h - margin) { state.flatLon = lon; state.panY += h / 2 - p.y; state.flatAuto = false; clampPan(); }
      draw(); return;
    }
    if (state.rotate) setRotate(false);
    const targetLat = Math.max(-80, Math.min(80, lat)); const startLat = state.lat0;
    const startLon = state.lon0; const deltaLon = ((lon - startLon + 540) % 360) - 180;
    if (state.focusFrame) cancelAnimationFrame(state.focusFrame);
    if (reduced || typeof requestAnimationFrame !== "function") { state.lat0 = targetLat; state.lon0 = startLon + deltaLon; draw(); return; }
    const started = performance.now(); const duration = 360;
    const step = (now) => {
      const t = Math.min(1, (now - started) / duration); const e = 1 - (1 - t) ** 3;
      state.lat0 = startLat + (targetLat - startLat) * e; state.lon0 = startLon + deltaLon * e; draw();
      state.focusFrame = t < 1 ? requestAnimationFrame(step) : null;
    };
    state.focusFrame = requestAnimationFrame(step);
  }

  /* ---------- data ---------- */
  async function load() {
    const ticket = ++state.ticket; const params = new URLSearchParams({ days: String(state.days) });
    if (state.asOf) params.set("as_of", state.asOf); if (state.query) params.set("q", state.query);
    $("intel-status").textContent = "Loading…";
    try {
      const payload = await call(`/api/v1/intel/overview?${params}`);
      if (ticket !== state.ticket) return;
      state.data = payload.data; $("intel-status").textContent = "";
      const pending = state.pendingSelect || (state.selected && state.selected.event_id); state.pendingSelect = null;
      const reference = state.selected && (state.selected.layer_ref || state.selected.kind_ref) ? state.selected : null;
      state.selected = pending ? state.data.events.find((e) => e.event_id === pending) || null : reference;
      renderAll();
    } catch (error) {
      if (ticket !== state.ticket) return;
      $("intel-status").textContent = `ERROR · ${error.message}`;
      if (!$("intel-panel").classList.contains("hidden")) { $("state-badge").textContent = "ERROR"; $("state-badge").dataset.state = "ERROR"; $("source-label").textContent = "global-intelligence · overview request failed"; }
    }
  }
  async function loadWorld() {
    if (state.world) return;
    try { const response = await fetch("/static/world-110m.json", { cache: "force-cache" }); state.world = await response.json(); $("intel-basemap").textContent = state.world.source; draw(); }
    catch { $("intel-basemap").textContent = "Basemap unavailable; events are still plotted on the graticule."; }
  }
  function setAsOf(value) {
    state.asOf = value; const slider = $("intel-timeline");
    if (!value) { slider.value = "100"; $("intel-timeline-label").textContent = "Now (latest acquired)"; return; }
    const span = state.days * DAY; const pct = Math.round(100 - ((Date.now() - Date.parse(value)) / span) * 100);
    slider.value = String(Math.max(0, Math.min(100, pct))); $("intel-timeline-label").textContent = `Replay: ${iso(value)}`;
  }
  function timelineChanged() {
    const pct = Number($("intel-timeline").value);
    if (pct >= 100) { setAsOf(null); load(); return; }
    const stamp = new Date(Date.now() - (1 - pct / 100) * state.days * DAY); stamp.setUTCSeconds(0, 0);
    state.asOf = stamp.toISOString(); $("intel-timeline-label").textContent = `Replay: ${iso(state.asOf)} (only records acquired by then)`; load();
  }
  function togglePlay() {
    const button = $("intel-play");
    if (state.playing) { clearInterval(state.playing); state.playing = null; button.textContent = "Play"; button.setAttribute("aria-pressed", "false"); return; }
    if (Number($("intel-timeline").value) >= 100) $("intel-timeline").value = "0";
    button.textContent = "Pause"; button.setAttribute("aria-pressed", "true");
    const step = () => { const slider = $("intel-timeline"); const next = Math.min(100, Number(slider.value) + 5); slider.value = String(next); timelineChanged(); if (next >= 100) togglePlay(); };
    step(); state.playing = setInterval(step, reduced ? 2500 : 1200);
  }

  /* ---------- research forms ---------- */
  function syntheticSeries(n, seed, drift, start) {
    const rows = []; let price = 100;
    for (let i = 0; i < n; i += 1) { const shock = Math.sin(i * 1.7 + seed) * 0.009 + Math.cos(i * 0.61 + seed * 2) * 0.006; price *= Math.exp(drift + shock); rows.push({ date: new Date(start + i * DAY).toISOString().replace(".000Z", "Z"), close: Number(price.toFixed(4)) }); }
    return rows;
  }
  function examples() {
    const start = Date.UTC(2026, 6, 1);
    const prices = syntheticSeries(90, 1, 0.0002, start); const market = syntheticSeries(90, 4, 0.0001, start);
    for (let i = 70; i < 75; i += 1) prices[i].close = Number((prices[i].close * (1 + 0.01 * (i - 69))).toFixed(4));
    $("intel-study-input").value = JSON.stringify({ instrument: "SYN-ENERGY", classification: "SYNTHETIC", source: "GIG authored synthetic price fixture", event_at: prices[70].date, estimation_window: 40, pre: 5, post: 10, prices, market_prices: market }, null, 2);
    $("intel-exposure-input").value = JSON.stringify({ nav: 1000000, holdings: [
      { instrument: "SYN-INTEGRATED-OIL", market_value: 180000, themes: ["crude_oil"] }, { instrument: "SYN-TANKERS", market_value: 60000, themes: ["tanker_shipping", "crude_oil"] },
      { instrument: "SYN-CONTAINER-LINE", market_value: 90000, themes: ["container_shipping"] }, { instrument: "SYN-AIRLINE", market_value: 70000, themes: ["aviation"] },
      { instrument: "SYN-DEFENSE-PRIME", market_value: 120000, themes: ["defense"] }, { instrument: "SYN-BROAD-INDEX", market_value: 400000, themes: [] }],
      shocks: { crude_oil: 0.06, tanker_shipping: -0.1, container_shipping: -0.08, aviation: -0.07 } }, null, 2);
    const now = Date.now(); const at = (days) => new Date(now - days * DAY).toISOString().replace(/\.\d{3}Z$/, "Z");
    const cite = [{ publisher: "GIG authored fixture", publisher_type: "OTHER", url: "https://example.org/gig-authored-fixture" }];
    const record = (id, branch, kind, title, region, lat, lon) => ({ record_id: id, branch, kind, title, summary: "Fictional record authored to demonstrate the research schema. It describes no real announcement.", region, country_codes: [],
      latitude: lat, longitude: lon, published_at: at(4), observed_start: at(6), observed_end: at(5), acquired_at: at(1), sources: cite, verification: "ANALYTICAL_SCENARIO", classification: "SYNTHETIC" });
    $("intel-admit-input").value = JSON.stringify({ records: [
      record("fixture-navy-1", "NAVY", "EXERCISE_ANNOUNCEMENT", "Authored fixture: announced multinational naval exercise (fictional)", "Arabian Sea", 20, 62),
      record("fixture-air-1", "AIR_FORCE", "AIRSPACE_RESTRICTION_NOTICE", "Authored fixture: temporary airspace restriction notice (fictional)", "Eastern Mediterranean", 34, 33),
      record("fixture-army-1", "ARMY", "EXERCISE_ANNOUNCEMENT", "Authored fixture: scheduled land exercise announcement (fictional)", "Baltic region", 56, 24),
      record("fixture-geo-1", "GEOPOLITICAL", "SANCTIONS_ANNOUNCEMENT", "Authored fixture: sanctions package announcement (fictional)", "Global", null, null)] }, null, 2);
  }
  function output(id, nodes) { const box = $(id); box.replaceChildren(...nodes); }
  async function runStudy() {
    let body; try { body = JSON.parse($("intel-study-input").value); } catch { output("intel-study-output", [el("p", "intel-error", "Inputs are not valid JSON.")]); return; }
    try {
      const result = (await call("/api/v1/intel/event-study", body)).data;
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg"); svg.setAttribute("viewBox", "0 0 400 140"); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "Cumulative abnormal return across the event window; exact values in the table.");
      const cars = result.window.map((r) => r.cumulative_abnormal_return); const lo = Math.min(0, ...cars), hi = Math.max(0, ...cars); const span = hi - lo || 1;
      const x = (i) => 20 + (360 * i) / Math.max(1, cars.length - 1); const y = (v) => 120 - (100 * (v - lo)) / span;
      const axis = document.createElementNS(svg.namespaceURI, "line"); axis.setAttribute("x1", "20"); axis.setAttribute("x2", "380"); axis.setAttribute("y1", String(y(0))); axis.setAttribute("y2", String(y(0))); axis.setAttribute("class", "intel-axis"); svg.append(axis);
      const zero = result.window.findIndex((r) => r.offset === 0); const mark = document.createElementNS(svg.namespaceURI, "line"); mark.setAttribute("x1", String(x(zero))); mark.setAttribute("x2", String(x(zero))); mark.setAttribute("y1", "15"); mark.setAttribute("y2", "125"); mark.setAttribute("class", "intel-event-mark"); svg.append(mark);
      const path = document.createElementNS(svg.namespaceURI, "polyline"); path.setAttribute("points", cars.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ")); path.setAttribute("class", "intel-car"); svg.append(path);
      const dl = el("dl", "intel-dl"); row(dl, "Model", result.model); row(dl, "Estimation window", `${iso(result.estimation.start)} → ${iso(result.estimation.end)} (${result.estimation.observations} obs, before the event window)`);
      row(dl, "CAR", `${(result.car * 100).toFixed(2)}%`); row(dl, "t-statistic", result.car_t_stat === null ? "Unknown" : result.car_t_stat.toFixed(2)); row(dl, "Data", `${result.data_classification} · ${result.source}`); row(dl, "Interpretation", "Descriptive single-event study; not a causal effect.");
      const table = el("table", "intel-table"); const head = el("tr"); ["Offset", "Date", "Return", "Abnormal", "CAR"].forEach((h) => { const th = el("th", null, h); th.scope = "col"; head.append(th); }); const thead = el("thead"); thead.append(head); const tbody = el("tbody");
      result.window.forEach((r) => { const tr = el("tr"); tr.append(el("td", null, r.offset), el("td", null, iso(r.date).slice(0, 10)), el("td", null, (r.return * 100).toFixed(2) + "%"), el("td", null, (r.abnormal_return * 100).toFixed(2) + "%"), el("td", null, (r.cumulative_abnormal_return * 100).toFixed(2) + "%")); tbody.append(tr); });
      table.append(thead, tbody); const wrap = el("div", "intel-table-wrap"); wrap.tabIndex = 0; wrap.setAttribute("role", "region"); wrap.setAttribute("aria-label", "Event window returns"); wrap.append(table);
      output("intel-study-output", [svg, dl, el("p", "intel-muted", result.limitations.join(" ")), wrap]); say(`Event study complete. CAR ${(result.car * 100).toFixed(2)} percent.`);
    } catch (error) { output("intel-study-output", [el("p", "intel-error", error.message)]); }
  }
  async function runExposure() {
    let body; try { body = JSON.parse($("intel-exposure-input").value); } catch { output("intel-exposure-output", [el("p", "intel-error", "Inputs are not valid JSON.")]); return; }
    try {
      const result = (await call("/api/v1/intel/exposure", body)).data;
      const table = el("table", "intel-table"); const thead = el("thead"); const head = el("tr"); ["Theme", "Market value", "Share of NAV"].forEach((h) => { const th = el("th", null, h); th.scope = "col"; head.append(th); }); thead.append(head);
      const tbody = el("tbody"); result.themes.forEach((t) => { const tr = el("tr"); tr.append(el("td", null, t.label), el("td", null, t.market_value.toLocaleString("en-US")), el("td", null, t.share_of_nav === null ? "Unknown" : `${(t.share_of_nav * 100).toFixed(1)}%`)); tbody.append(tr); });
      table.append(thead, tbody);
      const nodes = [table];
      if (result.scenario) { const dl = el("dl", "intel-dl"); row(dl, "Scenario P&L", `${result.scenario.pnl.toLocaleString("en-US", { maximumFractionDigits: 0 })} (${(result.scenario.pnl_share_of_nav * 100).toFixed(2)}% of NAV)`); row(dl, "Aggregation", "Most adverse shock per holding; overlapping tags are not compounded"); row(dl, "Classification", "Analytical scenario — analyst-supplied shocks, not a forecast"); nodes.push(dl); }
      nodes.push(el("p", "intel-muted", `${result.interpretation} Untagged value: ${result.untagged_market_value.toLocaleString("en-US")}. Basis: ${result.nav_basis.replaceAll("_", " ").toLowerCase()}.`));
      output("intel-exposure-output", nodes); say("Exposure calculated.");
    } catch (error) { output("intel-exposure-output", [el("p", "intel-error", error.message)]); }
  }
  async function admit() {
    let body; try { body = JSON.parse($("intel-admit-input").value); } catch { $("intel-admit-output").textContent = "Inputs are not valid JSON."; return; }
    try { const result = (await call("/api/v1/intel/admit", body)).data; $("intel-admit-output").textContent = `Admitted ${result.admitted} records (${result.new_revisions} new revisions).`; await load(); }
    catch (error) { $("intel-admit-output").textContent = `Rejected · ${error.message}`; }
  }
  async function refreshSource(source) {
    const status = $("intel-refresh-status"); status.textContent = `Requesting ${source}…`;
    try { const result = (await call("/api/v1/intel/refresh", { source_id: source })).data; status.textContent = `${source}: ${result.state}${result.received !== undefined ? ` · ${result.received} records, ${result.new_revisions} new revisions` : ""}${result.error_code ? ` · ${result.error_code}` : ""}${result.next_allowed_at ? ` · next allowed ${iso(result.next_allowed_at)}` : ""}`; await load(); }
    catch (error) { status.textContent = `${source}: ${error.message}`; }
  }
  async function demonstration() {
    try { const result = (await call("/api/v1/intel/demonstration", {})).data; $("intel-refresh-status").textContent = `Loaded ${result.admitted} authored demonstration events (labeled DEMONSTRATION).`; await load(); }
    catch (error) { $("intel-refresh-status").textContent = error.message; }
  }
  async function saveInvestigation() {
    const name = $("intel-inv-name").value.trim(); if (!name) { $("intel-inv-status").textContent = "Name the investigation first."; return; }
    const body = { layers: [...state.layers], notes: $("intel-inv-notes").value.slice(0, 2000) };
    if (state.asOf) body.as_of = state.asOf; if (state.selected && state.selected.event_id) { body.event_ids = [state.selected.event_id]; body.themes = (state.selected.research_links?.themes || []).map((t) => t.theme); }
    try { const result = (await call("/api/v1/intel/investigation", { name, body })).data; $("intel-inv-status").textContent = `Saved ${result.name} (${result.investigation_id.slice(0, 12)}…).`; await load(); }
    catch (error) { $("intel-inv-status").textContent = error.message; }
  }

  /* ---------- interaction ---------- */
  function hit(x, y) { let best = null, dist = Infinity; for (const p of state.projected) { const d = Math.hypot(p.x - x, p.y - y); if (d < p.r + 3 && d < dist) { best = p; dist = d; } } return best; }
  function point(event) { const r = $("intel-canvas").getBoundingClientRect(); return [event.clientX - r.left, event.clientY - r.top]; }
  function asSelection(target) { return target.kind === "event" ? target.item : { ...target.item, kind_ref: target.kind }; }
  function setRotate(on) {
    const button = $("intel-rotate"); if (state.rotate) { cancelAnimationFrame(state.rotate); state.rotate = null; }
    button.setAttribute("aria-pressed", String(on)); button.textContent = on ? "Pause rotation" : "Rotate globe";
    if (!on) return; let last = null;
    const frame = (t) => { if (last !== null) state.lon0 = ((state.lon0 + (t - last) * 0.006 + 540) % 360) - 180; last = t; draw(); state.rotate = requestAnimationFrame(frame); };
    state.rotate = requestAnimationFrame(frame);
  }
  function bind() {
    const canvas = $("intel-canvas");
    canvas.addEventListener("pointerdown", (event) => { if (event.button !== 0) return; setRotate(false); state.drag = { x: event.clientX, y: event.clientY, moved: false }; canvas.setPointerCapture(event.pointerId); });
    canvas.addEventListener("pointermove", (event) => {
      if (state.drag) {
        const dx = event.clientX - state.drag.x, dy = event.clientY - state.drag.y; if (Math.abs(dx) + Math.abs(dy) > 2) state.drag.moved = true;
        state.drag.x = event.clientX; state.drag.y = event.clientY;
        if (state.view === "globe") { state.lon0 -= dx * 0.35 / state.zoom; state.lat0 = Math.max(-80, Math.min(80, state.lat0 + dy * 0.35 / state.zoom)); } else { const { w, h } = geometry(); state.flatLon -= dx / flatScale(w, h); state.panY += dy; state.flatAuto = false; clampPan(); }
        draw(); return;
      }
      const [x, y] = point(event); const target = hit(x, y); const tip = $("intel-tooltip");
      canvas.style.cursor = target ? "pointer" : "grab";
      if (!target) { tip.classList.add("hidden"); return; }
      const item = target.item; tip.textContent = target.kind === "event" ? `${item.title}\n${item.category.replaceAll("_", " ")} · ${iso(item.observed_at)}\n${item.provider.replaceAll("_", " ")} · ${VERIFICATION_LABEL[item.verification] || item.verification}` : `${item.name}\n${target.kind === "chokepoint" ? "Chokepoint (approximate)" : "Port (approximate)"}`;
      tip.style.left = `${Math.min(x + 14, canvas.clientWidth - 260)}px`; tip.style.top = `${Math.max(4, y + 10)}px`; tip.classList.remove("hidden");
    });
    canvas.addEventListener("pointerup", (event) => { const drag = state.drag; state.drag = null; if (drag && !drag.moved) { const [x, y] = point(event); const target = hit(x, y); select(target ? asSelection(target) : null); } });
    canvas.addEventListener("pointercancel", () => { state.drag = null; });
    canvas.addEventListener("pointerleave", () => $("intel-tooltip").classList.add("hidden"));
    canvas.addEventListener("wheel", (event) => { event.preventDefault(); zoomBy(event.deltaY > 0 ? 0.9 : 1.1); draw(); }, { passive: false });
    canvas.addEventListener("keydown", (event) => {
      const step = 8 / state.zoom; const ev = events().filter((e) => e.latitude !== null); let handled = true;
      if (event.key === "ArrowLeft") { if (state.view === "globe") state.lon0 -= step; else { state.flatLon -= step; state.flatAuto = false; } }
      else if (event.key === "ArrowRight") { if (state.view === "globe") state.lon0 += step; else { state.flatLon += step; state.flatAuto = false; } }
      else if (event.key === "ArrowUp") { if (state.view === "globe") state.lat0 = Math.min(80, state.lat0 + step); else state.panY += 30; }
      else if (event.key === "ArrowDown") { if (state.view === "globe") state.lat0 = Math.max(-80, state.lat0 - step); else state.panY -= 30; }
      else if (event.key === "+" || event.key === "=") zoomBy(1.15);
      else if (event.key === "-") zoomBy(1 / 1.15);
      else if (event.key === "0") { state.zoom = 1; if (state.view === "globe") { state.lon0 = 40; state.lat0 = 18; } else resetFlat(); }
      else if ((event.key === "n" || event.key === "p") && ev.length) {
        const index = ev.findIndex((e) => state.selected && e.event_id === state.selected.event_id);
        const next = ev[(index + (event.key === "n" ? 1 : ev.length - 1) + (index < 0 && event.key === "p" ? 1 : 0)) % ev.length];
        if (state.view === "globe") { state.lon0 = next.longitude; state.lat0 = Math.max(-60, Math.min(60, next.latitude)); }
        select(next);
      } else if (event.key === "Escape") select(null);
      else handled = false;
      if (handled) { event.preventDefault(); setRotate(false); if (state.view !== "globe") clampPan(); draw(); }
    });
    $("intel-view-globe").addEventListener("click", () => setView("globe"));
    $("intel-view-flat").addEventListener("click", () => setView("flat"));
    $("intel-reset").addEventListener("click", () => { state.zoom = 1; state.lon0 = 40; state.lat0 = 18; resetFlat(); draw(); });
    $("intel-rotate").addEventListener("click", () => setRotate(!state.rotate));
    let searchTimer = null;
    $("intel-search").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { state.query = $("intel-search").value.trim().slice(0, 100); load(); }, 250); });
    $("intel-days").addEventListener("change", () => { state.days = Number($("intel-days").value); setAsOf(null); load(); });
    $("intel-timeline").addEventListener("change", timelineChanged);
    $("intel-play").addEventListener("click", togglePlay);
    $("intel-now").addEventListener("click", () => { if (state.playing) togglePlay(); setAsOf(null); load(); });
    $("intel-refresh-usgs").addEventListener("click", () => refreshSource("USGS_EARTHQUAKES"));
    $("intel-refresh-eonet").addEventListener("click", () => refreshSource("NASA_EONET"));
    $("intel-demo").addEventListener("click", demonstration);
    $("intel-admit-run").addEventListener("click", admit);
    $("intel-study-run").addEventListener("click", runStudy);
    $("intel-exposure-run").addEventListener("click", runExposure);
    $("intel-inv-save").addEventListener("click", saveInvestigation);
    if (typeof ResizeObserver !== "undefined") new ResizeObserver(() => draw()).observe($("intel-map"));
    examples();
  }
  // The flat map never zooms out past the fill scale, which would reopen the
  // empty bands; the globe may shrink slightly.
  function zoomBy(factor) {
    state.zoom = Math.max(state.view === "globe" ? 0.7 : 1, Math.min(8, state.zoom * factor));
    if (state.view !== "globe") clampPan();
  }
  function setView(view) {
    state.view = view; state.zoom = 1; resetFlat();
    $("intel-view-globe").setAttribute("aria-pressed", String(view === "globe")); $("intel-view-flat").setAttribute("aria-pressed", String(view === "flat"));
    $("intel-rotate").disabled = view !== "globe"; if (view !== "globe") setRotate(false); draw();
  }
  function show(visible) {
    const panel = $("intel-panel"); if (!panel) return;
    panel.classList.toggle("hidden", !visible);
    if (!visible) { setRotate(false); if (state.playing) togglePlay(); return; }
    if (!state.started) { state.started = true; bind(); loadWorld(); }
    if (state.data) renderTruth(state.data); else truthLoading();
    load();
  }
  function frame() {
    const { w, h } = geometry();
    const centre = state.view === "globe" ? project(state.lat0, state.lon0) : null;
    return { width: w, height: h, mapTop: state.view === "globe" ? null : project(FLAT_LAT, 0).y, mapBottom: state.view === "globe" ? null : project(-FLAT_LAT, 0).y, lonSpan: state.view === "globe" ? null : w / flatScale(w, h), centre };
  }
  window.GIGIntel = { show, state: () => ({ view: state.view, asOf: state.asOf, events: state.data ? state.data.event_count : null, selected: state.selected && (state.selected.event_id || state.selected.name), lon0: state.lon0, lat0: state.lat0, flatLon: state.flatLon, zoom: state.zoom, focusing: Boolean(state.focusFrame) }), frame, project: (lat, lon) => project(lat, lon), events: () => events().map(({ event_id, title, latitude, longitude }) => ({ event_id, title, latitude, longitude })), setView, select: (id) => select(state.data.events.find((e) => e.event_id === id) || null) };
})();
