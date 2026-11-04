/* GIG shared 3D engine: projected canvas rendering with real camera, picking and cleanup.
 * Scenes are plain data: primitives in normalized [-1, 1] space plus axis metadata.
 * No external libraries, no network access, no HTML sinks. */
"use strict";
(() => {
  const TAU = Math.PI * 2;
  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
  const reducedMotion = () => window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const SCALES = {
    // Perceptually ordered sequential ramp (dark blue to pale yellow), legible on black.
    sequential: ["#1d2f6f", "#2a4f9f", "#2f7fbf", "#33a3b8", "#5fc19b", "#a8d672", "#f2e45c"],
    // Gain/loss diverging ramp: red losses, neutral grey, green gains.
    pnl: ["#b8323a", "#f05259", "#8a6a6e", "#5b6573", "#5e8f7c", "#16c784", "#0f9a66"],
    // Correlation: yellow negative, neutral grey, blue positive (no good/bad implication).
    correlation: ["#f2be4c", "#b39552", "#5b6573", "#5577b5", "#408bff"],
  };
  function hexToRgb(hex) { const n = parseInt(hex.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  function ramp(name, t) {
    const stops = SCALES[name] || SCALES.sequential;
    const x = clamp(Number.isFinite(t) ? t : 0, 0, 1) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(x));
    const a = hexToRgb(stops[i]); const b = hexToRgb(stops[i + 1]); const f = x - i;
    return `rgb(${Math.round(a[0] + (b[0] - a[0]) * f)},${Math.round(a[1] + (b[1] - a[1]) * f)},${Math.round(a[2] + (b[2] - a[2]) * f)})`;
  }
  function scaler(min, max) {
    const span = max - min || 1;
    return { to: (v) => ((v - min) / span) * 2 - 1, from: (n) => min + ((n + 1) / 2) * span, min, max };
  }
  function format(value, digits = 3) {
    if (value === null || value === undefined) return "UNKNOWN";
    if (typeof value !== "number") return String(value);
    if (!Number.isFinite(value)) return "UNKNOWN";
    const abs = Math.abs(value);
    if (abs !== 0 && (abs >= 1e6 || abs < 1e-3)) return value.toExponential(2);
    return value.toLocaleString("en-US", { maximumFractionDigits: digits });
  }

  function create(canvas, options = {}) {
    const ctx = canvas.getContext("2d");
    const camera = { yaw: -0.75, pitch: 0.55, zoom: 1, panX: 0, panY: 0, ortho2d: false };
    const initial = { ...camera };
    const view = { width: 0, height: 0, dpr: 1, scale: 1, offsetX: 0, offsetY: 0 };
    let placed = [];
    let pointLabels = [];
    let scene = { primitives: [], axes: null };
    let projected = [];
    let hovered = null; let selected = null; let frame = 0; let orbiting = false; let last = 0; let destroyed = false;
    let drag = null;
    const listeners = [];
    const on = (target, type, fn, opts) => { target.addEventListener(type, fn, opts); listeners.push([target, type, fn, opts]); };

    function resize() {
      const rect = canvas.getBoundingClientRect();
      view.dpr = Math.min(2, window.devicePixelRatio || 1);
      view.width = Math.max(1, rect.width); view.height = Math.max(1, rect.height);
      canvas.width = Math.round(view.width * view.dpr); canvas.height = Math.round(view.height * view.dpr);
      fit(); draw();
    }
    // Fit the cube and its label anchors to the canvas for the current camera
    // orientation. Done on resize, scene change, reset and 2D toggle only, so the
    // scene does not breathe while the user orbits.
    const ANCHORS = [];
    for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) ANCHORS.push([x, y, z]);
    for (const s of [-1, 1]) ANCHORS.push([0, 1.32 * s, -1], [1.36 * s, 0, -1], [1, 1.14 * s, -1], [1.16 * s, 1, -1]);
    ANCHORS.push([-1.2, -1.2, 1.18]);
    // Floor axes sit on whichever parallel floor edge is nearer the viewer, so
    // their titles and ticks fall outside the data instead of behind it.
    function nearSides() {
      const nearer = (a, b) => (rotate(a)[2] < rotate(b)[2] - 1e-9 ? 1 : -1);
      return { sx: nearer([1, 0, -1], [-1, 0, -1]), sy: nearer([0, 1, -1], [0, -1, -1]) };
    }
    function fit() {
      if (!view.width) return;
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const point of ANCHORS) {
        if (camera.ortho2d && point[2] > 1) continue;
        const [x, y, depth] = rotate(point); const perspective = camera.ortho2d ? 1 : 3.4 / (3.4 + depth);
        minX = Math.min(minX, x * perspective); maxX = Math.max(maxX, x * perspective);
        minY = Math.min(minY, -y * perspective); maxY = Math.max(maxY, -y * perspective);
      }
      const marginX = Math.min(48, view.width * 0.08); const marginY = Math.min(28, view.height * 0.06);
      view.scale = Math.max(1, Math.min((view.width - 2 * marginX) / (maxX - minX), (view.height - 2 * marginY) / (maxY - minY),
        Math.min(view.width, view.height) * 0.42));
      view.offsetX = -((minX + maxX) / 2) * view.scale; view.offsetY = -((minY + maxY) / 2) * view.scale;
    }
    function rotate([x, y, z]) {
      const yaw = camera.ortho2d ? 0 : camera.yaw; const pitch = camera.ortho2d ? Math.PI / 2 : camera.pitch;
      const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      const rx = x * cy - y * sy; const ry = x * sy + y * cy;
      return [rx, ry * sp + z * cp, ry * cp - z * sp];
    }
    function project(point) {
      const [x, y, depth] = rotate(point);
      const perspective = camera.ortho2d ? 1 : 3.4 / (3.4 + depth);
      const s = view.scale * camera.zoom * perspective;
      return [view.width / 2 + view.offsetX + camera.panX + x * s, view.height / 2 + view.offsetY + camera.panY - y * s, depth];
    }
    // Labels stay inside the canvas, carry a dark halo so geometry behind them
    // cannot make them illegible, and are skipped when `avoid` is set and they
    // would collide with a label already placed this frame.
    function text(value, x, y, color = "#a6b2c3", align = "center", size = 11, avoid = false) {
      ctx.font = `${size}px Inter, "Segoe UI", system-ui, sans-serif`;
      const width = ctx.measureText(value).width; const pad = 4;
      let left = align === "center" ? x - width / 2 : align === "right" ? x - width : x;
      left = clamp(left, pad, Math.max(pad, view.width - width - pad));
      const top = clamp(y, size / 2 + pad, Math.max(size / 2 + pad, view.height - size / 2 - pad));
      const box = [left - 2, top - size / 2 - 1, left + width + 2, top + size / 2 + 1];
      if (avoid && placed.some((b) => box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1])) return false;
      placed.push(box);
      ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.lineJoin = "round";
      ctx.strokeStyle = "rgba(5,6,8,0.88)"; ctx.lineWidth = 3; ctx.strokeText(value, left, top);
      ctx.fillStyle = color; ctx.fillText(value, left, top);
      return true;
    }
    function drawAxisLines() {
      if (!scene.axes) return;
      ctx.lineWidth = 1;
      const edge = (a, b, color = "#26303d") => { const p = project(a), q = project(b); ctx.strokeStyle = color; ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke(); };
      for (let i = -1; i <= 1.001; i += 0.5) {
        edge([i, -1, -1], [i, 1, -1], "#1a222d"); edge([-1, i, -1], [1, i, -1], "#1a222d");
      }
      const { sx, sy } = nearSides();
      edge([-1, sy, -1], [1, sy, -1], "#3a4656"); edge([sx, -1, -1], [sx, 1, -1], "#3a4656"); edge([-1, -1, -1], [-1, -1, 1], "#3a4656");
    }
    function drawAxisLabels() {
      const axes = scene.axes; if (!axes) return;
      const keys = ["x", "y", "z"].filter((key) => axes[key] && !(key === "z" && camera.ortho2d));
      const { sx, sy } = nearSides();
      keys.forEach((key) => {
        const axis = axes[key];
        const at = key === "x" ? [0, 1.32 * sy, -1] : key === "y" ? [1.36 * sx, 0, -1] : [-1.2, -1.2, 1.18];
        const p = project(at); text(`${axis.label}${axis.unit ? ` (${axis.unit})` : ""}`, p[0], p[1], "#a6b2c3", "center", 11);
      });
      keys.forEach((key) => {
        const axis = axes[key];
        const ticks = axis.ticks || (axis.scale ? [-1, 0, 1].map((n) => [n, axis.scale.from(n)]) : []);
        ticks.forEach(([n, value]) => {
          const at = key === "x" ? [n, 1.14 * sy, -1] : key === "y" ? [1.16 * sx, n, -1] : [-1.12, -1.12, n];
          const p = project(at); text(format(value, 2), p[0], p[1], "#8292a5", "center", 10, true);
        });
      });
    }
    function drawPrimitive(item) {
      const pts = item.screen;
      const highlight = item === selected || item === hovered;
      ctx.globalAlpha = item.alpha ?? 1;
      if (item.type === "point") {
        const r = (item.radius || 3.5) * (highlight ? 1.6 : 1);
        ctx.fillStyle = item.color; ctx.beginPath(); ctx.arc(pts[0][0], pts[0][1], r, 0, TAU); ctx.fill();
        if (highlight) { ctx.strokeStyle = "#f2f5f9"; ctx.lineWidth = 1.5; ctx.stroke(); }
        if (item.label) pointLabels.push({ item, r, highlight });
      } else if (item.type === "line") {
        ctx.strokeStyle = item.color; ctx.lineWidth = (item.width || 1.2) * (highlight ? 2 : 1);
        if (item.dash) ctx.setLineDash(item.dash);
        ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.stroke(); ctx.setLineDash([]);
      } else if (item.type === "quad") {
        ctx.fillStyle = item.color; ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.closePath(); ctx.fill();
        if (item.stroke || highlight) { ctx.strokeStyle = highlight ? "#f2f5f9" : item.stroke; ctx.lineWidth = highlight ? 1.5 : 0.6; ctx.stroke(); }
      }
      ctx.globalAlpha = 1;
    }
    // Point labels try right, left, above and below their point; a label that
    // collides everywhere is omitted (hover, selection and the data table still
    // name it) unless its point is selected or hovered.
    function drawPointLabels() {
      pointLabels.sort((a, b) => b.highlight - a.highlight);
      for (const { item, r, highlight } of pointLabels) {
        const [x, y] = item.screen[0]; const gap = r + 4;
        const spots = [[x + gap, y, "left"], [x - gap, y, "right"], [x, y - gap - 6, "center"], [x, y + gap + 6, "center"]];
        const drawn = spots.some(([sx, sy, align]) => text(item.label, sx, sy, "#f2f5f9", align, 11, true));
        if (!drawn && highlight) text(item.label, spots[0][0], spots[0][1], "#f2f5f9", "left", 11);
      }
    }
    function expand(item) {
      if (item.type !== "bar") return [item];
      const [x, y, z0, z1] = item.box; const w = item.half || 0.04;
      const c = [[x - w, y - w], [x + w, y - w], [x + w, y + w], [x - w, y + w]];
      const faces = [[0, 1], [1, 2], [2, 3], [3, 0]].map(([a, b]) => ({ type: "quad", parent: item, pts: [[...c[a], z0], [...c[b], z0], [...c[b], z1], [...c[a], z1]], color: item.side || item.color, data: item.data, alpha: item.alpha }));
      faces.push({ type: "quad", parent: item, pts: c.map((p) => [...p, z1]), color: item.color, data: item.data, alpha: item.alpha, stroke: "#050608" });
      return faces;
    }
    function draw() {
      if (destroyed || !view.width) return;
      ctx.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
      ctx.clearRect(0, 0, view.width, view.height);
      placed = [];
      drawAxisLines();
      projected = [];
      for (const raw of scene.primitives) {
        for (const item of expand(raw)) {
          item.screen = item.pts.map(project);
          item.depth = item.screen.reduce((sum, p) => sum + p[2], 0) / item.screen.length;
          projected.push(item);
        }
      }
      projected.sort((a, b) => b.depth - a.depth || (a.type === "point") - (b.type === "point"));
      pointLabels = [];
      projected.forEach(drawPrimitive);
      drawPointLabels();
      drawAxisLabels();
      if (options.onDraw) options.onDraw(ctx, project, view);
    }
    function pick(x, y) {
      let best = null; let bestDistance = 14;
      for (let i = projected.length - 1; i >= 0; i -= 1) {
        const item = projected[i]; if (!item.data) continue;
        if (item.type === "quad" && inside(x, y, item.screen)) return item.parent || item;
        const pts = item.screen;
        for (const p of item.type === "line" ? [] : pts) {
          const d = Math.hypot(p[0] - x, p[1] - y);
          if (item.type === "point" && d < bestDistance) { best = item; bestDistance = d; }
        }
      }
      return best;
    }
    function inside(x, y, poly) {
      let hit = false;
      for (let i = 0, j = poly.length - 1; i < poly.length; j = i, i += 1) {
        const [xi, yi] = poly[i]; const [xj, yj] = poly[j];
        if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi || 1e-9) + xi) hit = !hit;
      }
      return hit;
    }
    function selectables() { return scene.primitives.filter((p) => p.data && p.selectable !== false); }
    function setSelected(item, announce = true) {
      selected = item; draw();
      if (options.onSelect && announce) options.onSelect(item ? item.data : null);
    }
    function tick(time) {
      frame = 0; if (!orbiting || destroyed) return;
      const dt = last ? Math.min(0.05, (time - last) / 1000) : 0; last = time;
      camera.yaw = (camera.yaw + dt * 0.12) % TAU; draw(); frame = requestAnimationFrame(tick);
    }
    function setOrbit(value) {
      orbiting = Boolean(value) && !reducedMotion(); last = 0;
      if (orbiting && !frame) frame = requestAnimationFrame(tick);
      if (!orbiting && frame) { cancelAnimationFrame(frame); frame = 0; }
      if (options.onOrbit) options.onOrbit(orbiting);
    }
    on(canvas, "pointerdown", (event) => { canvas.setPointerCapture(event.pointerId); drag = { x: event.clientX, y: event.clientY, pan: event.shiftKey || event.button === 2, moved: false }; });
    on(canvas, "pointermove", (event) => {
      const rect = canvas.getBoundingClientRect(); const x = event.clientX - rect.left; const y = event.clientY - rect.top;
      if (drag) {
        const dx = event.clientX - drag.x; const dy = event.clientY - drag.y; drag.x = event.clientX; drag.y = event.clientY;
        if (Math.abs(dx) + Math.abs(dy) > 1) drag.moved = true;
        if (drag.pan || camera.ortho2d) { camera.panX += dx; camera.panY += dy; } else { camera.yaw -= dx * 0.008; camera.pitch = clamp(camera.pitch + dy * 0.008, 0.05, 1.5); }
        draw(); return;
      }
      const item = pick(x, y);
      if (item !== hovered) { hovered = item; draw(); }
      if (options.onHover) options.onHover(item ? item.data : null, x, y);
    });
    on(canvas, "pointerup", (event) => {
      const rect = canvas.getBoundingClientRect();
      if (drag && !drag.moved) setSelected(pick(event.clientX - rect.left, event.clientY - rect.top));
      drag = null;
    });
    on(canvas, "pointerleave", () => { hovered = null; if (options.onHover) options.onHover(null); draw(); });
    on(canvas, "contextmenu", (event) => event.preventDefault());
    on(canvas, "wheel", (event) => { event.preventDefault(); camera.zoom = clamp(camera.zoom * Math.exp(-event.deltaY * 0.0012), 0.35, 6); draw(); }, { passive: false });
    on(canvas, "keydown", (event) => {
      const step = 0.08; let handled = true;
      if (event.key === "ArrowLeft") { if (event.shiftKey) camera.panX -= 20; else camera.yaw -= step; }
      else if (event.key === "ArrowRight") { if (event.shiftKey) camera.panX += 20; else camera.yaw += step; }
      else if (event.key === "ArrowUp") { if (event.shiftKey) camera.panY -= 20; else camera.pitch = clamp(camera.pitch + step, 0.05, 1.5); }
      else if (event.key === "ArrowDown") { if (event.shiftKey) camera.panY += 20; else camera.pitch = clamp(camera.pitch - step, 0.05, 1.5); }
      else if (event.key === "+" || event.key === "=") camera.zoom = clamp(camera.zoom * 1.15, 0.35, 6);
      else if (event.key === "-" || event.key === "_") camera.zoom = clamp(camera.zoom / 1.15, 0.35, 6);
      else if (event.key === "0") { Object.assign(camera, initial, { ortho2d: camera.ortho2d }); fit(); }
      else if (event.key === "n" || event.key === "p" || event.key === "]" || event.key === "[") {
        const items = selectables(); if (!items.length) return;
        const index = items.indexOf(selected); const delta = event.key === "n" || event.key === "]" ? 1 : -1;
        setSelected(items[(index + delta + items.length) % items.length]); return void event.preventDefault();
      } else if (event.key === "Escape") { setSelected(null); }
      else handled = false;
      if (handled) { event.preventDefault(); draw(); }
    });
    const observer = typeof ResizeObserver === "function" ? new ResizeObserver(() => resize()) : null;
    if (observer) observer.observe(canvas); else on(window, "resize", resize);
    const motionQuery = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;
    const motionChange = () => { if (reducedMotion()) setOrbit(false); };
    if (motionQuery && motionQuery.addEventListener) motionQuery.addEventListener("change", motionChange);
    resize();
    return {
      setScene(next) { scene = next; if (selected && !next.primitives.some((p) => p.data === selected.data)) selected = null; hovered = null; fit(); draw(); },
      setOrbit, isOrbiting: () => orbiting,
      set2D(value) { camera.ortho2d = Boolean(value); if (value) setOrbit(false); fit(); draw(); },
      reset() { Object.assign(camera, initial, { ortho2d: camera.ortho2d }); fit(); draw(); },
      select(data) { setSelected(scene.primitives.find((p) => p.data === data) || null, false); },
      camera: () => ({ ...camera }), redraw: draw, project,
      destroy() {
        destroyed = true; if (frame) cancelAnimationFrame(frame); frame = 0;
        listeners.forEach(([target, type, fn, opts]) => target.removeEventListener(type, fn, opts));
        if (observer) observer.disconnect();
        if (motionQuery && motionQuery.removeEventListener) motionQuery.removeEventListener("change", motionChange);
        scene = { primitives: [], axes: null }; projected = [];
      },
    };
  }

  function legend(container, { scale, min, max, label, unit }) {
    container.replaceChildren();
    const title = document.createElement("span"); title.className = "legend-title"; title.textContent = `${label}${unit ? ` · ${unit}` : ""}`;
    const bar = document.createElement("span"); bar.className = "legend-bar";
    bar.style.background = `linear-gradient(90deg, ${(SCALES[scale] || SCALES.sequential).join(",")})`;
    const low = document.createElement("span"); low.className = "legend-value"; low.textContent = format(min);
    const high = document.createElement("span"); high.className = "legend-value"; high.textContent = format(max);
    container.append(title, low, bar, high);
  }

  window.GIG3D = { create, ramp, scaler, format, legend, SCALES, reducedMotion };
})();
