/*
 * GIG research terminal.
 *
 * The 3D factor cloud is rendered with a hand-rolled perspective projection on
 * a 2D canvas: no WebGL library, no CDN, so the UI works offline. Points hold a
 * target position and a current position and are eased toward the target every
 * frame, which is what makes axis changes and live ticks animate instead of
 * snapping.
 */
(() => {
  "use strict";

  const SPAN = 95;        // world half-extent for a +/-3 z-score axis
  const FOCAL = 780;
  const EASE = 0.12;

  const state = {
    points: [],
    byId: new Map(),
    live: new Map(),       // symbol -> { last, prev, chg }
    yaw: 0.62,
    pitch: -0.32,
    dist: 340,
    spin: true,
    dragging: false,
    axisY: "ivol",
    mode: "alpha",         // alpha | risk | held
    sector: "",
    mouse: { x: -1e4, y: -1e4 },
    hover: null,
    width: 0,
    height: 0,
    lastRisk: null,
    lastPipeline: null,
  };

  const el = (id) => document.getElementById(id);
  const canvas = el("scene");
  const ctx = canvas.getContext("2d");
  const tip = el("tip");

  /* ── formatting ──────────────────────────────────────────────── */

  const isNum = (v) => typeof v === "number" && isFinite(v);
  const fmt = (v, dp = 2) => (isNum(v) ? v.toFixed(dp) : "—");
  const fmtPct = (v, dp = 2) => (isNum(v) ? (v * 100).toFixed(dp) + "%" : "—");
  const fmtSigned = (v, dp = 2) => (isNum(v) ? (v >= 0 ? "+" : "") + v.toFixed(dp) : "—");

  function fmtAdv(v) {
    if (!isNum(v)) return "—";
    if (v >= 1e9) return "$" + (v / 1e9).toFixed(1) + "B";
    if (v >= 1e6) return "$" + (v / 1e6).toFixed(0) + "M";
    if (v >= 1e3) return "$" + (v / 1e3).toFixed(0) + "K";
    return "$" + v.toFixed(0);
  }

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  async function getJSON(url) {
    const res = await fetch(url, { headers: { Accept: "application/json" } });
    if (!res.ok) throw new Error(url + " -> " + res.status);
    return res.json();
  }

  /* ── colour ──────────────────────────────────────────────────── */

  const SHORT = [255, 59, 92];
  const MID = [96, 108, 128];
  const LONG = [34, 227, 154];

  const mix = (a, b, u) => [
    Math.round(a[0] + (b[0] - a[0]) * u),
    Math.round(a[1] + (b[1] - a[1]) * u),
    Math.round(a[2] + (b[2] - a[2]) * u),
  ];

  function scoreColor(p) {
    // Rank percentile, not the raw z-score: the z-score cross-section piles up
    // near zero, which renders as one grey blob with two coloured outliers.
    const pct = isNum(p.score_pct)
      ? p.score_pct
      : isNum(p.score)
        ? clamp((p.score + 2) / 4, 0, 1)
        : 0.5;
    return pct < 0.5 ? mix(SHORT, MID, pct * 2) : mix(MID, LONG, (pct - 0.5) * 2);
  }

  const rgba = (c, alpha) => `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;

  /* ── data -> world coordinates ───────────────────────────────── */

  function axisValue(p) {
    if (state.axisY === "live") {
      const tick = state.live.get(p.symbol);
      const chg = tick && isNum(tick.chg) ? tick.chg : p.chg;
      return isNum(chg) ? clamp(chg / 0.05, -3, 3) : 0;   // +/-5% fills the axis
    }
    if (state.axisY === "score") return isNum(p.score) ? p.score : 0;
    return isNum(p.ivol) ? p.ivol : 0;
  }

  function retarget() {
    const k = SPAN / 3;
    const risk = state.mode === "risk";
    for (const p of state.points) {
      if (risk) {
        p.tx = (isNum(p.pc1) ? p.pc1 : 0) * k;
        p.ty = (isNum(p.pc2) ? p.pc2 : 0) * k;
        p.tz = (isNum(p.pc3) ? p.pc3 : 0) * k;
      } else {
        p.tx = (isNum(p.momentum) ? p.momentum : 0) * k;
        p.ty = axisValue(p) * k;
        p.tz = (isNum(p.reversal) ? p.reversal : 0) * k;
      }
    }
  }

  function applyCloud(cloud) {
    const empty = el("stage-empty");
    if (!cloud || !cloud.available || !cloud.points.length) {
      empty.hidden = false;
      state.points = [];
      return;
    }
    empty.hidden = true;

    const advs = cloud.points.map((p) => p.adv).filter(isNum).sort((a, b) => a - b);
    const advRank = (v) => {
      if (!isNum(v) || !advs.length) return 0.35;
      let lo = 0;
      let hi = advs.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (advs[mid] < v) lo = mid + 1;
        else hi = mid;
      }
      return lo / Math.max(1, advs.length - 1);
    };

    const next = [];
    const seen = new Map();
    for (const raw of cloud.points) {
      const prev = state.byId.get(raw.symbol);
      const p = Object.assign({}, raw);
      p.size = 2.4 + advRank(raw.adv) * 4.2;
      p.pulse = prev ? prev.pulse : 0;
      // Keep the current position across refreshes so the cloud eases rather
      // than jumping when a new cross-section arrives.
      p.cx = prev ? prev.cx : (Math.random() - 0.5) * SPAN * 2.4;
      p.cy = prev ? prev.cy : (Math.random() - 0.5) * SPAN * 2.4;
      p.cz = prev ? prev.cz : (Math.random() - 0.5) * SPAN * 2.4;
      next.push(p);
      seen.set(p.symbol, p);
    }
    state.points = next;
    state.byId = seen;
    retarget();

    // sector filter options
    const select = el("sector-filter");
    const sectors = [...new Set(next.map((p) => p.sector))].sort();
    const current = select.value;
    select.innerHTML =
      '<option value="">all sectors</option>' +
      sectors.map((s) => `<option value="${s}">${s}</option>`).join("");
    select.value = sectors.includes(current) ? current : "";
    state.sector = select.value;

    el("pill-asof").textContent = "asof " + cloud.asof;
    el("pill-universe").textContent = cloud.names + " names";
    el("book-asof").textContent = cloud.asof;

    renderBook(cloud);
    renderRisk(cloud);
    renderRiskModel(cloud.risk);
    renderIC(cloud.factor_ic);
    renderSectorNet(cloud.sector_net || {});
    renderPipeline(cloud.pipeline || [], cloud.risk || {});
    renderScree(cloud.risk || {});
    renderExposure(cloud.risk || {});
    state.lastRisk = cloud.risk || null;
    state.lastPipeline = cloud.pipeline || null;
    const brand = el("brand-build");
    if (brand) {
      brand.textContent =
        (cloud.construction === "risk_constrained" ? "factor-neutral" : "equal-weight") +
        " · lambdarank · paper";
    }
    setModeLabels();
  }

  /* ── panels ──────────────────────────────────────────────────── */

  function renderBook(cloud) {
    const b = cloud.book || {};
    el("bk-gross").textContent = fmt(b.gross, 2);
    el("bk-net").textContent = fmtSigned(b.net, 3);
    el("bk-longs").textContent = b.longs ?? "—";
    el("bk-shorts").textContent = b.shorts ?? "—";
    const build = el("bk-build");
    if (build) {
      const label = {
        risk_constrained: "risk-neutral",
        quantile_equal_weight: "equal-wt",
      }[cloud.construction] || cloud.construction || "—";
      build.textContent = label;
    }

    const held = cloud.points.filter((p) => p.side !== "flat");
    held.sort((a, z) => (z.score ?? 0) - (a.score ?? 0));
    const row = (p) =>
      `<li title="${p.sector}"><span class="sym">${p.symbol}</span>` +
      `<span class="wt">${fmtPct(p.weight, 1)}</span>` +
      `<span class="sc">${fmtSigned(p.score, 2)}</span></li>`;
    el("list-long").innerHTML = held.filter((p) => p.side === "long").map(row).join("");
    el("list-short").innerHTML = held
      .filter((p) => p.side === "short")
      .reverse()
      .map(row)
      .join("");
  }

  function barRow(label, value, limit, text) {
    const pct = limit ? clamp((Math.abs(value) / limit) * 100, 0, 100) : 0;
    const cls = Math.abs(value) > limit ? "bar-over" : "bar-neutral";
    return `<div class="bar-row">
      <div class="bar-top"><span class="k">${label}</span><span class="v">${text}</span></div>
      <div class="bar-track"><div class="bar-fill ${cls}" style="width:${pct}%"></div></div>
    </div>`;
  }

  function renderRisk(cloud) {
    const b = cloud.book || {};
    const lim = (window.__gigLimits || {});
    const html = [
      barRow("gross leverage", b.gross ?? 0, lim.max_gross_leverage ?? 2,
        `${fmt(b.gross, 2)} / ${fmt(lim.max_gross_leverage, 2)}`),
      barRow("net exposure", b.net ?? 0, lim.max_net_exposure ?? 0.1,
        `${fmtSigned(b.net, 3)} / ${fmt(lim.max_net_exposure, 2)}`),
      barRow("max name", b.max_name ?? 0, lim.max_name_weight ?? 0.05,
        `${fmtPct(b.max_name, 1)} / ${fmtPct(lim.max_name_weight, 1)}`),
    ].join("");
    el("risk-bars").innerHTML = html;

    const breaches = cloud.breaches || [];
    el("breaches").innerHTML = breaches.length
      ? breaches
          .map(
            (b2) =>
              `<div class="breach">${b2.name} ${fmt(b2.value, 3)} &gt; ${fmt(b2.limit, 3)}</div>`
          )
          .join("")
      : '<div class="breach-none">all limits clear</div>';
  }

  function renderRiskModel(risk) {
    const note = el("risk-model-note");
    if (!risk || !risk.available) {
      note.textContent = "unavailable";
      el("risk-metrics").innerHTML =
        '<div class="m"><span class="m-k">needs history</span><span class="m-v">—</span></div>';
      el("risk-split").innerHTML = "";
      el("risk-contrib").innerHTML = "";
      return;
    }

    note.textContent = `${risk.n_factors} PC · ${risk.n_obs}d`;

    const cell = (k, v, cls = "") =>
      `<div class="m"><span class="m-k">${k}</span><span class="m-v ${cls}">${v}</span></div>`;
    // Predicted vs realized on the same window is the model's own honesty check.
    const gap =
      isNum(risk.ex_ante_vol) && isNum(risk.realized_vol) && risk.realized_vol > 0
        ? risk.ex_ante_vol / risk.realized_vol
        : null;
    el("risk-metrics").innerHTML = [
      cell("predicted vol", fmtPct(risk.ex_ante_vol, 1)),
      cell("realized vol", fmtPct(risk.realized_vol, 1)),
      cell("effective names", fmt(risk.effective_names, 1)),
      cell("pred / real", isNum(gap) ? gap.toFixed(2) + "x" : "—"),
    ].join("");

    const sysPct = clamp((risk.systematic_share || 0) * 100, 0, 100);
    el("risk-split").innerHTML = `<div class="bar-row">
      <div class="bar-top">
        <span class="k">systematic / specific</span>
        <span class="v">${fmtPct(risk.systematic_share, 0)} / ${fmtPct(risk.specific_share, 0)}</span>
      </div>
      <div class="bar-track"><div class="bar-fill bar-neutral" style="width:${sysPct}%"></div></div>
    </div>`;

    el("risk-contrib").innerHTML = (risk.top_contributors || [])
      .map(
        (c) =>
          `<li title="${c.sector}"><span class="sym">${c.symbol}</span>` +
          `<span class="wt">${fmtPct(c.weight, 1)}</span>` +
          `<span class="sc">${fmtPct(c.share, 1)}</span></li>`
      )
      .join("");
  }

  function renderIC(ic) {
    if (!ic) return;
    const rows = Object.entries(ic).sort(
      (a, b) => Math.abs(b[1].ic_mean ?? 0) - Math.abs(a[1].ic_mean ?? 0)
    );
    el("ic-bars").innerHTML = rows
      .map(([name, s]) => {
        const mean = s.ic_mean ?? 0;
        const pct = clamp((Math.abs(mean) / 0.05) * 100, 2, 100);
        const cls = mean >= 0 ? "bar-pos" : "bar-neg";
        const t = isNum(s.ic_tstat_nw) ? `t ${s.ic_tstat_nw.toFixed(1)}` : "t —";
        return `<div class="bar-row">
          <div class="bar-top"><span class="k">${name}</span><span class="v">${fmt(mean, 4)} · ${t}</span></div>
          <div class="bar-track"><div class="bar-fill ${cls}" style="width:${pct}%"></div></div>
        </div>`;
      })
      .join("");
  }

  function renderResearch(r) {
    if (!r || !r.available) {
      el("metrics").innerHTML =
        '<div class="m"><span class="m-k">no run yet</span><span class="m-v">—</span></div>';
      el("research-note").textContent =
        "Run: python -m gig backtest --source yahoo --no-ml";
      return;
    }
    const m = r.metrics || {};
    const cell = (k, v, cls = "") =>
      `<div class="m"><span class="m-k">${k}</span><span class="m-v ${cls}">${v}</span></div>`;
    const sign = (v) => (isNum(v) ? (v >= 0 ? "good" : "bad") : "");
    el("metrics").innerHTML = [
      cell("sharpe", fmt(m.sharpe, 2), sign(m.sharpe)),
      cell("ann return", fmtPct(m.ann_return, 1), sign(m.ann_return)),
      cell("ann vol", fmtPct(m.ann_vol, 1)),
      cell("max dd", fmtPct(m.max_drawdown, 1), "bad"),
      cell("turnover", fmt(m.ann_turnover, 1) + "x"),
      cell("deflated sharpe", fmt(m.dsr, 3)),
    ].join("");
    el("research-src").textContent = (r.source || "—") + " · " + (r.modified || "");
    el("research-note").textContent =
      r.source === "synthetic"
        ? "Synthetic panel: the premium is planted. This measures the pipeline, not an edge."
        : "Current listing tape, not survivorship-free. Treat as research, not live performance.";
  }

  function renderStatus(s) {
    window.__gigLimits = s.limits || {};
    const db = el("pill-db");
    const bars = (s.tables || {}).bars || 0;
    db.textContent = "db " + (bars ? bars.toLocaleString() + " bars" : "empty");
    db.className = "pill " + (bars ? "pill-ok" : "pill-bad");

    const keys = s.keys || {};
    const live = keys.alpaca || keys.finnhub;
    const feed = el("pill-feed");
    feed.textContent = keys.alpaca ? "feed alpaca iex" : keys.finnhub ? "feed finnhub" : "feed none";
    feed.className = "pill " + (live ? "pill-ok" : "pill-warn");

    const ml = s.ml || {};
    const mlp = el("pill-ml");
    if (mlp) {
      mlp.textContent = ml.lightgbm ? "ml lightgbm" : ml.sklearn ? "ml sklearn" : "ml none";
      mlp.className = "pill " + (ml.lightgbm || ml.sklearn ? "pill-ok" : "pill-warn");
    }

    const ol = s.ollama || {};
    const olp = el("pill-ollama");
    if (olp) {
      olp.textContent = ol.available ? "desk ollama" : "desk structured";
      olp.className = "pill " + (ol.available ? "pill-ok" : "pill-warn");
    }

    const paper = el("pill-paper");
    if (paper) {
      paper.textContent = keys.alpaca ? "paper ready" : "paper keys?";
      paper.className = "pill " + (keys.alpaca ? "pill-ok" : "pill-warn");
    }
  }

  function renderSectorNet(sectorNet) {
    const host = el("sector-bars");
    if (!host) return;
    const lim = (window.__gigLimits || {}).max_sector_net || 0.1;
    const rows = Object.entries(sectorNet || {})
      .map(([k, v]) => [k, v])
      .sort((a, b) => Math.abs(b[1] || 0) - Math.abs(a[1] || 0))
      .slice(0, 8);
    if (!rows.length) {
      host.innerHTML = '<div class="breach-none">no sector net yet</div>';
      return;
    }
    host.innerHTML = rows
      .map(([name, value]) => {
        const pct = clamp((Math.abs(value) / lim) * 100, 0, 100);
        const cls = Math.abs(value) > lim ? "bar-over" : value >= 0 ? "bar-pos" : "bar-neg";
        return `<div class="bar-row">
          <div class="bar-top"><span class="k">${name}</span><span class="v">${fmtSigned(value, 3)}</span></div>
          <div class="bar-track"><div class="bar-fill ${cls}" style="width:${pct}%"></div></div>
        </div>`;
      })
      .join("");
  }

  function renderPipeline(pipeline, risk) {
    const host = el("pipe-steps");
    const eq = el("pipe-eq");
    if (eq) eq.textContent = (risk && risk.equation) || "Σ = BB′ + D";
    if (!host) return;
    if (!pipeline || !pipeline.length) {
      host.innerHTML = '<div class="pipe-step"><span class="pipe-k">loading</span><span class="pipe-v">signal path</span></div>';
      return;
    }
    host.innerHTML = pipeline
      .map(
        (s) =>
          `<div class="pipe-step"><span class="pipe-k">${s.label || s.id}</span>` +
          `<span class="pipe-v" title="${s.detail || ""}">${s.detail || ""}</span></div>`
      )
      .join("");
  }

  function renderScree(risk) {
    const c = el("scree");
    if (!c) return;
    const ctx2 = c.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = c.parentElement.clientWidth || 200;
    const h = 72;
    c.width = w * dpr;
    c.height = h * dpr;
    c.style.width = w + "px";
    c.style.height = h + "px";
    ctx2.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx2.clearRect(0, 0, w, h);
    const shares = (risk && risk.eigenvalue_share) || [];
    if (!shares.length) {
      ctx2.fillStyle = "#556070";
      ctx2.font = "10px IBM Plex Mono, monospace";
      ctx2.fillText("PCA scree after cloud loads", 10, 40);
      return;
    }
    const n = shares.length;
    const gap = 6;
    const barW = Math.max(10, (w - 16 - gap * (n - 1)) / n);
    shares.forEach((s, i) => {
      const bh = Math.max(2, s * (h - 22));
      const x = 8 + i * (barW + gap);
      const y = h - 10 - bh;
      ctx2.fillStyle = i === 0 ? "#6cb6ff" : i === 1 ? "#b8e63b" : "#8b97a8";
      ctx2.fillRect(x, y, barW, bh);
      ctx2.fillStyle = "#9aa6b5";
      ctx2.font = "9px IBM Plex Mono, monospace";
      ctx2.fillText("PC" + (i + 1), x, h - 2);
      ctx2.fillStyle = "#e8eef6";
      ctx2.fillText((100 * s).toFixed(0) + "%", x, y - 3);
    });
  }

  function renderExposure(risk) {
    const host = el("exposure-bars");
    if (!host) return;
    const exp = (risk && risk.factor_exposure) || {};
    const rows = Object.entries(exp);
    if (!rows.length) {
      host.innerHTML = '<div class="breach-none">no B′w yet</div>';
      return;
    }
    const maxAbs = Math.max(...rows.map(([, v]) => Math.abs(v || 0)), 1e-12);
    host.innerHTML = rows
      .map(([name, value]) => {
        const pct = clamp((Math.abs(value) / maxAbs) * 100, 0, 100);
        const cls = Math.abs(value) < 1e-6 ? "bar-neutral" : value >= 0 ? "bar-pos" : "bar-neg";
        return `<div class="bar-row">
          <div class="bar-top"><span class="k">${name}</span><span class="v">${fmtSigned(value, 4)}</span></div>
          <div class="bar-track"><div class="bar-fill ${cls}" style="width:${pct}%"></div></div>
        </div>`;
      })
      .join("");
  }

  function setModeLabels() {
    const blurb = el("stage-blurb");
    const lx = el("legend-x");
    const ly = el("legend-y");
    const lz = el("legend-z");
    const note = el("legend-note");
    const ctlY = el("ctl-y");
    if (state.mode === "risk") {
      if (blurb) blurb.textContent = "PCA risk space · PC1 · PC2 · PC3 · α colour · Σ = BB′ + D";
      if (lx) lx.textContent = "PC1 (systematic)";
      if (ly) ly.textContent = "PC2";
      if (lz) lz.textContent = "PC3";
      if (note) note.textContent = "exposures from statistical risk model · ring = held";
      if (ctlY) ctlY.style.display = "none";
    } else {
      if (blurb) {
        blurb.textContent =
          state.mode === "held"
            ? "held names only · neutralized factors · alpha colour"
            : "neutralized momentum · ivol · reversal · alpha colour";
      }
      if (lx) lx.textContent = "12-1 momentum";
      if (ly) {
        ly.textContent =
          { ivol: "idiosyncratic vol", live: "live % change", score: "alpha score" }[state.axisY] ||
          "idiosyncratic vol";
      }
      if (lz) lz.textContent = "short-term reversal";
      if (note) note.textContent = "rank-normalized · ring = held · pulse = quote";
      if (ctlY) ctlY.style.display = "";
    }
  }

  function renderSpark(experiments) {
    const c = el("spark");
    if (!c) return;
    const ctx2 = c.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = c.parentElement.clientWidth || 280;
    const h = 64;
    c.width = w * dpr;
    c.height = h * dpr;
    c.style.width = w + "px";
    c.style.height = h + "px";
    ctx2.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx2.clearRect(0, 0, w, h);

    const runs = (experiments && experiments.runs) || [];
    const ys = runs.map((r) => r.sharpe).filter(isNum);
    if (ys.length < 2) {
      ctx2.fillStyle = "#556070";
      ctx2.font = "10px IBM Plex Mono, monospace";
      ctx2.fillText("run more backtests to see the path", 10, 34);
      return;
    }
    const min = Math.min(...ys, 0);
    const max = Math.max(...ys, 0);
    const span = max - min || 1;
    ctx2.strokeStyle = "#1a2330";
    ctx2.beginPath();
    ctx2.moveTo(0, h * (1 - (0 - min) / span));
    ctx2.lineTo(w, h * (1 - (0 - min) / span));
    ctx2.stroke();

    ctx2.beginPath();
    ys.forEach((y, i) => {
      const x = (i / (ys.length - 1)) * (w - 8) + 4;
      const yy = h - 6 - ((y - min) / span) * (h - 12);
      if (i === 0) ctx2.moveTo(x, yy);
      else ctx2.lineTo(x, yy);
    });
    ctx2.strokeStyle = "#7dd3fc";
    ctx2.lineWidth = 1.5;
    ctx2.stroke();
    const last = ys[ys.length - 1];
    ctx2.fillStyle = last >= 0 ? "#3dffa8" : "#ff4d6d";
    ctx2.font = "10px IBM Plex Mono, monospace";
    ctx2.fillText("Sharpe " + last.toFixed(2), 8, 14);
  }

  function renderPaper(p) {
    const metrics = el("paper-metrics");
    const drift = el("paper-drift");
    const note = el("paper-note");
    const open = el("paper-open");
    if (!metrics) return;
    if (!p || !p.available) {
      metrics.innerHTML =
        '<div class="m"><span class="m-k">status</span><span class="m-v">offline</span></div>';
      if (drift) drift.innerHTML = "";
      if (open) open.textContent = "—";
      if (note) note.textContent = (p && (p.hint || p.error)) || "Alpaca keys not loaded";
      return;
    }
    const acct = p.account || {};
    const cell = (k, v, cls = "") =>
      `<div class="m"><span class="m-k">${k}</span><span class="m-v ${cls}">${v}</span></div>`;
    metrics.innerHTML = [
      cell("NAV", isNum(p.nav) ? p.nav.toLocaleString(undefined, { maximumFractionDigits: 0 }) : "—"),
      cell("held", String(p.n_positions ?? "—")),
      cell("worst drift", fmtPct(p.worst_drift, 2)),
      cell("session", p.market_open === true ? "open" : p.market_open === false ? "closed" : "—"),
    ].join("");
    if (open) open.textContent = acct.paper ? "paper" : "account";
    if (drift) {
      drift.innerHTML = (p.drift || [])
        .slice(0, 8)
        .map(
          (r) =>
            `<li><span class="sym">${r.symbol}</span>` +
            `<span class="wt">${fmtPct(r.current_weight, 1)}</span>` +
            `<span class="sc">${fmtPct(r.drift, 1)}</span></li>`
        )
        .join("");
    }
  }

  function deskAppend(cls, text) {
    const chat = el("desk-chat");
    if (!chat) return;
    const div = document.createElement("div");
    div.className = "desk-msg " + cls;
    div.textContent = text;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
  }

  async function askDesk(question) {
    if (!question) return;
    deskAppend("user", question);
    deskAppend("muted", "thinking…");
    const chat = el("desk-chat");
    const thinking = chat ? chat.lastChild : null;
    try {
      const res = await fetch("/api/ollama/brief", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      if (thinking) thinking.remove();
      if (!data.available) {
        deskAppend("err", data.error || data.hint || "Desk unavailable");
        return;
      }
      const tag = data.source === "ollama" ? "" : " [structured]";
      deskAppend("assistant", (data.answer || "(empty reply)") + tag);
    } catch (err) {
      if (thinking) thinking.remove();
      deskAppend("err", String(err));
    }
  }

  function wireDesk() {
    const form = el("desk-form");
    const input = el("desk-input");
    const prompts = el("desk-prompts");
    if (form && input) {
      form.addEventListener("submit", (e) => {
        e.preventDefault();
        const q = input.value.trim();
        input.value = "";
        askDesk(q);
      });
    }
    if (prompts) {
      getJSON("/api/ollama/status")
        .then((s) => {
          const st = el("desk-status");
          if (st) {
            st.textContent = s.available
              ? s.model || "ollama"
              : "structured · install ollama for LLM";
          }
          prompts.innerHTML = (s.prompts || [])
            .map((p) => `<button type="button" class="prompt-chip">${p}</button>`)
            .join("");
          prompts.querySelectorAll(".prompt-chip").forEach((btn) => {
            btn.addEventListener("click", () => askDesk(btn.textContent));
          });
        })
        .catch(() => {
          const st = el("desk-status");
          if (st) st.textContent = "offline";
        });
    }
  }

  /* ── tape ────────────────────────────────────────────────────── */

  function renderTape(payload) {
    const label = el("tape-label");
    if (label) {
      // Feed source stays in the header pill — no banner on the tape.
      label.hidden = true;
    }
    if (!payload || !payload.available) {
      return;
    }

    const strip = el("tape-strip");
    if (!strip) return;
    const html = [];
    for (const q of payload.quotes) {
      const px = isNum(q.last) ? q.last : isNum(q.bid) && isNum(q.ask) ? (q.bid + q.ask) / 2 : null;
      const prev = state.live.get(q.symbol);
      let chg = prev && isNum(prev.chg) ? prev.chg : null;
      let moved = false;

      if (isNum(px)) {
        const base = prev && isNum(prev.base) ? prev.base : px;
        chg = base ? (px - base) / base : 0;
        moved = !!prev && isNum(prev.last) && prev.last !== px;
        state.live.set(q.symbol, { base, last: px, chg });
        if (moved) {
          const point = state.byId.get(q.symbol);
          if (point) point.pulse = 1;
        }
      }

      const dir = isNum(chg) && chg > 0 ? "up" : isNum(chg) && chg < 0 ? "down" : "";
      html.push(
        `<span class="tick ${dir} ${moved ? "flash" : ""}">` +
          `<span class="t-sym">${q.symbol}</span>` +
          `<span class="t-px">${isNum(px) ? px.toFixed(2) : "—"}</span>` +
          `<span class="t-sym">${isNum(chg) ? fmtPct(chg, 2) : ""}</span></span>`
      );
    }
    strip.innerHTML = html.join("");
    if (state.axisY === "live") retarget();
  }

  /* ── 3D projection ───────────────────────────────────────────── */

  function project(x, y, z) {
    const cy = Math.cos(state.yaw);
    const sy = Math.sin(state.yaw);
    const cp = Math.cos(state.pitch);
    const sp = Math.sin(state.pitch);

    const x1 = x * cy - z * sy;
    const z1 = x * sy + z * cy;
    const y2 = y * cp - z1 * sp;
    const z2 = y * sp + z1 * cp;

    const depth = z2 + state.dist;
    if (depth < 30) return null;
    const f = FOCAL / depth;
    return {
      sx: state.width / 2 + x1 * f,
      sy: state.height / 2 - y2 * f,
      depth,
      f,
    };
  }

  function drawFrame() {
    const step = SPAN / 2.5;

    const origin = project(0, 0, 0);
    if (!origin) return;

    const axes = [
      {
        dir: [1, 0, 0],
        color: "#ef6b7a",
        label: state.mode === "risk" ? "PC1" : "momentum",
      },
      {
        dir: [0, 1, 0],
        color: "#9bd071",
        label: state.mode === "risk" ? "PC2" : yLabel(),
      },
      {
        dir: [0, 0, 1],
        color: "#5fb2f5",
        label: state.mode === "risk" ? "PC3" : "reversal",
      },
    ];
    ctx.font = "10px ui-monospace, monospace";
    for (const ax of axes) {
      const end = project(
        ax.dir[0] * SPAN * 1.16,
        ax.dir[1] * SPAN * 1.16,
        ax.dir[2] * SPAN * 1.16
      );
      if (!end) continue;
      ctx.strokeStyle = ax.color + "99";
      ctx.lineWidth = 1.2;
      line(origin, end);
      ctx.fillStyle = ax.color;
      ctx.fillText(ax.label, end.sx + 6, end.sy + 3);

      // Sigma ticks so the spread has a scale rather than being decorative.
      for (const tick of [-2, -1, 1, 2]) {
        const at = project(
          ax.dir[0] * tick * step,
          ax.dir[1] * tick * step,
          ax.dir[2] * tick * step
        );
        if (!at) continue;
        ctx.fillStyle = "rgba(150,170,200,0.5)";
        ctx.beginPath();
        ctx.arc(at.sx, at.sy, 1.4, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "rgba(150,170,200,0.34)";
        ctx.fillText(String(tick), at.sx + 4, at.sy - 4);
      }
    }
  }

  function line(a, b) {
    ctx.beginPath();
    ctx.moveTo(a.sx, a.sy);
    ctx.lineTo(b.sx, b.sy);
    ctx.stroke();
  }

  function yLabel() {
    if (state.axisY === "live") return "live %";
    if (state.axisY === "score") return "alpha";
    return "ivol";
  }

  function render() {
    if (state.spin && !state.dragging) state.yaw += 0.0016;

    ctx.clearRect(0, 0, state.width, state.height);
    ctx.fillStyle = "#0c1017";
    ctx.fillRect(0, 0, state.width, state.height);

    const drawable = [];
    for (const p of state.points) {
      p.cx += (p.tx - p.cx) * EASE;
      p.cy += (p.ty - p.cy) * EASE;
      p.cz += (p.tz - p.cz) * EASE;
      if (p.pulse > 0) p.pulse = Math.max(0, p.pulse - 0.02);

      const q = project(p.cx, p.cy, p.cz);
      if (!q) continue;
      p.sx = q.sx;
      p.sy = q.sy;
      p.depth = q.depth;
      p.r = Math.max(1.2, p.size * q.f * 0.62);
      p.dim =
        (state.sector && p.sector !== state.sector) ||
        (state.mode === "held" && p.side === "flat");
      // In risk mode, size held names a bit larger so the book reads clearly.
      if (state.mode === "risk" && p.side !== "flat") {
        p.r *= 1.15 + Math.min(0.5, Math.abs(p.weight || 0) * 8);
      }
      drawable.push(p);
    }

    drawFrame();
    drawable.sort((a, b) => b.depth - a.depth);

    // Nearest point to the cursor, in screen space.
    let hover = null;
    let bestDist = 18;
    for (const p of drawable) {
      if (p.dim) continue;
      const d = Math.hypot(p.sx - state.mouse.x, p.sy - state.mouse.y);
      if (d < bestDist) {
        bestDist = d;
        hover = p;
      }
    }
    state.hover = hover;

    const near = 170;
    const far = 640;
    for (const p of drawable) {
      p.fade = 1 - clamp((p.depth - near) / (far - near), 0, 1);
      p.color = scoreColor(p);
    }

    // Pass 1 — additive halos. Stacked low-alpha rings are far cheaper than a
    // radial gradient per point and read the same at this scale.
    ctx.globalCompositeOperation = "lighter";
    for (const p of drawable) {
      if (p.dim) continue;
      const held = p.side !== "flat";
      const base = (held ? 0.1 : 0.05) * (0.35 + p.fade * 0.65);
      for (const [scale, weight] of [[3.0, 0.5], [2.0, 0.8], [1.35, 1.0]]) {
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, p.r * scale, 0, Math.PI * 2);
        ctx.fillStyle = rgba(p.color, base * weight);
        ctx.fill();
      }
    }
    ctx.globalCompositeOperation = "source-over";

    // Pass 2 — cores, rings, pulses.
    for (const p of drawable) {
      if (p.dim) {
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, Math.max(1, p.r * 0.55), 0, Math.PI * 2);
        ctx.fillStyle = "rgba(120,135,160,0.13)";
        ctx.fill();
        continue;
      }

      ctx.beginPath();
      ctx.arc(p.sx, p.sy, p.r, 0, Math.PI * 2);
      ctx.fillStyle = rgba(p.color, 0.55 + p.fade * 0.45);
      ctx.fill();

      if (p.side !== "flat") {
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, p.r + 3, 0, Math.PI * 2);
        ctx.strokeStyle = rgba(p.side === "long" ? LONG : SHORT, 0.4 + p.fade * 0.5);
        ctx.lineWidth = 1.3;
        ctx.stroke();
      }

      if (p.pulse > 0) {
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, p.r + 3 + (1 - p.pulse) * 14, 0, Math.PI * 2);
        ctx.strokeStyle = rgba(p.color, p.pulse * 0.55);
        ctx.lineWidth = 1.1;
        ctx.stroke();
      }
    }

    if (hover) drawHover(hover);
    paintTip(hover);
    requestAnimationFrame(render);
  }

  function drawHover(p) {
    // Drop line to the floor grid: without it, depth is ambiguous in a
    // projected cloud and you cannot tell which point you are reading.
    const foot = project(p.cx, -SPAN, p.cz);
    if (foot) {
      ctx.save();
      ctx.setLineDash([2, 4]);
      ctx.strokeStyle = "rgba(255,255,255,0.28)";
      ctx.lineWidth = 1;
      line({ sx: p.sx, sy: p.sy }, foot);
      ctx.restore();
      ctx.beginPath();
      ctx.arc(foot.sx, foot.sy, 2, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,255,255,0.4)";
      ctx.fill();
    }

    ctx.beginPath();
    ctx.arc(p.sx, p.sy, p.r + 7, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255,255,255,0.8)";
    ctx.lineWidth = 1.3;
    ctx.stroke();

    ctx.fillStyle = "#fff";
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillText(p.symbol, p.sx + p.r + 10, p.sy - p.r - 4);
  }

  function paintTip(p) {
    if (!p) {
      tip.hidden = true;
      return;
    }
    const tick = state.live.get(p.symbol);
    const px = tick && isNum(tick.last) ? tick.last : p.close;
    const chg = tick && isNum(tick.chg) ? tick.chg : p.chg;
    const riskRows =
      state.mode === "risk"
        ? `<div class="tip-row"><span>PC1 / 2 / 3</span><b>${fmtSigned(p.pc1, 1)} / ${fmtSigned(p.pc2, 1)} / ${fmtSigned(p.pc3, 1)}</b></div>`
        : `<div class="tip-row"><span>mom / rev</span><b>${fmtSigned(p.momentum, 1)} / ${fmtSigned(p.reversal, 1)}</b></div>` +
          `<div class="tip-row"><span>ivol</span><b>${fmtSigned(p.ivol, 1)}</b></div>`;
    tip.innerHTML =
      `<div class="tip-sym">${p.symbol}</div>` +
      `<div class="tip-sec">${p.sector}</div>` +
      `<div class="tip-row"><span>last</span><b>${isNum(px) ? px.toFixed(2) : "—"}</b></div>` +
      `<div class="tip-row"><span>chg</span><b>${fmtPct(chg, 2)}</b></div>` +
      `<div class="tip-row"><span>alpha z</span><b>${fmtSigned(p.score, 2)}</b></div>` +
      riskRows +
      `<div class="tip-row"><span>adv</span><b>${fmtAdv(p.adv)}</b></div>` +
      `<div class="tip-row"><span>weight</span><b>${p.side === "flat" ? "flat" : fmtPct(p.weight, 2)}</b></div>`;
    tip.hidden = false;
    const pad = 14;
    const rect = tip.getBoundingClientRect();
    let x = p.sx + pad;
    let y = p.sy + pad;
    if (x + rect.width > state.width) x = p.sx - rect.width - pad;
    if (y + rect.height > state.height) y = p.sy - rect.height - pad;
    tip.style.left = Math.max(4, x) + "px";
    tip.style.top = Math.max(4, y) + "px";
  }

  /* ── events ──────────────────────────────────────────────────── */

  function resize() {
    const wrap = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    state.width = wrap.clientWidth;
    state.height = wrap.clientHeight;
    canvas.width = state.width * dpr;
    canvas.height = state.height * dpr;
    canvas.style.width = state.width + "px";
    canvas.style.height = state.height + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function wireEvents() {
    window.addEventListener("resize", resize);

    canvas.addEventListener("pointerdown", (e) => {
      state.dragging = true;
      state.lastX = e.clientX;
      state.lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointerup", (e) => {
      state.dragging = false;
      canvas.releasePointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
      const rect = canvas.getBoundingClientRect();
      state.mouse.x = e.clientX - rect.left;
      state.mouse.y = e.clientY - rect.top;
      if (!state.dragging) return;
      state.yaw += (e.clientX - state.lastX) * 0.006;
      state.pitch = clamp(state.pitch + (e.clientY - state.lastY) * 0.005, -1.35, 1.35);
      state.lastX = e.clientX;
      state.lastY = e.clientY;
    });
    canvas.addEventListener("pointerleave", () => {
      state.mouse.x = -1e4;
      state.mouse.y = -1e4;
    });
    canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        state.dist = clamp(state.dist * (1 + e.deltaY * 0.0012), 150, 900);
      },
      { passive: false }
    );

    el("axis-y").addEventListener("change", (e) => {
      state.axisY = e.target.value;
      el("legend-y").textContent =
        { ivol: "idiosyncratic vol", live: "live % change", score: "alpha score" }[state.axisY];
      retarget();
    });

    el("sector-filter").addEventListener("change", (e) => {
      state.sector = e.target.value;
    });

    document.querySelectorAll(".mode-tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".mode-tab").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.mode = btn.dataset.mode || "alpha";
        setModeLabels();
        retarget();
      });
    });

    const spin = el("btn-spin");
    spin.addEventListener("click", () => {
      state.spin = !state.spin;
      spin.dataset.on = state.spin ? "1" : "0";
    });

    el("btn-reset").addEventListener("click", () => {
      state.yaw = 0.62;
      state.pitch = -0.32;
      state.dist = 340;
    });

    const refresh = el("btn-refresh");
    if (refresh) {
      refresh.addEventListener("click", async () => {
        try {
          await fetch("/api/cache/clear", { method: "POST" });
        } catch (_) {}
        await refreshStatus();
        await refreshCloud();
        await refreshPaper();
        try {
          renderResearch(await getJSON("/api/research"));
          renderSpark(await getJSON("/api/experiments"));
        } catch (_) {}
      });
    }
  }

  function startClock() {
    const tickClock = () => {
      el("clock").textContent = new Date().toLocaleTimeString("en-US", { hour12: false });
    };
    tickClock();
    setInterval(tickClock, 1000);
  }

  /* ── boot ────────────────────────────────────────────────────── */

  async function refreshCloud() {
    try {
      applyCloud(await getJSON("/api/cloud"));
    } catch (err) {
      console.error("cloud", err);
    }
  }

  async function refreshTape() {
    try {
      renderTape(await getJSON("/api/tape?limit=40"));
    } catch (err) {
      console.error("tape", err);
    }
  }

  async function refreshStatus() {
    try {
      renderStatus(await getJSON("/api/status"));
    } catch (err) {
      console.error("status", err);
    }
  }

  async function refreshPaper() {
    try {
      renderPaper(await getJSON("/api/paper?limit=120"));
    } catch (err) {
      console.error("paper", err);
    }
  }

  async function boot() {
    resize();
    // Browsers restore <select> values across reloads, which would leave the
    // controls showing one view while `state` renders another.
    el("axis-y").value = state.axisY;
    el("sector-filter").value = "";
    state.sector = "";
    wireEvents();
    wireDesk();
    startClock();
    requestAnimationFrame(render);

    await refreshStatus();
    await refreshCloud();
    try {
      renderResearch(await getJSON("/api/research"));
      renderSpark(await getJSON("/api/experiments"));
    } catch (err) {
      console.error("research", err);
    }
    refreshTape();
    refreshPaper();

    setInterval(refreshTape, 5000);
    setInterval(refreshStatus, 30000);
    setInterval(refreshCloud, 120000);
    setInterval(refreshPaper, 45000);
  }

  boot();
})();
