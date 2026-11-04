"""Chromium regression checks for the nine defects found in the 2026-11-04 visual review.

With ``--url`` it drives an already running terminal (``python -m gig serve``);
otherwise it starts an isolated local service with a temporary state directory,
as CI does. Every request that is not to that origin is aborted and recorded.
Sections use the terminal's labelled "Synthetic research demo" environment.
Global Intelligence uses whatever the server has already acquired; when that is
nothing, the labelled authored demonstration events are loaded. No provider is
contacted.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

from browser_terminal_check import ROOT, wait_ready
from playwright.sync_api import sync_playwright

OVERLAP = """(boxes) => { const bad = [];
  for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
    const a = boxes[i].box, b = boxes[j].box;
    if (a[0] < b[2] - 0.5 && a[2] > b[0] + 0.5 && a[1] < b[3] - 0.5 && a[3] > b[1] + 0.5) bad.push([boxes[i].text, boxes[j].text]);
  } return bad; }"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Running terminal to drive; omitted, an isolated server is started")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    with contextlib.ExitStack() as stack:
        url = args.url.rstrip("/") if args.url else stack.enter_context(isolated_server(args.report.parent))
        return run(url, args.report)


@contextlib.contextmanager
def isolated_server(log_dir: Path):
    with tempfile.TemporaryDirectory(prefix="gig-stage-a-browser-") as temporary:
        directory = Path(temporary)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        env = {k: v for k, v in os.environ.items() if not any(x in k for x in ("ALPACA", "FINNHUB", "FRED_API", "FINBERT_MODEL_DIR"))}
        env.update(PYTHONPATH=str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", ""),
                   GIG_BROKER_CONNECTIVITY_ENABLED="false", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        log_dir.mkdir(parents=True, exist_ok=True)
        code = ("import uvicorn;from pathlib import Path;from gig.service.app import create_app;"
                f"uvicorn.run(create_app(state_dir=Path({str(directory / 'state')!r})),host='127.0.0.1',port={port},log_level='warning')")
        with (log_dir / "stage-a-server.txt").open("w") as log:
            server = subprocess.Popen([sys.executable, "-c", code], env=env, cwd=directory, stdout=log, stderr=subprocess.STDOUT)
            try:
                wait_ready(url, server)
                yield url
            finally:
                server.terminate()
                server.wait(timeout=10)


def run(url: str, report_path: Path) -> int:
    checks, errors, blocked = [], [], []

    def check(defect, name, passed, detail=None):
        checks.append({"defect": defect, "name": name, "passed": bool(passed), "detail": detail})

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])

        def route(r):
            if r.request.url.startswith(url + "/"):
                r.continue_()
            else:
                blocked.append(r.request.url)
                r.abort()

        def open_page(viewport, reduced="no-preference"):
            context = browser.new_context(viewport=viewport, reduced_motion=reduced)
            context.route("**/*", route)
            page = context.new_page()
            page.on("pageerror", lambda e: errors.append(str(e)))
            return context, page

        def go(page, section):
            page.locator(f'#section-nav a[href="#{section}"]').click()
            page.wait_for_function(
                f"""document.querySelector('#section-nav a[href="#{section}"]').getAttribute('aria-current') === 'page'
                    && !document.getElementById('refresh').disabled""", timeout=30000)
            page.wait_for_timeout(500)

        def lab(page, module):
            page.wait_for_function(f"document.getElementById('lab-tab-{module}')?.getAttribute('aria-selected') === 'true'"
                                   " && document.getElementById('lab-status').classList.contains('hidden')", timeout=20000)
            page.wait_for_timeout(300)
            return page.evaluate("GIGModels.engine().layout()")

        def start(viewport, reduced="no-preference"):
            context, page = open_page(viewport, reduced)
            page.goto(url + "/#command-center", wait_until="networkidle")
            page.wait_for_function("!document.getElementById('refresh').disabled", timeout=30000)
            page.locator("#data-mode").select_option("synthetic")
            page.wait_for_function("!document.getElementById('refresh').disabled", timeout=30000)
            page.wait_for_timeout(500)
            return context, page

        context, page = start({"width": 1920, "height": 1080})

        # 1. The shared truth strip follows the visible section.
        before = page.locator("#source-label").inner_text()
        go(page, "global-intelligence")
        page.wait_for_function("document.getElementById('state-badge').textContent !== 'LOADING'", timeout=30000)
        gi = {k: page.locator(f"#{k}").inner_text() for k in ("state-badge", "source-label", "asof-label", "truth-notice")}
        check(1, "GI replaces the previous section's provenance", gi["source-label"].startswith("global-intelligence") and gi["source-label"] != before, {"before": before, **gi})
        check(1, "GI truth strip never claims LIVE", "LIVE" not in gi["state-badge"] and "Not a live" in gi["truth-notice"], gi)
        go(page, "risk")
        after = page.locator("#source-label").inner_text()
        check(1, "leaving GI restores the section's own provenance", not after.startswith("global-intelligence"), after)

        # 3. Legacy charts: every placed label is clear of the others and all axis labels survive.
        for section in ("command-center", "risk"):
            go(page, section)
            for dimension in ("3d", "2d"):
                page.locator(f"#view-{dimension}").click()
                page.wait_for_timeout(200)
                labels = json.loads(page.locator("#analytics-chart").get_attribute("data-labels") or "[]")
                overlaps = page.evaluate(OVERLAP, labels)
                axes = 3 if dimension == "3d" else 2
                check(3, f"{section} {dimension}: chart labels do not overlap", not overlaps, overlaps)
                check(3, f"{section} {dimension}: titles and both end ticks placed for every axis", len(labels) >= 3 * axes, [b["text"] for b in labels])
            page.locator("#view-3d").click()

        # 8. Metric cards: readable labels and no lone card on its own row.
        go(page, "risk")
        labels = page.locator("#metrics .metric-label > span:first-child").all_inner_texts()
        machine = [t for t in labels if any(w in t.split() for w in ("Es", "Var", "Nav", "Currency"))]
        check(8, "risk metric labels are humanized", not machine and "Historical VaR (95%)" in labels and "Synthetic NAV" in labels, labels)
        rows = page.evaluate("[...document.querySelectorAll('#metrics .metric-card')].map(c => Math.round(c.getBoundingClientRect().top))")
        check(8, "five risk metric cards share one row at 1920px", len(set(rows)) == 1 and len(rows) == 5, rows)
        go(page, "models")
        labels = page.locator("#metrics .metric-label > span:first-child").all_inner_texts()
        check(8, "model metric labels are humanized", "Smoother uses future data" in labels and "Kalman log-likelihood" in labels, labels)

        # 2. Risk terrain legend reports the actual P&L range, with a zero tick.
        go(page, "portfolio")
        lab(page, "risk-terrain")
        expected = page.evaluate("""async () => { const r = await (await fetch('/api/v1/viz/risk-terrain?mode=synthetic')).json();
            const v = Object.values(r.data.horizons).flat().map(p => p.model_pnl); const lo = Math.min(...v), hi = Math.max(...v);
            return { lo, hi, low: GIG3D.format(lo), high: GIG3D.format(hi) }; }""")
        shown = page.locator("#lab-legend .legend-value").all_inner_texts()
        zero = page.locator("#lab-legend .legend-zero").count()
        check(2, "risk legend min/max equal the payload P&L extremes", shown == [expected["low"], expected["high"]], {"shown": shown, **expected})
        check(2, "risk legend marks zero inside the range", zero == 1 if expected["lo"] < 0 < expected["hi"] else zero == 0, zero)

        # 4. Vertical-axis minimum tick is present in the vol surface and order book.
        for section, module in (("options", "vol-surface"), ("markets", "order-book")):
            go(page, section)
            layout = lab(page, module)
            z = [b for b in layout["labels"] if b["axis"] == "z"]
            check(4, f"{module}: vertical-axis minimum tick placed", any(b["at"] == min(t["at"] for t in z) == -1 for b in z) if z else False, z)
            check(4, f"{module}: 3D labels do not overlap", not page.evaluate(OVERLAP, layout["labels"]))

        # 5. Correlation markers do not hide one another.
        go(page, "signals")
        layout = lab(page, "correlation-network")
        nodes = [p for p in layout["points"] if p["declutter"]]
        hidden = [(a["label"], b["label"]) for i, a in enumerate(nodes) for b in nodes[i + 1:]
                  if ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5 < a["r"] + b["r"] - 0.5]
        check(5, "correlation nodes do not overlap on screen", nodes and not hidden, {"nodes": [n["label"] for n in nodes], "overlapping": hidden})
        check(5, "SYN002 is drawn as its own marker", any(n["label"] == "SYN002" for n in nodes))
        covered = [(b["text"], n["label"]) for b in layout["labels"] for n in nodes if b["text"] != n["label"]
                   and b["box"][0] < n["x"] + n["r"] and b["box"][2] > n["x"] - n["r"] and b["box"][1] < n["y"] + n["r"] and b["box"][3] > n["y"] - n["r"]]
        check(5, "no label covers another node's marker", not covered, covered)
        canvas = page.locator("#lab-canvas")
        box = canvas.bounding_box()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] / 2 + 160, box["y"] + box["height"] / 2 + 40, steps=8)
        page.mouse.up()
        layout = page.evaluate("GIGModels.engine().layout()")
        nodes = [p for p in layout["points"] if p["declutter"]]
        hidden = [(a["label"], b["label"]) for i, a in enumerate(nodes) for b in nodes[i + 1:]
                  if ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5 < a["r"] + b["r"] - 0.5]
        check(5, "correlation nodes stay separated after rotation", not hidden, hidden)

        # 6 and 7. Globe orientation on selection; flat map fills the canvas and wraps.
        go(page, "global-intelligence")
        page.wait_for_function("GIGIntel.state().events !== null", timeout=30000)
        if not page.evaluate("GIGIntel.events().some(e => e.latitude !== null)"):
            with page.expect_response(lambda r: r.url.endswith("/api/v1/intel/demonstration")):
                page.locator("#intel-demo").click()
            page.wait_for_function("GIGIntel.events().some(e => e.latitude !== null)", timeout=30000)
        page.evaluate("document.getElementById('intel-map').scrollIntoView({block: 'center'})")
        sample = page.evaluate("GIGIntel.events()")
        geo = [e for e in sample if e["latitude"] is not None and e["longitude"] is not None]
        check(6, "acquired events are available to select", len(geo) > 0, {"visible_events": len(sample), "geolocated": len(geo)})
        if geo:
            far = max(geo, key=lambda e: abs(((e["longitude"] - 40 + 540) % 360) - 180))
            page.evaluate("id => GIGIntel.select(id)", far["event_id"])
            page.wait_for_function("!GIGIntel.state().focusing", timeout=5000)
            frame = page.evaluate("GIGIntel.frame()")
            p = page.evaluate("([lat, lon]) => GIGIntel.project(lat, lon)", [far["latitude"], far["longitude"]])
            dx, dy = p["x"] - frame["width"] / 2, p["y"] - frame["height"] / 2
            limited = abs(far["latitude"]) > 80
            check(6, "selecting an event turns the globe to centre it",
                  p["visible"] and abs(dx) < 2 and (abs(dy) < 2 or limited), {"event": far["title"], "lat": far["latitude"], "lon": far["longitude"], "offset": [round(dx, 1), round(dy, 1)]})
            page.evaluate("GIGIntel.setView('flat')")
            page.wait_for_timeout(150)
            frame = page.evaluate("GIGIntel.frame()")
            check(7, "flat map leaves no empty band above or below", frame["mapTop"] <= 0.5 and frame["mapBottom"] >= frame["height"] - 0.5, frame)
            edges = page.evaluate("""() => { const c = document.getElementById('intel-canvas'); const ctx = c.getContext('2d');
                const y = Math.round(c.height / 2); const sample = (x) => [...ctx.getImageData(x, y, 1, 1).data].slice(0, 3);
                return { left: sample(1), right: sample(c.width - 2) }; }""")
            check(7, "flat map reaches both canvas edges (no letterbox)", edges["left"] != [5, 6, 8] and edges["right"] != [5, 6, 8], edges)
            visible = page.evaluate("""(events) => { const f = GIGIntel.frame();
                const half = f.lonSpan / 2; const centre = GIGIntel.state().flatLon;
                const inside = (c) => events.filter(e => Math.abs(((e.longitude - c + 540) % 360) - 180) <= half).length;
                return { centre, shown: events.filter(e => { const p = GIGIntel.project(e.latitude, e.longitude); return p.x >= 0 && p.x <= f.width; }).length,
                         greenwich: inside(0), total: events.length, span: f.lonSpan }; }""", geo)
            check(7, "default flat-map centre shows at least as many events as a Greenwich centre", visible["shown"] >= visible["greenwich"], visible)
            canvas = page.locator("#intel-canvas")
            box = canvas.bounding_box()
            shift = ((visible["centre"] - 180 + 540) % 360 - 180) * frame["width"] / visible["span"]
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            page.mouse.down()
            page.mouse.move(box["x"] + box["width"] / 2 + shift, box["y"] + box["height"] / 2, steps=12)
            page.mouse.up()
            pair = page.evaluate("""() => { const a = GIGIntel.project(-17, 179.5), b = GIGIntel.project(-17, -179.5), f = GIGIntel.frame();
                return { centre: GIGIntel.state().flatLon, gap: Math.abs(a.x - b.x), mid: (a.x + b.x) / 2, width: f.width }; }""")
            check(7, "dragged to the antimeridian, the Pacific is contiguous", pair["gap"] < 12 and abs(pair["mid"] - pair["width"] / 2) < 20, pair)
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            page.mouse.down()
            page.mouse.move(box["x"] + box["width"] / 2 + 3000, box["y"] + box["height"] / 2 + 900, steps=20)
            page.mouse.up()
            frame = page.evaluate("GIGIntel.frame()")
            check(7, "panning cannot open an empty band", frame["mapTop"] <= 0.5 and frame["mapBottom"] >= frame["height"] - 0.5, frame)
            canvas.focus()
            page.keyboard.press("-")
            page.keyboard.press("-")
            frame = page.evaluate("GIGIntel.frame()")
            check(7, "zooming out cannot open an empty band", frame["mapTop"] <= 0.5 and frame["mapBottom"] >= frame["height"] - 0.5, frame)
            near_edge = min(geo, key=lambda e: abs(page.evaluate("([a, b]) => GIGIntel.project(a, b).x", [e["latitude"], e["longitude"]])))
            page.evaluate("id => GIGIntel.select(id)", near_edge["event_id"])
            page.wait_for_timeout(100)
            p = page.evaluate("([lat, lon]) => GIGIntel.project(lat, lon)", [near_edge["latitude"], near_edge["longitude"]])
            check(6, "selecting an edge event on the flat map brings it into view", 40 <= p["x"] <= frame["width"] - 40 and 0 <= p["y"] <= frame["height"], p)
            page.evaluate("GIGIntel.setView('globe')")
        context.close()

        # 6 under reduced motion: an immediate jump, no animation.
        context, page = start({"width": 1440, "height": 900}, reduced="reduce")
        go(page, "global-intelligence")
        page.wait_for_function("GIGIntel.state().events !== null", timeout=30000)
        sample = page.evaluate("GIGIntel.events()")
        geo = [e for e in sample if e["latitude"] is not None]
        if geo:
            page.evaluate("id => GIGIntel.select(id)", geo[-1]["event_id"])
            state = page.evaluate("GIGIntel.state()")
            check(6, "reduced motion centres the globe without animating", not state["focusing"] and abs(((state["lon0"] - geo[-1]["longitude"] + 540) % 360) - 180) < 0.01, state)
        context.close()

        # 8 and 9 on a phone-width viewport.
        context, page = start({"width": 390, "height": 844})
        fit = page.evaluate("""() => { const i = document.getElementById('replay-asof'); const s = getComputedStyle(i);
            const c = document.createElement('canvas').getContext('2d'); c.font = `${s.fontSize} ${s.fontFamily}`;
            const room = i.clientWidth - parseFloat(s.paddingLeft) - parseFloat(s.paddingRight);
            return { placeholder: i.placeholder, text: c.measureText(i.placeholder).width, room }; }""")
        check(9, "point-in-time placeholder fits the mobile input", fit["text"] <= fit["room"], fit)
        go(page, "risk")
        spans = page.evaluate("""[...document.querySelectorAll('#metrics .metric-card')].map(c => Math.round(c.getBoundingClientRect().width))""")
        check(8, "mobile metric grid leaves no lone half-width card", len(spans) % 2 == 0 or spans[-1] > spans[0] * 1.5 or len(set(spans)) == 1, spans)
        overflow = page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
        check(9, "no horizontal overflow at 390px", overflow)
        context.close()
        browser.close()

    failed = [c for c in checks if not c["passed"]]
    report = {"url": url, "checks": checks, "javascript_errors": errors, "blocked_external_requests": blocked,
              "passed": len(checks) - len(failed), "failed": len(failed)}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    for c in checks:
        print(("PASS" if c["passed"] else "FAIL"), f"[{c['defect']}]", c["name"], "" if c["passed"] else json.dumps(c["detail"])[:400])
    print(f"{report['passed']} passed, {report['failed']} failed, {len(errors)} JS errors, {len(blocked)} blocked requests")
    return 1 if failed or errors or blocked else 0


if __name__ == "__main__":
    sys.exit(main())
