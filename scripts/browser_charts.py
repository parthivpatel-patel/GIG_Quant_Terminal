"""Chromium acceptance for the Market Charts workspace, command palette and collapsible navigation.

With ``--url`` it drives an already running terminal, read-only. Otherwise it
starts an isolated local service with a temporary state directory (as CI
does). In that mode it also admits a small, explicitly labelled SYNTHETIC bar
set through the workspace API, to exercise the admitted-local path. Every
request that is not to the local origin is aborted and recorded.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

from browser_stage_a_regressions import isolated_server
from playwright.sync_api import sync_playwright

CLEAR_ONCE = "if (!sessionStorage.getItem('gig-test-cleared')) { localStorage.clear(); sessionStorage.setItem('gig-test-cleared', '1'); }"
PIXELS = """(id) => { const c = document.getElementById(id); const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  let h = 0, lit = 0; for (let i = 0; i < d.length; i += 4 * 11) { h = (h * 31 + d[i] + d[i + 1] * 3 + d[i + 2] * 7) >>> 0; if (d[i] + d[i + 1] + d[i + 2] > 120) lit++; }
  return { hash: h, lit }; }"""


def admitted_rows():
    """Two labelled SYNTHETIC instruments, 80 daily bars each from 2026-06-01."""
    rows = []
    start = dt.datetime(2026, 6, 1, 20, tzinfo=dt.UTC)
    for name, level, slope in (("ADMITTED.A", 50.0, 0.12), ("ADMITTED.B", 80.0, -0.05)):
        close = level
        for i in range(80):
            stamp = (start + dt.timedelta(days=i)).isoformat()
            open_ = close
            close = round(level + slope * i + 1.5 * ((i * 7919) % 13 - 6) / 6, 2)
            rows.append({"instrument_id": name, "timestamp": stamp, "available_at": stamp, "open": round(open_, 2),
                         "high": round(max(open_, close) + 0.6, 2), "low": round(min(open_, close) - 0.6, 2), "close": close,
                         "volume": 10_000 + (i * 37) % 900})
    return {"rows": rows, "source": "browser-acceptance-fixture", "classification": "SYNTHETIC",
            "corporate_action_policy": "RAW", "currency": "USD"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Running terminal to drive read-only; omitted, an isolated server is started")
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--axe", type=Path, help="Integrity-verified axe-core script (manifest.json alongside)")
    args = parser.parse_args()
    with contextlib.ExitStack() as stack:
        isolated = not args.url
        url = args.url.rstrip("/") if args.url else stack.enter_context(isolated_server(args.evidence_dir))
        return run(url, args.evidence_dir, args.axe, isolated)


def run(url: str, out: Path, axe: Path | None, isolated: bool) -> int:
    shots = out / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)
    checks, errors, blocked, captured, accessibility = [], [], [], [], []
    axe_manifest = None
    if axe:
        axe_manifest = json.loads((axe.parent / "manifest.json").read_text())
        if hashlib.sha256(axe.read_bytes()).hexdigest() != axe_manifest["script_sha256"]:
            raise SystemExit("axe-core script does not match its pinned SHA-256")

    def check(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])

        def route(r):
            if r.request.url.startswith(url + "/"):
                r.continue_()
            else:
                blocked.append(r.request.url)
                r.abort()

        def open_page(viewport, scale=1, storage=None):
            context = browser.new_context(viewport=viewport, device_scale_factor=scale)
            context.route("**/*", route)
            if storage:
                context.add_init_script(storage)
            page = context.new_page()
            page.on("pageerror", lambda e: errors.append(str(e)))
            return context, page

        def shot(page, name, locator=None):
            path = shots / f"{name}.png"
            (page.locator(locator).screenshot(path=str(path)) if locator else page.screenshot(path=str(path)))
            captured.append(path.name)

        def charts_ready(page):
            page.wait_for_function("window.GIGCharts && !['LOADING', ''].includes(document.getElementById('state-badge').textContent)"
                                   " && document.querySelector('#section-nav a[href=\"#charts\"]').getAttribute('aria-current') === 'page'",
                                   timeout=30000)
            page.wait_for_timeout(250)

        def mode(page, value):
            page.locator("#data-mode").select_option(value)
            expected = "SYNTHETIC" if value == "synthetic" else None
            page.wait_for_function("(c) => { const s = GIGCharts.state(); return s.payloadState && (!c || s.classification === c); }",
                                   arg=expected, timeout=30000)
            charts_ready(page)

        pixels = lambda page, canvas="chart-price": page.evaluate(PIXELS, canvas)  # noqa: E731
        cstate = lambda page: page.evaluate("GIGCharts.state()")  # noqa: E731

        context, page = open_page({"width": 1600, "height": 1000}, scale=2, storage=CLEAR_ONCE)
        page.goto(url + "/#charts", wait_until="networkidle")
        charts_ready(page)
        links = page.locator("#section-nav a")
        hrefs = [links.nth(i).get_attribute("href") for i in range(links.count())]
        check("navigation lists fifteen sections ending with Market Charts", len(hrefs) == 15 and hrefs[-1] == "#charts" and "#observatory" in hrefs, hrefs)
        badge = page.locator("#state-badge").inner_text()
        check("local mode without admitted bars is UNAVAILABLE and says why", badge == "UNAVAILABLE"
              and "No admitted bars" in page.locator("#charts-status").inner_text(), badge)
        source = page.locator("#source-label").inner_text()
        check("unavailable chart claims no source or classification", source == "charts · no admitted bars for this selection", source)
        shot(page, "22-charts-local-unavailable", "#charts-panel")

        mode(page, "synthetic")
        st = cstate(page)
        strip = page.locator("#source-label").inner_text()
        check("synthetic demo loads 504 labelled daily bars, last 120 visible", st["bars"] == 504 and st["range"] == [384, 504]
              and st["classification"] == "SYNTHETIC" and "SYNTHETIC" in strip, {"state": st, "strip": strip})
        check("truth notice says fictional, not market observations", "not market observations" in page.locator("#truth-notice").inner_text())
        payload = page.evaluate("fetch('/api/v1/charts/bars?mode=synthetic&symbol=SYN001&timeframe=1D').then(r => r.json())")
        bars = payload["data"]["bars"]
        canvas = page.locator("#chart-price")
        box = canvas.bounding_box()
        plot_w = box["width"] - 64
        target = 384 + 60
        x = box["x"] + (60 + 0.5) * plot_w / 120
        page.mouse.move(x, box["y"] + box["height"] * 0.4)
        page.wait_for_timeout(100)
        readout = page.locator("#chart-readout").inner_text()
        st = cstate(page)
        expected_close = f"{bars['close'][target]:,.2f}"
        check("crosshair reads the bar under the pointer from the payload", st["cursor"] == target and bars["t"][target][:10] in readout
              and expected_close in readout, {"cursor": st["cursor"], "readout": readout, "expected_close": expected_close})
        rsi = payload["data"]["indicators"]["rsi14"][target]
        check("readout shows the backend RSI value at the cursor", f"{rsi:,.2f}" in readout, {"rsi": rsi, "readout": readout})
        shot(page, "20-charts-candles-synthetic", "#charts-panel")

        hashes = {}
        for kind in ("candles", "ohlc", "line", "area"):
            page.locator(f'[data-chart-type="{kind}"]').click()
            page.wait_for_timeout(80)
            hashes[kind] = pixels(page)["hash"]
        check("candles, OHLC, line and area render differently", len(set(hashes.values())) == 4, hashes)
        page.locator('[data-chart-type="candles"]').click()
        before = pixels(page)["hash"]
        page.locator('[data-chart-overlay="bb"]').check()
        page.locator('[data-chart-overlay="vwap20"]').check()
        page.wait_for_timeout(80)
        check("Bollinger and VWAP overlays draw", pixels(page)["hash"] != before and "BB up" in page.locator("#chart-readout").inner_text())
        lower = {}
        for value in ("macd", "atr14", "rv20", "rsi14"):
            page.locator("#chart-lower-select").select_option(value)
            page.wait_for_timeout(80)
            lower[value] = pixels(page, "chart-lower")["hash"]
        check("RSI, MACD, ATR and realized volatility panes render", len(set(lower.values())) == 4, lower)

        page.mouse.move(box["x"] + plot_w * 0.5, box["y"] + box["height"] * 0.4)
        page.mouse.wheel(0, -500)
        page.wait_for_timeout(120)
        zoomed = cstate(page)["range"]
        check("wheel zoom narrows the visible range around the pointer", zoomed[1] - zoomed[0] < 120 and zoomed[0] > 384, zoomed)
        page.mouse.move(box["x"] + plot_w * 0.5, box["y"] + 120)
        page.mouse.down()
        page.mouse.move(box["x"] + plot_w * 0.5 + 300, box["y"] + 120, steps=10)
        page.mouse.up()
        panned = cstate(page)["range"]
        check("dragging right pans to earlier bars", panned[0] < zoomed[0] and panned[1] - panned[0] == zoomed[1] - zoomed[0], {"zoomed": zoomed, "panned": panned})
        canvas.focus()
        page.keyboard.press("0")
        page.keyboard.press("ArrowLeft")
        page.keyboard.press("ArrowLeft")
        st = cstate(page)
        check("keyboard: 0 resets, arrows move the crosshair", st["range"] == [384, 504] and st["cursor"] == 501, st)
        check("keyboard crosshair is announced", bars["t"][501][:10] in page.locator("#charts-live").inner_text())

        page.locator("#chart-lower-select").select_option("macd")
        page.locator("#chart-compare").select_option("SYN005")
        page.wait_for_function("GIGCharts.state().compare === 'SYN005'", timeout=15000)
        page.wait_for_timeout(150)
        compare_box = page.locator("#chart-compare-canvas").bounding_box()
        page.mouse.move(compare_box["x"] + (30 + 0.5) * (compare_box["width"] - 64) / 120, compare_box["y"] + 60)
        page.wait_for_timeout(80)
        st = cstate(page)
        readout = page.locator("#chart-readout").inner_text()
        check("comparison pane shares the time axis and crosshair", st["cursor"] == 384 + 30 and "SYN005" in readout
              and page.locator("#chart-compare-wrap").is_visible(), {"cursor": st["cursor"], "readout": readout})
        shot(page, "21-charts-compare-macd", "#charts-panel")

        splitter = page.locator("#chart-splitter").bounding_box()
        page.mouse.move(splitter["x"] + 200, splitter["y"] + 4)
        page.mouse.down()
        page.mouse.move(splitter["x"] + 200, splitter["y"] - 60, steps=6)
        page.mouse.up()
        grown = cstate(page)["lowerHeight"]
        page.locator("#chart-splitter").focus()
        page.keyboard.press("ArrowDown")
        check("indicator pane resizes by drag and keyboard", grown == 214 and cstate(page)["lowerHeight"] == 198, grown)

        page.locator('[data-chart-type="area"]').click()
        page.locator("#chart-layout-name").fill("Momentum review")
        page.locator("#chart-layout-save").click()
        page.locator('[data-chart-type="line"]').click()
        page.locator('[data-chart-timeframe="1W"]').click()
        page.wait_for_function("GIGCharts.state().timeframe === '1W' && GIGCharts.state().bars === 102", timeout=15000)
        check("weekly timeframe aggregates to 102 bars", cstate(page)["bars"] == 102)
        page.locator("#chart-layouts").select_option("Momentum review")
        page.wait_for_function("GIGCharts.state().timeframe === '1D' && GIGCharts.state().bars === 504", timeout=15000)
        st = cstate(page)
        check("saved layout restores type, timeframe, indicators and comparison", st["type"] == "area" and st["lower"] == "macd"
              and st["compare"] == "SYN005" and "bb" in st["overlays"] and st["lowerHeight"] == 198, st)
        page.reload(wait_until="networkidle")
        charts_ready(page)
        mode(page, "synthetic")
        check("layouts persist across reloads", "Momentum review" in page.locator("#chart-layouts").inner_text() and cstate(page)["type"] == "area")

        page.keyboard.press("Control+k")
        page.wait_for_selector("#palette:not(.hidden)")
        page.wait_for_function("GIGPalette.results().some(r => r.kind === 'Instrument')", timeout=10000)
        page.keyboard.type("syn007")
        page.wait_for_timeout(100)
        results = page.evaluate("GIGPalette.results()")
        check("palette searches instruments", results and results[0] == {"kind": "Instrument", "label": "SYN007"}, results[:3])
        shot(page, "24-command-palette")
        page.keyboard.press("Enter")
        page.wait_for_function("GIGCharts.state().symbol === 'SYN007'", timeout=15000)
        check("choosing an instrument opens it in the chart and closes the palette", page.locator("#palette").is_hidden())
        page.keyboard.press("Meta+k")
        page.keyboard.type("risk")
        page.keyboard.press("Enter")
        page.wait_for_function("location.hash === '#risk' && document.getElementById('section-title').textContent === 'Risk'", timeout=10000)
        check("palette navigates to sections", page.locator("#section-title").inner_text() == "Risk")
        page.keyboard.press("Control+k")
        page.keyboard.type("place order")
        page.wait_for_timeout(80)
        check("palette offers no order command", not page.evaluate("GIGPalette.results()"))
        page.keyboard.press("Escape")
        check("Escape closes the palette", page.locator("#palette").is_hidden())

        page.goto(url + "/#charts", wait_until="networkidle")
        charts_ready(page)
        page.locator("#nav-toggle").click()
        width = page.evaluate("document.querySelector('.sidebar').getBoundingClientRect().width")
        check("navigation collapses to icons", width == 72 and page.locator("#nav-toggle").get_attribute("aria-expanded") == "false", width)
        shot(page, "25-nav-collapsed-charts")
        page.reload(wait_until="networkidle")
        check("collapsed navigation persists", page.evaluate("document.body.classList.contains('nav-collapsed')"))
        page.locator("#nav-toggle").click()
        check("navigation expands again", page.evaluate("document.querySelector('.sidebar').getBoundingClientRect().width") > 150)
        context.close()

        if isolated:
            context, page = open_page({"width": 1600, "height": 1000}, scale=2, storage=CLEAR_ONCE)
            page.goto(url + "/#charts", wait_until="networkidle")
            charts_ready(page)
            response = page.evaluate("""(body) => fetch('/api/v1/workspace/bars', {method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(body)}).then(r => r.json())""", admitted_rows())
            check("labelled fixture bars admitted through the workspace API", response["state"] == "AVAILABLE", response.get("error"))
            page.locator("#replay-asof").fill("2026-08-20T00:00:00+00:00")
            page.locator("#replay-asof").dispatch_event("change")
            page.locator("#freshness-policy").select_option("604800")
            page.wait_for_function("GIGCharts.state().bars > 0", timeout=15000)
            st = cstate(page)
            strip = page.locator("#source-label").inner_text()
            check("admitted bars chart as known at the cutoff, with their own label", st["bars"] == 80 and st["classification"] == "SYNTHETIC"
                  and "local-admitted:browser-acceptance-fixture" in strip and page.locator("#state-badge").inner_text() == "AVAILABLE", {"state": st, "strip": strip})
            page.locator("#replay-asof").fill("2026-07-10T00:00:00+00:00")
            page.locator("#replay-asof").dispatch_event("change")
            page.wait_for_function("GIGCharts.state().bars === 39", timeout=15000)
            check("an earlier cutoff hides later bars", cstate(page)["bars"] == 39)
            page.locator("#replay-asof").fill("2026-10-30T00:00:00+00:00")
            page.locator("#replay-asof").dispatch_event("change")
            page.wait_for_function("document.getElementById('state-badge').textContent === 'STALE'", timeout=15000)
            check("bars older than the freshness policy are shown as STALE", "STALE" in page.locator("#truth-notice").inner_text())
            page.locator("#replay-asof").fill("2026-08-20T00:00:00+00:00")
            page.locator("#replay-asof").dispatch_event("change")
            page.wait_for_function("document.getElementById('state-badge').textContent === 'AVAILABLE'", timeout=15000)
            shot(page, "23-charts-local-admitted", "#charts-panel")
            context.close()

        context, page = open_page({"width": 390, "height": 844}, scale=2, storage=CLEAR_ONCE)
        page.goto(url + "/#charts", wait_until="networkidle")
        charts_ready(page)
        mode(page, "synthetic")
        check("no horizontal overflow at 390px", page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth"))
        check("price pane renders at 390px", pixels(page)["lit"] > 200)
        page.evaluate("document.getElementById('chart-readout').scrollIntoView({block: 'start'})")
        shot(page, "26-mobile-charts")
        context.close()

        if axe:
            for theme in ("dark", "light"):
                context, page = open_page({"width": 1440, "height": 1000}, storage=CLEAR_ONCE + f"localStorage.setItem('gig-theme', {json.dumps(theme)});")
                page.goto(url + "/#charts", wait_until="networkidle")
                charts_ready(page)
                mode(page, "synthetic")
                page.keyboard.press("Control+k")
                page.wait_for_timeout(200)
                page.add_script_tag(path=str(axe))
                scan = page.evaluate("async()=>await axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa']}})")
                accessibility.append({"section": "charts+palette", "theme": theme, "violations": scan["violations"], "passes": len(scan["passes"])})
                check(f"charts with palette open, {theme}: axe WCAG A/AA automated rules", not scan["violations"],
                      [(v["id"], len(v["nodes"]), v["nodes"][0]["target"]) for v in scan["violations"]])
                context.close()
        browser.close()

    failed = [c for c in checks if not c["passed"]]
    report = {"url": url if not isolated else "isolated-local-server", "checks": checks, "javascript_errors": errors,
              "blocked_external_requests": blocked, "screenshots": captured, "accessibility": accessibility, "axe_asset": axe_manifest,
              "broker_connections": 0, "orders_submitted": 0, "passed": len(checks) - len(failed), "failed": len(failed)}
    (out / "charts-report.json").write_text(json.dumps(report, indent=2) + "\n")
    for c in checks:
        print("PASS" if c["passed"] else "FAIL", c["name"], "" if c["passed"] else json.dumps(c["detail"])[:500])
    print(f"{report['passed']} passed, {report['failed']} failed, {len(errors)} JS errors, {len(blocked)} blocked requests")
    if errors:
        print("JS errors:", errors[:5])
    return 1 if failed or errors or blocked else 0


if __name__ == "__main__":
    sys.exit(main())
