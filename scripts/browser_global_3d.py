"""Actual-browser acceptance for the 3D Analytics Lab and Global Intelligence.

An isolated local service is started from source with a temporary state
directory. The browser may reach only that origin; every other request is
aborted and counted. With ``--provider-refresh`` the *server* makes the explicit
registry requests to USGS and NASA EONET and the outcome is recorded as found;
without it no provider is contacted. Screenshots are real Chromium captures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

from browser_terminal_check import ROOT, wait_ready
from playwright.sync_api import sync_playwright

SECTIONS = ("command-center", "markets", "signals", "portfolio", "risk", "models", "news",
            "stat-arb", "options", "execution", "research", "operations", "global-intelligence")
MODULES = {"options": "vol-surface", "markets": "order-book", "portfolio": "risk-terrain",
           "signals": "correlation-network", "stat-arb": "stat-arb", "models": "model-evaluation"}
PIXELS = """(id) => { const c = document.getElementById(id); const ctx = c.getContext('2d');
  const d = ctx.getImageData(0, 0, c.width, c.height).data; let h = 0, lit = 0;
  for (let i = 0; i < d.length; i += 4 * 13) { const v = d[i] + d[i + 1] * 3 + d[i + 2] * 7; h = (h * 31 + v) >>> 0; if (Math.abs(d[i] - 5) + Math.abs(d[i + 1] - 6) + Math.abs(d[i + 2] - 8) > 30) lit++; }
  return { hash: h, lit, width: c.width, height: c.height }; }"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--screenshots", type=Path, default=None, help="Directory for the PNG views")
    parser.add_argument("--provider-refresh", action="store_true", help="Let the server request USGS and NASA EONET once")
    parser.add_argument("--axe", type=Path, help="Integrity-verified axe-core script (manifest.json alongside)")
    args = parser.parse_args()
    axe_manifest = None
    if args.axe:
        axe_manifest = json.loads((args.axe.parent / "manifest.json").read_text())
        if hashlib.sha256(args.axe.read_bytes()).hexdigest() != axe_manifest["script_sha256"]:
            raise SystemExit("axe-core script does not match its pinned SHA-256")
    out = args.evidence_dir.absolute()
    out.mkdir(parents=True, exist_ok=True)
    shots = (args.screenshots or out / "screenshots").absolute()
    shots.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []
    errors: list[str] = []
    blocked: list[str] = []
    captured: list[str] = []
    report = {"runner": "real-playwright-chromium", "checks": checks, "javascript_errors": errors,
              "blocked_external_requests": blocked, "screenshots": captured, "broker_connections": 0,
              "browser_provider_connections": 0, "server_provider_refresh": args.provider_refresh,
              "public_deployment": False, "formal_accessibility_certification": False,
              "axe_asset": axe_manifest, "accessibility": []}

    def check(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    with tempfile.TemporaryDirectory(prefix="gig-global-3d-browser-") as temporary:
        directory = Path(temporary)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        env = {k: v for k, v in os.environ.items() if not any(x in k for x in ("ALPACA", "FINNHUB", "FRED_API", "FINBERT_MODEL_DIR"))}
        env.update(PYTHONPATH=str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", ""),
                   GIG_BROKER_CONNECTIVITY_ENABLED="false", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   XDG_CACHE_HOME=str(directory / "browser-cache"), GSETTINGS_BACKEND="memory")
        log = (out / "server.txt").open("w")
        code = ("import uvicorn;from pathlib import Path;from gig.service.app import create_app;"
                f"uvicorn.run(create_app(state_dir=Path({str(directory / 'state')!r})),host='127.0.0.1',port={port},log_level='warning')")
        server = subprocess.Popen([sys.executable, "-c", code], env=env, cwd=directory, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_ready(url, server)
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True, timeout=30000, args=["--no-sandbox", "--enable-precise-memory-info"], env=env)
                report["browser_version"] = browser.version

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
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    return context, page

                def shot(page, name, locator=None, full=False):
                    path = shots / f"{name}.png"
                    (page.locator(locator).screenshot(path=str(path)) if locator else page.screenshot(path=str(path), full_page=full))
                    captured.append(path.name)

                def pixels(page, canvas="lab-canvas"):
                    return page.evaluate(PIXELS, canvas)

                def lab_ready(page, module):
                    page.wait_for_function(f"document.getElementById('lab-tab-{module}')?.getAttribute('aria-selected') === 'true'"
                                           " && document.getElementById('lab-status').classList.contains('hidden')", timeout=20000)
                    page.wait_for_timeout(250)

                context, page = open_page({"width": 1440, "height": 1000})
                page.goto(url + "/#command-center", wait_until="networkidle")
                page.wait_for_function("!document.getElementById('refresh').disabled")
                links = page.locator("#section-nav a")
                check("fourteen sections: the twelve originals, Global Intelligence and Market Charts",
                      links.count() == 14 and [links.nth(i).get_attribute("href")[1:] for i in range(13)] == list(SECTIONS)
                      and links.nth(13).get_attribute("href") == "#charts")
                lab_ready(page, "correlation-network")
                shot(page, "01-command-center")
                shot(page, "02-navigation", "aside.sidebar")
                background = page.evaluate("getComputedStyle(document.body).backgroundColor")
                check("deep-black page background #050608", background == "rgb(5, 6, 8)", background)

                for section, module in MODULES.items():
                    page.locator(f'#section-nav a[href="#{section}"]').click()
                    lab_ready(page, module)
                    base = pixels(page)
                    rows = page.locator("#lab-table tbody tr").count()
                    check(f"{module}: renders non-blank canvas from AVAILABLE synthetic envelope",
                          base["lit"] > 200 and "SYNTHETIC" in page.locator("#lab-provenance").inner_text(), base)
                    check(f"{module}: exact-value table populated", rows > 0, rows)
                    canvas = page.locator("#lab-canvas")
                    page.evaluate("document.getElementById('lab-canvas').scrollIntoView({block: 'center'})")
                    page.wait_for_timeout(100)
                    base = pixels(page)
                    box = canvas.bounding_box()
                    page.mouse.move(box["x"] + box["width"] * .5, box["y"] + box["height"] * .5)
                    page.mouse.down()
                    page.mouse.move(box["x"] + box["width"] * .5 + 120, box["y"] + box["height"] * .5 + 30, steps=6)
                    page.mouse.up()
                    rotated = pixels(page)
                    check(f"{module}: drag rotates the camera", rotated["hash"] != base["hash"])
                    canvas.focus()
                    page.keyboard.press("0")
                    page.mouse.wheel(0, -400)
                    page.wait_for_timeout(100)
                    check(f"{module}: wheel zoom changes the view", pixels(page)["hash"] != rotated["hash"])
                    page.keyboard.press("0")
                    page.keyboard.press("n")
                    detail = page.locator("#lab-details").inner_text()
                    check(f"{module}: keyboard N selects an item and fills details", "Click or press" not in detail, detail[:120])
                    check(f"{module}: selection announced to assistive technology", page.locator("#lab-live").inner_text().startswith("Selected"))
                    page.keyboard.press("Escape")
                    page.locator("#lab-view-2d").click()
                    flat = pixels(page)
                    check(f"{module}: 2D alternative renders", flat["hash"] != base["hash"] and flat["lit"] > 100)
                    page.locator("#lab-view-3d").click()
                    if section == "options":
                        before = pixels(page)["hash"]
                        page.locator("#lab-filters select").nth(0).select_option("prior")
                        check("vol-surface: snapshot filter re-renders", pixels(page)["hash"] != before)
                    if section == "markets":
                        page.locator("#lab-filters select").nth(0).select_option("bids")
                        check("order-book: side filter keeps only bids", page.locator("#lab-table tbody tr").count() < rows
                              and "ASK" not in page.locator("#lab-table tbody").inner_text())
                        page.locator("#lab-filters select").nth(0).select_option("both")
                    if section == "signals":
                        slider = page.locator("#lab-filters input[type=range]")
                        before = pixels(page)["hash"]
                        slider.fill("0.6")
                        check("correlation-network: |rho| threshold filter re-renders", pixels(page)["hash"] != before)
                    page.keyboard.press("0")
                    page.wait_for_timeout(100)
                    names = {"options": "04-options-vol-surface", "markets": "05-order-book-landscape", "portfolio": "06-portfolio-risk-terrain",
                             "signals": "07-correlation-network", "stat-arb": "08-stat-arb-landscape", "models": "09-model-evaluation"}
                    shot(page, names[section], "#lab-panel")
                    if section == "markets":
                        shot(page, "03-markets")
                    if section == "portfolio":
                        shot(page, "16-portfolio-and-risk")

                page.locator('#section-nav a[href="#options"]').click()
                lab_ready(page, "vol-surface")
                page.locator("#lab-orbit").click()
                orbiting = page.evaluate("window.GIGModels.engine().isOrbiting()")
                first = pixels(page)["hash"]
                page.wait_for_timeout(400)
                check("orbit animation runs when requested", orbiting and pixels(page)["hash"] != first)
                page.locator("#lab-orbit").click()
                paused = pixels(page)["hash"]
                page.wait_for_timeout(300)
                check("pause stops the animation", not page.evaluate("window.GIGModels.engine().isOrbiting()") and pixels(page)["hash"] == paused)
                page.locator("#lab-mode").select_option("local")
                page.wait_for_function("document.getElementById('lab-status').textContent.startsWith('UNAVAILABLE')")
                check("local mode without admitted evidence shows UNAVAILABLE and renders nothing",
                      page.locator("#lab-table tbody tr").count() == 0 and page.locator("#lab-provenance").inner_text() == "No data rendered")
                page.locator("#lab-mode").select_option("synthetic")
                lab_ready(page, "vol-surface")

                heap0 = page.evaluate("performance.memory ? performance.memory.usedJSHeapSize : null")
                for _ in range(4):
                    for module in ("vol-surface", "order-book", "risk-terrain", "correlation-network", "stat-arb", "model-evaluation"):
                        page.locator(f"#lab-tab-{module}").click()
                        lab_ready(page, module)
                page.evaluate("window.gc && window.gc()")
                heap1 = page.evaluate("performance.memory ? performance.memory.usedJSHeapSize : null")
                chart_canvases = page.locator("#charts-panel canvas").count()
                check("24 module switches reuse one canvas and keep heap bounded",
                      page.locator("canvas").count() - chart_canvases == 3 and chart_canvases == 3
                      and (heap0 is None or heap1 - heap0 < 40_000_000),
                      {"heap_before": heap0, "heap_after": heap1, "chart_canvases": chart_canvases})
                page.set_viewport_size({"width": 1000, "height": 900})
                page.wait_for_timeout(300)
                resized = pixels(page)
                check("canvas follows browser resize", resized["width"] < 1000, resized)
                page.set_viewport_size({"width": 1440, "height": 1000})
                page.locator('#section-nav a[href="#news"]').click()
                page.wait_for_function("document.getElementById('section-title').textContent === 'NLP / News'")
                check("sections without a module hide the lab", page.locator("#lab-panel").is_hidden(), page.evaluate("[location.hash, document.getElementById('lab-panel').className]"))

                # Global Intelligence
                page.locator('#section-nav a[href="#global-intelligence"]').click()
                page.wait_for_function("document.getElementById('intel-asof').textContent.includes('window')")
                page.wait_for_function("document.getElementById('intel-basemap').textContent.includes('Natural Earth')")
                check("standard terminal panels hidden in Global Intelligence",
                      page.locator("#workbench").is_hidden() and page.locator("#lab-panel").is_hidden() and page.locator("#intel-panel").is_visible())
                check("integrated sources start UNAVAILABLE before any acquisition",
                      all(t == "UNAVAILABLE" for t in page.locator("#intel-health .intel-badge").all_inner_texts()))
                refresh = {}
                if args.provider_refresh:
                    for source, button in (("USGS_EARTHQUAKES", "#intel-refresh-usgs"), ("NASA_EONET", "#intel-refresh-eonet")):
                        with page.expect_response(lambda r: r.url.endswith("/api/v1/intel/refresh"), timeout=60000) as exchange:
                            page.locator(button).click()
                        refresh[source] = exchange.value.json().get("data")
                        page.wait_for_function("document.getElementById('intel-status').textContent === ''")
                    report["provider_refresh"] = refresh
                    acquired = [k for k, v in refresh.items() if v and v.get("state") == "ACQUIRED"]
                    semantics = page.locator("#intel-semantics").inner_text()
                    check("acquired providers are labeled DELAYED, never LIVE",
                          all(f"{k.replace('_', ' ')}\nDELAYED" in semantics for k in acquired) and "LIVE" not in semantics.replace("DELIVER", ""), semantics)
                    again = page.evaluate("fetch('/api/v1/intel/refresh',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source_id:'USGS_EARTHQUAKES'})}).then(r=>r.json())")
                    check("immediate repeat request is deferred by the refresh interval",
                          not acquired or again["data"]["state"] == "DEFERRED", again["data"].get("state"))
                with page.expect_response(lambda r: r.url.endswith("/api/v1/intel/demonstration")):
                    page.locator("#intel-demo").click()
                page.wait_for_function("document.getElementById('intel-semantics').textContent.includes('DEMONSTRATION')")
                with page.expect_response(lambda r: r.url.endswith("/api/v1/intel/admit")) as exchange:
                    page.locator("#intel-admit-run").click()
                check("authored synthetic defense records admitted", exchange.value.status == 200 and exchange.value.json()["data"]["admitted"] == 4)
                page.wait_for_function("document.getElementById('intel-semantics').textContent.includes('ADMITTED REPORTS')")
                semantics = page.locator("#intel-semantics").inner_text()
                check("admitted synthetic records labeled DEMONSTRATION, not HISTORICAL", "ADMITTED REPORTS\nDEMONSTRATION" in semantics, semantics)
                check("no LIVE semantic badge is rendered anywhere", page.locator(".intel-badge.live").count() == 0)
                unsafe = page.evaluate("""() => {
                    const record = JSON.parse(document.getElementById('intel-admit-input').value).records[0];
                    const post = (r) => fetch('/api/v1/intel/admit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({records:[r]})}).then(x=>x.status);
                    const now = new Date().toISOString();
                    return post({...record, record_id:'x1', callsign:'TEST'}).then((a) => post({...record, record_id:'x2', published_at: now, acquired_at: now}).then((b) => [a, b])); }""")
                check("tracking fields and sub-24-hour records rejected", unsafe == [422, 422], unsafe)
                for layer in ("maritime", "aviation", "imagery"):
                    check(f"{layer} layer disabled with an explicit blocker",
                          page.locator(f'#intel-layers input[value="{layer}"]').is_disabled())
                feed_all = int(page.locator("#intel-feed-count").inner_text())
                page.locator('#intel-layers input[value="hazards"]').uncheck()
                feed_less = int(page.locator("#intel-feed-count").inner_text())
                check("layer filter removes hazard events", feed_less < feed_all, [feed_all, feed_less])
                page.locator('#intel-layers input[value="hazards"]').check()
                page.locator("#intel-search").fill("demonstration: cyclone")
                page.wait_for_function("document.getElementById('intel-feed-count').textContent === '1'")
                check("title search filters server-side", True)
                page.locator("#intel-search").fill("")
                page.wait_for_function(f"Number(document.getElementById('intel-feed-count').textContent) === {feed_all}")
                globe = pixels(page, "intel-canvas")
                check("globe renders basemap and events", globe["lit"] > 300, globe)
                canvas = page.locator("#intel-canvas")
                page.evaluate("document.getElementById('intel-canvas').scrollIntoView({block: 'center'})")
                canvas.focus()
                for _ in range(6):
                    page.keyboard.press("n")
                    if page.locator("#intel-details").inner_text().find("Authored demonstration: cyclone") >= 0:
                        break
                details = page.locator("#intel-details").inner_text()
                check("event details show source, observation, publication, acquisition, delay, uncertainty, verification",
                      all(k in details for k in ("Source", "Observation time", "Publication time", "Acquired by GIG", "Reporting delay", "Geographic uncertainty"))
                      and ("DEMONSTRATION" in details or "PROVIDER" in details.upper()), details[:200])
                check("research links stated as hypotheses, not causal", "not causal" in details.lower())
                shot(page, "10-global-intelligence-globe", ".intel-grid")
                page.locator("#intel-view-flat").click()
                flat = pixels(page, "intel-canvas")
                check("flat map projection renders", flat["hash"] != globe["hash"] and flat["lit"] > 300)
                box = canvas.bounding_box()
                page.mouse.move(box["x"] + 300, box["y"] + 200)
                page.mouse.down()
                page.mouse.move(box["x"] + 380, box["y"] + 230, steps=4)
                page.mouse.up()
                check("flat map pans on drag", pixels(page, "intel-canvas")["hash"] != flat["hash"])
                page.locator("#intel-reset").click()
                shot(page, "11-global-intelligence-flat-map", ".intel-grid")
                page.locator("#intel-view-globe").click()
                for layer, name in (("maritime", "12-maritime-intelligence"), ("aviation", "13-civil-aviation-intelligence"), ("imagery", "14-satellite-imagery")):
                    page.locator(f'.intel-why[data-layer="{layer}"]').click()
                    text = page.locator("#intel-details").inner_text()
                    check(f"{layer} view states blocker and alternatives without positions",
                          "UNAVAILABLE" in text and "AVAILABLE INSTEAD" in text.upper() and "Blocker" in text, text[:160])
                    shot(page, name, ".intel-grid")
                page.locator("#intel-defense-tabs [data-branch=NAVY]").click()
                navy = page.locator("#intel-defense-records").inner_text()
                check("Navy research shows the admitted record with verification and synthetic label",
                      "ANALYTICAL SCENARIO" in navy and "AUTHORED SYNTHETIC" in navy and "±111 km" in navy, navy[:200])
                for branch in ("ARMY", "AIR_FORCE"):
                    page.locator(f"#intel-defense-tabs [data-branch={branch}]").click()
                    check(f"{branch} research category populated", page.locator("#intel-defense-records .intel-record").count() == 1)
                shot(page, "19-geopolitical-defense-workspace", ".intel-defense")
                with page.expect_response(lambda r: r.url.endswith("/api/v1/intel/event-study")) as exchange:
                    page.locator("#intel-study-run").click()
                study = exchange.value.json()["data"]
                check("event study estimates strictly before the event window",
                      study["uses_post_event_data_for_estimation"] is False and study["interpretation"] == "DESCRIPTIVE_SINGLE_EVENT_NOT_CAUSAL")
                with page.expect_response(lambda r: r.url.endswith("/api/v1/intel/exposure")) as exchange:
                    page.locator("#intel-exposure-run").click()
                exposure = exchange.value.json()["data"]
                check("scenario exposure labeled ANALYTICAL_SCENARIO", exposure["scenario"]["classification"] == "ANALYTICAL_SCENARIO")
                page.wait_for_function("document.getElementById('intel-exposure-output').textContent.includes('Scenario P&L')")
                shot(page, "15-financial-impact-research", ".intel-columns >> nth=1")
                shot(page, "20-geopolitical-market-risk-monitor", ".intel-columns >> nth=0")
                shot(page, "21-source-coverage-and-permissions", "#intel-coverage")
                page.locator("#intel-inv-name").fill("Hormuz shipping attention")
                page.locator("#intel-inv-save").click()
                page.wait_for_function("document.getElementById('intel-inv-list').textContent.includes('Hormuz shipping attention')")
                check("investigation saved and listed", True)
                count_now = int(page.locator("#intel-feed-count").inner_text())
                page.locator("#intel-timeline").fill("0")
                page.locator("#intel-timeline").dispatch_event("change")
                page.wait_for_function("document.getElementById('intel-asof').textContent.startsWith('Replay cutoff')")
                replay = page.locator("#intel-semantics").inner_text()
                count_then = int(page.locator("#intel-feed-count").inner_text())
                check("point-in-time replay excludes records acquired after the cutoff", count_then < count_now, [count_now, count_then])
                check("replayed integrated sources are HISTORICAL or UNAVAILABLE", all(s in ("HISTORICAL", "UNAVAILABLE") for s in
                      page.locator("#intel-health .intel-badge").all_inner_texts()), replay)
                page.locator("#intel-now").click()
                page.wait_for_function("document.getElementById('intel-asof').textContent.startsWith('As of')")
                context.close()

                context, page = open_page({"width": 1440, "height": 1000}, reduced="reduce")
                page.goto(url + "/#options", wait_until="networkidle")
                lab_ready(page, "vol-surface")
                page.locator("#lab-orbit").click()
                check("reduced motion: orbit request is refused", not page.evaluate("window.GIGModels.engine().isOrbiting()"))
                context.close()

                for name, viewport in (("17-mobile-layout", {"width": 390, "height": 844}), ("18-tablet-layout", {"width": 820, "height": 1180})):
                    context, page = open_page(viewport)
                    page.goto(url + "/#options", wait_until="networkidle")
                    lab_ready(page, "vol-surface")
                    overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
                    check(f"{name}: no horizontal page overflow", overflow <= 1, overflow)
                    page.locator("#lab-panel").scroll_into_view_if_needed()
                    shot(page, name)
                    page.goto(url + "/#global-intelligence", wait_until="networkidle")
                    page.wait_for_function("document.getElementById('intel-asof').textContent.includes('window')")
                    page.wait_for_timeout(300)
                    overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
                    check(f"{name}: Global Intelligence fits without horizontal overflow", overflow <= 1, overflow)
                    shot(page, name + "-global-intelligence")
                    context.close()

                # The renderer draws whatever the API returns: a modified payload
                # must change both the canvas and the exact-value table.
                context, page = open_page({"width": 1440, "height": 1000})
                page.goto(url + "/#options", wait_until="networkidle")
                lab_ready(page, "vol-surface")
                original = pixels(page)
                context.close()
                context, page = open_page({"width": 1440, "height": 1000})

                def modified(r):
                    response = r.fetch()
                    body = response.json()
                    for point in body["data"]["snapshots"]["current"]:
                        if point["implied_vol"] is not None:
                            point["implied_vol"] *= 1.5
                    body["data"]["snapshots"]["current"][0]["implied_vol"] = 0.987654
                    r.fulfill(response=response, body=json.dumps(body))
                page.route("**/api/v1/viz/vol-surface*", modified)
                page.goto(url + "/#options", wait_until="networkidle")
                lab_ready(page, "vol-surface")
                changed = pixels(page)
                legend = page.locator("#lab-legend").inner_text()
                page.locator("#lab-table-toggle").click()
                try:
                    page.wait_for_function("document.querySelector('#lab-table tbody')?.textContent.includes('0.98765')", timeout=5000)
                except Exception:  # noqa: BLE001 - the check below records the failure with the table text
                    pass
                table = page.locator("#lab-table tbody").inner_text()
                check("renderer follows its payload: modified implied vols change the canvas, legend and table",
                      changed["hash"] != original["hash"] and "0.988" in legend and "0.98765" in table,
                      {"legend": legend, "table": table[:160], "original": original["hash"], "modified": changed["hash"]})
                context.close()

                if args.axe:
                    for section, ready in (("options", "document.getElementById('lab-status').classList.contains('hidden')"),
                                           ("global-intelligence", "document.getElementById('intel-asof').textContent.includes('window')")):
                        for theme in ("dark", "light"):
                            context, page = open_page({"width": 1440, "height": 1000})
                            page.add_init_script(f"localStorage.setItem('gig-theme', {json.dumps(theme)})")
                            page.goto(url + "/#" + section, wait_until="networkidle")
                            page.wait_for_function(ready, timeout=20000)
                            page.wait_for_timeout(300)
                            page.add_script_tag(path=str(args.axe))
                            scan = page.evaluate("async()=>await axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa']}})")
                            report["accessibility"].append({"section": section, "theme": theme, "version": scan["testEngine"]["version"],
                                                            "violations": scan["violations"], "incomplete": len(scan["incomplete"]),
                                                            "passes": len(scan["passes"])})
                            check(f"{section} {theme}: axe WCAG A/AA automated rules", not scan["violations"],
                                  [(v["id"], len(v["nodes"])) for v in scan["violations"]])
                            context.close()

                context, page = open_page({"width": 1440, "height": 1000})
                for section in SECTIONS:
                    page.goto(url + "/#" + section, wait_until="networkidle")
                    page.wait_for_timeout(200)
                    check(f"{section} renders without JavaScript errors", not errors and page.locator("#section-title").inner_text() != "")
                context.close()
                browser.close()
        finally:
            server.terminate()
            server.wait(timeout=10)
            log.close()
    check("no JavaScript exceptions", not errors, errors)
    check("no browser requests outside the local origin", not blocked, blocked)
    report["passed"] = sum(c["passed"] for c in checks)
    report["failed"] = sum(not c["passed"] for c in checks)
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"{report['passed']} passed, {report['failed']} failed")
    for c in checks:
        if not c["passed"]:
            print("FAIL", c["name"], c["detail"])
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
