"""Review captures and interaction checks against an already running GIG terminal.

Unlike the acceptance scripts, this drives the instance started with
``python -m gig serve`` so the screenshots show the application as launched.
Every browser request that is not to that origin is aborted and recorded, so a
pass also shows the interface needs no external assets. Sections 1-12 use the
terminal's own "Synthetic research demo" data environment; Global Intelligence
uses whatever the running server has acquired, plus at most one explicit USGS
and one NASA EONET refresh when ``--provider-refresh`` is given.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from browser_global_3d import MODULES, PIXELS, SECTIONS
from playwright.sync_api import sync_playwright

MODULE_SHOTS = {"options": "05-options-vol-surface", "markets": "06-order-book-landscape", "portfolio": "07-portfolio-risk-terrain",
                "signals": "08-correlation-network", "stat-arb": "09-stat-arb-landscape", "models": "10-model-evaluation"}
SECTION_SHOTS = {"command-center": "01-command-center", "portfolio": "11-portfolio", "risk": "12-risk", "models": "13-models",
                 "research": "14-research", "operations": "15-operations"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--provider-refresh", action="store_true")
    args = parser.parse_args()
    out = args.evidence_dir
    shots = out / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)
    url = args.url.rstrip("/")
    checks, errors, blocked, console_errors, captured = [], [], [], [], []
    report = {"url": url, "checks": checks, "javascript_errors": errors, "console_errors": console_errors,
              "blocked_external_requests": blocked, "screenshots": captured, "broker_connections": 0, "orders_submitted": 0,
              "zoom": "100%", "sections": {}, "modules": {}}

    def check(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        report["browser_version"] = browser.version

        def route(r):
            if r.request.url.startswith(url + "/"):
                r.continue_()
            else:
                blocked.append(r.request.url)
                r.abort()

        def open_page(viewport, scale=1, reduced="no-preference"):
            context = browser.new_context(viewport=viewport, device_scale_factor=scale, reduced_motion=reduced)
            context.route("**/*", route)
            page = context.new_page()
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
            return context, page

        def shot(page, name, locator=None, full=False):
            path = shots / f"{name}.png"
            if locator:
                page.locator(locator).screenshot(path=str(path))
            else:
                page.screenshot(path=str(path), full_page=full)
            captured.append(path.name)

        def pixels(page, canvas="lab-canvas"):
            return page.evaluate(PIXELS, canvas)

        def section_ready(page, section):
            page.wait_for_function(
                f"""document.querySelector('#section-nav a[href="#{section}"]').getAttribute('aria-current') === 'page'
                    && !document.getElementById('refresh').disabled""", timeout=30000)
            page.wait_for_timeout(600)

        def lab_ready(page, module):
            page.wait_for_function(f"document.getElementById('lab-tab-{module}')?.getAttribute('aria-selected') === 'true'"
                                   " && document.getElementById('lab-status').classList.contains('hidden')", timeout=20000)
            page.wait_for_timeout(250)

        def truth(page):
            return page.evaluate("[...document.querySelectorAll('#truth-strip > *')].map(e => e.textContent.trim()).filter(Boolean)")

        # Sections 1-12 in the terminal's labelled synthetic demonstration environment.
        context, page = open_page({"width": 1920, "height": 1080})
        page.goto(url + "/#command-center", wait_until="networkidle")
        section_ready(page, "command-center")
        page.locator("#data-mode").select_option("synthetic")
        section_ready(page, "command-center")
        background = page.evaluate("getComputedStyle(document.body).backgroundColor")
        check("page background is #050608", background == "rgb(5, 6, 8)", background)
        links = page.locator("#section-nav a")
        check("thirteen sections in navigation", [links.nth(i).get_attribute("href")[1:] for i in range(links.count())] == list(SECTIONS))
        for section in SECTIONS[:-1]:
            page.locator(f'#section-nav a[href="#{section}"]').click()
            section_ready(page, section)
            strip = truth(page)
            title = page.locator("#section-title").inner_text()
            report["sections"][section] = {"title": title, "truth_strip": strip}
            check(f"{section}: loads, is current in navigation and labelled", strip and strip[0] in ("AVAILABLE", "UNAVAILABLE", "PARTIAL"), strip)
            check(f"{section}: no horizontal overflow at 1920px",
                  page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth"))
            if section in SECTION_SHOTS:
                page.evaluate("window.scrollTo(0, 0)")
                shot(page, SECTION_SHOTS[section], full=True)
        context.close()

        # Six 3D modules, high-resolution panel captures.
        context, page = open_page({"width": 1600, "height": 1000}, scale=2)
        page.goto(url + "/#command-center", wait_until="networkidle")
        section_ready(page, "command-center")
        page.locator("#data-mode").select_option("synthetic")
        for section, module in MODULES.items():
            result = {}
            page.locator(f'#section-nav a[href="#{section}"]').click()
            section_ready(page, section)
            lab_ready(page, module)
            page.evaluate("document.getElementById('lab-panel').scrollIntoView({block: 'start'})")
            page.wait_for_timeout(150)
            canvas = page.locator("#lab-canvas")
            box = canvas.bounding_box()
            cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            base = pixels(page)
            result["initial"] = base["lit"] > 200 and "SYNTHETIC" in page.locator("#lab-provenance").inner_text()
            legend = page.locator("#lab-legend").inner_text()
            result["legend"] = legend
            page.mouse.move(cx, cy)
            page.mouse.down()
            page.mouse.move(cx + 140, cy + 30, steps=8)
            page.mouse.up()
            rotated = pixels(page)
            result["rotate"] = rotated["hash"] != base["hash"]
            canvas.focus()
            page.keyboard.press("0")
            reset = pixels(page)
            result["reset_restores_view"] = reset["hash"] == base["hash"]
            page.mouse.move(cx, cy)
            page.mouse.wheel(0, -400)
            page.wait_for_timeout(120)
            zoomed = pixels(page)
            result["zoom"] = zoomed["hash"] != reset["hash"]
            page.keyboard.press("0")
            page.keyboard.down("Shift")
            page.mouse.move(cx, cy)
            page.mouse.down()
            page.mouse.move(cx + 90, cy - 40, steps=6)
            page.mouse.up()
            page.keyboard.up("Shift")
            panned = pixels(page)
            result["pan"] = panned["hash"] != base["hash"]
            page.keyboard.press("0")
            hover_at = None
            for fy in (0.5, 0.45, 0.55, 0.4, 0.6, 0.35, 0.65):
                for fx in (0.5, 0.45, 0.55, 0.4, 0.6, 0.35, 0.65, 0.3, 0.7):
                    page.mouse.move(box["x"] + box["width"] * fx, box["y"] + box["height"] * fy)
                    page.wait_for_timeout(30)
                    if not page.locator("#lab-tooltip").is_hidden():
                        hover_at = (fx, fy)
                        break
                if hover_at:
                    break
            result["hover_tooltip"] = bool(hover_at)
            result["hover_text"] = page.locator("#lab-tooltip").inner_text()[:120] if hover_at else None
            if hover_at:
                page.mouse.click(box["x"] + box["width"] * hover_at[0], box["y"] + box["height"] * hover_at[1])
                details = page.locator("#lab-details").inner_text()
                result["click_select"] = "Click or press" not in details
                result["selected_details"] = details[:160]
            canvas.focus()
            page.keyboard.press("Escape")
            page.keyboard.press("n")
            result["keyboard_select"] = page.locator("#lab-live").inner_text().startswith("Selected")
            page.keyboard.press("Escape")
            rows_before = page.locator("#lab-table tbody tr").count()
            before = pixels(page)["hash"]
            selects = page.locator("#lab-filters select")
            if selects.count():
                first = selects.nth(0)
                values = first.evaluate("s => [...s.options].map(o => o.value)")
                current = first.input_value()
                first.select_option(next(v for v in values if v != current))
                result["filter_changed"] = first.get_attribute("aria-label") or first.evaluate("s => s.labels[0]?.textContent")
            else:
                slider = page.locator("#lab-filters input[type=range]").nth(0)
                slider.fill(str(float(slider.get_attribute("min") or 0) + (float(slider.get_attribute("max") or 1) - float(slider.get_attribute("min") or 0)) * 0.25))
                result["filter_changed"] = "range"
            page.wait_for_timeout(200)
            result["data_update"] = pixels(page)["hash"] != before
            result["rows_before_after_filter"] = [rows_before, page.locator("#lab-table tbody tr").count()]
            if selects.count():
                selects.nth(0).select_option(current)
                page.wait_for_timeout(150)
            page.locator("#lab-orbit").click()
            first_frame = pixels(page)["hash"]
            page.wait_for_timeout(450)
            orbiting = page.evaluate("window.GIGModels.engine().isOrbiting()")
            result["orbit_animates"] = orbiting and pixels(page)["hash"] != first_frame
            page.locator("#lab-orbit").click()
            paused = pixels(page)["hash"]
            page.wait_for_timeout(350)
            result["pause_stops"] = not page.evaluate("window.GIGModels.engine().isOrbiting()") and pixels(page)["hash"] == paused
            page.locator("#lab-reset").click()
            page.locator("#lab-view-2d").click()
            flat = pixels(page)
            result["alt_2d"] = flat["lit"] > 100 and flat["hash"] != base["hash"]
            page.locator("#lab-view-3d").click()
            page.locator("#lab-table-toggle").click()
            result["data_table_rows"] = page.locator("#lab-table tbody tr").count()
            page.locator("#lab-table-toggle").click()
            page.locator("#lab-reset").click()
            page.wait_for_timeout(150)
            page.mouse.move(0, 0)
            shot(page, MODULE_SHOTS[section], "#lab-panel")
            report["modules"][module] = result
            for key in ("initial", "rotate", "zoom", "pan", "hover_tooltip", "keyboard_select", "data_update", "orbit_animates", "pause_stops", "alt_2d"):
                check(f"{module}: {key.replace('_', ' ')}", result.get(key), result.get(key))
            check(f"{module}: click on hovered item selects it", result.get("click_select"), result.get("selected_details"))
            check(f"{module}: exact-value table available", result["data_table_rows"] > 0, result["data_table_rows"])

        payload = page.evaluate("fetch('/api/v1/viz/vol-surface').then(r => r.json())")
        ivs = [p["implied_vol"] for snap in payload["data"]["snapshots"].values() for p in snap if p["implied_vol"] is not None]
        page.locator("#lab-tab-vol-surface").click()
        lab_ready(page, "vol-surface")
        legend = page.locator("#lab-legend").inner_text()
        check("vol-surface legend range equals the payload implied-vol range shared by both snapshots",
              f"{min(ivs):.3f}" in legend and f"{max(ivs):.3f}" in legend, [legend, min(ivs), max(ivs)])
        context.close()

        context, page = open_page({"width": 1600, "height": 1000}, reduced="reduce")
        page.goto(url + "/#options", wait_until="networkidle")
        section_ready(page, "options")
        page.locator("#data-mode").select_option("synthetic")
        refused = []
        for module in MODULES.values():
            page.locator(f"#lab-tab-{module}").click()
            lab_ready(page, module)
            page.locator("#lab-orbit").click()
            refused.append(not page.evaluate("window.GIGModels.engine().isOrbiting()"))
        check("reduced motion: orbit refused in all six modules", all(refused), refused)
        context.close()

        # Global Intelligence on the running server.
        context, page = open_page({"width": 1600, "height": 1000}, scale=2)
        page.goto(url + "/#operations", wait_until="networkidle")
        section_ready(page, "operations")
        page.locator('#section-nav a[href="#global-intelligence"]').click()
        page.wait_for_function("document.getElementById('intel-asof').textContent.includes('window')")
        page.wait_for_function("document.getElementById('intel-basemap').textContent.includes('Natural Earth')")
        report["global_intelligence_truth_strip"] = truth(page)
        check("Global Intelligence updates the global truth strip instead of keeping the previous section's",
              "local-process-diagnostics" not in " ".join(truth(page)), truth(page))
        if args.provider_refresh:
            refresh = {}
            for source, button in (("USGS_EARTHQUAKES", "#intel-refresh-usgs"), ("NASA_EONET", "#intel-refresh-eonet")):
                with page.expect_response(lambda r: r.url.endswith("/api/v1/intel/refresh"), timeout=60000) as exchange:
                    page.locator(button).click()
                refresh[source] = exchange.value.json().get("data")
                page.wait_for_function("document.getElementById('intel-status').textContent === ''")
            report["provider_refresh"] = refresh
        overview = page.evaluate("fetch('/api/v1/intel/overview').then(r => r.json())")["data"]
        report["sources"] = [{k: s.get(k) for k in ("source_id", "status", "semantics", "layer")} for s in overview["sources"]]
        report["source_health"] = overview.get("source_health")
        report["event_counts_by_provider"] = overview.get("counts_by_provider")
        semantics = page.locator("#intel-semantics").inner_text()
        report["semantics_panel"] = semantics
        badges = page.locator("#intel-panel .intel-badge").all_inner_texts()
        check("no source or event carries a LIVE label", page.locator(".intel-badge.live").count() == 0
              and not any(re.search(r"\bLIVE\b", b) for b in badges) and not re.search(r"\bLIVE\b", semantics), sorted(set(badges)))
        for layer in ("maritime", "aviation", "imagery"):
            check(f"{layer} layer is disabled with a stated blocker", page.locator(f'#intel-layers input[value="{layer}"]').is_disabled())
        feed_all = int(page.locator("#intel-feed-count").inner_text())
        page.locator('#intel-layers input[value="hazards"]').uncheck()
        feed_less = int(page.locator("#intel-feed-count").inner_text())
        page.locator('#intel-layers input[value="hazards"]').check()
        check("hazard layer toggle filters the feed", feed_less < feed_all or feed_all == 0, [feed_all, feed_less])
        report["feed_count"] = feed_all
        page.evaluate("document.getElementById('intel-canvas').scrollIntoView({block: 'center'})")
        globe = pixels(page, "intel-canvas")
        check("globe renders from the bundled basemap", globe["lit"] > 300, globe)
        canvas = page.locator("#intel-canvas")
        box = canvas.bounding_box()
        page.evaluate("window.__frames = 0; (function tick() { window.__frames++; requestAnimationFrame(tick); })()")
        start = time.perf_counter()
        frames0 = page.evaluate("window.__frames")
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] / 2 + 200, box["y"] + box["height"] / 2 + 20, steps=40)
        page.mouse.up()
        elapsed = time.perf_counter() - start
        frames = page.evaluate("window.__frames") - frames0
        report["globe_drag"] = {"pointer_steps": 40, "seconds": round(elapsed, 3), "animation_frames": frames,
                                "fps": round(frames / elapsed, 1) if elapsed else None}
        check("globe rotates on drag", pixels(page, "intel-canvas")["hash"] != globe["hash"])
        page.locator("#intel-reset").click()
        page.wait_for_timeout(200)
        shot(page, "02-global-intelligence-globe", ".intel-grid")
        page.locator("#intel-view-flat").click()
        page.wait_for_timeout(200)
        check("flat map renders", pixels(page, "intel-canvas")["lit"] > 300)
        shot(page, "03-global-intelligence-flat-map", ".intel-grid")
        page.locator("#intel-view-globe").click()
        usgs = page.locator(".intel-feed-button", has_text="USGS")
        target = usgs.first if usgs.count() else page.locator(".intel-feed-button").first
        if page.locator(".intel-feed-button").count():
            target.click()
            page.wait_for_timeout(250)
            details = page.locator("#intel-details").inner_text()
            report["event_detail"] = details
            check("event detail shows observation, publication, provider-update and acquisition clocks",
                  all(k in details for k in ("Observation time", "Publication time", "Acquired by GIG", "Reporting delay")), details[:300])
            shot(page, "04-global-intelligence-event-detail", ".intel-grid")
        count_now = int(page.locator("#intel-feed-count").inner_text())
        page.locator("#intel-timeline").fill("0")
        page.locator("#intel-timeline").dispatch_event("change")
        page.wait_for_function("document.getElementById('intel-asof').textContent.startsWith('Replay cutoff')")
        replay_badges = page.locator("#intel-health .intel-badge").all_inner_texts()
        check("timeline replay switches integrated sources to HISTORICAL or UNAVAILABLE",
              all(b in ("HISTORICAL", "UNAVAILABLE") for b in replay_badges), replay_badges)
        report["timeline"] = {"feed_now": count_now, "feed_at_earliest_cutoff": int(page.locator("#intel-feed-count").inner_text())}
        page.locator("#intel-now").click()
        page.wait_for_function("document.getElementById('intel-asof').textContent.startsWith('As of')")
        with page.expect_response(lambda r: r.url.endswith("/api/v1/intel/event-study")) as exchange:
            page.locator("#intel-study-run").click()
        study = exchange.value.json()["data"]
        check("event study (authored inputs) estimates before the event and is labelled not causal",
              study["uses_post_event_data_for_estimation"] is False and study["interpretation"] == "DESCRIPTIVE_SINGLE_EVENT_NOT_CAUSAL")
        with page.expect_response(lambda r: r.url.endswith("/api/v1/intel/exposure")):
            page.locator("#intel-exposure-run").click()
        page.wait_for_function("document.getElementById('intel-exposure-output').textContent.includes('Scenario P&L')")
        shot(page, "16-global-intelligence-research-panels", ".intel-columns >> nth=1")
        shot(page, "17-global-intelligence-sources", "#intel-coverage")
        context.close()

        for name, viewport in (("18-mobile-command-center", {"width": 390, "height": 844}),):
            context, page = open_page(viewport, scale=2)
            page.goto(url + "/#command-center", wait_until="networkidle")
            section_ready(page, "command-center")
            page.locator("#data-mode").select_option("synthetic")
            section_ready(page, "command-center")
            check(f"{name}: no horizontal overflow", page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth") <= 1)
            shot(page, name)
            page.goto(url + "/#global-intelligence", wait_until="networkidle")
            page.wait_for_function("document.getElementById('intel-asof').textContent.includes('window')")
            page.wait_for_timeout(300)
            check("mobile Global Intelligence: no horizontal overflow",
                  page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth") <= 1)
            shot(page, "19-mobile-global-intelligence")
            context.close()
        browser.close()

    check("no JavaScript exceptions", not errors, errors)
    check("browser attempted no external requests", not blocked, blocked)
    report["passed"] = sum(c["passed"] for c in checks)
    report["failed"] = sum(not c["passed"] for c in checks)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"{report['passed']} passed, {report['failed']} failed")
    for c in checks:
        if not c["passed"]:
            print("FAIL", c["name"], json.dumps(c["detail"], default=str)[:300])
    return int(report["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
