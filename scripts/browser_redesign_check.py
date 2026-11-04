"""Admitted-depth, motion and layout checks in actual isolated Chromium."""
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

from browser_terminal_check import CONTRAST_JS, ROOT, wait_ready
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--axe", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.axe.parent / "manifest.json").read_text())
    assert hashlib.sha256(args.axe.read_bytes()).hexdigest() == manifest["script_sha256"]
    output = args.evidence_dir.absolute()
    output.mkdir(parents=True, exist_ok=True)
    checks, errors, accessibility = [], [], []

    def check(name, condition, detail=None):
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    with tempfile.TemporaryDirectory(prefix="gig-redesign-browser-") as temporary:
        directory = Path(temporary)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        env = {k: v for k, v in os.environ.items() if not any(t in k for t in ("ALPACA", "FINNHUB", "FRED_API", "FINBERT_MODEL_DIR"))}
        env.update(PYTHONPATH=str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", ""), GIG_BROKER_CONNECTIVITY_ENABLED="false")
        code = ("import uvicorn;from pathlib import Path;from gig.service.app import create_app;"
                f"uvicorn.run(create_app(state_dir=Path({str(directory / 'state')!r})),host='127.0.0.1',port={port},log_level='warning')")
        with (output / "server.txt").open("w") as log:
            server = subprocess.Popen([sys.executable, "-c", code], env=env, cwd=directory, stdout=log, stderr=subprocess.STDOUT)
            try:
                wait_ready(url, server)
                with sync_playwright() as pw:
                    browser = pw.chromium.launch(headless=True, args=["--no-sandbox"], env=env)
                    version = browser.version
                    context = browser.new_context(viewport={"width": 1440, "height": 1100}, reduced_motion="reduce")
                    context.route("**/*", lambda route: route.continue_() if route.request.url.startswith(url + "/") else route.abort())
                    page = context.new_page()
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(url + "/#markets", wait_until="networkidle")
                    page.wait_for_function("!document.getElementById('refresh').disabled")
                    check("no synthetic depth on initial load", page.locator("#depth-content").is_hidden() and "own evidence" in page.locator("#depth-empty").inner_text())
                    page.locator("#analysis-engine").select_option("depth")
                    page.wait_for_function("!document.getElementById('refresh').disabled")
                    page.locator("#replay-asof").fill("2026-11-02T15:49:00Z")
                    page.locator("#replay-asof").press("Tab")
                    page.locator("#analytics-kind").select_option("workspace-depth")
                    with page.expect_response(lambda r: r.url.endswith("/api/v1/workspace/depth")) as response:
                        page.locator("#run-analysis").click()
                    page.wait_for_function("!document.getElementById('run-analysis').disabled")
                    check("depth admission succeeds through explicit browser action", response.value.status == 200 and response.value.json()["state"] == "AVAILABLE")
                    check("large bid ask ladder uses admitted levels", page.locator("#depth-levels tr").count() == 16 and page.locator("#depth-content").is_visible())
                    check("cumulative curves use two admitted sides", page.locator("#depth-chart polyline").count() == 2)
                    check("depth provenance remains synthetic and caller supplied", "SYNTHETIC" in page.locator("#depth-provenance").inner_text() and "CALLER_SUPPLIED" in page.locator("#depth-provenance").inner_text())
                    check("depth ladder has a substantial scrollable viewport", page.locator(".depth-table-wrap").bounding_box()["height"] >= 350)
                    for theme in ("dark", "light"):
                        page.evaluate("theme=>document.documentElement.dataset.theme=theme", arg=theme)
                        contrasts = page.evaluate(CONTRAST_JS)
                        check(theme + " depth visible contrast", all(x["passed"] for x in contrasts), [x for x in contrasts if not x["passed"]])
                        page.add_script_tag(path=str(args.axe))
                        result = page.evaluate("async()=>await axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa']}})")
                        accessibility.append({"theme": theme, "version": result["testEngine"]["version"], "violations": result["violations"], "incomplete": result["incomplete"]})
                        check(theme + " depth automated accessibility", not result["violations"], [x["id"] for x in result["violations"]])
                        page.evaluate("window.scrollTo(0,0)")
                        page.screenshot(path=str(output / f"depth-{theme}.png"), full_page=True)
                    for width in (320, 375, 768, 1024, 1920):
                        page.set_viewport_size({"width": width, "height": 1000})
                        check(f"depth {width}px no horizontal overflow", page.evaluate("document.documentElement.scrollWidth<=innerWidth"))
                        if width == 320:
                            page.evaluate("window.scrollTo(0,0)")
                            page.screenshot(path=str(output / "depth-mobile-320.png"), full_page=True)
                            page.screenshot(path=str(output / "depth-mobile-320-viewport.png"))
                    page.set_viewport_size({"width": 1440, "height": 1100})
                    page.route("**/api/v1/terminal/*", lambda route: route.abort())
                    page.locator("#refresh").click()
                    page.wait_for_function("!document.getElementById('refresh').disabled")
                    check("transport error clears previous successful depth", page.locator("#state-badge").inner_text() == "ERROR" and page.locator("#depth-content").is_hidden())
                    page.unroute("**/api/v1/terminal/*")
                    page.locator("#freshness-policy").select_option("3600")
                    page.locator("#replay-asof").fill("2026-11-02T17:51:00Z")
                    page.locator("#replay-asof").press("Tab")
                    page.locator("#refresh").click()
                    page.wait_for_function("!document.getElementById('refresh').disabled")
                    check("stale depth stays explicitly stale", page.locator("#state-badge").inner_text() == "STALE" and "STALE" in page.locator("#depth-state").inner_text())
                    page.locator("#depth-instrument").fill("NO_ADMITTED_BOOK")
                    page.locator("#depth-instrument").press("Tab")
                    page.wait_for_function("!document.getElementById('refresh').disabled")
                    check("missing book never falls back to another instrument", page.locator("#state-badge").inner_text() == "UNAVAILABLE" and page.locator("#depth-levels tr").count() == 0)
                    page.locator("#data-mode").select_option("synthetic")
                    page.wait_for_function("!document.getElementById('refresh').disabled")
                    check("reduced motion starts orbit paused", page.locator("#orbit-toggle").get_attribute("aria-pressed") == "false")
                    page.emulate_media(reduced_motion="no-preference")
                    page.locator("#orbit-toggle").click()
                    canvas = page.locator("#analytics-chart")
                    before = canvas.screenshot()
                    page.wait_for_timeout(600)
                    check("slow orbit moves actual analytical canvas", before != canvas.screenshot() and ".045" in page.locator("#motion-state").inner_text())
                    page.locator("#orbit-toggle").click()
                    paused = canvas.screenshot()
                    page.wait_for_timeout(250)
                    check("pause control freezes camera", paused == canvas.screenshot() and page.locator("#orbit-toggle").get_attribute("aria-pressed") == "false")
                    page.locator("#orbit-toggle").click()
                    page.emulate_media(reduced_motion="reduce")
                    page.wait_for_function("document.getElementById('orbit-toggle').getAttribute('aria-pressed')==='false'")
                    check("reduced motion preference immediately pauses camera", "Reduced motion" in page.locator("#motion-state").inner_text())
                    check("all twelve navigation sections retained plus Global Intelligence", page.locator("#section-nav a").count() == 13 and page.locator('#section-nav a[href="#global-intelligence"]').count() == 1)
                    check("no JavaScript exceptions", not errors, errors)
                    browser.close()
            finally:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)
    report = {"chromium_version": version, "checks": checks, "accessibility": accessibility,
              "javascript_errors": errors, "passed": sum(x["passed"] for x in checks), "failed": sum(not x["passed"] for x in checks),
              "source_classification": "AUTHORED_SYNTHETIC_DEPTH_FIXTURE", "broker_connections": 0,
              "public_deployment": False, "formal_accessibility_certification": False}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"checks", "accessibility"}}, indent=2))
    for row in checks:
        if not row["passed"]:
            print("FAIL", row["name"], row["detail"])
    return int(report["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
