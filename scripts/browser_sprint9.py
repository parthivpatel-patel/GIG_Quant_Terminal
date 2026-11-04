"""Actual-browser Sprint 9 cost-evidence, lineage and operational checks, fixture-only.

An isolated local service is started from source. The browser may reach only
that origin; every other request is aborted and counted. No provider, broker or
public deployment is contacted. Results are written as JSON with screenshots.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from browser_terminal_check import ROOT, wait_ready
from playwright.sync_api import sync_playwright

SECTIONS = ("command-center", "markets", "signals", "portfolio", "risk", "models", "news",
            "stat-arb", "options", "execution", "research", "operations")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.evidence_dir.absolute()
    out.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []
    errors: list[str] = []
    blocked: list[str] = []
    report = {"runner": "real-playwright-chromium", "checks": checks, "javascript_errors": errors,
              "blocked_external_requests": blocked, "broker_connections": 0, "provider_connections": 0,
              "public_deployment": False, "input_classification": "AUTHORED_SYNTHETIC_FIXTURES",
              "formal_accessibility_certification": False}

    def check(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    with tempfile.TemporaryDirectory(prefix="gig-sprint9-browser-") as temporary:
        directory = Path(temporary)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        env = {k: v for k, v in os.environ.items() if not any(x in k for x in ("ALPACA", "FINNHUB", "FRED_API", "FINBERT_MODEL_DIR"))}
        env.update(PYTHONPATH=str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", ""),
                   GIG_BROKER_CONNECTIVITY_ENABLED="false", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   XDG_CACHE_HOME=str(directory / "browser-cache"), GSETTINGS_BACKEND="memory")
        server = None
        log = (out / "server.txt").open("w")

        def start():
            nonlocal server
            if server is not None:
                server.terminate()
                server.wait(timeout=10)
            code = ("import uvicorn;from pathlib import Path;from gig.service.app import create_app;"
                    f"uvicorn.run(create_app(state_dir=Path({str(directory / 'state')!r})),host='127.0.0.1',port={port},log_level='warning')")
            server = subprocess.Popen([sys.executable, "-c", code], env=env, cwd=directory, stdout=log, stderr=subprocess.STDOUT)
            wait_ready(url, server)

        try:
            start()
            # Operational/security boundaries over raw HTTP, independent of the UI.
            for asset in ("/", "/static/app.js", "/static/styles.css"):
                with urllib.request.urlopen(url + asset, timeout=5) as response:
                    headers = response.headers
                    check(f"static {asset} served with no-store and nosniff",
                          response.status == 200 and "no-store" in headers["Cache-Control"]
                          and headers["X-Content-Type-Options"] == "nosniff")
            request = urllib.request.Request(url + "/api/v1/workspace/bars", data=b"{}", method="POST",
                                             headers={"Content-Type": "application/json", "Origin": "http://evil.invalid"})
            try:
                urllib.request.urlopen(request, timeout=5)
                check("cross-origin workspace write rejected", False)
            except urllib.error.HTTPError as exc:
                check("cross-origin workspace write rejected", exc.code == 403, exc.code)
            request = urllib.request.Request(url + "/api/v1/analytics/cost-evidence", data=b"{not json", method="POST",
                                             headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(request, timeout=5)
                check("malformed JSON rejected without echo", False)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode()
                check("malformed JSON rejected without echo", exc.code == 400 and "not json" not in body, exc.code)

            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True, timeout=30000, args=["--no-sandbox"], env=env)
                report["browser_version"] = browser.version
                context = browser.new_context(viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")

                def route(r):
                    if r.request.url.startswith(url + "/"):
                        r.continue_()
                    else:
                        blocked.append(r.request.url)
                        r.abort()
                context.route("**/*", route)
                page = context.new_page()
                page.on("pageerror", lambda error: errors.append(str(error)))
                ready = "!document.getElementById('refresh').disabled"
                page.goto(url + "/#execution", wait_until="networkidle")
                page.wait_for_function(ready)

                def calculate(kind):
                    page.locator("#analytics-kind").select_option(kind)
                    with page.expect_response(lambda r: r.url.endswith("/api/v1/analytics/" + kind)) as exchange:
                        page.locator("#run-analysis").click()
                    page.wait_for_function("!document.getElementById('run-analysis').disabled")
                    return exchange.value.status, exchange.value.json()

                status, cost = calculate("cost-evidence")
                data = cost.get("data") or {}
                classes = data.get("classes", {})
                check("cost evidence browser calculation", status == 200 and cost["state"] == "AVAILABLE", status)
                check("estimated, simulated and observed classes stay separate",
                      classes.get("ESTIMATED", {}).get("count") == 1 and classes.get("SIMULATED", {}).get("count") == 1
                      and classes.get("OBSERVED_HISTORICAL", {}).get("count") == 1)
                check("broker reconciliation shown unavailable",
                      classes.get("BROKER_RECONCILED", {}).get("reason") == "BROKER_DISCONNECTED")
                check("simulated fills never promoted", data.get("claims", {}).get("simulated_promoted_by_prints") is False)
                output = page.locator("#analytics-output").inner_text()
                check("cost evidence rendered in workbench output", "BROKER_DISCONNECTED" in output)
                page.screenshot(path=str(out / "cost-evidence.png"), full_page=False)
                first_hash = data.get("report_sha256")

                status, lineage = calculate("provenance")
                check("lineage eligibility browser calculation",
                      status == 200 and lineage["data"]["eligibility"] == "UNKNOWN", lineage.get("data", {}).get("eligibility"))
                check("UCI archive is not claimed point-in-time", lineage["data"]["point_in_time_claimed"] is False)

                page.locator("#analytics-kind").select_option("cost-evidence")
                page.locator("#analytics-input").fill("{\"currency\": \"USD\"}")
                with page.expect_response(lambda r: r.url.endswith("/api/v1/analytics/cost-evidence")) as exchange:
                    page.locator("#run-analysis").click()
                page.wait_for_function("!document.getElementById('run-analysis').disabled")
                check("invalid request shows classified error", exchange.value.status == 422
                      and "INVALID_REQUEST" in page.locator("#analytics-output").inner_text())

                start()  # restart: deterministic replay after process replacement
                page.reload(wait_until="networkidle")
                page.wait_for_function(ready)
                status, again = calculate("cost-evidence")
                check("restart then deterministic replay of cost report", status == 200 and again["data"]["report_sha256"] == first_hash)

                for section in SECTIONS:
                    page.locator(f'#section-nav a[href="#{section}"]').click()
                    page.wait_for_function(
                        f"""document.querySelector('#section-nav a[href="#{section}"]').getAttribute('aria-current') === 'page'
                            && {ready} && document.getElementById('state-badge').textContent.trim() !== 'LOADING'""")
                    badge = page.locator("#state-badge").inner_text()
                    check(f"{section} renders a truth state", badge in {"AVAILABLE", "UNAVAILABLE", "UNKNOWN", "STALE", "ERROR"}, badge)
                browser.close()
        finally:
            if server is not None:
                server.terminate()
                server.wait(timeout=10)
            log.close()
    check("no JavaScript exceptions", not errors, errors)
    check("no external requests attempted", not blocked, blocked)
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
