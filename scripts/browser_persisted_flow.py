"""Actual local Chromium analyst flow; authored inputs and offline execution only."""
from __future__ import annotations

import argparse
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
    parser.add_argument("--axe", type=Path)
    args = parser.parse_args()
    evidence = args.evidence_dir.absolute()
    evidence.mkdir(parents=True, exist_ok=True)
    checks, errors = [], []
    report = {"checks": checks, "javascript_errors": errors, "broker_connections": 0,
              "classification": "AUTHORED_SYNTHETIC_OFFLINE_WORKFLOW", "formal_accessibility_certified": False}

    def check(name, value, detail=None):
        checks.append({"name": name, "passed": bool(value), "detail": detail})

    with tempfile.TemporaryDirectory(prefix="gig-analyst-browser-") as directory:
        directory = Path(directory)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        env = {k: v for k, v in os.environ.items() if not any(word in k for word in ("ALPACA", "FINNHUB", "FRED_API", "FINBERT_MODEL_DIR"))}
        env.update(PYTHONPATH=str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", ""), GIG_BROKER_CONNECTIVITY_ENABLED="false")
        code = ("import uvicorn;from pathlib import Path;from gig.service.app import create_app;"
                f"uvicorn.run(create_app(state_dir=Path({str(directory / 'state')!r})),host='127.0.0.1',port={port},log_level='warning')")
        with (evidence / "server.txt").open("w") as log:
            server = subprocess.Popen([sys.executable, "-c", code], env=env, cwd=directory, stdout=log, stderr=subprocess.STDOUT)
            try:
                wait_ready(url, server)
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"], env=env)
                    report["chromium_version"] = browser.version
                    context = browser.new_context(viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")
                    context.route("**/*", lambda route: route.continue_() if route.request.url.startswith(url + "/") else route.abort())
                    page = context.new_page()
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(url + "/#research", wait_until="networkidle")
                    page.wait_for_function("!document.getElementById('refresh').disabled")
                    responses = {}
                    actions = {"pipeline-bars": "bars", "pipeline-universe": "universe", "research-dataset": "research-dataset",
                               "research-model": "research-model", "research-predict": "research-predict", "research-allocate": "research-allocate",
                               "research-prepare": "research-prepare", "offline-dispatch": "execution-engine", "offline-fill": "execution-engine",
                               "portfolio-accounting": "portfolio-accounting", "schedule-twap": "execution-engine", "advance-clock": "execution-engine"}
                    for action, endpoint in actions.items():
                        page.locator("#analytics-kind").select_option("workspace-" + action)
                        with page.expect_response(lambda response, endpoint=endpoint: response.url.endswith("/api/v1/workspace/" + endpoint)) as exchange:
                            page.locator("#run-analysis").click()
                        page.wait_for_function("!document.getElementById('run-analysis').disabled")
                        response = exchange.value.json()
                        responses[action] = response
                        check(action + " explicit browser action", exchange.value.status == 200 and response["state"] == "AVAILABLE",
                              {"status": exchange.value.status, "state": response["state"]})
                        if response["state"] != "AVAILABLE":
                            raise AssertionError("Offline analyst action failed: " + action)
                    ids = page.locator("#workflow-identities").inner_text()
                    check("returned dataset run and allocation identities connected", all(responses[k]["data"][field] in ids for k, field in
                          [("research-dataset", "dataset_id"), ("research-model", "run_id"), ("research-allocate", "allocation_id")]))
                    check("acknowledgement creates no positions", responses["offline-dispatch"]["data"]["positions"] == {})
                    check("partial fill creates simulated position", bool(responses["offline-fill"]["data"]["positions"]))
                    check("saved model inference does not retrain", responses["research-predict"]["data"]["metrics"]["training_performed"] is False)
                    page.locator("#replay-asof").fill("2026-11-02T15:49:00Z")
                    page.locator("#replay-asof").press("Tab")
                    page.wait_for_function("!document.getElementById('refresh').disabled")
                    for engine, section in [("persisted", "models"), ("persisted", "signals"), ("allocation", "portfolio"),
                                            ("accounting", "portfolio"), ("accounting", "risk"), ("execution-engine", "execution")]:
                        page.locator("#analysis-engine").select_option(engine)
                        page.wait_for_function("!document.getElementById('refresh').disabled")
                        with page.expect_response(lambda response, section=section, engine=engine:
                                                  "/api/v1/terminal/" + section + "?" in response.url and "engine=" + engine in response.url):
                            if page.url.endswith("#" + section):
                                page.locator("#refresh").click()
                            else:
                                page.locator(f'a[href="#{section}"]').click()
                        page.wait_for_function("!document.getElementById('refresh').disabled")
                        state = page.locator("#state-badge").inner_text()
                        check(engine + " " + section + " terminal view", state == "AVAILABLE", state)
                    report["accessibility"] = []
                    for theme in ("dark", "light"):
                        page.evaluate("theme=>document.documentElement.dataset.theme=theme", arg=theme)
                        contrasts = page.evaluate(CONTRAST_JS)
                        check(theme + " analyst visible contrast", all(row["passed"] for row in contrasts), [r for r in contrasts if not r["passed"]])
                        if args.axe:
                            import hashlib
                            manifest = json.loads((args.axe.parent / "manifest.json").read_text())
                            assert hashlib.sha256(args.axe.read_bytes()).hexdigest() == manifest["script_sha256"]
                            page.add_script_tag(path=str(args.axe))
                            result = page.evaluate("async()=>await axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa']}})")
                            report["accessibility"].append({"theme": theme, "version": result["testEngine"]["version"], "violations": result["violations"], "incomplete": result["incomplete"]})
                            check(theme + " analyst automated accessibility", not result["violations"])
                        page.evaluate("window.scrollTo(0,0)")
                        page.screenshot(path=str(evidence / f"analyst-{theme}.png"), full_page=True)
                    for width in (320, 375, 768, 1024, 1920):
                        page.set_viewport_size({"width": width, "height": 900})
                        check(f"analyst {width}px no document overflow", page.evaluate("document.documentElement.scrollWidth<=innerWidth"))
                    check("no JavaScript exceptions", not errors)
                    browser.close()
            finally:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)
    report["passed"] = sum(row["passed"] for row in checks)
    report["failed"] = sum(not row["passed"] for row in checks)
    (evidence / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"checks", "accessibility"}}, indent=2))
    for row in checks:
        if not row["passed"]:
            print("FAIL", row["name"], row["detail"])
    return int(report["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
