"""Real Chromium persisted pairs/options/IC analyst workflow, using authored data.

All HTTP writes are explicit browser actions against an isolated loopback service.
No external page requests, accounts, credentials, or brokerage operations are used.
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

from browser_terminal_check import CONTRAST_JS, ROOT, wait_ready
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--axe", type=Path)
    args = parser.parse_args()
    output = args.evidence_dir.absolute()
    output.mkdir(parents=True, exist_ok=True)
    checks, errors, accessibility, responses = [], [], [], {}
    report = {"runner": "real-playwright-chromium", "checks": checks, "javascript_errors": errors,
              "accessibility": accessibility, "broker_connections": 0, "public_deployment": False,
              "input_classification": "AUTHORED_SYNTHETIC_AND_MODELED_FIXTURES",
              "formal_accessibility_certification": False}
    if args.axe:
        manifest = json.loads((args.axe.parent / "manifest.json").read_text())
        assert hashlib.sha256(args.axe.read_bytes()).hexdigest() == manifest["script_sha256"]
        report["axe_asset"] = manifest

    def check(name, condition, detail=None):
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    with tempfile.TemporaryDirectory(prefix="gig-sprint5-browser-") as temporary:
        directory = Path(temporary)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        env = {key: value for key, value in os.environ.items()
               if not any(word in key for word in ("ALPACA", "FINNHUB", "FRED_API", "FINBERT_MODEL_DIR"))}
        env.update(PYTHONPATH=str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", ""),
                   GIG_BROKER_CONNECTIVITY_ENABLED="false", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        code = ("import uvicorn;from pathlib import Path;from gig.service.app import create_app;"
                f"uvicorn.run(create_app(state_dir=Path({str(directory / 'state')!r})),host='127.0.0.1',port={port},log_level='warning')")
        with (output / "server.txt").open("w") as log:
            server = subprocess.Popen([sys.executable, "-c", code], env=env, cwd=directory,
                                      stdout=log, stderr=subprocess.STDOUT)
            try:
                wait_ready(url, server)
                with sync_playwright() as pw:
                    browser = pw.chromium.launch(headless=True, args=["--no-sandbox"], env=env)
                    report["chromium_version"] = browser.version
                    context = browser.new_context(viewport={"width": 1440, "height": 1100}, reduced_motion="reduce")
                    context.route("**/*", lambda route: route.continue_() if route.request.url.startswith(url + "/") else route.abort())
                    page = context.new_page()
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(url + "/#research", wait_until="networkidle")
                    ready = "!document.getElementById('refresh').disabled"
                    page.wait_for_function(ready)

                    def action(kind, endpoint=None, *, expect="AVAILABLE", change=None):
                        endpoint = endpoint or kind
                        page.locator("#analytics-kind").select_option("workspace-" + kind)
                        if change:
                            payload = json.loads(page.locator("#analytics-input").input_value())
                            payload.update(change)
                            page.locator("#analytics-input").fill(json.dumps(payload))
                        with page.expect_response(lambda r: r.url.endswith("/api/v1/workspace/" + endpoint)) as exchange:
                            page.locator("#run-analysis").click()
                        page.wait_for_function("!document.getElementById('run-analysis').disabled")
                        response = exchange.value.json()
                        responses[kind] = response
                        check(kind + " explicit Chromium submission", exchange.value.status == 200 and response["state"] == expect,
                              {"status": exchange.value.status, "state": response["state"]})
                        if exchange.value.status != 200 or response["state"] != expect:
                            raise AssertionError("Unexpected analyst action state: " + kind)
                        return response

                    action("pipeline-bars", "bars")
                    action("pipeline-universe", "universe")
                    dataset = action("research-dataset")
                    model = action("research-model")
                    pair = action("pairs-research")
                    pair_replay = action("pairs-replay")
                    check("pair replay retains exact immutable experiment", pair == pair_replay)
                    check("pair diagnostics retain fit evidence without trading approval",
                          bool(pair["data"]["folds"]) and pair["data"]["trading_approved"] is False)
                    signal = action("research-signals")
                    signal_replay = action("signals-replay")
                    check("signal replay retains exact IC evidence", signal == signal_replay)
                    check("IC state does not invent a calibrated confidence",
                          signal["data"]["scoring_state"] in {"AVAILABLE", "UNKNOWN"}
                          and all(row["confidence"]["state"] == "UNKNOWN" for row in signal["data"]["diagnostics"]))
                    admitted = action("options-admit")
                    option = action("options-research")
                    option_replay = action("options-replay")
                    check("options replay retains exact original quote and model identity", option == option_replay)
                    check("options inversion uses all authored fixture quotes",
                          option["data"]["metrics"]["valid_models"] == 12 and option["data"]["input_classification"] == "SYNTHETIC")
                    check("Greek exposure uses explicit simulated positions", option["data"]["portfolio_exposure"]["state"] == "AVAILABLE")
                    check("CSM remains unavailable without authoritative definition", option["data"]["csm"]["state"] == "UNAVAILABLE")
                    # A separate explicit single-stock simulated account matches the
                    # authored options spot and observation clock exactly. The
                    # hypothetical options positions remain unfinanced.
                    client_id = "authored-overlay-underlying-1"
                    action("execution-engine", change={"client_id": client_id, "symbol": "SYN101",
                        "quantity": "1", "reference_price": "110", "price_asof": "2026-11-02T15:49:00Z",
                        "now": "2026-11-02T15:49:00Z", "max_notional": "1000", "max_position_quantity": "10"})
                    ack = action("offline-dispatch", "execution-engine")
                    check("overlay fixture acknowledgement remains unfilled", ack["data"]["positions"] == {})
                    action("offline-fill", "execution-engine", change={"client_id": client_id,
                        "price": "110", "quantity": "1", "fee": "0.10"})
                    account = action("portfolio-accounting", change={"marks": {"SYN101": {
                        "price": "110", "as_of": "2026-11-02T15:49:00Z"}},
                        "source_identity": "authored-overlay-mark-110", "input_classification": "SYNTHETIC"})
                    check("reconciled checkpoint retains exact simulated equity position",
                          account["data"]["checkpoint_id"] in page.locator("#workflow-identities").inner_text())
                    oms_url = url + "/api/v1/terminal/execution?mode=local&engine=execution-engine&as_of=2026-11-02T15%3A49%3A00Z&max_age_seconds=3600"
                    before_overlay = context.request.get(oms_url).json()
                    overlay = action("options-overlay")
                    overlay_replay = action("options-overlay-replay")
                    check("hypothetical overlay replays exact saved evidence", overlay == overlay_replay)
                    check("options financing remains explicitly unknown",
                          overlay["data"]["overlay_financing"]["state"] == "UNKNOWN"
                          and overlay["data"]["overlay_financing"]["financed_nav"] is None)
                    check("overlay evaluation leaves offline OMS evidence unchanged",
                          before_overlay == context.request.get(oms_url).json())
                    check("overlay identity uses its own replay field",
                          overlay["data"]["overlay_id"] in page.locator("#workflow-identities").inner_text())
                    ids = page.locator("#workflow-identities").inner_text()
                    check("CPU, pairs, options and signal identities remain distinct and visible",
                          all(value in ids for value in [dataset["data"]["dataset_id"], model["data"]["run_id"],
                              pair["data"]["run_id"], admitted["data"]["snapshot_id"], option["data"]["run_id"], signal["data"]["signal_id"]]))
                    check("session comparison retains submitted identities", page.locator("#experiment-comparison").is_visible()
                          and "Session-only comparison" in page.locator("#comparison-policy").inner_text())
                    page.locator("#replay-asof").fill("2026-11-02T15:49:00Z")
                    page.locator("#replay-asof").press("Tab")
                    page.wait_for_function(ready)
                    for engine, section in [("pairs", "stat-arb"), ("pairs", "models"),
                                            ("signals-research", "signals"), ("options-research", "options"),
                                            ("options-research", "portfolio"), ("options-research", "risk"),
                                            ("options-overlay", "portfolio"), ("options-overlay", "risk")]:
                        page.locator("#analysis-engine").select_option(engine)
                        page.wait_for_function(ready)
                        with page.expect_response(lambda r, engine=engine, section=section:
                                "/api/v1/terminal/" + section + "?" in r.url and "engine=" + engine in r.url):
                            if page.url.endswith("#" + section):
                                page.locator("#refresh").click()
                            else:
                                page.locator(f'a[href="#{section}"]').click()
                        page.wait_for_function(ready)
                        current_state = page.locator("#state-badge").inner_text()
                        check(engine + " " + section + " serves persisted backend evidence", current_state == "AVAILABLE", current_state)
                        if engine == "options-overlay":
                            check("overlay view retains financing warning in " + section,
                                  "financing UNKNOWN" in page.locator("#truth-notice").inner_text())
                        if section == "options":
                            check("options surface plots admitted coordinates with model labels",
                                  "modeled midpoint IV" in page.locator("#chart-title").inner_text()
                                  and "ACT/365F" in page.locator("#chart-legend").inner_text())
                            page.locator("#evidence-table-view").select_option("term_structure")
                            check("analyst can inspect actual saved term structure", page.locator("#observations-table tbody tr").count() == 2)
                            page.locator("#evidence-table-view").select_option("modeled_scenarios")
                            check("analyst can inspect explicit modeled scenarios", page.locator("#observations-table tbody tr").count() == 9)
                    for theme in ("dark", "light"):
                        page.evaluate("theme=>document.documentElement.dataset.theme=theme", arg=theme)
                        contrast = page.evaluate(CONTRAST_JS)
                        check(theme + " visible text contrast", all(row["passed"] for row in contrast), [r for r in contrast if not r["passed"]])
                        if args.axe:
                            if page.locator("#context-expand").get_attribute("aria-expanded") != "true":
                                page.locator("#context-expand").click()
                            page.add_script_tag(path=str(args.axe))
                            result = page.evaluate("async()=>await axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa']}})")
                            accessibility.append({"theme": theme, "version": result["testEngine"]["version"],
                                                  "violations": result["violations"], "incomplete": result["incomplete"]})
                            check(theme + " automated accessibility", not result["violations"], [r["id"] for r in result["violations"]])
                        if page.locator("#context-expand").get_attribute("aria-expanded") == "true":
                            page.locator("#context-expand").click()
                        page.evaluate("window.scrollTo(0,0)")
                        page.screenshot(path=str(output / f"sprint5-{theme}.png"), full_page=True)
                    for width in (320, 375, 768, 1024, 1920):
                        page.set_viewport_size({"width": width, "height": 1000})
                        check(f"{width}px research workflow without document overflow", page.evaluate("document.documentElement.scrollWidth<=innerWidth"))
                        if width == 320:
                            page.evaluate("window.scrollTo(0,0)")
                            page.screenshot(path=str(output / "sprint5-mobile-320.png"), full_page=True)
                    page.set_viewport_size({"width": 1440, "height": 1100})
                    page.locator("#replay-asof").fill("2026-11-02T17:50:00Z")
                    page.locator("#replay-asof").press("Tab")
                    page.locator("#freshness-policy").select_option("3600")
                    page.wait_for_function(ready)
                    check("historical research remains explicitly stale", page.locator("#state-badge").inner_text() == "STALE")
                    page.route("**/api/v1/terminal/*", lambda route: route.abort())
                    page.locator("#refresh").click()
                    page.wait_for_function(ready)
                    check("transport failure clears previous saved research rows",
                          page.locator("#state-badge").inner_text() == "ERROR" and page.locator("#observations-table tbody tr").count() == 0)
                    page.unroute("**/api/v1/terminal/*")
                    check("all twelve sections retained plus Global Intelligence, Observatory, Market Charts and Market Flow", page.locator("#section-nav a").count() == 16 and page.locator('#section-nav a[href="#market-flow"]').count() == 1 and page.locator('#section-nav a[href="#global-intelligence"]').count() == 1 and page.locator('#section-nav a[href="#observatory"]').count() == 1 and page.locator('#section-nav a[href="#charts"]').count() == 1)
                    check("reduced-motion camera remains paused", page.locator("#orbit-toggle").get_attribute("aria-pressed") == "false")
                    check("no JavaScript exceptions", not errors, errors)
                    browser.close()
            except Exception as exc:
                check("workflow completed without interruption", False, type(exc).__name__ + ": " + str(exc))
            finally:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)
    report["passed"] = sum(row["passed"] for row in checks)
    report["failed"] = sum(not row["passed"] for row in checks)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "workflow-responses.json").write_text(json.dumps(responses, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"checks", "accessibility"}}, indent=2))
    for row in checks:
        if not row["passed"]:
            print("FAIL", row["name"], row["detail"])
    return int(report["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
