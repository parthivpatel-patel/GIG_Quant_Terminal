"""Real Chromium persisted funded accounting/economics/microstructure analyst workflow, using authored data.

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
from decimal import Decimal
from pathlib import Path

from browser_terminal_check import CONTRAST_JS, ROOT, wait_ready
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--axe", type=Path)
    parser.add_argument("--skip-funded-risk", action="store_true", help="Verify other workstreams while the funded-risk backend is unavailable; reports this exclusion explicitly.")
    args = parser.parse_args()
    output = args.evidence_dir.absolute()
    output.mkdir(parents=True, exist_ok=True)
    checks, errors, accessibility, responses = [], [], [], {}
    report = {"runner": "real-playwright-chromium", "checks": checks, "javascript_errors": errors,
              "accessibility": accessibility, "broker_connections": 0, "public_deployment": False,
              "input_classification": "AUTHORED_SYNTHETIC_AND_MODELED_FIXTURES",
              "formal_accessibility_certification": False}
    report["funded_risk_workflow"] = "SKIPPED_BY_ARGUMENT" if args.skip_funded_risk else "INCLUDED"
    if args.axe:
        manifest = json.loads((args.axe.parent / "manifest.json").read_text())
        assert hashlib.sha256(args.axe.read_bytes()).hexdigest() == manifest["script_sha256"]
        report["axe_asset"] = manifest

    def check(name, condition, detail=None):
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    with tempfile.TemporaryDirectory(prefix="gig-sprint6-browser-") as temporary:
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

                    def action(kind, endpoint=None, *, expect="AVAILABLE", change=None, status=200):
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
                        check(kind + " explicit Chromium submission", exchange.value.status == status and response["state"] == expect,
                              {"status": exchange.value.status, "state": response["state"]})
                        if exchange.value.status != status or response["state"] != expect:
                            raise AssertionError("Unexpected analyst action state: " + kind)
                        return response

                    legacy_url = url + "/api/v1/terminal/execution?mode=local&engine=execution-engine&as_of=2026-12-02T15%3A49%3A00Z&max_age_seconds=3600"
                    legacy_before = context.request.get(legacy_url).json()
                    check("new funded view starts unavailable", context.request.get(url + "/api/v1/terminal/portfolio?engine=funded-account&mode=local").json()["state"] == "UNAVAILABLE")
                    action("pipeline-bars", "bars")
                    action("pipeline-universe", "universe")
                    action("research-dataset")
                    action("research-model")
                    economics = action("economic-research")
                    check("economic report replays exact saved research", economics == action("economic-replay"))
                    check("economic diagnostics never claim validated profit", economics["data"]["economically_validated"] is False and economics["data"]["production_suitable"] is False)
                    check("economic cost family retains all explicit combinations", len(economics["data"]["rows"]) == 4 and economics["data"]["family"]["declared_trials"] == 4)
                    action("microstructure-admit")
                    micro = action("microstructure-research")
                    check("microstructure replay retains immutable event and research identity", micro == action("microstructure-replay"))
                    check("actual admitted prints retain SYNTHETIC provenance", micro["data"]["input_classification"] == "SYNTHETIC" and micro["data"]["metrics"]["prints"] == 12)
                    check("print capacity is not represented as executable liquidity", micro["data"]["capacity"]["executable_capacity"]["state"] == "UNKNOWN")
                    action("options-admit")
                    action("options-research")
                    initialized = action("funded-initialize", "funded-account")
                    check("initial simulated cash is exact", Decimal(initialized["data"]["account"]["cash"]) == Decimal("100000"))
                    action("funded-register", "funded-account")
                    reserved = action("funded-reserve", "funded-account")
                    check("premium multiplier and fee budget are reserved before execution", Decimal(reserved["data"]["account"]["reserved_cash"]) == Decimal("602"))
                    ack = action("funded-dispatch", "funded-account")
                    check("funded acknowledgement is not a fill", ack["data"]["account"]["effective_positions"] == {} and Decimal(ack["data"]["account"]["cash"]) == Decimal("100000"))
                    first = action("funded-fill", "funded-account")
                    check("partial option fill debits multiplier premium plus fee", Decimal(first["data"]["account"]["cash"]) == Decimal("99707.4503"))
                    check("remaining intention keeps exact cash reservation", Decimal(first["data"]["account"]["reserved_cash"]) == Decimal("301.9"))
                    check("duplicate fill request returns its original snapshot", first == action("funded-fill", "funded-account"))
                    second_event = {"event_id": "authored-funded-fill-2", "client_id": "authored-funded-put-1", "kind": "FILL", "occurred_at": "2026-11-02T15:49:00Z", "quantity": "1", "price": "2.924497", "fee": "0.10"}
                    second = action("funded-fill", "funded-account", change={"event": second_event})
                    check("full fill releases remaining reservation", Decimal(second["data"]["account"]["reserved_cash"]) == 0)
                    marked = action("funded-mark", "funded-account")
                    check("funded positions reconcile cash NAV and fee expense", Decimal(marked["data"]["account"]["nav"]) == Decimal("99999.80") and Decimal(marked["data"]["account"]["cash"]) == Decimal("99414.9006"))
                    check("funding scope does not imply certification", marked["data"]["account"]["fully_funded_certified"] is False)
                    risk = None
                    if not args.skip_funded_risk:
                        risk = action("funded-risk")
                        check("funded risk replays exact saved actual-position evidence", risk == action("funded-risk-replay"))
                        check("risk remains an unapproved simulation result", risk["data"]["trading_approved"] is False)
                    check("submicrosecond execution timestamp is rejected explicitly", action("funded-initialize", "funded-account", change={"now": "2026-11-02T15:49:00.000000001Z"}, expect="ERROR", status=422)["data"] is None)
                    settled = action("funded-settle", "funded-account")
                    account = settled["data"]["account"]
                    check("European cash settlement conserves premium fee and intrinsic flows", Decimal(account["cash"]) == Decimal("101414.9006") and Decimal(account["nav"]) == Decimal("101414.9006"))
                    check("expiry closes effective quantity without inventing an OMS fill", Decimal(account["effective_positions"]["AUTHORED:SYN101:0:110:PUT"]) == 0 and len(settled["data"]["executions"]) == 2)
                    replay = action("funded-replay", "funded-account", change={"now": "2026-12-02T15:49:00Z"})
                    check("funded immutable replay preserves complete lifecycle snapshot", settled == replay)
                    check("funded simulation leaves legacy account untouched", legacy_before == context.request.get(legacy_url).json())
                    page.locator("#replay-asof").fill("2026-11-02T15:49:00Z")
                    page.locator("#replay-asof").press("Tab")
                    page.wait_for_function(ready)
                    for engine, section in [("economic-research", "research"), ("economic-research", "models"),
                                            ("microstructure-research", "stat-arb"), ("microstructure-research", "execution"),
                                            ("funded-risk", "risk"), ("funded-risk", "portfolio"),
                                            ("funded-account", "portfolio"), ("funded-account", "options"), ("funded-account", "execution")]:
                        if engine == "funded-risk" and args.skip_funded_risk:
                            continue
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
                        check(engine + " " + section + " serves saved backend evidence", current_state == "AVAILABLE", current_state)
                        if engine == "funded-account":
                            check("funded scope remains explicit in " + section, "no global funding certification" in page.locator("#truth-notice").inner_text())
                    ids = page.locator("#workflow-identities").inner_text()
                    identities = [settled["data"]["snapshot_id"], economics["data"]["report_id"], micro["data"]["report_id"]]
                    if risk:
                        identities.append(risk["data"]["report_id"])
                    check("saved workflow identities remain separate", all(value in ids for value in identities))
                    # Restart the actual local service on the same durable state.
                    server.terminate()
                    server.wait(timeout=10)
                    server = subprocess.Popen([sys.executable, "-c", code], env=env, cwd=directory, stdout=log, stderr=subprocess.STDOUT)
                    wait_ready(url, server)
                    recovered = action("funded-replay", "funded-account", change={"now": "2026-12-02T15:49:00Z", "snapshot_id": settled["data"]["snapshot_id"]})
                    check("actual service restart replays identical cash option and settlement state", recovered == settled)
                    if not args.skip_funded_risk:
                        page.locator("#analysis-engine").select_option("funded-risk")
                        page.wait_for_function(ready)
                        with page.expect_response(lambda r: "/api/v1/terminal/risk?" in r.url and "engine=funded-risk" in r.url):
                            page.locator('a[href="#risk"]').click()
                        page.wait_for_function(ready)
                        check("funded stress comparison uses actual saved current and asynchronous P&L", "asynchronous stress" in page.locator("#chart-title").inner_text())
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
                        page.screenshot(path=str(output / f"sprint6-{theme}.png"), full_page=True)
                    for width in (320, 375, 768, 1024, 1920):
                        page.set_viewport_size({"width": width, "height": 1000})
                        check(f"{width}px research workflow without document overflow", page.evaluate("document.documentElement.scrollWidth<=innerWidth"))
                        if width == 320:
                            check("mobile financial scalars remain unbroken and readable", page.evaluate("[...document.querySelectorAll('.metric-value')].filter(e=>e.textContent.length<=15).every(e=>{const r=document.createRange();r.selectNodeContents(e);return r.getClientRects().length===1})"))
                        if width == 320:
                            page.evaluate("window.scrollTo(0,0)")
                            page.screenshot(path=str(output / "sprint6-mobile-320.png"), full_page=True)
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
                    check("all twelve sections retained plus Global Intelligence and Market Charts", page.locator("#section-nav a").count() == 15 and page.locator('#section-nav a[href="#global-intelligence"]').count() == 1 and page.locator('#section-nav a[href="#observatory"]').count() == 1 and page.locator('#section-nav a[href="#charts"]').count() == 1)
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
