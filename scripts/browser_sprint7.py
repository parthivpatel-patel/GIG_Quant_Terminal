"""Actual Chromium lifecycle and independent-statement workflows, fixture-only.

All requests target an isolated local service. Public-provider fetching is never
invoked by this browser check; synthetic CSV admission tests that distinct path.
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
    parser.add_argument("--skip-observed", action="store_true", help="Component-only check; records observed-workflow exclusion explicitly.")
    args = parser.parse_args()
    out = args.evidence_dir.absolute()
    out.mkdir(parents=True, exist_ok=True)
    checks, errors, responses, accessibility = [], [], {}, []
    report = {"runner": "real-playwright-chromium", "checks": checks, "javascript_errors": errors,
              "accessibility": accessibility, "broker_connections": 0, "provider_connections": 0,
              "public_deployment": False, "input_classification": "AUTHORED_SYNTHETIC_FIXTURES",
              "observed_research": "SKIPPED_BY_ARGUMENT" if args.skip_observed else "INCLUDED",
              "formal_accessibility_certification": False}
    if args.axe:
        manifest = json.loads((args.axe.parent/"manifest.json").read_text())
        assert hashlib.sha256(args.axe.read_bytes()).hexdigest() == manifest["script_sha256"]
        report["axe_asset"] = manifest

    def check(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    with tempfile.TemporaryDirectory(prefix="gig-sprint7-browser-") as temporary:
        directory = Path(temporary)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        env = {key: value for key, value in os.environ.items() if not any(x in key for x in ("ALPACA", "FINNHUB", "FRED_API", "FINBERT_MODEL_DIR"))}
        env.update(PYTHONPATH=str(ROOT/"src")+os.pathsep+env.get("PYTHONPATH", ""),
                   GIG_BROKER_CONNECTIVITY_ENABLED="false", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        now = "2026-11-02T15:49:00Z"
        split_at, pay_at = "2026-11-04T15:49:00Z", "2026-11-06T15:49:00Z"
        server = None
        with (out/"server.txt").open("w") as log:
            def start(name):
                nonlocal server
                if server is not None:
                    server.terminate()
                    server.wait(timeout=10)
                code = ("import uvicorn;from pathlib import Path;from gig.service.app import create_app;"
                        f"uvicorn.run(create_app(state_dir=Path({str(directory/name)!r})),host='127.0.0.1',port={port},log_level='warning')")
                server = subprocess.Popen([sys.executable, "-c", code], env=env, cwd=directory, stdout=log, stderr=subprocess.STDOUT)
                wait_ready(url, server)

            try:
                start("statement-state")
                with sync_playwright() as pw:
                    browser = pw.chromium.launch(headless=True, args=["--no-sandbox"], env=env)
                    report["chromium_version"] = browser.version
                    context = browser.new_context(viewport={"width": 1440, "height": 1100}, reduced_motion="reduce")
                    context.route("**/*", lambda route: route.continue_() if route.request.url.startswith(url+"/") else route.abort())
                    page = context.new_page()
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(url+"/#research", wait_until="networkidle")
                    ready = "!document.getElementById('refresh').disabled"
                    page.wait_for_function(ready)

                    def action(kind, endpoint=None, *, change=None, expected="AVAILABLE", status=200):
                        endpoint = endpoint or kind
                        page.locator("#analytics-kind").select_option("workspace-"+kind)
                        if change:
                            payload = json.loads(page.locator("#analytics-input").input_value())
                            payload.update(change)
                            page.locator("#analytics-input").fill(json.dumps(payload))
                        with page.expect_response(lambda r: r.url.endswith("/api/v1/workspace/"+endpoint)) as exchange:
                            page.locator("#run-analysis").click()
                        page.wait_for_function("!document.getElementById('run-analysis').disabled")
                        value = exchange.value.json()
                        responses[kind] = value
                        okay = exchange.value.status == status and value["state"] == expected
                        check(kind+" explicit browser action", okay, {"status": exchange.value.status, "state": value["state"]})
                        if not okay:
                            raise AssertionError("Unexpected action response: "+kind)
                        return value

                    def stock_fill(client, event_id, *, quantity="10", price="10", fee="0.2", at=now, settlement=None):
                        change = {"now": at, "event": {"event_id": event_id, "client_id": client, "kind": "FILL", "occurred_at": at,
                                  "quantity": quantity, "price": price, "fee": fee}}
                        if settlement:
                            change["settlement_at"] = settlement
                        return action("funded-fill", "funded-account", change=change)

                    def stock_setup(*, lifecycle=False):
                        action("funded-initialize", "funded-account", change={"initial_cash": "1000"})
                        action("funded-register", "funded-account", change={"instruments": [{"instrument_id": "SYN101", "kind": "EQUITY", "currency": "USD", "multiplier": 1}]})
                        if lifecycle:
                            action("lifecycle-configure", "funded-lifecycle")
                        reserved = action("funded-reserve", "funded-account", change={"client_id": "authored-statement-stock", "instrument_id": "SYN101", "quantity": "10", "limit_price": "10", "fee_budget": "1"})
                        check("paid equity cash and fees reserve exactly", Decimal(reserved["data"]["account"]["reserved_cash"]) == 101)
                        ack = action("funded-dispatch", "funded-account")
                        check("equity acknowledgement is not a fill", ack["data"]["account"]["effective_positions"] == {})
                        stock_fill("authored-statement-stock", "authored-statement-fill", settlement="2026-11-03T21:00:00Z" if lifecycle else None)
                        marked = action("funded-mark", "funded-account", change={"marks": {"SYN101": {"price": "10", "as_of": now}}})
                        check("hand-calculated paid stock cash and NAV", Decimal(marked["data"]["account"]["cash"]) == Decimal("899.8") and Decimal(marked["data"]["account"]["nav"]) == Decimal("999.8"))
                        return marked

                    marked = stock_setup()
                    action("statement-admit")
                    matched = action("statement-reconcile")
                    check("independent literal statement agrees with engine", matched["data"]["reconciliation_state"] == "MATCH" and matched["data"]["fixture_match"] is True)
                    check("fixture agreement does not claim broker reconciliation", matched["data"]["external_reconciled"] is False)
                    check("statement comparison exactly replays", matched == action("statement-replay"))
                    page.locator("#analytics-kind").select_option("workspace-statement-admit")
                    bad = json.loads(page.locator("#analytics-input").input_value())
                    bad["statement_id"] = "authored-discrepant-equity-1"
                    bad["balances"]["cash"] = "900"
                    action("statement-admit", change=bad)
                    discrepancy = action("statement-reconcile")
                    check("cash discrepancy remains unresolved", discrepancy["data"]["reconciliation_state"] == "DISCREPANCY" and not discrepancy["data"]["fixture_match"] and not discrepancy["data"]["external_reconciled"])
                    check("saved discrepancies replay without changing account", discrepancy == action("statement-replay"))
                    restart_id = discrepancy["data"]["report_id"]
                    start("statement-state")
                    restarted = action("statement-replay", change={"report_id": restart_id})
                    check("actual service restart preserves discrepancy evidence", restarted == discrepancy)
                    check("statement comparison leaves exact funded balances unchanged", context.request.post(url+"/api/v1/workspace/funded-account", data={"operation": "state", "now": now}).json()["data"]["account"]["cash"] == marked["data"]["account"]["cash"])
                    budget = action("funded-factor-risk", "funded-risk", change={"options_run_ids": [], "marks": {"SYN101": {"price": "10", "as_of": now}}})
                    check("assumed factor budget uses actual paid equity position", budget["data"]["sector_factor_risk"]["factor_abs_budget_usd"]["market"] == 100)
                    check("declared factor loading is never represented as calibrated", budget["data"]["sector_factor_risk"]["calibrated_factor_model"] is False)

                    start("lifecycle-state")
                    page.reload(wait_until="networkidle")
                    page.wait_for_function(ready)
                    stock_setup(lifecycle=True)
                    admitted = action("lifecycle-split", "funded-lifecycle")
                    action("lifecycle-apply", "funded-lifecycle")
                    check("split uses selected admitted immutable revision", admitted["data"]["action_hash"] in page.locator("#workflow-identities").inner_text())
                    mark = action("funded-mark", "funded-account", change={"now": split_at, "marks": {"SYN101": {"price": "5", "as_of": split_at}}})
                    check("two-for-one split preserves value and basis", Decimal(mark["data"]["account"]["effective_positions"]["SYN101"]) == 20 and Decimal(mark["data"]["account"]["nav"]) == Decimal("999.8"))
                    action("lifecycle-dividend", "funded-lifecycle")
                    dividend = action("lifecycle-apply", "funded-lifecycle", change={"now": pay_at})
                    check("record-date entitlement credits only declared pay-date cash", Decimal(dividend["data"]["account"]["cash"]) == Decimal("919.8"))
                    final_mark = action("funded-mark", "funded-account", change={"now": pay_at, "marks": {"SYN101": {"price": "5", "as_of": pay_at}}})
                    check("dividend cash and paid stock NAV conserve", Decimal(final_mark["data"]["account"]["nav"]) == Decimal("1019.8"))
                    replay = action("lifecycle-replay", "funded-lifecycle", change={"now": pay_at})
                    check("lifecycle replay preserves exact saved funded state", replay["data"]["account"] == final_mark["data"]["account"])
                    action("funded-reserve", "funded-account", change={"now": pay_at, "price_asof": pay_at, "client_id": "authored-unsettled-sale", "instrument_id": "SYN101", "side": "sell", "quantity": "5", "limit_price": "5", "fee_budget": "1"})
                    action("funded-dispatch", "funded-account", change={"now": pay_at})
                    settlement_at = "2026-11-09T21:00:00Z"
                    sold = stock_fill("authored-unsettled-sale", "authored-unsettled-sale-fill", quantity="5", price="5", at=pay_at, settlement=settlement_at)
                    account = sold["data"]["account"]
                    check("sale cash is recorded but proceeds remain unavailable before settlement", Decimal(account["cash"]) == Decimal("944.6") and Decimal(account["available_cash"]) == Decimal("919.6") and Decimal(account["unsettled_sale_receivables"]) == 25)
                    settled = action("lifecycle-state", "funded-lifecycle", change={"now": settlement_at})
                    check("settlement releases only previously booked sale proceeds", Decimal(settled["data"]["account"]["cash"]) == Decimal("944.6") and Decimal(settled["data"]["account"]["available_cash"]) == Decimal("944.6"))
                    final_mark = action("funded-mark", "funded-account", change={"now": settlement_at, "marks": {"SYN101": {"price": "5", "as_of": settlement_at}}})
                    check("partial sale fees preserve exact final stock NAV", Decimal(final_mark["data"]["account"]["nav"]) == Decimal("1019.6") and Decimal(final_mark["data"]["account"]["effective_positions"]["SYN101"]) == 15)
                    replay = action("lifecycle-replay", "funded-lifecycle", change={"now": settlement_at})
                    check("settled sale and remaining positions replay exactly", replay["data"]["account"] == final_mark["data"]["account"])
                    if not args.skip_observed:
                        admitted_reference = action("observed-admit", "observed-reference")
                        researched = action("observed-research", "observed-reference")
                        check("reference fixture and research retain separate identities", admitted_reference["data"]["snapshot_id"] != researched["data"]["report_id"])
                        check("reference research replays exactly", researched == action("observed-replay", "observed-reference"))
                        check("reference fixture is never labeled provider-observed", researched["provenance"]["classification"] in {"SYNTHETIC", "MODEL_ESTIMATE"})
                        evaluated = action("observed-evaluate", "observed-reference")
                        check("baseline comparison retains original report and research boundary", evaluated["data"]["report_id"] == researched["data"]["report_id"] and not evaluated["data"]["economically_validated"] and evaluated["data"]["portfolio_pnl_state"] == "UNAVAILABLE")
                        check("baseline uncertainty uses paired date blocks", evaluated["data"]["comparisons"]["zero_return"]["method"] == "CIRCULAR_MOVING_BLOCK_BOOTSTRAP_PAIRED_OBSERVATION_DATES")
                        start("lifecycle-state")
                        check("baseline evidence survives actual service restart", evaluated == action("observed-evaluation-replay", "observed-reference"))
                    page.locator("#analysis-engine").select_option("funded-lifecycle")
                    page.locator("#replay-asof").fill(settlement_at)
                    page.locator("#replay-asof").press("Tab")
                    page.locator('a[href="#portfolio"]').click()
                    page.wait_for_function(ready)
                    page.locator("#refresh").click()
                    page.wait_for_function(ready)
                    check("lifecycle terminal shows persisted cash book", page.locator("#state-badge").inner_text() == "AVAILABLE")
                    for theme in ("dark", "light"):
                        page.evaluate("theme=>document.documentElement.dataset.theme=theme", arg=theme)
                        contrast = page.evaluate(CONTRAST_JS)
                        check(theme+" visible text contrast", all(row["passed"] for row in contrast), [r for r in contrast if not r["passed"]])
                        if args.axe:
                            page.add_script_tag(path=str(args.axe))
                            result = page.evaluate("async()=>await axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa']}})")
                            accessibility.append({"theme": theme, "version": result["testEngine"]["version"], "violations": result["violations"], "incomplete": result["incomplete"]})
                            check(theme+" automated accessibility", not result["violations"], [r["id"] for r in result["violations"]])
                        page.evaluate("window.scrollTo(0,0)")
                        page.screenshot(path=str(out/f"sprint7-{theme}.png"), full_page=True)
                    for width in (320, 375, 768, 1024, 1920):
                        page.set_viewport_size({"width": width, "height": 1000})
                        check(f"{width}px lifecycle layout without overflow", page.evaluate("document.documentElement.scrollWidth<=innerWidth"))
                        if width == 320:
                            page.evaluate("window.scrollTo(0,0)")
                            page.screenshot(path=str(out/"sprint7-mobile-320.png"), full_page=True)
                    check("all twelve terminal sections retained", page.locator("#section-nav a").count() == 12)
                    check("reduced-motion camera remains paused", page.locator("#orbit-toggle").get_attribute("aria-pressed") == "false")
                    check("no JavaScript exceptions", not errors, errors)
                    browser.close()
            except Exception as exc:
                check("workflow completed without interruption", False, type(exc).__name__+": "+str(exc))
            finally:
                if server is not None:
                    server.terminate()
                    try:
                        server.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=5)
    report["passed"] = sum(row["passed"] for row in checks)
    report["failed"] = sum(not row["passed"] for row in checks)
    (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    (out/"workflow-responses.json").write_text(json.dumps(responses, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"checks", "accessibility"}}, indent=2))
    for row in checks:
        if not row["passed"]:
            print("FAIL", row["name"], row["detail"])
    return int(report["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
