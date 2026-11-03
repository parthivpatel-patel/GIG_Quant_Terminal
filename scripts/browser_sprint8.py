"""Actual browser Sprint 8 accrual, scoring and archive workflows, fixture-only.

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
    parser.add_argument("--browser", choices=("chromium", "firefox"), default="chromium")
    parser.add_argument("--skip-observed", action="store_true", help="Component-only check; records observed-workflow exclusion explicitly.")
    args = parser.parse_args()
    out = args.evidence_dir.absolute()
    out.mkdir(parents=True, exist_ok=True)
    checks, errors, responses, accessibility = [], [], {}, []
    report = {"runner": "real-playwright-"+args.browser, "browser_name": args.browser, "checks": checks, "javascript_errors": errors,
              "accessibility": accessibility, "broker_connections": 0, "provider_connections": 0,
              "public_deployment": False, "input_classification": "AUTHORED_SYNTHETIC_FIXTURES",
              "browser_launch_timeout_ms": 30000,
              "observed_research": "SKIPPED_BY_ARGUMENT" if args.skip_observed else "INCLUDED",
              "formal_accessibility_certification": False}
    if args.axe:
        manifest = json.loads((args.axe.parent/"manifest.json").read_text())
        assert hashlib.sha256(args.axe.read_bytes()).hexdigest() == manifest["script_sha256"]
        report["axe_asset"] = manifest

    def check(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    with tempfile.TemporaryDirectory(prefix="gig-sprint8-browser-") as temporary:
        directory = Path(temporary)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        env = {key: value for key, value in os.environ.items() if not any(x in key for x in ("ALPACA", "FINNHUB", "FRED_API", "FINBERT_MODEL_DIR"))}
        env.update(PYTHONPATH=str(ROOT/"src")+os.pathsep+env.get("PYTHONPATH", ""),
                   GIG_BROKER_CONNECTIVITY_ENABLED="false", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   XDG_CACHE_HOME=str(directory/"browser-cache"), GSETTINGS_BACKEND="memory")
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
                    browser = getattr(pw, args.browser).launch(headless=True, timeout=30000, args=["--no-sandbox"] if args.browser == "chromium" else [], env=env)
                    report["browser_version"] = browser.version
                    if args.browser == "chromium":
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

                    def view(engine, section, cutoff, *, identity_key=None, identity_value=None):
                        page.locator("#analysis-engine").select_option(engine)
                        page.locator("#replay-asof").fill(cutoff)
                        page.locator("#replay-asof").press("Tab")
                        page.locator(f'#section-nav a[href="#{section}"]').click()
                        page.wait_for_function(ready)
                        with page.expect_response(lambda r: r.url.startswith(url+"/api/v1/terminal/"+section+"?") and "engine="+engine in r.url) as exchange:
                            page.locator("#refresh").click()
                        page.wait_for_function(ready)
                        value = exchange.value.json()
                        check(engine+" "+section+" section contract", exchange.value.status == 200 and value["state"] == "AVAILABLE", {"status": exchange.value.status, "state": value["state"]})
                        if identity_key:
                            check(section+" exact saved identity rendered", value.get("data", {}).get(identity_key) == identity_value)
                        check(section+" terminal is available", page.locator("#state-badge").inner_text() == "AVAILABLE")
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
                    ex_at = "2026-11-05T15:49:00Z"
                    accrued = action("lifecycle-accrue", "funded-lifecycle")
                    check("ex-date recognizes income and nonspendable receivable", Decimal(accrued["data"]["account"]["cash"]) == Decimal("899.8") and Decimal(accrued["data"]["account"]["dividend_receivable"]) == 20 and Decimal(accrued["data"]["account"]["dividend_income"]) == 20)
                    accrued_mark = action("funded-mark", "funded-account", change={"now": ex_at, "marks": {"SYN101": {"price": "5", "as_of": ex_at}}})
                    check("accrued NAV includes receivable once", Decimal(accrued_mark["data"]["account"]["nav"]) == Decimal("1019.8") and Decimal(accrued_mark["data"]["account"]["available_cash"]) == Decimal("899.8"))
                    check("financial summary discloses modeled receivable", "Dividend receivable" in page.locator("#analysis-summary-values").inner_text() and "not available cash" in page.locator("#analysis-summary-note").inner_text())
                    view("funded-lifecycle", "portfolio", ex_at)
                    check("receivable evidence table is available", page.locator('#evidence-table-view option[value="dividend_receivables"]').count() == 1)
                    page.locator("#evidence-table-view").select_option("dividend_receivables")
                    page.screenshot(path=str(out/"accrued-dividend-desktop.png"), full_page=True)
                    start("lifecycle-state")
                    accrual_replay = action("lifecycle-replay", "funded-lifecycle", change={"now": ex_at, "snapshot_id": accrued_mark["data"]["snapshot_id"]})
                    check("actual restart preserves accrued account exactly", accrual_replay["data"]["account"] == accrued_mark["data"]["account"])
                    dividend = action("lifecycle-apply", "funded-lifecycle", change={"now": pay_at})
                    check("payment clears receivable without recognizing income twice", Decimal(dividend["data"]["account"]["dividend_receivable"]) == 0 and Decimal(dividend["data"]["account"]["dividend_income"]) == 20)
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
                    start("cancel-state")
                    page.reload(wait_until="networkidle")
                    page.wait_for_function(ready)
                    stock_setup(lifecycle=True)
                    action("lifecycle-dividend", "funded-lifecycle")
                    original_accrual = action("lifecycle-accrue", "funded-lifecycle")
                    check("second account recognizes original ten-share entitlement", Decimal(original_accrual["data"]["account"]["dividend_receivable"]) == 10)
                    current_accrual = action("funded-mark", "funded-account", change={"now": ex_at, "marks": {"SYN101": {"price": "10", "as_of": ex_at}}})
                    page.locator("#analytics-kind").select_option("workspace-statement-admit")
                    literal = json.loads(page.locator("#analytics-input").input_value())
                    literal.update(statement_id="authored-accrued-statement", observed_at=ex_at, available_at=ex_at)
                    literal["balances"].update(nav="1009.8", dividend_income="10", dividend_receivable="10", settled_cash="899.8", unsettled_sale_receivables="0", unsettled_buy_payables="0")
                    literal["trade_settlements"] = [{"execution_id": "authored-statement-fill", "settlement_at": "2026-11-03T21:00:00Z"}]
                    literal["settled_trade_positions"] = {"SYN101": "10"}
                    # Supplier quantities, money and clocks are literal independent expectations.
                    # The opaque engine event identity links records; it does not provide amounts.
                    literal["lifecycle_events"] = [{"event_id": original_accrual["data"]["lifecycle_events"][-1]["event_id"], "kind": "DIVIDEND_ACCRUAL", "instrument_id": "SYN101", "action_id": "authored-dividend-SYN101", "applied_at": ex_at, "effective_at": ex_at, "known_at": split_at, "record_at": ex_at, "pay_at": pay_at, "cash_amount": "0", "entitlement_quantity": "10", "amount_per_share": "1", "income_amount": "10", "receivable_amount": "10", "total_income": "10", "status": "ACTIVE"}]
                    literal["dividend_receivables"] = [{"action_id": "authored-dividend-SYN101", "revision": 1, "instrument_id": "SYN101", "source": "AUTHORED_SYNTHETIC_ACTION_FIXTURE", "known_at": split_at, "effective_at": ex_at, "record_at": ex_at, "pay_at": pay_at, "entitlement_quantity": "10", "recognized_income": "10", "paid_cash": "0", "outstanding_receivable": "10", "status": "ACTIVE"}]
                    action("statement-admit", change=literal)
                    accrued_statement = action("statement-reconcile", change={"now": ex_at, "snapshot_id": current_accrual["data"]["snapshot_id"]})
                    check("independent accrued-dividend literal statement matches", accrued_statement["data"]["reconciliation_state"] == "MATCH" and not accrued_statement["data"]["external_reconciled"])
                    wrong = json.loads(json.dumps(literal))
                    wrong["statement_id"] = "authored-wrong-receivable"
                    wrong["balances"]["dividend_receivable"] = "0"
                    action("statement-admit", change=wrong)
                    wrong_report = action("statement-reconcile", change={"now": ex_at, "snapshot_id": current_accrual["data"]["snapshot_id"]})
                    check("incorrect dividend receivable is a discrepancy", wrong_report["data"]["reconciliation_state"] == "DISCREPANCY")
                    missing = json.loads(json.dumps(literal))
                    missing["statement_id"] = "authored-missing-receivable"
                    missing.pop("dividend_receivables")
                    action("statement-admit", change=missing)
                    missing_report = action("statement-reconcile", change={"now": ex_at, "snapshot_id": current_accrual["data"]["snapshot_id"]})
                    check("missing dividend source coverage remains unknown", missing_report["data"]["reconciliation_state"] == "UNKNOWN")
                    start("cancel-state")
                    check("accrued statement survives service restart and replays", accrued_statement == action("statement-replay", change={"report_id": accrued_statement["data"]["report_id"]}))
                    action("lifecycle-dividend-revise", "funded-lifecycle")
                    revised_at = "2026-11-05T16:00:00Z"
                    revised_accrual = action("lifecycle-accrue", "funded-lifecycle", change={"now": revised_at})
                    check("rate revision compensates rather than duplicates income", Decimal(revised_accrual["data"]["account"]["dividend_receivable"]) == 5 and Decimal(revised_accrual["data"]["account"]["dividend_income"]) == 5 and Decimal(revised_accrual["data"]["account"]["cash"]) == Decimal("899.8"))
                    action("lifecycle-dividend-cancel", "funded-lifecycle")
                    cancel_at = "2026-11-05T16:30:00Z"
                    cancelled_accrual = action("lifecycle-accrue", "funded-lifecycle", change={"now": cancel_at})
                    check("cancellation reverses the modeled receivable and income", Decimal(cancelled_accrual["data"]["account"]["dividend_receivable"]) == 0 and Decimal(cancelled_accrual["data"]["account"]["dividend_income"]) == 0 and Decimal(cancelled_accrual["data"]["account"]["cash"]) == Decimal("899.8"))
                    action("lifecycle-apply", "funded-lifecycle", change={"now": pay_at})
                    cancelled_mark = action("funded-mark", "funded-account", change={"now": pay_at, "marks": {"SYN101": {"price": "10", "as_of": pay_at}}})
                    check("cancelled payment cannot invent cash or NAV", Decimal(cancelled_mark["data"]["account"]["cash"]) == Decimal("899.8") and Decimal(cancelled_mark["data"]["account"]["nav"]) == Decimal("999.8"))
                    check("funding remains explicitly uncertified", cancelled_mark["data"]["account"]["fully_funded_certified"] is False)
                    if not args.skip_observed:
                        admitted_reference = action("observed-admit", "observed-reference")
                        view("observed-reference", "markets", admitted_reference["data"]["acquired_at"], identity_key="snapshot_id", identity_value=admitted_reference["data"]["snapshot_id"])
                        researched = action("observed-research", "observed-reference")
                        check("reference fixture and research retain separate identities", admitted_reference["data"]["snapshot_id"] != researched["data"]["report_id"])
                        check("reference research replays exactly", researched == action("observed-replay", "observed-reference"))
                        check("reference fixture is never labeled provider-observed", researched["provenance"]["classification"] in {"SYNTHETIC", "MODEL_ESTIMATE"})
                        evaluated = action("observed-evaluate", "observed-reference")
                        check("baseline comparison retains original report and research boundary", evaluated["data"]["report_id"] == researched["data"]["report_id"] and not evaluated["data"]["economically_validated"] and evaluated["data"]["portfolio_pnl_state"] == "UNAVAILABLE")
                        check("baseline uncertainty uses paired date blocks", evaluated["data"]["comparisons"]["zero_return"]["method"] == "CIRCULAR_MOVING_BLOCK_BOOTSTRAP_PAIRED_OBSERVATION_DATES")
                        for section in ("models", "research", "signals"):
                            view("observed-reference", section, evaluated["data"]["as_of"], identity_key="evaluation_id", identity_value=evaluated["data"]["evaluation_id"])
                        start("cancel-state")
                        check("baseline evidence survives actual service restart", evaluated == action("observed-evaluation-replay", "observed-reference"))
                        for model, family, table in (("lightgbm", "CROSS_SECTIONAL_RANKING", "ranking_quality_by_date"), ("garch", "CONDITIONAL_VARIANCE", "variance_scoring_observations")):
                            action("observed-research", "observed-reference", change={"model": model, "training_observations": 30, "test_observations": 10})
                            score = action("observed-evaluate", "observed-reference", change={"bootstrap_replications": 100})
                            check(model+" uses its proper scoring family", score["data"].get("scoring_family") == family and "matched_date_equal_weight_model_mse" not in (score["data"].get("metrics") or {}))
                            check(model+" remains research-only", score["data"]["economically_validated"] is False and score["data"]["trading_approved"] is False)
                            view("observed-reference", "models", score["data"]["as_of"], identity_key="evaluation_id", identity_value=score["data"]["evaluation_id"])
                            check(model+" table displays actual saved rows", page.locator(f'#evidence-table-view option[value="{table}"]').count() == 1)
                            page.locator("#evidence-table-view").select_option(table)
                            check(model+" selected scoring table has rows", page.locator("#observations-table tbody tr").count() > 0)
                            page.locator("#view-2d").click()
                            check(model+" 2D analytical view remains accessible", page.locator("#view-2d").get_attribute("aria-pressed") == "true")
                            page.locator("#view-3d").click()
                            page.locator("#analytics-chart").press("ArrowRight")
                            check(model+" 3D keyboard camera remains available", page.locator("#view-3d").get_attribute("aria-pressed") == "true")
                            page.locator("#analytics-chart").evaluate("node=>node.blur()")
                            page.evaluate("window.scrollTo(0,0)")
                            page.screenshot(path=str(out/(model+"-scoring-desktop.png")), full_page=True)
                            start("cancel-state")
                            check(model+" score exactly replays after restart", score == action("observed-evaluation-replay", "observed-reference"))
                    equity = action("equity-admit", "equity-reference")
                    check("weekly equity fixture remains synthetic and ineligible", equity["data"]["classification"] == "SYNTHETIC" and equity["data"]["point_in_time_backtest_eligible"] is False and equity["data"]["funded_allocation_eligible"] is False)
                    check("weekly equity source does not invent timestamps or currency", equity["data"]["rows"][0]["observation_time"] is None and equity["data"]["rows"][0]["currency"] is None and equity["data"]["rows"][0]["volume"] == 10000)
                    check("future source labels are quarantined", "next_weeks_close" not in equity["data"]["rows"][0] and "days_to_next_dividend" not in equity["data"]["rows"][0])
                    for section in ("markets", "research", "operations"):
                        view("observed-equity", section, equity["data"]["acquired_at"], identity_key="snapshot_id", identity_value=equity["data"]["snapshot_id"])
                    page.screenshot(path=str(out/"weekly-equity-desktop.png"), full_page=True)
                    check("weekly equity provenance limits are visible", "permanent security" in page.locator("#truth-notice").inner_text() and "Not eligible" in page.locator("#truth-notice").inner_text())
                    start("cancel-state")
                    check("weekly evidence replays after actual restart", equity == action("equity-replay", "equity-reference"))
                    page.locator("#analysis-engine").select_option("funded-lifecycle")
                    page.locator("#replay-asof").fill(pay_at)
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
                        page.screenshot(path=str(out/f"sprint8-{theme}.png"), full_page=True)
                    for width in (320, 375, 768, 1024, 1920):
                        page.set_viewport_size({"width": width, "height": 1000})
                        check(f"{width}px lifecycle layout without overflow", page.evaluate("document.documentElement.scrollWidth<=innerWidth"))
                        if width == 320:
                            page.evaluate("window.scrollTo(0,0)")
                            page.screenshot(path=str(out/"sprint8-mobile-320.png"), full_page=True)
                    check("all twelve terminal sections retained", page.locator("#section-nav a").count() == 12)
                    check("reduced-motion camera remains paused", page.locator("#orbit-toggle").get_attribute("aria-pressed") == "false")
                    page.locator("#analysis-engine").select_option("default")
                    page.locator("#replay-asof").fill("")
                    page.locator("#replay-asof").press("Tab")
                    for section in ("command-center", "markets", "signals", "portfolio", "risk", "models", "news", "stat-arb", "options", "execution", "research", "operations"):
                        with page.expect_response(lambda r, section=section: r.url.startswith(url+"/api/v1/terminal/"+section+"?") and "engine=" not in r.url):
                            page.locator(f'#section-nav a[href="#{section}"]').click()
                        page.wait_for_function(ready)
                        check(section+" remains functional with explicit state", page.locator("#state-badge").inner_text() in {"AVAILABLE", "UNAVAILABLE", "UNKNOWN", "STALE", "ERROR"})
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
