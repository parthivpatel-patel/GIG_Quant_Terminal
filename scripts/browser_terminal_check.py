"""Real Chromium terminal checks against an isolated loopback research service.

Requires Playwright/Chromium and an optional integrity-verified axe-core script.
No user browser profile, broker credentials, provider feed, or public port is used.
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
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ("command-center", "markets", "signals", "portfolio", "risk", "models",
            "news", "stat-arb", "options", "execution", "research", "operations")
CONTRAST_JS = r"""() => {
  const context=document.createElement('canvas').getContext('2d');
  context.canvas.width=context.canvas.height=1;
  const rgba=color=>{context.clearRect(0,0,1,1);context.fillStyle=color;
    context.fillRect(0,0,1,1);return [...context.getImageData(0,0,1,1).data].map((v,i)=>i===3?v/255:v)};
  const blend=(a,b)=>[...a.slice(0,3).map((v,i)=>v*a[3]+b[i]*(1-a[3])),1];
  const luminance=c=>c.slice(0,3).map(v=>{v/=255;return v<=.04045?v/12.92:((v+.055)/1.055)**2.4})
    .reduce((v,c,i)=>v+c*[.2126,.7152,.0722][i],0);
  const output=[];
  for(const node of document.querySelectorAll('body *')) {
    const rect=node.getBoundingClientRect(),style=getComputedStyle(node);
    if(!rect.width||!rect.height||style.visibility==='hidden'||node.closest('[aria-hidden="true"]'))continue;
    const direct=[...node.childNodes].filter(x=>x.nodeType===Node.TEXT_NODE).map(x=>x.textContent).join('').trim();
    if(!direct||['SCRIPT','STYLE','OPTION'].includes(node.tagName))continue;
    const lineage=[];for(let p=node;p;p=p.parentElement)lineage.unshift(p);
    let background=[255,255,255,1],opacity=1;
    for(const p of lineage){const s=getComputedStyle(p);background=blend(rgba(s.backgroundColor),background);opacity*=+s.opacity;}
    const text=rgba(style.color);text[3]*=opacity;const foreground=blend(text,background);
    const a=luminance(foreground),b=luminance(background),ratio=(Math.max(a,b)+.05)/(Math.min(a,b)+.05);
    const size=parseFloat(style.fontSize),weight=parseInt(style.fontWeight)||400;
    const threshold=size>=24||(size>=18.66&&weight>=700)?3:4.5;
    output.push({selector:node.id?'#'+node.id:node.tagName.toLowerCase()+'.'+node.className,
      text:direct.slice(0,80),ratio:Number(ratio.toFixed(3)),threshold,passed:ratio+.01>=threshold});
  }
  return output;
}"""


def wait_ready(url: str, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("The isolated local server stopped before readiness")
        try:
            with urllib.request.urlopen(url + "/healthz", timeout=1) as response:
                if json.load(response)["ok"]:
                    return
        except OSError:
            time.sleep(.1)
    raise RuntimeError("The isolated local service did not become ready")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--axe", type=Path)
    args = parser.parse_args()
    if args.axe:
        manifest = json.loads((args.axe.parent / "manifest.json").read_text())
        if hashlib.sha256(args.axe.read_bytes()).hexdigest() != manifest["script_sha256"]:
            raise ValueError("Accessibility script integrity mismatch")
    evidence = args.evidence_dir.absolute()
    evidence.mkdir(parents=True, exist_ok=True)
    checks, errors, accessibility = [], [], []
    report = {"runner": "real-playwright-chromium", "checks": checks,
              "browser_errors": errors, "accessibility": accessibility,
              "broker_connections": 0, "public_deployment": False,
              "formal_accessibility_certification": False,
              "axe_asset": manifest if args.axe else None}

    def check(name: str, condition: bool, detail=None):
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    with tempfile.TemporaryDirectory(prefix="gig-browser-state-") as temporary:
        directory = Path(temporary)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        environment = dict(os.environ)
        for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "GIG_ALPACA_API_KEY",
                     "GIG_ALPACA_SECRET_KEY", "FRED_API_KEY", "FINNHUB_API_KEY",
                     "GIG_FRED_API_KEY", "GIG_FINNHUB_API_KEY", "GIG_FINBERT_MODEL_DIR"):
            environment.pop(name, None)
        environment.update({"PYTHONPATH": str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", ""),
                            "GIG_BROKER_CONNECTIVITY_ENABLED": "false",
                            "GIG_DATA_DIR": str(directory / "data"),
                            "GIG_DB_PATH": str(directory / "data" / "isolated.duckdb"),
                            "GIG_RESULTS_DIR": str(directory / "results")})
        code = ("from pathlib import Path;import uvicorn;from gig.service.app import create_app;"
                f"uvicorn.run(create_app(state_dir=Path({str(directory / 'state')!r})),"
                f"host='127.0.0.1',port={port},log_level='warning')")
        with (evidence / "server.txt").open("w") as log:
            process = subprocess.Popen([sys.executable, "-c", code], cwd=directory,
                                       env=environment, stdout=log, stderr=subprocess.STDOUT)
            try:
                wait_ready(url, process)
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"], env=environment)
                    report["browser_version"] = browser.version
                    context = browser.new_context(viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")
                    context.route("**/*", lambda route: route.continue_() if route.request.url.startswith(url + "/") else route.abort())
                    page = context.new_page()
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(url, wait_until="networkidle")
                    page.wait_for_function("document.getElementById('state-badge').textContent!=='LOADING'")
                    check("initial environment is admitted local data", page.locator("#data-mode").input_value() == "local")

                    def navigate(section):
                        with page.expect_response(lambda response: f"/api/v1/terminal/{section}?" in response.url):
                            if page.url.endswith("#" + section) or (section == "command-center" and "#" not in page.url):
                                page.locator("#refresh").click()
                            else:
                                page.locator(f'a.nav-link[href="#{section}"]').click()
                        page.wait_for_function("id=>document.querySelector('.nav-link[aria-current=page]').hash==='#'+id&&document.getElementById('state-badge').textContent!=='LOADING'&&!document.getElementById('refresh').disabled", arg=section)

                    for mode in ("local", "synthetic"):
                        with page.expect_response(lambda response, mode=mode: "/api/v1/terminal/" in response.url and "mode=" + mode in response.url):
                            page.locator("#data-mode").select_option(mode)
                        page.wait_for_function("!document.getElementById('refresh').disabled")
                        for section in SECTIONS:
                            navigate(section)
                            state = page.locator("#state-badge").inner_text()
                            check(f"{mode} section {section} responds", state in {"AVAILABLE", "UNAVAILABLE", "UNKNOWN", "STALE"}, state)
                            if mode == "synthetic" and section != "operations":
                                check(f"{section} synthetic source disclosure", "SYNTHETIC" in page.locator("#truth-notice").inner_text())

                    navigate("markets")
                    canvas = page.locator("#analytics-chart")
                    check("factor view has finite backend observations", page.locator("#chart-empty").is_hidden())
                    first = canvas.screenshot()
                    canvas.focus()
                    page.keyboard.press("ArrowRight")
                    check("keyboard rotates real canvas", first != canvas.screenshot())
                    page.locator("#view-2d").click()
                    check("2D analytical alternative", page.locator("#view-2d").get_attribute("aria-pressed") == "true" and "2D projection" in page.locator("#chart-legend").inner_text())
                    page.locator("#view-3d").click()
                    navigate("options")
                    check("volatility surface is actual rendered chart", page.locator("#chart-empty").is_hidden() and "volatility" in page.locator("#chart-legend").inner_text().lower())

                    # The real browser loads and posts the terminal's authored
                    # fixture JSON. No data is auto-admitted by changing sections.
                    page.locator("#data-mode").select_option("local")
                    for action in ("bars", "news", "research", "simulate"):
                        page.locator("#analytics-kind").select_option("workspace-" + action)
                        page.locator("#run-analysis").click()
                        page.wait_for_function("!document.getElementById('run-analysis').disabled")
                        output = page.locator("#analytics-output").inner_text()
                        check(f"real browser workspace {action}", output.startswith("AVAILABLE"), output[:120])
                    page.locator("#replay-asof").fill("2026-09-29T20:00:00+00:00")
                    page.locator("#replay-asof").press("Tab")
                    for section in ("markets", "signals", "portfolio", "risk", "models", "news", "stat-arb", "research", "execution"):
                        navigate(section)
                        check(f"persisted local {section}", page.locator("#state-badge").inner_text() == "AVAILABLE")

                    for kind in ("calendar", "corporate-actions"):
                        page.locator("#analytics-kind").select_option(kind)
                        page.locator("#run-analysis").click()
                        page.wait_for_function("!document.getElementById('run-analysis').disabled")
                        check(f"real browser {kind} calculation", page.locator("#analytics-output").inner_text().startswith("AVAILABLE"))

                    # Execute the newly integrated controls through browser fetch,
                    # including a real CPU training result followed by saved inference.
                    sequence_model = None
                    for action in ("ingest", "universe", "train-sequence", "predict-sequence", "finbert", "execution-engine"):
                        page.locator("#analytics-kind").select_option("workspace-" + action)
                        if action == "predict-sequence" and sequence_model:
                            request = json.loads(page.locator("#analytics-input").input_value())
                            request["model_id"] = sequence_model
                            page.locator("#analytics-input").fill(json.dumps(request))
                        with page.expect_response(lambda response, action=action: response.url.endswith("/api/v1/workspace/" + action)) as exchange:
                            page.locator("#run-analysis").click()
                        page.wait_for_function("!document.getElementById('run-analysis').disabled")
                        payload = exchange.value.json()
                        expected = {"ERROR"} if action == "finbert" else {"AVAILABLE"}
                        matches = payload.get("state") in expected and payload.get("state") in page.locator("#analytics-output").inner_text()
                        if action == "finbert":
                            matches = matches and payload.get("data") is None and exchange.value.status == 503 and payload.get("error", {}).get("code") == "DEPENDENCY_UNAVAILABLE"
                        check("real browser integrated workspace " + action, matches, {"status": exchange.value.status, "state": payload.get("state"), "expected": sorted(expected)})
                        if action == "train-sequence" and payload.get("state") == "AVAILABLE":
                            sequence_model = payload["data"]["model_id"]

                    # Explicit local fixture setup for the new persisted rates view.
                    # These authored quotes are SYNTHETIC, not a public acquisition.
                    from gig.data.ingestion import ProviderWorkspace

                    xml = b'<feed xmlns:m="urn:metadata" xmlns:d="urn:data">'
                    for day, delta in (("2025-01-02", 0), ("2025-01-03", .1)):
                        xml += (f'<entry><m:properties><d:NEW_DATE>{day}T00:00:00</d:NEW_DATE>'
                                f'<d:BC_10YEAR>{4.25+delta}</d:BC_10YEAR>'
                                f'<d:BC_2YEAR>{4.5+delta}</d:BC_2YEAR>'
                                '<d:BC_3MONTH>4.6</d:BC_3MONTH></m:properties></entry>').encode()
                    xml += b'</feed>'
                    ProviderWorkspace(directory / "state").ingest_treasury_bytes(
                        xml, year=2025, observed_at="2026-09-29T20:00:00Z")
                    navigate("markets")
                    with page.expect_response(lambda response: "engine=provider" in response.url) as exchange:
                        page.locator("#analysis-engine").select_option("provider")
                    page.wait_for_function("!document.getElementById('refresh').disabled")
                    rates = exchange.value.json()
                    check("persisted rates chart retains synthetic fixture truth",
                          rates.get("state") == "AVAILABLE" and rates["provenance"]["classification"] == "SYNTHETIC"
                          and len(rates["data"]["chart"]["points"]) == 6)
                    check("observed par-yield chart renders with declared units",
                          page.locator("#chart-empty").is_hidden()
                          and "percent per year" in page.locator("#chart-legend").inner_text())
                    canvas = page.locator("#analytics-chart")
                    first = canvas.screenshot()
                    canvas.focus()
                    page.keyboard.press("ArrowLeft")
                    check("persisted yield surface rotates actual canvas", first != canvas.screenshot())
                    page.locator("#view-2d").click()
                    check("persisted yields have a 2D projection and exact table",
                          "2D projection" in page.locator("#chart-legend").inner_text()
                          and page.locator("#observations-table tbody tr").count() == 6)
                    page.locator("#view-3d").click()
                    submitted = []

                    def track_ingestion(request):
                        if request.method == "POST" and request.url.endswith("/api/v1/workspace/ingest"):
                            submitted.append(request.url)

                    page.on("request", track_ingestion)
                    page.locator("#analytics-kind").select_option("workspace-treasury")
                    check("public Treasury control declares acquisition and requires explicit submission",
                          json.loads(page.locator("#analytics-input").input_value()) == {
                              "provider": "us_treasury", "mode": "PUBLIC_OBSERVED", "year": 2025}
                          and "fixed official Treasury" in page.locator("#workbench-scope").inner_text()
                          and not submitted)
                    page.remove_listener("request", track_ingestion)
                    for theme in ("dark", "light"):
                        page.evaluate("theme=>{localStorage.setItem('gig-theme',theme);document.documentElement.dataset.theme=theme}", arg=theme)
                        page.locator("#refresh").click()
                        page.wait_for_function("!document.getElementById('refresh').disabled")
                        contrast = page.evaluate(CONTRAST_JS)
                        failures = [row for row in contrast if not row["passed"]]
                        accessibility.append({"theme": theme, "computed_contrast": contrast})
                        check(theme + " visible text contrast", not failures, failures)
                        if args.axe:
                            page.add_script_tag(path=str(args.axe))
                            scan = page.evaluate("async()=>await axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa']}})")
                            accessibility[-1]["axe"] = {"version": scan["testEngine"]["version"], "violations": scan["violations"], "incomplete": scan["incomplete"], "passes": len(scan["passes"])}
                            check(theme + " axe WCAG A/AA automated rules", not scan["violations"], [v["id"] for v in scan["violations"]])
                        page.screenshot(path=str(evidence / f"desktop-{theme}.png"), full_page=True)

                    page.set_viewport_size({"width": 320, "height": 800})
                    page.locator("#analytics-kind").select_option("workspace-research")
                    page.screenshot(path=str(evidence / "mobile-320.png"), full_page=True)
                    check("320px page has no horizontal overflow", page.evaluate("document.documentElement.scrollWidth<=innerWidth"), page.evaluate("({width:innerWidth,scrollWidth:document.documentElement.scrollWidth})"))
                    controls = page.locator("#analytics-kind,#analytics-input,#load-example,#run-analysis").evaluate_all("nodes=>nodes.map(node=>{const r=node.getBoundingClientRect();return {id:node.id,left:r.left,right:r.right,width:r.width}})")
                    check("320px workbench controls stay visible without panel clipping", all(row["left"] >= 0 and row["right"] <= 320 for row in controls), controls)
                    page.locator(".skip-link").focus()
                    bounds = page.locator(".skip-link").bounding_box()
                    check("mobile skip link stays within viewport", bounds["x"] >= 0 and bounds["x"] + bounds["width"] <= 320, bounds)
                    previous = page.url
                    page.keyboard.press("Enter")
                    check("skip navigation preserves selected section", page.url == previous and page.locator("#content").evaluate("node=>node===document.activeElement"))
                    for width in (375, 768, 1024, 1920):
                        page.set_viewport_size({"width": width, "height": 1000})
                        check(f"{width}px responsive page has no horizontal overflow",
                              page.evaluate("document.documentElement.scrollWidth<=innerWidth"),
                              page.evaluate("({width:innerWidth,scrollWidth:document.documentElement.scrollWidth})"))
                    page.set_viewport_size({"width": 1440, "height": 1000})

                    # Intercept only isolated local requests to exercise failure
                    # and loading presentation in a real rendering engine.
                    pattern = "**/api/v1/terminal/*"
                    page.route(pattern, lambda route: route.abort("failed"))
                    page.locator("#refresh").click()
                    page.wait_for_function("!document.getElementById('refresh').disabled")
                    check("network failure clears old success data", page.locator("#state-badge").inner_text() == "ERROR" and page.locator("#metrics").locator(".metric-card").count() == 0)
                    page.unroute(pattern)
                    page.route(pattern, lambda route: (time.sleep(.5), route.continue_()))
                    page.locator("#refresh").click()
                    check("real loading disables refresh", page.locator("#refresh").is_disabled())
                    page.wait_for_function("!document.getElementById('refresh').disabled")
                    page.unroute(pattern)
                    for engine in ("sequence", "finbert", "provider", "execution-engine"):
                        with page.expect_response(lambda response, engine=engine: response.url.startswith(url + "/api/v1/terminal/") and "engine=" + engine in response.url) as exchange:
                            page.locator("#analysis-engine").select_option(engine)
                        page.wait_for_function("document.getElementById('state-badge').textContent!=='LOADING'")
                        payload = exchange.value.json()
                        check("local evidence selection routes " + engine, payload.get("state") == page.locator("#state-badge").inner_text(), {"status": exchange.value.status, "state": payload.get("state"), "scope": "request routing and displayed availability, not trained-model certification"})
                    with page.expect_response(lambda response: "engine=provider" in response.url):
                        page.locator("#analysis-engine").select_option("provider")
                    page.wait_for_function("!document.getElementById('refresh').disabled")
                    navigate("research")
                    page.locator("#universe-id").fill("authored-synthetic-universe")
                    with page.expect_response(lambda response: "universe_id=authored-synthetic-universe" in response.url) as exchange:
                        page.locator("#universe-id").press("Tab")
                    page.wait_for_function("!document.getElementById('refresh').disabled")
                    check("browser replays explicitly selected point-in-time universe", exchange.value.json().get("state") == "AVAILABLE" and page.locator("#state-badge").inner_text() == "AVAILABLE")
                    page.locator("#data-mode").select_option("synthetic")
                    page.wait_for_function("document.getElementById('state-badge').textContent!=='LOADING'")
                    check("synthetic mode disables local model selection", page.locator("#analysis-engine").is_disabled() and page.locator("#analysis-engine").input_value() == "default")
                    check("no browser JavaScript exceptions", not errors, errors)
                    browser.close()
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    report["passed"] = sum(row["passed"] for row in checks)
    report["failed"] = sum(not row["passed"] for row in checks)
    (evidence / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"checks", "accessibility"}}, indent=2))
    for row in checks:
        if not row["passed"]:
            print("FAIL", row["name"], json.dumps(row["detail"]))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
