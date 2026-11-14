"""Real Chromium acceptance for corpus-local latent/hybrid retrieval and financial mentions, offline only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from browser_stage_a_regressions import isolated_server
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--axe", type=Path)
    parser.add_argument("--executable")
    args = parser.parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    checks, errors, external, accessibility = [], [], [], []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})

    with isolated_server(args.evidence_dir) as url, sync_playwright() as pw:
        options = {"headless": True, "args": ["--no-sandbox"]}
        if args.executable:
            options["executable_path"] = args.executable
        browser = pw.chromium.launch(**options)
        version = browser.version
        context = browser.new_context(viewport={"width": 1600, "height": 1100}, reduced_motion="reduce")

        def restrict(route):
            if route.request.url.startswith(url+"/"):
                route.continue_()
            else:
                external.append(route.request.url)
                route.abort()

        context.route("**/*", restrict)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url+"/#research", wait_until="networkidle")
        page.wait_for_function("!document.getElementById('refresh').disabled")

        def action(name, payload=None):
            page.locator("#analytics-kind").select_option("workspace-financial-"+name)
            if payload is not None:
                page.locator("#analytics-input").fill(json.dumps(payload))
            with page.expect_response(lambda r: r.url.endswith("/api/v1/workspace/financial-intelligence")) as response:
                page.locator("#run-analysis").click()
            page.wait_for_function("!document.getElementById('run-analysis').disabled")
            return response.value.status, response.value.json()

        status, admitted = action("reference")
        check("explicit permitted reference admission", status == 200 and admitted["state"] == "AVAILABLE")
        check("original publication remains UNKNOWN", admitted["data"]["metadata"]["publication_state"] == "UNKNOWN")
        check("CC0 archive citation is retained", admitted["data"]["metadata"]["license"] == "CC0-1.0")
        status, result = action("search")
        check("BM25 finds the actual financial record", status == 200 and result["data"]["rows"][0]["observation_date"] == "2023-01-03")
        check("no fabricated semantic or generative model", result["data"]["semantic_state"] == result["data"]["generative_state"] == "UNAVAILABLE")
        status, answer = action("answer")
        check("extractive answer quotes the normalized source", status == 200 and answer["data"]["answer"] == answer["data"]["rows"][0]["text"])
        check("actual Treasury value has a raw XML citation", "BC_10YEAR yield 3.79 percent" in answer["data"]["answer"] and "<d:BC_10YEAR" in answer["data"]["rows"][0]["source_excerpt"])
        check("answer is readable with passage identity", "3.79" in page.locator("#analysis-summary").inner_text() and answer["data"]["rows"][0]["passage_id"] in page.locator("#analysis-summary").inner_text())
        check("scrolling cited passage is keyboard focusable", page.locator("#analysis-summary-values dd").evaluate_all("nodes=>nodes.filter(n=>n.scrollHeight>n.clientHeight+1||n.scrollWidth>n.clientWidth+1).every(n=>n.tabIndex===0)"))
        snapshot = answer["data"]["snapshot_id"]
        check("snapshot joins the investigation identity strip", snapshot in page.locator("#workflow-identities").inner_text())
        status, replay = action("replay")
        check("saved search replays exactly", status == 200 and replay == answer)
        page.reload(wait_until="networkidle")
        page.wait_for_function("!document.getElementById('refresh').disabled")
        status, reopened = action("replay", {"operation": "replay", "snapshot_id": snapshot})
        check("explicit saved identity survives reload", status == 200 and reopened == answer)
        status, latent = action("latent")
        check("explicit corpus-local latent model really fitted", status == 200 and latent["data"]["model"]["model_kind"] == "CORPUS_LOCAL_LSA" and len(latent["data"]["model_id"]) == 64)
        check("effective latent dimensions are disclosed", 1 <= latent["data"]["model"]["dimensions"] <= latent["data"]["model"]["requested_dimensions"] == 8)
        check("latent scope does not imply pretrained language understanding", "not pretrained" in " ".join(latent["data"]["model"]["limitations"]))
        hybrid_request = {"operation": "answer", "query": "10-year Treasury yield 2023-12-29", "as_of": latent["data"]["as_of"], "limit": 3, "retrieval_mode": "hybrid", "latent_dimensions": 8, "extract_facts": True}
        status, hybrid = action("hybrid", hybrid_request)
        check("hybrid honors exact late date despite tied latent vectors", status == 200 and hybrid["data"]["rows"][0]["observation_date"] == "2023-12-29")
        check("explicit hybrid rank-fusion evidence", hybrid["data"]["fusion"]["method"] == "RECIPROCAL_RANK_FUSION" and hybrid["data"]["retrieval_mode"] == "hybrid")
        check("saved model identity is inspectable in terminal", hybrid["data"]["model_id"] in page.locator("#analysis-summary").inner_text())
        facts = hybrid["data"]["financial_nlp"]["facts"]
        ten_year = next(f for f in facts if f["tenor_months"] == "120" and f["observation_date"] == "2023-12-29")
        check("Decimal yield mention has exact XML citation", ten_year["decimal_value"] in ten_year["provenance"]["source_quote"] and "<d:BC_10YEAR" in ten_year["provenance"]["source_quote"] and ten_year["truth_state"] == "UNKNOWN")
        check("financial facts remain parsed mentions not issuer verification", hybrid["data"]["financial_nlp"]["issuer_linking_state"] == "UNKNOWN" and hybrid["data"]["financial_nlp"]["sentiment_state"] == "UNAVAILABLE")
        check("NLP limitations are readable", "not verified issuer facts" in page.locator("#analysis-summary").inner_text() and "text mentions only" in page.locator("#analysis-summary").inner_text())
        hybrid_id = hybrid["data"]["snapshot_id"]
        status, hybrid_replay = action("replay")
        check("hybrid model and citations replay exactly", status == 200 and hybrid_replay == hybrid)
        page.reload(wait_until="networkidle")
        page.wait_for_function("!document.getElementById('refresh').disabled")
        status, restored_hybrid = action("replay", {"operation": "replay", "snapshot_id": hybrid_id})
        check("hybrid and NLP survive page restart", status == 200 and restored_hybrid == hybrid)
        for theme in ("dark", "light"):
            page.evaluate("theme=>document.documentElement.dataset.theme=theme", arg=theme)
            if args.axe:
                manifest = json.loads((args.axe.parent/"manifest.json").read_text())
                assert hashlib.sha256(args.axe.read_bytes()).hexdigest() == manifest["script_sha256"]
                page.add_script_tag(path=str(args.axe))
                result = page.evaluate("async()=>await axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa']}})")
                accessibility.append({"theme": theme, "version": result["testEngine"]["version"], "violations": result["violations"], "incomplete": result["incomplete"]})
                check(theme+" automated accessibility", not result["violations"])
            page.locator("#workbench").screenshot(path=str(args.evidence_dir/("financial-latent-"+theme+".png")))
        for width in (320, 375, 768, 1440):
            page.set_viewport_size({"width": width, "height": 900})
            check(str(width)+"px no overflow", page.evaluate("document.documentElement.scrollWidth<=innerWidth"))
        status, absent = action("answer", {"operation": "answer", "query": "absent-nonexistent-term", "as_of": answer["data"]["as_of"]})
        check("no match produces no answer", status == 200 and absent["data"]["answer_state"] == "UNAVAILABLE" and absent["data"]["answer"] is None)
        check("no stale source quote remains", "3.79" not in page.locator("#analysis-summary").inner_text())
        status, historical = action("search", {"operation": "search", "query": "yield", "as_of": "2023-12-31T00:00:00Z"})
        check("retrospective archive cannot impersonate historic knowledge", status == 200 and historical["state"] == "UNAVAILABLE" and historical["data"] is None)
        status, unknown = action("latent", {**hybrid_request, "retrieval_mode": "latent", "query": "absent-nonexistent-term"})
        check("unseen latent vocabulary produces no invented answer", status == 200 and unknown["data"]["latent_query_state"] == "UNKNOWN_VOCABULARY" and unknown["data"]["answer"] is None)
        status, rejected = action("latent", {**hybrid_request, "retrieval_mode": "latent", "latent_dimensions": 17})
        check("latent resource overflow is rejected safely", status == 422 and rejected["state"] == "ERROR" and rejected["error"]["code"] == "INVALID_REQUEST")
        check("all sixteen existing sections retained", page.locator("#section-nav a").count() == 16)
        check("zero JavaScript errors", not errors)
        check("zero external browser requests", not external)
        browser.close()
    report = {"checks": checks, "passed": sum(c["passed"] for c in checks), "failed": sum(not c["passed"] for c in checks),
              "chromium_version": version, "accessibility": accessibility, "javascript_errors": errors, "external_requests": external,
              "source": "PINNED_CC0_TREASURY_ARCHIVE_NOT_NEW_ACQUISITION", "formal_accessibility_certified": False}
    (args.evidence_dir/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return int(report["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
