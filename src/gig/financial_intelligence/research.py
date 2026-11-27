"""Cited statement analysis, multi-document digests, explicit monitoring and investigations.

Everything here is extractive and deterministic. Numbers are Decimal mentions
quoted from cited passages, never verified issuer facts. Nothing is generated
by a language model, nothing runs in the background, and text found in a
document is never executed as an instruction.
"""
from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

from gig.financial_intelligence.retrieval import rank_candidates
from gig.financial_intelligence.store import FinancialCorpus, _sha, digest
from gig.service.serialization import dumps, loads
from gig.timestamps import aware_timestamp

IMPLEMENTATION = "gig.financial-research-automation.v1"
MAX_QUESTIONS, MAX_PERIODS, MAX_ROWS = 8, 8, 512
ITEMS = {
    "revenue": ("revenue", "revenues", "total revenue", "total revenues", "net sales", "net revenue", "net revenues"),
    "cost_of_revenue": ("cost of revenue", "cost of revenues", "cost of sales", "cost of goods sold"),
    "gross_profit": ("gross profit",),
    "operating_income": ("operating income", "income from operations", "operating profit"),
    "net_income": ("net income", "net earnings", "net profit"),
    "diluted_eps": ("diluted eps", "diluted earnings per share", "earnings per share, diluted"),
    "total_assets": ("total assets",),
    "total_liabilities": ("total liabilities",),
    "shareholders_equity": ("total shareholders' equity", "total stockholders' equity", "shareholders' equity",
                            "stockholders' equity", "total equity"),
    "operating_cash_flow": ("net cash provided by operating activities", "operating cash flow", "cash flow from operations"),
    "capital_expenditures": ("capital expenditures", "purchases of property and equipment"),
    "cash": ("cash and cash equivalents",),
}
ALIASES = {alias: item for item, aliases in ITEMS.items() for alias in aliases}
PER_SHARE = {"diluted_eps"}
PERIOD = re.compile(r"(?:(?P<part>Q[1-4]|H[12])\s*)?(?:FY\s*)?(?P<year>(?:19|20)[0-9]{2})", re.I)
SCALE = re.compile(r"\bin\s+(?P<scale>thousands|millions|billions)\b(?:\s+of\s+(?P<currency>USD|U\.S\.\s+dollars|dollars))?"
                   r"|\b(?P<currency2>USD)\s+(?P<scale2>thousands|millions|billions)\b", re.I)
SCALES = {"thousands": Decimal(1000), "millions": Decimal(1000000), "billions": Decimal(1000000000)}
VALUE = re.compile(r"(?P<open>\()?\s*(?:US\$|\$)?\s*(?P<number>(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?)\s*(?P<close>\))?")
MISSING = {"", "-", "—", "–", "n/a", "na", "nm"}
RATIOS = (("gross_margin", "gross_profit", "revenue"), ("operating_margin", "operating_income", "revenue"),
          ("net_margin", "net_income", "revenue"), ("liabilities_to_equity", "total_liabilities", "shareholders_equity"))
LIMITS = ["Values are parsed text mentions with citations, not verified issuer facts, audited figures or XBRL",
          "A stated scale applies to every later row of the document until another scale statement appears",
          "Only the listed line-item aliases, period labels and explicit scale statements are recognized",
          "No language model generated or interpreted any text; document instructions are never executed"]


def _cells(text: str) -> list[str]:
    parts = re.split(r"\s*\|\s*", text) if "|" in text else re.split(r"\t+|\s{2,}", text.strip())
    return [part.strip() for part in parts]


def _period(cell: str):
    match = PERIOD.fullmatch(cell.strip())
    if match is None:
        return None
    part = (match["part"] or "FY").upper()
    return {"label": f"{part}{match['year']}", "year": int(match["year"]),
            "order": int(match["year"])*10+(int(part[1]) if part[0] == "Q" else {"FY": 0, "H1": 1, "H2": 2}[part]),
            "kind": "FISCAL_YEAR" if part == "FY" else "HALF" if part[0] == "H" else "QUARTER"}


def _number(cell: str):
    """Exact Decimal or None. Parentheses mean a negative; a partial match is not a number."""
    match = VALUE.fullmatch(cell.strip())
    if match is None or bool(match["open"]) != bool(match["close"]):
        return None
    try:
        value = Decimal(match["number"].replace(",", ""))
    except InvalidOperation:
        return None
    return -value if match["open"] else value


def _text(value: Decimal) -> str:
    return format(value.normalize(), "f") if value == value.to_integral() else format(value, "f")


def _ratio(numerator: Decimal, denominator: Decimal) -> str | None:
    if denominator == 0:
        return None
    with localcontext() as context:
        context.prec = 28
        return _text((numerator/denominator).quantize(Decimal("0.000001")))


def statements(corpus: FinancialCorpus, document_id: str) -> dict[str, Any]:
    """Line items by period for one immutable document, with derived ratios and consistency checks."""
    document = corpus.document(document_id)
    periods: list[dict[str, Any]] = []
    scale, currency, scale_citation = None, None, None
    facts: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    order: dict[str, int] = {}
    kinds: dict[str, str] = {}
    for passage in document["passages"][:MAX_ROWS]:
        stated = SCALE.search(passage["text"])
        if stated:
            scale = SCALES[(stated["scale"] or stated["scale2"]).lower()]
            currency = "USD" if (stated["currency"] or stated["currency2"]) else "UNSPECIFIED_CURRENCY"
            scale_citation = passage["passage_id"]
        cells = _cells(passage["text"])
        if len(cells) < 2:
            continue
        header = [_period(cell) for cell in cells[1:]]
        if all(header) and len({p["label"] for p in header}) == len(header) <= MAX_PERIODS:
            periods = header
            continue
        label = re.sub(r"\s*[:*†]+$", "", cells[0]).casefold().replace("’", "'")
        item = ALIASES.get(label)
        if item is None or not periods or len(cells)-1 != len(periods):
            continue
        for period, cell in zip(periods, cells[1:], strict=True):
            if cell.casefold() in MISSING:
                continue
            quoted = _number(cell)
            if quoted is None:
                continue
            scaled = quoted if item in PER_SHARE or scale is None else quoted*scale
            fact = {"item": item, "label_mention": cells[0], "period": period["label"], "period_kind": period["kind"],
                    "quoted": cell, "decimal_value": _text(scaled),
                    "unit": "PER_SHARE_UNSPECIFIED_CURRENCY" if item in PER_SHARE else currency or "UNSPECIFIED",
                    "scale_state": "NOT_APPLIED_PER_SHARE" if item in PER_SHARE else "STATED" if scale else "UNSPECIFIED",
                    "scale_passage_id": None if item in PER_SHARE else scale_citation,
                    "kind": "PARSED_STATEMENT_MENTION_NOT_VERIFIED_FACT",
                    "citation": {k: passage[k] for k in ("document_id", "passage_id", "source_start", "source_end",
                                                         "raw_sha256", "source_url")}}
            key = (item, period["label"])
            if key in seen:
                if seen[key]["decimal_value"] != fact["decimal_value"]:
                    conflicts.append({"item": item, "period": period["label"], "first": seen[key]["decimal_value"],
                                      "later": fact["decimal_value"], "later_passage_id": passage["passage_id"]})
                continue
            seen[key] = fact
            facts.append(fact)
            order[period["label"]], kinds[period["label"]] = period["order"], period["kind"]
    values = {key: Decimal(fact["decimal_value"]) for key, fact in seen.items()}
    labels = sorted(order, key=lambda label: order[label])
    derived, checks = [], []
    for label in labels:
        for name, top, bottom in RATIOS:
            if (top, label) in values and (bottom, label) in values:
                derived.append({"metric": name, "period": label, "value": _ratio(values[top, label], values[bottom, label]),
                                "unit": "RATIO", "inputs": [top, bottom], "kind": "ARITHMETIC_ON_PARSED_MENTIONS"})
        if ("operating_cash_flow", label) in values and ("capital_expenditures", label) in values:
            spend = values["capital_expenditures", label]
            derived.append({"metric": "free_cash_flow", "period": label,
                            "value": _text(values["operating_cash_flow", label]-abs(spend)), "unit": currency or "UNSPECIFIED",
                            "inputs": ["operating_cash_flow", "capital_expenditures"],
                            "convention": "OPERATING_CASH_FLOW_MINUS_ABSOLUTE_CAPITAL_EXPENDITURES",
                            "kind": "ARITHMETIC_ON_PARSED_MENTIONS"})
        for name, total, parts in (("gross_profit_identity", "gross_profit", ("revenue", "cost_of_revenue")),
                                   ("balance_sheet_identity", "total_assets", ("total_liabilities", "shareholders_equity"))):
            if all((k, label) in values for k in (total, *parts)):
                first, second = (values[k, label] for k in parts)
                expected = first-abs(second) if name == "gross_profit_identity" else first+second
                checks.append({"check": name, "period": label, "reported": _text(values[total, label]),
                               "recomputed": _text(expected), "difference": _text(values[total, label]-expected),
                               "state": "CONSISTENT" if values[total, label] == expected else "INCONSISTENT"})
    for item in ITEMS:
        for kind in ("FISCAL_YEAR", "HALF", "QUARTER"):
            # Like-for-like only: the same period part one year earlier, never a sequential quarter.
            for label in [p for p in labels if kinds[p] == kind and (item, p) in values]:
                prior = next((p for p in labels if kinds[p] == kind and order[p] == order[label]-10 and (item, p) in values), None)
                if prior and values[item, prior] > 0:
                    derived.append({"metric": item+"_year_over_year_growth", "period": label, "prior_period": prior,
                                    "value": _ratio(values[item, label]-values[item, prior], values[item, prior]),
                                    "unit": "RATIO", "inputs": [item], "kind": "ARITHMETIC_ON_PARSED_MENTIONS"})
    metadata = document["metadata"]
    body = {"implementation": IMPLEMENTATION, "operation": "statements", "document_id": metadata["document_id"],
            "document_title": metadata["title"], "classification": metadata["classification"], "as_of": metadata["known_at"],
            "periods": labels, "facts": facts, "derived": derived, "consistency_checks": checks, "conflicts": conflicts,
            "statement_state": "AVAILABLE" if facts else "NO_RECOGNIZED_STATEMENT_ROWS",
            "recognized_items": sorted({fact["item"] for fact in facts}), "supported_items": sorted(ITEMS),
            "xbrl_state": "UNAVAILABLE", "issuer_identity_state": "UNKNOWN", "content_truth_state": "UNKNOWN",
            "generative_state": "UNAVAILABLE", "trading_approved": False, "limitations": LIMITS}
    return {**body, "record_id": digest(body)}


def _passages(corpus: FinancialCorpus, ids: list[str], at):
    output = []
    for identity in ids:
        document = corpus.document(identity)
        if aware_timestamp(document["metadata"]["known_at"]) > at:
            raise ValueError("future financial evidence cannot enter retrieval")
        output.append(document)
    return output


def multi_document_digest(corpus: FinancialCorpus, query: str, as_of: str, *, per_document: int = 2,
                          expand: bool = False, document_ids: list[str] | None = None) -> dict[str, Any] | None:
    """Top cited passages per visible document plus cross-document statement agreement."""
    at = aware_timestamp(as_of)
    if type(per_document) is not int or not 1 <= per_document <= 5:
        raise ValueError("digest requires one to five passages per document")
    ids = corpus._visible_ids(at) if document_ids is None else [_sha(i) for i in document_ids]
    rank_candidates([], query, expand=expand)
    if not ids:
        return None
    documents = _passages(corpus, ids, at)
    ranked, terms = rank_candidates([p for d in documents for p in d["passages"]], query, expand=expand)
    sections, unmatched = [], []
    reported: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for document in documents:
        metadata = document["metadata"]
        rows = [p for p in ranked if p["document_id"] == metadata["document_id"]][:per_document]
        if not rows:
            unmatched.append(metadata["document_id"])
        else:
            sections.append({"document_id": metadata["document_id"], "title": metadata["title"],
                             "source_url": metadata["source_url"], "known_at": metadata["known_at"],
                             "classification": metadata["classification"], "rows": rows})
        for fact in statements(corpus, metadata["document_id"])["facts"]:
            reported.setdefault((fact["item"], fact["period"]), []).append(
                {"document_id": metadata["document_id"], "decimal_value": fact["decimal_value"], "unit": fact["unit"],
                 "passage_id": fact["citation"]["passage_id"]})
    sections.sort(key=lambda s: (-s["rows"][0]["score"], s["document_id"]))
    comparison = [{"item": item, "period": period, "reports": reports,
                   "state": "SINGLE_DOCUMENT" if len(reports) == 1 else
                            "AGREE" if len({(r["decimal_value"], r["unit"]) for r in reports}) == 1 else "DIFFER"}
                  for (item, period), reports in sorted(reported.items())]
    body = {"implementation": IMPLEMENTATION, "operation": "digest", "query": query, "as_of": at.isoformat(),
            "per_document": per_document, "expand": expand, "expanded_terms": terms, "corpus_document_ids": ids,
            "corpus_identity": digest(ids), "sections": sections, "unmatched_document_ids": unmatched,
            "documents_matched": len(sections), "documents_visible": len(ids), "statement_comparison": comparison,
            "statement_disagreements": sum(c["state"] == "DIFFER" for c in comparison),
            "digest_kind": "EXTRACTIVE_MULTI_DOCUMENT_DIGEST_NOT_GENERATED", "score_kind": "BM25_RELEVANCE_NOT_FINANCIAL_CONFIDENCE",
            "digest_state": "AVAILABLE" if sections else "NO_MATCH", "generative_state": "UNAVAILABLE",
            "content_truth_state": "UNKNOWN", "trading_approved": False,
            "classification": "SYNTHETIC" if any(d["metadata"]["classification"] == "SYNTHETIC" for d in documents) else "HISTORICAL",
            "limitations": LIMITS}
    return {**body, "record_id": digest(body)}


def _settings(payload: dict[str, Any]) -> dict[str, Any]:
    return {"limit": payload.get("limit", 5), "expand": payload.get("expand", False),
            "retrieval_mode": payload.get("retrieval_mode", "bm25"), "latent_dimensions": payload.get("latent_dimensions", 16)}


def watch_create(corpus: FinancialCorpus, query: str, as_of: str, **settings) -> dict[str, Any] | None:
    """Pin a saved search as the baseline of an explicitly checked watch."""
    settings = _settings(settings)
    baseline = corpus.search(query, as_of, **settings)
    if baseline is None:
        return None
    body = {"implementation": IMPLEMENTATION, "operation": "watch", "query": query, "settings": settings,
            "as_of": baseline["as_of"], "baseline_snapshot_id": baseline["snapshot_id"],
            "baseline_passage_ids": [r["passage_id"] for r in baseline["rows"]],
            "baseline_document_ids": baseline["corpus_document_ids"], "classification": baseline["classification"],
            "monitoring_state": "EXPLICIT_CHECK_ONLY_NO_DAEMON", "source_acquisition_state": "NOT_RUNNING",
            "trading_approved": False}
    return {**body, "record_id": digest(body)}


def watch_check(corpus: FinancialCorpus, watch: dict[str, Any], as_of: str) -> dict[str, Any]:
    """Rerun a watch at a later explicit cutoff and report what changed against its baseline."""
    at = aware_timestamp(as_of)
    if watch.get("operation") != "watch" or at < aware_timestamp(watch["as_of"]):
        raise ValueError("watch check requires a saved watch and a cutoff at or after its baseline")
    current = corpus.search(watch["query"], at.isoformat(), **watch["settings"])
    if current is None:
        raise ValueError("watch corpus is unavailable at the requested cutoff")
    before, after = set(watch["baseline_passage_ids"]), [r["passage_id"] for r in current["rows"]]
    new_rows = [r for r in current["rows"] if r["passage_id"] not in before]
    documents = set(watch["baseline_document_ids"])
    body = {"implementation": IMPLEMENTATION, "operation": "watch-check", "watch_id": watch["record_id"],
            "query": watch["query"], "as_of": current["as_of"], "baseline_as_of": watch["as_of"],
            "baseline_snapshot_id": watch["baseline_snapshot_id"], "check_snapshot_id": current["snapshot_id"],
            "new_rows": new_rows, "new_passage_ids": [r["passage_id"] for r in new_rows],
            "dropped_passage_ids": sorted(before-set(after)),
            "new_document_ids": sorted(set(current["corpus_document_ids"])-documents),
            "retired_document_ids": sorted(documents-set(current["corpus_document_ids"])),
            "classification": current["classification"], "monitoring_state": "EXPLICIT_CHECK_ONLY_NO_DAEMON",
            "source_acquisition_state": "NOT_RUNNING", "notification_state": "NOT_SENT_LOCAL_RECORD_ONLY",
            "trading_approved": False}
    body["alert_state"] = "CHANGED" if any(body[k] for k in ("new_passage_ids", "dropped_passage_ids",
                                                             "new_document_ids", "retired_document_ids")) else "UNCHANGED"
    return {**body, "record_id": digest(body)}


def investigate(corpus: FinancialCorpus, title: str, questions: list[str], as_of: str) -> dict[str, Any] | None:
    """Run a bounded list of explicit questions and file one cited, replayable report."""
    if (not isinstance(title, str) or not 1 <= len(title) <= 256 or not isinstance(questions, list)
            or not 1 <= len(questions) <= MAX_QUESTIONS or len(set(questions)) != len(questions)
            or any(not isinstance(q, str) for q in questions)):
        raise ValueError("investigation requires a title and one to eight distinct questions")
    findings: list[dict[str, Any]] = []
    cited: list[str] = []
    for question in questions:
        result = corpus.search(question, as_of, limit=3, answer=True, extract_facts=True)
        if result is None:
            return None
        findings.append({"question": question, "snapshot_id": result["snapshot_id"], "answer": result["answer"],
                         "answer_state": result["answer_state"], "answer_kind": "EXTRACTIVE_PASSAGE_ONLY",
                         "citations": [{k: r[k] for k in ("document_id", "passage_id", "source_start", "source_end",
                                                          "raw_sha256", "source_url", "score")} for r in result["rows"]],
                         "parsed_mention_count": len(result["financial_nlp"]["facts"])})
        cited.extend(r["document_id"] for r in result["rows"])
    reviewed = []
    for identity in sorted(set(cited)):
        summary = statements(corpus, identity)
        reviewed.append({"document_id": identity, "document_title": summary["document_title"],
                         "statement_record_id": summary["record_id"], "statement_state": summary["statement_state"],
                         "recognized_items": summary["recognized_items"], "periods": summary["periods"],
                         "inconsistent_checks": [c for c in summary["consistency_checks"] if c["state"] == "INCONSISTENT"],
                         "conflicts": summary["conflicts"]})
    at = aware_timestamp(as_of)
    body = {"implementation": IMPLEMENTATION, "operation": "investigation", "title": title, "as_of": at.isoformat(),
            "questions": questions, "findings": findings, "documents_reviewed": reviewed,
            "answered": sum(f["answer_state"] == "AVAILABLE" for f in findings),
            "unanswered_questions": [f["question"] for f in findings if f["answer_state"] != "AVAILABLE"],
            "report_kind": "EXTRACTIVE_CITED_FINDINGS_NOT_GENERATED_ANALYSIS",
            "autonomy_state": "EXPLICIT_SINGLE_REQUEST_NO_BACKGROUND_AGENT",
            "director_binding_state": "NOT_BOUND_HUMAN_APPROVED_OBJECTIVE_REQUIRED",
            "document_instruction_policy": "NEVER_EXECUTED", "generative_state": "UNAVAILABLE",
            "content_truth_state": "UNKNOWN", "trading_approved": False,
            "classification": "SYNTHETIC" if any(corpus.document(i)["metadata"]["classification"] == "SYNTHETIC"
                                                 for i in corpus._visible_ids(at)) else "HISTORICAL",
            "limitations": LIMITS}
    return {**body, "record_id": digest(body)}


def save(corpus: FinancialCorpus, record: dict[str, Any]) -> dict[str, Any]:
    """Persist one immutable research record under its content hash."""
    body = {k: v for k, v in record.items() if k != "record_id"}
    identity, wire = digest(body), dumps(body)
    if identity != record["record_id"]:
        raise ValueError("financial research record identity differs from its content")
    with closing(corpus._connect(write=True)) as db:
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute("INSERT OR IGNORE INTO research_records VALUES(?,?,?)", (identity, body["operation"], wire))
            stored = db.execute("SELECT body FROM research_records WHERE id=?", (identity,)).fetchone()
            if stored is None or stored[0] != wire:
                raise ValueError("financial research record conflicts with immutable content")
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise
    return record


def load(corpus: FinancialCorpus, record_id: str) -> dict[str, Any]:
    identity = _sha(record_id)
    if not corpus.path.exists():
        raise FileNotFoundError("financial research record unavailable")
    try:
        with closing(corpus._connect()) as db:
            row = db.execute("SELECT body FROM research_records WHERE id=?", (identity,)).fetchone()
    except sqlite3.OperationalError:
        row = None
    if row is None:
        raise FileNotFoundError("financial research record unavailable")
    body = loads(row[0])
    if digest(body) != identity or body.get("implementation") != IMPLEMENTATION:
        raise ValueError("financial research record integrity failed")
    return {**body, "record_id": identity}


def replay(corpus: FinancialCorpus, record_id: str) -> dict[str, Any]:
    """Recompute a saved record from its pinned inputs; any difference is an integrity failure."""
    saved = load(corpus, record_id)
    operation = saved["operation"]
    again: dict[str, Any] | None
    if operation == "statements":
        again = statements(corpus, saved["document_id"])
    elif operation == "digest":
        again = multi_document_digest(corpus, saved["query"], saved["as_of"], per_document=saved["per_document"],
                                      expand=saved["expand"], document_ids=saved["corpus_document_ids"])
    elif operation == "watch":
        corpus.replay(saved["baseline_snapshot_id"])
        again = saved
    elif operation == "watch-check":
        baseline, current = corpus.replay(saved["baseline_snapshot_id"]), corpus.replay(saved["check_snapshot_id"])
        before = {r["passage_id"] for r in baseline["rows"]}
        if (saved["new_passage_ids"] != [r["passage_id"] for r in current["rows"] if r["passage_id"] not in before]
                or saved["new_document_ids"] != sorted(set(current["corpus_document_ids"])-set(baseline["corpus_document_ids"]))):
            raise ValueError("financial research replay changed")
        again = saved
    elif operation == "investigation":
        for finding in saved["findings"]:
            snapshot = corpus.replay(finding["snapshot_id"])
            if snapshot["answer"] != finding["answer"] or snapshot["query"] != finding["question"]:
                raise ValueError("financial research replay changed")
        for reviewed in saved["documents_reviewed"]:
            if statements(corpus, reviewed["document_id"])["record_id"] != reviewed["statement_record_id"]:
                raise ValueError("financial research replay changed")
        again = saved
    elif operation in ("transcript", "compare"):
        from gig.financial_intelligence import documents

        again = (documents.transcript(corpus, saved["document_id"]) if operation == "transcript"
                 else documents.compare(corpus, saved["base_document_id"], saved["revised_document_id"]))
    else:
        raise ValueError("unsupported financial research record")
    if again is None or again["record_id"] != saved["record_id"]:
        raise ValueError("financial research replay changed")
    return {**saved, "replay_state": "REPRODUCED_FROM_PINNED_INPUTS"}
