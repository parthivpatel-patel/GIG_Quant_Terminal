"""Official CFTC weekly aggregate positioning; never prints or trade flow.

Acquisition is explicit, fixed-host and bounded. Normal reading uses a pinned
public-domain government report. Report dates are dates, not publication clocks.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PROVIDER = "CFTC"
DATASET_ID = "gpe5-46if"
CONTRACT_CODE = "099741"
MARKET = "EURO FX - CHICAGO MERCANTILE EXCHANGE"
UNITS = "(CONTRACTS OF EUR 125,000)"
POLICY_URL = "https://www.cftc.gov/WebPolicy/index.htm"
INFO_URL = "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm"
METADATA_URL = "https://publicreporting.cftc.gov/api/views/gpe5-46if.json"
QUERY = {"$where": "cftc_contract_market_code = '099741' AND report_date_as_yyyy_mm_dd between '2023-01-01T00:00:00' and '2023-12-31T23:59:59'",
         "$order": "report_date_as_yyyy_mm_dd ASC", "$limit": "100"}
SOURCE_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json?" + urllib.parse.urlencode(QUERY)
FIXTURE = Path(__file__).parents[1] / "research_org/fixtures/observed_cftc_2023.json"
PROVENANCE = FIXTURE.with_name("observed_cftc_2023.metadata.json")
MAX_BYTES = 500_000


@dataclass(frozen=True)
class ContractIdentity:
    """Code-reviewed report identity, including exact source units."""
    code: str
    market: str
    units: str
    commodity_code: str


EURO_FX = ContractIdentity(CONTRACT_CODE, MARKET, UNITS, "099")
SP500_EMINI = ContractIdentity("13874A", "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE",
                              "($50 X S&P 500 INDEX)", "138")
SUPPORTED_IDENTITIES = (EURO_FX, SP500_EMINI)

CATEGORIES = (
    ("dealer", "dealer_positions_long_all", "dealer_positions_short_all", "dealer_positions_spread_all"),
    ("asset_manager", "asset_mgr_positions_long", "asset_mgr_positions_short", "asset_mgr_positions_spread"),
    ("leveraged_money", "lev_money_positions_long", "lev_money_positions_short", "lev_money_positions_spread"),
    ("other_reportable", "other_rept_positions_long", "other_rept_positions_short", "other_rept_positions_spread"),
    ("nonreportable", "nonrept_positions_long_all", "nonrept_positions_short_all", None),
)


def _integer(value: Any) -> int:
    if not isinstance(value, str) or re.fullmatch(r"0|[1-9][0-9]{0,11}", value) is None:
        raise ValueError("CFTC positions require exact nonnegative integer strings")
    return int(value)


def canonical_date(value: Any) -> date:
    if not isinstance(value, str) or len(value) != 10:
        raise ValueError("canonical report date required")
    try:
        parsed = date.fromisoformat(value)
    except (ValueError, TypeError):
        raise ValueError("canonical report date required") from None
    if parsed.isoformat() != value:
        raise ValueError("canonical report date required")
    return parsed


def normalize_rows(rows: Any, *, identity: ContractIdentity = EURO_FX) -> list[dict[str, Any]]:
    if identity not in SUPPORTED_IDENTITIES:
        raise ValueError("unregistered CFTC contract identity")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 100:
        raise ValueError("bounded CFTC observation list required")
    result: list[dict[str, Any]] = []
    seen: set[date] = set()
    identities: set[str] = set()
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("CFTC record required")
        clock = raw.get("report_date_as_yyyy_mm_dd")
        if not isinstance(clock, str) or re.fullmatch(r"2023-\d{2}-\d{2}T00:00:00\.000", clock) is None:
            raise ValueError("CFTC report date-only source encoding required")
        observed = canonical_date(clock[:10])
        report_week = observed.isocalendar().week
        if raw.get("yyyy_report_week_ww") != f"2023 Report Week {report_week:02}":
            raise ValueError("inconsistent CFTC reporting week identity")
        report_identity = raw.get("id")
        if (raw.get("cftc_contract_market_code") != identity.code or raw.get("market_and_exchange_names") != identity.market
                or raw.get("contract_units") != identity.units or raw.get("futonly_or_combined") != "FutOnly"
                or not isinstance(raw.get("cftc_market_code"), str) or raw["cftc_market_code"].strip() != "CME"
                or not isinstance(raw.get("cftc_commodity_code"), str) or raw["cftc_commodity_code"].strip() != identity.commodity_code
                or report_identity != observed.strftime("%y%m%d") + identity.code + "F"):
            raise ValueError("unsupported or inconsistent official contract identity")
        if observed in seen or report_identity in identities:
            raise ValueError("duplicate CFTC report identity")
        seen.add(observed)
        identities.add(report_identity)
        oi = _integer(raw.get("open_interest_all"))
        categories: list[dict[str, Any]] = []
        for category, long_key, short_key, spread_key in CATEGORIES:
            long, short = _integer(raw.get(long_key)), _integer(raw.get(short_key))
            spread = _integer(raw.get(spread_key)) if spread_key else None
            if long > oi or short > oi or (spread is not None and spread > oi):
                raise ValueError("position exceeds reported open interest")
            categories.append({"id": category, "long": long, "short": short, "spread": spread, "net": long - short})
        report_long = sum(c["long"] + (c["spread"] or 0) for c in categories[:-1])
        report_short = sum(c["short"] + (c["spread"] or 0) for c in categories[:-1])
        if (report_long != _integer(raw.get("tot_rept_positions_long_all"))
                or report_short != _integer(raw.get("tot_rept_positions_short"))
                or report_long + categories[-1]["long"] != oi
                or report_short + categories[-1]["short"] != oi):
            raise ValueError("CFTC open-interest conservation failed")
        result.append({"report_id": report_identity, "report_date": observed.isoformat(), "report_week": report_week, "open_interest": oi,
                       "observation_time": None, "publication_time": "UNKNOWN", "categories": categories})
    return sorted(result, key=lambda row: row["report_date"])


def analyze(rows: list[dict[str, Any]], report_through: str | None = None) -> list[dict[str, Any]]:
    """Derived descriptive metrics use strictly earlier report observations.

    No publication-vintage eligibility is implied by sorting observation dates.
    A cutoff selects a retrospective sample, not a knowledge-time reconstruction.
    """
    cutoff = canonical_date(report_through) if report_through is not None else None
    prior: dict[str, list[int]] = {category[0]: [] for category in CATEGORIES}
    result: list[dict[str, Any]] = []
    previous_date = None
    for row in rows:
        observed = canonical_date(row["report_date"])
        if previous_date is not None and observed <= previous_date:
            raise ValueError("positioning analysis requires strictly increasing report dates")
        previous_date = observed
        if cutoff is not None and observed > cutoff:
            continue
        categories = []
        # Official reporting weeks handle holiday shifts (e.g. July 3, 2023).
        # Missing reporting weeks remain gaps, not apparent one-week changes.
        consecutive = bool(result) and row["report_week"] == result[-1]["report_week"] + 1
        for category in row["categories"]:
            history = prior[category["id"]]
            value = category["net"]
            percentile = ((sum(p < value for p in history) + .5 * sum(p == value for p in history)) / len(history)
                          if len(history) >= 8 else None)
            categories.append({**category, "net_share_of_open_interest": value / row["open_interest"] if row["open_interest"] else None,
                               "weekly_net_change": value - history[-1] if history and consecutive else None,
                               "prior_only_percentile": percentile, "prior_observations": len(history)})
            history.append(value)
        result.append({**row, "categories": categories})
    return result


def load_positions() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = FIXTURE.read_bytes()
    if len(raw) > MAX_BYTES:
        raise ValueError("CFTC source exceeds bound")
    meta = json.loads(PROVENANCE.read_text())
    if hashlib.sha256(raw).hexdigest() != meta["raw_sha256"]:
        raise ValueError("CFTC source hash mismatch")
    if (meta.get("dataset_id") != DATASET_ID or meta.get("source_url") != SOURCE_URL
            or meta.get("license_url") != POLICY_URL or meta.get("contract_market_code") != CONTRACT_CODE
            or meta.get("source_authenticity") != "OFFICIAL_HTTPS_FETCH"):
        raise ValueError("CFTC provenance identity mismatch")
    from gig.timestamps import aware_timestamp
    aware_timestamp(meta["acquired_at"])
    rows = normalize_rows(json.loads(raw))
    if len(rows) != 52 or rows[0]["report_date"] != "2023-01-03" or rows[-1]["report_date"] != "2023-12-26":
        raise ValueError("incomplete pinned 2023 CFTC report sample")
    return meta, rows


def dataset_readiness() -> dict[str, Any]:
    try:
        meta, rows = load_positions()
    except (OSError, ValueError, KeyError, TypeError):
        return {"state": "UNAVAILABLE", "reason": "Pinned CFTC dataset missing or rejected; no observations substituted."}
    return {"state": "READY", "dataset_id": DATASET_ID, "observations": len(rows), "raw_sha256": meta["raw_sha256"]}


class _CFTCRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlsplit(newurl)
        if (parsed.scheme != "https" or parsed.hostname != "publicreporting.cftc.gov"
                or parsed.port not in {None, 443} or parsed.username is not None or parsed.password is not None):
            raise ValueError("CFTC redirect outside approved host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_positions_2023() -> tuple[bytes, dict[str, Any]]:
    """Explicit public acquisition only. Never called by status, UI or workers."""
    from datetime import UTC, datetime
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "GIG-Quant-Terminal/0.1 public-research", "Accept": "application/json"})
    try:
        with urllib.request.build_opener(_CFTCRedirect()).open(request, timeout=30) as response:
            if response.status != 200:
                raise ValueError("CFTC acquisition unavailable")
            raw = response.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise ValueError("CFTC source exceeds bound")
            normalize_rows(json.loads(raw))
            return raw, {"status": 200, "url": response.geturl(), "acquired_at": datetime.now(UTC).isoformat(),
                         "raw_sha256": hashlib.sha256(raw).hexdigest()}
    except Exception:
        raise ValueError("CFTC acquisition unavailable") from None
