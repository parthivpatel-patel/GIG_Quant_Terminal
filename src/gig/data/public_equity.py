"""Bounded UCI weekly equity observations, with future fields quarantined.

Repository-distributed historical observations are not direct exchange-verified
quotes. Source symbols, weekly dates and dollar signs are preserved without
inventing permanent security IDs, intraday clocks, currencies or PIT vintages.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
import urllib.request
import zipfile
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from urllib.parse import urlsplit

from gig.service.workspace import _fields
from gig.timestamps import aware_timestamp

PROVIDER = "UCI_DOW_JONES_WEEKLY_2011"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/312/dow+jones+index.zip"
INFORMATION_URL = "https://archive.ics.uci.edu/dataset/312/dow+jones+index"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
DOI = "10.24432/C5788V"
ATTRIBUTION = (
    "Source: Michael Brown, Dow Jones Index [Dataset], UCI Machine Learning Repository, "
    "2013, DOI 10.24432/C5788V. CC BY 4.0. Brown, M. S., Pelosi, M. and Dirska, H. "
    "(2013), Dynamic-radius Species-conserving Genetic Algorithm for the Financial "
    "Forecasting of Dow Jones Index Stocks, LNAI 7988, 27-41. GIG parses selected "
    "weekly observations and excludes future-label/dividend columns; these changes "
    "are not UCI or exchange outputs."
)
SYMBOLS = frozenset(["AA", "AXP", "BA", "BAC", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "HD", "HPQ", "IBM", "INTC", "JNJ", "JPM", "KRFT", "KO", "MCD", "MMM", "MRK", "MSFT", "PFE", "PG", "T", "TRV", "UTX", "VZ", "WMT", "XOM"])
COLUMNS = ("quarter", "stock", "date", "open", "high", "low", "close", "volume", "percent_change_price",
           "percent_change_volume_over_last_wk", "previous_weeks_volume", "next_weeks_open", "next_weeks_close",
           "percent_change_next_weeks_price", "days_to_next_dividend", "percent_return_next_dividend")
QUARANTINED = ("next_weeks_open", "next_weeks_close", "percent_change_next_weeks_price",
               "days_to_next_dividend", "percent_return_next_dividend")


def request_window(payload):
    _fields(payload, {"provider", "symbols", "start", "end"}, {"provider", "symbols", "start", "end"})
    symbols = payload["symbols"]
    if payload["provider"] != PROVIDER or not isinstance(symbols, list) or not 1 <= len(symbols) <= 5:
        raise ValueError("fixed UCI provider and one to five source symbol labels required")
    if any(not isinstance(s, str) or s not in SYMBOLS for s in symbols) or len(set(symbols)) != len(symbols):
        raise ValueError("distinct supported historical source labels required")
    dates = []
    for key in ("start", "end"):
        value = payload[key]
        if not isinstance(value, str) or len(value) != 10:
            raise ValueError("canonical weekly observation date bounds required")
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value or not date(2011, 1, 7) <= parsed <= date(2011, 6, 24):
            raise ValueError("only documented January-June 2011 archive coverage supported")
        dates.append(parsed)
    if dates[0] >= dates[1]:
        raise ValueError("increasing bounded historical window required")
    return {"provider": PROVIDER, "symbols": sorted(symbols), "start": dates[0].isoformat(), "end": dates[1].isoformat()}


def _price(raw):
    if not isinstance(raw, str) or re.fullmatch(r"\$[0-9]{1,8}(?:\.[0-9]{1,8})?", raw) is None:
        raise ValueError("bounded source dollar-prefixed exact price required")
    try:
        value = Decimal(raw[1:])
    except InvalidOperation as exc:
        raise ValueError("exact observed price required") from exc
    if not value.is_finite() or value <= 0:
        raise ValueError("positive finite source price required")
    return value


def parse_equity_csv(text, request, acquired_at, *, observed=False):
    window = request_window(request)
    acquired = aware_timestamp(acquired_at)
    if acquired.date() < date.fromisoformat(window["end"]):
        raise ValueError("acquisition cannot precede source observation coverage")
    if not isinstance(text, str) or len(text.encode()) > 500_000:
        raise ValueError("bounded weekly source CSV required")
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != COLUMNS:
        raise ValueError("exact documented UCI weekly source columns required")
    rows, seen, counts, source_rows, auxiliary_missing = [], set(), dict.fromkeys(window["symbols"], 0), 0, 0
    for number, row in enumerate(reader):
        if number >= 1000:
            raise ValueError("source weekly row bound exceeded")
        source_rows += 1
        if None in row or any(value is None for value in row.values()):
            raise ValueError("truncated or extra source columns")
        auxiliary_missing += sum(row[name] == "" for name in COLUMNS if name not in {"stock", "date", "open", "high", "low", "close", "volume"})
        symbol = row["stock"]
        if symbol not in SYMBOLS:
            raise ValueError("unknown source symbol label")
        try:
            observed_date = datetime.strptime(row["date"], "%m/%d/%Y").date()
        except (TypeError, ValueError) as exc:
            raise ValueError("documented source weekly date required") from exc
        if not date(2011, 1, 7) <= observed_date <= date(2011, 6, 24):
            raise ValueError("weekly date outside documented archive")
        key = symbol, observed_date
        if key in seen:
            raise ValueError("duplicate symbol and weekly observation")
        seen.add(key)
        values = {name: _price(row[name]) for name in ("open", "high", "low", "close")}
        if not values["low"] <= min(values["open"], values["close"]) <= max(values["open"], values["close"]) <= values["high"]:
            raise ValueError("inconsistent observed weekly OHLC bounds")
        if re.fullmatch(r"[0-9]{1,15}", row["volume"]) is None:
            raise ValueError("exact nonnegative source weekly volume required")
        volume = int(row["volume"])
        if symbol not in counts or not window["start"] <= observed_date.isoformat() <= window["end"]:
            continue
        counts[symbol] += 1
        rows.append({"instrument_id": "DJIA_MEMBER_LABEL:"+symbol, "source_symbol": symbol,
                     "week_ending_date": observed_date.isoformat(), "source_date": row["date"], "granularity": "WEEKLY",
                     **{name: str(value) for name, value in values.items()}, "volume": volume,
                     "price_unit": "SOURCE_DOLLAR_SIGN", "currency": None, "currency_state": "UNKNOWN",
                     "observation_time": None, "observation_time_state": "UNKNOWN",
                     "acquired_at": acquired.isoformat(), "knowledge_at": acquired.isoformat()})
    if any(count < 2 for count in counts.values()) or len(rows) > 150:
        raise ValueError("each requested label needs two to twenty-five weekly observations")
    rows.sort(key=lambda row: (row["week_ending_date"], row["source_symbol"]))
    return {"schema_version": "uci-dow-jones-weekly-v1", **window, "rows": rows,
            "classification": "HISTORICAL" if observed else "SYNTHETIC", "acquired_at": acquired.isoformat(),
            "source_url": SOURCE_URL if observed else None, "information_url": INFORMATION_URL if observed else None,
            "source_authenticity": "OFFICIAL_UCI_HTTPS_ARCHIVE_NOT_DIRECT_EXCHANGE_VERIFIED" if observed else "AUTHORED_OFFLINE_FIXTURE",
            "attribution": ATTRIBUTION if observed else "Authored synthetic UCI-schema protocol fixture; not UCI observations.",
            "license": "CC-BY-4.0" if observed else "AUTHORED_FIXTURE", "license_url": LICENSE_URL if observed else None,
            "dataset_doi": DOI if observed else None, "observation_semantics": "SOURCE_REPORTED_WEEKLY_AGGREGATES_NOT_INTRADAY_QUOTES",
            "instrument_identity_state": "SOURCE_LABEL_ONLY_NO_PERMANENT_SECURITY_ID",
            "historical_publication_time": None, "historical_publication_time_state": "UNKNOWN",
            "historical_revision_knowledge_state": "UNKNOWN", "corporate_actions_state": "UNAVAILABLE",
            "price_adjustment_basis_state": "UNKNOWN", "historical_universe_membership_state": "UNKNOWN",
            "point_in_time_backtest_eligible": False, "funded_allocation_eligible": False,
            "economically_validated": False, "trading_approved": False,
            "quarantined_future_columns": list(QUARANTINED), "quarantine_policy": "PRESERVED_ONLY_IN_RAW_SOURCE_NEVER_NORMALIZED_FEATURES",
            "diagnostics": {"source_rows": source_rows, "selected_rows": len(rows), "selected_labels": len(counts),
                            "observed_week_dates": len({row["week_ending_date"] for row in rows}), "per_label_rows": counts,
                            "imputed_values": 0, "source_auxiliary_missing_cells": auxiliary_missing,
                            "source_missing_auxiliary_values_not_imputed": True}}


class _UCIRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlsplit(newurl)
        if parsed.scheme != "https" or parsed.hostname != "archive.ics.uci.edu" or parsed.username or parsed.password:
            raise ValueError("equity evidence redirect outside fixed official archive")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_public_equity(payload):
    window = request_window(payload)
    opener = urllib.request.build_opener(_UCIRedirect())
    responses = {}
    values = {}
    for name, url, limit in (("license_page", INFORMATION_URL, 500_000), ("archive", SOURCE_URL, 250_000)):
        request = urllib.request.Request(url, headers={"User-Agent": "GIG-Quant-Terminal/0.1 permitted archive research"})
        with opener.open(request, timeout=25) as response:
            if response.status != 200 or urlsplit(response.geturl()).hostname != "archive.ics.uci.edu":
                raise ValueError("official equity archive evidence unavailable")
            raw = response.read(limit+1)
            if len(raw) > limit:
                raise ValueError("equity archive evidence byte bound exceeded")
            values[name] = raw
            responses[name] = {"status": 200, "url": response.geturl(), "sha256": hashlib.sha256(raw).hexdigest(),
                               "acquired_at": datetime.now(UTC).isoformat()}
    license_text = values["license_page"].decode("utf-8")
    if "creativecommons.org/licenses/by/4.0/legalcode" not in license_text or "CC BY 4.0" not in license_text:
        raise ValueError("explicit official dataset reuse license not verified")
    raw = values["archive"]
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        if ({entry.filename for entry in entries} != {"dow_jones_index.data", "dow_jones_index.names"}
                or len(entries) != 2 or any(entry.file_size > 500_000 for entry in entries)):
            raise ValueError("unexpected bounded official equity archive")
        csv_bytes = archive.read("dow_jones_index.data")
    acquired = responses["archive"]["acquired_at"]
    document = parse_equity_csv(csv_bytes.decode("utf-8"), window, acquired, observed=True)
    document.update(raw_sha256=hashlib.sha256(raw).hexdigest(), csv_sha256=hashlib.sha256(csv_bytes).hexdigest(),
                    source_responses=responses)
    return document, raw


def admit_equity_fixture(payload):
    required = {"provider", "symbols", "start", "end", "csv_text", "acquired_at"}
    _fields(payload, required, required)
    window = {key: payload[key] for key in ("provider", "symbols", "start", "end")}
    document = parse_equity_csv(payload["csv_text"], window, payload["acquired_at"])
    raw = payload["csv_text"].encode()
    document.update(raw_sha256=hashlib.sha256(raw).hexdigest(), csv_sha256=hashlib.sha256(raw).hexdigest())
    return document, raw
