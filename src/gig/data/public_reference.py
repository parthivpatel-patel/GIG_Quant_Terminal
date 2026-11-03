"""Official ECB daily reference observations, without invented market clocks."""
from __future__ import annotations

import csv
import hashlib
import io
import math
import urllib.request
import zipfile
from datetime import UTC, date, datetime
from urllib.parse import urlsplit

from gig.service.workspace import _fields
from gig.timestamps import aware_timestamp

PROVIDER = "ECB_REFERENCE_FX"
SOURCE_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
LICENSE_URL = "https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html"
INFO_URL = "https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html"
CURRENCIES = frozenset({"USD", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"})
ATTRIBUTION = "Source: European Central Bank. ECB information is available free of charge. Derived features, returns and models are modifications by GIG, not ECB outputs."


def request_window(payload):
    currencies = payload.get("currencies")
    if payload.get("provider") != PROVIDER or not isinstance(currencies, list) or not 1 <= len(currencies) <= 3:
        raise ValueError("explicit ECB provider and one to three currencies required")
    if any(not isinstance(c, str) or c not in CURRENCIES for c in currencies) or len(set(currencies)) != len(currencies):
        raise ValueError("distinct supported reference quote currencies required")
    dates = []
    for field in ("start", "end"):
        value = payload.get(field)
        if not isinstance(value, str) or len(value) != 10:
            raise ValueError("exact ISO observation dates required")
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError("canonical daily observation date required")
        dates.append(parsed)
    if not dates[0] < dates[1] or (dates[1]-dates[0]).days > 366:
        raise ValueError("bounded historical date window of at most one year required")
    return {"provider": PROVIDER, "currencies": sorted(currencies), "start": dates[0].isoformat(), "end": dates[1].isoformat()}


def parse_csv(text, request, acquired_at, *, observed):
    window = request_window(request)
    acquired = aware_timestamp(acquired_at)
    if date.fromisoformat(window["end"]) > acquired.date():
        raise ValueError("reference observation window lies after acquisition")
    if not isinstance(text, str) or len(text.encode()) > 8_000_000:
        raise ValueError("bounded CSV reference evidence required")
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    if (reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames))
            or not {"Date", *window["currencies"]} <= set(reader.fieldnames)):
        raise ValueError("ECB date and requested currency columns required")
    rows, seen, missing = [], set(), 0
    for number, raw in enumerate(reader):
        if number >= 10000:
            raise ValueError("reference source row bound exceeded")
        if raw.get("Date") is None:
            raise ValueError("missing source observation date")
        event = date.fromisoformat(raw["Date"])
        if event.isoformat() != raw["Date"]:
            raise ValueError("noncanonical source observation date")
        if not window["start"] <= raw["Date"] <= window["end"]:
            continue
        if raw["Date"] in seen:
            raise ValueError("duplicate source observation date")
        seen.add(raw["Date"])
        for currency in window["currencies"]:
            value = raw.get(currency)
            if value is None:
                raise ValueError("truncated reference record")
            value = value.strip()
            price = None if value in {"", "N/A", "-"} else float(value)
            if price is not None and (not math.isfinite(price) or not 0 < price < 1e9):
                raise ValueError("positive finite bounded observed reference value required")
            missing += price is None
            rows.append({"instrument_id": "ECB:EXR:"+currency+":PER_EUR", "observation_date": raw["Date"],
                         "quote_currency": currency, "base_currency": "EUR", "value": price,
                         "observation_time": None, "observation_time_state": "UNKNOWN",
                         "acquired_at": acquired.isoformat(), "knowledge_at": acquired.isoformat(),
                         "historical_publication_time": None, "historical_publication_state": "UNKNOWN",
                         "volume": None, "volume_state": "UNAVAILABLE"})
    if not 2 <= len(seen) <= 300:
        raise ValueError("requested reference window has insufficient or oversized observations")
    rows.sort(key=lambda r: (r["observation_date"], r["instrument_id"]))
    return {"schema_version": "ecb-reference-evidence-v1", **window, "rows": rows,
        "classification": "HISTORICAL" if observed else "SYNTHETIC", "acquired_at": acquired.isoformat(),
        "source_url": SOURCE_URL if observed else None, "source_authenticity": "OFFICIAL_HTTPS_FETCH" if observed else "AUTHORED_OFFLINE_FIXTURE_NOT_PROVIDER_OBSERVED",
        "attribution": ATTRIBUTION, "license_url": LICENSE_URL, "reference_information_url": INFO_URL,
        "observation_semantics": "DAILY_REFERENCE_QUOTE_FOREIGN_CURRENCY_PER_EUR_INFORMATION_ONLY_NOT_EXECUTABLE",
        "historical_knowledge": "UNKNOWN_NO_POINT_IN_TIME_PUBLICATION_VINTAGES",
        "point_in_time_backtest_eligible": False, "funded_allocation_eligible": False,
        "diagnostics": {"observed_dates": len(seen), "rows": len(rows), "missing_quotes": missing,
                        "volume_supplied": False, "intraday_time_supplied": False, "imputed_values": 0}}


class _ECBRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlsplit(newurl)
        if parsed.scheme != "https" or parsed.hostname != "www.ecb.europa.eu":
            raise ValueError("reference provider redirect outside fixed official host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_reference(payload):
    request = request_window(payload)
    opener = urllib.request.build_opener(_ECBRedirect())
    wire_request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "GIG-Quant-Terminal/0.1 public-reference research", "Accept": "application/zip"})
    with opener.open(wire_request, timeout=30) as response:
        if response.status != 200:
            raise ValueError("reference source did not return a successful response")
        raw = response.read(4_000_001)
        if len(raw) > 4_000_000:
            raise ValueError("compressed reference evidence exceeds bound")
        acquired = datetime.now(UTC).isoformat()
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            if len(entries) != 1 or entries[0].filename != "eurofxref-hist.csv" or entries[0].file_size > 8_000_000:
                raise ValueError("unexpected bounded official reference archive")
            csv_bytes = archive.read(entries[0])
        document = parse_csv(csv_bytes.decode("utf-8-sig"), request, acquired, observed=True)
        document["raw_sha256"] = hashlib.sha256(raw).hexdigest()
        document["csv_sha256"] = hashlib.sha256(csv_bytes).hexdigest()
        document["response"] = {"status": 200, "url": response.geturl(), "content_type": response.headers.get("Content-Type")}
        return document, raw


def admit_fixture(payload):
    required = {"provider", "currencies", "start", "end", "acquired_at", "csv_text"}
    _fields(payload, required, required)
    document = parse_csv(payload["csv_text"], payload, payload["acquired_at"], observed=False)
    raw = payload["csv_text"].encode()
    document["raw_sha256"] = hashlib.sha256(raw).hexdigest()
    document["csv_sha256"] = document["raw_sha256"]
    return document, raw
