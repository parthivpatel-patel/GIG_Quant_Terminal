"""Bounded provider ingestion, acquisition-time evidence, and durable replay.

Alpaca is an offline protocol fixture here: no key, environment credential, or
account connection is accepted. Treasury observations are official public rates,
not equity bars. Acquisition time is never backdated to a historical event.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from gig.artifacts import write_immutable
from gig.data.calendars import ExchangeCalendar
from gig.data.validated import DataSnapshot, PointInTimeStore, admit_bars
from gig.research.rates import TENOR_YEARS, curve_analytics, quoted_yield
from gig.service.contracts import Contract, State
from gig.service.serialization import dumps, loads, public_error
from gig.timestamps import aware_timestamp

MAX_PAGES = 10
MAX_PAGE_BYTES = 1_048_576
MAX_TREASURY_BYTES = 4_194_304
_TREASURY_ENDPOINT = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
)
_TREASURY_LICENSE_SOURCE = (
    "https://catalog.data.gov/dataset/interest-rate-statistics-daily-treasury-yield-curve-rates"
)
_SYMBOL = re.compile(r"[A-Z][A-Z0-9.-]{0,14}\Z")
_TENORS = frozenset(TENOR_YEARS)


def _bytes(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()


def _hash(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _text(value: Any, bound: int = 128) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > bound
        or any(ord(c) < 32 for c in value)
    ):
        raise ValueError("bounded nonempty identity text required")
    return value


def _fields(payload: Any, allowed: set[str], required: set[str]) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) - allowed or not required <= set(payload):
        raise ValueError("unknown or missing provider request fields")
    return payload


def _integer(value: Any, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("integer parameter exceeds declared bounds")
    return value


@dataclass(frozen=True, slots=True)
class Instrument:
    instrument_id: str
    symbol: str
    calendar: str
    currency: str

    def __post_init__(self) -> None:
        _text(self.instrument_id)
        if not isinstance(self.symbol, str) or _SYMBOL.fullmatch(self.symbol) is None:
            raise ValueError("provider symbol must be explicit uppercase ASCII")
        if self.calendar != "XNYS":
            raise ValueError("this equity adapter requires an explicit XNYS session mapping")
        if self.currency != "USD":
            raise ValueError("this US equity adapter requires explicit USD identity")


@dataclass(frozen=True, slots=True)
class Page:
    body: bytes
    observed_at: str
    status: int = 200
    retry_after_seconds: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.body, bytes) or len(self.body) > MAX_PAGE_BYTES:
            raise ValueError("page body exceeds declared bound")
        aware_timestamp(self.observed_at)
        _integer(self.status, 100, 599)
        _integer(self.retry_after_seconds, 0, 60)


Transport = Callable[[str | None], Page]


def provider_capabilities() -> list[dict[str, Any]]:
    return [
        {
            "provider": "alpaca_iex",
            "feed": "IEX",
            "consolidated_sip": False,
            "classification": "HISTORICAL",
            "asset_class": "US_EQUITY",
            "timeframes": ["1Min", "1Day"],
            "provider_timestamp_semantics": "BAR_START_OR_NEW_YORK_SESSION_LABEL",
            "normalized_timestamp_semantics": "CLOSING_OBSERVATION",
            "daily_calendar_rules": "EXPLICIT_XNYS_INSTALLED_VERSION",
            "credential_required": True,
            "connection_state": "BLOCKED_PENDING_APPROVAL",
            "account_connection": False,
            "offline_fixture_classification": "SYNTHETIC",
            "provider_verified": False,
        },
        {
            "provider": "us_treasury",
            "feed": "DAILY_TREASURY_PAR_YIELD_CURVE",
            "classification": "HISTORICAL",
            "asset_class": "INTEREST_RATE",
            "units": "percent_per_year",
            "credential_required": False,
            "license": "CC0-1.0; source attribution retained",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "license_source_url": _TREASURY_LICENSE_SOURCE,
            "supported_tenors": sorted(_TENORS),
            "event_time_precision": "DATE_ONLY",
            "historical_publication_time": "UNKNOWN",
        },
    ]


class AlpacaIEXAdapter:
    """Offline parser for documented Alpaca historical IEX bar pagination."""

    source = "alpaca_iex:offline-fixture"

    @staticmethod
    def request_url(
        instruments: list[Instrument],
        start: str,
        end: str,
        timeframe: str,
        cursor: str | None = None,
    ) -> str:
        if timeframe not in {"1Min", "1Day"}:
            raise ValueError("unsupported equity timeframe")
        first, last = aware_timestamp(start), aware_timestamp(end)
        if first > last or (last - first).days > 366:
            raise ValueError("equity request window must be ordered and at most 366 days")
        if not 1 <= len(instruments) <= 32 or len({x.symbol for x in instruments}) != len(
            instruments
        ):
            raise ValueError("unique explicit instrument mappings required")
        if cursor is not None:
            _text(cursor, 512)
        query = {
            "symbols": ",".join(x.symbol for x in instruments),
            "start": first.isoformat(),
            "end": last.isoformat(),
            "timeframe": timeframe,
            "feed": "iex",
            "adjustment": "raw",
            "limit": "1000",
            "sort": "asc",
        }
        if cursor is not None:
            query["page_token"] = cursor
        return "https://data.alpaca.markets/v2/stocks/bars?" + urllib.parse.urlencode(query)

    @classmethod
    def parse(
        cls, page: Page, instruments: list[Instrument], *, start: str, end: str, timeframe: str
    ) -> tuple[DataSnapshot, str | None]:
        if page.status != 200:
            raise ValueError("successful provider page required")
        cls.request_url(instruments, start, end, timeframe)
        raw = loads(page.body)
        _fields(raw, {"bars", "next_page_token"}, {"bars"})
        if not isinstance(raw["bars"], dict):
            raise ValueError("bars must be keyed by declared symbols")
        mapping = {x.symbol: x for x in instruments}
        if set(raw["bars"]) - mapping.keys():
            raise ValueError("provider returned undeclared instrument identity")
        if len({x.instrument_id for x in instruments}) != len(instruments):
            raise ValueError("instrument identities must be distinct")
        lower, upper, observed = (
            aware_timestamp(start),
            aware_timestamp(end),
            aware_timestamp(page.observed_at),
        )
        records = []
        seen: set[tuple[str, str]] = set()
        calendar = ExchangeCalendar("XNYS") if timeframe == "1Day" else None
        for symbol, bars in raw["bars"].items():
            if not isinstance(bars, list):
                raise ValueError("bar list required")
            for bar in bars:
                _fields(
                    bar, {"t", "o", "h", "l", "c", "v", "n", "vw"}, {"t", "o", "h", "l", "c", "v"}
                )
                stamp = aware_timestamp(bar["t"])
                if not lower <= stamp <= upper:
                    raise ValueError("provider bar lies outside requested interval")
                if calendar is not None:
                    # Daily provider timestamps label the New York session; the
                    # closing observation uses that explicit exchange schedule.
                    label = stamp.tz_convert("America/New_York").date().isoformat()
                    sessions = calendar.schedule(label, label)["rows"]
                    if len(sessions) != 1:
                        raise ValueError("daily provider bar does not identify an exchange session")
                    close = aware_timestamp(sessions[0]["close"])
                else:
                    close = stamp + pd.Timedelta(minutes=1)
                if close > observed:
                    raise ValueError("bar was acquired before its closing observation")
                key = (mapping[symbol].instrument_id, close.isoformat())
                if key in seen:
                    raise ValueError("duplicate event identity within provider page")
                seen.add(key)
                records.append(
                    {
                        "instrument_id": key[0],
                        "timestamp": key[1],
                        "available_at": observed.isoformat(),
                        "open": bar["o"],
                        "high": bar["h"],
                        "low": bar["l"],
                        "close": bar["c"],
                        "volume": bar["v"],
                    }
                )
                if len(records) > 1000:
                    raise ValueError("provider page exceeds 1000 bars")
        token = raw.get("next_page_token")
        if token is not None:
            _text(token, 512)
        snapshot = admit_bars(
            records,
            source=cls.source,
            classification="SYNTHETIC",
            corporate_action_policy="RAW",
            currency="USD",
        )
        return snapshot, token


class IngestionStore:
    """Page outbox: prepare bytes, idempotently admit bars, then advance cursor.

    A crash between bar admission and checkpoint advancement replays the same
    immutable acquired bytes and knowledge time, never inventing a new revision.
    """

    def __init__(self, base_dir: Path) -> None:
        self.base = Path(base_dir).absolute()
        if any(p.is_symlink() for p in (self.base, *self.base.parents)):
            raise ValueError("provider state cannot use symlink paths")
        self.base.mkdir(parents=True, exist_ok=True)
        self.path = self.base / "ingestion.sqlite"
        if self.path.is_symlink() or (self.base / "bars.sqlite").is_symlink():
            raise ValueError("provider state cannot redirect through symlinks")
        with self._connect() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS ingestion_jobs (job TEXT PRIMARY KEY, config TEXT NOT NULL, state TEXT NOT NULL, cursor TEXT, pages INTEGER NOT NULL, rows INTEGER NOT NULL, attempts INTEGER NOT NULL, retry_at TEXT, pending TEXT)"
            )
            con.execute(
                "CREATE TABLE IF NOT EXISTS ingestion_pages (digest TEXT PRIMARY KEY, job TEXT NOT NULL, payload BLOB NOT NULL, complete INTEGER NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA synchronous=FULL")
        return con

    def start(self, job_id: str, config: dict[str, Any]) -> dict[str, Any]:
        _text(job_id)
        _fields(
            config,
            {"instruments", "start", "end", "timeframe", "fixture_sha256"},
            {"instruments", "start", "end", "timeframe"},
        )
        instruments = [Instrument(**row) for row in config["instruments"]]
        AlpacaIEXAdapter.request_url(
            instruments, config["start"], config["end"], config["timeframe"]
        )
        encoded = dumps(config)
        with self._connect() as con:
            old = con.execute("SELECT config FROM ingestion_jobs WHERE job=?", (job_id,)).fetchone()
            if old is not None and old[0] != encoded:
                raise ValueError("job identity conflicts with preserved provider intent")
            con.execute(
                "INSERT OR IGNORE INTO ingestion_jobs VALUES (?,?,'READY',NULL,0,0,0,NULL,NULL)",
                (job_id, encoded),
            )
        return self.status(job_id)

    def status(self, job_id: str) -> dict[str, Any]:
        with self._connect() as con:
            row = con.execute("SELECT * FROM ingestion_jobs WHERE job=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError("unknown provider job")
        return {
            k: row[k]
            for k in ("job", "state", "cursor", "pages", "rows", "attempts", "retry_at", "pending")
        }

    def step(
        self,
        job_id: str,
        transport: Transport,
        *,
        now: str,
        after_admission: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        at = aware_timestamp(now)
        with self._connect() as con:
            row = con.execute("SELECT * FROM ingestion_jobs WHERE job=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError("unknown provider job")
        if row["state"] in {"COMPLETED", "ERROR"}:
            return self.status(job_id)
        if row["retry_at"] and at < aware_timestamp(row["retry_at"]):
            return self.status(job_id)
        config = loads(row["config"])
        instruments = [Instrument(**item) for item in config["instruments"]]
        if row["pending"]:
            with self._connect() as con:
                record = con.execute(
                    "SELECT payload FROM ingestion_pages WHERE digest=? AND job=?",
                    (row["pending"], job_id),
                ).fetchone()
            if record is None:
                raise ValueError("pending provider receipt missing")
            if not isinstance(record[0], bytes):
                raise ValueError("pending provider receipt integrity failure")
            encoded = record[0]
            if hashlib.sha256(encoded).hexdigest() != row["pending"]:
                raise ValueError("pending provider receipt integrity failure")
            receipt = loads(encoded)
            page = Page(receipt["body"].encode(), receipt["observed_at"])
        else:
            try:
                page = transport(row["cursor"])
            except Exception:
                page = Page(b"", at.isoformat(), status=503)
            if aware_timestamp(page.observed_at) > at:
                raise ValueError("provider acquisition time lies in the future")
            if page.status in {429, 500, 502, 503, 504}:
                attempts = row["attempts"] + 1
                wait = max(page.retry_after_seconds, min(60, 2**attempts))
                state = "ERROR" if attempts >= 4 else "WAITING"
                retry = (at + pd.Timedelta(seconds=wait)).isoformat()
                with self._connect() as con:
                    con.execute(
                        "UPDATE ingestion_jobs SET state=?,attempts=?,retry_at=? WHERE job=?",
                        (state, attempts, retry, job_id),
                    )
                return self.status(job_id)
            if page.status != 200:
                with self._connect() as con:
                    con.execute("UPDATE ingestion_jobs SET state='ERROR' WHERE job=?", (job_id,))
                return self.status(job_id)
            # Parse before preparing any durable evidence or changing the lake.
            _, next_token = AlpacaIEXAdapter.parse(
                page,
                instruments,
                start=config["start"],
                end=config["end"],
                timeframe=config["timeframe"],
            )
            if next_token is not None:
                with self._connect() as con:
                    cursors = [
                        loads(bytes(item[0]))["cursor"]
                        for item in con.execute(
                            "SELECT payload FROM ingestion_pages WHERE job=?", (job_id,)
                        )
                    ]
                if next_token in [row["cursor"], *cursors] or row["pages"] + 1 >= MAX_PAGES:
                    raise ValueError("cyclic or excessive provider pagination")
            receipt = {
                "job": job_id,
                "cursor": row["cursor"],
                "body": page.body.decode("utf-8"),
                "observed_at": aware_timestamp(page.observed_at).isoformat(),
                "config_sha256": _hash(config),
            }
            encoded = dumps(receipt).encode()
            digest = hashlib.sha256(encoded).hexdigest()
            with self._connect() as con:
                changed = con.execute(
                    "UPDATE ingestion_jobs SET pending=?,state='ADMITTING' WHERE job=? AND pending IS NULL AND cursor IS ? AND pages=?",
                    (digest, job_id, row["cursor"], row["pages"]),
                )
                if changed.rowcount != 1:
                    raise ValueError(
                        "provider job changed concurrently; retry after reconciliation"
                    )
                con.execute(
                    "INSERT OR IGNORE INTO ingestion_pages VALUES (?,?,?,0)",
                    (digest, job_id, encoded),
                )
        if (
            receipt["job"] != job_id
            or receipt["cursor"] != row["cursor"]
            or receipt["config_sha256"] != _hash(config)
        ):
            raise ValueError("provider receipt differs from admitted job")
        snapshot, token = AlpacaIEXAdapter.parse(
            page,
            instruments,
            start=config["start"],
            end=config["end"],
            timeframe=config["timeframe"],
        )
        if token is not None:
            with self._connect() as con:
                prior_cursors = [
                    loads(bytes(item[0]))["cursor"]
                    for item in con.execute(
                        "SELECT payload FROM ingestion_pages WHERE job=?", (job_id,)
                    )
                ]
            if token in prior_cursors or row["pages"] + 1 >= MAX_PAGES:
                raise ValueError("cyclic or excessive provider pagination")
        counts = PointInTimeStore(self.base / "bars.sqlite").ingest(snapshot)
        if after_admission is not None:
            after_admission()
        with self._connect() as con:
            con.execute(
                "UPDATE ingestion_pages SET complete=1 WHERE digest=?", (row["pending"] or digest,)
            )
            con.execute(
                "UPDATE ingestion_jobs SET state=?,cursor=?,pages=pages+1,rows=rows+?,attempts=0,retry_at=NULL,pending=NULL WHERE job=? AND pending=? AND pages=?",
                (
                    "COMPLETED" if token is None else "READY",
                    token,
                    len(snapshot.rows),
                    job_id,
                    row["pending"] or digest,
                    row["pages"],
                ),
            )
        result = self.status(job_id)
        result["last_page_admission"] = counts
        return result


class UniverseStore:
    """Immutable full-membership revisions; empty known universes stay explicit."""

    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self.path = Path(path).absolute()
        if any(p.is_symlink() for p in (self.path, *self.path.parents)):
            raise ValueError("universe state cannot use symlinks")
        if read_only:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS universe_revisions (universe TEXT, effective TEXT, available TEXT, digest TEXT, payload BLOB, PRIMARY KEY(universe,effective,available))"
            )

    def ingest(self, payload: dict[str, Any]) -> str:
        _fields(
            payload,
            {"universe_id", "source", "effective_at", "available_at", "members"},
            {"universe_id", "source", "effective_at", "available_at", "members"},
        )
        _text(payload["universe_id"])
        _text(payload["source"])
        effective, available = (
            aware_timestamp(payload["effective_at"]),
            aware_timestamp(payload["available_at"]),
        )
        if not isinstance(payload["members"], list) or len(payload["members"]) > 1000:
            raise ValueError("universe membership exceeds bounds")
        members = sorted(
            [asdict(Instrument(**item)) for item in payload["members"]],
            key=lambda item: item["instrument_id"],
        )
        if len({x["instrument_id"] for x in members}) != len(members) or len(
            {x["symbol"] for x in members}
        ) != len(members):
            raise ValueError("duplicate universe identity")
        body = {
            **payload,
            "effective_at": effective.isoformat(),
            "available_at": available.isoformat(),
            "members": members,
        }
        encoded = _bytes(body)
        digest = hashlib.sha256(encoded).hexdigest()
        with sqlite3.connect(self.path) as con:
            con.execute("PRAGMA synchronous=FULL")
            old = con.execute(
                "SELECT digest FROM universe_revisions WHERE universe=? AND effective=? AND available=?",
                (payload["universe_id"], effective.isoformat(), available.isoformat()),
            ).fetchone()
            if old is not None and old[0] != digest:
                raise ValueError("conflicting universe revision")
            con.execute(
                "INSERT OR IGNORE INTO universe_revisions VALUES (?,?,?,?,?)",
                (
                    payload["universe_id"],
                    effective.isoformat(),
                    available.isoformat(),
                    digest,
                    encoded,
                ),
            )
        return digest

    def snapshot(self, universe_id: str, as_of: str) -> dict[str, Any]:
        cutoff = aware_timestamp(as_of)
        selected = None
        with sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True) as con:
            for row in con.execute(
                "SELECT effective,available,digest,payload FROM universe_revisions WHERE universe=?",
                (_text(universe_id),),
            ):
                if not isinstance(row[3], bytes):
                    raise ValueError("universe evidence integrity failure")
                raw = row[3]
                if hashlib.sha256(raw).hexdigest() != row[2]:
                    raise ValueError("universe evidence integrity failure")
                payload = loads(raw)
                if (
                    payload["effective_at"] != row[0]
                    or payload["available_at"] != row[1]
                    or payload["universe_id"] != universe_id
                ):
                    raise ValueError("universe index differs from evidence")
                if aware_timestamp(row[0]) <= cutoff and aware_timestamp(row[1]) <= cutoff:
                    key = (aware_timestamp(row[0]), aware_timestamp(row[1]))
                    if selected is None or key > selected[0]:
                        selected = (key, payload, row[2])
        if selected is None:
            return {"state": "UNAVAILABLE", "members": None}
        return {"state": "AVAILABLE", **selected[1], "source_sha256": selected[2]}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("provider redirects are not permitted")


def _treasury_url(year: int) -> str:
    year = _integer(year, 1990, datetime.now(UTC).year)
    return (
        _TREASURY_ENDPOINT
        + "?"
        + urllib.parse.urlencode(
            {"data": "daily_treasury_yield_curve", "field_tdr_date_value": str(year)}
        )
    )


def fetch_treasury_year(year: int) -> tuple[bytes, str]:
    request = urllib.request.Request(
        _treasury_url(year),
        headers={
            "Accept": "application/atom+xml",
            "User-Agent": "GIG-Quant-Terminal historical public-data research",
        },
    )
    with urllib.request.build_opener(_NoRedirect()).open(request, timeout=20) as response:
        body = response.read(MAX_TREASURY_BYTES + 1)
    if len(body) > MAX_TREASURY_BYTES:
        raise ValueError("Treasury response exceeds bound")
    return body, aware_timestamp(datetime.now(UTC)).isoformat()


def parse_treasury(body: bytes, observed_at: str) -> list[dict[str, Any]]:
    if (
        not isinstance(body, bytes)
        or len(body) > MAX_TREASURY_BYTES
        or b"<!DOCTYPE" in body.upper()
        or b"<!ENTITY" in body.upper()
    ):
        raise ValueError("unsafe or oversized Treasury XML")
    cutoff = aware_timestamp(observed_at)
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        raise ValueError("malformed Treasury XML") from None
    rows = []
    seen = set()
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "properties":
            continue
        names = [child.tag.rsplit("}", 1)[-1] for child in element]
        if len(set(names)) != len(names):
            raise ValueError("duplicate Treasury property")
        properties = {child.tag.rsplit("}", 1)[-1]: child.text for child in element}
        rawdate = properties.get("NEW_DATE")
        if not isinstance(rawdate, str):
            raise ValueError("Treasury observation date absent")
        event = date.fromisoformat(rawdate.split("T")[0]).isoformat()
        if date.fromisoformat(event) > cutoff.date():
            raise ValueError("Treasury event date lies in future")
        for tenor in sorted(_TENORS):
            value = properties.get(tenor)
            if value is None:
                continue
            rate = quoted_yield(value)
            identity = (event, tenor)
            if identity in seen:
                raise ValueError("duplicate Treasury observation identity")
            seen.add(identity)
            rows.append(
                {
                    "event_date": event,
                    "tenor": tenor,
                    "yield_percent": str(rate),
                    "available_at": cutoff.isoformat(),
                    "publication_time": "UNKNOWN",
                    "source": "US_TREASURY",
                }
            )
            if len(rows) > 5000:
                raise ValueError("Treasury observation count exceeds bound")
    return sorted(rows, key=lambda r: (r["event_date"], r["tenor"]))


class ProviderWorkspace:
    """Application-owned paths and explicit fixture/public-data write actions."""

    def __init__(self, base_dir: Path) -> None:
        self.base = Path(base_dir).absolute()
        if any(p.is_symlink() for p in (self.base, *self.base.parents)):
            raise ValueError("provider workspace cannot use symlinks")

    @staticmethod
    def _contract(
        data: dict[str, Any] | None,
        *,
        classification: str = "SNAPSHOT",
        state: State = State.AVAILABLE,
        source: str = "provider-capabilities",
        as_of: str | None = None,
    ) -> Contract:
        return Contract.admit(
            state,
            data,
            source=source,
            classification=classification,
            scope="workspace.providers",
            as_of=as_of,
            evidence_state="PROVIDER_PROTOCOL_FIXTURE"
            if classification == "SYNTHETIC"
            else "PUBLIC_SOURCE_ACQUISITION"
            if source == "US_TREASURY"
            else "CAPABILITY_DECLARATION",
            error=(
                public_error("DATA_UNAVAILABLE")
                if state == State.UNAVAILABLE
                else public_error("INTERNAL_ERROR")
                if state == State.ERROR
                else None
            ),
        )

    def ingest(self, payload: dict[str, Any]) -> Contract:
        if not isinstance(payload, dict):
            raise ValueError("provider request must be an object")
        payload = loads(_bytes(payload))
        if payload.get("provider") == "alpaca_iex":
            if payload.get("mode") != "OFFLINE_FIXTURE":
                return Contract.admit(
                    State.UNAVAILABLE,
                    None,
                    source="alpaca_iex",
                    classification="UNAVAILABLE",
                    scope="workspace.providers",
                    error=public_error("BROKER_DISABLED"),
                )
            _fields(
                payload,
                {
                    "provider",
                    "mode",
                    "job_id",
                    "instruments",
                    "start",
                    "end",
                    "timeframe",
                    "pages",
                    "now",
                },
                {
                    "provider",
                    "mode",
                    "job_id",
                    "instruments",
                    "start",
                    "end",
                    "timeframe",
                    "pages",
                    "now",
                },
            )
            if (
                not isinstance(payload["pages"], list)
                or not 1 <= len(payload["pages"]) <= MAX_PAGES
            ):
                raise ValueError("fixture page count exceeds bounds")
            at = aware_timestamp(payload["now"]).isoformat()
            pages = []
            for page in payload["pages"]:
                _fields(
                    page,
                    {"body", "observed_at", "status", "retry_after_seconds"},
                    {"body", "observed_at"},
                )
                pages.append(
                    Page(
                        _bytes(page["body"]),
                        page["observed_at"],
                        page.get("status", 200),
                        page.get("retry_after_seconds", 0),
                    )
                )
            config = {key: payload[key] for key in ("instruments", "start", "end", "timeframe")}
            config["fixture_sha256"] = _hash(payload["pages"])
            store = IngestionStore(self.base)
            store.start(payload["job_id"], config)

            def fixture(cursor):
                if cursor is None:
                    index = 0
                else:
                    matching = [
                        i + 1
                        for i, p in enumerate(pages)
                        if loads(p.body).get("next_page_token") == cursor
                    ]
                    if len(matching) != 1:
                        raise ValueError("fixture cursor cannot be resolved")
                    index = matching[0]
                if index >= len(pages):
                    raise ValueError("fixture missing next page")
                return pages[index]

            for _ in range(MAX_PAGES):
                state = store.step(payload["job_id"], fixture, now=at)
                if state["state"] != "READY":
                    break
            outer_state = (
                State.ERROR
                if state["state"] == "ERROR"
                else State.UNKNOWN
                if state["state"] not in {"COMPLETED"}
                else State.AVAILABLE
                if state["rows"]
                else State.UNAVAILABLE
            )
            if outer_state != State.AVAILABLE:
                return self._contract(
                    None,
                    classification="SYNTHETIC"
                    if outer_state == State.ERROR
                    else "UNKNOWN"
                    if outer_state == State.UNKNOWN
                    else "UNAVAILABLE",
                    state=outer_state,
                    source=AlpacaIEXAdapter.source,
                    as_of=at,
                )
            return self._contract(
                {
                    "title": "Offline IEX protocol ingestion",
                    "metrics": state,
                    "rows": [],
                    "capability": provider_capabilities()[0],
                },
                classification="SYNTHETIC",
                as_of=at,
            )
        if payload.get("provider") == "us_treasury":
            _fields(payload, {"provider", "mode", "year"}, {"provider", "mode", "year"})
            if payload["mode"] != "PUBLIC_OBSERVED":
                raise ValueError("Treasury transport requires explicit PUBLIC_OBSERVED mode")
            body, observed = fetch_treasury_year(payload["year"])
            return self.ingest_treasury_bytes(
                body, observed_at=observed, year=payload["year"], public_observed=True
            )
        raise ValueError("unknown provider")

    def ingest_treasury_bytes(
        self, body: bytes, *, observed_at: str, year: int, public_observed: bool = False
    ) -> Contract:
        _integer(year, 1990, datetime.now(UTC).year)
        if not isinstance(public_observed, bool):
            raise ValueError("acquisition mode must be explicit")
        classification = "HISTORICAL" if public_observed else "SYNTHETIC"
        observed_at = aware_timestamp(observed_at).isoformat()
        rows = parse_treasury(body, observed_at)
        if any(not row["event_date"].startswith(str(year) + "-") for row in rows):
            raise ValueError("Treasury response year differs from request")
        digest = hashlib.sha256(body).hexdigest()
        directory = self.base / "provider_receipts"
        if directory.is_symlink() or (self.base / "ingestion.sqlite").is_symlink():
            raise ValueError("provider state path cannot redirect")
        metadata = {
            "source": "US_TREASURY",
            "receipt": digest,
            "available_at": observed_at,
            "classification": classification,
            "transport": "PUBLIC_HTTPS_ACQUISITION" if public_observed else "OFFLINE_FIXTURE",
            "year": year,
            "historical_publication_time": "UNKNOWN",
            "event_time_precision": "DATE_ONLY",
            "request_url": _treasury_url(year) if public_observed else None,
            "license": "CC0-1.0" if public_observed else "OFFLINE_TEST_FIXTURE",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/"
            if public_observed
            else None,
            "license_source_url": _TREASURY_LICENSE_SOURCE if public_observed else None,
        }
        metadata_bytes = _bytes(metadata)
        metadata_hash = hashlib.sha256(metadata_bytes).hexdigest()
        candidate = self._contract(
            {
                "title": "Treasury rate admission",
                "metrics": {"observations": len(rows), "year": year},
                "rows": rows[-25:],
                "source_sha256": digest,
                "rows_total": len(rows),
                "rows_complete": len(rows) <= 25,
            },
            classification=classification,
            source="US_TREASURY" if public_observed else "US_TREASURY:offline-fixture",
            as_of=observed_at,
        )
        del candidate  # Entire response must cross the wire before any evidence changes.
        self.base.mkdir(parents=True, exist_ok=True)
        write_immutable(directory / (digest + ".xml"), body)
        write_immutable(directory / (metadata_hash + ".json"), metadata_bytes)
        with sqlite3.connect(self.base / "ingestion.sqlite") as con:
            con.execute("PRAGMA synchronous=FULL")
            con.execute(
                "CREATE TABLE IF NOT EXISTS treasury_observations (event TEXT,tenor TEXT,available TEXT,payload TEXT,receipt TEXT,PRIMARY KEY(event,tenor,available))"
            )
            con.execute(
                "CREATE TABLE IF NOT EXISTS treasury_acquisitions (receipt TEXT,available TEXT,metadata_hash TEXT NOT NULL,PRIMARY KEY(receipt,available))"
            )
            prior = con.execute(
                "SELECT metadata_hash FROM treasury_acquisitions WHERE receipt=? AND available=?",
                (digest, observed_at),
            ).fetchone()
            if prior is not None and prior[0] != metadata_hash:
                raise ValueError("conflicting Treasury acquisition provenance")
            con.execute(
                "INSERT OR IGNORE INTO treasury_acquisitions VALUES (?,?,?)",
                (digest, observed_at, metadata_hash),
            )
            for row in rows:
                encoded = dumps(row)
                key = (row["event_date"], row["tenor"], row["available_at"])
                old = con.execute(
                    "SELECT payload FROM treasury_observations WHERE event=? AND tenor=? AND available=?",
                    key,
                ).fetchone()
                if old is not None and old[0] != encoded:
                    raise ValueError("conflicting Treasury acquisition identity")
                con.execute(
                    "INSERT OR IGNORE INTO treasury_observations VALUES (?,?,?,?,?)",
                    (*key, encoded, digest),
                )
        return (
            self._contract(
                {
                    "title": "Official Treasury historical rates",
                    "metrics": {
                        "observations": len(rows),
                        "year": year,
                        "equity_bars_fabricated": False,
                    },
                    "rows": rows[-25:],
                    "source_sha256": digest,
                    "rows_total": len(rows),
                    "rows_complete": len(rows) <= 25,
                    "license": provider_capabilities()[1]["license"],
                },
                classification=classification,
                source="US_TREASURY" if public_observed else "US_TREASURY:offline-fixture",
                as_of=observed_at,
            )
            if rows
            else self._contract(
                None, state=State.UNAVAILABLE, classification="UNAVAILABLE", source="US_TREASURY"
            )
        )

    def status(self) -> Contract:
        jobs = []
        path = self.base / "ingestion.sqlite"
        if path.exists():
            if path.is_symlink():
                raise ValueError("provider state cannot redirect")
            with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as con:
                if con.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='ingestion_jobs'"
                ).fetchone():
                    jobs = [
                        {"job_id": r[0], "state": r[1], "pages": r[2], "bars": r[3]}
                        for r in con.execute(
                            "SELECT job,state,pages,rows FROM ingestion_jobs ORDER BY job LIMIT 100"
                        )
                    ]
        return self._contract(
            {
                "title": "Provider capabilities and ingestion",
                "metrics": {"broker_connections": False, "jobs": len(jobs)},
                "rows": provider_capabilities(),
                "jobs": jobs,
            }
        )

    def rates(
        self, as_of: str, *, limit: int = 100, max_age_seconds: float | None = None
    ) -> Contract:
        """Read acquisition-time revisions; DATE_ONLY events never become fake clocks."""
        cutoff = aware_timestamp(as_of)
        _integer(limit, 1, 1000)
        if max_age_seconds is not None and (
            isinstance(max_age_seconds, bool)
            or not isinstance(max_age_seconds, (int, float))
            or not 0 <= max_age_seconds <= 366 * 86400
        ):
            raise ValueError("finite explicit freshness limit required")
        path = self.base / "ingestion.sqlite"

        def missing():
            return self._contract(
                None, state=State.UNAVAILABLE, classification="UNAVAILABLE", source="US_TREASURY"
            )

        if not path.exists():
            return missing()
        if path.is_symlink():
            raise ValueError("provider state cannot redirect")
        selected: dict[tuple[str, str, str], dict[str, Any]] = {}
        anchors: dict[tuple[str, str], tuple[dict[str, Any], set[bytes]]] = {}
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as con:
            if not con.execute(
                "SELECT 1 FROM sqlite_master WHERE name='treasury_observations'"
            ).fetchone():
                return missing()
            if not con.execute(
                "SELECT 1 FROM sqlite_master WHERE name='treasury_acquisitions'"
            ).fetchone():
                raise ValueError("unanchored legacy Treasury provenance; reingest source evidence")
            observed_count = 0
            for event, tenor, available, payload, digest in con.execute(
                "SELECT * FROM treasury_observations"
            ):
                observed_count += 1
                if observed_count > 50000:
                    raise ValueError("rate archive exceeds declared read bound")
                if aware_timestamp(available) > cutoff:
                    continue
                if not isinstance(digest, str) or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
                    raise ValueError("invalid provider receipt identity")
                if (self.base / "provider_receipts").is_symlink():
                    raise ValueError("provider evidence cannot redirect")
                key = (digest, available)
                if key not in anchors:
                    record = con.execute(
                        "SELECT metadata_hash FROM treasury_acquisitions WHERE receipt=? AND available=?",
                        key,
                    ).fetchone()
                    if record is None:
                        raise ValueError("rate acquisition provenance missing")
                    if (
                        not isinstance(record[0], str)
                        or re.fullmatch(r"[a-f0-9]{64}", record[0]) is None
                    ):
                        raise ValueError("invalid acquisition receipt identity")
                    metadata_path = self.base / "provider_receipts" / (record[0] + ".json")
                    if metadata_path.is_symlink():
                        raise ValueError("rate acquisition cannot redirect")
                    encoded = metadata_path.read_bytes()
                    if hashlib.sha256(encoded).hexdigest() != record[0]:
                        raise ValueError("rate acquisition integrity failure")
                    metadata = loads(encoded)
                    if (
                        metadata["receipt"] != digest
                        or metadata["available_at"] != available
                        or metadata["classification"] not in {"HISTORICAL", "SYNTHETIC"}
                    ):
                        raise ValueError("rate acquisition differs from index")
                    evidence_path = self.base / "provider_receipts" / (digest + ".xml")
                    if evidence_path.is_symlink():
                        raise ValueError("Treasury source cannot redirect")
                    body = evidence_path.read_bytes()
                    if hashlib.sha256(body).hexdigest() != digest:
                        raise ValueError("Treasury source evidence integrity failure")
                    anchors[key] = (
                        metadata,
                        {_bytes(row) for row in parse_treasury(body, available)},
                    )
                metadata, valid_rows = anchors[key]
                row = loads(payload)
                if (row["event_date"], row["tenor"], row["available_at"]) != (
                    event,
                    tenor,
                    available,
                ) or _bytes(row) not in valid_rows:
                    raise ValueError("Treasury observation differs from source evidence")
                identity = (event, tenor, metadata["classification"])
                if identity not in selected or aware_timestamp(available) > aware_timestamp(
                    selected[identity]["available_at"]
                ):
                    selected[identity] = {
                        **row,
                        "classification": metadata["classification"],
                        "source_sha256": digest,
                    }
        if not selected:
            return missing()
        classifications = {row["classification"] for row in selected.values()}
        if len(classifications) != 1 or max_age_seconds is None:
            return self._contract(
                None, state=State.UNKNOWN, classification="UNKNOWN", source="US_TREASURY"
            )
        classification = classifications.pop()
        allrows = [selected[k] for k in sorted(selected)]
        rows = allrows[-limit:]
        observed = max(aware_timestamp(row["available_at"]) for row in allrows)
        state = (
            State.STALE
            if (cutoff - observed).total_seconds() > max_age_seconds
            else State.AVAILABLE
        )
        analytics = curve_analytics(rows)
        return self._contract(
            {
                "title": "Treasury yield observations",
                **analytics,
                "metrics": {"observations": len(allrows), **analytics["metrics"]},
                "rows": rows,
                "rows_total": len(allrows),
                "rows_complete": len(rows) == len(allrows),
                "units": "percent_per_year",
                "event_time_precision": "DATE_ONLY",
                "historical_publication_time": "UNKNOWN",
                "freshness_basis": "ACQUISITION_TIME",
                "query_as_of": cutoff.isoformat(),
            },
            classification=classification,
            source="US_TREASURY"
            if classification == "HISTORICAL"
            else "US_TREASURY:offline-fixture",
            state=state,
            as_of=observed.isoformat(),
        )

    def universe(self, payload: dict[str, Any]) -> Contract:
        payload = loads(_bytes(payload))
        store = UniverseStore(self.base / "universe.sqlite")
        digest = store.ingest(payload)
        return self._contract(
            {
                "title": "Point-in-time universe admission",
                "metrics": {"members": len(payload["members"])},
                "rows": [],
                "source_sha256": digest,
            }
        )

    def universe_snapshot(self, universe_id: str, as_of: str) -> Contract:
        path = self.base / "universe.sqlite"
        if not path.exists():
            return self._contract(None, state=State.UNAVAILABLE, classification="UNAVAILABLE")
        result = UniverseStore(path, read_only=True).snapshot(universe_id, as_of)
        return (
            self._contract(
                {
                    **result,
                    "title": "Point-in-time universe",
                    "metrics": {"members": len(result["members"])},
                    "rows": result["members"],
                },
                classification="SNAPSHOT",
                source="declared-universe:" + result["source"],
                as_of=result["available_at"],
            )
            if result["state"] == "AVAILABLE"
            else self._contract(None, state=State.UNAVAILABLE, classification="UNAVAILABLE")
        )
