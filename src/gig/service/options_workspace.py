"""Immutable historical option evidence and European research, never an order path.

Caller knowledge/acquisition clocks are evidence declarations, not vendor proof.
Valuation uses the observation clock and ACT/365F, even when reconstructed later.
"""
from __future__ import annotations

import hashlib
import math
import platform
import sqlite3
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

from gig.artifacts import write_immutable
from gig.data.validated import CLASSIFICATIONS
from gig.options import (
    OptionPosition,
    OptionQuote,
    SurfacePoint,
    black_scholes,
    diagnose_call_slice,
    implied_volatility,
    interpolate_total_variance,
    portfolio_exposure,
    scenario_grid,
    validate_surface,
)
from gig.service.contracts import Contract, State
from gig.service.serialization import dumps, loads, public_error
from gig.service.workspace import TerminalWorkspace, _age, _fields, _number, _text
from gig.timestamps import aware_timestamp

VERSION = "persisted-european-options-v1"
SECTIONS = {"options", "research", "models", "portfolio", "risk"}
SNAPSHOT_KEYS = {"source", "classification", "feed", "underlying_id", "currency", "event_at",
                 "acquired_at", "knowledge_at", "spot", "spot_source_identity", "spot_event_at",
                 "spot_knowledge_at", "quotes"}
QUOTE_KEYS = {"contract_id", "expiry", "strike", "right", "multiplier", "style", "bid", "ask",
              "volume", "open_interest"}
RUN_KEYS = {"source", "classification", "underlying_id", "observation_at", "as_of", "max_age_seconds",
            "rate", "dividend_yield", "positions", "spot_shocks", "volatility_shocks", "surface_query"}


def _identity(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("canonical options artifact identity required")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def _count(value: Any, *, nullable: bool = False, signed: bool = False) -> int | None:
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or abs(value) > 1_000_000_000_000 or (not signed and value < 0):
        raise ValueError("bounded exact integer count required")
    return value


def _bounded_number(value: Any, *, positive: bool = False) -> float:
    answer = _number(value, positive=positive)
    if abs(answer) > 1e12:
        raise ValueError("options numeric bound exceeded")
    return answer


class OptionsWorkspace:
    def __init__(self, base_dir: Path):
        self.workspace = TerminalWorkspace(base_dir)

    def _path(self, name: str) -> Path:
        return self.workspace._path("options-research/" + name)

    @staticmethod
    def _admit(payload: dict[str, Any]) -> dict[str, Any]:
        _fields(payload, SNAPSHOT_KEYS, SNAPSHOT_KEYS)
        document = loads(dumps(payload))
        for key in ("source", "feed", "underlying_id", "spot_source_identity"):
            _text(document[key], bound=128)
        if not isinstance(document["classification"], str) or document["classification"] not in CLASSIFICATIONS or document["currency"] != "USD":
            raise ValueError("explicit input classification and USD option units required")
        for key in ("event_at", "acquired_at", "knowledge_at", "spot_event_at", "spot_knowledge_at"):
            document[key] = aware_timestamp(document[key]).isoformat()
        event, acquired, known = (aware_timestamp(document[k]) for k in ("event_at", "acquired_at", "knowledge_at"))
        if not event <= acquired <= known:
            raise ValueError("observation must precede acquisition and revision knowledge")
        if document["spot_event_at"] != document["event_at"] or not event <= aware_timestamp(document["spot_knowledge_at"]) <= known:
            raise ValueError("synchronous observed spot with explicit available provenance required")
        document["spot"] = _bounded_number(document["spot"], positive=True)
        quotes = document["quotes"]
        if not isinstance(quotes, list) or not 1 <= len(quotes) <= 200:
            raise ValueError("one to 200 option quotes required")
        identities, economic = set(), set()
        for quote in quotes:
            _fields(quote, QUOTE_KEYS, QUOTE_KEYS)
            contract = _text(quote["contract_id"], bound=128)
            quote["expiry"] = aware_timestamp(quote["expiry"]).isoformat()
            expiry = aware_timestamp(quote["expiry"])
            if not event < expiry or expiry.value-event.value > 30*366*86400*1_000_000_000:
                raise ValueError("expiry must follow the quote within 30 years")
            quote["strike"] = _bounded_number(quote["strike"], positive=True)
            if not isinstance(quote["right"], str) or quote["right"] not in {"CALL", "PUT"} or quote["style"] != "EUROPEAN":
                raise ValueError("only explicit European CALL/PUT research contracts supported")
            multiplier = _count(quote["multiplier"])
            if multiplier is None or not 1 <= multiplier <= 1_000_000:
                raise ValueError("positive bounded integer contract multiplier required")
            quote["bid"], quote["ask"] = _bounded_number(quote["bid"]), _bounded_number(quote["ask"])
            if not 0 <= quote["bid"] <= quote["ask"]:
                raise ValueError("nonnegative noncrossed quotes required")
            for key in ("volume", "open_interest"):
                quote[key] = _count(quote[key], nullable=True)
            terms = (quote["expiry"], quote["strike"], quote["right"], quote["style"])
            if contract in identities or terms in economic:
                raise ValueError("duplicate contract identity or economic contract terms")
            identities.add(contract)
            economic.add(terms)
        document["quotes"] = sorted(quotes, key=lambda row: row["contract_id"])
        return document

    @staticmethod
    def _snapshot_contract(document: dict[str, Any]) -> Contract:
        return Contract.admit(State.AVAILABLE,
            {"title": "Admitted historical European options evidence", **document,
             "rows": document["quotes"], "metrics": {"quotes": len(document["quotes"]), "vendor_verified": False},
             "notes": ["Input classification is caller-declared. Admission establishes content identity, not market coverage or entitlements."]},
            source=document["source"], classification=document["classification"], as_of=document["knowledge_at"],
            scope="workspace.options.evidence", source_identity=document["snapshot_id"],
            instrument_identity=document["underlying_id"], evidence_state="CALLER_SUPPLIED_NOT_VENDOR_VERIFIED")

    def ingest(self, payload: dict[str, Any]) -> Contract:
        document = self._admit(payload)
        identity = _digest(document)
        document["snapshot_id"] = identity
        result = self._snapshot_contract(document)
        index = self._path("options.sqlite")
        index.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(index) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("CREATE TABLE IF NOT EXISTS snapshots(source TEXT,underlying TEXT,event TEXT,acquired TEXT,known TEXT,digest TEXT,classification TEXT,PRIMARY KEY(source,underlying,event,known))")
            connection.execute("CREATE TABLE IF NOT EXISTS contracts(contract TEXT PRIMARY KEY,terms TEXT NOT NULL)")
            previous = connection.execute("SELECT digest FROM snapshots WHERE source=? AND underlying=? AND event=? AND known=?",
                tuple(document[k] for k in ("source", "underlying_id", "event_at", "knowledge_at"))).fetchone()
            if previous is not None and previous[0] != identity:
                raise ValueError("conflicting option revision at identical knowledge time")
            if previous is None and connection.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] >= 1000:
                raise ValueError("options snapshot retention bound exceeded")
            for quote in document["quotes"]:
                terms = dumps({"underlying_id": document["underlying_id"], "currency": document["currency"],
                    **{k: quote[k] for k in ("expiry", "strike", "right", "multiplier", "style")}})
                existing = connection.execute("SELECT terms FROM contracts WHERE contract=?", (quote["contract_id"],)).fetchone()
                if existing is not None and existing[0] != terms:
                    raise ValueError("contract identity cannot silently change economic terms")
                connection.execute("INSERT OR IGNORE INTO contracts VALUES(?,?)", (quote["contract_id"], terms))
            write_immutable(self._path(f"snapshots/{identity}.json"), dumps(document).encode())
            connection.execute("INSERT OR IGNORE INTO snapshots VALUES(?,?,?,?,?,?,?)",
                tuple(document[k] for k in ("source", "underlying_id", "event_at", "acquired_at", "knowledge_at")) + (identity, document["classification"]))
        return result

    def _snapshot(self, identity: str) -> dict[str, Any]:
        identity = _identity(identity)
        document = loads(self._path(f"snapshots/{identity}.json").read_bytes())
        body = {k: v for k, v in document.items() if k != "snapshot_id"}
        if _digest(body) != identity or document.get("snapshot_id") != identity or self._admit(body) != body:
            raise ValueError("immutable option snapshot integrity failure")
        path = self._path("options.sqlite")
        with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True) as connection:
            row = connection.execute("SELECT acquired,digest,classification FROM snapshots WHERE source=? AND underlying=? AND event=? AND known=?",
                tuple(document[k] for k in ("source", "underlying_id", "event_at", "knowledge_at"))).fetchone()
            if row != (document["acquired_at"], identity, document["classification"]):
                raise ValueError("option snapshot index differs from immutable evidence")
        return document

    @staticmethod
    def _request(payload: dict[str, Any]) -> dict[str, Any]:
        _fields(payload, RUN_KEYS, RUN_KEYS - {"positions", "spot_shocks", "volatility_shocks", "surface_query"})
        request = loads(dumps(payload))
        for key in ("source", "underlying_id"):
            _text(request[key], bound=128)
        if not isinstance(request["classification"], str) or request["classification"] not in CLASSIFICATIONS:
            raise ValueError("explicit options source classification required")
        for key in ("observation_at", "as_of"):
            request[key] = aware_timestamp(request[key]).isoformat()
        cutoff, maximum = _age(request["as_of"], request["max_age_seconds"])
        if maximum is None or aware_timestamp(request["observation_at"]) > cutoff:
            raise ValueError("explicit observation cutoff and freshness policy required")
        request["max_age_seconds"] = maximum
        for key in ("rate", "dividend_yield"):
            request[key] = _bounded_number(request[key])
            if not -1 <= request[key] <= 1:
                raise ValueError("continuous rates must lie between -1 and 1")
        positions = request.setdefault("positions", [])
        if not isinstance(positions, list) or len(positions) > 200:
            raise ValueError("bounded explicit simulated option positions required")
        ids = set()
        for position in positions:
            _fields(position, {"contract_id", "quantity"}, {"contract_id", "quantity"})
            _text(position["contract_id"], bound=128)
            quantity = _count(position["quantity"], signed=True)
            if quantity is None or abs(quantity) > 1_000_000 or position["contract_id"] in ids:
                raise ValueError("unique bounded signed contract positions required")
            ids.add(position["contract_id"])
        positions.sort(key=lambda p: p["contract_id"])
        for key, default in (("spot_shocks", [-.1, 0, .1]), ("volatility_shocks", [-.02, 0, .02])):
            values = request.setdefault(key, default)
            if not isinstance(values, list) or not 1 <= len(values) <= 25:
                raise ValueError("one to 25 explicit shocks per axis required")
            request[key] = [_bounded_number(value) for value in values]
            if len(set(request[key])) != len(values):
                raise ValueError("duplicate scenario shocks are not admitted")
        if any(not -1 < x <= 10 for x in request["spot_shocks"]) or any(abs(x) > 5 for x in request["volatility_shocks"]):
            raise ValueError("scenario shocks outside bounded research domain")
        if "surface_query" in request:
            query = request["surface_query"]
            _fields(query, {"log_forward_moneyness", "ttm"}, {"log_forward_moneyness", "ttm"})
            query["log_forward_moneyness"] = _bounded_number(query["log_forward_moneyness"])
            query["ttm"] = _bounded_number(query["ttm"], positive=True)
        return request

    def _select(self, request: dict[str, Any]) -> dict[str, Any] | None:
        path = self._path("options.sqlite")
        if not path.is_file():
            return None
        with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True) as connection:
            rows = connection.execute("SELECT acquired,known,digest,classification FROM snapshots WHERE source=? AND underlying=? AND event=? LIMIT 1001",
                tuple(request[k] for k in ("source", "underlying_id", "observation_at"))).fetchall()
        if len(rows) > 1000:
            raise ValueError("options source read bound exceeded")
        cutoff = aware_timestamp(request["as_of"])
        candidates = [(aware_timestamp(known), digest) for acquired, known, digest, classification in rows
            if aware_timestamp(acquired) <= cutoff and aware_timestamp(known) <= cutoff and classification == request["classification"]]
        if not candidates:
            return None
        return self._snapshot(max(candidates)[1])

    @staticmethod
    def _pcr(quotes: list[dict[str, Any]], field: str) -> dict[str, Any]:
        values = {right: [q[field] for q in quotes if q["right"] == right] for right in ("CALL", "PUT")}
        known = all(values[right] and all(v is not None for v in values[right]) for right in values)
        calls = sum(values["CALL"]) if known else None
        puts = sum(values["PUT"]) if known else None
        return {"value": puts/calls if calls else None, "state": "AVAILABLE" if calls else "UNKNOWN",
                "put_total": puts, "call_total": calls, "unit": "put/call ratio",
                "definition": f"Sum admitted PUT {field} / sum admitted CALL {field}; same synchronous chain; any missing count or zero call denominator is UNKNOWN."}

    @staticmethod
    def _compute(document: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        event = aware_timestamp(document["event_at"])
        spot, rate, div = document["spot"], request["rate"], request["dividend_yield"]
        rows, points, valid = [], [], {}
        slices: dict[str, list[dict[str, Any]]] = {}
        for quote in document["quotes"]:
            ttm = (aware_timestamp(quote["expiry"]).value-event.value)/(365*86400*1_000_000_000)
            midpoint = (quote["bid"]+quote["ask"])/2
            row = {**quote, "ttm": ttm, "midpoint": midpoint, "midpoint_classification": "MODEL_ESTIMATE",
                   "source_snapshot_id": document["snapshot_id"], "state": "AVAILABLE"}
            try:
                iv = implied_volatility(midpoint, spot, quote["strike"], rate, div, ttm, quote["right"] == "CALL")
                if iv == 0:
                    raise ValueError("zero IV excludes smooth Greeks")
                row.update({"implied_volatility": iv, "greeks": black_scholes(spot, quote["strike"], rate, div, iv, ttm, quote["right"] == "CALL"),
                    "log_forward_moneyness": math.log(quote["strike"]/spot)-(rate-div)*ttm})
                valid[quote["contract_id"]] = row
                if quote["right"] == "CALL":
                    points.append(SurfacePoint(row["log_forward_moneyness"], ttm, iv, document["snapshot_id"]))
                    slices.setdefault(quote["expiry"], []).append(row)
            except ValueError:
                row.update({"state": "ERROR", "implied_volatility": None, "greeks": None, "error_code": "OPTION_IV_UNAVAILABLE"})
            rows.append(row)
        term, skew, arbitrage = [], [], []
        for expiry, values in sorted(slices.items()):
            ordered = sorted(values, key=lambda row: row["strike"])
            nearest = min(values, key=lambda row: abs(row["log_forward_moneyness"]))
            term.append({"expiry": expiry, "ttm": nearest["ttm"], "strike": nearest["strike"],
                         "log_forward_moneyness": nearest["log_forward_moneyness"], "implied_volatility": nearest["implied_volatility"],
                         "definition": "Nearest admitted CALL to forward ATM; strike may differ across expiries."})
            width = ordered[-1]["log_forward_moneyness"]-ordered[0]["log_forward_moneyness"]
            skew.append({"expiry": expiry, "value": (ordered[-1]["implied_volatility"]-ordered[0]["implied_volatility"])/width if width else None,
                         "state": "AVAILABLE" if width else "UNKNOWN", "unit": "decimal IV/log-forward-moneyness",
                         "definition": "Endpoint secant of admitted CALL IV against ln(K/F), not fixed-delta market risk reversal."})
            arbitrage.append({"expiry": expiry, **diagnose_call_slice([OptionQuote(q["strike"], q["midpoint"], document["snapshot_id"]) for q in ordered],
                spot=spot, rate=rate, div=div, ttm=nearest["ttm"])})
        vts = [{"near_expiry": left["expiry"], "far_expiry": right["expiry"],
                "value": (right["implied_volatility"]-left["implied_volatility"])/(right["ttm"]-left["ttm"]),
                "unit": "decimal IV/year", "state": "AVAILABLE"}
               for left, right in zip(term, term[1:], strict=False)]
        positions = []
        for position in request["positions"]:
            selected = valid.get(position["contract_id"])
            if selected is None:
                raise ValueError("position references absent or invalid option model")
            positions.append(OptionPosition(selected["contract_id"], document["underlying_id"], selected["strike"], selected["ttm"],
                selected["implied_volatility"], position["quantity"], selected["multiplier"], selected["right"] == "CALL", document["currency"], document["snapshot_id"]))
        exposure = portfolio_exposure(positions, spot=spot, rate=rate, div=div)
        scenarios = scenario_grid(positions, spot=spot, rate=rate, div=div, spot_shocks=request["spot_shocks"], volatility_shocks=request["volatility_shocks"])
        surface = {"points": [{"log_forward_moneyness": p.log_forward_moneyness, "ttm": p.ttm, "volatility": p.volatility,
                               "source": p.source, "classification": "MODEL_ESTIMATE"} for p in points],
                   "diagnostics": asdict(validate_surface(points)), "classification": "MODEL_ESTIMATE", "right": "CALL",
                   "method": "Observed CALL coordinates with BSM-inverted midpoint IV; linear total variance query within hull only.",
                   "continuous_arbitrage_certified": False}
        if "surface_query" in request:
            try:
                surface["query"] = {"state": "AVAILABLE", **interpolate_total_variance(points, **request["surface_query"])}
            except ValueError:
                surface["query"] = {"state": "UNAVAILABLE", "value": None, "code": "SURFACE_QUERY_UNAVAILABLE"}
        return {"schema_version": VERSION, "request": request, "source": document["source"],
                "implementation_provenance": {"python": platform.python_version(), "numpy": version("numpy"), "scipy": version("scipy"),
                    "pricing_kernel": "gig.options.black_scholes", "iv_solver": "scipy.optimize.brentq", "max_iv_decimal": 10.0},
                "input_classification": document["classification"], "source_snapshot_id": document["snapshot_id"],
                "underlying_id": document["underlying_id"], "as_of": request["as_of"], "observation_at": document["event_at"],
                "acquired_at": document["acquired_at"], "knowledge_at": document["knowledge_at"],
                "spot_source_identity": document["spot_source_identity"], "spot_event_at": document["spot_event_at"], "spot_knowledge_at": document["spot_knowledge_at"],
                "valuation_clock": document["event_at"], "spot": spot, "rows": rows, "surface": surface,
                "term_structure": term, "skew": skew, "slice_diagnostics": arbitrage,
                "pcr_volume": OptionsWorkspace._pcr(document["quotes"], "volume"),
                "pcr_open_interest": OptionsWorkspace._pcr(document["quotes"], "open_interest"),
                "vts": {"definition": "VTS here means adjacent-expiry slope of nearest-admitted-CALL-to-ATM IV over ACT/365F maturity; not a proprietary standardized signal.", "rows": vts, "state": "AVAILABLE" if vts else "UNAVAILABLE"},
                "csm": {"state": "UNAVAILABLE", "value": None, "definition": "CSM authoritative definition UNSUPPLIED; no substitute formula invented."},
                "portfolio_exposure": exposure, "scenarios": scenarios,
                "metrics": {"admitted_quotes": len(rows), "valid_models": len(valid), "invalid_models": len(rows)-len(valid),
                            "simulated_positions": len(positions), "trading_approved": False},
                "assumptions": {"exercise": "EUROPEAN", "pricing": "BLACK_SCHOLES_MERTON", "day_count": "ACT/365F",
                    "rates": "Continuous annualized decimal supplied by researcher; not an observed yield curve.",
                    "dividends": "Constant continuous annualized dividend yield supplied by researcher; not a discrete dividend forecast.",
                    "spot": "Synchronous caller-admitted underlying observation; no interpolated spot or live mark.",
                    "iv": "Bracketed midpoint inversion, max_vol=10; midpoint and all IV/Greeks are modeled, not executable quotes.",
                    "costs": "No execution or transaction costs in model valuations. Scenario P&L is modeled, not realized."},
                "trading_approved": False, "economically_validated": False, "broker_observed": False,
                "feed": document["feed"], "notes": ["Caller evidence is not vendor verified and does not establish OPRA coverage or entitlements.",
                    "Reconstructed research known at the requested cutoff is not proof this research model existed historically.",
                    "Portfolio inputs are simulated signed contract counts and are not OMS or broker positions."]}

    @staticmethod
    def _contract(artifact: dict[str, Any], section: str = "options", *, state: State | None = None) -> Contract:
        if state is None:
            request = artifact["request"]
            age = (aware_timestamp(request["as_of"]).value-aware_timestamp(artifact["observation_at"]).value)/1e9
            state = State.STALE if age > request["max_age_seconds"] else State.AVAILABLE
        data = {"title": "Persisted European options research", **artifact}
        if section == "portfolio":
            data["rows"] = artifact["portfolio_exposure"]["positions"]
        elif section == "risk":
            data["rows"] = artifact["scenarios"]["scenarios"]
        return Contract.admit(state, data if state in {State.AVAILABLE, State.STALE} else None,
            source=artifact["source"], classification="MODEL_ESTIMATE", as_of=artifact["as_of"],
            scope=f"workspace.options.{section}", source_identity=artifact["source_snapshot_id"],
            model_identity=artifact["run_id"], model_version=VERSION, instrument_identity=artifact["underlying_id"],
            evidence_state="HASH_VERIFIED_CALLER_EVIDENCE_RESEARCH", readiness="RESEARCH_ONLY")

    @staticmethod
    def _missing() -> Contract:
        return Contract.admit(State.UNAVAILABLE, None, source="historical-options-research", classification="UNAVAILABLE",
            scope="workspace.options.research", error=public_error("DATA_UNAVAILABLE"))

    def run(self, payload: dict[str, Any]) -> Contract:
        request = self._request(payload)
        document = self._select(request)
        if document is None:
            return self._missing()
        artifact = self._compute(document, request)
        encoded = dumps(artifact).encode()
        identity = hashlib.sha256(encoded).hexdigest()
        artifact["run_id"] = identity
        result = self._contract(artifact)
        write_immutable(self._path(f"runs/{identity}.json"), encoded)
        with sqlite3.connect(self._path("options.sqlite")) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY,as_of TEXT NOT NULL,snapshot_id TEXT NOT NULL)")
            previous = connection.execute("SELECT as_of,snapshot_id FROM runs WHERE run_id=?", (identity,)).fetchone()
            expected = (artifact["as_of"], artifact["source_snapshot_id"])
            if previous is not None and previous != expected:
                raise ValueError("options run index conflicts with immutable artifact")
            if previous is None and connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] >= 1000:
                raise ValueError("options experiment retention bound exceeded")
            connection.execute("INSERT OR IGNORE INTO runs VALUES(?,?,?)", (identity, *expected))
        return result

    def load(self, run_id: str) -> dict[str, Any]:
        run_id = _identity(run_id)
        encoded = self._path(f"runs/{run_id}.json").read_bytes()
        artifact = loads(encoded)
        if hashlib.sha256(encoded).hexdigest() != run_id or dumps(artifact).encode() != encoded or artifact.get("schema_version") != VERSION:
            raise ValueError("options experiment integrity failure")
        with sqlite3.connect(self._path("options.sqlite").as_uri()+"?mode=ro", uri=True) as connection:
            row = connection.execute("SELECT as_of,snapshot_id FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row != (artifact["as_of"], artifact["source_snapshot_id"]):
            raise ValueError("options experiment index differs from immutable run")
        self._snapshot(artifact["source_snapshot_id"])
        return {**artifact, "run_id": run_id}

    def replay(self, payload: dict[str, Any]) -> Contract:
        _fields(payload, {"run_id"}, {"run_id"})
        try:
            artifact = self.load(payload["run_id"])
        except FileNotFoundError:
            return self._missing()
        document = self._select(artifact["request"])
        if document is None or document["snapshot_id"] != artifact["source_snapshot_id"]:
            raise ValueError("original option research knowledge window changed")
        recomputed = self._compute(document, artifact["request"])
        if _digest(recomputed) != artifact["run_id"]:
            raise ValueError("options experiment deterministic replay mismatch")
        return self._contract(artifact)

    def section(self, section: str, as_of: Any, max_age_seconds: float | None) -> Contract | None:
        if section not in SECTIONS:
            return None
        cutoff, maximum = _age(as_of, max_age_seconds)
        path = self._path("options.sqlite")
        if not path.is_file():
            return None
        with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True) as connection:
            if connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='runs'").fetchone() is None:
                return None
            rows = connection.execute("SELECT run_id,as_of FROM runs LIMIT 1001").fetchall()
        if len(rows) > 1000:
            raise ValueError("options experiment read bound exceeded")
        candidates = [(aware_timestamp(at), identity) for identity, at in rows if aware_timestamp(at) <= cutoff]
        if not candidates:
            return None
        artifact = self.load(max(candidates)[1])
        age = (cutoff.value-aware_timestamp(artifact["observation_at"]).value)/1e9
        state = State.UNKNOWN if maximum is None else State.STALE if age > maximum else State.AVAILABLE
        return self._contract(artifact, section, state=state)
