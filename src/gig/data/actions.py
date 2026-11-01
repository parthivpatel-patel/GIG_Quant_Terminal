"""Caller-evidenced point-in-time corporate actions and exact back-adjustment.

These are introduced research contracts, not historical project authority.
No provider authentication, complete event coverage, exchange calendar, or
trading approval is inferred. Original bars always remain present unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

from gig.data.validated import admit_bars
from gig.timestamps import aware_timestamp

MAX_ACTIONS = 128
MAX_BARS = 500
IMPLEMENTATION = "gig.corporate-actions.v1"
KINDS = frozenset({"SPLIT", "CASH_DIVIDEND", "SPLIT_AND_DIVIDEND"})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(value: Any, name: str, bound: int = 256) -> str:
    if (not isinstance(value, str) or not value.strip() or len(value) > bound
            or any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in value)):
        raise ValueError(f"{name} requires bounded nonempty evidence text")
    return value


def _cash(value: Any, name: str) -> str:
    if (not isinstance(value, str) or not 1 <= len(value) <= 128
            or re.fullmatch(r"[+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value) is None):
        raise ValueError(f"{name} requires an exact bounded decimal string")
    try:
        number = Decimal(value)
    except InvalidOperation:
        raise ValueError(f"{name} requires a finite nonnegative decimal") from None
    if not number.is_finite() or number < 0:
        raise ValueError(f"{name} requires a finite nonnegative decimal")
    exponent = number.as_tuple().exponent
    if not isinstance(exponent, int) or abs(exponent) > 100 or len(number.as_tuple().digits) > 100:
        raise ValueError(f"{name} decimal precision exceeds declared bounds")
    return value


def _ratio(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10**9:
        raise ValueError("split terms require positive integers <= 1000000000")
    return value


def _currency(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 3 or not value.isascii() or not value.isalpha():
        raise ValueError("currency requires three ASCII letters")
    return value.upper()


def _action(raw: Mapping[str, Any]) -> dict[str, Any]:
    common = {"action_id", "instrument_id", "source", "kind", "effective_at", "available_at"}
    if not isinstance(raw, Mapping) or not common <= set(raw):
        raise ValueError("corporate action is missing common evidence fields")
    kind = raw["kind"]
    if not isinstance(kind, str) or kind not in KINDS:
        raise ValueError("unsupported corporate-action kind")
    fields = set(common)
    if kind in {"SPLIT", "SPLIT_AND_DIVIDEND"}:
        fields |= {"ratio_new", "ratio_old"}
    cash_name = "cash_per_pre_action_share" if kind == "SPLIT_AND_DIVIDEND" else "cash_per_share"
    if kind != "SPLIT":
        fields |= {cash_name, "currency", "reference_timestamp", "reference_close"}
    if set(raw) != fields:
        raise ValueError("unsupported, missing, or ambiguous corporate-action fields")
    row: dict[str, Any] = {name: _text(raw[name], name, 512 if name == "source" else 256)
           for name in ("action_id", "instrument_id", "source")}
    row.update(kind=kind, effective_at=aware_timestamp(raw["effective_at"]).isoformat(),
               available_at=aware_timestamp(raw["available_at"]).isoformat())
    # Announcements can precede effectiveness, and corrections can arrive later.
    if kind in {"SPLIT", "SPLIT_AND_DIVIDEND"}:
        row.update(ratio_new=_ratio(raw["ratio_new"]), ratio_old=_ratio(raw["ratio_old"]))
    if kind != "SPLIT":
        row[cash_name] = _cash(raw[cash_name], cash_name)
        if Fraction(Decimal(row[cash_name])) <= 0:
            raise ValueError("a dividend event requires positive declared cash")
        row["currency"] = _currency(raw["currency"])
        row["reference_timestamp"] = aware_timestamp(raw["reference_timestamp"]).isoformat()
        row["reference_close"] = _cash(raw["reference_close"], "reference_close")
        if Decimal(row["reference_close"]) <= 0:
            raise ValueError("dividend reference close must be positive")
        if aware_timestamp(row["reference_timestamp"]) >= aware_timestamp(row["effective_at"]):
            raise ValueError("dividend reference must strictly precede effectiveness")
    return row


@dataclass(frozen=True, slots=True)
class ActionSnapshot:
    payload: bytes
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload, bytes) or hashlib.sha256(self.payload).hexdigest() != self.sha256:
            raise ValueError("action snapshot identity mismatch")

    def as_dict(self) -> dict[str, Any]:
        return json.loads(self.payload)


def admit_actions(records: Iterable[Mapping[str, Any]]) -> ActionSnapshot:
    """Take a detached immutable canonical snapshot of bounded action revisions."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    identities: dict[tuple[str, str], tuple[str, str]] = {}
    for raw in records:
        if len(rows) >= MAX_ACTIONS:
            raise ValueError("corporate-action admission exceeds 128 revisions")
        row = _action(raw)
        identity = row["source"], row["action_id"], row["available_at"]
        if identity in seen:
            raise ValueError("duplicate source/action/revision identity")
        seen.add(identity)
        stable_key = row["source"], row["action_id"]
        stable_value = row["instrument_id"], row["kind"]
        if identities.setdefault(stable_key, stable_value) != stable_value:
            raise ValueError("action identity cannot change instrument or kind across revisions")
        rows.append(row)
    rows.sort(key=lambda row: (row["source"], row["action_id"], aware_timestamp(row["available_at"])))
    payload = _canonical({"schema_version": IMPLEMENTATION, "evidence_state": "CALLER_SUPPLIED_NOT_VENDOR_VERIFIED",
                          "coverage": "CALLER_DECLARED_EVENTS_ONLY", "rows": rows})
    return ActionSnapshot(payload, hashlib.sha256(payload).hexdigest())


def _visible_actions(snapshot: ActionSnapshot, as_of: Any) -> list[dict[str, Any]]:
    cutoff = aware_timestamp(as_of)
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in snapshot.as_dict()["rows"]:
        if aware_timestamp(row["available_at"]) <= cutoff:
            key = row["source"], row["action_id"]
            prior = latest.get(key)
            if prior is None or aware_timestamp(row["available_at"]) > aware_timestamp(prior["available_at"]):
                latest[key] = row
    visible = [row for row in latest.values() if aware_timestamp(row["effective_at"]) <= cutoff]
    visible.sort(key=lambda row: (row["instrument_id"], aware_timestamp(row["effective_at"]), row["action_id"]))
    return visible


class ActionStore:
    """Atomic, restart-safe action revision evidence; no market-data connection."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS action_revisions (source TEXT NOT NULL, action_id TEXT NOT NULL, available_at TEXT NOT NULL, payload BLOB NOT NULL, sha256 TEXT NOT NULL, PRIMARY KEY(source,action_id,available_at))")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def ingest(self, snapshot: ActionSnapshot) -> dict[str, int]:
        data = snapshot.as_dict()
        checked = admit_actions(data["rows"])
        if checked.payload != snapshot.payload:
            raise ValueError("noncanonical action snapshot")
        inserted = replayed = 0
        with self._connect() as connection:
            for row in data["rows"]:
                payload = _canonical(row)
                identity = row["source"], row["action_id"], row["available_at"]
                older = connection.execute("SELECT payload FROM action_revisions WHERE source=? AND action_id=? LIMIT 1", identity[:2]).fetchone()
                if older is not None:
                    old_row = json.loads(bytes(older[0]))
                    if (old_row["instrument_id"], old_row["kind"]) != (row["instrument_id"], row["kind"]):
                        raise ValueError("action identity cannot change instrument or kind across revisions")
                prior = connection.execute("SELECT payload,sha256 FROM action_revisions WHERE source=? AND action_id=? AND available_at=?", identity).fetchone()
                if prior:
                    if bytes(prior[0]) != payload or prior[1] != hashlib.sha256(payload).hexdigest():
                        raise ValueError("conflicting action revision; preserve later knowledge timestamp")
                    replayed += 1
                else:
                    connection.execute("INSERT INTO action_revisions VALUES (?,?,?,?,?)", (*identity, payload, hashlib.sha256(payload).hexdigest()))
                    inserted += 1
        return {"inserted": inserted, "replayed": replayed}

    def snapshot(self, as_of: Any, *, source: str | None = None) -> ActionSnapshot:
        cutoff = aware_timestamp(as_of)
        rows = []
        with self._connect() as connection:
            query = "SELECT source,action_id,available_at,payload,sha256 FROM action_revisions"
            params: tuple[str, ...] = ()
            if source is not None:
                query += " WHERE source=?"
                params = (source,)
            for stored in connection.execute(query, params):
                payload = bytes(stored[3])
                if hashlib.sha256(payload).hexdigest() != stored[4]:
                    raise ValueError("stored action evidence is corrupt")
                row = _action(json.loads(payload))
                if tuple(stored[:3]) != (row["source"], row["action_id"], row["available_at"]):
                    raise ValueError("stored action identity differs from source evidence")
                if aware_timestamp(row["available_at"]) <= cutoff:
                    rows.append(row)
                    if len(rows) > MAX_ACTIONS:
                        raise ValueError("action snapshot exceeds revision bound; select a narrower source store")
        admitted = admit_actions(rows)
        return admit_actions(_visible_actions(admitted, cutoff))


def _fraction(value: Fraction) -> dict[str, str]:
    numerator, denominator = str(value.numerator), str(value.denominator)
    if max(len(numerator), len(denominator)) > 128:
        raise ValueError("adjustment exact-rational precision exceeds 128 digits per term")
    return {"numerator": numerator, "denominator": denominator}


def adjust_bars(records: Iterable[Mapping[str, Any]], actions: Iterable[Mapping[str, Any]], *,
                as_of: Any, mode: str, source: str, classification: str, currency: str) -> dict[str, Any]:
    """Back-adjust raw caller bars using effective, knowable corporate actions.

    Split price factor is old/new; volume multiplies new/old. Total-return
    cash factors use the actual first post-effective closing price and reproduce
    wealth-reinvestment returns exactly. Combined cash is per PRE-action share.
    Derived availability records all later evidence used; raw times never move.
    """
    if mode not in {"SPLIT_ADJUSTED", "TOTAL_RETURN_ADJUSTED"}:
        raise ValueError("explicit split or total-return adjustment mode required")
    cutoff = aware_timestamp(as_of)
    raw_records: list[Mapping[str, Any]] = []
    for raw_record in records:
        if len(raw_records) >= MAX_BARS:
            raise ValueError("adjustment request exceeds 500 raw bar revisions")
        raw_records.append(raw_record)
    admitted = admit_bars(raw_records, source=source, classification=classification,
                          corporate_action_policy="RAW", currency=_currency(currency))
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in admitted.rows:
        if aware_timestamp(row["timestamp"]) > cutoff or aware_timestamp(row["available_at"]) > cutoff:
            continue
        key = row["instrument_id"], row["timestamp"]
        prior = latest.get(key)
        if prior is None or aware_timestamp(row["available_at"]) > aware_timestamp(prior["available_at"]):
            latest[key] = row
    rows = sorted(latest.values(), key=lambda row: (row["instrument_id"], aware_timestamp(row["timestamp"])))
    admitted_actions = admit_actions(actions)
    visible_actions = _visible_actions(admitted_actions, cutoff)
    if not rows:
        return {"state": "UNAVAILABLE", "classification": classification, "rows": [],
                "as_of": cutoff.isoformat(), "implementation": IMPLEMENTATION}
    instruments = {row["instrument_id"] for row in rows}
    if any(row["instrument_id"] not in instruments for row in visible_actions):
        raise ValueError("action instrument identity is absent from admitted visible bars")
    seen_events: set[tuple[str, str]] = set()
    source_by_instrument: dict[str, str] = {}
    for event in visible_actions:
        key = event["instrument_id"], event["effective_at"]
        if key in seen_events:
            raise ValueError("simultaneous separate events are ambiguous; supply one explicit combined event")
        seen_events.add(key)
        prior_source = source_by_instrument.setdefault(event["instrument_id"], event["source"])
        if prior_source != event["source"]:
            raise ValueError("mixed corporate-action providers for one instrument require an explicit reconciliation decision")
    factors = []
    action_applications = []
    for event in visible_actions:
        split = Fraction(event["ratio_new"], event["ratio_old"]) if event["kind"] != "CASH_DIVIDEND" else Fraction(1)
        price_factor = 1 / split
        derived_available = max(aware_timestamp(event["available_at"]), aware_timestamp(event["effective_at"]))
        post_timestamp = None
        if event["kind"] != "SPLIT" and mode == "TOTAL_RETURN_ADJUSTED":
            if event["currency"] != _currency(currency):
                raise ValueError("dividend currency differs from source bars")
            preceding = [row for row in rows if row["instrument_id"] == event["instrument_id"]
                         and aware_timestamp(row["timestamp"]) < aware_timestamp(event["effective_at"])]
            if not preceding:
                raise ValueError("dividend requires an actual prior source closing bar")
            reference = preceding[-1]
            if reference["timestamp"] != event["reference_timestamp"]:
                raise ValueError("dividend reference must be the latest known pre-effective source bar")
            actual_close = Fraction(Decimal(str(reference["close"])))
            if actual_close != Fraction(Decimal(event["reference_close"])):
                raise ValueError("dividend reference price differs from original source evidence")
            cash_name = "cash_per_pre_action_share" if event["kind"] == "SPLIT_AND_DIVIDEND" else "cash_per_share"
            cash = Fraction(Decimal(event[cash_name]))
            if cash >= actual_close:
                raise ValueError("cash dividend would produce a nonpositive adjustment factor")
            following = [row for row in rows if row["instrument_id"] == event["instrument_id"]
                         and aware_timestamp(row["timestamp"]) >= aware_timestamp(event["effective_at"])]
            if not following:
                raise ValueError("total-return dividend adjustment requires a knowable post-effective closing bar")
            ex_bar = following[0]
            intervening = [other for other in visible_actions if other["instrument_id"] == event["instrument_id"]
                           and aware_timestamp(reference["timestamp"]) < aware_timestamp(other["effective_at"])
                           <= aware_timestamp(ex_bar["timestamp"])]
            if len(intervening) != 1:
                raise ValueError("dividend valuation interval spans another action; use a combined event or intervening closing bar")
            ex_close = Fraction(Decimal(str(ex_bar["close"])))
            price_factor = ex_close / (split * ex_close + cash)
            post_timestamp = aware_timestamp(ex_bar["timestamp"])
            derived_available = max(derived_available, aware_timestamp(reference["available_at"]),
                                    aware_timestamp(ex_bar["available_at"]))
        applied = event["kind"] != "CASH_DIVIDEND" or mode == "TOTAL_RETURN_ADJUSTED"
        action_applications.append({"source": event["source"], "action_id": event["action_id"],
                                    "applied": applied, "derived_available_at": derived_available.isoformat(),
                                    "post_effective_observation": None if post_timestamp is None else post_timestamp.isoformat()})
        if applied:
            factors.append((event, price_factor, split, derived_available, post_timestamp))
    output = []
    for row in rows:
        price_factor, volume_factor = Fraction(1), Fraction(1)
        adjusted_available = aware_timestamp(row["available_at"])
        later_observations = False
        for event, price, split, knowledge_time, post_time in factors:
            if event["instrument_id"] == row["instrument_id"] and aware_timestamp(row["timestamp"]) < aware_timestamp(event["effective_at"]):
                price_factor *= price
                volume_factor *= split
                adjusted_available = max(adjusted_available, knowledge_time)
                later_observations = later_observations or aware_timestamp(event["effective_at"]) > aware_timestamp(row["timestamp"]) or (post_time is not None and post_time > aware_timestamp(row["timestamp"]))
        result = dict(row)
        result["adjusted_available_at"] = adjusted_available.isoformat()
        result["uses_later_observations"] = later_observations
        result["derived_available_after_original_timestamp"] = adjusted_available > aware_timestamp(row["timestamp"])
        result["price_factor_exact"] = _fraction(price_factor)
        result["volume_factor_exact"] = _fraction(volume_factor)
        for name in ("open", "high", "low", "close"):
            exact = Fraction(Decimal(str(row[name]))) * price_factor
            _fraction(exact)
            estimate = float(exact)
            if not math.isfinite(estimate) or estimate <= 0:
                raise ValueError("adjusted price is outside finite positive binary64 range")
            result["adjusted_" + name] = estimate
            if name == "close":
                result["adjusted_close_exact"] = _fraction(exact)
        result["adjusted_volume_exact"] = _fraction(Fraction(int(row["volume"])) * volume_factor)
        output.append(result)
    source_identity = _hash({"source": source, "classification": classification, "currency": _currency(currency), "rows": rows})
    actions_identity = _hash(visible_actions)
    configuration = {"implementation": IMPLEMENTATION, "as_of": cutoff.isoformat(), "mode": mode,
                     "source_snapshot_sha256": source_identity, "action_snapshot_sha256": actions_identity,
                     "cash_convention": "EX_EVENT_WEALTH_REINVESTMENT_FIRST_VISIBLE_CLOSE"}
    result = {"state": "AVAILABLE", "classification": classification, "mode": mode,
              "source": source, "currency": _currency(currency), "as_of": cutoff.isoformat(),
              "rows": output, "actions": visible_actions, "action_applications": action_applications, "configuration": configuration,
              "transformation_sha256": _hash(configuration), "evidence_state": "CALLER_SUPPLIED_NOT_VENDOR_VERIFIED",
              "coverage": "CALLER_DECLARED_EVENTS_ONLY_NOT_INDEPENDENTLY_COMPLETE",
              "price_estimate_encoding": "binary64_with_exact_rational_factor_and_close",
              "adjusted_volume_encoding": "exact_rational_integer_strings", "trading_approved": False,
              "retrospective_adjustment": True, "feature_availability_field": "adjusted_available_at",
              "uses_later_observations": any(row["uses_later_observations"] for row in output)}
    if len(_canonical(result)) > 900_000:
        raise ValueError("adjustment result exceeds bounded response size")
    return result
