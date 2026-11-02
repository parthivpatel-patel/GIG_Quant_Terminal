"""Admitted multi-level depth evidence; no inferred book or provider connection."""
from __future__ import annotations

import hashlib
import sqlite3
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from gig.artifacts import write_immutable
from gig.data.validated import CLASSIFICATIONS
from gig.execution.oms import _decimal, _exact, _text
from gig.service.contracts import Contract, State
from gig.service.serialization import dumps, loads, public_error
from gig.service.workspace import TerminalWorkspace, _age, _fields
from gig.timestamps import aware_timestamp


class DepthWorkspace:
    def __init__(self, base_dir: Path):
        self.workspace = TerminalWorkspace(base_dir)

    def _path(self, name):
        return self.workspace._path("market-depth/" + name)

    @staticmethod
    @_exact
    def _contract(document, state=State.AVAILABLE):
        bids, asks = document["bids"], document["asks"]
        best_bid, best_ask = Decimal(bids[0]["price"]), Decimal(asks[0]["price"])
        bid_quantity = sum((Decimal(row["quantity"]) for row in bids), Decimal(0))
        ask_quantity = sum((Decimal(row["quantity"]) for row in asks), Decimal(0))
        total = bid_quantity + ask_quantity
        data = {"title": "Admitted market depth · " + document["instrument_id"],
                "instrument_id": document["instrument_id"],
                "metrics": {"best_bid": str(best_bid), "best_ask": str(best_ask),
                            "spread": {"value": str(best_ask-best_bid), "unit": document["currency"]},
                            "displayed_bid_quantity": str(bid_quantity), "displayed_ask_quantity": str(ask_quantity),
                            "displayed_depth_imbalance": {"value": float(Fraction(bid_quantity-ask_quantity)/Fraction(total)) if total else None,
                                                "state": "AVAILABLE" if total else "UNKNOWN", "unit": "fraction of admitted displayed quantity"}},
                "rows": [{"side": side, "level": i+1, **row} for side in ("bids", "asks")
                         for i, row in enumerate(document[side])],
                "depth": document, "feed": document["feed"], "event_at": document["event_at"],
                "notes": ["Caller-admitted displayed depth; provider authenticity and entitlements are unverified.",
                          "A snapshot does not establish executable liquidity, venue coverage, SIP or OPRA."]}
        return Contract.admit(state, data if state in {State.AVAILABLE, State.STALE} else None,
            source=document["source"], classification=document["classification"], as_of=document["available_at"],
            scope="terminal.market-depth", instrument_identity=document["instrument_id"],
            source_identity=document["snapshot_id"], evidence_state="CALLER_SUPPLIED_NOT_VENDOR_VERIFIED")

    def ingest(self, payload: dict[str, Any]) -> Contract:
        keys = {"instrument_id", "source", "classification", "feed", "currency", "event_at", "available_at", "sequence", "bids", "asks"}
        _fields(payload, keys, keys)
        document = loads(dumps(payload))
        for key in ("instrument_id", "source", "feed"):
            _text(document[key], key)
        if document["classification"] not in CLASSIFICATIONS or document["currency"] != "USD":
            raise ValueError("explicit source classification and USD quote units required")
        sequence = document["sequence"]
        if isinstance(sequence, bool) or not isinstance(sequence, int) or not 0 <= sequence < 2**63:
            raise ValueError("bounded nonnegative sequence required")
        for key in ("event_at", "available_at"):
            document[key] = aware_timestamp(document[key]).isoformat()
        if aware_timestamp(document["available_at"]) < aware_timestamp(document["event_at"]):
            raise ValueError("depth cannot be known before observation")
        for side in ("bids", "asks"):
            levels = document[side]
            if not isinstance(levels, list) or not 1 <= len(levels) <= 50:
                raise ValueError("one to 50 levels required on each side")
            prices = []
            for level in levels:
                _fields(level, {"price", "quantity"}, {"price", "quantity"})
                level["price"] = str(_decimal(level["price"]))
                level["quantity"] = str(_decimal(level["quantity"], positive=False))
                prices.append(Decimal(level["price"]))
            if len(set(prices)) != len(prices) or prices != sorted(prices, reverse=side == "bids"):
                raise ValueError("source levels must be unique and strictly ordered")
        if Decimal(document["bids"][0]["price"]) > Decimal(document["asks"][0]["price"]):
            raise ValueError("crossed depth is not admitted")
        identity = hashlib.sha256(dumps(document).encode()).hexdigest()
        document["snapshot_id"] = identity
        result = self._contract(document)
        directory, index = self._path("snapshots"), self._path("depth.sqlite")
        directory.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(index) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("CREATE TABLE IF NOT EXISTS snapshots(source TEXT,instrument TEXT,sequence INTEGER,available TEXT,digest TEXT,PRIMARY KEY(source,instrument,sequence))")
            previous = connection.execute("SELECT digest FROM snapshots WHERE source=? AND instrument=? AND sequence=?",
                (document["source"], document["instrument_id"], sequence)).fetchone()
            if previous and previous[0] != identity:
                raise ValueError("depth sequence conflicts with immutable source evidence")
            if previous is None and connection.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] >= 1000:
                raise ValueError("depth snapshot retention bound exceeded")
            write_immutable(directory / (identity + ".json"), dumps(document).encode())
            connection.execute("INSERT OR IGNORE INTO snapshots VALUES(?,?,?,?,?)",
                (document["source"], document["instrument_id"], sequence, document["available_at"], identity))
        return result

    def section(self, section, as_of, maximum, *, instrument_id=None):
        if section not in {"markets", "execution", "command-center"}:
            return None
        cutoff, maximum = _age(as_of, maximum)
        def empty(state):
            return Contract.admit(state, None, source="admitted-market-depth", classification="UNAVAILABLE",
                scope="terminal.market-depth", error=public_error("DATA_UNAVAILABLE") if state == State.UNAVAILABLE else None)
        path = self._path("depth.sqlite")
        if not path.is_file() or instrument_id is None:
            return empty(State.UNAVAILABLE)
        _text(instrument_id, "instrument_id")
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
            rows = list(connection.execute("SELECT source,instrument,sequence,available,digest FROM snapshots WHERE instrument=? LIMIT 1001", (instrument_id,)))
        if len(rows) > 1000:
            raise ValueError("depth read bound exceeded")
        visible = [row for row in rows if aware_timestamp(row[3]) <= cutoff]
        if not visible:
            return empty(State.UNAVAILABLE)
        if len({row[0] for row in visible}) != 1 or maximum is None:
            return empty(State.UNKNOWN)
        source, instrument, sequence, available, identity = max(visible, key=lambda row: row[2])
        if not isinstance(identity, str) or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
            raise ValueError("invalid depth evidence identity")
        document = loads(self._path("snapshots/" + identity + ".json").read_bytes())
        body = {k: v for k, v in document.items() if k != "snapshot_id"}
        if (hashlib.sha256(dumps(body).encode()).hexdigest() != identity
                or document.get("snapshot_id") != identity
                or (source, instrument, sequence, available) != tuple(document[k] for k in ("source", "instrument_id", "sequence", "available_at"))):
            raise ValueError("depth index differs from preserved source evidence")
        state = State.STALE if (cutoff-aware_timestamp(document["event_at"])).total_seconds() > maximum else State.AVAILABLE
        return self._contract(document, state)
