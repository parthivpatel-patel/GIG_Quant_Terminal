"""Read-only OMS accounting with immutable, restart-safe portfolio checkpoints."""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Any

from gig.artifacts import write_immutable
from gig.execution.oms import OfflineOMS, _decimal, _text
from gig.portfolio.ledger import canonical_fills, reconcile_account, risk_report, signed_decimal
from gig.service.contracts import Contract, State
from gig.service.serialization import dumps, loads, public_error
from gig.timestamps import aware_timestamp


class PortfolioWorkspace:
    """One USD simulated account, initially flat, with an immutable capital base."""

    def __init__(self, base_dir: Path, *, oms: OfflineOMS | None = None):
        self.base_dir = Path(base_dir).absolute()
        self.oms = oms

    def _path(self, name: str) -> Path:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("portfolio paths must remain in the fixed workspace")
        path = self.base_dir / relative
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError("portfolio state cannot follow symbolic links")
        return path

    @staticmethod
    def _identity(value: Any) -> str:
        if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
            raise ValueError("invalid portfolio checkpoint identity")
        return value

    @staticmethod
    def _freshness(value: Any) -> float | None:
        if value is None:
            return None
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 366 * 86400):
            raise ValueError("portfolio requires a finite bounded freshness policy")
        return float(value)

    @staticmethod
    def _missing(state: State = State.UNAVAILABLE, *, as_of: Any = None) -> Contract:
        return Contract.admit(state, None, source="durable-simulated-portfolio",
                              classification="UNKNOWN" if state == State.UNKNOWN else "UNAVAILABLE",
                              as_of=as_of, scope="workspace.portfolio",
                              error=public_error("DATA_UNAVAILABLE") if state == State.UNAVAILABLE else None,
                              evidence_state="SIMULATED_NOT_BROKER_OBSERVED")

    def _execution_snapshot(self) -> tuple[list[dict[str, Any]], dict[str, str], Any] | None:
        path = self._path("execution-engine/oms.sqlite")
        if not path.is_file():
            return None
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
            connection.execute("BEGIN")
            count = connection.execute("SELECT COUNT(*) FROM executions").fetchone()[0]
            if count > 10000:
                raise ValueError("portfolio execution replay exceeds bound")
            fills = [dict(zip(("execution_id", "client_id", "symbol", "side", "quantity", "price",
                               "fee", "timestamp", "classification"), row, strict=True))
                     for row in connection.execute(
                         "SELECT e.execution_id,e.client_id,o.symbol,o.side,e.quantity,e.price,e.fee,e.timestamp,e.classification "
                         "FROM executions e LEFT JOIN orders o ON e.client_id=o.client_id ORDER BY e.rowid")]
            positions = {row[0]: row[1] for row in connection.execute("SELECT symbol,quantity FROM positions")}
            # This is a current-account snapshot, not a reconstruction that can
            # pretend a delayed execution was known at its occurrence time.
            known: list[Any] = []
            for table, column in (("events", "timestamp"), ("transport_events", "received_at"),
                                  ("twap_clock", "timestamp")):
                if not connection.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone():
                    continue
                count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                if count > 100000:
                    raise ValueError("execution knowledge archive exceeds read bound")
                known.extend(aware_timestamp(row[0]) for row in connection.execute(f"SELECT {column} FROM {table}"))
        for symbol, quantity in positions.items():
            _text(symbol, "instrument")
            signed_decimal(quantity)
        return canonical_fills(fills), positions, max(known) if known else None

    def load(self, checkpoint_id: str) -> dict[str, Any]:
        identity = self._identity(checkpoint_id)
        path = self._path(f"portfolio-ledger/checkpoints/{identity}.json")
        encoded = path.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != identity:
            raise ValueError("portfolio checkpoint integrity failure")
        data = loads(encoded)
        if not isinstance(data, dict) or data.get("schema_version") != "1":
            raise ValueError("invalid portfolio checkpoint")
        if dumps(data).encode() != encoded:
            raise ValueError("portfolio checkpoint is not canonical")
        return {**data, "checkpoint_id": identity}

    def _history(self) -> list[dict[str, Any]]:
        database = self._path("portfolio-ledger/index.sqlite")
        rows = []
        if database.is_file():
            with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
                rows = list(connection.execute("SELECT checkpoint_id,as_of,intent_sha256 FROM checkpoints ORDER BY rowid LIMIT 1001"))
        execution = self._path("execution-engine/oms.sqlite")
        if self.oms is not None:
            rows += self._execution_history(self.oms.connection)
        elif execution.is_file():
            with sqlite3.connect(execution.as_uri() + "?mode=ro", uri=True) as connection:
                rows += self._execution_history(connection)
        rows = list(dict.fromkeys(rows))
        if len(rows) > 1000:
            raise ValueError("portfolio checkpoint read bound exceeded")
        history = []
        for identity, timestamp, intent in rows:
            data = self.load(identity)
            if data["as_of"] != timestamp or data["intent_sha256"] != intent:
                raise ValueError("portfolio checkpoint differs from index")
            history.append(data)
        return sorted(history, key=lambda value: aware_timestamp(value["as_of"]))

    @staticmethod
    def _execution_history(connection):
        if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='portfolio_checkpoints'").fetchone():
            return []
        return [tuple(row) for row in connection.execute(
            "SELECT checkpoint_id,as_of,intent_sha256 FROM portfolio_checkpoints ORDER BY rowid LIMIT 1001")]

    @staticmethod
    def _contract(checkpoint: dict[str, Any], section: str, state: State | None = None) -> Contract:
        state = State(state or checkpoint["state"])
        identity = checkpoint["checkpoint_id"]
        account, risk = checkpoint["account"], checkpoint["risk"]
        rows = account["rows"] if section != "risk" else [
            {"control": key, "violation": key in risk["violations"], "limit": value}
            for key, value in risk["limits"].items()]
        if section == "risk":
            rows += [{"scenario": row["name"], "pnl": row["pnl"], "nav_return": row["nav_return"]}
                     for row in risk["scenarios"]]
        data = {"title": "Reconciled simulated portfolio" if section != "risk" else "Simulated portfolio risk",
                "metrics": {"nav": account["nav"], "cash": account["cash"],
                            "realized_gross_pnl": account["realized_gross_pnl"],
                            "unrealized_pnl": account["unrealized_pnl"], "fees": account["fees"],
                            "gross_exposure": risk["gross_exposure"], "net_exposure": risk["net_exposure"],
                            "current_drawdown": risk["current_drawdown"], "broker_connected": False},
                "rows": rows, "checkpoint_id": identity, "as_of": checkpoint["as_of"],
                "account": account, "risk": risk, "source_identity": checkpoint["request"].get("source_identity"),
                "input_classification": checkpoint["request"].get("input_classification", "UNKNOWN"),
                "classification": "SIMULATED", "mark_evidence_state": "CALLER_SUPPLIED_NOT_VENDOR_VERIFIED",
                "trading_approved": False}
        return Contract.admit(state, data if state in (State.AVAILABLE, State.STALE) else None,
                              source="durable-simulated-portfolio", classification="SYNTHETIC",
                              as_of=checkpoint["observed_at"], scope=f"workspace.{section}",
                              source_identity=identity, evidence_state="SIMULATED_NOT_BROKER_OBSERVED",
                              error=public_error("INVALID_REQUEST") if state == State.ERROR else None)

    def run(self, payload: dict[str, Any]) -> Contract:
        if not isinstance(payload, dict):
            raise ValueError("portfolio request must be an object")
        request = loads(dumps(payload))
        required = {"operation", "initial_cash", "now", "marks", "max_age_seconds"}
        allowed = required | {"metadata", "limits", "scenarios", "confidence", "source_identity", "input_classification"}
        if not required <= request.keys() or set(request) - allowed:
            raise ValueError("portfolio request has unsupported or missing fields")
        if request["operation"] not in {"mark", "reconcile"}:
            raise ValueError("unsupported portfolio operation")
        now = aware_timestamp(request["now"])
        request["now"] = now.isoformat()
        initial = _decimal(request["initial_cash"])
        request["initial_cash"] = str(initial)
        maximum = self._freshness(request["max_age_seconds"])
        if "source_identity" in request:
            _text(request["source_identity"], "source identity")
        if request.get("input_classification", "UNKNOWN") not in {
            "UNKNOWN", "SYNTHETIC", "HISTORICAL", "REAL_TIME", "DELAYED", "SNAPSHOT", "MODEL_ESTIMATE"}:
            raise ValueError("unsupported input classification")
        marks = request["marks"]
        if not isinstance(marks, dict) or len(marks) > 64:
            raise ValueError("portfolio marks require a bounded mapping")
        for symbol, mark in marks.items():
            _text(symbol, "instrument")
            if not isinstance(mark, dict) or set(mark) != {"price", "as_of"}:
                raise ValueError("marks require price and original timestamp only")
            mark["price"] = str(_decimal(mark["price"]))
            at = aware_timestamp(mark["as_of"])
            if at > now:
                raise ValueError("portfolio mark lies in the future")
            mark["as_of"] = at.isoformat()
        source = self._execution_snapshot()
        if source is None:
            return self._missing()
        fills, positions, knowledge_time = source
        if ((knowledge_time is not None and knowledge_time > now)
                or any(aware_timestamp(fill["timestamp"]) > now for fill in fills)):
            return self._missing(State.UNKNOWN, as_of=now.isoformat())
        held = {symbol for symbol, quantity in positions.items() if signed_decimal(quantity)}
        if held - marks.keys():
            return self._missing()
        if maximum is None:
            return self._missing(State.UNKNOWN, as_of=now.isoformat())
        history = self._history()
        current_evidence = {fill["execution_id"]: fill for fill in fills}
        for previous in history:
            if Decimal(previous["initial_cash"]) != initial:
                raise ValueError("initial simulation cash cannot change after ledger admission")
            for fill in previous["execution_evidence"]:
                if current_evidence.get(fill["execution_id"]) != fill:
                    raise ValueError("execution history changed after portfolio checkpoint")
        source_hash = hashlib.sha256(dumps(fills).encode()).hexdigest()
        intent = hashlib.sha256(dumps({"request": request, "executions_sha256": source_hash}).encode()).hexdigest()
        for previous in history:
            if previous["intent_sha256"] == intent:
                return self._contract(previous, "portfolio")
        if history and now < max(aware_timestamp(item["as_of"]) for item in history):
            raise ValueError("portfolio simulation clock cannot move backward")
        account = reconcile_account(fills, str(initial), {symbol: row["price"] for symbol, row in marks.items()})
        expected = {row["instrument_id"]: Decimal(row["quantity"]) for row in account["rows"] if Decimal(row["quantity"])}
        actual = {symbol: signed_decimal(quantity) for symbol, quantity in positions.items() if signed_decimal(quantity)}
        mismatch = expected != actual
        risk = risk_report(account, metadata=request.get("metadata", {}), limits=request.get("limits", {}),
                           nav_history=[(item["as_of"], item["nav"]) for item in history],
                           now=now.isoformat(), confidence=request.get("confidence", .95),
                           scenarios=request.get("scenarios", []))
        if mismatch:
            risk["violations"].append("OMS_POSITION_MISMATCH")
        observed = min((aware_timestamp(marks[symbol]["as_of"]) for symbol in held), default=now)
        stale = any((now - aware_timestamp(marks[symbol]["as_of"])).total_seconds() > maximum for symbol in held)
        state = State.ERROR if risk["violations"] else State.STALE if stale else State.AVAILABLE
        document = {"schema_version": "1", "as_of": now.isoformat(), "observed_at": observed.isoformat(),
                    "intent_sha256": intent, "executions_sha256": source_hash, "execution_evidence": fills,
                    "initial_cash": str(initial), "nav": account["nav"], "cash": account["cash"],
                    "positions": account["rows"], "account": account, "risk": risk, "request": request,
                    "state": state.value, "oms_positions": positions, "classification": "SIMULATED"}
        encoded = dumps(document).encode()
        identity = hashlib.sha256(encoded).hexdigest()
        checkpoint = {**document, "checkpoint_id": identity}
        contract = self._contract(checkpoint, "portfolio")
        self._path("portfolio-ledger/checkpoints")
        write_immutable(self._path(f"portfolio-ledger/checkpoints/{identity}.json"), encoded)
        if self.oms is not None:
            if not self.oms.connection.in_transaction:
                raise ValueError("execution-linked checkpoints require the OMS transaction")
            self.oms.connection.execute("CREATE TABLE IF NOT EXISTS portfolio_checkpoints(checkpoint_id TEXT PRIMARY KEY,as_of TEXT NOT NULL,intent_sha256 TEXT NOT NULL UNIQUE)")
            if len(history) >= 1000:
                raise ValueError("portfolio supports at most 1000 retained checkpoints")
            self.oms.connection.execute("INSERT OR IGNORE INTO portfolio_checkpoints VALUES(?,?,?)",
                                        (identity, now.isoformat(), intent))
            return contract
        with sqlite3.connect(self._path("portfolio-ledger/index.sqlite")) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("CREATE TABLE IF NOT EXISTS checkpoints(checkpoint_id TEXT PRIMARY KEY,as_of TEXT NOT NULL,intent_sha256 TEXT NOT NULL UNIQUE)")
            count = connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
            if count >= 1000:
                raise ValueError("portfolio supports at most 1000 retained checkpoints")
            # The database lock closes the multi-process publication race.
            indexed = list(connection.execute("SELECT checkpoint_id,as_of FROM checkpoints"))
            for prior, timestamp in indexed:
                saved = self.load(prior)
                if Decimal(saved["initial_cash"]) != initial or aware_timestamp(timestamp) > now:
                    raise ValueError("concurrent portfolio state conflicts with request")
                for fill in saved["execution_evidence"]:
                    if current_evidence.get(fill["execution_id"]) != fill:
                        raise ValueError("concurrent execution evidence differs")
            connection.execute("INSERT OR IGNORE INTO checkpoints VALUES(?,?,?)", (identity, now.isoformat(), intent))
        return contract

    def section(self, section: str, as_of: Any, max_age_seconds: float | None) -> Contract | None:
        if section not in {"portfolio", "risk"}:
            return None
        cutoff, maximum = aware_timestamp(as_of), self._freshness(max_age_seconds)
        history = [item for item in self._history() if aware_timestamp(item["as_of"]) <= cutoff]
        if not history:
            return None
        selected = max(enumerate(history), key=lambda pair: (aware_timestamp(pair[1]["as_of"]), pair[0]))[1]
        state = State(selected["state"])
        if state != State.ERROR:
            state = (State.UNKNOWN if maximum is None else State.STALE
                     if state == State.STALE or (cutoff - aware_timestamp(selected["observed_at"])).total_seconds() > maximum
                     else State.AVAILABLE)
        return self._contract(selected, section, state)
