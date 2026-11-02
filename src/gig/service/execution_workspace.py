"""Fixed-path offline EMS workspace; no broker object or credentials are accepted."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from gig.execution.offline_adapter import OfflinePaperAdapter
from gig.execution.oms import OfflineOMS, OrderError, _decimal, _text
from gig.execution.outbox import ExecutionOutbox, event_value
from gig.execution.scheduler import DurableTWAPScheduler
from gig.execution.twap import plan_twap
from gig.service.contracts import Contract, State
from gig.service.serialization import dumps, public_error
from gig.timestamps import aware_timestamp


def _stamp(value: Any) -> datetime:
    stamp = aware_timestamp(value)
    if stamp.nanosecond:
        raise OrderError("offline execution timestamps require microsecond precision")
    return stamp.to_pydatetime()


class ExecutionWorkspace:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir).absolute()

    def _path(self, name: str) -> Path:
        path = self.base_dir / name
        if any(p.is_symlink() for p in (path, self.base_dir, *self.base_dir.parents)):
            raise OrderError("execution state paths must not be symbolic links")
        return path

    @staticmethod
    def _contract(
        data: dict[str, Any] | None,
        at: datetime,
        *,
        state: State = State.AVAILABLE,
        scope: str = "execution",
    ) -> Contract:
        return Contract.admit(
            state,
            data if state in (State.AVAILABLE, State.STALE) else None,
            source="durable-offline-execution-engine",
            classification="SYNTHETIC",
            as_of=at.isoformat(),
            scope=f"workspace.{scope}",
            evidence_state="SIMULATED_NOT_BROKER_OBSERVED",
            readiness="RESEARCH_ONLY",
            error=public_error("INVALID_REQUEST")
            if state == State.ERROR
            else public_error("DATA_UNAVAILABLE")
            if state == State.UNAVAILABLE
            else None,
        )

    def _snapshot(self, oms: OfflineOMS, outbox: ExecutionOutbox) -> dict[str, Any]:
        scheduler = DurableTWAPScheduler(oms, outbox, create=False)
        total = oms.connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        ids = [
            row[0]
            for row in oms.connection.execute(
                "SELECT client_id FROM orders ORDER BY created_at,client_id LIMIT 100"
            )
        ]
        return {
            "title": "Durable offline execution",
            "description": "Simulated execution evidence; no brokerage account is connected.",
            "metrics": {
                "order_count": total,
                "kill_switch": oms.kill_switch,
                "broker_connected": False,
                "nav": None,
                "pnl": None,
                "simulation_clock": scheduler.clock,
            },
            "rows": [oms.order(client) for client in ids],
            "positions": oms.positions(),
            "outbox": outbox.intents()[:100],
            "recovery": outbox.recover(),
            "schedules": scheduler.schedules(),
            "rows_complete": total <= 100,
            "trading_approved": False,
            "classification": "SIMULATED",
        }

    def run(self, payload: dict[str, Any]) -> Contract:
        dumps(payload)
        request = json.loads(json.dumps(payload, allow_nan=False, ensure_ascii=False))
        if not isinstance(request, dict) or not {"operation", "now"} <= request.keys():
            raise OrderError("execution request requires operation and explicit now")
        op, at = request["operation"], _stamp(request["now"])
        if not isinstance(op, str):
            raise OrderError("execution operation must be a supported string")
        order_fields = {
            "client_id",
            "symbol",
            "side",
            "quantity",
            "reference_price",
            "price_asof",
            "max_notional",
            "max_position_quantity",
            "limit_price",
        }
        required_order = order_fields - {"limit_price"}
        controls = {"disconnected", "fault", "reject_next", "queries_unknown"}
        shapes = {
            "prepare": (order_fields, required_order),
            "dispatch": (controls | {"limit"}, set()),
            "event": ({"event"}, {"event"}),
            "fill": ({"client_id", "quantity", "price", "fee"}, {"client_id", "quantity", "price"}),
            "cancel": ({"client_id"}, {"client_id"}),
            "retry-absent": ({"client_id"}, {"client_id"}),
            "reconcile": (controls, set()),
            "status": (set(), set()),
            "replay": (set(), set()),
            "kill-switch": ({"engaged"}, {"engaged"}),
            "twap": (
                {"client_id", "quantity", "start", "end", "slices", "lot_size"},
                {"client_id", "quantity", "start", "end", "slices"},
            ),
            "schedule-twap": (
                {"client_id", "symbol", "side", "quantity", "start", "end", "slices",
                 "lot_size", "max_notional", "max_position_quantity", "limit_price"},
                {"client_id", "symbol", "side", "quantity", "start", "end", "slices",
                 "max_notional", "max_position_quantity"},
            ),
            "advance-clock": ({"marks"}, {"marks"}),
            "cancel-schedule": ({"client_id"}, {"client_id"}),
        }
        if op not in shapes:
            raise OrderError("unsupported offline execution operation")
        allowed, required = shapes[op]
        values = {key: value for key, value in request.items() if key not in {"operation", "now"}}
        if set(values) - allowed or not required <= values.keys():
            raise OrderError("unsupported or missing execution fields")
        if "client_id" in values:
            _text(values["client_id"], "client_id")
        if op == "prepare":
            _text(values["symbol"], "symbol")
            if not isinstance(values["side"], str) or values["side"] not in {"buy", "sell"}:
                raise OrderError("unsupported order side")
            for key in ("quantity", "reference_price", "max_notional", "max_position_quantity"):
                _decimal(values[key])
            if values.get("limit_price") is not None:
                _decimal(values["limit_price"])
            values["price_asof"] = _stamp(values["price_asof"])
        elif op == "event":
            values["event"] = event_value(values["event"])
        elif op == "fill":
            for key in ("quantity", "price"):
                _decimal(values[key])
            _decimal(values.get("fee", "0"), positive=False)
        elif op == "kill-switch" and not isinstance(values["engaged"], bool):
            raise OrderError("kill switch requires an explicit boolean")
        elif op == "schedule-twap":
            DurableTWAPScheduler._intent(values)
        elif op == "advance-clock":
            DurableTWAPScheduler._marks(values["marks"])
        elif op == "twap":
            values["start"], values["end"] = _stamp(values["start"]), _stamp(values["end"])
            if (
                isinstance(values["slices"], bool)
                or not isinstance(values["slices"], int)
                or not 1 <= values["slices"] <= 100
            ):
                raise OrderError("workspace TWAP supports one to 100 slices")
            return self._contract(
                {
                    "title": "Offline TWAP plan",
                    "rows": [asdict(row) for row in plan_twap(**values)],
                    "broker_connected": False,
                    "trading_approved": False,
                },
                at,
            )
        for key in controls - {"fault"}:
            if key in values and not isinstance(values[key], bool):
                raise OrderError("offline fault controls require booleans")
        if values.get("fault") not in (None, "before_accept", "after_accept"):
            raise OrderError("unsupported offline fault")
        path = self._path("oms.sqlite")
        if op not in {"prepare", "schedule-twap"} and not path.is_file():
            return self._contract(None, at, state=State.UNAVAILABLE)
        if op == "schedule-twap" and not path.is_file() and _stamp(values["start"]) < at:
            raise OrderError("schedule start precedes admission clock")
        self._path("counterparty.sqlite")
        with OfflineOMS(path) as oms:
            outbox = ExecutionOutbox(oms)
            scheduler = DurableTWAPScheduler(
                oms, outbox, create=op in {"schedule-twap", "advance-clock", "cancel-schedule"}
            )
            scheduler.ensure_not_before(at)
            adapter = OfflinePaperAdapter(self._path("counterparty.sqlite"))
            try:
                adapter.disconnected = values.pop("disconnected", False)
                adapter.fault = values.pop("fault", None)
                adapter.reject_next = values.pop("reject_next", False)
                adapter.queries_unknown = values.pop("queries_unknown", False)
                if op == "prepare":
                    count = oms.connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
                    known = oms.connection.execute(
                        "SELECT 1 FROM orders WHERE client_id=?", (values["client_id"],)
                    ).fetchone()
                    if count >= 100 and known is None:
                        raise OrderError("offline workspace supports at most 100 retained orders")
                    result = outbox.admit_order(**values, now=at)
                elif op == "dispatch":
                    result = outbox.dispatch(adapter, now=at, **values)
                elif op == "event":
                    result = outbox.ingest_event(values["event"], received_at=at)
                elif op == "fill":
                    event = adapter.fill(**values, now=at)
                    result = outbox.ingest_event(event, received_at=at)
                elif op == "cancel":
                    result = outbox.request_cancel(**values, now=at)
                elif op == "retry-absent":
                    result = outbox.retry_absent(**values, now=at)
                elif op == "reconcile":
                    result = outbox.reconcile(adapter, now=at)
                elif op == "kill-switch":
                    oms.set_kill_switch(values["engaged"], now=at)
                    result = {"engaged": oms.kill_switch}
                elif op == "schedule-twap":
                    result = scheduler.schedule(values, now=at)
                elif op == "advance-clock":
                    result = scheduler.advance(values["marks"], now=at)
                elif op == "cancel-schedule":
                    result = scheduler.cancel(values["client_id"], now=at)
                elif op == "replay":
                    result = {"events": outbox.replay_events()}
                else:
                    result = outbox.recover()
                data = self._snapshot(oms, outbox)
                data["operation_result"] = result
                state = (
                    State.ERROR
                    if result.get("state") == "ERROR"
                    else State.UNKNOWN
                    if result.get("state") == "UNKNOWN"
                    else State.AVAILABLE
                )
                return self._contract(data, at, state=state)
            finally:
                adapter.close()

    def section(self, section: str, as_of: Any, max_age_seconds: float | None) -> Contract | None:
        if section not in {"execution", "portfolio", "risk"}:
            return None
        at = _stamp(as_of)
        if max_age_seconds is not None and (
            isinstance(max_age_seconds, bool)
            or not math.isfinite(max_age_seconds)
            or max_age_seconds < 0
        ):
            raise OrderError("invalid execution view freshness")
        path = self._path("oms.sqlite")
        if not path.is_file():
            return self._contract(None, at, state=State.UNAVAILABLE)
        with OfflineOMS(path) as oms:
            outbox = ExecutionOutbox(oms)
            timestamps = [
                datetime.fromisoformat(row[0])
                for row in oms.connection.execute("SELECT timestamp FROM events")
            ]
            if not timestamps:
                return self._contract(None, at, state=State.UNAVAILABLE)
            observed = max(timestamps)
            if observed > at:
                # This is current execution state, not a fabricated historical
                # ledger. Historical event replay is exposed separately.
                return self._contract(None, at, state=State.UNKNOWN)
            data = self._snapshot(oms, outbox)
            if section == "portfolio":
                data["rows"] = [
                    {"instrument_id": key, "quantity": value, "classification": "SIMULATED"}
                    for key, value in oms.positions().items()
                ]
            elif section == "risk":
                data["rows"] = [
                    {"control": "kill_switch", "engaged": oms.kill_switch},
                    {
                        "control": "reconciliation_required",
                        "engaged": outbox.recover()["requires_reconciliation"],
                    },
                ]
            state = (
                State.UNKNOWN
                if max_age_seconds is None
                else State.STALE
                if (at - observed).total_seconds() > max_age_seconds
                else State.AVAILABLE
            )
            return self._contract(data, observed, state=state, scope=f"terminal.{section}")
