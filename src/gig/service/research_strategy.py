"""Offline research strategy simulation from a saved buy/hold/sell screen.

Builds an allocation, prepares durable intents, and optionally acknowledges
and fills them in the offline OMS. Nothing here enables broker connectivity
or trading approval.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from gig.artifacts import write_immutable
from gig.research.model_pipeline import sha256
from gig.service.allocation_workspace import ResearchAllocationWorkspace
from gig.service.contracts import Contract, State
from gig.service.execution_workspace import ExecutionWorkspace
from gig.service.research_screen import ResearchScreenWorkspace
from gig.service.serialization import dumps, loads, public_error
from gig.service.workspace import TerminalWorkspace, _fields
from gig.timestamps import aware_timestamp

IMPLEMENTATION = "gig.research-strategy.v1"


def _hash(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode()).hexdigest()


class ResearchStrategyWorkspace:
    def __init__(self, base_dir: Path):
        self.workspace = TerminalWorkspace(base_dir)
        self.base_dir = self.workspace.base_dir
        self.screens = ResearchScreenWorkspace(base_dir)
        self.allocations = ResearchAllocationWorkspace(base_dir)
        self.execution = ExecutionWorkspace(self.workspace._path("execution-engine"))

    def _path(self, name: str) -> Path:
        return self.workspace._path("research-strategies/" + name)

    @staticmethod
    def _contract(document: dict[str, Any], *, state: State = State.AVAILABLE) -> Contract:
        return Contract.admit(
            state,
            {**document, "title": "Unapproved offline research strategy simulation"}
            if state in {State.AVAILABLE, State.STALE} else None,
            source=document["source"],
            classification="MODEL_ESTIMATE",
            as_of=document["as_of"],
            scope="workspace.research-strategy",
            source_identity=document["dataset_id"],
            model_identity=document["strategy_id"],
            model_version=IMPLEMENTATION,
            evidence_state="HASH_VERIFIED_OFFLINE_STRATEGY_SIMULATION",
            readiness="RESEARCH_ONLY",
        )

    def run(self, payload: dict[str, Any]) -> Contract:
        allowed = {"screen_id", "now", "initial_cash", "n_long", "n_short", "gross_leverage",
                   "max_name_weight", "lot_size", "max_age_seconds", "annualization", "metadata",
                   "limits", "commission_bps", "slippage_bps", "price_tick", "max_turnover",
                   "max_transaction_cost", "fill"}
        fields = _fields(payload, allowed, {"screen_id", "now", "initial_cash", "n_long", "n_short",
                                            "gross_leverage", "max_name_weight", "lot_size",
                                            "max_age_seconds", "annualization", "metadata", "limits"})
        fill = fields.get("fill", False)
        if not isinstance(fill, bool):
            raise ValueError("fill must be a boolean")
        try:
            screen = self.screens.load(fields["screen_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        allocation_payload = {key: fields[key] for key in fields if key not in {"fill"}}
        allocation_payload["screen_id"] = screen["screen_id"]
        allocated = self.allocations.build(allocation_payload).to_dict()
        if allocated["state"] != State.AVAILABLE.value or not isinstance(allocated.get("data"), dict):
            return Contract.admit(allocated["state"], allocated.get("data"),
                                  source="persisted-research-strategy",
                                  classification=allocated["provenance"]["classification"],
                                  as_of=allocated["provenance"].get("as_of"),
                                  scope="workspace.research-strategy",
                                  error=allocated.get("error"),
                                  readiness="RESEARCH_ONLY")
        plan = allocated["data"]
        prepared = self.allocations.prepare({
            "allocation_id": plan["allocation_id"], "now": fields["now"],
        }).to_dict()
        if prepared["state"] != State.AVAILABLE.value or not isinstance(prepared.get("data"), dict):
            return Contract.admit(prepared["state"], prepared.get("data"),
                                  source="persisted-research-strategy",
                                  classification=prepared["provenance"]["classification"],
                                  as_of=prepared["provenance"].get("as_of"),
                                  scope="workspace.research-strategy",
                                  error=prepared.get("error"),
                                  readiness="RESEARCH_ONLY")
        prep = prepared["data"]
        fills: list[dict[str, Any]] = []
        if fill:
            self.execution.run({"operation": "dispatch", "now": fields["now"]})
            for client, row in zip(prep["prepared_client_ids"], prep["rows"], strict=True):
                filled = self.execution.run({
                    "operation": "fill", "client_id": client, "quantity": row["quantity"],
                    "price": row["reference_price"], "fee": "0", "now": fields["now"],
                }).to_dict()
                if filled["state"] != State.AVAILABLE.value:
                    raise ValueError("offline fill failed during research simulation")
                fills.append({"client_id": client, "instrument_id": row["instrument_id"],
                              "side": row["side"], "quantity": row["quantity"],
                              "price": row["reference_price"]})
            self.execution.run({"operation": "reconcile", "now": fields["now"]})
        body = {
            "schema_version": IMPLEMENTATION,
            "as_of": aware_timestamp(fields["now"]).isoformat(),
            "dataset_id": screen["dataset_id"],
            "screen_id": screen["screen_id"],
            "signal_id": screen["signal_id"],
            "allocation_id": plan["allocation_id"],
            "source": screen["source"],
            "input_classification": screen["input_classification"],
            "prepared_client_ids": list(prep["prepared_client_ids"]),
            "allocation_rows": plan["rows"],
            "fills": fills,
            "fill_requested": fill,
            "broker_connected": False,
            "trading_approved": False,
            "economically_validated": False,
            "classification": "SIMULATED_PLAN",
            "metrics": {
                "prepared_orders": len(prep["prepared_client_ids"]),
                "filled_orders": len(fills),
                "turnover_nav": plan["metrics"]["turnover_nav"],
                "trading_approved": False,
            },
            "notes": [
                "Simulation uses the offline OMS only. Broker connectivity stays disabled.",
                "Fills reuse the allocation reference prices and are not market observations.",
                "No trading approval or funded certification is established.",
            ],
        }
        identity = _hash(body)
        document = {**body, "strategy_id": identity}
        self._path("runs").mkdir(parents=True, exist_ok=True)
        write_immutable(self._path("runs/" + identity + ".json"), dumps(document).encode())
        return self._contract(document)

    def load(self, strategy_id: str) -> dict[str, Any]:
        identity = sha256(strategy_id)
        path = self._path("runs/" + identity + ".json")
        if not path.is_file():
            raise FileNotFoundError("research strategy unavailable")
        document = loads(path.read_bytes())
        body = {k: v for k, v in document.items() if k != "strategy_id"}
        if document.get("strategy_id") != identity or _hash(body) != identity or body.get("schema_version") != IMPLEMENTATION:
            raise ValueError("strategy artifact integrity mismatch")
        return document

    def replay(self, payload: dict[str, Any]) -> Contract:
        _fields(payload, {"strategy_id"}, {"strategy_id"})
        try:
            document = self.load(payload["strategy_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        return self._contract(document)
