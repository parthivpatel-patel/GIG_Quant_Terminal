"""Offline research strategy simulation from a saved buy/hold/sell screen.

Builds an allocation, prepares durable intents, and optionally acknowledges
and fills them in the offline OMS. Position sizing converts research weights
into capped notionals and lot quantities. Nothing here enables broker
connectivity or trading approval.
"""
from __future__ import annotations

import hashlib
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from gig.artifacts import write_immutable
from gig.data.validated import PointInTimeStore
from gig.research.model_pipeline import sha256
from gig.service.allocation_workspace import ResearchAllocationWorkspace
from gig.service.contracts import Contract, State
from gig.service.dataset_workspace import PersistedDatasetWorkspace
from gig.service.execution_workspace import ExecutionWorkspace
from gig.service.research_screen import ResearchScreenWorkspace
from gig.service.serialization import dumps, loads, public_error
from gig.service.workspace import TerminalWorkspace, _fields
from gig.timestamps import aware_timestamp

IMPLEMENTATION = "gig.research-strategy.v1"
SIZING = "gig.research-strategy.sizing.v1"


def _hash(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def _positive(value: Any, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ValueError(f"{name} must be a positive decimal")
    number = Decimal(str(value))
    if number <= 0 or not number.is_finite():
        raise ValueError(f"{name} must be a positive decimal")
    return number


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
    def _contract(document: dict[str, Any], *, state: State = State.AVAILABLE,
                  title: str = "Unapproved offline research strategy simulation",
                  model_version: str = IMPLEMENTATION,
                  evidence: str = "HASH_VERIFIED_OFFLINE_STRATEGY_SIMULATION",
                  identity_key: str = "strategy_id") -> Contract:
        return Contract.admit(
            state,
            {**document, "title": title}
            if state in {State.AVAILABLE, State.STALE} else None,
            source=document["source"],
            classification="MODEL_ESTIMATE",
            as_of=document["as_of"],
            scope="workspace.research-strategy",
            source_identity=document["dataset_id"],
            model_identity=document[identity_key],
            model_version=model_version,
            evidence_state=evidence,
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

    def size(self, payload: dict[str, Any]) -> Contract:
        """Convert screen research weights into capped research lot sizes.

        Prices come from admitted bar evidence. Vol scaling is optional and
        caller-supplied. No order is prepared or submitted.
        """
        fields = _fields(payload, {"screen_id", "now", "portfolio_notional", "max_name_notional",
                                   "lot_size", "max_age_seconds", "vol_scale"},
                         {"screen_id", "now", "portfolio_notional", "max_name_notional", "lot_size"})
        try:
            screen = self.screens.load(fields["screen_id"])
            dataset = PersistedDatasetWorkspace(self.base_dir).load(screen["dataset_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        at = aware_timestamp(fields["now"])
        portfolio = _positive(fields["portfolio_notional"], "portfolio_notional")
        name_cap = _positive(fields["max_name_notional"], "max_name_notional")
        lot = int(fields["lot_size"])
        if lot < 1:
            raise ValueError("lot_size must be a positive integer")
        maximum = int(fields.get("max_age_seconds", 86_400))
        if maximum < 0:
            raise ValueError("max_age_seconds must be non-negative")
        scale = Decimal("1")
        if "vol_scale" in fields:
            scale = _positive(fields["vol_scale"], "vol_scale")
            if scale > Decimal("10"):
                raise ValueError("vol_scale exceeds the research bound")
        actionable = [row for row in screen["rows"] if float(row["research_weight"]) != 0.0]
        if not actionable:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        symbols = sorted({row["instrument_id"] for row in actionable})
        source_policy = dataset["provenance"]["stored_corporate_action_policy"]
        snapshot = PointInTimeStore(self.workspace._path("bars.sqlite"), read_only=True).snapshot(
            at, source=dataset["source"])
        marks: dict[str, dict[str, str]] = {}
        for row in snapshot["rows"]:
            symbol = row["instrument_id"]
            if (symbol in symbols and row["classification"] == dataset["classification"]
                    and row["currency"] == dataset.get("currency", "USD")
                    and row["corporate_action_policy"] == source_policy
                    and (symbol not in marks
                         or aware_timestamp(row["timestamp"]) > aware_timestamp(marks[symbol]["as_of"]))):
                marks[symbol] = {"price": str(row["close"]), "as_of": row["timestamp"]}
        if set(marks) != set(symbols):
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        for mark in marks.values():
            age = (at - aware_timestamp(mark["as_of"])).total_seconds()
            if age > maximum:
                return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                      classification="UNAVAILABLE", scope="workspace.research-strategy",
                                      error=public_error("DATA_UNAVAILABLE"))
        sized: list[dict[str, Any]] = []
        for row in sorted(actionable, key=lambda item: item["instrument_id"]):
            weight = Decimal(str(row["research_weight"]))
            raw = (weight * portfolio * scale).copy_abs()
            notional = min(raw, name_cap)
            price = Decimal(marks[row["instrument_id"]]["price"])
            quantity = int(Fraction(notional) / Fraction(price) / Fraction(lot)) * lot
            if quantity <= 0:
                continue
            sized.append({
                "instrument_id": row["instrument_id"],
                "stance": row["stance"],
                "research_weight": float(row["research_weight"]),
                "side": "buy" if weight > 0 else "sell",
                "target_notional": str(notional),
                "uncapped_notional": str(raw),
                "capped": raw > name_cap,
                "quantity": str(quantity),
                "reference_price": marks[row["instrument_id"]]["price"],
                "price_asof": marks[row["instrument_id"]]["as_of"],
                "trading_approved": False,
            })
        if not sized:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        body = {
            "schema_version": SIZING,
            "as_of": at.isoformat(),
            "dataset_id": screen["dataset_id"],
            "screen_id": screen["screen_id"],
            "signal_id": screen["signal_id"],
            "source": screen["source"],
            "input_classification": screen["input_classification"],
            "request": {
                "portfolio_notional": str(portfolio),
                "max_name_notional": str(name_cap),
                "lot_size": lot,
                "max_age_seconds": maximum,
                "vol_scale": str(scale),
            },
            "rows": sized,
            "marks": marks,
            "broker_connected": False,
            "trading_approved": False,
            "economically_validated": False,
            "classification": "SIMULATED_PLAN",
            "metrics": {
                "names": len(sized),
                "gross_notional": str(sum((Decimal(row["target_notional"]) for row in sized), Decimal(0))),
                "capped_names": sum(1 for row in sized if row["capped"]),
                "trading_approved": False,
            },
            "notes": [
                "Position sizes are research lot targets from screen weights and admitted marks.",
                "max_name_notional caps each name; vol_scale is caller-supplied and not estimated here.",
                "No broker order, OMS prepare, or trading approval is established.",
            ],
        }
        identity = _hash(body)
        document = {**body, "size_id": identity}
        self._path("sizes").mkdir(parents=True, exist_ok=True)
        write_immutable(self._path("sizes/" + identity + ".json"), dumps(document).encode())
        return self._contract(document, title="Unapproved research position sizes",
                              model_version=SIZING,
                              evidence="HASH_VERIFIED_RESEARCH_POSITION_SIZING",
                              identity_key="size_id")

    def load_size(self, size_id: str) -> dict[str, Any]:
        identity = sha256(size_id)
        path = self._path("sizes/" + identity + ".json")
        if not path.is_file():
            raise FileNotFoundError("research position size unavailable")
        document = loads(path.read_bytes())
        body = {k: v for k, v in document.items() if k != "size_id"}
        if document.get("size_id") != identity or _hash(body) != identity or body.get("schema_version") != SIZING:
            raise ValueError("position size artifact integrity mismatch")
        return document

    def replay_size(self, payload: dict[str, Any]) -> Contract:
        _fields(payload, {"size_id"}, {"size_id"})
        try:
            document = self.load_size(payload["size_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        return self._contract(document, title="Unapproved research position sizes",
                              model_version=SIZING,
                              evidence="HASH_VERIFIED_RESEARCH_POSITION_SIZING",
                              identity_key="size_id")

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
