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

import pandas as pd

from gig.artifacts import write_immutable
from gig.data.validated import PointInTimeStore
from gig.research.model_pipeline import sha256
from gig.risk.analytics import portfolio_risk
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
CYCLE = "gig.research-strategy.cycle.v1"
WATCH = "gig.research-strategy.watch.v1"


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

    def cycle(self, payload: dict[str, Any]) -> Contract:
        """One research cycle: discover from runs → size → offline simulate.

        Broker connectivity stays disabled. The cycle never grants trading
        approval. Size and simulate steps reuse the same screen identity.
        """
        allowed = {"run_ids", "now", "lookback", "min_ic_observations", "buy_threshold", "sell_threshold",
                   "previous_screen_id", "portfolio_notional", "max_name_notional", "vol_scale",
                   "target_volatility", "max_expected_shortfall", "initial_cash", "n_long", "n_short",
                   "gross_leverage", "max_name_weight", "lot_size", "max_age_seconds", "annualization",
                   "metadata", "limits", "commission_bps", "slippage_bps", "price_tick", "max_turnover",
                   "max_transaction_cost", "fill", "document_id", "nlp_profile"}
        fields = _fields(payload, allowed, {"run_ids", "now", "portfolio_notional", "max_name_notional",
                                            "initial_cash", "n_long", "n_short", "gross_leverage",
                                            "max_name_weight", "lot_size", "max_age_seconds",
                                            "annualization", "metadata", "limits"})
        discover_payload = {key: fields[key] for key in fields
                            if key in {"run_ids", "now", "lookback", "min_ic_observations",
                                       "buy_threshold", "sell_threshold", "previous_screen_id"}}
        discovered = self.screens.discover(discover_payload).to_dict()
        if discovered["state"] != State.AVAILABLE.value or not isinstance(discovered.get("data"), dict):
            return Contract.admit(discovered["state"], discovered.get("data"),
                                  source="persisted-research-strategy",
                                  classification=discovered["provenance"]["classification"],
                                  as_of=discovered["provenance"].get("as_of"),
                                  scope="workspace.research-strategy",
                                  error=discovered.get("error"),
                                  readiness="RESEARCH_ONLY")
        screen = discovered["data"]
        enrich_data = None
        if "document_id" in fields:
            enrich_payload = {"screen_id": screen["screen_id"], "document_id": fields["document_id"]}
            if "nlp_profile" in fields:
                enrich_payload["nlp_profile"] = fields["nlp_profile"]
            enriched = self.screens.enrich_mentions(enrich_payload).to_dict()
            if enriched["state"] != State.AVAILABLE.value or not isinstance(enriched.get("data"), dict):
                return Contract.admit(enriched["state"], enriched.get("data"),
                                      source="persisted-research-strategy",
                                      classification=enriched["provenance"]["classification"],
                                      as_of=enriched["provenance"].get("as_of"),
                                      scope="workspace.research-strategy",
                                      error=enriched.get("error"),
                                      readiness="RESEARCH_ONLY")
            enrich_data = enriched["data"]
        size_payload = {
            "screen_id": screen["screen_id"], "now": fields["now"],
            "portfolio_notional": fields["portfolio_notional"],
            "max_name_notional": fields["max_name_notional"],
            "lot_size": fields["lot_size"],
        }
        for key in ("max_age_seconds", "vol_scale", "target_volatility", "max_expected_shortfall",
                    "annualization"):
            if key in fields:
                size_payload[key] = fields[key]
        sized = self.size(size_payload).to_dict()
        if sized["state"] != State.AVAILABLE.value or not isinstance(sized.get("data"), dict):
            return Contract.admit(sized["state"], sized.get("data"),
                                  source="persisted-research-strategy",
                                  classification=sized["provenance"]["classification"],
                                  as_of=sized["provenance"].get("as_of"),
                                  scope="workspace.research-strategy",
                                  error=sized.get("error"),
                                  readiness="RESEARCH_ONLY")
        size_only = {"run_ids", "lookback", "min_ic_observations", "buy_threshold", "sell_threshold",
                     "previous_screen_id", "portfolio_notional", "max_name_notional", "vol_scale",
                     "target_volatility", "max_expected_shortfall", "document_id", "nlp_profile"}
        simulate_payload = {key: fields[key] for key in fields if key not in size_only}
        simulate_payload["screen_id"] = screen["screen_id"]
        simulated = self.run(simulate_payload).to_dict()
        if simulated["state"] != State.AVAILABLE.value or not isinstance(simulated.get("data"), dict):
            return Contract.admit(simulated["state"], simulated.get("data"),
                                  source="persisted-research-strategy",
                                  classification=simulated["provenance"]["classification"],
                                  as_of=simulated["provenance"].get("as_of"),
                                  scope="workspace.research-strategy",
                                  error=simulated.get("error"),
                                  readiness="RESEARCH_ONLY")
        strategy = simulated["data"]
        body = {
            "schema_version": CYCLE,
            "as_of": strategy["as_of"],
            "dataset_id": screen["dataset_id"],
            "screen_id": screen["screen_id"],
            "signal_id": screen["signal_id"],
            "size_id": sized["data"]["size_id"],
            "enrichment_id": enrich_data.get("enrichment_id") if enrich_data is not None else None,
            "strategy_id": strategy["strategy_id"],
            "allocation_id": strategy["allocation_id"],
            "source": screen["source"],
            "input_classification": screen["input_classification"],
            "discovery": screen.get("discovery"),
            "size_metrics": sized["data"]["metrics"],
            "fi_metrics": enrich_data["metrics"] if enrich_data is not None else None,
            "strategy_metrics": strategy["metrics"],
            "prepared_client_ids": strategy["prepared_client_ids"],
            "fills": strategy["fills"],
            "fill_requested": strategy["fill_requested"],
            "broker_connected": False,
            "trading_approved": False,
            "economically_validated": False,
            "classification": "SIMULATED_PLAN",
            "metrics": {
                "research_buys": screen["metrics"]["research_buys"],
                "research_sells": screen["metrics"]["research_sells"],
                "sized_names": sized["data"]["metrics"]["names"],
                "fi_matched": (enrich_data["metrics"]["matched_count"] if enrich_data is not None else 0),
                "prepared_orders": strategy["metrics"]["prepared_orders"],
                "filled_orders": strategy["metrics"]["filled_orders"],
                "trading_approved": False,
            },
            "notes": [
                "Research cycle chains discovery, optional FI enrich, position sizing, and offline simulation.",
                "Sizing is advisory relative to the separate allocation/OMS simulation path.",
                "Broker connectivity stays disabled; trading stays unapproved.",
            ],
        }
        identity = _hash(body)
        document = {**body, "cycle_id": identity}
        self._path("cycles").mkdir(parents=True, exist_ok=True)
        write_immutable(self._path("cycles/" + identity + ".json"), dumps(document).encode())
        return self._contract(document, title="Unapproved research opportunity cycle",
                              model_version=CYCLE,
                              evidence="HASH_VERIFIED_RESEARCH_OPPORTUNITY_CYCLE",
                              identity_key="cycle_id")

    def load_cycle(self, cycle_id: str) -> dict[str, Any]:
        identity = sha256(cycle_id)
        path = self._path("cycles/" + identity + ".json")
        if not path.is_file():
            raise FileNotFoundError("research cycle unavailable")
        document = loads(path.read_bytes())
        body = {k: v for k, v in document.items() if k != "cycle_id"}
        if document.get("cycle_id") != identity or _hash(body) != identity or body.get("schema_version") != CYCLE:
            raise ValueError("research cycle artifact integrity mismatch")
        return document

    def replay_cycle(self, payload: dict[str, Any]) -> Contract:
        _fields(payload, {"cycle_id"}, {"cycle_id"})
        try:
            document = self.load_cycle(payload["cycle_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        return self._contract(document, title="Unapproved research opportunity cycle",
                              model_version=CYCLE,
                              evidence="HASH_VERIFIED_RESEARCH_OPPORTUNITY_CYCLE",
                              identity_key="cycle_id")

    def watch(self, payload: dict[str, Any]) -> Contract:
        """Research watch pass: monitor successive signals, size latest, optional options scan.

        Explicit caller invocation only — not a live market subscription.
        """
        allowed = {"signal_ids", "buy_threshold", "sell_threshold", "now", "portfolio_notional",
                   "max_name_notional", "lot_size", "max_age_seconds", "vol_scale", "target_volatility",
                   "max_expected_shortfall", "annualization", "options_run_ids", "document_id",
                   "nlp_profile"}
        fields = _fields(payload, allowed, {"signal_ids", "now", "portfolio_notional",
                                            "max_name_notional", "lot_size"})
        monitor_payload = {key: fields[key] for key in fields
                           if key in {"signal_ids", "buy_threshold", "sell_threshold"}}
        monitored = self.screens.monitor(monitor_payload).to_dict()
        if monitored["state"] != State.AVAILABLE.value or not isinstance(monitored.get("data"), dict):
            return Contract.admit(monitored["state"], monitored.get("data"),
                                  source="persisted-research-strategy",
                                  classification=monitored["provenance"]["classification"],
                                  as_of=monitored["provenance"].get("as_of"),
                                  scope="workspace.research-strategy",
                                  error=monitored.get("error"),
                                  readiness="RESEARCH_ONLY")
        monitor = monitored["data"]
        screen_id = monitor["latest_screen_id"]
        enrich_data = None
        if "document_id" in fields:
            enrich_payload = {"screen_id": screen_id, "document_id": fields["document_id"]}
            if "nlp_profile" in fields:
                enrich_payload["nlp_profile"] = fields["nlp_profile"]
            enriched = self.screens.enrich_mentions(enrich_payload).to_dict()
            if enriched["state"] != State.AVAILABLE.value or not isinstance(enriched.get("data"), dict):
                return Contract.admit(enriched["state"], enriched.get("data"),
                                      source="persisted-research-strategy",
                                      classification=enriched["provenance"]["classification"],
                                      as_of=enriched["provenance"].get("as_of"),
                                      scope="workspace.research-strategy",
                                      error=enriched.get("error"),
                                      readiness="RESEARCH_ONLY")
            enrich_data = enriched["data"]
        size_payload = {
            "screen_id": screen_id, "now": fields["now"],
            "portfolio_notional": fields["portfolio_notional"],
            "max_name_notional": fields["max_name_notional"],
            "lot_size": fields["lot_size"],
        }
        for key in ("max_age_seconds", "vol_scale", "target_volatility", "max_expected_shortfall",
                    "annualization"):
            if key in fields:
                size_payload[key] = fields[key]
        sized = self.size(size_payload).to_dict()
        if sized["state"] != State.AVAILABLE.value or not isinstance(sized.get("data"), dict):
            return Contract.admit(sized["state"], sized.get("data"),
                                  source="persisted-research-strategy",
                                  classification=sized["provenance"]["classification"],
                                  as_of=sized["provenance"].get("as_of"),
                                  scope="workspace.research-strategy",
                                  error=sized.get("error"),
                                  readiness="RESEARCH_ONLY")
        scan_data = None
        if "options_run_ids" in fields:
            scanned = self.screens.scan_options({
                "screen_id": screen_id, "options_run_ids": fields["options_run_ids"],
            }).to_dict()
            if scanned["state"] != State.AVAILABLE.value or not isinstance(scanned.get("data"), dict):
                return Contract.admit(scanned["state"], scanned.get("data"),
                                      source="persisted-research-strategy",
                                      classification=scanned["provenance"]["classification"],
                                      as_of=scanned["provenance"].get("as_of"),
                                      scope="workspace.research-strategy",
                                      error=scanned.get("error"),
                                      readiness="RESEARCH_ONLY")
            scan_data = scanned["data"]
        body = {
            "schema_version": WATCH,
            "as_of": sized["data"]["as_of"],
            "dataset_id": monitor["dataset_id"],
            "monitor_id": monitor["monitor_id"],
            "screen_id": screen_id,
            "size_id": sized["data"]["size_id"],
            "enrichment_id": enrich_data.get("enrichment_id") if enrich_data is not None else None,
            "scan_id": scan_data["scan_id"] if scan_data is not None else None,
            "source": monitor["source"],
            "input_classification": monitor["input_classification"],
            "timeline": monitor["timeline"],
            "size_metrics": sized["data"]["metrics"],
            "fi_metrics": enrich_data["metrics"] if enrich_data is not None else None,
            "options_metrics": scan_data["metrics"] if scan_data is not None else None,
            "broker_connected": False,
            "trading_approved": False,
            "economically_validated": False,
            "classification": "SIMULATED_PLAN",
            "metrics": {
                "screens": monitor["metrics"]["screens"],
                "timing_entries": monitor["metrics"]["timing_entries"],
                "timing_exits": monitor["metrics"]["timing_exits"],
                "sized_names": sized["data"]["metrics"]["names"],
                "fi_matched": (enrich_data["metrics"]["matched_count"] if enrich_data is not None else 0),
                "options_actionable": (scan_data["metrics"]["actionable"] if scan_data is not None else 0),
                "trading_approved": False,
            },
            "notes": [
                "Watch pass chains monitor → optional FI enrich → size → optional options scan.",
                "It is not a live market subscription or broker order path.",
                "Trading stays unapproved.",
            ],
        }
        identity = _hash(body)
        document = {**body, "watch_id": identity}
        self._path("watches").mkdir(parents=True, exist_ok=True)
        write_immutable(self._path("watches/" + identity + ".json"), dumps(document).encode())
        return self._contract(document, title="Unapproved research opportunity watch pass",
                              model_version=WATCH,
                              evidence="HASH_VERIFIED_RESEARCH_OPPORTUNITY_WATCH",
                              identity_key="watch_id")

    def load_watch(self, watch_id: str) -> dict[str, Any]:
        identity = sha256(watch_id)
        path = self._path("watches/" + identity + ".json")
        if not path.is_file():
            raise FileNotFoundError("research watch unavailable")
        document = loads(path.read_bytes())
        body = {k: v for k, v in document.items() if k != "watch_id"}
        if document.get("watch_id") != identity or _hash(body) != identity or body.get("schema_version") != WATCH:
            raise ValueError("research watch artifact integrity mismatch")
        return document

    def replay_watch(self, payload: dict[str, Any]) -> Contract:
        _fields(payload, {"watch_id"}, {"watch_id"})
        try:
            document = self.load_watch(payload["watch_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        return self._contract(document, title="Unapproved research opportunity watch pass",
                              model_version=WATCH,
                              evidence="HASH_VERIFIED_RESEARCH_OPPORTUNITY_WATCH",
                              identity_key="watch_id")

    def size(self, payload: dict[str, Any]) -> Contract:
        """Convert screen research weights into capped research lot sizes.

        Prices come from admitted bar evidence. ``vol_scale`` may be caller-
        supplied, or derived from ``target_volatility`` against the saved
        dataset return panel. No order is prepared or submitted.
        """
        fields = _fields(payload, {"screen_id", "now", "portfolio_notional", "max_name_notional",
                                   "lot_size", "max_age_seconds", "vol_scale", "target_volatility",
                                   "max_expected_shortfall", "annualization"},
                         {"screen_id", "now", "portfolio_notional", "max_name_notional", "lot_size"})
        if "vol_scale" in fields and "target_volatility" in fields:
            raise ValueError("provide vol_scale or target_volatility, not both")
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
        annualization = int(fields.get("annualization", 252))
        if annualization < 1:
            raise ValueError("annualization must be a positive integer")
        scale = Decimal("1")
        scale_source = "IDENTITY"
        risk_metrics: dict[str, Any] | None = None
        if "vol_scale" in fields:
            scale = _positive(fields["vol_scale"], "vol_scale")
            if scale > Decimal("10"):
                raise ValueError("vol_scale exceeds the research bound")
            scale_source = "CALLER"
        actionable = [row for row in screen["rows"] if float(row["research_weight"]) != 0.0]
        if not actionable:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        symbols = sorted({row["instrument_id"] for row in actionable})
        need_risk = "target_volatility" in fields or "max_expected_shortfall" in fields
        if need_risk:
            panel = pd.DataFrame(dataset["panel_rows"])
            if "return_1" not in panel:
                return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                      classification="UNAVAILABLE", scope="workspace.research-strategy",
                                      error=public_error("DATA_UNAVAILABLE"))
            complete = panel.pivot(index="timestamp", columns="instrument_id", values="return_1").reindex(
                columns=symbols).dropna()
            if len(complete) < 20:
                return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                      classification="UNAVAILABLE", scope="workspace.research-strategy",
                                      error=public_error("DATA_UNAVAILABLE"))
            weights = [float(next(row["research_weight"] for row in actionable if row["instrument_id"] == symbol))
                       for symbol in symbols]
            risk_metrics = portfolio_risk(complete.to_numpy(), weights, annualization=annualization)
        if "target_volatility" in fields:
            target = _positive(fields["target_volatility"], "target_volatility")
            if target > Decimal("5"):
                raise ValueError("target_volatility exceeds the research bound")
            assert risk_metrics is not None
            realized = Decimal(str(risk_metrics["annualized_volatility"]))
            if realized <= 0:
                return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                      classification="UNAVAILABLE", scope="workspace.research-strategy",
                                      error=public_error("DATA_UNAVAILABLE"))
            scale = min(target / realized, Decimal("10"))
            scale_source = "TARGET_VOLATILITY"
        if "max_expected_shortfall" in fields:
            bound = _positive(fields["max_expected_shortfall"], "max_expected_shortfall")
            if bound > Decimal("5"):
                raise ValueError("max_expected_shortfall exceeds the research bound")
            assert risk_metrics is not None
            expected = Decimal(str(risk_metrics["loss_risk"]["expected_shortfall"]))
            if expected > 0 and expected > bound:
                scale = min(scale * (bound / expected), Decimal("10"))
                scale_source = (scale_source + "+MAX_EXPECTED_SHORTFALL"
                                if scale_source != "IDENTITY" else "MAX_EXPECTED_SHORTFALL")
            risk_metrics = {**risk_metrics, "max_expected_shortfall": str(bound),
                            "expected_shortfall_bound_applied": expected > bound}
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
        request = {
            "portfolio_notional": str(portfolio),
            "max_name_notional": str(name_cap),
            "lot_size": lot,
            "max_age_seconds": maximum,
            "vol_scale": str(scale),
            "vol_scale_source": scale_source,
            "annualization": annualization,
        }
        if "target_volatility" in fields:
            request["target_volatility"] = str(_positive(fields["target_volatility"], "target_volatility"))
        if "max_expected_shortfall" in fields:
            request["max_expected_shortfall"] = str(
                _positive(fields["max_expected_shortfall"], "max_expected_shortfall"))
        body = {
            "schema_version": SIZING,
            "as_of": at.isoformat(),
            "dataset_id": screen["dataset_id"],
            "screen_id": screen["screen_id"],
            "signal_id": screen["signal_id"],
            "source": screen["source"],
            "input_classification": screen["input_classification"],
            "request": request,
            "rows": sized,
            "marks": marks,
            "risk": {"state": "AVAILABLE", "data": risk_metrics} if risk_metrics is not None
            else {"state": "NOT_REQUESTED", "data": None},
            "broker_connected": False,
            "trading_approved": False,
            "economically_validated": False,
            "classification": "SIMULATED_PLAN",
            "metrics": {
                "names": len(sized),
                "gross_notional": str(sum((Decimal(row["target_notional"]) for row in sized), Decimal(0))),
                "capped_names": sum(1 for row in sized if row["capped"]),
                "vol_scale": str(scale),
                "vol_scale_source": scale_source,
                "trading_approved": False,
            },
            "notes": [
                "Position sizes are research lot targets from screen weights and admitted marks.",
                "max_name_notional caps each name; target_volatility and max_expected_shortfall scale notionals by panel risk.",
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
