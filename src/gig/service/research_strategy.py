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

import numpy as np
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
EXIT_EVAL = "gig.research-strategy.exit-eval.v1"
FEEDBACK = "gig.research-strategy.feedback.v1"
INVALIDATION = "gig.research-strategy.invalidation.v1"
SCHEDULE = "gig.research-strategy.schedule.v1"
TICK = "gig.research-strategy.schedule-tick.v1"
COST = "gig.research-strategy.estimate-cost.v1"
COST_REBALANCE = "gig.research-strategy.cost-rebalance.v1"
_INVALIDATION_REASONS = frozenset({
    "SIGNAL_EXPIRED", "MANUAL", "REGIME_BREAK", "DATA_STALE", "FEEDBACK_STOP", "ALPHA_DECAY",
})
_SCHEDULE_ACTIONS = frozenset({"cycle", "watch"})


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
        if self._invalidation_hits(screen_id=screen["screen_id"]):
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
        """One research cycle: discover → size → offline simulate.

        Discovery accepts saved model ``run_ids``, a factor ``dataset_id``
        (including ``market_neutral_momentum``), or a ``pair_run_id``. Broker
        connectivity stays disabled. The cycle never grants trading approval.
        """
        allowed = {"run_ids", "dataset_id", "pair_run_id", "mode", "momentum_lookback",
                   "reversion_lookback", "momentum_weight", "reversion_weight", "now", "lookback",
                   "min_ic_observations", "buy_threshold", "sell_threshold", "previous_screen_id",
                   "portfolio_notional", "max_name_notional", "vol_scale", "target_volatility",
                   "max_expected_shortfall", "regime_short_bars", "regime_long_bars",
                   "max_gross_exposure", "max_adv_participation", "adv_lookback_bars",
                   "previous_size_id", "max_size_turnover", "initial_cash", "n_long", "n_short",
                   "gross_leverage", "max_name_weight", "lot_size", "max_age_seconds",
                   "annualization", "metadata", "limits", "commission_bps", "slippage_bps",
                   "price_tick", "max_turnover", "max_transaction_cost", "fill", "document_id",
                   "nlp_profile", "estimate_cost", "spread_bps", "impact_eta", "findings_id",
                   "cost_rebalance", "rebalance_mode",
                   "min_trade_notional", "max_one_way_bps", "max_cost_nav", "alpha_scale"}
        fields = _fields(payload, allowed, {"now", "portfolio_notional", "max_name_notional",
                                            "initial_cash", "n_long", "n_short", "gross_leverage",
                                            "max_name_weight", "lot_size", "max_age_seconds",
                                            "annualization", "metadata", "limits"})
        estimate = fields.get("estimate_cost", False)
        if not isinstance(estimate, bool):
            raise ValueError("estimate_cost must be a boolean")
        sources = [key for key in ("run_ids", "dataset_id", "pair_run_id") if key in fields]
        if len(sources) != 1:
            raise ValueError("cycle requires exactly one of run_ids, dataset_id, or pair_run_id")
        discover_keys = {"run_ids", "dataset_id", "pair_run_id", "mode", "momentum_lookback",
                         "reversion_lookback", "momentum_weight", "reversion_weight", "now",
                         "lookback", "min_ic_observations", "buy_threshold", "sell_threshold",
                         "previous_screen_id"}
        discover_payload = {key: fields[key] for key in fields if key in discover_keys}
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
                    "regime_short_bars", "regime_long_bars", "max_gross_exposure",
                    "max_adv_participation", "previous_size_id", "max_size_turnover",
                    "annualization", "findings_id"):
            if key in fields:
                size_payload[key] = fields[key]
        if "max_adv_participation" in fields and "adv_lookback_bars" in fields:
            size_payload["adv_lookback_bars"] = fields["adv_lookback_bars"]
        sized = self.size(size_payload).to_dict()
        if sized["state"] != State.AVAILABLE.value or not isinstance(sized.get("data"), dict):
            return Contract.admit(sized["state"], sized.get("data"),
                                  source="persisted-research-strategy",
                                  classification=sized["provenance"]["classification"],
                                  as_of=sized["provenance"].get("as_of"),
                                  scope="workspace.research-strategy",
                                  error=sized.get("error"),
                                  readiness="RESEARCH_ONLY")
        cost_data = None
        if estimate:
            cost_payload: dict[str, Any] = {
                "size_id": sized["data"]["size_id"], "now": fields["now"],
            }
            for key in ("previous_size_id", "spread_bps", "commission_bps", "impact_eta",
                        "adv_lookback_bars"):
                if key in fields:
                    cost_payload[key] = fields[key]
            costed = self.estimate_cost(cost_payload).to_dict()
            if costed["state"] != State.AVAILABLE.value or not isinstance(costed.get("data"), dict):
                return Contract.admit(costed["state"], costed.get("data"),
                                      source="persisted-research-strategy",
                                      classification=costed["provenance"]["classification"],
                                      as_of=costed["provenance"].get("as_of"),
                                      scope="workspace.research-strategy",
                                      error=costed.get("error"),
                                      readiness="RESEARCH_ONLY")
            cost_data = costed["data"]
        rebalance = fields.get("cost_rebalance", False)
        if not isinstance(rebalance, bool):
            raise ValueError("cost_rebalance must be a boolean")
        rebalance_data = None
        if rebalance:
            if cost_data is None:
                # ensure cost exists when rebalance requested without estimate_cost
                cost_payload = {
                    "size_id": sized["data"]["size_id"], "now": fields["now"],
                }
                for key in ("previous_size_id", "spread_bps", "commission_bps", "impact_eta",
                            "adv_lookback_bars"):
                    if key in fields:
                        cost_payload[key] = fields[key]
                costed = self.estimate_cost(cost_payload).to_dict()
                if costed["state"] != State.AVAILABLE.value or not isinstance(costed.get("data"), dict):
                    return Contract.admit(costed["state"], costed.get("data"),
                                          source="persisted-research-strategy",
                                          classification=costed["provenance"]["classification"],
                                          as_of=costed["provenance"].get("as_of"),
                                          scope="workspace.research-strategy",
                                          error=costed.get("error"),
                                          readiness="RESEARCH_ONLY")
                cost_data = costed["data"]
            reb_payload: dict[str, Any] = {
                "size_id": sized["data"]["size_id"],
                "cost_id": cost_data["cost_id"],
                "now": fields["now"],
                "mode": fields.get("rebalance_mode", "SKIP_BELOW_COST"),
            }
            for key in ("min_trade_notional", "max_one_way_bps", "max_cost_nav", "alpha_scale"):
                if key in fields:
                    reb_payload[key] = fields[key]
            rebalanced = self.cost_rebalance(reb_payload).to_dict()
            if rebalanced["state"] != State.AVAILABLE.value or not isinstance(rebalanced.get("data"), dict):
                return Contract.admit(rebalanced["state"], rebalanced.get("data"),
                                      source="persisted-research-strategy",
                                      classification=rebalanced["provenance"]["classification"],
                                      as_of=rebalanced["provenance"].get("as_of"),
                                      scope="workspace.research-strategy",
                                      error=rebalanced.get("error"),
                                      readiness="RESEARCH_ONLY")
            rebalance_data = rebalanced["data"]
        size_only = {"run_ids", "dataset_id", "pair_run_id", "mode", "momentum_lookback",
                     "reversion_lookback", "momentum_weight", "reversion_weight", "lookback",
                     "min_ic_observations", "buy_threshold", "sell_threshold", "previous_screen_id",
                     "portfolio_notional", "max_name_notional", "vol_scale", "target_volatility",
                     "max_expected_shortfall", "regime_short_bars", "regime_long_bars",
                     "max_gross_exposure", "max_adv_participation", "adv_lookback_bars",
                     "previous_size_id", "max_size_turnover", "document_id", "nlp_profile",
                     "estimate_cost", "spread_bps", "impact_eta", "findings_id",
                     "cost_rebalance", "rebalance_mode",
                     "min_trade_notional", "max_one_way_bps", "max_cost_nav", "alpha_scale"}
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
            "cost_id": cost_data["cost_id"] if cost_data is not None else None,
            "rebalance_id": rebalance_data["rebalance_id"] if rebalance_data is not None else None,
            "enrichment_id": enrich_data.get("enrichment_id") if enrich_data is not None else None,
            "strategy_id": strategy["strategy_id"],
            "allocation_id": strategy["allocation_id"],
            "source": screen["source"],
            "input_classification": screen["input_classification"],
            "discovery": screen.get("discovery"),
            "size_metrics": sized["data"]["metrics"],
            "cost_metrics": cost_data["metrics"] if cost_data is not None else None,
            "rebalance_metrics": rebalance_data["metrics"] if rebalance_data is not None else None,
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
                "estimated_cost_nav": (
                    None if cost_data is None else cost_data["metrics"]["cost_nav"]),
                "rebalance_trades_skipped": (
                    None if rebalance_data is None else rebalance_data["metrics"]["trades_skipped"]),
                "prepared_orders": strategy["metrics"]["prepared_orders"],
                "filled_orders": strategy["metrics"]["filled_orders"],
                "trading_approved": False,
            },
            "notes": [
                "Research cycle chains discovery, optional FI enrich, position sizing, optional pre-trade TCA, optional cost-aware rebalance, and offline simulation.",
                "Discovery may use model ensembles or cross-sectional factor ranks including market-neutral momentum.",
                "Sizing is advisory relative to the separate allocation/OMS simulation path.",
                "Pre-trade costs stay ESTIMATED and are never mixed with simulated fills.",
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
                   "max_expected_shortfall", "regime_short_bars", "regime_long_bars",
                   "max_gross_exposure", "max_adv_participation", "adv_lookback_bars",
                   "previous_size_id", "max_size_turnover", "annualization", "options_run_ids",
                   "document_id", "nlp_profile"}
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
                    "regime_short_bars", "regime_long_bars", "max_gross_exposure",
                    "max_adv_participation", "adv_lookback_bars", "previous_size_id",
                    "max_size_turnover", "annualization"):
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

    def exit_eval(self, payload: dict[str, Any]) -> Contract:
        """Historical stop-loss / take-profit path evaluation on saved research sizes.

        Uses the saved dataset return panel after the size decision clock.
        Results are research-only path diagnostics, not broker exits.
        """
        fields = _fields(payload, {"size_id", "stop_loss_pct", "take_profit_pct", "horizon_bars"},
                         {"size_id", "stop_loss_pct", "take_profit_pct", "horizon_bars"})
        stop = _positive(fields["stop_loss_pct"], "stop_loss_pct")
        take = _positive(fields["take_profit_pct"], "take_profit_pct")
        if stop > Decimal("1") or take > Decimal("5"):
            raise ValueError("stop_loss_pct or take_profit_pct exceeds the research bound")
        horizon = int(fields["horizon_bars"])
        if not 1 <= horizon <= 252:
            raise ValueError("horizon_bars must lie between 1 and 252")
        try:
            sized = self.load_size(fields["size_id"])
            dataset = PersistedDatasetWorkspace(self.base_dir).load(sized["dataset_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        if self._invalidation_hits(screen_id=sized["screen_id"], size_id=sized["size_id"]):
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        panel = pd.DataFrame(dataset["panel_rows"])
        if "return_1" not in panel or "timestamp" not in panel:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        decision = aware_timestamp(sized["as_of"])
        rows_out: list[dict[str, Any]] = []
        for row in sized["rows"]:
            symbol = row["instrument_id"]
            series = panel.loc[panel["instrument_id"] == symbol, ["timestamp", "return_1"]].copy()
            series["timestamp"] = pd.to_datetime(series["timestamp"], utc=True)
            series = series.sort_values("timestamp")
            forward = series.loc[series["timestamp"] > decision].head(horizon)
            if forward.empty:
                rows_out.append({
                    "instrument_id": symbol, "side": row["side"], "outcome": "NO_FORWARD_PATH",
                    "bars_observed": 0, "path_return": None, "hit_bar": None,
                    "stop_loss_pct": str(stop), "take_profit_pct": str(take),
                    "trading_approved": False,
                })
                continue
            sign = 1.0 if row["side"] == "buy" else -1.0
            wealth = 1.0
            outcome = "HORIZON"
            hit_bar = None
            path_return = None
            for index, (_, point) in enumerate(forward.iterrows(), start=1):
                wealth *= 1.0 + sign * float(point["return_1"])
                path_return = wealth - 1.0
                if path_return <= -float(stop):
                    outcome = "STOP_LOSS"
                    hit_bar = index
                    break
                if path_return >= float(take):
                    outcome = "TAKE_PROFIT"
                    hit_bar = index
                    break
            rows_out.append({
                "instrument_id": symbol, "side": row["side"], "outcome": outcome,
                "bars_observed": int(len(forward) if hit_bar is None else hit_bar),
                "path_return": None if path_return is None else float(path_return),
                "hit_bar": hit_bar, "stop_loss_pct": str(stop), "take_profit_pct": str(take),
                "trading_approved": False,
            })
        counts = {key: sum(1 for row in rows_out if row["outcome"] == key)
                  for key in ("STOP_LOSS", "TAKE_PROFIT", "HORIZON", "NO_FORWARD_PATH")}
        body = {
            "schema_version": EXIT_EVAL,
            "as_of": sized["as_of"],
            "dataset_id": sized["dataset_id"],
            "screen_id": sized["screen_id"],
            "size_id": sized["size_id"],
            "source": sized["source"],
            "input_classification": sized["input_classification"],
            "request": {
                "stop_loss_pct": str(stop), "take_profit_pct": str(take),
                "horizon_bars": horizon,
            },
            "rows": rows_out,
            "broker_connected": False,
            "trading_approved": False,
            "economically_validated": False,
            "classification": "MODEL_ESTIMATE",
            "metrics": {**counts, "names": len(rows_out), "trading_approved": False},
            "notes": [
                "Exit evaluation walks the saved return panel after the size decision clock.",
                "Stops and targets are research barriers on path returns; they are not broker orders.",
                "Missing forward paths remain NO_FORWARD_PATH and are never filled with zeros.",
            ],
        }
        identity = _hash(body)
        document = {**body, "exit_eval_id": identity}
        self._path("exit-evals").mkdir(parents=True, exist_ok=True)
        write_immutable(self._path("exit-evals/" + identity + ".json"), dumps(document).encode())
        return self._contract(document, title="Unapproved research stop/take-profit evaluation",
                              model_version=EXIT_EVAL,
                              evidence="HASH_VERIFIED_RESEARCH_EXIT_EVALUATION",
                              identity_key="exit_eval_id")

    def load_exit_eval(self, exit_eval_id: str) -> dict[str, Any]:
        identity = sha256(exit_eval_id)
        path = self._path("exit-evals/" + identity + ".json")
        if not path.is_file():
            raise FileNotFoundError("research exit evaluation unavailable")
        document = loads(path.read_bytes())
        body = {k: v for k, v in document.items() if k != "exit_eval_id"}
        if (document.get("exit_eval_id") != identity or _hash(body) != identity
                or body.get("schema_version") != EXIT_EVAL):
            raise ValueError("research exit evaluation integrity mismatch")
        return document

    def replay_exit_eval(self, payload: dict[str, Any]) -> Contract:
        _fields(payload, {"exit_eval_id"}, {"exit_eval_id"})
        try:
            document = self.load_exit_eval(payload["exit_eval_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        return self._contract(document, title="Unapproved research stop/take-profit evaluation",
                              model_version=EXIT_EVAL,
                              evidence="HASH_VERIFIED_RESEARCH_EXIT_EVALUATION",
                              identity_key="exit_eval_id")

    def feedback(self, payload: dict[str, Any]) -> Contract:
        """Aggregate historical exit-eval outcomes into research performance feedback."""
        fields = _fields(payload, {"exit_eval_ids"}, {"exit_eval_ids"})
        ids = fields["exit_eval_ids"]
        if not isinstance(ids, list) or not 1 <= len(ids) <= 32 or len(set(ids)) != len(ids):
            raise ValueError("feedback requires one to thirty-two distinct exit evaluation identities")
        evaluations: list[dict[str, Any]] = []
        try:
            for identity in ids:
                evaluations.append(self.load_exit_eval(identity))
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        dataset_ids = {item["dataset_id"] for item in evaluations}
        if len(dataset_ids) != 1:
            raise ValueError("exit evaluations must share one dataset identity")
        totals = {"STOP_LOSS": 0, "TAKE_PROFIT": 0, "HORIZON": 0, "NO_FORWARD_PATH": 0, "names": 0}
        path_returns: list[float] = []
        for item in evaluations:
            for key in ("STOP_LOSS", "TAKE_PROFIT", "HORIZON", "NO_FORWARD_PATH"):
                totals[key] += int(item["metrics"].get(key, 0))
            totals["names"] += int(item["metrics"].get("names", 0))
            for row in item["rows"]:
                if row.get("path_return") is not None:
                    path_returns.append(float(row["path_return"]))
        observed = totals["STOP_LOSS"] + totals["TAKE_PROFIT"] + totals["HORIZON"]
        mean_path = (sum(path_returns) / len(path_returns)) if path_returns else None
        body = {
            "schema_version": FEEDBACK,
            "as_of": max(item["as_of"] for item in evaluations),
            "dataset_id": next(iter(dataset_ids)),
            "exit_eval_ids": [sha256(i) for i in ids],
            "source": evaluations[-1]["source"],
            "input_classification": evaluations[-1]["input_classification"],
            "broker_connected": False,
            "trading_approved": False,
            "economically_validated": False,
            "classification": "MODEL_ESTIMATE",
            "metrics": {
                **totals,
                "evaluations": len(evaluations),
                "observed_paths": observed,
                "take_profit_rate": (totals["TAKE_PROFIT"] / observed) if observed else None,
                "stop_loss_rate": (totals["STOP_LOSS"] / observed) if observed else None,
                "mean_path_return": mean_path,
                "trading_approved": False,
            },
            "notes": [
                "Feedback aggregates saved exit-eval path outcomes only.",
                "Rates are descriptive research diagnostics, not calibrated expectancy.",
                "No broker order or trading approval is established.",
            ],
        }
        identity = _hash(body)
        document = {**body, "feedback_id": identity}
        self._path("feedback").mkdir(parents=True, exist_ok=True)
        write_immutable(self._path("feedback/" + identity + ".json"), dumps(document).encode())
        return self._contract(document, title="Unapproved research strategy performance feedback",
                              model_version=FEEDBACK,
                              evidence="HASH_VERIFIED_RESEARCH_STRATEGY_FEEDBACK",
                              identity_key="feedback_id")

    def load_feedback(self, feedback_id: str) -> dict[str, Any]:
        identity = sha256(feedback_id)
        path = self._path("feedback/" + identity + ".json")
        if not path.is_file():
            raise FileNotFoundError("research strategy feedback unavailable")
        document = loads(path.read_bytes())
        body = {k: v for k, v in document.items() if k != "feedback_id"}
        if (document.get("feedback_id") != identity or _hash(body) != identity
                or body.get("schema_version") != FEEDBACK):
            raise ValueError("research strategy feedback integrity mismatch")
        return document

    def replay_feedback(self, payload: dict[str, Any]) -> Contract:
        _fields(payload, {"feedback_id"}, {"feedback_id"})
        try:
            document = self.load_feedback(payload["feedback_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        return self._contract(document, title="Unapproved research strategy performance feedback",
                              model_version=FEEDBACK,
                              evidence="HASH_VERIFIED_RESEARCH_STRATEGY_FEEDBACK",
                              identity_key="feedback_id")

    def _invalidation_hits(self, *, screen_id: str | None = None,
                           size_id: str | None = None) -> list[dict[str, Any]]:
        directory = self._path("invalidations")
        if not directory.is_dir():
            return []
        hits: list[dict[str, Any]] = []
        for path in sorted(directory.glob("*.json")):
            document = loads(path.read_bytes())
            if ((screen_id is not None and document.get("screen_id") == screen_id)
                    or (size_id is not None and document.get("size_id") == size_id)):
                hits.append(document)
        return hits

    def invalidate(self, payload: dict[str, Any]) -> Contract:
        """Record a research strategy invalidation that blocks later size/simulate use."""
        fields = _fields(payload, {"screen_id", "size_id", "reason", "now", "notes"},
                         {"reason", "now"})
        reason = fields["reason"]
        if reason not in _INVALIDATION_REASONS:
            raise ValueError("reason must be a supported research invalidation reason")
        if ("screen_id" in fields) == ("size_id" in fields):
            raise ValueError("invalidate requires exactly one of screen_id or size_id")
        screen_id = None
        size_id = None
        dataset_id = None
        source = None
        input_classification = None
        try:
            if "size_id" in fields:
                sized = self.load_size(fields["size_id"])
                size_id = sized["size_id"]
                screen_id = sized["screen_id"]
                dataset_id = sized["dataset_id"]
                source = sized["source"]
                input_classification = sized["input_classification"]
            else:
                screen = self.screens.load(fields["screen_id"])
                screen_id = screen["screen_id"]
                dataset_id = screen["dataset_id"]
                source = screen["source"]
                input_classification = screen["input_classification"]
                if reason == "SIGNAL_EXPIRED":
                    expired = bool(screen.get("signal_expired")) or any(
                        row.get("expiration_state") == "EXPIRED" for row in screen.get("rows", []))
                    if not expired:
                        raise ValueError("SIGNAL_EXPIRED requires an expired research screen")
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        if self._invalidation_hits(screen_id=screen_id, size_id=size_id):
            raise ValueError("research strategy lineage is already invalidated")
        notes = fields.get("notes")
        if notes is not None and (not isinstance(notes, str) or not 1 <= len(notes) <= 240):
            raise ValueError("notes must be a short string when provided")
        body = {
            "schema_version": INVALIDATION,
            "as_of": aware_timestamp(fields["now"]).isoformat(),
            "dataset_id": dataset_id,
            "screen_id": screen_id,
            "size_id": size_id,
            "reason": reason,
            "source": source,
            "input_classification": input_classification,
            "broker_connected": False,
            "trading_approved": False,
            "economically_validated": False,
            "classification": "MODEL_ESTIMATE",
            "usable_for_simulation": False,
            "metrics": {"trading_approved": False},
            "notes": [
                "Invalidation permanently blocks later size and simulate ops on this lineage.",
                "It does not cancel offline OMS intents or approve trading.",
                *([notes] if notes else []),
            ],
        }
        identity = _hash(body)
        document = {**body, "invalidation_id": identity}
        self._path("invalidations").mkdir(parents=True, exist_ok=True)
        write_immutable(self._path("invalidations/" + identity + ".json"), dumps(document).encode())
        return self._contract(document, title="Unapproved research strategy invalidation",
                              model_version=INVALIDATION,
                              evidence="HASH_VERIFIED_RESEARCH_STRATEGY_INVALIDATION",
                              identity_key="invalidation_id")

    def load_invalidation(self, invalidation_id: str) -> dict[str, Any]:
        identity = sha256(invalidation_id)
        path = self._path("invalidations/" + identity + ".json")
        if not path.is_file():
            raise FileNotFoundError("research strategy invalidation unavailable")
        document = loads(path.read_bytes())
        body = {k: v for k, v in document.items() if k != "invalidation_id"}
        if (document.get("invalidation_id") != identity or _hash(body) != identity
                or body.get("schema_version") != INVALIDATION):
            raise ValueError("research strategy invalidation integrity mismatch")
        return document

    def replay_invalidation(self, payload: dict[str, Any]) -> Contract:
        _fields(payload, {"invalidation_id"}, {"invalidation_id"})
        try:
            document = self.load_invalidation(payload["invalidation_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        return self._contract(document, title="Unapproved research strategy invalidation",
                              model_version=INVALIDATION,
                              evidence="HASH_VERIFIED_RESEARCH_STRATEGY_INVALIDATION",
                              identity_key="invalidation_id")

    def schedule(self, payload: dict[str, Any]) -> Contract:
        """Persist a bounded research scan schedule for explicit later ticks.

        Cadence is advisory. Nothing runs in the background; each tick must be
        requested. Broker connectivity stays disabled.
        """
        fields = _fields(payload, {"action", "cadence_seconds", "now", "request", "max_ticks",
                                   "exit_eval", "stop_loss_pct", "take_profit_pct", "horizon_bars",
                                   "feedback_every_n_ticks",
                                   "min_ic_ir", "min_ic_half_life_steps", "alpha_decay_action"},
                         {"action", "cadence_seconds", "now", "request"})
        action = fields["action"]
        if action not in _SCHEDULE_ACTIONS:
            raise ValueError("schedule action must be cycle or watch")
        cadence = int(fields["cadence_seconds"])
        if not 60 <= cadence <= 86_400:
            raise ValueError("cadence_seconds must lie between 60 and 86400")
        maximum = int(fields.get("max_ticks", 32))
        if not 1 <= maximum <= 128:
            raise ValueError("max_ticks must lie between 1 and 128")
        request = fields["request"]
        if not isinstance(request, dict) or "now" in request:
            raise ValueError("schedule request must be an object without a now clock")
        do_exit = fields.get("exit_eval", False)
        if not isinstance(do_exit, bool):
            raise ValueError("exit_eval must be a boolean")
        feedback_every = int(fields.get("feedback_every_n_ticks", 0))
        if not 0 <= feedback_every <= 32:
            raise ValueError("feedback_every_n_ticks must lie between 0 and 32")
        if feedback_every and not do_exit:
            raise ValueError("feedback_every_n_ticks requires exit_eval")
        exit_request = None
        if do_exit:
            for key in ("stop_loss_pct", "take_profit_pct", "horizon_bars"):
                if key not in fields:
                    raise ValueError(f"exit_eval requires {key}")
            stop = _positive(fields["stop_loss_pct"], "stop_loss_pct")
            take = _positive(fields["take_profit_pct"], "take_profit_pct")
            if stop > Decimal("1") or take > Decimal("5"):
                raise ValueError("stop_loss_pct or take_profit_pct exceeds the research bound")
            horizon = int(fields["horizon_bars"])
            if not 1 <= horizon <= 252:
                raise ValueError("horizon_bars must lie between 1 and 252")
            exit_request = {
                "stop_loss_pct": str(stop),
                "take_profit_pct": str(take),
                "horizon_bars": horizon,
            }
        decay_action = fields.get("alpha_decay_action", "soft_skip")
        if decay_action not in {"soft_skip", "invalidate"}:
            raise ValueError("alpha_decay_action must be soft_skip or invalidate")
        min_ir = fields.get("min_ic_ir")
        if min_ir is not None:
            min_ir = float(min_ir)
            if min_ir != min_ir:  # NaN
                raise ValueError("min_ic_ir must be a finite decimal")
        min_hl = fields.get("min_ic_half_life_steps")
        if min_hl is not None:
            min_hl = float(min_hl)
            if not (min_hl > 0):
                raise ValueError("min_ic_half_life_steps must be positive")
        origin = aware_timestamp(fields["now"])
        body = {
            "schema_version": SCHEDULE,
            "as_of": origin.isoformat(),
            "dataset_id": request.get("dataset_id") or request.get("run_ids") or "schedule",
            "action": action,
            "cadence_seconds": cadence,
            "max_ticks": maximum,
            "ticks_completed": 0,
            "next_due": origin.isoformat(),
            "request": request,
            "exit_eval": do_exit,
            "exit_eval_request": exit_request,
            "feedback_every_n_ticks": feedback_every,
            "min_ic_ir": min_ir,
            "min_ic_half_life_steps": min_hl,
            "alpha_decay_action": decay_action,
            "source": "persisted-research-strategy-schedule",
            "input_classification": "MODEL_ESTIMATE",
            "broker_connected": False,
            "trading_approved": False,
            "economically_validated": False,
            "classification": "MODEL_ESTIMATE",
            "metrics": {"ticks_completed": 0, "exit_eval": do_exit,
                        "feedback_every_n_ticks": feedback_every,
                        "alpha_decay_gated": min_ir is not None or min_hl is not None,
                        "trading_approved": False},
            "notes": [
                "Schedules are durable research plans; ticks are explicit and CPU-local.",
                "Optional exit_eval runs stop/take path diagnostics on each tick size_id.",
                "Optional feedback_every_n_ticks aggregates successful exit-evals into research feedback.",
                "Optional min_ic_ir / min_ic_half_life_steps soft-stop or invalidate when signal diagnostics decay.",
                "No background worker, broker path, or trading approval is established.",
            ],
        }
        # Pin a stable dataset identity when the template carries one.
        if isinstance(request.get("dataset_id"), str):
            body["dataset_id"] = sha256(request["dataset_id"])
        elif isinstance(request.get("run_ids"), list) and request["run_ids"]:
            body["dataset_id"] = sha256("|".join(sorted(sha256(i) for i in request["run_ids"])))
        elif isinstance(request.get("signal_ids"), list) and request["signal_ids"]:
            body["dataset_id"] = _hash(sorted(sha256(i) for i in request["signal_ids"]))
        else:
            body["dataset_id"] = _hash({"action": action, "request": request})
        identity = _hash(body)
        document = {**body, "schedule_id": identity}
        self._path("schedules").mkdir(parents=True, exist_ok=True)
        write_immutable(self._path("schedules/" + identity + ".json"), dumps(document).encode())
        return self._contract(document, title="Unapproved research strategy scan schedule",
                              model_version=SCHEDULE,
                              evidence="HASH_VERIFIED_RESEARCH_STRATEGY_SCHEDULE",
                              identity_key="schedule_id")

    def load_schedule(self, schedule_id: str) -> dict[str, Any]:
        identity = sha256(schedule_id)
        path = self._path("schedules/" + identity + ".json")
        if not path.is_file():
            raise FileNotFoundError("research strategy schedule unavailable")
        document = loads(path.read_bytes())
        body = {k: v for k, v in document.items() if k != "schedule_id"}
        if (document.get("schedule_id") != identity or _hash(body) != identity
                or body.get("schema_version") != SCHEDULE):
            raise ValueError("research strategy schedule integrity mismatch")
        return document


    def _alpha_decay_tripped(self, plan: dict[str, Any], screen_id: str | None) -> dict[str, Any] | None:
        """Evaluate optional schedule alpha-decay thresholds against signal diagnostics."""
        min_ir = plan.get("min_ic_ir")
        min_hl = plan.get("min_ic_half_life_steps")
        if min_ir is None and min_hl is None:
            return None
        if not screen_id:
            return None
        action = plan.get("alpha_decay_action", "soft_skip")
        if action not in {"soft_skip", "invalidate"}:
            raise ValueError("alpha_decay_action must be soft_skip or invalidate")
        try:
            screen = self.screens.load(screen_id)
            from gig.service.signal_workspace import SignalWorkspace
            signal = SignalWorkspace(self.base_dir).load(screen["signal_id"])
        except FileNotFoundError:
            return None
        diagnostics = signal.get("diagnostics") or []
        ir_values = [float(d["ic_ir"]) for d in diagnostics
                     if d.get("ic_ir_state") == "AVAILABLE" and d.get("ic_ir") is not None]
        hl_values = [float(d["ic_half_life"]["half_life_steps"]) for d in diagnostics
                     if isinstance(d.get("ic_half_life"), dict)
                     and d["ic_half_life"].get("state") == "AVAILABLE"
                     and d["ic_half_life"].get("half_life_steps") is not None]
        tripped = False
        reasons: list[str] = []
        if min_ir is not None and ir_values and all(value < float(min_ir) for value in ir_values):
            tripped = True
            reasons.append("MIN_IC_IR")
        if min_hl is not None and hl_values and all(value < float(min_hl) for value in hl_values):
            tripped = True
            reasons.append("MIN_IC_HALF_LIFE")
        if not tripped:
            return None
        return {
            "alpha_decayed": True,
            "alpha_decay_action": action,
            "alpha_decay_reasons": reasons,
            "min_ic_ir": None if min_ir is None else float(min_ir),
            "min_ic_half_life_steps": None if min_hl is None else float(min_hl),
            "observed_ic_ir": ir_values,
            "observed_half_life_steps": hl_values,
            "signal_id": signal["signal_id"],
            "screen_id": screen["screen_id"],
        }

    def tick_schedule(self, payload: dict[str, Any]) -> Contract:
        """Run one due schedule action and persist a tick receipt.

        If ``now`` is before ``next_due``, returns STALE without executing.
        After ``max_ticks``, further ticks are UNAVAILABLE.
        """
        from datetime import timedelta

        fields = _fields(payload, {"schedule_id", "now"}, {"schedule_id", "now"})
        try:
            plan = self.load_schedule_state(fields["schedule_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        clock = aware_timestamp(fields["now"])
        due = aware_timestamp(plan["next_due"])
        if int(plan["ticks_completed"]) >= int(plan["max_ticks"]):
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        if plan.get("alpha_decayed"):
            return Contract.admit(
                State.STALE,
                {
                    **plan,
                    "title": "Research schedule soft-stopped after alpha decay",
                    "tick_state": "ALPHA_DECAYED",
                    "observed_at": clock.isoformat(),
                    "trading_approved": False,
                },
                source=plan["source"],
                classification="MODEL_ESTIMATE",
                as_of=clock.isoformat(),
                scope="workspace.research-strategy",
                source_identity=str(plan["dataset_id"]),
                model_identity=plan["schedule_id"],
                model_version=TICK,
                evidence_state="HASH_VERIFIED_RESEARCH_STRATEGY_SCHEDULE_TICK",
                readiness="RESEARCH_ONLY",
            )
        if clock < due:
            return Contract.admit(
                State.STALE,
                {
                    **plan,
                    "title": "Research schedule tick not yet due",
                    "tick_state": "NOT_DUE",
                    "observed_at": clock.isoformat(),
                    "trading_approved": False,
                },
                source=plan["source"],
                classification="MODEL_ESTIMATE",
                as_of=clock.isoformat(),
                scope="workspace.research-strategy",
                source_identity=str(plan["dataset_id"]),
                model_identity=plan["schedule_id"],
                model_version=TICK,
                evidence_state="HASH_VERIFIED_RESEARCH_STRATEGY_SCHEDULE_TICK",
                readiness="RESEARCH_ONLY",
            )
        request = {**plan["request"], "now": clock.isoformat()}
        # Chain the prior tick size so later cycles/watches measure turnover and delta TCA.
        if "previous_size_id" not in request and plan.get("last_size_id"):
            request["previous_size_id"] = plan["last_size_id"]
        if plan["action"] == "cycle":
            executed = self.cycle(request).to_dict()
        else:
            executed = self.watch(request).to_dict()
        if executed["state"] != State.AVAILABLE.value or not isinstance(executed.get("data"), dict):
            return Contract.admit(executed["state"], executed.get("data"),
                                  source="persisted-research-strategy",
                                  classification=executed["provenance"]["classification"],
                                  as_of=executed["provenance"].get("as_of"),
                                  scope="workspace.research-strategy",
                                  error=executed.get("error"),
                                  readiness="RESEARCH_ONLY")
        result = executed["data"]
        completed = int(plan["ticks_completed"]) + 1
        next_due = (clock + timedelta(seconds=int(plan["cadence_seconds"]))).isoformat()
        exit_eval_id = None
        feedback_id = None
        exit_eval_ids = list(plan.get("exit_eval_ids") or [])
        if plan.get("exit_eval") and result.get("size_id") and plan.get("exit_eval_request"):
            eval_payload = {
                "size_id": result["size_id"],
                **plan["exit_eval_request"],
            }
            evaluated = self.exit_eval(eval_payload).to_dict()
            if evaluated["state"] == State.AVAILABLE.value and isinstance(evaluated.get("data"), dict):
                exit_eval_id = evaluated["data"]["exit_eval_id"]
                if exit_eval_id not in exit_eval_ids:
                    exit_eval_ids.append(exit_eval_id)
                exit_eval_ids = exit_eval_ids[-32:]
                every = int(plan.get("feedback_every_n_ticks") or 0)
                if every > 0 and len(exit_eval_ids) >= every and len(exit_eval_ids) % every == 0:
                    window = exit_eval_ids[-every:]
                    fed = self.feedback({"exit_eval_ids": window}).to_dict()
                    if fed["state"] == State.AVAILABLE.value and isinstance(fed.get("data"), dict):
                        feedback_id = fed["data"]["feedback_id"]
        decay = self._alpha_decay_tripped(plan, result.get("screen_id"))
        invalidation_id = None
        if decay is not None and decay["alpha_decay_action"] == "invalidate" and result.get("screen_id"):
            invalidated = self.invalidate({
                "screen_id": result["screen_id"],
                "reason": "ALPHA_DECAY",
                "now": clock.isoformat(),
                "notes": "Schedule alpha-decay gate tripped on signal diagnostics",
            }).to_dict()
            if invalidated["state"] == State.AVAILABLE.value and isinstance(invalidated.get("data"), dict):
                invalidation_id = invalidated["data"]["invalidation_id"]
        body = {
            "schema_version": TICK,
            "as_of": clock.isoformat(),
            "dataset_id": result.get("dataset_id", plan["dataset_id"]),
            "schedule_id": plan["schedule_id"],
            "action": plan["action"],
            "tick_index": completed,
            "max_ticks": plan["max_ticks"],
            "next_due": next_due,
            "cycle_id": result.get("cycle_id"),
            "watch_id": result.get("watch_id"),
            "screen_id": result.get("screen_id"),
            "size_id": result.get("size_id"),
            "previous_size_id": request.get("previous_size_id"),
            "cost_id": result.get("cost_id"),
            "exit_eval_id": exit_eval_id,
            "feedback_id": feedback_id,
            "strategy_id": result.get("strategy_id"),
            "invalidation_id": invalidation_id,
            "alpha_decay": decay,
            "source": result.get("source", plan["source"]),
            "input_classification": result.get("input_classification", plan["input_classification"]),
            "broker_connected": False,
            "trading_approved": False,
            "economically_validated": False,
            "classification": "SIMULATED_PLAN",
            "metrics": {
                "tick_index": completed,
                "prepared_orders": (result.get("metrics") or {}).get("prepared_orders"),
                "estimated_cost_nav": (result.get("metrics") or {}).get("estimated_cost_nav"),
                "size_chained": bool(request.get("previous_size_id")),
                "exit_eval_attached": exit_eval_id is not None,
                "feedback_attached": feedback_id is not None,
                "exit_eval_count": len(exit_eval_ids),
                "alpha_decayed": decay is not None,
                "alpha_decay_action": None if decay is None else decay["alpha_decay_action"],
                "trading_approved": False,
            },
            "notes": [
                "Schedule tick executed one research cycle or watch pass under an explicit clock.",
                "next_due advances by cadence_seconds from the tick clock.",
                "Later ticks inject previous_size_id from the prior tick when the schedule request omits it.",
                "Optional cycle estimate_cost attaches ESTIMATED pre-trade TCA on the sized book.",
                "Optional exit_eval and feedback hooks are soft-fail; UNAVAILABLE post-hooks leave null ids.",
                "Optional min_ic_ir / min_ic_half_life_steps soft-stop or invalidate when signal diagnostics decay.",
                "No background scheduler or trading approval is established.",
            ],
        }
        identity = _hash(body)
        document = {**body, "tick_id": identity}
        self._path("schedule-ticks").mkdir(parents=True, exist_ok=True)
        write_immutable(self._path("schedule-ticks/" + identity + ".json"), dumps(document).encode())
        # Advance mutable schedule bookkeeping in a companion pointer file.
        pointer = {
            "schedule_id": plan["schedule_id"],
            "ticks_completed": completed,
            "next_due": next_due,
            "last_tick_id": identity,
            "last_size_id": result.get("size_id"),
            "last_cost_id": result.get("cost_id"),
            "last_exit_eval_id": exit_eval_id if exit_eval_id is not None else plan.get("last_exit_eval_id"),
            "exit_eval_ids": exit_eval_ids,
            "last_feedback_id": feedback_id if feedback_id is not None else plan.get("last_feedback_id"),
            "alpha_decayed": bool(decay is not None or plan.get("alpha_decayed")),
            "alpha_decay_tick_id": identity if decay is not None else plan.get("alpha_decay_tick_id"),
            "invalidation_id": invalidation_id if invalidation_id is not None else plan.get("invalidation_id"),
        }
        pointer_path = self._path("schedules/" + plan["schedule_id"] + ".pointer.json")
        pointer_path.write_bytes(dumps(pointer).encode())
        return self._contract(document, title="Unapproved research strategy schedule tick",
                              model_version=TICK,
                              evidence="HASH_VERIFIED_RESEARCH_STRATEGY_SCHEDULE_TICK",
                              identity_key="tick_id")

    def load_schedule_state(self, schedule_id: str) -> dict[str, Any]:
        plan = self.load_schedule(schedule_id)
        pointer_path = self._path("schedules/" + plan["schedule_id"] + ".pointer.json")
        if pointer_path.is_file():
            pointer = loads(pointer_path.read_bytes())
            if pointer.get("schedule_id") != plan["schedule_id"]:
                raise ValueError("schedule pointer integrity mismatch")
            return {**plan, "ticks_completed": pointer["ticks_completed"],
                    "next_due": pointer["next_due"], "last_tick_id": pointer.get("last_tick_id"),
                    "last_size_id": pointer.get("last_size_id"),
                    "last_cost_id": pointer.get("last_cost_id"),
                    "last_exit_eval_id": pointer.get("last_exit_eval_id"),
                    "exit_eval_ids": pointer.get("exit_eval_ids") or [],
                    "last_feedback_id": pointer.get("last_feedback_id"),
                    "alpha_decayed": pointer.get("alpha_decayed", False),
                    "alpha_decay_tick_id": pointer.get("alpha_decay_tick_id"),
                    "invalidation_id": pointer.get("invalidation_id")}
        return plan

    def due_schedules(self, payload: dict[str, Any]) -> Contract:
        """List hash-verified schedules that are due at ``now`` for agent ticks.

        Advisory inventory only. Does not execute ticks, start workers, or
        approve trading. Agents must call ``schedule-tick`` explicitly per row
        (or ``tick-due`` for a bounded batch).
        """
        fields = _fields(payload, {"now", "limit"}, {"now"})
        clock = aware_timestamp(fields["now"])
        limit = int(fields.get("limit", 32))
        if not 1 <= limit <= 128:
            raise ValueError("limit must lie between 1 and 128")
        rows = self._due_schedule_rows(clock, limit)
        body = {
            "schema_version": "gig.research-strategy.due-schedules.v1",
            "as_of": clock.isoformat(),
            "dataset_id": "due-schedules",
            "source": "persisted-research-strategy-schedule",
            "input_classification": "MODEL_ESTIMATE",
            "broker_connected": False,
            "trading_approved": False,
            "economically_validated": False,
            "classification": "MODEL_ESTIMATE",
            "rows": rows,
            "metrics": {"due": len(rows), "trading_approved": False},
            "notes": [
                "Due schedules are advisory for explicit agent ticks.",
                "No background worker or trading approval is established.",
            ],
        }
        return Contract.admit(
            State.AVAILABLE, body, source=body["source"],
            classification="MODEL_ESTIMATE", as_of=clock.isoformat(),
            scope="workspace.research-strategy",
            source_identity="due-schedules",
            model_identity=_hash({"as_of": clock.isoformat(), "rows": rows}),
            model_version="gig.research-strategy.due-schedules.v1",
            evidence_state="HASH_VERIFIED_RESEARCH_STRATEGY_DUE_SCHEDULES",
            readiness="RESEARCH_ONLY",
        )

    def _due_schedule_rows(self, clock, limit: int) -> list[dict[str, Any]]:
        directory = self._path("schedules")
        rows: list[dict[str, Any]] = []
        if not directory.is_dir():
            return rows
        for path in sorted(directory.glob("*.json")):
            if path.name.endswith(".pointer.json"):
                continue
            try:
                plan = self.load_schedule_state(path.stem)
            except (FileNotFoundError, ValueError):
                continue
            if int(plan["ticks_completed"]) >= int(plan["max_ticks"]):
                continue
            if plan.get("alpha_decayed"):
                continue
            due = aware_timestamp(plan["next_due"])
            if clock < due:
                continue
            rows.append({
                "schedule_id": plan["schedule_id"],
                "action": plan["action"],
                "next_due": plan["next_due"],
                "ticks_completed": plan["ticks_completed"],
                "max_ticks": plan["max_ticks"],
                "dataset_id": plan["dataset_id"],
            })
            if len(rows) >= limit:
                break
        return rows

    def tick_due_schedules(self, payload: dict[str, Any]) -> Contract:
        """Execute one tick for each due schedule under an explicit agent clock.

        Bounded batch for orchestrators. Failures are recorded per schedule;
        the batch itself stays research-only and never starts a worker.
        """
        fields = _fields(payload, {"now", "limit"}, {"now"})
        clock = aware_timestamp(fields["now"])
        limit = int(fields.get("limit", 8))
        if not 1 <= limit <= 32:
            raise ValueError("limit must lie between 1 and 32")
        due = self._due_schedule_rows(clock, limit)
        ticks: list[dict[str, Any]] = []
        for row in due:
            result = self.tick_schedule({
                "schedule_id": row["schedule_id"], "now": clock.isoformat(),
            }).to_dict()
            entry: dict[str, Any] = {
                "schedule_id": row["schedule_id"],
                "action": row["action"],
                "state": result["state"],
            }
            data = result.get("data")
            if isinstance(data, dict):
                entry["tick_id"] = data.get("tick_id")
                entry["tick_index"] = data.get("tick_index")
                entry["cycle_id"] = data.get("cycle_id")
                entry["watch_id"] = data.get("watch_id")
            if result.get("error") is not None:
                entry["error"] = result["error"]
            ticks.append(entry)
        available = sum(1 for tick in ticks if tick["state"] == State.AVAILABLE.value)
        body = {
            "schema_version": "gig.research-strategy.tick-due.v1",
            "as_of": clock.isoformat(),
            "dataset_id": "tick-due",
            "source": "persisted-research-strategy-schedule",
            "input_classification": "MODEL_ESTIMATE",
            "broker_connected": False,
            "trading_approved": False,
            "economically_validated": False,
            "classification": "MODEL_ESTIMATE",
            "ticks": ticks,
            "metrics": {
                "attempted": len(ticks),
                "available": available,
                "trading_approved": False,
            },
            "notes": [
                "tick-due runs one explicit schedule-tick per due plan under a shared clock.",
                "Per-schedule failures do not abort the batch inventory.",
                "No background worker or trading approval is established.",
            ],
        }
        return Contract.admit(
            State.AVAILABLE, body, source=body["source"],
            classification="MODEL_ESTIMATE", as_of=clock.isoformat(),
            scope="workspace.research-strategy",
            source_identity="tick-due",
            model_identity=_hash({"as_of": clock.isoformat(), "ticks": ticks}),
            model_version="gig.research-strategy.tick-due.v1",
            evidence_state="HASH_VERIFIED_RESEARCH_STRATEGY_TICK_DUE",
            readiness="RESEARCH_ONLY",
        )

    def size(self, payload: dict[str, Any]) -> Contract:
        """Convert screen research weights into capped research lot sizes.

        Prices come from admitted bar evidence. ``vol_scale`` may be caller-
        supplied, or derived from ``target_volatility`` against the saved
        dataset return panel. No order is prepared or submitted.
        """
        fields = _fields(payload, {"screen_id", "now", "portfolio_notional", "max_name_notional",
                                   "lot_size", "max_age_seconds", "vol_scale", "target_volatility",
                                   "max_expected_shortfall", "regime_short_bars", "regime_long_bars",
                                   "max_gross_exposure", "max_adv_participation", "adv_lookback_bars",
                                   "previous_size_id", "max_size_turnover", "annualization",
                                   "findings_id"},
                         {"screen_id", "now", "portfolio_notional", "max_name_notional", "lot_size"})
        if "vol_scale" in fields and "target_volatility" in fields:
            raise ValueError("provide vol_scale or target_volatility, not both")
        if ("regime_short_bars" in fields) != ("regime_long_bars" in fields):
            raise ValueError("regime_short_bars and regime_long_bars must be provided together")
        if "max_adv_participation" in fields and "adv_lookback_bars" not in fields:
            raise ValueError("max_adv_participation requires adv_lookback_bars")
        if "adv_lookback_bars" in fields and "max_adv_participation" not in fields:
            raise ValueError("adv_lookback_bars requires max_adv_participation")
        if "max_size_turnover" in fields and "previous_size_id" not in fields:
            raise ValueError("max_size_turnover requires previous_size_id")
        try:
            screen = self.screens.load(fields["screen_id"])
            dataset = PersistedDatasetWorkspace(self.base_dir).load(screen["dataset_id"])
            previous_size = (self.load_size(fields["previous_size_id"])
                             if "previous_size_id" in fields else None)
            findings = (self.screens.load_fi_findings(fields["findings_id"])
                        if "findings_id" in fields else None)
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        if findings is not None and findings["screen_id"] != screen["screen_id"]:
            raise ValueError("findings_id must reference the same screen_id")
        if self._invalidation_hits(screen_id=screen["screen_id"]):
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        if previous_size is not None:
            if previous_size["dataset_id"] != screen["dataset_id"]:
                raise ValueError("previous_size_id must share the screen dataset")
            if self._invalidation_hits(size_id=previous_size["size_id"]):
                return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                      classification="UNAVAILABLE",
                                      scope="workspace.research-strategy",
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
        if any(row.get("expiration_state") == "EXPIRED" for row in screen["rows"]):
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        screen_rows = list(screen["rows"])
        findings_applied = False
        if findings is not None:
            by_name = {row["instrument_id"]: row for row in findings["rows"]}
            overlaid: list[dict[str, Any]] = []
            for row in screen_rows:
                overlay = by_name.get(row["instrument_id"])
                if overlay is None:
                    overlaid.append(row)
                    continue
                updated = dict(row)
                if overlay.get("stance_override"):
                    updated["stance"] = overlay["stance_override"]
                    updated["research_weight"] = 0.0
                    updated["fi_gate_reason"] = overlay.get("gate_reason")
                elif overlay.get("adjusted_weight") is not None:
                    updated["research_weight"] = float(overlay["adjusted_weight"])
                    updated["fi_weight_adjusted"] = True
                if overlay.get("mention_count"):
                    updated["fi_mention_count"] = int(overlay["mention_count"])
                overlaid.append(updated)
            screen_rows = overlaid
            findings_applied = True
        actionable = [row for row in screen_rows if float(row["research_weight"]) != 0.0]
        if not actionable:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        symbols = sorted({row["instrument_id"] for row in actionable})
        need_risk = ("target_volatility" in fields or "max_expected_shortfall" in fields
                     or "regime_short_bars" in fields)
        complete = None
        weights = None
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
        if "regime_short_bars" in fields:
            short_bars = int(fields["regime_short_bars"])
            long_bars = int(fields["regime_long_bars"])
            if not 5 <= short_bars < long_bars <= 252:
                raise ValueError("regime windows must satisfy 5 <= short < long <= 252")
            assert complete is not None and weights is not None and risk_metrics is not None
            if len(complete) < long_bars:
                return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                      classification="UNAVAILABLE", scope="workspace.research-strategy",
                                      error=public_error("DATA_UNAVAILABLE"))
            series = complete.to_numpy() @ np.asarray(weights, dtype=float)
            short_vol = float(np.std(series[-short_bars:], ddof=1))
            long_vol = float(np.std(series[-long_bars:], ddof=1))
            if not (short_vol == short_vol and long_vol == long_vol) or long_vol <= 0:
                return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                      classification="UNAVAILABLE", scope="workspace.research-strategy",
                                      error=public_error("DATA_UNAVAILABLE"))
            ratio = Decimal(str(long_vol / short_vol)) if short_vol > 0 else Decimal("1")
            regime_scale = min(max(ratio, Decimal("0.1")), Decimal("10"))
            scale = min(scale * regime_scale, Decimal("10"))
            scale_source = (scale_source + "+TRAILING_VOL_REGIME"
                            if scale_source != "IDENTITY" else "TRAILING_VOL_REGIME")
            risk_metrics = {
                **risk_metrics,
                "regime": {
                    "short_bars": short_bars, "long_bars": long_bars,
                    "short_vol": short_vol, "long_vol": long_vol,
                    "regime_scale": str(regime_scale),
                    "method": "DESCRIPTIVE_TRAILING_VOLATILITY_NOT_HMM",
                },
            }
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
        gross_cap_applied = False
        if "max_gross_exposure" in fields:
            gross_cap = _positive(fields["max_gross_exposure"], "max_gross_exposure")
            if gross_cap > Decimal("10"):
                raise ValueError("max_gross_exposure exceeds the research bound")
            gross = sum((Decimal(row["target_notional"]) for row in sized), Decimal(0))
            exposure = gross / portfolio
            if exposure > gross_cap:
                factor = gross_cap / exposure
                rescaled: list[dict[str, Any]] = []
                for row in sized:
                    notional = Decimal(row["target_notional"]) * factor
                    price = Decimal(row["reference_price"])
                    quantity = int(Fraction(notional) / Fraction(price) / Fraction(lot)) * lot
                    if quantity <= 0:
                        continue
                    updated = {**row, "target_notional": str(notional), "quantity": str(quantity),
                               "gross_scaled": True}
                    rescaled.append(updated)
                sized = rescaled
                gross_cap_applied = True
                if not sized:
                    return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                          classification="UNAVAILABLE",
                                          scope="workspace.research-strategy",
                                          error=public_error("DATA_UNAVAILABLE"))
        adv_cap_applied = False
        if "max_adv_participation" in fields:
            participation = _positive(fields["max_adv_participation"], "max_adv_participation")
            if participation > Decimal("1"):
                raise ValueError("max_adv_participation must lie in (0, 1]")
            lookback = int(fields["adv_lookback_bars"])
            if not 5 <= lookback <= 252:
                raise ValueError("adv_lookback_bars must lie between 5 and 252")
            panel = pd.DataFrame(dataset["panel_rows"])
            if not {"timestamp", "instrument_id", "close", "volume"}.issubset(panel.columns):
                return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                      classification="UNAVAILABLE", scope="workspace.research-strategy",
                                      error=public_error("DATA_UNAVAILABLE"))
            panel = panel.copy()
            panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
            panel = panel.loc[panel["timestamp"] <= pd.Timestamp(at)]
            panel["close"] = pd.to_numeric(panel["close"], errors="coerce")
            panel["volume"] = pd.to_numeric(panel["volume"], errors="coerce")
            panel = panel.dropna(subset=["close", "volume"])
            panel = panel.loc[panel["volume"] > 0]
            limited: list[dict[str, Any]] = []
            for row in sized:
                symbol = row["instrument_id"]
                history = panel.loc[panel["instrument_id"] == symbol].sort_values("timestamp")
                if len(history) < lookback:
                    return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                          classification="UNAVAILABLE",
                                          scope="workspace.research-strategy",
                                          error=public_error("DATA_UNAVAILABLE"))
                recent = history.tail(lookback)
                dollar = (recent["close"] * recent["volume"]).astype(float)
                adv = Decimal(str(float(dollar.mean())))
                if adv <= 0:
                    return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                          classification="UNAVAILABLE",
                                          scope="workspace.research-strategy",
                                          error=public_error("DATA_UNAVAILABLE"))
                cap = adv * participation
                notional = min(Decimal(row["target_notional"]), cap)
                price = Decimal(row["reference_price"])
                quantity = int(Fraction(notional) / Fraction(price) / Fraction(lot)) * lot
                if quantity <= 0:
                    continue
                updated = {
                    **row,
                    "target_notional": str(notional),
                    "quantity": str(quantity),
                    "adv_dollar": str(adv),
                    "adv_cap": str(cap),
                    "adv_capped": Decimal(row["target_notional"]) > cap,
                }
                if updated["adv_capped"]:
                    adv_cap_applied = True
                limited.append(updated)
            sized = limited
            if not sized:
                return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                      classification="UNAVAILABLE", scope="workspace.research-strategy",
                                      error=public_error("DATA_UNAVAILABLE"))
        turnover_nav = None
        turnover_cap_applied = False
        if previous_size is not None:
            prior = {
                row["instrument_id"]: Decimal(row["target_notional"]) * (
                    Decimal(1) if row["side"] == "buy" else Decimal(-1))
                for row in previous_size["rows"]
            }
            current = {
                row["instrument_id"]: Decimal(row["target_notional"]) * (
                    Decimal(1) if row["side"] == "buy" else Decimal(-1))
                for row in sized
            }
            names = set(prior) | set(current)
            turnover = sum((abs(current.get(name, Decimal(0)) - prior.get(name, Decimal(0)))
                            for name in names), Decimal(0)) / Decimal(2)
            turnover_nav = turnover / portfolio
            if "max_size_turnover" in fields:
                turnover_cap = _positive(fields["max_size_turnover"], "max_size_turnover")
                if turnover_cap > Decimal("8"):
                    raise ValueError("max_size_turnover must lie in (0, 8]")
                if turnover_nav > turnover_cap and turnover > 0:
                    # Blend toward the previous signed notionals so |Δ|/2 ≤ cap * NAV.
                    # Solve: (1-a)*turnover_nav = turnover_cap  => a = 1 - cap/turnover_nav.
                    blend = Decimal(1) - (turnover_cap / turnover_nav)
                    blended: list[dict[str, Any]] = []
                    for name in sorted(names):
                        signed = ((Decimal(1) - blend) * current.get(name, Decimal(0))
                                  + blend * prior.get(name, Decimal(0)))
                        if signed == 0:
                            continue
                        notional = abs(signed)
                        if name in marks:
                            price = Decimal(marks[name]["price"])
                            price_asof = marks[name]["as_of"]
                        else:
                            # Fall back to previous mark when a closed name lacks a current mark.
                            prior_mark = next(row for row in previous_size["rows"]
                                              if row["instrument_id"] == name)
                            price = Decimal(prior_mark["reference_price"])
                            price_asof = previous_size["as_of"]
                        quantity = int(Fraction(notional) / Fraction(price) / Fraction(lot)) * lot
                        if quantity <= 0:
                            continue
                        prior_row = next((row for row in sized if row["instrument_id"] == name), None)
                        base = prior_row or {
                            "instrument_id": name,
                            "stance": "HOLD",
                            "research_weight": 0.0,
                            "uncapped_notional": str(notional),
                            "capped": False,
                            "reference_price": str(price),
                            "price_asof": price_asof,
                            "trading_approved": False,
                        }
                        blended.append({
                            **base,
                            "side": "buy" if signed > 0 else "sell",
                            "target_notional": str(notional),
                            "quantity": str(quantity),
                            "turnover_blended": True,
                        })
                    sized = blended
                    turnover_cap_applied = True
                    if not sized:
                        return Contract.admit(State.UNAVAILABLE, None,
                                              source="persisted-research-strategy",
                                              classification="UNAVAILABLE",
                                              scope="workspace.research-strategy",
                                              error=public_error("DATA_UNAVAILABLE"))
                    current = {
                        row["instrument_id"]: Decimal(row["target_notional"]) * (
                            Decimal(1) if row["side"] == "buy" else Decimal(-1))
                        for row in sized
                    }
                    names = set(prior) | set(current)
                    turnover = sum((abs(current.get(name, Decimal(0)) - prior.get(name, Decimal(0)))
                                    for name in names), Decimal(0)) / Decimal(2)
                    turnover_nav = turnover / portfolio
        gross = sum((Decimal(row["target_notional"]) for row in sized), Decimal(0))
        net = sum((Decimal(row["target_notional"]) * (1 if row["side"] == "buy" else -1)
                   for row in sized), Decimal(0))
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
        if "regime_short_bars" in fields:
            request["regime_short_bars"] = int(fields["regime_short_bars"])
            request["regime_long_bars"] = int(fields["regime_long_bars"])
        if "max_gross_exposure" in fields:
            request["max_gross_exposure"] = str(_positive(fields["max_gross_exposure"], "max_gross_exposure"))
            request["gross_exposure_cap_applied"] = gross_cap_applied
        if "max_adv_participation" in fields:
            request["max_adv_participation"] = str(
                _positive(fields["max_adv_participation"], "max_adv_participation"))
            request["adv_lookback_bars"] = int(fields["adv_lookback_bars"])
            request["adv_participation_cap_applied"] = adv_cap_applied
        if previous_size is not None:
            request["previous_size_id"] = previous_size["size_id"]
            request["size_turnover_nav"] = str(turnover_nav)
            if "max_size_turnover" in fields:
                request["max_size_turnover"] = str(
                    _positive(fields["max_size_turnover"], "max_size_turnover"))
                request["size_turnover_cap_applied"] = turnover_cap_applied
        if findings is not None:
            request["findings_id"] = findings["findings_id"]
            request["findings_policy"] = findings["policy"]
            request["findings_applied"] = findings_applied
        body = {
            "schema_version": SIZING,
            "as_of": at.isoformat(),
            "dataset_id": screen["dataset_id"],
            "screen_id": screen["screen_id"],
            "signal_id": screen["signal_id"],
            "previous_size_id": previous_size["size_id"] if previous_size is not None else None,
            "findings_id": findings["findings_id"] if findings is not None else None,
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
                "gross_notional": str(gross),
                "net_notional": str(net),
                "gross_exposure": str(gross / portfolio),
                "net_exposure": str(net / portfolio),
                "capped_names": sum(1 for row in sized if row["capped"]),
                "vol_scale": str(scale),
                "vol_scale_source": scale_source,
                "gross_exposure_cap_applied": gross_cap_applied,
                "adv_participation_cap_applied": adv_cap_applied,
                "size_turnover_nav": None if turnover_nav is None else str(turnover_nav),
                "size_turnover_cap_applied": turnover_cap_applied,
                "findings_applied": findings_applied,
                "trading_approved": False,
            },
            "notes": [
                "Position sizes are research lot targets from screen weights and admitted marks.",
                "max_name_notional caps each name; target_volatility, max_expected_shortfall, max_gross_exposure, and max_adv_participation constrain notionals.",
                "ADV caps use trailing close×volume means from the saved panel; they are descriptive liquidity bounds.",
                "previous_size_id measures one-way NAV turnover; max_size_turnover blends toward the prior signed notionals.",
                "Optional findings_id applies FI gate-hold / weight-adjust overlays before sizing; the screen artifact is unchanged.",
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

    def estimate_cost(self, payload: dict[str, Any]) -> Contract:
        """Pre-trade TCA on research sizes using square-root impact against ADV.

        Trades are the full size book from flat cash, or the signed notional
        delta versus ``previous_size_id``. Costs stay ESTIMATED; nothing here
        observes fills, connects a broker, or approves trading.
        """
        fields = _fields(payload, {"size_id", "now", "previous_size_id", "spread_bps",
                                   "commission_bps", "impact_eta", "adv_lookback_bars"},
                         {"size_id", "now"})
        try:
            sized = self.load_size(fields["size_id"])
            previous = (self.load_size(fields["previous_size_id"])
                        if "previous_size_id" in fields else None)
            dataset = PersistedDatasetWorkspace(self.base_dir).load(sized["dataset_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        if self._invalidation_hits(screen_id=sized["screen_id"], size_id=sized["size_id"]):
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        if previous is not None:
            if previous["dataset_id"] != sized["dataset_id"]:
                raise ValueError("previous_size_id must share the size dataset")
            if self._invalidation_hits(size_id=previous["size_id"]):
                return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                      classification="UNAVAILABLE",
                                      scope="workspace.research-strategy",
                                      error=public_error("DATA_UNAVAILABLE"))
        at = aware_timestamp(fields["now"])
        spread_bps = _positive(fields.get("spread_bps", "4"), "spread_bps")
        commission_bps = _positive(fields.get("commission_bps", "1"), "commission_bps")
        if spread_bps > Decimal("500") or commission_bps > Decimal("500"):
            raise ValueError("spread_bps and commission_bps must lie in (0, 500]")
        impact_eta = _positive(fields.get("impact_eta", "0.10"), "impact_eta")
        if impact_eta > Decimal("10"):
            raise ValueError("impact_eta must lie in (0, 10]")
        lookback = int(fields.get("adv_lookback_bars", 20))
        if not 5 <= lookback <= 252:
            raise ValueError("adv_lookback_bars must lie between 5 and 252")
        portfolio = Decimal(sized["request"]["portfolio_notional"])
        current = {
            row["instrument_id"]: Decimal(row["target_notional"]) * (
                Decimal(1) if row["side"] == "buy" else Decimal(-1))
            for row in sized["rows"]
        }
        prior = {}
        if previous is not None:
            prior = {
                row["instrument_id"]: Decimal(row["target_notional"]) * (
                    Decimal(1) if row["side"] == "buy" else Decimal(-1))
                for row in previous["rows"]
            }
        marks = {row["instrument_id"]: row for row in sized["rows"]}
        if previous is not None:
            for row in previous["rows"]:
                marks.setdefault(row["instrument_id"], row)
        names = sorted(set(current) | set(prior))
        panel = pd.DataFrame(dataset["panel_rows"])
        if not {"timestamp", "instrument_id", "close", "volume"}.issubset(panel.columns):
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        panel = panel.copy()
        panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
        panel = panel.loc[panel["timestamp"] <= pd.Timestamp(at)]
        panel["close"] = pd.to_numeric(panel["close"], errors="coerce")
        panel["volume"] = pd.to_numeric(panel["volume"], errors="coerce")
        panel = panel.dropna(subset=["close", "volume"])
        panel = panel.loc[panel["volume"] > 0]
        bp = Decimal("0.0001")
        rows_out: list[dict[str, Any]] = []
        total_notional = Decimal(0)
        total_fees = Decimal(0)
        total_spread = Decimal(0)
        total_impact = Decimal(0)
        for name in names:
            delta = current.get(name, Decimal(0)) - prior.get(name, Decimal(0))
            if delta == 0:
                continue
            trade_notional = abs(delta)
            mark = marks[name]
            price = Decimal(mark["reference_price"])
            if price <= 0:
                raise ValueError("reference_price must be positive")
            quantity = trade_notional / price
            history = panel.loc[panel["instrument_id"] == name].sort_values("timestamp")
            if len(history) < lookback:
                return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                      classification="UNAVAILABLE",
                                      scope="workspace.research-strategy",
                                      error=public_error("DATA_UNAVAILABLE"))
            recent = history.tail(lookback)
            adv = Decimal(str(float((recent["close"] * recent["volume"]).astype(float).mean())))
            if adv <= 0:
                return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                      classification="UNAVAILABLE",
                                      scope="workspace.research-strategy",
                                      error=public_error("DATA_UNAVAILABLE"))
            participation = trade_notional / adv
            impact_bps = impact_eta * participation.sqrt() * Decimal(10_000)
            fees = trade_notional * commission_bps * bp
            spread_cost = trade_notional * spread_bps * bp / Decimal(2)
            impact_cost = trade_notional * impact_bps * bp
            one_way_bps = commission_bps + (spread_bps / Decimal(2)) + impact_bps
            side = "BUY" if delta > 0 else "SELL"
            rows_out.append({
                "instrument_id": name,
                "side": side,
                "trade_notional": str(trade_notional),
                "quantity": str(quantity),
                "reference_price": str(price),
                "adv_dollar": str(adv),
                "participation": str(participation),
                "spread_bps": str(spread_bps),
                "commission_bps": str(commission_bps),
                "impact_eta": str(impact_eta),
                "impact_bps": str(impact_bps),
                "one_way_bps": str(one_way_bps),
                "transaction_fees": str(fees),
                "spread_cost": str(spread_cost),
                "market_impact": str(impact_cost),
                "evidence_class": "ESTIMATED",
                "model": "half-spread + commission + square-root impact",
                "trading_approved": False,
            })
            total_notional += trade_notional
            total_fees += fees
            total_spread += spread_cost
            total_impact += impact_cost
        total_cost = total_fees + total_spread + total_impact
        turnover_nav = (total_notional / Decimal(2) / portfolio) if portfolio > 0 else Decimal(0)
        # A zero-delta book is an AVAILABLE zero-cost estimate (no rebalance trades),
        # not an unavailable research artifact — schedule ticks rely on this.
        body = {
            "schema_version": COST,
            "as_of": at.isoformat(),
            "dataset_id": sized["dataset_id"],
            "screen_id": sized["screen_id"],
            "signal_id": sized["signal_id"],
            "size_id": sized["size_id"],
            "previous_size_id": previous["size_id"] if previous is not None else None,
            "source": sized["source"],
            "input_classification": sized["input_classification"],
            "request": {
                "spread_bps": str(spread_bps),
                "commission_bps": str(commission_bps),
                "impact_eta": str(impact_eta),
                "adv_lookback_bars": lookback,
                "portfolio_notional": str(portfolio),
                "trade_basis": "DELTA_VS_PREVIOUS" if previous is not None else "OPEN_FROM_FLAT",
            },
            "rows": rows_out,
            "broker_connected": False,
            "trading_approved": False,
            "economically_validated": False,
            "classification": "MODEL_ESTIMATE",
            "metrics": {
                "names": len(rows_out),
                "trade_notional": str(total_notional),
                "turnover_nav": str(turnover_nav),
                "transaction_fees": str(total_fees),
                "spread_cost": str(total_spread),
                "market_impact": str(total_impact),
                "total_estimated_cost": str(total_cost),
                "total_estimated_cost_bps": (
                    None if total_notional == 0 else str(total_cost / total_notional * Decimal(10_000))),
                "cost_nav": str(total_cost / portfolio),
                "evidence_class": "ESTIMATED",
                "trading_approved": False,
            },
            "notes": [
                "Pre-trade costs use half-spread, commission, and square-root impact against trailing ADV.",
                "ADV is the mean of close×volume over adv_lookback_bars ending at the request clock.",
                "Estimates are never summed with simulated or observed fills.",
                "No broker order, OMS prepare, or trading approval is established.",
            ],
        }
        identity = _hash(body)
        document = {**body, "cost_id": identity}
        self._path("costs").mkdir(parents=True, exist_ok=True)
        write_immutable(self._path("costs/" + identity + ".json"), dumps(document).encode())
        return self._contract(document, title="Unapproved research pre-trade cost estimate",
                              model_version=COST,
                              evidence="HASH_VERIFIED_RESEARCH_PRETRADE_COST",
                              identity_key="cost_id")

    def load_cost(self, cost_id: str) -> dict[str, Any]:
        identity = sha256(cost_id)
        path = self._path("costs/" + identity + ".json")
        if not path.is_file():
            raise FileNotFoundError("research cost estimate unavailable")
        document = loads(path.read_bytes())
        body = {k: v for k, v in document.items() if k != "cost_id"}
        if document.get("cost_id") != identity or _hash(body) != identity or body.get("schema_version") != COST:
            raise ValueError("cost estimate artifact integrity mismatch")
        return document

    def replay_cost(self, payload: dict[str, Any]) -> Contract:
        _fields(payload, {"cost_id"}, {"cost_id"})
        try:
            document = self.load_cost(payload["cost_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        return self._contract(document, title="Unapproved research pre-trade cost estimate",
                              model_version=COST,
                              evidence="HASH_VERIFIED_RESEARCH_PRETRADE_COST",
                              identity_key="cost_id")

    def cost_rebalance(self, payload: dict[str, Any]) -> Contract:
        """Propose a cost-aware research book from a sized target and TCA estimate.

        Modes skip expensive or tiny trades, shrink turnover when cost_nav exceeds a
        bound, or gate trades whose ESTIMATED cost exceeds an uncalibrated alpha budget.
        Emits a research proposal only — no OMS prepare, broker, or trading approval.
        """
        modes = frozenset({"SKIP_BELOW_COST", "SHRINK_TO_COST", "ALPHA_COST_GATE"})
        fields = _fields(payload, {"size_id", "cost_id", "now", "mode", "previous_size_id",
                                   "spread_bps", "commission_bps", "impact_eta",
                                   "adv_lookback_bars", "min_trade_notional", "max_one_way_bps",
                                   "max_cost_nav", "alpha_scale"},
                         {"size_id", "now", "mode"})
        mode = fields["mode"]
        if mode not in modes:
            raise ValueError("mode must be SKIP_BELOW_COST, SHRINK_TO_COST, or ALPHA_COST_GATE")
        try:
            sized = self.load_size(fields["size_id"])
            if "cost_id" in fields:
                cost = self.load_cost(fields["cost_id"])
            else:
                cost_payload = {"size_id": fields["size_id"], "now": fields["now"]}
                for key in ("previous_size_id", "spread_bps", "commission_bps", "impact_eta",
                            "adv_lookback_bars"):
                    if key in fields:
                        cost_payload[key] = fields[key]
                costed = self.estimate_cost(cost_payload).to_dict()
                if costed["state"] != State.AVAILABLE.value or not isinstance(costed.get("data"), dict):
                    return Contract.admit(costed["state"], costed.get("data"),
                                          source="persisted-research-strategy",
                                          classification=costed["provenance"]["classification"],
                                          as_of=costed["provenance"].get("as_of"),
                                          scope="workspace.research-strategy",
                                          error=costed.get("error"),
                                          readiness="RESEARCH_ONLY")
                cost = costed["data"]
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        if cost["size_id"] != sized["size_id"]:
            raise ValueError("cost_id must reference the same size_id")
        if cost["dataset_id"] != sized["dataset_id"]:
            raise ValueError("cost_id must share the size dataset")
        if self._invalidation_hits(screen_id=sized["screen_id"], size_id=sized["size_id"]):
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        previous = None
        previous_id = cost.get("previous_size_id")
        if previous_id:
            try:
                previous = self.load_size(previous_id)
            except FileNotFoundError:
                return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                      classification="UNAVAILABLE",
                                      scope="workspace.research-strategy",
                                      error=public_error("DATA_UNAVAILABLE"))
            if previous["dataset_id"] != sized["dataset_id"]:
                raise ValueError("previous_size_id must share the size dataset")
            if self._invalidation_hits(size_id=previous["size_id"]):
                return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                      classification="UNAVAILABLE",
                                      scope="workspace.research-strategy",
                                      error=public_error("DATA_UNAVAILABLE"))
        at = aware_timestamp(fields["now"])
        portfolio = Decimal(sized["request"]["portfolio_notional"])
        target = {
            row["instrument_id"]: Decimal(row["target_notional"]) * (
                Decimal(1) if row["side"] == "buy" else Decimal(-1))
            for row in sized["rows"]
        }
        weights = {row["instrument_id"]: Decimal(str(row.get("research_weight", 0)))
                   for row in sized["rows"]}
        marks = {row["instrument_id"]: row for row in sized["rows"]}
        prior: dict[str, Decimal] = {}
        if previous is not None:
            prior = {
                row["instrument_id"]: Decimal(row["target_notional"]) * (
                    Decimal(1) if row["side"] == "buy" else Decimal(-1))
                for row in previous["rows"]
            }
            for row in previous["rows"]:
                marks.setdefault(row["instrument_id"], row)
        cost_by_name = {row["instrument_id"]: row for row in cost["rows"]}
        min_trade = Decimal(str(fields.get("min_trade_notional", "0")))
        if min_trade < 0 or not min_trade.is_finite():
            raise ValueError("min_trade_notional must be a non-negative decimal")
        max_bps = None
        if "max_one_way_bps" in fields:
            max_bps = _positive(fields["max_one_way_bps"], "max_one_way_bps")
            if max_bps > Decimal("10_000"):
                raise ValueError("max_one_way_bps must lie in (0, 10000]")
        max_cost_nav: Decimal | None = None
        if mode == "SHRINK_TO_COST":
            max_cost_nav = _positive(fields.get("max_cost_nav", "0.001"), "max_cost_nav")
            if max_cost_nav > Decimal("1"):
                raise ValueError("max_cost_nav must lie in (0, 1]")
        alpha_scale: Decimal | None = None
        if mode == "ALPHA_COST_GATE":
            alpha_scale = _positive(fields.get("alpha_scale", "0.01"), "alpha_scale")
            if alpha_scale > Decimal("10"):
                raise ValueError("alpha_scale must lie in (0, 10]")

        proposed = dict(target)
        actions: dict[str, tuple[str, str | None]] = {}
        cost_nav = Decimal(cost["metrics"]["cost_nav"])
        scale = Decimal(1)

        if mode == "SHRINK_TO_COST":
            assert max_cost_nav is not None
            if cost_nav > max_cost_nav and cost_nav > 0:
                scale = max_cost_nav / cost_nav
                if scale > 1:
                    scale = Decimal(1)
                for name in set(target) | set(prior):
                    delta = target.get(name, Decimal(0)) - prior.get(name, Decimal(0))
                    proposed[name] = prior.get(name, Decimal(0)) + scale * delta
                for name in set(proposed):
                    if proposed[name] == 0 and name not in target and name not in prior:
                        del proposed[name]

        for name in set(target) | set(prior) | set(cost_by_name):
            tgt = target.get(name, Decimal(0))
            prv = prior.get(name, Decimal(0))
            prop = proposed.get(name, tgt)
            delta = tgt - prv
            trade_notional = abs(delta)
            skip_reason = None
            action = "KEEP"
            if trade_notional == 0 and abs(prop - prv) == 0:
                action = "KEEP"
                proposed[name] = prv
            elif mode == "SHRINK_TO_COST":
                if scale < 1 and trade_notional > 0:
                    action = "SHRINK"
                    skip_reason = "COST_NAV_SHRINK"
                elif abs(prop - prv) > 0:
                    action = "TRADE"
                else:
                    action = "KEEP"
            elif mode == "SKIP_BELOW_COST":
                row = cost_by_name.get(name)
                if row is None or trade_notional == 0:
                    action = "KEEP" if abs(tgt - prv) == 0 else "TRADE"
                    proposed[name] = tgt
                else:
                    one_way = Decimal(row["one_way_bps"])
                    notional = Decimal(row["trade_notional"])
                    if min_trade > 0 and notional < min_trade:
                        action, skip_reason = "SKIP", "BELOW_MIN_TRADE_NOTIONAL"
                        proposed[name] = prv
                    elif max_bps is not None and one_way > max_bps:
                        action, skip_reason = "SKIP", "ABOVE_MAX_ONE_WAY_BPS"
                        proposed[name] = prv
                    else:
                        action = "TRADE"
                        proposed[name] = tgt
            else:  # ALPHA_COST_GATE
                assert alpha_scale is not None
                row = cost_by_name.get(name)
                if row is None or trade_notional == 0:
                    action = "KEEP" if abs(tgt - prv) == 0 else "TRADE"
                    proposed[name] = tgt
                else:
                    trade_est = (Decimal(row["transaction_fees"]) + Decimal(row["spread_cost"])
                                 + Decimal(row["market_impact"]))
                    weight = abs(weights.get(name, Decimal(0)))
                    budget = weight * alpha_scale * portfolio
                    if trade_est > budget:
                        action, skip_reason = "SKIP", "ALPHA_COST_GATE"
                        proposed[name] = prv
                    else:
                        action = "TRADE"
                        proposed[name] = tgt
            actions[name] = (action, skip_reason)

        rows_out: list[dict[str, Any]] = []
        trades_proposed = 0
        trades_skipped = 0
        total_est = Decimal(0)
        trade_notional_sum = Decimal(0)
        for name in sorted(set(proposed) | set(actions)):
            signed = proposed.get(name, Decimal(0))
            if signed == 0 and name not in target and name not in prior:
                continue
            mark = marks.get(name)
            if mark is None:
                continue
            price = Decimal(mark["reference_price"])
            notional = abs(signed)
            quantity = (notional / price) if price > 0 else Decimal(0)
            action, skip_reason = actions.get(name, ("KEEP", None))
            if action == "TRADE" or action == "SHRINK":
                trades_proposed += 1
            elif action == "SKIP":
                trades_skipped += 1
            row_cost = cost_by_name.get(name)
            est: Decimal | None = None
            trade_n: Decimal | None = None
            if row_cost is not None and action in {"TRADE", "SHRINK"}:
                base_est = (Decimal(row_cost["transaction_fees"]) + Decimal(row_cost["spread_cost"])
                            + Decimal(row_cost["market_impact"]))
                est = base_est * scale if action == "SHRINK" else base_est
                trade_n = abs(signed - prior.get(name, Decimal(0)))
                total_est += est
                trade_notional_sum += trade_n
            elif row_cost is not None and action == "SKIP":
                trade_n = Decimal(row_cost["trade_notional"])
            entry = {
                "instrument_id": name,
                "side": "buy" if signed > 0 else ("sell" if signed < 0 else "flat"),
                "target_notional": str(notional),
                "quantity": str(quantity),
                "reference_price": str(price),
                "action": action,
                "trade_notional": None if trade_n is None else str(trade_n),
                "estimated_cost": None if est is None else str(est),
                "skip_reason": skip_reason,
                "research_weight": float(weights.get(name, Decimal(0))),
                "trading_approved": False,
            }
            rows_out.append(entry)
        # Drop pure flat KEEP zeros with no prior/target interest
        rows_out = [r for r in rows_out if not (
            r["side"] == "flat" and r["action"] == "KEEP" and Decimal(r["target_notional"]) == 0)]
        turnover = Decimal(0)
        for name in set(proposed) | set(prior):
            turnover += abs(proposed.get(name, Decimal(0)) - prior.get(name, Decimal(0)))
        turnover = turnover / Decimal(2)
        turnover_nav = turnover / portfolio if portfolio > 0 else Decimal(0)
        body = {
            "schema_version": COST_REBALANCE,
            "as_of": at.isoformat(),
            "dataset_id": sized["dataset_id"],
            "screen_id": sized["screen_id"],
            "signal_id": sized["signal_id"],
            "size_id": sized["size_id"],
            "previous_size_id": previous["size_id"] if previous is not None else None,
            "cost_id": cost["cost_id"],
            "source": sized["source"],
            "input_classification": sized["input_classification"],
            "request": {
                "mode": mode,
                "min_trade_notional": str(min_trade),
                "max_one_way_bps": None if max_bps is None else str(max_bps),
                "max_cost_nav": None if max_cost_nav is None else str(max_cost_nav),
                "alpha_scale": None if alpha_scale is None else str(alpha_scale),
                "shrink_scale": str(scale),
                "portfolio_notional": str(portfolio),
                "trade_basis": cost["request"]["trade_basis"],
            },
            "rows": rows_out,
            "broker_connected": False,
            "trading_approved": False,
            "economically_validated": False,
            "classification": "MODEL_ESTIMATE",
            "metrics": {
                "names": len(rows_out),
                "trades_proposed": trades_proposed,
                "trades_skipped": trades_skipped,
                "turnover_nav": str(turnover_nav),
                "trade_notional": str(trade_notional_sum),
                "total_estimated_cost": str(total_est),
                "cost_nav": str(total_est / portfolio if portfolio > 0 else Decimal(0)),
                "input_cost_nav": str(cost_nav),
                "evidence_class": "ESTIMATED",
                "trading_approved": False,
            },
            "notes": [
                "Cost-aware rebalance proposes KEEP/TRADE/SKIP/SHRINK actions from ESTIMATED TCA.",
                "ALPHA_COST_GATE uses uncalibrated research_weight × alpha_scale × portfolio as a dollar budget.",
                "Proposals never prepare OMS intents, connect a broker, or approve trading.",
                "Sized books are not rewritten; this artifact is advisory relative to size_id.",
            ],
        }
        identity = _hash(body)
        document = {**body, "rebalance_id": identity}
        self._path("rebalances").mkdir(parents=True, exist_ok=True)
        write_immutable(self._path("rebalances/" + identity + ".json"), dumps(document).encode())
        return self._contract(document, title="Unapproved research cost-aware rebalance proposal",
                              model_version=COST_REBALANCE,
                              evidence="HASH_VERIFIED_RESEARCH_COST_REBALANCE",
                              identity_key="rebalance_id")

    def load_rebalance(self, rebalance_id: str) -> dict[str, Any]:
        identity = sha256(rebalance_id)
        path = self._path("rebalances/" + identity + ".json")
        if not path.is_file():
            raise FileNotFoundError("research cost rebalance unavailable")
        document = loads(path.read_bytes())
        body = {k: v for k, v in document.items() if k != "rebalance_id"}
        if (document.get("rebalance_id") != identity or _hash(body) != identity
                or body.get("schema_version") != COST_REBALANCE):
            raise ValueError("cost rebalance artifact integrity mismatch")
        return document

    def replay_rebalance(self, payload: dict[str, Any]) -> Contract:
        _fields(payload, {"rebalance_id"}, {"rebalance_id"})
        try:
            document = self.load_rebalance(payload["rebalance_id"])
        except FileNotFoundError:
            return Contract.admit(State.UNAVAILABLE, None, source="persisted-research-strategy",
                                  classification="UNAVAILABLE", scope="workspace.research-strategy",
                                  error=public_error("DATA_UNAVAILABLE"))
        return self._contract(document, title="Unapproved research cost-aware rebalance proposal",
                              model_version=COST_REBALANCE,
                              evidence="HASH_VERIFIED_RESEARCH_COST_REBALANCE",
                              identity_key="rebalance_id")


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
