"""Saved research -> constrained offline targets -> atomic durable intents.

Planning never submits or fills. Every mark comes from admitted source evidence,
and worst-case asynchronous fills must fit the explicit exposure policy.
"""
from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterable
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd

from gig.artifacts import write_immutable
from gig.data.validated import PointInTimeStore
from gig.execution.oms import TERMINAL, OfflineOMS, _decimal, _exact
from gig.execution.outbox import ExecutionOutbox
from gig.execution.scheduler import DurableTWAPScheduler
from gig.portfolio.construct import dollar_neutral_quantiles
from gig.research.model_pipeline import integer, sha256
from gig.risk.analytics import portfolio_risk
from gig.service.contracts import Contract, State
from gig.service.execution_workspace import _stamp
from gig.service.serialization import dumps, loads, public_error
from gig.service.workspace import TerminalWorkspace, _age, _fields, _number
from gig.timestamps import aware_timestamp


def _hash(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def _worst_signed(initial: Fraction, changes: Iterable[Fraction]) -> Fraction:
    values = list(changes)
    positive = initial + sum((v for v in values if v > 0), Fraction(0))
    negative = initial + sum((v for v in values if v < 0), Fraction(0))
    return max(abs(positive), abs(negative))


class ResearchAllocationWorkspace:
    def __init__(self, base_dir: Path):
        self.workspace = TerminalWorkspace(base_dir)
        self.base_dir = self.workspace.base_dir

    def _path(self, name: str) -> Path:
        return self.workspace._path(f"allocations/{name}")

    @staticmethod
    def _contract(data, at, *, state=State.AVAILABLE, identity=None):
        return Contract.admit(state, data if state in (State.AVAILABLE, State.STALE) else None,
                              source="persisted-research-offline-allocation",
                              classification="MODEL_ESTIMATE", as_of=at,
                              scope="workspace.research-allocation", source_identity=identity,
                              evidence_state="SIMULATED_NOT_BROKER_OBSERVED",
                              error=public_error("DATA_UNAVAILABLE") if state == State.UNAVAILABLE
                              else public_error("INVALID_REQUEST") if state == State.ERROR else None)

    def load(self, allocation_id: str) -> dict[str, Any]:
        document = loads(self._path(f"{sha256(allocation_id)}.json").read_bytes())
        body = {k: v for k, v in document.items() if k != "allocation_id"}
        if document.get("allocation_id") != allocation_id or _hash(body) != allocation_id:
            raise ValueError("allocation integrity mismatch")
        # A valid hash cannot make an invalidated research lineage reusable.
        # This read precedes every cached view and OMS preparation mutation.
        kind = document.get("source_kind", "run_id")
        source_id = sha256(document.get("research_source_identity") or document.get("run_id"))
        if kind == "run_id":
            from gig.service.persisted_research import PersistedResearchWorkspace
            source = PersistedResearchWorkspace(self.base_dir).load(source_id)
        elif kind == "signal_id":
            from gig.service.signal_workspace import SignalWorkspace
            source = SignalWorkspace(self.base_dir).load(source_id)
        elif kind == "pair_run_id":
            from gig.service.pairs_workspace import PairsResearchWorkspace
            source = PairsResearchWorkspace(self.base_dir).load(source_id)
        else:
            raise ValueError("unsupported saved allocation research source")
        if source["dataset_id"] != document["dataset_id"]:
            raise ValueError("saved allocation differs from its research dataset")
        return document

    def _positions(self) -> tuple[dict[str, Decimal], bool]:
        path = self.workspace._path("execution-engine/oms.sqlite")
        if not path.is_file():
            return {}, False
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
            positions = {symbol: Decimal(quantity) for symbol, quantity in
                         connection.execute("SELECT symbol,quantity FROM positions")}
            open_orders = any(state not in TERMINAL for (state,) in connection.execute("SELECT state FROM orders"))
        return positions, open_orders

    @_exact
    def build(self, payload: dict[str, Any]) -> Contract:
        _fields(payload, {"run_id", "signal_id", "pair_run_id", "max_turnover", "max_transaction_cost", "now", "initial_cash", "n_long", "n_short", "gross_leverage",
                          "max_name_weight", "lot_size", "max_age_seconds", "annualization",
                          "metadata", "limits", "commission_bps", "slippage_bps", "price_tick"},
                {"now", "initial_cash", "n_long", "n_short", "gross_leverage",
                 "max_name_weight", "lot_size", "max_age_seconds", "annualization", "metadata", "limits"})
        request = loads(dumps(payload))
        sources = set(request) & {"run_id", "signal_id", "pair_run_id"}
        if len(sources) != 1:
            raise ValueError("allocation requires exactly one saved model, signal or pair source")
        source_key = next(iter(sources))
        source_id = sha256(request[source_key])
        at, maximum = _age(request["now"], request["max_age_seconds"])
        if maximum is None:
            raise ValueError("allocation requires an explicit freshness bound")
        cash = _decimal(request["initial_cash"])
        lot = _decimal(request["lot_size"])
        price_tick = _decimal(request["price_tick"]) if "price_tick" in request else None
        long_count = integer(request["n_long"], 1, 16, "n_long")
        short_count = integer(request["n_short"], 1, 16, "n_short")
        annualization = integer(request["annualization"], 1, 525600, "annualization")
        gross, name = _number(request["gross_leverage"], positive=True), _number(request["max_name_weight"], positive=True)
        if gross > 4 or name > 1:
            raise ValueError("allocation leverage or concentration exceeds research bounds")
        costs = {k: _number(request.get(k, 0)) for k in ("commission_bps", "slippage_bps")}
        if any(not 0 <= value <= 1000 for value in costs.values()):
            raise ValueError("modeled transaction costs exceed bounds")
        turnover_cap = _number(request["max_turnover"]) if "max_turnover" in request else None
        if turnover_cap is not None and not 0 <= turnover_cap <= 8:
            raise ValueError("turnover cap must be between zero and eight NAV units")
        transaction_cost_cap = _decimal(request["max_transaction_cost"], positive=False) if "max_transaction_cost" in request else None
        metadata, limits = request["metadata"], request["limits"]
        if not isinstance(metadata, dict) or len(metadata) > 32 or not isinstance(limits, dict):
            raise ValueError("bounded explicit instrument metadata and limits required")
        _fields(limits, {"max_gross", "max_abs_net", "max_name_weight", "max_sector_gross", "max_factor_exposure", "min_cash"},
                {"max_gross", "max_abs_net", "max_name_weight", "max_sector_gross", "max_factor_exposure", "min_cash"})
        scalar_limits = {key: _number(limits[key]) for key in ("max_gross", "max_abs_net", "max_name_weight", "max_sector_gross")}
        factors = limits["max_factor_exposure"]
        if not isinstance(factors, dict) or len(factors) > 16 or any(not isinstance(k, str) or not k for k in factors):
            raise ValueError("explicit bounded factor limits required")
        factor_limits = {key: _number(value) for key, value in factors.items()}
        minimum_cash = _decimal(limits["min_cash"], positive=False)
        if any(v < 0 or v > 4 for v in [*scalar_limits.values(), *factor_limits.values()]):
            raise ValueError("invalid exposure limits")
        from gig.service.dataset_workspace import PersistedDatasetWorkspace
        from gig.service.portfolio_workspace import PortfolioWorkspace
        from gig.service.risk_coordination import reconcile_with_risk_controls

        try:
            if source_key == "run_id":
                from gig.service.persisted_research import PersistedResearchWorkspace
                run = PersistedResearchWorkspace(self.base_dir).load(source_id)
            elif source_key == "signal_id":
                from gig.service.signal_workspace import SignalWorkspace
                run = SignalWorkspace(self.base_dir).load(source_id)
            else:
                from gig.service.pairs_workspace import PairsResearchWorkspace
                run = PairsResearchWorkspace(self.base_dir).load(source_id)
            dataset = PersistedDatasetWorkspace(self.base_dir).load(run["dataset_id"])
        except FileNotFoundError:
            return self._contract(None, at.isoformat(), state=State.UNAVAILABLE)
        if not run["usable_for_portfolio"] or aware_timestamp(run["as_of"]) > at:
            return self._contract(None, at.isoformat(), state=State.UNAVAILABLE)
        current, open_orders = self._positions()
        if open_orders:
            return self._contract(None, at.isoformat(), state=State.UNKNOWN)
        pair_weights: dict[str, float] | None = None
        if source_key == "pair_run_id":
            proposals = [p for p in run["proposals"] if p["state"] == "AVAILABLE"]
            if not proposals:
                return self._contract(None, at.isoformat(), state=State.UNAVAILABLE)
            latest = max(aware_timestamp(p["timestamp"]) for p in proposals)
            pair_weights = {}
            for proposal in proposals:
                if aware_timestamp(proposal["timestamp"]) == latest:
                    for symbol, weight in proposal["weights"].items():
                        pair_weights[symbol] = pair_weights.get(symbol, 0.) + _number(weight)
            scores = {s: w for s, w in pair_weights.items() if w != 0.}
        else:
            rows = run["rows"]
            if not rows:
                return self._contract(None, at.isoformat(), state=State.UNAVAILABLE)
            latest = max(aware_timestamp(row["timestamp"]) for row in rows)
            scores = {row["instrument_id"]: _number(row["prediction"]) for row in rows
                      if aware_timestamp(row["timestamp"]) == latest}
        if latest > at or (at - latest).total_seconds() > maximum:
            return self._contract(None, latest.isoformat(), state=State.UNAVAILABLE)
        if len(scores) < long_count + short_count or max(scores.values()) == min(scores.values()):
            return self._contract(None, at.isoformat(), state=State.UNAVAILABLE)
        symbols = sorted(set(scores) | set(current))
        for symbol in symbols:
            item = metadata.get(symbol)
            if not isinstance(item, dict) or set(item) != {"sector", "factors"} or not isinstance(item["sector"], str) or not item["sector"]:
                raise ValueError("every instrument requires explicit sector/factor metadata")
            if not isinstance(item["factors"], dict) or not set(factors) <= item["factors"].keys():
                raise ValueError("required factor exposure is unknown")
            for value in item["factors"].values():
                _number(value)
        source_policy = dataset["provenance"]["stored_corporate_action_policy"]
        snapshot = PointInTimeStore(self.workspace._path("bars.sqlite"), read_only=True).snapshot(at, source=dataset["source"])
        marks: dict[str, dict[str, str]] = {}
        for row in snapshot["rows"]:
            symbol = row["instrument_id"]
            if (symbol in symbols and row["classification"] == dataset["classification"]
                    and row["currency"] == request.get("currency", dataset.get("currency", "USD"))
                    and row["corporate_action_policy"] == source_policy
                    and (symbol not in marks or aware_timestamp(row["timestamp"]) > aware_timestamp(marks[symbol]["as_of"]))):
                marks[symbol] = {"price": str(row["close"]), "as_of": row["timestamp"], "source_sha256": row["snapshot_hash"]}
        if set(marks) != set(symbols) or any((at - aware_timestamp(v["as_of"])).total_seconds() > maximum for v in marks.values()):
            return self._contract(None, at.isoformat(), state=State.UNAVAILABLE)
        if price_tick is not None:
            for mark in marks.values():
                mark["source_price"] = mark["price"]
                quote = Decimal(round(Fraction(Decimal(mark["price"])) / Fraction(price_tick))) * price_tick
                mark["price"] = str(_decimal(quote))
        nav, actual_cash = cash, cash
        if self.workspace._path("execution-engine/oms.sqlite").is_file():
            response = reconcile_with_risk_controls(self.base_dir, {"operation": "reconcile", "initial_cash": str(cash),
                "now": at.isoformat(), "marks": {k: {a: v[a] for a in ("price", "as_of")} for k, v in marks.items()},
                "max_age_seconds": maximum, "metadata": metadata, "limits": limits,
                "source_identity": source_id, "input_classification": dataset["classification"]}).to_dict()
            if response["state"] != "AVAILABLE":
                return self._contract(None, at.isoformat(), state=State.ERROR if response["state"] == "ERROR" else State.UNKNOWN)
            account = PortfolioWorkspace(self.base_dir).load(response["provenance"]["source_identity"])
            nav, actual_cash = Decimal(account["nav"]), Decimal(account["cash"])
        if nav <= 0:
            return self._contract(None, at.isoformat(), state=State.ERROR)
        if pair_weights is None:
            weights = dollar_neutral_quantiles(pd.Series(scores).sort_index(), n_long=long_count, n_short=short_count,
                                               max_name_weight=name, gross_leverage=gross)
        else:
            # Preserve the pair hedge ratio and signs, reducing all legs by one
            # common factor. Never turn hedge weights into independent ranks.
            weights = pd.Series(pair_weights).sort_index()
            if (weights > 0).sum() != long_count or (weights < 0).sum() != short_count:
                raise ValueError("pair proposal leg counts differ from explicit allocation request")
            total_gross, maximum_name = float(weights.abs().sum()), float(weights.abs().max())
            if total_gross <= 0 or maximum_name <= 0:
                return self._contract(None, at.isoformat(), state=State.UNAVAILABLE)
            weights *= min(1., gross/total_gross, name/maximum_name)
        targets = {symbol: int(Fraction(Decimal(str(weights.get(symbol, 0)))) * Fraction(nav)
                               / Fraction(Decimal(marks[symbol]["price"])) / Fraction(lot)) * lot for symbol in symbols}
        prior_weights: dict[str, Fraction] = {s: Fraction(current.get(s, Decimal(0)) * Decimal(marks[s]["price"])) / Fraction(nav) for s in symbols}
        target_weights: dict[str, Fraction] = {s: Fraction(targets[s] * Decimal(marks[s]["price"])) / Fraction(nav) for s in symbols}
        deltas = {s: target_weights[s] - prior_weights[s] for s in symbols}
        worst = {s: max(abs(prior_weights[s]), abs(target_weights[s])) for s in symbols}
        violations = []
        def check(label, value, cap):
            if value > Fraction(Decimal(str(cap))):
                violations.append(label)
        check("gross", sum(worst.values()), scalar_limits["max_gross"])
        check("concentration", max(worst.values()), scalar_limits["max_name_weight"])
        prior_net = sum(prior_weights.values(), Fraction(0))
        check("asynchronous_net", _worst_signed(prior_net, deltas.values()), scalar_limits["max_abs_net"])
        for sector in {metadata[s]["sector"] for s in symbols}:
            check("sector:" + sector, sum(worst[s] for s in symbols if metadata[s]["sector"] == sector), scalar_limits["max_sector_gross"])
        for factor, cap in factor_limits.items():
            initial = sum((prior_weights[s] * Fraction(Decimal(str(metadata[s]["factors"][factor]))) for s in symbols), Fraction(0))
            changes = [deltas[s] * Fraction(Decimal(str(metadata[s]["factors"][factor]))) for s in symbols]
            check("asynchronous_factor:" + factor, _worst_signed(initial, changes), cap)
        traded_value = {s: (targets[s] - current.get(s, Decimal(0))) * Decimal(marks[s]["price"]) for s in symbols}
        modeled_cost = sum(abs(v) for v in traded_value.values()) * Decimal(str(sum(costs.values()))) / 10000
        turnover = sum((abs(v) for v in deltas.values()), Fraction(0))
        if turnover_cap is not None:
            check("turnover", turnover, turnover_cap)
        if transaction_cost_cap is not None and modeled_cost > transaction_cost_cap:
            violations.append("transaction_cost")
        buying_cash = sum(max(v, Decimal(0)) for v in traded_value.values())
        if actual_cash - buying_cash - modeled_cost < minimum_cash:
            violations.append("worst_case_cash")
        if violations:
            return self._contract(None, at.isoformat(), state=State.ERROR)
        planned = []
        for s in symbols:
            delta = targets[s] - current.get(s, Decimal(0))
            if delta:
                planned.append({"instrument_id": s, "quantity": str(abs(delta)), "side": "buy" if delta > 0 else "sell",
                                "prior_quantity": str(current.get(s, Decimal(0))), "target_quantity": str(targets[s]),
                                "weight": float(target_weights[s]), "reference_price": marks[s]["price"], "price_asof": marks[s]["as_of"],
                                "max_position_quantity": str(max(abs(targets[s]), abs(current.get(s, Decimal(0))), lot))})
        if not planned:
            return self._contract(None, at.isoformat(), state=State.UNAVAILABLE)
        panel = pd.DataFrame(dataset["panel_rows"])
        risk: dict[str, Any] = {"state": "UNAVAILABLE", "data": None}
        if "return_1" in panel:
            complete = panel.pivot(index="timestamp", columns="instrument_id", values="return_1").reindex(columns=symbols).dropna()
            if len(complete) >= 20:
                risk = {"state": "AVAILABLE", "data": portfolio_risk(complete.to_numpy(),
                    [float(target_weights[s]) for s in symbols], annualization=annualization)}
        body = {"schema_version": "1", "as_of": at.isoformat(), "run_id": source_id if source_key == "run_id" else None,
                "source_kind": source_key, "research_source_identity": source_id, "dataset_id": run["dataset_id"],
                "request": request, "model_ids": run["model_ids"], "score_as_of": latest.isoformat(),
                "source_classification": dataset["classification"],
                "rows": planned, "marks": marks, "risk": risk, "prior_positions": {s: str(q) for s, q in current.items()},
                "metrics": {"nav": str(nav), "cash": str(actual_cash), "modeled_transaction_cost": str(modeled_cost),
                            "turnover_nav": float(turnover),
                            "gross": float(sum(abs(v) for v in target_weights.values())), "net": float(sum(target_weights.values())),
                            "score_age_seconds": (at - latest).total_seconds(),
                            "worst_case_fill_gross": float(sum(worst.values())), "broker_connected": False},
                "execution_quote_policy": "CALLER_TICK_ROUND_HALF_EVEN_SIMULATED" if price_tick is not None else "EXACT_SOURCE_VALUE",
                "trading_approved": False, "classification": "SIMULATED_PLAN", "economically_validated": False}
        identity = _hash(body)
        document = {**body, "allocation_id": identity}
        contract = self._contract({**document, "title": "Constrained offline research allocation"}, at.isoformat(), identity=identity)
        write_immutable(self._path(identity + ".json"), dumps(document).encode())
        with sqlite3.connect(self._path("index.sqlite")) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("CREATE TABLE IF NOT EXISTS allocations (identity TEXT PRIMARY KEY, as_of TEXT NOT NULL)")
            connection.execute("INSERT OR IGNORE INTO allocations VALUES(?,?)", (identity, at.isoformat()))
        return contract

    @_exact
    def prepare(self, payload: dict[str, Any]) -> Contract:
        _fields(payload, {"allocation_id", "now"}, {"allocation_id", "now"})
        document = self.load(payload["allocation_id"])
        at = _stamp(payload["now"])
        if aware_timestamp(at) < aware_timestamp(document["as_of"]):
            raise ValueError("execution cannot precede allocation knowledge")
        path = self.workspace._path("execution-engine/oms.sqlite")
        with OfflineOMS(path) as oms:
            outbox = ExecutionOutbox(oms)
            oms.connection.execute("CREATE TABLE IF NOT EXISTS allocation_preparations (identity TEXT PRIMARY KEY, wire BLOB NOT NULL)")
            with oms._transaction():
                prior = oms.connection.execute("SELECT wire FROM allocation_preparations WHERE identity=?", (document["allocation_id"],)).fetchone()
                if prior:
                    return Contract(bytes(prior[0]))
                DurableTWAPScheduler(oms, outbox, create=False).ensure_not_before(at)
                if oms.kill_switch:
                    raise ValueError("offline kill switch is engaged")
                positions, opened = self._positions()
                if opened or {s: str(q) for s, q in positions.items()} != document["prior_positions"]:
                    raise ValueError("account state changed; construct a new allocation")
                count = oms.connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
                if count + len(document["rows"]) > 100:
                    raise ValueError("offline retained-order bound exceeded")
                clients = []
                for i, row in enumerate(document["rows"]):
                    client = "allocation:" + document["allocation_id"] + ":" + str(i)
                    px, qty = Decimal(row["reference_price"]), Decimal(row["quantity"])
                    outbox.admit_order(client, row["instrument_id"], row["side"], row["quantity"],
                        reference_price=row["reference_price"], price_asof=_stamp(row["price_asof"]),
                        max_notional=str(px * qty), max_position_quantity=row["max_position_quantity"], now=at)
                    clients.append(client)
                contract = self._contract({"title": "Prepared durable offline allocation", "allocation_id": document["allocation_id"],
                    "prepared_client_ids": clients, "rows": document["rows"], "broker_connected": False,
                    "trading_approved": False, "classification": "SIMULATED_INTENT"}, at.isoformat(), identity=document["allocation_id"])
                oms.connection.execute("INSERT INTO allocation_preparations VALUES(?,?)", (document["allocation_id"], contract.wire))
                return contract

    def section(self, section: str, as_of: Any, maximum: float | None) -> Contract | None:
        if section not in {"portfolio", "risk"}:
            return None
        at, maximum = _age(as_of, maximum)
        path = self._path("index.sqlite")
        if not path.is_file():
            return None
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
            choices = [(aware_timestamp(t), i) for i, t in connection.execute("SELECT identity,as_of FROM allocations") if aware_timestamp(t) <= at]
        if not choices:
            return None
        observed, identity = max(choices)
        document = self.load(identity)
        if aware_timestamp(document["as_of"]) != observed:
            raise ValueError("allocation index differs from preserved evidence")
        state = State.UNKNOWN if maximum is None else State.STALE if (at-observed).total_seconds() > maximum else State.AVAILABLE
        return self._contract({**document, "title": "Offline planned portfolio" if section == "portfolio" else "Offline planned risk"}, observed.isoformat(), state=state, identity=identity)
