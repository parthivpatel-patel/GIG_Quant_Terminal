"""Scoped multi-asset model risk on actual isolated offline-account positions.

An admitted research model supplies per-unit Greeks; its hypothetical positions
are never imported. Risk evidence and kill controls share the existing OMS
transaction. Neither a risk report nor its replay approves trading.
"""
from __future__ import annotations

import hashlib
import sqlite3
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from gig.artifacts import write_immutable
from gig.execution.oms import TERMINAL, OfflineOMS, _decimal
from gig.options import black_scholes
from gig.risk.analytics import historical_loss_risk
from gig.service.contracts import Contract, State
from gig.service.execution_workspace import _stamp
from gig.service.funded_workspace import FundedWorkspace
from gig.service.options_workspace import OptionsWorkspace
from gig.service.serialization import dumps, loads, public_error
from gig.service.workspace import TerminalWorkspace, _age, _fields, _number, _text
from gig.timestamps import aware_timestamp

VERSION = "funded-scoped-risk-v1"
LIMITS = {"max_gross_notional", "max_name_notional", "max_abs_delta_notional", "max_abs_gamma_dollar",
          "max_abs_vega_per_point", "max_abs_theta_per_day", "max_scenario_loss", "min_available_cash"}


def _hash(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def _seconds(later, earlier):
    return (aware_timestamp(later).value-aware_timestamp(earlier).value)/1e9


class RiskUnavailable(ValueError):
    pass


class FundedRiskWorkspace:
    def __init__(self, base_dir: Path, *, fault_hook=None):
        self.base_dir = Path(base_dir).absolute()
        self.workspace = TerminalWorkspace(base_dir)
        self.book = FundedWorkspace(base_dir)
        self.hook = fault_hook or (lambda _: None)

    def _path(self, name):
        return self.workspace._path("funded-account/risk/" + name)

    @staticmethod
    def _schema(oms):
        oms.connection.execute("CREATE TABLE IF NOT EXISTS funded_risk_policy(name TEXT PRIMARY KEY,wire TEXT NOT NULL)")
        oms.connection.execute("CREATE TABLE IF NOT EXISTS funded_risk_reports(report_id TEXT PRIMARY KEY,as_of TEXT NOT NULL)")

    @staticmethod
    def _request(payload):
        required = {"now", "marks", "options_run_ids", "max_age_seconds", "limits", "scenarios"}
        _fields(payload, required | {"liquidity", "max_participation_rate"}, required)
        request = loads(dumps(payload))
        request["now"] = _stamp(request["now"]).isoformat()
        _, maximum = _age(request["now"], request["max_age_seconds"])
        if maximum is None or maximum > 86400:
            raise ValueError("explicit risk freshness of at most one day required")
        request["max_age_seconds"] = maximum
        _fields(request["limits"], LIMITS, LIMITS)
        for key in LIMITS:
            value = _number(request["limits"][key])
            if not 0 <= value <= 1e15:
                raise ValueError("finite bounded nonnegative USD risk caps required")
            request["limits"][key] = value
        ids = request["options_run_ids"]
        if not isinstance(ids, list) or len(ids) > 8:
            raise ValueError("at most eight saved option-model runs required")
        if any(not isinstance(i, str) or len(i) != 64 or any(c not in "0123456789abcdef" for c in i) for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("distinct canonical model identities required")
        request["options_run_ids"] = sorted(ids)
        marks = request["marks"]
        if not isinstance(marks, dict) or len(marks) > 64:
            raise ValueError("bounded explicit risk marks required")
        for symbol, mark in marks.items():
            _text(symbol, bound=128)
            _fields(mark, {"price", "as_of"}, {"price", "as_of"})
            mark["price"] = str(_decimal(mark["price"], positive=False))
            mark["as_of"] = aware_timestamp(mark["as_of"]).isoformat()
            if aware_timestamp(mark["as_of"]) > aware_timestamp(request["now"]):
                raise ValueError("future mark is unavailable")
        scenarios = request["scenarios"]
        if not isinstance(scenarios, list) or not 1 <= len(scenarios) <= 25:
            raise ValueError("one to 25 explicit bounded scenarios required")
        names = set()
        for scenario in scenarios:
            _fields(scenario, {"name", "spot_shocks", "vol_shocks", "elapsed_days"}, {"name", "spot_shocks", "vol_shocks", "elapsed_days"})
            _text(scenario["name"], bound=128)
            if scenario["name"] in names:
                raise ValueError("duplicate scenario identity")
            names.add(scenario["name"])
            elapsed = _number(scenario["elapsed_days"])
            if not 0 <= elapsed <= 366:
                raise ValueError("scenario time shock outside supported domain")
            for key in ("spot_shocks", "vol_shocks"):
                if not isinstance(scenario[key], dict) or len(scenario[key]) > 64:
                    raise ValueError("bounded explicit per-underlying shocks required")
                for symbol, value in scenario[key].items():
                    _text(symbol, bound=128)
                    value = _number(value)
                    if not (-1 < value <= 10 if key == "spot_shocks" else -5 <= value <= 5):
                        raise ValueError("scenario shock outside supported domain")
        liquidity = request.setdefault("liquidity", {})
        if not isinstance(liquidity, dict) or len(liquidity) > 64:
            raise ValueError("bounded liquidity evidence required")
        for symbol, item in liquidity.items():
            _text(symbol, bound=128)
            _fields(item, {"executed_volume", "as_of", "source_identity", "kind"}, {"executed_volume", "as_of", "source_identity", "kind"})
            if item["kind"] != "EXECUTED_PRINTS" or not 0 <= _number(item["executed_volume"]) <= 1e12:
                raise ValueError("executed-print volume cannot be inferred from displayed depth")
            item["as_of"] = aware_timestamp(item["as_of"]).isoformat()
            _text(item["source_identity"], bound=128)
        if "max_participation_rate" in request and not 0 < _number(request["max_participation_rate"]) <= 1:
            raise ValueError("participation fraction must lie in (0,1]")
        return request

    def _units(self, snapshot, request):
        now, maximum = aware_timestamp(request["now"]), request["max_age_seconds"]
        models = {}
        for identity in request["options_run_ids"]:
            artifact = OptionsWorkspace(self.base_dir).load(identity)
            if aware_timestamp(artifact["as_of"]) > now or _seconds(now, artifact["observation_at"]) > maximum:
                raise RiskUnavailable("option-model knowledge or freshness unavailable")
            selected = OptionsWorkspace(self.base_dir)._select({**artifact["request"], "as_of": now.isoformat()})
            if selected is None or selected["snapshot_id"] != artifact["source_snapshot_id"]:
                raise RiskUnavailable("option observation revised; new research required")
            for quote in artifact["rows"]:
                if quote["contract_id"] in models:
                    raise ValueError("ambiguous duplicate option-model identity")
                models[quote["contract_id"]] = (artifact, quote)
        units = {}
        for symbol, terms in snapshot["instruments"].items():
            mark = request["marks"].get(symbol)
            if mark is None:
                continue
            age = _seconds(now, mark["as_of"])
            if age < 0 or age > maximum:
                continue
            multiplier, price = terms["multiplier"], float(mark["price"])
            if terms["kind"] == "EQUITY":
                if price <= 0:
                    raise RiskUnavailable("equity price unavailable")
                underlying, spot = symbol, price
                values = {"delta": 1., "gamma": 0., "vega": 0., "theta": 0., "price": price}
                option = None
            else:
                if symbol not in models:
                    continue
                artifact, quote = models[symbol]
                if (quote["state"] != "AVAILABLE" or quote["right"] != terms["right"]
                        or Decimal(str(quote["strike"])) != Decimal(terms["strike"])
                        or quote["multiplier"] != multiplier or quote["style"] != terms["style"]
                        or aware_timestamp(quote["expiry"]) != aware_timestamp(terms["expiry"])
                        or artifact["underlying_id"] != terms["underlying_id"]):
                    raise RiskUnavailable("actual contract and saved model terms disagree")
                underlying, spot = terms["underlying_id"], artifact["spot"]
                spot_mark = request["marks"].get(underlying)
                if (spot_mark is None or Decimal(spot_mark["price"]) != Decimal(str(spot))
                        or aware_timestamp(spot_mark["as_of"]) != aware_timestamp(artifact["observation_at"])):
                    raise RiskUnavailable("underlying spot/clock must match admitted option evidence")
                ttm = (aware_timestamp(terms["expiry"])-now).total_seconds()/(365*86400)
                if ttm <= 0:
                    raise RiskUnavailable("expired Greeks unsupported before settlement")
                option = {"spot": spot, "strike": float(terms["strike"]), "rate": artifact["request"]["rate"],
                          "div": artifact["request"]["dividend_yield"], "vol": quote["implied_volatility"],
                          "ttm": ttm, "call": terms["right"] == "CALL"}
                values = black_scholes(**option)
            changes = {}
            for scenario in request["scenarios"]:
                if underlying not in scenario["spot_shocks"] or underlying not in scenario["vol_shocks"]:
                    raise RiskUnavailable("every modeled underlying needs explicit stress coordinates")
                shock = scenario["spot_shocks"][underlying]
                if option is None:
                    change = price*shock
                else:
                    changed = {**option, "spot": spot*(1+shock), "vol": option["vol"]+scenario["vol_shocks"][underlying],
                               "ttm": option["ttm"]-scenario["elapsed_days"]/365}
                    if changed["ttm"] <= 0 or changed["vol"] <= 0:
                        raise RiskUnavailable("scenario reaches unsupported expiry or volatility")
                    change = black_scholes(**changed)["price"]-values["price"]
                changes[scenario["name"]] = change*multiplier
            units[symbol] = {"underlying_id": underlying, "mark_value": price*multiplier,
                "delta_notional": values["delta"]*spot*multiplier, "gamma_dollar": values["gamma"]*spot*spot*multiplier,
                "vega_per_point": values["vega"]*multiplier, "theta_per_day": values["theta"]*multiplier,
                "model_mark_basis_difference": (values["price"]-price)*multiplier,
                "scenario_pnl": changes, "multiplier": multiplier}
        return units

    def _historical(self, oms, snapshot, request):
        now = aware_timestamp(request["now"])
        day = 86400*1_000_000_000
        samples = snapshot.get("risk_nav_samples")
        if samples is None:
            selected = {}
            entries = oms.connection.execute("SELECT snapshot_id,as_of FROM funded_snapshots ORDER BY rowid LIMIT 1001").fetchall()
            if len(entries) > 1000:
                raise ValueError("historical funded NAV read bound exceeded")
            for identity, stamp in entries:
                at = aware_timestamp(stamp)
                if at > now or at.value % day != now.value % day:
                    continue
                source = self.book.load(identity)
                account = source["account"]
                if (account["mark_state"] != "AVAILABLE" or account["funding_state"] != "AVAILABLE"
                        or account["nav"] is None or Decimal(account["nav"]) <= 0 or account["pending_expiries"]
                        or any(Decimal(r["quantity"]) and aware_timestamp(r["mark_asof"]) != at for r in account["rows"])):
                    continue
                selected[at.value] = {"snapshot_id": identity, "as_of": stamp, "nav": account["nav"], "initial_cash": account["initial_cash"]}
            samples = [selected[k] for k in sorted(selected)][-21:]
            snapshot["risk_nav_samples"] = samples
        if len(samples) < 21 or aware_timestamp(samples[-1]["as_of"]) != now:
            return {"state": "UNAVAILABLE", "var": None, "es": None, "reason": "NEED_21_CURRENT_COMPARABLE_DAILY_FUNDED_NAV_MARKS"}
        if (len({s["initial_cash"] for s in samples}) != 1
                or any(aware_timestamp(b["as_of"]).value-aware_timestamp(a["as_of"]).value != day for a, b in zip(samples, samples[1:], strict=False))):
            return {"state": "UNAVAILABLE", "var": None, "es": None, "reason": "IRREGULAR_HISTORY_OR_DIFFERENT_CAPITAL_BASE"}
        returns = [float(Fraction(Decimal(b["nav"])-Decimal(a["nav"]))/Fraction(Decimal(a["nav"])))
                   for a, b in zip(samples, samples[1:], strict=False)]
        risk = historical_loss_risk(returns, .99)
        return {"state": "AVAILABLE", **risk, "es": risk["expected_shortfall"],
                "period": "EXACT_24_HOUR_SIMULATED_ACCOUNT_MARKS", "source_snapshot_ids": [s["snapshot_id"] for s in samples],
                "scope": "FUNDED_SIMULATION_NAV_RETURNS_NO_UNREPORTED_EXTERNAL_CASH_FLOWS",
                "economic_validation": False}

    def _evaluate(self, oms, request, candidate=None, *, snapshot_override=None):
        at = _stamp(request["now"])
        clock = self.book._config(oms, "clock") if oms is not None else None
        if clock is not None and aware_timestamp(clock) > aware_timestamp(at):
            raise RiskUnavailable("risk clock precedes current account")
        if oms is not None:
            for table, field in (("events", "timestamp"), ("transport_events", "received_at")):
                if oms.connection.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone():
                    values = oms.connection.execute(f"SELECT {field} FROM {table} LIMIT 100001").fetchall()
                    if len(values) > 100000 or any(aware_timestamp(v[0]) > aware_timestamp(at) for v in values):
                        raise RiskUnavailable("risk cannot precede execution knowledge")
        snapshot = snapshot_override if snapshot_override is not None else self.book._snapshot(oms, at, request["marks"], request["max_age_seconds"])
        account, units = snapshot["account"], self._units(snapshot, request)
        quantities = {key: float(value) for key, value in account["effective_positions"].items() if Decimal(value)}
        orders = [o for o in snapshot["orders"] if o["state"] not in TERMINAL]
        existing_clients = {o["client_id"] for o in snapshot["orders"]}
        if candidate is not None and candidate["client_id"] not in existing_clients:
            orders.append({"symbol": candidate["instrument_id"], "side": candidate["side"],
                           "quantity": candidate["quantity"], "filled": "0", "funded_limit_price": candidate["limit_price"],
                           "fee_budget": candidate["fee_budget"], "client_id": candidate["client_id"]})
        changes: dict[str, list[float]] = {}
        for order in orders:
            remaining = float(Decimal(order["quantity"])-Decimal(order["filled"]))
            changes.setdefault(order["symbol"], []).append(remaining*(1 if order["side"] == "buy" else -1))
        required = sorted(set(quantities) | set(changes))
        if set(required) - units.keys() or account["pending_expiries"] or account["mark_state"] != "AVAILABLE":
            raise RiskUnavailable("position or order risk exposure unavailable")
        if oms is not None:
            stored = self.book._config(oms, "marks", {"marks": {}})["marks"]
            for symbol in quantities:
                if symbol not in stored or stored[symbol] != request["marks"][symbol]:
                    raise RiskUnavailable("configured risk marks differ from current funded marks")
        worst_quantity = {s: max(abs(quantities.get(s, 0.)+sum(max(0., v) for v in changes.get(s, []))),
                                 abs(quantities.get(s, 0.)+sum(min(0., v) for v in changes.get(s, [])))) for s in required}
        measures = {"max_gross_notional": sum(worst_quantity[s]*abs(units[s]["mark_value"]) for s in required),
                    "max_name_notional": max((worst_quantity[s]*abs(units[s]["mark_value"]) for s in required), default=0.)}
        by_underlying = {}
        for symbol in required:
            underlying = units[symbol]["underlying_id"]
            by_underlying[underlying] = by_underlying.get(underlying, 0.)+worst_quantity[symbol]*abs(units[symbol]["mark_value"])
        measures["max_name_notional"] = max(by_underlying.values(), default=0.)
        for metric, field in (("max_abs_delta_notional", "delta_notional"), ("max_abs_gamma_dollar", "gamma_dollar"),
                              ("max_abs_vega_per_point", "vega_per_point"), ("max_abs_theta_per_day", "theta_per_day")):
            # Conservative per-leg absolute budgets prevent asynchronous fills or
            # cross-underlying cancellation from hiding exposure.
            measures[metric] = sum(worst_quantity[s]*abs(units[s][field]) for s in required)
        scenarios = []
        entry_cost = 0.
        for order in orders:
            symbol = order["symbol"]
            remaining = float(Decimal(order["quantity"])-Decimal(order["filled"]))
            direction = 1 if order["side"] == "buy" else -1
            paid = sum((Decimal(f["fee"]) for f in snapshot["executions"] if f["client_id"] == order["client_id"]), Decimal(0))
            entry_cost += max(0., direction*(float(order["funded_limit_price"])*units[symbol]["multiplier"]-units[symbol]["mark_value"]))*remaining
            entry_cost += max(0., float(Decimal(order["fee_budget"])-paid))
        for scenario in request["scenarios"]:
            name = scenario["name"]
            current = sum(quantities.get(s, 0.)*units[s]["scenario_pnl"][name] for s in required)
            worst = current+sum(min(0., change*units[s]["scenario_pnl"][name]) for s in required for change in changes.get(s, []))-entry_cost
            scenarios.append({"name": name, "current_model_pnl": current, "worst_async_model_pnl": worst})
        measures["max_scenario_loss"] = max((max(0., -r["worst_async_model_pnl"]) for r in scenarios), default=0.)
        violations = [key for key, value in measures.items() if value > request["limits"][key]]
        available = Decimal(account["available_cash"])
        if candidate is not None and candidate["client_id"] not in existing_clients:
            amount = Decimal(candidate["quantity"])*Decimal(candidate["limit_price"])*snapshot["instruments"][candidate["instrument_id"]]["multiplier"] if candidate["side"] == "buy" else Decimal(0)
            available -= amount+Decimal(candidate["fee_budget"])
        if available < Decimal(str(request["limits"]["min_available_cash"])) or account["funding_state"] != "AVAILABLE":
            violations.append("min_available_cash_or_funding")
        liquidity_rows = []
        for symbol in sorted(required):
            item = request["liquidity"].get(symbol)
            known = item is not None and 0 <= _seconds(request["now"], item["as_of"]) <= request["max_age_seconds"]
            ratio = worst_quantity[symbol]/item["executed_volume"] if known and item["executed_volume"] > 0 else None
            liquidity_rows.append({"instrument_id": symbol, "participation": ratio, "state": "AVAILABLE" if ratio is not None else "UNKNOWN"})
            if "max_participation_rate" in request and (ratio is None or ratio > request["max_participation_rate"]):
                violations.append("liquidity:"+symbol)
        rows = [{"instrument_id": s, "actual_quantity": quantities.get(s, 0.), "worst_working_quantity": worst_quantity[s], **units[s]} for s in sorted(required)]
        historical = self._historical(oms, snapshot, request)
        return {"schema_version": VERSION, "as_of": request["now"], "request": request, "account_source_identity": snapshot["source_identity"],
            "account_cash": account["cash"], "account_nav": account["nav"], "available_cash": account["available_cash"],
            "account_snapshot": snapshot, "underlying_gross_notional": by_underlying,
            "rows": rows, "units": units, "scenarios": scenarios, "metrics": measures, "violations": violations,
            "state": "ERROR" if violations else "AVAILABLE", "liquidity": liquidity_rows,
            "historical_var_es": historical,
            "risk_scope": "CONSERVATIVE_LONG_ONLY_MODEL_BUDGETS_AND_ASYNCHRONOUS_ORDERS",
            "liquidity_evidence_state": "CALLER_DECLARED_EXECUTED_PRINTS_NOT_EXECUTABLE_LIQUIDITY",
            "model_assumptions": "BSM_FIXED_ADMITTED_IV_CONTINUOUS_RATES_AND_DIVIDENDS; OPTION_MARK_BASIS_DIFFERENCES_DISCLOSED",
            "trading_approved": False, "fully_funded_certified": False, "economically_validated": False}

    @staticmethod
    def _missing():
        return Contract.admit(State.UNAVAILABLE, None, source="funded-scoped-risk", classification="UNAVAILABLE",
                              scope="workspace.funded-risk", error=public_error("DATA_UNAVAILABLE"))

    @staticmethod
    def _contract(document, state=None):
        state = State(state or document["state"])
        return Contract.admit(state, {"title": "Actual offline-account scoped model risk", **document} if state in {State.AVAILABLE, State.STALE} else None,
            source="funded-scoped-risk", classification="MODEL_ESTIMATE", as_of=document["as_of"],
            scope="workspace.funded-risk", source_identity=document["account_source_identity"], model_identity=document["report_id"],
            model_version=VERSION, evidence_state="ACTUAL_SIMULATED_POSITIONS_WITH_RESEARCH_MODELS", readiness="RESEARCH_ONLY",
            error=public_error("INVALID_REQUEST") if state == State.ERROR else None)

    def _publish(self, oms, body):
        identity = _hash(body)
        document = {**body, "report_id": identity}
        result = self._contract(document)
        write_immutable(self._path(identity+".json"), dumps(body).encode())
        if oms.connection.execute("SELECT COUNT(*) FROM funded_risk_reports").fetchone()[0] >= 1000 and not oms.connection.execute("SELECT 1 FROM funded_risk_reports WHERE report_id=?", (identity,)).fetchone():
            raise ValueError("risk registry retention bound exceeded")
        oms.connection.execute("INSERT OR IGNORE INTO funded_risk_reports VALUES(?,?)", (identity, body["as_of"]))
        self.hook("risk_indexed")
        if body["state"] == "ERROR" and not oms.kill_switch:
            oms.set_kill_switch(True, now=_stamp(body["as_of"]))
        self.hook("risk_controls_applied")
        return result

    def run(self, payload):
        request = self._request(payload)
        path = self.book._path("oms.sqlite")
        if not path.is_file():
            return self._missing()
        with OfflineOMS(path) as oms, oms._transaction():
            self._schema(oms)
            try:
                body = self._evaluate(oms, request)
            except (RiskUnavailable, FileNotFoundError):
                return self._missing()
            result = self._publish(oms, body)
            oms.connection.execute("INSERT INTO funded_risk_policy VALUES('active',?) ON CONFLICT(name) DO UPDATE SET wire=excluded.wire", (dumps(request),))
            return result

    def load(self, report_id):
        if not isinstance(report_id, str) or len(report_id) != 64 or any(c not in "0123456789abcdef" for c in report_id):
            raise ValueError("canonical risk identity required")
        wire = self._path(report_id+".json").read_bytes()
        body = loads(wire)
        if hashlib.sha256(wire).hexdigest() != report_id or body.get("schema_version") != VERSION:
            raise ValueError("risk evidence integrity failure")
        path = self.book._path("oms.sqlite")
        with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True) as db:
            entry = db.execute("SELECT as_of FROM funded_risk_reports WHERE report_id=?", (report_id,)).fetchone()
        if entry is None or entry[0] != body["as_of"]:
            raise ValueError("risk index differs from immutable evidence")
        for identity in body["request"]["options_run_ids"]:
            OptionsWorkspace(self.base_dir).load(identity)
        for sample in body.get("account_snapshot", {}).get("risk_nav_samples", []):
            source = self.book.load(sample["snapshot_id"])
            if (source["as_of"], source["account"]["nav"], source["account"]["initial_cash"]) != (sample["as_of"], sample["nav"], sample["initial_cash"]):
                raise ValueError("risk historical NAV evidence differs")
        if "account_snapshot" in body and self._evaluate(None, body["request"], snapshot_override=body["account_snapshot"]) != body:
            raise ValueError("saved risk calculation differs from immutable replay")
        return {**body, "report_id": report_id}

    def replay(self, payload):
        _fields(payload, {"report_id"}, {"report_id"})
        try:
            return self._contract(self.load(payload["report_id"]))
        except FileNotFoundError:
            return self._missing()

    def section(self, section, as_of, max_age_seconds):
        if section not in {"risk", "portfolio", "options", "execution"}:
            return None
        cutoff, maximum = _age(as_of, max_age_seconds)
        path = self.book._path("oms.sqlite")
        if not path.is_file():
            return None
        with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True) as db:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name='funded_risk_reports'").fetchone():
                return None
            entries = db.execute("SELECT report_id,as_of FROM funded_risk_reports ORDER BY rowid LIMIT 1001").fetchall()
        if len(entries) > 1000:
            raise ValueError("risk read bound exceeded")
        choices = [(aware_timestamp(at), n, identity) for n, (identity, at) in enumerate(entries) if aware_timestamp(at) <= cutoff]
        if not choices:
            return None
        observed, _, identity = max(choices)
        document = self.load(identity)
        state = State(document["state"])
        if state != State.ERROR:
            state = State.UNKNOWN if maximum is None else State.STALE if _seconds(cutoff, observed) > maximum else State.AVAILABLE
        return self._contract(document, state)


def _policy(oms):
    if not oms.connection.execute("SELECT 1 FROM sqlite_master WHERE name='funded_risk_policy'").fetchone():
        return None
    row = oms.connection.execute("SELECT wire FROM funded_risk_policy WHERE name='active'").fetchone()
    return loads(row[0]) if row else None


def guard_funded_order(base_dir, oms, candidate, now):
    policy = _policy(oms)
    if policy is None:
        return
    workspace = FundedRiskWorkspace(base_dir)
    request = {**policy, "now": now.isoformat()}
    body = workspace._evaluate(oms, request, candidate)
    if body["violations"]:
        raise ValueError("configured model-risk budgets reject this proposed order")


def reconcile_funded_risk(base_dir, oms, now):
    policy = _policy(oms)
    if policy is None:
        return
    workspace = FundedRiskWorkspace(base_dir)
    request = {**policy, "now": now.isoformat()}
    try:
        body = workspace._evaluate(oms, request)
    except (ValueError, FileNotFoundError):
        body = {"schema_version": VERSION, "as_of": request["now"], "request": request,
                "account_source_identity": "UNAVAILABLE", "state": "ERROR", "rows": [],
                "violations": ["POST_EVENT_MODEL_RISK_UNAVAILABLE"], "trading_approved": False}
    workspace._publish(oms, body)
