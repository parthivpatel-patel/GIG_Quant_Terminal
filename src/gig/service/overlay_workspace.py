"""Read-only hypothetical option overlays on hash-verified simulated equity books.

The sum of equity NAV and option model value is an unfinanced modeled value,
never a funded account NAV, executable portfolio, or permission to trade.
"""
from __future__ import annotations

import hashlib
import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Any

from gig.artifacts import write_immutable
from gig.execution.oms import _exact
from gig.portfolio.ledger import signed_decimal
from gig.service.contracts import Contract, State
from gig.service.options_workspace import OptionsWorkspace
from gig.service.portfolio_workspace import PortfolioWorkspace
from gig.service.serialization import dumps, loads, public_error
from gig.service.workspace import TerminalWorkspace, _age, _fields, _number
from gig.timestamps import aware_timestamp

VERSION = "simulated-equity-hypothetical-options-overlay-v1"
LIMITS = {"max_abs_delta_notional", "max_abs_gamma_dollar", "max_abs_vega_per_point", "max_abs_scenario_loss"}
SECTIONS = {"portfolio", "risk", "research", "options"}


def _hash(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def _age_seconds(later: Any, earlier: Any) -> float:
    return (aware_timestamp(later).value-aware_timestamp(earlier).value)/1e9


class _Unavailable(ValueError):
    def __init__(self, evidence: str, state: State = State.UNAVAILABLE):
        self.evidence, self.state = evidence, state


class OptionsOverlayWorkspace:
    def __init__(self, base_dir: Path):
        self.workspace = TerminalWorkspace(base_dir)
        self.base_dir = self.workspace.base_dir

    def _path(self, name: str) -> Path:
        return self.workspace._path("options-overlays/" + name)

    @staticmethod
    def _request(payload: dict[str, Any]) -> dict[str, Any]:
        fields = {"checkpoint_id", "options_run_id", "now", "max_age_seconds", "limits"}
        _fields(payload, fields, fields)
        request = loads(dumps(payload))
        request["checkpoint_id"] = PortfolioWorkspace._identity(request["checkpoint_id"])
        request["options_run_id"] = PortfolioWorkspace._identity(request["options_run_id"])
        now, maximum = _age(request["now"], request["max_age_seconds"])
        if maximum is None:
            raise ValueError("overlay requires an explicit freshness policy")
        request["now"], request["max_age_seconds"] = now.isoformat(), maximum
        _fields(request["limits"], LIMITS, LIMITS)
        for key in LIMITS:
            value = _number(request["limits"][key])
            if not 0 <= value <= 1e15:
                raise ValueError("overlay USD limits must be finite and bounded nonnegative values")
            request["limits"][key] = value
        return request

    @staticmethod
    def _missing(evidence: str = "MISSING_PARENT_EVIDENCE", state: State = State.UNAVAILABLE) -> Contract:
        return Contract.admit(state, None, source="simulated-equity-hypothetical-options-overlay",
            classification="UNKNOWN" if state == State.UNKNOWN else "UNAVAILABLE", scope="workspace.options-overlay",
            evidence_state=evidence, readiness="RESEARCH_ONLY",
            error=public_error("DATA_UNAVAILABLE") if state == State.UNAVAILABLE else None)

    def _parents(self, request: dict[str, Any], *, current: bool) -> tuple[dict[str, Any], dict[str, Any]]:
        ledger = PortfolioWorkspace(self.base_dir)
        history = ledger._history()
        matches = [item for item in history if item["checkpoint_id"] == request["checkpoint_id"]]
        if not matches:
            raise _Unavailable("PORTFOLIO_CHECKPOINT_NOT_PUBLISHED")
        checkpoint = matches[0]
        options = OptionsWorkspace(self.base_dir).load(request["options_run_id"])
        now = aware_timestamp(request["now"])
        maximum = request["max_age_seconds"]
        if (aware_timestamp(checkpoint["as_of"]) > now or aware_timestamp(options["as_of"]) > now
                or aware_timestamp(options["knowledge_at"]) > now or aware_timestamp(options["acquired_at"]) > now):
            raise _Unavailable("PARENT_EVIDENCE_LIES_AFTER_CUTOFF", State.UNKNOWN)
        if checkpoint["state"] != State.AVAILABLE:
            raise _Unavailable("PORTFOLIO_CHECKPOINT_NOT_AVAILABLE")
        if checkpoint["account"]["currency"] != "USD":
            raise _Unavailable("UNSUPPORTED_ACCOUNT_CURRENCY")
        if (_age_seconds(now, checkpoint["observed_at"]) > maximum
                or _age_seconds(now, options["observation_at"]) > maximum):
            raise _Unavailable("STALE_PARENT_OBSERVATIONS")
        if OptionsWorkspace._contract(options).to_dict()["state"] != State.AVAILABLE:
            raise _Unavailable("OPTIONS_RESEARCH_NOT_AVAILABLE")
        if options["portfolio_exposure"]["state"] != "AVAILABLE" or options["scenarios"]["state"] != "AVAILABLE":
            raise _Unavailable("OPTION_POSITION_OR_SCENARIO_EXPOSURE_UNKNOWN", State.UNKNOWN)
        if options["portfolio_exposure"]["currency"] != "USD":
            raise _Unavailable("UNSUPPORTED_OPTIONS_CURRENCY")
        if current:
            candidates = [item for item in history if aware_timestamp(item["as_of"]) <= now]
            latest = max(enumerate(candidates), key=lambda pair: (aware_timestamp(pair[1]["as_of"]), pair[0]))[1]
            if latest["checkpoint_id"] != request["checkpoint_id"]:
                raise _Unavailable("OLDER_PORTFOLIO_CHECKPOINT_SELECTED")
            source = ledger._execution_snapshot()
            if source is None:
                raise _Unavailable("CURRENT_OMS_UNAVAILABLE")
            fills, positions, known = source
            if ((known is not None and known > now) or any(aware_timestamp(fill["timestamp"]) > now for fill in fills)):
                raise _Unavailable("CURRENT_EXECUTION_EVIDENCE_AFTER_CUTOFF", State.UNKNOWN)
            if _hash(fills) != checkpoint["executions_sha256"] or fills != checkpoint["execution_evidence"]:
                raise _Unavailable("CURRENT_EXECUTIONS_DIFFER_FROM_CHECKPOINT")
            actual = {symbol: signed_decimal(value) for symbol, value in positions.items()}
            saved = {symbol: signed_decimal(value) for symbol, value in checkpoint["oms_positions"].items()}
            if actual != saved:
                raise _Unavailable("CURRENT_OMS_POSITIONS_DIFFER_FROM_CHECKPOINT")
            selected = OptionsWorkspace(self.base_dir)._select({**options["request"], "as_of": now.isoformat()})
            if selected is None or selected["snapshot_id"] != options["source_snapshot_id"]:
                raise _Unavailable("OPTIONS_SOURCE_REVISION_CHANGED_AT_CUTOFF")
        held = [row for row in checkpoint["account"]["rows"] if signed_decimal(row["quantity"])]
        if any(row["instrument_id"] != options["underlying_id"] for row in held):
            raise _Unavailable("OTHER_EQUITY_UNDERLYINGS_NOT_MODELED")
        for row in held:
            mark = checkpoint["request"]["marks"].get(row["instrument_id"])
            if (mark is None or signed_decimal(mark["price"]) != Decimal(str(options["spot"]))
                    or aware_timestamp(mark["as_of"]) != aware_timestamp(options["observation_at"])):
                raise _Unavailable("EQUITY_AND_OPTIONS_MARKS_ARE_NOT_SYNCHRONOUS")
        return checkpoint, options

    @staticmethod
    @_exact
    def _compute(checkpoint: dict[str, Any], options: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        account = checkpoint["account"]
        exposure = options["portfolio_exposure"]
        held = [row for row in account["rows"] if signed_decimal(row["quantity"])]
        quantity = sum((signed_decimal(row["quantity"]) for row in held), Decimal(0))
        spot = Decimal(str(options["spot"]))
        equity_nav, cash = Decimal(account["nav"]), Decimal(account["cash"])
        option_value = Decimal(str(exposure["price"]))
        augmented = equity_nav + option_value
        delta_units = quantity+Decimal(str(exposure["delta"]))
        delta_notional = delta_units*spot
        gamma_dollar = Decimal(str(exposure["gamma"]))*spot*spot
        vega = Decimal(str(exposure["vega"]))
        rows = []
        for scenario in options["scenarios"]["scenarios"]:
            equity_pnl = quantity*spot*Decimal(str(scenario["spot_shock"]))
            combined = equity_pnl+Decimal(str(scenario["model_pnl"]))
            rows.append({"spot_shock": scenario["spot_shock"], "volatility_shock": scenario["volatility_shock"],
                "equity_pnl": str(equity_pnl), "option_model_pnl": scenario["model_pnl"],
                "combined_model_pnl": str(combined), "combined_model_value": str(augmented+combined),
                "classification": "MODEL_ESTIMATE"})
        if not rows:
            raise ValueError("available overlay requires explicit option scenario evidence")
        worst_loss = max((max(Decimal(0), -Decimal(row["combined_model_pnl"])) for row in rows), default=Decimal(0))
        measures = {"max_abs_delta_notional": abs(delta_notional), "max_abs_gamma_dollar": abs(gamma_dollar),
            "max_abs_vega_per_point": abs(vega), "max_abs_scenario_loss": worst_loss}
        violations = [key for key in sorted(LIMITS) if measures[key] > Decimal(str(request["limits"][key]))]
        observed = min(aware_timestamp(checkpoint["observed_at"]), aware_timestamp(options["observation_at"]))
        return {"schema_version": VERSION, "request": request, "as_of": request["now"], "observed_at": observed.isoformat(),
            "checkpoint_id": checkpoint["checkpoint_id"], "options_run_id": options["run_id"],
            "options_snapshot_id": options["source_snapshot_id"], "executions_sha256": checkpoint["executions_sha256"],
            "underlying_id": options["underlying_id"], "currency": "USD", "spot": str(spot),
            "equity_quantity": str(quantity), "equity_cash": str(cash), "equity_nav": str(equity_nav),
            "option_model_value": str(option_value), "modeled_overlay_value": str(augmented),
            "metrics": {"equity_nav": str(equity_nav), "option_model_value": str(option_value),
                "modeled_overlay_value": str(augmented), "delta_notional": str(delta_notional),
                "gamma_dollar": str(gamma_dollar), "vega_per_point": str(vega), "worst_scenario_loss": str(worst_loss)},
            "greeks": {"delta_units": str(delta_units), "delta_notional": str(delta_notional),
                "gamma_dollar": str(gamma_dollar), "vega_per_point": str(vega)}, "rows": rows,
            "limits": request["limits"], "violations": violations, "state": "ERROR" if violations else "AVAILABLE",
            "overlay_financing": {"state": "UNKNOWN", "cash_adjustment": None, "financed_nav": None,
                "definition": "Hypothetical option positions are not financed or executed. Equity cash is unchanged; modeled equity NAV plus option value is not a funded account NAV."},
            "units": {"equity_cash": "USD exact simulated ledger value", "equity_nav": "USD exact simulated ledger value",
                "delta_units": "underlying units", "delta_notional": "USD; (equity quantity + option model delta)*spot",
                "gamma_dollar": "USD; option model gamma*spot^2 (one half applies only to a quadratic return-scenario approximation)",
                "vega_per_point": "USD/one absolute volatility percentage point", "combined_model_pnl": "USD full option BSM repricing plus equity spot-shock P&L"},
            "input_classifications": {"equity_account": "SIMULATED", "equity_marks": checkpoint["request"].get("input_classification", "UNKNOWN"),
                "option_quotes": options["input_classification"], "option_positions": "HYPOTHETICAL", "combined_values": "MODEL_ESTIMATE"},
            "option_model_provenance": options["implementation_provenance"],
            "source_identity": _hash({"checkpoint_id": checkpoint["checkpoint_id"], "options_run_id": options["run_id"]}),
            "scenario_scope": "SINGLE_UNDERLYING_EQUITY_AND_HYPOTHETICAL_EUROPEAN_OPTIONS_ONLY",
            "risk_policy_scope": "HYPOTHETICAL_RESEARCH_ONLY_NO_KILL_SWITCH_OR_ORDER_APPROVAL",
            "historical_tail_risk": {"state": "UNAVAILABLE", "var": None, "es": None, "definition": "A scenario grid is not a probability distribution or empirical historical VaR/ES."},
            "outstanding_order_exposure": {"state": "UNKNOWN", "value": None,
                "definition": "Only reconciled filled equity positions are modeled; outstanding orders and future fills are outside this hypothetical snapshot."},
            "replay_scope": "IMMUTABLE_SOURCE_CHECKPOINTS_NOT_CURRENT_ACCOUNT_REAPPROVAL",
            "trading_approved": False, "economically_validated": False, "broker_observed": False,
            "notes": ["No cash debit, option fill, funded NAV, option position reconciliation, or broker position is inferred.",
                "Option valuations and Greeks retain numerical model precision; equity quantities and cash retain exact Decimal accounting.",
                "Equity financing, dividends, stock borrow, option transaction costs and liquidity are outside these immediate spot/vol shocks.",
                "All nonzero equity positions must have the modeled underlying; unmodeled assets are rejected rather than assigned zero risk."]}

    @staticmethod
    def _contract(document: dict[str, Any], section: str = "risk", state: State | None = None) -> Contract:
        state = State(state or document["state"])
        return Contract.admit(state,
            {"title": "Simulated equity and hypothetical option overlay", **document}
            if state in {State.AVAILABLE, State.STALE} else None,
            source="simulated-equity-hypothetical-options-overlay", classification="MODEL_ESTIMATE",
            as_of=document["as_of"], scope=f"workspace.options-overlay.{section}",
            source_identity=document["source_identity"], model_identity=document["overlay_id"], model_version=VERSION,
            instrument_identity=document["underlying_id"], evidence_state="HASH_VERIFIED_UNFINANCED_HYPOTHETICAL_OVERLAY",
            readiness="RESEARCH_ONLY", error=public_error("INVALID_REQUEST") if state == State.ERROR else None)

    def run(self, payload: dict[str, Any]) -> Contract:
        request = self._request(payload)
        try:
            checkpoint, options = self._parents(request, current=True)
        except FileNotFoundError:
            return self._missing()
        except _Unavailable as missing:
            return self._missing(missing.evidence, missing.state)
        document = self._compute(checkpoint, options, request)
        identity = _hash(document)
        document["overlay_id"] = identity
        result = self._contract(document)
        index = self._path("overlays.sqlite")
        index.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(index) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("CREATE TABLE IF NOT EXISTS overlays(overlay_id TEXT PRIMARY KEY,as_of TEXT NOT NULL,checkpoint_id TEXT NOT NULL,options_run_id TEXT NOT NULL)")
            old = connection.execute("SELECT as_of,checkpoint_id,options_run_id FROM overlays WHERE overlay_id=?", (identity,)).fetchone()
            expected = (document["as_of"], request["checkpoint_id"], request["options_run_id"])
            if old is not None and old != expected:
                raise ValueError("overlay index conflicts with immutable artifact")
            if old is None and connection.execute("SELECT COUNT(*) FROM overlays").fetchone()[0] >= 1000:
                raise ValueError("overlay retention bound exceeded")
            write_immutable(self._path(f"runs/{identity}.json"), dumps(document).encode())
            connection.execute("INSERT OR IGNORE INTO overlays VALUES(?,?,?,?)", (identity, *expected))
        return result

    def load(self, overlay_id: str) -> dict[str, Any]:
        identity = PortfolioWorkspace._identity(overlay_id)
        encoded = self._path(f"runs/{identity}.json").read_bytes()
        document = loads(encoded)
        body = {k: v for k, v in document.items() if k != "overlay_id"}
        if (document.get("overlay_id") != identity or _hash(body) != identity or dumps(document).encode() != encoded
                or document.get("schema_version") != VERSION):
            raise ValueError("options overlay integrity failure")
        with sqlite3.connect(self._path("overlays.sqlite").as_uri()+"?mode=ro", uri=True) as connection:
            row = connection.execute("SELECT as_of,checkpoint_id,options_run_id FROM overlays WHERE overlay_id=?", (identity,)).fetchone()
        if row != (document["as_of"], document["checkpoint_id"], document["options_run_id"]):
            raise ValueError("overlay index differs from immutable artifact")
        request = self._request(document["request"])
        if request != document["request"]:
            raise ValueError("options overlay request is not canonical")
        try:
            checkpoint, options = self._parents(request, current=False)
        except _Unavailable:
            raise ValueError("historical overlay parent evidence no longer matches") from None
        if self._compute(checkpoint, options, request) != body:
            raise ValueError("overlay evidence differs from deterministic replay")
        return document

    def replay(self, payload: dict[str, Any]) -> Contract:
        _fields(payload, {"overlay_id"}, {"overlay_id"})
        try:
            document = self.load(payload["overlay_id"])
        except FileNotFoundError:
            return self._missing()
        return self._contract(document)

    def section(self, section: str, as_of: Any, max_age_seconds: float | None) -> Contract | None:
        if section not in SECTIONS:
            return None
        cutoff, maximum = _age(as_of, max_age_seconds)
        index = self._path("overlays.sqlite")
        if not index.is_file():
            return None
        with sqlite3.connect(index.as_uri()+"?mode=ro", uri=True) as connection:
            rows = connection.execute("SELECT overlay_id,as_of FROM overlays ORDER BY rowid LIMIT 1001").fetchall()
        if len(rows) > 1000:
            raise ValueError("overlay registry read bound exceeded")
        candidates = [(aware_timestamp(at), number, identity) for number, (identity, at) in enumerate(rows) if aware_timestamp(at) <= cutoff]
        if not candidates:
            return None
        document = self.load(max(candidates)[2])
        state = State(document["state"])
        if state != State.ERROR:
            state = State.UNKNOWN if maximum is None else State.STALE if _age_seconds(cutoff, document["observed_at"]) > maximum else State.AVAILABLE
        return self._contract(document, section, state)
