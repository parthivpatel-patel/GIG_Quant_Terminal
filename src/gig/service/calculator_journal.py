"""Content-addressed journal for the stateless research calculators.

Saving is explicit. The record is the calculator kind and the exact inputs.
Replay runs the calculator again and refuses the record if the result differs.
A saved dataset can supply one series; the extracted numbers are pinned in the
record. Nothing here reads an account, a feed, or an order.
"""
from __future__ import annotations

import hashlib
import math
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from gig.service.contracts import Contract, State
from gig.service.dataset_workspace import PersistedDatasetWorkspace
from gig.service.quant_analytics import ANALYSES, run
from gig.service.serialization import dumps, loads, public_error
from gig.service.workspace import TerminalWorkspace, _fields

IMPLEMENTATION = "gig.calculator-journal.v1"
MAX_RECORDS = 128
# The argument is filled from the dataset. The allowed fields are the only
# panel columns that may be copied, and they are not imputed.
# argument, allowed panel fields, and whether one column or an aligned matrix is copied.
DATASET_BINDINGS = {
    "cornish-fisher-var": ("returns", frozenset({"return_1"}), "series"),
    "stationary-bootstrap": ("series", frozenset({"return_1", "close"}), "series"),
    "roll-spread": ("prices", frozenset({"close"}), "series"),
    "ledoit-wolf": ("returns", frozenset({"return_1"}), "panel"),
    "minimum-cvar": ("scenarios", frozenset({"return_1"}), "panel"),
}
COVARIANCE_BINDINGS = frozenset({"hierarchical-risk-parity", "maximum-diversification"})
SIZE_BINDINGS = frozenset({
    "historical-var-components", "liquidity-var", "almgren-chriss", "participation-schedule",
})
SCREEN_BINDINGS = frozenset({"black-litterman", "risk-parity"})
OPTIONS_BINDINGS = frozenset({"svi-smile", "option-greeks-advanced"})
SIZE_RECORD_BINDINGS = frozenset({"conditional-stress"})
SCHEMA = """
CREATE TABLE IF NOT EXISTS records(
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    body TEXT NOT NULL
)
"""


def _identity(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("calculator record identity must be a lowercase SHA-256")
    return value


def panel_series(rows: Any, instrument_id: str, field: str) -> tuple[list[float], int]:
    """Copy one instrument column in stored order. Only a leading return warmup may be absent."""
    if not isinstance(instrument_id, str) or not instrument_id or len(instrument_id) > 64:
        raise ValueError("instrument identity must be a short string")
    if field not in {"return_1", "close"}:
        raise ValueError("dataset field must be return_1 or close")
    if not isinstance(rows, list):
        raise ValueError("dataset panel is unavailable")
    selected = [row for row in rows if isinstance(row, dict) and row.get("instrument_id") == instrument_id]
    if not selected:
        raise ValueError("dataset has no panel rows for that instrument")
    missing: list[int] = []
    values: list[float] = []
    for index, row in enumerate(selected):
        value = row.get(field)
        if value is None:
            missing.append(index)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("dataset field is not a finite number")
        values.append(float(value))
    if field == "return_1":
        if missing != list(range(len(missing))):
            raise ValueError("return_1 gap is not a leading warmup")
    elif missing:
        raise ValueError("close series has a missing observation")
    if len(values) < 2:
        raise ValueError("dataset series is too short")
    return values, len(missing)


def aligned_returns(rows: Any, instrument_ids: Any, field: str) -> tuple[list[list[float]], int]:
    """Keep timestamps where every requested instrument has a finite return. Do not fill the rest."""
    if field != "return_1":
        raise ValueError("aligned returns use return_1")
    if not isinstance(instrument_ids, list) or not 2 <= len(instrument_ids) <= 64 or len(set(instrument_ids)) != len(instrument_ids):
        raise ValueError("aligned returns need 2 to 64 distinct instruments")
    if any(not isinstance(item, str) or not item or len(item) > 64 for item in instrument_ids):
        raise ValueError("instrument identity must be a short string")
    if not isinstance(rows, list):
        raise ValueError("dataset panel is unavailable")
    order: list[str] = []
    seen: dict[str, dict[str, Any]] = {}
    requested = set(instrument_ids)
    for row in rows:
        if not isinstance(row, dict) or row.get("instrument_id") not in requested:
            continue
        stamp = row.get("timestamp")
        if not isinstance(stamp, str):
            raise ValueError("dataset timestamp is unavailable")
        if stamp not in seen:
            seen[stamp] = {}
            order.append(stamp)
        if row["instrument_id"] in seen[stamp]:
            raise ValueError("dataset repeats an instrument at one timestamp")
        seen[stamp][row["instrument_id"]] = row.get(field)
    matrix: list[list[float]] = []
    skipped = 0
    for stamp in order:
        cells = [seen[stamp].get(ident) for ident in instrument_ids]
        if all(value is None for value in cells):
            skipped += 1
            continue
        if any(value is None for value in cells):
            raise ValueError("aligned return panel has a partial observation")
        numbers: list[float] = []
        for value in cells:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("dataset field is not a finite number")
            numbers.append(float(value))
        matrix.append(numbers)
    if len(matrix) < 2:
        raise ValueError("dataset series is too short")
    return matrix, skipped


class CalculatorJournal:
    def __init__(self, directory: Path):
        self.path = TerminalWorkspace(directory)._path("calculator-journal")/"journal.sqlite"

    def _connect(self, *, write: bool = False) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path)
        db.execute("PRAGMA synchronous=FULL")
        if write:
            db.execute("PRAGMA journal_mode=DELETE")
            db.executescript(SCHEMA)
        return db

    def save(self, kind: str, inputs: dict[str, Any], *, source: dict[str, Any] | None = None) -> dict[str, Any]:
        if kind not in ANALYSES:
            raise ValueError("unsupported calculator")
        if not isinstance(inputs, dict):
            raise ValueError("calculator inputs must be an object")
        if source is not None and not isinstance(source, dict):
            raise ValueError("calculator source must be an object")
        result = run(kind, inputs)
        body = {"implementation": IMPLEMENTATION, "operation": "calculator", "kind": kind, "inputs": inputs,
                "result": result, "classification": "MODEL_ESTIMATE", "trading_approved": False,
                "generative_state": "UNAVAILABLE", "persistence": "LOCAL_CONTENT_ADDRESSED_RECORD"}
        if source is not None:
            body["source"] = source
        wire = dumps(body)
        identity = hashlib.sha256(wire.encode()).hexdigest()
        with closing(self._connect(write=True)) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                stored = db.execute("SELECT body FROM records WHERE id=?", (identity,)).fetchone()
                if stored is None:
                    count = db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
                    if count >= MAX_RECORDS:
                        raise ValueError("calculator journal capacity exhausted")
                    db.execute("INSERT INTO records VALUES(?,?,?)", (identity, kind, wire))
                elif stored[0] != wire:
                    raise ValueError("calculator record conflicts with immutable content")
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK")
                raise
        return {**body, "record_id": identity}

    def replay(self, record_id: str) -> dict[str, Any]:
        identity = _identity(record_id)
        if not self.path.exists():
            raise FileNotFoundError("calculator record unavailable")
        try:
            with closing(sqlite3.connect(self.path.as_uri()+"?mode=ro", uri=True)) as db:
                row = db.execute("SELECT body FROM records WHERE id=?", (identity,)).fetchone()
        except sqlite3.OperationalError:
            row = None
        if row is None:
            raise FileNotFoundError("calculator record unavailable")
        body = loads(row[0])
        if dumps(body) != row[0] or hashlib.sha256(row[0].encode()).hexdigest() != identity \
                or body.get("implementation") != IMPLEMENTATION:
            raise ValueError("calculator record integrity failed")
        again = run(body["kind"], body["inputs"])
        if dumps(again) != dumps(body["result"]):
            raise ValueError("calculator replay changed")
        return {**body, "record_id": identity, "replay_state": "REPRODUCED_FROM_PINNED_INPUTS"}


class CalculatorJournalWorkspace:
    def __init__(self, directory: Path):
        self.directory = directory
        self.journal = CalculatorJournal(directory)

    @staticmethod
    def _contract(data, *, state=State.AVAILABLE):
        return Contract.admit(state, data, source="user-supplied-research-inputs",
                              classification="MODEL_ESTIMATE" if data else "UNAVAILABLE",
                              scope="workspace.calculator-journal", readiness="RESEARCH_ONLY",
                              evidence_state="RESEARCH_OUTPUT",
                              error=public_error("DATA_UNAVAILABLE") if state == State.UNAVAILABLE else None)

    def run(self, payload: dict[str, Any]) -> Contract:
        if not isinstance(payload, dict) or payload.get("operation") not in (
                "save", "replay", "from-dataset", "from-record", "from-size", "from-screen",
                "from-options", "from-size-record"):
            raise ValueError(
                "calculator journal operation must be save, replay, from-dataset, from-record, "
                "from-size, from-screen, from-options or from-size-record")
        try:
            if payload["operation"] == "save":
                fields = _fields(payload, {"operation", "kind", "inputs"}, {"operation", "kind", "inputs"})
                if not isinstance(fields["kind"], str):
                    raise ValueError("calculator kind must be a string")
                return self._contract(self.journal.save(fields["kind"], fields["inputs"]))
            if payload["operation"] == "from-dataset":
                return self._contract(self._from_dataset(payload))
            if payload["operation"] == "from-record":
                return self._contract(self._from_record(payload))
            if payload["operation"] == "from-size":
                return self._contract(self._from_size(payload))
            if payload["operation"] == "from-screen":
                return self._contract(self._from_screen(payload))
            if payload["operation"] == "from-options":
                return self._contract(self._from_options(payload))
            if payload["operation"] == "from-size-record":
                return self._contract(self._from_size_record(payload))
            fields = _fields(payload, {"operation", "record_id"}, {"operation", "record_id"})
            return self._contract(self.journal.replay(fields["record_id"]))
        except FileNotFoundError:
            return self._contract(None, state=State.UNAVAILABLE)

    def _from_dataset(self, payload: dict[str, Any]) -> dict[str, Any]:
        fields = _fields(payload, {"operation", "dataset_id", "kind", "instrument_id", "instruments", "field", "inputs"},
                         {"operation", "dataset_id", "kind", "field", "inputs"})
        kind = fields["kind"]
        if not isinstance(kind, str) or kind not in DATASET_BINDINGS:
            raise ValueError("calculator cannot be bound to a saved dataset")
        argument, allowed_fields, mode = DATASET_BINDINGS[kind]
        if fields["field"] not in allowed_fields:
            raise ValueError("dataset field is not valid for that calculator")
        inputs = fields["inputs"]
        if not isinstance(inputs, dict) or argument in inputs:
            raise ValueError("calculator inputs must leave the dataset argument unbound")
        has_one = "instrument_id" in fields
        has_many = "instruments" in fields
        if mode == "series" and (not has_one or has_many):
            raise ValueError("this calculator binds one instrument_id")
        if mode == "panel" and (not has_many or has_one):
            raise ValueError("this calculator binds an instruments list")
        document = PersistedDatasetWorkspace(self.directory).load(fields["dataset_id"])
        if mode == "series":
            series, skipped = panel_series(document["panel_rows"], fields["instrument_id"], fields["field"])
            source = {"dataset_id": document["dataset_id"], "dataset_classification": document["classification"],
                      "instrument_id": fields["instrument_id"], "field": fields["field"], "argument": argument,
                      "observations": len(series), "leading_missing_observations": skipped,
                      "extraction": "PANEL_ROWS_IN_STORED_ORDER_NO_IMPUTATION"}
            return self.journal.save(kind, {**inputs, argument: series}, source=source)
        matrix, skipped = aligned_returns(document["panel_rows"], fields["instruments"], fields["field"])
        source = {"dataset_id": document["dataset_id"], "dataset_classification": document["classification"],
                  "instruments": list(fields["instruments"]), "field": fields["field"], "argument": argument,
                  "observations": len(matrix), "assets": len(fields["instruments"]),
                  "skipped_all_missing_timestamps": skipped,
                  "extraction": "ALIGNED_RETURN_PANEL_NO_IMPUTATION"}
        return self.journal.save(kind, {**inputs, argument: matrix}, source=source)

    def _from_record(self, payload: dict[str, Any]) -> dict[str, Any]:
        fields = _fields(payload, {"operation", "record_id", "kind", "inputs"},
                         {"operation", "record_id", "kind", "inputs"})
        kind = fields["kind"]
        if not isinstance(kind, str) or kind not in COVARIANCE_BINDINGS:
            raise ValueError("calculator cannot bind a saved covariance")
        inputs = fields["inputs"]
        if not isinstance(inputs, dict) or "covariance" in inputs:
            raise ValueError("calculator inputs must leave the covariance unbound")
        parent = self.journal.replay(fields["record_id"])
        if parent["kind"] != "ledoit-wolf" or "covariance" not in parent["result"]:
            raise ValueError("covariance must come from a saved Ledoit-Wolf record")
        covariance = parent["result"]["covariance"]
        source = {"parent_record_id": parent["record_id"], "parent_kind": "ledoit-wolf",
                  "argument": "covariance", "extraction": "PINNED_LEDOIT_WOLF_COVARIANCE"}
        parent_source = parent.get("source")
        if isinstance(parent_source, dict):
            for key in ("dataset_id", "dataset_classification", "instruments"):
                if key in parent_source:
                    source[key] = parent_source[key]
        return self.journal.save(kind, {**inputs, "covariance": covariance}, source=source)

    def _from_size(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Bind research position sizes into risk or execution calculators.

        ``historical-var-components`` uses signed size weights over the aligned
        return panel. ``liquidity-var``, ``almgren-chriss``, and
        ``participation-schedule`` bind the largest absolute name using panel
        ADV/volume and return volatility. No order path.
        """
        fields = _fields(payload, {"operation", "size_id", "kind", "field", "inputs",
                                   "adv_lookback_bars", "instrument_id"},
                         {"operation", "size_id", "kind", "inputs"})
        kind = fields["kind"]
        if not isinstance(kind, str) or kind not in SIZE_BINDINGS:
            raise ValueError("calculator cannot be bound to research position sizes")
        inputs = fields["inputs"]
        if not isinstance(inputs, dict):
            raise ValueError("calculator inputs must be an object")
        from gig.service.research_strategy import ResearchStrategyWorkspace
        sized = ResearchStrategyWorkspace(self.directory).load_size(fields["size_id"])
        dataset = PersistedDatasetWorkspace(self.directory).load(sized["dataset_id"])
        if kind == "historical-var-components":
            if fields.get("field", "return_1") != "return_1":
                raise ValueError("dataset field is not valid for size-bound risk")
            if "asset_returns" in inputs or "weights" in inputs:
                raise ValueError("calculator inputs must leave asset_returns and weights unbound")
            instruments = [row["instrument_id"] for row in sized["rows"]]
            portfolio = float(sized["request"]["portfolio_notional"])
            if portfolio <= 0:
                raise ValueError("portfolio notional must be positive")
            weights = []
            for row in sized["rows"]:
                notional = float(row["target_notional"])
                signed = notional if row["side"] == "buy" else -notional
                weights.append(signed / portfolio)
            matrix, skipped = aligned_returns(dataset["panel_rows"], instruments, "return_1")
            if len(matrix) < 20:
                raise ValueError("aligned return panel is too short for size-bound risk")
            source = {
                "size_id": sized["size_id"], "dataset_id": dataset["dataset_id"],
                "dataset_classification": dataset["classification"],
                "screen_id": sized["screen_id"], "instruments": instruments,
                "field": "return_1", "observations": len(matrix),
                "assets": len(instruments), "skipped_all_missing_timestamps": skipped,
                "extraction": "RESEARCH_SIZE_WEIGHTS_ALIGNED_RETURN_PANEL_NO_IMPUTATION",
                "trading_approved": False,
            }
            return self.journal.save(kind, {**inputs, "asset_returns": matrix, "weights": weights},
                                     source=source)
        lookback = int(fields.get("adv_lookback_bars", 20))
        if not 5 <= lookback <= 252:
            raise ValueError("adv_lookback_bars must lie between 5 and 252")
        rows = list(sized["rows"])
        if "instrument_id" in fields:
            rows = [row for row in rows if row["instrument_id"] == fields["instrument_id"]]
            if not rows:
                raise ValueError("instrument_id is not present in the research size")
        selected = max(rows, key=lambda row: float(row["target_notional"]))
        notional = float(selected["target_notional"])
        price = float(selected["reference_price"])
        if notional <= 0 or price <= 0:
            raise ValueError("selected size row must have positive notional and price")
        shares = notional / price
        panel_rows = dataset["panel_rows"]
        if not isinstance(panel_rows, list):
            raise ValueError("dataset panel is unavailable")
        closes: list[float] = []
        volumes: list[float] = []
        returns: list[float] = []

        def _finite(value: Any) -> float | None:
            if isinstance(value, bool):
                return None
            if isinstance(value, (int, float)):
                number = float(value)
            elif isinstance(value, str):
                try:
                    number = float(value)
                except ValueError:
                    return None
            else:
                return None
            return number if math.isfinite(number) else None

        for row in panel_rows:
            if not isinstance(row, dict) or row.get("instrument_id") != selected["instrument_id"]:
                continue
            close = _finite(row.get("close"))
            volume = _finite(row.get("volume"))
            ret = _finite(row.get("return_1"))
            if close is not None and volume is not None and volume > 0 and close > 0:
                closes.append(close)
                volumes.append(volume)
            if ret is not None:
                returns.append(ret)
        if len(closes) < lookback or len(returns) < 20:
            raise ValueError("selected instrument lacks enough panel history for size binding")
        recent_closes = closes[-lookback:]
        recent_volumes = volumes[-lookback:]
        adv = sum(c * v for c, v in zip(recent_closes, recent_volumes, strict=True)) / lookback
        if adv <= 0:
            raise ValueError("average daily value must be positive")
        mean_ret = sum(returns) / len(returns)
        variance = sum((value - mean_ret) ** 2 for value in returns) / (len(returns) - 1)
        volatility = math.sqrt(max(variance, 0.0))
        if volatility <= 0:
            raise ValueError("return volatility must be positive")
        if kind == "liquidity-var":
            if any(key in inputs for key in
                   ("position_value", "volatility", "average_daily_value")):
                raise ValueError(
                    "calculator inputs must leave position_value, volatility and "
                    "average_daily_value unbound")
            bound = {
                "position_value": notional,
                "volatility": volatility,
                "average_daily_value": adv,
                **inputs,
            }
            extraction = "RESEARCH_SIZE_LARGEST_NAME_LIQUIDITY_VAR"
        elif kind == "participation-schedule":
            if any(key in inputs for key in ("shares", "volumes")):
                raise ValueError("calculator inputs must leave shares and volumes unbound")
            whole_shares = int(float(selected["quantity"]))
            if whole_shares < 1:
                raise ValueError("participation schedule needs a positive whole-share quantity")
            volume_forecast = [max(1, int(value)) for value in volumes[-lookback:]]
            if not volume_forecast:
                raise ValueError("participation schedule needs a volume forecast")
            bound = {
                "shares": whole_shares,
                "volumes": volume_forecast,
                **inputs,
            }
            extraction = "RESEARCH_SIZE_LARGEST_NAME_PARTICIPATION_SCHEDULE"
        else:
            if any(key in inputs for key in ("shares", "volatility")):
                raise ValueError("calculator inputs must leave shares and volatility unbound")
            if "temporary_impact" not in inputs or "horizon" not in inputs or "intervals" not in inputs:
                raise ValueError("almgren-chriss binding requires horizon, intervals and temporary_impact")
            bound = {
                "shares": shares,
                "volatility": volatility * price,
                **inputs,
            }
            extraction = "RESEARCH_SIZE_LARGEST_NAME_ALMGREN_CHRISS"
        source = {
            "size_id": sized["size_id"], "dataset_id": dataset["dataset_id"],
            "dataset_classification": dataset["classification"],
            "screen_id": sized["screen_id"],
            "instrument_id": selected["instrument_id"],
            "adv_lookback_bars": lookback,
            "observations": len(returns),
            "extraction": extraction,
            "trading_approved": False,
        }
        return self.journal.save(kind, bound, source=source)

    def _from_screen(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Bind a research screen and Ledoit-Wolf covariance into allocation research.

        Black-Litterman views are absolute unit views on actionable names using
        research_weight as the uncalibrated view return. Risk parity uses equal
        risk budgets over the covariance instruments. Neither path approves trading.
        """
        fields = _fields(payload, {"operation", "screen_id", "record_id", "kind", "inputs"},
                         {"operation", "screen_id", "record_id", "kind", "inputs"})
        kind = fields["kind"]
        if not isinstance(kind, str) or kind not in SCREEN_BINDINGS:
            raise ValueError("calculator cannot be bound to a research screen")
        inputs = fields["inputs"]
        if not isinstance(inputs, dict):
            raise ValueError("calculator inputs must be an object")
        parent = self.journal.replay(fields["record_id"])
        if parent["kind"] != "ledoit-wolf" or "covariance" not in parent["result"]:
            raise ValueError("covariance must come from a saved Ledoit-Wolf record")
        parent_source = parent.get("source")
        if not isinstance(parent_source, dict) or "instruments" not in parent_source:
            raise ValueError("Ledoit-Wolf record must pin its instrument order")
        instruments = list(parent_source["instruments"])
        covariance = parent["result"]["covariance"]
        if len(covariance) != len(instruments):
            raise ValueError("covariance dimension does not match pinned instruments")
        from gig.service.research_screen import ResearchScreenWorkspace
        screen = ResearchScreenWorkspace(self.directory).load(fields["screen_id"])
        if screen.get("dataset_id") != parent_source.get("dataset_id"):
            raise ValueError("screen and covariance must share one dataset identity")
        by_name = {row["instrument_id"]: row for row in screen["rows"]}
        if any(name not in by_name for name in instruments):
            raise ValueError("covariance instruments must appear on the research screen")
        count = len(instruments)
        market_weights = [1.0 / count] * count
        bound: dict[str, Any] = {"covariance": covariance}
        if kind == "risk-parity":
            if any(key in inputs for key in ("covariance", "budgets")):
                raise ValueError("calculator inputs must leave covariance and budgets unbound")
            bound["budgets"] = list(market_weights)
            bound.update(inputs)
        else:
            if any(key in inputs for key in
                   ("covariance", "market_weights", "views", "view_returns")):
                raise ValueError(
                    "calculator inputs must leave covariance, market_weights, views and "
                    "view_returns unbound")
            views: list[list[float]] = []
            view_returns: list[float] = []
            for index, name in enumerate(instruments):
                row = by_name[name]
                if row.get("stance") == "RESEARCH_HOLD" or float(row.get("research_weight", 0)) == 0.0:
                    continue
                vector = [0.0] * count
                vector[index] = 1.0
                views.append(vector)
                view_returns.append(float(row["research_weight"]))
            if not views:
                raise ValueError("screen has no actionable research views for Black-Litterman")
            bound.update({
                "market_weights": market_weights,
                "views": views,
                "view_returns": view_returns,
                **inputs,
            })
        source = {
            "screen_id": screen["screen_id"],
            "parent_record_id": parent["record_id"],
            "parent_kind": "ledoit-wolf",
            "dataset_id": screen["dataset_id"],
            "dataset_classification": parent_source.get("dataset_classification",
                                                       screen.get("input_classification")),
            "instruments": instruments,
            "argument": "covariance",
            "extraction": "RESEARCH_SCREEN_VIEWS_PINNED_LEDOIT_WOLF_COVARIANCE",
            "trading_approved": False,
            "view_count": len(bound.get("views", [])),
        }
        return self.journal.save(kind, bound, source=source)

    def _from_options(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Bind a saved options research run into SVI or higher-order Greeks.

        SVI uses admitted CALL rows with finite IV and log-forward moneyness;
        total variance is ``iv**2 * ttm``. Greeks bind one admitted contract's
        inverted IV. Research-only; no OPRA or trading path.
        """
        fields = _fields(payload, {"operation", "options_run_id", "kind", "expiry", "contract_id", "inputs"},
                         {"operation", "options_run_id", "kind", "inputs"})
        kind = fields["kind"]
        if not isinstance(kind, str) or kind not in OPTIONS_BINDINGS:
            raise ValueError("calculator cannot be bound to a saved options run")
        inputs = fields["inputs"]
        if not isinstance(inputs, dict):
            raise ValueError("calculator inputs must be an object")
        from gig.service.options_workspace import OptionsWorkspace
        options = OptionsWorkspace(self.directory).load(fields["options_run_id"])
        if kind == "option-greeks-advanced":
            if any(key in inputs for key in ("spot", "strike", "vol", "ttm", "call", "rate", "div")):
                raise ValueError(
                    "calculator inputs must leave spot, strike, vol, ttm, call, rate and div unbound")
            contract_id = fields.get("contract_id")
            rows = [
                row for row in options.get("rows", [])
                if row.get("state") == "AVAILABLE" and row.get("implied_volatility") is not None
                and row.get("ttm") is not None and float(row["ttm"]) > 0
                and row.get("strike") is not None
            ]
            if contract_id is not None:
                rows = [row for row in rows if row.get("contract_id") == contract_id]
            if not rows:
                raise ValueError("options run has no admitted contract for higher-order Greeks")
            selected = min(rows, key=lambda row: abs(float(row.get("log_forward_moneyness") or 0.0)))
            request = options.get("request") or {}
            bound = {
                "spot": float(options["spot"]),
                "strike": float(selected["strike"]),
                "rate": float(request.get("rate", 0.0)),
                "div": float(request.get("dividend_yield", 0.0)),
                "vol": float(selected["implied_volatility"]),
                "ttm": float(selected["ttm"]),
                "call": selected.get("right") == "CALL",
                **inputs,
            }
            source = {
                "options_run_id": options["run_id"],
                "underlying_id": options["underlying_id"],
                "contract_id": selected["contract_id"],
                "expiry": selected.get("expiry"),
                "argument": "spot/strike/vol/ttm",
                "extraction": "ADMITTED_OPTION_IV_HIGHER_ORDER_GREEKS",
                "trading_approved": False,
                "options_trading_approved": False,
            }
            return self.journal.save(kind, bound, source=source)
        if any(key in inputs for key in ("log_moneyness", "total_variance")):
            raise ValueError("calculator inputs must leave log_moneyness and total_variance unbound")
        rows = [row for row in options.get("rows", [])
                if row.get("right") == "CALL" and row.get("state") == "AVAILABLE"
                and row.get("implied_volatility") is not None
                and row.get("log_forward_moneyness") is not None
                and row.get("ttm") is not None and float(row["ttm"]) > 0]
        if "expiry" in fields:
            expiry = fields["expiry"]
            rows = [row for row in rows if row.get("expiry") == expiry]
        else:
            expiries = sorted({row["expiry"] for row in rows})
            if not expiries:
                raise ValueError("options run has no admitted CALL smile for SVI")
            expiry = expiries[0]
            rows = [row for row in rows if row.get("expiry") == expiry]
        ordered = sorted(rows, key=lambda row: float(row["log_forward_moneyness"]))
        if len(ordered) < 5:
            raise ValueError("SVI binding needs at least five admitted CALL quotes on one expiry")
        log_moneyness = [float(row["log_forward_moneyness"]) for row in ordered]
        if len(set(log_moneyness)) != len(log_moneyness):
            raise ValueError("options smile has duplicate log-moneyness on the selected expiry")
        total_variance = [
            float(row["implied_volatility"]) ** 2 * float(row["ttm"]) for row in ordered
        ]
        source = {
            "options_run_id": options["run_id"],
            "underlying_id": options["underlying_id"],
            "expiry": expiry,
            "quotes": len(ordered),
            "argument": "log_moneyness/total_variance",
            "extraction": "ADMITTED_CALL_IV_TOTAL_VARIANCE_ONE_EXPIRY",
            "trading_approved": False,
            "options_trading_approved": False,
        }
        return self.journal.save(kind, {**inputs, "log_moneyness": log_moneyness,
                                        "total_variance": total_variance}, source=source)

    def _from_size_record(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Bind research sizes and a Ledoit-Wolf covariance into conditional stress.

        Weights follow signed size notionals over portfolio notional. Covariance
        stays pinned from the parent record. Caller supplies shocked_assets/shocks.
        """
        fields = _fields(payload, {"operation", "size_id", "record_id", "kind", "inputs"},
                         {"operation", "size_id", "record_id", "kind", "inputs"})
        kind = fields["kind"]
        if not isinstance(kind, str) or kind not in SIZE_RECORD_BINDINGS:
            raise ValueError("calculator cannot bind research sizes with a covariance record")
        inputs = fields["inputs"]
        if not isinstance(inputs, dict) or any(key in inputs for key in ("weights", "covariance")):
            raise ValueError("calculator inputs must leave weights and covariance unbound")
        if "shocked_assets" not in inputs or "shocks" not in inputs:
            raise ValueError("conditional stress requires shocked_assets and shocks")
        from gig.service.research_strategy import ResearchStrategyWorkspace
        sized = ResearchStrategyWorkspace(self.directory).load_size(fields["size_id"])
        parent = self.journal.replay(fields["record_id"])
        if parent["kind"] != "ledoit-wolf" or "covariance" not in parent["result"]:
            raise ValueError("covariance must come from a saved Ledoit-Wolf record")
        parent_source = parent.get("source")
        if not isinstance(parent_source, dict) or "instruments" not in parent_source:
            raise ValueError("Ledoit-Wolf record must pin its instrument order")
        instruments = list(parent_source["instruments"])
        size_weights = {
            row["instrument_id"]: (
                float(row["target_notional"]) / float(sized["request"]["portfolio_notional"])
                if row["side"] == "buy"
                else -float(row["target_notional"]) / float(sized["request"]["portfolio_notional"])
            )
            for row in sized["rows"]
        }
        if any(name not in size_weights for name in instruments):
            raise ValueError("covariance instruments must appear in the research size")
        if sized["dataset_id"] != parent_source.get("dataset_id"):
            raise ValueError("size and covariance must share one dataset identity")
        weights = [size_weights[name] for name in instruments]
        covariance = parent["result"]["covariance"]
        if len(covariance) != len(instruments):
            raise ValueError("covariance dimension does not match pinned instruments")
        source = {
            "size_id": sized["size_id"],
            "screen_id": sized["screen_id"],
            "parent_record_id": parent["record_id"],
            "parent_kind": "ledoit-wolf",
            "dataset_id": sized["dataset_id"],
            "dataset_classification": parent_source.get("dataset_classification",
                                                       sized.get("input_classification")),
            "instruments": instruments,
            "argument": "weights/covariance",
            "extraction": "RESEARCH_SIZE_WEIGHTS_PINNED_LEDOIT_WOLF_COVARIANCE",
            "trading_approved": False,
        }
        return self.journal.save(kind, {**inputs, "weights": weights, "covariance": covariance},
                                 source=source)
