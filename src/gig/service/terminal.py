"""Read-only terminal integration and bounded, explicitly selected local research.

Synthetic mode is opt-in. Its fictional instruments and deterministic data never
represent market observations, broker activity, or economically validated alpha.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from gig.service.contracts import Contract, State
from gig.service.serialization import public_error

SECTIONS = ("command-center", "markets", "signals", "portfolio", "risk", "models",
            "news", "stat-arb", "options", "execution", "research", "operations")
SYNTHETIC_ASOF = "2026-09-30T20:00:00+00:00"


def _response(data: dict, *, scope: str, synthetic: bool = False,
              classification: str = "MODEL_ESTIMATE", source: str = "local-research",
              as_of: str | None = None) -> Contract:
    return Contract.admit(State.AVAILABLE, data, source=source,
                          classification="SYNTHETIC" if synthetic else classification,
                          as_of=SYNTHETIC_ASOF if synthetic else as_of, scope=scope,
                          evidence_state="RESEARCH_OUTPUT", readiness="RESEARCH_ONLY")


def unavailable(scope: str, *, code: str = "DATA_UNAVAILABLE") -> Contract:
    return Contract.admit(State.UNAVAILABLE, None, source="local", classification="UNAVAILABLE",
                          scope=scope, error=public_error(code))


def _fixture() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(18209317)
    returns = rng.normal(0, .012, (180, 12)) + rng.normal(0, .004, (180, 1))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    symbols = [f"SYN{i + 1:03d}" for i in range(12)]
    factors = np.column_stack((prices[-1] / prices[-21] - 1,
                               -returns[-5:].sum(axis=0), returns[-60:].std(axis=0)))
    factors = (factors - factors.mean(axis=0)) / factors.std(axis=0)
    scores = .5 * factors[:, 0] + .3 * factors[:, 1] - .2 * factors[:, 2]
    return returns, prices, symbols, factors, scores


def _factor_chart(symbols: list[str], factors: np.ndarray, scores: np.ndarray) -> dict:
    return {"kind": "factor-space", "title": "Fictional instruments in factor space",
            "axes": [{"key": key, "label": label, "unit": "cross-sectional z-score"}
                     for key, label in zip(("x", "y", "z"),
                                           ("20-day momentum", "5-day reversal", "60-day volatility"), strict=True)],
            "points": [{"symbol": symbol, "x": factors[i, 0], "y": factors[i, 1],
                        "z": factors[i, 2], "score": scores[i]}
                       for i, symbol in enumerate(symbols)]}


def section_snapshot(section: str, mode: str = "local") -> Contract:
    if section not in SECTIONS or mode not in ("local", "synthetic"):
        raise ValueError("unsupported section or mode")
    if mode == "local":
        return _local_section(section)
    returns, prices, symbols, factors, scores = _fixture()
    from gig.portfolio.construct import dollar_neutral_quantiles
    weights = dollar_neutral_quantiles(pd.Series(scores, index=symbols), 3, 3, .05, .3).to_numpy()
    data: dict[str, Any] = {
        "title": section.replace("-", " ").title(),
        "description": "Deterministic synthetic research; fictional instruments; no trading approval.",
        "metrics": {}, "rows": [],
    }
    if section in ("command-center", "markets", "signals", "portfolio"):
        data["rows"] = [{"symbol": symbol, "price": prices[-1, i],
                         "daily_return": float(np.expm1(returns[-1, i])),
                         "x": factors[i, 0], "y": factors[i, 1], "z": factors[i, 2],
                         "research_score": scores[i], "target_weight": weights[i],
                         "approved_for_trading": False, "data_classification": "SYNTHETIC"}
                        for i, symbol in enumerate(symbols)]
        data["metrics"] = {"instruments": len(symbols), "observations": returns.size,
                           "gross_target": float(np.abs(weights).sum()),
                           "net_target": float(weights.sum()), "broker_connected": False}
        data["chart"] = _factor_chart(symbols, factors, scores)
    elif section == "risk":
        from gig.options import OptionPosition, scenario_grid
        daily_pnl = np.expm1(returns) @ weights * 100_000
        losses = -daily_pnl
        threshold = float(np.quantile(losses, .95))
        data["metrics"] = {"historical_var_95_currency": threshold,
                           "historical_es_95_currency": float(losses[losses >= threshold].mean()),
                           "confidence": .95, "holding_period_days": 1, "synthetic_nav": 100_000}
        # Reprice the same synthetic European call at every scenario, rather
        # than drawing a decorative polynomial and calling it a risk surface.
        positions = [OptionPosition(instrument="SYN001-C100", underlying="SYN001",
                                    strike=100, ttm=.5, volatility=.2, quantity=10,
                                    source="authored-synthetic-fixture", call=True)]
        grid = scenario_grid(positions, spot=100, spot_shocks=[-.2, -.1, 0, .1, .2],
                             volatility_shocks=[-.05, 0, .05, .1])
        data["rows"] = grid["scenarios"]
        data["chart"] = {"kind": "risk-surface", "title": "Synthetic option repricing scenarios",
                         "axes": [{"key": "spot_shock", "label": "Spot shock", "unit": "decimal"},
                                  {"key": "volatility_shock", "label": "Volatility shock", "unit": "absolute annualized decimal"},
                                  {"key": "model_pnl", "label": "Model P&L", "unit": "currency"}], "points": grid["scenarios"]}
    elif section == "options":
        from gig.options import black_scholes
        rows = []
        for tenor in (.1, .25, .5, 1.0, 2.0):
            for strike in (75, 85, 95, 100, 105, 115, 125):
                vol = .18 + .25 * (np.log(strike / 100)) ** 2 + .02 * np.sqrt(tenor)
                option_value = black_scholes(100, strike, .03, 0, float(vol), tenor)
                rows.append({"strike": strike, "tenor_years": tenor, "iv": float(vol),
                             "price": option_value["price"], "price_kind": "MODEL_ESTIMATE"})
        data["rows"] = rows
        data["metrics"] = {"spot_assumption": 100, "risk_free_assumption": .03,
                           "observed_option_quotes": 0, "VTS": "AUTHORITY_UNSUPPLIED",
                           "CSM": "AUTHORITY_UNSUPPLIED", "OPRA": "UNAVAILABLE"}
        data["chart"] = {"kind": "volatility-surface", "title": "Assumed synthetic volatility surface",
                         "points": rows}
    elif section == "models":
        from gig.ml.state_space import kalman_filter, rts_smooth
        from gig.research.workflow import model_capabilities
        model = kalman_filter(returns[:60, :1], [[1]], [[1]], [[1e-6]], [[1e-4]], [0], [[1e-4]])
        smooth = rts_smooth(model)
        data["metrics"] = {"kalman_log_likelihood": model.log_likelihood,
                           "smoothed_uses_future_observations": True,
                           "economic_validation": "UNAVAILABLE"}
        data["rows"] = [{"model": key, **value} for key, value in model_capabilities().items()]
        data["diagnostics"] = {"filtered_mean": model.filtered_mean[-1].tolist(),
                               "offline_smoothed_mean": smooth.mean[-1].tolist()}
    elif section == "stat-arb":
        from gig.research.stat_arb import evaluate_pair, fit_pair, mean_reversion_diagnostics
        fit = fit_pair(prices[:, 0], prices[:, 1], dependent=symbols[0],
                       independent=symbols[1], training_rows=120)
        evaluation = evaluate_pair(fit, prices[120:, 0], prices[120:, 1])
        diagnostics = mean_reversion_diagnostics(evaluation["spread"])
        data["metrics"] = {"log_price_hedge_elasticity": fit.beta, "intercept": fit.alpha,
                           "training_spread_std": fit.spread_std, "economic_validation": "UNAVAILABLE",
                           "training_rows": 120, "holdout_rows": 60}
        data["diagnostics"] = diagnostics
        data["rows"] = [{"observation": i, "spread": float(value)}
                        for i, value in enumerate(evaluation["spread"])]
    elif section == "news":
        from gig.nlp.workflow import analyze_articles
        articles = [{"article_id": "synthetic-1", "source": "authored-synthetic-fixture",
                     "published_at": SYNTHETIC_ASOF, "available_at": SYNTHETIC_ASOF,
                     "url": "", "title": "SYN001 reports earnings growth",
                     "body": "SYN001 reports profit growth. This is an authored synthetic research example.",
                     "license": "authored synthetic fixture"}]
        result = analyze_articles(articles, as_of=SYNTHETIC_ASOF, aliases={"SYN001": "SYN001"})
        data["rows"] = result["articles"]
        data["metrics"] = {"articles": len(data["rows"]), "finbert": "UNAVAILABLE",
                           "external_news_ingested": 0}
    elif section == "research":
        # Cross-sectional fixture only, explicitly descriptive and in-sample.
        ic = float(pd.Series(scores).corr(pd.Series(returns[-1]), method="spearman"))
        data["metrics"] = {"descriptive_rank_correlation": ic,
                           "sample_names": 12, "evidence": "IN_SAMPLE_SYNTHETIC_ONLY"}
        data["rows"] = [{"symbol": symbol, "score": scores[i], "same_day_return": returns[-1, i]}
                        for i, symbol in enumerate(symbols)]
        data["description"] += " Correlation is in-sample, not predictive performance. Use the purged workflow for evaluation."
    elif section == "execution":
        from gig.execution.twap import plan_twap
        start = datetime(2026, 9, 30, 14, tzinfo=UTC)
        plan = plan_twap("synthetic-parent", "100", start=start,
                         end=start + timedelta(seconds=300), slices=5)
        data["rows"] = [asdict(child) for child in plan]
        data["metrics"] = {"broker_connected": False, "real_orders_sent": 0,
                           "planner": "OFFLINE_TWAP", "kill_switch": "NO_BROKER_AUTHORITY"}
    else:
        return _local_section(section)
    return _response(data, scope=f"terminal.{section}", synthetic=True,
                     source="deterministic-synthetic-v1")


def _local_section(section: str) -> Contract:
    from gig.service import snapshot
    scope = f"terminal.{section}"
    if section == "operations":
        from gig.ops.maintenance import startup_report
        report = startup_report()
        return _response({"title": "Operations", "description": "Local diagnostics; report availability does not establish service readiness.",
                          "metrics": {key: value for key, value in report.items() if key != "dependencies"},
                          "rows": [{"dependency": key, "available": value}
                                   for key, value in report["dependencies"].items()]},
                         scope=scope, source="local-process-diagnostics", classification="SNAPSHOT",
                         as_of=datetime.now(UTC).isoformat())
    if section == "models":
        from gig.research.workflow import model_capabilities
        return _response({"title": "Model capabilities", "description": "Dependency, implementation, trained state, and approval are separate.",
                          "metrics": {"trading_approved": False},
                          "rows": [{"model": key, **value} for key, value in model_capabilities().items()]},
                         scope=scope, classification="SNAPSHOT", source="local-capability-registry")
    if section == "execution":
        return unavailable(scope, code="BROKER_DISABLED")
    if section == "research":
        raw = snapshot.research_snapshot()
        if not raw.get("available"):
            return unavailable(scope)
        classification = "SYNTHETIC" if str(raw.get("source")).lower() == "synthetic" else "UNKNOWN"
        # Unverified legacy source claims cannot be promoted to market evidence.
        if classification == "UNKNOWN":
            return Contract.admit(State.UNKNOWN, None, source="legacy-research-artifact", scope=scope)
        return _response({"title": "Persisted research", "metrics": raw["metrics"],
                          "rows": [{"factor": key, "ic": value} for key, value in raw["factor_ic"].items()],
                          "evidence": raw.get("evidence"), "source_text": raw.get("source_text")},
                         scope=scope, classification=classification, source="local-research-artifact")
    # Legacy bars lack the revision/source provenance needed for these contracts.
    # Do not infer admission or freshness from a populated database or API key.
    return unavailable(scope)


def _bounded_array(value: Any, *, maximum: int = 2000, dimensions: int = 2) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim > dimensions or array.size > maximum or not np.isfinite(array).all():
        raise ValueError("invalid or oversized numeric input")
    return array


def analytics(kind: str, payload: dict[str, Any]) -> Contract:
    """Compute supported numerical research; no imports contact providers."""
    scope = f"analytics.{kind}"
    result: dict[str, Any]
    if kind == "calendar":
        from gig.data.calendars import calendar_analytics

        return _response(calendar_analytics(payload), scope=scope,
                         source="installed-exchange-calendar-rules", as_of=payload.get("as_of"))
    if kind == "corporate-actions":
        from gig.data.actions import adjust_bars

        required = {"rows", "actions", "as_of", "mode", "source", "classification", "currency"}
        if set(payload) != required:
            raise ValueError("corporate action request fields do not match the contract")
        result = adjust_bars(payload["rows"], payload["actions"], as_of=payload["as_of"],
                             mode=payload["mode"], source=payload["source"],
                             classification=payload["classification"], currency=payload["currency"])
        if result.get("state") == "UNAVAILABLE":
            return unavailable(scope)
        return _response(result, scope=scope, source="caller-evidenced-corporate-action-research",
                         as_of=payload["as_of"])
    if kind == "cost-evidence":
        from gig.execution.cost_evidence import cost_evidence_report

        report = cost_evidence_report(payload)
        return _response(report, scope=scope, source="caller-supplied-cost-evidence",
                         classification="MODEL_ESTIMATE", as_of=report["as_of"])
    if kind == "provenance":
        from gig.data.provenance import assess_eligibility, documented_sources

        if set(payload) - {"records", "record_id", "as_of", "purpose"} or not {"record_id", "as_of", "purpose"} <= set(payload):
            raise ValueError("provenance request fields do not match the contract")
        records = payload.get("records")
        if records is None:
            records = documented_sources()
        result = assess_eligibility(records, record_id=payload["record_id"],
                                    as_of=payload["as_of"], purpose=payload["purpose"])
        return _response(result, scope=scope, source="lineage-records", classification="SNAPSHOT",
                         as_of=result["as_of"])
    if kind == "options":
        from gig.options import black_scholes, implied_volatility
        allowed = {"spot", "strike", "rate", "div", "vol", "ttm", "call", "market_price"}
        if set(payload) - allowed or not {"spot", "strike"} <= set(payload):
            raise ValueError("invalid option fields")
        args = {key: value for key, value in payload.items() if key != "market_price"}
        if "call" in args and not isinstance(args["call"], bool):
            raise ValueError("call must be a boolean")
        result = black_scholes(**args)
        if "market_price" in payload:
            args.pop("vol", None)
            result["implied_volatility"] = implied_volatility(payload["market_price"], **args)
        return _response(result, scope=scope, source="user-supplied-option-assumptions")
    if kind == "kalman":
        from gig.ml.state_space import kalman_filter, rts_smooth
        allowed = {"observations", "transition", "observation", "process_covariance",
                   "observation_covariance", "initial_mean", "initial_covariance", "smooth"}
        if set(payload) - allowed or not allowed - {"smooth"} <= set(payload):
            raise ValueError("invalid state-space fields")
        for key, value in payload.items():
            if key != "smooth":
                _bounded_array(value, maximum=2000)
        model = kalman_filter(**{key: value for key, value in payload.items() if key != "smooth"})
        result = asdict(model)
        if payload.get("smooth") is True:
            result["offline_smoothing"] = asdict(rts_smooth(model))
        return _response(result, scope=scope, source="user-supplied-state-space-data")
    if kind == "garch":
        from gig.ml.volatility import fit_garch11
        if set(payload) - {"returns", "max_iterations"} or "returns" not in payload:
            raise ValueError("invalid GARCH fields")
        _bounded_array(payload["returns"], maximum=2000, dimensions=1)
        iterations = payload.get("max_iterations", 100)
        if isinstance(iterations, bool) or not isinstance(iterations, int) or not 1 <= iterations <= 200:
            raise ValueError("iterations out of range")
        garch_fit = fit_garch11(payload["returns"], max_iterations=iterations)
        result = asdict(garch_fit)
        result["conditional_variances"] = garch_fit.model.conditional_variances(payload["returns"])
        result["next_variance"] = garch_fit.model.forecast(payload["returns"])
        return _response(result, scope=scope, source="user-supplied-returns")
    if kind == "hmm":
        from gig.ml.regimes import fit_gaussian_hmm, state_duration_diagnostics
        if set(payload) - {"observations", "states", "max_iterations"} or "observations" not in payload:
            raise ValueError("invalid HMM fields")
        observations = _bounded_array(payload["observations"], maximum=1000)
        states, iterations = payload.get("states", 2), payload.get("max_iterations", 50)
        if not isinstance(states, int) or isinstance(states, bool) or not 1 <= states <= 8:
            raise ValueError("states out of range")
        if not isinstance(iterations, int) or isinstance(iterations, bool) or not 1 <= iterations <= 200:
            raise ValueError("iterations out of range")
        hmm_fit = fit_gaussian_hmm(observations, states=states, max_iterations=iterations)
        result = asdict(hmm_fit)
        inference = hmm_fit.model.infer(observations)
        result["filtered_probabilities"] = inference.filtered_probabilities
        result["offline_smoothed_probabilities"] = inference.smoothed_probabilities
        result["smoothing_uses_future_observations"] = True
        result["duration_diagnostics"] = state_duration_diagnostics(hmm_fit.model.decode(observations), hmm_fit.model.transition)
        return _response(result, scope=scope, source="user-supplied-observations")
    if kind == "microstructure":
        from gig.microstructure import Quote, Trade, lee_ready, match_quotes, order_flow_imbalance
        if set(payload) - {"quotes", "trades", "lag_ns", "max_age_ns"}:
            raise ValueError("invalid event fields")
        quotes, trades = payload.get("quotes", []), payload.get("trades", [])
        if not isinstance(quotes, list) or not isinstance(trades, list) or max(len(quotes), len(trades)) > 1000:
            raise ValueError("event bounds exceeded")
        q = [Quote(**row) for row in quotes]
        t = [Trade(**row) for row in trades]
        matches = match_quotes(t, q, lag_ns=payload.get("lag_ns", 0),
                               max_age_ns=payload.get("max_age_ns", 5_000_000_000))
        return _response({"classifications": [asdict(value) for value in lee_ready(matches)],
                          "order_flow_imbalance": order_flow_imbalance(q)},
                         scope=scope, source="user-supplied-event-data")
    raise ValueError("unsupported analytics kind")
