"""Data for the six interactive 3D analytical modules.

Synthetic mode computes real analytics (numerical IV inversion, BSM scenario
repricing, rolling correlations, causal Kalman hedges, HMM/GARCH/LightGBM
walk-forward diagnostics) on authored deterministic fixtures. Every payload is
labeled SYNTHETIC. Local mode never fabricates inputs: without admitted evidence
for a module it returns UNAVAILABLE with the reason.

The synthetic outputs are deterministic, so ``scripts/build_viz_fixtures.py``
stores them in ``viz_synthetic.json`` and requests serve that file; the HMM and
GARCH fits take seconds. Tests recompute every module and compare.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from gig.service.contracts import Contract, State
from gig.service.serialization import public_error

PRECOMPUTED = Path(__file__).with_name("viz_synthetic.json")
MODULES = ("vol-surface", "order-book", "risk-terrain", "correlation-network", "stat-arb", "model-evaluation")
_ASOF = "2026-09-30T20:00:00+00:00"


def _fixture() -> tuple[np.ndarray, np.ndarray, list[str]]:
    from gig.service.terminal import _fixture as terminal_fixture
    returns, prices, symbols, _, _ = terminal_fixture()
    return returns, prices, symbols


@lru_cache(maxsize=1)
def vol_surface() -> dict[str, Any]:
    from gig.options import black_scholes, implied_volatility
    spot, rate = 100.0, 0.03
    strikes = [70, 80, 85, 90, 95, 100, 105, 110, 115, 120, 130]
    tenors = [0.08, 0.17, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

    def authored_vol(k: float, t: float, shift: float) -> float:
        m = math.log(k / spot)
        return 0.17 + shift + 0.32 * m * m / math.sqrt(t) - 0.07 * m + 0.015 * math.sqrt(t)

    snapshots = {}
    for label, shift in (("current", 0.0), ("prior", -0.025)):
        points = []
        for t in tenors:
            for k in strikes:
                call = k >= spot
                price = float(black_scholes(spot, k, rate, 0, authored_vol(k, t, shift), t, call)["price"])
                iv = implied_volatility(price, spot, k, rate, 0, t, call)
                greeks = black_scholes(spot, k, rate, 0, float(iv), t, call) if iv is not None else None
                points.append({"strike": k, "moneyness": math.log(k / spot), "tenor_years": t,
                               "option_type": "CALL" if call else "PUT", "authored_price": price,
                               "implied_vol": None if iv is None else float(iv),
                               "vega": None if greeks is None else float(greeks["vega"])})
        snapshots[label] = points
    return {"module": "vol-surface", "title": "Implied volatility surface",
            "axes": {"x": {"key": "moneyness", "label": "Log-moneyness ln(K/S)", "unit": "log ratio"},
                     "y": {"key": "tenor_years", "label": "Maturity", "unit": "years"},
                     "z": {"key": "implied_vol", "label": "Implied volatility", "unit": "annualized decimal"}},
            "grid": {"x": [math.log(k / spot) for k in strikes], "y": tenors},
            "snapshots": snapshots, "comparison": ["current", "prior"],
            "color_scale": {"kind": "sequential", "label": "Implied volatility (annualized)"},
            "method": "Authored BSM prices inverted numerically to IV (European, r=3%, no dividends).",
            "spot": spot, "observed_option_quotes": 0}


@lru_cache(maxsize=1)
def order_book() -> dict[str, Any]:
    rng = np.random.default_rng(90211)
    steps, levels, tick = 36, 15, 0.01
    mid = 50.0
    frames = []
    for t in range(steps):
        mid += float(rng.normal(0, 0.008))
        best_bid = round(mid - tick / 2 - 0.005, 2)
        best_ask = round(best_bid + 0.02, 2)
        bids = [{"price": round(best_bid - i * tick, 2), "quantity": int(200 + 90 * i + rng.integers(0, 160))} for i in range(levels)]
        asks = [{"price": round(best_ask + i * tick, 2), "quantity": int(200 + 90 * i + rng.integers(0, 160))} for i in range(levels)]
        prints = int(rng.poisson(6))
        frames.append({"step": t, "observed_at": f"2026-09-30T14:{t // 6:02d}:{(t % 6) * 10:02d}+00:00",
                       "bids": bids, "asks": asks, "trade_prints": prints,
                       "displayed_imbalance": (sum(b["quantity"] for b in bids) - sum(a["quantity"] for a in asks))
                       / (sum(b["quantity"] for b in bids) + sum(a["quantity"] for a in asks))})
    return {"module": "order-book", "title": "Displayed depth landscape",
            "axes": {"x": {"key": "price", "label": "Price level", "unit": "USD"},
                     "y": {"key": "step", "label": "Snapshot sequence", "unit": "10-second steps"},
                     "z": {"key": "quantity", "label": "Displayed quantity", "unit": "shares"}},
            "frames": frames, "instrument": "SYN-BOOK",
            "layers": {"displayed_depth": "AUTHORED_SYNTHETIC_DEPTH", "trade_activity": "AUTHORED_SYNTHETIC_PRINT_COUNTS",
                       "inferred_liquidity": "NOT_COMPUTED"},
            "color_scale": {"kind": "diverging-side", "bid": "positive green", "ask": "negative red"},
            "method": "Authored deterministic depth snapshots; displayed quantity is not executable liquidity."}


@lru_cache(maxsize=1)
def risk_terrain() -> dict[str, Any]:
    from gig.options import OptionPosition, scenario_grid
    positions = [OptionPosition("SYN001-C105", "SYN001", 105, 0.5, 0.22, 20, source="authored-synthetic-fixture"),
                 OptionPosition("SYN001-P90", "SYN001", 90, 0.5, 0.26, 15, call=False, source="authored-synthetic-fixture"),
                 OptionPosition("SYN001-C120", "SYN001", 120, 1.0, 0.20, 10, source="authored-synthetic-fixture")]
    spot_shocks = [round(-0.3 + 0.05 * i, 2) for i in range(13)]
    vol_shocks = [round(-0.09 + 0.03 * i, 2) for i in range(10)]
    horizons = {}
    for label, elapsed in (("today", 0.0), ("plus_30_days", 30 / 365)):
        grid = scenario_grid(positions, spot=100, spot_shocks=spot_shocks, volatility_shocks=vol_shocks, rate=0.03,
                             elapsed_years=elapsed)
        horizons[label] = [{"spot_shock": s["spot_shock"], "volatility_shock": s["volatility_shock"],
                            "model_pnl": s["model_pnl"]} for s in grid["scenarios"]]
    premiums: list[dict[str, Any]] = []
    from gig.options import black_scholes
    for position in positions:
        value = float(black_scholes(100, position.strike, 0.03, 0, position.volatility, position.ttm, position.call)["price"]) \
            * position.quantity * position.multiplier
        premiums.append({"instrument": position.instrument, "premium_value": value})
    total = sum(row["premium_value"] for row in premiums)
    for row in premiums:
        row["share"] = row["premium_value"] / total
    return {"module": "risk-terrain", "title": "Portfolio stress terrain",
            "axes": {"x": {"key": "spot_shock", "label": "Spot shock", "unit": "relative decimal"},
                     "y": {"key": "volatility_shock", "label": "Volatility shock", "unit": "absolute decimal"},
                     "z": {"key": "model_pnl", "label": "Model P&L", "unit": "USD"}},
            "grid": {"x": spot_shocks, "y": vol_shocks}, "horizons": horizons, "comparison": ["today", "plus_30_days"],
            "risk_limit": {"loss_limit": -0.5 * total, "label": "Authored loss limit (50% of premium)"},
            "concentration": premiums, "long_premium_total": total,
            "color_scale": {"kind": "diverging", "label": "Model P&L (green gain, red loss)"},
            "method": "Full BSM repricing of long, fully paid European options; scenario P&L is a model difference."}


@lru_cache(maxsize=1)
def correlation_network() -> dict[str, Any]:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    returns, _, symbols = _fixture()
    windows = {}
    for label, end, length in (("latest_60", 180, 60), ("latest_120", 180, 120), ("prior_60", 120, 60)):
        sample = returns[end - length:end]
        corr = np.corrcoef(sample, rowvar=False)
        distance = np.sqrt(np.clip(2 * (1 - corr), 0, None))
        np.fill_diagonal(distance, 0)
        n = len(symbols)
        j = np.eye(n) - np.ones((n, n)) / n
        b = -0.5 * j @ (distance ** 2) @ j
        values, vectors = np.linalg.eigh(b)
        order = np.argsort(values)[::-1][:3]
        coords = vectors[:, order] * np.sqrt(np.clip(values[order], 0, None))
        signs = np.sign(coords[np.argmax(np.abs(coords), axis=0), range(3)])
        coords = coords * np.where(signs == 0, 1, signs)
        clusters = fcluster(linkage(squareform(distance, checks=False), "average"), t=3, criterion="maxclust")
        windows[label] = {"observations": length, "end_index": end,
                          "nodes": [{"symbol": s, "x": float(coords[i, 0]), "y": float(coords[i, 1]), "z": float(coords[i, 2]),
                                     "cluster": int(clusters[i]), "volatility": float(sample[:, i].std(ddof=1))}
                                    for i, s in enumerate(symbols)],
                          "edges": [{"a": symbols[i], "b": symbols[k], "rho": float(corr[i, k])}
                                    for i in range(n) for k in range(i + 1, n)],
                          "standard_error_at_zero": 1 / math.sqrt(length - 3)}
    return {"module": "correlation-network", "title": "Asset correlation network",
            "axes": {"x": {"label": "MDS dimension 1"}, "y": {"label": "MDS dimension 2"}, "z": {"label": "MDS dimension 3"}},
            "windows": windows, "comparison": ["latest_60", "prior_60"],
            "color_scale": {"kind": "diverging", "label": "Pearson correlation (blue positive, yellow negative)"},
            "method": "Pearson correlation of daily log returns; classical MDS on sqrt(2(1-rho)); average-linkage clusters.",
            "interpretation": "Correlation describes co-movement in the sample; it does not imply causation."}


@lru_cache(maxsize=1)
def stat_arb() -> dict[str, Any]:
    from gig.ml.regimes import fit_gaussian_hmm
    from gig.research.stat_arb import kalman_hedge
    _, prices, symbols = _fixture()
    hedge = kalman_hedge(prices[:, 0], prices[:, 1])
    innovation = np.asarray(hedge["innovation"], dtype=float)
    variance = np.asarray(hedge["innovation_variance"], dtype=float)
    z = innovation / np.sqrt(variance)
    changes = np.diff(z)[:, None]
    hmm = fit_gaussian_hmm(changes, states=2, max_iterations=100)
    filtered = np.asarray(hmm.model.infer(changes).filtered_probabilities)
    regime = [None] + [int(np.argmax(row)) for row in filtered]
    beta = np.asarray(hedge["beta"], dtype=float)
    points = [{"t": i, "beta": float(beta[i]), "zscore": float(z[i]), "regime": regime[i],
               "regime_probability": None if i == 0 else float(filtered[i - 1].max())} for i in range(len(z))]
    return {"module": "stat-arb", "title": "Statistical arbitrage landscape", "pair": [symbols[0], symbols[1]],
            "axes": {"x": {"key": "t", "label": "Observation", "unit": "trading days"},
                     "y": {"key": "beta", "label": "Causal Kalman hedge elasticity", "unit": "log-price elasticity"},
                     "z": {"key": "zscore", "label": "Standardized innovation", "unit": "z-score"}},
            "points": points, "burn_in": 20, "regime_model": "2-state Gaussian HMM on z-score changes (filtered, causal)",
            "color_scale": {"kind": "categorical", "label": "Filtered regime"},
            "method": "Causal Kalman [alpha, beta] filter; pre-observation innovations; no smoothing.",
            "economic_validation": "UNAVAILABLE"}


@lru_cache(maxsize=1)
def model_evaluation() -> dict[str, Any]:
    from gig.ml.regimes import fit_gaussian_hmm
    from gig.ml.state_space import kalman_filter
    from gig.ml.volatility import fit_garch11
    returns, prices, symbols = _fixture()
    series = returns[:, 0]
    folds = [(60, 90), (90, 120), (120, 150), (150, 180)]
    rows: list[dict[str, Any]] = []
    for f, (train_end, test_end) in enumerate(folds):
        train, test = series[:train_end], series[train_end:test_end]
        fit = fit_garch11(train, max_iterations=200)
        variances = np.asarray(fit.model.conditional_variances(series[:test_end]))[train_end:test_end]
        squared = np.maximum(test ** 2, 1e-12)

        def qlike(h: np.ndarray, sq: np.ndarray = squared) -> float:
            return float(np.mean(sq / h - np.log(sq / h) - 1))
        rows.append({"model": "GARCH(1,1)", "family": "variance_qlike", "fold": f, "value": qlike(variances),
                     "baseline": qlike(np.full(len(test), train.var(ddof=1))), "lower_is_better": True})
        kf = kalman_filter(series[:test_end, None], [[1]], [[1]], [[1e-6]], [[float(train.var())]], [0], [[1e-4]])
        predicted = np.asarray(kf.predicted_mean)[train_end:test_end, 0]
        rows.append({"model": "Kalman local level", "family": "forecast_mse", "fold": f,
                     "value": float(np.mean((test - predicted) ** 2)), "baseline": float(np.mean(test ** 2)),
                     "lower_is_better": True})
        hmm2 = fit_gaussian_hmm(train[:, None], states=2, max_iterations=100)
        hmm1 = fit_gaussian_hmm(train[:, None], states=1, max_iterations=10)
        ll2 = hmm2.model.infer(series[:test_end, None]).log_likelihood - hmm2.model.infer(train[:, None]).log_likelihood
        ll1 = hmm1.model.infer(series[:test_end, None]).log_likelihood - hmm1.model.infer(train[:, None]).log_likelihood
        rows.append({"model": "Gaussian HMM (2 state)", "family": "regime_loglik_per_obs", "fold": f,
                     "value": float(ll2 / len(test)), "baseline": float(ll1 / len(test)), "lower_is_better": False})
    try:
        import lightgbm as lgb
        from scipy.stats import spearmanr
        lookback = np.stack([np.log(prices[20:-1] / prices[:-21]), returns[20:-1], np.abs(returns[16:-5])], axis=-1)
        target = returns[21:]
        for f, (train_end, test_end) in enumerate(folds):
            x_train = lookback[:train_end - 21].reshape(-1, 3)
            y_train = target[:train_end - 21].reshape(-1)
            model = lgb.LGBMRegressor(n_estimators=60, learning_rate=0.05, num_leaves=7, min_child_samples=20,
                                      random_state=7, deterministic=True, n_jobs=1, verbose=-1)
            model.fit(x_train, y_train)
            ics, base = [], []
            for day in range(train_end - 21, test_end - 21):
                prediction = model.predict(lookback[day])
                ics.append(spearmanr(prediction, target[day]).statistic)
                base.append(spearmanr(lookback[day][:, 0], target[day]).statistic)
            rows.append({"model": "LightGBM ranker", "family": "ranking_ic", "fold": f, "value": float(np.nanmean(ics)),
                         "baseline": float(np.nanmean(base)), "lower_is_better": False})
        lightgbm_state = "COMPUTED"
    except ImportError:
        lightgbm_state = "DEPENDENCY_UNAVAILABLE"
    return {"module": "model-evaluation", "title": "Model evaluation landscape",
            "axes": {"x": {"key": "model", "label": "Model"}, "y": {"key": "fold", "label": "Walk-forward fold"},
                     "z": {"key": "value", "label": "Metric (family-specific)"}},
            "rows": rows, "families": {"variance_qlike": "Variance forecast QLIKE (lower better)",
                                       "forecast_mse": "One-step return forecast MSE (lower better)",
                                       "regime_loglik_per_obs": "Out-of-sample log-likelihood per observation (higher better)",
                                       "ranking_ic": "Cross-sectional Spearman IC (higher better)"},
            "not_computed": {"BiLSTM": "Training is run explicitly in the CPU sequence research workflow, not in this view.",
                             "LightGBM": None if lightgbm_state == "COMPUTED" else lightgbm_state},
            "folds": [{"fold": i, "train_end": a, "test_end": b} for i, (a, b) in enumerate(folds)],
            "color_scale": {"kind": "diverging", "label": "Model minus training-only baseline (green better)"},
            "method": "Expanding-window folds on authored synthetic returns; every family is scored separately.",
            "interpretation": "Forecast, ranking and regime metrics are not trading profitability or economic validation."}


BUILDERS = {"vol-surface": vol_surface, "order-book": order_book, "risk-terrain": risk_terrain,
            "correlation-network": correlation_network, "stat-arb": stat_arb, "model-evaluation": model_evaluation}


@lru_cache(maxsize=1)
def precomputed() -> dict[str, Any]:
    try:
        return dict(json.loads(PRECOMPUTED.read_text())["modules"])
    except FileNotFoundError:
        return {}


def synthetic(name: str) -> dict[str, Any]:
    stored = precomputed().get(name)
    return dict(stored) if stored is not None else BUILDERS[name]()


def module(name: str, mode: str) -> Contract:
    if name not in BUILDERS or mode not in ("local", "synthetic"):
        raise ValueError("unsupported module or mode")
    scope = f"viz.{name}"
    if mode == "local":
        return Contract.admit(State.UNAVAILABLE, None, source="local", classification="UNAVAILABLE", scope=scope,
                              error=public_error("DATA_UNAVAILABLE"))
    return Contract.admit(State.AVAILABLE, synthetic(name), source="deterministic-synthetic-v1",
                          classification="SYNTHETIC", as_of=_ASOF, scope=scope,
                          evidence_state="RESEARCH_OUTPUT", readiness="RESEARCH_ONLY")
