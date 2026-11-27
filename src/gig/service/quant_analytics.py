"""Registry of stateless quantitative research calculators.

One declaration per analysis drives the HTTP contract, the generated
workbench option and its example. Calculators only transform the request:
they read no account, place no order and persist nothing.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from gig.data import bars
from gig.execution import schedule
from gig.options import models as option_models
from gig.portfolio import construction
from gig.research import cross_section, overfitting
from gig.risk import var_models


@dataclass(frozen=True)
class Analysis:
    key: str
    label: str
    function: Callable[..., dict]
    required: tuple[str, ...]
    optional: tuple[str, ...]
    headline: tuple[tuple[str, str], ...]
    scope: str
    example: dict[str, Any]


RESEARCH_ONLY = " Research estimate on the inputs you supply; no order, allocation or trading approval."
_COV = [[.04, .018, .006, .002], [.018, .0625, .01, .004], [.006, .01, .0225, .003], [.002, .004, .003, .01]]
_RETURNS = [[.012, .008, -.004, .001], [-.02, -.031, .006, .002], [.007, .011, .002, -.001], [.015, .02, -.003, .0],
            [-.009, -.012, .008, .003], [.003, .001, .001, .002], [-.026, -.04, .011, .001], [.018, .022, -.006, -.002],
            [.004, .009, .0, .001], [-.011, -.006, .004, .002], [.021, .027, -.008, .0], [-.005, -.013, .003, .001],
            [.009, .004, .002, -.001], [-.016, -.022, .007, .003], [.011, .017, -.002, .0], [.002, -.003, .001, .001],
            [-.03, -.045, .012, .002], [.014, .019, -.005, -.001], [.006, .01, -.001, .001], [-.008, -.004, .005, .002]]
_SERIES = [row[0] for row in _RETURNS]+[.005, -.013, .008, -.021, .01, .003, -.007, .016, -.002, .004, -.035, .012]
_SMILE_K = [-.4, -.3, -.2, -.1, 0., .1, .2, .3, .4]
_SMILE_W = [.0854, .0721, .0592, .0472, .0378, .0338, .0352, .0392, .0441]
# Black-Scholes values at 25% volatility, spot 100, rate 3%, six months: a synthetic strip.
_STRIKES = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
_PUTS = [.0057, .094, .644, 2.4509, 6.2715, 12.2609, 19.9803, 28.7972, 38.1977, 47.8689, 57.6531]
_CALLS = [40.8989, 31.1362, 21.8351, 13.7908, 7.7603, 3.8986, 1.7669, .7326, .282, .1021, .0352]
_TRADES = [{"timestamp_ns": 1_000_000_000*t, "price": p, "size": s}
           for t, p, s in ((1, 100., 200), (2, 100.02, 100), (4, 99.98, 300), (61, 100.05, 150), (63, 100.1, 250),
                           (118, 100.08, 100), (125, 100.2, 400), (190, 100.15, 200))]
_BARS = [{"timestamp_ns": 86_400_000_000_000*d, "open": o, "high": h, "low": low, "close": c, "volume": v}
         for d, o, h, low, c, v in ((1, 100, 101, 99.5, 100.5, 1000), (2, 100.5, 102, 100, 101.5, 1200),
                                    (3, 101.5, 101.8, 100.9, 101, 900), (4, 101, 103, 100.8, 102.6, 1500),
                                    (5, 102.6, 102.9, 99, 99.4, 0), (8, 99.4, 100.2, 104, 100, 1100))]


def _declare(*rows: Analysis) -> dict[str, Analysis]:
    return {row.key: row for row in rows}


ANALYSES = _declare(
    Analysis("ledoit-wolf", "Covariance shrinkage (Ledoit-Wolf)", construction.ledoit_wolf, ("returns",), (),
             (("Shrinkage intensity", "shrinkage"), ("Condition number", "condition_number")),
             "Shrink a sample covariance toward a scaled identity. 2-2,000 rows of 2-64 assets." + RESEARCH_ONLY,
             {"returns": _RETURNS}),
    Analysis("hierarchical-risk-parity", "Hierarchical risk parity", construction.hierarchical_risk_parity,
             ("covariance",), (), (("Portfolio volatility", "volatility"), ("Inverse-variance volatility", "inverse_variance_volatility")),
             "Cluster assets by correlation and split risk down the tree without inverting the covariance. "
             "2-64 assets, positive definite covariance." + RESEARCH_ONLY, {"covariance": _COV}),
    Analysis("minimum-cvar", "Minimum-CVaR allocation", construction.minimum_cvar, ("scenarios",),
             ("confidence", "max_weight", "minimum_return"),
             (("Tail loss (CVaR)", "cvar"), ("Equal-weight CVaR", "equal_weight_cvar"), ("Expected return", "expected_return")),
             "Long-only fully invested weights minimizing expected tail loss over your return scenarios. "
             "In-sample scenarios only; no out-of-sample claim." + RESEARCH_ONLY,
             {"scenarios": _RETURNS, "confidence": .9, "max_weight": .6}),
    Analysis("efficient-frontier", "Efficient frontier (long-only)", construction.efficient_frontier,
             ("expected_returns", "covariance"), ("points", "max_weight"), (("Assets", "assets"),),
             "Minimum-variance portfolios across attainable returns under a weight cap. Expected returns are "
             "your assumptions, not forecast evidence." + RESEARCH_ONLY,
             {"expected_returns": [.08, .1, .05, .03], "covariance": _COV, "points": 6, "max_weight": .7}),
    Analysis("parametric-var", "Parametric VaR and components", var_models.parametric, ("weights", "covariance"),
             ("mean", "confidence", "horizon"), (("Value at risk", "var"), ("Expected shortfall", "expected_shortfall")),
             "Normal value at risk and expected shortfall with per-asset components that sum to the total. "
             "Losses are positive." + RESEARCH_ONLY,
             {"weights": [400000, 300000, 200000, 100000], "covariance": [[v/252 for v in row] for row in _COV],
              "confidence": .99, "horizon": 10}),
    Analysis("cornish-fisher-var", "Cornish-Fisher VaR", var_models.cornish_fisher, ("returns",), ("confidence",),
             (("Adjusted VaR", "var"), ("Normal VaR", "normal_var"), ("Historical VaR", "historical_var")),
             "Adjust the normal quantile for skewness and kurtosis of one return series (30-5,000 values). The "
             "result states whether the expansion is still a valid quantile function." + RESEARCH_ONLY,
             {"returns": _SERIES, "confidence": .95}),
    Analysis("historical-var-components", "Scenario VaR and tail contributions", var_models.historical_components,
             ("asset_returns", "weights"), ("confidence",),
             (("Value at risk", "var"), ("Expected shortfall", "expected_shortfall"), ("Worst scenario loss", "worst_loss")),
             "Scenario value at risk and expected shortfall with each asset's exact share of the tail loss. "
             "20-5,000 scenarios." + RESEARCH_ONLY,
             {"asset_returns": _RETURNS, "weights": [.4, .3, .2, .1], "confidence": .9}),
    Analysis("monte-carlo-var", "Monte Carlo VaR", var_models.monte_carlo, ("weights", "covariance"),
             ("mean", "confidence", "horizon", "paths", "seed", "degrees_of_freedom"),
             (("Value at risk", "var"), ("Expected shortfall", "expected_shortfall")),
             "Seeded Gaussian or covariance-matched Student-t simulation. Sampling error is not estimated; rerun "
             "with other seeds." + RESEARCH_ONLY,
             {"weights": [.4, .3, .2, .1], "covariance": _COV, "confidence": .99, "paths": 20000, "seed": 7,
              "degrees_of_freedom": 5}),
    Analysis("var-backtest", "VaR exception backtest", var_models.backtest, ("returns", "var_forecasts"),
             ("confidence",),
             (("Exceptions", "exceptions"), ("Expected exceptions", "expected_exceptions"),
              ("Kupiec p-value", "kupiec_p_value"), ("Zone", "basel_style_zone")),
             "Kupiec coverage, Christoffersen independence and a Basel-style zone for forecasts made before each "
             "return. Asymptotic tests with low power in short samples." + RESEARCH_ONLY,
             {"returns": _SERIES, "var_forecasts": [.025]*len(_SERIES), "confidence": .95}),
    Analysis("almgren-chriss", "Optimal execution schedule (Almgren-Chriss)", schedule.almgren_chriss,
             ("shares", "horizon", "intervals", "volatility", "temporary_impact"),
             ("permanent_impact", "fixed_cost", "risk_aversion", "frontier"),
             (("Expected cost", "expected_cost"), ("Cost standard deviation", "cost_standard_deviation"),
              ("Even-schedule expected cost", "uniform_expected_cost")),
             "Mean-variance optimal liquidation under linear impact and an arithmetic price. Impact and volatility "
             "are your assumptions; fills are assumed certain." + RESEARCH_ONLY,
             {"shares": 1000000, "horizon": 5, "intervals": 5, "volatility": .95, "temporary_impact": 2.5e-6,
              "permanent_impact": 2.5e-7, "fixed_cost": .0625, "risk_aversion": 2e-6, "frontier": [0, 1e-6, 1e-5]}),
    Analysis("participation-schedule", "Volume participation schedule", schedule.participation_schedule,
             ("shares", "volumes"), ("max_participation",),
             (("Scheduled shares", "scheduled_shares"), ("Unfilled shares", "unfilled_shares")),
             "Whole-share slices proportional to your volume forecast, capped at a participation rate. Shares the "
             "cap cannot place are reported unfilled." + RESEARCH_ONLY,
             {"shares": 50000, "volumes": [120000, 80000, 60000, 70000, 150000], "max_participation": .1}),
    Analysis("shortfall-simulation", "Implementation shortfall simulation", schedule.simulate_shortfall,
             ("trades", "interval", "volatility", "temporary_impact"),
             ("permanent_impact", "fixed_cost", "paths", "seed"),
             (("Mean shortfall", "mean"), ("Standard error", "standard_error"), ("Analytic expected cost", "analytic_expected_cost")),
             "Seeded simulation of any schedule under the Almgren-Chriss price model, beside its analytic moments. "
             "Simulated, not observed execution." + RESEARCH_ONLY,
             {"trades": [300000, 250000, 200000, 150000, 100000], "interval": 1, "volatility": .95,
              "temporary_impact": 2.5e-6, "permanent_impact": 2.5e-7, "fixed_cost": .0625, "paths": 10000, "seed": 7}),
    Analysis("american-option", "American option (binomial tree)", option_models.american_binomial,
             ("spot", "strike"), ("rate", "div", "vol", "ttm", "call", "steps"),
             (("American value", "price"), ("Early-exercise premium", "early_exercise_premium"),
              ("European Black-Scholes value", "black_scholes_price")),
             "American and European values on one Cox-Ross-Rubinstein tree with the exercise boundary. Continuous "
             "dividend yield only." + RESEARCH_ONLY,
             {"spot": 100, "strike": 105, "rate": .05, "div": .02, "vol": .25, "ttm": .75, "call": False, "steps": 400}),
    Analysis("option-greeks-advanced", "Second-order option Greeks", option_models.higher_order_greeks,
             ("spot", "strike"), ("rate", "div", "vol", "ttm", "call"),
             (("Vanna", "vanna"), ("Volga", "volga"), ("Charm", "charm")),
             "Closed-form vanna, volga, charm, speed and zomma in raw units (per 1.00 volatility, per year)."
             + RESEARCH_ONLY, {"spot": 100, "strike": 110, "rate": .03, "div": .01, "vol": .22, "ttm": .5, "call": True}),
    Analysis("svi-smile", "SVI smile fit and arbitrage checks", option_models.fit_svi,
             ("log_moneyness", "total_variance"), (), (("Fit error (RMSE)", "root_mean_square_error"),),
             "Fit one expiry's total implied variance with raw SVI, then check the butterfly condition, wing slopes "
             "and implied density. A local fit; not an arbitrage-constrained calibration." + RESEARCH_ONLY,
             {"log_moneyness": _SMILE_K, "total_variance": _SMILE_W}),
    Analysis("variance-swap", "Variance swap replication", option_models.variance_swap_strike,
             ("strikes", "put_prices", "call_prices", "forward", "rate", "ttm"), (),
             (("Fair volatility", "fair_volatility"), ("Fair variance", "fair_variance")),
             "Fair variance from an out-of-the-money option strip. Edge-strike shares show how much the truncated "
             "wings matter." + RESEARCH_ONLY,
             {"strikes": _STRIKES, "put_prices": _PUTS, "call_prices": _CALLS, "forward": 101.5113, "rate": .03, "ttm": .5}),
    Analysis("backtest-overfitting", "Probability of backtest overfitting", overfitting.probability_of_backtest_overfitting,
             ("performance",), ("partitions",),
             (("Probability of backtest overfitting", "probability_of_backtest_overfitting"),
              ("Out-of-sample loss probability", "out_of_sample_loss_probability")),
             "How often the best in-sample trial lands in the bottom half out of sample, across all balanced "
             "splits. Measures selection risk among the trials you supply, not profitability." + RESEARCH_ONLY,
             {"performance": _RETURNS, "partitions": 4}),
    Analysis("track-record-length", "Minimum track record length", overfitting.minimum_track_record_length,
             ("sharpe",), ("benchmark", "skew", "kurtosis", "confidence"), (("Observations needed", "whole_observations"),),
             "Observations needed before a per-period Sharpe ratio exceeds a benchmark at a confidence. Assumes one "
             "unselected trial and stationary returns." + RESEARCH_ONLY,
             {"sharpe": .1, "benchmark": 0, "skew": -.5, "kurtosis": 5, "confidence": .95}),
    Analysis("fama-macbeth", "Fama-MacBeth regression", cross_section.fama_macbeth, ("returns", "exposures"), ("lags",),
             (("Periods", "periods"), ("Average R-squared", "average_r_squared")),
             "Period-by-period regression of returns on exposures with Newey-West standard errors on the average "
             "slopes. Balanced panel; exposures must predate each return." + RESEARCH_ONLY,
             {"returns": [row[:4] for row in _RETURNS[:8]],
              "exposures": [[[(a*7+t*3) % 5/5-.4] for a in range(4)] for t in range(8)], "lags": 1}),
    Analysis("trade-bars", "Bars from trade prints", bars.aggregate_trades, ("trades", "kind", "threshold"), (),
             (("Bars", "complete_bars"),),
             "Build time, tick, volume or dollar bars from ordered prints. Prints are never split or reordered; the "
             "last bar is marked incomplete." + RESEARCH_ONLY,
             {"trades": _TRADES, "kind": "TIME", "threshold": 60_000_000_000}),
    Analysis("bar-quality", "Bar data quality report", bars.quality_report, ("bars",),
             ("outlier_threshold", "stale_run", "gap_multiple"),
             (("Errors", "errors"), ("Warnings", "warnings"), ("Structurally usable", "structurally_usable")),
             "Flag impossible prices, inconsistent highs and lows, duplicate or out-of-order timestamps, gaps, "
             "stale closes and return outliers. Calendar-unaware; nothing is repaired." + RESEARCH_ONLY,
             {"bars": _BARS}),
)


def run(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    analysis = ANALYSES[kind]
    if set(payload)-set(analysis.required)-set(analysis.optional) or not set(analysis.required) <= set(payload):
        raise ValueError("request fields do not match the analysis contract")
    result = analysis.function(**payload)
    result["headline"] = [{"label": label, "value": result[key]} for label, key in analysis.headline]
    return result
