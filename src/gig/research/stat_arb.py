"""Bounded pair research with fit/test separation and honest optional inference.

No broker execution, economic validation, or in-sample significance promotion.
Engle–Granger/ADF require statsmodels; descriptive OLS is not a replacement test.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from inspect import signature
from itertools import combinations
from numbers import Integral
from typing import NotRequired, TypedDict

import numpy as np


class PairEvaluation(TypedDict):
    state: str
    spread: np.ndarray
    zscore: np.ndarray | None
    model_digest: str
    trading_approved: bool
    classification: str
    beta_units: str


class SignificanceResult(TypedDict):
    state: str
    statistic: float | None
    pvalue: float | None
    code: NotRequired[str]
    method: NotRequired[str]
    sample_count: NotRequired[int]
    critical_values: NotRequired[dict[str, float]]
    null: NotRequired[str]
    multiple_testing_adjusted: NotRequired[bool]
    trading_approved: NotRequired[bool]
    used_lag: NotRequired[int]
    nobs: NotRequired[int]
    post_selection_inference_validated: NotRequired[bool]


class PairFold(TypedDict):
    training_start: int
    training_end_exclusive: int
    test_start: int
    test_end_exclusive: int
    model_digest: str
    alpha: float
    beta: float


class WalkForwardPairResult(TypedDict):
    state: str
    folds: list[PairFold]
    evaluation_start: int
    spread: list[float]
    zscore: list[float | None]
    classification: str
    transaction_costs_applied: bool
    economic_validation: bool
    trading_approved: bool


def _series(
    value: Sequence[float], name: str, *, positive: bool = False, min_size: int = 3
) -> np.ndarray:
    out = np.array(value, dtype=float, copy=True)
    if out.ndim != 1 or len(out) < min_size or len(out) > 1_000_000 or not np.isfinite(out).all():
        raise ValueError(
            f"{name} requires a bounded finite one-dimensional series with at least {min_size} observations"
        )
    if positive and np.any(out <= 0):
        raise ValueError(f"{name} requires positive prices")
    return out


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _paired(y, x, *, positive=True, min_size=3) -> tuple[np.ndarray, np.ndarray]:
    y = _series(y, "dependent", positive=positive, min_size=min_size)
    x = _series(x, "independent", positive=positive, min_size=min_size)
    if y.shape != x.shape:
        raise ValueError("paired series must have identical shapes and prior alignment")
    return y, x


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    # Centering protects intercept estimation and rejects unidentifiable hedge.
    xc, yc = x - x.mean(), y - y.mean()
    denominator = float(xc @ xc)
    if denominator <= np.finfo(float).eps * max(1.0, float(x @ x)):
        raise ValueError("independent series lacks identifiable variation")
    beta = float(xc @ yc) / denominator
    alpha = float(y.mean() - beta * x.mean())
    if not math.isfinite(alpha) or not math.isfinite(beta):
        raise ValueError("pair regression numerical overflow")
    return alpha, beta


@dataclass(frozen=True, slots=True)
class PairModel:
    dependent: str
    independent: str
    alpha: float
    beta: float
    spread_mean: float
    spread_std: float
    training_rows: int
    training_digest: str
    log_prices: bool
    evidence_state: str = "RESEARCH_FIT_ONLY"
    trading_approved: bool = False


def fit_pair(
    dependent_prices,
    independent_prices,
    *,
    dependent: str = "Y",
    independent: str = "X",
    training_rows: int | None = None,
    log_prices: bool = True,
) -> PairModel:
    """OLS y=alpha+beta*x fitted only to prefix [:training_rows].

    Inputs must already be point-in-time aligned; rows after the training prefix
    cannot influence parameters or fingerprint. Log-price beta is an elasticity,
    not a share hedge ratio; level-price beta uses underlying price units.
    """
    raw_y = np.asarray(dependent_prices)
    raw_x = np.asarray(independent_prices)
    if raw_y.ndim != 1 or raw_x.shape != raw_y.shape or len(raw_y) > 1_000_000:
        raise ValueError("bounded aligned one-dimensional price arrays required")
    if not isinstance(log_prices, bool):
        raise ValueError("log_prices must be boolean")
    if (
        any(not isinstance(v, str) or not v.strip() for v in (dependent, independent))
        or dependent == independent
    ):
        raise ValueError("distinct nonempty instrument identities are required")
    n = len(raw_y) if training_rows is None else _integer(training_rows, "training_rows", 3)
    if n > len(raw_y):
        raise ValueError("training_rows exceeds available observations")
    y, x = _paired(raw_y[:n], raw_x[:n])
    if log_prices:
        y, x = np.log(y), np.log(x)
    alpha, beta = _ols(y, x)
    spread = y - alpha - beta * x
    std = float(np.std(spread, ddof=1))
    if not math.isfinite(std):
        raise ValueError("spread variance overflow")
    digest = hashlib.sha256()
    digest.update((dependent + "\0" + independent + "\0" + str(log_prices)).encode("utf-8"))
    digest.update(np.asarray(np.column_stack((y, x)), dtype="<f8").tobytes())
    return PairModel(
        dependent,
        independent,
        alpha,
        beta,
        float(spread.mean()),
        std,
        n,
        digest.hexdigest(),
        log_prices,
    )


def evaluate_pair(model: PairModel, dependent_prices, independent_prices) -> PairEvaluation:
    """Evaluate fixed fitted coefficients; never refit using evaluation rows."""
    y, x = _paired(dependent_prices, independent_prices, min_size=1)
    if model.log_prices:
        y, x = np.log(y), np.log(x)
    spread = y - model.alpha - model.beta * x
    if not np.isfinite(spread).all():
        raise ValueError("spread arithmetic overflow")
    z = (
        None
        if model.spread_std <= np.finfo(float).eps
        else (spread - model.spread_mean) / model.spread_std
    )
    return {
        "state": "AVAILABLE" if z is not None else "UNKNOWN",
        "spread": spread,
        "zscore": z,
        "model_digest": model.training_digest,
        "trading_approved": False,
        "classification": "RESEARCH_ESTIMATE",
        "beta_units": "log_price_elasticity" if model.log_prices else "price_ratio",
    }


def mean_reversion_diagnostics(spread, *, step_years: float = 1 / 252) -> dict[str, object]:
    """Descriptive AR(1) fit and exact discretization-derived OU half-life.

    phi in (0,1) is required for conventional continuous OU interpretation.
    This is not a stationarity significance test. Half-life is not evidence
    of a profitable tradable strategy.
    """
    values = _series(spread, "spread", min_size=5)
    if not math.isfinite(step_years) or step_years <= 0:
        raise ValueError("step_years must be finite and positive")
    try:
        intercept, phi = _ols(values[1:], values[:-1])
    except ValueError:
        return {
            "state": "UNKNOWN",
            "phi": None,
            "half_life_steps": None,
            "half_life_years": None,
            "stationarity_tested": False,
            "sample_count": len(values),
        }
    residual = values[1:] - intercept - phi * values[:-1]
    sigma = float(np.std(residual, ddof=2))
    compatible = 0 < phi < 1
    half_life = math.log(0.5) / math.log(phi) if compatible else None
    return {
        "state": "AVAILABLE",
        "phi": phi,
        "ar_intercept": intercept,
        "residual_std": sigma,
        "ou_compatible": compatible,
        "half_life_steps": half_life,
        "half_life_years": None if half_life is None else half_life * step_years,
        "long_run_mean": intercept / (1 - phi) if compatible else None,
        "stationarity_tested": False,
        "sample_count": len(values),
        "classification": "DESCRIPTIVE_ESTIMATE",
    }


def cointegration_test(
    dependent_prices, independent_prices, *, log_prices: bool = True
) -> SignificanceResult:
    """Engle–Granger with constant, AIC lag selection; optional statsmodels.

    Null hypothesis is no cointegration. P-values are unadjusted until a caller
    accounts for pair screening. Missing dependency is reported, never replaced
    with correlation or plain residual regression p-values.
    """
    y, x = _paired(dependent_prices, independent_prices, min_size=30)
    if not isinstance(log_prices, bool):
        raise ValueError("log_prices must be boolean")
    if log_prices:
        y, x = np.log(y), np.log(x)
    try:
        from statsmodels.tsa.stattools import coint
    except ImportError:
        return {
            "state": "UNAVAILABLE",
            "code": "DEPENDENCY_STATSMODELS_MISSING",
            "statistic": None,
            "pvalue": None,
            "method": "ENGLE_GRANGER",
            "sample_count": len(y),
        }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            statistic, pvalue, critical = coint(y, x, trend="c", autolag="aic")
        except Exception:
            return {
                "state": "ERROR",
                "code": "COINTEGRATION_FIT_FAILED",
                "statistic": None,
                "pvalue": None,
            }
    if caught or not np.isfinite([statistic, pvalue, *critical]).all():
        return {
            "state": "ERROR",
            "code": "COINTEGRATION_NUMERICAL_DIAGNOSTIC",
            "statistic": None,
            "pvalue": None,
        }
    return {
        "state": "AVAILABLE",
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "critical_values": dict(zip(("1%", "5%", "10%"), map(float, critical), strict=True)),
        "method": "ENGLE_GRANGER",
        "null": "NO_COINTEGRATION",
        "sample_count": len(y),
        "multiple_testing_adjusted": False,
        "trading_approved": False,
    }


def adf_test(spread) -> SignificanceResult:
    """ADF with constant and AIC lags; valid null is a unit root, not no alpha."""
    values = _series(spread, "spread", min_size=30)
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return {
            "state": "UNAVAILABLE",
            "code": "DEPENDENCY_STATSMODELS_MISSING",
            "statistic": None,
            "pvalue": None,
        }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            # statsmodels 0.15 warns about a future tuple-result default. Its
            # explicit compatibility option avoids that nonnumerical warning;
            # older supported versions do not expose the option. Numerical
            # warnings still make the result fail closed below.
            compatibility = (
                {"result_object": False}
                if "result_object" in signature(adfuller).parameters
                else {}
            )
            statistic, pvalue, used_lag, nobs, critical, _ = adfuller(
                values, regression="c", autolag="AIC", **compatibility
            )
        except Exception:
            return {"state": "ERROR", "code": "ADF_FIT_FAILED", "statistic": None, "pvalue": None}
    if caught or not np.isfinite([statistic, pvalue, *critical.values()]).all():
        return {
            "state": "ERROR",
            "code": "ADF_NUMERICAL_DIAGNOSTIC",
            "statistic": None,
            "pvalue": None,
        }
    return {
        "state": "AVAILABLE",
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "used_lag": int(used_lag),
        "nobs": int(nobs),
        "critical_values": critical,
        "null": "UNIT_ROOT",
        "post_selection_inference_validated": False,
    }


def select_pairs(
    prices, *, training_rows: int, alpha: float = 0.05, log_prices: bool = True
) -> dict[str, object]:
    """Bounded all-pair screening with Bonferroni family-wise correction.

    prices is a chronological uniquely indexed DataFrame. Only its training
    prefix is read. Point-in-time universe membership remains caller authority.
    Missing significance backend leaves every pair unselected.
    """
    training_rows = _integer(training_rows, "training_rows", 30)
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    if not hasattr(prices, "columns") or not hasattr(prices, "iloc"):
        raise ValueError("prices must be a chronologically indexed DataFrame")
    if (
        not prices.index.is_unique
        or not prices.index.is_monotonic_increasing
        or not prices.columns.is_unique
        or len(prices.columns) > 64
        or training_rows > len(prices)
    ):
        raise ValueError("bounded unique chronological data required")
    prefix = prices.iloc[:training_rows]
    pair_count = len(prefix.columns) * (len(prefix.columns) - 1) // 2
    rows = []
    for dependent, independent in combinations(prefix.columns, 2):
        y = prefix[dependent].to_numpy(dtype=float)
        x = prefix[independent].to_numpy(dtype=float)
        try:
            model = fit_pair(
                y, x, dependent=str(dependent), independent=str(independent), log_prices=log_prices
            )
            test = cointegration_test(y, x, log_prices=log_prices)
            pvalue = test["pvalue"]
            adjusted = None if pvalue is None else min(1.0, pvalue * pair_count)
            rows.append(
                {
                    "dependent": str(dependent),
                    "independent": str(independent),
                    "state": test["state"],
                    "code": test.get("code"),
                    "alpha": model.alpha,
                    "beta": model.beta,
                    "model_digest": model.training_digest,
                    "pvalue": test.get("pvalue"),
                    "bonferroni_pvalue": adjusted,
                    "selected": adjusted is not None and adjusted < alpha,
                }
            )
        except ValueError:
            rows.append(
                {
                    "dependent": str(dependent),
                    "independent": str(independent),
                    "state": "ERROR",
                    "code": "PAIR_INPUT_INVALID_OR_UNIDENTIFIABLE",
                    "selected": False,
                }
            )
    return {
        "state": "AVAILABLE" if any(r["state"] == "AVAILABLE" for r in rows) else "UNAVAILABLE",
        "pairs": rows,
        "family_size": pair_count,
        "training_rows": training_rows,
        "correction": "BONFERRONI_ALL_SCREENED_PAIRS",
        "trading_approved": False,
    }


def kalman_hedge(
    dependent_prices,
    independent_prices,
    *,
    process_variance: float = 1e-5,
    observation_variance: float = 1e-2,
    log_prices: bool = True,
    initial_alpha: float = 0.0,
    initial_beta: float = 1.0,
) -> dict[str, object]:
    """Causal random-walk [intercept,beta] filter; no full-sequence smoothing.

    Returns pre-observation innovation for research signal construction. Initial
    coefficients are caller assumptions, never fit on future observations.
    """
    from gig.ml.state_space import kalman_filter

    y, x = _paired(dependent_prices, independent_prices)
    if not isinstance(log_prices, bool):
        raise ValueError("log_prices must be boolean")
    if log_prices:
        y, x = np.log(y), np.log(x)
    if (
        not all(
            math.isfinite(v)
            for v in (process_variance, observation_variance, initial_alpha, initial_beta)
        )
        or process_variance < 0
        or observation_variance <= 0
    ):
        raise ValueError(
            "finite means, nonnegative process variance, positive observation variance required"
        )
    h = np.stack((np.ones(len(x)), x), axis=1)[:, None, :]
    result = kalman_filter(
        y[:, None],
        np.eye(2),
        h,
        np.eye(2) * process_variance,
        np.array([[observation_variance]]),
        [initial_alpha, initial_beta],
        np.eye(2),
    )
    return {
        "state": "AVAILABLE",
        "alpha": result.filtered_mean[:, 0],
        "beta": result.filtered_mean[:, 1],
        "innovation": result.innovations[:, 0],
        "innovation_variance": result.innovation_covariance[:, 0, 0],
        "evidence_state": "CAUSAL_FILTERED_RESEARCH",
        "uses_future_observations": False,
        "beta_units": "log_price_elasticity" if log_prices else "price_ratio",
        "trading_approved": False,
    }


def walk_forward_pair(
    dependent_prices,
    independent_prices,
    *,
    training_window: int,
    test_window: int = 20,
    log_prices: bool = True,
) -> WalkForwardPairResult:
    """Rolling fit-before-test spread/z-score evaluation; no backtest P&L claim."""
    y, x = _paired(dependent_prices, independent_prices)
    training_window = _integer(training_window, "training_window", 5)
    test_window = _integer(test_window, "test_window")
    if training_window >= len(y):
        raise ValueError("training_window leaves no evaluation observations")
    if len(range(training_window, len(y), test_window)) > 1000:
        raise ValueError("walk-forward research exceeds 1000 fold budget")
    spreads: list[float] = []
    zscores: list[float | None] = []
    folds: list[PairFold] = []
    for start in range(training_window, len(y), test_window):
        end = min(start + test_window, len(y))
        train_start = start - training_window
        model = fit_pair(y[train_start:start], x[train_start:start], log_prices=log_prices)
        evaluation = evaluate_pair(model, y[start:end], x[start:end])
        spreads.extend(evaluation["spread"].tolist())
        zscores.extend(
            [None] * (end - start)
            if evaluation["zscore"] is None
            else evaluation["zscore"].tolist()
        )
        folds.append(
            {
                "training_start": train_start,
                "training_end_exclusive": start,
                "test_start": start,
                "test_end_exclusive": end,
                "model_digest": model.training_digest,
                "alpha": model.alpha,
                "beta": model.beta,
            }
        )
    return {
        "state": "AVAILABLE",
        "folds": folds,
        "evaluation_start": training_window,
        "spread": spreads,
        "zscore": zscores,
        "classification": "OUT_OF_SAMPLE_RESEARCH_ESTIMATE",
        "transaction_costs_applied": False,
        "economic_validation": False,
        "trading_approved": False,
    }


def backtest_pair(
    dependent_prices,
    independent_prices,
    *,
    training_window: int,
    test_window: int = 20,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    cost_bps: float = 10.0,
    gross_limit: float = 1.0,
    log_prices: bool = True,
    eligibility=None,
) -> dict[str, object]:
    """Bounded OOS mean-reversion rehearsal with next-close execution lag.

    At close t, a fixed training-prefix model generates a target for close t+1.
    That target earns the following interval's arithmetic returns. Targets use
    constant initial-capital notional, not reinvested/compounded wealth. Turnover
    includes price drift of prior holdings. Costs are an explicit linear charge;
    no borrow, impact, spread, available volume, or economic validation is assumed.
    An optional caller-admitted eligibility mask can block new exposure. It is
    not an independently validated regime or liquidity classifier.
    """
    y, x = _paired(dependent_prices, independent_prices)
    if len(y) > 20_000:
        raise ValueError("pair backtest exceeds 20000 observations")
    parameters = (entry_z, exit_z, cost_bps, gross_limit)
    if (
        any(isinstance(v, (bool, np.bool_)) or not math.isfinite(v) for v in parameters)
        or not 0 <= exit_z < entry_z
        or cost_bps < 0
        or not 0 < gross_limit <= 2
    ):
        raise ValueError("require 0<=exit_z<entry_z, nonnegative costs and gross in (0,2]")
    research = walk_forward_pair(
        y, x, training_window=training_window, test_window=test_window, log_prices=log_prices
    )
    allowed = np.ones(len(y), dtype=bool) if eligibility is None else np.asarray(eligibility)
    if allowed.shape != y.shape or allowed.dtype != np.bool_:
        raise ValueError("eligibility must be a same-length explicit boolean mask")
    start = int(research["evaluation_start"])
    price_matrix = np.column_stack((y, x))
    desired = np.zeros(2)
    previous = np.zeros(2)
    previous_asset_returns = np.zeros(2)
    direction = 0
    rows = []
    folds = iter(research["folds"])
    fold = next(folds)
    for decision in range(start, len(y) - 2):
        while decision >= fold["test_end_exclusive"]:
            fold = next(folds)
        score = research["zscore"][decision - start]
        if not allowed[decision] or score is None:
            direction = 0
        elif direction == 0:
            direction = -1 if score >= entry_z else (1 if score <= -entry_z else 0)
        elif abs(score) <= exit_z or score * direction > 0:
            direction = 0
        beta = float(fold["beta"])
        if direction:
            # Log beta is a return elasticity. Level beta is a share hedge and
            # must be translated into dollar exposure using decision-time prices.
            ratio = beta if log_prices else beta * x[decision] / y[decision]
            desired = gross_limit * direction * np.array([1.0, -ratio]) / (1 + abs(ratio))
        else:
            desired = np.zeros(2)
        execution = decision + 1
        asset_returns = price_matrix[execution + 1] / price_matrix[execution] - 1
        drifted = previous * (1 + previous_asset_returns)
        turnover = float(np.abs(desired - drifted).sum())
        gross_return = float(desired @ asset_returns)
        cost = turnover * cost_bps / 10_000
        net = gross_return - cost
        if not all(math.isfinite(v) for v in (turnover, gross_return, cost, net)):
            raise ValueError("pair backtest valuation overflow")
        rows.append(
            {
                "decision_row": decision,
                "execution_row": execution,
                "return_end_row": execution + 1,
                "training_end_exclusive": fold["training_end_exclusive"],
                "model_digest": fold["model_digest"],
                "decision_zscore": score,
                "weights": desired.tolist(),
                "gross_return": gross_return,
                "turnover": turnover,
                "cost": cost,
                "net_return": net,
            }
        )
        previous, previous_asset_returns = desired.copy(), asset_returns.copy()
    return {
        "state": "AVAILABLE" if rows else "UNAVAILABLE",
        "rows": rows,
        "cost_bps": cost_bps,
        "gross_limit": gross_limit,
        "remaining_weights": previous.tolist(),
        "terminal_liquidation_applied": False,
        "execution_assumption": "ONE_BAR_LAG_TO_NEXT_CLOSE",
        "capital_assumption": "CONSTANT_INITIAL_CAPITAL_NOTIONAL_NO_COMPOUNDING",
        "transaction_costs_applied": True,
        "borrow_and_capacity_validated": False,
        "classification": "OUT_OF_SAMPLE_RESEARCH_ESTIMATE",
        "economic_validation": False,
        "trading_approved": False,
    }
