"""Options research with explicit model/market separation and numerical bounds.

Prices use currency per underlying unit; times use years (caller day-count);
volatility is annualized decimal. No market feed or proprietary metrics implied.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import brentq
from scipy.stats import norm

from gig.assets.options import black_scholes as _legacy_black_scholes


class UnavailableExposure(TypedDict):
    state: Literal["UNAVAILABLE"]
    classification: Literal["MODEL_ESTIMATE"]
    positions: list[dict[str, float | str]]


class AvailableExposure(TypedDict):
    state: Literal["AVAILABLE"]
    classification: Literal["MODEL_ESTIMATE"]
    underlying: str
    currency: str
    spot: float
    positions: list[dict[str, float | str]]
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    units: dict[str, str]


class ScenarioRow(TypedDict):
    spot_shock: float
    volatility_shock: float
    spot: float
    value: float
    model_pnl: float


class ScenarioBase(TypedDict):
    state: str
    classification: str
    scenarios: list[ScenarioRow]


class ScenarioResult(ScenarioBase, total=False):
    base_value: float
    underlying: str
    currency: str
    elapsed_years: float
    spot_shock_unit: str
    volatility_shock_unit: str
    sources: list[str]


def _finite(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric")
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return value


def _inputs(spot: float, strike: float, rate: float, div: float, ttm: float, call: bool) -> tuple[float, ...]:
    if not isinstance(call, bool):
        raise ValueError("call must be boolean")
    return (_finite(spot, "spot", positive=True), _finite(strike, "strike", positive=True),
            _finite(rate, "rate"), _finite(div, "div"), _finite(ttm, "ttm", positive=True))


def _discounted(spot: float, strike: float, rate: float, div: float, ttm: float) -> tuple[float, float]:
    try:
        discounted_spot = spot * math.exp(-div * ttm)
        discounted_strike = strike * math.exp(-rate * ttm)
    except OverflowError as exc:
        raise ValueError("discount factors outside numerical bounds") from exc
    if not all(math.isfinite(x) and x > 0 for x in (discounted_spot, discounted_strike)):
        raise ValueError("discounted values outside numerical bounds")
    return discounted_spot, discounted_strike


def black_scholes(
    spot: float, strike: float, rate: float = 0.0, div: float = 0.0,
    vol: float = 0.2, ttm: float = 1.0, call: bool = True,
) -> dict[str, float | str]:
    """Reuse existing BSM kernel after strict admission; add rho and units.

    Vega/rho are per one percentage point; theta per calendar day. Expired and
    zero-vol contracts are excluded because classical smooth Greeks do not have
    one uniform boundary convention. Results are MODEL_ESTIMATE, never observed.
    """
    spot, strike, rate, div, ttm = _inputs(spot, strike, rate, div, ttm, call)
    vol = _finite(vol, "vol", positive=True)
    _, discounted_strike = _discounted(spot, strike, rate, div, ttm)
    try:
        result = _legacy_black_scholes(spot, strike, rate, div, vol, ttm, call)
        rho = (ttm * discounted_strike * norm.cdf(result["d2"]) if call
               else -ttm * discounted_strike * norm.cdf(-result["d2"])) / 100.0
        result["rho"] = float(rho)
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError("BSM inputs outside numerical bounds") from exc
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("BSM arithmetic produced nonfinite results")
    return {**result, "classification": "MODEL_ESTIMATE", "vega_unit": "price/vol_percentage_point",
            "theta_unit": "price/calendar_day", "rho_unit": "price/rate_percentage_point"}


def implied_volatility(
    price: float, spot: float, strike: float, rate: float = 0.0, div: float = 0.0,
    ttm: float = 1.0, call: bool = True, *, max_vol: float = 10.0,
    price_tolerance: float = 1e-10,
) -> float:
    """Bracketed BSM inversion with European no-arbitrage bounds.

    Exact discounted intrinsic yields 0. Upper-bound prices imply infinite vol
    and are rejected. A quote with IV above max_vol is rejected explicitly.
    """
    spot, strike, rate, div, ttm = _inputs(spot, strike, rate, div, ttm, call)
    price = _finite(price, "price")
    max_vol = _finite(max_vol, "max_vol", positive=True)
    price_tolerance = _finite(price_tolerance, "price_tolerance", positive=True)
    ds, dk = _discounted(spot, strike, rate, div, ttm)
    lower = max(0.0, ds - dk if call else dk - ds)
    upper = ds if call else dk
    if price < lower or price >= upper:
        raise ValueError("price outside finite-vol European no-arbitrage bounds")
    if price == lower:
        return 0.0

    def residual(vol: float) -> float:
        return float(black_scholes(spot, strike, rate, div, vol, ttm, call)["price"]) - price

    lo = 1e-12
    hi = max_vol
    if residual(hi) < 0:
        raise ValueError("implied volatility exceeds declared max_vol")
    if residual(lo) >= 0:
        raise ValueError("implied volatility below numerical resolution")
    answer = float(brentq(residual, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200))
    if abs(residual(answer)) > price_tolerance * max(1.0, price):
        raise ValueError("implied-volatility repricing failed tolerance")
    return answer


@dataclass(frozen=True, slots=True)
class OptionQuote:
    strike: float
    price: float
    source: str
    classification: str = "OBSERVED_INPUT"

    def __post_init__(self) -> None:
        object.__setattr__(self, "strike", _finite(self.strike, "strike", positive=True))
        object.__setattr__(self, "price", _finite(self.price, "price"))
        if self.price < 0:
            raise ValueError("option price must be nonnegative")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source attribution required")


@dataclass(frozen=True, slots=True)
class SurfacePoint:
    log_forward_moneyness: float  # ln(K/F), dimensionless
    ttm: float  # years
    volatility: float  # annualized decimal
    source: str
    classification: str = "MODEL_ESTIMATE"

    def __post_init__(self) -> None:
        for name in ("log_forward_moneyness", "ttm", "volatility"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, positive=name != "log_forward_moneyness"))
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source attribution required")


@dataclass(frozen=True, slots=True)
class SurfaceDiagnostics:
    state: str
    point_count: int
    calendar_violations: tuple[tuple[float, float, float], ...]
    scope: str = "TOTAL_VARIANCE_CALENDAR_AT_OBSERVED_MONEYNESS_ONLY"
    butterfly_checked: bool = False


def diagnose_call_slice(
    quotes: Sequence[OptionQuote], *, spot: float, rate: float, div: float, ttm: float,
    tolerance: float = 1e-10,
) -> dict[str, object]:
    """Discrete bounds, monotonicity and convexity at one European call expiry.

    These necessary discrete conditions do not certify a continuous surface.
    Callers must supply quotes for the same underlying/expiry/currency/time.
    """
    _inputs(spot, 1.0, rate, div, ttm, True)
    tolerance = _finite(tolerance, "tolerance", positive=True)
    rows = sorted(quotes, key=lambda q: q.strike)
    if len({q.strike for q in rows}) != len(rows):
        raise ValueError("duplicate strikes in expiry slice")
    ds, _ = _discounted(spot, 1.0, rate, div, ttm)
    df = math.exp(-rate * ttm)
    violations: list[dict[str, object]] = []
    for q in rows:
        lo = max(0.0, ds - q.strike * df)
        if q.price < lo - tolerance or q.price > ds + tolerance:
            violations.append({"code": "PRICE_BOUND", "strike": q.strike})
    slopes = []
    for left, right in zip(rows, rows[1:], strict=False):
        slope = (right.price - left.price) / (right.strike - left.strike)
        slopes.append(slope)
        if slope > tolerance or slope < -df - tolerance:
            violations.append({"code": "VERTICAL_SPREAD_BOUND", "strike": right.strike})
    for i in range(1, len(slopes)):
        if slopes[i] < slopes[i - 1] - tolerance:
            violations.append({"code": "DISCRETE_CONVEXITY", "strike": rows[i].strike})
    return {"state": "ERROR" if violations else ("AVAILABLE" if len(rows) >= 3 else "UNKNOWN"),
            "violations": violations, "point_count": len(rows), "continuous_arbitrage_certified": False,
            "sources": sorted({q.source for q in rows}), "scope": "SINGLE_EXPIRY_EUROPEAN_CALLS"}


def validate_surface(points: Sequence[SurfacePoint], *, tolerance: float = 1e-12) -> SurfaceDiagnostics:
    """Reject duplicates and diagnose decreasing total variance at fixed ln(K/F).

    No rounding of moneyness coordinates: only exactly matching coordinates are
    compared. Missing cross-expiry coordinates remain unchecked, not interpolated
    into apparently observed evidence.
    """
    tolerance = _finite(tolerance, "tolerance", positive=True)
    if len(points) > 10_000:
        raise ValueError("maximum surface point count exceeded")
    keys = [(p.log_forward_moneyness, p.ttm) for p in points]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate surface coordinates")
    groups: dict[float, list[SurfacePoint]] = {}
    for p in points:
        if not math.isfinite(p.volatility * p.volatility * p.ttm):
            raise ValueError("total variance overflow")
        groups.setdefault(p.log_forward_moneyness, []).append(p)
    violations = []
    comparisons = 0
    for k, rows in groups.items():
        ordered = sorted(rows, key=lambda p: p.ttm)
        for before, after in zip(ordered, ordered[1:], strict=False):
            comparisons += 1
            if after.volatility ** 2 * after.ttm < before.volatility ** 2 * before.ttm - tolerance:
                violations.append((k, before.ttm, after.ttm))
    state = "ERROR" if violations else ("AVAILABLE" if comparisons else "UNKNOWN")
    return SurfaceDiagnostics(state, len(points), tuple(violations))


def interpolate_total_variance(
    points: Sequence[SurfacePoint], log_forward_moneyness: float, ttm: float,
) -> dict[str, object]:
    """Linear interpolation of total variance within convex hull; no extrapolation.

    Returns a model estimate with all source identities. It does not claim that
    interpolated slices preserve butterfly arbitrage constraints.
    """
    k = _finite(log_forward_moneyness, "log_forward_moneyness")
    ttm = _finite(ttm, "ttm", positive=True)
    diagnostic = validate_surface(points)
    if diagnostic.state == "ERROR":
        raise ValueError("calendar-violating input surface")
    if len(points) < 3:
        raise ValueError("at least three non-collinear surface points required")
    coordinates = np.asarray([(p.log_forward_moneyness, p.ttm) for p in points], dtype=float)
    variances = np.asarray([p.volatility ** 2 * p.ttm for p in points], dtype=float)
    try:
        value = float(LinearNDInterpolator(coordinates, variances)(k, ttm))
    except Exception as exc:
        # Qhull diagnostics are unsuitable API content; preserve a stable error.
        raise ValueError("surface coordinates do not support 2D interpolation") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("query outside interpolation hull or invalid variance")
    return {"volatility": math.sqrt(value / ttm), "total_variance": value,
            "log_forward_moneyness": k, "ttm": ttm, "classification": "MODEL_ESTIMATE",
            "method": "LINEAR_TOTAL_VARIANCE", "sources": sorted({p.source for p in points}),
            "arbitrage_certified": False}


@dataclass(frozen=True, slots=True)
class OptionPosition:
    instrument: str
    underlying: str
    strike: float
    ttm: float
    volatility: float
    quantity: float
    multiplier: float = 100.0
    call: bool = True
    currency: str = "USD"
    source: str = "unspecified"

    def __post_init__(self) -> None:
        for name in ("instrument", "underlying", "currency", "source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty identity")
        for name in ("strike", "ttm", "volatility", "quantity", "multiplier"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, positive=name != "quantity"))
        if not isinstance(self.call, bool):
            raise ValueError("call must be boolean")


def portfolio_exposure(
    positions: Sequence[OptionPosition], *, spot: float, rate: float = 0.0, div: float = 0.0,
) -> AvailableExposure | UnavailableExposure:
    """Aggregate model PV and Greeks for one underlying and currency.

    Quantity is signed contracts. Delta uses underlying units; gamma is delta
    change per spot currency unit, vega/rho per percentage point, theta per day.
    No netting across currencies or underlyings and no brokerage holdings claim.
    """
    if len(positions) > 10_000:
        raise ValueError("maximum position count exceeded")
    if not positions:
        return {"state": "UNAVAILABLE", "classification": "MODEL_ESTIMATE", "positions": []}
    if len({(p.underlying, p.currency) for p in positions}) != 1:
        raise ValueError("aggregate requires one underlying and currency")
    if len({p.instrument for p in positions}) != len(positions):
        raise ValueError("duplicate option instrument identity")
    keys = ("price", "delta", "gamma", "vega", "theta", "rho")
    totals: dict[str, list[float]] = {key: [] for key in keys}
    rows: list[dict[str, float | str]] = []
    for p in positions:
        value = black_scholes(spot, p.strike, rate, div, p.volatility, p.ttm, p.call)
        signed_units = p.quantity * p.multiplier
        row = {key: float(value[key]) * signed_units for key in keys}
        if not all(math.isfinite(v) for v in row.values()):
            raise ValueError("position exposure overflow")
        for key in keys:
            totals[key].append(row[key])
        rows.append({"instrument": p.instrument, "quantity": p.quantity,
                     "multiplier": p.multiplier, "source": p.source, **row})
    try:
        aggregate = {key: math.fsum(values) for key, values in totals.items()}
    except OverflowError as exc:
        raise ValueError("aggregate exposure overflow") from exc
    return {"state": "AVAILABLE", "classification": "MODEL_ESTIMATE",
            "underlying": positions[0].underlying, "currency": positions[0].currency,
            "spot": float(spot), "positions": rows,
            "price": aggregate["price"], "delta": aggregate["delta"], "gamma": aggregate["gamma"],
            "vega": aggregate["vega"], "theta": aggregate["theta"], "rho": aggregate["rho"],
            "units": {"price": "currency", "delta": "underlying_units", "gamma": "underlying_units/spot_currency",
                      "vega": "currency/vol_percentage_point", "theta": "currency/calendar_day",
                      "rho": "currency/rate_percentage_point"}}


def scenario_grid(
    positions: Sequence[OptionPosition], *, spot: float,
    spot_shocks: Sequence[float], volatility_shocks: Sequence[float],
    rate: float = 0.0, div: float = 0.0, elapsed_years: float = 0.0,
) -> ScenarioResult:
    """Full BSM repricing grid; scenario P&L is a model difference, never realized.

    Spot shocks are relative decimal changes; volatility shocks are absolute
    decimal changes. Smooth-Greek expiry domain requires elapsed_years smaller
    than every remaining maturity. Negative shocked vol is rejected.
    """
    scenario_count = len(spot_shocks) * len(volatility_shocks)
    if scenario_count > 10_000 or scenario_count * len(positions) > 100_000:
        raise ValueError("maximum scenario count exceeded")
    elapsed_years = _finite(elapsed_years, "elapsed_years")
    if elapsed_years < 0:
        raise ValueError("elapsed_years must be nonnegative")
    base = portfolio_exposure(positions, spot=spot, rate=rate, div=div)
    if base["state"] != "AVAILABLE" or not spot_shocks or not volatility_shocks:
        return {"state": "UNAVAILABLE", "classification": "MODEL_ESTIMATE", "scenarios": []}
    if any(p.ttm <= elapsed_years for p in positions):
        raise ValueError("scenario reaches expiry outside smooth BSM domain")
    spots = [_finite(x, "spot_shock") for x in spot_shocks]
    vols = [_finite(x, "volatility_shock") for x in volatility_shocks]
    if any(x <= -1 for x in spots):
        raise ValueError("spot shock must exceed -1")
    if any(p.volatility + dv <= 0 for p in positions for dv in vols):
        raise ValueError("shocked volatility must remain positive")
    rows: list[ScenarioRow] = []
    for ds in spots:
        for dv in vols:
            changed = [OptionPosition(p.instrument, p.underlying, p.strike, p.ttm - elapsed_years,
                                      p.volatility + dv, p.quantity, p.multiplier, p.call, p.currency, p.source)
                       for p in positions]
            exposure = portfolio_exposure(changed, spot=spot * (1 + ds), rate=rate, div=div)
            if exposure["state"] != "AVAILABLE":
                raise ValueError("scenario exposure is unavailable")
            pnl = exposure["price"] - base["price"]
            if not math.isfinite(pnl):
                raise ValueError("scenario P&L overflow")
            rows.append({"spot_shock": ds, "volatility_shock": dv,
                         "spot": exposure["spot"], "value": exposure["price"], "model_pnl": pnl})
    return {"state": "AVAILABLE", "classification": "MODEL_ESTIMATE", "base_value": base["price"],
            "underlying": base["underlying"], "currency": base["currency"], "elapsed_years": elapsed_years,
            "spot_shock_unit": "relative_decimal", "volatility_shock_unit": "absolute_decimal",
            "sources": sorted({p.source for p in positions}), "scenarios": rows}
