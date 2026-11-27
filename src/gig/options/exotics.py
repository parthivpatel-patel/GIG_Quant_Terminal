"""Stochastic-volatility, barrier and multi-leg option research.

Heston (1993) European values by one Fourier integral, Reiner-Rubinstein
single-barrier values under Black-Scholes, and the exact expiry profile of a
multi-leg position. Units follow ``analytics``: prices per underlying unit,
time in years, annualized decimal volatility and variance. Every result is a
MODEL_ESTIMATE on caller inputs; no quote is fetched and nothing is calibrated.
"""
from __future__ import annotations

import cmath
import math

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from gig import checks
from gig.options.analytics import black_scholes, implied_volatility
from gig.options.models import _contract

MAX_LEGS = 16
# Broadie, Glasserman and Kou (1997): -zeta(1/2)/sqrt(2*pi).
DISCRETE_MONITORING_SHIFT = 0.5825971579390107


def _heston_characteristic(z: complex, ttm: float, v0: float, kappa: float, theta: float, xi: float, rho: float) -> complex:
    """E[exp(iz log(S_T/F))] in the branch-cut-free form of Albrecher et al. (2007)."""
    beta = kappa-1j*rho*xi*z
    d = cmath.sqrt(beta*beta+xi*xi*(1j*z+z*z))
    g = (beta-d)/(beta+d)
    decay = cmath.exp(-d*ttm)
    variance_term = (beta-d)/(xi*xi)*(1-decay)/(1-g*decay)
    level_term = kappa*theta/(xi*xi)*((beta-d)*ttm-2*cmath.log((1-g*decay)/(1-g)))
    return cmath.exp(level_term+variance_term*v0)


def heston(spot, strike, rate=0., div=0., ttm=1., call=True, *, initial_variance, mean_reversion, long_run_variance,
           vol_of_variance, correlation) -> dict:
    """European value under Heston dynamics and the Black-Scholes volatility that reproduces it."""
    spot, strike, rate, div, _, ttm = _contract(spot, strike, rate, div, 1., ttm, call)
    v0 = checks.number(initial_variance, "initial_variance", positive=True, high=25.)
    kappa = checks.number(mean_reversion, "mean_reversion", positive=True, high=100.)
    theta = checks.number(long_run_variance, "long_run_variance", positive=True, high=25.)
    xi = checks.number(vol_of_variance, "vol_of_variance", positive=True, high=10.)
    rho = checks.number(correlation, "correlation", low=-.999, high=.999)
    forward = spot*math.exp((rate-div)*ttm)
    k = math.log(forward/strike)

    def integrand(u: float) -> float:
        return (cmath.exp(1j*u*k)*_heston_characteristic(u-.5j, ttm, v0, kappa, theta, xi, rho)).real/(u*u+.25)

    upper = 50.
    while abs(_heston_characteristic(upper-.5j, ttm, v0, kappa, theta, xi, rho))/(upper*upper+.25) > 1e-13:
        upper *= 2
        if upper > 1e5:
            raise ValueError("Heston integrand does not decay; maturity or variance parameters are too small")
    integral, error = quad(integrand, 0., upper, limit=2000, epsabs=1e-9, epsrel=1e-9)
    discount = math.exp(-rate*ttm)
    call_value = discount*(forward-math.sqrt(forward*strike)/math.pi*integral)
    put_value = call_value-discount*(forward-strike)
    tolerance = 1e-7*max(spot, strike)
    lower = discount*max(forward-strike, 0.)
    if not math.isfinite(call_value) or error*discount*math.sqrt(forward*strike)/math.pi > tolerance \
            or not lower-tolerance <= call_value <= discount*forward+tolerance:
        raise ValueError("Heston integration did not converge inside the no-arbitrage bounds")
    call_value = min(max(call_value, lower), discount*forward)
    put_value = max(call_value-discount*(forward-strike), 0.)
    value = call_value if call else put_value
    try:
        implied = implied_volatility(value, spot, strike, rate, div, ttm, call)
    except ValueError:
        implied = None
    return {"method": "HESTON_1993_SINGLE_FOURIER_INTEGRAL", "price": value, "call": call_value, "put": put_value,
            "implied_volatility": implied, "forward": forward,
            "integration": {"upper_limit": upper, "absolute_error_estimate": error},
            "feller_condition_satisfied": 2*kappa*theta >= xi*xi,
            "expected_average_variance": theta+(v0-theta)*(1-math.exp(-kappa*ttm))/(kappa*ttm),
            "parameter_limits": "CALLER_PARAMETERS_NOT_CALIBRATED_TO_ANY_QUOTE", "classification": "MODEL_ESTIMATE"}


def barrier(spot, strike, barrier_level, rate=0., div=0., vol=.2, ttm=1., call=True, *, direction, knock,
            monitoring_dates=None) -> dict:
    """Single-barrier European option under Black-Scholes, without rebate.

    Continuous monitoring uses the Reiner-Rubinstein formulas. ``monitoring_dates``
    applies the Broadie-Glasserman-Kou barrier shift, an approximation for
    equally spaced discrete monitoring.
    """
    spot, strike, rate, div, vol, ttm = _contract(spot, strike, rate, div, vol, ttm, call)
    level = checks.number(barrier_level, "barrier_level", positive=True, high=1e12)
    if direction not in ("UP", "DOWN") or knock not in ("IN", "OUT"):
        raise ValueError("direction must be UP or DOWN and knock must be IN or OUT")
    up = direction == "UP"
    if (level <= spot) if up else (level >= spot):
        raise ValueError("an up barrier must be above spot and a down barrier below spot")
    effective = level
    if monitoring_dates is not None:
        dates = checks.integer(monitoring_dates, "monitoring_dates", 1, 100_000)
        effective = level*math.exp((1 if up else -1)*DISCRETE_MONITORING_SHIFT*vol*math.sqrt(ttm/dates))
    root = vol*math.sqrt(ttm)
    lam = (rate-div+.5*vol*vol)/(vol*vol)
    carry, discount = spot*math.exp(-div*ttm), strike*math.exp(-rate*ttm)
    ratio = effective/spot
    reflected_spot, reflected_strike = carry*ratio**(2*lam), discount*ratio**(2*lam-2)
    y = math.log(effective*effective/(spot*strike))/root+lam*root
    x1 = math.log(spot/effective)/root+lam*root
    y1 = math.log(effective/spot)/root+lam*root
    n = norm.cdf
    vanilla = float(black_scholes(spot, strike, rate, div, vol, ttm, call)["price"])
    if call and not up:
        if effective <= strike:
            knocked_in = reflected_spot*n(y)-reflected_strike*n(y-root)
        else:
            knocked_in = vanilla-(carry*n(x1)-discount*n(x1-root)-reflected_spot*n(y1)+reflected_strike*n(y1-root))
    elif call:
        if effective <= strike:
            knocked_in = vanilla
        else:
            knocked_in = (carry*n(x1)-discount*n(x1-root)-reflected_spot*(n(-y)-n(-y1))
                          + reflected_strike*(n(-y+root)-n(-y1+root)))
    elif up:
        if effective >= strike:
            knocked_in = -reflected_spot*n(-y)+reflected_strike*n(-y+root)
        else:
            knocked_in = vanilla-(-carry*n(-x1)+discount*n(-x1+root)+reflected_spot*n(-y1)-reflected_strike*n(-y1+root))
    elif effective >= strike:
        knocked_in = vanilla
    else:
        knocked_in = (-carry*n(-x1)+discount*n(-x1+root)+reflected_spot*(n(y)-n(y1))
                      - reflected_strike*(n(y-root)-n(y1-root)))
    knocked_in = min(max(float(knocked_in), 0.), vanilla)
    knocked_out = vanilla-knocked_in
    return {"method": "REINER_RUBINSTEIN_SINGLE_BARRIER", "price": knocked_in if knock == "IN" else knocked_out,
            "knock_in": knocked_in, "knock_out": knocked_out, "vanilla": vanilla, "barrier_level": level,
            "effective_barrier": effective,
            "monitoring": "CONTINUOUS" if monitoring_dates is None else "DISCRETE_BROADIE_GLASSERMAN_KOU_APPROXIMATION",
            "model_limits": "CONSTANT_VOLATILITY_NO_REBATE_NO_SMILE_OR_JUMP_ADJUSTMENT", "classification": "MODEL_ESTIMATE"}


def strategy(legs, *, multiplier=1.) -> dict:
    """Exact profit and loss at expiry of calls, puts and underlying held to one expiry."""
    if not isinstance(legs, list) or not 1 <= len(legs) <= MAX_LEGS:
        raise ValueError(f"legs must be a list of 1 to {MAX_LEGS} positions")
    scale = checks.number(multiplier, "multiplier", positive=True, high=1e6)
    parsed = []
    for leg in legs:
        if not isinstance(leg, dict) or set(leg)-{"kind", "strike", "quantity", "premium"} \
                or leg.get("kind") not in ("CALL", "PUT", "UNDERLYING"):
            raise ValueError("each leg needs kind CALL, PUT or UNDERLYING with quantity, premium and (for options) strike")
        quantity = checks.number(leg.get("quantity"), "quantity", low=-1e9, high=1e9)
        premium = checks.number(leg.get("premium"), "premium", low=0., high=1e12)
        if quantity == 0 or (leg["kind"] == "UNDERLYING") != ("strike" not in leg):
            raise ValueError("quantity must be nonzero; options need a strike and the underlying must not have one")
        strike = 0. if leg["kind"] == "UNDERLYING" else checks.number(leg["strike"], "strike", positive=True, high=1e12)
        parsed.append((leg["kind"], strike, quantity, premium))
    cost = sum(quantity*premium for _, _, quantity, premium in parsed)

    def profit(price: float) -> float:
        value = 0.
        for kind, strike, quantity, _ in parsed:
            value += quantity*(price if kind == "UNDERLYING" else max(price-strike, 0.) if kind == "CALL" else max(strike-price, 0.))
        return (value-cost)*scale

    # Profit is linear between strikes, so its extremes and zeros lie at the knots or on the final ray.
    knots = sorted({0., *(strike for kind, strike, _, _ in parsed if kind != "UNDERLYING")})
    values = [profit(knot) for knot in knots]
    final_slope = scale*sum(quantity for kind, _, quantity, _ in parsed if kind != "PUT")
    breakevens = [knot for knot, value in zip(knots, values, strict=True) if value == 0]
    for index in range(len(knots)-1):
        low, high = values[index], values[index+1]
        if low*high < 0:
            breakevens.append(knots[index]-low*(knots[index+1]-knots[index])/(high-low))
    if values[-1]*final_slope < 0:
        breakevens.append(knots[-1]-values[-1]/final_slope)
    flat_at_zero = any(low == 0 and high == 0 for low, high in zip(values, values[1:], strict=False)) \
        or (values[-1] == 0 and final_slope == 0)
    span = max(knots[-1], *(premium for kind, _, _, premium in parsed if kind == "UNDERLYING"), 1.)
    grid = sorted({*knots, *breakevens, *np.linspace(0., 2*span, 41).tolist()})
    return {"method": "EXACT_PIECEWISE_LINEAR_EXPIRY_PROFILE", "legs": len(parsed), "multiplier": scale,
            "net_premium": cost*scale, "premium_direction": "DEBIT" if cost > 0 else "CREDIT" if cost < 0 else "ZERO",
            "max_profit": None if final_slope > 0 else max(values), "max_profit_state": "UNBOUNDED" if final_slope > 0 else "BOUNDED",
            "max_loss": None if final_slope < 0 else -min(min(values), 0.),
            "max_loss_state": "UNBOUNDED" if final_slope < 0 else "BOUNDED",
            "breakevens": sorted(set(breakevens)), "breakeven_count": len(set(breakevens)),
            "zero_profit_interval_present": flat_at_zero,
            "profit_at_zero_underlying": values[0], "slope_above_highest_strike": final_slope,
            "profile": [{"underlying": price, "profit": profit(price)} for price in grid],
            "profile_limits": "EXPIRY_ONLY_SAME_EXPIRY_NO_EARLY_EXERCISE_FINANCING_FEES_OR_MARGIN",
            "classification": "MODEL_ESTIMATE"}
