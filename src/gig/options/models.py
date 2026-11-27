"""Option model research beyond European Black-Scholes values.

American exercise on a Cox-Ross-Rubinstein tree, second-order Greeks, raw SVI
smile calibration with static-arbitrage diagnostics and the implied density,
and discrete variance-swap replication. Units follow ``analytics``: prices per
underlying unit, time in years, annualized decimal volatility. Every result is
a MODEL_ESTIMATE on caller inputs; no quote is fetched or implied.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

from gig import checks
from gig.options.analytics import black_scholes

MAX_STEPS = 2000
MAX_QUOTES = 500


def _contract(spot, strike, rate, div, vol, ttm, call) -> tuple[float, float, float, float, float, float]:
    if not isinstance(call, bool):
        raise ValueError("call must be boolean")
    return (checks.number(spot, "spot", positive=True, high=1e12), checks.number(strike, "strike", positive=True, high=1e12),
            checks.number(rate, "rate", low=-1., high=10.), checks.number(div, "div", low=-1., high=10.),
            checks.number(vol, "vol", positive=True, high=10.), checks.number(ttm, "ttm", positive=True, high=100.))


def american_binomial(spot, strike, rate=0., div=0., vol=.2, ttm=1., call=True, *, steps=200) -> dict:
    """American and European values on one Cox-Ross-Rubinstein tree."""
    spot, strike, rate, div, vol, ttm = _contract(spot, strike, rate, div, vol, ttm, call)
    n = checks.integer(steps, "steps", 2, MAX_STEPS)
    dt = ttm/n
    up = math.exp(vol*math.sqrt(dt))
    growth, discount = math.exp((rate-div)*dt), math.exp(-rate*dt)
    p = (growth-1/up)/(up-1/up)
    if not 0 < p < 1:
        raise ValueError("steps too few for a risk-neutral probability between 0 and 1")
    sign = 1. if call else -1.

    def prices(step: int) -> np.ndarray:
        return spot*up**(2.*np.arange(step+1)-step)

    def payoff(step: int) -> np.ndarray:
        return np.maximum(sign*(prices(step)-strike), 0.)

    american = european = payoff(n)
    levels: dict[int, np.ndarray] = {}
    boundary: list[dict] = []
    for step in range(n-1, -1, -1):
        hold = discount*(p*american[1:]+(1-p)*american[:-1])
        european = discount*(p*european[1:]+(1-p)*european[:-1])
        exercise = payoff(step)
        # Indifference (for example a zero-rate deep in-the-money put) is not early exercise.
        early = exercise > hold*(1+1e-12)+1e-14
        american = np.maximum(hold, exercise)
        # Odd and even steps sit on different price grids; one grid keeps the boundary free of sawtooth.
        if early.any() and step % 2 == 0:
            critical = prices(step)[early]
            boundary.append({"time": step*dt, "critical_price": float(critical.min() if call else critical.max())})
        if step <= 2:
            levels[step] = american
    s1, s2 = prices(1), prices(2)
    upper, lower = (levels[2][2]-levels[2][1])/(s2[2]-s2[1]), (levels[2][1]-levels[2][0])/(s2[1]-s2[0])
    reference = black_scholes(spot, strike, rate, div, vol, ttm, call)
    return {"method": "COX_ROSS_RUBINSTEIN", "steps": n, "risk_neutral_probability": p,
            "price": float(american[0]), "european_tree_price": float(european[0]),
            "early_exercise_premium": float(american[0]-european[0]),
            "black_scholes_price": reference["price"],
            "tree_discretization_error": float(european[0]-reference["price"]),
            "delta": float((levels[1][1]-levels[1][0])/(s1[1]-s1[0])),
            "gamma": float((upper-lower)/(.5*(s2[2]-s2[0]))),
            "theta_per_year": float((levels[2][1]-american[0])/(2*dt)),
            "exercise_boundary": boundary[::-1][:: max(1, len(boundary)//50)],
            "exercise_boundary_nodes": len(boundary), "exercise_boundary_resolution": "TREE_NODE_PRICES_AT_EVEN_STEPS",
            "dividend_model": "CONTINUOUS_YIELD_NO_DISCRETE_DIVIDENDS", "classification": "MODEL_ESTIMATE"}


def higher_order_greeks(spot, strike, rate=0., div=0., vol=.2, ttm=1., call=True) -> dict:
    """Closed-form Black-Scholes cross and second-order sensitivities in raw units."""
    spot, strike, rate, div, vol, ttm = _contract(spot, strike, rate, div, vol, ttm, call)
    root = vol*math.sqrt(ttm)
    d1 = (math.log(spot/strike)+(rate-div+.5*vol*vol)*ttm)/root
    d2 = d1-root
    carry, density = math.exp(-div*ttm), float(norm.pdf(d1))
    gamma = carry*density/(spot*root)
    vega = spot*carry*density*math.sqrt(ttm)
    drift = carry*density*(2*(rate-div)*ttm-d2*root)/(2*ttm*root)
    charm = (div*carry*norm.cdf(d1) if call else -div*carry*norm.cdf(-d1))-drift
    result = {"d1": d1, "d2": d2, "gamma": gamma, "vega": vega,
              "vanna": -carry*density*d2/vol, "volga": vega*d1*d2/vol, "charm": float(charm),
              "speed": -gamma/spot*(d1/root+1), "zomma": gamma*(d1*d2-1)/vol}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Greeks outside numerical bounds")
    return {**result, "units": {"vega": "price per 1.00 volatility", "vanna": "delta per 1.00 volatility",
                                "volga": "vega per 1.00 volatility", "charm": "delta per year of elapsed time",
                                "speed": "gamma per underlying unit", "zomma": "gamma per 1.00 volatility"},
            "classification": "MODEL_ESTIMATE"}


def svi_total_variance(k, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    """Raw SVI total implied variance at log-moneyness ``k``."""
    shifted = np.asarray(k, dtype=float)-m
    return a+b*(rho*shifted+np.sqrt(shifted**2+sigma**2))


def _svi_shape(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float):
    shifted = k-m
    root = np.sqrt(shifted**2+sigma**2)
    w = a+b*(rho*shifted+root)
    first, second = b*(rho+shifted/root), b*sigma**2/root**3
    # Gatheral's butterfly function: negative values imply a negative density.
    g = (1-k*first/(2*w))**2-first**2/4*(1/w+.25)+second/2
    return w, g


def svi_diagnostics(parameters: dict, *, low: float = -1.5, high: float = 1.5) -> dict:
    """Static-arbitrage checks and the implied density of one raw SVI slice."""
    a, b, rho, m, sigma = (parameters[key] for key in ("a", "b", "rho", "m", "sigma"))
    floor = a+b*sigma*math.sqrt(1-rho*rho)
    grid = np.linspace(low, high, 2001)
    result = {"minimum_total_variance": floor, "wing_slope": b*(1+abs(rho)),
              "wing_slope_within_lee_bound": bool(b*(1+abs(rho)) <= 4), "grid": [low, high]}
    if floor <= 0:
        return {**result, "butterfly_arbitrage_free_on_grid": False, "minimum_butterfly_function": None,
                "density_mass_on_grid": None}
    w, g = _svi_shape(grid, a, b, rho, m, sigma)
    density = g/np.sqrt(2*math.pi*w)*np.exp(-.5*(-grid/np.sqrt(w)-.5*np.sqrt(w))**2)
    return {**result, "minimum_butterfly_function": float(g.min()),
            "butterfly_arbitrage_free_on_grid": bool(g.min() >= 0),
            "density_mass_on_grid": float(np.sum(.5*(density[1:]+density[:-1])*np.diff(grid))),
            "density": [{"log_moneyness": float(x), "density": float(y)} for x, y in zip(grid[::50], density[::50], strict=True)]}


def fit_svi(log_moneyness, total_variance) -> dict:
    """Least-squares raw SVI fit of one expiry from deterministic starting points.

    A local optimizer from a fixed start grid; the best residual is reported
    with its diagnostics. A good fit is not evidence that quotes were tradable.
    """
    k = checks.array(log_moneyness, "log_moneyness", dimensions=1, maximum=MAX_QUOTES, minimum=5)
    w = checks.array(total_variance, "total_variance", dimensions=1, maximum=MAX_QUOTES, minimum=5)
    if k.shape != w.shape or np.any(w <= 0) or len(np.unique(k)) != len(k) or np.abs(k).max() > 5:
        raise ValueError("need matching distinct log-moneyness within 5 and positive total variance")
    span, top = float(k.max()-k.min()), float(w.max())
    lower = [-top, 0., -.999, float(k.min())-span, 1e-4]
    upper = [top, 4., .999, float(k.max())+span, 10*max(span, 1.)]
    best = None
    for m in np.quantile(k, (.25, .5, .75)):
        for sigma in (.05*span, .2*span, .6*span):
            for rho in (-.5, 0., .5):
                start = [.5*float(w.min()), max((top-float(w.min()))/span, 1e-3), rho, float(m), max(sigma, 2e-4)]
                fit = least_squares(lambda x: svi_total_variance(k, *x)-w, start, bounds=(lower, upper),
                                    xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=2000)
                if best is None or fit.cost < best.cost:
                    best = fit
    assert best is not None
    parameters = dict(zip(("a", "b", "rho", "m", "sigma"), map(float, best.x), strict=True))
    fitted = svi_total_variance(k, **parameters)
    pad = .25*span
    return {"method": "RAW_SVI_LEAST_SQUARES", "quotes": len(k), "starts": 27, "parameters": parameters,
            "root_mean_square_error": float(np.sqrt(np.mean((fitted-w)**2))),
            "maximum_absolute_error": float(np.abs(fitted-w).max()), "fitted_total_variance": fitted.tolist(),
            "diagnostics": svi_diagnostics(parameters, low=float(k.min())-pad, high=float(k.max())+pad),
            "optimizer_limits": "LOCAL_LEAST_SQUARES_NOT_A_GLOBAL_OR_ARBITRAGE_CONSTRAINED_FIT",
            "classification": "MODEL_ESTIMATE"}


def variance_swap_strike(strikes, put_prices, call_prices, forward, rate, ttm) -> dict:
    """Fair variance from a discrete out-of-the-money option strip.

    The standard log-contract replication: ``2/T * sum(dK/K^2 * exp(rT) * Q)``
    less the forward adjustment at the first strike at or below the forward.
    """
    k = checks.array(strikes, "strikes", dimensions=1, maximum=MAX_QUOTES, minimum=3)
    puts = checks.array(put_prices, "put_prices", dimensions=1, maximum=MAX_QUOTES, minimum=3)
    calls = checks.array(call_prices, "call_prices", dimensions=1, maximum=MAX_QUOTES, minimum=3)
    f, r = checks.number(forward, "forward", positive=True, high=1e12), checks.number(rate, "rate", low=-1., high=10.)
    t = checks.number(ttm, "ttm", positive=True, high=100.)
    if not k.shape == puts.shape == calls.shape or np.any(k <= 0) or np.any(np.diff(k) <= 0):
        raise ValueError("strikes must be positive and strictly increasing with one put and call each")
    if np.any(puts < 0) or np.any(calls < 0) or not k[0] <= f <= k[-1]:
        raise ValueError("prices must be nonnegative and strikes must bracket the forward")
    at = int(np.searchsorted(k, f, side="right")-1)
    quotes = np.where(k < k[at], puts, calls)
    quotes[at] = .5*(puts[at]+calls[at])
    width = np.gradient(k)
    terms = 2/t*width/k**2*math.exp(r*t)*quotes
    variance = float(terms.sum()-(f/k[at]-1)**2/t)
    if variance <= 0:
        raise ValueError("option strip implies a nonpositive variance")
    return {"method": "DISCRETE_LOG_CONTRACT_REPLICATION", "strikes": len(k), "reference_strike": float(k[at]),
            "fair_variance": variance, "fair_volatility": math.sqrt(variance),
            # Large edge shares mean the strip is too narrow for the wings it omits.
            "lowest_strike_share": float(terms[0]/terms.sum()), "highest_strike_share": float(terms[-1]/terms.sum()),
            "replication_limits": "DISCRETE_TRUNCATED_STRIP_NO_JUMP_OR_DIVIDEND_ADJUSTMENT",
            "classification": "MODEL_ESTIMATE"}
