"""Stationary Gaussian GARCH(1,1) likelihood and CPU maximum likelihood fitting.

Returns must be decimal returns sampled at a single, documented cadence.
Conditional variance and forecasts are in squared-return units per observation.
Gaussian likelihood is a distributional assumption, not economic validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from gig.ml.state_space import _freeze


def _returns(values, *, minimum: int = 1) -> np.ndarray:
    x = np.array(values, dtype=float, copy=True)
    if x.ndim != 1 or len(x) < minimum or len(x) > 1_000_000 or not np.isfinite(x).all():
        raise ValueError(f"returns must be finite 1D with {minimum}..1000000 observations")
    return x


@dataclass(frozen=True)
class GARCH11:
    omega: float
    alpha: float
    beta: float
    mean: float = 0.0

    def __post_init__(self) -> None:
        values = (self.omega, self.alpha, self.beta, self.mean)
        if not all(np.ndim(v) == 0 and np.isfinite(v) for v in values):
            raise ValueError("GARCH parameters must be finite scalars")
        if self.omega <= 0 or min(self.alpha, self.beta) < 0 or self.alpha + self.beta >= 1:
            raise ValueError("require omega>0, alpha,beta>=0 and alpha+beta<1")
        if not np.isfinite(self.unconditional_variance):
            raise ValueError("GARCH unconditional variance overflows")

    @property
    def unconditional_variance(self) -> float:
        return float(self.omega / (1.0 - self.alpha - self.beta))

    def conditional_variances(self, returns, *, initial_variance: float | None = None) -> np.ndarray:
        """h[0] is unconditional variance unless caller explicitly supplies it."""
        x = _returns(returns)
        initial = self.unconditional_variance if initial_variance is None else initial_variance
        if not np.ndim(initial) == 0 or not np.isfinite(initial) or initial <= 0:
            raise ValueError("initial variance must be finite positive")
        h = np.empty(len(x))
        h[0] = initial
        with np.errstate(over="ignore", invalid="ignore"):
            innovations = (x - self.mean) ** 2
            for t in range(1, len(x)):
                h[t] = self.omega + self.alpha * innovations[t - 1] + self.beta * h[t - 1]
        if not np.isfinite(h).all() or (h <= 0).any() or not np.isfinite(innovations).all():
            raise ValueError("GARCH variance numerical overflow")
        return _freeze(h)

    def log_likelihood(self, returns, *, initial_variance: float | None = None) -> float:
        x = _returns(returns)
        h = self.conditional_variances(x, initial_variance=initial_variance)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            value = -0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + (x - self.mean) ** 2 / h)
        if not np.isfinite(value):
            raise ValueError("GARCH likelihood numerical overflow")
        return float(value)

    def forecast(self, returns, horizon: int = 1) -> np.ndarray:
        """Expected variance for next horizon steps after the supplied observations."""
        if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 100_000:
            raise ValueError("horizon must be an integer 1..100000")
        x = _returns(returns)
        h = self.conditional_variances(x)
        out = np.empty(horizon)
        with np.errstate(over="ignore", invalid="ignore"):
            out[0] = self.omega + self.alpha * (x[-1] - self.mean) ** 2 + self.beta * h[-1]
            for step in range(1, horizon):
                out[step] = self.omega + (self.alpha + self.beta) * out[step - 1]
        if not np.isfinite(out).all() or (out <= 0).any():
            raise ValueError("GARCH forecast numerical overflow")
        return _freeze(out)


@dataclass(frozen=True)
class GARCHFit:
    model: GARCH11
    log_likelihood: float
    converged: bool
    iterations: int
    observations: int
    optimizer_status: int
    # Generic status code only: do not relay provider/optimizer diagnostics to UI.
    stop_reason: str
    evidence_state: str = "TRAINING_SAMPLE_ONLY"


def fit_garch11(returns, *, max_iterations: int = 500, tolerance: float = 1e-8) -> GARCHFit:
    """Deterministic constrained Gaussian MLE with three starting persistence values.

    Uses SLSQP on normalized returns for numerical conditioning. Reports optimizer
    success explicitly; nonconvergence is never silently promoted to a fitted model.
    omega is strictly positive and alpha+beta is bounded below one by 1e-6.
    """
    x = _returns(returns, minimum=20)
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or not 1 <= max_iterations <= 10_000:
        raise ValueError("max_iterations must be an integer 1..10000")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite positive")
    with np.errstate(over="ignore", invalid="ignore"):
        center, scale = float(x.mean()), float(x.std())
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= np.finfo(float).tiny ** 0.25:
        raise ValueError("GARCH training requires nondegenerate return variance")
    z = (x - center) / scale
    bounds = [(1e-9, 100.0), (0.0, 0.999999), (0.0, 0.999999), (-10.0, 10.0)]
    constraints = [{"type": "ineq", "fun": lambda p: 0.999999 - p[1] - p[2]}]

    def objective(params) -> float:
        try:
            candidate = GARCH11(*map(float, params))
            return -candidate.log_likelihood(z)
        except ValueError:
            return 1e100

    results = []
    for alpha, beta in ((0.05, 0.9), (0.1, 0.5), (0.01, 0.1)):
        result = minimize(
            objective, np.array([1 - alpha - beta, alpha, beta, 0.0]),
            method="SLSQP", bounds=bounds, constraints=constraints,
            options={"maxiter": max_iterations, "ftol": tolerance},
        )
        feasible = (
            np.isfinite(result.fun) and result.fun < 1e99 and np.isfinite(result.x).all()
            and result.x[0] > 0 and min(result.x[1:3]) >= 0 and sum(result.x[1:3]) <= 0.999999 + 1e-12
        )
        if feasible:
            results.append(result)
    if not results:
        raise ValueError("GARCH optimization did not produce finite feasible parameters")
    successful = [result for result in results if result.success]
    best = min(successful or results, key=lambda result: result.fun)
    omega, alpha, beta, mean = map(float, best.x)
    model = GARCH11(omega * scale ** 2, alpha, beta, center + mean * scale)
    return GARCHFit(
        model, model.log_likelihood(x), bool(best.success), int(best.nit), len(x),
        int(best.status), "CONVERGED" if best.success else "OPTIMIZER_NOT_CONVERGED",
    )
