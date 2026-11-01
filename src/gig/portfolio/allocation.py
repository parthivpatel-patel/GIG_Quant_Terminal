"""Additional bounded allocation methods; existing mean-variance stays canonical."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

MAX_ASSETS = 512
MAX_VIEWS = 512


def _covariance(values) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 2 or not len(raw) or raw.shape[0] != raw.shape[1] or len(raw) > MAX_ASSETS:
        raise ValueError("covariance must be a nonempty square matrix with at most 512 assets")
    covariance = np.array(raw, dtype=float, copy=True)
    if not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T, rtol=0, atol=1e-12):
        raise ValueError("covariance must be finite and symmetric")
    if np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError("covariance must be positive definite; regularization must be explicit")
    return covariance


def _vector(values, n: int, name: str):
    result = np.array(values, dtype=float, copy=True)
    if result.shape != (n,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector with one value per asset")
    return result


def risk_parity(covariance, *, budgets=None, max_weight: float = 1.0) -> dict:
    """Long-only fully invested risk budgeting, reporting constrained residuals.

    A binding max-weight cap can prevent exact risk parity; feasibility and the
    remaining budget error are reported separately. Solver failure raises.
    """
    cov = _covariance(covariance)
    n = len(cov)
    if not np.isfinite(max_weight) or not 1/n <= max_weight <= 1:
        raise ValueError("max_weight must permit a long-only fully invested portfolio")
    budget = np.full(n, 1/n) if budgets is None else _vector(budgets, n, "budgets")
    if np.any(budget <= 0) or not np.isclose(budget.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError("risk budgets must be positive and sum to one")
    scaled = cov/np.max(np.diag(cov))

    def objective(weights):
        variance = weights @ scaled @ weights
        fractions = weights*(scaled @ weights)/variance
        return float(np.sum((fractions-budget)**2))

    result = minimize(objective, np.full(n, 1/n), method="SLSQP",
                      bounds=[(0., max_weight)]*n,
                      constraints=[{"type": "eq", "fun": lambda w: w.sum()-1}],
                      options={"ftol": 1e-14, "maxiter": 2000})
    if not result.success:
        raise RuntimeError("risk budgeting optimizer failed")
    weights = result.x
    if abs(weights.sum()-1) > 1e-8 or np.min(weights) < -1e-9 or np.max(weights) > max_weight+1e-9:
        raise RuntimeError("risk budgeting returned infeasible weights")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        variance = float(weights @ cov @ weights)
        fractions = weights*(cov @ weights)/variance
    if not np.isfinite(variance) or variance <= 0 or not np.isfinite(fractions).all():
        raise ValueError("risk budgeting arithmetic overflow")
    error = float(np.max(np.abs(fractions-budget)))
    return {"weights": weights.tolist(), "risk_fractions": fractions.tolist(),
            "target_budgets": budget.tolist(), "max_budget_error": error,
            "state": "RISK_BUDGET_MATCHED" if error < 1e-6 else "CONSTRAINED_APPROXIMATION",
            "volatility": float(np.sqrt(variance)), "max_weight": max_weight,
            "gross": float(np.abs(weights).sum())}


def black_litterman(covariance, market_weights, views, view_returns, *,
                   risk_aversion: float = 2.5, tau: float = 0.05, view_covariance=None) -> dict:
    """Gaussian Black-Litterman posterior, with explicit view uncertainty.

    Without supplied Omega, its diagonal is ``diag(P tau Sigma P')``. Inputs
    must use a common return horizon; rates/returns are excess returns. Returns
    posterior mean and predictive covariance; it does not silently optimize or
    approve trades. Market weights must be long-only and sum to one.
    """
    cov = _covariance(covariance)
    n = len(cov)
    weights = _vector(market_weights, n, "market_weights")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError("market weights must be long-only and sum to one")
    p = np.array(views, dtype=float, copy=True)
    if p.ndim != 2 or p.shape[1] != n or not 1 <= len(p) <= MAX_VIEWS or not np.isfinite(p).all():
        raise ValueError("views must be a finite nonempty views by assets matrix")
    q = _vector(view_returns, len(p), "view_returns")
    if not np.isfinite(tau) or tau <= 0 or not np.isfinite(risk_aversion) or risk_aversion <= 0:
        raise ValueError("tau and risk_aversion must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore"):
        prior = risk_aversion*(cov @ weights)
        uncertainty = tau*cov
        projected_uncertainty = p @ uncertainty @ p.T
    if not all(np.isfinite(v).all() for v in (prior, uncertainty, projected_uncertainty)):
        raise ValueError("Black-Litterman prior arithmetic overflow")
    omega = np.diag(np.diag(projected_uncertainty)) if view_covariance is None else _covariance(view_covariance)
    if omega.shape != (len(p), len(p)) or np.any(np.diag(omega) <= 0):
        raise ValueError("view covariance must match views and be positive definite")
    with np.errstate(over="ignore", invalid="ignore"):
        system = projected_uncertainty + omega
        if not np.isfinite(system).all():
            raise ValueError("Black-Litterman view uncertainty overflow")
        gain = np.linalg.solve(system, p @ uncertainty).T
        mean = prior + gain @ (q-p @ prior)
        posterior_uncertainty = uncertainty - gain @ p @ uncertainty
        posterior_uncertainty = posterior_uncertainty/2 + posterior_uncertainty.T/2
        predictive = cov+posterior_uncertainty
    if not all(np.isfinite(v).all() for v in (mean, posterior_uncertainty, predictive)):
        raise ValueError("Black-Litterman posterior arithmetic overflow")
    return {"expected_excess_returns": mean.tolist(),
            "predictive_covariance": predictive.tolist(),
            "mean_uncertainty": posterior_uncertainty.tolist(),
            "equilibrium_prior": prior.tolist(), "tau": tau,
            "view_uncertainty_source": "supplied" if view_covariance is not None else "diagonal_prior_view_variance"}


def fractional_kelly(expected_excess, covariance, *, fraction: float = 0.25,
                     max_gross: float = 1.0, max_name: float = 0.25, allow_short: bool = False) -> dict:
    """Safeguarded Gaussian small-return Kelly approximation.

    This is ``fraction * inv(Sigma) * mu`` followed by explicit caps, not a
    universal log-utility optimum. Mean/covariance horizons must match. Capping
    changes the optimum; actual return tails and model uncertainty remain.
    """
    cov = _covariance(covariance)
    mu = _vector(expected_excess, len(cov), "expected_excess")
    if type(allow_short) is not bool:
        raise ValueError("allow_short must be an explicit boolean")
    if not all(np.isfinite(v) and v > 0 for v in (fraction, max_gross, max_name)) or fraction > 1:
        raise ValueError("require fraction in (0,1] and positive finite caps")
    with np.errstate(over="ignore", invalid="ignore"):
        weights = fraction*np.linalg.solve(cov, mu)
    # Overflow must be caught before capping; clipping infinity to a plausible
    # finite position would conceal that the optimization never produced one.
    if not np.isfinite(weights).all():
        raise ValueError("Kelly unconstrained allocation overflow")
    weights = np.clip(weights, -max_name if allow_short else 0, max_name)
    gross = float(np.abs(weights).sum())
    if gross > max_gross:
        weights *= max_gross/gross
    if not np.isfinite(weights).all():
        raise ValueError("Kelly allocation overflow")
    return {"weights": weights.tolist(), "gross": float(np.abs(weights).sum()),
            "net": float(weights.sum()), "fraction": fraction, "max_name": max_name,
            "max_gross": max_gross, "allow_short": allow_short,
            "model": "gaussian_small_return_capped_approximation", "trading_approved": False}
