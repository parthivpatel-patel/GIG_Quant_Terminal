"""
Univariate Gaussian hidden Markov model.

States are statistical emission clusters. They are not labeled bull or bear
before the fit. After fitting, states may be described by their estimated mean
as a post-hoc interpretation. That label is not an input to the estimator.

`filtered` is P(state_t | returns through t) and is the causal series.
`smoothed` is P(state_t | the full sample) and must not be used as a
same-day trading signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True, slots=True)
class GaussianHMMResult:
    n_states: int
    means: np.ndarray
    stds: np.ndarray
    transition: np.ndarray
    start: np.ndarray
    log_likelihood: float
    filtered: np.ndarray
    smoothed: np.ndarray
    iterations: int
    converged: bool

    def as_dict(self) -> dict:
        order = np.argsort(self.means)
        labels = [f"mean_rank_{i}" for i in range(self.n_states)]
        ranked = {
            labels[i]: {
                "state": int(order[i]),
                "mean": float(self.means[order[i]]),
                "std": float(self.stds[order[i]]),
            }
            for i in range(self.n_states)
        }
        last = self.filtered[-1]
        return {
            "n_states": self.n_states,
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "transition": self.transition.tolist(),
            "start": self.start.tolist(),
            "log_likelihood": float(self.log_likelihood),
            "iterations": self.iterations,
            "converged": self.converged,
            "filtered_last": last.tolist(),
            "most_likely_state": int(np.argmax(last)),
            "post_hoc_by_mean": ranked,
            "label_policy": "states are unlabeled during fit; mean_rank is assigned after estimation",
            "filtered_uses": "returns through t only",
            "smoothed_uses": "full sample — research display only",
        }


def _log_emit(returns: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    var = np.maximum(stds, 1e-8) ** 2
    x = returns[:, None]
    return -0.5 * (np.log(2.0 * np.pi) + np.log(var) + (x - means) ** 2 / var)


def _forward(log_emit: np.ndarray, log_start: np.ndarray, log_trans: np.ndarray) -> np.ndarray:
    t_len, k = log_emit.shape
    log_alpha = np.empty((t_len, k))
    log_alpha[0] = log_start + log_emit[0]
    for t in range(1, t_len):
        log_alpha[t] = log_emit[t] + logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0)
    return log_alpha


def _backward(log_emit: np.ndarray, log_trans: np.ndarray) -> np.ndarray:
    t_len, k = log_emit.shape
    log_beta = np.zeros((t_len, k))
    for t in range(t_len - 2, -1, -1):
        log_beta[t] = logsumexp(log_trans + log_emit[t + 1] + log_beta[t + 1], axis=1)
    return log_beta


def _init_params(returns: np.ndarray, n_states: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    quantiles = np.linspace(0, 1, n_states + 1)
    edges = np.quantile(returns, quantiles)
    means = np.empty(n_states)
    stds = np.empty(n_states)
    for k in range(n_states):
        lo, hi = edges[k], edges[k + 1]
        if k == n_states - 1:
            chunk = returns[(returns >= lo) & (returns <= hi)]
        else:
            chunk = returns[(returns >= lo) & (returns < hi)]
        if chunk.size < 2:
            chunk = returns
        means[k] = float(np.mean(chunk))
        stds[k] = float(max(np.std(chunk), 1e-4))
    start = np.full(n_states, 1.0 / n_states)
    trans = np.full((n_states, n_states), 0.05 / max(n_states - 1, 1))
    np.fill_diagonal(trans, 0.95)
    trans = trans / trans.sum(axis=1, keepdims=True)
    return means, stds, start, trans


def fit_gaussian_hmm(
    returns: np.ndarray,
    n_states: int = 2,
    n_iter: int = 40,
    tol: float = 1e-5,
) -> GaussianHMMResult:
    """Fit a Gaussian HMM by expectation-maximization. Requires a finite sample."""
    series = np.asarray(returns, dtype=float)
    series = series[np.isfinite(series)]
    if n_states < 2:
        raise ValueError("n_states must be at least 2")
    if series.size < max(40, 15 * n_states):
        raise ValueError("not enough finite returns to fit an HMM")

    means, stds, start, trans = _init_params(series, n_states)
    prev_ll = -np.inf
    iterations = 0
    converged = False
    log_alpha = np.empty((series.size, n_states))
    log_beta = np.empty_like(log_alpha)
    smoothed = np.empty_like(log_alpha)

    for iteration in range(1, n_iter + 1):
        iterations = iteration
        log_emit = _log_emit(series, means, stds)
        log_start = np.log(np.maximum(start, 1e-300))
        log_trans = np.log(np.maximum(trans, 1e-300))
        log_alpha = _forward(log_emit, log_start, log_trans)
        log_beta = _backward(log_emit, log_trans)
        ll = float(logsumexp(log_alpha[-1]))
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        smoothed = np.exp(log_gamma)

        start = smoothed[0] / smoothed[0].sum()
        numer = np.zeros((n_states, n_states))
        for t in range(series.size - 1):
            log_xi = (
                log_alpha[t][:, None]
                + log_trans
                + log_emit[t + 1]
                + log_beta[t + 1]
            )
            log_xi -= logsumexp(log_xi)
            numer += np.exp(log_xi)
        denom = numer.sum(axis=1, keepdims=True)
        trans = np.divide(numer, np.maximum(denom, 1e-300))
        trans = np.clip(trans, 1e-8, None)
        trans /= trans.sum(axis=1, keepdims=True)

        for k in range(n_states):
            weight = smoothed[:, k]
            mass = float(weight.sum())
            if mass < 1e-8:
                continue
            means[k] = float(np.average(series, weights=weight))
            var = float(np.average((series - means[k]) ** 2, weights=weight))
            stds[k] = float(np.sqrt(max(var, 1e-8)))

        if abs(ll - prev_ll) < tol * (1.0 + abs(prev_ll)):
            converged = True
            prev_ll = ll
            break
        prev_ll = ll

    filtered = np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))
    return GaussianHMMResult(
        n_states=n_states,
        means=means.copy(),
        stds=stds.copy(),
        transition=trans.copy(),
        start=start.copy(),
        log_likelihood=float(prev_ll),
        filtered=filtered,
        smoothed=smoothed.copy(),
        iterations=iterations,
        converged=converged,
    )
