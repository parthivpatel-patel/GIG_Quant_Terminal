"""Linear Gaussian filtering and offline Rauch–Tung–Striebel smoothing.

Filtered states use observations through t only. Smoothed states use the whole
supplied sequence and must never be passed to a historical trading decision.
All matrix inputs and outputs are copied; no identity/mutation based fit cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _array(value, ndim: int, name: str) -> np.ndarray:
    out = np.array(value, dtype=float, copy=True)
    if out.ndim != ndim or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return out


def _freeze(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value)
    # A bytes-backed array cannot be made writeable again with setflags().
    # Merely clearing WRITEABLE on an owning ndarray permits that bypass.
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


def _covariance(value, n: int, name: str, *, positive: bool = False) -> np.ndarray:
    out = _array(value, 2, name)
    if out.shape != (n, n) or not np.allclose(out, out.T, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{name} must be a symmetric {n}x{n} covariance")
    eigenvalues = np.linalg.eigvalsh(out)
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(out))))
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    if positive:
        try:
            np.linalg.cholesky(out)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"{name} must be positive definite") from exc
    return (out + out.T) / 2


@dataclass(frozen=True)
class KalmanResult:
    filtered_mean: np.ndarray
    filtered_covariance: np.ndarray
    predicted_mean: np.ndarray
    predicted_covariance: np.ndarray
    innovations: np.ndarray
    innovation_covariance: np.ndarray
    log_likelihood: float
    transition: np.ndarray
    evidence_state: str = "RESEARCH_FILTERED"


@dataclass(frozen=True)
class SmoothingResult:
    mean: np.ndarray
    covariance: np.ndarray
    evidence_state: str = "OFFLINE_FULL_SEQUENCE_SMOOTHED"
    uses_future_observations: bool = True


def kalman_filter(
    observations,
    transition,
    observation,
    process_covariance,
    observation_covariance,
    initial_mean,
    initial_covariance,
) -> KalmanResult:
    """Filter y[t] = H[t] x[t] + noise, x[t] = F x[t-1] + noise.

    initial_mean/covariance describe x[0] *before* observing y[0]. There is
    no transition or Q addition before the first observation. H can be a
    constant (m,n) matrix or a per-observation (T,m,n) design. Observations
    must be finite; missing observations require explicit preprocessing.
    """
    y = _array(observations, 2, "observations")
    x0 = _array(initial_mean, 1, "initial_mean")
    t_count, m = y.shape
    n = x0.size
    if t_count < 1 or m < 1 or n < 1 or t_count > 1_000_000 or n > 128 or m > 128:
        raise ValueError("empty or oversized state-space dimensions")
    if t_count * (2 * n * n + m * m + 2 * n + m) > 12_000_000:
        raise ValueError("state-space output exceeds the 96MB numerical-array budget")
    f = _array(transition, 2, "transition")
    if f.shape != (n, n):
        raise ValueError("transition shape differs from state dimension")
    h = np.array(observation, dtype=float, copy=True)
    if h.shape not in {(m, n), (t_count, m, n)} or not np.isfinite(h).all():
        raise ValueError("observation must have shape (m,n) or (T,m,n), all finite")
    q = _covariance(process_covariance, n, "process_covariance")
    r = _covariance(observation_covariance, m, "observation_covariance", positive=True)
    p0 = _covariance(initial_covariance, n, "initial_covariance")
    means = np.empty((t_count, n))
    covs = np.empty((t_count, n, n))
    predicted = np.empty_like(means)
    predicted_cov = np.empty_like(covs)
    innovations = np.empty_like(y)
    innovation_cov = np.empty((t_count, m, m))
    x, p = x0, p0
    loglik = 0.0
    for t in range(t_count):
        if t:
            x = f @ x
            p = f @ p @ f.T + q
        ht = h if h.ndim == 2 else h[t]
        predicted[t], predicted_cov[t] = x, p
        v = y[t] - ht @ x
        s = ht @ p @ ht.T + r
        s = (s + s.T) / 2
        try:
            chol = np.linalg.cholesky(s)
            gain = np.linalg.solve(s, ht @ p).T
            whitened = np.linalg.solve(chol, v)
        except np.linalg.LinAlgError as exc:
            raise ValueError("innovation covariance is singular or not positive definite") from exc
        x = x + gain @ v
        # Joseph form preserves covariance symmetry and numerical PSD.
        residual_map = np.eye(n) - gain @ ht
        p = residual_map @ p @ residual_map.T + gain @ r @ gain.T
        p = (p + p.T) / 2
        contribution = -0.5 * (
            m * np.log(2 * np.pi) + 2 * np.log(np.diag(chol)).sum()
            + whitened @ whitened
        )
        loglik += float(contribution)
        if not np.isfinite(x).all() or not np.isfinite(p).all() or not np.isfinite(loglik):
            raise ValueError("state-space numerical overflow")
        means[t], covs[t] = x, p
        innovations[t], innovation_cov[t] = v, s
    return KalmanResult(
        filtered_mean=_freeze(means), filtered_covariance=_freeze(covs),
        predicted_mean=_freeze(predicted), predicted_covariance=_freeze(predicted_cov),
        innovations=_freeze(innovations), innovation_covariance=_freeze(innovation_cov),
        log_likelihood=loglik, transition=_freeze(f),
    )


def rts_smooth(result: KalmanResult) -> SmoothingResult:
    """Offline RTS smoother. Singular predicted covariance is rejected explicitly."""
    means = np.array(result.filtered_mean, copy=True)
    covs = np.array(result.filtered_covariance, copy=True)
    for t in range(len(means) - 2, -1, -1):
        try:
            gain = np.linalg.solve(
                result.predicted_covariance[t + 1],
                result.transition @ result.filtered_covariance[t],
            ).T
        except np.linalg.LinAlgError as exc:
            raise ValueError("RTS requires nonsingular predicted covariance") from exc
        means[t] += gain @ (means[t + 1] - result.predicted_mean[t + 1])
        covs[t] += gain @ (covs[t + 1] - result.predicted_covariance[t + 1]) @ gain.T
        covs[t] = (covs[t] + covs[t].T) / 2
    if not np.isfinite(means).all() or not np.isfinite(covs).all():
        raise ValueError("RTS numerical overflow")
    return SmoothingResult(_freeze(means), _freeze(covs))
