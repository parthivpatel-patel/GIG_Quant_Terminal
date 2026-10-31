"""CPU diagonal-Gaussian HMM with explicit causal and offline inference.

The model is fitted only on the supplied training sequence. It does not split
data or grant trading authority. Callers must enforce point-in-time training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from gig.ml.state_space import _freeze


def _observations(value) -> np.ndarray:
    out = np.array(value, dtype=float, copy=True)
    if out.ndim == 1:
        out = out[:, None]
    if (
        out.ndim != 2 or min(out.shape) < 1 or out.shape[0] > 100_000
        or out.shape[1] > 128 or not np.isfinite(out).all()
    ):
        raise ValueError("observations must be finite, nonempty (T,D), T<=100000, D<=128")
    return out


def _log_probabilities(values: np.ndarray) -> np.ndarray:
    out = np.full_like(values, -np.inf)
    np.log(values, out=out, where=values > 0)
    return out


@dataclass(frozen=True)
class HMMInference:
    filtered_probabilities: np.ndarray
    smoothed_probabilities: np.ndarray
    transition_counts: np.ndarray
    log_likelihood: float
    # smoothed_probabilities and transition_counts use future observations.
    evidence_state: str = "RESEARCH_INFERENCE"


@dataclass(frozen=True)
class GaussianHMM:
    initial: np.ndarray
    transition: np.ndarray
    means: np.ndarray
    variances: np.ndarray

    def __post_init__(self) -> None:
        initial = np.array(self.initial, dtype=float, copy=True)
        transition = np.array(self.transition, dtype=float, copy=True)
        means = np.array(self.means, dtype=float, copy=True)
        variances = np.array(self.variances, dtype=float, copy=True)
        k = initial.size
        if initial.ndim != 1 or not 1 <= k <= 32:
            raise ValueError("initial must be a probability vector with 1..32 states")
        if transition.shape != (k, k) or means.ndim != 2 or means.shape[0] != k:
            raise ValueError("HMM dimensions disagree")
        if means.shape[1] < 1 or means.shape[1] > 128 or variances.shape != means.shape:
            raise ValueError("HMM diagonal covariance dimensions disagree")
        if not all(np.isfinite(a).all() for a in (initial, transition, means, variances)):
            raise ValueError("HMM parameters must be finite")
        if (
            (initial < 0).any() or (transition < 0).any()
            or (initial > 1).any() or (transition > 1).any()
            or not np.isclose(initial.sum(), 1.0, rtol=0, atol=1e-12)
            or not np.allclose(transition.sum(axis=1), 1.0, rtol=0, atol=1e-12)
            or (variances <= 0).any()
        ):
            raise ValueError("invalid HMM probabilities or variances")
        for name, value in zip(
            ("initial", "transition", "means", "variances"),
            (initial, transition, means, variances), strict=True,
        ):
            object.__setattr__(self, name, _freeze(value))

    def _emissions(self, observations: np.ndarray) -> np.ndarray:
        if observations.shape[1] != self.means.shape[1]:
            raise ValueError("observation feature count differs from model")
        if observations.shape[0] * self.means.size > 4_000_000:
            raise ValueError("HMM emission workspace exceeds four million scalar cells")
        with np.errstate(over="ignore", invalid="ignore"):
            squared = (observations[:, None, :] - self.means[None, :, :]) ** 2
            out = -0.5 * (
                np.log(2 * np.pi * self.variances)[None, :, :]
                + squared / self.variances[None, :, :]
            ).sum(axis=2)
        if not np.isfinite(out).all():
            raise ValueError("Gaussian emission numerical overflow")
        return out

    def infer(self, observations) -> HMMInference:
        """Log-domain forward/backward; filtered[t] is causal given fitted parameters."""
        x = _observations(observations)
        emissions = self._emissions(x)
        t_count, k = emissions.shape
        log_transition = _log_probabilities(self.transition)
        alpha = np.empty((t_count, k))
        scales = np.empty(t_count)
        alpha[0] = _log_probabilities(self.initial) + emissions[0]
        scales[0] = logsumexp(alpha[0])
        alpha[0] -= scales[0]
        for t in range(1, t_count):
            alpha[t] = emissions[t] + logsumexp(alpha[t - 1, :, None] + log_transition, axis=0)
            scales[t] = logsumexp(alpha[t])
            alpha[t] -= scales[t]
        beta = np.zeros_like(alpha)
        counts = np.zeros((k, k))
        for t in range(t_count - 2, -1, -1):
            beta[t] = logsumexp(
                log_transition + emissions[t + 1, None, :] + beta[t + 1, None, :], axis=1,
            ) - scales[t + 1]
            joint = alpha[t, :, None] + log_transition + emissions[t + 1, None, :] + beta[t + 1, None, :]
            counts += np.exp(joint - logsumexp(joint))
        gamma_log = alpha + beta
        gamma_log -= logsumexp(gamma_log, axis=1)[:, None]
        loglik = float(scales.sum())
        if not np.isfinite(loglik):
            raise ValueError("HMM likelihood numerical overflow")
        return HMMInference(
            _freeze(np.exp(alpha)), _freeze(np.exp(gamma_log)), _freeze(counts), loglik,
        )

    def decode(self, observations) -> np.ndarray:
        """Viterbi most likely complete path; uses the full sequence (offline only)."""
        x = _observations(observations)
        emissions = self._emissions(x)
        log_transition = _log_probabilities(self.transition)
        values = _log_probabilities(self.initial) + emissions[0]
        previous = np.zeros(emissions.shape, dtype=np.int64)
        for t in range(1, len(x)):
            candidates = values[:, None] + log_transition
            previous[t] = np.argmax(candidates, axis=0)
            values = candidates[previous[t], np.arange(len(self.initial))] + emissions[t]
        path = np.empty(len(x), dtype=np.int64)
        path[-1] = np.argmax(values)
        for t in range(len(x) - 2, -1, -1):
            path[t] = previous[t + 1, path[t + 1]]
        return _freeze(path)


@dataclass(frozen=True)
class HMMFit:
    model: GaussianHMM
    likelihood_history: tuple[float, ...]
    converged: bool
    iterations: int
    stop_reason: str
    evidence_state: str = "TRAINING_SAMPLE_ONLY"


def fit_gaussian_hmm(
    observations, states: int = 2, *, max_iterations: int = 100,
    tolerance: float = 1e-6, variance_floor: float = 1e-8,
) -> HMMFit:
    """Deterministic Baum–Welch EM with diagonal emissions, no hidden fallback.

    Local likelihood optimization is not evidence of the number of true regimes
    or an economically useful strategy. Sorting fitted means canonicalizes labels.
    """
    x = _observations(observations)
    if isinstance(states, bool) or not isinstance(states, int) or not 1 <= states <= 32 or len(x) < 2 * states:
        raise ValueError("states must be an integer 1..32 with >=2 observations per state")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or not 1 <= max_iterations <= 1000:
        raise ValueError("max_iterations must be an integer 1..1000")
    if not np.isfinite(tolerance) or tolerance <= 0 or not np.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("tolerance and variance_floor must be finite positive")
    means = np.quantile(x, (np.arange(states) + 0.5) / states, axis=0)
    variances = np.tile(np.maximum(x.var(axis=0), variance_floor), (states, 1))
    transition = np.full((states, states), 1.0 / states)
    if states > 1:
        transition = np.full((states, states), 0.1 / (states - 1))
        np.fill_diagonal(transition, 0.9)
    model = GaussianHMM(np.full(states, 1.0 / states), transition, means, variances)
    inference = model.infer(x)
    history = [inference.log_likelihood]
    converged, reason = False, "MAX_ITERATIONS"
    for _ in range(max_iterations):
        gamma = inference.smoothed_probabilities
        occupancy = gamma.sum(axis=0)
        new_means, new_variances = np.array(model.means), np.array(model.variances)
        for state in range(states):
            if occupancy[state] > 1e-12:
                new_means[state] = (gamma[:, state, None] * x).sum(axis=0) / occupancy[state]
                new_variances[state] = np.maximum(
                    (gamma[:, state, None] * (x - new_means[state]) ** 2).sum(axis=0) / occupancy[state],
                    variance_floor,
                )
        counts = np.array(inference.transition_counts)
        sums = counts.sum(axis=1)
        new_transition = np.array(model.transition)
        active = sums > 1e-12
        new_transition[active] = counts[active] / sums[active, None]
        candidate = GaussianHMM(gamma[0], new_transition, new_means, new_variances)
        candidate_inference = candidate.infer(x)
        improvement = candidate_inference.log_likelihood - history[-1]
        if improvement < -1e-8 * (1 + abs(history[-1])):
            reason = "LIKELIHOOD_DECREASE_REJECTED"
            break
        if improvement < 0:
            converged, reason = True, "NUMERICAL_LIKELIHOOD_PLATEAU"
            break
        model, inference = candidate, candidate_inference
        history.append(inference.log_likelihood)
        if improvement <= tolerance * (1 + abs(history[-2])):
            converged, reason = True, "RELATIVE_LIKELIHOOD_TOLERANCE"
            break
    order = np.argsort(model.means[:, 0], kind="stable")
    model = GaussianHMM(
        model.initial[order], model.transition[np.ix_(order, order)],
        model.means[order], model.variances[order],
    )
    return HMMFit(model, tuple(history), converged, len(history) - 1, reason)


def state_duration_diagnostics(path, transition) -> list[dict]:
    """Observed runs and geometric expected durations, with edge censoring.

    Sequence-edge runs may be left/right censored. They are recorded separately
    and excluded from complete-run means. Absorbing expected duration is None
    with ABSORBING state, never JSON infinity. Units are observation steps.
    """
    states = np.asarray(path)
    matrix = np.array(transition, dtype=float, copy=True)
    if states.ndim != 1 or states.size == 0 or states.dtype.kind not in "iu":
        raise ValueError("path must be a nonempty integer vector")
    if (
        matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1
        or not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix > 1).any()
        or not np.allclose(matrix.sum(axis=1), 1.0, rtol=0, atol=1e-12)
        or (states < 0).any() or (states >= len(matrix)).any()
    ):
        raise ValueError("invalid transition matrix or state identifiers")
    boundaries = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1, len(states)]
    runs = []
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        runs.append({
            "state": int(states[start]), "length": int(end - start),
            "start_index": int(start), "end_index_exclusive": int(end),
            "left_censored": i == 0, "right_censored": i == len(boundaries) - 2,
        })
    diagnostics = []
    for state in range(len(matrix)):
        own = [run for run in runs if run["state"] == state]
        complete = [run["length"] for run in own if not run["left_censored"] and not run["right_censored"]]
        absorbing = matrix[state, state] == 1
        diagnostics.append({
            "state": state, "unit": "OBSERVATION_STEPS", "runs": own,
            "observed_steps": sum(run["length"] for run in own),
            "complete_run_count": len(complete),
            "complete_run_mean": float(np.mean(complete)) if complete else None,
            "expected_duration": None if absorbing else float(1.0 / (1.0 - matrix[state, state])),
            "expected_duration_state": "ABSORBING" if absorbing else "AVAILABLE",
        })
    return diagnostics
