"""Families 31-33: bounded dependence, tail and change-point research on return histories.

Outputs are model diagnostics on admitted histories. None is a calibrated
risk limit; exceedance and scoring statistics are reported so failures show.
"""
from __future__ import annotations

import numpy as np

from gig.ml.bounded.core import Family, register, return_score, unavailable
from gig.ml.bounded.supervised import pinball, vector

TAIL_LEVEL, THRESHOLD_QUANTILE, MIN_EXCEEDANCES = .01, .9, 10
SHAPE_BOUNDS = (-.5, .95)  # regular maximum-likelihood region with a finite expected shortfall
HAZARD, RUN_LIMIT, RECENT = 1/50, 256, 5
PRIOR = {"mean": 0., "kappa": 1., "alpha": 1., "beta": 1.}


def _normal_scores(reference, z):
    """Training empirical distribution mapped to normal scores; holdout ranks never refit it."""
    from scipy.special import ndtri
    count = len(reference)
    rank = np.column_stack([np.searchsorted(np.sort(reference[:, j]), z[:, j], side="right") for j in range(z.shape[1])])
    return ndtri((rank+.5)/(count+1))


def _copula_fit(z, _target, _seed):
    correlation = np.corrcoef(_normal_scores(z, z).T)
    return {"correlation": correlation.tolist(), "training_sorted": np.sort(z, axis=0).tolist()}, None


def _copula_parts(payload, width):
    correlation = vector(payload, "correlation", (width, width))
    rows = payload.get("training_sorted")
    reference = vector(payload, "training_sorted", (len(rows) if isinstance(rows, list) else 0, width))
    if (not np.allclose(correlation, correlation.T, atol=1e-12) or not np.allclose(np.diag(correlation), 1., atol=1e-9)
            or np.linalg.eigvalsh(correlation).min() <= 1e-8):
        raise ValueError("saved copula correlation must be a positive-definite correlation matrix")
    return correlation, reference


def _copula_replay(payload, z):
    correlation, reference = _copula_parts(payload, z.shape[1])
    scores = _normal_scores(reference, z)
    density = -.5*np.linalg.slogdet(correlation)[1]-.5*np.einsum("ij,jk,ik->i", scores, np.linalg.inv(correlation)-np.eye(z.shape[1]), scores)
    return np.column_stack([density, np.zeros(len(z))])


def _copula_check(payload, z, _target):
    correlation, reference = _copula_parts(payload, z.shape[1])
    if not np.array_equal(reference, np.sort(z, axis=0)) or not np.allclose(correlation, np.corrcoef(_normal_scores(z, z).T), atol=1e-10):
        raise ValueError("saved copula is not the normal-score correlation of its training panel")


def _copula_score(_observed, outputs, _train, _payload):
    model = float(np.mean(outputs["prediction"]))
    return {"holdout_copula_log_density": model, "holdout_independence_baseline_log_density": 0.,
            "holdout_improvement_over_baseline": model, "holdout_beats_baseline_fraction": float(model > 0),
            "tail_dependence": 0.}


def _gpd_loglike(excess, shape, scale):
    if scale <= 0 or np.any(1+shape*excess/scale <= 0):
        return -np.inf
    if abs(shape) < 1e-9:
        return float(-len(excess)*np.log(scale)-excess.sum()/scale)
    return float(-len(excess)*np.log(scale)-(1+1/shape)*np.log1p(shape*excess/scale).sum())


def _tail_parts(payload):
    threshold, shape, scale = (vector(payload, k, ()) for k in ("threshold", "shape", "scale"))
    count, total = payload.get("exceedances"), payload.get("training_rows")
    if (scale <= 0 or type(count) is not int or type(total) is not int or not MIN_EXCEEDANCES <= count < total
            or not SHAPE_BOUNDS[0] <= shape <= SHAPE_BOUNDS[1]):
        raise ValueError("saved tail model needs a positive scale, bounded shape and at least ten exceedances")
    return float(threshold), float(shape), float(scale), count, total


def _tail_levels(threshold, shape, scale, count, total):
    ratio = total*TAIL_LEVEL/count
    quantile = threshold+scale*(-np.log(ratio) if abs(shape) < 1e-9 else (ratio**-shape-1)/shape)
    shortfall = quantile/(1-shape)+(scale-shape*threshold)/(1-shape) if shape < 1 else np.nan
    return quantile, shortfall


def _tail_fit(z, _target, _seed):
    from scipy.optimize import minimize
    from scipy.stats import genpareto
    loss = -z[:, 0]
    threshold = float(np.quantile(loss, THRESHOLD_QUANTILE))
    excess = loss[loss > threshold]-threshold
    if len(excess) < MIN_EXCEEDANCES:
        raise ValueError("peaks-over-threshold requires at least ten training exceedances")
    def objective(parameters):
        value = -float(genpareto.logpdf(excess, parameters[0], 0., parameters[1]).sum())
        return value if np.isfinite(value) else 1e12
    result = minimize(objective, [.1, float(excess.mean())], method="Nelder-Mead", bounds=[SHAPE_BOUNDS, (1e-9, 1e6)],
                      options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000, "maxfev": 8000})
    shape, scale = (float(v) for v in result.x)
    return ({"threshold": threshold, "shape": float(shape), "scale": float(scale), "exceedances": len(excess),
             "training_rows": len(loss), "optimizer_converged": bool(result.success), "training_empirical_quantile": float(np.quantile(z[:, 0], TAIL_LEVEL))}, None)


def _tail_replay(payload, z):
    quantile, shortfall = _tail_levels(*_tail_parts(payload))
    return np.tile([-quantile, -shortfall, vector(payload, "training_empirical_quantile", ())], (len(z), 1))


def _tail_check(payload, z, _target):
    """Threshold, exceedances and the Pareto likelihood optimum are recomputed from training rows."""
    threshold, shape, scale, count, total = _tail_parts(payload)
    loss = -z[:, 0]
    excess = loss[loss > threshold]-threshold
    if (total != len(loss) or count != len(excess) or not np.isclose(threshold, np.quantile(loss, THRESHOLD_QUANTILE), atol=1e-12)
            or not np.isclose(vector(payload, "training_empirical_quantile", ()), np.quantile(z[:, 0], TAIL_LEVEL), atol=1e-12)):
        raise ValueError("saved tail threshold or exceedances differ from the training history")
    base = _gpd_loglike(excess, shape, scale)
    for a, b in [(shape+1e-3, scale), (shape-1e-3, scale), (shape, scale*1.001), (shape, scale/1.001)]:
        if SHAPE_BOUNDS[0] <= a <= SHAPE_BOUNDS[1] and _gpd_loglike(excess, a, b) > base+1e-6:
            raise ValueError("saved generalized Pareto parameters are not a likelihood optimum")


def _tail_score(observed, outputs, _train, payload):
    model, baseline = outputs["prediction"], outputs["baseline"]
    return {"nominal_tail_probability": TAIL_LEVEL,
            "holdout_violation_fraction": float(np.mean(observed < model)),
            "holdout_empirical_baseline_violation_fraction": float(np.mean(observed < baseline)),
            "holdout_quantile_loss": pinball(observed, model, TAIL_LEVEL),
            "holdout_empirical_baseline_quantile_loss": pinball(observed, baseline, TAIL_LEVEL),
            "tail_shape": float(payload["shape"]), "expected_shortfall_finite_fraction": float(payload["shape"] < 1),
            "coverage_tested": False}


def _tail_uncertainty(outputs):
    return {"value": None, "expected_shortfall": outputs["expected_shortfall"],
            "state": "UNCONDITIONAL_TRAINING_TAIL_NOT_CALIBRATED_RISK_LIMIT", "method": "PEAKS_OVER_THRESHOLD_GENERALIZED_PARETO",
            "tail_probability": TAIL_LEVEL, "unit": "arithmetic_return", "calibrated": False, "predictive_interval": False}


def _run_lengths(z):
    """Adams-MacKay run-length filter with a Normal-Inverse-Gamma prior on standardized returns."""
    from scipy.special import gammaln
    mean, kappa, alpha, beta = (np.array([PRIOR[k]]) for k in ("mean", "kappa", "alpha", "beta"))
    weight, out = np.array([1.]), np.empty((len(z), 3))
    for index, value in enumerate(z):
        freedom, spread = 2*alpha, beta*(kappa+1)/(alpha*kappa)
        density = np.exp(gammaln((freedom+1)/2)-gammaln(freedom/2)-.5*np.log(np.pi*freedom*spread)
                         - (freedom+1)/2*np.log1p((value-mean)**2/(freedom*spread)))
        grown = weight*density
        weight = np.r_[HAZARD*grown.sum(), (1-HAZARD)*grown][:RUN_LIMIT]
        weight /= weight.sum()
        updated = (kappa*mean+value)/(kappa+1)
        beta = np.r_[PRIOR["beta"], beta+kappa*(value-mean)**2/(2*(kappa+1))][:RUN_LIMIT]
        mean = np.r_[PRIOR["mean"], updated][:RUN_LIMIT]
        kappa = np.r_[PRIOR["kappa"], kappa+1][:RUN_LIMIT]
        alpha = np.r_[PRIOR["alpha"], alpha+.5][:RUN_LIMIT]
        out[index] = weight@mean, weight[:RECENT].sum(), weight@np.arange(len(weight))
    return out


def _change_fit(z, _target, _seed):
    return {"training_final": _run_lengths(z[:, 0])[-1].tolist()}, None


def _change_replay(payload, z):
    vector(payload, "training_final", (3,))
    return _run_lengths(z[:, 0])


def _change_check(payload, z, _target):
    if not np.allclose(vector(payload, "training_final", (3,)), _run_lengths(z[:, 0])[-1], rtol=1e-10, atol=1e-12):
        raise ValueError("saved change-point summary differs from the filter on its training history")


def _change_score(observed, outputs, train_target, payload):
    return {**return_score(observed, outputs, train_target, payload),
            "holdout_mean_recent_change_probability": float(np.mean(outputs["recent_change_probability"])),
            "holdout_mean_expected_run_length": float(np.mean(outputs["expected_run_length"]))}


def _change_uncertainty(outputs):
    return {"value": None, "recent_change_probability": outputs["recent_change_probability"],
            "expected_run_length": outputs["expected_run_length"],
            "state": "MODEL_POSTERIOR_NOT_CALIBRATED_CHANGE_PROBABILITY", "method": "ADAMS_MACKAY_RUN_LENGTH_POSTERIOR",
            "unit": "arithmetic_return", "calibrated": False, "predictive_interval": False}


_HISTORY = " Requires a one-observation horizon and an unchanged training history at replay. Allocation, promotion and trading remain unapproved."
register(Family(
    key="gaussian_copula", number=31, capability="copula_dependence", label="Gaussian copula dependence · cross-instrument returns",
    scope="Bounded CPU Gaussian copula on the balanced cross-instrument return panel: training empirical margins, normal-score correlation, 2–8 instruments, 30–256 training dates, 16 fits. Reports holdout copula log-density against independence. A Gaussian copula has zero tail dependence; Student-t and vine copulas are unavailable." + _HISTORY,
    kind="JOINT_COPULA_LOG_DENSITY", fit=_copula_fit, replay=_copula_replay, check=_copula_check,
    backend="scipy.special.ndtri + numpy.corrcoef", requires=("scipy",), data="PANEL", joint=True, min_train=30,
    max_columns=8, outputs=("prediction", "baseline"), affine=(), return_units=False,
    parameters={"margins": "TRAINING_EMPIRICAL", "copula": "GAUSSIAN"}, score=_copula_score,
    uncertainty=unavailable("COPULA_DENSITY_IS_NOT_A_PREDICTIVE_INTERVAL", "log_density")))
register(Family(
    key="evt_pot", number=32, capability="extreme_value_tail", label="Extreme-value tail (POT) · return history",
    scope="Bounded CPU peaks-over-threshold on each instrument's admitted return history: generalized Pareto maximum likelihood (shape bounded to [-0.5, 0.95]) on losses above the training 90% threshold, at least 100 training rows and 10 exceedances, 16 fits. Emits an unconditional 1% return quantile and expected shortfall; holdout violations and quantile loss are compared with the training empirical quantile. Not a calibrated risk limit." + _HISTORY,
    kind="NEXT_OBSERVATION_RETURN_QUANTILE_1PCT", fit=_tail_fit, replay=_tail_replay, check=_tail_check,
    backend="scipy.stats.genpareto likelihood + scipy.optimize.minimize (bounded Nelder-Mead)", requires=("scipy",), data="SERIES", min_train=100,
    outputs=("prediction", "expected_shortfall", "baseline"), affine=("prediction", "expected_shortfall", "baseline"),
    return_units=False, parameters={"threshold_quantile": THRESHOLD_QUANTILE, "tail_probability": TAIL_LEVEL,
                                    "min_exceedances": MIN_EXCEEDANCES, "shape_bounds": list(SHAPE_BOUNDS)},
    score=_tail_score, uncertainty=_tail_uncertainty))
register(Family(
    key="bocpd", number=33, capability="bayesian_change_point", label="Bayesian online change point · return history",
    scope="Bounded CPU Bayesian online change-point detection (Adams–MacKay) on each instrument's admitted return history: Normal-Inverse-Gamma prior on training-standardized returns, constant hazard 1/50, run length truncated at 256, 20–256 training rows, 16 fits. Emits the filtered next-return mean, the posterior probability of a change within five observations and the expected run length; posterior probabilities are not calibrated." + _HISTORY,
    kind="NEXT_OBSERVATION_RETURN_ESTIMATE", fit=_change_fit, replay=_change_replay, check=_change_check,
    backend="gig.ml.bounded.tails (NumPy Adams-MacKay recursion, scipy.special.gammaln)", requires=("scipy",),
    data="SERIES", outputs=("prediction", "recent_change_probability", "expected_run_length"),
    parameters={"hazard": HAZARD, "run_limit": RUN_LIMIT, "recent_window": RECENT, "prior": dict(PRIOR)},
    score=_change_score, uncertainty=_change_uncertainty, license="MIT"))
