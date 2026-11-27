"""Families 22-25: bounded one-step forecasts from admitted return histories.

Each replay is a prefix-causal recursion: output row t is the forecast of
observation t+1 from observations 0..t only. Fits are actual statsmodels
estimators; nothing is smoothed with future observations.
"""
from __future__ import annotations

import warnings

import numpy as np

from gig.ml.bounded.core import Family, register, return_score, unavailable
from gig.ml.bounded.supervised import vector

ITERATIONS = 200


def _ahead(result):
    """Library one-step-ahead values aligned so row t holds the forecast of t+1."""
    inside = np.asarray(result.predict())
    return np.concatenate([inside[1:], np.asarray(result.forecast(1)).reshape((1, *inside.shape[1:]))])


def _ar_parameters(payload):
    mu, phi, sigma2 = vector(payload, "mean", ()), vector(payload, "phi", (2,)), vector(payload, "sigma2", ())
    if sigma2 <= 0 or not (abs(phi[1]) < 1 and phi[0]+phi[1] < 1 and phi[1]-phi[0] < 1):
        raise ValueError("saved autoregression is not stationary with positive innovation variance")
    return float(mu), phi, float(sigma2)


def _ar_loglike(y, mu, phi, sigma2):
    """Exact Gaussian likelihood of a stationary AR(2), from the stationary law of the first two rows."""
    rho = phi[0]/(1-phi[1])
    gamma = sigma2*(1-phi[1])/((1+phi[1])*((1-phi[1])**2-phi[0]**2))
    d, start = y-mu, y[:2]-mu
    quad = (start[0]**2-2*rho*start[0]*start[1]+start[1]**2)/(gamma*(1-rho**2))
    residual = d[2:]-phi[0]*d[1:-1]-phi[1]*d[:-2]
    return float(-.5*(2*np.log(2*np.pi)+2*np.log(gamma)+np.log(1-rho**2)+quad)
                 - .5*(len(residual)*np.log(2*np.pi*sigma2)+residual@residual/sigma2))


def _arima_fit(z, _target, _seed):
    from statsmodels.tsa.arima.model import ARIMA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ARIMA(z[:, 0], order=(2, 0, 0), trend="c").fit()
    mu, first, second, sigma2 = (float(v) for v in result.params)
    return ({"mean": mu, "phi": [first, second], "sigma2": sigma2,
             "optimizer_converged": bool(result.mle_retvals.get("converged", False)),
             "training_loglike": float(result.llf)}, _ahead(result)[:, None])


def _arima_replay(payload, z):
    mu, phi, _ = _ar_parameters(payload)
    d = z[:, 0]-mu
    out = mu+phi[0]*d
    out[1:] += phi[1]*d[:-1]
    out[0] = mu+phi[0]/(1-phi[1])*d[0]
    return out[:, None]


def _arima_check(payload, z, _target):
    """Saved parameters must be a local maximum of the independent exact likelihood."""
    mu, phi, sigma2 = _ar_parameters(payload)
    base = _ar_loglike(z[:, 0], mu, phi, sigma2)
    if not np.isclose(base, vector(payload, "training_loglike", ()), rtol=1e-7, atol=1e-6):
        raise ValueError("saved autoregression disagrees with its training likelihood")
    for index in range(4):
        for sign in (-1., 1.):
            trial = np.array([mu, *phi, sigma2])
            trial[index] += sign*1e-3
            with np.errstate(all="ignore"):
                other = _ar_loglike(z[:, 0], trial[0], trial[1:3], trial[3])
            if np.isfinite(other) and other > base+1e-4:
                raise ValueError("saved autoregression is not a likelihood optimum on its training history")


def _ets_parameters(payload):
    alpha, level = vector(payload, "smoothing_level", ()), vector(payload, "initial_level", ())
    if not 0 < alpha < 1:
        raise ValueError("saved smoothing level must lie strictly inside (0, 1)")
    return float(alpha), float(level)


def _ets_levels(alpha, level, y):
    out = np.empty(len(y))
    for index, value in enumerate(y):
        level += alpha*(value-level)
        out[index] = level
    return out


def _ets_fit(z, _target, _seed):
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ETSModel(z[:, 0], error="add", trend=None, seasonal=None).fit(disp=False, maxiter=ITERATIONS)
    alpha, level = (float(v) for v in result.params[:2])
    return ({"smoothing_level": alpha, "initial_level": level,
             "optimizer_converged": bool(result.mle_retvals.get("converged", False))}, _ahead(result)[:, None])


def _ets_replay(payload, z):
    return _ets_levels(*_ets_parameters(payload), z[:, 0])[:, None]


def _ets_check(payload, z, _target):
    """Additive-error ETS maximum likelihood is least squares of one-step errors."""
    alpha, level = _ets_parameters(payload)
    y = z[:, 0]

    def sse(a, start):
        return float(np.sum((y-np.r_[start, _ets_levels(a, start, y)[:-1]])**2))
    base = sse(alpha, level)
    for a, start in [(alpha+1e-3, level), (alpha-1e-3, level), (alpha, level+1e-3), (alpha, level-1e-3)]:
        if 1e-4 <= a <= 1-1e-4 and sse(a, start) < base-1e-5*(1+base):
            raise ValueError("saved exponential smoothing is not a least-squares optimum on its training history")


def _var_fit(z, _target, _seed):
    from statsmodels.tsa.api import VAR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = VAR(z).fit(1, trend="c")
    params = np.asarray(result.params)
    expected = np.vstack([np.asarray(result.fittedvalues), result.forecast(z[-1:], 1)])
    return {"intercept": params[0].tolist(), "coefficients": params[1:].tolist()}, expected[:, :, None]


def _var_replay(payload, z):
    width = z.shape[1]
    return (vector(payload, "intercept", (width,))+z@vector(payload, "coefficients", (width, width)))[:, :, None]


def _var_check(payload, z, _target):
    """Equation-by-equation least-squares normal equations on the training panel."""
    design = np.c_[np.ones(len(z)-1), z[:-1]]
    residual = z[1:]-_var_replay(payload, z[:-1])[:, :, 0]
    if np.max(np.abs(design.T@residual)) > 1e-7*len(z):
        raise ValueError("saved vector autoregression does not satisfy its training normal equations")


def _factor_parameters(payload, width):
    loadings, noise = vector(payload, "loadings", (width,)), vector(payload, "idiosyncratic_variance", (width,))
    phi = vector(payload, "factor_autoregression", ())
    if np.any(noise <= 0) or not abs(phi) < 1:
        raise ValueError("saved dynamic factor needs positive idiosyncratic variances and a stationary factor")
    return loadings, noise, float(phi)


def _factor_fit(z, _target, _seed):
    from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
    width = z.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = DynamicFactor(z, k_factors=1, factor_order=1).fit(disp=False, maxiter=ITERATIONS)
    params = np.asarray(result.params)
    return ({"loadings": params[:width].tolist(), "idiosyncratic_variance": params[width:2*width].tolist(),
             "factor_autoregression": float(params[-1]), "training_loglike": float(result.llf),
             "optimizer_converged": bool(result.mle_retvals.get("converged", False))}, _ahead(result)[:, :, None])


def _factor_filter(payload, z):
    """Scalar-state Kalman filter with the stationary prior; returns forecasts and log-likelihood."""
    loadings, noise, phi = _factor_parameters(payload, z.shape[1])
    state, variance, out, loglike = 0., 1/(1-phi**2), np.empty_like(z), 0.
    for index, row in enumerate(z):
        innovation = row-loadings*state
        covariance = np.outer(loadings, loadings)*variance+np.diag(noise)
        solved = np.linalg.solve(covariance, np.c_[loadings, innovation])
        loglike -= .5*(len(row)*np.log(2*np.pi)+np.linalg.slogdet(covariance)[1]+innovation@solved[:, 1])
        gain = variance*solved[:, 0]
        state, variance = state+gain@innovation, (1-gain@loadings)*variance
        state, variance = phi*state, phi*phi*variance+1
        out[index] = loadings*state
    return out, float(loglike)


def _factor_replay(payload, z):
    return _factor_filter(payload, z)[0][:, :, None]


def _factor_check(payload, z, _target):
    """The independent filter must reproduce the fitted training likelihood."""
    loglike = _factor_filter(payload, z)[1]
    if not np.isclose(loglike, vector(payload, "training_loglike", ()), rtol=1e-7, atol=1e-6):
        raise ValueError("saved dynamic factor parameters disagree with their training likelihood")


def _history_score(observed, outputs, train_target, payload):
    return {**return_score(observed, outputs, train_target, payload),
            "optimizer_converged_fraction": float(bool(payload.get("optimizer_converged", True)))}


_COMMON = (" Row t forecasts observation t+1 from observations up to t only; requires a one-observation horizon and an"
           " unchanged training history at replay. Uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.")
register(Family(
    key="arima", number=22, capability="arima", label="AR(2) ARIMA forecast · return history",
    scope="Bounded CPU ARIMA(2,0,0) with mean by exact Gaussian maximum likelihood on each instrument's admitted return history: 30–256 training rows, 16 fits. Differencing, MA and seasonal terms are unavailable." + _COMMON,
    kind="NEXT_OBSERVATION_RETURN_ESTIMATE", fit=_arima_fit, replay=_arima_replay, check=_arima_check,
    backend="statsmodels.tsa.arima.model.ARIMA", requires=("statsmodels",), data="SERIES", min_train=30,
    parameters={"order": [2, 0, 0], "trend": "c", "estimator": "EXACT_GAUSSIAN_MLE"}, score=_history_score,
    uncertainty=unavailable("INNOVATION_VARIANCE_IS_A_TRAINING_PARAMETER_NOT_A_CALIBRATED_INTERVAL"), tolerance=1e-8))
register(Family(
    key="ets", number=23, capability="exponential_smoothing", label="Exponential smoothing ETS(A,N,N) · return history",
    scope="Bounded CPU additive-error simple exponential smoothing (no trend, no seasonality) fitted by maximum likelihood on each instrument's admitted return history: 30–256 training rows, 16 fits." + _COMMON,
    kind="NEXT_OBSERVATION_RETURN_ESTIMATE", fit=_ets_fit, replay=_ets_replay, check=_ets_check,
    backend="statsmodels.tsa.exponential_smoothing.ets.ETSModel", requires=("statsmodels",), data="SERIES",
    min_train=30, parameters={"error": "add", "trend": None, "seasonal": None, "maxiter": ITERATIONS},
    score=_history_score,
    uncertainty=unavailable("SMOOTHED_LEVEL_HAS_NO_CALIBRATED_PREDICTIVE_INTERVAL"), tolerance=1e-9))
register(Family(
    key="var", number=24, capability="vector_autoregression", label="VAR(1) forecast · cross-instrument returns",
    scope="Bounded CPU VAR(1) with intercept by least squares on the balanced cross-instrument return panel: 2–8 instruments on identical decision times, 30–256 training dates, 16 fits. Cointegration (VECM) and higher lags are unavailable." + _COMMON,
    kind="NEXT_OBSERVATION_RETURN_ESTIMATE", fit=_var_fit, replay=_var_replay, check=_var_check,
    backend="statsmodels.tsa.api.VAR", requires=("statsmodels",), data="PANEL", min_train=30, max_columns=8,
    parameters={"lags": 1, "trend": "c"},
    uncertainty=unavailable("RESIDUAL_COVARIANCE_IS_NOT_A_CALIBRATED_INTERVAL"), tolerance=1e-9))
register(Family(
    key="dynamic_factor", number=25, capability="dynamic_factor", label="Dynamic factor forecast · cross-instrument returns",
    scope="Bounded CPU one-factor AR(1) dynamic factor model by state-space maximum likelihood on the balanced cross-instrument return panel: 2–8 instruments, 30–256 training dates, 200 optimizer iterations, 16 fits. Forecasts use filtered (never smoothed) factor states; optimizer non-convergence is reported." + _COMMON,
    kind="NEXT_OBSERVATION_RETURN_ESTIMATE", fit=_factor_fit, replay=_factor_replay, check=_factor_check,
    backend="statsmodels.tsa.statespace.dynamic_factor.DynamicFactor", requires=("statsmodels",), data="PANEL",
    min_train=30, max_columns=8, parameters={"k_factors": 1, "factor_order": 1, "maxiter": ITERATIONS},
    score=_history_score,
    uncertainty=unavailable("FILTERED_FACTOR_VARIANCE_IS_NOT_A_CALIBRATED_INTERVAL"), tolerance=1e-6))
