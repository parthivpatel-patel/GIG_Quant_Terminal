"""Additional bounded families on the shared fit/replay contract.

ARD, seasonal ARIMA, a rank-1 VECM and spectral clustering. Each fit is the
named library estimator. Replay is saved algebra or an explicit filter, never
a refit. Hawkes, N-HiTS, TFT and PatchTST are not approximated here.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from gig.ml.bounded.core import Family, register, unavailable
from gig.ml.bounded.latent import _structure_score
from gig.ml.bounded.supervised import vector

ARD_ITERATIONS, ARD_TOL, ARD_THRESHOLD = 300, 1e-6, 1e4
SEASON, SARIMA_ITERATIONS = 4, 200
SPECTRAL_CLUSTERS, SPECTRAL_GAMMA = 3, 1.0
_RESEARCH = " Uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved."
_CAUSAL = (" Row t forecasts observation t+1 from observations up to t only; requires a one-observation horizon and an"
           " unchanged training history at replay. Uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.")


def _ard_fit(z, y, _seed):
    from sklearn.linear_model import ARDRegression
    estimator = ARDRegression(max_iter=ARD_ITERATIONS, tol=ARD_TOL, threshold_lambda=ARD_THRESHOLD).fit(z, y)
    if estimator.n_iter_ >= ARD_ITERATIONS or not np.any(np.abs(estimator.coef_) > 0):
        raise ValueError("ARD regression did not converge with at least one retained coefficient")
    return ({"coef": estimator.coef_.tolist(), "intercept": float(estimator.intercept_),
             "weight_precision": estimator.lambda_.tolist(), "noise_precision": float(estimator.alpha_),
             "iterations": int(estimator.n_iter_)}, estimator.predict(z)[:, None])


def _ard_replay(payload, z):
    return (z@vector(payload, "coef", (z.shape[1],)) + vector(payload, "intercept", ()))[:, None]


def _ard_check(payload, z, y):
    """Posterior mean at the saved precisions, after the library's centering step."""
    precision = vector(payload, "weight_precision", (z.shape[1],))
    alpha = float(vector(payload, "noise_precision", ()))
    coef = vector(payload, "coef", (z.shape[1],))
    if alpha <= 0 or np.any(precision <= 0):
        raise ValueError("saved ARD precisions must be positive")
    keep = precision < ARD_THRESHOLD
    if not np.any(keep) or np.any(coef[~keep] != 0):
        raise ValueError("saved ARD coefficients disagree with the pruning threshold")
    centered_x, centered_y = z-z.mean(0), y-y.mean()
    gram = centered_x[:, keep].T@centered_x[:, keep]
    sigma = np.linalg.inv(np.diag(precision[keep])+alpha*gram)
    expected = alpha*sigma@centered_x[:, keep].T@centered_y
    intercept = float(y.mean()-z.mean(0)@coef)
    if (not np.allclose(coef[keep], expected, rtol=1e-8, atol=1e-8)
            or not np.isclose(intercept, float(vector(payload, "intercept", ())), rtol=0, atol=1e-10)):
        raise ValueError("saved ARD coefficients are not the posterior mean at the saved precisions")


def _transition(phi, seasonal):
    transition = np.zeros((5, 5))
    transition[0, 0], transition[0, 1] = phi, 1.
    transition[1, 2] = transition[2, 3] = transition[3, 4] = 1.
    transition[3, 0], transition[4, 0] = seasonal, -phi*seasonal
    return transition


def _sarima_parameters(payload):
    mu = float(vector(payload, "intercept", ()))
    phi = float(vector(payload, "ar", ()))
    seasonal = float(vector(payload, "seasonal_ar", ()))
    sigma2 = float(vector(payload, "sigma2", ()))
    if sigma2 <= 0 or not (abs(phi) < 1 and abs(seasonal) < 1):
        raise ValueError("saved seasonal autoregression is not stationary with positive innovation variance")
    transition = _transition(phi, seasonal)
    if np.max(np.abs(np.linalg.eigvals(transition))) >= 1-1e-8:
        raise ValueError("saved seasonal autoregression companion is not stationary")
    return mu, phi, seasonal, sigma2, transition


def _sarima_filter(payload, y):
    """Stationary Kalman filter in the statsmodels seasonal-AR state order."""
    mu, _phi, _seasonal, sigma2, transition = _sarima_parameters(payload)
    shock = np.zeros(5)
    shock[0] = 1.
    drift = np.zeros(5)
    drift[0] = mu
    mean = np.linalg.solve(np.eye(5)-transition, drift)
    covariance = solve_discrete_lyapunov(transition, np.outer(shock, shock)*sigma2)
    predicted = np.empty(len(y)+1)
    loglike = 0.
    for index, value in enumerate(y):
        predicted[index] = mean[0]
        innovation = value-mean[0]
        variance = covariance[0, 0]
        if variance <= 0:
            raise ValueError("saved seasonal autoregression has a nonpositive forecast variance")
        loglike -= .5*(np.log(2*np.pi)+np.log(variance)+innovation*innovation/variance)
        gain = covariance[:, 0]/variance
        mean = mean+gain*innovation
        covariance = covariance-np.outer(gain, covariance[0])
        mean = transition@mean+drift
        covariance = transition@covariance@transition.T+np.outer(shock, shock)*sigma2
    predicted[-1] = mean[0]
    return predicted, float(loglike)


def _sarima_fit(z, _target, _seed):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y = z[:, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = SARIMAX(y, order=(1, 0, 0), seasonal_order=(1, 0, 0, SEASON), trend="c").fit(
            disp=False, maxiter=SARIMA_ITERATIONS)
    if list(result.param_names) != ["intercept", "ar.L1", "ar.S.L4", "sigma2"]:
        raise ValueError("seasonal ARIMA parameter order differs from the declared state equation")
    if not result.mle_retvals.get("converged", False):
        raise ValueError("seasonal ARIMA did not converge within the fixed iteration budget")
    mu, phi, seasonal, sigma2 = (float(v) for v in result.params)
    payload = {"intercept": mu, "ar": phi, "seasonal_ar": seasonal, "sigma2": sigma2,
               "training_loglike": None, "optimizer_converged": True}
    predicted, loglike = _sarima_filter(payload, y)
    library = np.concatenate([np.asarray(result.predict())[1:], np.asarray(result.forecast(1))])
    if not np.allclose(predicted[1:], library, rtol=1e-8, atol=1e-8) or not np.isclose(loglike, result.llf, rtol=1e-8, atol=1e-8):
        raise ValueError("seasonal ARIMA filter does not reproduce the library fit")
    payload["training_loglike"] = loglike
    return payload, predicted[1:, None]


def _sarima_replay(payload, z):
    return _sarima_filter(payload, z[:, 0])[0][1:, None]


def _sarima_check(payload, z, _target):
    predicted, loglike = _sarima_filter(payload, z[:, 0])
    if predicted.shape != (len(z)+1,) or not np.isclose(loglike, vector(payload, "training_loglike", ()), rtol=1e-8, atol=1e-8):
        raise ValueError("saved seasonal autoregression disagrees with its training likelihood")


def _vecm_parts(payload, width):
    beta = vector(payload, "beta", (width,))
    alpha = vector(payload, "alpha", (width,))
    gamma = vector(payload, "gamma", (width, width))
    constant = float(vector(payload, "coint_constant", ()))
    if not np.isclose(beta[0], 1., rtol=0, atol=1e-8):
        raise ValueError("saved cointegrating vector must be normalized on the first series")
    return beta, alpha, gamma, constant


def _vecm_replay(payload, z):
    beta, alpha, gamma, constant = _vecm_parts(payload, z.shape[1])
    out = np.empty_like(z)
    out[0] = z[0]
    for index in range(1, len(z)):
        equilibrium = float(beta@z[index]+constant)
        change = alpha*equilibrium+gamma@(z[index]-z[index-1])
        out[index] = z[index]+change
    return out[:, :, None]


def _vecm_fit(z, _target, _seed):
    from statsmodels.tsa.vector_ar.vecm import VECM
    if z.shape[1] < 2:
        raise ValueError("VECM requires at least two series")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = VECM(z, k_ar_diff=1, coint_rank=1, deterministic="ci").fit()
    beta, alpha = np.asarray(result.beta, dtype=float), np.asarray(result.alpha, dtype=float)
    scale = float(beta[0, 0])
    if not np.isfinite(scale) or abs(scale) < 1e-8:
        raise ValueError("VECM could not normalize the cointegrating vector on the first series")
    beta, alpha = (beta/scale).ravel(), (alpha*scale).ravel()
    constant = float(np.asarray(result.const_coint).reshape(-1)[0]/scale)
    payload = {"beta": beta.tolist(), "alpha": alpha.tolist(), "gamma": np.asarray(result.gamma, dtype=float).tolist(),
               "coint_constant": constant}
    expected = _vecm_replay(payload, z)[:, :, 0]
    library = np.vstack([np.asarray(result.fittedvalues), np.asarray(result.predict(steps=1))])
    if not np.allclose(expected[1:], library, rtol=1e-8, atol=1e-8):
        raise ValueError("VECM replay does not reproduce the library fit")
    return payload, expected[:, :, None]


def _vecm_check(payload, z, _target):
    """Given the normalized cointegrating vector, alpha and gamma are least squares."""
    beta, alpha, gamma, constant = _vecm_parts(payload, z.shape[1])
    design, target = [], []
    for index in range(1, len(z)-1):
        design.append(np.r_[float(beta@z[index]+constant), z[index]-z[index-1]])
        target.append(z[index+1]-z[index])
    residual = np.asarray(target)-np.asarray(design)@np.column_stack([alpha, gamma]).T
    if np.max(np.abs(np.asarray(design).T@residual)) > 1e-6*len(z):
        raise ValueError("saved VECM loadings do not satisfy the training normal equations")


def _spectral_labels(z, seed):
    from sklearn.cluster import SpectralClustering
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SpectralClustering(n_clusters=SPECTRAL_CLUSTERS, affinity="rbf", gamma=SPECTRAL_GAMMA,
                                  assign_labels="discretize", random_state=seed).fit_predict(z)


def _spectral_centers(z, seed):
    labels = _spectral_labels(z, seed)
    if len(set(labels.tolist())) != SPECTRAL_CLUSTERS:
        raise ValueError("spectral clustering produced an empty cluster on its training rows")
    centers = np.array([z[labels == cluster].mean(0) for cluster in range(SPECTRAL_CLUSTERS)])
    return centers[np.lexsort(centers.T[::-1])]


def _nearest(centers, z):
    distance = ((z[:, None, :]-centers)**2).sum(2)
    return distance.argmin(1), distance.min(1)


def _spectral_fit(z, _target, seed):
    if z.shape[1] < 2:
        raise ValueError("spectral clustering requires at least two varying features")
    centers = _spectral_centers(z, seed)
    label, distance = _nearest(centers, z)
    return {"centers": centers.tolist(), "seed": int(seed)}, np.column_stack([distance, (z*z).sum(1), label])


def _spectral_replay(payload, z):
    label, distance = _nearest(vector(payload, "centers", (SPECTRAL_CLUSTERS, z.shape[1])), z)
    return np.column_stack([distance, (z*z).sum(1), label])


def _spectral_check(payload, z, _target):
    seed = payload.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("saved spectral clustering seed is invalid")
    centers = _spectral_centers(z, seed)
    if not np.allclose(centers, vector(payload, "centers", (SPECTRAL_CLUSTERS, z.shape[1])), rtol=1e-8, atol=1e-8):
        raise ValueError("saved spectral centers are not the means of the training spectral partition")


register(Family(
    key="ard", number=41, capability="automatic_relevance_determination",
    label="Automatic relevance determination · saved dataset",
    scope="Bounded CPU ARD regression on admitted features: evidence iteration with fixed priors, coefficients pruned at precision 10,000, at most 300 iterations, 20–256 training rows, 16 fits. A fit that keeps no coefficient is refused. The posterior mean is not a calibrated predictive interval." + _RESEARCH,
    kind="FORWARD_RETURN_ESTIMATE", fit=_ard_fit, replay=_ard_replay, check=_ard_check,
    backend="sklearn.linear_model.ARDRegression",
    parameters={"max_iter": ARD_ITERATIONS, "tol": ARD_TOL, "threshold_lambda": ARD_THRESHOLD},
    uncertainty=unavailable("ARD_NOISE_PRECISION_IS_NOT_A_CALIBRATED_INTERVAL"), tolerance=1e-8))
register(Family(
    key="sarima", number=42, capability="sarima", label="Seasonal ARIMA · return history",
    scope="Bounded CPU SARIMA(1,0,0)(1,0,0,4) with intercept by Gaussian state-space maximum likelihood on each instrument's admitted return history: 48–256 training rows, 200 iterations, 16 fits. Differencing and moving-average terms are unavailable. The seasonal period is fixed at 4, not estimated." + _CAUSAL,
    kind="NEXT_OBSERVATION_RETURN_ESTIMATE", fit=_sarima_fit, replay=_sarima_replay, check=_sarima_check,
    backend="statsmodels.tsa.statespace.sarimax.SARIMAX", requires=("statsmodels",), data="SERIES", min_train=48,
    parameters={"order": [1, 0, 0], "seasonal_order": [1, 0, 0, SEASON], "trend": "c", "maxiter": SARIMA_ITERATIONS},
    uncertainty=unavailable("INNOVATION_VARIANCE_IS_A_TRAINING_PARAMETER_NOT_A_CALIBRATED_INTERVAL"), tolerance=1e-8))
register(Family(
    key="vecm", number=43, capability="vecm", label="VECM one-step forecast · cross-instrument series",
    scope="Bounded CPU VECM with one lag difference, one cointegrating relation and a constant inside that relation, on the balanced admitted panel after training-only standardization: 2–8 series, 40–256 rows, 16 fits. The rank is fixed, not tested. The first forecast, before a difference exists, is the no-change value. This is not a price-level cointegration claim unless the admitted series are levels." + _CAUSAL,
    kind="NEXT_OBSERVATION_RETURN_ESTIMATE", fit=_vecm_fit, replay=_vecm_replay, check=_vecm_check,
    backend="statsmodels.tsa.vector_ar.vecm.VECM", requires=("statsmodels",), data="PANEL", min_train=40,
    max_columns=8, parameters={"k_ar_diff": 1, "coint_rank": 1, "deterministic": "ci"},
    uncertainty=unavailable("COINTEGRATING_RESIDUAL_IS_NOT_A_CALIBRATED_INTERVAL"), tolerance=1e-8))
register(Family(
    key="spectral", number=44, capability="spectral_clustering", label="Spectral clustering · saved dataset",
    scope="Bounded CPU spectral clustering on admitted features: fixed RBF affinity, three clusters, discretized normalized-Laplacian labels, 20–256 training rows, 16 fits. Saved centers are the training-cluster means. A new row receives the nearest saved center and does not rebuild the graph. Cluster labels are not market regimes."
           + " Labels are never read. Outputs are descriptive diagnostics, not return forecasts; uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.",
    kind="NEAREST_CENTER_SQUARED_DISTANCE", fit=_spectral_fit, replay=_spectral_replay, check=_spectral_check,
    backend="sklearn.cluster.SpectralClustering", outputs=("prediction", "baseline", "cluster"),
    parameters={"n_clusters": SPECTRAL_CLUSTERS, "affinity": "rbf", "gamma": SPECTRAL_GAMMA, "assign_labels": "discretize"},
    uncertainty=unavailable("HARD_CLUSTER_ASSIGNMENT_HAS_NO_PROBABILITY", "standardized_squared_distance"),
    score=_structure_score(False), tolerance=1e-8, supervised=False, return_units=False, affine=()))
