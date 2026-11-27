"""Families 26-30 and 40: bounded unsupervised structure on admitted features.

No labels are read. Each replay also emits the matching no-structure
baseline, so out-of-sample folds report whether learned structure generalizes.
Outputs are descriptive research diagnostics, never return forecasts.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from gig.ml.bounded.core import Family, register, unavailable
from gig.ml.bounded.supervised import vector

GMM_COMPONENTS, GMM_ITERATIONS, GMM_FLOOR = 2, 512, 1e-6
CLUSTERS, KMEANS_STARTS, KMEANS_ITERATIONS = 3, 4, 300
COMPONENTS = 2
ICA_ITERATIONS, NMF_ITERATIONS = 512, 2000
AE_STEPS, AE_RATE, AE_DECAY = 300, .02, 1e-4


def _needs(z, columns, name):
    if z.shape[1] < columns:
        raise ValueError(f"{name} requires at least {columns} varying training features")


def _canonical_rows(matrix):
    """Largest-magnitude weight of every component positive; a fixed sign convention."""
    pivot = matrix[np.arange(len(matrix)), np.argmax(np.abs(matrix), axis=1)]
    return matrix*np.where(pivot < 0, -1., 1.)[:, None]


def _structure_score(better_when_higher):
    def score(_observed, outputs, _train, payload):
        model, baseline = float(np.mean(outputs["prediction"])), float(np.mean(outputs["baseline"]))
        gain = model-baseline if better_when_higher else baseline-model
        result = {"holdout_model_score": model, "holdout_no_structure_baseline_score": baseline,
                  "holdout_improvement_over_baseline": gain, "holdout_beats_baseline_fraction": float(gain > 0)}
        if "converged" in payload:
            result["optimizer_converged_fraction"] = float(bool(payload["converged"]))
        return result
    return score


def _gmm_fit(z, _target, seed):
    from sklearn.mixture import GaussianMixture
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GaussianMixture(n_components=GMM_COMPONENTS, covariance_type="diag", reg_covar=GMM_FLOOR, tol=1e-8,
                                max_iter=GMM_ITERATIONS, n_init=1, random_state=seed).fit(z)
    order = np.lexsort(model.means_.T[::-1])
    payload = {"weights": model.weights_[order].tolist(), "means": model.means_[order].tolist(),
               "variances": model.covariances_[order].tolist(), "converged": bool(model.converged_),
               "iterations": int(model.n_iter_)}
    posterior = model.predict_proba(z)[:, order]
    return payload, np.column_stack([model.score_samples(z), _gaussian(z), posterior.argmax(1), posterior.max(1)])


def _gaussian(z):
    return -.5*(z*z).sum(1)-.5*z.shape[1]*np.log(2*np.pi)


def _gmm_parts(payload, width):
    weights = vector(payload, "weights", (GMM_COMPONENTS,))
    means, variances = (vector(payload, k, (GMM_COMPONENTS, width)) for k in ("means", "variances"))
    if np.any(weights <= 0) or abs(weights.sum()-1) > 1e-9 or np.any(variances < GMM_FLOOR*.999):
        raise ValueError("saved mixture needs positive weights summing to one and floored variances")
    return weights, means, variances


def _gmm_replay(payload, z):
    weights, means, variances = _gmm_parts(payload, z.shape[1])
    joint = (np.log(weights)-.5*np.log(2*np.pi*variances).sum(1)
             - .5*(((z[:, None, :]-means)**2)/variances).sum(2))
    top = joint.max(1)
    density = top+np.log(np.exp(joint-top[:, None]).sum(1))
    posterior = np.exp(joint-density[:, None])
    return np.column_stack([density, _gaussian(z), posterior.argmax(1), posterior.max(1)])


def _gmm_check(payload, z, _target):
    """A converged EM fit is a fixed point: one more M-step returns the saved moments."""
    weights, means, variances = _gmm_parts(payload, z.shape[1])
    if not payload.get("converged"):
        return
    joint = np.log(weights)-.5*np.log(2*np.pi*variances).sum(1)-.5*(((z[:, None, :]-means)**2)/variances).sum(2)
    posterior = np.exp(joint-joint.max(1)[:, None])
    posterior /= posterior.sum(1)[:, None]
    mass = posterior.sum(0)
    updated = posterior.T@z/mass[:, None]
    if not np.allclose(mass/len(z), weights, atol=1e-3) or not np.allclose(updated, means, atol=1e-2):
        raise ValueError("saved mixture is not an expectation-maximization fixed point of its training rows")


def _nearest(centers, z):
    distance = ((z[:, None, :]-centers)**2).sum(2)
    return distance.argmin(1), distance.min(1)


def _kmeans_fit(z, _target, seed):
    from sklearn.cluster import KMeans
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = KMeans(n_clusters=CLUSTERS, n_init=KMEANS_STARTS, max_iter=KMEANS_ITERATIONS, tol=0.,
                       algorithm="lloyd", random_state=seed).fit(z)
    centers = model.cluster_centers_[np.lexsort(model.cluster_centers_.T[::-1])]
    label, distance = _nearest(centers, z)
    return ({"centers": centers.tolist(), "converged": bool(model.n_iter_ < KMEANS_ITERATIONS),
             "iterations": int(model.n_iter_)}, np.column_stack([distance, (z*z).sum(1), label]))


def _kmeans_replay(payload, z):
    label, distance = _nearest(vector(payload, "centers", (CLUSTERS, z.shape[1])), z)
    return np.column_stack([distance, (z*z).sum(1), label])


def _kmeans_check(payload, z, _target):
    """Lloyd convergence: every center is the mean of the training rows nearest to it."""
    centers = vector(payload, "centers", (CLUSTERS, z.shape[1]))
    label, _ = _nearest(centers, z)
    if len(set(label.tolist())) != CLUSTERS:
        raise ValueError("saved clustering has an empty cluster on its training rows")
    if payload.get("converged") and not np.allclose(
            np.array([z[label == c].mean(0) for c in range(CLUSTERS)]), centers, atol=1e-8):
        raise ValueError("saved centers are not the means of their training clusters")


def _projection(components, mean, z):
    scores = (z-mean)@components.T
    error = ((z-mean-scores@components)**2).sum(1)
    return np.column_stack([error, ((z-mean)**2).sum(1), scores])


def _pca_fit(z, _target, _seed):
    from sklearn.decomposition import PCA
    _needs(z, COMPONENTS+1, "principal components")
    model = PCA(n_components=COMPONENTS, svd_solver="full").fit(z)
    components = _canonical_rows(model.components_)
    return ({"components": components.tolist(), "mean": model.mean_.tolist(),
             "explained_variance_ratio": model.explained_variance_ratio_.tolist()},
            _projection(components, model.mean_, z))


def _pca_replay(payload, z):
    return _projection(vector(payload, "components", (COMPONENTS, z.shape[1])), vector(payload, "mean", (z.shape[1],)), z)


def _pca_check(payload, z, _target):
    """Saved directions must be the leading orthonormal eigenvectors of the training covariance."""
    components, mean = vector(payload, "components", (COMPONENTS, z.shape[1])), vector(payload, "mean", (z.shape[1],))
    covariance = np.cov(z.T, bias=True)
    values = np.linalg.eigvalsh(covariance)[::-1]
    explained = np.einsum("ij,jk,ik->i", components, covariance, components)
    if (not np.allclose(mean, z.mean(0), atol=1e-9) or not np.allclose(components@components.T, np.eye(COMPONENTS), atol=1e-8)
            or not np.allclose(explained, values[:COMPONENTS], rtol=1e-7, atol=1e-9)
            or not np.allclose(components@covariance, explained[:, None]*components, atol=1e-7)):
        raise ValueError("saved principal directions are not leading eigenvectors of the training covariance")


def _ica_fit(z, _target, seed):
    from sklearn.decomposition import FastICA
    _needs(z, COMPONENTS, "independent components")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = FastICA(n_components=COMPONENTS, whiten="unit-variance", fun="logcosh", max_iter=ICA_ITERATIONS,
                        tol=1e-4, random_state=seed).fit(z)
    unmixing = _canonical_rows(model.components_)
    sources = (z-model.mean_)@unmixing.T
    centered = sources-sources.mean(0)
    kurtosis = (centered**4).mean(0)/(centered**2).mean(0)**2-3
    order = np.argsort(-np.abs(kurtosis), kind="stable")
    return ({"unmixing": unmixing[order].tolist(), "mean": model.mean_.tolist(),
             "training_excess_kurtosis": kurtosis[order].tolist(), "iterations": int(model.n_iter_),
             "converged": bool(model.n_iter_ < ICA_ITERATIONS)}, sources[:, order])


def _ica_replay(payload, z):
    return (z-vector(payload, "mean", (z.shape[1],)))@vector(payload, "unmixing", (COMPONENTS, z.shape[1])).T


def _ica_check(payload, z, _target):
    """Whitening constraint: training sources are uncorrelated with unit variance."""
    sources = _ica_replay(payload, z)
    if (not np.allclose(vector(payload, "mean", (z.shape[1],)), z.mean(0), atol=1e-9)
            or not np.allclose(np.cov(sources.T, bias=True), np.eye(COMPONENTS), atol=1e-6)):
        raise ValueError("saved unmixing does not whiten its training rows")


def _ica_score(_observed, outputs, _train, payload):
    first, second = outputs["prediction"], outputs["source_2"]

    def kurtosis(v):
        c = v-v.mean()
        return float((c**4).mean()/(c**2).mean()**2-3) if len(v) > 3 and np.ptp(v) > 0 else None
    related = float(abs(np.corrcoef(first, second)[0, 1])) if len(first) > 2 and np.ptp(first) > 0 and np.ptp(second) > 0 else None
    return {"holdout_source_absolute_correlation": related, "holdout_source_1_excess_kurtosis": kurtosis(first),
            "holdout_source_2_excess_kurtosis": kurtosis(second), "independence_tested": False,
            "optimizer_converged_fraction": float(bool(payload.get("converged")))}


def _nmf_codes(basis, z):
    from scipy.optimize import nnls
    return np.array([nnls(basis.T, row)[0] for row in z])


def _nmf_outputs(basis, profile, z):
    codes = _nmf_codes(basis, z)
    return np.column_stack([((z-codes@basis)**2).sum(1), ((z-profile)**2).sum(1), codes])


def _nmf_fit(z, _target, seed):
    from sklearn.decomposition import NMF
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = NMF(n_components=COMPONENTS, init="nndsvda", solver="cd", max_iter=NMF_ITERATIONS, tol=1e-8,
                    random_state=seed).fit(z)
    basis = model.components_
    scale = np.linalg.norm(basis, axis=1)
    if np.any(scale <= 0):
        raise ValueError("nonnegative factorization produced an empty component; no parts are reported")
    basis = basis/scale[:, None]
    basis = basis[np.lexsort(basis.T[::-1])]
    return ({"basis": basis.tolist(), "training_profile": z.mean(0).tolist(),
             "converged": bool(model.n_iter_ < NMF_ITERATIONS), "iterations": int(model.n_iter_)},
            np.column_stack([((z-model.transform(z)@model.components_)**2).sum(1), ((z-z.mean(0))**2).sum(1),
                             np.full((len(z), COMPONENTS), np.nan)]))


def _nmf_parts(payload, width):
    basis, profile = vector(payload, "basis", (COMPONENTS, width)), vector(payload, "training_profile", (width,))
    if np.any(basis < 0) or not np.allclose(np.linalg.norm(basis, axis=1), 1., atol=1e-9):
        raise ValueError("saved parts must be nonnegative unit-norm components")
    return basis, profile


def _nmf_replay(payload, z):
    return _nmf_outputs(*_nmf_parts(payload, z.shape[1]), z)


def _nmf_check(payload, z, _target):
    """Basis stationarity: with optimal nonnegative codes, no basis entry can descend further."""
    basis, profile = _nmf_parts(payload, z.shape[1])
    codes = _nmf_codes(basis, z)
    scale = np.maximum(np.linalg.norm(codes, axis=0), 1e-12)
    gradient = (codes.T@(codes@basis-z))/scale[:, None]
    violation = np.where(basis > 1e-9, np.abs(gradient), np.maximum(-gradient, 0.))
    if not np.allclose(profile, z.mean(0), atol=1e-9) or (payload.get("converged") and violation.max() > 1e-2*np.sqrt(len(z))):
        raise ValueError("saved parts are not a stationary nonnegative factorization of the training rows")


def _ae_forward(payload, z):
    width = z.shape[1]
    hidden = np.tanh(z@vector(payload, "encoder_weights", (width, COMPONENTS))+vector(payload, "encoder_bias", (COMPONENTS,)))
    output = hidden@vector(payload, "decoder_weights", (COMPONENTS, width))+vector(payload, "decoder_bias", (width,))
    return np.column_stack([((z-output)**2).sum(1), (z*z).sum(1), hidden])


def _ae_fit(z, _target, seed):
    import torch
    _needs(z, COMPONENTS+1, "the autoencoder")
    torch.manual_seed(seed)
    data = torch.tensor(z, dtype=torch.float64)
    encoder = torch.nn.Linear(z.shape[1], COMPONENTS, dtype=torch.float64)
    decoder = torch.nn.Linear(COMPONENTS, z.shape[1], dtype=torch.float64)
    optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=AE_RATE, weight_decay=AE_DECAY)
    for _ in range(AE_STEPS):
        optimizer.zero_grad()
        loss = ((decoder(torch.tanh(encoder(data)))-data)**2).sum(1).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        hidden = torch.tanh(encoder(data))
        expected = torch.column_stack([((decoder(hidden)-data)**2).sum(1), (data*data).sum(1), hidden]).numpy()
    return ({"encoder_weights": encoder.weight.detach().numpy().T.tolist(), "encoder_bias": encoder.bias.detach().tolist(),
             "decoder_weights": decoder.weight.detach().numpy().T.tolist(), "decoder_bias": decoder.bias.detach().tolist(),
             "final_training_loss": float(loss.detach())}, expected)


_NOTE = " Labels are never read. Outputs are descriptive diagnostics, not return forecasts; uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved."
_COMMON: dict[str, Any] = dict(supervised=False, return_units=False, affine=())
register(Family(
    key="gmm", number=26, capability="gaussian_mixture_regimes", label="Gaussian mixture regimes · saved dataset",
    scope="Bounded CPU two-component diagonal Gaussian mixture on admitted features: 20–256 training rows, 16 features, 512 EM iterations, seeded, 16 fits. Reports holdout log-density against a single-Gaussian baseline; component labels are statistical clusters, not validated market regimes." + _NOTE,
    kind="FEATURE_LOG_DENSITY", fit=_gmm_fit, replay=_gmm_replay, check=_gmm_check, backend="sklearn.mixture.GaussianMixture",
    outputs=("prediction", "baseline", "component", "component_probability"),
    parameters={"n_components": GMM_COMPONENTS, "covariance_type": "diag", "reg_covar": GMM_FLOOR, "max_iter": GMM_ITERATIONS},
    uncertainty=unavailable("COMPONENT_POSTERIOR_IS_MODEL_PROBABILITY_NOT_CALIBRATED_CONFIDENCE", "log_density"),
    score=_structure_score(True), tolerance=1e-8, **_COMMON))
register(Family(
    key="kmeans", number=27, capability="kmeans_clustering", label="K-means clustering · saved dataset",
    scope="Bounded CPU K-means with three clusters on admitted features: 4 seeded starts, Lloyd iterations to exact convergence, 20–256 training rows, 16 fits. Reports holdout distance to the nearest center against the one-cluster baseline. Spectral clustering is unavailable." + _NOTE,
    kind="NEAREST_CENTER_SQUARED_DISTANCE", fit=_kmeans_fit, replay=_kmeans_replay, check=_kmeans_check,
    backend="sklearn.cluster.KMeans", outputs=("prediction", "baseline", "cluster"),
    parameters={"n_clusters": CLUSTERS, "n_init": KMEANS_STARTS, "max_iter": KMEANS_ITERATIONS, "algorithm": "lloyd"},
    uncertainty=unavailable("HARD_CLUSTER_ASSIGNMENT_HAS_NO_PROBABILITY", "standardized_squared_distance"),
    score=_structure_score(False), tolerance=1e-9, **_COMMON))
register(Family(
    key="pca", number=28, capability="principal_components", label="Principal components · saved dataset",
    scope="Bounded CPU two-component PCA on admitted features (3–16 varying features): training-only directions with a fixed sign convention, 20–256 training rows, 16 fits. Reports holdout reconstruction error against the training-mean baseline." + _NOTE,
    kind="RECONSTRUCTION_SQUARED_ERROR", fit=_pca_fit, replay=_pca_replay, check=_pca_check,
    backend="sklearn.decomposition.PCA", outputs=("prediction", "baseline", "score_1", "score_2"),
    parameters={"n_components": COMPONENTS, "svd_solver": "full"},
    uncertainty=unavailable("PROJECTION_HAS_NO_PREDICTIVE_DISTRIBUTION", "standardized_squared_error"),
    score=_structure_score(False), tolerance=1e-9, **_COMMON))
register(Family(
    key="ica", number=29, capability="independent_components", label="Independent components · saved dataset",
    scope="Bounded CPU FastICA with two sources on admitted features: log-cosh contrast, 512 iterations, seeded; a fit that exhausts its iterations is reported as unconverged and its sources are only a whitening rotation. Sources are ordered by training excess kurtosis with a fixed sign convention. Statistical independence is not tested." + _NOTE,
    kind="INDEPENDENT_COMPONENT_SOURCE", fit=_ica_fit, replay=_ica_replay, check=_ica_check,
    backend="sklearn.decomposition.FastICA", outputs=("prediction", "source_2"),
    parameters={"n_components": COMPONENTS, "fun": "logcosh", "max_iter": ICA_ITERATIONS, "whiten": "unit-variance"},
    uncertainty=unavailable("SOURCE_SIGNAL_HAS_NO_PREDICTIVE_DISTRIBUTION", "unit_variance_source"),
    score=_ica_score, tolerance=1e-8, **_COMMON))
register(Family(
    key="nmf", number=30, capability="nonnegative_matrix_factorization", label="Nonnegative factorization · nonnegative features",
    scope="Bounded CPU two-part nonnegative matrix factorization for datasets whose every feature is nonnegative; signed features are refused, never shifted. Training-maximum scaling, NNDSVDa start, 2000 coordinate-descent iterations, 16 fits. Reports holdout reconstruction error against the training-mean profile." + _NOTE,
    kind="RECONSTRUCTION_SQUARED_ERROR", fit=_nmf_fit, replay=_nmf_replay, check=_nmf_check,
    backend="sklearn.decomposition.NMF", outputs=("prediction", "baseline", "part_1", "part_2"), standardize=False,
    parameters={"n_components": COMPONENTS, "init": "nndsvda", "solver": "cd", "max_iter": NMF_ITERATIONS},
    uncertainty=unavailable("PARTS_HAVE_NO_PREDICTIVE_DISTRIBUTION", "max_scaled_squared_error"),
    score=_structure_score(False), tolerance=1e-4, **_COMMON))
register(Family(
    key="autoencoder", number=40, capability="autoencoder", label="Autoencoder anomaly score · saved dataset",
    scope="Bounded CPU autoencoder on admitted features (3–16 varying features): one two-unit tanh bottleneck, 300 full-batch Adam steps, seeded, one thread, 16 fits. Reports holdout reconstruction error against the training-mean baseline; a high error is an unvalidated anomaly score. Variational models are unavailable." + _NOTE,
    kind="RECONSTRUCTION_SQUARED_ERROR", fit=_ae_fit, replay=_ae_forward, backend="torch.nn.Linear + torch.optim.Adam",
    requires=("torch",), outputs=("prediction", "baseline", "code_1", "code_2"),
    parameters={"bottleneck": COMPONENTS, "activation": "tanh", "steps": AE_STEPS, "learning_rate": AE_RATE, "weight_decay": AE_DECAY},
    uncertainty=unavailable("RECONSTRUCTION_ERROR_IS_NOT_A_CALIBRATED_ANOMALY_PROBABILITY", "standardized_squared_error"),
    score=_structure_score(False), tolerance=1e-9, **_COMMON))
