# Family11: bounded uniform nearest-neighbor regression

The `knn` persisted-data adapter actually calls the existing BSD-3-Clause
`sklearn.neighbors.KNeighborsRegressor.fit` on CPU. This is one regression
variant of one family: classifiers, distance weighting, other metrics and
hyperparameter searches remain unimplemented. Eleven families are implemented;
29 additions remain planned. A nonparametric KNN fit retains observations and
labels; it does **not** optimize a coefficient vector or learn neural weights.

`research-model` accepts `model: "knn"` with the existing admitted `dataset_id`,
`min_train`, `test_size`, `embargo` and `seed` fields. The seed is common request
metadata; this fixed deterministic estimator has no stochastic operation.
K=5, uniform weights, brute-force search, squared Euclidean distance and one
CPU thread are fixed. Feature population means/scales use only the fold's
purged training rows. Exact constant features retain their exact level and
scale1. Targets remain in arithmetic-return units and are not standardized.

The complete plan is admitted before the first fit:20..256purged retained
observations per fit,16features,16total fits across instruments/folds and at
most2,000prediction rows per inference call. No observations or folds are
silently dropped. All retained rows have chronological indices0..n-1, assigned
after purging. The low-level fit API requires that chronological order from its
caller; the persisted adapter sorts rows explicitly before folding.

Fit-time sklearn neighbor queries request **all** retained neighbors, not merely
five candidates. GIG orders them by `(binary64 squared Euclidean distance,
chronological retained training index)` and averages the first five labels in
that order. This makes duplicate observations and equidistant ties, including
ties at the fifth neighbor, deterministic. The local
`sklearn.config_context(enable_cython_pairwise_dist=False, working_memory=8)`
selects SciPy distance arithmetic; no global sklearn configuration is changed.
Squared distances are computed directly, avoiding a rounded square-root step
that could change ties. There is no distance tolerance or approximate search.

Safe non-executable JSON records bounded raw training observations and labels,
training-only means/scales, retained indices, fixed settings, tie policy,
preprocessing policy, training-transform hash, code hash, library versions and
research flags. Replay validates shapes, strict finite non-boolean numbers,
fixed parameter types, exact training-derived transforms and identity. It then
uses SciPy `cdist(metric="sqeuclidean")` and the same canonical ordering.
Actual sklearn all-neighbor inference and JSON replay must agree exactly at
fit time; OOS tests also compare genuine sklearn queries and independent scalar
distances. Restart inference performs no fit and loads no pickle. Immutable
registry hashes protect all saved bytes; structural validation does not
independently establish dataset authenticity or economic value.

Outcome revisions require both label end and knowledge timestamp strictly
before the fold test start. Future features, labels and outcome revisions cannot
alter an earlier fitted scaler, retained training sample or prediction. Existing
admission rules continue to preserve source classification, acquisition/revision
knowledge, feature availability and sequence lineage. No external dataset,
weights, new package, paid service, broker connection or order is required.

Each fold reports OOS arithmetic-return MSE, MAE and MSE against the **training
mean** baseline. Negative findings remain visible. Neither neighbor distance nor
label spread is a calibrated prediction interval. Uncertainty is `value: null`,
`UNAVAILABLE_NO_PREDICTIVE_DISTRIBUTION`, `calibrated: false` and
`predictive_interval: false`.

`FITTED_BOUNDED_KNN`, `usable_for_portfolio=false`, `trading_approved=false`
and `production_promoted=false` are explicit. Fresh and saved inference exclude
KNN from portfolio eligibility even if a wrapper's fit-state label is changed
to `CONVERGED`. Existing allocation/ensemble rejection and worker approval/lease
controls remain in place. These models are research artifacts without economic
validation, promotion permission or an approved Director objective. Authored
fixtures are SYNTHETIC, not observed market evidence.

`tests/test_knn.py` covers genuine sklearn fitting/all-neighbor queries and
independent scalar references, duplicates/equidistant/constant inputs, strict
numeric/state/resource rejection, caller-mutation isolation, training-only
scaling, future-data/revision invariance, strict knowledge purge, correct metrics,
no-refit JSON/restart/HTTP inference, capability disclosure and allocation
rejection. Parent-managed terminal and browser checks cover the workbench action.

`fully_funded_certified=false`; `T4_NOT_CERTIFIED`; §139 UNSUPPLIED. Physical
ARM64/Ampere and current container acceptance remain unverified. Strict $0.
