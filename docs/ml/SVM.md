# Family10: bounded fixed-RBF epsilon-SVR research

The `svr` persisted-data adapter actually fits `sklearn.svm.SVR` using existing
BSD-3-Clause scikit-learn CPU code. This implements one regression variant of
one family. Classifiers, additional kernels and parameter searches remain
unimplemented; the inventory has ten implemented families and30planned additions.

`research-model` accepts `model: "svr"` with the existing admitted `dataset_id`,
`min_train`, `test_size`, `embargo` and `seed` fields. Seed remains common request
metadata: this deterministic libsvm solver has no stochastic seed. Fixed settings
are RBF gamma=1/feature-count, C=1, epsilon=0.1, tolerance=0.000001, shrinking
false,16MiB kernel cache and10,000maximum iterations. These values operate on
standardized targets. Feature and target population means/scales come only from
the purged training rows of that fold. Exact constant columns use their exact
level and scale1, including constant targets with zero support vectors.

The complete chronological plan is admitted before the first fit:20..128purged
training rows per fit, at most16features,16total fits across instruments/folds,
and2,000prediction rows per inference call. This bounds the quadratic kernel
work. No folds or observations are silently discarded, no hyperparameter search
runs, and a nonconverged fit raises an error before immutable publication. CPU
thread limits apply to fits and numerical replay. No downloads, new libraries,
external data, credentials or broker connections are needed.

Each training label uses the revision whose outcome end and knowledge timestamp
are both strictly before the test start. Future revisions cannot enter earlier
training identities. Future features and targets cannot affect an earlier fitted
scaler or model. The existing dataset retains acquisition/provenance and feature
availability requirements; this adapter does not alter those admission controls.

Persisted non-executable JSON records training-only transforms, training values,
support indices/vectors, dual coefficients, intercept, fixed settings, convergence
and iteration status, versions, code hash and training-transform identity. Replay
validates bounded shapes/numbers, exact training transforms, support ownership,
dual bounds and research-only flags, then computes
`(exp(-gamma * squared_distance) @ dual + intercept) * target_scale + target_mean`.
Actual sklearn predictions are compared with independent JSON RBF replay at fit
time. Inference after restart performs no fit and loads no pickle. Immutable
registry hashes protect all saved bytes; the structural checks do not claim to
prove optimization optimality or dataset authenticity independently.

Each fold reports arithmetic-return OOS MSE, MAE and MSE against its **training
mean** baseline. Negative comparisons remain in the report. Epsilon is a loss
tube, not a confidence interval or return error bound. Uncertainty is explicitly
`value: null`, `UNAVAILABLE_NO_PREDICTIVE_DISTRIBUTION`, `calibrated: false` and
`predictive_interval: false`; no zero uncertainty or coverage claim is invented.

Models use `FITTED_BOUNDED_SVR`, `usable_for_portfolio=false`,
`trading_approved=false` and `production_promoted=false`. Both fresh and saved
inference explicitly exclude SVR from portfolio eligibility even if a wrapper's
fit-state label changes. Existing allocation/ensemble and worker approval/lease
controls remain in place. This is statistical research, not economic validation,
automatic model promotion or an approved Director objective. The authored test
fixture is SYNTHETIC, not observed market evidence.

`tests/test_svm.py` checks actual sklearn and independent scalar RBF references,
training-only scaling, constant inputs/targets, no-fit JSON replay, adversarial
numeric/state/work-bound rejection, convergence failure, future-revision
invariance, strict label-knowledge purge, correct fold metrics, restart inference
and research fences. The existing workbench exposes one explicit saved-dataset
SVR action through the Models/Research/Signals contracts.

`fully_funded_certified=false`; `T4_NOT_CERTIFIED`; §139 UNSUPPLIED. Physical
ARM64/Ampere and current container acceptance remain unverified. Strict $0.
