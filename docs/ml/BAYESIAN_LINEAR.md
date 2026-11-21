# Family12: bounded empirical-Bayes linear regression

The persisted `bayesian_ridge` adapter actually fits the existing permissively
licensed CPU `sklearn.linear_model.BayesianRidge`. This implements one Bayesian
linear family variant, not ARD, MCMC, a classifier or a new pretrained model.
ARD and other variants remain unavailable. Twelve families are implemented;
28 additions remain planned, subject to their individual admission contracts.

Training admits20..256rows,1..16features and at most16total instrument/fold
fits, with a fixed512iteration cap and one CPU thread. Entire fold work is
preflighted before fitting. Hyperprior shape/rate parameters are1e-6, tolerance
1e-6, intercept enabled and no user hyperparameter search. Reaching the iteration
cap rejects publication. Source records actual iteration count and library/code
identities. The common request seed is metadata; this fixed estimator is not
stochastic. No rows, features or folds are silently dropped to fit budgets.

Feature population means/scales use only purged training observations; exact
constant columns retain their level and unit scale. Targets are arithmetic
returns, not standardized scores. Outcome end and revision knowledge must both
precede the validation origin strictly. Forecast features cannot use observations
unavailable at their decision time. Future features and label revisions do not
alter earlier fitted states. This is retrospective research, not proof that a
model was available or published at the historical decision time.

The empirical-Bayes fit estimates one weight precision and one noise precision.
Its conditional posterior assumes a Gaussian linear relationship with IID,
homoskedastic Gaussian noise. Precisions are plug-in estimates; their uncertainty
is omitted. Saved non-executable JSON includes raw training data, train-only
transforms, coefficients/intercept, posterior covariance, precisions, fixed
settings, bounded iteration count, provenance and research flags. No pickle,
weight download, new dependency or remote inference is used.

Saved inference validates strict finite non-boolean numbers, resource bounds,
fixed parameters, train-derived transforms, positive symmetric covariance and
posterior algebra. For standardized queries z it computes `z·coef+intercept`
and variance `z·Sigma·z + 1/noise_precision`. Covariance is checked against
`(weight_precision I + noise_precision X_centered' X_centered)^-1`; coefficients
and intercept must agree with that posterior. Immutable registry hashes bind
the actual bytes. Structural algebra checks do not establish source authenticity
or statistical/economic validity. Inference is bounded to2,000query rows and
never refits. State and caller data do not share mutable arrays.

Each fold reports OOS MSE/MAE, a training-mean MSE baseline, Gaussian negative
log predictive density, a training-only mean/variance Gaussian baseline,
nominal95%conditional interval coverage and width. A zero training variance
leaves baseline density UNAVAILABLE/null instead of inventing a variance floor.
Coverage is an OOS diagnostic, not calibration approval. No validation labels
fit a transform, prior, precision or baseline. Negative findings remain saved.

Uncertainty is `MODEL_STD_NOT_OOS_CALIBRATED`, in arithmetic-return units, with
`calibrated=false` and `hyperparameter_uncertainty_included=false`. Gaussian
intervals may be badly misspecified under heavy tails, regime changes or model
selection. They are not trading confidence or validated economic risk bounds.
Fresh and saved results retain `usable_for_portfolio=false`,
`trading_approved=false`, `production_promoted=false`, even if a wrapper's
fit-state label is changed to CONVERGED. Existing allocation, worker approvals,
lease fencing, risk controls and execution boundaries remain intact.

Tests use genuine sklearn fits and OOS predictions, an independent posterior
matrix inverse and SciPy Gaussian scoring references. They cover constants,
collinearity, strict input/state rejection, accepted/rejected resource limits,
failure-injected nonconvergence, caller isolation, exact knowledge cutoffs,
future-vintage invariance, no-refit process/workspace replay, API/terminal output
and allocation rejection. Real Chromium checks exercise the explicit workflow,
both themes and responsive layouts. Authored fixtures remain SYNTHETIC and
prove no market profitability. Current full acceptance evidence is separate.

BSD-3-Clause sklearn and existing NumPy/SciPy CPU libraries only. Strict$0,
no broker/orders, paid services, deployment or automatic model promotion.
FinBERT noncommercial research-only;fully_funded_certified=false;
T4_NOT_CERTIFIED;§139UNSUPPLIED. Physical ARM64/Ampere and container acceptance
remain unverified/blocked.
