# Family13: bounded binomial logit research

This implements one generalized-linear-model variant: actual existing CPU
`sklearn.linear_model.LogisticRegression` with a Bernoulli likelihood, logit link
and L2 coefficient regularization. Other GLM links/likelihoods are unavailable.
It is not a return regressor, count model, calibrated classifier or trading
decision rule. Family13 adds one family, not separate families for each label.

The fixed label rule is **admitted forward arithmetic return strictly greater
than zero**. Zero/nonpositive returns are class0. Purging uses original outcome
end and revision knowledge strictly before the fold origin before converting
labels. Raw return targets and the label-rule identity remain in output rows;
the admitted dataset is never mutated. This rule is fixed before validation,
not tuned on OOS outcomes. Acquisition, feature availability and revision clocks
remain governed by the existing point-in-time dataset admission contract.

Training admits20..256rows,1..16features,at least3observations per class and at
most16total fits. Entire plans, including class adequacy, are admitted before
the first fit. Train-only population standardization, fixed C=1,lbfgs,tolerance
1e-8,512iterations and one CPU thread bound work. The intercept is unpenalized;
there is no class reweighting, tuning or threshold search. Reaching the iteration
cap rejects publication. An exact initial optimum may legitimately require
zero iterations; the objective-gradient check still applies.

Safe JSON preserves raw labelled training observations, transforms, coefficients,
intercept, class order, fixed settings, label/link/likelihood/penalty identities,
iteration count, code/library provenance and execution restrictions. Replay
validates strict finite non-boolean inputs, exact training transforms, immutable
identity and the regularized likelihood gradient. Inference uses stable sigmoid
algebra, at most2,000query rows and no refit or pickle. It agrees with actual
sklearn probabilities and an independent SciPy optimizer of the declared
penalized likelihood. Caller inputs/state cannot mutate later fits or replay.

OOS evaluation reports **Brier score and Bernoulli log loss**, with train-only
positive-frequency baselines. Stable logit-based loss avoids inventing an
epsilon probability when a sigmoid rounds to0or1. Return-MSE is explicitly
null/UNAVAILABLE_DIFFERENT_OUTPUT_UNITS; probability residuals are named Brier
score. Five predeclared calibration bins retain counts, mean probabilities and
observed rates. Empty bins are null/UNAVAILABLE_NO_OBSERVATIONS. These small
diagnostics do not certify calibration, predictive value or economic usefulness.

The output kind is FORWARD_DIRECTION_PROBABILITY. Parameter uncertainty remains
null/UNAVAILABLE; conditional Bernoulli outcome variance p(1-p) is separately
labelled and is not parameter confidence. Fresh/saved results have
usable_for_portfolio=false,probability_calibrated=false,trading_approved=false
and production_promoted=false, including relabelled wrapper fit states.
No probability threshold dispatches allocations or orders. Existing lease,
approval, risk, account and execution safeguards remain unchanged.

Tests cover actual fitting, independent likelihood solutions, constants and
zero-iteration optima, limits, malformed/state attacks, injected nonconvergence,
train-only probability baselines, unknown empty bins, future-vintage invariance,
strict cutoff purging, original-return label provenance, persisted restart
without fitting, HTTP contracts and allocation rejection. Actual Chromium covers
the explicit saved-dataset workflow and probability/unit distinctions. Authored
fixtures stay SYNTHETIC; negative and inconclusive results remain research data.

Existing BSD sklearn/NumPy/SciPy CPU stack only; no new dependency, weights,
provider, paid service, broker/order, production promotion or deployment.
FinBERT noncommercial research-only;fully_funded_certified=false;
T4_NOT_CERTIFIED;§139UNSUPPLIED. PhysicalARM64/Ampere and current-container
acceptance remain unverified/blocked. Strict$0.
