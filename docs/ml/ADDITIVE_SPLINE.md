# Family14: bounded additive cubic-spline research

This family implements one actual CPU generalized-additive variant, using the
existing BSD-licensed sklearn SplineTransformer and Ridge. Each nonconstant
training input has five uniform knots, degree-three basis and six retained
basis columns. A fixed alpha=1 ridge with unpenalized intercept fits the combined
additive design using SVD. There are no interactions, tuning, knot search,
penalized-likelihood uncertainty or automatic model promotion.

Training-only population standardization, ranges and knots are persisted with
raw paired training observations. Constant training inputs are excluded and
recorded explicitly; new variation in those inputs rejects inference rather
than silently inventing an effect. All-constant inputs reject fitting. Linear
boundary extrapolation is explicit and is not evidence of reliable forecasts
outside the observed training range. Entire purged fold plans, including
constant-feature admissibility, are checked before the first fit.

Work admits20..256training rows,1..16features,96basis columns,16total fits and
2,000query rows, with one CPU thread. There are no fallback estimators or weight
downloads. Safe JSON preserves settings, learned knots, coefficients, intercept,
training transforms, library/code identity, strict execution restrictions and
constant-input policy. Replay verifies finite strict types/shapes, exact
training transforms, uniform knot policy, identity and the regularized ridge
normal equations. SciPy BSpline reconstructs predictions without sklearn refit
or executable serialization. The identity hash is not a digital signature.

Independent recursive Cox-de Boor basis and derivative calculations plus an
independent normal-equation solution agree with actual sklearn fitting and
JSON replay, including boundary extrapolation. Tests cover exact resource
bounds, constants, zero targets, malformed and rehashed state, train-only
transforms, strict label/feature availability, later vintage invariance,
no-refit restart, HTTP and allocation rejection. Saved OOS evaluation uses
return MSE/MAE and training-only mean baselines. Predictive uncertainty is null
and UNAVAILABLE; additive components are not causal financial attribution.

The existing saved-dataset workbench exposes this explicit model through HTTP,
registry, replay and terminal selection. Actual Chromium159 verifies the saved
workflow, unavailable uncertainty, research-only allocation rejection, dark/light
axe checks and responsive widths. Authored fixture observations remain
SYNTHETIC, and statistical/economic validation is a separate unfulfilled gate.
Negative and inconclusive results must be retained. No production promotion or
brokerage activity is enabled.

This is family14, not additional families for each spline or regression variant.
Fourteen families are implemented;26remain planned. No new dependency, provider,
credential, paid service or deployment. FinBERT remains noncommercial research
only;fully_funded_certified=false;T4_NOT_CERTIFIED;historical§139UNSUPPLIED.
PhysicalARM64/Ampere unverified; current-container acceptance capacity blocked.
