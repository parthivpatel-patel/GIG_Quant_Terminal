# Family09: bounded Random Forest and ExtraTrees research

Both actual sklearn regressors implement one family, not two new families.
The inventory now has nine implemented families and31planned additions.

`research-model` accepts `random_forest` or `extra_trees` on a saved admitted
dataset. Each variant fits16trees, depth≤4, minimum3leaf observations, squared
error, all available features, one CPU thread and an explicit seed. Random
Forest uses its actual bootstrap samples; ExtraTrees uses all training rows.
The complete purged plan is bounded before the first fit:16fits across all
instruments/folds,256training rows per fit,16features and2,000prediction rows.
No fold is silently dropped, sampled or replaced when a bound fails.

Tree inputs explicitly use sklearn's **float32 feature boundary**. There is no
fitted scaler; training-only feature minima/maxima are retained as diagnostics.
Targets and outputs use binary64 research arithmetic. This is not a change to
Decimal cash accounting or timestamp precision. Labels and revisions pass the
existing strict knowledge-before-test-start purge. Future revisions are excluded
from earlier model identities and inputs.

Actual fitted children, feature indices, thresholds, node means and training
sample indices are persisted as non-executable JSON. Replay checks bounded
forward-only tree topology, dimensions, source-training hashes and leaf means
against the stored actual training samples before inference. Fit-time replay is
also compared with actual sklearn inference. No pickle, weight download,
substitute estimator or external service is used. Current sklearn1.8.0 is the
existing BSD-3-Clause dependency; versions/code hashes are recorded in models.

OOS return MSE and MAE and a training-only mean baseline are reported by fold.
Spread across correlated trees is **not calibrated confidence**, a predictive
interval, model-selection significance or economic validation. Zero spread can
be a known arithmetic result, not evidence of certainty. Negative fold results
remain in saved reports. The fixture is SYNTHETIC, not observed equity evidence.

Both variants use `FITTED_BOUNDED_FOREST`, `usable_for_portfolio=false`,
`trading_approved=false`, and `production_promoted=false`. Existing allocation
and ensemble controls reject these results. No order preparation or broker
activity is available through this adapter. It is not an automatically approved
Research Director objective.

The existing workbench exposes two explicit saved-dataset actions, carries a
server-verified dataset identity from manual training into saved inference, and
uses the unchanged Models/Research/Signals contracts. Browser tests exercise both
variants, exact saved inference after reload, allocator rejection, provenance,
dark/light automated accessibility and responsive widths. Actual screenshots and
research results are preserved in `docs/evidence/ml-batch2/`.

`fully_funded_certified=false`; `T4_NOT_CERTIFIED`; §139 UNSUPPLIED. No current
container or physical ARM64 acceptance is inferred from these CPU tests.
