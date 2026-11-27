# Bounded research families (shared contract)

`gig.ml.bounded` holds one registry (`FAMILIES`) for inventory families 15, 16,
19, 22-33 and 35-40. A registry entry declares an actual library fit, a
non-executable JSON replay and, where the mathematics allows, a check that
re-derives the saved parameters from the saved training rows. Generic code in
`gig/ml/bounded/core.py` and `gig/research/bounded.py` owns everything else, so
every family has the same boundary:

- CPU, one thread, fixed seed, fixed hyperparameters (no search), 20..256
  training rows (more where a method needs them), at most 16 columns, 16 fits
  and 2,000 replay rows.
- Training-only transforms. Constant training features are excluded, not
  imputed. Signed inputs are never shifted for the nonnegative family.
- Saved state is JSON with a SHA-256 over every replay-relevant field. Replay
  never refits and is compared with the library's own training-row output at
  fit time. Tampered and re-hashed states are rejected by the per-family check.
- Purged walk-forward folds use only label revisions known before the test
  start. The whole work plan is admitted before the first fit.
- Uncertainty is null/uncalibrated, `usable_for_portfolio` is false for every
  fresh and saved estimate, and allocation is rejected.

## Data modes

| Mode | Input | Families |
| --- | --- | --- |
| FEATURES | admitted feature rows per instrument | huber, ransac, quantile, adaboost, mlp, gmm, kmeans, pca, ica, nmf, autoencoder |
| SERIES | admitted one-observation return history per instrument | arima, ets, evt_pot, bocpd, tcn, esn, nbeats, attention |
| PANEL | balanced cross-instrument return matrix | var, dynamic_factor, gaussian_copula |

SERIES and PANEL replays are prefix-causal: output row t uses observations up to
t only and is the estimate for t+1. They require a one-observation target
horizon, a contiguous purged training prefix and, at replay, the unchanged
training history. Unbalanced panels are refused, never filled.

## Scores

Return-unit families report holdout MSE/MAE against the training-mean forecast.
Structure families (26-31, 40) emit descriptive diagnostics, not return
forecasts; return MSE is `UNAVAILABLE_DIFFERENT_OUTPUT_UNITS` and each fold
reports the holdout score against its no-structure baseline (single Gaussian,
one cluster, training mean, independence). The tail family reports holdout
violation rate and quantile loss against the training empirical quantile.
Fold scores are aggregated in `metrics.family`; underperformance is kept.

## Adding a family

Add one `register(Family(...))` entry with `fit`, `replay` and (when possible)
`check`, run `python scripts/sync_bounded_families.py` to regenerate the
workbench table and options, and add an independent reference to
`tests/test_bounded_families.py`. The contract, persisted, HTTP, Node and
Chromium checks pick the new entry up from the registry.

## Not implemented

XGBoost (17), CatBoost (18), NGBoost (20) and EBM (21) need new packages and a
source/license/ARM review. Hawkes intensity (34) stays BLOCKED until permitted
event timestamps exist. Variant limits are stated in each family's scope text.
