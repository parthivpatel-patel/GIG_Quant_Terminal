# Bounded research families (shared contract)

`gig.ml.bounded` holds one registry (`FAMILIES`) for inventory families 15-33
and 35-40. A registry entry declares an actual library fit, a
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
| FEATURES | admitted feature rows per instrument | huber, ransac, quantile, xgboost, catboost, adaboost, ngboost, ebm, mlp, gmm, kmeans, pca, ica, nmf, autoencoder |
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

## Optional boosted engines (17, 18, 20, 21)

These four need `pip install "gig[boosting]"`. Without a library its family is
UNAVAILABLE and refuses to fit; nothing is substituted. Prediction never needs
the library: trees, bins and scores are saved as JSON and replayed in NumPy.

| Family | Distribution (tested) | License | Review notes |
| --- | --- | --- | --- |
| 17 XGBoost | `xgboost-cpu` 3.4.1, 3.2.0 | Apache-2.0 | The `xgboost` distribution depends on `nvidia-nccl` on Linux, so the CPU-only distribution is required. Python 3.11 resolves to 3.2.0. |
| 18 CatBoost | `catboost` 1.2.10 | Apache-2.0 | Numeric features only; no categorical target statistics. Installs plotting packages it does not need here. |
| 20 NGBoost | `ngboost` 0.6.1 | Apache-2.0 | Pure Python on scikit-learn trees. |
| 21 EBM | `interpret-core` 0.7.8 | MIT | License is stated in the project repository; the PyPI metadata field is empty. |

Licenses and platform wheels were read from PyPI metadata on 2026-11-27. Linux
aarch64 wheels are published for the two compiled engines, but no run on
physical ARM64 hardware has been made.

## Not implemented

Hawkes intensity (34) stays BLOCKED until permitted event timestamps exist. Variant limits are stated in each family's scope text.
