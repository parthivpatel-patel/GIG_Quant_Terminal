# Saved reference-model baseline comparisons

The existing observed-reference workflow now supports an immutable predictive
evaluation, followed by replay. This is a retrospective diagnostic on saved
predictions, not an executable portfolio backtest or economic validation.

```json
{"operation":"evaluate","report_id":"<saved report hash>","bootstrap_replications":500,"block_length":5,"seed":17}
```

```json
{"operation":"evaluation-replay","evaluation_id":"<saved evaluation hash>"}
```

Both operations use `POST /api/v1/workspace/observed-reference`. The terminal's
workbench fills successful parent identities into the next example. Saved
evaluations appear through the existing observed-reference Research, Models and
Signals views. Evaluation availability uses its actual creation timestamp,
separately from parent acquisition; an earlier cutoff cannot expose a later
evaluation. Replay verifies source, parent report, saved models, publication
index and deterministic calculation without retraining or fetching data.

For return estimates, zero-return and per-model training-mean forecasts provide
baselines. Training means come only from the original frozen training targets.
All comparisons use identical available model/target rows within each date;
dates have equal weight. Missing fits remain excluded and disclosed as PARTIAL,
never imputed or treated as zero loss. A circular moving-block bootstrap samples
whole paired dates, retaining the cross-currency observations together. Block
length, seed, repetitions and unadjusted percentile intervals are disclosed.
Temporal stationarity and completeness of the tested model family are not
established; these are not calibrated confidence or adjusted alpha claims.

The acquired 2025 ECB USD/GBP/JPY window produced these saved diagnostics:

| Saved model | Matched-date model MSE | Zero-return MSE on matching rows | Coverage |
| --- | ---: | ---: | --- |
| Ridge | 0.0000258235 | 0.0000204240 | 240 rows / 80 dates |
| HMM | 0.0000202840 | 0.0000200257 | 220 rows / 80 dates; 20 unavailable |
| Kalman | 0.0000214629 | 0.0000204240 | 240 rows / 80 dates |
| BiLSTM | 0.0381910 | 0.0000204240 | 240 rows / 80 dates |

Each was also worse than its training-mean baseline on matching dates. The
HMM/Kalman zero-baseline intervals include zero; no statistical superiority or
inferiority claim follows from the point estimates alone. No hyperparameters
were selected using this report. GARCH and LightGBM explicitly return an
UNAVAILABLE return-score diagnostic: conditional variance and rank scores cannot
be compared with return forecasts through MSE. Model-appropriate variance/ranking
diagnostics remain subsequent scope.

These findings preserve negative evidence. Date-only reference quotations lack
historical publication vintages, actual tradable quotes, executions and volume.
Capacity, costs, account P&L, historical point-in-time validity and economically
validated alpha remain UNKNOWN/UNAVAILABLE or false as appropriate. Research
output never becomes approved trading intent through this workflow.

Focused scalar tests independently verify .16 model loss versus .01 baselines,
constant paired intervals, perfect-prediction boundaries, missing coverage,
invalid parents/settings/clocks, immutable replay and publication failure.
Real Chromium tests exercise both actions and restart replay. Exact acquired
source diagnostics and saved report identities are in
`docs/evidence/sprint7/observed-baseline-summary.json`.
