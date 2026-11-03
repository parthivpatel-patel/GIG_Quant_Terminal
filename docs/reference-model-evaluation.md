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
were selected using this report. Original v1 GARCH and LightGBM evaluations retain
their UNAVAILABLE return-score diagnostic: conditional variance and rank scores
cannot be compared with return forecasts through MSE. Those historical artifacts
remain unchanged. Sprint 8 adds separately versioned, model-appropriate diagnostics.

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

## Sprint 8 model-appropriate evaluation

New evaluations use `reference-model-appropriate-evaluation-v2`. Replay dispatches
on the stored version; all six original acquired v1 evaluations are checked
against their original identities and values. Existing source snapshots, models,
reports and v1 evaluations are never refitted, rewritten or reclassified.
Model scoring does not submit signals or execution intents.

LightGBM scores each complete held-out currency cross-section using linear-gain
NDCG and average-rank Spearman information coefficient. Gains are the ordinal
target-return ranks used by the saved LambdaRank training convention. Equal
predicted scores receive expected tie ordering rather than an arbitrary ticker
ordering. Constant targets produce UNKNOWN NDCG/IC; constant predictions produce
UNKNOWN IC rather than zero. Integers retain their order above binary64's exact
integer range, and nonnumeric/boolean evidence rejects instead of being coerced.

Two ranking baselines are disclosed: uniformly tied scores, and each currency's
training-only mean return. New model states freeze those means with the original
training evidence. Older states reconstruct them only from the original
hash-verified source snapshot and exact training-row hash. A mismatched training
hash rejects. Missing source/statistics leave the training baseline and universe
coverage UNKNOWN. An incomplete held-out cross-section is excluded as a whole;
it is never silently scored as a smaller claimed universe. Paired comparisons
use loss `1 - NDCG` on the same dates and retain unadjusted bootstrap assumptions.

GARCH scores strictly positive saved conditional variances with
`QLIKE = log(v) + (r - fitted_training_mean)^2 / v` and Gaussian negative log
likelihood `0.5 * (log(2*pi) + QLIKE)`. The squared innovation is a noisy proxy
for latent variance, not observed volatility. Proxy squared-error units are
return-to-the-fourth; they are not return MSE. The baseline is the original
training second moment around that same frozen fitted mean. A degenerate zero
training baseline leaves its comparison UNKNOWN without an epsilon replacement;
valid model scores remain available. Nonpositive/nonfinite model variance rejects.
Scoring means weight available rows equally, whereas paired baseline differences
weight dates equally using identical available rows within each date.

Calibration output is explicitly DIAGNOSTIC_ONLY: standardized innovation mean
and second moment, summed squared-innovation/forecast-variance ratio and assumed
Gaussian 95% interval coverage. Conditional normality, stationarity and calibrated
confidence are not established. HMM diagnostics disclose every saved model's
stop reason, convergence, expected/available/excluded rows and original training
boundary. BiLSTM and Kalman retain the paired OOS return comparisons above.

All six existing acquired ECB reports were evaluated and replayed offline without
refitting, provider requests or any change to the original source index. A
separate standard-library calculation reproduced NDCG, source-derived training
ranking baselines, GARCH QLIKE and all four return-estimator MSE comparisons:

| Diagnostic on the original acquired window | Saved result |
| --- | --- |
| LightGBM NDCG / mean Spearman IC, 80 dates | 0.815723 / 0.0125 |
| Ranking loss difference versus uniformly tied scores | -0.005770; interval [-0.039118, 0.023874] |
| Ranking loss difference versus training-mean rank | -0.022286; interval [-0.055775, 0.014028] |
| GARCH mean QLIKE, 240 rows / 80 dates | -9.848400 |
| GARCH QLIKE difference versus training-only variance | -0.126234; interval [-0.225296, -0.052545] |
| GARCH standardized innovation second moment | 0.870145 |
| Assumed Gaussian 95% interval empirical coverage | 0.966667; calibrated confidence UNKNOWN |
| HMM convergence | 11 converged, 1 nonconverged, 20 excluded rows |

Intervals are unadjusted 95% circular date-block bootstrap percentiles. Both
ranking intervals include zero. The sampled GARCH loss is lower on this limited
window; this does not demonstrate a calibrated volatility model, economic
profitability, tradable liquidity, alpha or production suitability. The original
negative return-estimator findings remain unchanged. Exact evaluation/model
identities and independently checked values are in
`docs/evidence/sprint8/observed-model-appropriate-summary.json` in the release evidence; no model parameter files or restricted FinBERT weights are
included in that proof.
