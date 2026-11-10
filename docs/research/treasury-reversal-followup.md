# Treasury reversal follow-up

This is an exploratory follow-up to the preserved 2023 Treasury 10-year
lagged-sign result: 105 matches among 228 eligible pairs, below one half. That
negative finding motivated testing the opposite sign. The same full archive
was already inspected, so chronological evaluation here is **not independent
confirmation** and cannot remove post-selection bias.

The source is the existing official Treasury par-yield XML with SHA-256
`99f80b80cfeff0a93e339735d72672080d9209073b5fe814c1252e88d1428227`.
Its admitted provenance records acquisition at `2026-11-05T03:01:08Z` and the
Data.gov CC0 license field. Historical publication and revision knowledge
clocks remain UNKNOWN. Observation-date ordering does not establish a tradable
point-in-time dataset. Yields are not transaction prices.

The rule predicts the opposite sign of the previous completed yield change.
Three fixed expanding training windows precede chronological validation. A
purged boundary keeps the last training target strictly before the first
validation feature date. A majority-sign baseline is estimated only from
training targets; ties predict up. Flat changes are excluded from directional
accuracy scoring after observation, never imputed or treated as knowable in
advance.

| Validation dates | Training pairs | Validation pairs | Reversal hits | Baseline hits |
|---|---:|---:|---:|---:|
| 2023-07-03–2023-08-29 | 112 | 39 | 19 | 23 |
| 2023-08-30–2023-10-27 | 151 | 37 | 21 | 19 |
| 2023-10-30–2023-12-29 | 188 | 39 | 21 | 16 |
| Total | — | 115 | 61 | 58 |

Reversal accuracy is 53.04%, versus 50.43% for the training-only baseline.
The two-sided exact binomial p-value against one half is 0.5760, with a nominal
95% interval of 43.51%–62.41%. The paired discordance test against the baseline
has p=0.7877. Both are IID reference calculations; serial dependence can
invalidate nominal significance or coverage.

A bounded, deterministic within-fold circular moving-block bootstrap uses
256 draws, block length five and seed 67. Its 95% interval for the paired
accuracy difference is approximately −8.70 to +15.33 percentage points.
Sixty-four distinct circular target shifts give a diagnostic upper-tail
fraction of 0.3692. Cyclic exchangeability is not established; this is not an
exact market null test. These small diagnostics do not validate economic value.

Holm correction covers the declared local family of parent persistence,
follow-up reversal versus chance, and follow-up versus the training-only
baseline. Adjusted p-values are approximately 0.7805, 1, and 1. This does not
correct for every prior repository experiment or erase the selection history.

An independent Decimal-loop implementation reconstructs predictions, counts,
training baselines and temporal boundaries without consuming the evaluator's
feature arrays. It is computational verification on the same observations,
not independent data or external human review. The saved report includes this
skeptic verdict, causal prediction records, configuration, provenance,
uncertainty and a SHA-256 result hash.

The result remains **INCONCLUSIVE**. Return, Sharpe, drawdown, executable cost,
and allocation eligibility remain unavailable. No broker order, model
promotion, or autonomous code change occurs.

Reproduce after CLI integration:

```sh
python -m gig.research_org --compute treasury-10y-reversal-robustness-2023-v1
```

The direct interface is
`gig.research_org.treasury_followup.run_followup() -> (config_hash, report)`.
`evaluate_series(series)` is pure and supports causal perturbation tests;
`challenge(series, metrics)` independently checks numerical and temporal
integrity. Fixed fixture scope rejects missing, duplicate, malformed,
nonfinite, wrong-tenor and insufficient eligible observations explicitly.
