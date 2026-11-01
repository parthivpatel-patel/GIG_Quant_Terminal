# Optional CPU research model validation

This batch exercises the actual installed LightGBM and statsmodels backends.
It does not substitute a baseline estimator, simulate a third-party fit, fetch
model weights, contact a provider, or produce an approved trading signal.
Every dataset is a deterministic synthetic research fixture.

## Observed environment

| Component | Version exercised |
| --- | --- |
| LightGBM | 4.7.0 |
| statsmodels | 0.15.0 |
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| SciPy | 1.17.0 |

The optional dependencies were supplied through the normal package mechanism,
not copied from another project. Runtime tests limit BLAS/OpenMP threads to one.
The existing purged LightGBM workflow already selects `n_jobs=1`, a fixed seed,
and deterministic training. The ranking tests use the real `lightgbm.train`
function and inspect its objective and query groups while prohibiting a
scikit-learn substitution.

## Verified capabilities

The real regression workflow trains LightGBM on expanding chronological
prefixes, purges labels whose end times reach the next test block, and computes
feature transforms on the admitted training rows. The fixture contains 95
observations and three features, producing two folds. Its three-day label
horizon removes two otherwise eligible labels at each split. Tests check those
boundaries and independently recompute the training-only feature means.

Model states include the actual LightGBM version and serialized booster text.
Their JSON bytes match their SHA-256 identities. Reloaded registered models
reproduce every held-out prediction to an absolute tolerance of `1e-12`.
Repeated training reproduces model/run identities and predictions. Changing a
future held-out label cannot change the first fitted model or its predictions.
The registry and artifacts retain `trading_approved: false` and
`economically_validated: false`.

The real cross-sectional ranker uses the `lambdarank` objective, not regression
under a ranking label. Every query group contains the eight names on one
synthetic observation date. Training rows and the explicit two-observation gap
have no predictions. Only complete held-out blocks receive finite scores.
Changing later features and labels preserves earlier held-out predictions.
This ranker uses a caller-selected fixed embargo; it does not itself accept
label-end timestamps. Callers must choose that gap consistently with the label
horizon. The separate purged workflow verifies explicit label-end knowledge.

For statistical inference, a 400-observation synthetic fixture consists of a
random walk `X`, a stationary AR(1) residual with coefficient `0.4`, the designed
pair `Y = 3 + 1.5 X + residual`, and an independent random walk `Z`.

| Engle–Granger fixture | Statistic | Unadjusted p-value |
| --- | ---: | ---: |
| Designed pair `Y` against `X` | -13.633566902761546 | 1.662546414409752e-24 |
| Independent `Z` against `X` | -2.6380088012679375 | 0.2226616743388961 |

Those values are deterministic numerical regression references for the tested
backend. They are not evidence about any actual instruments. An independent
least-squares calculation of the augmented level-term t-statistic also agrees
with the real ADF result to nine decimal places. The real ADF distinguishes the
stationary residual fixture from this random-walk fixture. Three-pair screening
applies Bonferroni correction across the entire family, and changes after its
training prefix cannot affect the selected pairs or their fitted identities.
No general test-power, post-selection validity, or economic claim follows from
these constructed examples.

## Reproduced backend compatibility repair

statsmodels 0.15 emits a `FutureWarning` when `adfuller` is called without an
explicit result representation. The existing diagnostic boundary treated every
warning as a reason to reject the output, so otherwise valid ADF fits returned
`ERROR / ADF_NUMERICAL_DIAGNOSTIC`.

The adapter now requests `result_object=False` when the installed function
signature supports that compatibility option. Older signatures still receive
the original arguments. This avoids the documented representation-migration
warning without suppressing warning categories or weakening numerical checks.
The legacy-signature compatibility test passes, and genuine exact-collinearity
warnings in Engle–Granger still produce a classified error with null statistics
and p-values. An older statsmodels runtime was not installed or executed; its
signature compatibility was tested using a narrow callback.

## Missing-dependency behavior

Explicit LightGBM selections raise an unavailable-dependency error before any
ridge/scikit-learn substitution. A previously registered LightGBM model also
refuses inference if that dependency is unavailable. Missing statsmodels returns
`UNAVAILABLE` with null statistics and p-values; correlation or OLS significance
is never relabeled as Engle–Granger or ADF. These absence paths are verified with
controlled import/capability failures.

When a runtime is genuinely absent, tests that require its numerical engine
report an explicit unittest skip; the dependency-absence contract tests still
run. In the environment described above, all 16 new tests ran with zero skips.

## Reproduction and limits

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python -m unittest discover -s tests -p test_optional_research_models.py -v
```

The new tests and the existing statistical-arbitrage suite passed together:
30 tests, zero failures, errors, or skips. Ruff, Python compilation, and the
focused source type check also pass. Evidence logs and an XML test report are
retained with the development checkpoint.

This evidence supports implemented and tested local CPU research capabilities.
It does not establish approved live data, valid trading universes, realistic
execution costs, out-of-sample economic performance, broker readiness, trained
BiLSTM/FinBERT capabilities, or institutional certification.
