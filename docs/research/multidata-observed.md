# Observed central-bank reference risk studies

These are two distinct bounded historical datasets and two distinct research
methods. They supplement the existing Treasury studies without replacing their
negative or inconclusive findings. Neither study establishes profitable alpha,
a tradable point-in-time signal, model promotion, executable prices, or funded
allocation eligibility.

## Acquisition and permitted scope

The ECB source was fetched successfully from
`https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip`. Its original CSV
bytes are preserved in a clearly marked base64 JSON wrapper. The historical
archive SHA, exact CSV SHA and wrapper SHA are recorded separately. The selected
2023 EUR-base USD/GBP/JPY panel contains **255 observation dates**. Source and
normalized fixture identities are checked before any experiment runs.

The [ECB disclaimer](https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html)
was fetched with HTTP 200. It permits reuse of information obtained directly
from the ECB subject to accurate reproduction, source attribution, a free-source
notice where content is sold, and identification of modifications. The
[reference-rate information page](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html)
states that these rates are for information and strongly discourages transaction
use. GIG attributes the ECB and identifies its derived calculations as its own.

The Bank of Canada source was fetched successfully from
`https://www.bankofcanada.ca/valet/observations/FXUSDCAD/json?start_date=2023-01-01&end_date=2023-12-31`.
The unchanged original JSON contains **248 daily-average observations**, series
identity `FXUSDCAD`, units `Canadian dollars per US dollar`, and its own link to
Bank terms. The known local artifact-persistence timestamp is used as the
acquisition clock and explicitly described as such. It is not a retroactively
invented publication timestamp. A subsequent request returned HTTP 503; that
failure is preserved and did not replace the successful pinned observations.

The [Bank terms](https://www.bankofcanada.ca/terms/) were fetched with HTTP 200.
They permit freely using, copying, distributing and transmitting website
content, with source attribution, disclosure of changes, accuracy and a
free-source notice for paid redistribution. Exceptions apply to logos,
banknotes and third-party/licensed material. This study uses only the
Bank-published averages and includes attribution and modification notices. It
does not acquire or redistribute underlying LSEG quotes or Bank logos.
The [methodology page](https://www.bankofcanada.ca/rates/exchange/background-information-on-foreign-exchange-rates/)
describes these indicative averages as a public good for statistical, analytical
and informational use, not transactional benchmarks. Their computed averages
must not be confused with observed executed trades.

Permission-page URLs, retrieval evidence, HTML hashes and scoped summaries are
retained in `research_org/fixtures/observed_source_permissions.json`.
Normalized fixtures and original source artifacts are independently hash gated.
No credential, subscription, paid data service or brokerage account was used.

## Chronological methods

Both methods use three fixed expanding training windows. Training targets end
strictly before the first validation feature date, with one purged observation.
No missing date, bank holiday, quote or volume is imputed. Publication and
revision-knowledge vintages remain **UNKNOWN**. Observation-date ordering is
verified, but it does not make the historical data eligible for decision-time
trading reconstruction. These are exploratory, non-preregistered studies, not
independent confirmation of prior research.

ECB risk stability estimates full three-currency covariance using training
reference log changes. The benchmark is a diagonal covariance estimated from
the same training sample and mean. A fixed `1e-12` eigenvalue floor preserves
positive definiteness. Validation uses a multivariate Gaussian covariance
log score (`log determinant + quadratic residual`), not return MSE. Held-out
correlations and Frobenius changes are descriptive stability diagnostics.

| ECB validation period | Training changes | Validation changes |
| --- | ---: | ---: |
| 2023-06-22–2023-08-23 | 118 | 45 |
| 2023-08-24–2023-10-25 | 163 | 45 |
| 2023-10-26–2023-12-29 | 208 | 45 |

The full covariance mean score is **-29.853299**, versus **-29.741033** for the
training-only diagonal benchmark; lower is better. The paired difference is
**-0.112266**, with exploratory 95% block-bootstrap interval
**[-0.244013, 0.019117]**. It crosses zero. The approximate centered-bootstrap
diagnostic p-value is 0.0973. No reliable superiority or economic claim is made.

Bank of Canada variance calibration uses fixed EWMA coefficients `0.94` and
`0.97`, conditional mean zero, and completed daily USD/CAD reference changes.
The baseline is training-only constant squared-change variance. Scores are
variance QLIKE in the form `log(forecast variance) + squared change / forecast
variance`; the omitted target-only additive constant cancels in comparisons
and allows a flat observed change to be scored without inventing a floor on the
observed target. Squared reference changes are a noisy variance proxy.
Alpha `0.94` is the declared primary method; the outcome does not select it.
Both declared candidates are retained and independently checked.

| Bank of Canada validation period | Training changes | Validation changes |
| --- | ---: | ---: |
| 2023-06-23–2023-08-28 | 118 | 45 |
| 2023-08-29–2023-11-02 | 163 | 45 |
| 2023-11-03–2023-12-29 | 208 | 38 |

The primary EWMA mean score is **-10.244119**, versus **-10.356258** for the
training-only constant benchmark. **EWMA underperforms** on this score. Its
paired difference is **+0.112139**, with exploratory 95% block-bootstrap interval
**[-0.038678, 0.348897]**. The second candidate also underperforms
(score -10.305216). Both Holm-adjusted diagnostic p-values are approximately
0.5603. The correction covers only these two declared variance comparisons,
not every archived experiment in the repository.

Each uncertainty calculation uses 256 fixed-seed, within-fold circular
moving-block bootstrap draws, block length five. This is an approximate
short-sample diagnostic; dependence and model misspecification can invalidate
nominal coverage. Raw estimates, intervals, both candidates and negative
findings remain in the saved reports. Both verdicts are **INCONCLUSIVE**.

## Independent skepticism and replay

The Skeptic uses a separate Decimal-log and explicit-loop implementation of
training moments, EWMA recursions and forecasts. It checks prediction identity,
training boundaries, fold metadata, scores, aggregate scores and positive
covariances; covariance scores use linear solves in the independent reference.
Both EWMA candidates receive separate checks. This is computational reference
independence on the same data, not a second dataset, human review or separately
trained AI model.

The source loader validates original response hashes, exact quote identities,
normalization, positive finite values, date ordering and acquisition scope.
`dataset_readiness()` performs those checks without fitting a model or making
a network request. The worker runs pinned artifacts offline only. Reports
preserve configuration, source/fixture identity, measured scores, uncertainty,
Skeptic outputs, unavailable allocation state and a result content hash.

```python
from gig.research_org.observed_experiments import run_ecb_risk, run_boc_volatility
config_id, ecb_report = run_ecb_risk()
config_id, boc_report = run_boc_volatility()
```

The focused tests check numerical references, deterministic same-environment
replay, exact source identity, unknown economic states, future-data
perturbations, tampered predictions/scores, malformed dates, nonfinite values,
corrupt or unavailable source files and preservation of the prior Treasury
105/228 finding. Bitwise floating-point hashes may depend on the numerical
library environment; cross-platform economic or institutional certification is
not inferred from matching local replay hashes.
