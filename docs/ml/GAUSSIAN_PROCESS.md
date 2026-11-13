# Family 08: bounded Gaussian-process research

This is the first addition after the seven existing primary families. The other
32 additions in [the inventory](EXPANSION_INVENTORY.md) remain planned. This
adapter is not registered as an automatically approved Research Director job.

## Implemented workflow

The existing `research-model` HTTP contract accepts `model=gaussian_process`
with a saved, admitted dataset ID. The workbench has an explicit Gaussian-process
choice that carries forward the selected dataset identity. `research-predict`
reopens immutable model JSON and computes conditional Gaussian predictions;
replay does not fit a model or deserialize executable objects. Model, Research
and Signals views use the existing persisted-data contracts.

Training uses actual `sklearn.gaussian_process.GaussianProcessRegressor`, a fixed
unit-amplitude/unit-length RBF kernel, noise variance `1e-4` in standardized
label units, no optimizer, one CPU thread, and train-only feature/label scaling.
There are no pretrained assets, downloads, paid dependencies or new services.
Scikit-learn is an existing project dependency; the installed 1.8.0 distribution
is BSD-3-Clause. NumPy/SciPy remain their existing BSD-licensed dependencies.

## Admission and limits

Every fold uses the existing strict decision-time label-vintage filter. Labels
and revisions must be known before the fold's test start. Future revision
histories do not enter the fitted model identity. Later observations do not
change earlier training statistics, fitted identities or predictions.

The complete plan is validated before the first fit: at most **16 total fits**
across instruments and folds, **128 training rows per fit**, **16 features**,
**2,000 inference rows**, and the existing dataset/time/instrument bounds. Plans
outside these limits fail explicitly; no subsampling or substitute model is
introduced. Invalid, nonfinite, coercive, inconsistent and corrupted arrays or
saved kernel factors are rejected. Saved factors/weights are checked against
independently rebuilt kernel equations before inference.

Per-fold reports include OOS MSE, training-only constant-mean baseline MSE and
descriptive 95% coverage. Posterior predictive standard deviation includes the
explicit assumed noise and is **not calibrated confidence**. Coverage from a
small retrospective fixture is not a calibrated interval, multiple-testing
correction, economic validation, risk limit or an approved signal.

`fit_state=FITTED_FIXED_KERNEL`, `usable_for_portfolio=false`,
`trading_approved=false`, `production_promoted=false`. Existing allocation and
ensemble gates reject this research output. No OMS intents are created.

## Evidence scope

The first execution uses the existing explicitly SYNTHETIC two-instrument
persisted-bars fixture: 6 fitted states, 72 OOS rows, 1.15 seconds measured for
the complete fit/publish call on this x86_64 CPU environment. The exact timing
and artifact identity are recorded in `docs/evidence/ml-batch1/research-result.json`.
This is functional evidence, not observed-market predictive validation or a
latency guarantee. No new observed market-data source is claimed.

Independent NumPy conditional-Gaussian references, adversarial numeric/state
inputs, future-data and revision tests, saved-state restart, HTTP contracts and
allocation rejection are in `tests/test_gaussian_process.py`. Existing all-family
replay and vintage tests now also execute this family. The actual Chromium
workflow preserves every existing section, tests explicit replay after reload,
checks dark/light axe rules and four responsive widths. Browser evidence is
from the actual application with external browser requests blocked.
