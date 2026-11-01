# State models: numerical contracts and research workflow

These modules are independent implementations from the available GitHub baseline.
They do not recover or certify historical Phase 7 work. No `native_fit_cache`
implementation exists in the inspected `src/` baseline. New models do not use an
identity-based fit cache. Admitted model arrays and returned inference arrays are
copied and marked read-only; caller mutation does not alter model parameters.

## Linear Gaussian state space

`gig.ml.state_space.kalman_filter` implements the standard linear Gaussian
observation update using a Cholesky innovation solve and Joseph covariance update.
Initial mean/covariance describe state zero **before observing the first sample**;
there is no transition/process-noise addition before sample zero. Observation
matrices can be constant or vary at each observation, supporting dynamic linear
regressions such as research hedge ratios. Process and initial covariance must be
symmetric positive semidefinite; measurement covariance must be positive definite.
Missing/nonfinite observations are rejected explicitly. Covariance and innovation
overflow is rejected. Numerical output arrays are limited to 96MB in aggregate.

`rts_smooth` computes Rauch–Tung–Striebel full-sequence posterior means and marginal
covariances. Singular predicted covariance is explicitly rejected. Smoothed
results are tagged `OFFLINE_FULL_SEQUENCE_SMOOTHED` and
`uses_future_observations=True`. They are research diagnostics and must not be
treated as historical causal signals. Filtered outputs are prefix-causal given
fixed model parameters; fitting those parameters still requires a training split.

Numerical references assemble an independent joint Gaussian posterior precision
matrix and compare its inverse/solution with the smoother. Scalar hand references
also check filter means, covariance and Gaussian innovation likelihood. Tests
cover multivariate time-varying designs, covariance PSD and causal prefixes.

## Gaussian HMM

`gig.ml.regimes.GaussianHMM` uses finite diagonal Gaussian emissions, log-domain
forward/backward recursions and Viterbi decoding. Zero transition/initial
probabilities are supported. `filtered_probabilities[t]` uses observations through
time t; `smoothed_probabilities`, transition counts and Viterbi paths use the full
supplied sequence. Models allow 1–32 states and up to 128 features; sequences
are bounded at 100,000 observations and the emission workspace at 4 million cells.

`fit_gaussian_hmm` implements deterministic Baum–Welch maximum-likelihood EM,
with quantile initialization, a declared variance floor and canonical ascending
first-feature mean labels. It reports likelihood history, convergence, iteration
count and a stop code. Empty numerical components retain their previous emission
parameters. Material likelihood decreases are rejected; convergence is never
inferred from parameter presence. EM is a local optimization and does not prove
the true number of regimes, statistical significance or economic value.

`state_duration_diagnostics` reports observed runs in observation steps. Runs at
the first/last sequence edges carry left/right censoring flags and are excluded
from complete-run mean durations. A geometric expected duration is
`1/(1-transition[state,state])`. Absorbing states return a null expected duration
and `ABSORBING` state instead of JSON infinity. An absent complete run remains
null rather than zero. Exact reference tests enumerate every possible state path
in a small model and independently compare likelihood, posterior probabilities,
transition counts and Viterbi paths.

## Gaussian GARCH(1,1)

`gig.ml.volatility.GARCH11` implements

    h[t] = omega + alpha * (return[t-1] - mean)^2 + beta * h[t-1]

with `omega>0`, `alpha,beta>=0`, `alpha+beta<1`. Initial variance defaults to the
unconditional variance `omega/(1-alpha-beta)`, with an explicit alternative
accepted for continuation evaluation. Returns are decimal units; variance is
squared return per observation. Cadence must be consistent and supplied by the
research dataset; functions do not infer annualization or a trading calendar.
Forecast one uses the last observed innovation; later forecasts use expected
variance recursion `omega+(alpha+beta)*previous_forecast`.

`fit_garch11` performs Gaussian maximum-likelihood optimization using CPU SciPy
SLSQP, normalized training returns and three deterministic persistence starts.
It estimates the constant mean together with omega/alpha/beta. The optimizer
enforces positive omega and persistence at most 0.999999. Fit outputs explicitly
report optimizer success, iteration count, training likelihood and stop code.
Nonconverged feasible outputs remain `converged=False`. Degenerate variance,
nonfinite observations and numerical overflow are rejected. Hand recursion,
likelihood scale identities and deterministic synthetic MLE references are tested.

## Integrated reproducible research

`gig.research.state_models.run_state_model_research` fits these models on an
explicit chronological prefix and evaluates a nonempty holdout. It accepts
25–10,000 finite observations, 1–8 HMM states, explicit source identity and aware
strictly increasing timestamp strings. It preserves original timestamp strings,
normalizes independent UTC timestamps without discarding nanosecond precision,
hashes exact little-endian binary64 inputs and records the training boundary.

GARCH holdout inference continues the last training conditional variance. HMM
holdout output exposes only causal forward probabilities. Kalman parameters use a
declared training-variance heuristic (process variance 1% of measurement variance),
not an undocumented parameter fit. RTS is used only for training diagnostics.
Model identities depend on training data/configuration/fitted parameters only;
run identities include the full result. Altering future holdout observations
cannot change a fitted model identity or earlier causal holdout outputs.

`ResearchRegistry` stores complete content-addressed JSON artifacts in SQLite
transactions with WAL and full synchronous writes. Replaying identical evidence
is idempotent; changed payloads under old identities and corruption on retrieval
are rejected. This is local durable state, **not an independent external backup**.
Caller-provided source attribution is explicitly not independently verified.

Example (caller must supply permitted data):

```python
from gig.research.state_models import ResearchRegistry, run_state_model_research

artifact = run_state_model_research(
    returns, timestamps,
    training_end=500,
    source_id="your-permitted-dataset:snapshot-sha256",
)
run_id = ResearchRegistry("data/research.sqlite").save(artifact)
```

All artifacts remain `RESEARCH_ONLY`, `NOT_APPROVED` for trading and
`REPRODUCIBLE_RESEARCH_NOT_ECONOMICALLY_VALIDATED`. No LightGBM, BiLSTM or FinBERT
substitute is introduced. Parameter convergence and numerical reference agreement
do not establish out-of-sample profitability or institutional certification.

Verification in the available CPU environment:

```sh
PYTHONPATH=src python -m unittest discover -s tests -p 'test_phase7_models.py' -v
PYTHONPATH=src python -m unittest discover -s tests -p 'test_state_model_research.py' -v
```
