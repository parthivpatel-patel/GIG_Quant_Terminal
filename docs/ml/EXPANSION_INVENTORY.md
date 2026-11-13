# Forty-family research expansion inventory

Baseline: verified PR #13 merge `53405c066d5ef63573c0a570ad781d7f66cdbae2`.
Post-merge Python 3.11, Python 3.12 and frontend CI all passed before this inventory
was prepared. This is a research implementation inventory, not a claim of forty
trained, economically validated or approved trading models.

Implementation update: families08(Gaussian process) and09(Random Forest /
ExtraTrees) are implemented in bounded CPU batches. There are nine implemented
families and31planned additions. RF and ExtraTrees are two actually fitted
variants of one family, not two independent statistical hypotheses. See
[Gaussian process](GAUSSIAN_PROCESS.md) and [forests](FORESTS.md).

## Seven existing primary families

| ID | Family | Actual implementation and evidence | Limitation |
| --- | --- | --- | --- |
| 01 | Ridge regression | `research/workflow.py`, `research/model_pipeline.py`; purged training, JSON registry and inference | Linear baseline; uncertainty not OOS calibrated |
| 02 | LightGBM | `ml/ranker.py`, persisted date-grouped LambdaRank adapter | Rank score is not an expected return; optional library |
| 03 | Bidirectional LSTM | `ml/sequence.py`, `service/sequence_workspace.py`; actual CPU fits, bounded historical windows, JSON state | No future-window inference; not production approved |
| 04 | Gaussian HMM | `ml/regimes.py`, persisted state-model adapter | Convergence and partial folds must remain explicit |
| 05 | GARCH(1,1) | `ml/volatility.py`; constrained likelihood fitting and saved variance forecasts | Variance is not alpha; adequate sample required |
| 06 | Kalman / RTS | `ml/state_space.py`; filtering and training-only smoothing | Parameter assumptions are disclosed, not automatically calibrated |
| 07 | FinBERT | `nlp/news.py`, `nlp/model_assets.py`, `service/sentiment_workspace.py` | ProsusAI checkpoint CC-BY-NC-SA-4.0; noncommercial research only; actual assets currently absent |

These are the six persisted numerical/sequence families plus the separate
specialist FinBERT workflow. Existing HistGradientBoosting fallback, factor/risk
PCA, IC-weighted composition, VADER, pairs diagnostics and Ollama briefings are
retained components, not fictitious newly implemented families. PPO/RL and other
registered-not-run labels are not counted as existing fitted engines.

## Thirty-three planned additions

Rows identify proposed engine families. Related boosting implementations and
linear objectives are not independent statistical hypotheses: comparisons on the
same question require one declared multiple-testing family. Classifier/regressor,
kernel, lag and architecture variants are grouped in their row. A name in this
inventory never registers executable code or starts training.

| ID | Family / grouped variants | Candidate CPU implementation | Admission or licensing gate |
| --- | --- | --- | --- |
| 08 | Gaussian process regression | scikit-learn fixed RBF, no optimizer | First batch; BSD-3-Clause verified locally; bounded matrix work, no pretrained weights |
| 09 | Random forest / ExtraTrees | IMPLEMENTED: actual sklearn, JSON tree replay |16trees/depth4,256training rows,16total fits; descriptive disagreement, no allocation approval |
| 10 | Support vector machines / SVR | scikit-learn | Explicit kernel and quadratic resource limits |
| 11 | Nearest-neighbor models | scikit-learn | Training-only scaling; bounded retained observations |
| 12 | Bayesian linear / ARD regression | scikit-learn | Posterior assumptions and OOS interval calibration |
| 13 | Generalized linear models | statsmodels / scikit-learn | Link/likelihood must match label type |
| 14 | Generalized additive models | spline basis plus existing linear kernels | Reuse supported BSD components; fixed training-only knots |
| 15 | Robust regression / Huber / RANSAC | scikit-learn | Disclose robust-loss and outlier policy; no hindsight filtering |
| 16 | Linear quantile regression | scikit-learn / SciPy | Pinball loss, quantile crossing and coverage checks |
| 17 | XGBoost | candidate CPU tree engine | Separate source/license/ARM compatibility verification before installation |
| 18 | CatBoost | candidate CPU ordered boosting | License and categorical availability/target-encoding review |
| 19 | AdaBoost | scikit-learn | Bounded weak learners; training-only weighting |
| 20 | Natural-gradient distributional boosting | candidate NGBoost | Package/license review; proper distributional scores |
| 21 | Explainable boosting machines | candidate interpret engine | Package/license review; training-only interactions |
| 22 | ARIMA / seasonal ARIMA | statsmodels | Causal differencing, stationarity and forecast-origin checks |
| 23 | Exponential smoothing / ETS | statsmodels | Explicit seasonality and training-only initialization |
| 24 | VAR / VECM | statsmodels | Synchronized admitted series, lag and cointegration validation |
| 25 | Dynamic factor forecasting | statsmodels | Train-only loadings; missingness and identification diagnostics |
| 26 | Gaussian mixture regimes | scikit-learn | Unsupervised probabilities are not return forecasts |
| 27 | K-means / spectral regime clustering | scikit-learn | Label stability, causal transforms and OOS assignment |
| 28 | PCA research adapter | Reuse `risk/factor_model.py` components | Existing mathematics; add a persisted research wrapper, do not rebuild PCA |
| 29 | Independent component analysis | scikit-learn | Fit-only transforms, sign/permutation and convergence checks |
| 30 | Nonnegative matrix factorization | scikit-learn | Nonnegative admissible features; no arbitrary shift of signed returns |
| 31 | Copula dependence models | reviewed SciPy-based implementation | Marginals, tails and proper OOS dependence scores |
| 32 | Extreme-value tail models | SciPy / reviewed POT implementation | Sufficient tail observations; reject unsupported VaR/ES |
| 33 | Bayesian change-point models | reviewed CPU implementation | Sequential filtering; historical regimes cannot use future smoothing |
| 34 | Hawkes event intensity | reviewed bounded likelihood implementation | BLOCKED until actual permitted event/print timestamps exist; CFTC counts are not events |
| 35 | Multilayer perceptron | scikit-learn / CPU Torch | Fixed architecture/epochs; training-only normalization |
| 36 | Temporal CNN / TCN | CPU Torch | Causal receptive field, bounded sequence/window work |
| 37 | Reservoir / echo-state networks | reviewed NumPy implementation | Fixed seeded reservoir; stable spectral dynamics |
| 38 | Residual-basis neural forecasting / N-BEATS / N-HiTS | candidate CPU Torch implementation | Architecture variants share evaluation family; no weight download |
| 39 | Temporal attention / TFT / PatchTST | candidate small CPU Torch implementation | Compute and dataset adequacy gate; no production or large-model claim |
| 40 | Autoencoder / variational autoencoder | CPU Torch / reviewed implementation | Reconstruction/likelihood metrics; anomaly score is not approved alpha |

Candidate library names are not license approvals or installed implementations.
Only locally verified core dependency licenses are relied on in the first batch;
new packages, pretrained weights and datasets need their own source/terms review.
Commercial authorization never follows merely from local inference.

## First manageable batch and common acceptance contract

Implement only family 08 initially: fixed-kernel Gaussian process regression,
actual scikit-learn CPU training, bounded dimensions/fits, purged persisted-data
integration, saved non-executable JSON state, model-specific uncertainty and
independent numerical references. No optimizer, downloads, broker connections,
orders or automatic model promotion. Family09 is now implemented in batch2;31later additions remain planned.

Every family must preserve acquisition/revision knowledge, feature availability,
purged label selection, source classification, training-only preprocessing,
code/library/configuration identity, immutable storage, restart replay and
sanitized errors. Unknown uncertainty is null, not zero. Calibration, benchmark,
placebo and economic evidence are separate gates; negative findings are retained.
A fixed compute budget is an admission condition, not a suggestion.

FinBERT remains research-only; fully_funded_certified=false; T4_NOT_CERTIFIED;
historical §139 UNSUPPLIED. Physical ARM64/Ampere and current container acceptance
remain unverified. Strict $0: existing CPU and approved free libraries only.
