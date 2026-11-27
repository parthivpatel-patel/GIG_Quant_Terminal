# Forty-family research expansion inventory

Baseline: verified PR #13 merge `53405c066d5ef63573c0a570ad781d7f66cdbae2`.
Post-merge Python 3.11, Python 3.12 and frontend CI all passed before this inventory
was prepared. This is a research implementation inventory, not a claim of forty
trained, economically validated or approved trading models.

Implementation update: families08(Gaussian process),09(Random Forest /
ExtraTrees),10(fixed-RBF epsilon-SVR),11(uniform K=5 nearest-neighbor regression)
12(empirical-Bayes ridge regression) and13(binomial logit probability research)
and14(additive cubic-spline regression) are implemented in bounded CPU batches.
Families 15-33 and 35-40 (25 families, 26 fitted variants) are implemented
behind one shared bounded contract; see [bounded families](BOUNDED_FAMILIES.md).
There are 39 implemented families; 17, 18 and 21 need the optional `boosting` extra and are
UNAVAILABLE without it. 34 remains BLOCKED on permitted event timestamps. RF and ExtraTrees
are two actually fitted variants of one family, not two independent statistical hypotheses. See
[Gaussian process](GAUSSIAN_PROCESS.md), [forests](FORESTS.md), [SVR](SVM.md),
[nearest neighbors](KNN.md), [Bayesian linear](BAYESIAN_LINEAR.md)
[binomial GLM](BINOMIAL_GLM.md) and [additive splines](ADDITIVE_SPLINE.md).

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
| 10 | Support vector machines / SVR | IMPLEMENTED: actual sklearn fixed-RBF epsilon-SVR, JSON support-vector replay | One regressor;128training rows/16fits, train-only scaling, null uncertainty, no allocation approval |
| 11 | Nearest-neighbor models | IMPLEMENTED: actual sklearn uniform K=5 regression; JSON retained-observation replay |20..256training rows/16fits; train-only scaling, canonical chronological distance ties, null uncertainty, no allocation approval |
| 12 | Bayesian linear / ARD regression | IMPLEMENTED: actual sklearn BayesianRidge; JSON posterior replay; ARD unavailable |20..256rows/16fits/512iterations; conditional Gaussian intervals uncalibrated; proper OOS density/coverage diagnostics; allocation rejected |
| 13 | Generalized linear models | IMPLEMENTED: actual sklearn regularized binomial logit; safe JSON probability replay |20..256rows/16fits/512iterations; directional labels, Brier/log-loss and train-only baselines; uncalibrated; no allocation approval; other likelihoods unavailable |
| 14 | Generalized additive models | IMPLEMENTED: actual sklearn cubic SplineTransformer plus Ridge, safe JSON/SciPy replay |20..256rows/16fits/96basis columns; fixed training-only uniform knots and linear boundary extrapolation; null uncertainty, no interactions or allocation approval |
| 15 | Robust regression / Huber / RANSAC | IMPLEMENTED: actual sklearn HuberRegressor and RANSACRegressor, JSON replay | Two variants of one family; Huber stationarity and RANSAC consensus least squares rechecked; uncertainty null |
| 16 | Linear quantile regression | IMPLEMENTED: actual sklearn QuantileRegressor (HiGHS) at 10/50/90% | Pinball loss vs training quantiles, crossing count and uncalibrated coverage reported; band is not calibrated |
| 17 | XGBoost | IMPLEMENTED: actual `xgboost-cpu` exact trees, JSON tree replay | 32 depth-2 trees; Apache-2.0 CPU-only distribution (the `xgboost` distribution pulls a proprietary NVIDIA runtime on Linux and is not used); aarch64 wheels published, physical ARM64 run unverified; uncertainty null |
| 18 | CatBoost | IMPLEMENTED: actual catboost ordered boosting, JSON oblivious-tree replay | 32 depth-2 trees; Apache-2.0; numeric features only, so no categorical target statistics are computed; aarch64 wheels published, physical ARM64 run unverified; uncertainty null |
| 19 | AdaBoost | IMPLEMENTED: actual sklearn AdaBoost.R2, saved trees and weighted-median replay | 16 depth-2 trees; training-only reweighting; uncertainty null |
| 20 | Natural-gradient distributional boosting | IMPLEMENTED: NGBoost algorithm for a Normal on sklearn trees, JSON stage replay | At most 32 stages; implemented natively because the `ngboost` package pins pandas below 3 through lifelines; agreement with `ngboost` 0.6.1 is tested in an isolated CI step; Gaussian log score, CRPS and 90% coverage against the unconditional training Normal; standard deviation uncalibrated |
| 21 | Explainable boosting machines | IMPLEMENTED: actual interpret-core EBM, JSON bin/score replay | Main effects plus at most two pairs selected on the training fold; MIT (stated in the project repository; the PyPI metadata field is empty); uncertainty null |
| 22 | ARIMA / seasonal ARIMA | IMPLEMENTED: actual statsmodels ARIMA(2,0,0) exact MLE, recursion replay | AR(2) with mean only; differencing, MA and seasonal terms unavailable |
| 23 | Exponential smoothing / ETS | IMPLEMENTED: actual statsmodels ETS(A,N,N), level recursion replay | No trend or seasonality; training-only initial level |
| 24 | VAR / VECM | IMPLEMENTED: actual statsmodels VAR(1) on the balanced return panel | 2..8 instruments on identical decision times; VECM/cointegration and higher lags unavailable |
| 25 | Dynamic factor forecasting | IMPLEMENTED: actual statsmodels one-factor AR(1) DynamicFactor MLE, independent Kalman replay | Filtered (never smoothed) states; optimizer convergence reported per fold |
| 26 | Gaussian mixture regimes | IMPLEMENTED: actual sklearn two-component diagonal GaussianMixture | Holdout log-density vs single Gaussian; components are clusters, not validated regimes or forecasts |
| 27 | K-means / spectral regime clustering | IMPLEMENTED: actual sklearn KMeans (3 clusters) | Holdout nearest-center distance vs one cluster; spectral clustering unavailable |
| 28 | PCA research adapter | IMPLEMENTED: actual sklearn PCA (2 components) persisted research wrapper | Holdout reconstruction error vs training mean; needs 3+ varying features; `risk/factor_model.py` unchanged |
| 29 | Independent component analysis | IMPLEMENTED: actual sklearn FastICA (2 sources) | Fixed sign/order convention; unconverged fits reported; independence not tested |
| 30 | Nonnegative matrix factorization | IMPLEMENTED: actual sklearn NMF (2 parts), SciPy NNLS replay | Refuses any signed feature; never shifts returns |
| 31 | Copula dependence models | IMPLEMENTED: Gaussian copula from training empirical margins (SciPy/NumPy) | Holdout copula log-density vs independence; zero tail dependence; t/vine copulas unavailable |
| 32 | Extreme-value tail models | IMPLEMENTED: peaks-over-threshold generalized Pareto MLE (SciPy), shape bounded to [-0.5, 0.95] | Needs 100+ training rows and 10+ exceedances; unconditional 1% quantile/ES; not a calibrated risk limit |
| 33 | Bayesian change-point models | IMPLEMENTED: Adams-MacKay online filter (NumPy/SciPy), brute-force reference test | Sequential filtering only; posterior change probability uncalibrated |
| 34 | Hawkes event intensity | reviewed bounded likelihood implementation | BLOCKED until actual permitted event/print timestamps exist; CFTC counts are not events |
| 35 | Multilayer perceptron | IMPLEMENTED: actual sklearn MLPRegressor (8 tanh units, L-BFGS), NumPy replay | Fixed architecture; exhausted iteration budget recorded |
| 36 | Temporal CNN / TCN | IMPLEMENTED: CPU Torch causal dilated convolutions (receptive field 8), NumPy replay | 200 fixed Adam steps; one-step return-history forecast only |
| 37 | Reservoir / echo-state networks | IMPLEMENTED: seeded 32-unit NumPy reservoir, closed-form ridge readout | Spectral radius 0.9 enforced at replay; only the readout is trained |
| 38 | Residual-basis neural forecasting / N-BEATS / N-HiTS | IMPLEMENTED: CPU Torch generic N-BEATS (2 blocks), NumPy replay | Trend/seasonality bases and N-HiTS unavailable; no weight download |
| 39 | Temporal attention / TFT / PatchTST | IMPLEMENTED: CPU Torch single-head attention over 8 returns, NumPy replay | One attention layer; TFT/PatchTST and pretrained transformers unavailable |
| 40 | Autoencoder / variational autoencoder | IMPLEMENTED: CPU Torch 2-unit tanh autoencoder, NumPy replay | Reconstruction error vs training mean; anomaly score unvalidated; VAE unavailable |

Candidate library names are not license approvals or installed implementations.
Only locally verified core dependency licenses are relied on in the first batch;
new packages, pretrained weights and datasets need their own source/terms review.
Commercial authorization never follows merely from local inference.

## First manageable batch and common acceptance contract

Implement only family 08 initially: fixed-kernel Gaussian process regression,
actual scikit-learn CPU training, bounded dimensions/fits, purged persisted-data
integration, saved non-executable JSON state, model-specific uncertainty and
independent numerical references. No optimizer, downloads, broker connections,
orders or automatic model promotion. Family09 is implemented in batch2; family10
is implemented in batch3 as one bounded SVR regressor; family11 is implemented in
batch4 as one bounded uniform K=5 nearest-neighbor regressor retaining training
observations and labels, without optimized weights. Family12 is implemented in
batch5 as bounded empirical-Bayes ridge, not ARD. Family13 is implemented in
batch6 as a bounded binomial logit for directional probabilities, not expected
returns. Family14 is implemented in batch7 as fixed cubic additive splines plus
ridge, without interactions or calibrated intervals. Twenty-six later additions
remain planned.

Every family must preserve acquisition/revision knowledge, feature availability,
purged label selection, source classification, training-only preprocessing,
code/library/configuration identity, immutable storage, restart replay and
sanitized errors. Unknown uncertainty is null, not zero. Calibration, benchmark,
placebo and economic evidence are separate gates; negative findings are retained.
A fixed compute budget is an admission condition, not a suggestion.

FinBERT remains research-only; fully_funded_certified=false; T4_NOT_CERTIFIED;
historical §139 UNSUPPLIED. Physical ARM64/Ampere and current container acceptance
remain unverified. Strict $0: existing CPU and approved free libraries only.
