/* GIG local research terminal. Backend truth states and source evidence are retained. */
"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const sections = [
    { id: "command-center", label: "Command Center", icon: "M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z", description: "Portfolio intelligence, research, and operational controls." },
    { id: "markets", label: "Markets", icon: "M3 17l5-5 4 3 8-10M15 5h5v5M3 21h18", description: "Market observations, instrument identity, freshness, and source provenance." },
    { id: "signals", label: "Signals", icon: "M3 12h4l3-8 4 16 3-8h4", description: "Factor research and model scores. Research outputs require trading approval." },
    { id: "portfolio", label: "Portfolio", icon: "M3 7h18v13H3zM8 7V3h8v4M3 12h18M10 12v3h4v-3", description: "Portfolio construction, constraints, allocation, and exposure evidence." },
    { id: "risk", label: "Risk", icon: "M12 3l8 3v6c0 5-8 9-8 9s-8-4-8-9V6zM12 8v5M12 16h.01", description: "Risk estimates, stress scenarios, and valuation assumptions." },
    { id: "models", label: "Models", icon: "M5 5h4v4H5zM15 5h4v4h-4zM10 16h4v4h-4zM7 9v4h10V9M12 13v3", description: "Model capability, versions, diagnostics, and reproducible CPU research." },
    { id: "news", label: "NLP / News", icon: "M4 4h13v16H4zM17 8h3v12h-3M7 7h7M7 11h7M7 15h4", description: "Attributed financial text, entity links, and sentiment evidence." },
    { id: "stat-arb", label: "Statistical Arbitrage", icon: "M3 6c6 0 6 12 12 12h6M3 18c6 0 6-12 12-12h6", description: "Spread research, hedge ratios, mean reversion, and uncertainty." },
    { id: "options", label: "Options", icon: "M3 20h18M4 16c4-13 12-13 16 0M12 3v17", description: "European option estimates, volatility structures, and numerical diagnostics." },
    { id: "execution", label: "Execution", icon: "M4 7h15M15 3l4 4-4 4M20 17H5M9 13l-4 4 4 4", description: "Offline order lifecycle and reconciliation. Broker connectivity is disabled by default." },
    { id: "research", label: "Research", icon: "M9 3h6M10 3v6L4 19c-.5 1 .5 2 2 2h12c1.5 0 2.5-1 2-2L14 9V3M7 15h10", description: "Reproducible experiments, research artifacts, and evidence-backed analysis." },
    { id: "operations", label: "Operations", icon: "M3 12h4l3-7 4 14 3-7h4M4 3h16M4 21h16", description: "Dependency readiness, safe configuration, restart recovery, and operating status." },
    { id: "global-intelligence", label: "Global Intelligence", icon: "M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18M3 12h18M12 3c3 3 3 15 0 18M12 3c-3 3-3 15 0 18", description: "Hazards, geopolitical and delayed defense research with source, clocks, uncertainty and verification. Not a live or tactical feed." },
    { id: "observatory", label: "Research Observatory", icon: "M12 3v3M12 18v3M3 12h3M18 12h3M6 6l2 2M16 16l2 2M18 6l-2 2M8 16l-2 2M12 8a4 4 0 1 0 0 8a4 4 0 0 0 0-8", description: "Scheduler, backlog, and skeptic verdicts. Synthetic jobs stay labelled synthetic. The Treasury study uses the official 2023 par-yield file." },
    { id: "charts", label: "Market Charts", icon: "M5 3v18M12 3v18M19 3v18M3 7h4v9H3zM10 5h4v6h-4zM17 9h4v8h-4z", description: "Candlestick, OHLC, line and area charts with volume and backend-computed indicators, on admitted bars or the labelled synthetic demo." },
    { id: "market-flow", label: "Market Flow Intelligence", icon: "M3 17l4-8 5 6 4-10 5 6M3 21h18", description: "Source-access evidence and authored options scenarios. Observed options prints, sweeps and off-exchange activity are unavailable." },
  ];
  const validStates = new Set(["AVAILABLE", "UNAVAILABLE", "UNKNOWN", "ERROR", "STALE"]);
  const workspaceActions = { "workspace-bars": "bars", "workspace-news": "news", "workspace-research": "research", "workspace-simulate": "simulate", "workspace-ingest": "ingest", "workspace-treasury": "ingest", "workspace-universe": "universe", "workspace-train-sequence": "train-sequence", "workspace-predict-sequence": "predict-sequence", "workspace-finbert": "finbert", "workspace-execution-engine": "execution-engine" };
  Object.assign(workspaceActions, { "workspace-pipeline-bars": "bars", "workspace-pipeline-universe": "universe", "workspace-research-dataset": "research-dataset", "workspace-research-model": "research-model", "workspace-research-gpr": "research-model", "workspace-research-rf": "research-model", "workspace-research-xt": "research-model", "workspace-research-svr": "research-model", "workspace-research-knn": "research-model", "workspace-research-bayes": "research-model", "workspace-research-gam": "research-model", "workspace-research-glm": "research-model", "workspace-research-predict": "research-predict", "workspace-research-allocate": "research-allocate", "workspace-research-prepare": "research-prepare", "workspace-portfolio-accounting": "portfolio-accounting", "workspace-offline-dispatch": "execution-engine", "workspace-offline-fill": "execution-engine", "workspace-schedule-twap": "execution-engine", "workspace-advance-clock": "execution-engine" });
  Object.assign(workspaceActions, { "workspace-pairs-research": "pairs-research", "workspace-pairs-replay": "pairs-replay", "workspace-options-admit": "options-admit", "workspace-options-research": "options-research", "workspace-options-replay": "options-replay", "workspace-research-signals": "research-signals", "workspace-signals-replay": "signals-replay", "workspace-research-screen": "research-screen", "workspace-research-discover": "research-screen", "workspace-research-options": "research-screen", "workspace-research-options-scan": "research-screen", "workspace-research-monitor": "research-screen", "workspace-research-enrich": "research-screen", "workspace-screen-replay": "screen-replay", "workspace-allocate-signals": "research-allocate", "workspace-allocate-screen": "research-allocate", "workspace-allocate-pairs": "research-allocate", "workspace-research-strategy": "research-strategy", "workspace-research-size": "research-strategy", "workspace-research-cycle": "research-strategy", "workspace-research-watch": "research-strategy", "workspace-strategy-replay": "research-strategy", "workspace-size-replay": "research-strategy", "workspace-cycle-replay": "research-strategy", "workspace-watch-replay": "research-strategy" });
  Object.assign(workspaceActions, { "workspace-financial-reference": "financial-intelligence", "workspace-financial-search": "financial-intelligence", "workspace-financial-answer": "financial-intelligence", "workspace-financial-replay": "financial-intelligence", "workspace-financial-verify": "financial-intelligence", "workspace-financial-latent": "financial-intelligence", "workspace-financial-hybrid": "financial-intelligence", "workspace-financial-facts": "financial-intelligence", "workspace-financial-entity-protocol": "financial-intelligence", "workspace-financial-entities": "financial-intelligence" });
  const financialResearchKinds = ["workspace-financial-html", "workspace-financial-statements", "workspace-financial-digest", "workspace-financial-watch", "workspace-financial-watch-check", "workspace-financial-investigate", "workspace-financial-call", "workspace-financial-transcript", "workspace-financial-compare", "workspace-financial-mention-diff", "workspace-financial-record-replay"];
  const calculatorJournalKinds = ["workspace-calculator-save", "workspace-calculator-replay", "workspace-calculator-dataset", "workspace-calculator-covariance"];
  for (const kind of financialResearchKinds) workspaceActions[kind] = "financial-intelligence";
  for (const kind of calculatorJournalKinds) workspaceActions[kind] = "calculator-journal";
  workspaceActions["workspace-options-overlay"] = "options-overlay";
  // BEGIN GENERATED BOUNDED FAMILIES
  const boundedFamilies = [
    ["huber", "Huber robust regression · saved dataset", "Bounded CPU Huber regression (epsilon 1.35, fixed L2 1e-4) on admitted saved data: 20–256 training rows, 16 features, 16 fits. Outlier down-weighting uses training residuals only; no hindsight filtering. Uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.", 40],
    ["ransac", "RANSAC consensus regression · saved dataset", "Bounded CPU RANSAC least squares on admitted saved data: 64 seeded trials, majority consensus set, training-MAD threshold, 20–256 training rows and 16 fits. Excluded training rows are recorded, not deleted from evidence. Uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.", 40],
    ["quantile", "Linear quantile regression · 10/50/90", "Bounded CPU linear quantile regression at fixed 10%, 50% and 90% levels (HiGHS linear programs, no penalty): 20–256 training rows, 16 features, 16 folds. The estimate is a conditional median; the 10–90 band is NOT calibrated and quantiles may cross. Pinball loss, crossing and coverage are reported. Allocation, promotion and trading remain unapproved.", 40],
    ["adaboost", "AdaBoost.R2 regression · saved dataset", "Bounded CPU AdaBoost.R2 on admitted saved data: 16 depth-2 trees, learning rate 0.5, linear loss, seeded training-only reweighting, 20–256 training rows and 16 fits. The weighted-median estimate has no predictive distribution. Uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.", 40],
    ["mlp", "Multilayer perceptron · saved dataset", "Bounded CPU multilayer perceptron on admitted saved data: one 8-unit tanh layer, fixed L2 0.01, at most 256 L-BFGS iterations, seeded initialization, 20–256 training rows and 16 fits. An exhausted iteration budget is recorded, not hidden. Uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.", 40],
    ["xgboost", "XGBoost exact trees · saved dataset", "Bounded CPU XGBoost on admitted saved data: 32 depth-2 trees by the exact greedy method, learning rate 0.1, L2 1, minimum child weight 3, squared error, one thread, 20–256 training rows and 16 fits. Trees are exported to JSON and replayed without the library; the CPU-only package is used, never a GPU build. Uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.", 40],
    ["catboost", "CatBoost ordered boosting · saved dataset", "Bounded CPU CatBoost on admitted saved data: 32 depth-2 oblivious trees with ordered boosting, learning rate 0.1, L2 3, 32 borders, no bootstrap, one thread, 20–256 training rows and 16 fits. Only numeric features are admitted, so no categorical target statistics are computed. Trees are exported to JSON and replayed without the library. Uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.", 40],
    ["ngboost", "NGBoost algorithm · Normal distribution", "Bounded CPU natural-gradient boosting of a Normal distribution on admitted saved data: at most 32 stages of depth-2 trees for location and log scale, learning rate 0.05, full batches, NGBoost line search, 20–256 training rows and 16 fits. The published algorithm is implemented here on scikit-learn trees; the ngboost package is not a dependency. The standard deviation is a model output, NOT a calibrated interval; Gaussian log score, CRPS and 90% coverage are reported against the unconditional training Normal. Allocation, promotion and trading remain unapproved.", 40],
    ["ebm", "Explainable boosting machine · saved dataset", "Bounded CPU explainable boosting machine on admitted saved data: additive per-feature shape functions on at most 32 bins plus at most two pair terms chosen on the training fold only, one bag, at most 200 rounds with a seeded 15% training-fold stopping split, 20–256 training rows and 16 fits. Bins and scores are exported to JSON and replayed without the library; shape functions describe the fit, not causes. Uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.", 40],
    ["gmm", "Gaussian mixture regimes · saved dataset", "Bounded CPU two-component diagonal Gaussian mixture on admitted features: 20–256 training rows, 16 features, 512 EM iterations, seeded, 16 fits. Reports holdout log-density against a single-Gaussian baseline; component labels are statistical clusters, not validated market regimes. Labels are never read. Outputs are descriptive diagnostics, not return forecasts; uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.", 40],
    ["kmeans", "K-means clustering · saved dataset", "Bounded CPU K-means with three clusters on admitted features: 4 seeded starts, Lloyd iterations to exact convergence, 20–256 training rows, 16 fits. Reports holdout distance to the nearest center against the one-cluster baseline. Spectral clustering is the separate spectral family. Labels are never read. Outputs are descriptive diagnostics, not return forecasts; uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.", 40],
    ["pca", "Principal components · saved dataset", "Bounded CPU two-component PCA on admitted features (3–16 varying features): training-only directions with a fixed sign convention, 20–256 training rows, 16 fits. Reports holdout reconstruction error against the training-mean baseline. Labels are never read. Outputs are descriptive diagnostics, not return forecasts; uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.", 40],
    ["ica", "Independent components · saved dataset", "Bounded CPU FastICA with two sources on admitted features: log-cosh contrast, 512 iterations, seeded; a fit that exhausts its iterations is reported as unconverged and its sources are only a whitening rotation. Sources are ordered by training excess kurtosis with a fixed sign convention. Statistical independence is not tested. Labels are never read. Outputs are descriptive diagnostics, not return forecasts; uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.", 40],
    ["nmf", "Nonnegative factorization · nonnegative features", "Bounded CPU two-part nonnegative matrix factorization for datasets whose every feature is nonnegative; signed features are refused, never shifted. Training-maximum scaling, NNDSVDa start, 2000 coordinate-descent iterations, 16 fits. Reports holdout reconstruction error against the training-mean profile. Labels are never read. Outputs are descriptive diagnostics, not return forecasts; uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.", 40],
    ["autoencoder", "Autoencoder anomaly score · saved dataset", "Bounded CPU autoencoder on admitted features (3–16 varying features): one two-unit tanh bottleneck, 300 full-batch Adam steps, seeded, one thread, 16 fits. Reports holdout reconstruction error against the training-mean baseline; a high error is an unvalidated anomaly score. Variational models are unavailable. Labels are never read. Outputs are descriptive diagnostics, not return forecasts; uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.", 40],
    ["ard", "Automatic relevance determination · saved dataset", "Bounded CPU ARD regression on admitted features: evidence iteration with fixed priors, coefficients pruned at precision 10,000, at most 300 iterations, 20–256 training rows, 16 fits. A fit that keeps no coefficient is refused. The posterior mean is not a calibrated predictive interval. Uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.", 40],
    ["sarima", "Seasonal ARIMA · return history", "Bounded CPU SARIMA(1,0,0)(1,0,0,4) with intercept by Gaussian state-space maximum likelihood on each instrument's admitted return history: 48–256 training rows, 200 iterations, 16 fits. Differencing and moving-average terms are unavailable. The seasonal period is fixed at 4, not estimated. Row t forecasts observation t+1 from observations up to t only; requires a one-observation horizon and an unchanged training history at replay. Uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.", 48],
    ["vecm", "VECM one-step forecast · cross-instrument series", "Bounded CPU VECM with one lag difference, one cointegrating relation and a constant inside that relation, on the balanced admitted panel after training-only standardization: 2–8 series, 40–256 rows, 16 fits. The rank is fixed, not tested. The first forecast, before a difference exists, is the no-change value. This is not a price-level cointegration claim unless the admitted series are levels. Row t forecasts observation t+1 from observations up to t only; requires a one-observation horizon and an unchanged training history at replay. Uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.", 40],
    ["spectral", "Spectral clustering · saved dataset", "Bounded CPU spectral clustering on admitted features: fixed RBF affinity, three clusters, discretized normalized-Laplacian labels, 20–256 training rows, 16 fits. Saved centers are the training-cluster means. A new row receives the nearest saved center and does not rebuild the graph. Cluster labels are not market regimes. Labels are never read. Outputs are descriptive diagnostics, not return forecasts; uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.", 40],
    ["arima", "AR(2) ARIMA forecast · return history", "Bounded CPU ARIMA(2,0,0) with mean by exact Gaussian maximum likelihood on each instrument's admitted return history: 30–256 training rows, 16 fits. Differencing, MA and seasonal terms are unavailable. Row t forecasts observation t+1 from observations up to t only; requires a one-observation horizon and an unchanged training history at replay. Uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.", 40],
    ["ets", "Exponential smoothing ETS(A,N,N) · return history", "Bounded CPU additive-error simple exponential smoothing (no trend, no seasonality) fitted by maximum likelihood on each instrument's admitted return history: 30–256 training rows, 16 fits. Row t forecasts observation t+1 from observations up to t only; requires a one-observation horizon and an unchanged training history at replay. Uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.", 40],
    ["var", "VAR(1) forecast · cross-instrument returns", "Bounded CPU VAR(1) with intercept by least squares on the balanced cross-instrument return panel: 2–8 instruments on identical decision times, 30–256 training dates, 16 fits. One cointegrating relation is the separate VECM family. Higher lags are unavailable. Row t forecasts observation t+1 from observations up to t only; requires a one-observation horizon and an unchanged training history at replay. Uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.", 40],
    ["dynamic_factor", "Dynamic factor forecast · cross-instrument returns", "Bounded CPU one-factor AR(1) dynamic factor model by state-space maximum likelihood on the balanced cross-instrument return panel: 2–8 instruments, 30–256 training dates, 200 optimizer iterations, 16 fits. Forecasts use filtered (never smoothed) factor states; optimizer non-convergence is reported. Row t forecasts observation t+1 from observations up to t only; requires a one-observation horizon and an unchanged training history at replay. Uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.", 40],
    ["tcn", "Temporal convolution network · return history", "Bounded CPU temporal convolution network on each instrument's admitted return history: three causal dilated layers (kernel 2, dilations 1/2/4, 4 channels, 8-row receptive field), 200 full-batch Adam steps, seeded, one thread, 40–256 training rows, 16 fits. Row t forecasts observation t+1 from observations up to t only; requires a one-observation horizon and an unchanged training history at replay. Uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.", 40],
    ["nbeats", "N-BEATS generic blocks · return history", "Bounded CPU N-BEATS with two generic doubly-residual blocks (8-row lookback, two 8-unit ReLU layers each, one-step forecast): 200 full-batch Adam steps, seeded, one thread, 40–256 training rows, 16 fits. Trend/seasonality basis stacks are unavailable. Row t forecasts observation t+1 from observations up to t only; requires a one-observation horizon and an unchanged training history at replay. Uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.", 40],
    ["attention", "Single-head attention · return history", "Bounded CPU single-head self-attention over the last eight returns (4-dimensional embeddings, learned positions, newest observation as query): 200 full-batch Adam steps, seeded, one thread, 40–256 training rows, 16 fits. This is one attention layer, not a pretrained or multi-layer transformer. Row t forecasts observation t+1 from observations up to t only; requires a one-observation horizon and an unchanged training history at replay. Uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.", 40],
    ["esn", "Echo state network · return history", "Bounded CPU echo state network on each instrument's admitted return history: fixed seeded 32-unit tanh reservoir at spectral radius 0.9, 10-row washout, ridge readout (0.01) solved in closed form, 40–256 training rows, 16 fits. Only the readout is trained. Row t forecasts observation t+1 from observations up to t only; requires a one-observation horizon and an unchanged training history at replay. Uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.", 40],
    ["gaussian_copula", "Gaussian copula dependence · cross-instrument returns", "Bounded CPU Gaussian copula on the balanced cross-instrument return panel: training empirical margins, normal-score correlation, 2–8 instruments, 30–256 training dates, 16 fits. Reports holdout copula log-density against independence. A Gaussian copula has zero tail dependence; Student-t and vine copulas are unavailable. Requires a one-observation horizon and an unchanged training history at replay. Allocation, promotion and trading remain unapproved.", 40],
    ["evt_pot", "Extreme-value tail (POT) · return history", "Bounded CPU peaks-over-threshold on each instrument's admitted return history: generalized Pareto maximum likelihood (shape bounded to [-0.5, 0.95]) on losses above the training 90% threshold, at least 100 training rows and 10 exceedances, 16 fits. Emits an unconditional 1% return quantile and expected shortfall; holdout violations and quantile loss are compared with the training empirical quantile. Not a calibrated risk limit. Requires a one-observation horizon and an unchanged training history at replay. Allocation, promotion and trading remain unapproved.", 100],
    ["bocpd", "Bayesian online change point · return history", "Bounded CPU Bayesian online change-point detection (Adams–MacKay) on each instrument's admitted return history: Normal-Inverse-Gamma prior on training-standardized returns, constant hazard 1/50, run length truncated at 256, 20–256 training rows, 16 fits. Emits the filtered next-return mean, the posterior probability of a change within five observations and the expected run length; posterior probabilities are not calibrated. Requires a one-observation horizon and an unchanged training history at replay. Allocation, promotion and trading remain unapproved.", 40],
  ];
  // END GENERATED BOUNDED FAMILIES
  // BEGIN GENERATED QUANT ANALYTICS
  const quantAnalytics = [
    ["ledoit-wolf", "Covariance shrinkage (Ledoit-Wolf)", "Shrink a sample covariance toward a scaled identity. 2-2,000 rows of 2-64 assets. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"returns": [[0.012, 0.008, -0.004, 0.001], [-0.02, -0.031, 0.006, 0.002], [0.007, 0.011, 0.002, -0.001], [0.015, 0.02, -0.003, 0.0], [-0.009, -0.012, 0.008, 0.003], [0.003, 0.001, 0.001, 0.002], [-0.026, -0.04, 0.011, 0.001], [0.018, 0.022, -0.006, -0.002], [0.004, 0.009, 0.0, 0.001], [-0.011, -0.006, 0.004, 0.002], [0.021, 0.027, -0.008, 0.0], [-0.005, -0.013, 0.003, 0.001], [0.009, 0.004, 0.002, -0.001], [-0.016, -0.022, 0.007, 0.003], [0.011, 0.017, -0.002, 0.0], [0.002, -0.003, 0.001, 0.001], [-0.03, -0.045, 0.012, 0.002], [0.014, 0.019, -0.005, -0.001], [0.006, 0.01, -0.001, 0.001], [-0.008, -0.004, 0.005, 0.002]]}],
    ["hierarchical-risk-parity", "Hierarchical risk parity", "Cluster assets by correlation and split risk down the tree without inverting the covariance. 2-64 assets, positive definite covariance. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"covariance": [[0.04, 0.018, 0.006, 0.002], [0.018, 0.0625, 0.01, 0.004], [0.006, 0.01, 0.0225, 0.003], [0.002, 0.004, 0.003, 0.01]]}],
    ["minimum-cvar", "Minimum-CVaR allocation", "Long-only fully invested weights minimizing expected tail loss over your return scenarios. In-sample scenarios only; no out-of-sample claim. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"scenarios": [[0.012, 0.008, -0.004, 0.001], [-0.02, -0.031, 0.006, 0.002], [0.007, 0.011, 0.002, -0.001], [0.015, 0.02, -0.003, 0.0], [-0.009, -0.012, 0.008, 0.003], [0.003, 0.001, 0.001, 0.002], [-0.026, -0.04, 0.011, 0.001], [0.018, 0.022, -0.006, -0.002], [0.004, 0.009, 0.0, 0.001], [-0.011, -0.006, 0.004, 0.002], [0.021, 0.027, -0.008, 0.0], [-0.005, -0.013, 0.003, 0.001], [0.009, 0.004, 0.002, -0.001], [-0.016, -0.022, 0.007, 0.003], [0.011, 0.017, -0.002, 0.0], [0.002, -0.003, 0.001, 0.001], [-0.03, -0.045, 0.012, 0.002], [0.014, 0.019, -0.005, -0.001], [0.006, 0.01, -0.001, 0.001], [-0.008, -0.004, 0.005, 0.002]], "confidence": 0.9, "max_weight": 0.6}],
    ["efficient-frontier", "Efficient frontier (long-only)", "Minimum-variance portfolios across attainable returns under a weight cap. Expected returns are your assumptions, not forecast evidence. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"expected_returns": [0.08, 0.1, 0.05, 0.03], "covariance": [[0.04, 0.018, 0.006, 0.002], [0.018, 0.0625, 0.01, 0.004], [0.006, 0.01, 0.0225, 0.003], [0.002, 0.004, 0.003, 0.01]], "points": 6, "max_weight": 0.7}],
    ["parametric-var", "Parametric VaR and components", "Normal value at risk and expected shortfall with per-asset components that sum to the total. Losses are positive. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"weights": [400000, 300000, 200000, 100000], "covariance": [[0.00015873015873015873, 7.142857142857142e-05, 2.380952380952381e-05, 7.936507936507936e-06], [7.142857142857142e-05, 0.000248015873015873, 3.968253968253968e-05, 1.5873015873015872e-05], [2.380952380952381e-05, 3.968253968253968e-05, 8.928571428571429e-05, 1.1904761904761905e-05], [7.936507936507936e-06, 1.5873015873015872e-05, 1.1904761904761905e-05, 3.968253968253968e-05]], "confidence": 0.99, "horizon": 10}],
    ["cornish-fisher-var", "Cornish-Fisher VaR", "Adjust the normal quantile for skewness and kurtosis of one return series (30-5,000 values). The result states whether the expansion is still a valid quantile function. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"returns": [0.012, -0.02, 0.007, 0.015, -0.009, 0.003, -0.026, 0.018, 0.004, -0.011, 0.021, -0.005, 0.009, -0.016, 0.011, 0.002, -0.03, 0.014, 0.006, -0.008, 0.005, -0.013, 0.008, -0.021, 0.01, 0.003, -0.007, 0.016, -0.002, 0.004, -0.035, 0.012], "confidence": 0.95}],
    ["historical-var-components", "Scenario VaR and tail contributions", "Scenario value at risk and expected shortfall with each asset's exact share of the tail loss. 20-5,000 scenarios. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"asset_returns": [[0.012, 0.008, -0.004, 0.001], [-0.02, -0.031, 0.006, 0.002], [0.007, 0.011, 0.002, -0.001], [0.015, 0.02, -0.003, 0.0], [-0.009, -0.012, 0.008, 0.003], [0.003, 0.001, 0.001, 0.002], [-0.026, -0.04, 0.011, 0.001], [0.018, 0.022, -0.006, -0.002], [0.004, 0.009, 0.0, 0.001], [-0.011, -0.006, 0.004, 0.002], [0.021, 0.027, -0.008, 0.0], [-0.005, -0.013, 0.003, 0.001], [0.009, 0.004, 0.002, -0.001], [-0.016, -0.022, 0.007, 0.003], [0.011, 0.017, -0.002, 0.0], [0.002, -0.003, 0.001, 0.001], [-0.03, -0.045, 0.012, 0.002], [0.014, 0.019, -0.005, -0.001], [0.006, 0.01, -0.001, 0.001], [-0.008, -0.004, 0.005, 0.002]], "weights": [0.4, 0.3, 0.2, 0.1], "confidence": 0.9}],
    ["monte-carlo-var", "Monte Carlo VaR", "Seeded Gaussian or covariance-matched Student-t simulation. Sampling error is not estimated; rerun with other seeds. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"weights": [0.4, 0.3, 0.2, 0.1], "covariance": [[0.04, 0.018, 0.006, 0.002], [0.018, 0.0625, 0.01, 0.004], [0.006, 0.01, 0.0225, 0.003], [0.002, 0.004, 0.003, 0.01]], "confidence": 0.99, "paths": 20000, "seed": 7, "degrees_of_freedom": 5}],
    ["var-backtest", "VaR exception backtest", "Kupiec coverage, Christoffersen independence and a Basel-style zone for forecasts made before each return. Asymptotic tests with low power in short samples. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"returns": [0.012, -0.02, 0.007, 0.015, -0.009, 0.003, -0.026, 0.018, 0.004, -0.011, 0.021, -0.005, 0.009, -0.016, 0.011, 0.002, -0.03, 0.014, 0.006, -0.008, 0.005, -0.013, 0.008, -0.021, 0.01, 0.003, -0.007, 0.016, -0.002, 0.004, -0.035, 0.012], "var_forecasts": [0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025], "confidence": 0.95}],
    ["almgren-chriss", "Optimal execution schedule (Almgren-Chriss)", "Mean-variance optimal liquidation under linear impact and an arithmetic price. Impact and volatility are your assumptions; fills are assumed certain. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"shares": 1000000, "horizon": 5, "intervals": 5, "volatility": 0.95, "temporary_impact": 2.5e-06, "permanent_impact": 2.5e-07, "fixed_cost": 0.0625, "risk_aversion": 2e-06, "frontier": [0, 1e-06, 1e-05]}],
    ["participation-schedule", "Volume participation schedule", "Whole-share slices proportional to your volume forecast, capped at a participation rate. Shares the cap cannot place are reported unfilled. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"shares": 50000, "volumes": [120000, 80000, 60000, 70000, 150000], "max_participation": 0.1}],
    ["shortfall-simulation", "Implementation shortfall simulation", "Seeded simulation of any schedule under the Almgren-Chriss price model, beside its analytic moments. Simulated, not observed execution. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"trades": [300000, 250000, 200000, 150000, 100000], "interval": 1, "volatility": 0.95, "temporary_impact": 2.5e-06, "permanent_impact": 2.5e-07, "fixed_cost": 0.0625, "paths": 10000, "seed": 7}],
    ["american-option", "American option (binomial tree)", "American and European values on one Cox-Ross-Rubinstein tree with the exercise boundary. Continuous dividend yield only. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"spot": 100, "strike": 105, "rate": 0.05, "div": 0.02, "vol": 0.25, "ttm": 0.75, "call": false, "steps": 400}],
    ["option-greeks-advanced", "Second-order option Greeks", "Closed-form vanna, volga, charm, speed and zomma in raw units (per 1.00 volatility, per year). Research estimate on the inputs you supply; no order, allocation or trading approval.", {"spot": 100, "strike": 110, "rate": 0.03, "div": 0.01, "vol": 0.22, "ttm": 0.5, "call": true}],
    ["svi-smile", "SVI smile fit and arbitrage checks", "Fit one expiry's total implied variance with raw SVI, then check the butterfly condition, wing slopes and implied density. A local fit; not an arbitrage-constrained calibration. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"log_moneyness": [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4], "total_variance": [0.0854, 0.0721, 0.0592, 0.0472, 0.0378, 0.0338, 0.0352, 0.0392, 0.0441]}],
    ["variance-swap", "Variance swap replication", "Fair variance from an out-of-the-money option strip. Edge-strike shares show how much the truncated wings matter. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"strikes": [60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160], "put_prices": [0.0057, 0.094, 0.644, 2.4509, 6.2715, 12.2609, 19.9803, 28.7972, 38.1977, 47.8689, 57.6531], "call_prices": [40.8989, 31.1362, 21.8351, 13.7908, 7.7603, 3.8986, 1.7669, 0.7326, 0.282, 0.1021, 0.0352], "forward": 101.5113, "rate": 0.03, "ttm": 0.5}],
    ["backtest-overfitting", "Probability of backtest overfitting", "How often the best in-sample trial lands in the bottom half out of sample, across all balanced splits. Measures selection risk among the trials you supply, not profitability. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"performance": [[0.012, 0.008, -0.004, 0.001], [-0.02, -0.031, 0.006, 0.002], [0.007, 0.011, 0.002, -0.001], [0.015, 0.02, -0.003, 0.0], [-0.009, -0.012, 0.008, 0.003], [0.003, 0.001, 0.001, 0.002], [-0.026, -0.04, 0.011, 0.001], [0.018, 0.022, -0.006, -0.002], [0.004, 0.009, 0.0, 0.001], [-0.011, -0.006, 0.004, 0.002], [0.021, 0.027, -0.008, 0.0], [-0.005, -0.013, 0.003, 0.001], [0.009, 0.004, 0.002, -0.001], [-0.016, -0.022, 0.007, 0.003], [0.011, 0.017, -0.002, 0.0], [0.002, -0.003, 0.001, 0.001], [-0.03, -0.045, 0.012, 0.002], [0.014, 0.019, -0.005, -0.001], [0.006, 0.01, -0.001, 0.001], [-0.008, -0.004, 0.005, 0.002]], "partitions": 4}],
    ["track-record-length", "Minimum track record length", "Observations needed before a per-period Sharpe ratio exceeds a benchmark at a confidence. Assumes one unselected trial and stationary returns. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"sharpe": 0.1, "benchmark": 0, "skew": -0.5, "kurtosis": 5, "confidence": 0.95}],
    ["fama-macbeth", "Fama-MacBeth regression", "Period-by-period regression of returns on exposures with Newey-West standard errors on the average slopes. Balanced panel; exposures must predate each return. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"returns": [[0.012, 0.008, -0.004, 0.001], [-0.02, -0.031, 0.006, 0.002], [0.007, 0.011, 0.002, -0.001], [0.015, 0.02, -0.003, 0.0], [-0.009, -0.012, 0.008, 0.003], [0.003, 0.001, 0.001, 0.002], [-0.026, -0.04, 0.011, 0.001], [0.018, 0.022, -0.006, -0.002]], "exposures": [[[-0.4], [0.0], [0.4], [-0.2]], [[0.19999999999999996], [-0.4], [0.0], [0.4]], [[-0.2], [0.19999999999999996], [-0.4], [0.0]], [[0.4], [-0.2], [0.19999999999999996], [-0.4]], [[0.0], [0.4], [-0.2], [0.19999999999999996]], [[-0.4], [0.0], [0.4], [-0.2]], [[0.19999999999999996], [-0.4], [0.0], [0.4]], [[-0.2], [0.19999999999999996], [-0.4], [0.0]]], "lags": 1}],
    ["trade-bars", "Bars from trade prints", "Build time, tick, volume or dollar bars from ordered prints. Prints are never split or reordered; the last bar is marked incomplete. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"trades": [{"timestamp_ns": 1000000000, "price": 100.0, "size": 200}, {"timestamp_ns": 2000000000, "price": 100.02, "size": 100}, {"timestamp_ns": 4000000000, "price": 99.98, "size": 300}, {"timestamp_ns": 61000000000, "price": 100.05, "size": 150}, {"timestamp_ns": 63000000000, "price": 100.1, "size": 250}, {"timestamp_ns": 118000000000, "price": 100.08, "size": 100}, {"timestamp_ns": 125000000000, "price": 100.2, "size": 400}, {"timestamp_ns": 190000000000, "price": 100.15, "size": 200}], "kind": "TIME", "threshold": 60000000000}],
    ["bar-quality", "Bar data quality report", "Flag impossible prices, inconsistent highs and lows, duplicate or out-of-order timestamps, gaps, stale closes and return outliers. Calendar-unaware; nothing is repaired. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"bars": [{"timestamp_ns": 86400000000000, "open": 100, "high": 101, "low": 99.5, "close": 100.5, "volume": 1000}, {"timestamp_ns": 172800000000000, "open": 100.5, "high": 102, "low": 100, "close": 101.5, "volume": 1200}, {"timestamp_ns": 259200000000000, "open": 101.5, "high": 101.8, "low": 100.9, "close": 101, "volume": 900}, {"timestamp_ns": 345600000000000, "open": 101, "high": 103, "low": 100.8, "close": 102.6, "volume": 1500}, {"timestamp_ns": 432000000000000, "open": 102.6, "high": 102.9, "low": 99, "close": 99.4, "volume": 0}, {"timestamp_ns": 691200000000000, "open": 99.4, "high": 100.2, "low": 104, "close": 100, "volume": 1100}]}],
    ["heston-option", "Heston option", "European value under Heston (1993) dynamics by one Fourier integral, plus the Black-Scholes volatility that matches it. Parameters are yours; nothing is calibrated to a quote. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"spot": 100, "strike": 100, "rate": 0.03, "div": 0.01, "ttm": 0.5, "call": true, "initial_variance": 0.04, "mean_reversion": 2, "long_run_variance": 0.04, "vol_of_variance": 0.3, "correlation": -0.5}],
    ["barrier-option", "Barrier option", "Single-barrier European value under constant volatility, continuously monitored, or shifted for equally spaced discrete monitoring. No rebate and no smile. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"spot": 100, "strike": 100, "barrier_level": 80, "rate": 0.03, "div": 0.01, "vol": 0.2, "ttm": 0.5, "call": true, "direction": "DOWN", "knock": "OUT", "monitoring_dates": 50}],
    ["option-strategy", "Option strategy at expiry", "Exact expiry profit of calls, puts and the underlying held to one date, including breakevens and whether profit or loss is bounded. No early exercise, financing or fees. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"legs": [{"kind": "CALL", "strike": 100, "quantity": 1, "premium": 5}, {"kind": "CALL", "strike": 110, "quantity": -1, "premium": 2}], "multiplier": 100}],
    ["conditional-stress", "Conditional stress propagation", "Apply stated return shocks and let the other assets follow their conditional normal mean through the covariance you supply. Stress correlations are not modelled. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"weights": [0.4, 0.3, 0.2, 0.1], "covariance": [[0.04, 0.018, 0.006, 0.002], [0.018, 0.0625, 0.01, 0.004], [0.006, 0.01, 0.0225, 0.003], [0.002, 0.004, 0.003, 0.01]], "shocked_assets": [0], "shocks": [-0.2]}],
    ["liquidity-var", "Liquidation-horizon value at risk", "Normal value at risk while a position is sold in equal pieces at a participation cap, plus an exogenous half-spread cost. No price impact and no volume uncertainty. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"position_value": 5000000, "volatility": 0.02, "average_daily_value": 2000000, "max_participation": 0.2, "spread": 0.001, "spread_volatility": 0.0004, "spread_multiplier": 3, "confidence": 0.99}],
    ["cost-aware-rebalance", "Cost-aware rebalance", "Long-only mean-variance rebalance from current weights with proportional trading costs and an optional one-way turnover cap. Expected returns and costs are your assumptions. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"current_weights": [0.4, 0.3, 0.2, 0.1], "expected_returns": [0.08, 0.1, 0.05, 0.03], "covariance": [[0.04, 0.018, 0.006, 0.002], [0.018, 0.0625, 0.01, 0.004], [0.006, 0.01, 0.0225, 0.003], [0.002, 0.004, 0.003, 0.01]], "risk_aversion": 3, "cost": 0.002, "max_weight": 0.6}],
    ["maximum-diversification", "Maximum diversification", "Long-only weights that maximize weighted-average volatility over portfolio volatility. The covariance is your input; there is no return forecast. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"covariance": [[0.04, 0.018, 0.006, 0.002], [0.018, 0.0625, 0.01, 0.004], [0.006, 0.01, 0.0225, 0.003], [0.002, 0.004, 0.003, 0.01]], "max_weight": 0.5}],
    ["tracking-portfolio", "Minimum tracking error", "Long-only portfolio closest to a benchmark in tracking variance, with exclusions, a weight cap and an optional active-return floor. Ex ante only. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"benchmark_weights": [0.4, 0.3, 0.2, 0.1], "covariance": [[0.04, 0.018, 0.006, 0.002], [0.018, 0.0625, 0.01, 0.004], [0.006, 0.01, 0.0225, 0.003], [0.002, 0.004, 0.003, 0.01]], "excluded": [3], "max_weight": 0.5}],
    ["stationary-bootstrap", "Stationary bootstrap", "Percentile interval for the mean and Sharpe ratio by the Politis-Romano stationary bootstrap: geometric blocks wrapped around the series you supply. A resampling, not evidence that a strategy worked. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"series": [0.012, -0.02, 0.007, 0.015, -0.009, 0.003, -0.026, 0.018, 0.004, -0.011, 0.021, -0.005, 0.009, -0.016, 0.011, 0.002, -0.03, 0.014, 0.006, -0.008, 0.005, -0.013, 0.008, -0.021, 0.01, 0.003, -0.007, 0.016, -0.002, 0.004, -0.035, 0.012], "mean_block": 5, "replications": 1000, "seed": 7, "confidence": 0.95}],
    ["roll-spread", "Roll spread", "Implied spread from the autocovariance of price changes (Roll 1984). A nonnegative autocovariance is reported as not identified. Prices are yours; no quote is read. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"prices": [100.1174, 99.8087, 99.8919, 99.9248, 100.1427, 100.1425, 99.8113, 100.1187, 100.0383, 99.7504, 99.7622, 99.841, 99.8568, 100.1823, 100.1077, 100.2203, 100.1245, 100.1796, 100.1631, 100.1191, 99.7863, 99.7527, 100.0717, 100.0662]}],
    ["corwin-schultz", "Corwin-Schultz spread", "High-low spread for every pair of consecutive bars (Corwin and Schultz 2012). Overlapping windows are not independent samples, and a negative window is a noisy estimate, not a negative spread. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"high": [100.4624, 100.5642, 100.5275, 100.405, 100.6083, 100.6549, 100.2758, 100.4447, 100.3054, 100.2029, 100.245, 100.7685, 100.7373, 100.4617, 100.6014, 100.2955, 100.7236, 100.5519, 100.7053, 100.3347, 100.362, 100.3115, 100.7052, 100.3739], "low": [99.4676, 99.1851, 99.2999, 99.6378, 99.6083, 99.3083, 99.6968, 99.5334, 99.1625, 99.1498, 99.5721, 99.7437, 99.3342, 99.4593, 99.6152, 99.5668, 99.7317, 99.4096, 99.4971, 99.3463, 99.6834, 99.3158, 99.4995, 99.4166]}],
    ["risk-parity", "Risk parity", "Long-only fully invested risk budgets on the covariance you supply. A binding weight cap is reported as an approximation when the budgets cannot be met. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"covariance": [[0.04, 0], [0, 0.16]]}],
    ["black-litterman", "Black-Litterman posterior", "Gaussian Black-Litterman posterior from your covariance, long-only market weights and views. Excess returns must share one horizon. No portfolio is optimized from the posterior. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"covariance": [[0.04, 0.01], [0.01, 0.09]], "market_weights": [0.6, 0.4], "views": [[1, -1]], "view_returns": [-0.034999999999999976], "risk_aversion": 2.5, "tau": 0.05}],
    ["fractional-kelly", "Capped fractional Kelly", "Quarter-Kelly by default: fraction times the inverse covariance times your excess returns, then clipped by name and gross caps. Capping changes the unconstrained result. Not an order. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"expected_excess": [0.08, -0.04], "covariance": [[0.04, 0], [0, 0.04]], "fraction": 0.25, "max_name": 1}],
    ["drawdown", "Compounded drawdown", "Drawdown of compounded wealth from arithmetic returns, starting from wealth one. A return below -100 percent is refused. This is path accounting, not a forecast. Research estimate on the inputs you supply; no order, allocation or trading approval.", {"returns": [0.1, -0.2, 0.05]}],
  ];
  // END GENERATED QUANT ANALYTICS
  const boundedKind = (key) => `workspace-research-family-${key}`;
  const isBoundedKind = (kind) => kind.startsWith("workspace-research-family-");
  for (const [key] of boundedFamilies) workspaceActions[boundedKind(key)] = "research-model";
  workspaceActions["workspace-options-overlay-replay"] = "options-overlay-replay";
  Object.assign(workspaceActions, { "workspace-economic-research": "economic-research", "workspace-economic-replay": "economic-replay", "workspace-microstructure-admit": "microstructure-admit", "workspace-microstructure-research": "microstructure-research", "workspace-microstructure-replay": "microstructure-replay", "workspace-funded-risk": "funded-risk", "workspace-funded-risk-replay": "funded-risk-replay" });
  for (const operation of ["initialize", "register", "reserve", "dispatch", "fill", "cancel", "mark", "settle", "state", "replay"]) workspaceActions["workspace-funded-" + operation] = "funded-account";
  Object.assign(workspaceActions, { "workspace-lifecycle-configure": "funded-lifecycle", "workspace-lifecycle-split": "funded-lifecycle", "workspace-lifecycle-dividend": "funded-lifecycle", "workspace-lifecycle-apply": "funded-lifecycle", "workspace-lifecycle-state": "funded-lifecycle", "workspace-lifecycle-replay": "funded-lifecycle", "workspace-statement-admit": "statement-admit", "workspace-statement-reconcile": "statement-reconcile", "workspace-statement-replay": "statement-replay", "workspace-observed-fetch": "observed-reference", "workspace-observed-admit": "observed-reference", "workspace-observed-research": "observed-reference", "workspace-observed-replay": "observed-reference" });
  workspaceActions["workspace-observed-evaluate"] = "observed-reference";
  workspaceActions["workspace-observed-evaluation-replay"] = "observed-reference";
  workspaceActions["workspace-lifecycle-accrue"] = "funded-lifecycle";
  workspaceActions["workspace-lifecycle-dividend-revise"] = "funded-lifecycle";
  workspaceActions["workspace-lifecycle-dividend-cancel"] = "funded-lifecycle";
  workspaceActions["workspace-funded-factor-risk"] = "funded-risk";
  for (const operation of ["fetch", "admit", "replay"]) workspaceActions["workspace-equity-" + operation] = "equity-reference";
  const workflowReferences = { calculator_record_id: null, financial_snapshot_id: null, financial_passage_id: null, financial_verification_id: null, financial_document_id: null, financial_base_document_id: null, financial_record_id: null, financial_watch_id: null, equity_snapshot_id: null, lifecycle_action_hash: null, statement_hash: null, statement_report_id: null, observed_snapshot_id: null, observed_report_id: null, observed_evaluation_id: null, funded_snapshot_id: null, funded_risk_id: null, economic_report_id: null, microstructure_snapshot_id: null, microstructure_report_id: null, checkpoint_id: null, overlay_id: null, dataset_id: null, run_id: null, allocation_id: null, pair_run_id: null, options_snapshot_id: null, options_run_id: null, signal_id: null, screen_id: null, strategy_id: null, size_id: null, cycle_id: null, prepared_client_ids: [] };
  const comparisonResults = [];

  workspaceActions["workspace-depth"] = "depth";
  const state = { section: sections[0], mode: "local", payload: null, rows: [], columns: [], sort: null, request: null, chart: null, points: [], projected: [], angle: -.55, tilt: .40, zoom: 1, dimension: 3, drag: null };
  const motionPreference = typeof window.matchMedia === "function" ? window.matchMedia("(prefers-reduced-motion: reduce)") : { matches: true };
  state.orbit = !motionPreference.matches;
  let orbitFrame = null, orbitLast = null;
  const numberFormat = new Intl.NumberFormat("en-US", { maximumSignificantDigits: 7 });
  const ACRONYMS = { var: "VaR", es: "ES", nav: "NAV", pnl: "P&L", usd: "USD", id: "ID", ml: "ML", etf: "ETF", ols: "OLS", adf: "ADF", sma: "SMA", ema: "EMA", vwap: "VWAP", rsi: "RSI", macd: "MACD", atr: "ATR", api: "API", url: "URL", utc: "UTC", gi: "GI" };
  const human = (value) => String(value).replace(/_/g, " ").replace(/\b\w+/g, (word) => ACRONYMS[word.toLowerCase()] || word.charAt(0).toUpperCase() + word.slice(1));
  const METRIC_LABELS = {
    historical_var_95_currency: ["Historical VaR (95%)", "One-day loss quantile · currency units"],
    historical_es_95_currency: ["Historical ES (95%)", "Mean loss beyond VaR · currency units"],
    synthetic_nav: ["Synthetic NAV", "Notional portfolio value · currency units"],
    holding_period_days: ["Holding period", "Days"],
    smoothed_uses_future_observations: ["Smoother uses future data", "Offline diagnostic only; not a tradable signal"],
    kalman_log_likelihood: ["Kalman log-likelihood", null],
  };
  const isObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
  function scalarTag(value) {
    if (!isObject(value) || Object.keys(value).length !== 1) return null;
    if (typeof value.$integer === "string" && /^-?\d+$/.test(value.$integer)) return value.$integer;
    if (typeof value.$decimal === "string" && /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(value.$decimal)) return value.$decimal;
    return null;
  }
  function exactNumericOrder(left, right) {
    // Compare tagged decimals/integers by sign, decimal order, and significand.
    // Exponents are compared as BigInt; no expanded zero strings or binary64
    // conversion can round an identity or allocate an exponent-sized buffer.
    const parts = (value) => {
      const tagged = scalarTag(value);
      const text = tagged !== null ? tagged : typeof value === "number" && Number.isFinite(value) ? String(value) : null;
      if (text === null) return null;
      const match = /^([+-]?)(\d*)(?:\.(\d*))?(?:[eE]([+-]?\d+))?$/.exec(text);
      if (!match) return null;
      const digits = (match[2] + (match[3] || "")).replace(/^0+/, "");
      if (!digits) return { sign: 0, digits: "", order: 0n };
      const exponent = BigInt(match[4] || "0") - BigInt((match[3] || "").length);
      return { sign: match[1] === "-" ? -1 : 1, digits, order: BigInt(digits.length) + exponent };
    };
    const a = parts(left), b = parts(right);
    if (!a || !b) return null;
    if (a.sign !== b.sign) return a.sign < b.sign ? -1 : 1;
    if (!a.sign) return 0;
    if (a.order !== b.order) return (a.order < b.order ? -1 : 1) * a.sign;
    for (let index = 0; index < Math.max(a.digits.length, b.digits.length); index++) {
      const x = a.digits[index] || "0", y = b.digits[index] || "0";
      if (x !== y) return (x < y ? -1 : 1) * a.sign;
    }
    return 0;
  }
  function validContract(payload) {
    if (!isObject(payload) || payload.schema_version !== "1" || !validStates.has(payload.state) || !isObject(payload.provenance)) return false;
    if (typeof payload.provenance.source !== "string" || typeof payload.provenance.classification !== "string") return false;
    const hasData = ["AVAILABLE", "STALE"].includes(payload.state);
    if (hasData ? !isObject(payload.data) : payload.data !== null) return false;
    if (payload.state === "STALE" && typeof payload.provenance.as_of !== "string") return false;
    if (payload.state === "ERROR" && (!isObject(payload.error) || typeof payload.error.code !== "string" || typeof payload.error.message !== "string")) return false;
    if (payload.state === "AVAILABLE" && payload.error !== null) return false;
    return true;
  }
  function printable(value) {
    if (value === null || value === undefined) return "Unknown";
    if (typeof value === "number") return Number.isFinite(value) ? (Object.is(value, -0) ? "−0" : numberFormat.format(value)) : "Invalid numeric value";
    if (typeof value === "boolean") return value ? "Yes" : "No";
    if (scalarTag(value) !== null) return scalarTag(value);
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  }
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function nav() {
    const fragment = document.createDocumentFragment();
    sections.forEach((section) => {
      const link = el("a", "nav-link");
      link.href = `#${section.id}`;
      link.title = section.label;
      link.setAttribute("aria-label", section.label);
      const icon = el("span", "nav-icon");
      icon.setAttribute("aria-hidden", "true");
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("viewBox", "0 0 24 24");
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", section.icon);
      svg.append(path);
      icon.append(svg);
      link.append(icon, el("span", "", section.label));
      fragment.append(link);
    });
    $("section-nav").append(fragment);
    const mark = $("nav-count");
    if (mark) mark.textContent = `01—${String(sections.length).padStart(2, "0")}`;
  }
  function selectSection() {
    const id = location.hash.slice(1);
    state.section = sections.find((section) => section.id === id) || sections[0];
    document.querySelectorAll(".nav-link").forEach((link) => {
      if (link.hash === `#${state.section.id}`) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    $("section-title").textContent = state.section.label;
    $("breadcrumb").textContent = state.section.label;
    $("section-description").textContent = state.section.description;
    document.title = `${state.section.label} · GIG Quant Terminal`;
    const intelligence = state.section.id === "global-intelligence";
    const charts = state.section.id === "charts";
    const observatory = state.section.id === "observatory";
    const marketFlow = state.section.id === "market-flow";
    $("content").dataset.workspace = intelligence ? "intel" : charts ? "charts" : observatory ? "observatory" : marketFlow ? "market-flow" : "terminal";
    $("content").dataset.section = state.section.id;
    if (window.GIGIntel) window.GIGIntel.show(intelligence);
    if (window.GIGCharts) window.GIGCharts.show(charts);
    if (window.GIGObservatory) window.GIGObservatory.show(observatory);
    if (window.GIGMarketFlow) window.GIGMarketFlow.show(marketFlow);
    if (window.GIGModels) window.GIGModels.show(state.section.id);
    if (intelligence || charts || observatory || marketFlow) {
      if (state.request) state.request.abort();
      return;
    }
    $("workbench").classList.remove("hidden");
    if (state.section.id === "options") $("analytics-kind").value = "options";
    else if (state.section.id === "models") $("analytics-kind").value = "kalman";
    else if (state.section.id === "markets") $("analytics-kind").value = "workspace-bars";
    else if (state.section.id === "news") $("analytics-kind").value = "workspace-news";
    else if (state.section.id === "research") $("analytics-kind").value = "workspace-research";
    else if (state.section.id === "execution") $("analytics-kind").value = "workspace-simulate";
    loadExample();
    state.sort = null;
    $("table-search").value = "";
    loadSection();
  }
  async function loadSection() {
    if (state.section.id === "charts") { if (window.GIGCharts) window.GIGCharts.reload(); return; }
    if (state.section.id === "market-flow") { if (window.GIGMarketFlow) window.GIGMarketFlow.reload(); return; }
    if (state.section.id === "global-intelligence" || state.section.id === "observatory") return;
    $("universe-id").disabled = state.mode !== "local" || $("analysis-engine").value !== "provider";
    $("universe-control").classList.toggle("hidden", state.mode !== "local" || $("analysis-engine").value !== "provider");
    $("depth-control").classList.toggle("hidden", state.mode !== "local" || $("analysis-engine").value !== "depth");
    $("depth-panel").classList.toggle("hidden", !["markets", "execution", "command-center"].includes(state.section.id));
    renderDepth(null);
    if (state.request) state.request.abort();
    const controller = new AbortController();
    state.request = controller;
    $("refresh").disabled = true;
    $("state-badge").textContent = "LOADING";
    $("state-badge").removeAttribute("data-state");
    $("source-label").textContent = "Fetching backend evidence…";
    $("asof-label").textContent = "As of —";
    $("metrics").replaceChildren();
    $("context-content").replaceChildren();
    $("error-panel").classList.add("hidden");
    state.rows = [];
    state.tables = {};
    $("evidence-table-view").replaceChildren();
    $("evidence-table-control").classList.add("hidden");
    state.chart = null;
    renderTable();
    renderChart();
    const timeout = setTimeout(() => controller.abort("timeout"), 20000);
    try {
      let query = `mode=${encodeURIComponent(state.mode)}`;
      if (state.mode === "local") {
        const maximumAge = $("freshness-policy").value;
        if (!["3600", "86400", "604800"].includes(maximumAge)) throw new Error("Unsupported freshness policy.");
        query += `&max_age_seconds=${maximumAge}`;
        const engine = $("analysis-engine").value || "default";
        if (!["default", "sequence", "finbert", "provider", "execution-engine", "persisted", "accounting", "allocation", "depth", "pairs", "options-research", "signals-research", "options-overlay", "funded-account", "funded-risk", "economic-research", "microstructure-research", "funded-lifecycle", "independent-statement", "observed-reference", "observed-equity"].includes(engine)) throw new Error("Unsupported local evidence view.");
        if (engine !== "default") query += `&engine=${encodeURIComponent(engine)}`;
        const universe = $("universe-id").value;
        if (engine === "provider" && universe) {
          if (universe.length > 128) throw new Error("Universe identity exceeds supported length.");
          query += `&universe_id=${encodeURIComponent(universe)}`;
        }
        const depthInstrument = $("depth-instrument").value;
        if (engine === "depth" && depthInstrument) query += `&instrument_id=${encodeURIComponent(depthInstrument)}`;
        // The server admits aware timestamps. Keep the original text so
        // browser Date parsing cannot truncate nanosecond point-in-time data.
        const cutoff = $("replay-asof").value;
        if (cutoff !== "") query += `&as_of=${encodeURIComponent(cutoff)}`;
      }
      const response = await fetch(`/api/v1/terminal/${encodeURIComponent(state.section.id)}?${query}`, { signal: controller.signal, cache: "no-store", headers: { Accept: "application/json" } });
      const payload = await response.json();
      if (state.request !== controller) return;
      if (!validContract(payload) || (!response.ok && payload.state !== "ERROR")) throw new Error("Backend returned an unsupported truth-state contract.");
      state.payload = payload;
      renderPayload(payload);
    } catch (error) {
      if (state.request !== controller) return;
      state.payload = null;
      renderPayload({ state: "ERROR", data: null, provenance: { source: "Backend request", classification: "UNAVAILABLE", as_of: null }, error: { code: controller.signal.reason === "timeout" ? "REQUEST_TIMEOUT" : "REQUEST_FAILED", message: controller.signal.reason === "timeout" ? "Backend request timed out. Refresh to retry." : "The backend response could not be loaded. Check the local service and retry." } });
    } finally {
      clearTimeout(timeout);
      if (state.request === controller) $("refresh").disabled = false;
    }
  }
  function renderPayload(payload) {
    const provenance = isObject(payload.provenance) ? payload.provenance : {};
    const data = isObject(payload.data) ? payload.data : {};
    const classification = provenance.classification || "UNKNOWN";
    const synthetic = String(classification).toUpperCase() === "SYNTHETIC" || data.input_classification === "SYNTHETIC";
    $("truth-strip").classList.toggle("synthetic", synthetic);
    $("state-badge").textContent = payload.state;
    $("state-badge").dataset.state = payload.state;
    $("source-label").textContent = `${printable(provenance.source)} · ${classification}`;
    $("asof-label").textContent = `As of ${provenance.as_of === null || provenance.as_of === undefined ? "unknown" : String(provenance.as_of)}`;
    $("truth-notice").textContent = `${payload.state === "STALE" ? "STALE · Historical snapshot; freshness requirement is not satisfied. " : ""}${synthetic ? "SYNTHETIC INPUTS · Demonstration and research only · No broker-observed performance" : provenance.evidence_state === "CALLER_SUPPLIED_NOT_VENDOR_VERIFIED" ? "Caller-supplied provenance · Vendor authenticity unverified · Research only" : "Research output requires review before any trading use."}`;
    if (isObject(data.overlay_financing) && data.overlay_financing.state === "UNKNOWN") $("truth-notice").textContent += " Hypothetical options financing UNKNOWN · Modeled overlay value is not funded NAV.";
    if (typeof data.funding_scope === "string" || typeof data.account?.funding_scope === "string" || isObject(data.funding) || isObject(data.account?.funding)) $("truth-notice").textContent += " SIMULATED cash account · Funding is limited to the explicitly supported contract, cash and reservation rules; no global funding certification.";
    if (isObject(data.account) && typeof data.account.mark_state === "string" && data.account.mark_state !== "AVAILABLE") $("truth-notice").textContent += ` Mark valuation ${data.account.mark_state} · Missing or stale marks do not establish reconciled NAV.`;
    if (isObject(data.account) && Object.hasOwn(data.account, "dividend_receivable")) $("truth-notice").textContent += " Dividend receivables are modeled entitlements, not spendable cash; payment converts the recognized balance without earning the income twice.";
    if (typeof data.scoring_family === "string") $("truth-notice").textContent += ` ${human(data.scoring_family)} scoring · ${data.fit_diagnostics?.coverage_state === "PARTIAL" ? "Partial model coverage; missing folds are excluded explicitly. " : ""}Calibration diagnostics are not approved trading confidence.`;
    if ($("analysis-engine").value === "observed-reference") $("truth-notice").textContent += " Reference-rate observations have DATE precision only · Intraday time, volume and historical publication availability are UNKNOWN · Retrospective model research is not executable-price evidence.";
    if ($("analysis-engine").value === "observed-equity") $("truth-notice").textContent += " Weekly historical archive · Source labels are not permanent security identities · Currency, corporate actions, universe membership and historical publication vintages are UNKNOWN or UNAVAILABLE · Not eligible for funded allocation.";
    if (typeof data.reconciliation_state === "string") $("truth-notice").textContent += ` Independent offline statement comparison: ${data.reconciliation_state} · ${data.reconciliation_state === "MATCH" ? "Fixture agreement only; external account reconciliation remains unverified." : "Unresolved or unavailable evidence; the account is not reconciled."}`;
    if (data.historical_publication_state === "UNKNOWN" || data.historical_availability === "UNKNOWN") $("truth-notice").textContent += " Historical publication availability UNKNOWN · Retrospective research is not a past trading decision.";
    if (typeof data.risk_scope === "string") $("truth-notice").textContent += " Scoped simulated-account model risk · Unavailable liquidity and historical tail estimates remain explicit; no trading approval.";
    if (typeof data.economically_validated === "boolean" && data.economically_validated === false && (data.report_id || data.experiments)) $("truth-notice").textContent += " Economic research diagnostics do not establish profitable alpha or production suitability.";
    if (data.title) $("section-title").textContent = String(data.title);
    if (data.description) $("section-description").textContent = String(data.description);
    const errorMessage = payload.error && (payload.error.message || payload.error.code);
    if (errorMessage) {
      $("error-panel").textContent = `${payload.error.code || payload.state}: ${errorMessage}`;
      $("error-panel").classList.remove("hidden");
    } else if (["UNAVAILABLE", "UNKNOWN"].includes(payload.state)) {
      $("error-panel").textContent = "No admissible data satisfies this view and its time/freshness policy. Admit supported local evidence in the workbench, adjust the explicit cutoff, or select the synthetic research demo.";
      $("error-panel").classList.remove("hidden");
    }
    renderMetrics(data.metrics);
    $("metrics").classList.toggle("depth-metrics", isObject(data.depth));
    const context = [
      ["Source identity", provenance.source],
      ["Data classification", classification],
      ["Evidence state", payload.state],
      ["Source timestamp", provenance.as_of],
      ["Trading authority", "Research only. No trading approval is implied."],
    ];
    if (data.description) context.push(["Method & scope", data.description]);
    if (data.notes) context.push(["Research notes", Array.isArray(data.notes) ? data.notes.join("\n") : data.notes]);
    if (data.warnings) context.push(["Limitations", Array.isArray(data.warnings) ? data.warnings.join("\n") : data.warnings]);
    Object.entries(provenance).filter(([key]) => !["source", "classification", "as_of"].includes(key)).forEach(([key, value]) => context.push([human(key), value]));
    const extra = Object.entries(data).filter(([key]) => !["title", "description", "metrics", "rows", "chart", "notes", "warnings"].includes(key));
    extra.forEach(([key, value]) => context.push([human(key), value]));
    $("context-content").replaceChildren(...context.map(([label, value]) => {
      const entry = el("div", "context-entry");
      const content = el("div", "context-value", printable(value));
      if (isObject(value) || Array.isArray(value) || printable(value).length > 300) {
        content.setAttribute("tabindex", "0");
        content.setAttribute("role", "region");
        content.setAttribute("aria-label", "Scrollable " + label);
      }
      entry.append(el("div", "context-label", label), content);
      return entry;
    }));
    state.tables = { observations: Array.isArray(data.rows) ? data.rows.filter(isObject) : [] };
    const addTable = (name, rows) => { if (Array.isArray(rows) && rows.length) state.tables[name] = rows.filter(isObject); };
    if (Array.isArray(data.folds)) addTable("pair_screening", data.folds.flatMap((fold, index) => Array.isArray(fold.screening) ? fold.screening.map(row => ({ fold: index, ...row })) : []));
    addTable("allocation_proposals", data.proposals);
    addTable("model_diagnostics", data.diagnostics);
    addTable("term_structure", data.term_structure);
    addTable("skew", data.skew);
    if (isObject(data.vts)) addTable("term_slope", data.vts.rows);
    if (isObject(data.scenarios)) addTable("modeled_scenarios", data.scenarios.scenarios);
    if (isObject(data.portfolio_exposure)) addTable("simulated_greek_exposures", data.portfolio_exposure.positions);
    if (isObject(data.account)) addTable("cash_reservations", data.account.reservations);
    addTable("funded_order_lifecycle", data.orders);
    addTable("cash_settlements", data.settlements);
    addTable("best_quote_flow", data.ofi_rows);
    if (isObject(data.kyle_lambda)) addTable("signed_volume_intervals", data.kyle_lambda.intervals);
    if (Array.isArray(data.experiments)) addTable("oos_economic_periods", data.experiments.flatMap(experiment => Array.isArray(experiment.periods) ? experiment.periods.map(row => ({ run_id: experiment.run_id, ...row })) : []));
    if (Array.isArray(data.scenarios)) addTable("funded_risk_scenarios", data.scenarios);
    addTable("liquidity_diagnostics", data.liquidity);
    addTable("lifecycle_events", data.lifecycle_events);
    addTable("trade_settlement_schedule", data.trade_settlements);
    addTable("pending_corporate_actions", data.pending_corporate_actions);
    addTable("corporate_actions", data.actions);
    addTable("cash_entitlements", data.entitlements);
    addTable("dividend_receivables", data.dividend_receivables);
    if (isObject(data.account)) addTable("dividend_receivables", data.account.dividend_receivables);
    if (isObject(data.fit_diagnostics)) addTable("model_fit_diagnostics", data.fit_diagnostics.folds);
    addTable("variance_scoring_observations", data.score_rows);
    if (Array.isArray(data.date_scores)) addTable("ranking_quality_by_date", data.date_scores.map(row => ({ observation_date: row.observation_date, model_ndcg: row.model?.ndcg, model_spearman_ic: row.model?.spearman_ic, model_ic_state: row.model?.ic_state, uniform_tied_ndcg: row.uniform_tied_scores?.ndcg, training_mean_rank_ndcg: row.training_mean_return_rank?.ndcg, training_baseline_evidence: row.training_baseline_evidence })));
    addTable("excluded_ranking_dates", data.excluded_dates);
    addTable("excluded_model_predictions", data.excluded_rows);
    if (isObject(data.comparisons)) addTable("paired_baseline_comparisons", Object.entries(data.comparisons).map(([baseline, comparison]) => ({ baseline, ...(isObject(comparison) ? comparison : { value: comparison }) })));
    if (typeof data.scoring_family === "string" && isObject(data.metrics)) addTable("model_appropriate_scores", Object.entries(data.metrics).map(([metric, value]) => ({ metric, ...(isObject(value) && scalarTag(value) === null ? value : { value }) })));
    addTable("unsettled_trade_cash", data.unsettled_trades);
    addTable("statement_discrepancies", data.discrepancies);
    addTable("independent_statement_positions", data.statement_positions);
    if (isObject(data.sector_factor_risk)) {
      addTable("assumed_sector_budgets", Object.entries(data.sector_factor_risk.sector_delta_gross_usd || {}).map(([sector, delta_gross_usd]) => ({ sector, delta_gross_usd, evidence: "CALLER_DECLARED_NOT_INDEPENDENTLY_VALIDATED" })));
      addTable("assumed_factor_budgets", Object.entries(data.sector_factor_risk.factor_abs_budget_usd || {}).map(([factor, absolute_budget_usd]) => ({ factor, absolute_budget_usd, current_signed_exposure_usd: data.sector_factor_risk.current_signed_factor_exposure_usd?.[factor], calibrated_factor_model: false })));
    }
    if (isObject(data.lifecycle)) { addTable("cash_entitlements", data.lifecycle.entitlements); addTable("unsettled_trade_cash", data.lifecycle.unsettled_trades); }
    if (!state.tables.observations.length) {
      const candidate = Object.keys(state.tables).find(key => key !== "observations" && state.tables[key].length);
      if (candidate) delete state.tables.observations;
    }
    const keys = Object.keys(state.tables), tableView = $("evidence-table-view");
    tableView.replaceChildren(...keys.map(key => { const option = el("option", "", human(key)); option.value = key; return option; }));
    tableView.value = keys[0];
    $("evidence-table-control").classList.toggle("hidden", keys.length <= 1);
    state.rows = state.tables[keys[0]];
    state.chart = isObject(data.chart) ? data.chart : analyticalChart(data) || researchChart(state.rows);
    $("analysis-grid").classList.toggle("no-chart", !state.chart && ["AVAILABLE", "STALE"].includes(payload.state));
    renderDepth(data.depth || null, payload.state);
    renderTable();
    renderChart();
  }
  function analyticalChart(data) {
    if (data.scoring_family === "CROSS_SECTIONAL_RANKING" && Array.isArray(data.date_scores)) {
      return { kind: "ranking-quality", title: "Saved ranking quality by observation date · no portfolio return", axes: [
        { key: "date_index", label: "Observation date", unit: "ordered index; exact dates retained in table" },
        { key: "model_ndcg", label: "Model ranking quality", unit: "NDCG; higher is better" },
        { key: "baseline_ndcg", label: "Training-only baseline quality", unit: "NDCG; missing baseline remains unknown" }],
        points: data.date_scores.map((row, date_index) => ({ observation_date: row.observation_date, date_index, model_ndcg: row.model?.ndcg, baseline_ndcg: row.training_mean_return_rank?.ndcg })) };
    }
    if (data.scoring_family === "CONDITIONAL_VARIANCE" && Array.isArray(data.score_rows)) {
      return { kind: "variance-calibration", title: "Saved variance forecasts versus noisy squared-innovation proxy", axes: [
        { key: "observation_index", label: "Reference observation", unit: "ordered index; exact dates retained in table" },
        { key: "forecast_variance", label: "Forecast conditional variance", unit: "squared fractional reference return" },
        { key: "innovation_squared_proxy", label: "Squared innovation proxy", unit: "squared return; not observed latent volatility" }],
        points: data.score_rows.map((row, observation_index) => ({ ...row, observation_index })) };
    }
    if (typeof data.risk_scope === "string" && Array.isArray(data.scenarios)) {
      return { kind: "scenario-comparison", title: "Actual simulated positions · current and asynchronous stress P&L", axes: [
        { key: "scenario_index", label: "Explicit stress scenario", unit: "index; names retained in table" },
        { key: "current_model_pnl", label: "Current-position modeled P&L", unit: "USD" },
        { key: "worst_async_model_pnl", label: "Worst asynchronous modeled P&L", unit: "USD" }],
        points: data.scenarios.map((row, scenario_index) => ({ ...row, scenario_index })) };
    }
    if (Array.isArray(data.experiments) && Array.isArray(data.rows)) {
      const points = data.rows.map(row => ({ cost_bps: row.cost_bps, slippage_bps: row.slippage_bps, compounded_return: row.performance?.compounded_return, scenario_id: row.scenario_id }));
      return { kind: "cost-sensitivity", title: "Saved OOS cost and slippage sensitivity · research only", axes: [
        { key: "cost_bps", label: "Transaction cost assumption", unit: "bps of turnover" },
        { key: "slippage_bps", label: "Slippage assumption", unit: "bps of turnover" },
        { key: "compounded_return", label: "Modeled OOS return", unit: "arithmetic fraction; no profitability validation" }], points };
    }
    if (isObject(data.kyle_lambda) && Array.isArray(data.kyle_lambda.intervals)) {
      return { kind: "print-impact", title: "Admitted print-volume and quote-price changes · not executable capacity", axes: [
        { key: "signed_volume", label: "Estimated signed print volume", unit: "underlying units" },
        { key: "interval_index", label: "Nonoverlapping quote interval", unit: "index" },
        { key: "price_change", label: "Midquote price change", unit: "USD" }],
        points: data.kyle_lambda.intervals.map((row, interval_index) => ({ ...row, interval_index })) };
    }
    if (typeof data.overlay_id === "string" && Array.isArray(data.rows)) {
      const points = data.rows.map(row => ({ ...row, approximate_combined_pnl: typeof row.combined_model_pnl === "string" && row.combined_model_pnl.trim() !== "" ? Number(row.combined_model_pnl) : NaN }));
      return { kind: "risk-surface", title: "Hypothetical equity and options scenario P&L", axes: [
        { key: "spot_shock", label: "Spot shock", unit: "arithmetic fraction" },
        { key: "volatility_shock", label: "Volatility shock", unit: "annualized decimal" },
        { key: "approximate_combined_pnl", label: "Combined modeled P&L", unit: "USD; approximate plot, exact table" }], points };
    }
    if (isObject(data.surface) && Array.isArray(data.surface.points) && ["options", "models", "research"].includes(state.section.id)) {
      return { kind: "volatility-surface", title: "Admitted CALL coordinates · modeled midpoint IV", axes: [
        { key: "log_forward_moneyness", label: "Log forward moneyness", unit: "log(K/F)" },
        { key: "ttm", label: "Time to expiry", unit: "ACT/365F years" },
        { key: "volatility", label: "Modeled IV", unit: "annualized decimal" }], points: data.surface.points };
    }
    if (state.section.id === "risk" && isObject(data.scenarios) && Array.isArray(data.scenarios.scenarios)) {
      return { kind: "risk-surface", title: "Simulated options scenario P&L", axes: [
        { key: "spot_shock", label: "Spot shock", unit: "arithmetic fraction" },
        { key: "volatility_shock", label: "Volatility shock", unit: "annualized decimal" },
        { key: "model_pnl", label: "Modeled scenario P&L", unit: "quote currency" }], points: data.scenarios.scenarios };
    }
    const pairs = Array.isArray(data.rows) ? data.rows.filter(row => [row.innovation_zscore, row.kalman_beta, row.fold].every(value => typeof value === "number" && Number.isFinite(value))) : [];
    if (pairs.length) return { kind: "pair-diagnostics", title: "Causal hedge and standardized innovation", axes: [
      { key: "fold", label: "Validation fold", unit: "index" },
      { key: "kalman_beta", label: "Dynamic hedge beta", unit: "level-price hedge units" },
      { key: "innovation_zscore", label: "Innovation z-score", unit: "predicted standard deviations" }], points: pairs };
    return null;
  }
  function researchChart(rows) {
    const supported = new Set(["FORWARD_RETURN_ESTIMATE", "CROSS_SECTIONAL_RANK_SCORE", "NEXT_OBSERVATION_RETURN_ESTIMATE", "FILTERED_RETURN_STATE"]);
    const points = rows.filter(row => supported.has(row.prediction_kind) && [row.prediction, row.target, row.fold].every(value => typeof value === "number" && Number.isFinite(value)));
    if (!points.length) return null;
    return { kind: "model-validation", title: "Walk-forward predictions & outcomes", axes: [
      { key: "prediction", label: "Model output", unit: points[0].prediction_kind === "CROSS_SECTIONAL_RANK_SCORE" ? "ranking score" : "arithmetic return" },
      { key: "fold", label: "Validation fold", unit: "index" },
      { key: "target", label: "Realized forward return", unit: "arithmetic return" }], points };
  }
  function renderDepth(depth, availability = "UNAVAILABLE") {
    const usable = isObject(depth) && ["AVAILABLE", "STALE"].includes(availability) && Array.isArray(depth.bids) && Array.isArray(depth.asks);
    $("depth-state").textContent = usable ? `${availability} · ${depth.instrument_id}` : ["ERROR", "UNKNOWN"].includes(availability) ? availability : "UNAVAILABLE";
    $("depth-empty").classList.toggle("hidden", usable);
    $("depth-content").classList.toggle("hidden", !usable);
    $("depth-levels").replaceChildren();
    $("depth-summary").replaceChildren();
    $("depth-chart").replaceChildren();
    $("depth-provenance").textContent = usable ? `${depth.source} · ${depth.classification} · ${depth.feed} · Known ${depth.available_at}` : "No admitted depth snapshot selected. No order book is inferred from bars.";
    if (!usable) return;
    $("depth-summary").append(el("span", "", `Observation ${depth.event_at}`), el("span", "", `Sequence ${depth.sequence} · ${depth.bids.length} bid / ${depth.asks.length} ask levels`));
    const quantity = row => row && typeof row.quantity === "string" && Number.isFinite(Number(row.quantity)) ? Number(row.quantity) : 0;
    const maximum = Math.max(...depth.bids.map(quantity), ...depth.asks.map(quantity), 1);
    for (let index = 0; index < Math.max(depth.bids.length, depth.asks.length); index++) {
      const bid = depth.bids[index], ask = depth.asks[index];
      const tr = el("tr");
      for (const [value, side, row] of [[bid?.quantity, "bid", bid], [bid?.price, "bid", bid], [ask?.price, "ask", ask], [ask?.quantity, "ask", ask]]) {
        const cell = el("td", "depth-" + side, value === undefined ? "Unknown" : String(value));
        cell.title = value === undefined ? "Unknown" : String(value);
        cell.style.setProperty("--depth-width", `${Math.min(100, quantity(row) / maximum * 100)}%`);
        cell.style.setProperty("--depth-fill", `color-mix(in srgb,var(--${side === "bid" ? "green" : "red"}) 10%,transparent)`);
        tr.append(cell);
      }
      $("depth-levels").append(tr);
    }
    const cumulative = levels => { let total = 0; return levels.map(row => ({ price: Number(row.price), cumulative: total += quantity(row) })); };
    const bids = cumulative(depth.bids), asks = cumulative(depth.asks);
    const lo = Math.min(...bids.map(row => row.price)), hi = Math.max(...asks.map(row => row.price));
    const top = Math.max(bids[bids.length-1].cumulative, asks[asks.length-1].cumulative, 1);
    const x = price => 35 + (hi === lo ? .5 : (price-lo)/(hi-lo)) * 530;
    const y = volume => 218 - volume/top*175;
    for (const [levels, name] of [[bids, "buy"], [asks, "sell"]]) {
      const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
      const steps = [[x(levels[0].price), 218]];
      levels.forEach((row, index) => { if (index) steps.push([x(row.price), y(levels[index-1].cumulative)]); steps.push([x(row.price), y(row.cumulative)]); });
      steps.push([x(levels[levels.length-1].price), 218]);
      line.setAttribute("points", steps.map(row => row.join(",")).join(" "));
      line.setAttribute("class", "depth-" + name + "-area");
      $("depth-chart").append(line);
    }
    for (const [value, position] of [[depth.bids[depth.bids.length-1].price, 35], [depth.asks[depth.asks.length-1].price, 480]]) {
      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", String(position)); label.setAttribute("y", "249"); label.setAttribute("class", "depth-chart-label"); label.textContent = value;
      $("depth-chart").append(label);
    }
    $("depth-legend").textContent = `${depth.currency} price · Displayed quantity · Observed levels only; no interpolated quotes. Exact source values remain in the table.`;
  }
  function orbitControls() {
    $("orbit-toggle").textContent = state.orbit ? "Pause orbit" : "Resume orbit";
    $("orbit-toggle").setAttribute("aria-pressed", String(state.orbit));
    $("orbit-toggle").disabled = !state.points.length || state.dimension !== 3;
    $("motion-state").textContent = state.dimension === 2 ? "2D projection · orbit paused" : state.orbit ? "Slow orbit · 0.045 rad/s" : motionPreference.matches ? "Reduced motion · camera paused" : "Camera paused";
  }
  function setOrbit(enabled) {
    state.orbit = enabled; orbitLast = null;
    if (orbitFrame !== null && typeof window.cancelAnimationFrame === "function") window.cancelAnimationFrame(orbitFrame);
    orbitFrame = null;
    renderChart();
  }
  function orbitTick(stamp) {
    orbitFrame = null;
    if (!state.orbit || !state.points.length || state.dimension !== 3) return;
    if (orbitLast !== null) state.angle += Math.min((stamp-orbitLast)/1000, .05) * .045;
    orbitLast = stamp;
    renderChart();
  }
  function renderMetrics(metrics) {
    const entries = isObject(metrics) ? Object.entries(metrics) : Array.isArray(metrics) ? metrics.map((item, index) => [item.label || `Metric ${index + 1}`, item]) : [];
    const shown = entries.slice(0, 12);
    $("metrics").dataset.count = String(shown.length);
    $("metrics").replaceChildren(...shown.map(([key, metric]) => {
      const item = isObject(metric) && scalarTag(metric) === null ? metric : { value: metric };
      const value = printable(item.value);
      const [knownLabel, knownDetail] = METRIC_LABELS[key] || [];
      const card = el("article", "metric-card");
      const label = el("div", "metric-label");
      label.append(el("span", "", item.label || knownLabel || human(key)), el("span", "metric-marker"));
      const textValue = typeof item.value === "string" && !Number.isFinite(Number(item.value));
      const valueElement = el("div", `metric-value${value.length > 15 ? " long-value" : ""}${textValue ? " text-value" : ""}`, value);
      const detail = [item.unit, item.state, item.description || item.note].filter((part) => part !== null && part !== undefined && part !== "").join(" · ");
      card.append(label, valueElement, el("div", "metric-detail", detail || knownDetail || "Backend research metric"));
      return card;
    }));
  }
  function visibleRows() {
    const term = $("table-search").value.toLocaleLowerCase();
    const rows = state.rows.filter((row) => !term || Object.values(row).some((value) => printable(value).toLocaleLowerCase().includes(term)));
    if (state.sort) {
      const { key, direction } = state.sort;
      rows.sort((a, b) => {
        const x = a[key], y = b[key];
        if (x === null || x === undefined) return y === null || y === undefined ? 0 : 1;
        if (y === null || y === undefined) return -1;
        const exact = exactNumericOrder(x, y);
        const comparison = exact === null ? printable(x).localeCompare(printable(y)) : exact;
        return comparison * direction;
      });
    }
    return rows;
  }
  function renderTable() {
    const columns = [];
    state.rows.forEach((row) => Object.keys(row).forEach((key) => { if (!columns.includes(key)) columns.push(key); }));
    state.columns = columns;
    const header = el("tr");
    columns.forEach((key) => {
      const th = el("th");
      th.scope = "col";
      const sorted = state.sort && state.sort.key === key;
      if (sorted) th.setAttribute("aria-sort", state.sort.direction === 1 ? "ascending" : "descending");
      const button = el("button", "table-sort", `${human(key)}${sorted ? state.sort.direction === 1 ? " ↑" : " ↓" : ""}`);
      button.addEventListener("click", () => { state.sort = { key, direction: sorted ? -state.sort.direction : 1 }; renderTable(); });
      th.append(button);
      header.append(th);
    });
    $("observations-table").querySelector("thead").replaceChildren(header);
    const rows = visibleRows();
    $("observations-table").querySelector("tbody").replaceChildren(...rows.slice(0, 500).map((row) => {
      const tr = el("tr");
      columns.forEach((key) => {
        const value = row[key];
        const td = el("td", value === null || value === undefined ? "unknown" : typeof value === "number" ? "numeric" : "", printable(value));
        td.title = printable(value);
        tr.append(td);
      });
      return tr;
    }));
    $("row-count").textContent = numberFormat.format(state.rows.length);
    $("table-empty").classList.toggle("hidden", rows.length !== 0);
    $("table-wrap").classList.toggle("hidden", rows.length === 0);
    $("table-empty").textContent = state.rows.length ? "No observations match this filter." : "No supported observations in this response.";
    $("table-summary").textContent = `Showing ${Math.min(rows.length, 500)} of ${rows.length} matching observations · ${state.rows.length} total`;
    $("export-csv").disabled = !rows.length;
  }
  function exportCsv() {
    // Spreadsheet formulas are text-escaped; all exported fields are quoted.
    const csvValue = (value) => {
      let text = value === null || value === undefined ? "" : scalarTag(value) !== null ? scalarTag(value) : typeof value === "object" ? JSON.stringify(value) : String(value);
      if (scalarTag(value) !== null || (typeof value === "string" && (/^[\s\u0000-\u001f]*[=+\-@]/.test(text) || /^[\t\r]/.test(text)))) text = `'${text}`;
      return `"${text.replace(/"/g, '""')}"`;
    };
    const rows = visibleRows();
    const csv = [state.columns.map(csvValue).join(","), ...rows.map((row) => state.columns.map((key) => csvValue(row[key])).join(","))].join("\r\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const link = el("a");
    link.href = url;
    link.download = `gig-${state.section.id}-${state.mode}.csv`;
    link.setAttribute("aria-label", "Download visible research observations as CSV");
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function axisSpecs(chart) {
    const kind = String(chart.kind || "").replace(/_/g, "-");
    const definitions = kind.includes("volatility") ? [{ key: "strike", label: "Strike", unit: "currency" }, { key: "tenor_years", label: "Tenor", unit: "years" }, { key: "iv", label: "Implied volatility", unit: "annualized decimal" }] : kind.includes("risk") ? [{ key: "equity_shock", label: "Equity shock", unit: "decimal" }, { key: "volatility_shock", label: "Volatility shock", unit: "decimal" }, { key: "pnl", label: "Scenario P&L", unit: "model units" }] : [{ key: "x", label: "Factor X", unit: "standardized units" }, { key: "y", label: "Factor Y", unit: "standardized units" }, { key: "z", label: "Factor Z", unit: "standardized units" }];
    if (Array.isArray(chart.axes)) chart.axes.slice(0, 3).forEach((axis, index) => { if (typeof axis === "string") definitions[index].label = axis; else if (isObject(axis)) Object.assign(definitions[index], axis); });
    else if (isObject(chart.axes)) ["x", "y", "z"].forEach((axis, index) => { const supplied = chart.axes[axis]; if (typeof supplied === "string") definitions[index].label = supplied; else if (isObject(supplied)) Object.assign(definitions[index], supplied); });
    return definitions;
  }
  function project(x, y, z, width, height) {
    const scale = Math.min(width * .29, height * .33) * state.zoom;
    if (state.dimension === 2) return { x: width * .48 + x * scale, y: height * .51 - z * scale, depth: y };
    const a = state.angle, b = state.tilt;
    const horizontal = x * Math.cos(a) - y * Math.sin(a);
    const depth = x * Math.sin(a) + y * Math.cos(a);
    return { x: width * .48 + horizontal * scale, y: height * .51 - (z * Math.cos(b) - depth * Math.sin(b)) * scale, depth };
  }
  function renderChart() {
    const canvas = $("analytics-chart");
    const context = canvas.getContext("2d");
    if (!context) return;
    const bounds = canvas.getBoundingClientRect();
    const width = Math.max(bounds.width, 1), height = Math.max(bounds.height, 1);
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * pixelRatio);
    canvas.height = Math.round(height * pixelRatio);
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, width, height);
    const style = getComputedStyle(document.documentElement);
    const muted = style.getPropertyValue("--muted").trim();
    const grid = style.getPropertyValue("--grid").trim();
    const accent = style.getPropertyValue("--accent").trim();
    $("chart-tooltip").classList.add("hidden");
    state.projected = [];
    const chart = state.chart;
    const axes = chart ? axisSpecs(chart) : [];
    const rawPoints = chart && Array.isArray(chart.points) ? chart.points : [];
    const points = rawPoints.filter((point) => isObject(point) && axes.every((axis) => typeof point[axis.key] === "number" && Number.isFinite(point[axis.key]))).slice(0, 1600);
    state.points = points;
    orbitControls();
    $("chart-empty").classList.toggle("hidden", points.length !== 0);
    $("chart-title").textContent = chart && chart.title ? String(chart.title) : chart && chart.kind ? human(chart.kind) : `${state.section.label} analytical view`;
    if (!points.length) { $("chart-legend").textContent = "Only finite backend observations are plotted."; return; }
    $("chart-legend").textContent = axes.map((axis) => `${axis.label} (${axis.unit || "unspecified units"})`).join(" · ") + (state.dimension === 2 ? ` · 2D projection: ${axes[0].label} vs ${axes[2].label}; ${axes[1].label} retained in tooltips` : "") + (rawPoints.length > 1600 ? " · Plot limited to 1,600 observations; table retains rows" : "");
    const ranges = axes.map((axis) => {
      let lo = Infinity, hi = -Infinity;
      points.forEach((point) => { lo = Math.min(lo, point[axis.key]); hi = Math.max(hi, point[axis.key]); });
      return [lo, hi];
    });
    const normalize = (value, index) => ranges[index][0] === ranges[index][1] ? 0 : 2 * (value - ranges[index][0]) / (ranges[index][1] - ranges[index][0]) - 1;
    const line = (from, to, color = grid, lineWidth = 1) => {
      const a = project(...from, width, height), b = project(...to, width, height);
      context.beginPath(); context.strokeStyle = color; context.lineWidth = lineWidth; context.moveTo(a.x, a.y); context.lineTo(b.x, b.y); context.stroke();
    };
    for (let index = 0; index <= 4; index++) {
      const tick = -1 + index / 2;
      line([-1, tick, -1], [1, tick, -1]);
      line([tick, -1, -1], [tick, 1, -1]);
      if (state.dimension === 2) line([-1, -1, tick], [1, -1, tick]);
    }
    const origin = [-1, -1, -1];
    const endpoints = [[1.12, -1, -1], [-1, 1.12, -1], [-1, -1, 1.12]];
    const axisColors = [accent, style.getPropertyValue("--amber").trim(), style.getPropertyValue("--green").trim()];
    const halo = document.documentElement.dataset.theme === "light" ? "rgba(255,255,255,0.9)" : "rgba(5,6,8,0.9)";
    // Every axis starts at the shared origin corner, so labels try several
    // offsets and are skipped rather than drawn over an already placed label.
    const placed = [], labelLog = [];
    const place = (text, x, y, color, font, offsets, required = false) => {
      context.font = font;
      const w = context.measureText(text).width, h = parseInt(font, 10) || 11;
      for (const [dx, dy] of offsets) {
        const left = Math.max(4, Math.min(x + (dx === "left" ? -w - 6 : dx), width - w - 4));
        const base = Math.max(h + 2, Math.min(y + dy, height - 4));
        const box = [left - 2, base - h - 1, left + w + 2, base + 3];
        if (placed.some((b) => box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1])) continue;
        placed.push(box); labelLog.push({ text, box });
        context.lineJoin = "round"; context.lineWidth = 3; context.strokeStyle = halo; context.strokeText(text, left, base);
        context.fillStyle = color; context.fillText(text, left, base);
        return true;
      }
      if (required) { const [dx, dy] = offsets[0]; context.fillStyle = color; context.fillText(text, Math.max(4, x + (dx === "left" ? -w - 6 : dx)), y + dy); }
      return false;
    };
    const visibleAxes = [0, 1, 2].filter((index) => !(state.dimension === 2 && index === 1));
    visibleAxes.forEach((index) => line(origin, endpoints[index], axisColors[index], 1.2));
    state.chartLabels = { place };
    visibleAxes.forEach((index) => {
      const labelPoint = project(...endpoints[index], width, height);
      const label = axes[index].label;
      place(label.length > 24 ? `${label.slice(0, 22)}…` : label, labelPoint.x, labelPoint.y, axisColors[index], "12px 'SFMono-Regular', Consolas, monospace",
        [[7, -5], [7, 14], ["left", -5], ["left", 14]], true);
    });
    visibleAxes.forEach((index) => {
      const far = [...origin]; far[index] = 1;
      const farPoint = project(...far, width, height);
      place(printable(ranges[index][1]), farPoint.x, farPoint.y, muted, "11px 'SFMono-Regular', Consolas, monospace", [[4, index === 2 ? -7 : 13], [4, index === 2 ? 13 : -7], ["left", 13], ["left", -7]]);
    });
    const originPoint = project(...origin, width, height);
    visibleAxes.forEach((index) => {
      const near = project(...endpoints[index].map((v, i) => (i === index ? -1 + 0.18 : v)), width, height);
      const dx = near.x - originPoint.x, dy = near.y - originPoint.y;
      const ox = dx >= 0 ? 4 : "left", oy = dy >= 0 ? 15 : -6;
      place(printable(ranges[index][0]), near.x, near.y, muted, "11px 'SFMono-Regular', Consolas, monospace",
        [[ox, oy], [ox, -oy + 9], [ox === 4 ? "left" : 4, oy], [ox === 4 ? "left" : 4, -oy + 9], [4, 28], ["left", 28]]);
    });
    const projected = points.map((point, index) => {
      const coordinates = axes.map((axis, axisIndex) => normalize(point[axis.key], axisIndex));
      return { ...project(...coordinates, width, height), point, index, coordinates };
    });
    if (String(chart.kind).includes("surface")) {
      // Connect only observations sharing an actual sampled axis value; no interpolation.
      for (let direction = 0; direction < 2; direction++) {
        const groups = new Map();
        projected.forEach((point) => { const key = point.point[axes[direction].key]; if (!groups.has(key)) groups.set(key, []); groups.get(key).push(point); });
        groups.forEach((group) => {
          group.sort((a, b) => a.point[axes[1 - direction].key] - b.point[axes[1 - direction].key]);
          for (let index = 1; index < group.length; index++) line(group[index - 1].coordinates, group[index].coordinates, grid, 1);
        });
      }
    }
    projected.sort((a, b) => a.depth - b.depth);
    projected.forEach((point) => {
      const colorPosition = Math.max(0, Math.min(1, (point.coordinates[2] + 1) / 2));
      context.fillStyle = `hsl(${195 + colorPosition * 30} 65% ${document.documentElement.dataset.theme === "light" ? 40 : 55 + colorPosition * 17}%)`;
      context.beginPath();
      context.arc(point.x, point.y, String(chart.kind).includes("surface") ? 2.7 : 3.7, 0, Math.PI * 2);
      context.fill();
    });
    if (points.length <= 25) {
      projected.filter((point) => point.point.symbol).sort((a, b) => b.depth - a.depth).forEach((point) => {
        state.chartLabels.place(String(point.point.symbol), point.x, point.y, muted, "10px 'SFMono-Regular', Consolas, monospace", [[6, -5], ["left", -5], [6, 12], ["left", 12]]);
      });
    }
    state.projected = projected;
    canvas.dataset.labels = JSON.stringify(labelLog);
    canvas.setAttribute("aria-label", `${$("chart-title").textContent}. ${points.length} backend observations. ${axes.map((axis) => `${axis.label}, ${axis.unit || "units unspecified"}`).join("; ")}. ${state.dimension}D view. Underlying observations also available in the table. Drag or use arrow keys to rotate; plus and minus zoom.`);
    if (state.orbit && state.dimension === 3 && orbitFrame === null && typeof window.requestAnimationFrame === "function") orbitFrame = window.requestAnimationFrame(orbitTick);
  }
  function chartPointer(event) {
    const bounds = $("analytics-chart").getBoundingClientRect();
    const x = event.clientX - bounds.left, y = event.clientY - bounds.top;
    if (state.drag) { state.angle += (event.clientX - state.drag.x) / 180; state.tilt = Math.max(-1.2, Math.min(1.2, state.tilt + (event.clientY - state.drag.y) / 220)); state.drag = { x: event.clientX, y: event.clientY }; renderChart(); return; }
    let selected = null, distance = 12;
    state.projected.forEach((point) => { const d = Math.hypot(point.x - x, point.y - y); if (d < distance) { selected = point; distance = d; } });
    const tooltip = $("chart-tooltip");
    tooltip.classList.toggle("hidden", selected === null);
    if (selected) {
      tooltip.textContent = Object.entries(selected.point).slice(0, 10).map(([key, value]) => `${human(key)}: ${printable(value)}`).join("\n");
      tooltip.style.left = `${Math.max(4, Math.min(x + 15, bounds.width - 225))}px`;
      tooltip.style.top = `${Math.max(4, Math.min(y + 12, bounds.height - 140))}px`;
    }
  }
  const examples = {
    options: { spot: 100, strike: 100, rate: 0.05, div: 0, vol: 0.2, ttm: 1, call: true },
    kalman: { observations: [[1], [2], [1.5], [2.2], [2.4]], transition: [[1]], observation: [[1]], process_covariance: [[0.1]], observation_covariance: [[1]], initial_mean: [0], initial_covariance: [[1]], smooth: true },
    garch: { returns: [0.01, -0.015, 0.007, 0.02, -0.025, 0.012, -0.008, 0.009, -0.018, 0.013, 0.005, -0.009, 0.014, -0.021, 0.011, 0.008, -0.012, 0.017, -0.006, 0.009, -0.013, 0.004, 0.016, -0.008, 0.012, -0.019, 0.007, 0.015, -0.01, 0.006], max_iterations: 100 },
    hmm: { observations: [-0.02, -0.01, 0.01, 0.02, -0.01, 0.03, 0.02, -0.02], states: 2, max_iterations: 50 },
    microstructure: { quotes: [{ timestamp_ns: 100, bid: 99, ask: 101, bid_size: 10, ask_size: 12, instrument: "SYN001", source: "deterministic-demo", data_kind: "SYNTHETIC" }], trades: [{ timestamp_ns: 101, price: 101, size: 5, instrument: "SYN001", source: "deterministic-demo", data_kind: "SYNTHETIC" }] },
  };
  examples.calendar = { calendar: "XNYS", start: "2026-11-23", end: "2026-11-30", as_of: "2026-11-27T18:00:00+00:00", last_bar_close: "2026-11-25T21:00:00+00:00", max_missed_sessions: 0, grace_seconds: 0 };
  examples["corporate-actions"] = { source: "authored-synthetic-action-fixture", classification: "SYNTHETIC", currency: "USD", as_of: "2026-09-29T20:00:00+00:00", mode: "SPLIT_ADJUSTED", rows: [ { instrument_id: "SYN001", timestamp: "2026-09-28T20:00:00+00:00", available_at: "2026-09-28T20:00:00+00:00", open: 100, high: 101, low: 99, close: 100, volume: 1000000 }, { instrument_id: "SYN001", timestamp: "2026-09-29T20:00:00+00:00", available_at: "2026-09-29T20:00:00+00:00", open: 50, high: 50.5, low: 49.5, close: 50, volume: 2000000 } ], actions: [ { action_id: "authored-split-1", instrument_id: "SYN001", source: "authored-synthetic-action-fixture", kind: "SPLIT", effective_at: "2026-09-29T13:30:00+00:00", available_at: "2026-09-27T21:00:00+00:00", ratio_new: 2, ratio_old: 1 } ] };
  examples["cost-evidence"] = { currency: "USD", as_of: "2026-09-30T21:00:00+00:00", match_tolerance_seconds: 1, max_participation: "0.25",
    estimates: [{ estimate_id: "authored-estimate-1", instrument: "SYN001", side: "BUY", quantity: "100", reference_price: "50", spread_bps: "4", commission_bps: "1", impact_eta: "0.1", adv_value: "5000000" }],
    fills: [
      { fill_id: "authored-sim-1", instrument: "SYN001", side: "BUY", quantity: "100", price: "50.03", fees: "1.00", executed_at: "2026-09-30T14:00:05+00:00", knowledge_at: "2026-09-30T14:00:05+00:00", evidence_class: "SIMULATED", source: "offline-simulator", arrival: { bid: "49.98", ask: "50.02", observed_at: "2026-09-30T14:00:00+00:00", source: "authored-quote" }, post_trade: { mid: "50.04", observed_at: "2026-09-30T14:05:00+00:00", source: "authored-quote" } },
      { fill_id: "authored-hist-1", instrument: "SYN002", side: "SELL", quantity: "200", price: "20.00", fees: "0.50", executed_at: "2026-09-30T15:00:00+00:00", knowledge_at: "2026-09-30T16:00:00+00:00", evidence_class: "OBSERVED_HISTORICAL", source: "authored-execution-report", arrival: { bid: "20.01", ask: "20.03", observed_at: "2026-09-30T14:59:59+00:00", source: "authored-quote" } } ],
    prints: [
      { print_id: "authored-print-1", instrument: "SYN002", price: "20.00", size: "200", observed_at: "2026-09-30T15:00:00.400+00:00", source: "authored-tape" },
      { print_id: "authored-print-2", instrument: "SYN002", price: "20.02", size: "600", observed_at: "2026-09-30T14:59:59.500+00:00", source: "authored-tape" } ] };
  examples.provenance = { record_id: "UCI_DOW_JONES_WEEKLY_2011", as_of: "2026-11-04T00:00:00+00:00", purpose: "point_in_time_research" };
  const fixtureTimestamp = (day) => `2026-09-${String(day).padStart(2, "0")}T20:00:00+00:00`;
  examples["workspace-bars"] = { source: "authored-synthetic-workbench-v1", classification: "SYNTHETIC", corporate_action_policy: "RAW", currency: "USD", rows: Array.from({ length: 28 }, (_, index) => ["SYN001", "SYN002"].map((instrument_id, instrument) => { const close = instrument ? 50 - index * .2 : 100 + index * .5; return { instrument_id, timestamp: fixtureTimestamp(index + 1), available_at: fixtureTimestamp(index + 1), open: close - .2, high: close + 1, low: close - 1, close, volume: 1000000 }; })).flat() };
  examples["workspace-news"] = { as_of: fixtureTimestamp(28), aliases: { "SYN001": "SYN001" }, articles: [{ source: "authored-synthetic-workbench-v1", article_id: "synthetic-fixture-news-1", url: "", license: "Authored synthetic research fixture", title: "SYN001 reports profit growth", body: "This authored synthetic example mentions SYN001. Profit growth is a lexical research input, not an observed article or validated signal.", published_at: fixtureTimestamp(28), available_at: fixtureTimestamp(28), symbols: ["SYN001"] }] };
  examples["workspace-research"] = { source: "authored-synthetic-workbench-v1", classification: "SYNTHETIC", model: "ridge", min_train: 10, test_size: 5, embargo: 1, rows: Array.from({ length: 28 }, (_, index) => ({ timestamp: fixtureTimestamp(index + 1), available_at: fixtureTimestamp(index + 1), label_end: fixtureTimestamp(index + 2), features: { authored_factor: Math.sin(index / 3), lagged_feature: Math.cos(index / 4) }, target: .01 * Math.sin((index + 1) / 3) })) };
  examples["workspace-simulate"] = { source_id: "authored-synthetic-workbench-v1", classification: "SYNTHETIC", currency: "USD", run_id: "authored-synthetic-rehearsal-1", asof: fixtureTimestamp(28), now: fixtureTimestamp(28), scores: { SYN001: 1, SYN002: -1 }, prices: { SYN001: 113.5, SYN002: 44.6 }, sectors: { SYN001: "Authored sector A", SYN002: "Authored sector B" }, nav: 1000000, n_long: 1, n_short: 1 };
  const fixtureInstrument = { instrument_id: "SYN001", symbol: "SYN001", calendar: "XNYS", currency: "USD" };
  examples["workspace-ingest"] = { provider: "alpaca_iex", mode: "OFFLINE_FIXTURE", job_id: "authored-synthetic-iex-protocol-v1", instruments: [fixtureInstrument], start: "2026-09-28T14:30:00Z", end: "2026-09-28T14:31:00Z", timeframe: "1Min", now: "2026-09-28T14:32:00Z", pages: [{ observed_at: "2026-09-28T14:32:00Z", body: { bars: { SYN001: [{ t: "2026-09-28T14:30:00Z", o: 100, h: 101, l: 99, c: 100.5, v: 1000 }] }, next_page_token: null } }] };
  examples["workspace-universe"] = { universe_id: "authored-synthetic-universe", source: "authored-synthetic-workbench-v1", effective_at: fixtureTimestamp(28), available_at: fixtureTimestamp(28), members: [fixtureInstrument] };
  const sequenceStamp = hour => new Date(Date.UTC(2026, 10, 1, hour)).toISOString();
  examples["workspace-train-sequence"] = { source: "authored-synthetic-sequence-workbench-v1", classification: "SYNTHETIC", sequence_length: 4, train_size: 24, hidden_size: 4, epochs: 4, learning_rate: .02, seed: 17, rows: Array.from({ length: 32 }, (_, index) => ({ timestamp: sequenceStamp(index), available_at: sequenceStamp(index), label_end: sequenceStamp(index + 1), features: { momentum: Math.sin(index / 3), reversal: Math.cos(index / 4) }, target: .5 * Math.sin(index / 3) - .3 * Math.cos(index / 4) })) };
  examples["workspace-predict-sequence"] = { model_id: "0".repeat(64), rows: examples["workspace-train-sequence"].rows.map(({ timestamp, available_at, features }) => ({ timestamp, available_at, features })) };
  examples["workspace-treasury"] = { provider: "us_treasury", mode: "PUBLIC_OBSERVED", year: 2025 };
  examples["workspace-depth"] = { instrument_id: "SYN101", source: "authored-depth-workbench-v1", classification: "SYNTHETIC", feed: "CALLER_SUPPLIED", currency: "USD", sequence: 1, event_at: "2026-11-02T15:49:00Z", available_at: "2026-11-02T15:49:00Z", bids: Array.from({ length: 16 }, (_, index) => ({ price: (110-index*.02).toFixed(2), quantity: String(120+index*25) })), asks: Array.from({ length: 16 }, (_, index) => ({ price: (110.04+index*.02).toFixed(2), quantity: String(90+index*30) })) };
  examples["workspace-finbert"] = examples["workspace-news"];
  examples["workspace-execution-engine"] = { operation: "prepare", now: fixtureTimestamp(28), client_id: "authored-synthetic-offline-intent-1", symbol: "SYN001", side: "buy", quantity: "1", reference_price: "100", price_asof: fixtureTimestamp(28), max_notional: "1000", max_position_quantity: "10" };
  const pipelineStamp = index => new Date(Date.UTC(2026, 10, 2, 14, 30 + index)).toISOString();
  const pipelineSymbols = ["SYN101", "SYN102"];
  const pipelineMetadata = Object.fromEntries(pipelineSymbols.map(symbol => [symbol, { sector: "Authored sector", factors: { market: 1 } }]));
  const pipelineLimits = { max_gross: 1, max_abs_net: .5, max_name_weight: .3, max_sector_gross: 1, max_factor_exposure: { market: .5 }, min_cash: "0" };
  examples["workspace-pipeline-bars"] = { source: "authored-pipeline-workbench-v1", classification: "SYNTHETIC", corporate_action_policy: "RAW", currency: "USD", rows: Array.from({ length: 80 }, (_, index) => pipelineSymbols.map((instrument_id, instrument) => { const close = (100 + instrument * 30) * Math.exp(.001 * index + .018 * Math.sin(index / (3 + instrument)) + .007 * Math.cos(index * 1.73 + instrument)); return { instrument_id, timestamp: pipelineStamp(index), available_at: pipelineStamp(index), open: close, high: close, low: close, close, volume: 1000 + 7 * index + instrument }; })).flat() };
  examples["workspace-pipeline-universe"] = { universe_id: "authored-pipeline-universe-v1", source: "authored-universe-workbench-v1", effective_at: "2026-11-01T00:00:00Z", available_at: "2026-11-01T00:00:00Z", members: pipelineSymbols.map(instrument_id => ({ instrument_id, symbol: instrument_id, calendar: "XNYS", currency: "USD" })) };
  examples["workspace-research-dataset"] = { source: "authored-pipeline-workbench-v1", classification: "SYNTHETIC", currency: "USD", universe_id: "authored-pipeline-universe-v1", start: pipelineStamp(0), end: pipelineStamp(79), as_of: pipelineStamp(79), decision_delay_seconds: 0, horizon: 1, lookback: 5, adjustment: "RAW" };
  examples["workspace-financial-reference"] = { operation: "ingest-reference" };
  examples["workspace-financial-search"] = { operation: "search", query: "10-year Treasury yield 2023-01-03", as_of: new Date().toISOString(), limit: 3, expand: true };
  examples["workspace-financial-answer"] = { ...examples["workspace-financial-search"], operation: "answer" };
  examples["workspace-financial-latent"] = { ...examples["workspace-financial-answer"], retrieval_mode: "latent", latent_dimensions: 8 };
  examples["workspace-financial-hybrid"] = { ...examples["workspace-financial-answer"], retrieval_mode: "hybrid", latent_dimensions: 8, extract_facts: true };
  examples["workspace-financial-facts"] = { ...examples["workspace-financial-answer"], extract_facts: true };
  examples["workspace-financial-entity-protocol"] = { operation: "admit-authored", source_id: "gig-authored-entity-protocol-v2", title: "GIG authored entity syntax protocol", license: "GIG-AUTHORED", text: "GIG AUTHORED SYNTHETIC entity syntax protocol. NYSE:BRK.B and NASDAQ:AAPL are source mention examples only. NYSE:BRK.BOGUS is an unsupported whole candidate. Issuer identity, exchange membership and historical eligibility are UNKNOWN. No market observations or trading approval are supplied." };
  examples["workspace-financial-entities"] = { ...examples["workspace-financial-answer"], query: "entity syntax protocol", extract_facts: true, nlp_profile: "v2" };
  examples["workspace-financial-html"] = { operation: "admit-html", source_id: "gig-authored-northwind-fy2024", title: "Northwind Synthetic Co FY2024 results (GIG authored fixture)", license: "GIG-AUTHORED", html: "<html><body><h1>Northwind Synthetic Co FY2024 annual results</h1><p>GIG AUTHORED SYNTHETIC fixture, not an issuer filing. Revenue grew as subscription demand improved.</p><table><caption>Summary financial statements (USD millions)</caption><tr><th></th><th>FY2023</th><th>FY2024</th></tr><tr><td>Revenue</td><td>1,000</td><td>1,120</td></tr><tr><td>Cost of revenue</td><td>(600)</td><td>(650)</td></tr><tr><td>Gross profit</td><td>400</td><td>470</td></tr><tr><td>Operating income</td><td>150</td><td>190.5</td></tr><tr><td>Net income</td><td>(50)</td><td>80</td></tr><tr><td>Diluted EPS</td><td>(0.50)</td><td>0.78</td></tr><tr><td>Net cash provided by operating activities</td><td>120</td><td>210</td></tr><tr><td>Capital expenditures</td><td>(40)</td><td>(60)</td></tr><tr><td>Total assets</td><td>2,000</td><td>2,290</td></tr><tr><td>Total liabilities</td><td>1,200</td><td>1,350</td></tr><tr><td>Total shareholders' equity</td><td>800</td><td>940</td></tr></table><p>Management expects capital expenditures to moderate.</p></body></html>" };
  examples["workspace-financial-statements"] = { operation: "statements", document_id: "0".repeat(64) };
  examples["workspace-financial-digest"] = { operation: "synthesize", query: "revenue capital expenditures", as_of: new Date().toISOString(), per_document: 2 };
  examples["workspace-financial-watch"] = { operation: "watch-create", query: "capital expenditures", as_of: new Date().toISOString(), limit: 5 };
  examples["workspace-financial-watch-check"] = { operation: "watch-check", watch_id: "0".repeat(64), as_of: new Date().toISOString() };
  examples["workspace-financial-investigate"] = { operation: "investigate", title: "Northwind synthetic fixture review", questions: ["revenue", "capital expenditures", "net income"], as_of: new Date().toISOString() };
  examples["workspace-financial-call"] = {"operation": "admit-authored", "source_id": "gig-authored-northwind-q4-call", "title": "Northwind Synthetic Co Q4 2024 earnings call (GIG authored fixture)", "license": "GIG-AUTHORED", "text": "Northwind Synthetic Co fourth quarter 2024 earnings call (GIG authored synthetic fixture)\nPrepared Remarks\nOperator: Good morning and welcome to the Northwind Synthetic fourth quarter call.\nJane Doe - Chief Executive Officer: Thank you. Revenue grew 12% to 1,120 million. We expect full-year 2025 revenue growth of approximately 8% to 10%.\nMargins improved on mix. We will keep investing in the platform.\nRaj Patel, Chief Financial Officer: Operating income was 190.5 million. Capital expenditures could rise if demand holds. Free cash flow was 135 million.\nQuestion-and-Answer Session\nOperator: Our first question comes from Alex Kim.\nAlex Kim (Analyst, Example Securities): Can you size the 2025 capital expenditures? And what drives the margin outlook?\nRaj Patel: We plan to spend roughly 120 million. Pricing is stable.\nJane Doe: Mix should continue to help.\nOperator: The next question comes from Maria Silva.\nMaria Silva (Analyst, Sample Bank): Is there any change to the dividend?\nOperator: That concludes the call.\n"};
  examples["workspace-financial-transcript"] = { operation: "transcript", document_id: "0".repeat(64) };
  examples["workspace-financial-compare"] = { operation: "compare", base_document_id: "0".repeat(64), revised_document_id: "0".repeat(64) };
  examples["workspace-financial-mention-diff"] = { operation: "mention-diff", base_document_id: "0".repeat(64), revised_document_id: "0".repeat(64), nlp_profile: "v1" };
  examples["workspace-financial-record-replay"] = { operation: "research-replay", record_id: "0".repeat(64) };
  examples["workspace-financial-replay"] = { operation: "replay", snapshot_id: "0".repeat(64) };
  examples["workspace-financial-verify"] = { operation: "verify-citation", snapshot_id: "0".repeat(64), passage_id: "0".repeat(64) };
  examples["workspace-research-rf"] = { dataset_id: "0".repeat(64), model: "random_forest", min_train: 20, test_size: 10, embargo: 1, seed: 17 };
  examples["workspace-research-xt"] = { ...examples["workspace-research-rf"], model: "extra_trees" };
  examples["workspace-research-knn"] = { ...examples["workspace-research-rf"], model: "knn" };
  examples["workspace-research-bayes"] = { ...examples["workspace-research-rf"], model: "bayesian_ridge" };
  examples["workspace-research-gam"] = { ...examples["workspace-research-rf"], model: "additive_spline", min_train: 30 };
  examples["workspace-research-glm"] = { ...examples["workspace-research-rf"], model: "binomial_glm" };
  examples["workspace-research-svr"] = { ...examples["workspace-research-rf"], model: "svr" };
  for (const [key, , , minTrain] of boundedFamilies) examples[boundedKind(key)] = { ...examples["workspace-research-rf"], model: key, min_train: minTrain, seed: 42 };
  for (const [key, , , example] of quantAnalytics) examples[key] = example;
  examples["workspace-calculator-save"] = { operation: "save", kind: "track-record-length", inputs: { sharpe: .1, benchmark: 0, skew: -.5, kurtosis: 5, confidence: .95 } };
  examples["workspace-calculator-replay"] = { operation: "replay", record_id: "0".repeat(64) };
  examples["workspace-calculator-dataset"] = { operation: "from-dataset", dataset_id: "0".repeat(64), kind: "stationary-bootstrap", instrument_id: "US:AAA", field: "return_1", inputs: { mean_block: 5, replications: 100, seed: 1, confidence: .95 } };
  examples["workspace-calculator-covariance"] = { operation: "from-record", record_id: "0".repeat(64), kind: "hierarchical-risk-parity", inputs: {} };
  examples["workspace-research-gpr"] = { dataset_id: "0".repeat(64), model: "gaussian_process", min_train: 20, test_size: 10, embargo: 1, seed: 17 };
  examples["workspace-research-model"] = { dataset_id: "0".repeat(64), model: "ridge", min_train: 20, test_size: 10, embargo: 1, seed: 17 };
  examples["workspace-research-predict"] = { run_id: "0".repeat(64), dataset_id: "0".repeat(64) };
  examples["workspace-research-allocate"] = { run_id: "0".repeat(64), now: pipelineStamp(79), initial_cash: "100000", n_long: 1, n_short: 1, gross_leverage: .2, max_name_weight: .1, lot_size: "1", price_tick: "0.01", max_age_seconds: 3600, annualization: 98280, metadata: pipelineMetadata, limits: pipelineLimits, commission_bps: 5, slippage_bps: 5 };
  examples["workspace-research-prepare"] = { allocation_id: "0".repeat(64), now: pipelineStamp(79) };
  const pipelineMarks = Object.fromEntries(examples["workspace-pipeline-bars"].rows.slice(-2).map(row => [row.instrument_id, { price: row.close.toFixed(2), as_of: row.timestamp }]));
  examples["workspace-portfolio-accounting"] = { operation: "reconcile", initial_cash: "100000", now: pipelineStamp(79), marks: pipelineMarks, max_age_seconds: 60, metadata: pipelineMetadata, limits: pipelineLimits };
  examples["workspace-offline-dispatch"] = { operation: "dispatch", now: pipelineStamp(79) };
  examples["workspace-offline-fill"] = { operation: "fill", now: pipelineStamp(79), client_id: "copy-a-prepared-client-id", quantity: "1", price: pipelineMarks.SYN101.price, fee: "0.10" };
  examples["workspace-schedule-twap"] = { operation: "schedule-twap", now: pipelineStamp(79), client_id: "authored-offline-twap-v1", symbol: "SYN101", side: "buy", quantity: "2", start: pipelineStamp(79), end: pipelineStamp(81), slices: 2, lot_size: "1", max_notional: "1000", max_position_quantity: "1000" };
  examples["workspace-advance-clock"] = { operation: "advance-clock", now: pipelineStamp(79), marks: { SYN101: { reference_price: pipelineMarks.SYN101.price, price_asof: pipelineStamp(79) } } };
  examples["workspace-pairs-research"] = { dataset_id: "0".repeat(64), training_window: 30, test_size: 10, embargo: 1, alpha: .05, entry_z: 1.5, cost_bps: 10, borrow_bps: 0, max_gross: .2, process_variance: .00001, observation_variance: .01 };
  examples["workspace-pairs-replay"] = { run_id: "0".repeat(64) };
  examples["workspace-research-signals"] = { run_ids: ["0".repeat(64)], now: pipelineStamp(79), lookback: 20, min_ic_observations: 5 };
  examples["workspace-signals-replay"] = { signal_id: "0".repeat(64) };
  examples["workspace-research-screen"] = { signal_id: "0".repeat(64), buy_threshold: .25, sell_threshold: -.25 };
  examples["workspace-research-discover"] = { operation: "from-runs", run_ids: ["0".repeat(64)], now: pipelineStamp(79), lookback: 20, min_ic_observations: 5, buy_threshold: .25, sell_threshold: -.25 };
  examples["workspace-research-options"] = { operation: "link-options", screen_id: "0".repeat(64), options_run_id: "0".repeat(64) };
  examples["workspace-research-options-scan"] = { operation: "scan-options", screen_id: "0".repeat(64), options_run_ids: ["0".repeat(64), "1".repeat(64)] };
  examples["workspace-research-monitor"] = { operation: "monitor", signal_ids: ["0".repeat(64), "1".repeat(64)], buy_threshold: .25, sell_threshold: -.25 };
  examples["workspace-research-enrich"] = { operation: "enrich-mentions", screen_id: "0".repeat(64), document_id: "0".repeat(64), nlp_profile: "v2" };
  examples["workspace-screen-replay"] = { screen_id: "0".repeat(64) };
  examples["workspace-options-admit"] = {"source":"authored-bsm-chain-workbench-v1","classification":"SYNTHETIC","feed":"MODELED_FIXTURE_NOT_OPRA","underlying_id":"SYN101","currency":"USD","event_at":"2026-11-02T15:49:00+00:00","acquired_at":"2026-11-02T15:49:00+00:00","knowledge_at":"2026-11-02T15:49:00+00:00","spot":110,"spot_source_identity":"authored-synthetic-spot-110","spot_event_at":"2026-11-02T15:49:00+00:00","spot_knowledge_at":"2026-11-02T15:49:00+00:00","quotes":[{"contract_id":"AUTHORED:SYN101:0:100:CALL","expiry":"2026-12-02T15:49:00+00:00","strike":100,"right":"CALL","multiplier":100,"style":"EUROPEAN","bid":10.40472,"ask":10.42472,"volume":120,"open_interest":200},{"contract_id":"AUTHORED:SYN101:0:100:PUT","expiry":"2026-12-02T15:49:00+00:00","strike":100,"right":"PUT","multiplier":100,"style":"EUROPEAN","bid":0.248822,"ask":0.268822,"volume":120,"open_interest":200},{"contract_id":"AUTHORED:SYN101:0:110:CALL","expiry":"2026-12-02T15:49:00+00:00","strike":110,"right":"CALL","multiplier":100,"style":"EUROPEAN","bid":3.095022,"ask":3.115022,"volume":130,"open_interest":210},{"contract_id":"AUTHORED:SYN101:0:110:PUT","expiry":"2026-12-02T15:49:00+00:00","strike":110,"right":"PUT","multiplier":100,"style":"EUROPEAN","bid":2.914497,"ask":2.934497,"volume":130,"open_interest":210},{"contract_id":"AUTHORED:SYN101:0:120:CALL","expiry":"2026-12-02T15:49:00+00:00","strike":120,"right":"CALL","multiplier":100,"style":"EUROPEAN","bid":0.396894,"ask":0.416894,"volume":140,"open_interest":220},{"contract_id":"AUTHORED:SYN101:0:120:PUT","expiry":"2026-12-02T15:49:00+00:00","strike":120,"right":"PUT","multiplier":100,"style":"EUROPEAN","bid":10.191742,"ask":10.211742,"volume":140,"open_interest":220},{"contract_id":"AUTHORED:SYN101:1:100:CALL","expiry":"2027-01-02T15:49:00+00:00","strike":100,"right":"CALL","multiplier":100,"style":"EUROPEAN","bid":11.310087,"ask":11.330087,"volume":120,"open_interest":200},{"contract_id":"AUTHORED:SYN101:1:100:PUT","expiry":"2027-01-02T15:49:00+00:00","strike":100,"right":"PUT","multiplier":100,"style":"EUROPEAN","bid":0.993654,"ask":1.013654,"volume":120,"open_interest":200},{"contract_id":"AUTHORED:SYN101:1:110:CALL","expiry":"2027-01-02T15:49:00+00:00","strike":110,"right":"CALL","multiplier":100,"style":"EUROPEAN","bid":4.732927,"ask":4.752927,"volume":130,"open_interest":210},{"contract_id":"AUTHORED:SYN101:1:110:PUT","expiry":"2027-01-02T15:49:00+00:00","strike":110,"right":"PUT","multiplier":100,"style":"EUROPEAN","bid":4.366483,"ask":4.386483,"volume":130,"open_interest":210},{"contract_id":"AUTHORED:SYN101:1:120:CALL","expiry":"2027-01-02T15:49:00+00:00","strike":120,"right":"CALL","multiplier":100,"style":"EUROPEAN","bid":1.417172,"ask":1.437172,"volume":140,"open_interest":220},{"contract_id":"AUTHORED:SYN101:1:120:PUT","expiry":"2027-01-02T15:49:00+00:00","strike":120,"right":"PUT","multiplier":100,"style":"EUROPEAN","bid":11.000716,"ask":11.020716,"volume":140,"open_interest":220}]};
  examples["workspace-options-research"] = {"source":"authored-bsm-chain-workbench-v1","classification":"SYNTHETIC","underlying_id":"SYN101","observation_at":"2026-11-02T15:49:00+00:00","as_of":"2026-11-02T15:49:00+00:00","max_age_seconds":3600,"rate":0.03,"dividend_yield":0.01,"positions":[{"contract_id":"AUTHORED:SYN101:0:110:CALL","quantity":2},{"contract_id":"AUTHORED:SYN101:0:110:PUT","quantity":-1}],"spot_shocks":[-0.1,0,0.1],"volatility_shocks":[-0.02,0,0.02]};
  examples["workspace-options-replay"] = { run_id: "0".repeat(64) };
  examples["workspace-allocate-signals"] = { ...examples["workspace-research-allocate"], signal_id: "0".repeat(64) };
  delete examples["workspace-allocate-signals"].run_id;
  examples["workspace-allocate-screen"] = { ...examples["workspace-research-allocate"], screen_id: "0".repeat(64) };
  delete examples["workspace-allocate-screen"].run_id;
  examples["workspace-research-strategy"] = { ...examples["workspace-allocate-screen"], operation: "simulate", fill: true };
  examples["workspace-research-size"] = { operation: "size", screen_id: "0".repeat(64), now: pipelineStamp(79), portfolio_notional: "100000", max_name_notional: "25000", lot_size: 1, max_age_seconds: 86400, target_volatility: "0.15", annualization: 252 };
  examples["workspace-research-cycle"] = { ...examples["workspace-allocate-screen"], operation: "cycle", run_ids: ["0".repeat(64)], lookback: 20, min_ic_observations: 5, buy_threshold: .25, sell_threshold: -.25, portfolio_notional: "100000", max_name_notional: "25000", vol_scale: "1", fill: true };
  delete examples["workspace-research-cycle"].screen_id;
  examples["workspace-strategy-replay"] = { operation: "replay", strategy_id: "0".repeat(64) };
  examples["workspace-size-replay"] = { operation: "size-replay", size_id: "0".repeat(64) };
  examples["workspace-cycle-replay"] = { operation: "cycle-replay", cycle_id: "0".repeat(64) };
  examples["workspace-research-watch"] = { operation: "watch", signal_ids: ["0".repeat(64), "1".repeat(64)], buy_threshold: .25, sell_threshold: -.25, now: "2024-01-02T00:00:00Z", portfolio_notional: "100000", max_name_notional: "25000", lot_size: 1, vol_scale: "1", options_run_ids: ["0".repeat(64)] };
  examples["workspace-watch-replay"] = { operation: "watch-replay", watch_id: "0".repeat(64) };
  examples["workspace-allocate-pairs"] = { ...examples["workspace-research-allocate"], pair_run_id: "0".repeat(64) };
  delete examples["workspace-allocate-pairs"].run_id;
  examples["workspace-options-overlay"] = { checkpoint_id: "0".repeat(64), options_run_id: "0".repeat(64), now: pipelineStamp(79), max_age_seconds: 3600, limits: { max_abs_delta_notional: 100000, max_abs_gamma_dollar: 100000, max_abs_vega_per_point: 100000, max_abs_scenario_loss: 100000 } };
  examples["workspace-options-overlay-replay"] = { overlay_id: "0".repeat(64) };
  const fundedNow = "2026-11-02T15:49:00Z";
  const fundedOption = "AUTHORED:SYN101:0:110:PUT";
  examples["workspace-funded-initialize"] = { operation: "initialize", now: fundedNow, initial_cash: "100000" };
  examples["workspace-funded-register"] = { operation: "register", now: fundedNow, instruments: [
    { instrument_id: "SYN101", kind: "EQUITY", currency: "USD", multiplier: 1 },
    { instrument_id: fundedOption, kind: "OPTION", underlying_id: "SYN101", currency: "USD", multiplier: 100, expiry: "2026-12-02T15:49:00Z", strike: "110", right: "PUT", style: "EUROPEAN", settlement: "CASH" }
  ] };
  examples["workspace-funded-reserve"] = { operation: "reserve", now: fundedNow, client_id: "authored-funded-put-1", instrument_id: fundedOption, side: "buy", quantity: "2", limit_price: "3", price_asof: fundedNow, fee_budget: "2" };
  examples["workspace-funded-dispatch"] = { operation: "dispatch", now: fundedNow };
  examples["workspace-funded-fill"] = { operation: "event", now: fundedNow, event: { event_id: "authored-funded-fill-1", client_id: "authored-funded-put-1", kind: "FILL", occurred_at: fundedNow, quantity: "1", price: "2.924497", fee: "0.10" } };
  examples["workspace-funded-cancel"] = { operation: "cancel", now: fundedNow, client_id: "authored-funded-put-1" };
  examples["workspace-funded-mark"] = { operation: "mark", now: fundedNow, marks: { SYN101: { price: "110", as_of: fundedNow }, [fundedOption]: { price: "2.924497", as_of: fundedNow } }, max_age_seconds: 3600 };
  examples["workspace-funded-settle"] = { operation: "settle", now: "2026-12-02T15:49:00Z", settlement_id: "authored-funded-expiry-1", instrument_id: fundedOption, spot: "100", observed_at: "2026-12-02T15:49:00Z", available_at: "2026-12-02T15:49:00Z", fee: "0" };
  examples["workspace-funded-state"] = { operation: "state", now: fundedNow };
  examples["workspace-funded-replay"] = { operation: "replay", now: fundedNow, snapshot_id: "0".repeat(64) };
  examples["workspace-economic-research"] = { kind: "CPU", run_ids: ["0".repeat(64)], now: pipelineStamp(79), annualization: 98280, cost_bps_grid: [0, 10], slippage_bps_grid: [0, 5], notional: 10000, participation_limit: .1, declared_trials: 4, bootstrap_repetitions: 100, block_size: 5, seed: 17 };
  examples["workspace-economic-replay"] = { report_id: "0".repeat(64) };
  const microStamp = milliseconds => new Date(Date.UTC(2026, 10, 2, 15, 48) + milliseconds).toISOString();
  const microEvents = Array.from({ length: 12 }, (_, index) => {
    const event_at = microStamp(index * 1000), trade_at = microStamp(index * 1000 + 500);
    const bid = 99.9 + .02 * index, ask = 100.1 + .02 * index;
    return [{ event_id: `authored-quote-${index}`, kind: "QUOTE", event_at, acquired_at: event_at, knowledge_at: event_at, sequence: index * 2, bid, ask, bid_size: 10 + index, ask_size: 16 + index },
      { event_id: `authored-print-${index}`, kind: "TRADE", event_at: trade_at, acquired_at: trade_at, knowledge_at: trade_at, sequence: index * 2 + 1, price: index % 2 ? bid : ask, size: 1 + index % 3 }];
  }).flat();
  examples["workspace-microstructure-admit"] = { source: "authored-sprint6-events", classification: "SYNTHETIC", feed: "AUTHORED_PRINT_AND_QUOTE_FIXTURE", instrument_id: "SYN101", currency: "USD", events: microEvents };
  examples["workspace-microstructure-research"] = { source: "authored-sprint6-events", classification: "SYNTHETIC", feed: "AUTHORED_PRINT_AND_QUOTE_FIXTURE", instrument_id: "SYN101", start: microStamp(0), end: microStamp(11500), as_of: microStamp(11500), max_quote_age_ns: 5000000000, lag_ns: 0, bucket_volume: 5, window_buckets: 2, participation_rate: .1, proposed_quantity: 1 };
  examples["workspace-microstructure-replay"] = { report_id: "0".repeat(64) };
  examples["workspace-funded-risk"] = { now: fundedNow, max_age_seconds: 3600, options_run_ids: ["0".repeat(64)], marks: JSON.parse(JSON.stringify(examples["workspace-funded-mark"].marks)), limits: { max_gross_notional: 100000, max_name_notional: 50000, max_abs_delta_notional: 100000, max_abs_gamma_dollar: 1000000, max_abs_vega_per_point: 100000, max_abs_theta_per_day: 100000, max_scenario_loss: 50000, min_available_cash: 0 }, scenarios: [ { name: "authored-spot-down", spot_shocks: { SYN101: -.1 }, vol_shocks: { SYN101: .02 }, elapsed_days: 0 }, { name: "authored-spot-up", spot_shocks: { SYN101: .1 }, vol_shocks: { SYN101: 0 }, elapsed_days: 0 } ] };
  examples["workspace-funded-risk-replay"] = { report_id: "0".repeat(64) };
  examples["workspace-funded-factor-risk"] = { ...JSON.parse(JSON.stringify(examples["workspace-funded-risk"])), metadata: { SYN101: { sector: "Authored synthetic sector", factors: { market: 1 } } }, metadata_source_identity: "AUTHORED_NOT_OBSERVED_FACTOR_METADATA", metadata_available_at: fundedNow };
  Object.assign(examples["workspace-funded-factor-risk"].limits, { max_sector_gross_notional: 100000, max_abs_factor_exposure: { market: 100000 } });
  const lifecycleSplitAt = "2026-11-04T15:49:00Z", lifecyclePayAt = "2026-11-06T15:49:00Z";
  examples["workspace-lifecycle-configure"] = { operation: "configure", now: fundedNow, settlement_policy: "EXPLICIT_EQUITY_DATES" };
  examples["workspace-lifecycle-split"] = { operation: "admit", now: "2026-11-03T21:00:00Z", action: { action_id: "authored-split-SYN101", revision: 1, instrument_id: "SYN101", source: "AUTHORED_SYNTHETIC_ACTION_FIXTURE", kind: "SPLIT", known_at: "2026-11-03T21:00:00Z", effective_at: lifecycleSplitAt, record_at: lifecycleSplitAt, pay_at: lifecycleSplitAt, ratio_new: 2, ratio_old: 1 } };
  examples["workspace-lifecycle-dividend"] = { operation: "admit", now: lifecycleSplitAt, action: { action_id: "authored-dividend-SYN101", revision: 1, instrument_id: "SYN101", source: "AUTHORED_SYNTHETIC_ACTION_FIXTURE", kind: "CASH_DIVIDEND", known_at: lifecycleSplitAt, effective_at: "2026-11-05T15:49:00Z", record_at: "2026-11-05T15:49:00Z", pay_at: lifecyclePayAt, amount_per_share: "1" } };
  examples["workspace-lifecycle-accrue"] = { operation: "accrue", now: "2026-11-05T15:49:00Z", action_hash: "0".repeat(64) };
  examples["workspace-lifecycle-dividend-revise"] = { operation: "admit", now: "2026-11-05T16:00:00Z", action: { ...examples["workspace-lifecycle-dividend"].action, revision: 2, known_at: "2026-11-05T16:00:00Z", amount_per_share: "0.5" } };
  examples["workspace-lifecycle-dividend-cancel"] = { operation: "admit", now: "2026-11-05T16:30:00Z", action: { ...examples["workspace-lifecycle-dividend"].action, revision: 3, known_at: "2026-11-05T16:30:00Z", amount_per_share: "0", status: "CANCELLED" } };
  examples["workspace-lifecycle-apply"] = { operation: "apply", now: lifecycleSplitAt, action_hash: "0".repeat(64) };
  examples["workspace-lifecycle-state"] = { operation: "state", now: lifecyclePayAt };
  examples["workspace-lifecycle-replay"] = { operation: "replay", now: lifecyclePayAt, snapshot_id: "0".repeat(64) };
  examples["workspace-statement-admit"] = { source: "authored-statement-supplier", statement_id: "authored-equity-1", observed_at: fundedNow, available_at: fundedNow, currency: "USD", balances: { cash: "899.8", nav: "999.8", realized_gross_pnl: "0", unrealized_pnl: "0", fees: "0.2", reserved_cash: "0", available_cash: "899.8" }, positions: [{ instrument_id: "SYN101", kind: "EQUITY", currency: "USD", multiplier: 1, quantity: "10", cost_basis: "100", market_value: "100", unrealized_pnl: "0" }], orders: [{ client_id: "authored-statement-stock", instrument_id: "SYN101", side: "buy", quantity: "10", filled: "10", limit_price: "10", state: "FILLED", acknowledged: true, created_at: fundedNow, updated_at: fundedNow }], fills: [{ execution_id: "authored-statement-fill", client_id: "authored-statement-stock", instrument_id: "SYN101", side: "buy", quantity: "10", price: "10", fee: "0.2", occurred_at: fundedNow }], settlements: [], reservations: [] };
  examples["workspace-statement-reconcile"] = { statement_hash: "0".repeat(64), snapshot_id: "0".repeat(64), now: fundedNow };
  examples["workspace-statement-replay"] = { report_id: "0".repeat(64) };
  examples["workspace-observed-fetch"] = { operation: "fetch", provider: "ECB_REFERENCE_FX", currencies: ["USD", "GBP", "JPY"], start: "2025-01-01", end: "2025-12-31" };
  const referenceCsv = "Date,USD,GBP,JPY\n" + Array.from({ length: 100 }, (_, i) => `${new Date(Date.UTC(2025, 0, 1+i)).toISOString().slice(0, 10)},${[0, 1, 2].map(j => (1+j*10+.01*Math.sin(i/3+j)+i*.0001).toFixed(6)).join(",")}`).join("\n");
  examples["workspace-observed-admit"] = { operation: "admit", provider: "ECB_REFERENCE_FX", currencies: ["USD", "GBP", "JPY"], start: "2025-01-01", end: "2025-04-10", acquired_at: "2026-11-03T15:00:00Z", csv_text: referenceCsv };
  examples["workspace-observed-research"] = { operation: "research", snapshot_id: "0".repeat(64), model: "ridge", training_observations: 60, test_observations: 20, embargo: 1, seed: 17 };
  examples["workspace-observed-evaluate"] = { operation: "evaluate", report_id: "0".repeat(64), bootstrap_replications: 500, block_length: 5, seed: 17 };
  examples["workspace-observed-evaluation-replay"] = { operation: "evaluation-replay", evaluation_id: "0".repeat(64) };
  examples["workspace-equity-fetch"] = { operation: "fetch", provider: "UCI_DOW_JONES_WEEKLY_2011", symbols: ["AA", "IBM"], start: "2011-01-07", end: "2011-06-24" };
  const equityColumns = "quarter,stock,date,open,high,low,close,volume,percent_change_price,percent_change_volume_over_last_wk,previous_weeks_volume,next_weeks_open,next_weeks_close,percent_change_next_weeks_price,days_to_next_dividend,percent_return_next_dividend";
  const equityCsv = equityColumns + "\n" + [7, 14, 21].map(day => `1,AA,1/${day}/2011,$10.10,$12.00,$9.50,$11.25,10000,11.3861,,,$99.00,$101.00,2.0202,3,9`).join("\n");
  examples["workspace-equity-admit"] = { operation: "admit", provider: "UCI_DOW_JONES_WEEKLY_2011", symbols: ["AA"], start: "2011-01-07", end: "2011-01-21", acquired_at: "2026-11-03T19:00:00Z", csv_text: equityCsv };
  examples["workspace-equity-replay"] = { operation: "replay", snapshot_id: "0".repeat(64) };
  examples["workspace-observed-replay"] = { operation: "replay", report_id: "0".repeat(64) };
  function loadExample() {
    const kind = $("analytics-kind").value;
    const example = JSON.parse(JSON.stringify(examples[kind]));
    if (["workspace-financial-search", "workspace-financial-answer", "workspace-financial-latent", "workspace-financial-hybrid", "workspace-financial-facts", "workspace-financial-entities", "workspace-financial-digest", "workspace-financial-watch", "workspace-financial-watch-check", "workspace-financial-investigate"].includes(kind)) example.as_of = new Date().toISOString();
    if (["workspace-financial-statements", "workspace-financial-transcript"].includes(kind) && workflowReferences.financial_document_id) example.document_id = workflowReferences.financial_document_id;
    if (["workspace-financial-compare", "workspace-financial-mention-diff"].includes(kind) && workflowReferences.financial_base_document_id) Object.assign(example, { base_document_id: workflowReferences.financial_base_document_id, revised_document_id: workflowReferences.financial_document_id });
    if (kind === "workspace-financial-watch-check" && workflowReferences.financial_watch_id) example.watch_id = workflowReferences.financial_watch_id;
    if (kind === "workspace-financial-record-replay" && workflowReferences.financial_record_id) example.record_id = workflowReferences.financial_record_id;
    if (kind === "workspace-calculator-replay" && workflowReferences.calculator_record_id) example.record_id = workflowReferences.calculator_record_id;
    if (kind === "workspace-calculator-dataset" && workflowReferences.dataset_id) example.dataset_id = workflowReferences.dataset_id;
    if (kind === "workspace-calculator-covariance" && workflowReferences.calculator_record_id) example.record_id = workflowReferences.calculator_record_id;
    if (kind === "workspace-financial-replay" && workflowReferences.financial_snapshot_id) example.snapshot_id = workflowReferences.financial_snapshot_id;
    if (kind === "workspace-financial-verify") {
      if (workflowReferences.financial_snapshot_id) example.snapshot_id = workflowReferences.financial_snapshot_id;
      if (workflowReferences.financial_passage_id) example.passage_id = workflowReferences.financial_passage_id;
    }
    if (["workspace-lifecycle-apply", "workspace-lifecycle-accrue"].includes(kind) && workflowReferences.lifecycle_action_hash) example.action_hash = workflowReferences.lifecycle_action_hash;
    if (kind === "workspace-lifecycle-replay" && workflowReferences.funded_snapshot_id) example.snapshot_id = workflowReferences.funded_snapshot_id;
    if (kind === "workspace-statement-reconcile") { if (workflowReferences.statement_hash) example.statement_hash = workflowReferences.statement_hash; if (workflowReferences.funded_snapshot_id) example.snapshot_id = workflowReferences.funded_snapshot_id; }
    if (kind === "workspace-statement-replay" && workflowReferences.statement_report_id) example.report_id = workflowReferences.statement_report_id;
    if (kind === "workspace-observed-research" && workflowReferences.observed_snapshot_id) example.snapshot_id = workflowReferences.observed_snapshot_id;
    if (kind === "workspace-equity-replay" && workflowReferences.equity_snapshot_id) example.snapshot_id = workflowReferences.equity_snapshot_id;
    if (kind === "workspace-observed-evaluate" && workflowReferences.observed_report_id) example.report_id = workflowReferences.observed_report_id;
    if (kind === "workspace-observed-evaluation-replay" && workflowReferences.observed_evaluation_id) example.evaluation_id = workflowReferences.observed_evaluation_id;
    if (kind === "workspace-observed-replay" && workflowReferences.observed_report_id) example.report_id = workflowReferences.observed_report_id;

    if ((["workspace-research-model", "workspace-research-gpr", "workspace-research-rf", "workspace-research-xt", "workspace-research-svr", "workspace-research-knn", "workspace-research-bayes", "workspace-research-glm", "workspace-research-gam", "workspace-research-predict"].includes(kind) || isBoundedKind(kind)) && workflowReferences.dataset_id) example.dataset_id = workflowReferences.dataset_id;
    if (["workspace-research-predict", "workspace-research-allocate"].includes(kind) && workflowReferences.run_id) example.run_id = workflowReferences.run_id;
    if (kind === "workspace-pairs-research" && workflowReferences.dataset_id) example.dataset_id = workflowReferences.dataset_id;
    if (kind === "workspace-pairs-replay" && workflowReferences.pair_run_id) example.run_id = workflowReferences.pair_run_id;
    if (kind === "workspace-options-replay" && workflowReferences.options_run_id) example.run_id = workflowReferences.options_run_id;
    if (kind === "workspace-research-signals" && workflowReferences.run_id) example.run_ids = [workflowReferences.run_id];
    if (kind === "workspace-signals-replay" && workflowReferences.signal_id) example.signal_id = workflowReferences.signal_id;
    if (kind === "workspace-allocate-signals" && workflowReferences.signal_id) example.signal_id = workflowReferences.signal_id;
    if (kind === "workspace-allocate-screen" && workflowReferences.screen_id) example.screen_id = workflowReferences.screen_id;
    if (kind === "workspace-research-strategy" && workflowReferences.screen_id) example.screen_id = workflowReferences.screen_id;
    if (kind === "workspace-research-size" && workflowReferences.screen_id) example.screen_id = workflowReferences.screen_id;
    if (kind === "workspace-research-cycle" && workflowReferences.run_id) example.run_ids = [workflowReferences.run_id];
    if (kind === "workspace-research-enrich" && workflowReferences.screen_id) example.screen_id = workflowReferences.screen_id;
    if (kind === "workspace-research-enrich" && workflowReferences.financial_document_id) example.document_id = workflowReferences.financial_document_id;
    if (kind === "workspace-strategy-replay" && workflowReferences.strategy_id) example.strategy_id = workflowReferences.strategy_id;
    if (kind === "workspace-size-replay" && workflowReferences.size_id) example.size_id = workflowReferences.size_id;
    if (kind === "workspace-cycle-replay" && workflowReferences.cycle_id) example.cycle_id = workflowReferences.cycle_id;
    if (kind === "workspace-research-watch" && workflowReferences.signal_id && workflowReferences.signal_id_prev) example.signal_ids = [workflowReferences.signal_id_prev, workflowReferences.signal_id];
    if (kind === "workspace-research-watch" && workflowReferences.options_run_id) example.options_run_ids = [workflowReferences.options_run_id];
    if (kind === "workspace-watch-replay" && workflowReferences.watch_id) example.watch_id = workflowReferences.watch_id;
    if (kind === "workspace-research-screen" && workflowReferences.signal_id) example.signal_id = workflowReferences.signal_id;
    if (kind === "workspace-research-discover" && workflowReferences.run_id) example.run_ids = [workflowReferences.run_id];
    if (kind === "workspace-research-options" && workflowReferences.screen_id) example.screen_id = workflowReferences.screen_id;
    if (kind === "workspace-research-options" && workflowReferences.options_run_id) example.options_run_id = workflowReferences.options_run_id;
    if (kind === "workspace-research-options-scan" && workflowReferences.screen_id) example.screen_id = workflowReferences.screen_id;
    if (kind === "workspace-research-options-scan" && workflowReferences.options_run_id) example.options_run_ids = [workflowReferences.options_run_id];
    if (kind === "workspace-screen-replay" && workflowReferences.screen_id) example.screen_id = workflowReferences.screen_id;
    if (kind === "workspace-research-screen" && workflowReferences.screen_id) example.previous_screen_id = workflowReferences.screen_id;
    if (kind === "workspace-allocate-pairs" && workflowReferences.pair_run_id) example.pair_run_id = workflowReferences.pair_run_id;
    if (kind === "workspace-options-overlay" && workflowReferences.checkpoint_id) example.checkpoint_id = workflowReferences.checkpoint_id;
    if (kind === "workspace-options-overlay" && workflowReferences.options_run_id) example.options_run_id = workflowReferences.options_run_id;
    if (kind === "workspace-options-overlay-replay" && workflowReferences.overlay_id) example.overlay_id = workflowReferences.overlay_id;
    if (kind === "workspace-economic-research" && workflowReferences.run_id) example.run_ids = [workflowReferences.run_id];
    if (kind === "workspace-economic-replay" && workflowReferences.economic_report_id) example.report_id = workflowReferences.economic_report_id;
    if (kind === "workspace-microstructure-replay" && workflowReferences.microstructure_report_id) example.report_id = workflowReferences.microstructure_report_id;
    if (kind === "workspace-funded-replay" && workflowReferences.funded_snapshot_id) example.snapshot_id = workflowReferences.funded_snapshot_id;
    if (["workspace-funded-risk", "workspace-funded-factor-risk"].includes(kind) && workflowReferences.options_run_id) example.options_run_ids = [workflowReferences.options_run_id];
    if (kind === "workspace-funded-risk-replay" && workflowReferences.funded_risk_id) example.report_id = workflowReferences.funded_risk_id;
    if (kind === "workspace-research-prepare" && workflowReferences.allocation_id) example.allocation_id = workflowReferences.allocation_id;
    if (kind === "workspace-offline-fill" && workflowReferences.prepared_client_ids.length) example.client_id = workflowReferences.prepared_client_ids[0];
    $("analytics-input").value = JSON.stringify(example, null, 2);
    $("run-analysis").textContent = workspaceActions[kind] ? (kind === "workspace-simulate" ? "Run offline simulation" : kind === "workspace-research" ? "Run and save research" : "Admit and save locally") : "Run local calculation";
    if (kind === "workspace-train-sequence") $("run-analysis").textContent = "Train CPU research model";
    if (kind === "workspace-predict-sequence") $("run-analysis").textContent = "Run saved model inference";
    if (kind === "workspace-treasury") $("run-analysis").textContent = "Fetch public Treasury data";
    if (kind === "workspace-finbert") $("run-analysis").textContent = "Analyze with local licensed model";
    if (kind === "workspace-execution-engine") $("run-analysis").textContent = "Prepare offline intent";
    if (kind === "workspace-research-dataset") $("run-analysis").textContent = "Build persisted dataset";
    if (["workspace-research-model", "workspace-research-gpr", "workspace-research-rf", "workspace-research-xt", "workspace-research-svr", "workspace-research-knn", "workspace-research-bayes", "workspace-research-glm", "workspace-research-gam"].includes(kind) || isBoundedKind(kind)) $("run-analysis").textContent = "Run saved-data research";
    if (kind === "workspace-financial-verify") $("run-analysis").textContent = "Verify saved citation";
    if (kind === "workspace-financial-watch-check") $("run-analysis").textContent = "Check saved watch now";
    if (kind === "workspace-financial-record-replay") $("run-analysis").textContent = "Replay saved research record";
    if (kind === "workspace-financial-mention-diff") $("run-analysis").textContent = "Compare cited mentions";
    if (kind === "workspace-calculator-save") $("run-analysis").textContent = "Save calculation";
    if (kind === "workspace-calculator-replay") $("run-analysis").textContent = "Replay saved calculation";
    if (kind === "workspace-calculator-dataset") $("run-analysis").textContent = "Calculate from saved dataset";
    if (kind === "workspace-calculator-covariance") $("run-analysis").textContent = "Allocate from saved covariance";
    if (kind === "workspace-research-predict") $("run-analysis").textContent = "Replay saved model";
    if (kind === "workspace-research-allocate") $("run-analysis").textContent = "Construct offline targets";
    if (kind === "workspace-research-prepare") $("run-analysis").textContent = "Prepare offline target intents";
    if (kind === "workspace-portfolio-accounting") $("run-analysis").textContent = "Reconcile simulated account";
    if (kind === "workspace-offline-dispatch") $("run-analysis").textContent = "Acknowledge in offline counterparty";
    if (kind === "workspace-offline-fill") $("run-analysis").textContent = "Apply explicit simulated fill";
    if (kind === "workspace-schedule-twap") $("run-analysis").textContent = "Save offline TWAP schedule";
    if (kind === "workspace-advance-clock") $("run-analysis").textContent = "Advance explicit simulation clock";
    const sprint5Labels = { "workspace-options-overlay": "Evaluate hypothetical options overlay", "workspace-options-overlay-replay": "Replay saved options overlay", "workspace-pairs-research": "Run saved-data pair research", "workspace-pairs-replay": "Replay saved pair experiment", "workspace-options-admit": "Admit authored options fixture", "workspace-options-research": "Run historical options research", "workspace-options-replay": "Replay saved options research", "workspace-research-signals": "Score saved OOS model evidence", "workspace-signals-replay": "Replay saved research scores", "workspace-research-screen": "Label research buy/hold/sell", "workspace-research-discover": "Discover from saved model runs", "workspace-research-options": "Link options research candidates", "workspace-research-options-scan": "Scan options opportunities", "workspace-research-monitor": "Monitor successive research screens", "workspace-research-enrich": "Enrich screen with FI mentions", "workspace-screen-replay": "Replay research screen", "workspace-allocate-signals": "Construct targets from research scores", "workspace-allocate-screen": "Construct targets from research screen", "workspace-research-strategy": "Simulate offline strategy", "workspace-research-size": "Size research positions", "workspace-research-cycle": "Run research opportunity cycle", "workspace-strategy-replay": "Replay strategy simulation", "workspace-size-replay": "Replay research position sizes", "workspace-cycle-replay": "Replay research opportunity cycle", "workspace-research-watch": "Watch then size and scan", "workspace-watch-replay": "Replay research watch pass", "workspace-allocate-pairs": "Construct targets from eligible pair" };
    if (sprint5Labels[kind]) $("run-analysis").textContent = sprint5Labels[kind];
    const specialScopes = {
      "workspace-funded-initialize": "Initialize a separate SIMULATED USD cash account using the existing offline OMS and durable outbox classes. Cash is explicit and cannot be reset. No brokerage connection or real order is involved.",
      "workspace-funded-register": "Register explicit equity and European cash-settled option identities and multipliers for the SIMULATED account. Unsupported borrowing, exercise styles and liabilities must be rejected rather than assigned zero.",
      "workspace-funded-reserve": "Reserve premium cash and fee budgets for fully paid long positions before an offline intent is admitted. Sales may close owned inventory only; borrowing and new short liabilities are unsupported. Outstanding intentions retain reservations. A scoped check is not a global funding or production readiness claim.",
      "workspace-funded-dispatch": "Explicitly send durable intents only to the local SYNTHETIC counterparty. Acknowledgement is not a fill; no brokerage or paper order is submitted.",
      "workspace-funded-fill": "Apply an explicitly requested SIMULATED partial fill. Contract multiplier, premium cash flow and fees affect the durable account; this is distinct from a hypothetical research overlay.",
      "workspace-funded-cancel": "Explicitly cancel a remaining offline intention through the existing simulated lifecycle. Reservations release only under the supported confirmed transition.",
      "workspace-funded-mark": "Reconcile SIMULATED positions, exact cash, premium and fee flows using explicit fresh marks. Missing liabilities and unsupported financing remain unknown; no global funding certification is implied.",
      "workspace-funded-settle": "Apply the declared European CASH-settlement assumption using an explicit underlying observation at exact expiry. No physical delivery, American early exercise or observed broker assignment is inferred.",
      "workspace-funded-state": "Inspect the separate SIMULATED cash-account lifecycle and funding evidence. This does not fetch quotes, place orders or replace missing marks.",
      "workspace-funded-replay": "Verify and reload an immutable SIMULATED account snapshot. Historical replay does not reapprove a current portfolio or reset its capital.",
      "workspace-economic-research": "Evaluate saved admitted out-of-sample research under explicit cost, slippage, capacity and multiple-trial assumptions. Bootstrap/statistical diagnostics do not establish validated profitability, alpha or production suitability.",
      "workspace-economic-replay": "Replay the saved economic research report and its source/model identities. No missing or delisted security outcome is replaced with zero.",
      "workspace-funded-risk": "Evaluate the actual SIMULATED cash-account positions and outstanding reservations using saved option models, fresh exact marks and explicit risk budgets. Hypothetical positions in research runs are ignored. Missing executed-print liquidity remains UNKNOWN; declared model assumptions do not certify economic value.",
      "workspace-funded-risk-replay": "Replay a saved actual-account risk report by identity. Historical risk evidence does not reapprove a changed account, reset a kill switch or manufacture missing exposures.",
      "cost-evidence": "Decompose fills into fees, half-spread, slippage and measured post-trade price move against a causal arrival quote. ESTIMATED, SIMULATED and OBSERVED_HISTORICAL costs are never summed together; simulated fills are never promoted by matching prints; BROKER_RECONCILED is unavailable because no broker is connected.",
      "provenance": "Evaluate a lineage record and its ancestry for archive inspection, point-in-time research or funded allocation at a cutoff. Missing publication, revision, identity, currency or availability evidence yields UNKNOWN, never ELIGIBLE.",
      "workspace-microstructure-admit": "Admit explicitly authored SYNTHETIC quote and trade-print events with separate observation, acquisition and knowledge clocks. Quotes or displayed depth are not represented as executed liquidity.",
      "workspace-microstructure-research": "Compute causal quote/print diagnostics from admitted events: best-queue OFI, trade classification, Kyle regression and volume-bucket VPIN where supported. Capacity and impact estimates remain model assumptions, not executable liquidity.",
      "workspace-microstructure-replay": "Verify a saved microstructure report and source-event identities. Later quote corrections must not rewrite earlier trade-classification decisions.",
      "workspace-options-overlay": "Combine a saved reconciled SIMULATED equity checkpoint with hypothetical option positions from a saved options run. Held equity must match the same underlying, exact spot mark and observation clock. Explicit USD risk caps apply. Options financing remains UNKNOWN; this does not fund contracts, change OMS state, approve trading or submit an order.",
      "workspace-options-overlay-replay": "Verify and replay the saved hypothetical overlay by identity. Replayed modeled exposures do not establish funded portfolio NAV, broker positions or trading approval.",
      "workspace-pairs-research": "Use a saved admitted dataset to test chronological pair eligibility, cointegration, ADF, causal hedge states, and modeled costs. Rejected pairs remain rejected. Diagnostic significance is not economic validation or approval to trade.",
      "workspace-pairs-replay": "Reload and verify the saved pair experiment by its hash. Replay does not obtain new provider data or place orders.",
      "workspace-options-admit": "Explicitly admit an authored SYNTHETIC European BSM chain fixture with quotes, acquisition/knowledge time, spot provenance and contract identities. The fixture is not observed OPRA data and grants no vendor entitlement.",
      "workspace-options-research": "Select an admitted chain at its historical knowledge cutoff. IV, Greeks, surfaces and scenarios are European ACT/365F model estimates under the supplied rate and dividend assumptions. PCR and VTS carry explicit definitions; unavailable CSM remains unavailable.",
      "workspace-options-replay": "Verify and reload a saved options experiment. Original quote identities, modeled assumptions and simulated Greek exposures are retained after restart.",
      "workspace-research-signals": "Combine saved out-of-sample model evidence using only outcomes known at the decision time. IC diagnostics are research estimates; confidence is not a calibrated probability or trading approval.",
      "workspace-signals-replay": "Reload a hash-verified saved research score. Research scores do not automatically create portfolio targets or approve orders.",
            "workspace-research-screen": "Label RESEARCH_BUY / RESEARCH_HOLD / RESEARCH_SELL from a saved ensemble score with caller thresholds. Timing compares two saved screens. No order or trading approval.",
      "workspace-research-discover": "Build the IC-weighted ensemble from saved model runs and label research buy/hold/sell in one step. Still research-only; no order or trading approval.",
      "workspace-research-options": "Join a saved equity research screen to a saved European options run. CALL/PUT candidates follow RESEARCH_BUY/SELL only. No order or trading approval.",
      "workspace-research-options-scan": "Scan multiple saved options runs against one equity research screen and list CALL/PUT research candidates. No order or trading approval.",
      "workspace-research-monitor": "Chain saved signals into successive research screens and summarize entry/exit timing. Not a live feed. No order or trading approval.",
      "workspace-research-enrich": "Match Financial Intelligence entity mentions onto a saved research screen by exact lexical overlap. Mentions stay unlinked to issuers. No order or trading approval.",
      "workspace-screen-replay": "Reload a saved research screen and recompute its labels from the pinned signal. Any difference is an integrity failure.",
      "workspace-allocate-signals": "Construct explicit constrained offline targets from a saved research score artifact. Existing exposure, cash, transaction costs and asynchronous fill limits still apply.",
      "workspace-allocate-screen": "Construct offline targets from nonzero research_weight values on a saved buy/hold/sell screen. Still simulated and unapproved.",
      "workspace-research-strategy": "Allocate from a research screen, prepare offline intents, and optionally fill at reference prices in the offline OMS. Broker stays off; trading stays unapproved.",
      "workspace-research-size": "Convert screen research weights into capped lot sizes using admitted marks. Optional target_volatility derives vol_scale from the saved return panel. No OMS prepare or broker order.",
      "workspace-research-cycle": "One research cycle: discover from saved model runs, size positions, then allocate/prepare/optionally fill offline. Broker stays off; trading stays unapproved.",
      "workspace-strategy-replay": "Reload a saved offline strategy simulation. No new orders are created.",
      "workspace-size-replay": "Reload saved research position sizes by identity. Replay does not approve trading.",
      "workspace-cycle-replay": "Reload a saved research opportunity cycle by identity. Replay does not approve trading.",
      "workspace-research-watch": "Explicit watch pass: monitor successive saved signals, size the latest screen, optionally scan options. Not a live subscription; broker stays off.",
      "workspace-watch-replay": "Reload a saved research watch pass by identity. Replay does not approve trading.",
      "workspace-allocate-pairs": "Construct explicit constrained offline targets from an eligible saved pair proposal. Missing, rejected or inactive proposals return unavailable rather than manufacturing a trade.",
      "workspace-depth": "Persist an explicitly authored SYNTHETIC multi-level book. No depth is inferred from bars or connected to a provider. After submission, choose Admitted market depth and instrument SYN101 with its explicit replay cutoff.",
      "workspace-treasury": "Submitting this action fetches historical par yields from the fixed official Treasury public endpoint and persists original XML under CC0-1.0 with attribution. No credentials or broker connection. Select provider evidence in Markets after acquisition. Acquisition freshness does not mean current market quotes.",
      "workspace-ingest": "Offline authored SYNTHETIC IEX protocol fixture only. No Alpaca connection, credentials or provider-observed market data. IEX is not SIP.",
      "workspace-universe": "Persist authored SYNTHETIC universe membership with effective and knowledge timestamps. Caller identity claims are not provider verification.",
      "workspace-train-sequence": "Train and persist a real CPU BiLSTM on authored SYNTHETIC inputs. Model output is research only and is not an approved trading signal.",
      "workspace-predict-sequence": "Copy a saved model_id from training into this input. The all-zero placeholder returns UNAVAILABLE; it does not identify a trained model. Authored SYNTHETIC features remain research only.",
      "workspace-finbert": "Requires an operator-configured, hash-verified local model. Selected ProsusAI FinBERT weights are CC-BY-NC-SA-4.0 noncommercial research only. Missing assets return an explicit dependency error; no download or commercial approval is implied.",
      "workspace-execution-engine": "Prepare an authored SYNTHETIC offline order intent only. This does not dispatch it. To rehearse dispatch, explicitly edit operation and fields under the offline execution contract. No brokerage connection or real orders."
    };
    Object.assign(specialScopes, {
      "workspace-pipeline-bars": "Explicitly admit 160 authored SYNTHETIC minute bars for two instruments. The example is not provider-observed data. This setup action does not run a model or prepare an order.",
      "workspace-pipeline-universe": "Save the authored SYNTHETIC universe known before all fixture decisions. Use this with the pipeline bars setup, then build a persisted dataset.",
      "workspace-research-dataset": "Build a research dataset from persisted source observations and decision-time universe/revisions. No rows are generated by this action. Missing or unavailable knowledge fails explicitly; a later correction cannot rewrite an earlier feature decision.",
      "workspace-financial-reference": "Explicitly ingest the pinned CC0 Treasury XML document already packaged with GIG. Original source bytes and acquisition provenance are retained. No network, SEC identity, price feed, embeddings or language model is activated.",
      "workspace-financial-search": "Bounded financial-document BM25 research with optional authored terminology expansion. Only documents known by the explicit cutoff enter corpus statistics. Results retain exact source-byte citations. Relevance is not financial confidence; semantic retrieval is UNAVAILABLE.",
      "workspace-financial-answer": "Extractive source passage only, with original XML or text citations. This is not generative financial analysis. No match returns an unavailable answer; historical publication times remain UNKNOWN.",
      "workspace-financial-latent": "Fit a bounded corpus-local LSA retrieval model on visible permitted documents, then quote a cited passage. This is statistical latent retrieval, not pretrained language understanding. Dates and pure numbers are excluded from LSA features; near-identical yield records may tie. No generative model or trading approval.",
      "workspace-financial-hybrid": "Combine BM25 lexical ranks with corpus-local LSA cosine ranks using explicit reciprocal-rank fusion. Model and corpus identities are saved for replay. Scores are relevance diagnostics, not confidence. Parsed financial mentions retain citations; issuer truth and event occurrence remain UNKNOWN.",
      "workspace-financial-facts": "Extract bounded Decimal-valued financial mentions, units, tenors and entity/event cues from cited passages. Mention parsing does not verify an issuer claim or establish an event. Original bytes and knowledge time remain authoritative; no formula or document instruction executes.",
      "workspace-financial-entity-protocol": "Explicitly admit a GIG-authored SYNTHETIC entity syntax protocol. Source symbols are text examples, with issuer identity, exchange membership and historical eligibility UNKNOWN. This adds no market observations and cannot approve a trade.",
      "workspace-financial-html": "Explicitly admit one GIG-authored SYNTHETIC HTML document. Scripts, styles and markup never execute; each visible block becomes a passage cited to its original byte range, and table rows keep their cells. Nothing is fetched from a network and no issuer filing is acquired.",
      "workspace-financial-statements": "Read line items by period from the cited table rows of one admitted document, with exact Decimal values, stated scale, margins, like-for-like growth, free cash flow and accounting-identity checks. Values are parsed mentions, not verified issuer facts, audited figures or XBRL.",
      "workspace-financial-digest": "Quote the most relevant cited passages from every document known by the cutoff and compare statement values across documents. The digest is extractive; no language model writes or interprets text, and disagreement between documents is reported, never resolved.",
      "workspace-financial-watch": "Save a search as a watch baseline. Nothing runs in the background: the watch changes only when you explicitly check it, and no provider connection, schedule or notification is created.",
      "workspace-financial-watch-check": "Rerun a saved watch at an explicit later cutoff and list new or dropped cited passages and new or superseded documents against its baseline. The result is a local record only; no alert is sent and no trade is approved.",
      "workspace-financial-investigate": "Run up to eight explicit questions against documents known by the cutoff and file one immutable report of extractive cited findings, unanswered questions and statement checks. This is a single explicit request, not an autonomous agent; document text is never executed as an instruction.",
      "workspace-financial-call": "Explicitly admit one GIG-authored SYNTHETIC earnings-call transcript as exact text lines. It is an authored fixture, not an issuer's call; nothing is fetched from a network.",
      "workspace-financial-transcript": "Read speaker turns, prepared remarks, question-and-answer exchanges and forward-looking sentences from one admitted document, each cited to its source bytes. Roles come only from speaker labels and cues are fixed word lists; nothing is interpreted, scored for sentiment or verified.",
      "workspace-financial-compare": "Compare two admitted documents passage by passage: added, removed, reworded and number-only changes with citations, plus statement values that differ. Differences are between texts, not verified restatements; a moved passage shows as removed and added.",
      "workspace-financial-mention-diff": "Compare parsed percent, money, event and entity mentions in the first 20 passages of two admitted documents. A difference is a mention present on one side, not a verified restatement. v1 is the default parser; v2 is only used when nlp_profile is v2. Symbols stay source text.",
      "workspace-financial-record-replay": "Recompute a saved statement, digest, watch, watch check, investigation, transcript, comparison or mention difference from its pinned documents and saved queries. Any difference is reported as an integrity failure; nothing new is fitted or fetched.",
      "workspace-financial-verify": "Read-only research citation check. First select a successful saved query and passage. Replays the pinned query and checks original source bytes; it writes no new state and performs no fit. Hash integrity does not establish financial truth, source authenticity, rights, freshness, signatures or trading approval.",
      "workspace-financial-entities": "Use the explicit version-2 financial mention profile on cited source text. Complete qualified symbols and share classes remain unverified mentions; unsupported whole candidates are never partially linked. Saved snapshots pin their profile; published version-1 results retain their original rules.",
      "workspace-financial-replay": "Replay a saved financial search by hash against its original pinned corpus and cutoff. New document revisions cannot change old results. No monitoring or autonomous provider connection is running.",
      "workspace-research-rf": "Bounded Random Forest research on admitted saved data. Actual CPU training, 16 trees / depth 4, 256 training rows and 16 total fits. Tree disagreement is not calibrated confidence. No allocation or trading approval.",
      "workspace-research-xt": "Bounded ExtraTrees research on admitted saved data. One family with the Random Forest variant; 16 trees / depth 4 and one CPU thread. Tree disagreement is not calibrated confidence. No allocation or trading approval.",
      "workspace-research-gam": "Bounded CPU additive spline research with fixed cubic bases and ridge regularization. Knots and scaling use training data only. Linear boundary extrapolation; new variation in constant training features is unsupported. At most 256 training rows, 16 raw features, 96 basis columns and 16 fits. Uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.",
      "workspace-research-glm": "Bounded CPU binomial logit research on saved data: 20–256 training rows, at least 3 per class, 16 fits, 512 iterations, one CPU thread. Estimates uncalibrated directional probabilities, not expected returns. Zero/nonpositive returns are class 0. Parameter uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.",
      "workspace-research-bayes": "Bounded CPU empirical-Bayes ridge research on admitted saved data: 20–256 training rows, 16 features, 16 fits, at most 512 iterations, one CPU thread. Conditional Gaussian intervals are NOT OOS calibrated; learned plug-in precisions omit hyperparameter uncertainty. ARD is unavailable. Allocation, promotion and trading remain unapproved.",
      "workspace-research-knn": "Bounded CPU nearest-neighbor research on admitted saved data. Fitting retains 20–256 labelled rows and training-only scaling; five neighbors use deterministic distance/time-order ties. At most 16 fits, one CPU thread. Predictive uncertainty is UNAVAILABLE. Allocation, promotion and trading remain unapproved.",
      "workspace-research-svr": "Fit bounded CPU RBF support-vector regression to admitted saved data for research. Train-only feature and target scaling, purged labels, at most 128 training rows and 16 fits. Predictive uncertainty is UNAVAILABLE, not calibrated confidence. Research only; allocation and trading remain rejected.",
      "workspace-research-gpr": "Fixed-RBF Gaussian process research on a saved admitted dataset. CPU only; at most 128 training rows and 16 total fits. Train-only scaling and purged labels; posterior spread is assumption-dependent, not calibrated confidence. No allocation or trading approval.",
      "workspace-research-model": "Train real CPU research on a saved admitted dataset with purged chronological folds. Available choices include ridge, LightGBM ranking, BiLSTM, HMM, GARCH and Kalman. Source classification and uncertainty remain explicit; research results are not approved signals.",
      "workspace-research-predict": "Reload a saved model/run and admitted dataset by hash. IDs come from this session's last successful research actions, or can be edited explicitly. Zero placeholders are unavailable, not trained models.",
      "workspace-research-allocate": "Construct modeled offline targets from saved research scores, admitted marks and explicit limits. Worst-case asynchronous fills must fit the policy. This does not prepare, dispatch or fill an order.",
      "workspace-research-prepare": "Prepare durable intents for the selected allocation in the offline OMS. Preparation does not acknowledge or fill them. No broker connection, paper order or live order is available.",
      "workspace-portfolio-accounting": "Reconcile actual stored SIMULATED fills, exact cash and positions against explicitly supplied marks. Missing/stale values remain explicit. Risk estimates and simulated NAV are not broker-observed account values.",
      "workspace-offline-dispatch": "Send durable intents only to the local SYNTHETIC counterparty. An acknowledgement is not a fill. No network, account or broker order is involved.",
      "workspace-offline-fill": "Record one explicitly requested SIMULATED partial fill. Check the selected client ID, price, quantity and fee before submission. No broker activity occurs.",
      "workspace-schedule-twap": "Persist an exact offline TWAP plan under an explicit simulation clock. Scheduling never dispatches or fills a child order.",
      "workspace-advance-clock": "Advance the durable offline clock with explicitly fresh reference marks. Due children become idempotent intents; this does not dispatch or fill them. Backward time and stale marks are rejected."
    });
    Object.assign(specialScopes, {
      "workspace-funded-factor-risk": "Apply explicit sector and factor-exposure budgets to actual simulated positions. The example uses AUTHORED assumed sector/beta metadata, not observed issuer facts or calibrated factor loadings. Missing or stale metadata cannot become zero exposure. A drawdown budget requires separately adequate NAV history.",
      "workspace-lifecycle-configure": "Configure supported trade-date versus settlement-date cash policy only on a flat account before orders. Legacy immediate settlement is explicit; no borrowed cash is introduced.",
      "workspace-lifecycle-split": "Admit an authored SYNTHETIC split revision with separate source knowledge and effective clocks. Admission does not apply a position adjustment or fabricate an observed corporate action.",
      "workspace-lifecycle-dividend": "Admit an authored SYNTHETIC dividend revision. Record-date entitlement and pay-date cash remain explicit; later purchases cannot acquire an earlier entitlement.",
      "workspace-lifecycle-accrue": "Recognize the selected known dividend entitlement under its ex/record-date evidence. A receivable contributes to modeled NAV and income, but never to spendable cash. Ambiguous entitlement remains unknown.",
      "workspace-lifecycle-dividend-revise": "Admit an authored correction to the same dividend with a later source-knowledge clock. The example is revision 2 after the original revision 1; recognize its compensating receivable and income effect without a new cash credit.",
      "workspace-lifecycle-dividend-cancel": "Admit an explicit CANCELLED dividend revision at a later knowledge clock with zero rate. The example is revision 3 after revision 2. Cancellation must reverse the supported obligation exactly; it cannot erase earlier saved evidence or finance unsupported cash debits.",
      "workspace-lifecycle-apply": "Apply only the selected hash-verified lifecycle revision under the declared clock. Raw OMS fills remain preserved; unsupported fractional or working-order states reject.",
      "workspace-lifecycle-state": "Inspect the persisted lifecycle and supported cash-settlement state. Unsettled proceeds and unresolved obligations are not spendable cash.",
      "workspace-lifecycle-replay": "Reload an immutable lifecycle/account snapshot. Historical replay does not reapprove today's portfolio or rewrite source revisions.",
      "workspace-statement-admit": "Admit an independently authored offline statement fixture. The literal example assumes initial cash 1000, ten shares at 10 and fee 0.2. It is not generated from engine output or claimed to be a broker statement.",
      "workspace-statement-reconcile": "Compare the selected immutable statement with a funded snapshot. MATCH means fixture agreement only; DISCREPANCY and UNKNOWN never establish external reconciliation.",
      "workspace-statement-replay": "Verify and replay saved independent-statement discrepancies. A completed comparison is not evidence that unresolved cash, positions or lifecycle obligations agree.",
      "workspace-observed-fetch": "This explicit action fetches permitted official ECB daily reference-rate data. Observation precision is a DATE, not a fabricated intraday quote; volume, executable liquidity and historical publication availability remain UNKNOWN. Rates are not funded-account instruments.",
      "workspace-observed-admit": "Admit an authored SYNTHETIC CSV protocol fixture. These values are not ECB observations, OHLC bars or actual traded volume. Public observed-data acquisition is a separate explicit action.",
      "workspace-observed-research": "Run a local model on selected reference-rate evidence. Chronological reconstruction is performed after acquisition; it does not prove inputs were knowable at historical trading decisions or establish alpha.",
      "workspace-observed-evaluate": "Evaluate the saved model in its own units: return estimates against return baselines, LightGBM ranks with ranking metrics, and GARCH variance forecasts with variance scoring/calibration. Training-only references and missing folds remain explicit. Retrospective uncertainty does not establish profitable alpha or calibrated trading confidence.",
      "workspace-observed-evaluation-replay": "Verify the saved baseline comparison against its original report, model bytes and evaluation settings. Later experiments cannot alter this evidence.",
      "workspace-equity-fetch": "Explicitly fetch the permitted CC BY 4.0 UCI weekly equity archive. This is historical source evidence with source-label identities, not verified exchange quotes or a point-in-time security universe. Future labels and dividend predictors are quarantined.",
      "workspace-equity-admit": "Admit an authored SYNTHETIC weekly archive protocol fixture. The prices and volume are fixture observations; no source authenticity, permanent instrument identity, currency or historical vintage is inferred.",
      "workspace-equity-replay": "Verify and reload the saved weekly evidence with original acquisition and source hashes. Archive availability does not establish current quotes, corporate-action correctness or funded eligibility.",
      "workspace-observed-replay": "Reload the saved reference-rate model report with source identities and original daily units. Unknown original publication clocks remain unknown."
    });
    for (const [key, , scope] of boundedFamilies) specialScopes[boundedKind(key)] = scope;
    for (const [key, , scope] of quantAnalytics) specialScopes[key] = scope;
    specialScopes["workspace-calculator-save"] = "Save one registry calculation under the hash of its kind and inputs. Replay recomputes it. No order, allocation or trading approval.";
    specialScopes["workspace-calculator-replay"] = "Recompute a saved calculation from its pinned inputs. Any difference is an integrity failure. Nothing new is fitted or fetched.";
    specialScopes["workspace-calculator-dataset"] = "Copy one instrument series, or an aligned return panel for Ledoit-Wolf or minimum-CVaR, from a saved dataset. Missing values are not filled. Replay uses the pinned numbers. No order or trading approval.";
    $("workbench-scope").textContent = specialScopes[kind] || (workspaceActions[kind] ? "This explicit action writes caller-supplied evidence or simulated research state to this local service. Submitted source claims are not vendor-verified. No broker orders can be sent." : "This calculation returns model research estimates. It does not admit market observations or approve trading.");
  }
  async function runAnalysis() {
    const output = $("analytics-output");
    $("analysis-summary").classList.add("hidden");
    const sourceText = $("analytics-input").value;
    let inputs;
    try { inputs = JSON.parse(sourceText); if (!isObject(inputs)) throw new Error(); }
    catch { output.textContent = "Input must be a valid JSON object. No request was sent."; return; }
    const button = $("run-analysis");
    button.disabled = true;
    output.textContent = "Computing with the local backend…";
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 30000);
    try {
      const kind = $("analytics-kind").value;
      const action = workspaceActions[kind];
      const endpoint = action ? `/api/v1/workspace/${action}` : `/api/v1/analytics/${encodeURIComponent(kind)}`;
      const response = await fetch(endpoint, { method: "POST", headers: { "Content-Type": "application/json", Accept: "application/json" }, body: sourceText, signal: controller.signal, cache: "no-store" });
      const result = await response.json();
      if (!validContract(result) || (!response.ok && !["ERROR", "UNAVAILABLE"].includes(result.state))) throw new Error("Unsupported response contract.");
      renderAnalysisSummary(result);
      if (response.ok && result.state === "AVAILABLE" && isObject(result.data)) {
        const identity = value => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
        if (action === "funded-lifecycle" && identity(result.data.action_hash)) workflowReferences.lifecycle_action_hash = result.data.action_hash;
        if (action === "funded-lifecycle" && identity(result.data.snapshot_id)) workflowReferences.funded_snapshot_id = result.data.snapshot_id;
        if (action === "statement-admit" && identity(result.data.statement_hash)) workflowReferences.statement_hash = result.data.statement_hash;
        if (action === "statement-reconcile" && identity(result.data.report_id)) workflowReferences.statement_report_id = result.data.report_id;
        if (action === "observed-reference" && identity(result.data.snapshot_id) && ["fetch", "admit"].includes(inputs.operation)) workflowReferences.observed_snapshot_id = result.data.snapshot_id;
        if (action === "equity-reference" && identity(result.data.snapshot_id)) workflowReferences.equity_snapshot_id = result.data.snapshot_id;
        if (action === "observed-reference" && identity(result.data.evaluation_id)) workflowReferences.observed_evaluation_id = result.data.evaluation_id;
        if (action === "observed-reference" && identity(result.data.report_id)) workflowReferences.observed_report_id = result.data.report_id;

        if (action === "research-dataset" && identity(result.data.dataset_id)) { workflowReferences.dataset_id = result.data.dataset_id; workflowReferences.run_id = null; workflowReferences.pair_run_id = null; workflowReferences.signal_id = null; workflowReferences.screen_id = null; workflowReferences.allocation_id = null; workflowReferences.prepared_client_ids = []; }
        if (action === "research-model" && identity(result.data.run_id)) { if (identity(result.data.dataset_id)) workflowReferences.dataset_id = result.data.dataset_id; workflowReferences.run_id = result.data.run_id; workflowReferences.allocation_id = null; workflowReferences.prepared_client_ids = []; }
        if (action === "pairs-research" && identity(result.data.run_id)) workflowReferences.pair_run_id = result.data.run_id;
        if (action === "options-admit" && identity(result.data.snapshot_id)) workflowReferences.options_snapshot_id = result.data.snapshot_id;
        if (action === "options-research" && identity(result.data.run_id)) { workflowReferences.options_run_id = result.data.run_id; workflowReferences.overlay_id = null; }
        if (action === "research-signals" && identity(result.data.signal_id)) workflowReferences.signal_id = result.data.signal_id;
        if ((action === "research-screen" || action === "screen-replay") && identity(result.data.screen_id)) workflowReferences.screen_id = result.data.screen_id;
        if (action === "research-strategy" && identity(result.data.strategy_id)) workflowReferences.strategy_id = result.data.strategy_id;
        if (action === "research-strategy" && identity(result.data.size_id)) workflowReferences.size_id = result.data.size_id;
        if (action === "research-strategy" && identity(result.data.cycle_id)) workflowReferences.cycle_id = result.data.cycle_id;
        if (action === "research-strategy" && identity(result.data.watch_id)) workflowReferences.watch_id = result.data.watch_id;
        if (action === "research-strategy" && identity(result.data.screen_id)) workflowReferences.screen_id = result.data.screen_id;
        if (action === "portfolio-accounting" && identity(result.data.checkpoint_id)) { workflowReferences.checkpoint_id = result.data.checkpoint_id; workflowReferences.overlay_id = null; }
        if (action === "options-overlay" && identity(result.data.overlay_id)) workflowReferences.overlay_id = result.data.overlay_id;
        if (action === "funded-account" && identity(result.data.snapshot_id)) workflowReferences.funded_snapshot_id = result.data.snapshot_id;
        if (action === "economic-research" && identity(result.data.report_id)) workflowReferences.economic_report_id = result.data.report_id;
        if (action === "microstructure-admit" && identity(result.data.snapshot_id)) workflowReferences.microstructure_snapshot_id = result.data.snapshot_id;
        if (action === "microstructure-research" && identity(result.data.report_id)) workflowReferences.microstructure_report_id = result.data.report_id;
        if (action === "funded-risk" && identity(result.data.report_id)) workflowReferences.funded_risk_id = result.data.report_id;
        if (action === "financial-intelligence" && identity(result.data.snapshot_id)) {
          if (workflowReferences.financial_snapshot_id !== result.data.snapshot_id) { workflowReferences.financial_passage_id = null; workflowReferences.financial_verification_id = null; }
          workflowReferences.financial_snapshot_id = result.data.snapshot_id;
          if (Array.isArray(result.data.rows)) workflowReferences.financial_passage_id = identity(result.data.rows[0]?.passage_id) ? result.data.rows[0].passage_id : null;
        }
        if (action === "financial-intelligence" && identity(result.data.passage_id)) workflowReferences.financial_passage_id = result.data.passage_id;
        if (action === "financial-intelligence" && identity(result.data.verification_id)) workflowReferences.financial_verification_id = result.data.verification_id;
        if (action === "financial-intelligence" && identity(result.data.document_id) && isObject(result.data.metadata) && result.data.document_id !== workflowReferences.financial_document_id) {
          workflowReferences.financial_base_document_id = workflowReferences.financial_document_id;
          workflowReferences.financial_document_id = result.data.document_id;
        }
        if (action === "calculator-journal" && identity(result.data.record_id)) workflowReferences.calculator_record_id = result.data.record_id;
        if (action === "financial-intelligence" && identity(result.data.record_id)) {
          workflowReferences.financial_record_id = result.data.record_id;
          if (result.data.operation === "watch") workflowReferences.financial_watch_id = result.data.record_id;
        }
        rememberExperiment(action, result);
        if (action === "research-allocate" && identity(result.data.allocation_id)) { workflowReferences.allocation_id = result.data.allocation_id; workflowReferences.prepared_client_ids = []; }
        if (action === "research-prepare" && Array.isArray(result.data.prepared_client_ids)) workflowReferences.prepared_client_ids = result.data.prepared_client_ids.filter(value => typeof value === "string" && value.length <= 128);
        if (action === "depth" && typeof result.data.instrument_id === "string") $("depth-instrument").value = result.data.instrument_id;
        $("workflow-identities").textContent = ["calculator_record_id", "financial_snapshot_id", "financial_passage_id", "financial_verification_id", "financial_document_id", "financial_base_document_id", "financial_record_id", "financial_watch_id", "equity_snapshot_id", "dataset_id", "run_id", "pair_run_id", "options_snapshot_id", "options_run_id", "signal_id", "screen_id", "strategy_id", "allocation_id", "checkpoint_id", "overlay_id", "funded_snapshot_id", "funded_risk_id", "economic_report_id", "microstructure_snapshot_id", "microstructure_report_id", "lifecycle_action_hash", "statement_hash", "statement_report_id", "observed_snapshot_id", "observed_report_id", "observed_evaluation_id"].map(key => `${human(key)}: ${workflowReferences[key] || "No successful result selected"}`).join("\n");
      }
      output.textContent = `${response.ok ? "" : `HTTP ${response.status}\n`}${result.state}${action ? " · Local workspace action; no broker orders" : " · Research calculation"}\n${JSON.stringify(result, null, 2)}`;
      if (action && response.ok && result.state === "AVAILABLE" && state.mode === "local") await loadSection();
    } catch { $("analysis-summary").classList.add("hidden"); output.textContent = "Analysis response could not be loaded. Check the local service, supported input contract, and retry."; }
    finally { clearTimeout(timer); button.disabled = false; }
  }
  function renderAnalysisSummary(result) {
    const data = result.data;
    if (!["AVAILABLE", "STALE"].includes(result.state) || !isObject(data)) return;
    const entries = [];
    if (typeof data.verification_id === "string") {
      entries.push(["Verification scope", "Immutable byte provenance only"], ["Verification ID", data.verification_id], ["Saved query", data.snapshot_id], ["Passage", data.passage_id], ["Source bytes SHA-256", data.source_bytes_sha256], ["Quoted bytes SHA-256", data.quoted_bytes_sha256], ["Original source quote", data.source_excerpt], ["Normalized passage", data.normalized_passage], ["Text / byte relationship", data.direct_text_byte_match ? "Exact UTF-8 source text" : "Rendered text; original source bytes shown separately"], ["Original-byte range", `${data.source_start}–${data.source_end}`], ["Source license", data.source_license], ["Financial truth / issuer identity", `${data.content_truth_state} / ${data.issuer_identity_state}`], ["Signature / freshness", `${data.signature_state} / ${data.freshness_state}`], ["Persistence", "Read only · no new state written"]);
    }
    if (typeof data.record_id === "string" && typeof data.operation === "string") {
      entries.push(["Research record", data.record_id], ["Source scope", data.classification], ["Generative model", data.generative_state || "UNAVAILABLE"]);
      if (typeof data.replay_state === "string") entries.push(["Replay", data.replay_state]);
      if (data.operation === "statements") {
        entries.push(["Statement state", data.statement_state], ["Periods", data.periods.join(", ") || "None recognized"], ["Value kind", "Parsed cited mentions · not verified issuer facts"]);
        entries.push(["Cited line items", data.facts.slice(0, 24).map(fact => `${fact.item} ${fact.period}: ${fact.decimal_value} ${fact.unit} (quoted ${fact.quoted})`).join("\n") || "No recognized statement rows"]);
        if (data.derived.length) entries.push(["Derived arithmetic", data.derived.slice(0, 24).map(row => `${row.metric} ${row.period}: ${row.value ?? "UNAVAILABLE"}`).join("\n")]);
        if (data.consistency_checks.length) entries.push(["Accounting identity checks", data.consistency_checks.map(row => `${row.check} ${row.period}: ${row.state} (difference ${row.difference})`).join("\n")]);
        if (data.conflicts.length) entries.push(["Conflicting repeated rows", data.conflicts.map(row => `${row.item} ${row.period}: ${row.first} vs ${row.later}`).join("\n")]);
      }
      if (data.operation === "digest") {
        entries.push(["Digest", "Extractive cited passages · not generated"], ["Documents matched", `${data.documents_matched} of ${data.documents_visible}`], ["Statement disagreements", data.statement_disagreements]);
        entries.push(["Cited passages by document", data.sections.map(section => `${section.title}\n${section.rows.map(row => `  ${row.text} [${row.passage_id.slice(0, 12)}]`).join("\n")}`).join("\n") || "No matching passage"]);
        const differ = data.statement_comparison.filter(row => row.state === "DIFFER");
        if (differ.length) entries.push(["Values that differ across documents", differ.map(row => `${row.item} ${row.period}: ${row.reports.map(report => report.decimal_value).join(" vs ")}`).join("\n")]);
      }
      if (data.operation === "transcript") {
        entries.push(["Transcript state", data.transcript_state], ["Speaker identity", "Label text only · not verified"], ["Turns", data.turn_count], ["Management share of words", data.management_word_share ?? "UNAVAILABLE"]);
        entries.push(["Speakers", data.speakers.slice(0, 24).map(row => `${row.name} · ${row.role}${row.role_text ? ` (${row.role_text})` : ""}: ${row.turns} turns, ${row.words} words`).join("\n") || "None recognized"]);
        entries.push(["Question-and-answer exchanges", data.exchanges.slice(0, 24).map(row => `${row.analyst}: ${row.answered ? `answered by ${row.responders.join(", ")}` : "no management answer found"} (${row.question_words} / ${row.answer_words} words)`).join("\n") || "None recognized"]);
        entries.push(["Forward-looking sentences", data.forward_looking.slice(0, 24).map(row => `${row.speaker} [${row.cue}${row.hedged ? ", hedged" : ""}]: ${row.sentence} [${row.citation.passage_id.slice(0, 12)}]`).join("\n") || "No cue word found"], ["Cue meaning", "Words appear in the text · not verified or interpreted"]);
        if (isObject(data.financial_mentions)) {
          entries.push(["Cited numerical mentions", data.financial_mentions.facts.slice(0, 8).map(fact => `${fact.metric_mention}: ${fact.quoted_decimal} ${fact.unit}`).join("\n") || "None in the annotated turns"], ["Event mentions", data.financial_mentions.event_mentions.slice(0, 8).map(event => `${event.mention}: occurrence ${event.occurrence}`).join("\n") || "None in the annotated turns"], ["Mention coverage", data.financial_mentions.coverage]);
        }
      }
      if (data.operation === "mention-diff") {
        const mentionLine = row => `${row.metric_mention}: ${row.quoted_decimal} ${row.unit}`;
        entries.push(["Mention comparison", "Parsed source mentions · not verified restatements"], ["Parser", `${data.nlp_profile} · ${data.coverage}`], ["Shared numerical mentions", data.shared_fact_count]);
        entries.push(["Only in the first document", data.base_only_facts.map(mentionLine).join("\n") || "None"], ["Only in the second document", data.revised_only_facts.map(mentionLine).join("\n") || "None"]);
        entries.push(["Events only in the first document", data.base_only_events.map(row => `${row.mention}: occurrence ${row.occurrence}`).join("\n") || "None"], ["Events only in the second document", data.revised_only_events.map(row => `${row.mention}: occurrence ${row.occurrence}`).join("\n") || "None"]);
        entries.push(["Entity text only in one document", [...data.base_only_entities, ...data.revised_only_entities].map(row => `${row.mention}: identity ${row.instrument_identity_state}`).join("\n") || "None"]);
      }
      if (data.operation === "compare") {
        entries.push(["Comparison", "Text differences with citations · not an interpretation"], ["Unchanged passages", data.unchanged_passages], ["Added / removed", `${data.counts.ADDED} / ${data.counts.REMOVED}`], ["Reworded / number-only", `${data.counts.REWORDED} / ${data.counts.NUMERIC_ONLY}`]);
        entries.push(["Changes", data.changes.slice(0, 24).map(row => row.base && row.revised ? `${row.kind}\n  − ${row.base.text}\n  + ${row.revised.text}` : row.base ? `REMOVED\n  − ${row.base.text}` : `ADDED\n  + ${row.revised.text}`).join("\n") || "The documents have identical passages"]);
        entries.push(["Statement values that differ", data.statement_value_changes.map(row => `${row.item} ${row.period}: ${row.base} → ${row.revised} (difference ${row.difference})`).join("\n") || "None among recognized rows"]);
      }
      if (data.operation === "watch") entries.push(["Monitoring", "Explicit checks only · nothing runs in the background"], ["Baseline query", data.query], ["Baseline passages", data.baseline_passage_ids.length]);
      if (data.operation === "watch-check") {
        entries.push(["Watch state", data.alert_state], ["Monitoring", "Explicit check · no notification sent"], ["New passages", data.new_rows.map(row => row.text).join("\n") || "None"], ["Dropped passages", data.dropped_passage_ids.length], ["New documents", data.new_document_ids.length], ["Superseded documents", data.retired_document_ids.length]);
      }
            if (data.option_candidate) {
        entries.push(["Options link", data.link_id], ["Underlying", data.underlying_id], ["Equity stance", data.equity_stance || "None"], ["Option candidate", data.option_candidate], ["Admitted quotes", data.admitted_option_quotes]);
      }
      if (Array.isArray(data.rows) && data.rows[0] && data.rows[0].stance) {
        entries.push(["Research screen", data.screen_id], ["Signal", data.signal_id], ["Buys / sells / holds", `${data.metrics?.research_buys}/${data.metrics?.research_sells}/${data.metrics?.research_holds}`], ["Timing entries / exits / flips", `${data.metrics?.timing_entries}/${data.metrics?.timing_exits}/${data.metrics?.timing_flips}`]);
        data.rows.slice(0, 8).forEach(row => entries.push([row.instrument_id, `${row.stance} · weight ${row.research_weight}`]));
      }
      if (data.operation === "calculator") {
        entries.push(["Saved calculation", data.record_id], ["Calculator", data.kind], ["Persistence", "Local content-addressed record"], ["Replay", data.replay_state || "Saved from these inputs"]);
        if (isObject(data.source)) {
          if (data.source.dataset_id) entries.push(["Dataset", data.source.dataset_id], ["Series", data.source.instrument_id || (data.source.instruments || []).join(", ")], ["Field", data.source.field], ["Dataset classification", data.source.dataset_classification]);
          if (data.source.parent_record_id) entries.push(["Covariance record", data.source.parent_record_id]);
        }
        if (Array.isArray(data.result?.headline)) data.result.headline.forEach(row => entries.push([row.label, row.value]));
      }
      if (data.operation === "investigation") {
        entries.push(["Report", "Extractive cited findings · not generated analysis"], ["Answered", `${data.answered} of ${data.questions.length}`], ["Autonomy", "Single explicit request · no background agent"]);
        entries.push(["Findings", data.findings.map(finding => `${finding.question}\n  ${finding.answer ?? "No cited passage found"}${finding.citations[0] ? ` [${finding.citations[0].passage_id.slice(0, 12)}]` : ""}`).join("\n")]);
        const flagged = data.documents_reviewed.filter(row => row.inconsistent_checks.length || row.conflicts.length);
        entries.push(["Statement checks", flagged.map(row => `${row.document_title}: ${row.inconsistent_checks.map(check => `${check.check} ${check.period}`).join(", ") || "conflicting rows"}`).join("\n") || "No inconsistency found in recognized rows"]);
      }
    }
    if (typeof data.corpus_identity === "string" && typeof data.snapshot_id === "string") {
      entries.push(["Retrieval", data.retrieval_mode === "latent" ? "Corpus-local LSA · cosine relevance" : data.retrieval_mode === "hybrid" ? "Hybrid BM25 + corpus-local LSA · rank fusion" : "BM25 lexical · source citations"], ["Corpus", data.corpus_identity], ["Source scope", data.classification], ["Answer state", data.answer_state], ["Answer mode", data.answer_kind], ["Semantic / generative", `${data.semantic_state} / ${data.generative_state}`]);
      if (typeof data.model_id === "string") entries.push(["Saved retrieval model", data.model_id]);
      if (isObject(data.model) && data.model.model_kind === "CORPUS_LOCAL_LSA") entries.push(["Latent model scope", data.model.training_scope], ["Latent dimensions", data.model.dimensions], ["Feature policy", data.model.feature_policy], ["Latent query state", data.latent_query_state]);
      if (isObject(data.financial_nlp)) {
        entries.push(["Annotation profile", data.nlp_profile || "v1"], ["Annotation implementation", data.financial_nlp.implementation]);
        if (Array.isArray(data.financial_nlp.entity_mentions)) entries.push(["Cited entity mentions", data.financial_nlp.entity_mentions.slice(0, 8).map(item => item.mention || item.surface_text || "UNKNOWN").join("\n") || "No supported entity mentions"]);
        if (Array.isArray(data.financial_nlp.unsupported_mentions) && data.financial_nlp.unsupported_mentions.length) entries.push(["Unsupported mentions", data.financial_nlp.unsupported_mentions.slice(0, 8).map(item => `${item.provenance?.surface_text || "UNKNOWN"}: ${item.reason || "UNSUPPORTED"}`).join("\n")]);
        entries.push(["Financial NLP", "Original rule-based parsed mentions · not verified issuer facts"], ["Sentiment / issuer linking", `${data.financial_nlp.sentiment_state || "UNAVAILABLE"} / ${data.financial_nlp.issuer_linking_state || "UNKNOWN"}`], ["Event occurrence", "UNKNOWN · text mentions only"]);
        const facts = data.financial_nlp.facts || data.financial_nlp.mentions;
        if (Array.isArray(facts)) entries.push(["Cited numerical mentions", facts.slice(0, 8).map(fact => `${fact.metric_mention || fact.kind || fact.type || "Mention"}: ${fact.decimal_value ?? fact.value ?? "UNKNOWN"} ${fact.unit || ""}`).join("\n") || "No supported numerical mentions"]);
      }
      if (typeof data.answer === "string") entries.push(["Exact normalized source passage", data.answer]);
      if (Array.isArray(data.rows) && data.rows.length) entries.push(["Citation", data.rows[0].passage_id], ["Source URL", data.rows[0].source_url], ["Original-byte range", `${data.rows[0].source_start}–${data.rows[0].source_end}`]);
    }
    if (isObject(data.account)) {
      for (const [key, label] of [["cash", "Book cash · USD"], ["available_cash", "Available cash · USD"], ["dividend_receivable", "Dividend receivable · USD"], ["dividend_income", "Recognized dividend income · USD"], ["nav", "Modeled NAV · USD"], ["dividend_accrual_state", "Dividend evidence state"]]) {
        if (Object.hasOwn(data.account, key)) entries.push([label, data.account[key]]);
      }
    }
    if (typeof data.scoring_family === "string" || typeof data.evaluation_id === "string") {
      entries.push(["Scoring family", data.scoring_family || data.prediction_kind], ["Evaluation state", data.evaluation_state], ["Coverage", data.fit_diagnostics?.coverage_state || data.coverage_state]);
      if (isObject(data.fit_diagnostics)) entries.push(["Converged / total models", `${printable(data.fit_diagnostics.converged_models)} / ${printable(data.fit_diagnostics.models)}`]);
      if (isObject(data.metrics)) Object.entries(data.metrics).slice(0, 8).forEach(([key, value]) => entries.push([human(key), value]));
      entries.push(["Economic validation", data.economically_validated === false ? "Not established" : "Unknown"]);
    }
    if (typeof data.reconciliation_state === "string") entries.push(["Independent statement", data.reconciliation_state], ["External reconciliation", data.external_reconciled === false ? "Unverified" : "Unknown"]);
    const calculator = Array.isArray(data.headline) && data.headline.every(row => isObject(row) && typeof row.label === "string");
    if (calculator) {
      data.headline.forEach(row => entries.push([row.label, row.value]));
      Object.entries(data).filter(([key, value]) => key.endsWith("_limits") && typeof value === "string").forEach(([, value]) => entries.push(["Stated limits", value.replace(/_/g, " ").toLowerCase()]));
    }
    if (!entries.length) return;
    $("analysis-summary-values").replaceChildren(...entries.map(([label, value]) => {
      const entry = el("div", "analysis-summary-entry");
      entry.append(el("dt", "context-label", label), el("dd", "context-value", printable(value)));
      return entry;
    }));
    $("analysis-summary-title").textContent = data.operation === "calculator" ? "Saved calculation" : calculator ? "Calculation summary" : "Saved evidence summary";
    $("analysis-summary-note").textContent = data.operation === "calculator" ? "Saved from the inputs in this record and recomputed on replay. No order, allocation or trading approval is established." : calculator ? "Model estimate computed from the inputs you supplied. Nothing was saved or sent. No order, allocation or trading approval is established." : isObject(data.account) ? "SIMULATED supported-scope account. A dividend receivable is not available cash. Unknown valuation remains unknown; funding is not independently certified." : "Saved research diagnostics. Output units, missing folds and uncertainty remain explicit. No profitability, calibrated trading confidence or trading approval is established.";
    $("analysis-summary").classList.remove("hidden");
    $("analysis-summary-values").querySelectorAll("dd").forEach(value => {
      if (value.scrollHeight > value.clientHeight + 1 || value.scrollWidth > value.clientWidth + 1) value.tabIndex = 0;
    });
  }
  function rememberExperiment(action, result) {
    if (!["research-model", "pairs-research", "options-research", "research-signals", "economic-research", "microstructure-research"].includes(action)) return;
    const data = result.data, identity = data.run_id || data.signal_id || data.report_id;
    if (typeof identity !== "string" || !/^[a-f0-9]{64}$/.test(identity)) return;
    if (comparisonResults.some(item => item.identity === identity)) return;
    const entry = { identity, action, dataset: data.dataset_id || null, source: result.provenance.source, classification: data.input_classification || result.provenance.classification, as_of: result.provenance.as_of, metrics: isObject(data.metrics) ? data.metrics : {} };
    comparisonResults.push(entry);
    if (comparisonResults.length > 12) comparisonResults.shift();
    for (const field of ["compare-left", "compare-right"]) {
      const selection = $(field), old = selection.value;
      selection.replaceChildren(...comparisonResults.map(item => { const option = el("option", "", `${human(item.action)} · ${item.identity.slice(0, 12)}`); option.value = item.identity; return option; }));
      selection.value = comparisonResults.some(item => item.identity === old) ? old : entry.identity;
    }
    $("compare-right").value = entry.identity;
    if (comparisonResults.length > 1) $("compare-left").value = comparisonResults[comparisonResults.length - 2].identity;
    renderComparison();
  }
  function renderComparison() {
    const left = comparisonResults.find(item => item.identity === $("compare-left").value);
    const right = comparisonResults.find(item => item.identity === $("compare-right").value);
    $("experiment-comparison").classList.toggle("hidden", !left || !right);
    if (!left || !right) return;
    const sameDataset = left.dataset && left.dataset === right.dataset;
    $("comparison-policy").textContent = (sameDataset ? "Same admitted dataset identity." : "Different or unspecified datasets; results are not directly comparable.") + " Metrics retain each experiment's meaning. Session-only comparison; no economic validation or trading approval.";
    const rows = [["Experiment", left.action, right.action], ["Immutable identity", left.identity, right.identity], ["Dataset", left.dataset, right.dataset], ["Source", left.source, right.source], ["Input classification", left.classification, right.classification], ["As of", left.as_of, right.as_of]];
    const keys = [...new Set([...Object.keys(left.metrics), ...Object.keys(right.metrics)])].slice(0, 24);
    keys.forEach(key => rows.push([human(key), left.metrics[key], right.metrics[key]]));
    $("comparison-body").replaceChildren(...rows.map(values => { const row = el("tr"); values.forEach((value, index) => row.append(el(index ? "td" : "th", "", printable(value)))); return row; }));
  }
  function setTheme(theme) {
    document.documentElement.dataset.theme = theme;
    $("theme-toggle").setAttribute("aria-label", `Switch to ${theme === "dark" ? "light" : "dark"} theme`);
    try { localStorage.setItem("gig-theme", theme); } catch { /* Privacy mode may deny persistence. */ }
    renderChart();
  }
  nav();
  let storedTheme;
  try { storedTheme = localStorage.getItem("gig-theme"); } catch { /* Keep default theme. */ }
  setTheme(["light", "dark"].includes(storedTheme) ? storedTheme : "dark");
  const updateClock = () => { const now = new Date(); $("clock").dateTime = now.toISOString(); $("clock").textContent = `${now.toISOString().slice(11, 19)} UTC`; };
  updateClock(); setInterval(updateClock, 1000);
  $("skip-link").addEventListener("click", (event) => { event.preventDefault(); $("content").focus(); });
  $("theme-toggle").addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
  $("data-mode").addEventListener("change", () => { state.mode = $("data-mode").value; $("analysis-engine").disabled = state.mode !== "local"; if (state.mode !== "local") $("analysis-engine").value = "default"; loadSection(); });
  $("analysis-engine").addEventListener("change", loadSection);
  $("universe-id").addEventListener("change", loadSection);
  $("depth-instrument").addEventListener("change", loadSection);
  $("context-expand").addEventListener("click", () => { const expanded = $("context-expand").getAttribute("aria-expanded") !== "true"; $("context-expand").setAttribute("aria-expanded", String(expanded)); $("context-expand").textContent = expanded ? "Show less" : "Show full provenance"; $("analysis-grid").classList.toggle("context-expanded", expanded); });
  $("orbit-toggle").addEventListener("click", () => setOrbit(!state.orbit));
  if (typeof motionPreference.addEventListener === "function") motionPreference.addEventListener("change", () => { if (motionPreference.matches) setOrbit(false); });
  $("freshness-policy").addEventListener("change", loadSection);
  $("replay-asof").addEventListener("change", loadSection);
  $("refresh").addEventListener("click", loadSection);
  $("table-search").addEventListener("input", renderTable);
  $("evidence-table-view").addEventListener("change", () => {
    state.rows = state.tables && state.tables[$("evidence-table-view").value] || [];
    state.sort = null;
    $("table-search").value = "";
    renderTable();
  });
  $("export-csv").addEventListener("click", exportCsv);
  $("analytics-kind").addEventListener("change", loadExample);
  $("load-example").addEventListener("click", loadExample);
  $("run-analysis").addEventListener("click", runAnalysis);
  $("compare-left").addEventListener("change", renderComparison);
  $("compare-right").addEventListener("change", renderComparison);
  window.addEventListener("hashchange", selectSection);
  const setNav = (collapsed) => {
    document.body.classList.toggle("nav-collapsed", collapsed);
    $("nav-toggle").setAttribute("aria-expanded", String(!collapsed));
    $("nav-toggle").setAttribute("aria-label", collapsed ? "Expand navigation" : "Collapse navigation");
    $("nav-toggle").title = collapsed ? "Expand navigation" : "Collapse navigation";
    try { localStorage.setItem("gig-nav-collapsed", collapsed ? "1" : "0"); } catch { /* Keep the choice for this page only. */ }
  };
  let navCollapsed = false;
  try { navCollapsed = localStorage.getItem("gig-nav-collapsed") === "1"; } catch { /* Default expanded. */ }
  setNav(navCollapsed);
  $("nav-toggle").addEventListener("click", () => setNav(!document.body.classList.contains("nav-collapsed")));
  [2, 3].forEach((dimension) => $(`view-${dimension}d`).addEventListener("click", () => { state.dimension = dimension; [2, 3].forEach((d) => $(`view-${d}d`).setAttribute("aria-pressed", String(d === dimension))); renderChart(); }));
  $("chart-reset").addEventListener("click", () => { state.angle = -.55; state.tilt = .40; state.zoom = 1; renderChart(); });
  const canvas = $("analytics-chart");
  canvas.addEventListener("pointerenter", () => setOrbit(false));
  canvas.addEventListener("pointerdown", (event) => { if (event.button !== 0) return; setOrbit(false); state.drag = { x: event.clientX, y: event.clientY }; canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener("pointermove", chartPointer);
  canvas.addEventListener("pointerup", () => { state.drag = null; });
  canvas.addEventListener("pointercancel", () => { state.drag = null; });
  canvas.addEventListener("pointerleave", () => $("chart-tooltip").classList.add("hidden"));
  canvas.addEventListener("wheel", (event) => { if (!state.points.length) return; event.preventDefault(); state.zoom = Math.max(.45, Math.min(2.5, state.zoom * (event.deltaY > 0 ? .92 : 1.08))); renderChart(); }, { passive: false });
  canvas.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "+", "=", "-", "0"].includes(event.key)) return;
    event.preventDefault();
    setOrbit(false);
    if (event.key === "ArrowLeft") state.angle -= .1;
    if (event.key === "ArrowRight") state.angle += .1;
    if (event.key === "ArrowUp") state.tilt = Math.min(1.2, state.tilt + .1);
    if (event.key === "ArrowDown") state.tilt = Math.max(-1.2, state.tilt - .1);
    if (event.key === "+" || event.key === "=") state.zoom = Math.min(2.5, state.zoom * 1.1);
    if (event.key === "-") state.zoom = Math.max(.45, state.zoom / 1.1);
    if (event.key === "0") { state.angle = -.55; state.tilt = .40; state.zoom = 1; }
    renderChart();
  });
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(renderChart).observe($("chart-container"));
  else window.addEventListener("resize", renderChart);
  selectSection();
})();
