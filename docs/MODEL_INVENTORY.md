# Model inventory

Scope: what the code does when imported by `src/gig` or by `tests/`. Prototype files are listed so a filename is not mistaken for a runtime model.

Columns: file, implementation, real/stub/proxy, fitted, tested, imported by the maintained package, in the `gig serve` process, affects signals, affects weights, affects execution, disposition.

Status words: `EXPERIMENTAL`, `RESEARCH`, `VALIDATED`, `PAPER`, `CHALLENGER`, `CHAMPION`, `RETIRED`. Nothing in the code registry is `CHAMPION`. `model_inventory()` raises if a row claims that status.

## Maintained package

| Model | File | Implementation | Kind | Fitted | Tested | Imported | Runtime | Signals | Weights | Execution | Disposition |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 12-1 momentum, 21-day reversal, 63-day idio vol | `src/gig/factors/` | cross-sectional factors, neutralized | real | no (deterministic) | yes | yes | yes | yes | yes | no | preserve |
| Lagged IC combination | `src/gig/factors/combine.py` | expanding/EWM rank IC, lagged one day | real | yes | yes | yes | yes | yes | yes | no | preserve |
| Statistical factor risk | `src/gig/risk/factor_model.py` | `Σ = BB' + D` | real | yes | yes | yes | yes | no | yes | no | preserve |
| Factor-neutral optimizer | `src/gig/portfolio/optimize.py` | closed form, box + neutrality | real | no | yes | yes | yes | no | yes | yes, as the target of `gig trade` | preserve |
| Historical VaR and expected shortfall | `src/gig/risk/var.py` | empirical quantile and tail mean | real | no | yes | yes | yes | no | no | no | preserve |
| Walk-forward ranker | `src/gig/ml/ranker.py` | sklearn HistGBDT; LightGBM LambdaRank if `gig[ml]` is installed | real | yes, OOS folds | yes (LightGBM test skips when the library is absent) | yes | yes, when the backtest asks for ML | yes | yes | no | preserve |
| Lexicon sentiment | `src/gig/nlp/lexicon.py` | word list, optional VADER | real | no | yes | yes | yes | no | no | no | preserve |
| FinBERT | `src/gig/nlp/sentiment.py` `transformer_sentiment` | ProsusAI/finbert if `transformers` imports | optional real | weights are the vendor's, not fit here | no | yes, the function | no, unless a caller installs transformers and invokes it | no | no | no | keep optional; do not describe the lexicon as FinBERT |
| Gaussian HMM | `src/gig/models/hmm.py` | EM: start, transitions, means, stds. Filtered probs use returns through t. Smoothed path is full sample | real | yes | yes | yes | diagnostic route only | no | no | no | preserve as research |
| GARCH(1,1) | `src/gig/models/garch.py` | Gaussian MLE, L-BFGS-B, persistence constrained below 1 | real | yes | yes | yes | diagnostic route only | no | no | no | preserve as research |
| EWMA | `src/gig/models/garch.py` `ewma_variance` | RiskMetrics λ = 0.94 | real | no | yes | yes | diagnostic route only | no | no | no | preserve as the GARCH baseline |
| Kalman dynamic regression | `src/gig/models/kalman.py` | causal filter, Joseph-style covariance | real | yes | yes | yes | not called by the book | no | no | no | preserve; no pair book yet |
| Black-Litterman | `src/gig/portfolio/black_litterman.py` | `π = δΣw`, posterior precision blend | real | no | yes | yes | no | no | no | no | preserve; live book stays the factor optimizer |
| Fractional Kelly | `src/gig/portfolio/kelly.py` | ridge, then gross cap | real | no | yes | yes | no | no | no | no | preserve; must not override limits |
| Dynamic deleveraging | `src/gig/risk/deleverage.py` | logistic scale, floor 0.25 | real | no | yes | yes | diagnostic suggestion only | no | no | no | preserve; `applied_to_orders` is false |
| Almgren-Chriss | `src/gig/execution/schedule.py` | closed-form trajectory vs TWAP | real planner | no | yes | yes | no | no | no | no | preserve as a planner; do not call the limit slicer by this name |
| Marketable limit slices | `src/gig/execution/algo.py` | child orders through the touch | real | no | yes | yes | yes, on `gig trade run` | no | no | yes | preserve |
| Black-Scholes | `src/gig/assets/options.py` | price, Greeks, Newton IV | real | no | yes | yes | no chain loaded | no | no | no | preserve the pricer; no surface without a chain |
| C++ neutralization | `src/cpp/speed.cpp` | pybind11 | real | no | yes | yes | yes, if the extension built | yes | yes | no | preserve |

Registry rows match this table. `GET /api/models` returns them and sets `champion` to null.

## Named families that are not in the maintained runtime

| Name | Where a reader might look | What it actually is | Disposition |
|---|---|---|---|
| LightGBM | `src/gig/ml/ranker.py` | Used when installed. Default install is sklearn. | preserve the optional branch |
| BiLSTM | `deep_learning_engine.py` | Torch BiLSTM if torch imports; otherwise a NumPy LSTM cell. Not imported by `src/gig`. A failed forward path can emit `0.5`. | leave unwired; do not treat `0.5` as a forecast |
| HMM (prototype) | `math_engine.HiddenMarkovRegime` | EM updates means and stds from hardcoded bear/neutral/bull initialization. The M-step does not update `trans`. Labels are assigned before fitting. | do not call from the book; the maintained EM replaces it for research |
| GARCH proxy | `math_engine.VolatilityAnalysis.garch_proxy_vol` | Recursion with fixed `alpha=0.1`, `beta=0.85`. Not a likelihood. | label as EWMA-style; the MLE is `fit_garch11` |
| EGARCH | — | absent | do not add a stub |
| GJR-GARCH | — | absent | do not add a stub |
| Kalman (prototype) | `math_engine.KalmanFilter` | simplified dynamic regression, not used by `src/gig` | superseded for research by `src/gig/models/kalman.py` |
| FinBERT class | `nlp_engine.py` | falls back to a lexicon | not runtime |
| Classical factors beyond momentum, reversal, idio vol | scattered root engines | not on the book path | do not promote by filename |
| IC-adaptive weighting | `src/gig/factors/combine.py` only | lagged rank IC of the three factors | the live combiner; no extra IC engine |
| Stat-arb / cointegration / PCA residuals | — in `src/gig` | Kalman hedge ratio exists; no ADF screen, no residual book | research-only until a walk-forward pair test exists |
| OFI, Lee-Ready, Kyle lambda, VPIN | `orderflow_engine.py`, `research_agent.py`, `dark_pool_engine.py` | bar or synthetic reconstructions. Not tick/quote native. Not imported by `src/gig`. | do not display as TICK NATIVE |
| Options analytics beyond BSM | `Options_engine.py`, `src/gig/assets/options.py` | BSM is real. No chain, so no smile, term structure, PCR, or GEX. | pricer only |
| Black-Litterman (prototype) | `portfolio_optimizer.py` | not imported by `src/gig` | use `src/gig/portfolio/black_litterman.py` |
| Kelly (prototype) | `portfolio_optimizer.py`, called from `live_data_server.py` | not on the `gig serve` path | use `fractional_kelly` if a caller is added later |
| VaR / ES | `src/gig/risk/var.py` | historical | preserve; not a model-based VaR |
| Almgren-Chriss (prototype) | `execution_engine.py` | used by `live_data_server.py` | not the paper OMS |
| Dynamic deleveraging | `src/gig/risk/deleverage.py` | suggestion | not applied |

## What "in runtime" means

Imported by `gig serve` or by the equity long/short path that `gig backtest` and `gig trade` call. A module that is importable and unused by those commands is not in the runtime.
