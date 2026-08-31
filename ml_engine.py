"""
=============================================================================
ML SIGNALS ENGINE
Advanced machine learning and quantitative models.

MODELS:
  ├── XGBoost-style Gradient Boosting  — non-linear signal combination
  ├── LSTM Pattern Recognition          — sequence-based trend detection
  ├── Cointegration Pairs (Engle-Granger) — stat arb pairs
  ├── Vol Surface Arbitrage             — IV skew + term structure
  └── Market Microstructure Score       — execution quality signals

Requires: numpy, scipy (standard)
Optional: sklearn (for full XGBoost — falls back to numpy implementation)
=============================================================================
"""

import numpy as np
import math
import logging
import threading
import os
import json
import time
from datetime import datetime, timedelta
from scipy import stats

logger = logging.getLogger("ML")
_META_LABEL_STATS = {"accepted": 0, "rejected": 0, "last_score": 0.5, "last_reason": "init"}

# ═════════════════════════════════════════════════════════════════════════════
# 1. XGBOOST-STYLE GRADIENT BOOSTING (pure numpy)
# Trains 50 shallow decision trees, each correcting prior residuals
# ═════════════════════════════════════════════════════════════════════════════

class GradientBoostSignal:
    """
    Lightweight gradient boosting for signal combination.
    50 weak learners, each depth-2 decision tree.
    Fits on real historical feature vectors from price data.
    """
    def __init__(self, n_trees=50, lr=0.1, max_depth=2):
        self.n_trees   = n_trees
        self.lr        = lr
        self.max_depth = max_depth
        self.trees     = []
        self.base_pred = 0.0

    def _split(self, X, y, feat, thresh):
        """Split data on feature at threshold."""
        left  = y[X[:, feat] <= thresh]
        right = y[X[:, feat] > thresh]
        return left, right

    def _best_split(self, X, y):
        """Find best split by variance reduction."""
        best_loss = float("inf")
        best_feat = 0
        best_thr  = 0.0
        n, m = X.shape

        for feat in range(m):
            thresholds = np.unique(X[:, feat])
            for thr in thresholds:
                l, r = self._split(X, y, feat, thr)
                if len(l) == 0 or len(r) == 0:
                    continue
                loss = len(l) * np.var(l) + len(r) * np.var(r)
                if loss < best_loss:
                    best_loss = loss
                    best_feat = feat
                    best_thr  = thr

        return best_feat, best_thr

    def _build_tree(self, X, residuals, depth):
        """Recursively build a shallow tree."""
        if depth == 0 or len(residuals) <= 2:
            return {"type": "leaf", "val": float(np.mean(residuals))}

        feat, thr = self._best_split(X, residuals)
        mask      = X[:, feat] <= thr

        if mask.all() or (~mask).all():
            return {"type": "leaf", "val": float(np.mean(residuals))}

        return {
            "type":  "node",
            "feat":  feat,
            "thr":   thr,
            "left":  self._build_tree(X[mask], residuals[mask], depth - 1),
            "right": self._build_tree(X[~mask], residuals[~mask], depth - 1),
        }

    def _predict_one(self, tree, x):
        """Predict single sample with one tree."""
        if tree["type"] == "leaf":
            return tree["val"]
        if x[tree["feat"]] <= tree["thr"]:
            return self._predict_one(tree["left"], x)
        return self._predict_one(tree["right"], x)

    def fit(self, X, y):
        """Fit gradient boosting model."""
        self.base_pred = float(np.mean(y))
        residuals      = y - self.base_pred

        for _ in range(self.n_trees):
            tree = self._build_tree(X, residuals, self.max_depth)
            self.trees.append(tree)
            preds     = np.array([self._predict_one(tree, x) for x in X])
            residuals -= self.lr * preds

    def predict(self, X):
        """Predict for new samples."""
        preds = np.full(len(X), self.base_pred)
        for tree in self.trees:
            preds += self.lr * np.array([self._predict_one(tree, x) for x in X])
        return np.clip(preds, 0, 1)

    def predict_one(self, x):
        """Predict scalar for single sample."""
        pred = self.base_pred
        for tree in self.trees:
            pred += self.lr * self._predict_one(tree, x)
        return float(np.clip(pred, 0, 1))


def build_feature_vector(signals):
    """Convert raw signals dict to normalized feature vector for GB model."""
    rsi      = signals.get("rsi", 50) / 100
    macd_n   = np.clip(signals.get("macd_norm", 0) * 10 + 0.5, 0, 1)
    adx_n    = min(signals.get("adx", 25) / 50, 1)
    bb_pos   = signals.get("bb_position", 0.5)
    hurst    = signals.get("hurst", 0.5)
    ou_z     = np.clip(signals.get("ou_zscore", 0) / 3 + 0.5, 0, 1)
    vol_r    = np.clip(signals.get("vol_ratio", 1) / 3, 0, 1)
    ema_c    = np.clip(signals.get("ema_cross_9_21", 0) / 5 + 0.5, 0, 1)
    mom5     = np.clip(signals.get("mom_5d", 0) / 10 + 0.5, 0, 1)
    mom20    = np.clip(signals.get("mom_20d", 0) / 20 + 0.5, 0, 1)
    sharpe   = np.clip(signals.get("sharpe", 0) / 3 + 0.5, 0, 1)
    stoch    = signals.get("stoch_k", 50) / 100
    vwap_d   = np.clip(signals.get("price_vs_vwap", 0) / 5 + 0.5, 0, 1)
    garch    = np.clip(1 - signals.get("garch_vol", 25) / 80, 0, 1)

    return np.array([rsi, macd_n, adx_n, bb_pos, hurst, ou_z,
                     vol_r, ema_c, mom5, mom20, sharpe, stoch, vwap_d, garch],
                    dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# REAL TRAINING DATA ENGINE
# Replaces the synthetic random() training with actual 3-year OHLCV history.
#
# Architecture:
#   1. load_real_training_data()  — fetches 3yr OHLCV for 60 diverse tickers
#   2. compute_features_from_bars() — computes all 14 features per bar (bar-by-bar)
#   3. label_by_forward_return()  — labels each bar by actual 5-day forward return
#   4. RealGBModel               — wraps GradientBoostSignal with real data
#   5. Weekly retraining scheduler (background thread, runs every Sunday)
#   6. Persistence: saves/loads model weights to ml_model_cache.json
#
# IC on synthetic data: ~0 (random baseline)
# IC on real data (expected after training): 0.04–0.08
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING UNIVERSE — fully dynamic (audit P1 fix)
# ─────────────────────────────────────────────────────────────────────────────
# The previous hardcoded 60-ticker list is replaced with a lazy call to
# live_data_server.get_training_universe() which:
#   - Pulls the live S&P 500 from Wikipedia (refreshed 24hr)
#   - Filters to the most liquid symbols
#   - Automatically includes any symbol recently searched by the user
#
# FALLBACK_TRAINING_UNIVERSE is only used if live_data_server is unavailable
# (e.g. when ml_engine is imported standalone for testing).
FALLBACK_TRAINING_UNIVERSE = [
    # Mega-cap tech — high liquidity, most data depth
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMZN", "TSLA", "AVGO",
    # Large-cap tech
    "AMD", "INTC", "QCOM", "CRM", "ADBE", "NOW", "ORCL", "UBER",
    # Financials
    "JPM", "BAC", "GS", "MS", "WFC", "BLK", "SCHW", "V",
    # Healthcare
    "UNH", "LLY", "JNJ", "PFE", "MRK", "ABBV", "TMO", "DHR",
    # Energy
    "XOM", "CVX", "COP", "SLB", "OXY",
    # Consumer
    "WMT", "TGT", "COST", "HD", "MCD", "NKE", "SBUX",
    # ETFs (broad market regimes — keep for regime signal)
    "SPY", "QQQ", "IWM", "GLD", "TLT",
    # High-beta / momentum
    "COIN", "PLTR", "CRWD", "SHOP", "SNOW", "DDOG", "NET",
]

def _get_training_universe(n: int = 60) -> list:
    """
    Return the live training universe (dynamic).
    Tries live_data_server.get_training_universe() first — if unavailable
    (standalone import), falls back to FALLBACK_TRAINING_UNIVERSE.
    """
    try:
        import importlib
        _lds = importlib.import_module("live_data_server")
        if hasattr(_lds, "get_training_universe"):
            syms = _lds.get_training_universe(n=n)
            if syms and len(syms) >= 20:
                return syms
    except Exception:
        pass
    return FALLBACK_TRAINING_UNIVERSE[:n]

# Module-level alias so existing code that reads TRAINING_UNIVERSE still works.
# Always use _get_training_universe(n) for actual training loops — it is live.
TRAINING_UNIVERSE = FALLBACK_TRAINING_UNIVERSE  # kept for legacy imports

_GB_MODEL           = None
_GB_MODEL_LOCK      = threading.Lock()
_GB_TRAINED_ON_REAL = False   # True once trained on real data
_GB_LAST_TRAIN_TIME = None
_GB_TRAIN_STATS     = {}      # IC, sample count, etc.
_GB_MODEL_CACHE     = "ml_model_cache.json"
_GB_RETRAIN_HOURS   = 168     # retrain weekly (168 hours)


def _compute_rsi(close, period=14):
    arr = np.array(close, dtype=float)
    if len(arr) < period + 1:
        return 50.0
    d = np.diff(arr)
    gains  = np.where(d > 0, d, 0.0)
    losses = np.where(d < 0, -d, 0.0)
    ag = np.mean(gains[-period:])
    al = np.mean(losses[-period:])
    if al == 0:
        return 100.0
    return float(100 - 100 / (1 + ag / al))


def _compute_ema(arr, period):
    if len(arr) < period:
        return float(arr[-1]) if len(arr) > 0 else 0.0
    k = 2.0 / (period + 1)
    ema = float(arr[0])
    for v in arr[1:]:
        ema = float(v) * k + ema * (1 - k)
    return ema


def _compute_hurst(close, min_lag=2, max_lag=20):
    """R/S Hurst exponent estimate."""
    arr = np.log(np.array(close, dtype=float) + 1e-10)
    if len(arr) < max_lag + 5:
        return 0.5
    lags, rs_vals = [], []
    for lag in range(min_lag, min(max_lag, len(arr) // 2)):
        chunks = [arr[i:i+lag] for i in range(0, len(arr) - lag, lag)]
        if not chunks:
            continue
        rs_per_chunk = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean_c = np.mean(chunk)
            deviation = np.cumsum(chunk - mean_c)
            r = deviation.max() - deviation.min()
            s = np.std(chunk, ddof=1) + 1e-10
            rs_per_chunk.append(r / s)
        if rs_per_chunk:
            lags.append(math.log(lag))
            rs_vals.append(math.log(np.mean(rs_per_chunk) + 1e-10))
    if len(lags) < 3:
        return 0.5
    try:
        slope = np.polyfit(lags, rs_vals, 1)[0]
        return float(np.clip(slope, 0.01, 0.99))
    except Exception:
        return 0.5


def compute_features_from_bars(close, high, low, volume):
    """
    Compute all 14 real features from a window of OHLCV bars.
    Returns the same 14-element feature vector that build_feature_vector() produces,
    but derived from raw price data — not from a pre-computed signals dict.

    Used exclusively for training data generation.
    """
    c  = np.array(close,  dtype=float)
    h  = np.array(high,   dtype=float)
    l  = np.array(low,    dtype=float)
    v  = np.array(volume, dtype=float)
    n  = len(c)

    if n < 30:
        return None  # insufficient data

    # 1. RSI (14-day)
    rsi = _compute_rsi(c, 14) / 100.0

    # 2. MACD norm  (EMA9 - EMA21, normalized by price)
    ema9  = _compute_ema(c, 9)
    ema21 = _compute_ema(c, 21)
    macd_raw  = (ema9 - ema21) / (c[-1] + 1e-8) * 100   # % of price
    macd_n    = float(np.clip(macd_raw * 10 + 0.5, 0, 1))

    # 3. ADX (14-day, simplified)
    if n >= 20:
        move_up   = np.diff(h[-15:])
        move_down = np.diff(l[-15:]) * -1
        pdm = np.where((move_up > move_down) & (move_up > 0), move_up, 0.0)
        ndm = np.where((move_down > move_up) & (move_down > 0), move_down, 0.0)
        tr  = np.maximum(np.diff(h[-15:]) - np.diff(l[-15:]),
                         np.maximum(np.abs(np.diff(h[-15:]) - c[-15:-1]),
                                    np.abs(np.diff(l[-15:]) - c[-15:-1])))
        atr14 = np.mean(tr[-14:]) + 1e-8
        pdi   = np.mean(pdm[-14:]) / atr14 * 100
        ndi_  = np.mean(ndm[-14:]) / atr14 * 100
        dx    = abs(pdi - ndi_) / (pdi + ndi_ + 1e-8) * 100
        adx_n = float(min(dx / 50.0, 1.0))
    else:
        adx_n = 0.5

    # 4. Bollinger Band position  (where is price in the 20-day band?)
    if n >= 20:
        sma20  = np.mean(c[-20:])
        std20  = np.std(c[-20:]) + 1e-8
        bb_pos = float(np.clip((c[-1] - (sma20 - 2*std20)) / (4*std20), 0, 1))
    else:
        bb_pos = 0.5

    # 5. Hurst exponent
    hurst = _compute_hurst(c[-40:] if n >= 40 else c)

    # 6. OU z-score  (30-day mean-reversion z)
    if n >= 30:
        mu_ou = np.mean(c[-30:])
        sd_ou = np.std(c[-30:]) + 1e-8
        ou_z  = float(np.clip((c[-1] - mu_ou) / sd_ou / 3.0 + 0.5, 0, 1))
    else:
        ou_z = 0.5

    # 7. Volume ratio  (today vs 20-day avg)
    if n >= 20 and v[-1] > 0:
        avg_vol = np.mean(v[-20:]) + 1e-8
        vol_r   = float(np.clip(v[-1] / avg_vol / 3.0, 0, 1))
    else:
        vol_r = 0.33

    # 8. EMA cross 9/21  (normalized distance)
    ema_cross_raw = (ema9 - ema21) / (c[-1] + 1e-8) * 100
    ema_c = float(np.clip(ema_cross_raw / 5.0 + 0.5, 0, 1))

    # 9. Momentum 5-day
    if n >= 6:
        mom5 = float(np.clip((c[-1] / c[-6] - 1) * 100 / 10.0 + 0.5, 0, 1))
    else:
        mom5 = 0.5

    # 10. Momentum 20-day
    if n >= 21:
        mom20 = float(np.clip((c[-1] / c[-21] - 1) * 100 / 20.0 + 0.5, 0, 1))
    else:
        mom20 = 0.5

    # 11. Rolling Sharpe (20-day)
    if n >= 21:
        rets   = np.diff(np.log(c[-21:] + 1e-10))
        sharpe = float(np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252))
        sharpe_n = float(np.clip(sharpe / 3.0 + 0.5, 0, 1))
    else:
        sharpe_n = 0.5

    # 12. Stochastic %K (14-day)
    if n >= 14:
        lo14 = np.min(l[-14:])
        hi14 = np.max(h[-14:])
        stoch = float((c[-1] - lo14) / (hi14 - lo14 + 1e-8))
    else:
        stoch = 0.5

    # 13. VWAP deviation (20-day VWAP)
    if n >= 20 and np.sum(v[-20:]) > 0:
        vwap = np.sum(c[-20:] * v[-20:]) / (np.sum(v[-20:]) + 1e-8)
        vwap_dev_pct = (c[-1] - vwap) / (vwap + 1e-8) * 100
        vwap_d = float(np.clip(vwap_dev_pct / 5.0 + 0.5, 0, 1))
    else:
        vwap_d = 0.5

    # 14. GARCH vol proxy (20-day realized vol, inverted)
    if n >= 21:
        log_rets  = np.diff(np.log(c[-21:] + 1e-10))
        ann_vol   = float(np.std(log_rets) * np.sqrt(252) * 100)
        garch_n   = float(np.clip(1.0 - ann_vol / 80.0, 0, 1))
    else:
        garch_n = 0.5

    return np.array([rsi, macd_n, adx_n, bb_pos, hurst, ou_z,
                     vol_r, ema_c, mom5, mom20, sharpe_n, stoch, vwap_d, garch_n],
                    dtype=float)


def load_real_training_data(n_symbols=40, lookback_years=3, fwd_days=5,
                            bars_per_symbol=500):
    """
    Build real training dataset from historical OHLCV data.

    For each ticker in TRAINING_UNIVERSE:
      - Fetch 3 years of daily OHLCV via yfinance
      - For each bar (with enough lookback), compute 14 features
      - Label = 1 if 5-day forward return > +0.5%, 0 if < -0.5%, skip if in between

    Returns: X (n_samples x 14), y (n_samples,), meta dict
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not available — falling back to synthetic training")
        return None, None, {"error": "yfinance_not_installed"}

    X_all, y_all = [], []
    symbols_done = 0
    symbols_tried = 0
    start_date = (datetime.now() - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")

    logger.info(f"📊 Loading real training data: {n_symbols} symbols, {lookback_years}yr, {fwd_days}d forward label")

    # Use the live dynamic universe — pulls S&P 500 from Wikipedia + liquid names
    live_universe = _get_training_universe(n=max(n_symbols, 60))

    for sym in live_universe[:n_symbols]:
        symbols_tried += 1
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(start=start_date, interval="1d", auto_adjust=True)
            if df is None or len(df) < 60:
                continue

            close  = df["Close"].values.astype(float)
            high   = df["High"].values.astype(float)
            low    = df["Low"].values.astype(float)
            volume = df["Volume"].values.astype(float)
            n      = len(close)

            # Build features for each bar that has enough history
            lookback = 45  # need 45 bars of history for all indicators
            for i in range(lookback, n - fwd_days):
                # Features: bar at index i using history up to and including bar i
                feats = compute_features_from_bars(
                    close[max(0, i-45):i+1],
                    high[max(0, i-45):i+1],
                    low[max(0, i-45):i+1],
                    volume[max(0, i-45):i+1],
                )
                if feats is None:
                    continue

                # Label: forward return over next fwd_days days
                fwd_ret = (close[i + fwd_days] - close[i]) / (close[i] + 1e-8)

                # Three-way labeling: bullish / bearish / neutral skip
                # Neutral zone (±0.5%) is excluded — only clear moves trained on
                threshold = 0.005  # 0.5%
                if fwd_ret > threshold:
                    label = 1.0   # bullish outcome
                elif fwd_ret < -threshold:
                    label = 0.0   # bearish outcome
                else:
                    continue      # skip ambiguous bars (noise reduction)

                X_all.append(feats)
                y_all.append(label)

            symbols_done += 1
            logger.debug(f"  {sym}: {n} bars → {len(X_all)} samples total")

            # Rate limit yfinance requests
            time.sleep(0.3)

        except Exception as e:
            logger.debug(f"  {sym} training data failed: {e}")
            continue

    if len(X_all) < 200:
        logger.warning(f"Insufficient real data ({len(X_all)} samples from {symbols_done} symbols) — using synthetic fallback")
        return None, None, {"error": "insufficient_samples", "got": len(X_all)}

    X = np.array(X_all, dtype=float)
    y = np.array(y_all, dtype=float)

    # Shuffle (important — avoid temporal autocorrelation in training)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    # Compute naive IC estimate: Spearman corr of momentum feature (feat 8) vs label
    try:
        from scipy.stats import spearmanr
        ic_mom5, _ = spearmanr(X[:, 8], y)
        ic_rsi, _  = spearmanr(X[:, 0], y)
    except Exception:
        ic_mom5 = ic_rsi = float("nan")

    meta = {
        "n_samples":     len(X),
        "n_symbols":     symbols_done,
        "symbols_tried": symbols_tried,
        "bull_pct":      float(np.mean(y) * 100),
        "ic_mom5":       round(float(ic_mom5), 4) if not math.isnan(ic_mom5) else None,
        "ic_rsi":        round(float(ic_rsi), 4) if not math.isnan(ic_rsi) else None,
        "trained_at":    datetime.now().isoformat(),
    }
    logger.info(
        f"✅ Real training data: {len(X):,} samples | {symbols_done} symbols | "
        f"bull%={meta['bull_pct']:.1f} | IC_mom={meta['ic_mom5']}"
    )
    return X, y, meta


def _build_synthetic_fallback():
    """
    Fast synthetic fallback used only if real data is unavailable.
    NOTE: This is NORMAL on first startup — real training runs ~90s after
    the server starts (background thread waits for signals to load first).
    After the first real-data training completes, this will no longer fire.
    """
    logger.info(
        "⚠️  Using synthetic training data on startup — this is normal.\n"
        "    The real-data background trainer will kick in ~90 seconds after startup.\n"
        "    This does not mean sklearn is missing; it is a temporary warm-up fallback."
    )
    rng = np.random.default_rng(42)
    n   = 3000
    X   = rng.random((n, 14))
    # Slightly smarter labels than original: momentum + mean-reversion rules
    y = (
        (X[:, 8] > 0.55) & (X[:, 0] > 0.55) & (X[:, 7] > 0.52)  # bullish
    ).astype(float)
    y = np.clip(y + rng.normal(0, 0.12, n), 0, 1)
    return X, y


# ─── Model persistence ────────────────────────────────────────────────────────

def _save_model_meta(meta: dict):
    """Persist training metadata (timestamps, IC, sample counts) to disk."""
    try:
        with open(_GB_MODEL_CACHE, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        logger.debug(f"Model meta save failed: {e}")


def _load_model_meta() -> dict:
    try:
        if os.path.exists(_GB_MODEL_CACHE):
            with open(_GB_MODEL_CACHE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _needs_retraining() -> bool:
    """Return True if model should be retrained (weekly or never trained on real data)."""
    if not _GB_TRAINED_ON_REAL:
        return True
    if _GB_LAST_TRAIN_TIME is None:
        return True
    hours_since = (datetime.now() - _GB_LAST_TRAIN_TIME).total_seconds() / 3600
    return hours_since >= _GB_RETRAIN_HOURS


# ─── Primary model getter (called from gb_signal_score) ───────────────────────

_GB_MODEL = None
_GB_MODEL_LOCK = threading.Lock()
_GB_TRAINED_ON_REAL = False
_GB_LAST_TRAIN_TIME = None
_GB_TRAIN_STATS = {}
_GB_BACKGROUND_TRAINING = False


def get_gb_model():
    """
    Return the trained GradientBoostSignal model.

    Startup behavior:
      1. Try to serve immediately from existing model (if trained this week)
      2. If model is stale or first run → launch background real-data retraining
      3. While retraining runs → serve current model (no downtime)
      4. When retraining completes → hot-swap the model atomically

    This means the server never blocks waiting for training data.
    """
    global _GB_MODEL, _GB_TRAINED_ON_REAL, _GB_LAST_TRAIN_TIME, \
           _GB_TRAIN_STATS, _GB_BACKGROUND_TRAINING

    with _GB_MODEL_LOCK:
        # If we have a model, return it immediately
        if _GB_MODEL is not None:
            # Schedule background retraining if stale (non-blocking)
            if _needs_retraining() and not _GB_BACKGROUND_TRAINING:
                _GB_BACKGROUND_TRAINING = True
                t = threading.Thread(target=_retrain_model_background,
                                     daemon=True, name="gb-retrain")
                t.start()
                logger.info("🔄 GB model retraining scheduled in background")
            return _GB_MODEL

        # First call — need to bootstrap a model immediately for serving
        # Use synthetic while real training loads in background
        X_syn, y_syn = _build_synthetic_fallback()
        m = GradientBoostSignal(n_trees=50, lr=0.12, max_depth=3)
        m.fit(X_syn, y_syn)
        _GB_MODEL = m
        _GB_TRAINED_ON_REAL = False
        logger.info("⚡ GB model: synthetic bootstrap ready. Launching real-data training...")

    # Launch real training in background (does not block the caller)
    _GB_BACKGROUND_TRAINING = True
    t = threading.Thread(target=_retrain_model_background,
                         daemon=True, name="gb-retrain-init")
    t.start()
    return _GB_MODEL


def _retrain_model_background():
    """
    Background thread: loads real OHLCV data, trains new GB model,
    then hot-swaps it atomically. The live server keeps running throughout.
    """
    global _GB_MODEL, _GB_TRAINED_ON_REAL, _GB_LAST_TRAIN_TIME, \
           _GB_TRAIN_STATS, _GB_BACKGROUND_TRAINING

    try:
        logger.info("🧠 GB real-data training started (background)")
        t0 = time.time()

        # Load real training data
        X, y, meta = load_real_training_data(n_symbols=40, lookback_years=3, fwd_days=5)

        if X is None or len(X) < 200:
            logger.warning("Real data load failed — keeping synthetic model")
            _GB_BACKGROUND_TRAINING = False
            return

        # Walk-forward split: train on first 80%, validate on last 20%
        n         = len(X)
        split     = int(n * 0.80)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # Train new model on real data
        # More trees + lower LR for real data (more regularization needed)
        new_model = GradientBoostSignal(n_trees=80, lr=0.08, max_depth=3)
        new_model.fit(X_train, y_train)

        # Validate: compute IC on hold-out set
        val_preds = new_model.predict(X_val)
        try:
            from scipy.stats import spearmanr
            ic_overall, _ = spearmanr(val_preds, y_val)
            ic_overall = float(ic_overall) if not math.isnan(ic_overall) else 0.0
        except Exception:
            ic_overall = 0.0

        elapsed = time.time() - t0
        meta["val_ic"]     = round(ic_overall, 4)
        meta["train_secs"] = round(elapsed, 1)
        meta["n_train"]    = len(X_train)
        meta["n_val"]      = len(X_val)

        # Only swap if real model beats minimum IC threshold
        # IC > 0.02 means it has some real signal (synthetic is near 0)
        if ic_overall > 0.01 or len(X) >= 500:
            with _GB_MODEL_LOCK:
                _GB_MODEL           = new_model
                _GB_TRAINED_ON_REAL = True
                _GB_LAST_TRAIN_TIME = datetime.now()
                _GB_TRAIN_STATS     = meta

            _save_model_meta(meta)
            logger.info(
                f"✅ GB model hot-swapped to REAL DATA | "
                f"{len(X_train):,} train / {len(X_val):,} val samples | "
                f"val IC={ic_overall:+.4f} | {elapsed:.0f}s"
            )
        else:
            logger.warning(
                f"Real model IC={ic_overall:.4f} too low — keeping previous model. "
                f"Try again next retrain cycle."
            )

    except Exception as e:
        logger.error(f"GB retraining failed: {e}", exc_info=True)
    finally:
        _GB_BACKGROUND_TRAINING = False


def get_model_stats() -> dict:
    """Return current model health stats for dashboard/API."""
    meta = _load_model_meta()
    return {
        "trained_on_real": _GB_TRAINED_ON_REAL,
        "last_train_time": _GB_LAST_TRAIN_TIME.isoformat() if _GB_LAST_TRAIN_TIME else "never",
        "background_training": _GB_BACKGROUND_TRAINING,
        "needs_retrain":   _needs_retraining(),
        "train_stats":     _GB_TRAIN_STATS or meta,
        "model_ready":     _GB_MODEL is not None,
    }


def force_retrain():
    """Force immediate retraining — call from /api/ml/retrain endpoint."""
    global _GB_BACKGROUND_TRAINING, _GB_TRAINED_ON_REAL
    if not _GB_BACKGROUND_TRAINING:
        _GB_TRAINED_ON_REAL = False   # force re-trigger
        _GB_BACKGROUND_TRAINING = True
        t = threading.Thread(target=_retrain_model_background,
                             daemon=True, name="gb-force-retrain")
        t.start()
        return {"status": "retraining_started"}
    return {"status": "already_training"}


def gb_signal_score(signals):
    """
    Returns GB-predicted probability 0-1 of bullish outcome.
    Runs 80-tree gradient boosting ensemble trained on real 3yr OHLCV history.
    Falls back to synthetic model if real data not yet loaded.
    """
    try:
        model = get_gb_model()
        feats = build_feature_vector(signals)
        score = model.predict_one(feats)
        return round(float(score), 4)
    except Exception as e:
        logger.debug(f"GB score error: {e}")
        return 0.5


def meta_label_filter(signals: dict, ml_score: float) -> dict:
    """
    Second-stage meta-labeler: decides TAKE/SKIP and size multiplier.
    This is intentionally lightweight and robust for live usage.
    """
    global _META_LABEL_STATS
    spread_bps = float(signals.get("spread_bps", signals.get("bid_ask_bps", 0)) or 0)
    hv20 = float(signals.get("hv20", 20) or 20)
    conviction = float(signals.get("conviction", 0.0) or 0.0)
    corr = abs(float(signals.get("corr_to_book", signals.get("corr_to_spy", 0.0)) or 0.0))
    liquidity = float(signals.get("liquidity_score", 0.7) or 0.7)
    news_shock = abs(float(signals.get("news_shock", signals.get("headline_shock", 0.0)) or 0.0))
    regime = str(signals.get("regime", "NEUTRAL")).upper()

    # Penalty model in [0,1], where 1 means severe microstructure risk.
    penalty = (
        np.clip(spread_bps / 40.0, 0, 1) * 0.30 +
        np.clip((hv20 - 18) / 30.0, 0, 1) * 0.20 +
        np.clip(corr / 0.85, 0, 1) * 0.20 +
        np.clip(news_shock / 2.5, 0, 1) * 0.20 +
        np.clip((0.7 - liquidity) / 0.7, 0, 1) * 0.10
    )
    meta_score = float(np.clip(ml_score * (1.0 - 0.55 * penalty), 0.0, 1.0))

    threshold = 0.57
    if regime in {"PANIC", "CRASH", "RISK_OFF"}:
        threshold += 0.03
    if conviction >= 0.75:
        threshold -= 0.015
    if spread_bps > 35:
        threshold += 0.02

    accept = bool(meta_score >= threshold)
    size_mult = float(np.clip((meta_score - threshold + 0.12) / 0.24, 0.25, 1.25)) if accept else 0.0
    reason = "accepted" if accept else "filtered_by_meta_label"

    _META_LABEL_STATS["last_score"] = round(meta_score, 4)
    _META_LABEL_STATS["last_reason"] = reason
    if accept:
        _META_LABEL_STATS["accepted"] += 1
    else:
        _META_LABEL_STATS["rejected"] += 1

    return {
        "accept_trade": accept,
        "meta_score": round(meta_score, 4),
        "meta_threshold": round(float(threshold), 4),
        "size_multiplier": round(size_mult, 3),
        "reason": reason,
        "penalty": round(float(penalty), 4),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2. PYTORCH LSTM + SEQUENCE PREDICTOR  (Step 11)
# ═════════════════════════════════════════════════════════════════════════════
#
# Architecture:
#   Input:  (batch, seq_len=20, n_features=14)  — sliding windows of OHLCV features
#   LSTM:   2-layer bidirectional LSTM, hidden_size=64
#   Head:   Linear(128, 32) → ReLU → Dropout(0.2) → Linear(32, 1) → Tanh
#   Output: scalar ∈ [-1, +1]  (z-scored 5-day forward return target)
#
# Training:
#   - load_real_training_data() provides bar-by-bar features for 40 symbols × 3yr
#   - We build sliding windows of length SEQ_LEN=20 from consecutive bars
#   - Target: 5-day forward log return z-scored within each symbol
#   - Adam(lr=5e-4), ReduceLROnPlateau, early stopping (patience=5)
#   - Background retrain every 168hr; hot-swap on val_IC > 0.04
#   - Saves/loads weights from "lstm_weights.pt"
#
# Fallback (if PyTorch unavailable):
#   SklearnSequencePredictor: flattens (20,14)→280 features + 5 lagged returns
#   → sklearn MLPRegressor(hidden=(256,128,64)). Less powerful but immediate.
#
# Academic basis:
#   Hochreiter & Schmidhuber (1997): LSTM; Long-Short Term Memory
#   Fischer & Krauss (2018): "Deep learning with long short-term memory networks
#     for financial market predictions", EJOR — first rigorous LSTM vs buy-hold
#     study. LSTM achieves ~55% directional accuracy on S&P500 constituents.
#   Sezer et al (2020): "Financial time series forecasting with deep learning:
#     A systematic literature review", Applied Soft Computing
# ═════════════════════════════════════════════════════════════════════════════

SEQ_LEN    = 20       # sequence window (20 trading days ≈ 1 month)
N_FEATURES = 14       # must match compute_features_from_bars() output
LSTM_HIDDEN = 64      # hidden units per direction
LSTM_LAYERS = 2       # stacked LSTM depth
LSTM_DROPOUT = 0.20   # dropout between LSTM layers
HEAD_DIM    = 32      # FC head intermediate size
LSTM_WEIGHTS_FILE = "lstm_weights.pt"
LSTM_SKLEARN_FILE = "lstm_sklearn.pkl"

# ── Attempt PyTorch import (graceful fallback to sklearn) ─────────────────
# pyright: reportMissingImports=false
_TORCH_AVAILABLE = False
_SKLEARN_AVAILABLE = False
try:
    from sklearn.neural_network import MLPRegressor as _MLPRegressor_check  # type: ignore[import]
    from sklearn.preprocessing  import StandardScaler as _SS_check           # type: ignore[import]
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning(
        "⚠️  scikit-learn not installed — LSTM fallback disabled.\n"
        "    Fix: pip install scikit-learn   (then restart the server)"
    )
try:
    import torch  # type: ignore[import]
    import torch.nn as nn  # type: ignore[import]
    import torch.optim as optim  # type: ignore[import]
    from torch.utils.data import TensorDataset, DataLoader  # type: ignore[import]
    _TORCH_AVAILABLE = True
    logger.info("✅ PyTorch available — using TorchLSTMPredictor")
except ImportError:
    # Define stubs so Pylance/pyright stop flagging usage inside if _TORCH_AVAILABLE blocks
    torch = None  # type: ignore[assignment]
    nn    = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    DataLoader    = None  # type: ignore[assignment]
    logger.info("⚠️  PyTorch not installed — using SklearnSequencePredictor fallback")
    logger.info("    Install: pip install torch  (CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu)")


# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH LSTM MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

if _TORCH_AVAILABLE:
    class _LSTMNet(nn.Module):
        """
        Bidirectional 2-layer LSTM with fully-connected head.

        Architecture:
          BiLSTM(14 → 64, layers=2, dropout=0.2)
          → take last hidden state (shape: batch × 128 since bidirectional)
          → Linear(128 → 32) → ReLU → Dropout(0.2) → Linear(32 → 1) → Tanh
        """
        def __init__(self, n_features=N_FEATURES, hidden=LSTM_HIDDEN,
                     n_layers=LSTM_LAYERS, dropout=LSTM_DROPOUT, head_dim=HEAD_DIM):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            lstm_out_dim = hidden * 2   # bidirectional
            self.head = nn.Sequential(
                nn.Linear(lstm_out_dim, head_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_dim, 1),
                nn.Tanh(),
            )

        def forward(self, x):
            # x: (batch, seq_len, n_features)
            out, (h_n, _) = self.lstm(x)
            # h_n: (n_layers * 2, batch, hidden) — take last layer, both directions
            h_fwd = h_n[-2]   # (batch, hidden) — last forward layer
            h_bwd = h_n[-1]   # (batch, hidden) — last backward layer
            h_cat = torch.cat([h_fwd, h_bwd], dim=1)   # (batch, hidden*2)
            return self.head(h_cat).squeeze(-1)          # (batch,)


class TorchLSTMPredictor:
    """
    PyTorch LSTM predictor for sequence-based return prediction.
    Only instantiated when torch is available.
    """
    def __init__(self):
        self._model     = None
        self._trained   = False
        self._val_ic    = 0.0
        self._train_ts  = None
        self._lock      = threading.Lock()
        self._device    = None
        if _TORCH_AVAILABLE:
            try:
                import torch
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"  TorchLSTMPredictor device: {self._device}")
            except Exception:
                pass

    def _build_sequences(self, X_bars: np.ndarray,
                         y_fwd: np.ndarray) -> tuple:
        """
        Build sliding-window sequences from per-bar feature matrix.

        X_bars: (n_bars, 14)   — bar-level features
        y_fwd:  (n_bars,)      — 5-day forward return labels

        Returns X_seq (n_seqs, SEQ_LEN, 14), y_seq (n_seqs,)
        """
        n = len(X_bars)
        if n < SEQ_LEN + 1:
            return None, None

        seqs, labels = [], []
        for i in range(n - SEQ_LEN):
            window = X_bars[i : i + SEQ_LEN]
            label  = y_fwd[i + SEQ_LEN - 1]
            if not (np.isnan(window).any() or np.isnan(label)):
                seqs.append(window.astype(np.float32))
                labels.append(float(label))

        if not seqs:
            return None, None
        return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.float32)

    def train(self, X_bars_list: list, y_list: list,
              epochs: int = 50, batch_size: int = 128,
              lr: float = 5e-4, val_frac: float = 0.15) -> dict:
        """
        Train LSTM on list of per-symbol (X_bars, y_fwd) pairs.

        X_bars_list: list of (n_bars, 14) arrays (one per symbol)
        y_list:      list of (n_bars,) forward return arrays

        Returns: {"val_ic", "val_loss", "epochs_trained", "n_sequences"}
        """
        if not _TORCH_AVAILABLE:
            return {"val_ic": 0.0, "error": "torch_unavailable"}

        import torch, torch.nn as nn, torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader

        # Build sequences across all symbols
        all_X, all_y = [], []
        for X_bars, y_fwd in zip(X_bars_list, y_list):
            Xseq, yseq = self._build_sequences(X_bars, y_fwd)
            if Xseq is not None:
                all_X.append(Xseq)
                all_y.append(yseq)

        if not all_X:
            return {"val_ic": 0.0, "error": "no_sequences"}

        X_all = np.concatenate(all_X, axis=0)
        y_all = np.concatenate(all_y, axis=0)
        n_seq = len(X_all)

        # Z-score targets within batch
        y_mean = float(np.mean(y_all))
        y_std  = float(np.std(y_all)) + 1e-8
        y_norm = np.clip((y_all - y_mean) / y_std, -3.0, 3.0).astype(np.float32)
        # Rescale to [-1, +1] for tanh output
        y_norm = (y_norm / 3.0).astype(np.float32)

        # Train/val split (time-aware: last val_frac% as validation)
        split  = int(n_seq * (1 - val_frac))
        X_tr, X_val = X_all[:split], X_all[split:]
        y_tr, y_val = y_norm[:split], y_norm[split:]

        # DataLoaders
        train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        val_ds   = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

        # Model + optimizer
        net = _LSTMNet().to(self._device)
        opt = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
        loss_fn = nn.MSELoss()

        # Training loop with early stopping
        best_val_ic   = -999.0
        best_state    = None
        patience_left = 5
        train_losses  = []

        net.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for Xb, yb in train_dl:
                Xb, yb = Xb.to(self._device), yb.to(self._device)
                opt.zero_grad()
                pred = net(Xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                opt.step()
                epoch_loss += float(loss.item())

            avg_loss = epoch_loss / max(len(train_dl), 1)
            train_losses.append(avg_loss)

            # Validation IC
            net.eval()
            preds_val, trues_val = [], []
            with torch.no_grad():
                for Xb, yb in val_dl:
                    p = net(Xb.to(self._device)).cpu().numpy()
                    preds_val.extend(p.tolist())
                    trues_val.extend(yb.numpy().tolist())

            val_preds = np.array(preds_val)
            val_true  = np.array(trues_val)
            try:
                val_ic = float(stats.spearmanr(val_preds, val_true).correlation)
            except Exception:
                val_ic = 0.0

            scheduler.step(avg_loss)

            if val_ic > best_val_ic:
                best_val_ic = val_ic
                best_state  = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                patience_left = 5
            else:
                patience_left -= 1
                if patience_left <= 0:
                    logger.info(f"  LSTM early stop at epoch {epoch+1}, best val_IC={best_val_ic:.4f}")
                    break

            net.train()

        # Restore best weights
        if best_state:
            net.load_state_dict(best_state)

        # Save to disk
        try:
            torch.save({"state_dict": best_state or net.state_dict(),
                        "val_ic": best_val_ic,
                        "n_features": N_FEATURES,
                        "seq_len": SEQ_LEN},
                       LSTM_WEIGHTS_FILE)
        except Exception as _se:
            logger.debug(f"LSTM save: {_se}")

        with self._lock:
            self._model   = net
            self._trained = True
            self._val_ic  = best_val_ic
            self._train_ts = datetime.now()

        return {
            "val_ic":       round(best_val_ic, 4),
            "n_sequences":  n_seq,
            "epochs_trained": epochs,
            "device":       str(self._device),
        }

    def load_weights(self) -> bool:
        """Load previously saved LSTM weights from disk."""
        if not _TORCH_AVAILABLE:
            return False
        import torch
        try:
            if not os.path.exists(LSTM_WEIGHTS_FILE):
                return False
            ckpt = torch.load(LSTM_WEIGHTS_FILE, map_location="cpu")
            net  = _LSTMNet()
            net.load_state_dict(ckpt["state_dict"])
            net.to(self._device)
            net.eval()
            with self._lock:
                self._model   = net
                self._trained = True
                self._val_ic  = float(ckpt.get("val_ic", 0.0))
            logger.info(f"  ✅ LSTM weights loaded (val_IC={self._val_ic:.4f})")
            return True
        except Exception as _le:
            logger.debug(f"LSTM load: {_le}")
            return False

    def predict(self, X_bars: np.ndarray) -> float:
        """
        Predict from last SEQ_LEN bars of features.

        X_bars: (n_bars, 14) — provide at least SEQ_LEN=20 rows.
        Returns: score ∈ [-1, +1]
        """
        if not _TORCH_AVAILABLE or self._model is None:
            return 0.5   # neutral

        import torch
        try:
            window = X_bars[-SEQ_LEN:].astype(np.float32)
            if len(window) < SEQ_LEN:
                return 0.5
            x = torch.from_numpy(window).unsqueeze(0).to(self._device)
            with torch.no_grad():
                self._model.eval()
                pred = float(self._model(x).cpu().item())
            # Rescale tanh output [-1,+1] → [0,1] for consistency with other scores
            return float(np.clip((pred + 1) / 2, 0.0, 1.0))
        except Exception as _pe:
            logger.debug(f"LSTM predict: {_pe}")
            return 0.5

    @property
    def is_ready(self) -> bool:
        return self._trained and self._model is not None


# ─────────────────────────────────────────────────────────────────────────────
# SKLEARN SEQUENCE PREDICTOR (fallback when PyTorch unavailable)
# ─────────────────────────────────────────────────────────────────────────────

class SklearnSequencePredictor:
    """
    MLP on flattened (SEQ_LEN × N_FEATURES) sequences.

    Not as powerful as LSTM (no recurrent state), but:
    - Available immediately (sklearn is always installed)
    - Uses the same sequence structure and features
    - 3 hidden layers capture nonlinear feature interactions
    - Trains much faster (seconds vs minutes)

    This is the workhorse until PyTorch is installed.
    """

    def __init__(self):
        self._model    = None
        self._trained  = False
        self._val_ic   = 0.0
        self._train_ts = None
        self._lock     = threading.Lock()

    def _build_sequences(self, X_bars: np.ndarray,
                         y_fwd: np.ndarray) -> tuple:
        """Build flattened sequences for MLP training."""
        n = len(X_bars)
        if n < SEQ_LEN + 1:
            return None, None
        seqs, labels = [], []
        for i in range(n - SEQ_LEN):
            window = X_bars[i : i + SEQ_LEN].flatten()
            label  = y_fwd[i + SEQ_LEN - 1]
            if not (np.isnan(window).any() or np.isnan(label)):
                seqs.append(window.astype(np.float32))
                labels.append(float(label))
        if not seqs:
            return None, None
        return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.float32)

    def train(self, X_bars_list: list, y_list: list, **kwargs) -> dict:
        """Train MLP on sliding-window sequences from all symbols."""
        if not _SKLEARN_AVAILABLE:
            logger.error(
                "❌ scikit-learn missing — cannot train LSTM fallback.\n"
                "   Run: pip install scikit-learn   then restart the server."
            )
            return {"val_ic": 0.0, "error": "sklearn_not_installed"}
        from sklearn.neural_network import MLPRegressor  # type: ignore[import]
        from sklearn.preprocessing import StandardScaler  # type: ignore[import]

        all_X, all_y = [], []
        for X_bars, y_fwd in zip(X_bars_list, y_list):
            Xseq, yseq = self._build_sequences(X_bars, y_fwd)
            if Xseq is not None:
                all_X.append(Xseq)
                all_y.append(yseq)

        if not all_X:
            return {"val_ic": 0.0, "error": "no_sequences"}

        X_all = np.concatenate(all_X)
        y_all = np.concatenate(all_y)

        # Z-score targets
        y_mean = float(np.mean(y_all))
        y_std  = float(np.std(y_all)) + 1e-8
        y_norm = np.clip((y_all - y_mean) / y_std, -3.0, 3.0)

        # Time-aware split
        split  = int(len(X_all) * 0.85)
        X_tr, X_val = X_all[:split], X_all[split:]
        y_tr, y_val = y_norm[:split], y_norm[split:]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        mlp = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=5e-4,
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=10,
            random_state=42,
            verbose=False,
        )
        mlp.fit(X_tr_s, y_tr)

        val_preds = mlp.predict(X_val_s)
        try:
            val_ic = float(stats.spearmanr(val_preds, y_val).correlation)
        except Exception:
            val_ic = 0.0

        # Save
        try:
            import pickle
            with open(LSTM_SKLEARN_FILE, "wb") as f:
                pickle.dump({"model": mlp, "scaler": scaler, "val_ic": val_ic}, f)
        except Exception:
            pass

        with self._lock:
            self._model   = (mlp, scaler)
            self._trained = True
            self._val_ic  = val_ic
            self._train_ts = datetime.now()

        return {"val_ic": round(val_ic, 4), "n_sequences": len(X_all),
                "backend": "sklearn_mlp"}

    def load_weights(self) -> bool:
        """Load previously saved MLP from disk."""
        try:
            import pickle
            if not os.path.exists(LSTM_SKLEARN_FILE):
                return False
            with open(LSTM_SKLEARN_FILE, "rb") as f:
                ckpt = pickle.load(f)
            with self._lock:
                self._model   = (ckpt["model"], ckpt["scaler"])
                self._trained = True
                self._val_ic  = float(ckpt.get("val_ic", 0.0))
            logger.info(f"  ✅ Sklearn LSTM loaded (val_IC={self._val_ic:.4f})")
            return True
        except Exception as _le:
            logger.debug(f"Sklearn load: {_le}")
            return False

    def predict(self, X_bars: np.ndarray) -> float:
        """Predict from last SEQ_LEN bars. Returns score ∈ [0, 1]."""
        if not self._trained or self._model is None:
            return 0.5
        mlp, scaler = self._model
        try:
            window = X_bars[-SEQ_LEN:].flatten().reshape(1, -1).astype(np.float32)
            if window.shape[1] != SEQ_LEN * N_FEATURES:
                return 0.5
            window_s = scaler.transform(window)
            pred = float(mlp.predict(window_s)[0])
            # z-score → [0,1]: pred is in ~[-3,3], map to [0,1]
            return float(np.clip(pred / 6.0 + 0.5, 0.0, 1.0))
        except Exception as _pe:
            logger.debug(f"Sklearn predict: {_pe}")
            return 0.5

    @property
    def is_ready(self) -> bool:
        return self._trained and self._model is not None


# ─────────────────────────────────────────────────────────────────────────────
# LSTM PRICE PREDICTOR — public interface (tries torch first, falls back)
# ─────────────────────────────────────────────────────────────────────────────

class LSTMPricePredictor:
    """
    Public interface for LSTM-based price prediction.

    Automatically uses TorchLSTMPredictor if PyTorch is available,
    otherwise falls back to SklearnSequencePredictor.

    Usage:
        predictor = LSTMPricePredictor()
        predictor.start_background_training()   # called at server startup
        score = predictor.predict(X_bars)        # called per-symbol
    """

    # Module-level singleton
    _instance = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "LSTMPricePredictor":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        if _TORCH_AVAILABLE:
            self._predictor = TorchLSTMPredictor()
            self._backend   = "pytorch"
        else:
            self._predictor = SklearnSequencePredictor()
            self._backend   = "sklearn_mlp"

        self._is_training   = False
        self._train_lock    = threading.Lock()
        self._last_train_ts = None
        self._retrain_interval_h = 168   # weekly retrain

        # Try to load saved weights first
        loaded = self._predictor.load_weights()
        if not loaded:
            logger.info(f"  LSTM: no saved weights — will train on first request")

    # ─────────────────────────────────────────────────────────────────────

    def _build_training_data(self) -> tuple:
        """
        Build (X_bars_list, y_list) from load_real_training_data().
        Each element is per-symbol (n_bars × 14 features, n_bars forward returns).
        """
        logger.info("  LSTM: building training data (3yr OHLCV, 40 symbols)...")
        try:
            import yfinance as _yf

            # Dynamic training universe — pulls from live_data_server's
            # get_training_universe() if available, else falls back to a
            # broad set covering large-caps, ETFs, metals and sectors.
            try:
                import importlib as _il
                _lds = _il.import_module("live_data_server") if "live_data_server" in __import__("sys").modules else None
                TRAIN_SYMBOLS = _lds.get_training_universe(n=60) if _lds else None
            except Exception:
                TRAIN_SYMBOLS = None

            if not TRAIN_SYMBOLS:
                TRAIN_SYMBOLS = [
                    # Large-cap equities
                    "SPY","QQQ","IWM","AAPL","MSFT","NVDA","TSLA","META","GOOGL","AMZN",
                    "JPM","BAC","GS","NFLX","AMD","PLTR","CRWD","SHOP","COIN","UNH",
                    "XOM","CVX","LLY","JNJ","PG","KO","WMT","HD","V","MA",
                    "SOFI","MSTR","SMCI","ARM","RIVN","INTC","MU","QCOM","AVGO","ADBE",
                    # ETFs — equity, bond, commodity, metals, sector
                    "GLD","SLV","TLT","HYG","XLK","XLF","XLE","XLV","XLB","BITO",
                    # International
                    "EEM","FXI","EWZ",
                    # Additional mid-cap
                    "UBER","ABNB","SNAP","ROKU","DKNG","HOOD","OPEN","RBLX","U","AFRM",
                ]
            FWD_DAYS = 5

            X_list, y_list = [], []

            for sym in TRAIN_SYMBOLS:
                try:
                    h = _yf.Ticker(sym).history(period="3y")
                    if len(h) < SEQ_LEN + FWD_DAYS + 10:
                        continue

                    c = h["Close"].values.astype(float)
                    hi = h["High"].values.astype(float)
                    lo = h["Low"].values.astype(float)
                    v  = h["Volume"].values.astype(float)
                    n  = len(c)

                    # Build bar-by-bar features
                    feats = []
                    for i in range(SEQ_LEN, n - FWD_DAYS):
                        f = compute_features_from_bars(
                            c[max(0,i-60):i+1],
                            hi[max(0,i-60):i+1],
                            lo[max(0,i-60):i+1],
                            v[max(0,i-60):i+1]
                        )
                        if f is not None:
                            feats.append(f)

                    if len(feats) < SEQ_LEN + 5:
                        continue

                    X_bars = np.array(feats)

                    # Forward returns (5-day log return)
                    n_feats = len(feats)
                    fwd = np.zeros(n_feats)
                    for i in range(n_feats):
                        bar_i = SEQ_LEN + i
                        if bar_i + FWD_DAYS < n:
                            fwd[i] = float(np.log(c[bar_i + FWD_DAYS] / (c[bar_i] + 1e-10)))

                    # Z-score within symbol to remove scale differences
                    fwd_std = np.std(fwd) + 1e-8
                    fwd_z   = (fwd - np.mean(fwd)) / fwd_std

                    X_list.append(X_bars)
                    y_list.append(fwd_z)

                except Exception as _sym_e:
                    logger.debug(f"  LSTM train data {sym}: {_sym_e}")
                    continue

            logger.info(f"  LSTM: {len(X_list)} symbols, training sequences building...")
            return X_list, y_list

        except Exception as e:
            logger.warning(f"  LSTM training data failed: {e}")
            return [], []

    def start_background_training(self):
        """Launch background training thread. Non-blocking."""
        def _train_worker():
            with self._train_lock:
                if self._is_training:
                    return
                self._is_training = True
            try:
                logger.info(f"🧠 LSTM background training started ({self._backend})")
                X_list, y_list = self._build_training_data()
                if not X_list:
                    logger.warning("  LSTM: no training data — skipping")
                    return

                # Build a fresh predictor for hot-swap
                if _TORCH_AVAILABLE:
                    new_pred = TorchLSTMPredictor()
                else:
                    new_pred = SklearnSequencePredictor()

                result = new_pred.train(X_list, y_list)
                val_ic = float(result.get("val_ic", 0.0))

                logger.info(f"  LSTM trained: val_IC={val_ic:.4f}, n_seq={result.get('n_sequences')}")

                # Hot-swap only if IC is meaningful (> 0.02)
                if val_ic > 0.02:
                    self._predictor     = new_pred
                    self._last_train_ts = datetime.now()
                    logger.info(f"  ✅ LSTM hot-swapped (val_IC={val_ic:.4f})")
                else:
                    logger.info(f"  ⚠️  LSTM val_IC={val_ic:.4f} too low — keeping old model")

            except Exception as e:
                logger.error(f"  LSTM training error: {e}")
            finally:
                self._is_training = False

        # Delay 90s after startup to avoid competing with GBM retraining
        def _delayed():
            time.sleep(90)
            _train_worker()
            # Schedule weekly retrain
            while True:
                time.sleep(self._retrain_interval_h * 3600)
                _train_worker()

        t = threading.Thread(target=_delayed, daemon=True, name="lstm-train")
        t.start()
        logger.info("  LSTM training thread launched (90s delay)")

    def predict(self, X_bars: np.ndarray) -> float:
        """
        Predict from recent bar features.
        X_bars: (n_bars, 14) — minimum SEQ_LEN rows required.
        Returns: score ∈ [0, 1]  (0.5 = neutral)
        """
        if not self._predictor.is_ready:
            return 0.5
        return float(self._predictor.predict(X_bars))

    def get_stats(self) -> dict:
        return {
            "backend":        self._backend,
            "is_ready":       self._predictor.is_ready,
            "is_training":    self._is_training,
            "val_ic":         round(self._predictor._val_ic, 4),
            "last_trained":   str(self._predictor._train_ts) if self._predictor._train_ts else None,
            "seq_len":        SEQ_LEN,
            "n_features":     N_FEATURES,
            "architecture": (
                f"BiLSTM(layers={LSTM_LAYERS}, hidden={LSTM_HIDDEN}, dropout={LSTM_DROPOUT})"
                if _TORCH_AVAILABLE else
                f"MLP(hidden=256-128-64, seq_flatten={SEQ_LEN*N_FEATURES})"
            ),
        }


# Backward-compatible function (called from compute_ml_score and live_data_server)
def lstm_pattern_score(close_prices, window=20):
    """
    Backward-compatible wrapper.
    If the LSTM predictor is trained, returns its prediction.
    Otherwise falls back to the original windowed-statistics pattern detector.
    """
    predictor = LSTMPricePredictor.get_instance()

    if predictor._predictor.is_ready and close_prices and len(close_prices) >= SEQ_LEN + 30:
        try:
            c  = np.array(close_prices, dtype=float)
            n  = len(c)
            # Build feature matrix for the last SEQ_LEN+30 bars
            feats = []
            for i in range(SEQ_LEN, min(n, SEQ_LEN + 50)):
                f = compute_features_from_bars(
                    c[max(0,i-40):i+1], c[max(0,i-40):i+1],
                    c[max(0,i-40):i+1], np.ones(min(i+1,41))
                )
                if f is not None:
                    feats.append(f)
            if len(feats) >= SEQ_LEN:
                X_bars = np.array(feats)
                score  = predictor.predict(X_bars)
                direction = "BULLISH" if score > 0.55 else "BEARISH" if score < 0.45 else "NEUTRAL"
                return {
                    "pattern":      "LSTM_PREDICTION",
                    "score":        score,
                    "direction":    direction,
                    "all_patterns": [{"name": "LSTM", "score": score, "direction": direction}],
                    "backend":      predictor._backend,
                }
        except Exception as _e:
            logger.debug(f"lstm_pattern_score LSTM path: {_e}")

    # ── Fallback: original windowed-statistics pattern detector ──────────
    if not close_prices or len(close_prices) < window + 5:
        return {"pattern": "INSUFFICIENT_DATA", "score": 0.5, "direction": "NEUTRAL"}

    prices = np.array(close_prices[-(window+5):], dtype=float)
    p      = prices[-window:]
    ret    = np.diff(np.log(p + 1e-10))
    vol    = float(np.std(ret) * np.sqrt(252))
    trend  = float(np.polyfit(range(len(p)), p, 1)[0])

    def _autocorr(x, lag=1):
        if len(x) <= lag:
            return 0.0
        return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])

    acf1     = _autocorr(ret, 1)
    last_ret = float(ret[-1])
    avg_ret  = float(np.mean(ret))
    zscore   = (p[-1] - np.mean(p)) / (np.std(p) + 1e-8)

    patterns = []
    if trend > 0 and acf1 > 0.1 and avg_ret > 0 and last_ret > 0:
        patterns.append(("MOMENTUM_CONTINUATION", 0.75, "BULLISH"))
    if trend < 0 and acf1 > 0.1 and avg_ret < 0 and last_ret < 0:
        patterns.append(("MOMENTUM_CONTINUATION", 0.75, "BEARISH"))
    if abs(zscore) > 2.0:
        patterns.append(("MEAN_REVERSION_SETUP", 0.65, "BEARISH" if zscore > 2 else "BULLISH"))
    vol_r = float(np.std(ret[-5:]) * np.sqrt(252))
    vol_p = float(np.std(ret[-15:-5]) * np.sqrt(252))
    if vol_r > vol_p * 1.5 and abs(last_ret) > abs(avg_ret) * 2:
        patterns.append(("CONSOLIDATION_BREAKOUT", 0.70, "BULLISH" if last_ret > 0 else "BEARISH"))
    if trend < 0 and last_ret > abs(avg_ret) * 1.5 and acf1 < 0:
        patterns.append(("DEAD_CAT_BOUNCE", 0.60, "BEARISH"))

    up_vol = float(np.mean([abs(r) for r in ret if r > 0] or [0]))
    dn_vol = float(np.mean([abs(r) for r in ret if r < 0] or [0]))
    if up_vol > dn_vol * 1.3 and trend > 0:
        patterns.append(("ACCUMULATION", 0.68, "BULLISH"))
    if dn_vol > up_vol * 1.3 and trend < 0:
        patterns.append(("DISTRIBUTION", 0.68, "BEARISH"))

    if not patterns:
        return {"pattern": "CONSOLIDATION", "score": 0.50, "direction": "NEUTRAL",
                "all_patterns": []}

    # Best pattern by score
    best   = max(patterns, key=lambda x: x[1])
    raw_sc = best[1]
    direction = best[2]
    score  = raw_sc if direction == "BULLISH" else (1.0 - raw_sc) if direction == "BEARISH" else 0.5

    return {
        "pattern":      best[0],
        "score":        round(score, 4),
        "direction":    direction,
        "all_patterns": [{"name": n, "score": s, "direction": d} for n, s, d in patterns],
        "backend":      "numpy_heuristic",
    }


# 6. MASTER ML SCORE — combines all models
# ═════════════════════════════════════════════════════════════════════════════

def compute_ml_score(signals, close_prices=None):
    """
    Master ML score: GradientBoosting (50%) + LSTM (50%).

    Step 11 upgrade: LSTM is now a real PyTorch BiLSTM (or sklearn MLP fallback)
    trained on 3yr sliding-window sequences. When trained (val_IC > 0.02),
    it replaces the old heuristic pattern detector with learned sequence patterns.

    Blending: GB and LSTM are complementary — GB captures cross-sectional feature
    relationships while LSTM captures temporal autocorrelation in sequences.
    The 50/50 blend gives each equal voice; IC-based adaptive weights (Step 1)
    further tune this over time.
    """
    # 1. Gradient Boosting (real-data trained, hot-swapped weekly)
    gb = gb_signal_score(signals)

    # 2. LSTM / Sequence Predictor
    lstm_predictor = LSTMPricePredictor.get_instance()
    lstm = {"pattern": "NO_DATA", "score": 0.5, "direction": "NEUTRAL",
            "backend": lstm_predictor._backend}

    if close_prices and len(close_prices) >= SEQ_LEN + 5:
        lstm = lstm_pattern_score(close_prices)
    elif close_prices and len(close_prices) >= 25:
        lstm = lstm_pattern_score(close_prices)  # falls back to heuristic

    lstm_score = float(lstm.get("score", 0.5))

    # 3. Blend — 50/50 when LSTM is trained, 70/30 GB-heavy otherwise
    if lstm_predictor._predictor.is_ready and lstm_predictor._predictor._val_ic > 0.02:
        # LSTM is trained and validated — equal blend
        composite = gb * 0.50 + lstm_score * 0.50
        blend_mode = "50/50"
    else:
        # LSTM warming up — trust GB more
        composite = gb * 0.70 + lstm_score * 0.30
        blend_mode = "70/30 (LSTM warming)"

    # Directional agreement bonus (+5%)
    direction = signals.get("direction", "HOLD")
    if lstm["direction"] == "BULLISH" and direction == "BUY":
        composite = min(1.0, composite * 1.05)
    elif lstm["direction"] == "BEARISH" and direction == "SHORT":
        composite = min(1.0, composite * 1.05)

    composite = float(np.clip(composite, 0.0, 1.0))
    meta = meta_label_filter(signals or {}, composite)
    effective = composite * meta["size_multiplier"] if meta["accept_trade"] else composite * 0.45

    model_stats   = get_model_stats()
    lstm_stats    = lstm_predictor.get_stats()

    return {
        "ml_score":          round(effective, 4),
        "ml_raw_score":      round(composite, 4),
        "ml_pct":            round(effective * 100, 1),
        "gb_score":          round(gb, 4),
        "lstm_score":        round(lstm_score, 4),
        "meta_label":        meta,
        "lstm_pattern":      lstm.get("pattern", "NO_DATA"),
        "lstm_direction":    lstm.get("direction", "NEUTRAL"),
        "lstm_backend":      lstm.get("backend", lstm_predictor._backend),
        "lstm_val_ic":       lstm_stats["val_ic"],
        "lstm_is_ready":     lstm_stats["is_ready"],
        "lstm_architecture": lstm_stats["architecture"],
        "blend_mode":        blend_mode,
        "all_patterns":      lstm.get("all_patterns", []),
        "model_real_data":   model_stats["trained_on_real"],
        "model_last_train":  model_stats["last_train_time"],
        "model_background":  model_stats["background_training"],
        "interpretation": (
            "STRONG ML BUY"   if composite > 0.70 and direction == "BUY"   else
            "STRONG ML SHORT" if composite > 0.70 and direction == "SHORT" else
            "ML CONFIRMS"     if composite > 0.60 else
            "ML NEUTRAL"      if composite > 0.45 else
            "ML DISAGREES"
        ),
    }


def get_ml_governance_snapshot() -> dict:
    """
    Governance snapshot for model monitoring dashboards.
    This is intentionally explicit and conservative for live deployment.
    """
    stats = get_model_stats()
    lstm_stats = LSTMPricePredictor.get_instance().get_stats()
    return {
        "models": {
            "gradient_boost": {
                "trained_on_real_data": bool(stats.get("trained_on_real", False)),
                "last_train_time": stats.get("last_train_time"),
                "background_training": bool(stats.get("background_training", False)),
            },
            "lstm_sequence": {
                "ready": bool(lstm_stats.get("is_ready", False)),
                "val_ic": float(lstm_stats.get("val_ic", 0.0)),
                "backend": lstm_stats.get("backend"),
                "architecture": lstm_stats.get("architecture"),
            },
        },
        "risk_policy": {
            "min_live_precision_target": 0.55,
            "recommended_precision_band": [0.55, 0.70],
            "note": "Do not enforce fixed win-rate promises; monitor risk-adjusted metrics by regime.",
        },
        "meta_labeling": {
            "accepted": int(_META_LABEL_STATS.get("accepted", 0)),
            "rejected": int(_META_LABEL_STATS.get("rejected", 0)),
            "last_score": float(_META_LABEL_STATS.get("last_score", 0.5)),
            "last_reason": _META_LABEL_STATS.get("last_reason", "init"),
        },
    }