"""
=============================================================================
RENAISSANCE.IO — Configuration
=============================================================================
ALL secrets come from environment variables — never hardcode keys here.

Setup:
  1. Copy .env.example → .env
  2. Fill in your keys in .env
  3. Run:  pip install python-dotenv
  4. .env is loaded automatically at server start via python-dotenv
=============================================================================
"""

import os
from pathlib import Path

# ── Auto-load .env if present (python-dotenv) ────────────────────────────────
try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]
    # Walk up from this file's directory to find .env
    _here = Path(__file__).resolve().parent
    for _p in [_here, _here.parent, _here.parent.parent]:
        _env = _p / ".env"
        if _env.exists():
            load_dotenv(_env)
            break
except ImportError:
    pass  # python-dotenv not installed — keys must be set as real env vars


# ─────────────────────────────────────────────────────────────────────────────
# API KEYS — all read from environment, never hardcoded
# ─────────────────────────────────────────────────────────────────────────────
ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY",    "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL   = os.environ.get("ALPACA_BASE_URL",   "https://paper-api.alpaca.markets")
ALPACA_DATA_URL   = os.environ.get("ALPACA_DATA_URL",   "https://data.alpaca.markets")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
NEWS_API_KEY      = os.environ.get("NEWS_API_KEY",       "")
FRED_API_KEY      = os.environ.get("FRED_API_KEY",       "")
POLYGON_API_KEY   = os.environ.get("POLYGON_API_KEY",    "")

# ── Key validation helpers ────────────────────────────────────────────────────
def _key_ok(key: str) -> bool:
    return bool(key) and not key.startswith("YOUR_")

def check_keys() -> dict:
    """Return a dict of which keys are configured (True/False)."""
    return {
        "alpaca":     _key_ok(ALPACA_API_KEY) and _key_ok(ALPACA_SECRET_KEY),
        "anthropic":  _key_ok(ANTHROPIC_API_KEY),
        "news":       _key_ok(NEWS_API_KEY),
        "fred":       _key_ok(FRED_API_KEY),
        "polygon":    _key_ok(POLYGON_API_KEY),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TIMING
# ─────────────────────────────────────────────────────────────────────────────
LOOP_INTERVAL_SECONDS  = 75           # Scan interval (75s default — yfinance-safe)
MARKET_OPEN            = "09:30"
MARKET_CLOSE           = "16:00"
PRE_MARKET_OPEN        = "04:00"
AFTER_HOURS_CLOSE      = "20:00"

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL WEIGHTS  (overridden at runtime by AladdinScorer adaptive IC weights)
# These are the cold-start defaults only — not used after first IC update.
# ─────────────────────────────────────────────────────────────────────────────
SIGNAL_WEIGHTS = {
    "technical_composite":  0.15,
    "advanced_composite":   0.10,
    "ml_score":             0.18,
    "hurst_blended":        0.10,
    "ou_reversion":         0.08,
    "kalman_trend":         0.07,
    "vol_regime":           0.04,
    "news_sentiment":       0.10,
    "options_flow":         0.07,
    "macro_regime":         0.04,
    # Level-2 microstructure signals (Gap 3 partial fix)
    "l2_spread":            0.03,
    "l2_imbalance":         0.02,
    "l2_vwap_dev":          0.02,
}

SIGNAL_THRESHOLD    = 0.65     # Composite score must exceed this to trade
SHORT_THRESHOLD     = 0.30     # Below this → short signal (if enabled)

# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
MAX_PORTFOLIO_RISK    = 0.02    # 2% max daily loss
MAX_POSITION_SIZE     = 0.05    # 5% max per position
MAX_SECTOR_EXPOSURE   = 0.25    # 25% max per sector
MAX_CORRELATION       = 0.70    # Max correlation between positions
KELLY_FRACTION        = 0.25    # Fractional Kelly (conservative)
MAX_DRAWDOWN_STOP     = 0.05    # 5% drawdown → circuit breaker pause
DRAWDOWN_DELEVER_PCT  = 0.03    # 3% drawdown → start reducing position sizes
STOP_LOSS_PCT         = 0.02    # 2% stop loss per trade
TAKE_PROFIT_PCT       = 0.50    # 50% take profit per trade
MAX_OPEN_POSITIONS    = 5       # Max simultaneous positions
VAR_CONFIDENCE        = 0.95    # 95% VaR
PORTFOLIO_OPTIM       = True    # Enable mean-variance portfolio optimization

# ─────────────────────────────────────────────────────────────────────────────
# ML MODEL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
LOOKBACK_PERIODS      = 60      # Days of history for ML features
PREDICTION_HORIZON    = 5       # Predict 5-day forward return
ML_RETRAIN_HOURS      = 168     # Retrain models weekly (168 hours)
LSTM_SEQUENCE_LEN     = 30      # LSTM lookback window
MIN_TRAINING_SAMPLES  = 500     # Minimum samples to train

# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATOR PARAMS
# ─────────────────────────────────────────────────────────────────────────────
RSI_PERIOD       = 14
MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9
BB_PERIOD        = 20
BB_STD           = 2.0
ATR_PERIOD       = 14
EMA_SHORT        = 9
EMA_LONG         = 21
ADX_PERIOD       = 14
VWAP_PERIOD      = 1            # Daily VWAP

# ─────────────────────────────────────────────────────────────────────────────
# DATA FREQUENCY  (audit Gap 1 — highest-impact change)
# ─────────────────────────────────────────────────────────────────────────────
# "1d"  = daily bars (default — stable, works on free yfinance)
# "5m"  = 5-minute bars (recommended — 78x more observations, stronger IC)
# "1m"  = 1-minute bars (max resolution — requires Alpaca WebSocket)
BAR_INTERVAL          = "1d"    # Change to "5m" or "1m" to unlock intraday signals

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
LOG_DIR           = "logs"
LOG_LEVEL         = "INFO"
DASHBOARD_PORT    = 3000
SERVER_PORT       = 5001