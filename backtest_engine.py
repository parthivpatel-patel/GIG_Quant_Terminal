"""
=============================================================================
RENAISSANCE MEDALLION — MINIMUM VIABLE BACKTEST ENGINE  (Step 6)
=============================================================================

Architecture:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  BacktestEngine                                                     │
  │    ├── DataLoader         — yfinance OHLCV + feature computation   │
  │    ├── SignalComputer     — recomputes all 14 features per bar     │
  │    ├── WalkForwardRunner  — expanding-window train/test splits     │
  │    ├── TransactionCosts   — bid-ask spread + commission + impact   │
  │    ├── PortfolioSimulator — position sizing + equity curve         │
  │    └── PerformanceStats   — Sharpe, Calmar, IC, win-rate, etc.    │
  └─────────────────────────────────────────────────────────────────────┘

Walk-Forward Design (expanding window):
  ─────────────────────────────────────────────────────────────────────
  |<───── train ─────────>|<── test ──>|
                          |<─────── train ──────────>|<── test ──>|
                                        ...
  ─────────────────────────────────────────────────────────────────────
  Train: minimum 252 bars (1 year)
  Test:  63 bars (1 quarter)
  Overlap: 0 (strict OOS — test bars never leak into training)

Transaction Cost Model (3-layer):
  1. Bid-ask spread:  0.05% (large cap), 0.15% (mid), 0.30% (small/penny)
  2. Commission:      $1.00 per trade (typical broker)
  3. Market impact:   Almgren-Chriss proxy = 0.1% * sqrt(trade_size / adv)

Signal IC Analysis:
  Spearman rank IC computed per signal per out-of-sample window.
  EWMA IC tracked over rolling windows to detect signal decay.

Output:
  equity_curve:   list of {date, portfolio_value, drawdown}
  trades:         list of {symbol, entry, exit, pnl, signal_scores}
  ic_per_signal:  {signal_name: {ic, t_stat, p_value}}
  performance:    {sharpe, calmar, max_dd, win_rate, avg_hold, ...}
=============================================================================
"""

import numpy as np
import math
import logging
import threading
import time
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger("BACKTEST")

TRADING_DAYS = 252


# ─────────────────────────────────────────────────────────────────────────────
# 1. TRANSACTION COST MODEL
# ─────────────────────────────────────────────────────────────────────────────

class TransactionCostModel:
    """
    Three-layer transaction cost model.

    Layer 1 — Bid-ask spread (market microstructure):
        Large cap (>$10B MC):   ~0.03–0.08%
        Mid cap ($2–10B):       ~0.10–0.20%
        Small cap (<$2B):       ~0.20–0.40%
        We use conservative midpoints.

    Layer 2 — Commission:
        $1.00 flat per order (competitive broker rate)
        Expressed as % of trade value → declines with position size

    Layer 3 — Market impact (Almgren-Chriss simplified):
        impact_pct = η * sqrt(trade_value / ADV)
        η = 0.1 (empirical from institutional literature)
        ADV = 20-day average dollar volume
        Significant only for large positions relative to liquidity.
    """

    # Spread by cap category
    SPREAD_BPS = {
        "LARGE":  4,    # 0.04%
        "MID":    12,   # 0.12%
        "SMALL":  25,   # 0.25%
        "PENNY":  50,   # 0.50%
        "ETF":    2,    # 0.02%
    }
    COMMISSION_PER_TRADE = 1.00    # USD
    IMPACT_ETA           = 0.10    # Almgren-Chriss η

    @classmethod
    def cost_pct(cls, trade_value: float, cap_category: str = "LARGE",
                 adv: float = 1e8) -> float:
        """
        Total one-way transaction cost as fraction of trade value.

        Args:
            trade_value:  Dollar value of the trade
            cap_category: LARGE / MID / SMALL / PENNY / ETF
            adv:          20-day average dollar volume

        Returns:
            cost_pct: e.g. 0.0015 = 0.15% one-way cost
        """
        spread_pct = cls.SPREAD_BPS.get(cap_category.upper(), 12) / 10000
        comm_pct   = cls.COMMISSION_PER_TRADE / (trade_value + 1e-8)
        impact_pct = cls.IMPACT_ETA * math.sqrt(
            max(0, trade_value / (adv + 1e-8))
        )
        return float(spread_pct + comm_pct + impact_pct)

    @classmethod
    def round_trip_cost(cls, trade_value: float, cap_category: str = "LARGE",
                        adv: float = 1e8) -> float:
        """Full round-trip cost (entry + exit) as fraction."""
        one_way = cls.cost_pct(trade_value, cap_category, adv)
        return one_way * 2   # entry + exit


# ─────────────────────────────────────────────────────────────────────────────
# 2. SIGNAL COMPUTER — bar-by-bar feature extraction
# ─────────────────────────────────────────────────────────────────────────────

class BarSignalComputer:
    """
    Computes all signal features from a rolling window of OHLCV bars.
    Reuses the same math as ml_engine.compute_features_from_bars() but
    returns a named dict instead of a numpy array, and includes more signals.
    """

    @staticmethod
    def compute(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                volume: np.ndarray) -> Optional[Dict]:
        """
        Compute all signals from OHLCV window.
        Requires at least 45 bars.
        Returns dict with all signal values, or None if insufficient data.
        """
        n = len(close)
        if n < 45:
            return None

        c, h, l, v = (np.array(x, dtype=float) for x in (close, high, low, volume))

        sigs = {}

        # ── Momentum ──────────────────────────────────────────────────────
        sigs["mom_5d"]  = float((c[-1] / c[-6]  - 1) * 100) if n >= 6  else 0.0
        sigs["mom_20d"] = float((c[-1] / c[-21] - 1) * 100) if n >= 21 else 0.0
        sigs["mom_60d"] = float((c[-1] / c[-61] - 1) * 100) if n >= 61 else 0.0

        # ── RSI (14) ──────────────────────────────────────────────────────
        d = np.diff(c[-15:])
        gains  = np.where(d > 0, d, 0.0)
        losses = np.where(d < 0, -d, 0.0)
        ag, al = np.mean(gains[-14:]), np.mean(losses[-14:])
        sigs["rsi"] = float(100 - 100 / (1 + ag / (al + 1e-8)))

        # ── MACD ─────────────────────────────────────────────────────────
        def ema(arr, p):
            k = 2.0 / (p + 1)
            e = float(arr[0])
            for x in arr[1:]: e = float(x) * k + e * (1 - k)
            return e
        ema9 = ema(c, 9); ema21 = ema(c, 21)
        sigs["macd_norm"] = float((ema9 - ema21) / (c[-1] + 1e-8) * 100)
        sigs["ema_cross_9_21"] = sigs["macd_norm"]

        # ── Bollinger ─────────────────────────────────────────────────────
        if n >= 20:
            sma20 = np.mean(c[-20:]); std20 = np.std(c[-20:]) + 1e-8
            sigs["bb_position"] = float(np.clip((c[-1] - (sma20 - 2*std20)) / (4*std20), 0, 1))
            sigs["bb_width"] = float(4 * std20 / (sma20 + 1e-8) * 100)
        else:
            sigs["bb_position"] = 0.5; sigs["bb_width"] = 5.0

        # ── ADX ──────────────────────────────────────────────────────────
        if n >= 20:
            mu = np.diff(h[-15:]); md = np.diff(l[-15:]) * -1
            pdm = np.where((mu > md) & (mu > 0), mu, 0.0)
            ndm = np.where((md > mu) & (md > 0), md, 0.0)
            tr  = np.maximum(mu - md,
                  np.maximum(np.abs(mu - c[-15:-1]), np.abs(md - c[-15:-1])))
            atr = np.mean(tr[-14:]) + 1e-8
            dx  = abs(np.mean(pdm[-14:]) - np.mean(ndm[-14:])) / (
                      np.mean(pdm[-14:]) + np.mean(ndm[-14:]) + 1e-8) * 100
            sigs["adx"] = float(dx)
        else:
            sigs["adx"] = 20.0

        # ── Volatility ────────────────────────────────────────────────────
        log_ret = np.diff(np.log(c + 1e-10))
        sigs["hv20"] = float(np.std(log_ret[-20:]) * np.sqrt(TRADING_DAYS) * 100) if n >= 21 else 25.0
        sigs["hv5"]  = float(np.std(log_ret[-5:])  * np.sqrt(TRADING_DAYS) * 100) if n >= 6  else 25.0

        # ── Volume ────────────────────────────────────────────────────────
        avg_vol = np.mean(v[-20:]) + 1e-8
        sigs["vol_ratio"] = float(v[-1] / avg_vol)

        # ── Stochastic ────────────────────────────────────────────────────
        if n >= 14:
            lo14 = np.min(l[-14:]); hi14 = np.max(h[-14:])
            sigs["stoch_k"] = float((c[-1] - lo14) / (hi14 - lo14 + 1e-8) * 100)
        else:
            sigs["stoch_k"] = 50.0

        # ── VWAP deviation ────────────────────────────────────────────────
        if n >= 20 and np.sum(v[-20:]) > 0:
            vwap = np.sum(c[-20:] * v[-20:]) / np.sum(v[-20:])
            sigs["price_vs_vwap"] = float((c[-1] - vwap) / (vwap + 1e-8) * 100)
        else:
            sigs["price_vs_vwap"] = 0.0

        # ── Sharpe (20d) ──────────────────────────────────────────────────
        if n >= 21:
            r20 = np.diff(np.log(c[-21:] + 1e-10))
            sigs["sharpe"] = float(np.mean(r20) / (np.std(r20) + 1e-8) * np.sqrt(TRADING_DAYS))
        else:
            sigs["sharpe"] = 0.0

        # ── Hurst ────────────────────────────────────────────────────────
        prices40 = c[-40:] if n >= 40 else c
        try:
            lgs, rsv = [], []
            for lag in range(2, min(20, len(prices40) // 2)):
                chunks = [np.log(prices40[i:i+lag] + 1e-10)
                          for i in range(0, len(prices40) - lag, lag)]
                rsp = []
                for ch in chunks:
                    if len(ch) < 2: continue
                    dev = np.cumsum(ch - np.mean(ch))
                    R = dev.max() - dev.min()
                    S = np.std(ch, ddof=1) + 1e-10
                    rsp.append(R / S)
                if rsp:
                    lgs.append(math.log(lag))
                    rsv.append(math.log(np.mean(rsp) + 1e-10))
            if len(lgs) >= 3:
                slope = np.polyfit(lgs, rsv, 1)[0]
                sigs["hurst"] = float(np.clip(slope, 0.01, 0.99))
            else:
                sigs["hurst"] = 0.5
        except Exception:
            sigs["hurst"] = 0.5

        # ── OU z-score ────────────────────────────────────────────────────
        if n >= 30:
            mu_ou = np.mean(c[-30:]); sd_ou = np.std(c[-30:]) + 1e-8
            sigs["ou_zscore"] = float((c[-1] - mu_ou) / sd_ou)
        else:
            sigs["ou_zscore"] = 0.0

        # ── Composite direction signal [-1, +1] ──────────────────────────
        rsi_s  = (sigs["rsi"] - 50) / 50
        macd_s = np.clip(sigs["macd_norm"] * 5, -1, 1)
        mom_s  = np.clip(sigs["mom_5d"] / 5, -1, 1)
        bb_s   = (sigs["bb_position"] - 0.5) * 2

        h = sigs["hurst"]
        if h > 0.55:
            raw = 0.4 * mom_s + 0.3 * macd_s + 0.3 * rsi_s
        elif h < 0.45:
            raw = -0.5 * (sigs["ou_zscore"] / 3) + 0.3 * rsi_s + 0.2 * bb_s
        else:
            raw = 0.35 * mom_s + 0.35 * rsi_s + 0.30 * macd_s

        sigs["composite"]  = float(np.clip(raw, -1, 1))
        sigs["normalized"] = float((sigs["composite"] + 1) / 2)
        sigs["price"]      = float(c[-1])
        sigs["return_1d"]  = float(log_ret[-1]) if len(log_ret) > 0 else 0.0

        return sigs


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

class BacktestDataLoader:
    """
    Downloads and caches historical OHLCV data for backtesting.
    Uses yfinance with local disk cache (60-day TTL) to avoid re-downloading.
    """

    CACHE_DIR = "backtest_cache"
    CACHE_TTL_DAYS = 7

    def __init__(self):
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    def _cache_path(self, symbol: str, period: str) -> str:
        return os.path.join(self.CACHE_DIR, f"{symbol}_{period}.json")

    def load(self, symbol: str, period: str = "5y") -> Optional[Dict]:
        """
        Load OHLCV data for symbol.
        Returns: {dates, open, high, low, close, volume} as lists, or None.
        """
        path = self._cache_path(symbol, period)

        # Check disk cache
        if os.path.exists(path):
            try:
                with open(path) as f:
                    cached = json.load(f)
                age_days = (time.time() - cached.get("_ts", 0)) / 86400
                if age_days < self.CACHE_TTL_DAYS:
                    logger.debug(f"  Cache hit: {symbol} ({len(cached['close'])} bars)")
                    return {k: v for k, v in cached.items() if k != "_ts"}
            except Exception:
                pass

        # Download
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval="1d", auto_adjust=True)
            if df is None or len(df) < 100:
                return None

            data = {
                "symbol":  symbol,
                "dates":   [str(d)[:10] for d in df.index],
                "open":    df["Open"].values.astype(float).tolist(),
                "high":    df["High"].values.astype(float).tolist(),
                "low":     df["Low"].values.astype(float).tolist(),
                "close":   df["Close"].values.astype(float).tolist(),
                "volume":  df["Volume"].values.astype(float).tolist(),
            }

            # Persist
            try:
                with open(path, "w") as f:
                    json.dump({**data, "_ts": time.time()}, f)
            except Exception:
                pass

            logger.debug(f"  Downloaded: {symbol} ({len(data['close'])} bars)")
            return data

        except Exception as e:
            logger.warning(f"BacktestDataLoader {symbol}: {e}")
            return None

    def load_batch(self, symbols: List[str], period: str = "5y",
                   max_workers: int = 5) -> Dict[str, Dict]:
        """Load multiple symbols concurrently."""
        results = {}
        lock = threading.Lock()

        def _worker(sym):
            d = self.load(sym, period)
            if d:
                with lock:
                    results[sym] = d

        threads = [threading.Thread(target=_worker, args=(s,), daemon=True)
                   for s in symbols]
        for t in threads: t.start()
        for t in threads: t.join(timeout=30)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. WALK-FORWARD RUNNER
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardRunner:
    """
    Generates expanding-window train/test splits.

    Each fold:
      - Train:  bars [0 ... split_end]    (expanding window)
    - Test:   bars [split_end ... split_end + test_size]  (OOS)

    The expanding window means the model sees all history before its test period.
    This mirrors how a real system would be deployed: you have all past data
    at every point in time.
    """

    def __init__(self, min_train: int = 252, test_size: int = 63,
                 step_size: int = 21):
        """
        Args:
            min_train:  Minimum bars for first training window (252 = 1yr)
            test_size:  OOS test window size in bars (63 = ~1 quarter)
            step_size:  Bars to advance between folds (21 = monthly rolling)
        """
        self.min_train = min_train
        self.test_size = test_size
        self.step_size = step_size

    def splits(self, n: int) -> List[Tuple[int, int, int, int]]:
        """
        Generate (train_start, train_end, test_start, test_end) tuples.
        train_start is always 0 (expanding window, not rolling).
        """
        splits = []
        train_end = self.min_train
        while train_end + self.test_size <= n:
            test_start = train_end
            test_end   = min(train_end + self.test_size, n)
            splits.append((0, train_end, test_start, test_end))
            train_end += self.step_size
        return splits


class PurgedWalkForwardRunner(WalkForwardRunner):
    """
    Walk-forward runner with purge + embargo around the train/test boundary.
    Helps prevent target leakage in adjacent-bar features.
    """

    def __init__(self, min_train: int = 252, test_size: int = 63,
                 step_size: int = 21, purge_bars: int = 5, embargo_bars: int = 3):
        super().__init__(min_train=min_train, test_size=test_size, step_size=step_size)
        self.purge_bars = max(0, int(purge_bars))
        self.embargo_bars = max(0, int(embargo_bars))

    def splits(self, n: int) -> List[Tuple[int, int, int, int]]:
        splits = []
        train_end = self.min_train
        while train_end + self.test_size <= n:
            # Purge bars immediately before test starts from training set.
            purged_train_end = max(self.min_train, train_end - self.purge_bars)
            test_start = min(n, train_end + self.embargo_bars)
            test_end = min(test_start + self.test_size, n)
            if test_start < test_end and purged_train_end > 30:
                splits.append((0, purged_train_end, test_start, test_end))
            train_end += self.step_size
        return splits


# ─────────────────────────────────────────────────────────────────────────────
# 5. SIGNAL IC ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

class SignalICAnalyzer:
    """
    Computes Information Coefficient (Spearman rank correlation) for each
    signal vs forward returns.

    IC is the single most important metric for evaluating quantitative signals:
      - IC > 0.05:  Good signal (IC = 0.05 → Sharpe ≈ 0.8 if signal is independent)
      - IC > 0.10:  Excellent signal (rare in equity markets)
      - IC < 0.02:  No meaningful edge

    Also computes:
      - IR (Information Ratio): IC / std(IC) — consistency matters
      - ICIR > 0.5: signal reliable enough to trade
      - t-statistic: |IC| * sqrt(N) / std(IC) — statistical significance
    """

    SIGNAL_NAMES = [
        "composite", "mom_5d", "mom_20d", "rsi", "macd_norm",
        "bb_position", "adx", "vol_ratio", "stoch_k",
        "hurst", "ou_zscore", "sharpe", "hv20", "price_vs_vwap",
    ]

    @staticmethod
    def compute_ic(signal_vals: np.ndarray, fwd_returns: np.ndarray) -> Dict:
        """
        Spearman rank IC of signal vs forward return.
        Returns IC, t-stat, p-value, and a GOOD/WEAK/NOISE label.
        """
        # Remove NaN pairs
        mask = ~(np.isnan(signal_vals) | np.isnan(fwd_returns))
        s = signal_vals[mask]; f = fwd_returns[mask]
        n = len(s)
        if n < 20:
            return {"ic": 0.0, "t_stat": 0.0, "p_value": 1.0,
                    "n": n, "quality": "INSUFFICIENT_DATA"}

        # Spearman rank correlation
        try:
            from scipy.stats import spearmanr
            ic, p_val = spearmanr(s, f)
            ic = float(ic) if not math.isnan(ic) else 0.0
        except Exception:
            # Manual rank correlation
            rs  = np.argsort(np.argsort(s)).astype(float)
            rf  = np.argsort(np.argsort(f)).astype(float)
            rs -= rs.mean(); rf -= rf.mean()
            ic  = float(np.dot(rs, rf) / (np.linalg.norm(rs) * np.linalg.norm(rf) + 1e-10))
            p_val = 1.0

        # t-statistic: IC * sqrt(N-2) / sqrt(1 - IC^2)
        t_stat = ic * math.sqrt(max(n - 2, 1)) / math.sqrt(max(1 - ic**2, 1e-8))

        quality = ("STRONG"  if abs(ic) > 0.08 else
                   "GOOD"    if abs(ic) > 0.05 else
                   "WEAK"    if abs(ic) > 0.02 else
                   "NOISE")

        return {
            "ic":       round(ic, 5),
            "t_stat":   round(t_stat, 3),
            "p_value":  round(float(p_val), 4) if isinstance(p_val, float) else 0.05,
            "n":        n,
            "quality":  quality,
        }

    @classmethod
    def analyze_all(cls, bars_data: List[Dict], fwd_days: int = 5) -> Dict:
        """
        Compute IC for all signals across all OOS bars.
        bars_data: list of signal dicts (one per bar, OOS only)
        """
        n = len(bars_data)
        if n < fwd_days + 5:
            return {}

        # Extract forward returns
        closes = np.array([b.get("price", np.nan) for b in bars_data])
        valid  = ~np.isnan(closes)
        fwd_rets = np.full(n, np.nan)
        for i in range(n - fwd_days):
            if valid[i] and valid[i + fwd_days]:
                fwd_rets[i] = math.log(closes[i + fwd_days] / (closes[i] + 1e-10))

        ic_results = {}
        for sig_name in cls.SIGNAL_NAMES:
            sig_vals = np.array([b.get(sig_name, np.nan) for b in bars_data])
            # Only use bars where forward return is available
            mask = ~np.isnan(fwd_rets[:n - fwd_days])
            if mask.sum() < 20:
                continue
            ic_results[sig_name] = cls.compute_ic(
                sig_vals[:n - fwd_days][mask],
                fwd_rets[:n - fwd_days][mask]
            )

        # Sort by abs(IC) descending
        ic_results = dict(sorted(
            ic_results.items(),
            key=lambda x: abs(x[1].get("ic", 0)),
            reverse=True
        ))
        return ic_results


# ─────────────────────────────────────────────────────────────────────────────
# 6. PORTFOLIO SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioSimulator:
    """
    Bar-by-bar portfolio simulation with position sizing and transaction costs.

    Strategy: Long/Short on composite signal threshold
      - signal > +entry_threshold: go LONG  (buy)
      - signal < -entry_threshold: go SHORT (sell)
      - |signal| < exit_threshold: EXIT position

    Position sizing: Fixed fractional (Kelly-inspired)
      size = min(conviction * max_position, max_position)
      where conviction = |composite_signal|

    Risk management:
      - Max position: 20% of portfolio per stock
      - Stop loss: -8% from entry (hard stop)
      - Take profit: +15% from entry (partial exit)
      - Max drawdown circuit breaker: halt trading if DD > 25%
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        max_position_pct: float = 0.20,
        entry_threshold: float  = 0.35,
        exit_threshold:  float  = 0.15,
        stop_loss_pct:   float  = 0.08,
        take_profit_pct: float  = 0.15,
        max_drawdown_halt: float = 0.25,
    ):
        self.initial_capital     = initial_capital
        self.max_position_pct    = max_position_pct
        self.entry_threshold     = entry_threshold
        self.exit_threshold      = exit_threshold
        self.stop_loss_pct       = stop_loss_pct
        self.take_profit_pct     = take_profit_pct
        self.max_drawdown_halt   = max_drawdown_halt

    def simulate(
        self,
        bars: List[Dict],          # list of signal dicts with 'price' key
        cap_category: str = "LARGE",
        adv_estimate: float = 1e8,
    ) -> Dict:
        """
        Simulate single-symbol strategy over bar sequence.

        Returns:
            equity_curve: list of portfolio values per bar
            trades:       list of completed trade records
            metrics:      performance statistics
        """
        if len(bars) < 10:
            return {"equity_curve": [], "trades": [], "metrics": {}}

        capital        = self.initial_capital
        position       = 0.0      # shares held (negative = short)
        entry_price    = 0.0
        entry_bar      = 0
        entry_direction = 0       # +1 long, -1 short

        equity_curve = []
        trades       = []
        peak_equity  = capital
        halted       = False

        for i, bar in enumerate(bars):
            price = float(bar.get("price", 0))
            if price <= 0:
                equity_curve.append(capital + position * price)
                continue

            portfolio_val = capital + position * price
            equity_curve.append(portfolio_val)

            # Track peak for drawdown
            peak_equity = max(peak_equity, portfolio_val)
            drawdown    = (peak_equity - portfolio_val) / (peak_equity + 1e-8)

            # Circuit breaker
            if drawdown > self.max_drawdown_halt and not halted:
                halted = True
                logger.debug(f"  Circuit breaker: DD={drawdown:.1%}")

            if halted:
                continue

            composite = float(bar.get("composite", 0))

            # ── Check existing position exits ─────────────────────────────
            if position != 0:
                hold_bars = i - entry_bar
                pnl_pct   = (price - entry_price) / (entry_price + 1e-8) * entry_direction

                exit_signal = (
                    abs(composite) < self.exit_threshold or
                    pnl_pct <= -self.stop_loss_pct or
                    pnl_pct >= self.take_profit_pct or
                    (entry_direction > 0 and composite < -self.entry_threshold) or
                    (entry_direction < 0 and composite >  self.entry_threshold) or
                    hold_bars >= 20   # max hold 20 bars
                )

                if exit_signal:
                    # Close position
                    trade_value = abs(position * price)
                    cost_pct    = TransactionCostModel.cost_pct(trade_value, cap_category, adv_estimate)
                    cost        = trade_value * cost_pct

                    gross_pnl = position * (price - entry_price)
                    net_pnl   = gross_pnl - cost

                    capital += position * price - cost * np.sign(position)
                    trades.append({
                        "entry_bar":   entry_bar,
                        "exit_bar":    i,
                        "entry_price": round(entry_price, 4),
                        "exit_price":  round(price, 4),
                        "direction":   "LONG" if entry_direction > 0 else "SHORT",
                        "shares":      round(abs(position), 4),
                        "gross_pnl":   round(gross_pnl, 2),
                        "net_pnl":     round(net_pnl, 2),
                        "cost":        round(cost, 2),
                        "hold_bars":   hold_bars,
                        "pnl_pct":     round(pnl_pct * 100, 2),
                        "exit_reason": (
                            "STOP_LOSS"    if pnl_pct <= -self.stop_loss_pct else
                            "TAKE_PROFIT"  if pnl_pct >= self.take_profit_pct else
                            "SIGNAL_FLIP"  if (
                                (entry_direction > 0 and composite < -self.entry_threshold) or
                                (entry_direction < 0 and composite > self.entry_threshold)
                            ) else
                            "MAX_HOLD"     if hold_bars >= 20 else
                            "SIGNAL_FADE"
                        ),
                        "entry_signal": round(float(bar.get("composite", 0)), 4),
                    })
                    position        = 0.0
                    entry_price     = 0.0
                    entry_direction = 0

            # ── Check entry ───────────────────────────────────────────────
            if position == 0:
                if composite > self.entry_threshold:
                    direction = 1
                elif composite < -self.entry_threshold:
                    direction = -1
                else:
                    continue

                # Position sizing: fractional Kelly
                conviction    = min(abs(composite), 1.0)
                pos_value     = capital * self.max_position_pct * conviction
                shares        = (pos_value / price) * direction

                # Entry transaction cost
                trade_value = abs(shares * price)
                cost        = trade_value * TransactionCostModel.cost_pct(
                    trade_value, cap_category, adv_estimate
                )

                if capital < cost + abs(shares * price) * 0.1:
                    continue  # insufficient capital

                capital        -= cost
                position        = shares
                entry_price     = price
                entry_bar       = i
                entry_direction = direction

        # Force close any open position at last bar
        if position != 0 and bars:
            last_price  = float(bars[-1].get("price", entry_price))
            trade_value = abs(position * last_price)
            cost        = trade_value * TransactionCostModel.cost_pct(trade_value, cap_category, adv_estimate)
            gross_pnl   = position * (last_price - entry_price)
            net_pnl     = gross_pnl - cost
            capital    += position * last_price - cost * np.sign(position)
            trades.append({
                "entry_bar": entry_bar, "exit_bar": len(bars) - 1,
                "entry_price": round(entry_price, 4), "exit_price": round(last_price, 4),
                "direction": "LONG" if entry_direction > 0 else "SHORT",
                "shares": round(abs(position), 4),
                "gross_pnl": round(gross_pnl, 2), "net_pnl": round(net_pnl, 2),
                "cost": round(cost, 2), "hold_bars": len(bars) - 1 - entry_bar,
                "pnl_pct": round((last_price - entry_price) / (entry_price + 1e-8) * entry_direction * 100, 2),
                "exit_reason": "END_OF_TEST",
            })

        equity_curve[-1] = capital if equity_curve else capital

        metrics = PerformanceStats.compute(
            equity_curve=equity_curve,
            trades=trades,
            initial_capital=self.initial_capital,
        )
        return {"equity_curve": equity_curve, "trades": trades, "metrics": metrics}


# ─────────────────────────────────────────────────────────────────────────────
# 7. PERFORMANCE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

class PerformanceStats:
    """
    Full suite of institutional performance metrics.
    """

    @staticmethod
    def compute(equity_curve: List[float], trades: List[Dict],
                initial_capital: float = 100_000) -> Dict:
        """Compute all performance metrics from equity curve and trade list."""
        if not equity_curve or len(equity_curve) < 2:
            return {}

        eq = np.array(equity_curve, dtype=float)
        eq = eq[eq > 0]   # remove zeros
        if len(eq) < 2:
            return {}

        returns = np.diff(np.log(eq + 1e-10))
        n_bars  = len(eq)
        n_years = n_bars / TRADING_DAYS

        # ── Return metrics ────────────────────────────────────────────────
        total_return  = float(eq[-1] / (initial_capital + 1e-8) - 1)
        cagr          = float((eq[-1] / initial_capital) ** (1 / max(n_years, 0.01)) - 1)

        # ── Risk metrics ──────────────────────────────────────────────────
        ann_vol       = float(np.std(returns) * np.sqrt(TRADING_DAYS))
        sharpe        = float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(TRADING_DAYS))
        sortino_neg   = returns[returns < 0]
        sortino       = float(np.mean(returns) / (np.std(sortino_neg) + 1e-8) * np.sqrt(TRADING_DAYS)) \
                        if len(sortino_neg) > 0 else sharpe

        # Maximum drawdown
        peak    = np.maximum.accumulate(eq)
        dd_arr  = (eq - peak) / (peak + 1e-8)
        max_dd  = float(dd_arr.min())
        calmar  = float(cagr / (-max_dd + 1e-8)) if max_dd < 0 else 0.0

        # ── Trade metrics ─────────────────────────────────────────────────
        n_trades  = len(trades)
        if n_trades > 0:
            pnls    = np.array([t.get("net_pnl", 0) for t in trades])
            winners = pnls[pnls > 0]
            losers  = pnls[pnls < 0]
            win_rate = float(len(winners) / n_trades)
            avg_win  = float(np.mean(winners)) if len(winners) > 0 else 0.0
            avg_loss = float(np.mean(losers))  if len(losers)  > 0 else 0.0
            profit_factor = float(np.sum(winners) / (-np.sum(losers) + 1e-8))
            avg_hold = float(np.mean([t.get("hold_bars", 0) for t in trades]))
            expectancy = float(np.mean(pnls))
        else:
            win_rate = profit_factor = avg_win = avg_loss = avg_hold = expectancy = 0.0

        # ── Drawdown curve ────────────────────────────────────────────────
        dd_curve = dd_arr.tolist()

        return {
            # Returns
            "total_return_pct":  round(total_return * 100, 2),
            "cagr_pct":          round(cagr * 100, 2),
            "ann_vol_pct":       round(ann_vol * 100, 2),
            # Risk-adjusted
            "sharpe":            round(sharpe, 3),
            "sortino":           round(sortino, 3),
            "calmar":            round(calmar, 3),
            "max_drawdown_pct":  round(max_dd * 100, 2),
            # Trades
            "n_trades":          n_trades,
            "win_rate_pct":      round(win_rate * 100, 1),
            "profit_factor":     round(profit_factor, 3),
            "avg_win_usd":       round(avg_win, 2),
            "avg_loss_usd":      round(avg_loss, 2),
            "avg_hold_bars":     round(avg_hold, 1),
            "expectancy_usd":    round(expectancy, 2),
            # Meta
            "n_bars":            n_bars,
            "years_tested":      round(n_years, 2),
            "final_equity":      round(float(eq[-1]), 2),
            "initial_capital":   initial_capital,
            # Drawdown curve (downsampled for API)
            "drawdown_curve":    [round(x * 100, 2) for x in dd_curve[::5]],
        }


# ─────────────────────────────────────────────────────────────────────────────
# 8. MASTER BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Orchestrates the full walk-forward backtest pipeline.

    Usage:
        engine = BacktestEngine()
        result = engine.run(symbols=["AAPL","MSFT","NVDA"], years=3)

    Result structure:
        {
          "status":         "complete",
          "symbols_tested": [...],
          "aggregate": {
            "sharpe", "cagr_pct", "max_drawdown_pct", "win_rate_pct", ...
          },
          "per_symbol": {
            "AAPL": {
              "metrics":      {...},
              "equity_curve": [...],
              "ic_per_signal":{...},
              "trades":       [...],
              "n_folds":      4,
            },
            ...
          },
          "ic_summary":  {signal: mean_ic, ...},     # across all symbols
          "equity_curve": [...],                       # aggregate equity
          "run_seconds":  12.3,
        }
    """

    # Singleton backtest state (for background runs)
    _running    = False
    _last_result: Optional[Dict] = None
    _run_lock   = threading.Lock()

    def __init__(self):
        self.loader     = BacktestDataLoader()
        self.wf_runner  = PurgedWalkForwardRunner(
            min_train=252, test_size=63, step_size=21, purge_bars=5, embargo_bars=3
        )
        self.simulator  = PortfolioSimulator()
        self.ic_analyzer = SignalICAnalyzer()

    def run(
        self,
        symbols:     List[str],
        years:       int  = 3,
        cap_categories: Dict[str, str] = None,
        progress_cb  = None,     # optional callback(pct, message)
    ) -> Dict:
        """
        Full walk-forward backtest across all symbols.

        Args:
            symbols:        List of ticker symbols
            years:          Years of history to use (1–5)
            cap_categories: {symbol: "LARGE"/"MID"/"SMALL"/"PENNY"}
            progress_cb:    Optional callback for progress updates

        Returns:
            Full result dict (see class docstring)
        """
        t0 = time.time()
        period = f"{years}y"
        cap_map = cap_categories or {}

        def _progress(pct, msg):
            if progress_cb:
                try: progress_cb(pct, msg)
                except Exception: pass
            logger.info(f"  [{pct:.0f}%] {msg}")

        _progress(0, f"Starting backtest: {len(symbols)} symbols, {years}yr history")

        # Load data
        _progress(5, "Loading OHLCV data...")
        all_data = self.loader.load_batch(symbols, period=period, max_workers=6)
        loaded   = list(all_data.keys())
        _progress(20, f"Loaded {len(loaded)}/{len(symbols)} symbols")

        if not loaded:
            return {"status": "error", "error": "No data loaded", "symbols_tested": []}

        per_symbol   = {}
        all_oos_bars = []    # pooled OOS bars for aggregate IC
        agg_equity   = []    # aggregate equity curve (equal-weight)

        for idx, symbol in enumerate(loaded):
            pct = 20 + int(idx / len(loaded) * 60)
            _progress(pct, f"Processing {symbol}...")

            data = all_data[symbol]
            close  = np.array(data["close"],  dtype=float)
            high   = np.array(data["high"],   dtype=float)
            low    = np.array(data["low"],    dtype=float)
            volume = np.array(data["volume"], dtype=float)
            dates  = data.get("dates", [])
            n      = len(close)

            if n < self.wf_runner.min_train + self.wf_runner.test_size:
                logger.debug(f"  {symbol}: insufficient bars ({n}), skipping")
                continue

            cap = cap_map.get(symbol, "LARGE")
            adv = float(np.mean(close[-20:] * volume[-20:])) if n >= 20 else 1e8

            # Compute bar-by-bar signals (lookback = 45 bars)
            lookback   = 45
            bars_sigs  = []
            for i in range(lookback, n):
                sigs = BarSignalComputer.compute(
                    close[max(0, i-lookback):i+1],
                    high[max(0, i-lookback):i+1],
                    low[max(0, i-lookback):i+1],
                    volume[max(0, i-lookback):i+1],
                )
                if sigs:
                    sigs["date"] = dates[i] if i < len(dates) else ""
                    bars_sigs.append(sigs)

            if len(bars_sigs) < self.wf_runner.min_train:
                continue

            # Walk-forward splits
            splits = self.wf_runner.splits(len(bars_sigs))
            if not splits:
                continue

            oos_bars  = []   # collect all OOS bars
            all_trades = []
            fold_eq    = []
            n_folds    = 0

            for tr_s, tr_e, ts_s, ts_e in splits:
                oos = bars_sigs[ts_s:ts_e]
                if not oos:
                    continue

                sim_result = self.simulator.simulate(oos, cap_category=cap, adv_estimate=adv)
                if sim_result["equity_curve"]:
                    fold_eq.append(sim_result["equity_curve"])
                all_trades.extend(sim_result.get("trades", []))
                oos_bars.extend(oos)
                n_folds += 1

            if not oos_bars:
                continue

            # Splice fold equity curves end-to-end
            combined_eq = []
            running_cap  = self.simulator.initial_capital
            for fe in fold_eq:
                if not fe: continue
                scale = running_cap / (fe[0] + 1e-8)
                combined_eq.extend([v * scale for v in fe])
                running_cap = combined_eq[-1] if combined_eq else running_cap

            # Signal IC analysis on pooled OOS bars
            ic_results = SignalICAnalyzer.analyze_all(oos_bars, fwd_days=5)
            all_oos_bars.extend(oos_bars)

            # Metrics
            metrics = PerformanceStats.compute(
                combined_eq, all_trades, self.simulator.initial_capital
            )

            per_symbol[symbol] = {
                "metrics":       metrics,
                "equity_curve":  [round(v, 2) for v in combined_eq[::5]],  # downsampled
                "ic_per_signal": ic_results,
                "n_folds":       n_folds,
                "n_oos_bars":    len(oos_bars),
                "n_trades":      len(all_trades),
                "trades":        all_trades[-20:],   # last 20 trades only (API size)
                "dates":         dates,
            }

            if combined_eq:
                agg_equity.append(combined_eq)

        _progress(85, "Computing aggregate statistics...")

        # Aggregate equity curve (equal-weight across all symbols)
        if agg_equity:
            max_len = max(len(e) for e in agg_equity)
            agg_norm = []
            for e in agg_equity:
                # Normalize to start at 1.0
                base = e[0] if e[0] > 0 else 1.0
                norm = [v / base for v in e]
                # Pad to max_len with last value
                if len(norm) < max_len:
                    norm.extend([norm[-1]] * (max_len - len(norm)))
                agg_norm.append(norm[:max_len])
            agg_arr  = np.mean(agg_norm, axis=0) * self.simulator.initial_capital
            agg_eq   = agg_arr.tolist()
        else:
            agg_eq = []

        # Aggregate metrics
        agg_metrics = PerformanceStats.compute(
            agg_eq,
            [t for sym in per_symbol.values() for t in sym.get("trades", [])],
            self.simulator.initial_capital,
        ) if agg_eq else {}

        # IC summary across all symbols
        ic_summary = {}
        if all_oos_bars:
            ic_summary_raw = SignalICAnalyzer.analyze_all(all_oos_bars, fwd_days=5)
            ic_summary = {k: v.get("ic", 0) for k, v in ic_summary_raw.items()}

        elapsed = time.time() - t0
        _progress(100, f"Backtest complete in {elapsed:.1f}s")

        result = {
            "status":          "complete",
            "symbols_tested":  list(per_symbol.keys()),
            "n_symbols":       len(per_symbol),
            "aggregate":       agg_metrics,
            "per_symbol":      per_symbol,
            "ic_summary":      ic_summary,
            "equity_curve":    [round(v, 2) for v in agg_eq[::5]],  # downsampled
            "run_seconds":     round(elapsed, 1),
            "config": {
                "years":        years,
                "min_train":    self.wf_runner.min_train,
                "test_size":    self.wf_runner.test_size,
                "step_size":    self.wf_runner.step_size,
                "purge_bars":   getattr(self.wf_runner, "purge_bars", 0),
                "embargo_bars": getattr(self.wf_runner, "embargo_bars", 0),
                "entry_thresh": self.simulator.entry_threshold,
                "exit_thresh":  self.simulator.exit_threshold,
                "stop_loss":    self.simulator.stop_loss_pct,
                "take_profit":  self.simulator.take_profit_pct,
                "max_pos_pct":  self.simulator.max_position_pct,
            },
            "timestamp": datetime.now().isoformat(),
        }

        BacktestEngine._last_result = result
        return result

    @classmethod
    def run_background(cls, symbols: List[str], years: int = 3,
                       cap_categories: Dict = None) -> Dict:
        """
        Launch backtest in background thread.
        Returns immediately with status="running".
        Check _last_result for completion.
        """
        if cls._running:
            return {"status": "already_running"}

        def _worker():
            cls._running = True
            try:
                engine = cls()
                result = engine.run(symbols, years, cap_categories)
                cls._last_result = result
            except Exception as e:
                cls._last_result = {"status": "error", "error": str(e)}
            finally:
                cls._running = False

        t = threading.Thread(target=_worker, daemon=True, name="backtest")
        t.start()
        return {"status": "running", "message": f"Backtest started: {len(symbols)} symbols, {years}yr"}

    @classmethod
    def get_status(cls) -> Dict:
        """Return current backtest run status."""
        if cls._running:
            return {"status": "running"}
        if cls._last_result:
            return {"status": cls._last_result.get("status", "unknown"),
                    "has_result": True,
                    "symbols": cls._last_result.get("symbols_tested", []),
                    "run_seconds": cls._last_result.get("run_seconds")}
        return {"status": "idle", "has_result": False}
