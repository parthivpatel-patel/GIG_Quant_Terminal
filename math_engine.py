"""
=============================================================================
RENAISSANCE MEDALLION — ADVANCED MATH ENGINE v2.0
PhD+ Mathematical Arsenal:
  • Black-Scholes-Merton with full Greeks (Δ Γ Θ V ρ + Vanna Volga Charm Speed)
  • Kalman Filter (adaptive trend estimation)
  • Hidden Markov Model (regime detection)
  • Copula-based correlation (tail dependency)
  • Cointegration + Pairs Trading (Engle-Granger, Johansen)
  • Hurst Exponent (mean reversion / trend persistence)
  • Ornstein-Uhlenbeck process (mean-reversion speed / half-life)
  • Monte Carlo with Geometric Brownian Motion (10,000 paths)
  • Value at Risk + Conditional VaR (Expected Shortfall)
  • Kelly Criterion (full + fractional)
  • Entropy-weighted signal fusion
  • Correlation matrix eigenvalue decomposition (PCA-based factor exposure)
  • Skewness/Kurtosis regime adjustment
  • Implied Volatility surface (Newton-Raphson solver)
  • Volume Profile & VWAP deviation scoring
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize, linalg
from scipy.stats import norm, skew, kurtosis, jarque_bera
from scipy.special import ndtr
from typing import Dict, List, Optional, Tuple
import warnings
import threading
import time
import logging
from datetime import datetime
warnings.filterwarnings("ignore")

logger = logging.getLogger("MATH")

# ─── Constants ────────────────────────────────────────────────────────────────
TRADING_DAYS   = 252

# ─── Live Risk-Free Rate ──────────────────────────────────────────────────────
# Fetches the 13-week T-Bill yield (^IRX) from yfinance daily.
# Falls back to a hardcoded safe value only if fetch fails.
# This fixes the audit finding: RISK_FREE_RATE was hardcoded at 0.053 (stale).
_RFR_CACHE: dict = {"rate": 0.05, "ts": 0.0}   # {rate: float, ts: epoch}
_RFR_TTL   = 86_400   # refresh once per day

def get_risk_free_rate() -> float:
    """
    Return current annualised risk-free rate from 13-week T-Bill yield (^IRX).
    Cached 24 hours. Returns last valid value on failure — never raises.
    """
    now = time.time()
    if now - _RFR_CACHE["ts"] < _RFR_TTL:
        return _RFR_CACHE["rate"]
    try:
        import yfinance as _yf_rfr
        tk  = _yf_rfr.Ticker("^IRX")
        fi  = tk.fast_info
        irx = getattr(fi, "last_price", None)
        if irx and irx > 0:
            rate = float(irx) / 100.0   # ^IRX is quoted as percentage (e.g. 5.3 → 0.053)
            _RFR_CACHE["rate"] = rate
            _RFR_CACHE["ts"]   = now
            logger.debug(f"Risk-free rate updated: {rate:.4f} ({irx:.2f}% ^IRX)")
            return rate
    except Exception as _e:
        logger.debug(f"RFR fetch failed ({_e}) — using cached {_RFR_CACHE['rate']:.4f}")
    _RFR_CACHE["ts"] = now   # suppress retry for TTL even on failure
    return _RFR_CACHE["rate"]

# Module-level alias — always use get_risk_free_rate() inside functions
# so the value is fresh; this alias exists for legacy call-sites.
RISK_FREE_RATE = property(lambda self: get_risk_free_rate())   # type: ignore[assignment]
# For module-level access (BSM etc.), call get_risk_free_rate() directly


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BLACK-SCHOLES-MERTON + FULL GREEKS
# ═══════════════════════════════════════════════════════════════════════════════

class BlackScholesMerton:
    """
    Complete BSM implementation with all first, second, and third-order Greeks.
    Newton-Raphson Implied Volatility solver (converges in <10 iterations).
    """

    @staticmethod
    def d1d2(S, K, T, r, sigma):
        """Core BSM d1, d2 calculation."""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0, 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    @staticmethod
    def price(S, K, T, r, sigma, option_type="call"):
        """Theoretical option price."""
        d1, d2 = BlackScholesMerton.d1d2(S, K, T, r, sigma)
        if option_type.lower() == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def all_greeks(S, K, T, r, sigma, option_type="call"):
        """Compute all options Greeks in one pass."""
        if T <= 1e-6:
            return {}
        d1, d2 = BlackScholesMerton.d1d2(S, K, T, r, sigma)
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)
        sqrtT  = np.sqrt(T)
        discount = np.exp(-r * T)

        sign = 1 if option_type.lower() == "call" else -1

        # First-order Greeks
        delta  = sign * cdf_d1 if option_type.lower() == "call" else cdf_d1 - 1
        gamma  = pdf_d1 / (S * sigma * sqrtT)
        theta  = (-(S * pdf_d1 * sigma) / (2 * sqrtT)
                  - sign * r * K * discount * norm.cdf(sign * d2)) / TRADING_DAYS
        vega   = S * pdf_d1 * sqrtT / 100   # Per 1% IV change
        rho    = sign * K * T * discount * norm.cdf(sign * d2) / 100

        # Second-order Greeks
        vanna  = -pdf_d1 * d2 / sigma                          # dDelta/dVol
        volga  = vega * d1 * d2 / sigma                        # dVega/dVol (Vomma)
        charm  = (-pdf_d1 * (2*r*T - d2*sigma*sqrtT)
                  / (2*T*sigma*sqrtT)) / TRADING_DAYS          # dDelta/dTime

        # Third-order
        speed  = -gamma / S * (d1 / (sigma * sqrtT) + 1)      # dGamma/dS
        color  = (-gamma * (r*d1/(sigma*sqrtT)
                  + (2*T - 1 - d1*d2)/(2*T))) / TRADING_DAYS  # dGamma/dTime (Charm of Gamma)

        return {
            "delta": float(delta), "gamma": float(gamma),
            "theta": float(theta), "vega":  float(vega),
            "rho":   float(rho),   "vanna": float(vanna),
            "volga": float(volga), "charm": float(charm),
            "speed": float(speed), "color": float(color),
        }

    @staticmethod
    def implied_vol(market_price, S, K, T, r, option_type="call", tol=1e-5, max_iter=100):
        """Newton-Raphson IV solver — converges in ~5-8 iterations."""
        if market_price <= 0 or T <= 0:
            return None

        # Intrinsic value check
        intrinsic = max(0, S - K) if option_type == "call" else max(0, K - S)
        if market_price <= intrinsic:
            return None

        sigma = 0.3  # Initial guess
        for _ in range(max_iter):
            try:
                price = BlackScholesMerton.price(S, K, T, r, sigma, option_type)
                d1, _ = BlackScholesMerton.d1d2(S, K, T, r, sigma)
                vega  = S * norm.pdf(d1) * np.sqrt(T)
                if abs(vega) < 1e-10:
                    break
                diff  = price - market_price
                sigma -= diff / vega
                sigma = max(1e-6, min(sigma, 10.0))
                if abs(diff) < tol:
                    return float(sigma)
            except Exception:
                break
        return float(sigma)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. KALMAN FILTER — Adaptive Trend Estimation
# ═══════════════════════════════════════════════════════════════════════════════

class KalmanFilter:
    """
    Kalman Filter for dynamic linear regression in pairs trading.
    State: [beta, alpha] — continuously updated belief about hedge ratio.
    Used by Renaissance for cointegration tracking.
    """

    def __init__(self, observation_cov=0.001, transition_cov=1e-5):
        self.observation_cov  = observation_cov
        self.transition_cov   = transition_cov
        self.state            = None   # [beta, alpha]
        self.state_cov        = None   # 2x2 covariance matrix

    def initialize(self, n_factors=2):
        self.state    = np.zeros(n_factors)
        self.state_cov = np.eye(n_factors) * 1.0

    def update(self, x, y):
        """Update Kalman state with new (x, y) observation. Returns spread."""
        if self.state is None:
            self.initialize()

        F = np.array([x, 1.0])   # Observation vector
        Q = np.eye(2) * self.transition_cov
        R = self.observation_cov

        # Predict
        self.state_cov += Q

        # Innovation
        y_pred = F @ self.state
        S      = F @ self.state_cov @ F + R
        K      = self.state_cov @ F / S   # Kalman gain

        # Update
        self.state      = self.state + K * (y - y_pred)
        self.state_cov  = (np.eye(2) - np.outer(K, F)) @ self.state_cov

        spread = y - F @ self.state
        return float(spread), float(self.state[0]), float(self.state[1])

    def compute_spread_series(self, x_series, y_series):
        """Run full Kalman filter over price series. Returns hedge ratio + spread."""
        self.initialize()
        spreads, betas, alphas = [], [], []
        for x, y in zip(x_series, y_series):
            sp, b, a = self.update(x, y)
            spreads.append(sp)
            betas.append(b)
            alphas.append(a)
        return np.array(spreads), np.array(betas), np.array(alphas)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HIDDEN MARKOV MODEL — Market Regime Detection
# ═══════════════════════════════════════════════════════════════════════════════

class HiddenMarkovRegime:
    """
    2-state HMM for bull/bear regime classification.
    Uses Baum-Welch-inspired EM with Gaussian emissions.
    States: 0=BEAR (high vol, negative drift), 1=BULL (low vol, positive drift)
    """

    def __init__(self, n_states=3):
        self.n_states = n_states
        self.fitted   = False
        # State params: [mean_return, std_return] per regime
        self.means    = np.array([-0.001, 0.0005, 0.002])[:n_states]
        self.stds     = np.array([0.025, 0.010, 0.012])[:n_states]
        self.trans    = np.full((n_states, n_states), 1/n_states)
        self.pi       = np.full(n_states, 1/n_states)

    def fit(self, returns: np.ndarray, n_iter=20):
        """Fit HMM via EM algorithm on return series."""
        returns = np.array(returns, dtype=float)
        returns = returns[~np.isnan(returns)]
        T = len(returns)
        if T < 50:
            return self

        for iteration in range(n_iter):
            # E-step: Forward-Backward
            log_emit = np.zeros((T, self.n_states))
            for k in range(self.n_states):
                log_emit[:, k] = norm.logpdf(returns, self.means[k], self.stds[k] + 1e-8)

            # Forward pass
            log_alpha = np.zeros((T, self.n_states))
            log_alpha[0] = np.log(self.pi + 1e-300) + log_emit[0]
            for t in range(1, T):
                for k in range(self.n_states):
                    log_alpha[t, k] = log_emit[t, k] + np.log(
                        np.sum(np.exp(log_alpha[t-1]) * self.trans[:, k]) + 1e-300)

            # Backward pass
            log_beta = np.zeros((T, self.n_states))
            for t in range(T-2, -1, -1):
                for k in range(self.n_states):
                    log_beta[t, k] = np.log(np.sum(
                        self.trans[k] * np.exp(log_emit[t+1] + log_beta[t+1])) + 1e-300)

            # State responsibilities (gamma)
            log_gamma = log_alpha + log_beta
            log_gamma -= np.max(log_gamma, axis=1, keepdims=True)
            gamma     = np.exp(log_gamma)
            gamma    /= gamma.sum(axis=1, keepdims=True) + 1e-300

            # M-step
            self.pi = gamma[0] / (gamma[0].sum() + 1e-300)
            for k in range(self.n_states):
                w = gamma[:, k] + 1e-300
                self.means[k] = np.average(returns, weights=w)
                self.stds[k]  = np.sqrt(np.average((returns - self.means[k])**2, weights=w)) + 1e-6

        self.fitted = True
        return self

    def predict_regime(self, returns: np.ndarray) -> Tuple[np.ndarray, str, float]:
        """Return state probabilities and current regime label."""
        returns = np.array(returns, dtype=float)[-100:]
        T = len(returns)

        log_emit = np.zeros((T, self.n_states))
        for k in range(self.n_states):
            log_emit[:, k] = norm.logpdf(returns, self.means[k], self.stds[k] + 1e-8)

        # Viterbi decoding
        log_delta = np.zeros((T, self.n_states))
        log_delta[0] = np.log(self.pi + 1e-300) + log_emit[0]
        for t in range(1, T):
            for k in range(self.n_states):
                log_delta[t, k] = np.max(
                    log_delta[t-1] + np.log(self.trans[:, k] + 1e-300)
                ) + log_emit[t, k]

        probs = np.exp(log_delta[-1] - log_delta[-1].max())
        probs /= probs.sum()

        regime_idx = np.argmax(probs)
        labels = ["BEAR", "NEUTRAL", "BULL"][:self.n_states]
        return probs, labels[regime_idx], float(probs[regime_idx])


# ═══════════════════════════════════════════════════════════════════════════════
# 3b. CUSUM STRUCTURAL BREAK DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class CUSUMBreakDetector:
    """
    Page-CUSUM (Cumulative Sum) structural break detector.

    Detects when the underlying return-generating process has shifted —
    i.e. when the current market regime is no longer described by the HMM
    that was fitted on historical data.

    Two independent CUSUM statistics are tracked simultaneously:
      1. Mean CUSUM   — detects persistent drift change (trend regime shift)
      2. Vol CUSUM    — detects variance change (volatility regime shift)

    Either exceeding its threshold triggers a regime-change signal,
    which causes the HMM to retrain on the most recent data.

    Theory:
        S_t+ = max(0, S_{t-1}+ + (x_t - μ_0 - k))   ← detects upward mean shift
        S_t- = max(0, S_{t-1}- - (x_t - μ_0 + k))   ← detects downward mean shift
        Alarm when S_t > h  (threshold)

        k = allowance = 0.5 * shift_size_to_detect
        h = decision threshold (controls false-alarm rate)

    Parameters are calibrated for daily S&P 500 returns:
        mean_k: 0.001  (detect shifts of 0.2%/day = ~50%/yr annualized)
        vol_k:  0.005  (detect variance jumps of ~50%)
        h:      5.0    (corresponds to ~ARL=500 days in control)

    Reference: Page (1954), "Continuous Inspection Schemes"
    """

    def __init__(
        self,
        mean_k: float = 0.001,    # allowance for mean CUSUM (daily return units)
        vol_k:  float = 0.005,    # allowance for vol CUSUM (daily variance units)
        h:      float = 5.0,      # decision threshold
        warmup: int   = 30,       # minimum observations before alarm can trigger
    ):
        self.mean_k  = mean_k
        self.vol_k   = vol_k
        self.h       = h
        self.warmup  = warmup
        self._reset()

    def _reset(self):
        """Reset CUSUM statistics (called after HMM retrains)."""
        self.s_mean_up   = 0.0   # CUSUM for upward mean shift
        self.s_mean_dn   = 0.0   # CUSUM for downward mean shift
        self.s_vol       = 0.0   # CUSUM for variance shift
        self.n_obs       = 0
        self.alarm_count = 0
        self.last_alarm  = None
        self.alarm_history: list = []

    def update(self, ret: float, ref_mean: float = 0.0, ref_std: float = 0.012) -> bool:
        """
        Update CUSUM with new daily return observation.

        Args:
            ret:      Today's log return
            ref_mean: In-control mean (from fitted HMM state)
            ref_std:  In-control std dev (from fitted HMM state)

        Returns:
            True if structural break alarm triggered, False otherwise.
        """
        self.n_obs += 1
        if self.n_obs < self.warmup:
            return False

        # ── Mean CUSUM ──────────────────────────────────────────────────────
        # Standardize residual
        z = (ret - ref_mean) / (ref_std + 1e-10)
        self.s_mean_up = max(0.0, self.s_mean_up + z - self.mean_k / (ref_std + 1e-10))
        self.s_mean_dn = max(0.0, self.s_mean_dn - z - self.mean_k / (ref_std + 1e-10))

        # ── Vol CUSUM (squared residuals vs expected variance) ───────────────
        # Tracks when squared returns persistently exceed expected variance
        z2 = z**2 - 1.0   # expected value under N(0,σ²) is 1.0
        self.s_vol = max(0.0, self.s_vol + z2 - self.vol_k / (ref_std**2 + 1e-10))

        # ── Alarm check ────────────────────────────────────────────────────
        alarm = (
            self.s_mean_up > self.h or
            self.s_mean_dn > self.h or
            self.s_vol     > self.h * 1.5     # vol alarm needs higher threshold (more noisy)
        )

        if alarm:
            self.alarm_count += 1
            self.last_alarm   = datetime.now()
            self.alarm_history.append({
                "time":        self.last_alarm.isoformat(),
                "s_mean_up":   round(self.s_mean_up, 3),
                "s_mean_dn":   round(self.s_mean_dn, 3),
                "s_vol":       round(self.s_vol, 3),
                "trigger":     ("MEAN_UP"  if self.s_mean_up > self.h else
                                "MEAN_DN"  if self.s_mean_dn > self.h else
                                "VOL"),
            })
            self._reset()   # soft reset after alarm (keep warmup history)
            self.n_obs = self.warmup   # allow alarms immediately after reset

        return alarm

    def update_batch(self, returns: np.ndarray, ref_mean: float = 0.0,
                     ref_std: float = 0.012) -> Dict:
        """
        Process a batch of returns (e.g., new weekly data).
        Returns summary with alarm status and current statistics.
        """
        returns = np.array(returns, dtype=float)
        returns = returns[~np.isnan(returns)]
        alarms_fired = 0
        for r in returns:
            if self.update(r, ref_mean, ref_std):
                alarms_fired += 1

        return {
            "n_processed":   len(returns),
            "alarms_fired":  alarms_fired,
            "s_mean_up":     round(self.s_mean_up, 3),
            "s_mean_dn":     round(self.s_mean_dn, 3),
            "s_vol":         round(self.s_vol, 3),
            "threshold":     self.h,
            "pct_threshold": {
                "mean_up": round(self.s_mean_up / self.h * 100, 1),
                "mean_dn": round(self.s_mean_dn / self.h * 100, 1),
                "vol":     round(self.s_vol / (self.h * 1.5) * 100, 1),
            },
            "alarm_count":   self.alarm_count,
            "last_alarm":    self.last_alarm.isoformat() if self.last_alarm else None,
            "status":        "BREAK_DETECTED" if alarms_fired > 0 else "IN_CONTROL",
        }

    def get_stats(self) -> Dict:
        """Return current CUSUM state for dashboard/API."""
        return {
            "s_mean_up":   round(self.s_mean_up, 3),
            "s_mean_dn":   round(self.s_mean_dn, 3),
            "s_vol":       round(self.s_vol, 3),
            "threshold":   self.h,
            "n_obs":       self.n_obs,
            "alarm_count": self.alarm_count,
            "last_alarm":  self.last_alarm.isoformat() if self.last_alarm else None,
            "pct_to_alarm": round(
                max(self.s_mean_up, self.s_mean_dn) / self.h * 100, 1
            ),
            "pct_vol_to_alarm": round(self.s_vol / (self.h * 1.5) * 100, 1),
            "status": "ELEVATED" if max(
                self.s_mean_up, self.s_mean_dn
            ) > self.h * 0.6 else "NORMAL",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3c. REGIME MODEL — HMM + CUSUM + Weekly Retrain Scheduler
# ═══════════════════════════════════════════════════════════════════════════════

class RegimeModel:
    """
    Stateful market regime classifier with automatic structural break detection.

    Combines:
      1. HiddenMarkovRegime (3-state Baum-Welch EM)
      2. CUSUMBreakDetector (Page-CUSUM mean + vol break detection)
      3. Weekly retrain scheduler (background thread, every Sunday)
      4. SPY index fallback (regime always trained on broad market)

    Usage:
        regime = RegimeModel()   # singleton, lives in scanner
        regime.update(new_returns)          # feed new daily returns
        state = regime.get_regime()         # get current regime dict
        regime.force_retrain(returns)       # force immediate refit

    The regime output is used by AladdinScorer as the macro_regime signal.
    """

    RETRAIN_INTERVAL_HOURS = 168   # weekly

    def __init__(self):
        self.hmm     = HiddenMarkovRegime(n_states=3)
        self.cusum   = CUSUMBreakDetector()
        self.fitted  = False
        self._lock   = threading.Lock()

        self._last_retrain:  Optional[datetime] = None
        self._retrain_count  = 0
        self._background_running = False
        self._break_triggered    = False
        self._break_history:     list = []

        # Current regime state (updated on every predict call)
        self._current_regime   = "NEUTRAL"
        self._current_probs    = {"BEAR": 0.33, "NEUTRAL": 0.34, "BULL": 0.33}
        self._current_conf     = 0.5
        self._ref_mean         = 0.0      # in-control HMM mean (for CUSUM)
        self._ref_std          = 0.012    # in-control HMM std

        # Return buffer: rolling 252-day daily returns
        self._return_buffer:   list = []
        self._BUFFER_MAX       = 504      # 2 years of daily returns

        logger.info("RegimeModel initialized (HMM + CUSUM + weekly retrain)")

    # ── Public API ─────────────────────────────────────────────────────────

    def update(self, returns: np.ndarray) -> Dict:
        """
        Main entry point. Feed new batch of returns (called each scan cycle).

        1. Appends returns to rolling buffer
        2. Fits HMM if not yet fitted
        3. Runs CUSUM on new observations
        4. If CUSUM alarm → triggers HMM retrain
        5. Returns current regime dict

        Args:
            returns: Array of daily log returns (most recent last)

        Returns:
            Regime dict with label, probs, confidence, CUSUM stats
        """
        returns = np.array(returns, dtype=float)
        returns = returns[~np.isnan(returns)]

        if len(returns) < 10:
            return self._build_regime_dict()

        # Update rolling buffer
        self._return_buffer.extend(returns.tolist())
        if len(self._return_buffer) > self._BUFFER_MAX:
            self._return_buffer = self._return_buffer[-self._BUFFER_MAX:]

        # Initial fit if not done
        if not self.fitted and len(self._return_buffer) >= 60:
            self._fit(np.array(self._return_buffer))

        if not self.fitted:
            return self._build_regime_dict()

        # Run CUSUM on new returns (using current HMM in-control params)
        for r in returns[-20:]:   # only process most recent 20 bars per cycle
            alarm = self.cusum.update(r, self._ref_mean, self._ref_std)
            if alarm and not self._background_running:
                logger.warning(
                    f"⚠️  CUSUM structural break detected! "
                    f"s_mean_up={self.cusum.s_mean_up:.2f} "
                    f"s_vol={self.cusum.s_vol:.2f} — triggering HMM retrain"
                )
                self._break_triggered = True
                self._break_history.append({
                    "time":    datetime.now().isoformat(),
                    "n_obs":   self.cusum.n_obs,
                    "trigger": self.cusum.alarm_history[-1] if self.cusum.alarm_history else {},
                })
                self._schedule_retrain()

        # Weekly scheduled retrain check (even without CUSUM alarm)
        if self._needs_scheduled_retrain() and not self._background_running:
            logger.info("📅 Weekly HMM retrain scheduled")
            self._schedule_retrain()

        # Update regime prediction
        try:
            with self._lock:
                probs, label, conf = self.hmm.predict_regime(
                    np.array(self._return_buffer[-100:])
                )
            self._current_regime  = label
            self._current_conf    = conf
            labels = ["BEAR", "NEUTRAL", "BULL"][:len(probs)]
            self._current_probs   = {l: round(float(p), 3) for l, p in zip(labels, probs)}
        except Exception as e:
            logger.debug(f"Regime predict error: {e}")

        return self._build_regime_dict()

    def get_regime(self) -> Dict:
        """Return current regime dict (no update — read-only)."""
        return self._build_regime_dict()

    def force_retrain(self, returns: np.ndarray = None) -> Dict:
        """Force immediate synchronous retraining (blocks caller briefly)."""
        buf = np.array(returns if returns is not None else self._return_buffer)
        if len(buf) >= 50:
            self._fit(buf)
        return self.get_regime()

    def get_stats(self) -> Dict:
        """Return full diagnostic stats for /api/regime endpoint."""
        return {
            "regime":           self._current_regime,
            "probs":            self._current_probs,
            "confidence":       round(self._current_conf, 3),
            "fitted":           self.fitted,
            "last_retrain":     self._last_retrain.isoformat() if self._last_retrain else None,
            "retrain_count":    self._retrain_count,
            "background_running": self._background_running,
            "break_triggered":  self._break_triggered,
            "break_count":      len(self._break_history),
            "break_history":    self._break_history[-5:],
            "cusum":            self.cusum.get_stats(),
            "buffer_size":      len(self._return_buffer),
            "hmm_means":        [round(float(m*100), 4) for m in self.hmm.means],  # as % per day
            "hmm_stds":         [round(float(s*100), 4) for s in self.hmm.stds],
            "macro_score":      self._regime_to_macro_score(),
        }

    # ── Internal mechanics ─────────────────────────────────────────────────

    def _fit(self, returns: np.ndarray):
        """Fit HMM on returns array. Resets CUSUM after fitting."""
        try:
            with self._lock:
                self.hmm = HiddenMarkovRegime(n_states=3)
                self.hmm.fit(returns)
                self.fitted = True
                self._last_retrain = datetime.now()
                self._retrain_count += 1
                self._break_triggered = False

                # Extract BULL state (highest mean) as the in-control reference
                # CUSUM monitors deviations from the fitted neutral state
                state_order = np.argsort(self.hmm.means)   # BEAR < NEUTRAL < BULL
                neutral_idx = state_order[1]
                self._ref_mean = float(self.hmm.means[neutral_idx])
                self._ref_std  = float(self.hmm.stds[neutral_idx])

            # Reset CUSUM with new in-control parameters
            self.cusum._reset()
            self.cusum.n_obs = self.cusum.warmup  # allow alarms immediately

            logger.info(
                f"✅ HMM retrained: {len(returns)} obs | "
                f"means={[round(m*100,3) for m in self.hmm.means]}%/day | "
                f"ref_mean={self._ref_mean*100:.4f}% ref_std={self._ref_std*100:.3f}%"
            )
        except Exception as e:
            logger.error(f"HMM fit error: {e}", exc_info=True)

    def _schedule_retrain(self):
        """Launch background retraining (non-blocking)."""
        self._background_running = True
        buf_snapshot = list(self._return_buffer)   # snapshot to avoid mutation

        def _worker():
            try:
                arr = np.array(buf_snapshot)
                if len(arr) >= 50:
                    self._fit(arr)
            except Exception as e:
                logger.error(f"Background HMM retrain error: {e}")
            finally:
                self._background_running = False

        t = threading.Thread(target=_worker, daemon=True, name="hmm-retrain")
        t.start()

    def _needs_scheduled_retrain(self) -> bool:
        if not self.fitted:
            return True
        if self._last_retrain is None:
            return True
        hours = (datetime.now() - self._last_retrain).total_seconds() / 3600
        return hours >= self.RETRAIN_INTERVAL_HOURS

    def _regime_to_macro_score(self) -> float:
        """
        Convert regime label + probs → scalar [-1, +1] for AladdinScorer macro_regime slot.
        BULL=+0.6, NEUTRAL=0.0, BEAR=-0.6, weighted by confidence.
        """
        score_map = {"BULL": 0.6, "NEUTRAL": 0.0, "BEAR": -0.6}
        raw = sum(
            score_map.get(label, 0) * prob
            for label, prob in self._current_probs.items()
        )
        return float(np.clip(raw, -1.0, 1.0))

    def _build_regime_dict(self) -> Dict:
        return {
            "regime":      self._current_regime,
            "confidence":  round(self._current_conf, 3),
            "probs":       self._current_probs,
            "macro_score": round(self._regime_to_macro_score(), 4),
            "cusum_pct":   round(
                max(self.cusum.s_mean_up, self.cusum.s_mean_dn) / (self.cusum.h + 1e-8) * 100, 1
            ),
            "break_alert": self._break_triggered,
            "fitted":      self.fitted,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HURST EXPONENT — Mean Reversion vs Trend Persistence
# ═══════════════════════════════════════════════════════════════════════════════

class HurstExponent:
    """
    Hurst < 0.5 → Mean-Reverting (use RSI, Bollinger fade strategies)
    Hurst = 0.5 → Random Walk (no edge)
    Hurst > 0.5 → Trending (use momentum, breakout strategies)
    """

    @staticmethod
    def compute(prices: np.ndarray, min_lag=2, max_lag=50) -> float:
        """R/S analysis Hurst exponent."""
        if len(prices) < max_lag * 2:
            return 0.5

        try:
            prices = np.array(prices, dtype=float)
            lags   = range(min_lag, min(max_lag, len(prices)//4))
            tau    = []
            for lag in lags:
                chunks = [prices[i:i+lag] for i in range(0, len(prices)-lag, lag)]
                if len(chunks) < 2:
                    continue
                rs = []
                for chunk in chunks:
                    if len(chunk) < 2:
                        continue
                    mean  = np.mean(chunk)
                    diffs = chunk - mean
                    cumdev = np.cumsum(diffs)
                    R = cumdev.max() - cumdev.min()
                    S = np.std(chunk, ddof=1)
                    if S > 0:
                        rs.append(R / S)
                if rs:
                    tau.append(np.mean(rs))

            if len(tau) < 2:
                return 0.5

            lags_arr = np.array(list(range(min_lag, min_lag + len(tau))))
            log_lags = np.log(lags_arr[:len(tau)])
            log_tau  = np.log(np.array(tau) + 1e-10)

            hurst, _, _, _, _ = stats.linregress(log_lags, log_tau)
            return float(np.clip(hurst, 0.0, 1.0))
        except Exception:
            return 0.5

    @staticmethod
    def strategy_signal(hurst: float, momentum_signal: float, mr_signal: float) -> float:
        """
        Blend momentum vs mean-reversion based on Hurst regime.
        Hurst > 0.55: favor momentum
        Hurst < 0.45: favor mean reversion
        """
        if hurst > 0.55:
            weight_mom = (hurst - 0.5) * 4   # 0.0 to 2.0
            weight_mr  = 1 - weight_mom
        elif hurst < 0.45:
            weight_mr  = (0.5 - hurst) * 4
            weight_mom = 1 - weight_mr
        else:
            weight_mom = weight_mr = 0.5

        return float(np.clip(weight_mom * momentum_signal + weight_mr * mr_signal, -1, 1))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ORNSTEIN-UHLENBECK — Mean Reversion Speed & Half-Life
# ═══════════════════════════════════════════════════════════════════════════════

class OrnsteinUhlenbeck:
    """
    dX = κ(μ - X)dt + σdW
    κ = speed of mean reversion
    μ = long-run mean
    σ = volatility of the process

    Half-life = ln(2)/κ  (time to revert 50% toward mean)
    Used for: pairs trading entry/exit timing, spread mean reversion
    """

    @staticmethod
    def fit(series: np.ndarray) -> Dict:
        """Fit OU parameters via OLS on lagged regression."""
        series = np.array(series, dtype=float)
        series = series[~np.isnan(series)]
        if len(series) < 20:
            return {"kappa": 0, "mu": 0, "sigma": 0, "half_life": np.inf, "valid": False}

        y   = series[1:]
        x   = series[:-1]
        reg = stats.linregress(x, y)

        a     = reg.slope
        b     = reg.intercept
        kappa = -np.log(a) * TRADING_DAYS if a > 0 else 0
        mu    = b / (1 - a) if a != 1 else np.mean(series)
        resid = y - (a * x + b)
        sigma = np.std(resid) * np.sqrt(TRADING_DAYS)

        half_life = np.log(2) / kappa if kappa > 0 else np.inf

        return {
            "kappa":     float(kappa),
            "mu":        float(mu),
            "sigma":     float(sigma),
            "half_life": float(half_life),
            "valid":     kappa > 0 and half_life < 252,
            "current_z": float((series[-1] - mu) / (sigma / np.sqrt(TRADING_DAYS * 2) + 1e-8)),
        }

    @staticmethod
    def mean_reversion_signal(ou_params: Dict) -> float:
        """Signal based on OU z-score. Positive = buy (below mean), Negative = sell."""
        if not ou_params.get("valid"):
            return 0.0
        z = ou_params.get("current_z", 0)
        # z < -2: strong buy (2 std below mean)
        # z > +2: strong sell (2 std above mean)
        return float(np.clip(-z / 3, -1, 1))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MONTE CARLO — Full Distribution Simulation
# ═══════════════════════════════════════════════════════════════════════════════

class MonteCarlo:
    """
    Geometric Brownian Motion path simulation.
    10,000 paths for robust probability estimation.
    Used for: Options POP, EV, VaR, CVaR, barrier probability.
    """

    @staticmethod
    def simulate_paths(S0, mu, sigma, T_days, n_paths=10_000) -> np.ndarray:
        """
        Simulate GBM: S(t) = S0 * exp((μ - σ²/2)t + σ√t * Z)
        Returns: (n_paths,) array of terminal prices.
        """
        T      = T_days / TRADING_DAYS
        Z      = np.random.standard_normal(n_paths)
        ST     = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        return ST

    @staticmethod
    def options_pop(S0, strike, sigma, T_days, option_type="call",
                    mu=0.0, n_paths=10_000) -> Tuple[float, float]:
        """
        Probability of Profit for an option position.
        Returns: (POP %, Expected Value $)
        """
        ST    = MonteCarlo.simulate_paths(S0, mu, sigma, T_days, n_paths)
        T     = T_days / TRADING_DAYS

        # Theoretical premium via BSM
        premium = BlackScholesMerton.price(S0, strike, T, get_risk_free_rate(), sigma, option_type)

        if option_type == "call":
            payoffs = np.maximum(ST - strike, 0) - premium
        else:
            payoffs = np.maximum(strike - ST, 0) - premium

        pop = float(np.mean(payoffs > 0) * 100)
        ev  = float(np.mean(payoffs) * 100)   # Per contract = 100 shares
        return pop, ev

    @staticmethod
    def var_cvar(returns: np.ndarray, confidence=0.95, horizon=1) -> Dict:
        """
        Historical + Parametric VaR and CVaR (Expected Shortfall).
        Used for position sizing and risk limit checks.
        """
        returns = np.array(returns, dtype=float)
        returns = returns[~np.isnan(returns)]
        if len(returns) < 30:
            return {}

        # Parametric (normal assumption)
        mu_r  = np.mean(returns)
        sig_r = np.std(returns)
        var_p  = norm.ppf(1 - confidence) * sig_r * np.sqrt(horizon) - mu_r * horizon
        cvar_p = (sig_r * np.sqrt(horizon) * norm.pdf(norm.ppf(confidence))
                  / (1 - confidence))

        # Historical
        sorted_r = np.sort(returns)
        cutoff   = int(len(sorted_r) * (1 - confidence))
        var_h    = -sorted_r[cutoff]
        cvar_h   = -np.mean(sorted_r[:cutoff])

        return {
            "var_parametric":  float(var_p),
            "cvar_parametric": float(cvar_p),
            "var_historical":  float(var_h),
            "cvar_historical": float(cvar_h),
            "sharpe":          float(mu_r * TRADING_DAYS / (sig_r * np.sqrt(TRADING_DAYS) + 1e-8)),
            "sortino":         float(mu_r * TRADING_DAYS / (np.std(returns[returns < 0]) * np.sqrt(TRADING_DAYS) + 1e-8)),
            "skewness":        float(skew(returns)),
            "kurtosis":        float(kurtosis(returns)),
        }

    @staticmethod
    def barrier_probability(S0, barrier, sigma, T_days, direction="up") -> float:
        """P(price touches barrier before expiry) — for stop-loss / target analysis."""
        T  = T_days / TRADING_DAYS
        mu = get_risk_free_rate() - 0.5 * sigma**2
        if direction == "up":
            p = norm.cdf((np.log(barrier/S0) - mu*T) / (sigma*np.sqrt(T) + 1e-8))
        else:
            p = norm.cdf((-np.log(barrier/S0) + mu*T) / (sigma*np.sqrt(T) + 1e-8))
        return float(np.clip(p, 0, 1))


# ═══════════════════════════════════════════════════════════════════════════════
# 7. KELLY CRITERION — Optimal Position Sizing
# ═══════════════════════════════════════════════════════════════════════════════

class KellyCriterion:
    """
    Full Kelly: f* = (bp - q) / b
    Fractional Kelly (safer): f = f* * fraction
    Multi-asset Kelly via covariance matrix inversion.
    """

    @staticmethod
    def single_asset(win_prob: float, win_return: float,
                     loss_return: float, fraction: float = 0.25) -> float:
        """
        Optimal fraction of capital to bet.
        win_prob:    P(winning trade)
        win_return:  avg gain per winning trade (e.g. 0.05 = 5%)
        loss_return: avg loss per losing trade (e.g. 0.02 = 2%)
        fraction:    safety factor (0.25 = quarter-Kelly)
        """
        if win_return <= 0 or loss_return <= 0:
            return 0.0
        q     = 1 - win_prob
        b     = win_return / loss_return   # Odds ratio
        kelly = (b * win_prob - q) / b
        return float(np.clip(kelly * fraction, 0, 0.20))   # Max 20% per position

    @staticmethod
    def portfolio_kelly(expected_returns: np.ndarray,
                        cov_matrix: np.ndarray,
                        fraction: float = 0.25) -> np.ndarray:
        """
        Multi-asset Kelly via covariance matrix inverse.
        f* = Σ^-1 * μ / max_leverage
        """
        try:
            inv_cov = np.linalg.inv(cov_matrix + np.eye(len(cov_matrix)) * 1e-8)
            weights = inv_cov @ expected_returns
            # Normalize and apply fraction
            weights = weights / (np.sum(np.abs(weights)) + 1e-8)
            weights = weights * fraction
            return np.clip(weights, -0.20, 0.20)
        except Exception:
            return np.zeros(len(expected_returns))


# ═══════════════════════════════════════════════════════════════════════════════
# 8. COINTEGRATION — Pairs Trading Foundation
# ═══════════════════════════════════════════════════════════════════════════════

class Cointegration:
    """
    Engle-Granger cointegration test + spread z-score.
    Used to find pairs that move together long-term (statistical arbitrage).
    """

    @staticmethod
    def engle_granger_test(x: np.ndarray, y: np.ndarray) -> Dict:
        """
        Test for cointegration between two price series.
        Returns: p-value, hedge ratio, spread statistics.
        """
        from scipy import stats as scipy_stats

        x, y = np.array(x, dtype=float), np.array(y, dtype=float)
        n    = min(len(x), len(y))
        x, y = x[-n:], y[-n:]

        # OLS regression y = alpha + beta * x + epsilon
        slope, intercept, r, p, _ = scipy_stats.linregress(x, y)
        spread = y - (slope * x + intercept)

        # ADF test on spread (check stationarity = cointegration)
        # Simple ADF approximation
        dy    = np.diff(spread)
        y_lag = spread[:-1]
        adf_slope, _, _, adf_p, _ = scipy_stats.linregress(y_lag, dy)

        z_score = (spread[-1] - np.mean(spread)) / (np.std(spread) + 1e-8)

        return {
            "cointegrated":  adf_p < 0.05,
            "p_value":       float(adf_p),
            "hedge_ratio":   float(slope),
            "spread":        spread.tolist(),
            "spread_mean":   float(np.mean(spread)),
            "spread_std":    float(np.std(spread)),
            "z_score":       float(z_score),
            "half_life":     OrnsteinUhlenbeck.fit(spread).get("half_life", np.inf),
            "r_squared":     float(r**2),
        }

    @staticmethod
    def pairs_signal(z_score: float, entry_z=2.0, exit_z=0.5) -> Tuple[str, float]:
        """
        Generate trade signal from spread z-score.
        z < -entry_z → LONG spread (buy y, sell x)
        z > +entry_z → SHORT spread (sell y, buy x)
        |z| < exit_z → EXIT
        """
        if z_score < -entry_z:
            return "LONG_SPREAD", float((-z_score - entry_z) / 2)
        elif z_score > entry_z:
            return "SHORT_SPREAD", float((z_score - entry_z) / 2)
        elif abs(z_score) < exit_z:
            return "EXIT", 0.0
        return "HOLD", 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 9. VOLATILITY ANALYSIS — IV Rank, Surface, Premium
# ═══════════════════════════════════════════════════════════════════════════════

class VolatilityAnalysis:
    """
    Complete implied + historical volatility analysis.
    IV Rank, IV Percentile, Vol Premium, Expected Move, GARCH proxy.
    """

    @staticmethod
    def historical_vol(prices: np.ndarray, window=20) -> float:
        """20-day historical volatility (annualized)."""
        if len(prices) < window + 1:
            return 0.0
        log_returns = np.diff(np.log(prices))[-window:]
        return float(np.std(log_returns) * np.sqrt(TRADING_DAYS))

    @staticmethod
    def iv_rank(current_iv: float, iv_history: np.ndarray) -> float:
        """IV Rank: where current IV sits vs 52-week range."""
        iv_min = np.min(iv_history)
        iv_max = np.max(iv_history)
        if iv_max == iv_min:
            return 50.0
        return float((current_iv - iv_min) / (iv_max - iv_min) * 100)

    @staticmethod
    def iv_percentile(current_iv: float, iv_history: np.ndarray) -> float:
        """IV Percentile: % of past days where IV was lower than today."""
        return float(np.mean(iv_history < current_iv) * 100)

    @staticmethod
    def expected_move(S: float, iv: float, T_days: int) -> Dict:
        """Expected move based on implied volatility."""
        T  = T_days / TRADING_DAYS
        em = S * iv * np.sqrt(T)
        return {
            "one_sigma_up":    float(S + em),
            "one_sigma_down":  float(S - em),
            "two_sigma_up":    float(S + 2*em),
            "two_sigma_down":  float(S - 2*em),
            "expected_move":   float(em),
            "expected_move_pct": float(em / S * 100),
        }

    @staticmethod
    def garch_proxy_vol(returns: np.ndarray, alpha=0.1, beta=0.85) -> float:
        """
        Simple GARCH(1,1) proxy via exponential weighted variance.
        omega = (1 - alpha - beta) * long_run_var
        sigma²_t = omega + alpha * r²_{t-1} + beta * sigma²_{t-1}
        """
        if len(returns) < 20:
            return float(np.std(returns) * np.sqrt(TRADING_DAYS))

        omega    = (1 - alpha - beta) * np.var(returns)
        sigma_sq = np.var(returns)
        for r in returns:
            sigma_sq = omega + alpha * r**2 + beta * sigma_sq

        return float(np.sqrt(sigma_sq * TRADING_DAYS))

    @staticmethod
    def rv_iv_spread(
        close_prices: np.ndarray,
        atm_iv: float,
        vix: float = 20.0,
        rv_window_short: int = 10,
        rv_window_long:  int = 20,
    ) -> Dict:
        """
        RV vs IV Volatility Arbitrage Signal.
        Renaissance's most consistent alpha: IV systematically exceeds RV.
        The vol risk premium (VRP) is one of the most robust anomalies in finance,
        with Sharpe ~1.5 historically when traded systematically.

        Signal logic:
          VRP = ATM_IV - HV20   (positive = vol overpriced = sell premium)
          vol_arb_score = VRP / ATM_IV   (normalized: 0.15+ = strong opportunity)

        Strategy recommendations:
          VRP > 8 vol pts AND IV < 35 AND VIX < 25:  Iron Condor / Short Strangle
          VRP > 5 vol pts AND IV > 35:                 Put Credit Spread
          VRP > 3 vol pts:                              Short-dated Covered Call
          VRP < -3 vol pts (HV > IV):                  Buy Straddle (vol underpriced)

        Args:
            close_prices: Array of recent daily close prices (need ≥22 bars)
            atm_iv:       ATM implied volatility as decimal (e.g. 0.30 = 30%)
            vix:          Current VIX level
            rv_window_short: Short realized vol window (10d)
            rv_window_long:  Long realized vol window (20d)

        Returns dict with:
            rv10, rv20: Realized vol (annualized %, e.g. 22.5)
            iv_pct:     ATM IV as percentage
            vrp:        Vol risk premium = IV - RV20 (signed, in vol points)
            vrp_pct:    VRP / IV (normalized, how much IV exceeds RV)
            vol_arb_score: Composite signal [-1, +1] for AladdinScorer
            strategy:   Recommended trade structure
            conviction: STRONG / MODERATE / WEAK / NONE
        """
        if len(close_prices) < rv_window_long + 2:
            return {
                "rv10": 0.0, "rv20": 0.0, "iv_pct": atm_iv * 100,
                "vrp": 0.0, "vrp_pct": 0.0, "vol_arb_score": 0.0,
                "strategy": "INSUFFICIENT_DATA", "conviction": "NONE",
            }

        prices = np.array(close_prices, dtype=float)
        log_ret = np.diff(np.log(prices + 1e-10))

        # Realized vols (annualized %)
        rv10 = float(np.std(log_ret[-rv_window_short:]) * np.sqrt(TRADING_DAYS) * 100) \
               if len(log_ret) >= rv_window_short else 0.0
        rv20 = float(np.std(log_ret[-rv_window_long:])  * np.sqrt(TRADING_DAYS) * 100) \
               if len(log_ret) >= rv_window_long else rv10

        # GARCH-adjusted RV (give more weight to recent vol)
        garch_rv = VolatilityAnalysis.garch_proxy_vol(log_ret[-60:] if len(log_ret) >= 60 else log_ret) * 100

        # Use the maximum of short/long/garch as the RV estimate
        # (conservative: gives vol sellers the worst-case realized vol)
        rv_conservative = max(rv10, rv20, garch_rv)

        iv_pct = atm_iv * 100 if atm_iv <= 5 else atm_iv   # handle decimal vs pct

        # Vol Risk Premium
        vrp     = iv_pct - rv_conservative        # positive = IV expensive
        vrp_pct = vrp / (iv_pct + 1e-8)          # normalized

        # ── Strategy selection ─────────────────────────────────────────────
        # Based on VRP level, absolute IV, and VIX regime
        if vrp > 8 and iv_pct < 35 and vix < 25:
            strategy   = "IRON_CONDOR"            # Ideal: modest IV, wide VRP, calm VIX
            conviction = "STRONG"
        elif vrp > 8 and iv_pct >= 35:
            strategy   = "PUT_CREDIT_SPREAD"      # High IV: sell put spread for defined risk
            conviction = "STRONG"
        elif vrp > 5 and iv_pct < 50:
            strategy   = "SHORT_STRANGLE"         # Moderate VRP, manageable IV
            conviction = "MODERATE"
        elif vrp > 3:
            strategy   = "COVERED_CALL"           # Mild edge: collect some premium
            conviction = "WEAK"
        elif vrp < -5:
            strategy   = "BUY_STRADDLE"           # HV > IV: vol is cheap, buy it
            conviction = "MODERATE"
        elif vrp < -3:
            strategy   = "BUY_CALLS"              # Slight vol underpricing
            conviction = "WEAK"
        else:
            strategy   = "NO_EDGE"
            conviction = "NONE"

        # ── Composite signal for AladdinScorer vol_regime slot ────────────
        # Positive score = sell premium (IV > RV)
        # Negative score = buy vol (RV > IV)
        # Range [-1, +1]
        if vrp > 0:
            # Sell premium: stronger signal with higher normalized VRP
            raw_score = float(np.clip(vrp_pct * 2.5, 0, 1))
        else:
            # Buy vol: negative score, capped
            raw_score = float(np.clip(vrp_pct * 1.5, -1, 0))

        # VIX safety gate: if VIX > 30, premium selling is dangerous — reduce score
        if vix > 30 and raw_score > 0:
            raw_score *= 0.3
        elif vix > 25 and raw_score > 0:
            raw_score *= 0.6

        return {
            "rv10":          round(rv10, 2),
            "rv20":          round(rv20, 2),
            "rv_garch":      round(garch_rv, 2),
            "rv_conservative": round(rv_conservative, 2),
            "iv_pct":        round(iv_pct, 2),
            "vrp":           round(vrp, 2),               # Vol Risk Premium in vol points
            "vrp_pct":       round(vrp_pct * 100, 1),    # VRP as % of IV
            "vol_arb_score": round(raw_score, 4),         # for AladdinScorer
            "strategy":      strategy,
            "conviction":    conviction,
            "vix":           round(vix, 1),
            "label": (
                f"VRP: IV {iv_pct:.1f}% vs RV {rv_conservative:.1f}% → "
                f"{'+' if vrp>0 else ''}{vrp:.1f} pts ({strategy.replace('_',' ')})"
            ),
        }

    @staticmethod
    def iron_condor_strikes(
        spot: float,
        iv: float,
        dte: int,
        wing_width_sigma: float = 1.5,
        short_delta: float = 0.16,
    ) -> Dict:
        """
        Compute iron condor strike levels for premium selling.
        Uses log-normal distribution to find strikes at target delta.

        Args:
            spot: Current stock price
            iv:   ATM IV as decimal (0.30 = 30%)
            dte:  Days to expiration
            wing_width_sigma: Width of long wings in sigma multiples (1.5 = 1.5σ beyond short)
            short_delta: Target delta for short strikes (0.16 = ~1 std dev, 84% POP)

        Returns strike levels for the 4-leg condor + expected premium.
        """
        from scipy.stats import norm as _norm
        T = max(dte, 1) / TRADING_DAYS
        sig_T = iv * np.sqrt(T)

        # Short strikes at target delta (log-normal)
        short_call_K = spot * np.exp(_norm.ppf(1 - short_delta) * sig_T)
        short_put_K  = spot * np.exp(_norm.ppf(short_delta) * sig_T)

        # Long strikes 1.5σ further OTM (protection)
        wing = spot * iv * np.sqrt(T) * wing_width_sigma
        long_call_K  = short_call_K + wing
        long_put_K   = short_put_K  - wing

        # Round to nearest $0.50
        def r(x): return round(round(x * 2) / 2, 2)

        short_call_K = r(short_call_K)
        short_put_K  = r(short_put_K)
        long_call_K  = r(long_call_K)
        long_put_K   = r(long_put_K)

        # Expected premium (rough BSM estimate of short strikes)
        from scipy.special import ndtr as _ndtr
        def bsm_price_approx(S, K, T_, iv_, is_call):
            if T_ <= 0: return max(0, S-K) if is_call else max(0, K-S)
            d1 = (np.log(S/K) + 0.5*iv_**2*T_) / (iv_*np.sqrt(T_) + 1e-10)
            d2 = d1 - iv_*np.sqrt(T_)
            if is_call: return S*_ndtr(d1) - K*_ndtr(d2)
            else:       return K*_ndtr(-d2) - S*_ndtr(-d1)

        call_premium = bsm_price_approx(spot, short_call_K, T, iv, True)
        put_premium  = bsm_price_approx(spot, short_put_K,  T, iv, False)
        net_credit   = round((call_premium + put_premium) * 0.85, 2)   # 15% slip
        max_width    = round(short_call_K - short_put_K, 2)
        max_risk     = round(max(short_call_K - long_call_K, short_put_K - long_put_K) - net_credit, 2)

        return {
            "long_put_K":   long_put_K,
            "short_put_K":  short_put_K,
            "short_call_K": short_call_K,
            "long_call_K":  long_call_K,
            "net_credit":   net_credit,       # per share
            "net_credit_100": round(net_credit * 100, 2),   # per contract
            "max_risk":     max_risk,
            "max_width":    max_width,
            "pop_estimate": round((1 - 2 * short_delta) * 100, 1),   # approx POP
            "dte":          dte,
            "breakeven_up": round(short_call_K + net_credit, 2),
            "breakeven_dn": round(short_put_K  - net_credit, 2),
        }

    @staticmethod
    def strategy_regime(iv_rank: float, hurst: float, vix: float) -> Dict:
        """
        Combine IV rank + Hurst + VIX to select optimal strategy type.
        """
        if vix > 30:
            env = "crisis"
        elif vix > 20:
            env = "elevated"
        elif vix < 15:
            env = "compressed"
        else:
            env = "normal"

        if iv_rank < 35 and hurst > 0.55:
            rec = "BUY_MOMENTUM_OPTIONS"   # Trend + cheap vol
        elif iv_rank > 65:
            rec = "SELL_PREMIUM"           # Expensive vol → sell it
        elif hurst < 0.45:
            rec = "MEAN_REVERSION_SPREAD"  # Strong mean reversion
        elif env == "crisis":
            rec = "BUY_PUTS_HEDGE"         # Crisis → buy protection
        else:
            rec = "VERTICAL_SPREAD"        # Moderate vol → defined risk

        return {"vol_env": env, "strategy_rec": rec, "iv_rank": iv_rank, "hurst": hurst}


# ═══════════════════════════════════════════════════════════════════════════════
# 10. MI-ORTHOGONALIZED SIGNAL FUSION  (Step 10)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Problem with simple weighted average of 10 signals:
#   If "technical_composite" and "advanced_composite" are 90% correlated,
#   we are effectively double-counting the same information. The composite
#   is biased toward whatever cluster of correlated signals happens to fire.
#
# Solution: Mutual Information (MI) based redundancy detection.
#   1. Compute pairwise MI between all signal pairs from recent history
#   2. Build a redundancy matrix R[i,j] = normalized MI (0=independent, 1=identical)
#   3. For each signal, compute effective weight = base_weight / (1 + sum of MI with others)
#   4. Re-normalize effective weights to sum to 1
#   5. Apply effective weights to produce the MI-orthogonalized composite
#
# MI estimation: discrete KDE approach
#   - Discretize each signal into 10 equal-probability bins
#   - Estimate joint probability from co-occurrence counts
#   - MI(X;Y) = Σ p(x,y) log(p(x,y) / (p(x)*p(y)))
#   - Normalize by min(H(X), H(Y)) → NMI ∈ [0,1]
#
# Academic basis:
#   Kraskov, Stögbauer & Grassberger (2004): "Estimating mutual information"
#   de la Fuente et al (2004): MI vs Pearson in signal redundancy analysis
#   López de Prado (2020): "Machine Learning for Asset Managers" — chapter on
#   feature redundancy and the Correlation Shrinkage approach
#
# History buffer:
#   The MI matrix requires a history of (signal_name → list of past values).
#   MISignalFusion maintains a rolling 200-sample circular buffer per signal.
#   On first call (empty buffer), falls back to Pearson-correlation-based
#   redundancy as a bootstrap approximation.
# ═══════════════════════════════════════════════════════════════════════════════

class MISignalFusion:
    """
    Mutual Information-aware signal fusion.

    Detects which signals carry redundant information and discounts them,
    so the composite measures true independent information content.

    Usage:
        fusion = MISignalFusion()
        result = fusion.compute(signals_dict, weights_dict)
    """

    BUFFER_SIZE   = 200     # rolling history length per signal
    N_BINS        = 10      # discretization bins for MI estimation
    MI_FLOOR      = 1e-8    # avoid log(0)
    REDUNDANCY_CAP = 0.90   # cap NMI at 0.90 to avoid zeroing signals

    def __init__(self):
        # Rolling signal history: {signal_name: deque of float values}
        self._history: Dict[str, list] = {}
        self._mi_matrix: Dict[str, Dict[str, float]] = {}   # NMI cache
        self._mi_last_computed: float = 0.0
        self._MI_RECOMPUTE_INTERVAL = 300   # seconds: recompute MI every 5 min
        self._lock = threading.Lock()

    # ─────────────────────────────────────────────────────────────────────────
    # HISTORY BUFFER
    # ─────────────────────────────────────────────────────────────────────────

    def update_history(self, signals: Dict[str, float]):
        """Push current signal values into rolling history buffers."""
        with self._lock:
            for name, val in signals.items():
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    continue
                if name not in self._history:
                    self._history[name] = []
                buf = self._history[name]
                buf.append(float(np.clip(val, -1.0, 1.0)))
                if len(buf) > self.BUFFER_SIZE:
                    buf.pop(0)

    # ─────────────────────────────────────────────────────────────────────────
    # MI ESTIMATION
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _entropy(x: np.ndarray, n_bins: int = 10) -> float:
        """Shannon entropy H(X) from discretized 1D array."""
        counts, _ = np.histogram(x, bins=n_bins, range=(-1.0, 1.0))
        probs = counts / (counts.sum() + 1e-10)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs + 1e-10)))

    @staticmethod
    def _mutual_information(x: np.ndarray, y: np.ndarray,
                            n_bins: int = 10) -> float:
        """
        Estimate MI(X;Y) via discrete joint histogram.
        Returns normalized MI ∈ [0, 1] where 1 = identical.
        """
        if len(x) < 20 or len(y) < 20:
            return 0.0

        n = min(len(x), len(y))
        x, y = x[-n:], y[-n:]

        # Joint histogram
        joint, _, _ = np.histogram2d(x, y, bins=n_bins,
                                     range=[[-1,1],[-1,1]])
        joint = joint / (joint.sum() + 1e-10)

        # Marginals
        px = joint.sum(axis=1)
        py = joint.sum(axis=0)

        # MI = Σ p(x,y) log(p(x,y) / (p(x)*p(y)))
        outer = np.outer(px, py)
        mask  = (joint > 0) & (outer > 0)
        mi    = float(np.sum(joint[mask] * np.log(joint[mask] / outer[mask])))
        mi    = max(0.0, mi)

        # Normalize: NMI = MI / min(H(X), H(Y))
        hx = float(-np.sum(px[px>0] * np.log(px[px>0])))
        hy = float(-np.sum(py[py>0] * np.log(py[py>0])))
        denom = min(hx, hy)
        if denom < 1e-8:
            return 0.0

        nmi = min(1.0, mi / denom)
        return float(nmi)

    def _build_mi_matrix(self, signal_names: list) -> Dict[str, Dict[str, float]]:
        """
        Compute full NMI matrix for all signal pairs.
        Uses history buffers. Falls back to Pearson-based approximation
        if insufficient history (< 30 samples).
        """
        mi_mat: Dict[str, Dict[str, float]] = {n: {} for n in signal_names}

        for i, name_i in enumerate(signal_names):
            for j, name_j in enumerate(signal_names):
                if i >= j:
                    continue  # symmetric — only compute upper triangle

                hist_i = np.array(self._history.get(name_i, []))
                hist_j = np.array(self._history.get(name_j, []))

                if len(hist_i) >= 30 and len(hist_j) >= 30:
                    # Full MI estimation
                    nmi = self._mutual_information(hist_i, hist_j, self.N_BINS)
                else:
                    # Fallback: Pearson |r| as proxy (valid in Gaussian limit)
                    if len(hist_i) >= 5 and len(hist_j) >= 5:
                        n = min(len(hist_i), len(hist_j))
                        try:
                            r, _ = stats.pearsonr(hist_i[-n:], hist_j[-n:])
                            nmi  = float(min(1.0, abs(r)))
                        except Exception:
                            nmi = 0.0
                    else:
                        nmi = 0.0

                nmi = min(nmi, self.REDUNDANCY_CAP)
                mi_mat[name_i][name_j] = nmi
                mi_mat[name_j][name_i] = nmi

        return mi_mat

    # ─────────────────────────────────────────────────────────────────────────
    # EFFECTIVE WEIGHT COMPUTATION
    # ─────────────────────────────────────────────────────────────────────────

    def _effective_weights(self, base_weights: Dict[str, float],
                           mi_matrix: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Compute MI-adjusted effective weights.

        For each signal i:
            redundancy_i = Σ_{j≠i} base_weight_j * NMI(i, j)
            effective_w_i = base_weight_i / (1 + redundancy_i)

        Intuition: if signal A is 80% correlated with signals B and C,
        its effective weight shrinks because B and C are already capturing
        most of A's information content.

        After adjustment, weights are renormalized to sum to 1.
        """
        names = list(base_weights.keys())
        eff_w = {}

        for name in names:
            base_w     = float(base_weights[name])
            redundancy = 0.0

            for other in names:
                if other == name:
                    continue
                other_w = float(base_weights.get(other, 0))
                nmi     = float(mi_matrix.get(name, {}).get(other, 0.0))
                redundancy += other_w * nmi

            # Discount: signal with high redundancy gets smaller effective weight
            eff_w[name] = base_w / (1.0 + redundancy)

        # Renormalize to sum to 1
        total = sum(eff_w.values()) + 1e-10
        eff_w = {k: v / total for k, v in eff_w.items()}

        return eff_w

    # ─────────────────────────────────────────────────────────────────────────
    # MASTER COMPUTE
    # ─────────────────────────────────────────────────────────────────────────

    def compute(self, signals: Dict[str, float],
                base_weights: Dict[str, float]) -> Dict:
        """
        Compute MI-orthogonalized composite signal.

        Args:
            signals:      {signal_name: float ∈ [-1,+1]}
            base_weights: {signal_name: float > 0} (adaptive IC weights from AladdinScorer)

        Returns:
            composite:      MI-orthogonalized composite ∈ [-1,+1]
            effective_weights: actual weights after MI adjustment
            redundancy_pairs: top 3 most redundant signal pairs
            mi_matrix:      full NMI matrix (for diagnostics)
            naive_composite: simple weighted avg for comparison
        """
        # Update rolling history
        self.update_history(signals)

        names = [k for k in base_weights if k in signals]
        if not names:
            return {"composite": 0.0, "effective_weights": {}, "mi_matrix": {}}

        # Rebuild MI matrix if stale (every 5 minutes)
        now = time.time()
        with self._lock:
            if not self._mi_matrix or (now - self._mi_last_computed) > self._MI_RECOMPUTE_INTERVAL:
                self._mi_matrix          = self._build_mi_matrix(names)
                self._mi_last_computed   = now

            mi_matrix = dict(self._mi_matrix)

        # Effective weights
        sub_weights = {k: float(base_weights[k]) for k in names}
        eff_weights = self._effective_weights(sub_weights, mi_matrix)

        # Normalize signals
        norm_sigs = {}
        for k in names:
            v = signals.get(k, 0.0)
            norm_sigs[k] = float(np.clip(0.0 if v is None or np.isnan(float(v)) else v, -1, 1))

        # MI-orthogonalized composite
        composite = sum(eff_weights[k] * norm_sigs[k] for k in names)
        composite = float(np.clip(composite, -1.0, 1.0))

        # Naive composite for comparison (shows how much MI changes the answer)
        total_bw    = sum(sub_weights.values()) + 1e-10
        naive       = sum(sub_weights[k] * norm_sigs[k] for k in names) / total_bw
        naive       = float(np.clip(naive, -1.0, 1.0))

        # Top redundant pairs (for dashboard diagnostics)
        pairs = []
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                if j <= i: continue
                nmi = float(mi_matrix.get(ni, {}).get(nj, 0))
                if nmi > 0.05:
                    pairs.append((ni, nj, nmi))
        pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = [{"s1": a, "s2": b, "nmi": round(c, 4)}
                     for a, b, c in pairs[:5]]

        # Per-signal effective weight change
        weight_deltas = {
            k: round(eff_weights[k] - sub_weights[k] / total_bw, 4)
            for k in names
        }

        return {
            "composite":           round(composite, 4),
            "naive_composite":     round(naive, 4),
            "mi_adjustment":       round(composite - naive, 4),
            "effective_weights":   {k: round(v, 5) for k, v in eff_weights.items()},
            "weight_deltas":       weight_deltas,
            "redundancy_pairs":    top_pairs,
            "mi_matrix_size":      len(names),
            "history_samples":     min(len(self._history.get(n, [])) for n in names) if names else 0,
        }


# ── Module-level singleton (shared across all analyze_symbol() calls) ─────────
_MI_FUSION_SINGLETON: Optional["MISignalFusion"] = None

def get_mi_fusion() -> "MISignalFusion":
    """Return the module-level MISignalFusion singleton (thread-safe init)."""
    global _MI_FUSION_SINGLETON
    if _MI_FUSION_SINGLETON is None:
        _MI_FUSION_SINGLETON = MISignalFusion()
    return _MI_FUSION_SINGLETON


class SignalFusion:
    """
    Backward-compatible wrapper.
    entropy_weight() and weighted_composite() preserved for any code that uses them.
    New callers should use MISignalFusion directly or via get_mi_fusion().
    """

    @staticmethod
    def entropy_weight(signal_history: np.ndarray) -> float:
        """Shannon entropy-based signal weight. Low entropy = consistent = higher weight."""
        if len(signal_history) < 5:
            return 1.0
        bins    = np.linspace(-1, 1, 11)
        hist, _ = np.histogram(signal_history, bins=bins)
        probs   = hist / (hist.sum() + 1e-8)
        probs   = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return float(1 - entropy / np.log(10))

    @staticmethod
    def weighted_composite(signals: Dict[str, float],
                           weights: Dict[str, float],
                           signal_histories: Optional[Dict] = None) -> float:
        """Entropy-adjusted weighted composite (backward compat)."""
        total_w, total_s = 0.0, 0.0
        for key, sig in signals.items():
            if key not in weights:
                continue
            w = weights[key]
            if signal_histories and key in signal_histories:
                w *= SignalFusion.entropy_weight(np.array(signal_histories[key]))
            total_s += w * sig
            total_w += w
        return float(np.clip(total_s / max(total_w, 1e-8), -1, 1))

# ═══════════════════════════════════════════════════════════════════════════════
# 11. FAMA-FRENCH 5-FACTOR EXPOSURE & NEUTRALIZATION  (Step 9)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Fama-French (2015) five factors:
#   MKT-RF  — Market excess return (beta)
#   SMB     — Small Minus Big (size premium)
#   HML     — High Minus Low (value premium)
#   RMW     — Robust Minus Weak (profitability premium)
#   CMA     — Conservative Minus Aggressive (investment premium)
#
# Why neutralize?
#   A composite signal that scores a stock highly for high beta and small-cap
#   tilt has not found alpha — it has rediscovered known risk premia.
#   Neutralization isolates the idiosyncratic (stock-specific) component.
#
# Approach:
#   1. Build factor returns from ETF proxies (SPY, IWM, IWD, QUAL, MTUM)
#   2. Regress stock returns on 5 factors via OLS (60-day rolling window)
#   3. alpha_r2 = 1 - R² = idiosyncratic fraction
#   4. neutralized_signal = raw_signal * alpha_r2 (discount factor-driven signals)
#
# Academic:
#   Fama & French (2015): "A Five-Factor Asset Pricing Model", JFE
#   Harvey & Liu (2016): "...and the Cross-Section of Expected Returns", RFS
# ═══════════════════════════════════════════════════════════════════════════════

class FactorAnalysis:
    """
    Fama-French 5-Factor exposure analysis and signal neutralization.

    Backward-compatible: keeps original PCA methods (compute_factors,
    correlation_penalty) and adds FF5 regression and neutralization.
    """

    # ETF proxies for FF5 factors
    _ETF_PROXIES = {
        "MKT":  "SPY",
        "SMB":  "IWM",
        "HML":  "IWD",
        "RMW":  "QUAL",
        "CMA":  "MTUM",
    }

    _factor_cache: Dict    = {}
    _factor_cache_ts: float = 0.0
    _FACTOR_TTL = 86400

    _regression_cache: Dict = {}
    _REGRESSION_TTL = 4 * 3600

    # ─────────────────────────────────────────────────────────────────────────
    # ORIGINAL PCA METHODS (backward-compatible)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_factors(returns_matrix: np.ndarray, n_factors: int = 5) -> Dict:
        """PCA factor decomposition — kept for backward compatibility."""
        if returns_matrix.ndim < 2 or returns_matrix.shape[0] < 30 or returns_matrix.shape[1] < 2:
            return {}
        try:
            R   = returns_matrix - returns_matrix.mean(axis=0)
            cov = np.cov(R.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues  = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            n = min(n_factors, len(eigenvalues))
            ev = eigenvalues[:n] / (eigenvalues.sum() + 1e-10)
            return {"n_factors": n, "explained_var": ev.tolist(),
                    "total_explained": float(ev.sum()),
                    "loadings": eigenvectors[:, :n].tolist()}
        except Exception:
            return {}

    @staticmethod
    def correlation_penalty(symbol_returns: np.ndarray,
                            portfolio_returns: np.ndarray) -> float:
        """Position size penalty for high correlation with portfolio. Returns [0.3, 1.0]."""
        if len(portfolio_returns) < 20 or len(symbol_returns) < 20:
            return 1.0
        n = min(len(symbol_returns), len(portfolio_returns))
        r, _ = stats.pearsonr(symbol_returns[-n:], portfolio_returns[-n:])
        return float(max(0.3, 1.0 - abs(r) * 0.7))

    # ─────────────────────────────────────────────────────────────────────────
    # FF5 FACTOR RETURNS (ETF proxies)
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def get_factor_returns(cls, period: str = "1y") -> Dict:
        """
        Build daily FF5 factor returns from ETF proxies.
        Cached 24hr. Returns {MKT, SMB, HML, RMW, CMA, RF} as np.ndarrays.
        """
        now = time.time()
        if cls._factor_cache and (now - cls._factor_cache_ts) < cls._FACTOR_TTL:
            return cls._factor_cache

        factors: Dict = {}
        try:
            import yfinance as _yf
            from concurrent.futures import ThreadPoolExecutor, as_completed as _ac

            etfs = list(cls._ETF_PROXIES.values()) + ["BIL"]

            def _fetch(sym):
                try:
                    h = _yf.Ticker(sym).history(period=period)
                    if not h.empty:
                        c = h["Close"].values.astype(float)
                        return sym, np.diff(np.log(c + 1e-10))
                except Exception:
                    pass
                return sym, None

            raw: Dict = {}
            with ThreadPoolExecutor(max_workers=6) as pool:
                futs = {pool.submit(_fetch, s): s for s in etfs}
                for fut in _ac(futs, timeout=20):
                    try:
                        sym, ret = fut.result()
                        if ret is not None:
                            raw[sym] = ret
                    except Exception:
                        pass

            mkt = raw.get("SPY")
            if mkt is None:
                raise ValueError("SPY unavailable")

            rf  = raw.get("BIL", np.zeros(len(mkt)))
            n   = min(len(mkt), len(rf))

            def _excess(sym, n):
                r = raw.get(sym)
                if r is None:
                    return np.zeros(n)
                k = min(len(r), n)
                return r[-k:] - rf[-k:]

            mkt_rf = mkt[-n:] - rf[-n:]
            factors = {
                "MKT": mkt_rf,
                "SMB": _excess("IWM", n)  - mkt_rf,
                "HML": _excess("IWD", n)  - mkt_rf,
                "RMW": _excess("QUAL", n) - mkt_rf,
                "CMA": _excess("MTUM", n) - mkt_rf,
                "RF":  rf[-n:],
                "n_obs": n, "source": "ETF_PROXY",
            }
        except Exception as _fe:
            logger.debug(f"Factor download: {_fe}")
            n = 252
            factors = {k: np.zeros(n) for k in ["MKT","SMB","HML","RMW","CMA","RF"]}
            factors.update({"n_obs": n, "source": "FALLBACK_ZEROS"})

        cls._factor_cache    = factors
        cls._factor_cache_ts = time.time()
        return factors

    # ─────────────────────────────────────────────────────────────────────────
    # FF5 REGRESSION
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def regress_ff5(cls, symbol_returns: np.ndarray,
                    factor_returns: Dict = None,
                    window: int = 60) -> Dict:
        """
        OLS regression of stock excess returns on FF5 factors (60-day window).

        Returns alpha, betas, R², t-stats, residuals, factor contributions.
        """
        factor_names = ["MKT", "SMB", "HML", "RMW", "CMA"]

        if factor_returns is None:
            factor_returns = cls.get_factor_returns()

        try:
            r  = np.array(symbol_returns, dtype=float).flatten()
            rf = factor_returns.get("RF", np.zeros(len(r)))
            y  = r - rf[:len(r)]
            n  = min(len(y), window)
            y  = y[-n:]

            X_cols = []
            for fn in factor_names:
                f = factor_returns.get(fn, np.zeros(window))
                X_cols.append(f[-n:] if len(f) >= n else np.zeros(n))
            X     = np.column_stack(X_cols)
            X_int = np.column_stack([np.ones(n), X])

            betas_vec, _, _, _ = np.linalg.lstsq(X_int, y, rcond=None)
            alpha_daily = float(betas_vec[0])
            betas       = {fn: float(betas_vec[i+1]) for i, fn in enumerate(factor_names)}

            y_hat     = X_int @ betas_vec
            residuals = y - y_hat
            ss_tot    = np.var(y) * n + 1e-10
            ss_res    = np.sum(residuals**2)
            r2        = float(max(0.0, 1.0 - ss_res / ss_tot))

            sigma_resid = np.std(residuals) + 1e-10
            t_stats = {}
            for i, fn in enumerate(factor_names):
                t_stats[fn] = float(betas[fn] * (np.std(X[:, i]) + 1e-10) / sigma_resid * np.sqrt(n))

            total_var = np.var(y) + 1e-10
            factor_contrib = {fn: round(float(betas[fn]**2 * np.var(X[:, i])) / total_var, 4)
                              for i, fn in enumerate(factor_names)}

            return {
                "alpha_daily":       round(alpha_daily, 6),
                "alpha_annualized":  round(alpha_daily * TRADING_DAYS, 4),
                "betas":             {k: round(v, 4) for k, v in betas.items()},
                "r_squared":         round(r2, 4),
                "alpha_r2":          round(1.0 - r2, 4),
                "t_stats":           {k: round(v, 3) for k, v in t_stats.items()},
                "factor_contribution":factor_contrib,
                "residuals":         residuals.tolist(),
                "n_obs":             n,
            }

        except Exception as e:
            logger.debug(f"FF5 regression: {e}")
            return {"alpha_daily": 0.0, "alpha_annualized": 0.0,
                    "betas": {fn: 0.0 for fn in factor_names},
                    "r_squared": 0.0, "alpha_r2": 1.0,
                    "t_stats": {fn: 0.0 for fn in factor_names},
                    "factor_contribution": {fn: 0.0 for fn in factor_names},
                    "residuals": [], "n_obs": 0}

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL NEUTRALIZATION
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def neutralize_signal(cls, raw_signal: float, ff5_result: Dict,
                          mode: str = "alpha_weighted") -> Dict:
        """
        Discount raw composite signal by how much is FF5-factor-explained.

        neutralized_signal = raw_signal * max(0.1, alpha_r2)

        If R²=0.95 (mostly market beta), score shrinks to ~5% of original.
        If R²=0.20 (mostly idiosyncratic), score stays at ~80%.

        Returns: neutralized_signal, alpha_quality, mkt_beta, factor tilts.
        """
        alpha_r2       = float(ff5_result.get("alpha_r2", 1.0))
        betas          = ff5_result.get("betas", {})
        r2             = float(ff5_result.get("r_squared", 0.0))
        factor_contrib = ff5_result.get("factor_contribution", {})

        if   alpha_r2 >= 0.70: alpha_quality = "STRONG"
        elif alpha_r2 >= 0.50: alpha_quality = "GOOD"
        elif alpha_r2 >= 0.30: alpha_quality = "WEAK"
        else:                   alpha_quality = "FACTOR_DRIVEN"

        dominant = max(factor_contrib, key=lambda k: factor_contrib.get(k, 0), default="MKT")

        if mode == "beta_penalty":
            mkt_beta = abs(float(betas.get("MKT", 1.0)))
            discount = float(np.clip(1.5 / max(mkt_beta, 1.5), 0.3, 1.0))
        elif mode == "full":
            mkt_beta  = abs(float(betas.get("MKT", 1.0)))
            beta_mult = float(np.clip(1.5 / max(mkt_beta, 1.5), 0.3, 1.0))
            discount  = beta_mult * max(0.1, alpha_r2)
        else:   # alpha_weighted (default)
            discount = max(0.1, alpha_r2)

        neutralized = float(np.clip(raw_signal * discount, -1.0, 1.0))

        tilts = []
        for fn, (pl, nl) in [("MKT",("high-beta","low-beta")),
                              ("SMB",("small-cap","large-cap")),
                              ("HML",("value","growth")),
                              ("RMW",("profitable","weak-profit")),
                              ("CMA",("conservative","aggressive"))]:
            b = float(betas.get(fn, 0))
            if abs(b) > 0.3:
                tilts.append(f"{pl if b > 0 else nl} ({b:+.2f})")

        return {
            "neutralized_signal":  round(neutralized, 4),
            "raw_signal":          round(raw_signal, 4),
            "discount_applied":    round(1.0 - discount, 4),
            "alpha_quality":       alpha_quality,
            "alpha_r2":            round(alpha_r2, 4),
            "r_squared":           round(r2, 4),
            "dominant_factor":     dominant,
            "dominant_factor_pct": round(factor_contrib.get(dominant, 0), 4),
            "mkt_beta":            round(float(betas.get("MKT", 1.0)), 3),
            "smb_beta":            round(float(betas.get("SMB", 0.0)), 3),
            "hml_beta":            round(float(betas.get("HML", 0.0)), 3),
            "rmw_beta":            round(float(betas.get("RMW", 0.0)), 3),
            "cma_beta":            round(float(betas.get("CMA", 0.0)), 3),
            "alpha_annualized":    round(float(ff5_result.get("alpha_annualized", 0)), 4),
            "factor_tilts":        tilts,
            "factor_tilt_str":     " | ".join(tilts) if tilts else "none",
            "label": (
                f"FF5: alpha_quality={alpha_quality} | discount={1.0-discount:.0%} "
                f"| dominant={dominant} | MKT_beta={betas.get('MKT',0):+.2f}"
            ),
        }

    @classmethod
    def analyze_stock(cls, symbol_returns: np.ndarray,
                      raw_signal: float,
                      neutralization_mode: str = "alpha_weighted") -> Dict:
        """Full FF5 pipeline: factor download → regression → neutralize."""
        factors = cls.get_factor_returns()
        ff5     = cls.regress_ff5(symbol_returns, factors)
        neutral = cls.neutralize_signal(raw_signal, ff5, neutralization_mode)
        return {**ff5, **neutral}

    @classmethod
    def portfolio_factor_risk(cls, weights: Dict,
                              symbol_betas: Dict) -> Dict:
        """Aggregate portfolio FF5 factor exposure from per-symbol betas."""
        factor_names = ["MKT", "SMB", "HML", "RMW", "CMA"]
        port_betas   = {fn: 0.0 for fn in factor_names}
        total_w      = sum(abs(v) for v in weights.values()) + 1e-10
        for sym, w in weights.items():
            betas = symbol_betas.get(sym, {})
            for fn in factor_names:
                port_betas[fn] += (w / total_w) * float(betas.get(fn, 0))
        max_factor = max(port_betas, key=lambda k: abs(port_betas[k]))
        return {
            "portfolio_betas":   {k: round(v, 4) for k, v in port_betas.items()},
            "dominant_exposure": max_factor,
            "net_mkt_beta":      round(port_betas["MKT"], 3),
            "net_size_tilt":     round(port_betas["SMB"], 3),
            "net_value_tilt":    round(port_betas["HML"], 3),
        }

# 12. TECHNICAL INDICATOR SUITE — Advanced
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedTechnicals:
    """
    Extended technical indicator library beyond the basics.
    All signals normalized to [-1, +1].
    """

    @staticmethod
    def ichimoku(high, low, close) -> Dict:
        """Ichimoku Kinko Hyo cloud system."""
        h, l, c = np.array(high), np.array(low), np.array(close)
        if len(c) < 52:
            return {}

        tenkan  = (np.max(h[-9:])  + np.min(l[-9:]))  / 2   # Conversion (9)
        kijun   = (np.max(h[-26:]) + np.min(l[-26:])) / 2   # Base (26)
        senkou_a = (tenkan + kijun) / 2                       # Cloud A
        senkou_b = (np.max(h[-52:]) + np.min(l[-52:])) / 2  # Cloud B

        price_vs_cloud = 1.0 if c[-1] > max(senkou_a, senkou_b) else (
                        -1.0 if c[-1] < min(senkou_a, senkou_b) else 0.0)
        tk_cross = 1.0 if tenkan > kijun else -1.0 if tenkan < kijun else 0.0

        return {
            "price_vs_cloud": float(price_vs_cloud),
            "tk_cross":       float(tk_cross),
            "score":          float((price_vs_cloud + tk_cross) / 2),
        }

    @staticmethod
    def williams_r(high, low, close, period=14) -> float:
        """Williams %R oscillator. -100 = extremely oversold."""
        h, l, c = np.array(high[-period:]), np.array(low[-period:]), np.array(close)
        if len(h) < period:
            return 0.0
        highest = np.max(h)
        lowest  = np.min(l)
        if highest == lowest:
            return 0.0
        wr = (highest - c[-1]) / (highest - lowest) * -100
        # Normalize: -100 = buy (+1), 0 = sell (-1)
        return float(np.clip((wr + 50) / 50, -1, 1))

    @staticmethod
    def commodity_channel_index(high, low, close, period=20) -> float:
        """CCI — identifies cyclical trends."""
        h, l, c = np.array(high), np.array(low), np.array(close)
        if len(c) < period:
            return 0.0
        tp     = (h[-period:] + l[-period:] + c[-period:]) / 3
        tp_mean = np.mean(tp)
        md     = np.mean(np.abs(tp - tp_mean))
        cci    = (tp[-1] - tp_mean) / (0.015 * md + 1e-8)
        return float(np.clip(-cci / 200, -1, 1))  # Inverted: overbought = sell

    @staticmethod
    def on_balance_volume_signal(close, volume) -> float:
        """OBV trend signal — volume confirms price direction."""
        c, v = np.array(close), np.array(volume)
        if len(c) < 20:
            return 0.0
        obv = np.zeros(len(c))
        for i in range(1, len(c)):
            obv[i] = obv[i-1] + (v[i] if c[i] > c[i-1] else -v[i] if c[i] < c[i-1] else 0)
        # Trend: OBV going up while price flat = bullish divergence
        obv_trend = (obv[-1] - np.mean(obv[-20:])) / (np.std(obv[-20:]) + 1e-8)
        return float(np.clip(obv_trend / 3, -1, 1))

    @staticmethod
    def money_flow_index(high, low, close, volume, period=14) -> float:
        """MFI — volume-weighted RSI."""
        h, l, c, v = np.array(high), np.array(low), np.array(close), np.array(volume)
        if len(c) < period + 1:
            return 0.0
        tp = (h + l + c) / 3
        mf = tp * v
        pos_mf = np.sum([mf[i] for i in range(-period, 0) if tp[i] > tp[i-1]])
        neg_mf = np.sum([mf[i] for i in range(-period, 0) if tp[i] < tp[i-1]])
        if neg_mf == 0:
            return 1.0
        mfr = pos_mf / neg_mf
        mfi = 100 - 100 / (1 + mfr)
        return float((50 - mfi) / 50)

    @staticmethod
    def vwap_signal(close, volume, high, low) -> float:
        """VWAP deviation signal."""
        c, v, h, l = np.array(close), np.array(volume), np.array(high), np.array(low)
        if len(c) < 5:
            return 0.0
        tp   = (h + l + c) / 3
        vwap = np.cumsum(tp * v) / (np.cumsum(v) + 1e-8)
        deviation = (c[-1] - vwap[-1]) / (vwap[-1] + 1e-8)
        return float(np.clip(-deviation * 20, -1, 1))  # Below VWAP = buy signal

    @staticmethod
    def supertrend(high, low, close, period=10, multiplier=3.0) -> float:
        """SuperTrend indicator signal."""
        h, l, c = np.array(high), np.array(low), np.array(close)
        if len(c) < period + 1:
            return 0.0

        # ATR
        tr = np.maximum(h[1:] - l[1:],
             np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        atr = np.zeros(len(tr))
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period

        mid = (h[1:] + l[1:]) / 2
        upper = mid + multiplier * atr
        lower = mid - multiplier * atr

        trend = 1 if c[-1] > lower[-1] else -1
        return float(trend)

    @staticmethod
    def compute_all_advanced(df: pd.DataFrame) -> Dict[str, float]:
        """Compute all advanced technicals at once."""
        if df is None or len(df) < 55:
            return {}
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        v = df["volume"].values

        out = {}
        try:
            ich = AdvancedTechnicals.ichimoku(h, l, c)
            out["ichimoku"] = ich.get("score", 0)
        except Exception:
            pass
        try:
            out["williams_r"] = AdvancedTechnicals.williams_r(h, l, c)
        except Exception:
            pass
        try:
            out["cci"] = AdvancedTechnicals.commodity_channel_index(h, l, c)
        except Exception:
            pass
        try:
            out["obv"] = AdvancedTechnicals.on_balance_volume_signal(c, v)
        except Exception:
            pass
        try:
            out["mfi"] = AdvancedTechnicals.money_flow_index(h, l, c, v)
        except Exception:
            pass
        try:
            out["vwap"] = AdvancedTechnicals.vwap_signal(c, v, h, l)
        except Exception:
            pass
        try:
            out["supertrend"] = AdvancedTechnicals.supertrend(h, l, c)
        except Exception:
            pass

        if out:
            out["advanced_composite"] = float(np.mean(list(out.values())))

        return out


# ═══════════════════════════════════════════════════════════════════════════════
# 12b. CROSS-ASSET MACRO SCORER  (Step 7)
# ═══════════════════════════════════════════════════════════════════════════════

class MacroScorer:
    """
    Converts cross-asset market data into a single macro_regime score [-1, +1]
    for the AladdinScorer macro_regime slot.

    Five orthogonal signal sources (each contributes ≤ 0.2 to final score):

    1. Yield Curve (10Y–3M spread)
       +0.2  steep  (>1.5%): growth regime, historically bullish for equities
        0.0  flat   (0–0.5%): neutral
       -0.2  inverted (<0%): recession signal — negative 12–18m forward

    2. VIX Regime
       -0.2  VIX > 30 (crisis): forced selling, correlations go to 1
       -0.1  VIX 20–30 (elevated): caution
        0.0  VIX 15–20 (normal)
       +0.1  VIX < 15 (complacency): low vol = favorable carry environment

    3. VIX Term Structure (contango vs backwardation)
       Ratio = VIX9D / VIX3M
       < 0.85 (deep contango): calm, market expects vol to rise later but calm now → bullish
       > 1.10 (backwardation): near-term fear spikes above long-term → bearish/hedge

    4. Credit Spread Proxy (HYG vs LQD relative performance)
       HYG = iShares High Yield Corp Bond ETF
       LQD = iShares Investment Grade Corp Bond ETF
       Spread signal = HYG_5d_return - LQD_5d_return
       Positive (HY outperforms IG): risk-on, credit conditions easy → bullish equities
       Negative (HY underperforms): credit stress, risk-off → bearish equities

    5. Dollar (DXY) Momentum
       DXY 20d momentum (% change)
       Strongly positive DXY (+3% over 20 days): EM headwind, multinational EPS pressure
       Negative DXY: commodity tailwind, EM flows → bullish global equities

    Academic basis:
      Ilmanen (2011): cross-asset macro signals improve equity timing Sharpe by 0.3–0.5
      Koijen et al (2018): carry signals (yield curve) predict equity risk premiums
      Ang & Piazzesi (2003): yield curve factors explain 85% of bond return variation
    """

    # Signal weights (must sum to 1.0)
    WEIGHTS = {
        "yield_curve":    0.30,   # strongest individual IC historically
        "vix_level":      0.25,
        "vix_term":       0.15,
        "credit_spread":  0.20,
        "dxy_momentum":   0.10,
    }

    @staticmethod
    def yield_curve_score(spread_10y_3m: float) -> float:
        """
        Yield curve spread → [-1, +1] signal.
        10Y–3M spread: positive = normal, negative = inverted.

        Calibration based on NBER recession analysis:
          - Inversion (< 0) has predicted 8/8 post-WWII recessions
          - Average equity return in inversion year: -4% vs +12% in steep years
        """
        if spread_10y_3m >= 1.5:   return +0.8
        if spread_10y_3m >= 0.75:  return +0.4
        if spread_10y_3m >= 0.25:  return +0.1
        if spread_10y_3m >= -0.25: return -0.1
        if spread_10y_3m >= -0.75: return -0.5
        return -0.9   # deep inversion

    @staticmethod
    def vix_level_score(vix: float) -> float:
        """VIX level → [-1, +1]. High VIX = bearish."""
        if vix < 12:    return +0.6   # extreme complacency (slight warning)
        if vix < 15:    return +0.4
        if vix < 18:    return +0.2
        if vix < 22:    return  0.0
        if vix < 27:    return -0.3
        if vix < 35:    return -0.6
        return -1.0   # crisis

    @staticmethod
    def vix_term_score(vix9d: float, vix: float, vix3m: float) -> float:
        """
        VIX term structure: VIX9D / VIX3M ratio.
        Contango (<1): calm, normal premium — bullish
        Backwardation (>1): near-term fear above long-term — bearish
        """
        if vix9d <= 0 or vix3m <= 0:
            # Fallback: use VIX level-to-VIX3M if available, else neutral
            if vix > 0 and vix3m > 0:
                ratio = vix / vix3m
            else:
                return 0.0
        else:
            ratio = vix9d / vix3m

        if ratio < 0.80:   return +0.5   # deep contango — calmest signal
        if ratio < 0.90:   return +0.3
        if ratio < 0.97:   return +0.1
        if ratio < 1.03:   return  0.0
        if ratio < 1.10:   return -0.3
        if ratio < 1.20:   return -0.6
        return -1.0   # extreme backwardation (2020-style spike)

    @staticmethod
    def credit_spread_score(hyg_5d_ret: float, lqd_5d_ret: float) -> float:
        """
        Credit spread signal: HYG relative to LQD (high yield vs investment grade).
        When HY outperforms IG → credit conditions easy → risk-on → bullish equities.
        """
        spread = hyg_5d_ret - lqd_5d_ret   # positive = HY outperforms = risk-on
        if spread >= 1.5:    return +0.8
        if spread >= 0.5:    return +0.4
        if spread >= 0.0:    return +0.1
        if spread >= -0.5:   return -0.2
        if spread >= -1.5:   return -0.5
        return -0.9   # severe credit stress (HY crashes vs IG)

    @staticmethod
    def dxy_momentum_score(dxy_20d_change_pct: float) -> float:
        """
        DXY 20-day momentum → equity impact.
        Strong USD = headwind for SPX earnings and EM flows.
        Note: non-linear — a little USD strength is neutral, a lot is bearish.
        """
        d = dxy_20d_change_pct
        if d < -3.0:    return +0.6   # falling USD = broadly bullish
        if d < -1.5:    return +0.3
        if d < +1.5:    return  0.0   # neutral zone
        if d < +3.0:    return -0.3
        return -0.6    # dollar surge = EM/commodity/multinational headwind

    @classmethod
    def compute(
        cls,
        yield_10y:     float = 4.3,
        yield_3m:      float = 5.3,
        yield_2y:      float = 4.7,
        yield_30y:     float = 4.5,
        vix:           float = 20.0,
        vix9d:         float = 0.0,
        vix3m:         float = 0.0,
        hyg_5d_ret:    float = 0.0,
        lqd_5d_ret:    float = 0.0,
        dxy:           float = 104.0,
        dxy_20d_ago:   float = 0.0,
        crude_oil:     float = 75.0,
        gold:          float = 2000.0,
        fed_rate:      float = 5.25,
        cpi:           float = 3.2,
        unemployment:  float = 3.9,
        **kwargs,
    ) -> Dict:
        """
        Compute unified macro regime score from cross-asset inputs.

        Returns:
            macro_score:     [-1, +1] for AladdinScorer macro_regime slot
            macro_label:     STRONG_BULL / BULL / NEUTRAL / BEAR / CRISIS
            sub_scores:      breakdown per signal category
            regime_factors:  human-readable key drivers
            risk_indicators: specific warning flags
        """
        # ── Individual signal scores ───────────────────────────────────────
        spread_10y_3m = yield_10y - yield_3m
        spread_10y_2y = yield_10y - yield_2y   # secondary spread check
        dxy_mom = ((dxy - dxy_20d_ago) / (dxy_20d_ago + 1e-8) * 100
                   if dxy_20d_ago > 0 else 0.0)

        yc_score  = cls.yield_curve_score(spread_10y_3m)
        vix_score = cls.vix_level_score(vix)
        ts_score  = cls.vix_term_score(vix9d, vix, vix3m)
        cs_score  = cls.credit_spread_score(hyg_5d_ret, lqd_5d_ret)
        dx_score  = cls.dxy_momentum_score(dxy_mom)

        # ── Composite ─────────────────────────────────────────────────────
        w = cls.WEIGHTS
        raw = (
            w["yield_curve"]   * yc_score  +
            w["vix_level"]     * vix_score +
            w["vix_term"]      * ts_score  +
            w["credit_spread"] * cs_score  +
            w["dxy_momentum"]  * dx_score
        )
        macro_score = float(np.clip(raw, -1.0, 1.0))

        # ── Override: crisis gate ──────────────────────────────────────────
        # If multiple crisis signals fire simultaneously, hard-cap to bearish
        crisis_flags = sum([
            vix > 35,
            spread_10y_3m < -1.0,
            hyg_5d_ret - lqd_5d_ret < -2.0,
        ])
        if crisis_flags >= 2:
            macro_score = min(macro_score, -0.5)

        # ── Label ─────────────────────────────────────────────────────────
        if macro_score >= 0.4:    label = "STRONG_BULL"
        elif macro_score >= 0.15: label = "BULL"
        elif macro_score >= -0.15:label = "NEUTRAL"
        elif macro_score >= -0.4: label = "BEAR"
        else:                      label = "CRISIS"

        # ── Key drivers (top 2 most impactful) ───────────────────────────
        scored = sorted([
            ("yield_curve",   yc_score,  w["yield_curve"]),
            ("vix_level",     vix_score, w["vix_level"]),
            ("vix_term",      ts_score,  w["vix_term"]),
            ("credit_spread", cs_score,  w["credit_spread"]),
            ("dxy_momentum",  dx_score,  w["dxy_momentum"]),
        ], key=lambda x: abs(x[1] * x[2]), reverse=True)

        factors = []
        for name, score, weight in scored[:3]:
            direction = "bullish" if score > 0 else "bearish"
            if abs(score) > 0.05:
                factors.append(f"{name.replace('_', ' ')}: {direction} ({score:+.2f})")

        # ── Risk indicators ───────────────────────────────────────────────
        risks = []
        if spread_10y_3m < 0:
            risks.append(f"YIELD_CURVE_INVERTED ({spread_10y_3m:+.2f}%)")
        if spread_10y_2y < 0:
            risks.append(f"2Y10Y_INVERTED ({spread_10y_2y:+.2f}%)")
        if vix > 25:
            risks.append(f"VIX_ELEVATED ({vix:.1f})")
        if vix9d > 0 and vix3m > 0 and vix9d > vix3m * 1.05:
            risks.append(f"VIX_BACKWARDATION ({vix9d/vix3m:.2f}x)")
        if hyg_5d_ret - lqd_5d_ret < -1.0:
            risks.append(f"CREDIT_STRESS (HY-IG={hyg_5d_ret-lqd_5d_ret:.1f}%)")
        if dxy_mom > 2.5:
            risks.append(f"STRONG_DOLLAR (+{dxy_mom:.1f}% 20d)")

        # ── Real rate proxy ───────────────────────────────────────────────
        real_rate = fed_rate - cpi   # crude proxy for real Fed Funds rate
        # Very high real rates = contractionary, slightly bearish equities
        if real_rate > 2.0:
            macro_score = max(-1.0, macro_score - 0.1)
            risks.append(f"TIGHT_REAL_RATE (+{real_rate:.1f}%)")

        return {
            "macro_score":    round(macro_score, 4),
            "macro_label":    label,
            "sub_scores": {
                "yield_curve":    round(yc_score, 3),
                "vix_level":      round(vix_score, 3),
                "vix_term":       round(ts_score, 3),
                "credit_spread":  round(cs_score, 3),
                "dxy_momentum":   round(dx_score, 3),
            },
            "inputs": {
                "yield_10y":       round(yield_10y, 3),
                "yield_3m":        round(yield_3m, 3),
                "yield_2y":        round(yield_2y, 3),
                "spread_10y_3m":   round(spread_10y_3m, 3),
                "spread_10y_2y":   round(spread_10y_2y, 3),
                "vix":             round(vix, 2),
                "vix9d":           round(vix9d, 2) if vix9d > 0 else None,
                "vix3m":           round(vix3m, 2) if vix3m > 0 else None,
                "dxy":             round(dxy, 2),
                "dxy_momentum_pct":round(dxy_mom, 2),
                "hyg_5d_ret":      round(hyg_5d_ret, 3),
                "lqd_5d_ret":      round(lqd_5d_ret, 3),
                "real_rate":       round(real_rate, 2),
                "unemployment":    round(unemployment, 2),
                "cpi":             round(cpi, 2),
            },
            "regime_factors":  factors,
            "risk_indicators": risks,
            "crisis_flags":    crisis_flags,
            "weights":         cls.WEIGHTS,
        }

    @classmethod
    def from_macro_cache(cls, macro_data: Dict) -> Dict:
        """
        Convenience wrapper: compute MacroScorer from the /api/macro response dict.
        Handles missing fields gracefully with sensible defaults.
        """
        raw = macro_data.get("raw", {})

        def _val(key, default=0.0):
            d = macro_data.get(key) or raw.get(key) or {}
            if isinstance(d, dict):
                return float(d.get("value") or d.get("close") or default)
            try:
                return float(d)
            except Exception:
                return default

        # Yield curve
        y10  = _val("10Y_yield", 4.3)
        y3m  = _val("3M_yield",  5.3)
        y2y  = _val("2Y_yield",  4.7)
        y30  = _val("30Y_yield", 4.5)

        # Try direct keys if nested not present
        if y10 == 4.3:
            yc = macro_data.get("yield_curve", [])
            if isinstance(yc, list):
                for item in yc:
                    if item.get("tenor") == "10Y": y10 = float(item.get("yield", y10))
                    if item.get("tenor") == "3M":  y3m = float(item.get("yield", y3m))
                    if item.get("tenor") == "5Y":  pass
                    if item.get("tenor") == "30Y": y30 = float(item.get("yield", y30))

        vix  = _val("vix",  20.0)
        dxy  = _val("dxy",  104.0)

        # VIX term structure: fetch VIX9D and VIX3M from yfinance if not in cache
        vix9d = _val("vix9d", 0.0)
        vix3m = _val("vix3m", 0.0)

        # Credit spread: HYG vs LQD 5d returns
        hyg_5d = _val("hyg_5d_ret", 0.0)
        lqd_5d = _val("lqd_5d_ret", 0.0)
        # Fallback: use change_pct from raw
        if hyg_5d == 0.0:
            hyg_5d = float((raw.get("hyg") or {}).get("change_pct", 0.0))
        if lqd_5d == 0.0:
            lqd_5d = float((raw.get("lqd") or {}).get("change_pct", 0.0))

        dxy_ago = _val("dxy_20d_ago", 0.0)
        fed_rate     = _val("fed_rate",     5.25)
        cpi          = _val("cpi",          3.2)
        unemployment = _val("unemployment", 3.9)

        return cls.compute(
            yield_10y=y10, yield_3m=y3m, yield_2y=y2y, yield_30y=y30,
            vix=vix, vix9d=vix9d, vix3m=vix3m,
            hyg_5d_ret=hyg_5d, lqd_5d_ret=lqd_5d,
            dxy=dxy, dxy_20d_ago=dxy_ago,
            fed_rate=fed_rate, cpi=cpi, unemployment=unemployment,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 13. COMPOSITE SCORER — The Aladdin-Style Master Ranker
# ═══════════════════════════════════════════════════════════════════════════════

class AladdinScorer:
    """
    Combines ALL mathematical signals into one master score.
    Renaissance-style: 50+ factors → single actionable signal.
    Named after BlackRock's Aladdin, but built like Medallion Fund.

    ADAPTIVE: Weights are NOT static. They update weekly based on realized
    Information Coefficient (IC) of each signal vs actual forward returns.
    Weights are persisted to disk and survive server restarts.
    This is the core of how Renaissance compounded their edge over 35 years.
    """

    # Default weights — these are overwritten by adaptive IC-based weights
    # after sufficient trade history accumulates (typically 3-4 weeks).
    #
    # L2 microstructure signals (l2_spread, l2_imbalance, l2_vwap_dev) close
    # audit Gap 3 — Alpaca provides free real-time bid/ask data.
    # These start at low weight and will be IC-promoted if they add alpha.
    _DEFAULT_WEIGHTS = {
        "technical_composite":    0.15,
        "advanced_composite":     0.10,
        "ml_score":               0.18,
        "hurst_blended":          0.10,
        "ou_reversion":           0.08,
        "kalman_trend":           0.07,
        "vol_regime":             0.04,
        "news_sentiment":         0.10,
        "options_flow":           0.07,
        "macro_regime":           0.04,
        # ── Level-2 microstructure signals (Gap 3 partial fix) ─────────────
        # l2_spread    : bid-ask spread ratio — wide spread = informed flow
        # l2_imbalance : (bid_size - ask_size) / total — order pressure
        # l2_vwap_dev  : price deviation from VWAP — mean-reversion signal
        "l2_spread":              0.03,
        "l2_imbalance":           0.02,
        "l2_vwap_dev":            0.02,
    }

    # Weight bounds — no single signal can dominate or be zeroed out entirely
    _MIN_WEIGHT = 0.02   # floor: signal stays alive but minimal
    _MAX_WEIGHT = 0.35   # ceiling: no single signal above 35%

    # Live weights — mutated in place by update_weights_from_ic()
    WEIGHTS = dict(_DEFAULT_WEIGHTS)

    # IC tracking per signal — rolling EWM IC values
    _ic_ewm: Dict[str, float] = {}          # signal → current EWM-IC
    _ic_history: Dict[str, list] = {}       # signal → list of daily IC readings
    _ic_consecutive_weak: Dict[str, int] = {}   # signal → weeks of IC < threshold
    _ic_consecutive_strong: Dict[str, int] = {} # signal → weeks of IC > 0.10
    _weights_file: str = "aladdin_weights.json"
    _last_update: Optional[datetime] = None
    _update_lock = threading.Lock()

    # ── Persistence ───────────────────────────────────────────────────────────

    @classmethod
    def load_weights(cls):
        """Load adaptive weights from disk. Called at server startup."""
        try:
            import json, os
            if os.path.exists(cls._weights_file):
                with open(cls._weights_file, "r") as f:
                    data = json.load(f)
                loaded_w = data.get("weights", {})
                ic_data  = data.get("ic_ewm", {})
                last_upd = data.get("last_update", "")
                # Only apply if valid keys and weights sum > 0
                valid_keys = [k for k in loaded_w if k in cls._DEFAULT_WEIGHTS]
                if len(valid_keys) >= len(cls._DEFAULT_WEIGHTS) * 0.8:
                    for k in valid_keys:
                        cls.WEIGHTS[k] = float(np.clip(loaded_w[k],
                                                       cls._MIN_WEIGHT, cls._MAX_WEIGHT))
                    cls._ic_ewm = {k: float(v) for k, v in ic_data.items()}
                    logger.info(f"✅ AladdinScorer: loaded adaptive weights from {cls._weights_file}")
                    logger.info(f"   Top signal: {max(cls.WEIGHTS, key=cls.WEIGHTS.get)} "
                                f"= {max(cls.WEIGHTS.values()):.3f}")
                else:
                    logger.info("AladdinScorer: using default weights (insufficient history)")
        except Exception as e:
            logger.warning(f"AladdinScorer weight load failed (using defaults): {e}")

    @classmethod
    def save_weights(cls):
        """Persist current adaptive weights to disk."""
        try:
            import json
            data = {
                "weights":     cls.WEIGHTS,
                "ic_ewm":      cls._ic_ewm,
                "last_update": datetime.now().isoformat(),
                "note": "Auto-generated by AladdinScorer adaptive weight system. Do not edit manually.",
            }
            with open(cls._weights_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"AladdinScorer weight save failed: {e}")

    # ── IC Update (called by SignalDecayTracker after each closed trade) ──────

    @classmethod
    def update_signal_ic(cls, signal_name: str, predicted_direction: float,
                         actual_return_pct: float):
        """
        Update EWM-IC for one signal after a trade closes.

        Called from execution_engine.py after every position close.
        predicted_direction: +1 = predicted up, -1 = predicted down
        actual_return_pct:   realized P&L percentage (signed)

        IC contribution = sign(prediction) * sign(actual) → +1 or -1
        EWM-IC = 0.92 * IC_prev + 0.08 * IC_new   (≈30-trade half-life)
        """
        if signal_name not in cls._DEFAULT_WEIGHTS:
            return

        with cls._update_lock:
            # IC contribution: correct direction = +1, wrong = -1
            ic_contrib = float(np.sign(predicted_direction) * np.sign(actual_return_pct))

            # EWM update (alpha=0.08 → half-life ≈ 8 trades)
            prev_ic = cls._ic_ewm.get(signal_name, 0.0)
            new_ic  = 0.92 * prev_ic + 0.08 * ic_contrib
            cls._ic_ewm[signal_name] = float(new_ic)

            # Track history for statistics
            if signal_name not in cls._ic_history:
                cls._ic_history[signal_name] = []
            cls._ic_history[signal_name].append({
                "ic": new_ic,
                "ts": datetime.now().isoformat(),
            })
            # Keep last 200 readings
            cls._ic_history[signal_name] = cls._ic_history[signal_name][-200:]

    @classmethod
    def update_weights_from_ic(cls, force: bool = False):
        """
        Recalibrate WEIGHTS based on accumulated EWM-IC values.

        Called:
          - Every time a position closes (light update)
          - Weekly by background scheduler (full recalibration)

        Algorithm (mirrors Renaissance's adaptive weight system):
          1. Score each signal: new_weight ∝ max(IC, 0) ^ 0.5
             (square-root dampening prevents winner-take-all)
          2. Apply floor/ceiling bounds
          3. Renormalize to sum = 1.0
          4. Persist to disk
        """
        if not cls._ic_ewm:
            return  # No IC data yet — keep default weights

        min_ic_to_be_active = 0.02  # below this → signal gets floor weight only

        with cls._update_lock:
            new_weights = {}

            for signal_name, default_w in cls._DEFAULT_WEIGHTS.items():
                ic = cls._ic_ewm.get(signal_name, 0.0)  # EWM-IC, range [-1, +1]

                if ic >= min_ic_to_be_active:
                    # Active signal: weight ∝ sqrt(IC) × default_weight
                    # sqrt dampening: IC=0.04 → 0.2x, IC=0.09 → 0.3x, IC=0.16 → 0.4x
                    ic_score = float(np.sqrt(max(ic, 0.0)))
                    new_weights[signal_name] = default_w * (0.5 + ic_score * 2.5)
                elif ic >= -min_ic_to_be_active:
                    # Neutral signal: slight downweight but keep alive
                    new_weights[signal_name] = default_w * 0.6
                else:
                    # Negative IC signal: punish it but don't kill it
                    # (may recover — Renaissance keeps dormant signals at floor)
                    new_weights[signal_name] = cls._MIN_WEIGHT

            # Apply bounds
            for k in new_weights:
                new_weights[k] = float(np.clip(new_weights[k],
                                               cls._MIN_WEIGHT, cls._MAX_WEIGHT))

            # Renormalize to sum = 1.0
            total = sum(new_weights.values())
            if total > 0:
                for k in new_weights:
                    new_weights[k] = round(new_weights[k] / total, 6)

            cls.WEIGHTS = new_weights
            cls._last_update = datetime.now()
            cls.save_weights()

            logger.info(f"🔄 AladdinScorer weights recalibrated from IC data:")
            sorted_w = sorted(new_weights.items(), key=lambda x: -x[1])
            for name, w in sorted_w[:3]:
                ic_val = cls._ic_ewm.get(name, 0)
                logger.info(f"   {name}: weight={w:.4f}  IC={ic_val:+.4f}")

    @classmethod
    def get_ic_stats(cls) -> Dict:
        """Return current IC and weight stats for dashboard display."""
        stats = {}
        for name in cls._DEFAULT_WEIGHTS:
            stats[name] = {
                "weight":   round(cls.WEIGHTS.get(name, 0), 4),
                "ic_ewm":   round(cls._ic_ewm.get(name, 0), 4),
                "active":   cls._ic_ewm.get(name, 0) >= 0.02,
                "history_n": len(cls._ic_history.get(name, [])),
            }
        return {
            "signal_stats":  stats,
            "last_update":   cls._last_update.isoformat() if cls._last_update else "never",
            "total_trades_tracked": max(len(v) for v in cls._ic_history.values()) if cls._ic_history else 0,
        }

    @staticmethod
    def compute(signals: Dict) -> Dict:
        """
        Master composite score from all factor signals.

        Step 10 upgrade: uses MISignalFusion to detect and discount
        redundant signals before forming the composite.

        Pipeline:
          1. Sanitize + clip all 10 sub-signals to [-1, +1]
          2. Pass signals + adaptive IC weights to MISignalFusion.compute()
          3. MISignalFusion builds NMI matrix, adjusts effective weights,
             returns MI-orthogonalized composite
          4. Derive direction + conviction from composite
          5. Expose MI diagnostics (effective weights, redundant pairs) in output

        Falls back to naive weighted average if MI fusion errors.
        """
        raw_signals: Dict[str, float] = {}
        for k in AladdinScorer.WEIGHTS:
            val = signals.get(k, 0.0)
            try:
                v = float(val) if val is not None else 0.0
                raw_signals[k] = float(np.clip(0.0 if np.isnan(v) else v, -1, 1))
            except Exception:
                raw_signals[k] = 0.0

        # ── MI-orthogonalized composite (Step 10) ─────────────────────────
        mi_result: Dict = {}
        try:
            fusion    = get_mi_fusion()
            mi_result = fusion.compute(raw_signals, dict(AladdinScorer.WEIGHTS))
            composite = float(mi_result.get("composite", 0.0))
        except Exception as _mi_e:
            logger.debug(f"MISignalFusion fallback: {_mi_e}")
            # Graceful degradation: plain weighted average
            total_w = sum(AladdinScorer.WEIGHTS.values()) + 1e-10
            composite = float(np.clip(
                sum(AladdinScorer.WEIGHTS.get(k, 0) * v for k, v in raw_signals.items()) / total_w,
                -1, 1
            ))

        # Normalized to [0, 1] for dashboard
        normalized = (composite + 1) / 2

        # Direction thresholds (unchanged)
        if normalized > 0.62:
            direction = "BUY"
        elif normalized < 0.38:
            direction = "SHORT"
        else:
            direction = "HOLD"

        conviction = float(abs(normalized - 0.5) * 2)

        return {
            "composite":        composite,
            "normalized":       normalized,
            "direction":        direction,
            "conviction":       conviction,
            "sub_signals":      raw_signals,
            "weights_live":     dict(AladdinScorer.WEIGHTS),
            "ic_stats":         AladdinScorer.get_ic_stats(),
            # MI diagnostics (Step 10)
            "mi_composite":     float(mi_result.get("composite", composite)),
            "naive_composite":  float(mi_result.get("naive_composite", composite)),
            "mi_adjustment":    float(mi_result.get("mi_adjustment", 0.0)),
            "effective_weights":mi_result.get("effective_weights", {}),
            "weight_deltas":    mi_result.get("weight_deltas", {}),
            "redundancy_pairs": mi_result.get("redundancy_pairs", []),
            "mi_history_len":   int(mi_result.get("history_samples", 0)),
        }


def get_phd_math_status() -> Dict:
    """
    Expose advanced math/physics stack coverage for institutional diagnostics.
    """
    return {
        "stochastic_calculus": ["BlackScholesMerton", "MonteCarlo", "OrnsteinUhlenbeck"],
        "time_series_regime": ["HiddenMarkovRegime", "RegimeModel", "CUSUMBreakDetector", "KalmanFilter"],
        "optimization_risk": ["KellyCriterion", "SignalFusion", "MISignalFusion", "FactorAnalysis"],
        "volatility_derivatives": ["VolatilityAnalysis", "options_pop", "implied_vol"],
        "research_grade": "PhD-level quantitative stack enabled",
    }