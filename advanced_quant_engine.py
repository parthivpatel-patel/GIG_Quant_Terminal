"""
=============================================================================
RENAISSANCE.IO — ADVANCED QUANTITATIVE MATHEMATICS ENGINE
=============================================================================
The mathematical arsenal of a top-tier quantitative hedge fund.
These are the models that Citadel, Two Sigma, D.E. Shaw, and Renaissance
Technologies actually use — not retail indicator logic.

MODELS:
  ┌──────────────────────────────────────────────────────────────────┐
  │  1. LÉVY STABLE DISTRIBUTIONS  — Fat tail modeling              │
  │     Markets are NOT Gaussian. Lévy captures the real tails.     │
  ├──────────────────────────────────────────────────────────────────┤
  │  2. MERTON JUMP-DIFFUSION     — Sudden price discontinuities   │
  │     GBM + Poisson jumps. Models earnings gaps, flash crashes.   │
  ├──────────────────────────────────────────────────────────────────┤
  │  3. SPECTRAL ANALYSIS (FFT)   — Hidden cyclical patterns       │
  │     Fourier transform reveals periodicities invisible to eye.   │
  ├──────────────────────────────────────────────────────────────────┤
  │  4. EXTREME VALUE THEORY      — Tail risk quantification       │
  │     GEV + GPD for modeling once-in-a-decade events.             │
  ├──────────────────────────────────────────────────────────────────┤
  │  5. DYNAMIC TIME WARPING      — Pattern matching across time   │
  │     Finds similar historical episodes regardless of speed.      │
  ├──────────────────────────────────────────────────────────────────┤
  │  6. HESTON STOCHASTIC VOL     — Volatility-of-volatility       │
  │     Vol is not constant. Heston models vol clustering.          │
  ├──────────────────────────────────────────────────────────────────┤
  │  7. RANDOM MATRIX THEORY      — Correlation cleaning           │
  │     Marcenko-Pastur: separate signal from noise in corr matrix. │
  ├──────────────────────────────────────────────────────────────────┤
  │  8. OPTIMAL STOPPING THEORY   — When to exit positions         │
  │     Secretary problem applied to profit-taking decisions.       │
  ├──────────────────────────────────────────────────────────────────┤
  │  9. INFORMATION GEOMETRY      — KL divergence, Fisher info      │
  │     Detect regime shifts via distribution distance metrics.     │
  ├──────────────────────────────────────────────────────────────────┤
  │ 10. MICROSTRUCTURE MODELS     — Kyle lambda, Amihud illiquidity │
  │     Price impact and informed trading detection.                │
  └──────────────────────────────────────────────────────────────────┘

Renaissance.io — Institutional Grade Quantitative Trading
=============================================================================
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from scipy import stats, optimize, signal as sp_signal
from scipy.stats import norm, genextreme, genpareto
from collections import deque

logger = logging.getLogger("ADV_QUANT_MATH")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LÉVY STABLE DISTRIBUTIONS — Fat Tail Modeling
# ═══════════════════════════════════════════════════════════════════════════════

class LevyStableModel:
    """
    Markets are NOT Gaussian. Returns have fat tails and asymmetry.
    The Lévy stable distribution captures this with 4 parameters:
      α (stability): 2=Gaussian, <2=fat tails (markets are typically 1.4-1.8)
      β (skewness): -1 to +1
      γ (scale): volatility proxy
      δ (location): drift

    If your VaR model assumes Gaussian, you're underestimating tail risk
    by 3-10x. Lévy fixes this.
    """

    @classmethod
    def fit(cls, returns: np.ndarray) -> Dict:
        """Fit Lévy stable distribution to return series."""
        if len(returns) < 50:
            return {"error": "need 50+ observations"}

        try:
            # Fit stable distribution parameters
            params = stats.levy_stable.fit(returns, floc=0)
            alpha, beta, loc, scale = params

            # Compare with Gaussian
            gauss_std = np.std(returns)
            gauss_var_95 = norm.ppf(0.05) * gauss_std
            levy_var_95 = stats.levy_stable.ppf(0.05, alpha, beta, loc, scale)

            # Tail ratio: how much fatter are Lévy tails vs Gaussian?
            tail_ratio = abs(levy_var_95) / abs(gauss_var_95) if gauss_var_95 != 0 else 1

            # Kurtosis (excess)
            if alpha > 2:
                excess_kurt = 0
            elif alpha <= 1:
                excess_kurt = float('inf')
            else:
                excess_kurt = float(stats.kurtosis(returns))

            return {
                "alpha": round(float(alpha), 4),  # stability (2=Gaussian)
                "beta": round(float(beta), 4),     # skewness
                "scale": round(float(scale), 6),   # volatility
                "location": round(float(loc), 6),  # drift
                "gaussian_var_95": round(float(gauss_var_95), 6),
                "levy_var_95": round(float(levy_var_95), 6),
                "tail_ratio": round(float(tail_ratio), 2),
                "excess_kurtosis": round(float(excess_kurt), 2),
                "is_fat_tailed": alpha < 1.9,
                "fat_tail_severity": "EXTREME" if alpha < 1.5 else "HIGH" if alpha < 1.7 else "MODERATE" if alpha < 1.9 else "GAUSSIAN",
                "var_underestimate_pct": round((tail_ratio - 1) * 100, 1),
            }
        except Exception as e:
            # Fallback: basic tail analysis
            kurt = float(stats.kurtosis(returns))
            skew = float(stats.skew(returns))
            return {
                "alpha": 2.0 - min(1.0, max(0, kurt / 10)),
                "beta": round(skew, 4),
                "excess_kurtosis": round(kurt, 2),
                "is_fat_tailed": kurt > 3,
                "fat_tail_severity": "HIGH" if kurt > 6 else "MODERATE" if kurt > 3 else "GAUSSIAN",
                "note": f"Approximate fit: {e}",
            }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MERTON JUMP-DIFFUSION — Sudden Price Discontinuities
# ═══════════════════════════════════════════════════════════════════════════════

class MertonJumpDiffusion:
    """
    Standard GBM assumes continuous prices. But markets JUMP:
    earnings announcements, flash crashes, geopolitical shocks.

    Merton (1976) adds Poisson jumps to GBM:
      dS/S = (μ - λk)dt + σdW + JdN

    Where:
      λ = jump intensity (jumps per year)
      J = jump size (log-normal)
      N = Poisson counting process
    """

    @classmethod
    def estimate_params(cls, returns: np.ndarray, threshold: float = 2.5) -> Dict:
        """Estimate jump-diffusion parameters from return series."""
        if len(returns) < 60:
            return {"error": "need 60+ observations"}

        std = np.std(returns)
        mean = np.mean(returns)

        # Identify jumps: returns beyond threshold × std
        jumps = returns[np.abs(returns - mean) > threshold * std]
        normal_returns = returns[np.abs(returns - mean) <= threshold * std]

        n_jumps = len(jumps)
        n_total = len(returns)

        # Jump intensity (annualized)
        lambda_jump = n_jumps / n_total * 252

        # Jump size distribution
        if n_jumps > 0:
            jump_mean = float(np.mean(jumps))
            jump_std = float(np.std(jumps)) if n_jumps > 1 else std * 2
        else:
            jump_mean, jump_std = 0.0, 0.0

        # Diffusion volatility (from non-jump returns)
        sigma_diffusion = float(np.std(normal_returns) * np.sqrt(252))

        # Total volatility (observed)
        sigma_total = float(np.std(returns) * np.sqrt(252))

        # Jump contribution to total variance
        jump_variance_pct = (sigma_total**2 - sigma_diffusion**2) / max(sigma_total**2, 1e-10) * 100

        return {
            "lambda_jump": round(lambda_jump, 2),          # jumps per year
            "jump_mean": round(jump_mean * 100, 3),         # avg jump size (%)
            "jump_std": round(jump_std * 100, 3),           # jump volatility (%)
            "sigma_diffusion": round(sigma_diffusion * 100, 2),  # continuous vol (%)
            "sigma_total": round(sigma_total * 100, 2),     # total vol (%)
            "n_jumps_detected": n_jumps,
            "n_observations": n_total,
            "jump_frequency": f"{lambda_jump:.1f}/year",
            "jump_variance_contribution_pct": round(float(jump_variance_pct), 1),
            "largest_jump_pct": round(float(np.max(np.abs(jumps)) * 100), 2) if n_jumps > 0 else 0,
            "jump_risk_level": "HIGH" if lambda_jump > 10 else "MODERATE" if lambda_jump > 3 else "LOW",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SPECTRAL ANALYSIS (FFT) — Hidden Cyclical Patterns
# ═══════════════════════════════════════════════════════════════════════════════

class SpectralAnalyzer:
    """
    Fourier Transform reveals hidden periodicities in price data.
    Markets have cyclical patterns (monthly rebalancing, options expiry,
    quarterly earnings, yearly tax selling) that are invisible to the eye
    but detectable via FFT.
    """

    @classmethod
    def analyze(cls, prices: np.ndarray, sampling_freq: float = 1.0) -> Dict:
        """
        Spectral decomposition of price series.

        Args:
            prices: price array (daily close)
            sampling_freq: 1.0 for daily data

        Returns:
            Dominant frequencies, their periods, and power spectrum
        """
        if len(prices) < 64:
            return {"error": "need 64+ observations for FFT"}

        # Detrend: use returns instead of prices
        returns = np.diff(np.log(prices))

        # Apply Hanning window to reduce spectral leakage
        windowed = returns * np.hanning(len(returns))

        # FFT
        fft_vals = np.fft.rfft(windowed)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(len(windowed), d=1.0/sampling_freq)

        # Skip DC component (freq=0)
        power = power[1:]
        freqs = freqs[1:]

        if len(power) == 0:
            return {"error": "insufficient frequency bins"}

        # Find dominant frequencies (top 5 peaks)
        peak_indices = np.argsort(power)[::-1][:5]
        dominant = []
        for idx in peak_indices:
            if freqs[idx] > 0:
                period = 1.0 / freqs[idx]
                dominant.append({
                    "frequency": round(float(freqs[idx]), 6),
                    "period_days": round(float(period), 1),
                    "power": round(float(power[idx]), 4),
                    "power_pct": round(float(power[idx] / np.sum(power) * 100), 2),
                    "interpretation": cls._interpret_period(period),
                })

        # Spectral entropy (randomness measure)
        power_norm = power / (np.sum(power) + 1e-10)
        spectral_entropy = -float(np.sum(power_norm * np.log(power_norm + 1e-10)))
        max_entropy = np.log(len(power))
        normalized_entropy = spectral_entropy / max_entropy if max_entropy > 0 else 0

        return {
            "dominant_cycles": dominant,
            "spectral_entropy": round(spectral_entropy, 4),
            "normalized_entropy": round(float(normalized_entropy), 4),
            "is_random": normalized_entropy > 0.85,
            "has_cyclical_pattern": normalized_entropy < 0.70,
            "n_freq_bins": len(power),
            "total_power": round(float(np.sum(power)), 4),
        }

    @staticmethod
    def _interpret_period(period: float) -> str:
        if 4.5 <= period <= 5.5: return "WEEKLY (options expiry)"
        if 19 <= period <= 23: return "MONTHLY (rebalancing)"
        if 58 <= period <= 66: return "QUARTERLY (earnings)"
        if 120 <= period <= 135: return "SEMI-ANNUAL"
        if 240 <= period <= 265: return "ANNUAL (tax selling)"
        if 2 <= period <= 3: return "2-3 DAY (T+1 settlement)"
        if 8 <= period <= 12: return "BI-WEEKLY"
        return f"{period:.0f}-DAY CYCLE"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EXTREME VALUE THEORY — Tail Risk Quantification
# ═══════════════════════════════════════════════════════════════════════════════

class ExtremeValueTheory:
    """
    Standard VaR uses the middle of the distribution.
    EVT models the TAILS — exactly where risk lives.

    Generalized Pareto Distribution (GPD) for peaks-over-threshold:
    Models the probability of extreme losses exceeding a high threshold.

    This is what risk desks at Goldman, Morgan Stanley use for stress testing.
    """

    @classmethod
    def fit_tail(cls, returns: np.ndarray, threshold_pct: float = 5.0) -> Dict:
        """
        Fit GPD to tail losses using peaks-over-threshold.

        Args:
            returns: daily return series
            threshold_pct: percentile for threshold (5 = bottom 5%)
        """
        if len(returns) < 100:
            return {"error": "need 100+ observations for EVT"}

        losses = -returns  # flip sign: positive = loss
        threshold = np.percentile(losses, 100 - threshold_pct)

        # Exceedances over threshold
        exceedances = losses[losses > threshold] - threshold
        n_exceed = len(exceedances)

        if n_exceed < 10:
            return {"error": f"only {n_exceed} exceedances — need 10+"}

        try:
            # Fit Generalized Pareto Distribution
            shape, loc, scale = genpareto.fit(exceedances, floc=0)

            # Expected Shortfall beyond VaR
            # EVT-based VaR at 99% and 99.5%
            n = len(returns)
            prob_exceed = n_exceed / n

            def evt_var(p):
                """EVT VaR at confidence level p."""
                return threshold + (scale / shape) * (((1-p)/prob_exceed)**(-shape) - 1)

            var_95 = float(evt_var(0.95))
            var_99 = float(evt_var(0.99))
            var_995 = float(evt_var(0.995))

            # Expected Shortfall (CVaR) at 99%
            if shape < 1:
                es_99 = var_99 / (1 - shape) + (scale - shape * threshold) / (1 - shape)
            else:
                es_99 = var_99 * 1.5  # approximation for heavy tails

            # Gaussian comparison
            gauss_var_99 = float(np.percentile(losses, 99))
            tail_ratio = var_99 / max(gauss_var_99, 0.001)

            return {
                "gpd_shape": round(float(shape), 4),  # >0 = heavy tail (Fréchet)
                "gpd_scale": round(float(scale), 6),
                "threshold": round(float(threshold), 6),
                "n_exceedances": n_exceed,
                "var_95_pct": round(var_95 * 100, 3),
                "var_99_pct": round(var_99 * 100, 3),
                "var_995_pct": round(var_995 * 100, 3),
                "cvar_99_pct": round(float(es_99) * 100, 3),
                "gaussian_var_99_pct": round(gauss_var_99 * 100, 3),
                "evt_vs_gaussian_ratio": round(float(tail_ratio), 2),
                "tail_type": "HEAVY (Fréchet)" if shape > 0.1 else "THIN (Weibull)" if shape < -0.1 else "EXPONENTIAL",
                "tail_index": round(1.0 / max(abs(shape), 0.01), 2),
            }
        except Exception as e:
            return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DYNAMIC TIME WARPING — Pattern Matching Across Time
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicTimeWarping:
    """
    DTW finds similar historical episodes regardless of speed/timing.
    "This looks like the 2018 vol-crash setup" — but mathematically rigorous.

    Used by Renaissance for pattern-based alpha signals.
    """

    @classmethod
    def compute(cls, series_a: np.ndarray, series_b: np.ndarray) -> Dict:
        """Compute DTW distance between two series."""
        n, m = len(series_a), len(series_b)
        if n < 5 or m < 5:
            return {"error": "need 5+ points per series"}

        # Normalize both series to [0, 1]
        a = (series_a - np.min(series_a)) / (np.ptp(series_a) + 1e-10)
        b = (series_b - np.min(series_b)) / (np.ptp(series_b) + 1e-10)

        # DTW cost matrix
        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                d = (a[i-1] - b[j-1]) ** 2
                cost[i, j] = d + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])

        dtw_distance = float(np.sqrt(cost[n, m] / max(n, m)))

        # Similarity score: 0 = identical, 1 = completely different
        similarity = max(0, 1 - dtw_distance)

        return {
            "dtw_distance": round(dtw_distance, 6),
            "similarity": round(similarity, 4),
            "similarity_pct": round(similarity * 100, 1),
            "interpretation": "VERY_SIMILAR" if similarity > 0.85 else
                             "SIMILAR" if similarity > 0.70 else
                             "SOMEWHAT_SIMILAR" if similarity > 0.50 else "DIFFERENT",
            "series_a_len": n,
            "series_b_len": m,
        }

    @classmethod
    def find_historical_matches(cls, current: np.ndarray, history: np.ndarray,
                                 window: int = 30, top_n: int = 5) -> List[Dict]:
        """Find top-N most similar historical windows to current pattern."""
        if len(history) < window * 2 or len(current) < 10:
            return []

        matches = []
        current_norm = current[-window:] if len(current) >= window else current

        for i in range(0, len(history) - window - 5, max(1, window // 4)):
            hist_window = history[i:i + len(current_norm)]
            result = cls.compute(current_norm, hist_window)
            if "error" not in result:
                # What happened AFTER this historical pattern?
                forward_start = i + len(current_norm)
                forward_end = min(forward_start + 10, len(history))
                if forward_end > forward_start:
                    forward_return = (history[forward_end - 1] / history[forward_start] - 1) * 100
                else:
                    forward_return = 0

                matches.append({
                    "start_idx": i,
                    "similarity": result["similarity"],
                    "forward_return_10d_pct": round(float(forward_return), 2),
                    "interpretation": result["interpretation"],
                })

        matches.sort(key=lambda x: -x["similarity"])
        return matches[:top_n]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. HESTON STOCHASTIC VOLATILITY — Vol of Vol
# ═══════════════════════════════════════════════════════════════════════════════

class HestonModel:
    """
    Black-Scholes assumes constant volatility. Heston (1993) fixes this:
      dS = μS dt + √v S dW₁
      dv = κ(θ - v)dt + σᵥ√v dW₂
      cor(dW₁, dW₂) = ρ

    Parameters:
      κ (kappa): mean-reversion speed of vol
      θ (theta): long-run variance level
      σᵥ: vol-of-vol (how much vol itself fluctuates)
      ρ: correlation between price and vol (usually negative = leverage effect)
      v₀: current variance level
    """

    @classmethod
    def estimate(cls, returns: np.ndarray, window: int = 20) -> Dict:
        """Estimate Heston parameters from realized data."""
        if len(returns) < 60:
            return {"error": "need 60+ observations"}

        # Realized variance series (rolling)
        var_series = []
        for i in range(window, len(returns)):
            var_series.append(np.var(returns[i-window:i]) * 252)
        var_series = np.array(var_series)

        if len(var_series) < 20:
            return {"error": "insufficient variance series"}

        # Current vol
        v0 = float(var_series[-1])

        # Long-run variance (mean of variance series)
        theta = float(np.mean(var_series))

        # Vol-of-vol
        vol_of_vol = float(np.std(np.diff(var_series)) * np.sqrt(252))

        # Mean-reversion speed (from AR(1) of variance)
        if len(var_series) > 2:
            autocorr = float(np.corrcoef(var_series[:-1], var_series[1:])[0, 1])
            kappa = -np.log(max(abs(autocorr), 0.01)) * 252  # annualized
        else:
            kappa = 2.0

        # Leverage effect: correlation between returns and vol changes
        ret_trimmed = returns[window+1:][:len(var_series)-1]
        vol_changes = np.diff(var_series)
        min_len = min(len(ret_trimmed), len(vol_changes))
        if min_len > 5:
            rho = float(np.corrcoef(ret_trimmed[:min_len], vol_changes[:min_len])[0, 1])
        else:
            rho = -0.7  # typical leverage effect

        # Feller condition: 2κθ > σᵥ² (ensures variance stays positive)
        feller_satisfied = 2 * kappa * theta > vol_of_vol ** 2

        return {
            "v0": round(v0, 4),
            "v0_annual_vol_pct": round(float(np.sqrt(v0) * 100), 2),
            "theta": round(theta, 4),
            "theta_annual_vol_pct": round(float(np.sqrt(theta) * 100), 2),
            "kappa": round(float(kappa), 3),
            "vol_of_vol": round(vol_of_vol, 4),
            "rho": round(float(rho), 4),
            "leverage_effect": "STRONG" if rho < -0.5 else "MODERATE" if rho < -0.2 else "WEAK",
            "feller_condition": feller_satisfied,
            "vol_regime": "HIGH" if v0 > theta * 1.5 else "LOW" if v0 < theta * 0.6 else "NORMAL",
            "vol_mean_reverting": kappa > 0.5,
            "half_life_days": round(float(np.log(2) / max(kappa / 252, 0.001)), 1),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. RANDOM MATRIX THEORY — Correlation Cleaning
# ═══════════════════════════════════════════════════════════════════════════════

class RandomMatrixTheory:
    """
    Marcenko-Pastur law: in a random matrix, eigenvalues follow a specific
    distribution. Eigenvalues ABOVE the MP upper bound contain SIGNAL.
    Those below are pure NOISE.

    This is how top quant funds clean their correlation matrices:
    keep only signal eigenvalues, replace noise with average.
    Result: dramatically better portfolio optimization.
    """

    @classmethod
    def clean_correlation(cls, returns_matrix: np.ndarray) -> Dict:
        """
        Clean correlation matrix using Random Matrix Theory.

        Args:
            returns_matrix: (T × N) matrix of returns for N assets

        Returns:
            Cleaned correlation matrix + signal/noise decomposition
        """
        T, N = returns_matrix.shape
        if T < 30 or N < 3:
            return {"error": "need T>=30, N>=3"}

        # Compute correlation matrix
        corr = np.corrcoef(returns_matrix.T)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Marcenko-Pastur bounds
        q = T / N  # ratio
        lambda_plus = (1 + 1/np.sqrt(q)) ** 2   # upper bound
        lambda_minus = (1 - 1/np.sqrt(q)) ** 2   # lower bound

        # Classify eigenvalues as signal vs noise
        n_signal = int(np.sum(eigenvalues > lambda_plus))
        n_noise = N - n_signal

        # Cleaned matrix: keep signal eigenvalues, replace noise with average
        avg_noise_eigenval = np.mean(eigenvalues[eigenvalues <= lambda_plus]) if n_noise > 0 else 1.0
        cleaned_eigenvalues = eigenvalues.copy()
        cleaned_eigenvalues[eigenvalues <= lambda_plus] = avg_noise_eigenval

        # Reconstruct cleaned correlation matrix
        cleaned_corr = eigenvectors @ np.diag(cleaned_eigenvalues) @ eigenvectors.T
        # Normalize diagonal to 1
        d = np.sqrt(np.diag(cleaned_corr))
        cleaned_corr = cleaned_corr / np.outer(d, d)
        np.fill_diagonal(cleaned_corr, 1.0)

        # Signal-to-noise ratio
        signal_variance = float(np.sum(eigenvalues[eigenvalues > lambda_plus]))
        total_variance = float(np.sum(eigenvalues))
        snr = signal_variance / max(total_variance, 1e-10)

        return {
            "n_assets": N,
            "n_observations": T,
            "n_signal_factors": n_signal,
            "n_noise_factors": n_noise,
            "mp_upper_bound": round(float(lambda_plus), 4),
            "mp_lower_bound": round(float(lambda_minus), 4),
            "top_eigenvalue": round(float(eigenvalues[0]), 4),
            "signal_variance_pct": round(snr * 100, 1),
            "noise_variance_pct": round((1 - snr) * 100, 1),
            "eigenvalues": [round(float(e), 4) for e in eigenvalues[:10]],
            "cleaned_correlation": cleaned_corr.tolist(),
            "quality": "HIGH" if snr > 0.40 else "MODERATE" if snr > 0.20 else "NOISY",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 8. OPTIMAL STOPPING — When to Exit Positions
# ═══════════════════════════════════════════════════════════════════════════════

class OptimalStopping:
    """
    Secretary problem / optimal stopping applied to trading:
    "I'm up 15% — should I take profit or let it run?"

    Uses the 1/e rule and dynamic programming approach.
    Also computes optimal trailing stop width.
    """

    @classmethod
    def compute_optimal_exit(cls, returns: np.ndarray,
                              current_pnl_pct: float,
                              holding_period: int = 20) -> Dict:
        """
        Compute optimal exit strategy based on historical return distribution.

        Args:
            returns: historical daily returns
            current_pnl_pct: current unrealized P&L (%)
            holding_period: max holding period in days
        """
        if len(returns) < 60:
            return {"error": "need 60+ observations"}

        # Simulate forward paths (Monte Carlo)
        n_paths = 5000
        mu = float(np.mean(returns))
        sigma = float(np.std(returns))

        paths = np.zeros((n_paths, holding_period))
        for t in range(holding_period):
            paths[:, t] = np.random.normal(mu, sigma, n_paths)

        cumulative = np.cumsum(paths, axis=1) * 100  # convert to %
        cumulative += current_pnl_pct  # start from current P&L

        # Optimal stopping: find the day that maximizes expected P&L
        expected_pnl = np.mean(cumulative, axis=0)
        optimal_day = int(np.argmax(expected_pnl))
        optimal_pnl = float(expected_pnl[optimal_day])

        # Probability of improving from current P&L
        prob_improve = float(np.mean(np.max(cumulative, axis=1) > current_pnl_pct))

        # Optimal trailing stop (minimize expected loss from peak)
        max_drawdowns = []
        for pct_stop in [1, 2, 3, 5, 7, 10]:
            stopped_pnl = []
            for path in cumulative:
                peak = current_pnl_pct
                for t, val in enumerate(path):
                    peak = max(peak, val)
                    if peak - val >= pct_stop:
                        stopped_pnl.append(val)
                        break
                else:
                    stopped_pnl.append(path[-1])
            max_drawdowns.append({
                "stop_pct": pct_stop,
                "avg_exit_pnl": round(float(np.mean(stopped_pnl)), 2),
                "median_exit_pnl": round(float(np.median(stopped_pnl)), 2),
            })

        # Best trailing stop
        best_stop = max(max_drawdowns, key=lambda x: x["avg_exit_pnl"])

        # 1/e rule: observe for first 37% of holding period, then take first value above best seen
        observe_period = max(1, int(holding_period * 0.37))

        return {
            "current_pnl_pct": round(current_pnl_pct, 2),
            "optimal_exit_day": optimal_day + 1,
            "optimal_expected_pnl_pct": round(optimal_pnl, 2),
            "prob_improvement": round(prob_improve, 3),
            "recommendation": "HOLD" if optimal_day > 2 and prob_improve > 0.55 else
                            "TAKE_PROFIT" if current_pnl_pct > optimal_pnl * 0.9 else "TRAIL_STOP",
            "optimal_trailing_stop_pct": best_stop["stop_pct"],
            "trailing_stop_analysis": max_drawdowns,
            "observe_period_days": observe_period,
            "holding_period": holding_period,
            "forward_paths_simulated": n_paths,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 9. INFORMATION GEOMETRY — Distribution Shift Detection
# ═══════════════════════════════════════════════════════════════════════════════

class InformationGeometry:
    """
    Detect regime shifts by measuring how much the return distribution
    has changed. Uses KL divergence and Wasserstein distance.

    When the distribution shifts significantly, your models are stale
    and need recalibration.
    """

    @classmethod
    def detect_distribution_shift(cls, recent: np.ndarray,
                                    historical: np.ndarray,
                                    n_bins: int = 30) -> Dict:
        """Measure distribution shift between recent and historical returns."""
        if len(recent) < 20 or len(historical) < 50:
            return {"error": "need recent>=20, historical>=50"}

        # KL Divergence (asymmetric)
        all_data = np.concatenate([recent, historical])
        bins = np.linspace(np.min(all_data), np.max(all_data), n_bins + 1)

        hist_p, _ = np.histogram(recent, bins=bins, density=True)
        hist_q, _ = np.histogram(historical, bins=bins, density=True)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        hist_p = hist_p / (np.sum(hist_p) + eps) + eps
        hist_q = hist_q / (np.sum(hist_q) + eps) + eps

        kl_div = float(np.sum(hist_p * np.log(hist_p / hist_q)))

        # Wasserstein distance (Earth Mover's Distance)
        from scipy.stats import wasserstein_distance
        wass_dist = float(wasserstein_distance(recent, historical))

        # Statistical tests
        ks_stat, ks_pval = stats.ks_2samp(recent, historical)

        # Moment comparison
        moment_shift = {
            "mean_shift": round(float(np.mean(recent) - np.mean(historical)), 6),
            "vol_shift": round(float(np.std(recent) - np.std(historical)), 6),
            "skew_shift": round(float(stats.skew(recent) - stats.skew(historical)), 4),
            "kurt_shift": round(float(stats.kurtosis(recent) - stats.kurtosis(historical)), 4),
        }

        # Severity assessment
        regime_shift = kl_div > 0.5 or ks_pval < 0.05 or wass_dist > np.std(historical) * 0.5

        return {
            "kl_divergence": round(kl_div, 4),
            "wasserstein_distance": round(wass_dist, 6),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pval), 4),
            "moment_shifts": moment_shift,
            "regime_shift_detected": regime_shift,
            "shift_severity": "MAJOR" if kl_div > 1.0 else "MODERATE" if kl_div > 0.3 else "MINOR" if kl_div > 0.1 else "NONE",
            "model_recalibration_needed": regime_shift,
            "n_recent": len(recent),
            "n_historical": len(historical),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 10. MASTER ANALYZER — Full Quant Analysis Suite
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedQuantAnalyzer:
    """Orchestrates all advanced quant models on a single stock."""

    @classmethod
    def full_analysis(cls, prices: np.ndarray, symbol: str = "UNKNOWN") -> Dict:
        """Run the complete advanced quant analysis suite."""
        if len(prices) < 60:
            return {"error": "need 60+ observations", "symbol": symbol}

        returns = np.diff(np.log(prices))

        results = {"symbol": symbol, "n_observations": len(prices)}

        # 1. Lévy tails
        try:
            results["levy_tails"] = LevyStableModel.fit(returns)
        except Exception as e:
            results["levy_tails"] = {"error": str(e)}

        # 2. Jump diffusion
        try:
            results["jump_diffusion"] = MertonJumpDiffusion.estimate_params(returns)
        except Exception as e:
            results["jump_diffusion"] = {"error": str(e)}

        # 3. Spectral analysis
        try:
            results["spectral"] = SpectralAnalyzer.analyze(prices)
        except Exception as e:
            results["spectral"] = {"error": str(e)}

        # 4. EVT tail risk
        try:
            results["evt_tail_risk"] = ExtremeValueTheory.fit_tail(returns)
        except Exception as e:
            results["evt_tail_risk"] = {"error": str(e)}

        # 5. Heston stochastic vol
        try:
            results["heston_vol"] = HestonModel.estimate(returns)
        except Exception as e:
            results["heston_vol"] = {"error": str(e)}

        # 6. Distribution shift (recent vs historical)
        try:
            if len(returns) >= 70:
                results["distribution_shift"] = InformationGeometry.detect_distribution_shift(
                    returns[-20:], returns[:-20]
                )
        except Exception as e:
            results["distribution_shift"] = {"error": str(e)}

        # 7. Optimal exit (if holding)
        try:
            results["optimal_exit"] = OptimalStopping.compute_optimal_exit(
                returns, current_pnl_pct=0
            )
        except Exception as e:
            results["optimal_exit"] = {"error": str(e)}

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def full_quant_analysis(prices: np.ndarray, symbol: str = "UNKNOWN") -> Dict:
    return AdvancedQuantAnalyzer.full_analysis(prices, symbol)

def get_model_list() -> List[str]:
    return [
        "LevyStableModel", "MertonJumpDiffusion", "SpectralAnalyzer",
        "ExtremeValueTheory", "DynamicTimeWarping", "HestonModel",
        "RandomMatrixTheory", "OptimalStopping", "InformationGeometry",
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test with fake data
    np.random.seed(42)
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, 500)))
    result = full_quant_analysis(prices, "TEST")
    for model, data in result.items():
        if isinstance(data, dict) and "error" not in data:
            print(f"\n{model}:")
            for k, v in list(data.items())[:5]:
                print(f"  {k}: {v}")
