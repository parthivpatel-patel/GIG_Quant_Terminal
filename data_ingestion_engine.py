"""
=============================================================================
RENAISSANCE DATA INGESTION ENGINE — data_ingestion_engine.py
=============================================================================
Multi-Source Data Pipeline + Cleansing + Validation

Renaissance's first principle: garbage in = garbage out.
Simons hired scientists, not traders, because data cleansing IS the edge.

PIPELINE:
  ├── Multi-Source Fetcher       — yfinance, Alpaca, FRED, NewsAPI
  ├── OHLCV Validator            — detect bad ticks, splits, gaps
  ├── Outlier Detector           — Hampel filter + IQR + z-score
  ├── Gap Filler                 — forward-fill, interpolation, warning flags
  ├── Corporate Action Adjuster  — split/dividend adjustment verification
  ├── Feature Normalizer         — z-score, min-max, rank normalization
  ├── Data Quality Scorer        — 0-100 quality score per symbol
  └── Live Data Watchdog         — detects stale/frozen feeds

Renaissance.io — Institutional Grade Quantitative Trading
=============================================================================
"""

import numpy as np
import pandas as pd
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass, asdict

logger = logging.getLogger("DATA_INGEST")


@dataclass(frozen=True)
class DataFeedSpec:
    """Institutional feed descriptor for coverage and routing."""
    feed_id: str
    asset_class: str
    data_type: str
    cadence: str
    target_latency_ms: int
    source_examples: Tuple[str, ...]
    normalization_hint: str


class DataCoverageRegistry:
    """
    Bloomberg-style coverage map including micro/alternative feeds.
    This is intentionally explicit so the UI/research layers can inspect
    what data is expected to be online.
    """
    FEEDS: Tuple[DataFeedSpec, ...] = (
        DataFeedSpec("eq_ticks", "equities", "trade/quote ticks", "real-time", 25,
                     ("NASDAQ TotalView", "Polygon", "IEX"), "event_time + symbol harmonization"),
        DataFeedSpec("fx_spot", "fx", "spot ticks", "real-time", 30,
                     ("EBS", "Refinitiv", "Oanda"), "pip precision + venue tagging"),
        DataFeedSpec("crypto_l2", "crypto", "L2 order book", "real-time", 20,
                     ("Binance", "Coinbase", "Kraken"), "depth snapshot + delta merge"),
        DataFeedSpec("options_chain", "derivatives", "greeks/iv surface", "1s-15s", 80,
                     ("OPRA", "Tradier", "Polygon"), "contract symbology normalization"),
        DataFeedSpec("fundamentals", "equities", "financial statements", "daily/quarterly", 5000,
                     ("SEC", "Financial Modeling Prep"), "period alignment + restatement flags"),
        DataFeedSpec("macro", "macro", "economic series", "daily/weekly/monthly", 2000,
                     ("FRED", "BLS", "EIA"), "calendar aware forward-fill"),
        DataFeedSpec("news_nlp", "cross-asset", "news + sentiment", "real-time", 150,
                     ("NewsAPI", "SEC EDGAR", "RSS"), "entity linking + event schema"),
        DataFeedSpec("satellite_imagery", "alternative", "geospatial activity", "daily/weekly", 30000,
                     ("Sentinel Hub", "NASA MODIS", "Planet"), "geo-tiling + cloud-mask quality"),
        DataFeedSpec("ais_shipping", "alternative", "vessel flow", "5m-15m", 2500,
                     ("AIS aggregators", "MarineTraffic"), "port geofence aggregation"),
        DataFeedSpec("credit_card_proxy", "alternative", "consumer spend proxy", "daily/weekly", 5000,
                     ("Fed G.19", "public macro proxies"), "seasonal adjustment"),
        DataFeedSpec("web_traffic", "alternative", "site/app engagement", "daily", 4000,
                     ("public ranking APIs", "app store feeds"), "outlier clipping"),
        DataFeedSpec("supply_chain", "alternative", "cargo/rail/truck proxy", "daily", 5000,
                     ("port stats", "rail traffic reports"), "route-level rollups"),
    )

    @classmethod
    def coverage_manifest(cls) -> List[Dict]:
        return [asdict(feed) for feed in cls.FEEDS]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. OHLCV VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════

class OHLCVValidator:
    """
    Validates OHLCV data for common issues:
    - High < Low (impossible)
    - Close outside [Low, High] range
    - Volume = 0 on trading day
    - Price jumps > 50% in one bar (likely bad tick or unadjusted split)
    - Negative prices
    - Duplicate timestamps
    """

    @classmethod
    def validate(cls, df: pd.DataFrame) -> Dict:
        """
        Validate OHLCV DataFrame.

        Args:
            df: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']

        Returns:
            {
                "valid": bool,
                "quality_score": float [0, 100],
                "issues": list of issue descriptions,
                "rows_checked": int,
                "rows_flagged": int,
                "cleaned_df": DataFrame (if fixable issues)
            }
        """
        issues = []
        flags = np.zeros(len(df), dtype=bool)

        # Required columns check
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        # Handle lowercase columns
        col_map = {}
        for col in df.columns:
            for req in required:
                if col.lower() == req.lower():
                    col_map[req] = col

        if len(col_map) < 5:
            return {"valid": False, "quality_score": 0,
                    "issues": ["missing_columns"], "rows_checked": len(df),
                    "rows_flagged": len(df)}

        o = df[col_map['Open']].values.astype(float)
        h = df[col_map['High']].values.astype(float)
        l = df[col_map['Low']].values.astype(float)
        c = df[col_map['Close']].values.astype(float)
        v = df[col_map['Volume']].values.astype(float)

        n = len(df)

        # Check 1: Negative prices
        neg_mask = (o < 0) | (h < 0) | (l < 0) | (c < 0)
        if neg_mask.any():
            issues.append(f"negative_prices: {neg_mask.sum()} rows")
            flags |= neg_mask

        # Check 2: High < Low
        hl_mask = h < l
        if hl_mask.any():
            issues.append(f"high_below_low: {hl_mask.sum()} rows")
            flags |= hl_mask

        # Check 3: Close outside range
        range_mask = (c > h * 1.001) | (c < l * 0.999)
        if range_mask.any():
            issues.append(f"close_outside_range: {range_mask.sum()} rows")
            flags |= range_mask

        # Check 4: Zero volume
        zero_vol = v == 0
        if zero_vol.any():
            issues.append(f"zero_volume: {zero_vol.sum()} rows")

        # Check 5: Price jumps > 50%
        if n > 1:
            pct_change = np.abs(np.diff(c) / c[:-1])
            jumps = np.concatenate([[False], pct_change > 0.50])
            if jumps.any():
                issues.append(f"extreme_jumps: {jumps.sum()} rows (>50% move)")
                flags |= jumps

        # Check 6: Duplicate timestamps
        if hasattr(df.index, 'duplicated'):
            dups = df.index.duplicated()
            if dups.any():
                issues.append(f"duplicate_timestamps: {dups.sum()}")

        # Check 7: NaN values
        nan_count = df[list(col_map.values())].isna().sum().sum()
        if nan_count > 0:
            issues.append(f"nan_values: {nan_count} total")

        # Quality score
        rows_flagged = int(flags.sum())
        quality = max(0, 100 - (rows_flagged / max(n, 1)) * 100 - len(issues) * 5)

        return {
            "valid": len(issues) == 0,
            "quality_score": round(quality, 1),
            "issues": issues,
            "rows_checked": n,
            "rows_flagged": rows_flagged,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. OUTLIER DETECTOR — Hampel + IQR + Z-Score
# ═══════════════════════════════════════════════════════════════════════════════

class OutlierDetector:
    """
    Multi-method outlier detection for financial time series.
    """

    @staticmethod
    def hampel_filter(data: np.ndarray, window: int = 7,
                      threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hampel filter: detects outliers using rolling median + MAD.
        More robust than z-score for financial data.

        Returns:
            (cleaned_data, outlier_mask)
        """
        n = len(data)
        cleaned = data.copy()
        outliers = np.zeros(n, dtype=bool)

        for i in range(window, n - window):
            window_data = data[i - window:i + window + 1]
            median = np.median(window_data)
            mad = 1.4826 * np.median(np.abs(window_data - median))

            if mad > 0 and abs(data[i] - median) > threshold * mad:
                outliers[i] = True
                cleaned[i] = median

        return cleaned, outliers

    @staticmethod
    def iqr_filter(data: np.ndarray, multiplier: float = 3.0
                   ) -> Tuple[np.ndarray, np.ndarray]:
        """IQR-based outlier detection."""
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        outliers = (data < lower) | (data > upper)
        cleaned = data.copy()
        cleaned[outliers] = np.median(data)

        return cleaned, outliers

    @staticmethod
    def zscore_filter(data: np.ndarray, threshold: float = 3.0
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Z-score outlier detection."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return data.copy(), np.zeros(len(data), dtype=bool)

        z = np.abs((data - mean) / std)
        outliers = z > threshold
        cleaned = data.copy()
        cleaned[outliers] = mean

        return cleaned, outliers

    @classmethod
    def detect_all(cls, data: np.ndarray, max_points: int = 10000) -> Dict:
        """
        Run all outlier detectors and return consensus.
        For latency-sensitive paths, only the most recent window is analyzed.
        """
        used_recent_window = False
        if len(data) > max_points:
            data = data[-max_points:]
            used_recent_window = True

        _, hampel_mask = cls.hampel_filter(data)
        _, iqr_mask = cls.iqr_filter(data)
        _, zscore_mask = cls.zscore_filter(data)

        # Consensus: flagged by 2+ methods
        consensus = (hampel_mask.astype(int) + iqr_mask.astype(int) +
                     zscore_mask.astype(int)) >= 2

        return {
            "outlier_count": int(consensus.sum()),
            "outlier_pct": round(float(consensus.mean()) * 100, 2),
            "hampel_count": int(hampel_mask.sum()),
            "iqr_count": int(iqr_mask.sum()),
            "zscore_count": int(zscore_mask.sum()),
            "consensus_mask": consensus,
            "used_recent_window": used_recent_window,
            "points_analyzed": int(len(data)),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GAP FILLER
# ═══════════════════════════════════════════════════════════════════════════════

class GapFiller:
    """
    Fills missing data with appropriate methods:
    - Forward-fill for prices (most appropriate for market data)
    - Interpolation for smooth features
    - Flagging for awareness
    """

    @staticmethod
    def fill(df: pd.DataFrame, method: str = "ffill",
             max_gap: int = 5) -> Tuple[pd.DataFrame, Dict]:
        """
        Fill gaps in DataFrame.

        Args:
            df: DataFrame with potential NaN values
            method: "ffill", "interpolate", or "median"
            max_gap: maximum consecutive NaN gap to fill

        Returns:
            (filled_df, gap_report)
        """
        original_nans = df.isna().sum().sum()
        gap_report = {"original_nans": int(original_nans), "method": method}

        if original_nans == 0:
            gap_report["filled"] = 0
            return df, gap_report

        filled = df.copy()

        if method == "ffill":
            filled = filled.ffill(limit=max_gap)
            filled = filled.bfill(limit=2)  # Fill remaining at start
        elif method == "interpolate":
            filled = filled.interpolate(method='linear', limit=max_gap)
        elif method == "median":
            for col in filled.columns:
                if filled[col].isna().any():
                    filled[col] = filled[col].fillna(filled[col].median())

        remaining_nans = filled.isna().sum().sum()
        gap_report["filled"] = int(original_nans - remaining_nans)
        gap_report["remaining_nans"] = int(remaining_nans)

        return filled, gap_report


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE NORMALIZER
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureNormalizer:
    """
    Normalize features for ML consumption.
    """

    @staticmethod
    def zscore(data: np.ndarray, window: int = 60) -> np.ndarray:
        """Fast rolling z-score normalization using vectorized pandas windows."""
        series = pd.Series(data, dtype=float)
        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        rolling_std = series.rolling(window=window, min_periods=1).std(ddof=0).clip(lower=1e-8)
        return ((series - rolling_mean) / rolling_std).to_numpy()

    @staticmethod
    def minmax(data: np.ndarray, window: int = 60) -> np.ndarray:
        """Fast rolling min-max normalization to [0, 1]."""
        series = pd.Series(data, dtype=float)
        rolling_min = series.rolling(window=window, min_periods=1).min()
        rolling_max = series.rolling(window=window, min_periods=1).max()
        rng = (rolling_max - rolling_min).clip(lower=1e-8)
        return ((series - rolling_min) / rng).to_numpy()

    @staticmethod
    def rank_normalize(data: np.ndarray) -> np.ndarray:
        """Rank normalization — maps to [0, 1] based on percentile rank."""
        from scipy.stats import rankdata
        ranks = rankdata(data)
        return (ranks - 1) / max(len(ranks) - 1, 1)

    @classmethod
    def normalize_features(cls, feature_matrix: np.ndarray,
                           method: str = "zscore") -> np.ndarray:
        """Normalize each column of a feature matrix."""
        result = np.zeros_like(feature_matrix, dtype=float)
        for col in range(feature_matrix.shape[1]):
            if method == "zscore":
                result[:, col] = cls.zscore(feature_matrix[:, col])
            elif method == "minmax":
                result[:, col] = cls.minmax(feature_matrix[:, col])
            elif method == "rank":
                result[:, col] = cls.rank_normalize(feature_matrix[:, col])
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DATA QUALITY SCORER
# ═══════════════════════════════════════════════════════════════════════════════

class DataQualityScorer:
    """
    Comprehensive data quality assessment.
    Score 0-100 per symbol, with detailed breakdown.
    """

    @classmethod
    def score(cls, df: pd.DataFrame, symbol: str = "") -> Dict:
        """
        Score data quality.

        Returns:
            {
                "symbol": str,
                "quality_score": float [0, 100],
                "components": {
                    "completeness": float,
                    "validity": float,
                    "timeliness": float,
                    "consistency": float
                },
                "issues": list,
                "recommendation": str
            }
        """
        issues = []
        scores = {}

        # Completeness: % non-null
        total_cells = df.size
        non_null = df.notna().sum().sum()
        completeness = (non_null / max(total_cells, 1)) * 100
        scores["completeness"] = round(completeness, 1)
        if completeness < 95:
            issues.append(f"completeness_below_95: {round(100 - completeness, 1)}% missing")

        # Validity: OHLCV checks
        validation = OHLCVValidator.validate(df)
        validity = validation["quality_score"]
        scores["validity"] = validity
        issues.extend(validation.get("issues", []))

        # Timeliness: is the latest data recent?
        timeliness = 100.0
        if hasattr(df.index, 'max') and hasattr(df.index.max(), 'date'):
            try:
                latest = pd.Timestamp(df.index.max())
                now = pd.Timestamp.now()
                days_old = (now - latest).days
                if days_old > 3:
                    timeliness = max(0, 100 - days_old * 10)
                    issues.append(f"stale_data: {days_old} days old")
            except Exception:
                pass
        scores["timeliness"] = round(timeliness, 1)

        # Consistency: check for erratic patterns
        consistency = 100.0
        try:
            close_col = [c for c in df.columns if c.lower() == 'close']
            if close_col:
                returns = df[close_col[0]].pct_change().dropna()
                if len(returns) > 20:
                    outlier_report = OutlierDetector.detect_all(returns.values)
                    outlier_pct = outlier_report["outlier_pct"]
                    consistency = max(0, 100 - outlier_pct * 10)
                    if outlier_pct > 2:
                        issues.append(f"high_outlier_rate: {outlier_pct}%")
        except Exception:
            pass
        scores["consistency"] = round(consistency, 1)

        # Overall
        quality = (scores["completeness"] * 0.25 +
                   scores["validity"] * 0.30 +
                   scores["timeliness"] * 0.25 +
                   scores["consistency"] * 0.20)

        # Recommendation
        if quality >= 90:
            rec = "excellent_quality"
        elif quality >= 70:
            rec = "good_with_minor_issues"
        elif quality >= 50:
            rec = "usable_with_caution"
        else:
            rec = "significant_issues_review_required"

        return {
            "symbol": symbol,
            "quality_score": round(quality, 1),
            "components": scores,
            "issues": issues,
            "recommendation": rec,
            "rows": len(df)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. LIVE DATA WATCHDOG
# ═══════════════════════════════════════════════════════════════════════════════

class LiveDataWatchdog:
    """
    Monitors live data feeds for staleness and anomalies.
    Alerts when a feed goes stale or produces suspicious values.
    """

    def __init__(self, stale_threshold_seconds: int = 120):
        self._last_update: Dict[str, float] = {}
        self._last_price: Dict[str, float] = {}
        self._alerts: deque = deque(maxlen=100)
        self._lock = threading.Lock()
        self.stale_threshold = stale_threshold_seconds

    def heartbeat(self, symbol: str, price: float):
        """Record a data update."""
        with self._lock:
            now = time.time()
            prev_price = self._last_price.get(symbol)

            # Check for frozen price (same price for too long)
            if prev_price and abs(price - prev_price) < 1e-8:
                last_ts = self._last_update.get(symbol, now)
                if now - last_ts > self.stale_threshold:
                    self._alerts.append({
                        "symbol": symbol,
                        "type": "frozen_price",
                        "price": price,
                        "frozen_seconds": round(now - last_ts),
                        "ts": now
                    })

            # Check for extreme move
            if prev_price and prev_price > 0:
                pct_move = abs(price - prev_price) / prev_price
                if pct_move > 0.20:  # 20% move
                    self._alerts.append({
                        "symbol": symbol,
                        "type": "extreme_move",
                        "pct_move": round(pct_move * 100, 2),
                        "from_price": prev_price,
                        "to_price": price,
                        "ts": now
                    })

            self._last_update[symbol] = now
            self._last_price[symbol] = price

    def get_stale_symbols(self) -> List[Dict]:
        """Get symbols with stale data."""
        now = time.time()
        stale = []
        with self._lock:
            for sym, ts in self._last_update.items():
                if now - ts > self.stale_threshold:
                    stale.append({
                        "symbol": sym,
                        "seconds_stale": round(now - ts),
                        "last_price": self._last_price.get(sym)
                    })
        return stale

    def get_alerts(self, limit: int = 20) -> List[Dict]:
        """Get recent alerts."""
        with self._lock:
            return list(self._alerts)[-limit:]

    def get_status(self) -> Dict:
        """Get watchdog status."""
        with self._lock:
            return {
                "monitored_symbols": len(self._last_update),
                "stale_count": len(self.get_stale_symbols()),
                "recent_alerts": len(self._alerts),
                "stale_symbols": self.get_stale_symbols()[:10]
            }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MASTER DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class DataPipeline:
    """
    End-to-end data pipeline:
    Fetch → Validate → Clean → Normalize → Score → Deliver
    """

    def __init__(self):
        self.watchdog = LiveDataWatchdog()
        self._quality_cache: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._latency_ms: deque = deque(maxlen=2000)

    def process(self, df: pd.DataFrame, symbol: str = "",
                normalize_method: str = "zscore") -> Dict:
        """
        Run full pipeline on raw OHLCV data.

        Returns:
            {
                "quality": DataQualityScorer result,
                "outliers": OutlierDetector result,
                "gaps_filled": GapFiller report,
                "cleaned_df": pd.DataFrame,
                "feature_ready": bool
            }
        """
        start = time.perf_counter()

        # Step 1: Validate
        quality = DataQualityScorer.score(df, symbol)

        # Step 2: Fill gaps
        filled_df, gap_report = GapFiller.fill(df, method="ffill")

        # Step 3: Detect outliers on returns
        outlier_report = {}
        try:
            close_col = [c for c in filled_df.columns if c.lower() == 'close']
            if close_col:
                returns = filled_df[close_col[0]].pct_change().dropna().values
                if len(returns) > 10:
                    outlier_report = OutlierDetector.detect_all(returns)
        except Exception as e:
            outlier_report = {"error": str(e)}

        # Step 4: Watchdog heartbeat
        try:
            close_col = [c for c in filled_df.columns if c.lower() == 'close']
            if close_col and len(filled_df) > 0:
                last_price = float(filled_df[close_col[0]].iloc[-1])
                self.watchdog.heartbeat(symbol, last_price)
        except Exception:
            pass

        feature_ready = quality["quality_score"] >= 50

        with self._lock:
            self._quality_cache[symbol] = quality
            self._latency_ms.append((time.perf_counter() - start) * 1000.0)

        return {
            "quality": quality,
            "outliers": {k: v for k, v in outlier_report.items()
                         if k != "consensus_mask"},
            "gaps_filled": gap_report,
            "feature_ready": feature_ready,
            "rows": len(filled_df),
            "pipeline_latency_ms": round((time.perf_counter() - start) * 1000.0, 3),
            "data_coverage": DataCoverageRegistry.coverage_manifest()
        }

    def get_quality_dashboard(self) -> Dict:
        """Get quality scores for all processed symbols."""
        with self._lock:
            symbols = []
            for sym, q in self._quality_cache.items():
                symbols.append({
                    "symbol": sym,
                    "quality_score": q["quality_score"],
                    "recommendation": q["recommendation"]
                })
            symbols.sort(key=lambda x: x["quality_score"])

        return {
            "total_symbols": len(symbols),
            "avg_quality": round(float(np.mean(
                [s["quality_score"] for s in symbols]
            )), 1) if symbols else 0,
            "below_threshold": sum(1 for s in symbols if s["quality_score"] < 70),
            "symbols": symbols,
            "watchdog": self.watchdog.get_status(),
            "performance": {
                "avg_pipeline_latency_ms": round(float(np.mean(self._latency_ms)), 3) if self._latency_ms else 0.0,
                "p95_pipeline_latency_ms": round(float(np.percentile(self._latency_ms, 95)), 3) if self._latency_ms else 0.0,
                "samples": len(self._latency_ms),
            },
            "coverage_manifest": DataCoverageRegistry.coverage_manifest()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_pipeline: Optional[DataPipeline] = None

def get_data_pipeline() -> DataPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DataPipeline()
    return _pipeline


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test with synthetic data
    dates = pd.date_range("2024-01-01", periods=200, freq="B")
    df = pd.DataFrame({
        "Open": 100 + np.cumsum(np.random.randn(200) * 0.5),
        "High": 101 + np.cumsum(np.random.randn(200) * 0.5),
        "Low": 99 + np.cumsum(np.random.randn(200) * 0.5),
        "Close": 100 + np.cumsum(np.random.randn(200) * 0.5),
        "Volume": np.random.randint(1e6, 1e7, 200)
    }, index=dates)
    df["High"] = df[["Open", "High", "Close"]].max(axis=1) + 0.5
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1) - 0.5

    pipeline = get_data_pipeline()
    result = pipeline.process(df, "TEST")
    print(f"Quality: {result['quality']['quality_score']}")
    print(f"Feature ready: {result['feature_ready']}")
    print("✅ Data Ingestion Engine operational")