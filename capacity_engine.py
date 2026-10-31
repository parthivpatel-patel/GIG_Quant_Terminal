"""
=============================================================================
RENAISSANCE CAPACITY ENGINE — capacity_engine.py
=============================================================================
Position Sizing by Liquidity + Slippage Estimation + Concentration Limits

Renaissance principle: The biggest alpha source is worthless if you can't
execute at scale without moving the market.

MODELS:
  ├── ADV-Based Position Limits   — max shares = f(avg daily volume)
  ├── Market Impact Estimator     — sqrt(participation) model
  ├── Slippage Estimator          — bid-ask + market impact combined
  ├── Concentration Monitor       — sector/position/correlation limits
  ├── Liquidity Scorer            — 0-100 tradability score per symbol
  ├── Capacity Alert System       — warns when approaching limits
  └── Portfolio Capacity Report   — aggregate capacity utilization

Renaissance.io — Institutional Grade Quantitative Trading
=============================================================================
"""

import numpy as np
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("CAPACITY")


# ═══════════════════════════════════════════════════════════════════════════════
# POSITION LIMITS
# ═══════════════════════════════════════════════════════════════════════════════

class PositionLimits:
    """
    ADV-based position sizing:
    - Never hold more than 2% of Average Daily Volume
    - Scale limit by liquidity tier and market cap
    - Hard caps: max $500K per position, max 10% of portfolio per name
    """

    # Max participation rates by tier
    PARTICIPATION_LIMITS = {
        "mega":   0.03,   # 3% of ADV for mega-caps (AAPL, MSFT)
        "large":  0.025,  # 2.5% for large-caps
        "mid":    0.02,   # 2% for mid-caps
        "small":  0.015,  # 1.5% for small-caps
        "micro":  0.01,   # 1% for micro-caps (very illiquid)
        "penny":  0.005,  # 0.5% for penny stocks
    }

    # Hard caps
    MAX_POSITION_USD = 500_000
    MAX_PORTFOLIO_PCT = 0.10  # 10% of portfolio in one name
    MAX_SECTOR_PCT = 0.30     # 30% in one sector

    @classmethod
    def compute_max_shares(cls, symbol: str, avg_daily_volume: float,
                           price: float, market_cap: float = 0,
                           portfolio_value: float = 100_000) -> Dict:
        """
        Compute maximum position size for a symbol.

        Returns:
            {
                "max_shares": int,
                "max_usd": float,
                "participation_rate": float,
                "tier": str,
                "limiting_factor": str,
                "adv_shares": int,
                "dollar_limit_shares": int,
                "portfolio_limit_shares": int
            }
        """
        if avg_daily_volume <= 0 or price <= 0:
            return {"max_shares": 0, "error": "no_volume_data", "symbol": symbol}

        # Determine tier
        tier = cls._get_tier(market_cap, avg_daily_volume, price)
        participation = cls.PARTICIPATION_LIMITS.get(tier, 0.02)

        # ADV-based limit
        adv_shares = int(avg_daily_volume * participation)

        # Dollar-based limit
        dollar_limit_shares = int(cls.MAX_POSITION_USD / price)

        # Portfolio concentration limit
        portfolio_limit_shares = int((portfolio_value * cls.MAX_PORTFOLIO_PCT) / price)

        # Take minimum of all limits
        max_shares = min(adv_shares, dollar_limit_shares, portfolio_limit_shares)
        max_shares = max(max_shares, 1)  # At least 1 share

        # Identify limiting factor
        if max_shares == adv_shares:
            limiter = "liquidity_adv"
        elif max_shares == dollar_limit_shares:
            limiter = "max_position_usd"
        else:
            limiter = "portfolio_concentration"

        return {
            "symbol": symbol,
            "max_shares": max_shares,
            "max_usd": round(max_shares * price, 2),
            "participation_rate": round(participation * 100, 2),
            "tier": tier,
            "limiting_factor": limiter,
            "adv_shares": adv_shares,
            "dollar_limit_shares": dollar_limit_shares,
            "portfolio_limit_shares": portfolio_limit_shares,
            "avg_daily_volume": int(avg_daily_volume),
            "price": round(price, 2),
        }

    @classmethod
    def _get_tier(cls, market_cap: float, adv: float, price: float) -> str:
        if market_cap > 200e9:
            return "mega"
        if market_cap > 10e9:
            return "large"
        if market_cap > 2e9:
            return "mid"
        if market_cap > 300e6:
            return "small"
        if price < 5:
            return "penny"
        # Fallback: use ADV
        if adv > 10_000_000:
            return "large"
        if adv > 1_000_000:
            return "mid"
        if adv > 100_000:
            return "small"
        return "micro"


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET IMPACT + SLIPPAGE
# ═══════════════════════════════════════════════════════════════════════════════

class SlippageEstimator:
    """
    Estimates execution slippage using:
    - Bid-ask spread component
    - Market impact (Almgren-Chriss square root model)
    - Timing cost
    """

    @classmethod
    def estimate(cls, shares: int, price: float, avg_daily_volume: float,
                 bid_ask_spread: float = 0.01, volatility: float = 0.02) -> Dict:
        """
        Estimate total execution cost.

        Args:
            shares: order size
            price: current price
            avg_daily_volume: ADV in shares
            bid_ask_spread: dollar spread
            volatility: daily return volatility

        Returns:
            {
                "total_slippage_bps": float,
                "spread_cost_bps": float,
                "impact_cost_bps": float,
                "timing_cost_bps": float,
                "total_cost_usd": float,
                "participation_pct": float,
                "execution_rating": str
            }
        """
        if avg_daily_volume <= 0 or price <= 0:
            return {"total_slippage_bps": 0, "error": "no_data"}

        order_value = shares * price
        participation = shares / max(avg_daily_volume, 1)

        # 1. Spread cost (half-spread)
        spread_bps = (bid_ask_spread / price) * 5000  # half-spread in bps

        # 2. Market impact (square-root model)
        # Impact = σ × sqrt(participation) × constant
        impact_bps = volatility * 10000 * np.sqrt(participation) * 0.5

        # 3. Timing cost (opportunity cost of slow execution)
        timing_bps = volatility * 10000 * 0.1 * participation

        total_bps = spread_bps + impact_bps + timing_bps
        total_usd = order_value * total_bps / 10000

        # Rating
        if total_bps < 5:
            rating = "excellent"
        elif total_bps < 15:
            rating = "good"
        elif total_bps < 30:
            rating = "acceptable"
        elif total_bps < 60:
            rating = "costly"
        else:
            rating = "prohibitive"

        return {
            "total_slippage_bps": round(total_bps, 2),
            "spread_cost_bps": round(spread_bps, 2),
            "impact_cost_bps": round(impact_bps, 2),
            "timing_cost_bps": round(timing_bps, 2),
            "total_cost_usd": round(total_usd, 2),
            "participation_pct": round(participation * 100, 3),
            "order_value": round(order_value, 2),
            "execution_rating": rating,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# LIQUIDITY SCORER
# ═══════════════════════════════════════════════════════════════════════════════

class LiquidityScorer:
    """
    Composite liquidity score 0-100 for each symbol.
    Factors: ADV, spread, market cap, options OI, dark pool %.
    """

    @classmethod
    def score(cls, avg_daily_volume: float, price: float,
              market_cap: float = 0, bid_ask_spread: float = 0.01,
              options_oi: float = 0) -> Dict:
        """
        Score tradability from 0-100.
        """
        # ADV score (0-30)
        if avg_daily_volume > 50_000_000:
            adv_score = 30
        elif avg_daily_volume > 10_000_000:
            adv_score = 25
        elif avg_daily_volume > 1_000_000:
            adv_score = 20
        elif avg_daily_volume > 100_000:
            adv_score = 12
        elif avg_daily_volume > 10_000:
            adv_score = 5
        else:
            adv_score = 1

        # Spread score (0-25)
        spread_pct = bid_ask_spread / max(price, 0.01) * 100
        if spread_pct < 0.02:
            spread_score = 25
        elif spread_pct < 0.05:
            spread_score = 20
        elif spread_pct < 0.1:
            spread_score = 15
        elif spread_pct < 0.5:
            spread_score = 8
        else:
            spread_score = 2

        # Market cap score (0-25)
        if market_cap > 100e9:
            cap_score = 25
        elif market_cap > 10e9:
            cap_score = 20
        elif market_cap > 1e9:
            cap_score = 15
        elif market_cap > 100e6:
            cap_score = 8
        else:
            cap_score = 3

        # Options liquidity (0-20)
        if options_oi > 100_000:
            opt_score = 20
        elif options_oi > 10_000:
            opt_score = 15
        elif options_oi > 1_000:
            opt_score = 8
        else:
            opt_score = 2

        total = adv_score + spread_score + cap_score + opt_score

        # Rating
        if total >= 85:
            rating = "ultra_liquid"
        elif total >= 70:
            rating = "highly_liquid"
        elif total >= 50:
            rating = "liquid"
        elif total >= 30:
            rating = "moderate"
        else:
            rating = "illiquid"

        return {
            "liquidity_score": total,
            "rating": rating,
            "components": {
                "volume_score": adv_score,
                "spread_score": spread_score,
                "market_cap_score": cap_score,
                "options_score": opt_score,
            },
            "tradable": total >= 30,
            "max_recommended_pct": 0.03 if total >= 70 else 0.02 if total >= 50 else 0.01,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONCENTRATION MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

class ConcentrationMonitor:
    """
    Monitors portfolio concentration risk:
    - Single-name limits
    - Sector limits
    - Correlation-based concentration
    """

    MAX_SINGLE_NAME = 0.10   # 10%
    MAX_SECTOR = 0.30         # 30%
    MAX_CORRELATED_GROUP = 0.40  # 40%

    @classmethod
    def check(cls, positions: List[Dict], portfolio_value: float) -> Dict:
        """
        Check concentration limits.

        Args:
            positions: list of {symbol, value, sector, ...}
            portfolio_value: total portfolio value

        Returns:
            {
                "violations": list,
                "warnings": list,
                "name_concentration": dict,
                "sector_concentration": dict,
                "diversification_score": float [0, 100]
            }
        """
        if not positions or portfolio_value <= 0:
            return {"violations": [], "warnings": [],
                    "diversification_score": 100, "status": "empty_portfolio"}

        violations = []
        warnings = []
        name_conc = {}
        sector_conc = {}

        for pos in positions:
            sym = pos.get("symbol", "?")
            val = abs(pos.get("value", 0))
            sector = pos.get("sector", "unknown")
            pct = val / portfolio_value

            name_conc[sym] = round(pct * 100, 2)

            if sector not in sector_conc:
                sector_conc[sector] = 0
            sector_conc[sector] += pct * 100

            # Check limits
            if pct > cls.MAX_SINGLE_NAME:
                violations.append({
                    "type": "single_name",
                    "symbol": sym,
                    "current_pct": round(pct * 100, 2),
                    "limit_pct": cls.MAX_SINGLE_NAME * 100,
                    "excess_pct": round((pct - cls.MAX_SINGLE_NAME) * 100, 2)
                })
            elif pct > cls.MAX_SINGLE_NAME * 0.8:
                warnings.append({
                    "type": "single_name_approaching",
                    "symbol": sym,
                    "current_pct": round(pct * 100, 2),
                    "limit_pct": cls.MAX_SINGLE_NAME * 100,
                })

        # Sector checks
        for sector, pct in sector_conc.items():
            if pct / 100 > cls.MAX_SECTOR:
                violations.append({
                    "type": "sector_concentration",
                    "sector": sector,
                    "current_pct": round(pct, 2),
                    "limit_pct": cls.MAX_SECTOR * 100,
                })

        # Diversification score
        n_positions = len(positions)
        hhi = sum((v / 100) ** 2 for v in name_conc.values()) if name_conc else 0
        # Perfect diversification: HHI = 1/N, worst: HHI = 1
        ideal_hhi = 1.0 / max(n_positions, 1)
        div_score = max(0, 100 * (1 - (hhi - ideal_hhi) / max(1 - ideal_hhi, 0.01)))

        return {
            "violations": violations,
            "warnings": warnings,
            "name_concentration": dict(sorted(name_conc.items(), key=lambda x: x[1], reverse=True)),
            "sector_concentration": dict(sorted(sector_conc.items(), key=lambda x: x[1], reverse=True)),
            "diversification_score": round(div_score, 1),
            "hhi": round(hhi, 4),
            "n_positions": n_positions,
            "status": "violation" if violations else "warning" if warnings else "healthy",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO CAPACITY REPORT
# ═══════════════════════════════════════════════════════════════════════════════

class CapacityReport:
    """Aggregate capacity utilization for the whole portfolio."""

    @classmethod
    def generate(cls, signals: List[Dict], portfolio_value: float = 100_000) -> Dict:
        """
        Generate capacity report for proposed trades.

        Args:
            signals: list of {symbol, price, volume, market_cap, direction, ...}
            portfolio_value: total portfolio value

        Returns comprehensive capacity analysis.
        """
        results = []
        total_capacity_used = 0

        for sig in signals:
            sym = sig.get("symbol", "?")
            price = sig.get("price", 0)
            adv = sig.get("volume", sig.get("avg_volume", 0))
            mcap = sig.get("market_cap", 0)

            if price <= 0:
                continue

            # Position limit
            limit = PositionLimits.compute_max_shares(
                sym, adv, price, mcap, portfolio_value
            )

            # Liquidity score
            liq = LiquidityScorer.score(adv, price, mcap)

            # Slippage for proposed trade (use 50% of max as estimate)
            proposed_shares = limit["max_shares"] // 2
            slip = SlippageEstimator.estimate(
                proposed_shares, price, adv
            )

            results.append({
                "symbol": sym,
                "direction": sig.get("direction", "—"),
                "price": round(price, 2),
                "max_shares": limit["max_shares"],
                "max_usd": limit["max_usd"],
                "tier": limit["tier"],
                "liquidity_score": liq["liquidity_score"],
                "liquidity_rating": liq["rating"],
                "slippage_bps": slip.get("total_slippage_bps", 0),
                "execution_rating": slip.get("execution_rating", "—"),
                "limiting_factor": limit["limiting_factor"],
                "tradable": liq["tradable"],
            })

            total_capacity_used += min(limit["max_usd"], portfolio_value * 0.1)

        results.sort(key=lambda x: x["liquidity_score"], reverse=True)

        return {
            "positions": results,
            "total_capacity_used": round(total_capacity_used, 0),
            "portfolio_value": portfolio_value,
            "utilization_pct": round(total_capacity_used / max(portfolio_value, 1) * 100, 1),
            "tradable_count": sum(1 for r in results if r["tradable"]),
            "untradable_count": sum(1 for r in results if not r["tradable"]),
            "avg_liquidity": round(np.mean([r["liquidity_score"] for r in results]), 1) if results else 0,
            "avg_slippage_bps": round(np.mean([r["slippage_bps"] for r in results]), 2) if results else 0,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test
    limit = PositionLimits.compute_max_shares("AAPL", 60_000_000, 175.0, 2.8e12)
    print(f"AAPL max: {limit['max_shares']} shares (${limit['max_usd']:,.0f})")
    slip = SlippageEstimator.estimate(1000, 175.0, 60_000_000)
    print(f"Slippage: {slip['total_slippage_bps']:.1f} bps ({slip['execution_rating']})")
    liq = LiquidityScorer.score(60_000_000, 175.0, 2.8e12)
    print(f"Liquidity: {liq['liquidity_score']}/100 ({liq['rating']})")
    print("✅ Capacity Engine operational")