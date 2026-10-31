"""
=============================================================================
RENAISSANCE.IO — INSTITUTIONAL PORTFOLIO ENGINE
=============================================================================
Layer on top of portfolio_optimizer.py and execution_engine.py that adds:

  1. FACTOR-NEUTRAL CONSTRUCTION
     Neutralize market beta, sector, and style exposures so the portfolio
     only holds residual alpha — not factor bets.

  2. VOLATILITY TARGETING
     Scale gross exposure dynamically so portfolio realizes target vol.
     Low-vol environments → lever up. High-vol → de-lever.

  3. TURNOVER BUDGET
     Limit daily rebalancing cost. Trades only happen when the marginal
     alpha improvement exceeds the round-trip transaction cost.

  4. DYNAMIC POSITION SIZING
     Inverse-volatility weighting: volatile stocks get smaller positions.
     Scales by conviction, liquidity, and regime.

  5. EXECUTION ANALYTICS (TCA)
     Track slippage, implementation shortfall, fill quality per trade.
     Post-trade analysis for continuous execution improvement.

  6. PORTFOLIO ATTRIBUTION
     Decompose P&L into: alpha return + factor return + trading cost.

Renaissance.io — Institutional Grade Quantitative Trading
=============================================================================
"""

import numpy as np
import logging
import threading
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque

logger = logging.getLogger("INST_PORTFOLIO")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FACTOR-NEUTRAL PORTFOLIO CONSTRUCTOR
# ═══════════════════════════════════════════════════════════════════════════════

class FactorNeutralConstructor:
    """
    Takes raw alpha weights and adjusts them to be factor-neutral.

    Method:
      1. Estimate factor exposures (beta, sector, style) of raw portfolio
      2. Add offsetting positions to neutralize each factor
      3. Result: portfolio return ≈ pure alpha, not factor bets

    Why: If your "alpha" is just long-tech-short-energy, you're making a
    sector bet, not extracting alpha. Factor neutrality isolates true edge.
    """

    # Target exposures (0 = neutral)
    TARGET_BETA = 0.0          # Market-neutral
    TARGET_SECTOR_MAX = 0.05   # Max 5% net sector tilt
    BETA_TOLERANCE = 0.10      # Allow ±0.10 beta deviation

    @classmethod
    def neutralize(cls, raw_weights: Dict[str, float],
                   betas: Dict[str, float],
                   sectors: Dict[str, str],
                   volatilities: Dict[str, float]) -> Dict:
        """
        Adjust raw alpha weights to be approximately factor-neutral.

        Args:
            raw_weights: {symbol: weight} from alpha optimizer
            betas: {symbol: market_beta} estimated from regression
            sectors: {symbol: "Technology"|"Financials"|...}
            volatilities: {symbol: annualized_vol}

        Returns:
            Dict with adjusted_weights, exposures_before, exposures_after
        """
        if not raw_weights:
            return {"adjusted_weights": {}, "exposures_before": {}, "exposures_after": {}}

        symbols = list(raw_weights.keys())
        weights = np.array([raw_weights[s] for s in symbols])

        # ── Before: compute factor exposures ──────────────────────────────
        beta_arr = np.array([betas.get(s, 1.0) for s in symbols])
        port_beta_before = float(np.dot(weights, beta_arr))

        # Sector exposure
        sector_exposure_before = defaultdict(float)
        for s, w in zip(symbols, weights):
            sector_exposure_before[sectors.get(s, "unknown")] += w

        # ── Neutralize market beta ────────────────────────────────────────
        # Simple approach: scale long and short sides to equalize beta
        if abs(port_beta_before) > cls.BETA_TOLERANCE:
            # Hedge ratio: how much SPY exposure to add/subtract
            hedge_needed = -port_beta_before
            # Rather than adding SPY, we tilt existing weights
            # Reduce weights of high-beta longs, increase low-beta longs
            beta_centered = beta_arr - np.mean(beta_arr)
            adjustment = hedge_needed * beta_centered / (np.dot(beta_centered, beta_centered) + 1e-10)
            weights_adj = weights - adjustment * 0.5  # partial neutralization
        else:
            weights_adj = weights.copy()

        # ── Neutralize sector tilts ───────────────────────────────────────
        # Soft constraint: if any sector > TARGET_SECTOR_MAX, scale it down
        for i, s in enumerate(symbols):
            sec = sectors.get(s, "unknown")
            sec_total = sum(weights_adj[j] for j, sym in enumerate(symbols)
                          if sectors.get(sym, "unknown") == sec)
            if abs(sec_total) > cls.TARGET_SECTOR_MAX:
                scale = cls.TARGET_SECTOR_MAX / abs(sec_total)
                for j, sym in enumerate(symbols):
                    if sectors.get(sym, "unknown") == sec:
                        weights_adj[j] *= scale

        # ── After: recompute exposures ────────────────────────────────────
        port_beta_after = float(np.dot(weights_adj, beta_arr))
        sector_exposure_after = defaultdict(float)
        for s, w in zip(symbols, weights_adj):
            sector_exposure_after[sectors.get(s, "unknown")] += w

        # Build result
        adjusted = {symbols[i]: round(float(weights_adj[i]), 6) for i in range(len(symbols))
                    if abs(weights_adj[i]) > 0.001}

        return {
            "adjusted_weights": adjusted,
            "n_positions": len(adjusted),
            "gross_exposure": round(float(np.sum(np.abs(weights_adj))), 4),
            "net_exposure": round(float(np.sum(weights_adj)), 4),
            "exposures_before": {
                "beta": round(port_beta_before, 4),
                "sectors": dict(sector_exposure_before),
            },
            "exposures_after": {
                "beta": round(port_beta_after, 4),
                "sectors": dict(sector_exposure_after),
            },
            "beta_reduction": round(abs(port_beta_before) - abs(port_beta_after), 4),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. VOLATILITY TARGETING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class VolatilityTargeter:
    """
    Scales portfolio gross exposure to achieve target realized volatility.

    If target = 10% annual vol:
      - Current realized vol = 15% → scale to 10/15 = 0.67x
      - Current realized vol = 7%  → scale to 10/7 = 1.43x

    This is how Renaissance maintains consistent risk regardless of regime.
    """

    DEFAULT_TARGET_VOL = 0.10  # 10% annualized
    MAX_LEVERAGE = 2.0         # Never exceed 2x gross
    MIN_LEVERAGE = 0.2         # Never go below 0.2x gross
    VOL_LOOKBACK = 20          # 20-day realized vol
    SMOOTHING = 0.94           # EWM decay for vol estimate

    _realized_vol_ewm: float = 0.15  # initial estimate
    _lock = threading.Lock()

    @classmethod
    def compute_scaling(cls, portfolio_returns: List[float],
                        target_vol: float = None) -> Dict:
        """
        Compute leverage scaling factor to target volatility.

        Args:
            portfolio_returns: recent daily returns of the portfolio
            target_vol: annualized target volatility (default 10%)

        Returns:
            Dict with scaling_factor, realized_vol, target_vol, etc.
        """
        if target_vol is None:
            target_vol = cls.DEFAULT_TARGET_VOL

        with cls._lock:
            if len(portfolio_returns) < 5:
                return {
                    "scaling_factor": 1.0,
                    "realized_vol_ann": cls._realized_vol_ewm,
                    "target_vol_ann": target_vol,
                    "status": "insufficient_data",
                }

            returns = np.array(portfolio_returns[-cls.VOL_LOOKBACK:])

            # EWM volatility estimate
            weights = np.array([cls.SMOOTHING ** i for i in range(len(returns) - 1, -1, -1)])
            weights /= weights.sum()
            weighted_mean = np.dot(weights, returns)
            weighted_var = np.dot(weights, (returns - weighted_mean) ** 2)
            realized_vol_daily = np.sqrt(weighted_var)
            realized_vol_ann = realized_vol_daily * np.sqrt(252)

            # Update EWM
            cls._realized_vol_ewm = 0.9 * cls._realized_vol_ewm + 0.1 * realized_vol_ann

            # Scaling factor
            if realized_vol_ann > 0.001:
                raw_scaling = target_vol / realized_vol_ann
            else:
                raw_scaling = 1.0

            # Clip to bounds
            scaling = float(np.clip(raw_scaling, cls.MIN_LEVERAGE, cls.MAX_LEVERAGE))

            # Regime awareness: in extreme vol, be more conservative
            if realized_vol_ann > 0.30:  # >30% vol = crisis
                scaling *= 0.7
            elif realized_vol_ann > 0.20:  # >20% vol = stressed
                scaling *= 0.85

            return {
                "scaling_factor": round(scaling, 4),
                "realized_vol_daily": round(float(realized_vol_daily), 6),
                "realized_vol_ann": round(float(realized_vol_ann), 4),
                "target_vol_ann": round(target_vol, 4),
                "vol_ratio": round(float(target_vol / max(realized_vol_ann, 0.001)), 3),
                "status": "active",
                "regime_adjustment": scaling < raw_scaling,
            }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TURNOVER BUDGET CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════

class TurnoverBudgetController:
    """
    Controls rebalancing cost by limiting turnover.

    Only rebalance a position if:
      expected_alpha_gain > round_trip_cost × cost_multiplier

    This prevents over-trading on noisy signals (a major source of
    alpha decay in production quant systems).
    """

    MAX_DAILY_TURNOVER = 0.30    # 30% of portfolio per day max
    COST_MULTIPLIER = 2.0        # Alpha must exceed 2× cost to trade
    MIN_TRADE_SIZE = 0.005       # Don't rebalance below 0.5% weight change
    ROUND_TRIP_COST_BPS = 10     # 10 bps round-trip (spread + commission)

    _daily_turnover: float = 0.0
    _trade_log: deque = deque(maxlen=500)
    _lock = threading.Lock()

    @classmethod
    def filter_trades(cls, current_weights: Dict[str, float],
                      target_weights: Dict[str, float],
                      alpha_scores: Dict[str, float],
                      portfolio_value: float) -> Dict:
        """
        Filter proposed rebalancing trades through turnover budget.

        Returns only the trades that pass the cost-benefit threshold.
        """
        all_symbols = set(list(current_weights.keys()) + list(target_weights.keys()))
        approved_trades = {}
        rejected_trades = {}
        total_turnover = 0.0

        # Sort by alpha magnitude (highest alpha first — they get budget priority)
        trade_proposals = []
        for sym in all_symbols:
            curr_w = current_weights.get(sym, 0.0)
            tgt_w = target_weights.get(sym, 0.0)
            delta_w = tgt_w - curr_w

            if abs(delta_w) < cls.MIN_TRADE_SIZE:
                continue

            alpha = abs(alpha_scores.get(sym, 0.0))
            trade_proposals.append((sym, curr_w, tgt_w, delta_w, alpha))

        # Sort by alpha (descending) — best alpha trades get priority
        trade_proposals.sort(key=lambda x: -x[4])

        for sym, curr_w, tgt_w, delta_w, alpha in trade_proposals:
            turnover_this = abs(delta_w)

            # Check: would this exceed daily budget?
            if total_turnover + turnover_this > cls.MAX_DAILY_TURNOVER:
                rejected_trades[sym] = {
                    "reason": "turnover_budget_exceeded",
                    "delta_w": round(delta_w, 4),
                    "alpha": round(alpha, 4),
                }
                continue

            # Check: does alpha justify the cost?
            cost_bps = cls.ROUND_TRIP_COST_BPS
            alpha_bps = alpha * 10000  # convert to bps
            if alpha_bps < cost_bps * cls.COST_MULTIPLIER:
                rejected_trades[sym] = {
                    "reason": "insufficient_alpha_vs_cost",
                    "alpha_bps": round(alpha_bps, 1),
                    "cost_threshold_bps": round(cost_bps * cls.COST_MULTIPLIER, 1),
                }
                continue

            approved_trades[sym] = {
                "current_weight": round(curr_w, 4),
                "target_weight": round(tgt_w, 4),
                "delta_weight": round(delta_w, 4),
                "delta_dollars": round(delta_w * portfolio_value, 0),
                "direction": "BUY" if delta_w > 0 else "SELL",
                "alpha_bps": round(alpha_bps, 1),
                "estimated_cost_bps": cost_bps,
            }
            total_turnover += turnover_this

        with cls._lock:
            cls._daily_turnover = total_turnover

        return {
            "approved": approved_trades,
            "rejected": rejected_trades,
            "total_turnover_pct": round(total_turnover * 100, 2),
            "budget_remaining_pct": round((cls.MAX_DAILY_TURNOVER - total_turnover) * 100, 2),
            "n_approved": len(approved_trades),
            "n_rejected": len(rejected_trades),
            "estimated_total_cost_bps": round(total_turnover * cls.ROUND_TRIP_COST_BPS, 1),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DYNAMIC POSITION SIZER
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicPositionSizer:
    """
    Inverse-volatility weighted position sizing.

    Each position weight is scaled by:
      w_i = base_weight × (target_stock_vol / realized_vol_i) × conviction × liquidity_adj

    Volatile stocks get smaller positions. Illiquid stocks get smaller positions.
    High-conviction signals get larger positions.
    """

    TARGET_STOCK_CONTRIBUTION = 0.01  # 1% vol contribution per position
    MAX_SINGLE_POSITION = 0.10        # Never more than 10%
    MIN_SINGLE_POSITION = 0.005       # Never less than 0.5%

    @classmethod
    def size_positions(cls, symbols: List[str],
                       alpha_scores: Dict[str, float],
                       volatilities: Dict[str, float],
                       liquidities: Dict[str, float],
                       convictions: Dict[str, float],
                       portfolio_value: float) -> Dict:
        """
        Compute position sizes using inverse-vol weighting.

        Returns:
            {symbol: {weight, shares, dollar_value, vol_contribution, ...}}
        """
        if not symbols:
            return {"positions": [], "summary": {}}

        positions = []
        for sym in symbols:
            alpha = alpha_scores.get(sym, 0)
            vol = max(volatilities.get(sym, 0.30), 0.05)  # floor at 5%
            liq = liquidities.get(sym, 0.5)  # 0-1 liquidity score
            conv = convictions.get(sym, 0.5)  # 0-1 conviction

            # Inverse vol weight
            inv_vol_w = cls.TARGET_STOCK_CONTRIBUTION / vol

            # Scale by conviction (0.5 → 1x, 1.0 → 2x)
            conviction_mult = 0.5 + conv

            # Scale by liquidity (low liquidity → smaller position)
            liq_mult = 0.5 + liq * 0.5

            # Direction from alpha sign
            direction = "LONG" if alpha > 0 else "SHORT" if alpha < 0 else "FLAT"

            raw_weight = inv_vol_w * conviction_mult * liq_mult * abs(alpha) * 10
            weight = float(np.clip(raw_weight, cls.MIN_SINGLE_POSITION, cls.MAX_SINGLE_POSITION))

            if direction == "SHORT":
                weight = -weight

            dollar_value = weight * portfolio_value
            vol_contribution = abs(weight) * vol

            positions.append({
                "symbol": sym,
                "weight": round(weight, 5),
                "dollar_value": round(dollar_value, 0),
                "direction": direction,
                "alpha_score": round(alpha, 4),
                "volatility": round(vol, 4),
                "liquidity": round(liq, 3),
                "conviction": round(conv, 3),
                "vol_contribution": round(vol_contribution, 4),
                "inv_vol_factor": round(inv_vol_w, 4),
            })

        positions.sort(key=lambda x: -abs(x["weight"]))

        # Summary
        total_long = sum(p["weight"] for p in positions if p["weight"] > 0)
        total_short = abs(sum(p["weight"] for p in positions if p["weight"] < 0))
        gross = total_long + total_short
        net = total_long - total_short
        total_vol = sum(p["vol_contribution"] for p in positions)

        return {
            "positions": positions,
            "summary": {
                "n_long": sum(1 for p in positions if p["direction"] == "LONG"),
                "n_short": sum(1 for p in positions if p["direction"] == "SHORT"),
                "gross_exposure": round(gross, 4),
                "net_exposure": round(net, 4),
                "total_dollar_long": round(total_long * portfolio_value, 0),
                "total_dollar_short": round(total_short * portfolio_value, 0),
                "portfolio_vol_est": round(total_vol, 4),
                "avg_position_size": round(gross / max(len(positions), 1), 4),
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. EXECUTION ANALYTICS (TCA)
# ═══════════════════════════════════════════════════════════════════════════════

class TransactionCostAnalyzer:
    """
    Post-trade analytics for execution quality measurement.

    Tracks:
      - Implementation Shortfall (IS): decision_price vs fill_price
      - Slippage: expected vs realized
      - Fill rate and partial fill analysis
      - Timing cost: delay between signal and execution
      - Market impact estimation

    This is how institutional desks measure and improve execution.
    """

    _trade_log: deque = deque(maxlen=2000)
    _lock = threading.Lock()

    @classmethod
    def log_trade(cls, symbol: str, side: str, shares: int,
                  decision_price: float, fill_price: float,
                  expected_slippage_bps: float = 0,
                  fill_time_seconds: float = 0,
                  order_type: str = "market",
                  signal_strength: float = 0):
        """Log a completed trade for TCA analysis."""
        with cls._lock:
            slippage_bps = (fill_price - decision_price) / max(decision_price, 0.01) * 10000
            if side.upper() == "SELL":
                slippage_bps = -slippage_bps  # normalize: positive = bad

            implementation_shortfall = abs(slippage_bps)
            cost_dollars = abs(fill_price - decision_price) * shares

            cls._trade_log.append({
                "symbol": symbol,
                "side": side.upper(),
                "shares": shares,
                "decision_price": round(decision_price, 4),
                "fill_price": round(fill_price, 4),
                "slippage_bps": round(slippage_bps, 2),
                "expected_slippage_bps": round(expected_slippage_bps, 2),
                "implementation_shortfall_bps": round(implementation_shortfall, 2),
                "cost_dollars": round(cost_dollars, 2),
                "fill_time_seconds": round(fill_time_seconds, 1),
                "order_type": order_type,
                "signal_strength": round(signal_strength, 3),
                "timestamp": datetime.now().isoformat(),
                "ts": time.time(),
                "quality": "good" if implementation_shortfall < 5 else
                          "acceptable" if implementation_shortfall < 15 else "poor",
            })

    @classmethod
    def get_tca_report(cls, lookback_days: int = 30) -> Dict:
        """Generate TCA report for recent trades."""
        with cls._lock:
            trades = list(cls._trade_log)

        if not trades:
            return {
                "n_trades": 0,
                "avg_slippage_bps": 0,
                "total_cost_dollars": 0,
                "message": "No trades logged yet",
            }

        cutoff = time.time() - lookback_days * 86400
        recent = [t for t in trades if t["ts"] > cutoff]
        if not recent:
            recent = trades[-50:]  # fallback to last 50

        slippages = [t["slippage_bps"] for t in recent]
        is_vals = [t["implementation_shortfall_bps"] for t in recent]
        costs = [t["cost_dollars"] for t in recent]
        fill_times = [t["fill_time_seconds"] for t in recent]

        # Quality distribution
        quality_dist = defaultdict(int)
        for t in recent:
            quality_dist[t["quality"]] += 1

        # By order type
        by_type = defaultdict(list)
        for t in recent:
            by_type[t["order_type"]].append(t["slippage_bps"])

        type_avg = {k: round(float(np.mean(v)), 2) for k, v in by_type.items()}

        # Slippage vs expected
        expected_vs_actual = []
        for t in recent[-20:]:
            expected_vs_actual.append({
                "symbol": t["symbol"],
                "expected": t["expected_slippage_bps"],
                "actual": t["slippage_bps"],
                "delta": round(t["slippage_bps"] - t["expected_slippage_bps"], 2),
            })

        return {
            "n_trades": len(recent),
            "avg_slippage_bps": round(float(np.mean(slippages)), 2),
            "median_slippage_bps": round(float(np.median(slippages)), 2),
            "p95_slippage_bps": round(float(np.percentile(slippages, 95)), 2),
            "avg_implementation_shortfall_bps": round(float(np.mean(is_vals)), 2),
            "total_cost_dollars": round(float(np.sum(costs)), 2),
            "avg_fill_time_seconds": round(float(np.mean(fill_times)), 1),
            "quality_distribution": dict(quality_dist),
            "avg_slippage_by_type": type_avg,
            "expected_vs_actual": expected_vs_actual,
            "lookback_days": lookback_days,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PORTFOLIO ATTRIBUTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class PortfolioAttribution:
    """
    Decomposes portfolio P&L into:
      - Alpha return (residual after factor exposure)
      - Factor return (market, sector, style contributions)
      - Trading cost (slippage + commission)
      - Timing cost (signal-to-execution delay)

    This tells you exactly WHERE your returns come from.
    """

    @classmethod
    def attribute(cls, portfolio_return: float,
                  factor_returns: Dict[str, float],
                  factor_betas: Dict[str, float],
                  trading_cost_pct: float = 0) -> Dict:
        """
        Attribute portfolio return to sources.

        Args:
            portfolio_return: total portfolio return (daily, %)
            factor_returns: {factor_name: return_pct}
            factor_betas: {factor_name: portfolio_beta_to_factor}
            trading_cost_pct: total trading cost as % of portfolio
        """
        # Factor contribution = Σ(beta_i × factor_return_i)
        factor_contributions = {}
        total_factor_return = 0
        for fname, fret in factor_returns.items():
            beta = factor_betas.get(fname, 0)
            contribution = beta * fret
            factor_contributions[fname] = {
                "beta": round(beta, 4),
                "factor_return": round(fret, 4),
                "contribution": round(contribution, 4),
            }
            total_factor_return += contribution

        # Alpha = total - factor - cost
        alpha_return = portfolio_return - total_factor_return - trading_cost_pct

        return {
            "total_return_pct": round(portfolio_return, 4),
            "alpha_return_pct": round(alpha_return, 4),
            "factor_return_pct": round(total_factor_return, 4),
            "trading_cost_pct": round(trading_cost_pct, 4),
            "factor_contributions": factor_contributions,
            "alpha_share_pct": round(alpha_return / max(abs(portfolio_return), 0.0001) * 100, 1),
            "factor_share_pct": round(total_factor_return / max(abs(portfolio_return), 0.0001) * 100, 1),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MASTER ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class InstitutionalPortfolioPlatform:
    """
    Orchestrates the full institutional portfolio pipeline:

      Alpha scores → Position sizing → Factor neutralization →
      Vol targeting → Turnover filtering → Execution → TCA

    This is the production workflow of an institutional quant fund.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.factor_neutral = FactorNeutralConstructor
        self.vol_targeter = VolatilityTargeter
        self.turnover_ctrl = TurnoverBudgetController
        self.position_sizer = DynamicPositionSizer
        self.tca = TransactionCostAnalyzer
        self.attribution = PortfolioAttribution
        logger.info("✅ InstitutionalPortfolioPlatform initialized")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def construct_portfolio(self, alpha_scores: Dict[str, float],
                            volatilities: Dict[str, float],
                            betas: Dict[str, float],
                            sectors: Dict[str, str],
                            liquidities: Dict[str, float],
                            convictions: Dict[str, float],
                            portfolio_value: float,
                            current_weights: Dict[str, float] = None,
                            portfolio_returns: List[float] = None,
                            target_vol: float = 0.10) -> Dict:
        """
        Full institutional portfolio construction pipeline.
        """
        if not alpha_scores:
            return {"error": "no alpha scores provided"}

        # 1. Dynamic position sizing
        symbols = list(alpha_scores.keys())
        sizing = self.position_sizer.size_positions(
            symbols, alpha_scores, volatilities,
            liquidities, convictions, portfolio_value
        )

        raw_weights = {p["symbol"]: p["weight"] for p in sizing["positions"]}

        # 2. Factor neutralization
        neutral = self.factor_neutral.neutralize(
            raw_weights, betas, sectors, volatilities
        )
        adjusted_weights = neutral.get("adjusted_weights", raw_weights)

        # 3. Volatility targeting
        vol_scaling = {"scaling_factor": 1.0, "status": "no_data"}
        if portfolio_returns and len(portfolio_returns) >= 5:
            vol_scaling = self.vol_targeter.compute_scaling(
                portfolio_returns, target_vol
            )
            # Apply scaling to all weights
            scale = vol_scaling["scaling_factor"]
            adjusted_weights = {s: w * scale for s, w in adjusted_weights.items()}

        # 4. Turnover budget filtering
        turnover = {"n_approved": len(adjusted_weights), "n_rejected": 0}
        if current_weights:
            turnover = self.turnover_ctrl.filter_trades(
                current_weights, adjusted_weights, alpha_scores, portfolio_value
            )
            # Only keep approved trades
            final_weights = {}
            for sym in adjusted_weights:
                if sym in turnover.get("approved", {}):
                    final_weights[sym] = adjusted_weights[sym]
                elif sym in (current_weights or {}):
                    final_weights[sym] = current_weights[sym]  # keep existing
        else:
            final_weights = adjusted_weights

        # 5. Compute summary statistics
        long_w = sum(w for w in final_weights.values() if w > 0)
        short_w = abs(sum(w for w in final_weights.values() if w < 0))

        return {
            "final_weights": {k: round(v, 5) for k, v in final_weights.items() if abs(v) > 0.001},
            "position_sizing": sizing,
            "factor_neutralization": neutral,
            "vol_targeting": vol_scaling,
            "turnover_analysis": turnover,
            "summary": {
                "n_positions": len([w for w in final_weights.values() if abs(w) > 0.001]),
                "gross_exposure": round(long_w + short_w, 4),
                "net_exposure": round(long_w - short_w, 4),
                "long_exposure": round(long_w, 4),
                "short_exposure": round(short_w, 4),
                "portfolio_value": portfolio_value,
                "target_vol": target_vol,
                "vol_scaling_factor": vol_scaling.get("scaling_factor", 1.0),
            },
            "updated": datetime.now().isoformat(),
        }

    def get_tca_report(self, lookback_days: int = 30) -> Dict:
        return self.tca.get_tca_report(lookback_days)

    def get_execution_dashboard(self) -> Dict:
        """Full execution quality dashboard."""
        tca = self.tca.get_tca_report()
        return {
            "tca": tca,
            "turnover_budget": {
                "daily_limit_pct": self.turnover_ctrl.MAX_DAILY_TURNOVER * 100,
                "used_today_pct": round(self.turnover_ctrl._daily_turnover * 100, 2),
                "remaining_pct": round((self.turnover_ctrl.MAX_DAILY_TURNOVER - self.turnover_ctrl._daily_turnover) * 100, 2),
                "cost_multiplier": self.turnover_ctrl.COST_MULTIPLIER,
            },
            "vol_targeting": {
                "target_vol": self.vol_targeter.DEFAULT_TARGET_VOL,
                "current_realized_vol": round(self.vol_targeter._realized_vol_ewm, 4),
                "max_leverage": self.vol_targeter.MAX_LEVERAGE,
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_institutional_platform() -> InstitutionalPortfolioPlatform:
    return InstitutionalPortfolioPlatform.get_instance()

def construct_portfolio(**kwargs) -> Dict:
    return get_institutional_platform().construct_portfolio(**kwargs)

def get_tca_report(lookback_days: int = 30) -> Dict:
    return get_institutional_platform().get_tca_report(lookback_days)

def get_execution_dashboard() -> Dict:
    return get_institutional_platform().get_execution_dashboard()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    platform = get_institutional_platform()

    # Test position sizing
    result = platform.construct_portfolio(
        alpha_scores={"AAPL": 0.8, "MSFT": 0.6, "TSLA": -0.3, "JPM": 0.4},
        volatilities={"AAPL": 0.25, "MSFT": 0.22, "TSLA": 0.55, "JPM": 0.20},
        betas={"AAPL": 1.1, "MSFT": 1.0, "TSLA": 1.8, "JPM": 0.9},
        sectors={"AAPL": "Technology", "MSFT": "Technology", "TSLA": "Consumer Disc.", "JPM": "Financials"},
        liquidities={"AAPL": 0.95, "MSFT": 0.95, "TSLA": 0.90, "JPM": 0.85},
        convictions={"AAPL": 0.8, "MSFT": 0.6, "TSLA": 0.4, "JPM": 0.5},
        portfolio_value=100000,
    )
    print(json.dumps(result["summary"], indent=2))