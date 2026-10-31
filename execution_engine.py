"""
=============================================================================
EXECUTION ENGINE
Broker: Alpaca Markets (Paper + Live)
Order types: Market, Limit, Bracket (with stop-loss and take-profit)
Smart routing: TWAP for large orders
=============================================================================
"""

import logging
import time
import threading
import requests
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import sys
sys.path.append('..')
from config import *

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency — loaded on first trade close
_aladdin_scorer = None
def _get_aladdin():
    global _aladdin_scorer
    if _aladdin_scorer is None:
        try:
            from math_engine import AladdinScorer
            _aladdin_scorer = AladdinScorer
        except Exception:
            pass
    return _aladdin_scorer


class AlpacaBroker:
    """
    Alpaca Markets API wrapper.
    Paper trading by default — switch ALPACA_BASE_URL for live.
    """

    def __init__(self):
        self.base_url  = ALPACA_BASE_URL
        self.data_url  = ALPACA_DATA_URL
        self.headers   = {
            "APCA-API-KEY-ID":     ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
            "Content-Type":        "application/json",
        }
        self.session   = requests.Session()
        self.session.headers.update(self.headers)
        self._validate_connection()

    def _validate_connection(self):
        try:
            resp = self.session.get(f"{self.base_url}/v2/account", timeout=5)
            if resp.status_code == 200:
                acct = resp.json()
                logger.info(f"✅ Alpaca connected | Account: {acct.get('account_number')} | "
                            f"Status: {acct.get('status')} | "
                            f"Equity: ${float(acct.get('equity', 0)):,.2f}")
            else:
                logger.warning(f"⚠️ Alpaca connection issue: {resp.status_code} - {resp.text}")
        except Exception as e:
            logger.error(f"❌ Alpaca connection failed: {e}")

    # ─── Account Info ─────────────────────────────────────────────────────────

    def get_account(self) -> Dict:
        try:
            resp = self.session.get(f"{self.base_url}/v2/account", timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error(f"Account fetch failed: {e}")
        return {}

    def get_portfolio_value(self) -> float:
        acct = self.get_account()
        return float(acct.get("equity", 0))

    def get_positions(self) -> List[Dict]:
        try:
            resp = self.session.get(f"{self.base_url}/v2/positions", timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error(f"Positions fetch failed: {e}")
        return []

    def get_open_orders(self) -> List[Dict]:
        try:
            resp = self.session.get(f"{self.base_url}/v2/orders?status=open", timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error(f"Orders fetch failed: {e}")
        return []

    def is_market_open(self) -> bool:
        try:
            resp = self.session.get(f"{self.base_url}/v2/clock", timeout=5)
            if resp.status_code == 200:
                clock = resp.json()
                return clock.get("is_open", False)
        except Exception as e:
            logger.warning(f"Market clock check failed: {e}")
        return False

    def get_clock(self) -> Dict:
        try:
            resp = self.session.get(f"{self.base_url}/v2/clock", timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return {}

    # ─── Order Execution ──────────────────────────────────────────────────────

    def submit_bracket_order(
        self,
        symbol:       str,
        qty:          int,
        side:         str,          # "buy" or "sell"
        stop_loss:    float,
        take_profit:  float,
        time_in_force: str = "day",
    ) -> Dict:
        """
        Bracket order = entry + stop-loss + take-profit in one order.
        Automatically manages the position — no manual monitoring needed.
        """
        if qty < 1:
            return {"error": "Invalid quantity"}

        order_data = {
            "symbol":        symbol,
            "qty":           str(qty),
            "side":          side,
            "type":          "market",
            "time_in_force": time_in_force,
            "order_class":   "bracket",
            "stop_loss":     {"stop_price": str(round(stop_loss, 2))},
            "take_profit":   {"limit_price": str(round(take_profit, 2))},
        }

        try:
            resp = self.session.post(
                f"{self.base_url}/v2/orders",
                json=order_data,
                timeout=10
            )
            if resp.status_code in (200, 201):
                order = resp.json()
                logger.info(
                    f"✅ BRACKET ORDER: {side.upper()} {qty}x {symbol} | "
                    f"SL: ${stop_loss:.2f} | TP: ${take_profit:.2f} | "
                    f"OrderID: {order.get('id', 'N/A')}"
                )
                return order
            else:
                logger.error(f"Order rejected for {symbol}: {resp.status_code} - {resp.text}")
                return {"error": resp.text, "status_code": resp.status_code}
        except Exception as e:
            logger.error(f"Order submission failed for {symbol}: {e}")
            return {"error": str(e)}

    def submit_market_order(
        self,
        symbol:       str,
        qty:          int,
        side:         str,
        time_in_force: str = "day",
    ) -> Dict:
        """Simple market order."""
        order_data = {
            "symbol":        symbol,
            "qty":           str(qty),
            "side":          side,
            "type":          "market",
            "time_in_force": time_in_force,
        }
        try:
            resp = self.session.post(
                f"{self.base_url}/v2/orders",
                json=order_data,
                timeout=10
            )
            if resp.status_code in (200, 201):
                order = resp.json()
                logger.info(f"✅ MARKET ORDER: {side.upper()} {qty}x {symbol}")
                return order
            else:
                logger.error(f"Market order failed for {symbol}: {resp.text}")
                return {"error": resp.text}
        except Exception as e:
            logger.error(f"Market order exception for {symbol}: {e}")
            return {"error": str(e)}

    def close_position(self, symbol: str) -> Dict:
        """Close an entire position immediately."""
        try:
            resp = self.session.delete(
                f"{self.base_url}/v2/positions/{symbol}",
                timeout=10
            )
            if resp.status_code in (200, 201, 204):
                logger.info(f"✅ Position closed: {symbol}")
                return {"success": True, "symbol": symbol}
            else:
                logger.error(f"Close position failed for {symbol}: {resp.text}")
                return {"error": resp.text}
        except Exception as e:
            logger.error(f"Close position exception for {symbol}: {e}")
            return {"error": str(e)}

    def cancel_all_orders(self) -> bool:
        """Cancel all open orders (safety net)."""
        try:
            resp = self.session.delete(f"{self.base_url}/v2/orders", timeout=10)
            logger.info("🛑 All open orders cancelled")
            return resp.status_code in (200, 201, 204, 207)
        except Exception as e:
            logger.error(f"Cancel all orders failed: {e}")
            return False

    def twap_order(
        self,
        symbol:    str,
        total_qty: int,
        side:      str,
        slices:    int = 5,
        interval:  float = 60.0,
    ) -> List[Dict]:
        """
        Time-Weighted Average Price execution.
        Splits large orders into slices to minimize market impact.
        """
        orders = []
        slice_qty = total_qty // slices
        remainder = total_qty % slices

        for i in range(slices):
            qty = slice_qty + (remainder if i == slices - 1 else 0)
            if qty < 1:
                continue
            order = self.submit_market_order(symbol, qty, side)
            orders.append(order)
            logger.info(f"TWAP slice {i+1}/{slices}: {qty}x {symbol}")
            if i < slices - 1:
                time.sleep(interval)

        return orders

    # ─── Position Updates ─────────────────────────────────────────────────────

    def sync_positions(self, risk_engine) -> Dict:
        """
        Sync broker positions with our risk engine.
        Resolves discrepancies between tracked and actual positions.
        """
        try:
            broker_positions = self.get_positions()
            broker_symbols   = {p["symbol"] for p in broker_positions}
            local_symbols    = set(risk_engine.open_positions.keys())

            # Positions broker has that we don't track (shouldn't happen)
            for p in broker_positions:
                sym = p["symbol"]
                if sym not in risk_engine.open_positions:
                    risk_engine.open_positions[sym] = {
                        "symbol":      sym,
                        "qty":         abs(int(p.get("qty", 0))),
                        "entry_price": float(p.get("avg_entry_price", 0)),
                        "direction":   1 if p.get("side") == "long" else -1,
                        "pnl":         float(p.get("unrealized_pl", 0)),
                        "synced":      True,
                    }

            # Positions we track that broker doesn't have
            for sym in list(local_symbols - broker_symbols):
                logger.warning(f"⚠️ Position {sym} not found at broker — removing from local tracking")
                risk_engine.open_positions.pop(sym, None)

            # Update P&L from broker
            portfolio_value = self.get_portfolio_value()
            if portfolio_value > 0:
                risk_engine.update_portfolio_value(portfolio_value)

            return {
                "broker_positions": len(broker_positions),
                "local_positions":  len(risk_engine.open_positions),
                "portfolio_value":  portfolio_value,
            }

        except Exception as e:
            logger.error(f"Position sync failed: {e}")
            return {}


class ExecutionEngine:
    """
    High-level execution manager.
    Receives validated trade params from risk engine and submits orders.

    ADAPTIVE WEIGHT LOOP:
    When a position closes, we compute IC contribution for each signal that
    was active at entry time, then call AladdinScorer.update_signal_ic() and
    AladdinScorer.update_weights_from_ic() to recalibrate signal weights.
    This is the core of Renaissance's compounding edge.
    """

    def __init__(self, risk_engine):
        self.broker            = AlpacaBroker()
        self.risk              = risk_engine
        self.order_log:        List[Dict] = []
        self._signal_snapshots: Dict[str, Dict] = {}  # symbol -> signals at entry
        self._trades_since_recal = 0
        self._recal_every = 5   # full weight recalibration every 5 closed trades
        logger.info("✅ ExecutionEngine initialized with adaptive IC feedback loop")

    @staticmethod
    def _passes_cost_gate(trade_params: Dict) -> Dict:
        """
        Skip trades where expected edge does not exceed estimated execution cost.
        """
        expected_edge_bps = float(trade_params.get("expected_edge_bps", 0) or 0)
        spread_bps = float(trade_params.get("spread_bps", trade_params.get("bid_ask_bps", 8)) or 8)
        slippage_bps = float(trade_params.get("slippage_bps", 5) or 5)
        fee_bps = float(trade_params.get("fee_bps", 1.5) or 1.5)
        impact_bps = float(trade_params.get("impact_bps", 4.5) or 4.5)
        total_cost_bps = spread_bps + slippage_bps + fee_bps + impact_bps
        min_required_bps = max(8.0, total_cost_bps * 1.25)
        ok = expected_edge_bps >= min_required_bps
        return {
            "ok": ok,
            "expected_edge_bps": round(expected_edge_bps, 2),
            "estimated_cost_bps": round(total_cost_bps, 2),
            "min_required_bps": round(min_required_bps, 2),
        }

    def execute_trade(self, trade_params: Dict) -> Dict:
        """Execute a validated trade and snapshot signal values at entry.
        
        Step 12: If trade_params includes 'adv_shares' and 'daily_vol',
        builds an Almgren-Chriss execution plan and stores it for IS tracking.
        For large orders (>1% ADV), logs the AC schedule for staged execution.
        """
        symbol      = trade_params["symbol"]
        qty         = trade_params["qty"]
        direction   = trade_params["direction"]
        stop_loss   = trade_params["stop_loss"]
        take_profit = trade_params["take_profit"]
        side        = "buy" if direction == "BUY" else "sell"

        cost_gate = self._passes_cost_gate(trade_params)
        if not cost_gate["ok"]:
            result = {
                "status": "REJECTED",
                "reason": "cost_gate_failed",
                "symbol": symbol,
                "cost_gate": cost_gate,
            }
            self.order_log.append(result)
            logger.info(
                f"⛔ Cost gate {symbol}: edge={cost_gate['expected_edge_bps']:.1f}bps "
                f"< required={cost_gate['min_required_bps']:.1f}bps"
            )
            return result

        if not self.broker.is_market_open():
            logger.info(f"⏰ Market closed — queueing {symbol} for next open")
            return {"status": "QUEUED", "reason": "market_closed", "trade": trade_params, "cost_gate": cost_gate}

        # ── Almgren-Chriss execution plan (Step 12) ───────────────────────
        ac_plan = {}
        try:
            adv_shares  = float(trade_params.get("adv_shares", 0))
            daily_vol   = float(trade_params.get("daily_vol", _DEFAULT_SIGMA))
            price       = float(trade_params.get("entry_price", 50.0))
            conviction  = float(trade_params.get("conviction", 0.5))
            vix         = float(trade_params.get("vix", 20.0))
            price_hist  = trade_params.get("price_history")

            if adv_shares > 0 and qty > 0:
                ac_plan = AlmgrenChrissSched.build_plan(
                    symbol=symbol, total_shares=qty, side=side,
                    price=price, adv_shares=adv_shares,
                    daily_vol=daily_vol, conviction=conviction, vix=vix,
                    price_history=np.array(price_hist) if price_hist else None,
                )
                participation = qty / adv_shares
                if participation >= 0.01:
                    logger.info(
                        f"📋 AC-Schedule {symbol}: {ac_plan['urgency']} urgency | "
                        f"{ac_plan['n_periods']} periods | "
                        f"{ac_plan['expected_cost_bps']:.1f}bps est. IS | "
                        f"trajectory={ac_plan['trajectory']}"
                    )
                else:
                    logger.debug(f"AC-Schedule {symbol}: small order, single fill optimal")
        except Exception as _ac_e:
            logger.debug(f"AC plan {symbol}: {_ac_e}")

        order = self.broker.submit_bracket_order(
            symbol=symbol, qty=qty, side=side,
            stop_loss=stop_loss, take_profit=take_profit,
        )

        if "error" not in order:
            self.risk.register_trade(trade_params)

            # SNAPSHOT: capture signal sub-scores at entry for IC feedback later
            sub_signals = trade_params.get("sub_signals", {})
            direction_int = 1 if direction == "BUY" else -1
            self._signal_snapshots[symbol] = {
                "sub_signals":  sub_signals,
                "direction":    direction_int,
                "entry_price":  trade_params.get("entry_price", 0),
                "entry_time":   datetime.now().isoformat(),
                "ac_plan":      ac_plan,      # stored for IS tracking on exit
            }
            logger.info(f"📸 Entry snapshot: {symbol} | {len(sub_signals)} signals captured")

            result = {
                "status":    "EXECUTED",
                "order_id":  order.get("id"),
                "symbol":    symbol,
                "side":      side,
                "qty":       qty,
                "timestamp": datetime.now().isoformat(),
                "ac_plan":   ac_plan,
                "cost_gate": cost_gate,
            }
        else:
            result = {"status": "REJECTED", "reason": order.get("error"), "symbol": symbol}

        self.order_log.append(result)
        return result

    def execute_exit(self, symbol: str, price: float, reason: str = "") -> Dict:
        """
        Close a position AND feed realized return back into AladdinScorer IC tracker.
        Every trade close makes the adaptive weight system smarter.
        Step 12: also tracks implementation shortfall vs AC plan.
        """
        snapshot    = self._signal_snapshots.pop(symbol, None)
        pos         = self.risk.open_positions.get(symbol, {})
        entry_price = pos.get("entry_price", price)
        direction   = pos.get("direction", 1)

        result = self.broker.close_position(symbol)
        if "success" in result or "error" not in result:
            self.risk.close_position(symbol, price, reason)

            # IC FEEDBACK: compute realized return and update signal weights
            if snapshot and entry_price > 0:
                actual_return_pct = (price - entry_price) / entry_price * direction
                self._update_ic_from_trade(symbol, snapshot, actual_return_pct)

            # IS TRACKING (Step 12): compare exit vs AC plan if available
            ac_plan = (snapshot or {}).get("ac_plan", {})
            if ac_plan and entry_price > 0:
                try:
                    is_result = AlmgrenChrissSched.track_implementation_shortfall(
                        plan=ac_plan,
                        arrival_price=entry_price,
                        fills=[{"price": price, "shares": ac_plan.get("total_shares", 0)}],
                    )
                    logger.info(
                        f"📊 IS-Track {symbol}: realized={is_result['realized_is_bps']:.1f}bps "
                        f"vs plan={is_result['expected_is_bps']:.1f}bps "
                        f"| quality={is_result['fill_quality']}"
                    )
                    result["is_tracking"] = is_result
                except Exception as _is_e:
                    logger.debug(f"IS tracking {symbol}: {_is_e}")

        return result

    def _update_ic_from_trade(self, symbol: str, snapshot: Dict, actual_return_pct: float):
        """
        The adaptive weight update loop.
        For each signal in the composite, update EWM-IC based on whether the
        signal's direction prediction matched the actual return.
        Every 5 trades, recalibrate the full weight vector.
        """
        try:
            from math_engine import AladdinScorer as _AS
            sub_signals   = snapshot.get("sub_signals", {})
            direction_int = snapshot.get("direction", 1)
            updated = []

            for signal_name, signal_val in sub_signals.items():
                if signal_name not in _AS._DEFAULT_WEIGHTS:
                    continue
                predicted_direction = float(signal_val) * direction_int
                _AS.update_signal_ic(signal_name, predicted_direction, actual_return_pct)
                updated.append(signal_name)

            self._trades_since_recal += 1
            if self._trades_since_recal >= self._recal_every:
                _AS.update_weights_from_ic()
                self._trades_since_recal = 0

            sign_str = "✅ WIN" if actual_return_pct > 0 else "❌ LOSS"
            logger.info(
                f"🔄 IC feedback: {symbol} {sign_str} {actual_return_pct:+.2%} "
                f"| {len(updated)} signals updated"
            )
        except Exception as e:
            logger.warning(f"IC feedback failed for {symbol}: {e}")

    def manage_open_positions(self, quotes: Dict) -> List[Dict]:
        """Check all stops and close positions that hit targets."""
        exits  = self.risk.check_stop_losses(quotes)
        closed = []
        for exit_info in exits:
            symbol = exit_info["symbol"]
            price  = exit_info["price"]
            reason = exit_info["reason"]
            logger.info(f"🚨 {reason} hit for {symbol} at ${price:.2f}")
            result = self.execute_exit(symbol, price, reason)
            closed.append({"symbol": symbol, "reason": reason, "result": result})
        return closed

    def emergency_close_all(self):
        """Nuclear option — close everything and cancel all orders."""
        logger.critical("🚨 EMERGENCY CLOSE ALL POSITIONS")
        self.broker.cancel_all_orders()
        for symbol in list(self.risk.open_positions.keys()):
            self.broker.close_position(symbol)
        self.risk.open_positions.clear()
        logger.critical("✅ Emergency close complete")


# ═════════════════════════════════════════════════════════════════════════════
# ALMGREN-CHRISS OPTIMAL EXECUTION ENGINE  (Step 12)
# ═════════════════════════════════════════════════════════════════════════════
#
# Almgren & Chriss (2000): "Optimal execution of portfolio transactions"
# Journal of Risk — the seminal paper on execution cost minimization.
#
# Problem: Given that we need to trade X shares over T periods, what is the
# optimal trajectory x(t) that minimizes:
#
#   Total Cost = Market Impact Cost + Timing Risk
#
# Market Impact has two components:
#   1. Permanent impact: g(v) = γ · v  (permanently moves the price)
#      Each share traded permanently shifts the price by γ per unit
#   2. Temporary impact: h(v) = η · v  (bid-ask + immediate liquidity cost)
#      Only affects the trade being executed, not future prices
#
# Timing Risk: σ²·τ·x² — the risk of price moving against us while we wait
#   σ = daily volatility, τ = time per period, x = remaining shares
#
# The risk-aversion parameter λ controls the tradeoff:
#   λ → 0:  minimize impact only (VWAP-like, slow execution)
#   λ → ∞:  minimize timing risk (execute immediately)
#
# Optimal trajectory (closed-form solution):
#   x_j = X · sinh(κ(T-j·τ)) / sinh(κ·T)
#   where κ² = λ·σ² / (η·(1 + γ·T/(2η)))
#
#   This gives a hyperbolic decay: trade faster at the start and end,
#   slower in the middle — which minimizes the total cost function.
#
# Parameters estimated from market data:
#   η (temporary impact):  calibrated from bid-ask spread + typical slippage
#   γ (permanent impact):  calibrated from order-book depth and ADV
#   σ (volatility):        realized daily vol from price history
#   λ (risk aversion):     set by trader's urgency (0.001 low, 0.1 high)
#
# Extensions implemented:
#   - Volume-participation schedule (align with intraday volume curve)
#   - Implementation Shortfall tracking (IS = arrival price - VWAP)
#   - Urgency levels: LOW / MEDIUM / HIGH / URGENT
#   - Cost estimation: expected IS in bps before execution
#
# Reference:
#   Almgren & Chriss (2000), Almgren (2003) — "Optimal trading in dark pools"
#   Kissell & Glantz (2003) — "Optimal Trading Strategies"
# ═════════════════════════════════════════════════════════════════════════════

import math as _math

# Intraday volume curve (fraction of daily volume traded per 30-min bucket)
# Based on typical U-shaped volume pattern for US equities
# 13 buckets: 9:30–10:00, 10:00–10:30, ..., 15:30–16:00
_INTRADAY_VOL_CURVE = np.array([
    0.130,   # 9:30–10:00   (opening rush)
    0.085,   # 10:00–10:30
    0.070,   # 10:30–11:00
    0.060,   # 11:00–11:30
    0.055,   # 11:30–12:00
    0.050,   # 12:00–12:30  (midday lull)
    0.050,   # 12:30–13:00
    0.055,   # 13:00–13:30
    0.060,   # 13:30–14:00
    0.065,   # 14:00–14:30
    0.075,   # 14:30–15:00
    0.095,   # 15:00–15:30
    0.150,   # 15:30–16:00  (closing rush)
], dtype=float)
_INTRADAY_VOL_CURVE /= _INTRADAY_VOL_CURVE.sum()   # normalize to sum=1

# Urgency → λ (risk-aversion) mapping
_URGENCY_LAMBDA = {
    "LOW":    1e-4,    # 30+ periods, prioritize cost
    "MEDIUM": 5e-4,    # ~15 periods, balanced
    "HIGH":   2e-3,    # ~8 periods, lean toward speed
    "URGENT": 1e-2,    # 3–5 periods, minimize timing risk
}

# Default parameter estimates (calibrated for US large-cap)
_DEFAULT_ETA   = 2.5e-7    # temporary impact coefficient (≈ 2.5bps per 1% ADV)
_DEFAULT_GAMMA = 1.0e-7    # permanent impact coefficient
_DEFAULT_SIGMA = 0.015     # daily vol fallback if not provided (1.5%)


class AlmgrenChrissCostModel:
    """
    Almgren-Chriss (2000) market impact cost model.

    Estimates market impact parameters from observable market data and
    provides the closed-form optimal execution trajectory.

    All quantities are in "shares" and "price per share" units.
    """

    def __init__(self,
                 eta:   float = _DEFAULT_ETA,
                 gamma: float = _DEFAULT_GAMMA,
                 sigma: float = _DEFAULT_SIGMA):
        """
        Args:
            eta:   temporary impact coeff  [price/share per share/day]
            gamma: permanent impact coeff  [price/share per share/day]
            sigma: daily price volatility   [fraction, e.g. 0.015 = 1.5%]
        """
        self.eta   = eta
        self.gamma = gamma
        self.sigma = sigma

    @classmethod
    def calibrate(cls, price: float, adv_shares: float,
                  bid_ask_spread_pct: float = 0.0005,
                  daily_vol: float = None,
                  price_history: np.ndarray = None) -> "AlmgrenChrissCostModel":
        """
        Calibrate η, γ, σ from market observables.

        η (temporary impact):
          Estimated as half the bid-ask spread plus additional slippage
          for executing beyond ~0.5% of ADV.
          η ≈ spread / (2 · ADV_shares)  +  impact_factor

        γ (permanent impact):
          Estimated as ~0.5 · η (permanent impact is typically 50% of temporary
          for liquid large-cap; less for small-cap).

        σ (volatility):
          From rolling 20-day realized volatility if price history provided,
          else from supplied daily_vol or default.

        Args:
            price:              current stock price
            adv_shares:         average daily volume in shares
            bid_ask_spread_pct: bid-ask spread as fraction of price (e.g. 0.0005)
            daily_vol:          daily volatility (optional override)
            price_history:      recent closing prices for vol estimation
        """
        # ── Temporary impact η ────────────────────────────────────────────
        # η such that trading 1% of ADV moves price by ~1 bp above spread
        spread_cost   = bid_ask_spread_pct * price / 2
        impact_factor = price * 0.0001  # 1bp per unit for base calibration
        if adv_shares > 0:
            eta = (spread_cost + impact_factor) / max(adv_shares, 1)
        else:
            eta = _DEFAULT_ETA

        # ── Permanent impact γ ────────────────────────────────────────────
        # Permanent ≈ 50% of temporary for large-cap, 75% for small-cap
        gamma = eta * 0.50

        # ── Volatility σ ─────────────────────────────────────────────────
        if price_history is not None and len(price_history) >= 10:
            rets  = np.diff(np.log(np.array(price_history, dtype=float) + 1e-10))
            sigma = float(np.std(rets[-20:]) if len(rets) >= 20 else np.std(rets))
        elif daily_vol is not None:
            sigma = float(daily_vol)
        else:
            sigma = _DEFAULT_SIGMA

        return cls(eta=eta, gamma=gamma, sigma=max(sigma, 0.005))

    def optimal_trajectory(self, total_shares: int, n_periods: int,
                           lam: float = _URGENCY_LAMBDA["MEDIUM"],
                           period_minutes: float = 30.0) -> Dict:
        """
        Compute Almgren-Chriss optimal execution trajectory.

        Returns the number of shares to trade in each period to minimize:
            E[cost] + λ · Var[cost]

        The closed-form solution uses a hyperbolic sine schedule:
            x_j = X · sinh(κ(T - j·τ)) / sinh(κ·T)

        Args:
            total_shares:   total number of shares to execute (integer)
            n_periods:      number of time periods to spread execution over
            lam:            risk-aversion parameter (see _URGENCY_LAMBDA)
            period_minutes: minutes per period (default 30min)

        Returns:
            trajectory:     list of shares per period
            expected_cost:  expected implementation shortfall in bps
            cost_variance:  variance of implementation shortfall
            kappa:          decay parameter (higher = front-loaded)
            half_life_periods: how quickly the schedule front-loads
        """
        X     = float(abs(total_shares))
        T     = float(n_periods)
        tau   = period_minutes / 390.0   # period as fraction of trading day

        if X < 1 or T < 1:
            return self._trivial_trajectory(total_shares, n_periods)

        # κ² = λσ² / (η · (1 + γT/(2η)))  — Almgren-Chriss (2000) eq. (20)
        eta_eff = self.eta * (1 + self.gamma * T / (2 * self.eta + 1e-20))
        kappa_sq = (lam * self.sigma**2) / (eta_eff + 1e-20)
        kappa    = float(_math.sqrt(max(kappa_sq, 1e-12)))

        # Optimal trajectory: x_j = X · sinh(κ(T-j·τ)) / sinh(κ·T)
        sinh_kT = _math.sinh(kappa * T)
        if abs(sinh_kT) < 1e-10:
            # κ → 0: uniform TWAP (limit case)
            trajectory = [int(round(X / T))] * int(T)
        else:
            raw = []
            for j in range(int(T)):
                x_j = X * _math.sinh(kappa * (T - j * tau)) / sinh_kT
                raw.append(max(0.0, x_j))

            # Normalize to sum exactly to total_shares
            raw_sum = sum(raw) + 1e-10
            scaled  = [r * X / raw_sum for r in raw]
            trajectory = []
            remainder  = 0.0
            for s in scaled:
                alloc = s + remainder
                shares_this = int(alloc)
                remainder   = alloc - shares_this
                trajectory.append(shares_this)
            # Assign any remainder to first period
            trajectory[0] += int(round(remainder))

        # Ensure non-negative and cap each period
        trajectory = [max(0, t) for t in trajectory]
        # Actual sum might be off by 1 due to rounding — correct last period
        diff = int(total_shares) - sum(trajectory)
        if trajectory:
            trajectory[-1] = max(0, trajectory[-1] + diff)

        # ── Expected cost (implementation shortfall) ──────────────────────
        # E[IS] = γ/2 · X² + η · Σ v_j²  where v_j = shares_j / tau
        expected_is_shares = (self.gamma / 2) * X**2
        for sh in trajectory:
            v_j = sh / (tau + 1e-10)
            expected_is_shares += self.eta * v_j**2 * tau

        # Convert to basis points relative to total trade notional
        # notional ≈ price * X  →  we don't know price here, return in "shares² units"
        # Caller converts using price
        expected_cost_normalized = expected_is_shares / (X + 1e-10)

        # ── Variance of IS ────────────────────────────────────────────────
        # Var[IS] = σ² · τ · Σ x_remaining_j²
        remaining = float(X)
        var_is    = 0.0
        for sh in trajectory:
            var_is += self.sigma**2 * tau * remaining**2
            remaining = max(0.0, remaining - sh)

        # Half-life: when 50% of shares have been traded
        cumulative = 0.0
        half_life  = float(len(trajectory))
        for i, sh in enumerate(trajectory):
            cumulative += sh
            if cumulative >= X * 0.50:
                half_life = float(i + 1)
                break

        return {
            "trajectory":              trajectory,
            "n_periods":               len(trajectory),
            "total_shares":            sum(trajectory),
            "expected_cost_per_share": round(expected_cost_normalized, 6),
            "cost_variance":           round(var_is, 8),
            "kappa":                   round(kappa, 6),
            "half_life_periods":       round(half_life, 2),
            "front_load_pct":          round(trajectory[0] / (X + 1e-10) * 100, 1),
        }

    def _trivial_trajectory(self, total_shares: int, n_periods: int) -> Dict:
        """Fallback: uniform schedule."""
        n   = max(1, n_periods)
        q   = total_shares // n
        rem = total_shares % n
        traj = [q] * n
        traj[0] += rem
        return {"trajectory": traj, "n_periods": n,
                "total_shares": sum(traj), "kappa": 0.0,
                "expected_cost_per_share": 0.0, "cost_variance": 0.0,
                "half_life_periods": n / 2.0, "front_load_pct": 100.0 / n}

    def estimate_cost_bps(self, total_shares: int, price: float,
                          n_periods: int, lam: float) -> Dict:
        """
        Estimate total execution cost in basis points before trading.

        Returns expected IS (bps), variance of IS, and comparison to
        VWAP and immediate execution benchmarks.
        """
        traj_result = self.optimal_trajectory(total_shares, n_periods, lam)
        cost_per_share = float(traj_result.get("expected_cost_per_share", 0))

        notional = price * total_shares + 1e-10
        cost_bps = (cost_per_share * total_shares * price / notional) * 10_000

        # Immediate execution cost (no slicing): η · (X/1)² · 1 + γ/2 · X²
        immed_cost = self.eta * total_shares**2 + (self.gamma / 2) * total_shares**2
        immed_bps  = (immed_cost / notional) * 10_000

        # Pure TWAP cost (uniform): same formula but n_periods slices
        twap_cost_per = self.eta * (total_shares / n_periods)**2 * n_periods
        twap_bps = (twap_cost_per / notional) * 10_000

        return {
            "optimal_bps":    round(cost_bps, 2),
            "immediate_bps":  round(immed_bps, 2),
            "twap_bps":       round(twap_bps, 2),
            "saving_vs_immed": round(immed_bps - cost_bps, 2),
            "saving_vs_twap":  round(twap_bps - cost_bps, 2),
            "n_periods":       n_periods,
            "total_shares":    total_shares,
            "notional":        round(notional, 2),
        }


class AlmgrenChrissSched:
    """
    High-level schedule builder for the ExecutionEngine.

    Determines:
      1. Urgency level from signal conviction + volatility regime
      2. Optimal n_periods based on order size relative to ADV
      3. Per-period share counts (Almgren-Chriss trajectory)
      4. Volume-participation alignment (map periods to intraday vol curve)
      5. Expected cost estimate in bps

    The result is an "execution plan" dict that can be passed to AlpacaBroker
    for staged order submission.
    """

    @staticmethod
    def build_plan(symbol:         str,
                   total_shares:   int,
                   side:           str,          # "buy" or "sell"
                   price:          float,
                   adv_shares:     float,
                   daily_vol:      float,
                   conviction:     float = 0.5,  # AladdinScorer conviction [0,1]
                   vix:            float = 20.0,
                   price_history:  np.ndarray = None,
                   bid_ask_pct:    float = 0.0005,
                   ) -> Dict:
        """
        Build a full Almgren-Chriss execution plan.

        Args:
            symbol:         ticker symbol
            total_shares:   target shares to trade
            side:           "buy" or "sell"
            price:          current mid price
            adv_shares:     average daily volume (shares)
            daily_vol:      realized daily volatility
            conviction:     signal conviction 0–1 (higher = trade faster)
            vix:            current VIX (higher = trade faster)
            price_history:  recent closes for vol calibration
            bid_ask_pct:    spread as fraction of price

        Returns execution plan with:
            urgency, trajectory, cost_estimate, participation_rate, n_periods
        """
        # ── Determine urgency ─────────────────────────────────────────────
        # High conviction + high vol → URGENT (don't wait, price might move)
        # Low conviction + low vol  → LOW (take time, save impact cost)
        urgency_score = conviction * 0.6 + min(vix / 40.0, 1.0) * 0.4
        if   urgency_score > 0.75: urgency = "URGENT"
        elif urgency_score > 0.55: urgency = "HIGH"
        elif urgency_score > 0.35: urgency = "MEDIUM"
        else:                       urgency = "LOW"
        lam = _URGENCY_LAMBDA[urgency]

        # ── Determine n_periods ───────────────────────────────────────────
        # Participation rate: fraction of ADV to trade per day
        # Rule of thumb: keep each slice < 5–10% of intraday volume
        # to avoid moving the market.
        participation = total_shares / max(adv_shares, 1)

        if   participation < 0.005:  n_periods = 2    # < 0.5% ADV: 1 hour
        elif participation < 0.02:   n_periods = 4    # < 2% ADV: 2 hours
        elif participation < 0.05:   n_periods = 7    # < 5% ADV: 3.5 hours
        elif participation < 0.10:   n_periods = 10   # < 10% ADV: full day
        else:                         n_periods = 13   # > 10% ADV: multi-day caution

        # Override for urgency: compress schedule if URGENT
        if urgency == "URGENT":
            n_periods = max(2, n_periods // 3)
        elif urgency == "LOW":
            n_periods = min(13, n_periods + 3)

        # ── Calibrate cost model ──────────────────────────────────────────
        model = AlmgrenChrissCostModel.calibrate(
            price=price,
            adv_shares=adv_shares,
            bid_ask_spread_pct=bid_ask_pct,
            daily_vol=daily_vol,
            price_history=price_history,
        )

        # ── Build optimal trajectory ──────────────────────────────────────
        traj_result  = model.optimal_trajectory(total_shares, n_periods, lam)
        trajectory   = traj_result["trajectory"]

        # ── Volume-participation alignment ────────────────────────────────
        # Map trajectory periods to intraday buckets to match natural liquidity
        # This ensures we trade proportionally MORE when volume is high (open/close)
        n_traj = len(trajectory)
        if n_traj <= len(_INTRADAY_VOL_CURVE):
            # Remap trajectory to volume-weighted periods
            vol_weights = _INTRADAY_VOL_CURVE[:n_traj]
            vol_weights = vol_weights / vol_weights.sum()
            vol_weighted = [int(round(total_shares * w)) for w in vol_weights]
            # Blend 50/50: AC optimal vs volume-participation
            blended = []
            for ac, vw in zip(trajectory, vol_weighted):
                blended.append(int(round((ac + vw) / 2)))
            # Fix rounding
            diff = total_shares - sum(blended)
            blended[0] += diff
            trajectory = [max(0, b) for b in blended]
        else:
            # More periods than buckets — use pure AC
            pass

        # ── Cost estimate ─────────────────────────────────────────────────
        cost_est = model.estimate_cost_bps(total_shares, price, n_periods, lam)

        return {
            "symbol":              symbol,
            "side":                side,
            "total_shares":        total_shares,
            "urgency":             urgency,
            "lam":                 round(lam, 6),
            "n_periods":           len(trajectory),
            "trajectory":          trajectory,
            "kappa":               round(traj_result.get("kappa", 0), 6),
            "half_life_periods":   traj_result.get("half_life_periods", len(trajectory)/2),
            "front_load_pct":      round(trajectory[0] / (total_shares + 1e-10) * 100, 1),
            "participation_rate":  round(participation * 100, 3),
            "cost_estimate":       cost_est,
            "expected_cost_bps":   cost_est.get("optimal_bps", 0.0),
            "model_eta":           round(model.eta, 9),
            "model_gamma":         round(model.gamma, 9),
            "model_sigma":         round(model.sigma, 5),
            "label": (
                f"AC-Sched: {symbol} {side.upper()} {total_shares}sh "
                f"| {urgency} | {len(trajectory)} periods "
                f"| {cost_est.get('optimal_bps', 0):.1f}bps est. IS"
            ),
        }

    @staticmethod
    def track_implementation_shortfall(plan: Dict,
                                        arrival_price: float,
                                        fills: List[Dict]) -> Dict:
        """
        Compute realized implementation shortfall vs the AC plan.

        IS = Σ (fill_price - arrival_price) * shares  /  total_notional

        Args:
            plan:           execution plan from build_plan()
            arrival_price:  mid-price at decision time (before first fill)
            fills:          list of {"price": float, "shares": int}

        Returns:
            realized_is_bps: actual IS in basis points
            vs_plan_bps:     realized minus expected IS (slippage vs plan)
            fill_quality:    "EXCELLENT" / "GOOD" / "FAIR" / "POOR"
        """
        if not fills:
            return {"realized_is_bps": 0.0, "vs_plan_bps": 0.0,
                    "fill_quality": "NO_FILLS"}

        total_notional  = sum(f["price"] * f["shares"] for f in fills)
        total_shares    = sum(f["shares"] for f in fills) + 1e-10
        vwap_price      = total_notional / total_shares
        side            = plan.get("side", "buy").lower()

        if side == "buy":
            realized_is = (vwap_price - arrival_price) / (arrival_price + 1e-10) * 10_000
        else:
            realized_is = (arrival_price - vwap_price) / (arrival_price + 1e-10) * 10_000

        expected_bps = float(plan.get("expected_cost_bps", 0))
        vs_plan      = realized_is - expected_bps

        if   realized_is < expected_bps * 0.50: quality = "EXCELLENT"
        elif realized_is < expected_bps * 1.00: quality = "GOOD"
        elif realized_is < expected_bps * 1.50: quality = "FAIR"
        else:                                    quality = "POOR"

        return {
            "arrival_price":   arrival_price,
            "vwap_price":      round(vwap_price, 4),
            "realized_is_bps": round(realized_is, 2),
            "expected_is_bps": round(expected_bps, 2),
            "vs_plan_bps":     round(vs_plan, 2),
            "fill_quality":    quality,
            "n_fills":         len(fills),
        }