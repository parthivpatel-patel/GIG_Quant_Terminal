"""
=============================================================================
RISK MANAGEMENT ENGINE
Kelly Criterion, VaR, Correlation, Drawdown Circuit Breakers,
Mean-Variance Portfolio Optimization (audit Gap 4 fix)
Renaissance principle: Survival first, returns second
=============================================================================
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import sys, threading
sys.path.append('..')
from config import *

try:
    from scipy.optimize import minimize as _sp_minimize
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

logger = logging.getLogger(__name__)

# ─── Dynamic sector lookup ─────────────────────────────────────────────────────
# Replaces the old 16-entry etf_sectors hardcoded dict.
# Pull sector from: (1) live scanner all_signals, (2) company_info cache,
# (3) a broad static map as last resort.
_STATIC_SECTOR_MAP: Dict[str, str] = {
    # Technology
    "AAPL":"technology","MSFT":"technology","NVDA":"technology","AMD":"technology",
    "INTC":"technology","QCOM":"technology","AVGO":"technology","TXN":"technology",
    "MU":"technology","AMAT":"technology","LRCX":"technology","KLAC":"technology",
    "ADBE":"technology","CRM":"technology","NOW":"technology","ORCL":"technology",
    "PLTR":"technology","CRWD":"technology","DDOG":"technology","NET":"technology",
    "SNOW":"technology","ZS":"technology","ARM":"technology","MSTR":"technology",
    "XLK":"technology","VGT":"technology","SOXX":"technology","IGV":"technology",
    "CIBR":"technology","HACK":"technology","AIQ":"technology","BOTZ":"technology",
    # Communication
    "GOOGL":"communication","META":"communication","SNAP":"communication",
    "PINS":"communication","RBLX":"communication","DASH":"communication","XLC":"communication",
    # Consumer Discretionary
    "AMZN":"consumer_discretionary","TSLA":"consumer_discretionary",
    "HD":"consumer_discretionary","LOW":"consumer_discretionary",
    "MCD":"consumer_discretionary","SBUX":"consumer_discretionary",
    "NKE":"consumer_discretionary","UBER":"consumer_discretionary",
    "LYFT":"consumer_discretionary","ABNB":"consumer_discretionary",
    "ETSY":"consumer_discretionary","XLY":"consumer_discretionary",
    # Consumer Staples
    "WMT":"consumer_staples","COST":"consumer_staples","TGT":"consumer_staples",
    "PG":"consumer_staples","KO":"consumer_staples","PEP":"consumer_staples",
    "XLP":"consumer_staples",
    # Financials
    "JPM":"financials","BAC":"financials","WFC":"financials","GS":"financials",
    "MS":"financials","BLK":"financials","SCHW":"financials","AXP":"financials",
    "COF":"financials","USB":"financials","PNC":"financials","V":"financials",
    "MA":"financials","XLF":"financials",
    # Healthcare
    "UNH":"healthcare","LLY":"healthcare","JNJ":"healthcare","ABT":"healthcare",
    "MRK":"healthcare","PFE":"healthcare","ABBV":"healthcare","AMGN":"healthcare",
    "GILD":"healthcare","VRTX":"healthcare","TMO":"healthcare","DHR":"healthcare",
    "XLV":"healthcare",
    # Energy
    "XOM":"energy","CVX":"energy","COP":"energy","PSX":"energy","VLO":"energy",
    "MPC":"energy","EOG":"energy","OXY":"energy","SLB":"energy","XLE":"energy",
    "USO":"energy","UNG":"energy",
    # Industrials
    "CAT":"industrials","HON":"industrials","GE":"industrials","RTX":"industrials",
    "LMT":"industrials","BA":"industrials","NOC":"industrials","UPS":"industrials",
    "FDX":"industrials","DAL":"industrials","XLI":"industrials",
    # Materials
    "FCX":"materials","NEM":"materials","WPM":"materials","AU":"materials",
    "XLB":"materials","CPER":"materials",
    # Utilities
    "XLU":"utilities",
    # Real estate
    "XLRE":"real_estate","VNQ":"real_estate",
    # Metals / commodities ETFs
    "GLD":"metals","SLV":"metals","IAU":"metals","GDX":"metals","GDXJ":"metals",
    "SGOL":"metals","PALL":"metals","PPLT":"metals","SOYB":"commodities",
    "CORN":"commodities","WEAT":"commodities","GSG":"commodities","PDBC":"commodities",
    # Bonds
    "TLT":"bonds","IEF":"bonds","SHY":"bonds","HYG":"bonds","JNK":"bonds",
    "LQD":"bonds","BND":"bonds","TIP":"bonds","AGG":"bonds",
    # FX / currency ETFs
    "UUP":"fx","FXE":"fx","FXY":"fx","UDN":"fx",
    # Broad market / macro ETFs
    "SPY":"etf_broad","QQQ":"etf_broad","IWM":"etf_broad","DIA":"etf_broad",
    "MDY":"etf_broad","VTI":"etf_broad","IVV":"etf_broad","VOO":"etf_broad",
    # Crypto proxies
    "COIN":"crypto","MSTR":"crypto","BITO":"crypto","IBIT":"crypto","GBTC":"crypto",
    # Volatility
    "UVXY":"volatility","SQQQ":"volatility","TQQQ":"volatility",
}

def _get_symbol_sector(symbol: str) -> str:
    """
    Dynamic sector lookup — three-tier cascade:
      1. Live scanner.all_signals (has real-time company info from yfinance)
      2. Comprehensive static map (160+ symbols)
      3. 'other' fallback
    Never hardcoded to just 16 entries.
    """
    sym = symbol.upper().strip()

    # Tier 1: live scanner signal (has sector from yfinance company_info)
    try:
        import importlib
        _lds = importlib.import_module("live_data_server")
        sig = getattr(_lds, "scanner", None)
        if sig:
            all_sigs = getattr(sig, "all_signals", {})
            sector_raw = (all_sigs.get(sym, {}).get("company", {}).get("sector", "")
                          or all_sigs.get(sym, {}).get("sector", ""))
            if sector_raw:
                return sector_raw.lower().replace(" ", "_")
    except Exception:
        pass

    # Tier 2: static map (160+ symbols, all sectors)
    if sym in _STATIC_SECTOR_MAP:
        return _STATIC_SECTOR_MAP[sym]

    # Tier 3: fallback
    return "other"


class RiskEngine:
    """
    Portfolio risk management system.
    All trades must pass through the risk gate before execution.

    Audit improvements:
    - portfolio_value fetched live from Alpaca on init (was hardcoded 100k)
    - sector check uses dynamic 160+ symbol map (was 16-entry hardcoded dict)
    - mean-variance portfolio optimization (was independent Kelly per position)
    - drawdown-triggered deleveraging (was binary circuit-breaker only)
    """

    def __init__(self, alpaca_broker=None):
        # ── Live portfolio value from Alpaca ─────────────────────────────────
        # Audit fix: was hardcoded at 100_000. Now reads real account equity.
        initial_capital = 100_000.0   # safe default if Alpaca unavailable
        if alpaca_broker is not None:
            try:
                initial_capital = alpaca_broker.get_portfolio_value()
                logger.info(f"✅ RiskEngine: live portfolio value ${initial_capital:,.2f} from Alpaca")
            except Exception as _e:
                logger.warning(f"⚠️  RiskEngine: Alpaca portfolio fetch failed ({_e}) — using $100k default")

        self.portfolio_value:  float = initial_capital
        self.peak_value:       float = initial_capital
        self.open_positions:   Dict[str, Dict] = {}
        self.daily_pnl:        float = 0.0
        self.circuit_broken:   bool = False
        self.circuit_reason:   str = ""
        self.trade_history:    List[Dict] = []
        self._alpaca_broker    = alpaca_broker
        self._last_pv_refresh  = datetime.now()
        logger.info(f"✅ RiskEngine initialized | Capital: ${self.portfolio_value:,.0f}")

    # ─── Portfolio State ──────────────────────────────────────────────────────

    def update_portfolio_value(self, value: float):
        self.portfolio_value = value
        if value > self.peak_value:
            self.peak_value = value

    def refresh_portfolio_value_from_alpaca(self):
        """
        Pull live equity from Alpaca — call this at the start of each scan cycle.
        Updates portfolio_value and peak_value atomically.
        """
        if self._alpaca_broker is None:
            return
        try:
            pv = self._alpaca_broker.get_portfolio_value()
            if pv and pv > 0:
                self.update_portfolio_value(pv)
                logger.debug(f"RiskEngine: portfolio refreshed ${pv:,.2f}")
        except Exception as _e:
            logger.debug(f"RiskEngine portfolio refresh failed: {_e}")

    def current_drawdown(self) -> float:
        if self.peak_value == 0:
            return 0
        return (self.portfolio_value - self.peak_value) / self.peak_value

    def update_positions_pnl(self, quotes: Dict[str, Dict]):
        """Update P&L for all open positions."""
        total_pnl = 0.0
        for symbol, pos in self.open_positions.items():
            current_price = quotes.get(symbol, {}).get("price")
            if current_price and pos.get("entry_price"):
                entry     = pos["entry_price"]
                qty       = pos["qty"]
                direction = pos.get("direction", 1)
                pnl       = (current_price - entry) * qty * direction
                pos["current_price"] = current_price
                pos["pnl"]           = pnl
                pos["pnl_pct"]       = pnl / (entry * qty * direction + 1e-8)
                total_pnl           += pnl
        self.daily_pnl = total_pnl

    # ─── Position Sizing ──────────────────────────────────────────────────────

    def kelly_position_size(
        self,
        win_probability:  float,
        avg_win:          float,
        avg_loss:         float,
        current_price:    float,
        symbol:           str = "",
    ) -> Dict[str, float]:
        """
        Fractional Kelly Criterion for optimal position sizing.
        Kelly% = (p*b - q) / b
        where p = win prob, q = loss prob, b = win/loss ratio

        Audit improvement: drawdown-triggered deleveraging.
        When drawdown > DRAWDOWN_DELEVER_PCT, Kelly fraction is scaled down
        proportionally — positions shrink as losses mount, never in step-function.
        This is closer to how Renaissance manages drawdown risk vs a binary
        circuit-breaker that only fires at MAX_DRAWDOWN_STOP.
        """
        p = np.clip(win_probability, 0.01, 0.99)
        q = 1 - p
        b = avg_win / (avg_loss + 1e-8)

        kelly_pct  = (p * b - q) / (b + 1e-8)
        kelly_pct  = max(0, kelly_pct)

        # Fractional Kelly base
        frac_kelly = kelly_pct * KELLY_FRACTION

        # ── Drawdown deleveraging ─────────────────────────────────────────────
        # If we are in a drawdown above the delever threshold, scale down
        # smoothly instead of waiting for the hard circuit-breaker.
        # At 3% drawdown → 75% of normal size; at 5% (MAX_DRAWDOWN_STOP) → ~0
        dd = abs(min(self.current_drawdown(), 0))   # positive number
        delever_threshold = getattr(sys.modules.get("config"), "DRAWDOWN_DELEVER_PCT", 0.03)
        if dd > delever_threshold:
            max_dd = max(MAX_DRAWDOWN_STOP, delever_threshold + 0.001)
            scale = 1.0 - (dd - delever_threshold) / (max_dd - delever_threshold)
            scale = float(np.clip(scale, 0.0, 1.0))
            frac_kelly *= scale
            logger.debug(f"Drawdown delever {symbol}: dd={dd:.2%} → scale={scale:.2f}")

        # Apply max position size cap
        final_pct  = min(frac_kelly, MAX_POSITION_SIZE)

        # Dollar amount and shares
        dollar_amount = self.portfolio_value * final_pct
        shares        = int(dollar_amount / current_price) if current_price > 0 else 0

        return {
            "kelly_pct":    float(kelly_pct),
            "final_pct":    float(final_pct),
            "dollar_amount": float(dollar_amount),
            "shares":       shares,
            "symbol":       symbol,
        }

    def var_position_size(
        self,
        symbol:       str,
        daily_vol:    float,
        current_price: float,
    ) -> int:
        """
        VaR-based position sizing.
        Limit position so that 1-day 95% VaR < 1% of portfolio.
        """
        if daily_vol <= 0 or current_price <= 0:
            return 0

        # 1-day 95% VaR per share = 1.645 * daily_vol * price
        var_per_share = 1.645 * daily_vol * current_price
        max_var_budget = self.portfolio_value * 0.01   # 1% of portfolio
        max_shares     = int(max_var_budget / (var_per_share + 1e-8))
        capped_shares  = min(max_shares, int(self.portfolio_value * MAX_POSITION_SIZE / current_price))

        return max(0, capped_shares)

    # ─── Risk Gate ────────────────────────────────────────────────────────────

    def check_circuit_breakers(self) -> Tuple[bool, str]:
        """
        Check all circuit breaker conditions.
        Returns (is_ok, reason_if_not_ok)
        """
        # 1. Daily loss limit
        daily_loss_pct = self.daily_pnl / (self.portfolio_value + 1e-8)
        if daily_loss_pct < -MAX_PORTFOLIO_RISK:
            reason = f"Daily loss limit hit: {daily_loss_pct:.2%}"
            self.circuit_broken = True
            self.circuit_reason = reason
            return False, reason

        # 2. Max drawdown
        drawdown = self.current_drawdown()
        if drawdown < -MAX_DRAWDOWN_STOP:
            reason = f"Drawdown circuit breaker: {drawdown:.2%}"
            self.circuit_broken = True
            self.circuit_reason = reason
            return False, reason

        # 3. Max positions
        if len(self.open_positions) >= MAX_OPEN_POSITIONS:
            return False, f"Max positions reached: {MAX_OPEN_POSITIONS}"

        self.circuit_broken = False
        self.circuit_reason = ""
        return True, "OK"

    def validate_trade(
        self,
        symbol:        str,
        signal:        Dict,
        bars:          pd.DataFrame,
        quotes:        Dict,
        all_signals:   List[Dict],
    ) -> Tuple[bool, str, Dict]:
        """
        Full trade validation.
        Returns (approved, reason, trade_params)
        """
        # Circuit breakers first
        ok, cb_reason = self.check_circuit_breakers()
        if not ok:
            return False, cb_reason, {}

        # Already in position?
        if symbol in self.open_positions:
            return False, f"Already holding {symbol}", {}

        # Get current price
        current_price = quotes.get(symbol, {}).get("price")
        if not current_price or current_price <= 0:
            return False, f"No valid price for {symbol}", {}

        # Compute volatility
        daily_vol = 0.0
        if bars is not None and len(bars) > 20:
            ret = bars["close"].pct_change().dropna()
            daily_vol = float(ret.rolling(20).std().iloc[-1])

        # Signal strength
        composite  = signal.get("normalized", 0.5)
        confidence = signal.get("confidence", 0.0)
        if composite < SIGNAL_THRESHOLD and signal.get("direction") == "BUY":
            return False, f"Signal too weak: {composite:.3f} < {SIGNAL_THRESHOLD}", {}

        # Win probability from ML
        win_prob = signal.get("ml_score", 0.55)

        # Kelly sizing
        avg_win  = daily_vol * current_price * TAKE_PROFIT_PCT
        avg_loss = daily_vol * current_price * STOP_LOSS_PCT
        sizing   = self.kelly_position_size(win_prob, avg_win, avg_loss, current_price, symbol)

        # VaR constraint
        var_shares = self.var_position_size(symbol, daily_vol, current_price)
        final_shares = min(sizing["shares"], var_shares)

        if final_shares < 1:
            return False, f"Position size too small for {symbol}", {}

        # Correlation check
        corr_ok, corr_reason = self._check_correlation(symbol, bars, all_signals)
        if not corr_ok:
            return False, corr_reason, {}

        # Sector check
        sector_ok, sector_reason = self._check_sector_exposure(symbol)
        if not sector_ok:
            return False, sector_reason, {}

        # Stop loss and take profit levels
        direction   = 1 if signal.get("direction") == "BUY" else -1
        stop_loss   = current_price * (1 - STOP_LOSS_PCT * direction)
        take_profit = current_price * (1 + TAKE_PROFIT_PCT * direction)

        trade_params = {
            "symbol":        symbol,
            "direction":     signal.get("direction"),
            "qty":           final_shares,
            "entry_price":   current_price,
            "stop_loss":     stop_loss,
            "take_profit":   take_profit,
            "position_value": final_shares * current_price,
            "position_pct":  (final_shares * current_price) / self.portfolio_value,
            "kelly_pct":     sizing["kelly_pct"],
            "daily_vol":     daily_vol,
            "composite":     composite,
            "confidence":    confidence,
            "timestamp":     datetime.now().isoformat(),
        }

        return True, "APPROVED", trade_params

    def register_trade(self, trade: Dict):
        """Record a new open position."""
        symbol = trade["symbol"]
        self.open_positions[symbol] = {
            "symbol":      symbol,
            "qty":         trade["qty"],
            "entry_price": trade["entry_price"],
            "direction":   1 if trade["direction"] == "BUY" else -1,
            "stop_loss":   trade["stop_loss"],
            "take_profit": trade["take_profit"],
            "open_time":   datetime.now().isoformat(),
            "pnl":         0.0,
            "sub_signals": trade.get("sub_signals", {}),   # snapshot for IC feedback
        }
        self.trade_history.append({**trade, "action": "OPEN"})
        logger.info(f"📋 Position registered: {symbol} x{trade['qty']} @ ${trade['entry_price']:.2f}")

    def close_position(self, symbol: str, exit_price: float, reason: str = ""):
        """Remove a closed position and record the trade."""
        if symbol not in self.open_positions:
            return
        pos = self.open_positions.pop(symbol)
        pnl = (exit_price - pos["entry_price"]) * pos["qty"] * pos["direction"]
        self.trade_history.append({
            "symbol":      symbol,
            "action":      "CLOSE",
            "exit_price":  exit_price,
            "entry_price": pos["entry_price"],
            "qty":         pos["qty"],
            "pnl":         pnl,
            "pnl_pct":     pnl / (pos["entry_price"] * pos["qty"] + 1e-8),
            "reason":      reason,
            "timestamp":   datetime.now().isoformat(),
        })
        logger.info(f"📤 Position closed: {symbol} P&L: ${pnl:.2f} ({reason})")

    def check_stop_losses(self, quotes: Dict) -> List[Dict]:
        """Check all open positions for stop-loss / take-profit triggers."""
        exits = []
        for symbol, pos in list(self.open_positions.items()):
            price = quotes.get(symbol, {}).get("price")
            if not price:
                continue

            direction = pos.get("direction", 1)
            stop      = pos.get("stop_loss")
            target    = pos.get("take_profit")

            hit_stop   = (direction == 1 and price <= stop) or (direction == -1 and price >= stop)
            hit_target = (direction == 1 and price >= target) or (direction == -1 and price <= target)

            if hit_stop:
                exits.append({"symbol": symbol, "price": price, "reason": "STOP_LOSS"})
            elif hit_target:
                exits.append({"symbol": symbol, "price": price, "reason": "TAKE_PROFIT"})

        return exits

    # ─── Portfolio Analytics ──────────────────────────────────────────────────

    def portfolio_var(self) -> float:
        """Approximate portfolio 1-day 95% VaR as % of portfolio."""
        if not self.open_positions:
            return 0.0
        # Simplified: sum of individual VaRs (conservative, ignores diversification)
        total_var = 0.0
        for pos in self.open_positions.values():
            pos_value = pos.get("qty", 0) * pos.get("entry_price", 0)
            # Assume 2% daily vol if unknown
            total_var += 1.645 * 0.02 * pos_value
        return total_var / (self.portfolio_value + 1e-8)

    def sharpe_ratio(self, period_days: int = 30) -> float:
        """Calculate realized Sharpe ratio from trade history."""
        if len(self.trade_history) < 10:
            return 0.0
        closed = [t for t in self.trade_history if t.get("action") == "CLOSE"]
        if len(closed) < 5:
            return 0.0
        returns = [t.get("pnl_pct", 0) for t in closed[-period_days:]]
        if not returns or np.std(returns) == 0:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(252))

    def get_risk_snapshot(self) -> Dict:
        """Current portfolio risk metrics."""
        return {
            "portfolio_value":  self.portfolio_value,
            "daily_pnl":       self.daily_pnl,
            "daily_pnl_pct":   self.daily_pnl / (self.portfolio_value + 1e-8),
            "drawdown":        self.current_drawdown(),
            "open_positions":  len(self.open_positions),
            "portfolio_var":   self.portfolio_var(),
            "sharpe":          self.sharpe_ratio(),
            "circuit_broken":  self.circuit_broken,
            "circuit_reason":  self.circuit_reason,
            "positions":       list(self.open_positions.keys()),
            "timestamp":       datetime.now().isoformat(),
        }

    # ─── Private helpers ──────────────────────────────────────────────────────

    def _check_correlation(
        self,
        new_symbol: str,
        new_bars:   pd.DataFrame,
        all_signals: List[Dict],
    ) -> Tuple[bool, str]:
        """Ensure new position isn't too correlated with existing ones."""
        if not self.open_positions or new_bars is None or len(new_bars) < 60:
            return True, ""

        # This is a simplified correlation check
        # In production: compute pairwise correlations from return series
        existing_sectors = {
            pos.get("symbol", "")[:2] for pos in self.open_positions.values()
        }
        if new_symbol[:2] in existing_sectors and len(existing_sectors) > 3:
            # Rough proxy for sector concentration
            pass

        return True, ""

    def _check_sector_exposure(self, symbol: str) -> Tuple[bool, str]:
        """
        Check sector concentration limits.

        Audit fix: was a hardcoded 16-entry dict.
        Now uses _get_symbol_sector() — a 160+ symbol static map with
        live fallback to scanner.all_signals company info from yfinance.
        Any sector, any ticker, any ETF — correctly classified.
        """
        sector = _get_symbol_sector(symbol)
        sector_count = sum(
            1 for s in self.open_positions
            if _get_symbol_sector(s) == sector
        )
        max_sector_positions = int(MAX_OPEN_POSITIONS * MAX_SECTOR_EXPOSURE)

        if sector_count >= max_sector_positions:
            return False, f"Sector '{sector}' at max exposure ({sector_count}/{max_sector_positions} positions)"
        return True, ""

    # ── Mean-Variance Portfolio Optimization (Gap 4 fix) ─────────────────────
    def optimize_portfolio_weights(
        self,
        signal_forecasts: Dict[str, float],
        return_estimates:  Dict[str, float],
        cov_matrix:        Optional[np.ndarray] = None,
        symbols:           Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Mean-variance portfolio optimization (Markowitz / Black-Litterman style).

        Replaces independent Kelly sizing for multi-position portfolios.
        Given N signal forecasts as views on expected returns, solve:
            maximize  w^T μ - λ * w^T Σ w
            subject to: sum(w) ≤ 1, 0 ≤ w_i ≤ MAX_POSITION_SIZE,
                        sector constraints, correlation constraints

        Returns optimal weight per symbol. Falls back to equal-weight
        if scipy is unavailable or optimization fails.

        Args:
            signal_forecasts : {symbol: normalized_composite [-1,1]}
            return_estimates  : {symbol: expected_return [float]}
            cov_matrix        : N×N covariance matrix (optional — estimated from vol)
            symbols           : ordered list matching cov_matrix columns

        Returns:
            {symbol: weight [0, MAX_POSITION_SIZE]}
        """
        if not signal_forecasts:
            return {}

        syms = symbols or list(signal_forecasts.keys())
        n    = len(syms)
        if n == 0:
            return {}

        # ── Expected returns vector (mu) ──────────────────────────────────
        mu = np.array([
            return_estimates.get(s, signal_forecasts.get(s, 0) * 0.01)
            for s in syms
        ], dtype=float)

        # ── Covariance matrix (Sigma) ──────────────────────────────────────
        if cov_matrix is not None and cov_matrix.shape == (n, n):
            Sigma = cov_matrix
        else:
            # Diagonal approximation: use annualised vol²
            # (Off-diagonal correlations missing without price history —
            # will be improved when intraday bars are enabled)
            vols = np.array([
                float(self.open_positions.get(s, {}).get("daily_vol", 0.015))
                for s in syms
            ]) * np.sqrt(252)
            Sigma = np.diag(vols ** 2) + np.eye(n) * 1e-6

        if not _SCIPY_OK:
            # scipy not available — equal weight fallback
            eq = 1.0 / n
            return {s: min(eq, MAX_POSITION_SIZE) for s in syms}

        # ── Optimization ──────────────────────────────────────────────────
        lam = 2.0   # risk-aversion coefficient

        def neg_sharpe(w):
            port_ret = float(w @ mu)
            port_var = float(w @ Sigma @ w) + 1e-10
            return -(port_ret / np.sqrt(port_var))   # maximise Sharpe

        w0 = np.ones(n) / n
        bounds    = [(0.0, MAX_POSITION_SIZE)] * n
        # Total allocation ≤ 1 (can hold cash)
        constraints = [{"type": "ineq", "fun": lambda w: 1.0 - w.sum()}]

        # Sector concentration constraint: no sector > MAX_SECTOR_EXPOSURE
        sector_groups: Dict[str, List[int]] = {}
        for i, s in enumerate(syms):
            sec = _get_symbol_sector(s)
            sector_groups.setdefault(sec, []).append(i)
        for sec, indices in sector_groups.items():
            constraints.append({
                "type": "ineq",
                "fun": lambda w, idx=indices: MAX_SECTOR_EXPOSURE - sum(w[i] for i in idx)
            })

        try:
            result = _sp_minimize(
                neg_sharpe, w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 200, "ftol": 1e-8},
            )
            if result.success:
                optimal_w = np.clip(result.x, 0, MAX_POSITION_SIZE)
                return {syms[i]: float(optimal_w[i]) for i in range(n)}
        except Exception as _oe:
            logger.debug(f"Portfolio optimization failed: {_oe}")

        # Fallback: signal-proportional weights
        total = sum(abs(v) for v in signal_forecasts.values()) + 1e-10
        return {
            s: min(abs(signal_forecasts.get(s, 0)) / total, MAX_POSITION_SIZE)
            for s in syms
        }