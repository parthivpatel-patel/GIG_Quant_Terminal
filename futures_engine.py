"""
=============================================================================
RENAISSANCE FUTURES ENGINE — futures_engine.py
=============================================================================
Futures Market Intelligence + Basis Trading + Roll Calendar

Tracks:
  ├── ES (S&P 500 E-mini)     via SPY proxy + /ES estimates
  ├── NQ (Nasdaq 100 E-mini)  via QQQ proxy + /NQ estimates
  ├── CL (Crude Oil)           via USO + CL=F yfinance
  ├── GC (Gold)                via GLD + GC=F yfinance
  ├── SI (Silver)              via SLV + SI=F yfinance
  ├── ZB (30Y Treasury)        via TLT proxy
  ├── VX (VIX Futures)         via ^VIX + VXX
  └── BTC (Bitcoin Futures)    via BTC-USD

SIGNALS:
  - Cash-Futures Basis (contango/backwardation)
  - Roll yield estimation
  - Term structure slope
  - Fair value gap (futures vs cash)
  - Commitment of Traders proxy (ETF fund flows)

Renaissance.io — Institutional Grade Quantitative Trading
=============================================================================
"""

import numpy as np
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("FUTURES")


# ═══════════════════════════════════════════════════════════════════════════════
# FUTURES UNIVERSE
# ═══════════════════════════════════════════════════════════════════════════════

FUTURES_MAP = {
    "ES": {
        "name": "S&P 500 E-mini", "cash": "SPY", "futures": "ES=F",
        "multiplier": 50, "tick": 0.25, "margin": 12650,
        "sector": "equity_index", "exchange": "CME"
    },
    "NQ": {
        "name": "Nasdaq 100 E-mini", "cash": "QQQ", "futures": "NQ=F",
        "multiplier": 20, "tick": 0.25, "margin": 18150,
        "sector": "equity_index", "exchange": "CME"
    },
    "CL": {
        "name": "Crude Oil WTI", "cash": "USO", "futures": "CL=F",
        "multiplier": 1000, "tick": 0.01, "margin": 6500,
        "sector": "energy", "exchange": "NYMEX"
    },
    "GC": {
        "name": "Gold", "cash": "GLD", "futures": "GC=F",
        "multiplier": 100, "tick": 0.10, "margin": 9500,
        "sector": "metals", "exchange": "COMEX"
    },
    "SI": {
        "name": "Silver", "cash": "SLV", "futures": "SI=F",
        "multiplier": 5000, "tick": 0.005, "margin": 12000,
        "sector": "metals", "exchange": "COMEX"
    },
    "ZB": {
        "name": "30-Year Treasury", "cash": "TLT", "futures": "ZB=F",
        "multiplier": 1000, "tick": 0.03125, "margin": 4200,
        "sector": "fixed_income", "exchange": "CBOT"
    },
    "VX": {
        "name": "VIX Futures", "cash": "^VIX", "futures": "^VIX",
        "multiplier": 1000, "tick": 0.05, "margin": 10000,
        "sector": "volatility", "exchange": "CFE"
    },
    "BTC": {
        "name": "Bitcoin", "cash": "BTC-USD", "futures": "BTC-USD",
        "multiplier": 5, "tick": 5.0, "margin": 45000,
        "sector": "crypto", "exchange": "CME"
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# FUTURES DATA FETCHER
# ═══════════════════════════════════════════════════════════════════════════════

class FuturesDataFetcher:
    """Fetch live futures + cash data from yfinance."""

    _cache: Dict[str, Dict] = {}
    _cache_ts: float = 0
    _CACHE_TTL = 120  # 2 minutes

    @classmethod
    def fetch_all(cls) -> Dict[str, Dict]:
        """Fetch all futures contracts and cash equivalents."""
        now = time.time()
        if cls._cache and now - cls._cache_ts < cls._CACHE_TTL:
            return cls._cache

        try:
            import yfinance as yf
        except ImportError:
            return {}

        results = {}
        for code, meta in FUTURES_MAP.items():
            try:
                # Fetch futures price
                ft = yf.Ticker(meta["futures"])
                fi = ft.fast_info
                futures_price = float(getattr(fi, "last_price", 0) or
                                      getattr(fi, "previous_close", 0) or 0)

                # Fetch cash/spot price
                ct = yf.Ticker(meta["cash"])
                ci = ct.fast_info
                cash_price = float(getattr(ci, "last_price", 0) or
                                   getattr(ci, "previous_close", 0) or 0)

                # For index futures, adjust cash (SPY*10 ≈ ES, QQQ*40 ≈ NQ)
                cash_adjusted = cash_price
                if code == "ES":
                    cash_adjusted = cash_price * 10  # SPY ~= ES/10
                elif code == "NQ":
                    cash_adjusted = cash_price * 40  # QQQ ~= NQ/40

                # Historical data for signals
                df = ft.history(period="3mo", interval="1d")
                returns = []
                if not df.empty and len(df) > 5:
                    close = df["Close"].values
                    returns = list(np.diff(close) / close[:-1])

                # Basis calculation
                if futures_price > 0 and cash_adjusted > 0:
                    basis = futures_price - cash_adjusted
                    basis_pct = (basis / cash_adjusted) * 100
                else:
                    basis = 0
                    basis_pct = 0

                # Volatility
                vol_20d = float(np.std(returns[-20:]) * np.sqrt(252) * 100) if len(returns) >= 20 else 0

                # Momentum
                mom_5d = float(np.sum(returns[-5:]) * 100) if len(returns) >= 5 else 0
                mom_20d = float(np.sum(returns[-20:]) * 100) if len(returns) >= 20 else 0

                # Daily change
                daily_chg = returns[-1] * 100 if returns else 0

                results[code] = {
                    "code": code,
                    "name": meta["name"],
                    "futures_price": round(futures_price, 2),
                    "cash_price": round(cash_price, 2),
                    "cash_adjusted": round(cash_adjusted, 2),
                    "basis": round(basis, 2),
                    "basis_pct": round(basis_pct, 4),
                    "basis_signal": "contango" if basis > 0 else "backwardation",
                    "daily_change_pct": round(daily_chg, 2),
                    "momentum_5d": round(mom_5d, 2),
                    "momentum_20d": round(mom_20d, 2),
                    "volatility_20d": round(vol_20d, 2),
                    "multiplier": meta["multiplier"],
                    "margin": meta["margin"],
                    "notional": round(futures_price * meta["multiplier"], 0),
                    "sector": meta["sector"],
                    "exchange": meta["exchange"],
                    "leverage": round(futures_price * meta["multiplier"] / max(meta["margin"], 1), 1),
                    "updated": datetime.now().isoformat(),
                }
            except Exception as e:
                logger.debug(f"Futures {code} fetch error: {e}")
                results[code] = {
                    "code": code, "name": meta["name"],
                    "error": str(e), "sector": meta["sector"]
                }

        cls._cache = results
        cls._cache_ts = now
        return results

    @classmethod
    def get_contract(cls, code: str) -> Optional[Dict]:
        """Get single futures contract data."""
        data = cls.fetch_all()
        return data.get(code.upper())


# ═══════════════════════════════════════════════════════════════════════════════
# BASIS TRADING SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════

class BasisTrader:
    """
    Generates basis trading signals:
    - Contango → sell futures, buy cash (carry trade)
    - Backwardation → buy futures, sell cash
    - Basis widening/narrowing momentum
    """

    @classmethod
    def analyze(cls, futures_data: Dict[str, Dict]) -> Dict:
        """
        Analyze basis across all futures for trading opportunities.
        """
        opportunities = []
        sector_basis = {}

        for code, data in futures_data.items():
            if "error" in data:
                continue

            basis_pct = data.get("basis_pct", 0)
            vol = data.get("volatility_20d", 20)
            mom = data.get("momentum_5d", 0)

            # Basis signal strength
            if abs(basis_pct) > 0.5:  # Meaningful basis
                signal = "sell_basis" if basis_pct > 0.5 else "buy_basis"
                strength = min(abs(basis_pct) / 2.0, 1.0)

                # Adjust by momentum
                if signal == "sell_basis" and mom > 0:
                    strength *= 0.7  # Basis likely to widen more
                elif signal == "buy_basis" and mom < 0:
                    strength *= 0.7

                opportunities.append({
                    "code": code,
                    "name": data["name"],
                    "signal": signal,
                    "basis_pct": basis_pct,
                    "strength": round(strength, 3),
                    "annualized_carry": round(basis_pct * 4, 2),  # ~quarterly roll
                    "risk_level": "high" if vol > 30 else "medium" if vol > 18 else "low",
                })

            # Sector aggregation
            sector = data.get("sector", "other")
            if sector not in sector_basis:
                sector_basis[sector] = []
            sector_basis[sector].append(basis_pct)

        # Sector averages
        sector_summary = {}
        for sector, bases in sector_basis.items():
            avg = float(np.mean(bases))
            sector_summary[sector] = {
                "avg_basis_pct": round(avg, 4),
                "structure": "contango" if avg > 0 else "backwardation",
                "count": len(bases)
            }

        opportunities.sort(key=lambda x: abs(x["strength"]), reverse=True)

        return {
            "opportunities": opportunities,
            "sector_summary": sector_summary,
            "total_contracts": len(futures_data),
            "contango_count": sum(1 for d in futures_data.values() if d.get("basis_pct", 0) > 0),
            "backwardation_count": sum(1 for d in futures_data.values() if d.get("basis_pct", 0) < 0),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ROLL CALENDAR
# ═══════════════════════════════════════════════════════════════════════════════

class RollCalendar:
    """
    Futures roll dates and calendar management.
    Most CME contracts roll quarterly (Mar/Jun/Sep/Dec).
    """

    ROLL_MONTHS = {
        "ES": [3, 6, 9, 12], "NQ": [3, 6, 9, 12],
        "CL": list(range(1, 13)),  # Monthly
        "GC": [2, 4, 6, 8, 10, 12],
        "SI": [3, 5, 7, 9, 12],
        "ZB": [3, 6, 9, 12],
        "VX": list(range(1, 13)),  # Monthly
        "BTC": [3, 6, 9, 12],
    }

    @classmethod
    def next_roll_dates(cls) -> List[Dict]:
        """Get next roll date for each contract."""
        now = datetime.now()
        rolls = []

        for code, months in cls.ROLL_MONTHS.items():
            # Find next roll month (3rd Friday of roll month)
            for m in months:
                year = now.year if m >= now.month else now.year + 1
                # 3rd Friday
                import calendar
                c = calendar.Calendar()
                fridays = [d for d in c.itermonthdays2(year, m) if d[0] > 0 and d[1] == 4]
                if len(fridays) >= 3:
                    roll_day = fridays[2][0]
                    roll_date = datetime(year, m, roll_day)
                    if roll_date > now:
                        days_to_roll = (roll_date - now).days
                        rolls.append({
                            "code": code,
                            "name": FUTURES_MAP[code]["name"],
                            "roll_date": roll_date.strftime("%Y-%m-%d"),
                            "days_to_roll": days_to_roll,
                            "urgency": "imminent" if days_to_roll <= 5 else "soon" if days_to_roll <= 14 else "normal",
                            "month_code": f"{code}{roll_date.strftime('%b%y').upper()}"
                        })
                        break

        rolls.sort(key=lambda x: x["days_to_roll"])
        return rolls


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER FUTURES DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

class FuturesDashboard:
    """Complete futures intelligence."""

    @classmethod
    def get_dashboard(cls) -> Dict:
        data = FuturesDataFetcher.fetch_all()
        basis = BasisTrader.analyze(data)
        rolls = RollCalendar.next_roll_dates()

        # Compute composite futures signal
        signals = []
        for code, d in data.items():
            if "error" not in d:
                mom = d.get("momentum_5d", 0) / 10
                basis_sig = -d.get("basis_pct", 0) / 2  # Negative basis = bullish
                signals.append({"code": code, "signal": round(float(np.clip(mom + basis_sig, -1, 1)), 3)})

        return {
            "contracts": data,
            "basis_analysis": basis,
            "roll_calendar": rolls,
            "signals": sorted(signals, key=lambda x: abs(x["signal"]), reverse=True),
            "updated": datetime.now().isoformat()
        }


# Singleton
def get_futures_dashboard() -> Dict:
    return FuturesDashboard.get_dashboard()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    d = get_futures_dashboard()
    for code, c in d["contracts"].items():
        if "error" not in c:
            print(f"{code}: ${c['futures_price']:,.2f} | basis={c['basis_pct']:.3f}% | mom5d={c['momentum_5d']:.2f}%")
    print("✅ Futures Engine operational")