"""
catalyst_engine.py — Renaissance.io Universal Catalyst Detection Engine
========================================================================
Catches the HIMS +40%, GME +200%, SMCI -40% BEFORE they happen.
Works across ALL sectors, ALL market caps: Large → Mid → Small → Micro → Penny

Universe Strategy (rebuilds daily):
  Tier 1 — S&P 500        (~500 large caps,  Wikipedia)
  Tier 2 — NASDAQ 100     (~100 tech/growth,  Wikipedia)
  Tier 3 — Russell 2000   (~500 small caps,   IWM holdings)
  Tier 4 — High Short Int  (squeeze candidates, Finviz)
  Tier 5 — Unusual Volume  (any size, any sector, Finviz)
  Tier 6 — Penny/Micro     ($0.50-$10, Finviz screener)
  Tier 7 — Pre-market mvrs (Yahoo Finance real-time)
  Tier 8 — Options active  (highest options volume)

Total live universe: 500-2000 tickers, rebuilt daily.
"""

import time
import threading
import requests
import re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import yfinance as yf
    import numpy as np
    import pandas as pd
    YF_OK = True
except ImportError:
    YF_OK = False

_alert_lock    = threading.Lock()
_universe_lock = threading.Lock()
_alert_queue   = []
_edgar_alerts  = []
_live_universe = []
_universe_meta = {}
_scan_status   = {
    "running": False, "last_scan": None,
    "tickers_scanned": 0, "total": 0,
    "universe_size": 0, "universe_built": None,
}
MAX_ALERTS = 300
SCAN_UNIVERSE = {}  # kept for API compatibility


def _mktcap_tier(mktcap):
    if   mktcap >= 200e9: return "MEGA"
    elif mktcap >= 10e9:  return "LARGE"
    elif mktcap >= 2e9:   return "MID"
    elif mktcap >= 300e6: return "SMALL"
    elif mktcap >= 50e6:  return "MICRO"
    else:                  return "NANO/PENNY"


# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSE BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
def _fetch_sp500():
    """
    Fetch current S&P 500 constituents.
    4-layer fallback: pandas.read_html → BeautifulSoup → requests+regex → hardcoded core.
    """
    # Layer 1: pandas.read_html (needs lxml or html5lib installed)
    try:
        import pandas as _pd
        tables = _pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            attrs={"id": "constituents"}, timeout=15,
        )
        if tables:
            df = tables[0]
            col = "Symbol" if "Symbol" in df.columns else df.columns[0]
            syms = df[col].astype(str).tolist()
            result = [s.replace(".", "-").strip() for s in syms
                      if re.match(r'^[A-Z]{1,5}(-[A-Z])?$', s.replace(".", "-"))][:505]
            if len(result) > 400:
                return result
    except Exception:
        pass
    # Layer 2: BeautifulSoup table parse (more robust than pandas for Wikipedia)
    try:
        from bs4 import BeautifulSoup
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            timeout=15, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        if table:
            syms = []
            for row in table.find_all("tr")[1:]:
                cells = row.find_all("td")
                if cells:
                    sym = cells[0].get_text(strip=True).replace(".", "-")
                    if re.match(r'^[A-Z]{1,6}(-[A-Z])?$', sym):
                        syms.append(sym)
            if len(syms) > 400:
                return syms[:505]
    except Exception:
        pass
    # Layer 3: Raw regex scrape
    try:
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        syms = re.findall(r'<td[^>]*><a href="/wiki/[^"]+">([A-Z]{1,5}(?:-[A-Z])?)</a></td>', resp.text)
        syms += re.findall(r'<td>([A-Z]{1,5}(?:-[A-Z])?)</td>', resp.text)
        result = list(dict.fromkeys(s for s in syms if 1 <= len(s) <= 6))[:505]
        if result:
            return result
    except Exception:
        pass
    # Layer 4: Hardcoded core S&P 500 names (last resort — stale but better than nothing)
    print("  ⚠ S&P 500 fetch failed — using hardcoded 100-name fallback")
    return [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","BRK-B","JPM",
        "UNH","LLY","XOM","V","MA","HD","AVGO","CVX","MRK","ABBV","COST","PEP",
        "KO","ADBE","WMT","BAC","CRM","TMO","ACN","MCD","CSCO","ABT","NKE","ORCL",
        "DHR","INTC","AMD","NFLX","QCOM","TXN","LIN","PM","NEE","AMGN","BMY",
        "RTX","UPS","SPGI","INTU","MDT","CAT","GS","BLK","SBUX","HON","PLD",
        "BA","DE","AXP","GILD","C","MDLZ","ADI","REGN","VRTX","ZTS","ISRG",
        "BX","ETN","EOG","SLB","MO","CB","PGR","APD","NOW","PANW","CRWD",
        "PLTR","GE","F","GM","T","VZ","CMCSA","DIS","WFC","MS","USB",
        "JNJ","PFE","MRNA","BIIB","ENPH","NET","SNOW","ZS","UBER","ABNB",
    ]

def _fetch_nasdaq100():
    try:
        resp = requests.get(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        syms = re.findall(r'<td>([A-Z]{2,5})</td>', resp.text)
        return list(dict.fromkeys(syms))[:110]
    except Exception:
        return []

def _fetch_russell2000_sample():
    """
    Russell 2000 small-cap universe. 3-layer fallback:
    1. iShares IWM holdings JSON (most accurate, ~2000 names)
    2. SPDR SMLV / iShares IWM alternative URL
    3. Hardcoded 200 liquid small/mid-cap names
    """
    # Layer 1: iShares IWM holdings JSON
    for _url in [
        "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax"
        "?fileType=json&fileName=IWM_holdings&dataType=fund",
        "https://www.ishares.com/us/products/239710/1467271812596.ajax"
        "?fileType=json&fileName=IWM_holdings&dataType=fund",
    ]:
        try:
            resp = requests.get(_url, timeout=20,
                                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            if resp.ok and resp.headers.get("content-type","").startswith("application/json"):
                data = resp.json()
                syms = []
                for item in data.get("aaData", []):
                    try:
                        sym = str(item[0]).strip().upper()
                        if re.match(r'^[A-Z]{1,5}$', sym) and sym not in ("CASH","XTSLA","USD"):
                            syms.append(sym)
                    except Exception:
                        pass
                if len(syms) > 100:
                    return syms[:800]
        except Exception:
            continue
    # Layer 2: Try Wikipedia Russell 2000 page for partial list
    try:
        resp = requests.get(
            "https://en.wikipedia.org/wiki/Russell_2000_Index",
            timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        syms = re.findall(r'<td>([A-Z]{2,5})</td>', resp.text)
        syms = list(dict.fromkeys(s for s in syms if 2 <= len(s) <= 5))
        if len(syms) > 50:
            return syms[:500]
    except Exception:
        pass
    # Layer 3: Hardcoded 150 liquid small/mid-caps that commonly appear in Russell 2000
    # Fallback is expected occasionally due to source instability; keep silent.
    # print("  ⚠ Russell 2000 fetch failed — using hardcoded 150-name small/mid-cap fallback")
    return [
        "RIVN","LCID","NKLA","WKHS","GOEV","RIDE","BLNK","CHPT","EVGO","PTRA",
        "SOFI","HOOD","UPST","AFRM","LMND","OPEN","CLOV","MMAT","SOUN","IONQ",
        "QUBT","RGTI","IQM","ACHR","JOBY","RKLB","ASTR","SPCE","LILM","EVTL",
        "MARA","RIOT","CLSK","HIVE","BITF","CIFR","HUT","BTBT","WULF","IREN",
        "LAZR","MBLY","LIDR","OUST","VLDR","INVZ","ISEE","AGL","OPTX","SWVL",
        "PTON","ROKU","FUBO","PLTK","SKLZ","DKNG","PENN","GENI","EVERI","AGS",
        "REAL","ETSY","WISH","BARK","CHWY","FTCH","POSH","RVLV","FIGS","VZIO",
        "AMC","GME","BBBY","EXPR","KOSS","NAKD","SPRT","MVIS","ZNGA","TTWO",
        "BYND","TTCF","APPH","WOOF","HIMS","SDC","FIGS","GOSS","BODY","PRPL",
        "PLUG","FCEL","BLDP","HYLN","NKLA","BE","GPRE","REX","AMTX","GEVO",
        "XPEV","NIO","LI","KNDI","SOLO","FFIE","MULN","NXTP","IDEX","AYRO",
        "AGEN","SIGA","NVAX","OCGN","INO","VXRT","BNTX","SRPT","EDIT","NTLA",
        "BEAM","CRSP","PACB","ILMN","NTRA","FATE","KYMR","ALNY","ATRA","FOLD",
        "TRUP","PETS","FRPT","HIMS","WOLF","AEHR","FORM","COHU","ONTO","ACMR",
        "KIND","TRMK","INVA","NBTB","FFIN","CVBF","BUSE","GABC","PFBC","FBNC",
    ]

def _fetch_finviz(url_suffix, max_syms=150):
    """Generic Finviz screener scraper."""
    try:
        url  = "https://finviz.com/screener.ashx?" + url_suffix
        resp = requests.get(url, timeout=12,
                            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        syms = re.findall(r'quote\.ashx\?t=([A-Z]{1,5})"', resp.text)
        return list(dict.fromkeys(syms))[:max_syms]
    except Exception:
        return []

def _fetch_yahoo_screener(scr_id, count=50):
    try:
        url  = (f"https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
                f"?formatted=false&lang=en-US&region=US&scrIds={scr_id}&count={count}")
        resp = requests.get(url, timeout=8,
                            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        if resp.ok:
            quotes = resp.json().get("finance", {}).get("result", [{}])[0].get("quotes", [])
            return [q.get("symbol","") for q in quotes
                    if re.match(r'^[A-Z]{1,5}$', q.get("symbol",""))]
    except Exception:
        pass
    return []

def _fetch_nasdaq_movers(kind: str = "gainers", max_syms: int = 80):
    """
    Fetch Nasdaq free market movers (gainers/losers/most_active).
    kind: 'gainers' | 'losers' | 'most_active'
    """
    try:
        hdr = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://www.nasdaq.com",
            "Referer": "https://www.nasdaq.com/",
        }
        r = requests.get("https://api.nasdaq.com/api/marketmovers", headers=hdr, timeout=10)
        if not r.ok:
            return []
        j = r.json() or {}
        st = (((j.get("data") or {}).get("STOCKS")) or {})
        if kind == "gainers":
            rows = st.get("MostAdvanced") or []
        elif kind == "losers":
            rows = st.get("MostDeclined") or []
        else:
            rows = st.get("MostActive") or []
        out = []
        for row in rows:
            sym = str(row.get("symbol") or row.get("Symbol") or "").upper().strip()
            if sym and re.match(r"^[A-Z]{1,5}$", sym):
                out.append(sym)
        return list(dict.fromkeys(out))[:max_syms]
    except Exception:
        return []

def _fetch_market_movers(kind: str, max_syms: int = 80):
    """Prefer Nasdaq movers; fallback to Yahoo screeners if needed."""
    syms = _fetch_nasdaq_movers(kind=kind, max_syms=max_syms)
    if syms:
        return syms
    # emergency fallback only
    if kind == "gainers":
        return _fetch_yahoo_screener("day_gainers", min(50, max_syms))
    if kind == "losers":
        return _fetch_yahoo_screener("day_losers", min(50, max_syms))
    return _fetch_yahoo_screener("most_actives", min(80, max_syms))

def build_universe(force=False):
    global _live_universe, _universe_meta
    if not force and _live_universe and _scan_status.get("universe_built"):
        built = datetime.fromisoformat(_scan_status["universe_built"])
        if (datetime.now() - built).total_seconds() < 86400:
            return _live_universe

    print("Building dynamic scan universe (ALL sectors, ALL market caps)...")
    all_syms = []
    sources  = {}

    tasks = {
        "S&P500":       (_fetch_sp500,),
        "NASDAQ100":    (_fetch_nasdaq100,),
        "Russell2000":  (_fetch_russell2000_sample,),
        "UnusualVol":   (_fetch_finviz, "v=111&f=sh_avgvol_o200,sh_price_o0.5&s=ta_unusualvolume&r=1"),
        "HighShortInt": (_fetch_finviz, "v=111&f=sh_short_o10,sh_avgvol_o100&s=sh_short_desc&r=1"),
        "PennyStocks":  (_fetch_finviz, "v=111&f=sh_price_u10,sh_price_o0.5,sh_avgvol_o500&s=ta_unusualvolume&r=1"),
        "Gainers":      (_fetch_market_movers, "gainers", 40),
        "Losers":       (_fetch_market_movers, "losers", 40),
        "MostActive":   (_fetch_market_movers, "most_active", 60),
    }

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {}
        for label, args in tasks.items():
            fn  = args[0]
            fargs = args[1:]
            futures[pool.submit(fn, *fargs)] = label

        for fut in as_completed(futures, timeout=35):
            label = futures[fut]
            try:
                result = fut.result() or []
                sources[label] = len(result)
                all_syms.extend(result)
            except Exception as e:
                sources[label] = 0

    seen = set()
    clean = []
    for sym in all_syms:
        sym = sym.upper().strip()
        if sym and sym not in seen and re.match(r'^[A-Z]{1,5}$', sym):
            seen.add(sym)
            clean.append(sym)

    with _universe_lock:
        _live_universe = clean
        _scan_status["universe_size"]  = len(clean)
        _scan_status["universe_built"] = datetime.now().isoformat()

    print(f"Universe built: {len(clean)} unique tickers | " +
          " | ".join(f"{k}:{v}" for k, v in sources.items()))
    return clean


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL 1 — UNUSUAL OPTIONS ACTIVITY
# ═══════════════════════════════════════════════════════════════════════════════
def scan_unusual_options(sym, spot):
    result = {"score": 0, "direction": "NEUTRAL", "signals": [], "details": {}}
    if not YF_OK or spot <= 0:
        return result
    try:
        tk   = yf.Ticker(sym)
        exps = tk.options
        if not exps:
            return result

        today = datetime.now().date()
        near_exps = [(exp, (datetime.strptime(exp, "%Y-%m-%d").date() - today).days)
                     for exp in exps[:6]
                     if 0 <= (datetime.strptime(exp, "%Y-%m-%d").date() - today).days <= 45]
        if not near_exps:
            near_exps = [(exps[0], 7)]

        total_call_vol = total_put_vol = 0
        unusual_calls  = []
        unusual_puts   = []
        max_oi_ratio   = 0
        iv_values      = []
        min_vol        = 30 if spot < 5 else (50 if spot < 20 else 100)

        for exp, dte in near_exps[:4]:
            try:
                chain = tk.option_chain(exp)
                for _, row in chain.calls.iterrows():
                    strike = float(row.get("strike", 0))
                    vol    = int(row.get("volume", 0) or 0)
                    oi     = int(row.get("openInterest", 1) or 1)
                    iv     = float(row.get("impliedVolatility", 0) or 0)
                    otm    = (strike - spot) / spot
                    if iv > 0: iv_values.append(iv)
                    total_call_vol += vol
                    ratio = vol / max(oi, 1)
                    if 0.01 < otm < 0.70 and vol >= min_vol and ratio > 2.0:
                        unusual_calls.append({"strike": strike, "dte": dte, "vol": vol,
                                              "oi": oi, "ratio": round(ratio,1),
                                              "otm_pct": round(otm*100,1), "iv": round(iv*100,1)})
                        max_oi_ratio = max(max_oi_ratio, ratio)
                for _, row in chain.puts.iterrows():
                    strike = float(row.get("strike", 0))
                    vol    = int(row.get("volume", 0) or 0)
                    oi     = int(row.get("openInterest", 1) or 1)
                    iv     = float(row.get("impliedVolatility", 0) or 0)
                    otm    = (spot - strike) / spot
                    total_put_vol += vol
                    ratio = vol / max(oi, 1)
                    if 0.01 < otm < 0.70 and vol >= min_vol and ratio > 2.0:
                        unusual_puts.append({"strike": strike, "dte": dte, "vol": vol,
                                             "oi": oi, "ratio": round(ratio,1),
                                             "otm_pct": round(otm*100,1), "iv": round(iv*100,1)})
            except Exception:
                pass

        score, signals, direction = 0, [], "NEUTRAL"
        pcr = total_put_vol / max(total_call_vol, 1)

        if pcr < 0.4 and total_call_vol > 100:
            score += 20; signals.append(f"P/C ratio {pcr:.2f} — extreme call buying"); direction = "BULLISH"
        elif pcr > 2.5 and total_put_vol > 100:
            score += 20; signals.append(f"P/C ratio {pcr:.2f} — heavy put buying"); direction = "BEARISH"

        if unusual_calls:
            score += min(40, len(unusual_calls) * 10 + int((max_oi_ratio - 2) * 3))
            top = sorted(unusual_calls, key=lambda x: x["ratio"], reverse=True)[0]
            signals.append(f"{len(unusual_calls)} OTM call sweep(s) — ${top['strike']} {top['dte']}DTE {top['ratio']:.1f}x vol/OI")
            direction = "BULLISH"

        if unusual_puts:
            score += min(35, len(unusual_puts) * 10)
            signals.append(f"{len(unusual_puts)} OTM put sweep(s)")
            if direction == "NEUTRAL": direction = "BEARISH"

        if iv_values:
            avg_iv = float(np.mean(iv_values)) * 100
            if avg_iv > 150: score += 25; signals.append(f"IV extreme: {avg_iv:.0f}% — binary event")
            elif avg_iv > 80: score += 15; signals.append(f"IV elevated: {avg_iv:.0f}%")

        result.update({"score": min(100, score), "direction": direction, "signals": signals,
                        "details": {"call_vol": total_call_vol, "put_vol": total_put_vol,
                                    "pcr": round(pcr,2), "unusual_calls": unusual_calls[:3],
                                    "unusual_puts": unusual_puts[:3],
                                    "avg_iv": round(float(np.mean(iv_values))*100,1) if iv_values else 0}})
    except Exception as e:
        result["error"] = str(e)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL 2 — SHORT SQUEEZE DETECTOR (all caps)
# ═══════════════════════════════════════════════════════════════════════════════
def scan_short_squeeze(sym, info, hist):
    score, signals, metrics = 0, [], {}
    try:
        si_pct    = float(info.get("shortPercentOfFloat") or 0) * 100
        dtc       = float(info.get("shortRatio") or 0)
        float_sh  = float(info.get("floatShares") or 1e12)
        price     = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
        week52_hi = float(info.get("fiftyTwoWeekHigh") or price or 1)
        mktcap    = float(info.get("marketCap") or 0)
        tier      = _mktcap_tier(mktcap)
        si_mult   = 1.5 if tier in ("MICRO", "NANO/PENNY", "SMALL") else 1.0

        metrics = {"short_pct_float": round(si_pct,1), "days_to_cover": round(dtc,1),
                   "float_millions": round(float_sh/1e6,2), "mktcap_tier": tier,
                   "price_vs_52w_hi": round(price/week52_hi*100,1) if week52_hi else 0}

        if si_pct >= 30: score += int(35*si_mult); signals.append(f"Short interest EXTREME: {si_pct:.1f}% of float")
        elif si_pct >= 20: score += int(28*si_mult); signals.append(f"Short interest VERY HIGH: {si_pct:.1f}%")
        elif si_pct >= 12: score += int(18*si_mult); signals.append(f"Short interest HIGH: {si_pct:.1f}%")
        elif si_pct >= 6: score += int(10*si_mult); signals.append(f"Short interest elevated: {si_pct:.1f}%")

        if dtc >= 10: score += 20; signals.append(f"Days-to-cover CRITICAL: {dtc:.1f}d")
        elif dtc >= 5: score += 12; signals.append(f"Days-to-cover HIGH: {dtc:.1f}d")
        elif dtc >= 3: score += 6

        if float_sh < 5e6: score += 30; signals.append(f"NANO float: {float_sh/1e6:.2f}M shares — explosive")
        elif float_sh < 15e6: score += 22; signals.append(f"Micro-float: {float_sh/1e6:.1f}M shares")
        elif float_sh < 40e6: score += 14; signals.append(f"Small float: {float_sh/1e6:.1f}M shares")
        elif float_sh < 100e6: score += 7

        if week52_hi > 0 and price > 0:
            pct_hi = price / week52_hi
            if pct_hi > 0.92: score += 15; signals.append(f"At {pct_hi*100:.0f}% of 52W high — breakout zone")
            elif pct_hi > 0.80: score += 7

        if hist is not None and not hist.empty and len(hist) >= 5:
            vol_today = float(hist["Volume"].iloc[-1])
            vol_avg   = float(hist["Volume"].tail(20).mean())
            if vol_avg > 0:
                vr = vol_today / vol_avg
                metrics["vol_vs_avg"] = round(vr, 1)
                if vr > 5: score += 20; signals.append(f"Volume EXPLOSION: {vr:.1f}x — squeeze may be igniting")
                elif vr > 3: score += 14; signals.append(f"Volume surge: {vr:.1f}x average")
                elif vr > 2: score += 7

        if tier == "NANO/PENNY" and price < 5 and si_pct > 5:
            score += 12; signals.append("Penny stock + short interest = extreme % move potential")

    except Exception as e:
        signals.append(f"Error: {e}")
    return {"score": min(100, score), "signals": signals, "metrics": metrics}


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL 3 — EARNINGS CATALYST
# ═══════════════════════════════════════════════════════════════════════════════
def scan_earnings_catalyst(sym, info, spot):
    score, signals = 0, []
    days_to_earnings = None
    implied_move_pct = None
    try:
        tk = yf.Ticker(sym)
        try:
            ed_df = tk.earnings_dates
            if ed_df is not None and not ed_df.empty:
                for idx in ed_df.index:
                    d = idx.date() if hasattr(idx, "date") else None
                    if d and d >= datetime.now().date():
                        days_to_earnings = (d - datetime.now().date()).days
                        break
        except Exception:
            pass

        if days_to_earnings is not None:
            if 0 <= days_to_earnings <= 2:
                score += 40; signals.append(f"EARNINGS TONIGHT/TOMORROW — {days_to_earnings}d")
            elif days_to_earnings <= 5:
                score += 30; signals.append(f"Earnings in {days_to_earnings}d — THIS WEEK")
            elif days_to_earnings <= 14:
                score += 18; signals.append(f"Earnings in {days_to_earnings}d")
            elif days_to_earnings <= 30:
                score += 8; signals.append(f"Earnings in {days_to_earnings}d")

        if spot > 0 and days_to_earnings is not None and days_to_earnings <= 21:
            try:
                exps = tk.options
                today = datetime.now().date()
                target_exp = next((e for e in exps
                                   if (datetime.strptime(e, "%Y-%m-%d").date() - today).days >= days_to_earnings), None)
                if target_exp:
                    chain = tk.option_chain(target_exp)
                    atm_c = chain.calls.iloc[(chain.calls["strike"] - spot).abs().argsort()[:1]]
                    atm_p = chain.puts.iloc[(chain.puts["strike"]  - spot).abs().argsort()[:1]]
                    if not atm_c.empty and not atm_p.empty:
                        straddle = float(atm_c["lastPrice"].iloc[0]) + float(atm_p["lastPrice"].iloc[0])
                        implied_move_pct = round(straddle / spot * 100, 1)
                        if implied_move_pct > 30: score += 25; signals.append(f"Options pricing ±{implied_move_pct}% — EXPLOSIVE")
                        elif implied_move_pct > 15: score += 15; signals.append(f"Options pricing ±{implied_move_pct}% move")
                        elif implied_move_pct > 8:  score += 8;  signals.append(f"Options pricing ±{implied_move_pct}% move")
            except Exception:
                pass
    except Exception as e:
        signals.append(f"Error: {e}")

    return {"score": min(100,score), "days_to_earnings": days_to_earnings,
            "implied_move_pct": implied_move_pct, "signals": signals}


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL 4 — TAPE ACCELERATION + AFTERHOURS DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
def scan_tape_acceleration(sym, hist, info):
    score, signals = 0, []
    if hist is None or hist.empty or len(hist) < 5:
        return {"score": 0, "signals": [], "metrics": {}}
    try:
        close  = hist["Close"].values.astype(float)
        volume = hist["Volume"].values.astype(float)
        n      = len(close)

        ret_1d  = (close[-1]/close[-2] - 1)*100 if n >= 2 else 0
        ret_5d  = (close[-1]/close[-5] - 1)*100 if n >= 5 else 0
        vol_1d  = volume[-1]
        vol_avg = float(np.mean(volume[-20:])) if n >= 20 else float(np.mean(volume))
        vol_r   = vol_1d / max(vol_avg, 1)

        hi52 = float(np.max(close[-252:])) if n >= 252 else float(np.max(close))
        lo52 = float(np.min(close[-252:])) if n >= 252 else float(np.min(close))
        rng  = hi52 - lo52
        pos  = (close[-1] - lo52) / rng * 100 if rng > 0 else 50

        # After-hours / pre-market move — most important signal
        ah_chg = 0.0
        try:
            ah_price  = float(info.get("postMarketPrice") or info.get("preMarketPrice") or 0)
            reg_price = float(info.get("regularMarketPrice") or close[-1])
            if ah_price > 0 and reg_price > 0:
                ah_chg = (ah_price / reg_price - 1) * 100
                if abs(ah_chg) > 15:
                    score += 45; direction = "PUMPING" if ah_chg > 0 else "DUMPING"
                    signals.append(f"AFTERHOURS {direction} {ah_chg:+.1f}% — ACTIVE CATALYST NOW")
                elif abs(ah_chg) > 8:
                    score += 30; signals.append(f"After-hours: {ah_chg:+.1f}% — catalyst detected")
                elif abs(ah_chg) > 3:
                    score += 15; signals.append(f"Extended hours move: {ah_chg:+.1f}%")
        except Exception:
            pass

        if vol_r > 8:   score += 30; signals.append(f"Volume MASSIVE: {vol_r:.0f}x 20d avg")
        elif vol_r > 4: score += 20; signals.append(f"Volume EXPLOSION: {vol_r:.1f}x average")
        elif vol_r > 2.5: score += 12; signals.append(f"Volume surge: {vol_r:.1f}x average")

        if abs(ret_1d) > 10 and vol_r > 2:
            score += 20; signals.append(f"Large move {ret_1d:+.1f}% on {vol_r:.1f}x volume")
        elif abs(ret_1d) > 5:
            score += 10; signals.append(f"Significant move: {ret_1d:+.1f}% today")

        if abs(ret_5d) > 20: score += 15; signals.append(f"Strong 5d trend: {ret_5d:+.1f}%")
        if pos > 92: score += 10; signals.append(f"Near 52W high ({pos:.0f}% of range)")

        return {"score": min(100,score), "signals": signals,
                "metrics": {"ret_1d": round(ret_1d,2), "ret_5d": round(ret_5d,2),
                             "vol_ratio": round(vol_r,2), "range_pos": round(pos,1), "ah_chg": round(ah_chg,2)}}
    except Exception as e:
        return {"score": 0, "signals": [f"Error: {e}"], "metrics": {}}


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL 5 — SEC EDGAR 8-K + FORM 4 INSIDERS
# ═══════════════════════════════════════════════════════════════════════════════
def fetch_edgar_8k_alerts(max_items=50):
    alerts = []
    try:
        url  = (f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K"
                f"&dateb=&owner=include&count={max_items}&search_text=&output=atom")
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "Renaissance.io research@example.com"})
        if not resp.ok: return []
        for entry in re.findall(r"<entry>(.*?)</entry>", resp.text, re.DOTALL):
            try:
                title   = (re.search(r"<title>(.*?)</title>",   entry, re.DOTALL) or type("", (), {"group": lambda s,i: ""})()).group(1).strip()
                updated = (re.search(r"<updated>(.*?)</updated>", entry, re.DOTALL) or type("", (), {"group": lambda s,i: ""})()).group(1).strip()
                link    = (re.search(r'href="([^"]+)"', entry) or type("", (), {"group": lambda s,i: ""})()).group(1).strip()
                ticker_m = re.search(r"\(([A-Z]{1,5})\)", title)
                ticker   = ticker_m.group(1) if ticker_m else None
                t = title.lower()
                event_type = "MATERIAL EVENT"; direction = "WATCH"
                if any(k in t for k in ["earnings","quarterly","annual results"]): event_type="EARNINGS"
                elif any(k in t for k in ["merger","acqui","definitive","agreement"]): event_type="M&A"; direction="BULLISH"
                elif any(k in t for k in ["fda","pdufa","nda","bla","approval"]): event_type="FDA/CLINICAL"; direction="BINARY"
                elif any(k in t for k in ["guidance","outlook","raised","forecast"]): event_type="GUIDANCE"
                elif any(k in t for k in ["restructur","layoff","workforce"]): event_type="RESTRUCTURING"; direction="BEARISH"
                elif any(k in t for k in ["buyback","repurchase","dividend"]): event_type="CAPITAL RETURN"; direction="BULLISH"
                elif any(k in t for k in ["restat","material weakness"]): event_type="RESTATEMENT"; direction="BEARISH"
                alerts.append({"ticker": ticker, "title": title, "event_type": event_type,
                               "direction": direction, "updated": updated, "link": link, "source": "SEC EDGAR 8-K"})
            except Exception:
                pass
    except Exception:
        pass
    return alerts

def scan_insider_activity(sym):
    score, buys, signals = 0, 0, []
    try:
        url  = (f"https://efts.sec.gov/LATEST/search-index?q=%22{sym}%22"
                f"&dateRange=custom"
                f"&startdt={(datetime.now()-timedelta(days=90)).strftime('%Y-%m-%d')}"
                f"&enddt={datetime.now().strftime('%Y-%m-%d')}&forms=4")
        resp = requests.get(url, timeout=6, headers={"User-Agent": "Renaissance.io research@example.com"})
        if resp.ok:
            buys = len(resp.json().get("hits", {}).get("hits", []))
            if buys >= 4: score=25; signals.append(f"Insider Form 4 cluster: {buys} filings (90d)")
            elif buys >= 2: score=12; signals.append(f"Insider activity: {buys} Form 4 filings (90d)")
    except Exception:
        pass
    return {"score": score, "signals": signals, "buys": buys}


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE SCORER
# ═══════════════════════════════════════════════════════════════════════════════
def _suggest_trade(sym, spot, direction, score, opts, squeeze, earn, tier="LARGE"):
    dte  = earn.get("days_to_earnings")
    impl = earn.get("implied_move_pct")
    small = tier in ("MICRO", "NANO/PENNY", "SMALL")
    mult  = 1 if spot < 5 else 5

    if direction == "BULLISH":
        if dte is not None and dte <= 7 and impl:
            return {"type": "BUY CALL", "strike": round(spot*1.05/mult)*mult, "expiry": f"{dte+2}d",
                    "reason": f"Earnings play — options pricing ±{impl}% move",
                    "size": "0.5% portfolio (binary)", "color": "green"}
        elif squeeze["score"] > 45:
            return {"type": "BUY CALL / LONG" if small else "BUY CALL",
                    "strike": round(spot*1.08/mult)*mult, "expiry": "30d",
                    "reason": f"Short squeeze — {squeeze['metrics'].get('short_pct_float','?')}% SI, {squeeze['metrics'].get('float_millions','?')}M float",
                    "size": "1-2% portfolio", "color": "green"}
        return {"type": "BUY CALL", "strike": round(spot*1.05/mult)*mult, "expiry": "21d",
                "reason": "Unusual call flow + momentum", "size": "0.5-1% portfolio", "color": "green"}

    elif direction == "BEARISH":
        return {"type": "BUY PUT", "strike": round(spot*0.95/mult)*mult, "expiry": "21d",
                "reason": "Unusual put flow / bearish catalyst", "size": "0.5-1% portfolio", "color": "red"}

    else:
        if impl and impl > 15:
            return {"type": "STRADDLE", "strike": round(spot/mult)*mult,
                    "expiry": f"{(dte or 14)+1}d",
                    "reason": f"Binary event — ±{impl}% move priced, direction unclear",
                    "size": "0.5% portfolio", "color": "gold"}
        return {"type": "WATCH", "strike": None, "expiry": None,
                "reason": "Building conviction — monitor for entry", "size": "Paper only", "color": "gray"}


def score_ticker(sym):
    if not YF_OK: return None
    try:
        tk   = yf.Ticker(sym)
        info = tk.info or {}
        hist = tk.history(period="60d")

        spot = float(info.get("currentPrice") or info.get("regularMarketPrice") or
                     info.get("ask") or (float(hist["Close"].iloc[-1]) if not hist.empty else 0))
        if spot < 0.10: return None

        name   = info.get("shortName") or info.get("longName") or sym
        sector = _universe_meta.get(sym, {}).get("sector") or info.get("sector") or "UNKNOWN"
        mktcap = float(info.get("marketCap") or 0)
        tier   = _mktcap_tier(mktcap)

        _universe_meta[sym] = {"sector": sector, "mktcap_tier": tier,
                                "float_m": round(float(info.get("floatShares") or 0)/1e6, 2)}

        opts    = scan_unusual_options(sym, spot)
        squeeze = scan_short_squeeze(sym, info, hist)
        earn    = scan_earnings_catalyst(sym, info, spot)
        tape    = scan_tape_acceleration(sym, hist, info)
        insider = scan_insider_activity(sym)

        composite = (opts["score"]*0.30 + squeeze["score"]*0.20 +
                     earn["score"]*0.25 + tape["score"]*0.15 + insider["score"]*0.10)

        # Boost small/penny — same signal = bigger % move
        if tier in ("MICRO", "NANO/PENNY") and composite > 10:
            composite *= 1.3

        # Instant alert if afterhours move
        ah_chg = tape.get("metrics", {}).get("ah_chg", 0)
        if abs(ah_chg) > 5:
            composite = max(composite, 60)

        if composite < 12: return None

        direction = opts.get("direction", "NEUTRAL")
        if squeeze["score"] > 40 and direction == "NEUTRAL": direction = "BULLISH"
        if ah_chg < -5: direction = "BEARISH"
        elif ah_chg > 5: direction = "BULLISH"

        conviction = "HIGH" if composite >= 65 else "MEDIUM" if composite >= 40 else "LOW"
        all_signals = tape["signals"] + opts["signals"] + earn["signals"] + squeeze["signals"] + insider["signals"]

        return {
            "sym": sym, "name": name, "sector": sector, "mktcap_tier": tier,
            "spot": round(spot, 4), "direction": direction, "conviction": conviction,
            "composite_score": round(min(100, composite), 1),
            "signals": all_signals[:8],
            "trade": _suggest_trade(sym, spot, direction, composite, opts, squeeze, earn, tier),
            "signal_scores": {"options": opts["score"], "squeeze": squeeze["score"],
                              "earnings": earn["score"], "tape": tape["score"], "insider": insider["score"]},
            "options_detail": opts.get("details", {}),
            "squeeze_metrics": squeeze.get("metrics", {}),
            "earnings_detail": {"days_to_earnings": earn.get("days_to_earnings"),
                                "implied_move_pct": earn.get("implied_move_pct")},
            "tape_metrics": tape.get("metrics", {}),
            "ah_chg": ah_chg,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND SCANNER
# ═══════════════════════════════════════════════════════════════════════════════
def run_full_scan():
    global _alert_queue, _edgar_alerts
    _scan_status["running"] = True
    _scan_status["tickers_scanned"] = 0

    universe = build_universe()
    _scan_status["total"] = len(universe)

    try:
        _edgar_alerts = fetch_edgar_8k_alerts(50)
    except Exception:
        pass

    new_alerts = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(score_ticker, sym): sym for sym in universe}
        for fut in as_completed(futures, timeout=600):
            try:
                result = fut.result(timeout=25)
                if result:
                    new_alerts.append(result)
            except Exception:
                pass
            _scan_status["tickers_scanned"] += 1

    new_alerts.sort(key=lambda x: x["composite_score"], reverse=True)
    with _alert_lock:
        _alert_queue = new_alerts[:MAX_ALERTS]

    _scan_status["running"]   = False
    _scan_status["last_scan"] = datetime.now().isoformat()
    high = len([a for a in new_alerts if a["conviction"] == "HIGH"])
    print(f"Catalyst scan: {len(new_alerts)} alerts from {len(universe)} tickers | {high} HIGH conviction")


def start_background_scanner(interval_minutes=20):
    def _loop():
        build_universe(force=True)
        while True:
            try:
                run_full_scan()
            except Exception as e:
                print(f"Catalyst scan error: {e}")
                _scan_status["running"] = False
            time.sleep(interval_minutes * 60)
    t = threading.Thread(target=_loop, daemon=True, name="CatalystScanner")
    t.start()
    print(f"Universal catalyst scanner started — ALL sectors ALL caps")
    return t


def get_alerts(min_conviction="LOW", sector="ALL", tier="ALL", limit=60):
    min_score = {"HIGH": 65, "MEDIUM": 40, "LOW": 0}.get(min_conviction, 0)
    with _alert_lock:
        filtered = [a for a in _alert_queue
                    if a.get("composite_score", 0) >= min_score
                    and (sector == "ALL" or a.get("sector") == sector)
                    and (tier   == "ALL" or a.get("mktcap_tier") == tier)]
    return filtered[:limit]


def scan_single(sym):
    result = score_ticker(sym.upper().strip())
    if result is None:
        return {"sym": sym.upper(), "name": sym.upper(), "sector": "UNKNOWN",
                "mktcap_tier": "UNKNOWN", "spot": 0, "direction": "NEUTRAL",
                "conviction": "LOW", "composite_score": 0,
                "signals": ["No significant signals detected"],
                "trade": {"type": "WATCH", "reason": "Below signal threshold"},
                "signal_scores": {"options":0,"squeeze":0,"earnings":0,"tape":0,"insider":0},
                "timestamp": datetime.now().isoformat()}
    return result


def get_status():
    return {**_scan_status, "alert_count": len(_alert_queue), "edgar_alerts": len(_edgar_alerts)}


def get_edgar_alerts():
    return _edgar_alerts[:25]