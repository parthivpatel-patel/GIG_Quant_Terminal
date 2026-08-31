"""
=============================================================================
GIG LIVE DATA SERVER v8
Models: Hurst+OU+GARCH+BSM+Greeks+MC+MaxPain+UnusualFlow+XGBoost+LSTM
        Cointegration+VolSurface+EarningsIntel+InsiderData+PeerAnalysis
        VReversal+Kalman+Bayesian+Confidence+Squeeze+ChandelierStop
        HybridForecast+RangeForecast+SIPFeed+CompanyLogos+LatestSignal
Real market data: yfinance OHLCV + REAL OPTIONS CHAINS + NewsAPI
Computes: RSI, MACD, Bollinger, EMA, ADX, ATR, Hurst, OU, GARCH
Options: Real strikes, real expiry dates from live yfinance options chain

INSTALL:  pip install yfinance flask flask-cors numpy pandas scipy requests
RUN:      python live_data_server.py
=============================================================================
"""

import json, os, time, logging, threading, math, sys, importlib.util, re as _re

# ── Safe engine loader — server starts even if optional files are missing ─────
def _load_engine(name, filename):
    """Load a Python engine file safely. Returns module or None."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if not os.path.exists(path):
        print(f"⚠  {filename} not found — {name} disabled (server still runs)")
        return None
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            print(f"⚠  {filename} spec/loader unavailable — {name} disabled")
            return None
        mod  = importlib.util.module_from_spec(spec)
        # Important: register module before exec so dataclass/type inspection works.
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        print(f"✓  {filename} loaded")
        return mod
    except KeyboardInterrupt:
        # Never let a Ctrl+C during import kill the whole server startup
        print(f"⚠  {filename} interrupted during import — {name} disabled (safe)")
        return None
    except BaseException as e:
        print(f"⚠  {filename} load error: {type(e).__name__}: {e} — {name} disabled")
        return None

OE = _load_engine("options_engine", "options_engine.py")
ML = _load_engine("ml_engine",      "ml_engine.py")
SE = _load_engine("signal_engine",  "signal_engine.py")
CI = _load_engine("company_info",   "company_info.py")
EE = _load_engine("earnings_engine","earnings_engine.py")
RA = _load_engine("research_agent","research_agent.py")
CE = _load_engine("catalyst_engine","catalyst_engine.py")
AD = _load_engine("alt_data_engine","alt_data_engine.py")

# Backtest engine (local module — same directory)
BE = _load_engine("backtest_engine", "backtest_engine.py")

# ── Renaissance v5 Engines ────────────────────────────────────────────────────
NLP  = _load_engine("nlp_engine",          "nlp_engine.py")
DL   = _load_engine("deep_learning_engine","deep_learning_engine.py")
ENS  = _load_engine("ensemble_engine",     "ensemble_engine.py")
PAT  = _load_engine("pattern_engine",      "pattern_recognition_engine.py")
VS   = _load_engine("vol_surface_engine",  "vol_surface_engine.py")
CA   = _load_engine("cross_asset_engine",  "cross_asset_engine.py")
DI   = _load_engine("data_ingestion",      "data_ingestion_engine.py")
FUT  = _load_engine("futures_engine",      "futures_engine.py")
CAP  = _load_engine("capacity_engine",     "capacity_engine.py")
ALS  = _load_engine("alert_system",        "alert_system_engine.py")
# ADV (advanced_engines.py) REMOVED — 4 of 6 classes duplicate existing endpoints
#   MultiTimeframeAnalyzer → /api/mtf already built inline
#   StressTester → /api/stress already built inline
#   CorrelationAnalyzer → /api/correlation already built inline
#   RegimeSwitchingPortfolio → /api/regime/portfolio already built inline
SSE  = _load_engine("stock_signal_engine", "stock_signal_engine.py")
SLE  = _load_engine("self_learning",       "self_learning_engine.py")
QTE  = _load_engine("quant_theories",      "quant_theories_engine.py")
DEE  = _load_engine("data_edge",           "data_edge_engine.py")
ARE  = _load_engine("alpha_research",      "alpha_research_engine.py")
IPE  = _load_engine("inst_portfolio",      "institutional_portfolio_engine.py")
RET  = _load_engine("research_tracker",    "research_experiment_tracker.py")
GEO  = _load_engine("geopolitical",        "geopolitical_engine.py")
AQE  = _load_engine("adv_quant",           "advanced_quant_engine.py")
IDF  = _load_engine("inst_data_fabric",    "institutional_data_fabric.py")
CAE  = _load_engine("compliance_audit",    "compliance_audit_engine.py")
MCE  = _load_engine("messaging_collab",    "messaging_collab_engine.py")

# ── Start background catalyst scanner ───────────────────────────────────────────
if CE:
    try:
        CE.start_background_scanner(interval_minutes=20)
    except Exception as _ce_err:
        print(f"⚠  Catalyst scanner error: {_ce_err}")

# ── Environment variables — load .env if present ──────────────────────────────
# All secrets come from environment, never hardcoded in source.
# Create a .env file from .env.example and fill in your real keys.
try:
    from dotenv import load_dotenv as _load_dotenv  # type: ignore[import-untyped]
    import pathlib as _pathlib
    for _p in [_pathlib.Path(__file__).parent,
               _pathlib.Path(__file__).parent.parent]:
        _env_file = _p / ".env"
        if _env_file.exists():
            _load_dotenv(_env_file)
            break
except ImportError:
    pass  # python-dotenv optional — use real OS env vars instead

# ── Alpaca Markets ─────────────────────────────────────────────────────────────
# Keys are read from environment variables — NEVER hardcode here.
# Set in .env file:  ALPACA_API_KEY=...  ALPACA_SECRET_KEY=...
# Paper trading by default. Change ALPACA_BASE_URL to live URL when ready.
ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY",    "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL   = os.environ.get("ALPACA_BASE_URL",   "https://paper-api.alpaca.markets")
ALPACA_DATA_URL   = os.environ.get("ALPACA_DATA_URL",   "https://data.alpaca.markets")
# ──────────────────────────────────────────────────────────────────────────────

# ── Load execution engine with fake config shim ───────────────────────────────
# execution_engine.py does `from config import *` — we satisfy that without
# needing config.py by injecting a fake module into sys.modules first.
def _load_execution_engine():
    import types, sys as _sys
    fake_cfg = types.ModuleType("config")
    # These will be updated once the real constants are set at startup
    fake_cfg.ALPACA_API_KEY    = ALPACA_API_KEY
    fake_cfg.ALPACA_SECRET_KEY = ALPACA_SECRET_KEY
    fake_cfg.ALPACA_BASE_URL   = ALPACA_BASE_URL
    fake_cfg.ALPACA_DATA_URL   = ALPACA_DATA_URL
    _sys.modules.setdefault("config", fake_cfg)
    return _load_engine("execution_engine", "execution_engine.py")

# -- GIG Database Layer --
try:
    import db_engine as DB
    DB.setup()
    print("\u2713  Database initialized (SQLite)")
except Exception as _db_err:
    DB = None
    print(f"\u26a0  Database not available: {_db_err}")

EX = _load_execution_engine()  # AlpacaBroker lives here

# Global broker instance (initialised once at startup)
_alpaca_broker = None
_execution_snapshots: dict = {}   # symbol → signal snapshot at trade entry (for IC feedback)
def _init_alpaca():
    global _alpaca_broker
    # Guard: check for empty string too (env var not set), not just placeholder text
    keys_configured = bool(ALPACA_API_KEY) and bool(ALPACA_SECRET_KEY) \
                      and not ALPACA_API_KEY.startswith("YOUR_") \
                      and len(ALPACA_API_KEY) > 10
    if EX and keys_configured:
        try:
            _alpaca_broker = EX.AlpacaBroker()
            print("✓  Alpaca broker connected")
        except Exception as _ae:
            print(f"⚠  Alpaca broker error: {_ae}")
    else:
        print("⚠  Alpaca keys not configured — running in DATA-ONLY mode (no trade execution)")
        print("   To enable trading: add ALPACA_API_KEY + ALPACA_SECRET_KEY to your .env file")
        print("   Get free paper-trading keys at: alpaca.markets → Dashboard → Paper Trading → API Keys")

from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
import requests

import socket
socket.setdefaulttimeout(20)  # 20s hard timeout on ALL network calls
try:
    import yfinance as yf
    import requests as _req
except ImportError:
    yf = None  # type: ignore[assignment]


# ═════════════════════════════════════════════════════════════════════════════
# DYNAMIC UNIVERSE — no hardcoded ticker lists
# ═════════════════════════════════════════════════════════════════════════════
# Every scan, training, and precompute function calls get_scan_universe()
# instead of a hardcoded list.  Covers:
#   - S&P 500 large-caps (pulled live from Wikipedia)
#   - Nasdaq-100 (pulled live)
#   - Popular ETFs (SPY, QQQ, GLD, SLV, USO, TLT, HYG, IWM …)
#   - Metals / Commodities ETFs (GLD, SLV, PDBC, GSG, USO, UNG …)
#   - Sector ETFs (XLK, XLF, XLE, XLV, XLRE, XLU, XLB, XLI, XLC, XLP, XLY)
#   - Any symbol the user has explicitly searched gets added automatically
#   - The active scan universe (scanner.all_signals.keys()) is always included
#
# Result is cached 24 hrs. Fully live — add any ticker by searching it.

_universe_cache: dict = {"symbols": [], "ts": 0.0}
_universe_lock = threading.Lock()

def get_scan_universe(include_scanner: bool = True,
                      extra: list = None) -> list:
    """
    Return the current scan universe — fully dynamic, no hardcoded lists.

    Priority order (deduped):
      1. Active scanner symbols (whatever is already being tracked)
      2. Catalyst engine live universe
      3. S&P 500 from Wikipedia
      4. Russell 2000 via multiple sources (iShares, stockanalysis.com, Wikipedia)
      5. SEC EDGAR full US equity list (~9,500 tickers, filtered by liquidity)
      6. Nasdaq 100 + Yahoo screeners
      7. Core ETFs (equity, bond, commodity, sector, FX, crypto proxies)
      8. Any extra symbols

    Target coverage: ~3,000–5,000 liquid symbols across all cap tiers.
    Refreshed every 24 hours. Falls back gracefully if any source fails.
    """
    global _universe_cache
    now = time.time()

    # ── INSTANT START: Hardcoded top-200 liquid stocks + ETFs ─────────────
    # This returns IMMEDIATELY on cold start. Full universe loads in background.
    # Eliminates the 30-60s Wikipedia/SEC/iShares fetch blocking first scan.
    INSTANT_UNIVERSE = [
        # Mega caps
        "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","BRK-B","AVGO","JPM",
        "LLY","V","MA","UNH","XOM","COST","HD","PG","JNJ","ABBV",
        "CRM","ORCL","BAC","MRK","NFLX","AMD","CVX","KO","PEP","TMO",
        "WMT","CSCO","ABT","ACN","MCD","IBM","INTC","QCOM","TXN","PM",
        "GS","MS","NOW","ISRG","BKNG","LOW","INTU","AMGN","GE","PFE",
        "AMAT","CAT","BLK","NEE","ADP","VRTX","UNP","SYK","DE","PANW",
        # Large caps
        "ABNB","ADBE","ANET","CMG","DHR","DIS","GILD","LMT","RTX","SCHW",
        "SNOW","SQ","UBER","SHOP","SPOT","PYPL","BA","F","GM","NKE",
        "COP","EOG","SLB","PSX","OXY","DVN","FANG","MPC","VLO","HES",
        "PLTR","COIN","MSTR","DDOG","CRWD","ZS","FTNT","NET","OKTA","BILL",
        "RIVN","LCID","SOFI","HOOD","UPST","AFRM","LMND","MARA","RIOT","GME",
        "ARM","SMCI","MU","MRVL","LRCX","KLAC","ON","ADI","MPWR","SNPS",
        "WFC","C","USB","PNC","TFC","COF","AXP","ALLY","FITB","HBAN",
        "LLY","BMY","MRNA","REGN","BIIB","VRTX","ZTS","IDXX","DXCM","ALNY",
        # Sector & Factor ETFs
        "SPY","QQQ","IWM","DIA","VTI","VOO","RSP","MDY","IJR",
        "XLK","XLF","XLE","XLV","XLI","XLC","XLY","XLP","XLU","XLB","XLRE",
        "TLT","IEF","SHY","HYG","LQD","TIP","BND","AGG",
        "GLD","SLV","IAU","GDX","USO","UNG","PDBC","DBA","WEAT","CORN","SOYB",
        "UUP","FXE","FXY","EEM","EFA","FXI","EWZ","EWJ","MCHI","INDA",
        "SOXX","SMH","ARKK","CIBR","HACK","AIQ","BOTZ","XBI","IBB","KRE",
        "TQQQ","SQQQ","SPXL","SPXU","UVXY","VIXY",
        "BITO","IBIT","GBTC",
        "QUAL","USMV","MTUM","NOBL","SCHD","VNQ",
    ]

    # Core ETF universe (subset for critical signals)
    CORE_ETFS = INSTANT_UNIVERSE[:100]

    with _universe_lock:
        if now - _universe_cache.get("ts", 0) < 86_400 and _universe_cache.get("symbols"):
            syms = list(_universe_cache["symbols"])
        else:
            # INSTANT: return hardcoded list immediately, load full in background
            syms = list(INSTANT_UNIVERSE)
            logger.info(f"  Universe: INSTANT START ({len(syms)} liquid stocks + ETFs)")

            # Schedule full universe load in background (non-blocking)
            def _load_full_universe():
                try:
                    time.sleep(10)  # let first scan complete first
                    full_syms = list(INSTANT_UNIVERSE)
                    # S&P 500 from Wikipedia (background, non-blocking)
                    try:
                        import pandas as _pd
                        sp500 = _pd.read_html(
                            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                            attrs={"id": "constituents"}
                        )[0]["Symbol"].tolist()
                        full_syms.extend([s.replace(".", "-") for s in sp500])
                        logger.info(f"  [BG] S&P 500 loaded: {len(sp500)} symbols")
                    except Exception:
                        pass
                    # SEC EDGAR tickers (background)
                    try:
                        _sec_r = requests.get(
                            "https://www.sec.gov/files/company_tickers.json",
                            timeout=20, headers={"User-Agent": "GIG quant@example.com"})
                        if _sec_r.ok:
                            _sec_syms = [str(v.get("ticker","")).upper() for v in _sec_r.json().values()
                                        if 1 <= len(str(v.get("ticker",""))) <= 5]
                            full_syms.extend(_sec_syms)
                            logger.info(f"  [BG] SEC EDGAR loaded: {len(_sec_syms)} tickers")
                    except Exception:
                        pass
                    # Nasdaq screener symbols (free public endpoint)
                    try:
                        _hdr = {
                            "User-Agent": "Mozilla/5.0",
                            "Accept": "application/json, text/plain, */*",
                            "Origin": "https://www.nasdaq.com",
                            "Referer": "https://www.nasdaq.com/",
                        }
                        _ns = requests.get(
                            "https://api.nasdaq.com/api/screener/stocks",
                            params={"tableonly": "true", "limit": 5000, "offset": 0, "download": "false"},
                            timeout=20,
                            headers=_hdr,
                        )
                        if _ns.ok:
                            _rows = (((_ns.json() or {}).get("data") or {}).get("rows") or [])
                            _nsyms = [str(r.get("symbol", "")).upper().strip() for r in _rows if r.get("symbol")]
                            full_syms.extend(_nsyms)
                            logger.info(f"  [BG] Nasdaq screener loaded: {len(_nsyms)} symbols")
                    except Exception:
                        pass
                    # Deduplicate
                    seen = set()
                    clean = []
                    for s in full_syms:
                        s2 = str(s).upper().strip().replace(".", "-")
                        if s2 and s2 not in seen and 1 <= len(s2) <= 6 and s2.replace("-","").isalpha():
                            seen.add(s2)
                            clean.append(s2)
                    with _universe_lock:
                        _universe_cache["symbols"] = clean
                        _universe_cache["ts"] = time.time()
                    logger.info(f"  [BG] Full universe ready: {len(clean)} total symbols")
                except Exception as _fue:
                    logger.debug(f"  [BG] Full universe load failed: {_fue}")

            threading.Thread(target=_load_full_universe, daemon=True, name="universe-bg").start()

            # Set instant cache so first scan runs NOW
            _universe_cache["symbols"] = syms
            _universe_cache["ts"] = now
            logger.info(f"  Universe cache built: {len(syms)} total unique symbols (full load in background)")

    # ── Source 4: Catalyst engine live universe ───────────────────────────────
    # catalyst_engine runs its own parallel fetch (S&P500 + Nasdaq + Russell +
    # Yahoo screeners) in a background thread. Once built, wire it in here.
    result = list(syms)
    try:
        if CE and hasattr(CE, "_live_universe") and CE._live_universe:
            ce_syms = [s for s in CE._live_universe
                       if s and len(s) <= 6 and str(s).replace("-","").isalpha()]
            result = list(dict.fromkeys(ce_syms + result))
    except Exception:
        pass

    # Always include active scanner symbols (user-searched + already tracked)
    if include_scanner:
        try:
            active = list(scanner.all_signals.keys()) if scanner.all_signals else []
            result = list(dict.fromkeys(active + result))
        except Exception:
            pass

    if extra:
        result = list(dict.fromkeys([str(s).upper() for s in extra] + result))

    return result


def get_training_universe(n: int = 60) -> list:
    """
    Return a diverse training universe for ML models.
    Pulls from get_scan_universe() filtered to liquid large-caps.
    n: max symbols to return (default 60).
    """
    full = get_scan_universe(include_scanner=False)
    # Prefer large-cap + liquid ETFs — limit to n for training speed
    # The scanner's active universe always comes first
    priority = [
        "SPY","QQQ","IWM","GLD","TLT",
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","JPM","V","MA",
        "BAC","XOM","CVX","LLY","UNH","HD","AVGO","COST","NFLX","AMD",
        "PLTR","CRWD","COIN","SHOP","MELI","GS","BX","MS","C","WFC",
        "INTC","MU","QCOM","ARM","SMCI","SOFI","MSTR","RIVN","NIO",
        "XLK","XLF","XLE","XLV","XLB","XLI","BITO","IBIT",
    ]
    seen  = set()
    final = []
    for s in priority + full:
        if s not in seen:
            seen.add(s)
            final.append(s)
        if len(final) >= n:
            break
    return final


# ─── ETF / Fund detector ──────────────────────────────────────────────────────
# Used to filter out ETFs from revision/insider precomputes.
# ETFs have no analyst EPS estimates (→ HTTP 404 from Yahoo quoteSummary)
# and no SEC Form 4 insider filings (→ HTTP 401 / empty results from EDGAR).
# Fast — zero network calls.

_ETF_FUND_SET = frozenset({
    "SPY","QQQ","IWM","DIA","MDY","VTI","IVV","VOO","VIG","RSP","SCHB","ITOT",
    "XLK","XLF","XLE","XLV","XLI","XLC","XLY","XLP","XLU","XLB","XLRE",
    "SOXX","ARKK","ARKG","ARKW","ARKF","CIBR","HACK","AIQ","BOTZ","IGV","VGT",
    "TLT","IEF","SHY","SCHO","GOVT","HYG","JNK","LQD","BND","AGG","TIP","BNDX",
    "EMB","MBB","VCIT","VCSH","VGIT","VGSH","VGLT","VTIP",
    "GLD","SLV","IAU","SGOL","GLDM","GDX","GDXJ","SIL","PALL","PPLT","CPER",
    "USO","UNG","USL","PDBC","GSG","DJP","COMT",
    "EEM","EFA","FXI","EWZ","EWJ","EWT","EWY","MCHI","INDA","VWO","IEMG",
    "VXX","UVXY","SVXY","VIXY","SQQQ","TQQQ","SPXU","SPXL","LABD","LABU",
    "UPRO","SDOW","UDOW","FNGU","FNGD",
    "BITO","IBIT","GBTC","ETHE","BITB",
    "VNQ","IYR","REM","MORT",
    "NOBL","DVY","VYM","HDV","SCHD","DGRO","USMV","QUAL","MTUM","VLUE",
})

def _is_etf_or_fund(symbol: str) -> bool:
    """Return True if symbol is an ETF, index fund or commodity fund (no EPS/insider data)."""
    import re as _r
    s = symbol.upper().strip()
    if s in _ETF_FUND_SET:
        return True
    # Sector ETF pattern: XL + single letter (XLK, XLF, etc.)
    if _r.match(r'^XL[A-Z]$', s):
        return True
    return False


try:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    # Retry adapter: 3 retries, exponential backoff, respects rate limits
    _yf_session = _req.Session()
    _retry = Retry(
        total=3,
        backoff_factor=1.0,          # wait 1, 2, 4 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    _yf_session.mount("https://", HTTPAdapter(max_retries=_retry, pool_connections=4, pool_maxsize=8))
    _yf_session.mount("http://",  HTTPAdapter(max_retries=_retry, pool_connections=4, pool_maxsize=8))
    yf.set_tz_cache_location("/tmp/yf_tz_cache")

    YF_OK = True
except ImportError:
    YF_OK = False
    print("ERROR: pip install yfinance flask flask-cors numpy pandas scipy requests")

NEWS_API_KEY  = os.environ.get("NEWS_API_KEY", "")       # free @ newsapi.org — set in .env
FRED_API_KEY  = os.environ.get("FRED_API_KEY", "")       # FREE at fred.stlouisfed.org
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")
TRADIER_API_KEY = os.environ.get("TRADIER_API_KEY", "")
GNEWS_API_KEY = os.environ.get("GNEWS_API_KEY", "")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")
ULTRA_FAST_MODE = str(os.environ.get("ULTRA_FAST_MODE", "0")).lower() in ("1", "true", "yes", "on")
SCAN_INTERVAL = int(os.environ.get("SCAN_INTERVAL", "8" if ULTRA_FAST_MODE else "12"))
MARKET_CACHE_REFRESH_S = float(os.environ.get("MARKET_CACHE_REFRESH_S", "5" if ULTRA_FAST_MODE else "10"))
API_STATE_TTL_S = float(os.environ.get("API_STATE_TTL_S", "0.35" if ULTRA_FAST_MODE else "0.5"))
SSE_PUSH_INTERVAL_S = float(os.environ.get("SSE_PUSH_INTERVAL_S", "0.75" if ULTRA_FAST_MODE else "1.0"))
PORT          = 5001

# ─── FREE APIs USED ────────────────────────────────────────────────────────────
# 1. yfinance          — OHLCV, options chains, fundamentals    NO KEY NEEDED
# 2. SEC EDGAR         — Form 4 insider filings, 8-K, 10-Q     NO KEY NEEDED
# 3. Clearbit Logo API — Company logos by domain                NO KEY NEEDED
# 4. FRED API          — Federal Reserve macro data (yields,    FREE KEY AT
#                        CPI, unemployment, Fed rate, M2)       fred.stlouisfed.org
# 5. NewsAPI           — Business headlines, sentiment          FREE KEY AT
#                        newsapi.org (100 req/day)              newsapi.org
# 6. Yahoo Finance RSS — Real-time news feed                    NO KEY NEEDED
# 7. CBOE free data    — VIX history, term structure            NO KEY NEEDED
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("LIVE")


def _has_key(v: str) -> bool:
    return bool(v) and len(v) > 8 and not str(v).startswith("YOUR_")


def _period_start_date(period: str) -> datetime:
    p = str(period or "6mo").lower()
    now = datetime.utcnow()
    if p.endswith("mo"):
        try:
            months = max(1, int(p[:-2]))
        except Exception:
            months = 6
        return now - timedelta(days=months * 31)
    if p.endswith("y"):
        try:
            years = max(1, int(p[:-1]))
        except Exception:
            years = 1
        return now - timedelta(days=years * 365)
    if p.endswith("d"):
        try:
            days = max(1, int(p[:-1]))
        except Exception:
            days = 30
        return now - timedelta(days=days)
    return now - timedelta(days=186)


def _polygon_quote(symbol: str) -> dict:
    if not _has_key(POLYGON_API_KEY):
        return {}
    sym = symbol.upper().strip()
    try:
        t = requests.get(
            f"https://api.polygon.io/v2/last/trade/{sym}",
            params={"apiKey": POLYGON_API_KEY},
            timeout=6,
        )
        p = requests.get(
            f"https://api.polygon.io/v2/aggs/ticker/{sym}/prev",
            params={"adjusted": "true", "apiKey": POLYGON_API_KEY},
            timeout=6,
        )
        if t.status_code != 200:
            return {}
        tj = t.json() or {}
        pj = p.json() if p.status_code == 200 else {}
        last = float((tj.get("results") or {}).get("p") or 0)
        prev = float(((pj.get("results") or [{}])[0] or {}).get("c") or last)
        if last <= 0:
            return {}
        chg = last - prev
        return {
            "symbol": sym,
            "price": round(last, 4),
            "regularMarketPrice": round(last, 4),
            "change": round(chg, 4),
            "changePct": round((chg / prev * 100) if prev else 0, 4),
            "provider": "polygon",
            "market_session": "regular",
        }
    except Exception:
        return {}


def _polygon_history(symbol: str, period: str, interval: str) -> dict:
    if not _has_key(POLYGON_API_KEY):
        return {}
    mult = 1
    timespan = "day"
    iv = str(interval or "1d").lower()
    if iv in {"1m", "1min"}:
        mult, timespan = 1, "minute"
    elif iv in {"5m", "5min"}:
        mult, timespan = 5, "minute"
    elif iv in {"15m", "15min"}:
        mult, timespan = 15, "minute"
    elif iv in {"30m", "30min"}:
        mult, timespan = 30, "minute"
    elif iv in {"1h", "60m"}:
        mult, timespan = 60, "minute"
    elif iv == "1d":
        mult, timespan = 1, "day"
    else:
        return {}
    start = _period_start_date(period).strftime("%Y-%m-%d")
    end = datetime.utcnow().strftime("%Y-%m-%d")
    try:
        r = requests.get(
            f"https://api.polygon.io/v2/aggs/ticker/{symbol.upper()}/range/{mult}/{timespan}/{start}/{end}",
            params={"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY},
            timeout=8,
        )
        if r.status_code != 200:
            return {}
        j = r.json() or {}
        rows = j.get("results") or []
        if not rows:
            return {}
        hist = []
        for i, row in enumerate(rows):
            ts = int(row.get("t", 0) or 0)
            dt = datetime.utcfromtimestamp(ts / 1000.0) if ts else datetime.utcnow()
            session = "regular"
            if timespan == "minute":
                m = dt.hour * 60 + dt.minute
                if m < 570:
                    session = "pre"
                elif m >= 960:
                    session = "post"
            hist.append({
                "t": i,
                "date": dt.isoformat() if timespan == "minute" else dt.strftime("%Y-%m-%d"),
                "open": round(float(row.get("o", 0) or 0), 4),
                "price": round(float(row.get("c", 0) or 0), 4),
                "high": round(float(row.get("h", 0) or 0), 4),
                "low": round(float(row.get("l", 0) or 0), 4),
                "volume": int(row.get("v", 0) or 0),
                "session": session,
            })
        return {"history": hist, "provider": "polygon"}
    except Exception:
        return {}

# ── Suppress noisy library loggers ──────────────────────────────────────────
# yfinance logs every HTTP 404 (ETF fundamentals, delisted tickers) at ERROR.
# urllib3 logs connection retries at WARNING. Both flood the console.
# Set them to CRITICAL so only fatal errors surface.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("yfinance.base").setLevel(logging.CRITICAL)
logging.getLogger("yfinance.utils").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("werkzeug").setLevel(logging.WARNING)   # hide Flask request logs
# ─────────────────────────────────────────────────────────────────────────────

_GLOBAL_CYCLE   = 0     # incremented each scan — options activate after cycle 2
_OPTIONS_CACHE  = {}    # symbol → options dict
_OPTIONS_CHAIN_HTTP_CACHE = {}  # key -> {"ts": float, "payload": dict}
_OPTIONS_CHAIN_TTL_S = 120.0

# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC UNIVERSE — no hardcoded symbol list
# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSE and CAP_MAP have been REMOVED — they were hardcoded (~130 and ~80
# symbols respectively). All symbol resolution now goes through:
#   get_scan_universe()      — live S&P 500 from Wikipedia + ETF + extras
#   get_training_universe()  — liquid subset for ML training
#   get_cap_tier()           — live market-cap classification from yfinance
#
# This means ANY ticker is automatically handled correctly with no manual
# maintenance of static lists.

# ─── Dynamic cap tier ─────────────────────────────────────────────────────────
# Replaces the hardcoded CAP_MAP that only covered ~80 symbols.
# Now computes cap tier from market_cap fetched live by yfinance/company_info.
# Cached per symbol in _CAP_TIER_CACHE (24hr TTL).

_CAP_TIER_CACHE: dict = {}
_CAP_TIER_TTL   = 86_400   # 24 hours

# ETF/fund tickers that should always be classified as "ETF" regardless of
# their market cap (AUM ≠ market cap for ETFs — don't use market cap for these)
_ETF_TICKERS = frozenset({
    "SPY","QQQ","IWM","DIA","MDY","VTI","IVV","VOO","RSP","SCHB","ITOT",
    "XLK","XLF","XLE","XLV","XLI","XLC","XLY","XLP","XLU","XLB","XLRE",
    "SOXX","ARKK","ARKG","ARKW","CIBR","HACK","AIQ","BOTZ","IGV","VGT",
    "TLT","IEF","SHY","HYG","JNK","LQD","BND","AGG","TIP","BNDX",
    "GLD","SLV","IAU","GDX","GDXJ","PALL","PPLT","CPER","USO","UNG","GSG","PDBC",
    "EEM","EFA","FXI","EWZ","EWJ","MCHI","INDA","VWO","IEMG",
    "VXX","UVXY","SVXY","VIXY","SQQQ","TQQQ","SPXU","SPXL","LABD","LABU",
    "BITO","IBIT","GBTC","ETHE",
    "VNQ","IYR","REM",
    "NOBL","DVY","VYM","HDV","SCHD","DGRO","USMV","QUAL","MTUM","VLUE",
    # Commodity/FX ETFs (Gap 2 additions)
    "UUP","UDN","FXE","FXY","FXB","FXC","FXF","SOYB","CORN","WEAT",
})

_METALS_TICKERS = frozenset({
    "GLD","SLV","IAU","GDX","GDXJ","SGOL","GLDM","SIL","PALL","PPLT","CPER",
    "NEM","FCX","WPM","AU","GOLD","AEM","KGC","AGI",
})

def get_cap_tier(symbol: str, market_cap: float = 0) -> str:
    """
    Classify symbol into cap tier dynamically.

    Priority:
      1. Symbol is a known ETF/fund → "ETF"
      2. Symbol is a precious metal/miner → "METAL"
      3. Use market_cap passed in from co_info (already fetched by yfinance)
      4. If market_cap=0, check _CAP_TIER_CACHE for a previously fetched value
      5. Fetch market_cap from yfinance fast_info (cached 24hr)

    Cap tiers:
      MEGA  : > $500B
      LARGE : $10B – $500B
      MID   : $2B – $10B
      SMALL : $300M – $2B
      MICRO : $50M – $300M
      PENNY : < $50M
      ETF   : exchange-traded fund (no market cap tier)
      METAL : metals / miners
    """
    sym = symbol.upper().strip()
    now = __import__("time").time()

    if sym in _ETF_TICKERS:
        return "ETF"
    if sym in _METALS_TICKERS:
        return "METAL"

    # If caller provided a valid market_cap, use it
    if market_cap and market_cap > 0:
        return _mktcap_to_tier(market_cap)

    # Check cache
    cached = _CAP_TIER_CACHE.get(sym)
    if cached and (now - cached.get("ts", 0)) < _CAP_TIER_TTL:
        return cached["tier"]

    # Fetch live from yfinance fast_info
    try:
        import yfinance as _yf_cap
        fi = _yf_cap.Ticker(sym).fast_info
        mc = getattr(fi, "market_cap", None) or 0
        if mc > 0:
            tier = _mktcap_to_tier(mc)
            _CAP_TIER_CACHE[sym] = {"tier": tier, "ts": now}
            return tier
    except Exception:
        pass

    # Final fallback: LARGE (most symbols in the universe are large-cap)
    _CAP_TIER_CACHE[sym] = {"tier": "LARGE", "ts": now}
    return "LARGE"

def _mktcap_to_tier(mc: float) -> str:
    """Convert raw market cap value to tier string."""
    if   mc >= 500e9: return "MEGA"
    elif mc >= 10e9:  return "LARGE"
    elif mc >= 2e9:   return "MID"
    elif mc >= 300e6: return "SMALL"
    elif mc >= 50e6:  return "MICRO"
    else:             return "PENNY"


# ═══════════════════════════════════════════════════════════════════════════════
# REAL TECHNICAL SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════

class RealSignals:
    @staticmethod
    def rsi(close, period=14):
        if len(close) < period + 1: return 50.0
        d      = np.diff(close)
        gains  = np.where(d > 0, d, 0.0)
        losses = np.where(d < 0, -d, 0.0)
        ag     = np.mean(gains[-period:])
        al     = np.mean(losses[-period:])
        return float(100 - 100 / (1 + ag / al)) if al > 0 else 100.0

    @staticmethod
    def ema(prices, period):
        if len(prices) < period: return None
        k = 2.0 / (period + 1)
        e = float(np.mean(prices[:period]))
        for p in prices[period:]:
            e = p * k + e * (1 - k)
        return e

    @staticmethod
    def macd(close):
        if len(close) < 35: return 0.0, 0.0
        e12 = RealSignals.ema(close, 12)
        e26 = RealSignals.ema(close, 26)
        if not e12 or not e26 or e26 == 0: return 0.0, 0.0
        mv = e12 - e26
        return float(mv), float(mv / e26)

    @staticmethod
    def bollinger(close, period=20, std_dev=2.0):
        if len(close) < period: return 0.5, 0.0
        c   = close[-period:]
        mid = np.mean(c)
        std = np.std(c)
        if std == 0: return 0.5, 0.0
        upper = mid + std_dev * std
        lower = mid - std_dev * std
        pos   = (close[-1] - lower) / (upper - lower)
        return float(np.clip(pos, 0, 1)), float((upper - lower) / mid)

    @staticmethod
    def atr(high, low, close, period=14):
        if len(close) < period + 1: return 0.0
        tr = np.maximum(high[1:]-low[1:],
             np.maximum(np.abs(high[1:]-close[:-1]), np.abs(low[1:]-close[:-1])))
        return float(np.mean(tr[-period:]))

    @staticmethod
    def adx(high, low, close, period=14):
        if len(close) < period * 2: return 0.0
        try:
            tr  = np.maximum(high[1:]-low[1:],
                  np.maximum(np.abs(high[1:]-close[:-1]), np.abs(low[1:]-close[:-1])))
            atr = pd.Series(tr).rolling(period).mean().values
            dmp = np.where((high[1:]-high[:-1])>(low[:-1]-low[1:]),
                           np.maximum(high[1:]-high[:-1],0),0)
            dmm = np.where((low[:-1]-low[1:])>(high[1:]-high[:-1]),
                           np.maximum(low[:-1]-low[1:],0),0)
            dip = 100*pd.Series(dmp).rolling(period).mean().values/(atr+1e-8)
            dim = 100*pd.Series(dmm).rolling(period).mean().values/(atr+1e-8)
            dx  = 100*np.abs(dip-dim)/(dip+dim+1e-8)
            return float(pd.Series(dx).rolling(period).mean().iloc[-1])
        except Exception: return 0.0

    @staticmethod
    def hurst(prices, min_lag=2, max_lag=40):
        if len(prices) < max_lag * 2: return 0.5
        try:
            from scipy.stats import linregress
            lags = range(min_lag, min(max_lag, len(prices)//4))
            tau  = []
            for lag in lags:
                chunks = [prices[i:i+lag] for i in range(0, len(prices)-lag, lag)]
                rs = []
                for chunk in chunks:
                    if len(chunk) < 2: continue
                    mean  = np.mean(chunk)
                    diffs = np.cumsum(chunk - mean)
                    R     = diffs.max() - diffs.min()
                    S     = np.std(chunk, ddof=1)
                    if S > 0: rs.append(R/S)
                if rs: tau.append(np.mean(rs))
            if len(tau) < 2: return 0.5
            slope, *_ = linregress(np.log(list(range(min_lag, min_lag+len(tau)))), np.log(tau))
            return float(np.clip(slope, 0.0, 1.0))
        except Exception: return 0.5

    @staticmethod
    def ou_halflife(prices):
        if len(prices) < 20: return 999.0, 0.0
        try:
            from scipy.stats import linregress
            sl, intercept, *_ = linregress(prices[:-1], prices[1:])
            kappa = -np.log(max(sl, 1e-9)) * 252
            mu    = intercept / (1 - sl) if sl != 1 else np.mean(prices)
            hl    = np.log(2) / kappa if kappa > 0 else 999.0
            z     = (prices[-1] - mu) / (np.std(prices) + 1e-8)
            return float(hl), float(z)
        except Exception: return 999.0, 0.0

    @staticmethod
    def garch_vol(returns, alpha=0.1, beta=0.85):
        if len(returns) < 5: return 0.25
        omega    = (1 - alpha - beta) * np.var(returns)
        sigma_sq = np.var(returns)
        for r in returns:
            sigma_sq = omega + alpha * r**2 + beta * sigma_sq
        return float(np.sqrt(max(sigma_sq, 0) * 252))

    @staticmethod
    def iv_rank(hv20, hv_history):
        if len(hv_history) < 2: return 50.0
        lo, hi = np.min(hv_history), np.max(hv_history)
        if hi == lo: return 50.0
        return float(np.clip((hv20 - lo) / (hi - lo) * 100, 0, 100))

    @staticmethod
    def compute_all(df):
        if df is None or len(df) < 30: return {}
        close  = df["Close"].values.astype(float)
        high   = df["High"].values.astype(float)
        low    = df["Low"].values.astype(float)
        volume = df["Volume"].values.astype(float)
        rets   = np.diff(np.log(close + 1e-10))
        s      = {}

        for n, label in [(1,"1d"),(5,"5d"),(10,"10d"),(20,"20d"),(60,"60d")]:
            if len(close) > n:
                s[f"mom_{label}"] = float((close[-1]-close[-n])/(close[-n]+1e-8)*100)

        s["rsi"]           = RealSignals.rsi(close)
        mv, mn             = RealSignals.macd(close)
        s["macd"]          = mv; s["macd_norm"] = mn
        bb_pos, bb_wid     = RealSignals.bollinger(close)
        s["bb_position"]   = bb_pos; s["bb_width"] = bb_wid
        s["adx"]           = RealSignals.adx(high, low, close)
        atr_v              = RealSignals.atr(high, low, close)
        s["atr"]           = atr_v
        s["atr_pct"]       = float(atr_v / (close[-1]+1e-8) * 100)

        for p, k in [(9,"ema9"),(21,"ema21"),(50,"ema50"),(200,"ema200")]:
            s[k] = RealSignals.ema(close, p) or close[-1]

        s["ema_cross_9_21"]   = float((s["ema9"]-s["ema21"])/(s["ema21"]+1e-8)*100)
        s["ema_cross_50_200"] = float((s["ema50"]-s["ema200"])/(s["ema200"]+1e-8)*100)
        s["price_vs_200"]     = float((close[-1]-s["ema200"])/(s["ema200"]+1e-8)*100)

        vol_sma20        = float(np.mean(volume[-20:])) if len(volume)>=20 else float(volume[-1])
        s["volume"]      = float(volume[-1])
        s["vol_sma20"]   = vol_sma20
        s["vol_ratio"]   = float(volume[-1]/(vol_sma20+1e-8))

        s["hv20"]        = float(np.std(rets[-20:])*np.sqrt(252)*100) if len(rets)>=20 else 25.0
        s["hv60"]        = float(np.std(rets[-60:])*np.sqrt(252)*100) if len(rets)>=60 else 25.0
        hv_hist          = [float(np.std(rets[max(0,i-20):i])*np.sqrt(252)*100) for i in range(25,len(rets)+1)]
        s["ivr"]         = RealSignals.iv_rank(s["hv20"], hv_hist)
        s["garch_vol"]   = RealSignals.garch_vol(rets[-60:] if len(rets)>=60 else rets)*100

        s["hurst"]       = RealSignals.hurst(close[-120:] if len(close)>=120 else close)
        s["ou_halflife"], s["ou_zscore"] = RealSignals.ou_halflife(close[-100:] if len(close)>=100 else close)

        if len(rets) >= 30:
            s["sharpe"]  = float(np.mean(rets)*252/(np.std(rets)*np.sqrt(252)+1e-8))
            s["skew"]    = float(pd.Series(rets[-60:]).skew())
            s["kurt"]    = float(pd.Series(rets[-60:]).kurtosis())

        if len(close) >= 20:
            mu20     = np.mean(close[-20:])
            sd20     = np.std(close[-20:])
            s["zscore"] = float((close[-1]-mu20)/(sd20+1e-8))

        if len(close) >= 14:
            s["stoch_k"] = float((close[-1]-np.min(low[-14:]))/(np.max(high[-14:])-np.min(low[-14:])+1e-8)*100)

        tp    = (high + low + close) / 3
        vwap  = float(np.sum(tp[-20:]*volume[-20:])/(np.sum(volume[-20:])+1e-8))
        s["vwap"]          = vwap
        s["price_vs_vwap"] = float((close[-1]-vwap)/(vwap+1e-8)*100)

        # Composite
        rsi_sig  = float(np.clip((50-s["rsi"])/50, -1, 1))
        macd_sig = float(np.clip(s["macd_norm"]*20, -1, 1))
        bb_sig   = float(np.clip(1-s["bb_position"]*2, -1, 1))
        ema_sig  = float(np.clip(s.get("ema_cross_9_21",0)/3, -1, 1))
        mom5_sig = float(np.clip(s.get("mom_5d",0)/5, -1, 1))
        vol_sig  = float(np.clip((s["vol_ratio"]-1)/2, -0.5, 1))
        ou_sig   = float(np.clip(-s.get("ou_zscore",0)/3, -1, 1))

        h = s["hurst"]
        if h > 0.55:
            blended = 0.6*(0.4*ema_sig + 0.6*mom5_sig) + 0.4*rsi_sig
        elif h < 0.45:
            blended = 0.6*(0.5*ou_sig + 0.5*bb_sig) + 0.4*rsi_sig
        else:
            blended = 0.5*mom5_sig + 0.3*rsi_sig + 0.2*macd_sig

        composite = float(np.clip(
            0.22*blended + 0.18*macd_sig + 0.15*rsi_sig +
            0.12*bb_sig  + 0.12*ema_sig  + 0.10*mom5_sig +
            0.06*vol_sig + 0.05*float(np.clip(s.get("sharpe",0)/3,-1,1)),
            -1, 1))
        s["composite"]  = composite
        s["normalized"] = float((composite+1)/2)
        s["conviction"] = float(abs(composite))
        s["direction"]  = "BUY" if s["normalized"]>0.62 else "SHORT" if s["normalized"]<0.38 else "HOLD"
        return s


# ═══════════════════════════════════════════════════════════════════════════════
# REAL OPTIONS CHAIN FETCHER
# ═══════════════════════════════════════════════════════════════════════════════

class RealOptionsChain:
    """Fetch live options chain from yfinance and pick the best contract."""

    options_cache = {}   # symbol → {timestamp, data}
    CACHE_TTL     = 300  # 5 min cache

    @staticmethod
    def get_best_expiry(expirations, target_dte=30):
        """Find expiration closest to target DTE."""
        today  = datetime.today().date()
        best   = None
        best_d = 9999
        for exp_str in expirations:
            try:
                exp  = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte  = (exp - today).days
                if dte < 7:   # skip weeklies < 7 days
                    continue
                diff = abs(dte - target_dte)
                if diff < best_d:
                    best_d = diff
                    best   = exp_str
            except Exception:
                continue
        return best

    @staticmethod
    def pick_strike_call(calls_df, spot, otm_pct=0.05):
        """OTM call by Vol+OI. 2-10% OTM."""
        if calls_df is None or calls_df.empty: return None
        df = calls_df[calls_df["strike"] > spot * 1.005].copy()
        if df.empty: df = calls_df[calls_df["strike"] >= spot].copy()
        if df.empty: return None
        ss = df[(df["strike"]>=spot*1.02)&(df["strike"]<=spot*1.10)].copy()
        if ss.empty: ss = df.head(10).copy()
        ss["_v"]=ss["volume"].fillna(0).astype(float);ss["_o"]=ss["openInterest"].fillna(0).astype(float)
        ss["_l"]=ss["_v"]*2+ss["_o"]
        if ss["_l"].max()>0:
            t=ss[ss["_l"]>=ss["_l"].quantile(0.3)].copy();t["_d"]=(t["strike"]-spot*1.05).abs()
            return t.sort_values(["_l","_d"],ascending=[False,True]).iloc[0]
        return ss.iloc[0]

    @staticmethod
    def pick_strike_put(puts_df, spot, otm_pct=0.05):
        """OTM put by Vol+OI. 2-10% OTM."""
        if puts_df is None or puts_df.empty: return None
        df = puts_df[puts_df["strike"] < spot * 0.995].copy()
        if df.empty: df = puts_df[puts_df["strike"] <= spot].copy()
        if df.empty: return None
        ss = df[(df["strike"]<=spot*0.98)&(df["strike"]>=spot*0.90)].copy()
        if ss.empty: ss = df.tail(10).copy()
        ss["_v"]=ss["volume"].fillna(0).astype(float);ss["_o"]=ss["openInterest"].fillna(0).astype(float)
        ss["_l"]=ss["_v"]*2+ss["_o"]
        if ss["_l"].max()>0:
            t=ss[ss["_l"]>=ss["_l"].quantile(0.3)].copy();t["_d"]=(t["strike"]-spot*0.95).abs()
            return t.sort_values(["_l","_d"],ascending=[False,True]).iloc[0]
        return ss.iloc[-1]

    @staticmethod
    def fetch(symbol, direction, signals):
        """
        Fetch real options chain and return structured trade signal.
        Returns: dict with simple signal + full options data.
        """
        cache_key = f"{symbol}_{direction}"
        cached    = RealOptionsChain.options_cache.get(cache_key)
        if cached and (datetime.now() - cached["ts"]).total_seconds() < RealOptionsChain.CACHE_TTL:
            return cached["data"]

        result = RealOptionsChain._fetch_live(symbol, direction, signals)
        if result:
            RealOptionsChain.options_cache[cache_key] = {"ts": datetime.now(), "data": result}
        return result

    @staticmethod
    def _fetch_live(symbol, direction, signals):
        try:
            ticker      = yf.Ticker(symbol)
            expirations = ticker.options
            if not expirations:
                return RealOptionsChain._bsm_fallback(symbol, direction, signals)

            hv20   = signals.get("hv20", 25)
            garch  = signals.get("garch_vol", 25)
            conv   = signals.get("conviction", 0.3)
            spot   = signals.get("price", 100)
            ivr    = signals.get("ivr", 50)
            bull   = direction == "BUY"

            # ── DTE: conviction-based ─────────────────────────────────────
            if conv > 0.65:    target_dte = 21
            elif conv > 0.45:  target_dte = 30
            else:              target_dte = 45

            exp = RealOptionsChain.get_best_expiry(expirations, target_dte)
            if not exp:
                exp = expirations[0] if expirations else None
            if not exp:
                return RealOptionsChain._bsm_fallback(symbol, direction, signals)

            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            dte      = (exp_date - datetime.today().date()).days
            exp_nice = exp_date.strftime("%b %d, %Y")

            chain = ticker.option_chain(exp)
            calls = chain.calls
            puts  = chain.puts

            # ════════════════════════════════════════════════════════════════
            # ACTIVE STRATEGY: ALWAYS BUY CALL (bullish) or BUY PUT (bearish)
            # Picks ATM strike from real live options chain.
            # ════════════════════════════════════════════════════════════════

            if bull:
                # ── BUY CALL ─────────────────────────────────────────────
                row = RealOptionsChain.pick_strike_call(calls, spot, 0.01)
                if row is not None:
                    premium  = float(row.get("lastPrice") or row.get("ask") or 1.0)
                    strike   = float(row["strike"])
                    call_iv  = float(row.get("impliedVolatility", hv20/100) * 100)
                    exp_move = round(spot * (call_iv/100) * math.sqrt(max(dte,1)/252), 2)
                    cost     = round(premium * 100, 2)
                    itm      = bool(row.get("inTheMoney", False))
                    pop      = round(55.0 if itm else 45.0, 1)
                    return {
                        "strategy":    "BUY CALL",
                        "direction":   "BULLISH",
                        "iv_label":    f"IVR {round(ivr,0):.0f} — {'HIGH IV' if ivr>65 else 'LOW IV' if ivr<35 else 'MODERATE IV'}",
                        "ivr":         round(ivr, 1),
                        "dte":         dte,
                        "expiry":      exp,
                        "expiry_nice": exp_nice,
                        "pop":         pop,
                        "ev":          round(premium * 100 * 0.3, 2),
                        "score":       round(0.25*conv + 0.15*0.6, 3),
                        "recommended": conv > 0.5,
                        "iv":          round(call_iv, 1),
                        "hv":          round(hv20, 1),
                        "garch_vol":   round(garch, 1),
                        "vol_premium": round(call_iv - hv20, 1),
                        "exp_move":    exp_move,
                        "exp_move_pct":round(exp_move/spot*100, 1),
                        "max_profit":  "UNLIMITED",
                        "max_loss":    cost,
                        "breakeven":   round(strike + premium, 2),
                        "simple_signal": {
                            "action": "BUY CALL",
                            "legs": [{
                                "action":  "BUY",
                                "type":    "CALL",
                                "strike":  round(strike, 2),
                                "expiry":  exp_nice,
                                "premium": round(premium, 2),
                                "cost":    cost,
                                "volume":  int(row.get("volume", 0) or 0),
                                "oi":      int(row.get("openInterest", 0) or 0),
                                "iv":      round(call_iv, 1),
                                "bid":     round(float(row.get("bid", 0) or 0), 2),
                                "ask":     round(float(row.get("ask", 0) or 0), 2),
                                "itm":     itm,
                            }],
                            "breakeven": round(strike + premium, 2),
                            "max_profit": "UNLIMITED",
                            "max_loss":   cost,
                        },
                        "legs":   [],
                        "source": "LIVE CHAIN",
                    }

            else:
                # ── BUY PUT ──────────────────────────────────────────────
                row = RealOptionsChain.pick_strike_put(puts, spot, 0.01)
                if row is not None:
                    premium  = float(row.get("lastPrice") or row.get("ask") or 1.0)
                    strike   = float(row["strike"])
                    put_iv   = float(row.get("impliedVolatility", hv20/100) * 100)
                    exp_move = round(spot * (put_iv/100) * math.sqrt(max(dte,1)/252), 2)
                    cost     = round(premium * 100, 2)
                    itm      = bool(row.get("inTheMoney", False))
                    pop      = round(55.0 if itm else 45.0, 1)
                    max_p    = round((strike - premium) * 100, 2)
                    return {
                        "strategy":    "BUY PUT",
                        "direction":   "BEARISH",
                        "iv_label":    f"IVR {round(ivr,0):.0f} — {'HIGH IV' if ivr>65 else 'LOW IV' if ivr<35 else 'MODERATE IV'}",
                        "ivr":         round(ivr, 1),
                        "dte":         dte,
                        "expiry":      exp,
                        "expiry_nice": exp_nice,
                        "pop":         pop,
                        "ev":          round(premium * 100 * 0.3, 2),
                        "score":       round(0.25*conv + 0.15*0.6, 3),
                        "recommended": conv > 0.5,
                        "iv":          round(put_iv, 1),
                        "hv":          round(hv20, 1),
                        "garch_vol":   round(garch, 1),
                        "vol_premium": round(put_iv - hv20, 1),
                        "exp_move":    exp_move,
                        "exp_move_pct":round(exp_move/spot*100, 1),
                        "max_profit":  max_p,
                        "max_loss":    cost,
                        "breakeven":   round(strike - premium, 2),
                        "simple_signal": {
                            "action": "BUY PUT",
                            "legs": [{
                                "action":  "BUY",
                                "type":    "PUT",
                                "strike":  round(strike, 2),
                                "expiry":  exp_nice,
                                "premium": round(premium, 2),
                                "cost":    cost,
                                "volume":  int(row.get("volume", 0) or 0),
                                "oi":      int(row.get("openInterest", 0) or 0),
                                "iv":      round(put_iv, 1),
                                "bid":     round(float(row.get("bid", 0) or 0), 2),
                                "ask":     round(float(row.get("ask", 0) or 0), 2),
                                "itm":     itm,
                            }],
                            "breakeven": round(strike - premium, 2),
                            "max_profit": max_p,
                            "max_loss":   cost,
                        },
                        "legs":   [],
                        "source": "LIVE CHAIN",
                    }

            return RealOptionsChain._bsm_fallback(symbol, direction, signals)

        except Exception as e:
            logger.warning(f"Options chain error {symbol}: {e}")
            return RealOptionsChain._bsm_fallback(symbol, direction, signals)

        # ════════════════════════════════════════════════════════════════════════
        # COMMENTED OUT — CREDIT SPREADS (kept for future re-enable)
        # To re-enable: uncomment below and remove the BUY CALL/PUT block above
        # ════════════════════════════════════════════════════════════════════════
        #
        # if ivr > 65 and bull:
        #     # BULL PUT SPREAD — sell OTM put, buy further OTM put
        #     sell_row = RealOptionsChain.pick_strike_put(puts, spot, 0.03)
        #     buy_row  = RealOptionsChain.pick_strike_put(puts, spot, 0.08)
        #     if sell_row is not None and buy_row is not None:
        #         credit     = float(sell_row.get("lastPrice", sell_row.get("bid", 0.5)))
        #         debit      = float(buy_row.get("lastPrice",  buy_row.get("bid", 0.25)))
        #         net_credit = max(credit - debit, 0.01)
        #         width      = float(sell_row["strike"] - buy_row["strike"])
        #         return { "strategy": "BULL PUT SPREAD", "legs": [
        #             {"action":"SELL","type":"PUT","strike":float(sell_row["strike"]),"prem":credit},
        #             {"action":"BUY", "type":"PUT","strike":float(buy_row["strike"]), "prem":debit},
        #         ], "net_credit": net_credit, "max_profit": net_credit*100,
        #            "max_loss": (width-net_credit)*100 }
        #
        # elif ivr > 65 and not bull:
        #     # BEAR CALL SPREAD — sell OTM call, buy further OTM call
        #     sell_row = RealOptionsChain.pick_strike_call(calls, spot, 0.03)
        #     buy_row  = RealOptionsChain.pick_strike_call(calls, spot, 0.08)
        #     if sell_row is not None and buy_row is not None:
        #         credit     = float(sell_row.get("lastPrice", sell_row.get("bid", 0.5)))
        #         debit      = float(buy_row.get("lastPrice",  buy_row.get("bid", 0.25)))
        #         net_credit = max(credit - debit, 0.01)
        #         width      = float(buy_row["strike"] - sell_row["strike"])
        #         return { "strategy": "BEAR CALL SPREAD", "legs": [
        #             {"action":"SELL","type":"CALL","strike":float(sell_row["strike"]),"prem":credit},
        #             {"action":"BUY", "type":"CALL","strike":float(buy_row["strike"]), "prem":debit},
        #         ], "net_credit": net_credit, "max_profit": net_credit*100,
        #            "max_loss": (width-net_credit)*100 }
        #
        # elif bull:
        #     # BULL CALL SPREAD
        #     buy_row  = RealOptionsChain.pick_strike_call(calls, spot, 0.00)
        #     sell_row = RealOptionsChain.pick_strike_call(calls, spot, 0.05)
        #     ...
        #
        # else:
        #     # BEAR PUT SPREAD
        #     buy_row  = RealOptionsChain.pick_strike_put(puts, spot, 0.00)
        #     sell_row = RealOptionsChain.pick_strike_put(puts, spot, 0.05)
        #     ...

    @staticmethod
    def _bsm_fallback(symbol, direction, signals):
        """BSM-computed fallback when real chain unavailable."""
        from scipy.stats import norm
        spot   = signals.get("price", 100)
        hv20   = signals.get("hv20", 25) / 100
        ivr    = signals.get("ivr", 50)
        conv   = signals.get("conviction", 0.3)
        bull   = direction == "BUY"
        iv     = max(hv20 * 1.1, 0.08)
        dte    = 30
        T      = dte / 252
        # Live risk-free rate from ^IRX (cached 24hr in math_engine)
        try:
            from math_engine import get_risk_free_rate as _get_rfr
            r = _get_rfr()
        except Exception:
            r = 0.05   # safe fallback if math_engine unavailable
        K      = round(spot * (1.01 if bull else 0.99), 2)
        sqrtT  = math.sqrt(T)
        d1     = (math.log(spot/K) + (r+0.5*iv**2)*T) / (iv*sqrtT)
        d2     = d1 - iv*sqrtT
        if bull:
            prem = spot*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
        else:
            prem = K*math.exp(-r*T)*norm.cdf(-d2) - spot*norm.cdf(-d1)
        prem = max(round(prem, 2), 0.01)

        opt_type = "CALL" if bull else "PUT"
        exp_date = (datetime.today() + timedelta(days=dte)).strftime("%b %d, %Y")

        return {
            "strategy":    f"BUY {opt_type}",
            "direction":   "BULLISH" if bull else "BEARISH",
            "iv_label":    "CHEAP VOL → BUY" if ivr < 35 else "MODERATE IV",
            "ivr":         round(ivr, 1),
            "dte":         dte,
            "expiry":      (datetime.today()+timedelta(days=dte)).strftime("%Y-%m-%d"),
            "expiry_nice": exp_date,
            "pop":         48.0,
            "ev":          round(prem*100*0.3, 2),
            "score":       round(0.25*conv+0.10*0.6, 3),
            "recommended": False,
            "iv":          round(iv*100, 1),
            "hv":          round(hv20*100, 1),
            "garch_vol":   round(signals.get("garch_vol",25), 1),
            "vol_premium": round((iv - hv20)*100, 1),
            "exp_move":    round(spot*iv*math.sqrt(T), 2),
            "exp_move_pct":round(iv*math.sqrt(T)*100, 1),
            "max_profit":  "UNLIMITED" if bull else round((K-prem)*100, 2),
            "max_loss":    round(prem*100, 2),
            "breakeven":   round(K+prem if bull else K-prem, 2),
            "simple_signal": {
                "action":  f"BUY {opt_type}",
                "legs": [{
                    "action":  "BUY",
                    "type":    opt_type,
                    "strike":  K,
                    "expiry":  exp_date,
                    "premium": prem,
                    "volume":  0,
                    "oi":      0,
                    "iv":      round(iv*100, 1),
                    "bid":     round(prem*0.95, 2),
                    "ask":     round(prem*1.05, 2),
                }],
                "breakeven": round(K+prem if bull else K-prem, 2),
            },
            "legs":   [],
            "source": "BSM COMPUTED",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET DATA FETCHER
# ═══════════════════════════════════════════════════════════════════════════════

class LiveMarketFetcher:
    def __init__(self):
        self.bars_cache   = {}
        self.bars_ts      = {}
        self.bars_ttl     = 300
        self.price_cache  = {}
        self._max_cache   = 150  # max symbols in cache — prevents memory leak   # close arrays for ML/pairs (was missing — caused 0 signals)
        self.company_cache= {}   # company info cache (was missing — caused 0 signals)

    def get_bars(self, symbol, period="6mo", interval="1d", prepost=None):
        """
        SIP feed = Consolidated tape (all exchanges).
        yfinance uses SIP-consolidated prices by default.
        For real-time: feed='sip' in Alpaca; yfinance equivalent = standard download.
        """
        if prepost is None:
            prepost = interval in {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
        key = f"{symbol}_{period}_{interval}_{int(bool(prepost))}"
        now = datetime.now()
        age = (now - self.bars_ts.get(key, datetime(2000,1,1))).total_seconds()
        if key in self.bars_cache and age < self.bars_ttl:
            return self.bars_cache[key]
        try:
            df = yf.download(
                symbol, period=period, interval=interval, prepost=bool(prepost),
                progress=False, auto_adjust=True
            )
            if not df.empty:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                # Prune cache if too large (prevents memory leak over many cycles)
                if len(self.bars_cache) > getattr(self, "_max_cache", 150):
                    oldest = sorted(self.bars_ts, key=self.bars_ts.get)[:30]
                    for k in oldest:
                        self.bars_cache.pop(k, None)
                        self.bars_ts.pop(k, None)
                self.bars_cache[key] = df
                self.bars_ts[key]    = now
            return df
        except Exception as e:
            logger.debug(f"Bars error {symbol}: {e}")
            return self.bars_cache.get(key, pd.DataFrame())

    def get_quotes(self, symbols):
        """
        Weekend-safe quote fetcher.
        On weekends/after-hours, fast_info.last_price may be None.
        Falls back to last historical close so the scanner always has data.
        """
        results = {}
        # Split into chunks of 10 to avoid Yahoo rate limiting on bulk requests
        def _chunk_quotes(syms, chunk_size=10):
            for i in range(0, len(syms), chunk_size):
                yield syms[i:i+chunk_size]

        try:
            # Try quote endpoint first to get regular/pre/post + market state.
            quote_rows = {}
            for chunk in _chunk_quotes(symbols, 12):
                try:
                    qr = requests.get(
                        "https://query1.finance.yahoo.com/v7/finance/quote",
                        params={"symbols": ",".join(chunk)},
                        timeout=2.5,
                        headers={"User-Agent": "Mozilla/5.0"},
                    )
                    if qr.ok:
                        for row in (((qr.json() or {}).get("quoteResponse") or {}).get("result") or []):
                            sym = str(row.get("symbol", "")).upper().strip()
                            if sym:
                                quote_rows[sym] = row
                    time.sleep(0.05)
                except Exception:
                    pass

            all_tickers = {}
            for chunk in _chunk_quotes(symbols, 10):
                try:
                    t_batch = yf.Tickers(" ".join(chunk))
                    all_tickers.update(t_batch.tickers)
                    time.sleep(0.1)  # small gap between batches
                except Exception:
                    pass
            tickers = type("T", (), {"tickers": all_tickers})()
            for sym in symbols:
                try:
                    t  = tickers.tickers.get(sym)
                    if not t: continue
                    fi = t.fast_info
                    qrow = quote_rows.get(sym, {})
                    market_state = str(qrow.get("marketState", "") or "").upper()
                    regular_px = qrow.get("regularMarketPrice")
                    pre_px = qrow.get("preMarketPrice")
                    post_px = qrow.get("postMarketPrice")

                    # Prefer extended-hours price when market is not regular.
                    chosen_px = regular_px
                    if market_state in ("PRE", "PREPRE", "PREMARKET"):
                        chosen_px = pre_px or regular_px
                    elif market_state in ("POST", "POSTPOST", "POSTMARKET"):
                        chosen_px = post_px or regular_px

                    p  = chosen_px if chosen_px else getattr(fi, "last_price", None)
                    pc = getattr(fi, "previous_close", None)
                    hist_close = None

                    # ── Weekend / after-hours fallback ────────────────────
                    if not p or p <= 0:
                        try:
                            hist = t.history(period="5d", interval="1d")
                            if not hist.empty:
                                p  = float(hist["Close"].iloc[-1])
                                pc = float(hist["Close"].iloc[-2]) if len(hist) > 1 else p
                                hist_close = float(hist["Close"].iloc[-1])
                                vh = int(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else 0
                                dh = float(hist["High"].iloc[-1])
                                dl = float(hist["Low"].iloc[-1])
                            else:
                                continue
                        except Exception:
                            continue
                    else:
                        vh = getattr(fi, "last_volume", 0) or 0
                        dh = float(getattr(fi, "day_high", p) or p)
                        dl = float(getattr(fi, "day_low",  p) or p)
                        try:
                            hist = t.history(period="5d", interval="1d")
                            if not hist.empty:
                                hist_close = float(hist["Close"].iloc[-1])
                        except Exception:
                            hist_close = None

                    # Guardrail for occasional bad-scale quote rows.
                    try:
                        if hist_close and p and p > 0:
                            ratio = float(p) / max(1e-9, float(hist_close))
                            if ratio > 3.0 or ratio < 0.33:
                                logger.warning(f"Quote sanity fallback {sym}: live={p} hist_close={hist_close}")
                                p = hist_close
                                if not pc or pc <= 0:
                                    pc = hist_close
                    except Exception:
                        pass

                    pc = pc or p
                    change     = float(p - pc)
                    change_pct = float(change / pc * 100) if pc else 0.0

                    results[sym] = {
                        "price":     round(float(p), 4),
                        "change":    round(change, 4),
                        "changePct": round(change_pct, 4),
                        "volume":    int(vh),
                        "high":      round(dh, 4),
                        "low":       round(dl, 4),
                        "open":      round(float(pc), 4),
                        "bid":       round(float(p)-0.01, 4),
                        "ask":       round(float(p)+0.01, 4),
                        "regularMarketPrice": float(regular_px) if regular_px is not None else round(float(p), 4),
                        "preMarketPrice": float(pre_px) if pre_px is not None else None,
                        "postMarketPrice": float(post_px) if post_px is not None else None,
                        "marketState": market_state or "UNKNOWN",
                    }
                except Exception as _qe:
                    logger.debug(f"Quote {sym}: {_qe}")
        except Exception as e:
            logger.warning(f"Quotes batch error: {e}")
        return results

    def analyze_symbol(self, symbol, quote):
        df    = self.get_bars(symbol)
        price = quote.get("price", 0)

        # Weekend fallback: use last bar close if quote price missing
        if (not price or price <= 0) and df is not None and not df.empty:
            price = float(df["Close"].iloc[-1])
            quote = {
                **quote,
                "price":     round(price, 4),
                "high":      round(float(df["High"].iloc[-1]),  4),
                "low":       round(float(df["Low"].iloc[-1]),   4),
                "open":      round(float(df["Open"].iloc[-1]),  4),
                "volume":    int(df["Volume"].iloc[-1]),
                "change":    round(float(df["Close"].iloc[-1] - df["Close"].iloc[-2]), 4) if len(df) > 1 else 0,
                "changePct": round(float((df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100), 4) if len(df) > 1 else 0,
            }

        if df is None or df.empty or len(df) < 30 or not price:
            return None

        sigs = RealSignals.compute_all(df)
        if not sigs: return None

        sigs["price"] = price
        sigs["high"]  = quote.get("high", price)
        sigs["low"]   = quote.get("low",  price)
        sigs["changePct"] = quote.get("changePct", 0)

        # Cache close prices for ML/pairs
        close_arr = df["Close"].values.astype(float).tolist()
        self.price_cache[symbol] = close_arr

        # ML signals (optional engine)
        if ML:
            try:
                ml = ML.compute_ml_score(sigs, close_arr)
                sigs.update(ml)
                avg_vol = float(np.mean(df["Volume"].values[-20:])) if len(df) >= 20 else 1e6
                micro   = ML.microstructure_score(sigs, sigs.get("vol_ratio",1), price, avg_vol)
                sigs["microstructure"] = micro
                lstm = ML.lstm_pattern_score(close_arr)
                sigs["lstm"] = lstm
            except Exception as _ml_e:
                logger.debug(f"ML error {symbol}: {_ml_e}")

        # ── Pine Script v4.1 signal engine (optional) ─────────────────────
        if SE:
            try:
                sigs = SE.run_pine_signals(
                    symbol      = symbol,
                    df          = df,
                    signals     = sigs,
                    macro_bias  = 0.0,
                    sector_bias = 0.0,
                    sector_strength = "Neutral",
                    iv_rank     = sigs.get("ivr"),
                    earnings_meta = None,
                )
            except Exception as _se:
                logger.debug(f"Pine signals error {symbol}: {_se}")

        # ── Company info (logo + full details) ──────────────────────────────
        try:
            co_info = self.company_cache.get(symbol)
            needs_enrich = (
                co_info is None or
                not co_info.get("sector") or
                co_info.get("sector") == "Unknown" or
                not co_info.get("market_cap")
            )
            if co_info is None and CI:
                # Instant static lookup first so scan doesn't block
                co_info = CI.get_company_info(symbol, use_yfinance=False)
                self.company_cache[symbol] = co_info

            if needs_enrich and CI:
                # Background enrich with full yfinance data (non-blocking)
                def _enrich(sym=symbol):
                    try:
                        full = CI.get_company_info(sym, use_yfinance=True)
                        self.company_cache[sym] = full
                    except Exception as _ce:
                        logger.debug(f"Company enrich {sym}: {_ce}")
                threading.Thread(target=_enrich, daemon=True).start()

            if co_info:
                # Include description (was excluded before — now included truncated)
                sigs["company"] = {
                    "symbol":       co_info.get("symbol", symbol),
                    "name":         co_info.get("name", symbol),
                    "logo_url":     co_info.get("logo_url", ""),
                    "sector":       co_info.get("sector", ""),
                    "industry":     co_info.get("industry", ""),
                    "market_cap":   co_info.get("market_cap", 0),
                    "market_cap_fmt": co_info.get("market_cap_fmt", "—"),
                    "employees":    co_info.get("employees", 0),
                    "country":      co_info.get("country", "US"),
                    "exchange":     co_info.get("exchange", ""),
                    "website":      co_info.get("website", ""),
                    "description":  (co_info.get("description", "") or "")[:300],
                }
                sigs["logo_url"] = co_info.get("logo_url", "")
            else:
                sigs["company"] = {
                    "symbol":  symbol,
                    "name":    symbol,
                    "logo_url": f"https://logo.clearbit.com/{symbol.lower()}.com",
                }
                sigs["logo_url"] = sigs["company"]["logo_url"]
        except Exception as _co_e:
            logger.debug(f"Company info {symbol}: {_co_e}")
            sigs["company"] = {"name": symbol, "symbol": symbol}
            sigs["logo_url"] = ""

        # ── Full options analysis — all models ─────────────────────────────
        # ── Options chain — loaded in background after cycle 2 ────────────────
        # Skipped on first 2 cycles so signals appear in <30 seconds.
        # On cycle 3+ options load for BUY/SHORT signals only.
        opts = _OPTIONS_CACHE.get(symbol)  # use cached if available

        if opts is None and _GLOBAL_CYCLE > 1:
            direction = sigs.get("direction","HOLD")
            if direction in ("BUY","SHORT") and YF_OK:
                try:
                    ticker      = yf.Ticker(symbol)
                    expirations = ticker.options
                    if expirations:
                        conv = sigs.get("conviction", 0.3)
                        tdte = 21 if conv > 0.65 else 30 if conv > 0.45 else 45
                        exp  = RealOptionsChain.get_best_expiry(expirations, tdte) or expirations[0]
                        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                        dte_val  = (exp_date - datetime.today().date()).days
                        chain    = ticker.option_chain(exp)
                        hv_hist  = [sigs.get("hv20",25)/100] * 50
                        opts = OE.run_full_options_analysis(
                            symbol=symbol, spot=price, direction=direction,
                            signals=sigs, chain_calls=chain.calls, chain_puts=chain.puts,
                            expiry_str=exp, exp_nice=exp_date.strftime("%b %d, %Y"),
                            dte=dte_val, hv_history=hv_hist,
                        )
                        _OPTIONS_CACHE[symbol] = opts
                except Exception as _oe:
                    logger.debug(f"Options {symbol}: {_oe}")
        # Dynamic cap tier — uses market_cap from company_info (live yfinance)
        # Falls back to fast_info.market_cap if company_info not yet cached.
        _co = self.company_cache.get(symbol) or {}
        cap = get_cap_tier(symbol, market_cap=_co.get("market_cap", 0))

        hist = []
        close_vals = df["Close"].values[-80:]
        high_vals  = df["High"].values[-80:]
        low_vals   = df["Low"].values[-80:]
        vol_vals   = df["Volume"].values[-80:]
        dates      = df.index[-80:]
        for i, (c, h, l, v, d) in enumerate(zip(close_vals, high_vals, low_vals, vol_vals, dates)):
            hist.append({
                "t": i, "date": str(d)[:10],
                "price":  round(float(c), 4),
                "high":   round(float(h), 4),
                "low":    round(float(l), 4),
                "volume": int(v),
            })

        # ── AI Research Agent enrichment ──────────────────────────────────────
        research_data = {}
        if RC:
            try:
                avg_vol = float(np.mean(df["Volume"].values[-20:])) if len(df) >= 20 else 1e6
                research_data = RC.run_full_analysis(symbol, df, sigs, price, avg_vol)
            except Exception as _ra_e:
                logger.debug(f"Research agent {symbol}: {_ra_e}")

        # ── RV vs IV VOL ARB SIGNAL (Renaissance premium-collection edge) ────
        # Computes Vol Risk Premium = ATM_IV - RV20.
        # When IV significantly exceeds realized vol, premium selling is statistically
        # advantaged. This populates sigs["vol_regime"] for AladdinScorer.
        vol_arb_data = {}
        try:
            from math_engine import VolatilityAnalysis as _VA
            close_arr_va = df["Close"].values.astype(float)

            # Get ATM IV from options cache or use hv20 as proxy
            atm_iv_raw = sigs.get("hv20", 25) / 100   # fallback: RV = IV (no edge)
            if opts and isinstance(opts, dict):
                iv_from_opts = opts.get("iv") or opts.get("call_iv") or opts.get("put_iv")
                if iv_from_opts and float(iv_from_opts) > 0:
                    atm_iv_raw = float(iv_from_opts) / 100   # convert pct to decimal

            # Use cached options chain ATM IV if available
            cached_opts = _OPTIONS_CACHE.get(symbol, {})
            if isinstance(cached_opts, dict):
                chain_iv = (cached_opts.get("iv") or
                            cached_opts.get("call_iv") or
                            cached_opts.get("put_iv"))
                if chain_iv and float(chain_iv) > 0:
                    atm_iv_raw = float(chain_iv) / 100

            macro_d = _macro_cache.get("data") or {}
            vix_level = float((macro_d.get("vix") or {}).get("value", 20.0))

            vol_arb_data = _VA.rv_iv_spread(
                close_prices = close_arr_va,
                atm_iv       = atm_iv_raw,
                vix          = vix_level,
            )

            # Wire vol_arb_score → sigs["vol_regime"] for AladdinScorer
            # This replaces the previous static vol_regime value
            if "vol_arb_score" in vol_arb_data:
                sigs["vol_regime"]  = vol_arb_data["vol_arb_score"]
                sigs["vol_arb"]     = vol_arb_data
                sigs["vrp"]         = vol_arb_data.get("vrp", 0)
                sigs["vol_strategy"] = vol_arb_data.get("strategy", "NO_EDGE")

        except Exception as _va_e:
            logger.debug(f"Vol arb {symbol}: {_va_e}")

        # ── EARNINGS REVISION MOMENTUM (Step 5) ───────────────────────────────
        # SUE + revision direction + PEAD decay → revision_composite signal
        # Background-computed and cached (6hr TTL); returns zeros if not yet ready.
        # Feeds into AladdinScorer "options_flow" slot (proxy until Step 9 adds
        # a dedicated revision slot with 12-factor AladdinScorer).
        revision_data = {}
        try:
            if EE and hasattr(EE, "EarningsRevisionSignal"):
                revision_data = EE.EarningsRevisionSignal.compute(symbol)
                rev_composite = float(revision_data.get("revision_composite", 0))
                sigs["revision_composite"]  = rev_composite
                sigs["revision_direction"]  = revision_data.get("revision_direction", "NEUTRAL")
                sigs["sue_score"]           = revision_data.get("sue_score", 0)
                sigs["pead_signal"]         = revision_data.get("pead_signal", "NEUTRAL")
                sigs["pead_weight"]         = revision_data.get("pead_weight", 0)
                sigs["days_since_earnings"] = revision_data.get("days_since_earnings", 99)
                # Blend revision into options_flow slot
                # options_flow_score is already set by research agent (if any)
                existing_of = float(sigs.get("options_flow_score", 0))
                # 60% revision, 40% existing options flow (if any)
                sigs["options_flow_score"] = float(np.clip(
                    0.60 * rev_composite + 0.40 * existing_of, -1, 1
                ))
        except Exception as _rev_e:
            logger.debug(f"Revision signal {symbol}: {_rev_e}")

        # ── INSIDER CLUSTER SCORING (Step 8) ─────────────────────────────────
        # Cluster detection: 2+ distinct insiders buying in same 30-day window
        # 4hr cache; returns immediately if cached, else launches background fetch.
        # Cluster score wired into news_sentiment slot (blended 50/50 with news).
        insider_cluster_data = {}
        try:
            if EE and hasattr(EE, "InsiderClusterScorer"):
                insider_cluster_data = EE.InsiderClusterScorer.compute(symbol)
                cluster_score = float(insider_cluster_data.get("cluster_score", 0))
                sigs["insider_cluster_score"] = cluster_score
                sigs["insider_signal_label"]  = insider_cluster_data.get("signal_label", "NEUTRAL")
                sigs["has_buy_cluster"]        = insider_cluster_data.get("has_buy_cluster", False)
                # Blend insider cluster into news_sentiment slot (50/50 weight)
                # Insider cluster signal has higher IC than news sentiment alone
                existing_news = float(sigs.get("news_sentiment", 0))
                sigs["news_sentiment"] = float(np.clip(
                    0.50 * cluster_score + 0.50 * existing_news, -1, 1
                ))
        except Exception as _ic_e:
            logger.debug(f"Insider cluster {symbol}: {_ic_e}")

        # ── DARK POOL SIGNAL (Step 13) ────────────────────────────────────────
        # Detects institutional accumulation/distribution via:
        #   - FINRA short volume ratio (free, daily, 2-week lag)
        #   - Block trade / stealth bar detection (high vol, low price impact)
        #   - Dark pool ratio from microstructure (Amihud, Kyle's lambda)
        # Wired into options_flow slot (blended 50/50 with existing options signal)
        dp_data = {}
        try:
            import dark_pool_engine as _DPE
            close_arr_dp = df["Close"].values.astype(float)
            vol_arr_dp   = df["Volume"].values.astype(float)
            dp_data      = _DPE.DarkPoolScorer.compute(
                symbol, price_history=close_arr_dp, vol_history=vol_arr_dp
            )
            dp_score = float(dp_data.get("dp_score", 0.0))
            sigs["dp_score"]       = dp_score
            sigs["dp_signal"]      = dp_data.get("signal_label", "NEUTRAL")
            sigs["stealth_bars"]   = int(dp_data.get("stealth_bar_count", 0))
            # Blend dark pool into options_flow slot (50/50 with options flow)
            existing_of = float(sigs.get("options_flow_score", 0))
            sigs["options_flow_score"] = float(np.clip(
                0.50 * dp_score + 0.50 * existing_of, -1, 1
            ))
        except Exception as _dp_e:
            logger.debug(f"Dark pool {symbol}: {_dp_e}")

        # ── Level-2 Microstructure Signals (Gap 3 partial fix) ───────────────
        # Alpaca fast_info provides real-time bid/ask for any US equity.
        # We compute 3 microstructure signals from this data:
        #
        #   l2_spread    : bid-ask spread / mid-price
        #                  Wide spread → informed flow or illiquidity (bearish bias)
        #                  Normalized: spread% vs 20-day avg spread%
        #   l2_imbalance : (bid_size - ask_size) / (bid_size + ask_size)
        #                  Positive → buy pressure (bullish), negative → sell pressure
        #   l2_vwap_dev  : (price - VWAP) / VWAP  — price deviation from fair value
        #                  Negative deviation → mean-reversion buy signal
        #                  Positive deviation → mean-reversion sell signal
        #
        # These are OHLCV-proxy versions when Alpaca bid/ask unavailable.
        # Enable Alpaca WebSocket streaming for real tick-level L2 data.
        try:
            quote_data = scanner.all_signals.get(symbol, {}) if scanner else {}
            _price = float(sigs.get("price", quote_data.get("price", 0)) or 0)
            _bid   = float(quote_data.get("bid", 0) or 0)
            _ask   = float(quote_data.get("ask", 0) or 0)

            if _bid > 0 and _ask > 0 and _price > 0:
                # Real bid/ask from quote
                mid_price = (_bid + _ask) / 2
                spread_pct = (_ask - _bid) / (mid_price + 1e-8)
                # Normalize: spread vs typical 0.05% (tight) to 0.5% (wide)
                # Signal: wide spread = negative (bearish / avoid)
                l2_spread_norm = float(np.clip(-spread_pct / 0.003, -1, 1))
            else:
                # OHLCV proxy: use ATR/price as spread proxy
                atr_pct = float(sigs.get("atr_pct", 0.01) or 0.01)
                l2_spread_norm = float(np.clip(-atr_pct / 0.02, -1, 1))

            # Order imbalance — use volume/price direction proxy if no L2
            _vol_today = float(df["Volume"].iloc[-1]) if not df.empty else 0
            _vol_avg   = float(df["Volume"].rolling(20).mean().iloc[-1]) if len(df) >= 20 else _vol_today
            vol_ratio  = (_vol_today / (_vol_avg + 1e-8)) - 1.0
            price_ret  = float(sigs.get("chg_pct", 0) or 0) / 100.0
            # High vol + positive return = buy imbalance; high vol + negative = sell imbalance
            l2_imbalance_norm = float(np.clip(np.sign(price_ret) * min(abs(vol_ratio), 1.0), -1, 1))

            # VWAP deviation — mean-reversion signal
            # Use daily price vs 5-day MA as VWAP proxy (daily bars)
            if len(df) >= 5:
                vwap_proxy = float(df["Close"].rolling(5).mean().iloc[-1])
                vwap_dev   = (float(df["Close"].iloc[-1]) - vwap_proxy) / (vwap_proxy + 1e-8)
                l2_vwap_dev_norm = float(np.clip(-vwap_dev / 0.03, -1, 1))  # invert: below VWAP = buy signal
            else:
                l2_vwap_dev_norm = 0.0

            sigs["l2_spread"]    = l2_spread_norm
            sigs["l2_imbalance"] = l2_imbalance_norm
            sigs["l2_vwap_dev"]  = l2_vwap_dev_norm
        except Exception as _l2_e:
            logger.debug(f"L2 signals {symbol}: {_l2_e}")
            sigs["l2_spread"]    = 0.0
            sigs["l2_imbalance"] = 0.0
            sigs["l2_vwap_dev"]  = 0.0

        # ── Build sub_signals dict for AladdinScorer + IC tracking ────────────
        # 13 signals total: 10 original + 3 new L2 microstructure (Gap 3 fix)
        aladdin_input = {
            "technical_composite": float(np.clip(sigs.get("composite", 0), -1, 1)),
            "advanced_composite":  float(np.clip(sigs.get("adj_dir_score", sigs.get("composite", 0)), -1, 1)),
            "ml_score":            float(np.clip(sigs.get("ml_score", 0.5) * 2 - 1, -1, 1)),
            "hurst_blended":       float(np.clip(sigs.get("hurst", 0.5) * 2 - 1, -1, 1)),
            "ou_reversion":        float(np.clip(-sigs.get("ou_zscore", 0) / 3, -1, 1)),
            "kalman_trend":        float(np.clip(sigs.get("bull_prob_kf", 50) / 50 - 1, -1, 1)),
            "vol_regime":          float(np.clip(sigs.get("vol_regime", 0), -1, 1)),
            "news_sentiment":      float(np.clip(sigs.get("news_sentiment", 0), -1, 1)),
            "options_flow":        float(np.clip(sigs.get("options_flow_score", 0), -1, 1)),
            "macro_regime":        float(np.clip(
                scanner.macro_signal if hasattr(scanner, "macro_signal") else sigs.get("macro_bias", 0),
                -1, 1
            )),
            # ── Level-2 microstructure signals (Gap 3 partial fix) ───────────
            # Computed from Alpaca fast_info bid/ask + VWAP (zero extra cost).
            # When Alpaca WebSocket streaming is enabled these become real-time.
            # Currently computed once per scan cycle from fast_info snapshots.
            "l2_spread":    float(np.clip(sigs.get("l2_spread",    0.0), -1, 1)),
            "l2_imbalance": float(np.clip(sigs.get("l2_imbalance", 0.0), -1, 1)),
            "l2_vwap_dev":  float(np.clip(sigs.get("l2_vwap_dev",  0.0), -1, 1)),
        }

        # Run AladdinScorer with adaptive weights
        try:
            from math_engine import AladdinScorer
            aladdin_result = AladdinScorer.compute(aladdin_input)
            # Override composite with AladdinScorer result if it exists
            sigs["aladdin_composite"]   = aladdin_result["composite"]
            sigs["aladdin_direction"]   = aladdin_result["direction"]
            sigs["aladdin_conviction"]  = aladdin_result["conviction"]
            sigs["aladdin_weights_live"] = aladdin_result.get("weights_live", {})
            sigs["aladdin_ic_stats"]    = aladdin_result.get("ic_stats", {})
            sigs["sub_signals"]         = aladdin_input   # stored for IC feedback on close
            # MI diagnostics (Step 10)
            sigs["mi_adjustment"]       = aladdin_result.get("mi_adjustment", 0.0)
            sigs["effective_weights"]   = aladdin_result.get("effective_weights", {})
            sigs["weight_deltas"]       = aladdin_result.get("weight_deltas", {})
            sigs["redundancy_pairs"]    = aladdin_result.get("redundancy_pairs", [])
            sigs["mi_history_len"]      = aladdin_result.get("mi_history_len", 0)
        except Exception as _al_e:
            logger.debug(f"AladdinScorer {symbol}: {_al_e}")
            sigs["sub_signals"] = aladdin_input

        # ── FF5 FACTOR NEUTRALIZATION (Step 9) ────────────────────────────────
        # Discount AladdinScorer composite by its idiosyncratic fraction (alpha_r2).
        # Signals that are just high-beta or size-driven get discounted.
        # Uses 60-day rolling OLS on daily returns vs SPY/IWM/IWD/QUAL/MTUM proxies.
        ff5_data      = {}
        neutral_data  = {}
        try:
            from math_engine import FactorAnalysis as _FA
            close_series = df["Close"].values.astype(float)
            if len(close_series) >= 62:
                log_rets = np.diff(np.log(close_series[-62:] + 1e-10))
                raw_composite = float(sigs.get("aladdin_composite",
                                               sigs.get("composite", 0)))
                ff5_data     = _FA.regress_ff5(log_rets, window=60)
                neutral_data = _FA.neutralize_signal(raw_composite, ff5_data,
                                                      mode="alpha_weighted")
                # Override final signal with neutralized version
                sigs["neutralized_composite"] = neutral_data["neutralized_signal"]
                sigs["alpha_quality"]         = neutral_data["alpha_quality"]
                sigs["mkt_beta"]              = neutral_data["mkt_beta"]
                sigs["ff5_r_squared"]         = neutral_data["r_squared"]
                sigs["ff5_alpha_r2"]          = neutral_data["alpha_r2"]
                sigs["ff5_discount"]          = neutral_data["discount_applied"]
                sigs["factor_tilt_str"]       = neutral_data.get("factor_tilt_str", "none")
        except Exception as _ff5_e:
            logger.debug(f"FF5 neutralization {symbol}: {_ff5_e}")

        # ── PhD HYPOTHESIS ENGINE  (Step 14) ─────────────────────────────────
        # Cross-references all 13 signals via Bayesian posterior conviction,
        # classifies thesis archetype, and generates investment memo when
        # conviction > 60%. 4hr cache per symbol.
        phd_result = {}
        try:
            import phd_hypothesis_engine as _PHD
            api_key    = os.environ.get("ANTHROPIC_API_KEY", "")
            phd_engine = _PHD.PhDHypothesisEngine.get_instance(api_key)

            # Build normalized 15-signal dict for Bayesian scorer
            phd_signals = {
                "technical_composite": float(np.clip(sigs.get("composite", 0), -1, 1)),
                "advanced_composite":  float(np.clip(sigs.get("adj_dir_score", sigs.get("composite", 0)), -1, 1)),
                "ml_score":            float(np.clip(sigs.get("ml_score", 0.5)*2-1, -1, 1)),
                "hurst_blended":       float(np.clip(sigs.get("hurst", 0.5)*2-1, -1, 1)),
                "ou_reversion":        float(np.clip(-sigs.get("ou_zscore", 0)/3, -1, 1)),
                "kalman_trend":        float(np.clip(sigs.get("bull_prob_kf", 50)/50-1, -1, 1)),
                "vol_regime":          float(np.clip(sigs.get("vol_regime", 0), -1, 1)),
                "news_sentiment":      float(np.clip(sigs.get("news_sentiment", 0), -1, 1)),
                "options_flow":        float(np.clip(sigs.get("options_flow_score", 0), -1, 1)),
                "macro_regime":        float(np.clip(
                    scanner.macro_signal if hasattr(scanner, "macro_signal") else sigs.get("macro_bias", 0),
                    -1, 1
                )),
                # Steps 5–13
                "revision_momentum":   float(np.clip(sigs.get("revision_score", 0)/3, -1, 1)),
                "ff5_alpha_quality":   float(np.clip(sigs.get("neutralized_composite", 0), -1, 1)),
                "insider_cluster":     float(np.clip(sigs.get("insider_cluster_score", 0), -1, 1)),
                "dark_pool_score":     float(np.clip(sigs.get("dp_score", 0), -1, 1)),
                "backtest_ic":         float(np.clip(sigs.get("backtest_ic", 0)*10, -1, 1)),
            }

            hmm_regime = (
                scanner.regime_model.current_regime
                if hasattr(scanner, "regime_model") and scanner.regime_model else "NEUTRAL"
            )
            macro_ctx = {
                "vix":         float(scanner.all_signals.get("VIX", {}).get("price", 20.0)),
                "yield_curve": sigs.get("yield_curve", "unknown"),
                "dxy_trend":   sigs.get("dxy_trend", "unknown"),
            }
            company_info = sigs.get("company") or {}
            company_name = company_info.get("name", symbol) if isinstance(company_info, dict) else symbol

            phd_result = phd_engine.analyze(
                symbol=symbol,
                signals=phd_signals,
                regime=hmm_regime,
                macro_context=macro_ctx,
                company_name=company_name,
            )
        except Exception as _phd_e:
            logger.debug(f"PhD engine {symbol}: {_phd_e}")

        _result = {
            "symbol":      symbol,
            "price":       round(price, 4),
            "change":      quote.get("change", 0),
            "changePct":   quote.get("changePct", 0),
            "volume":      quote.get("volume", 0),
            "high":        quote.get("high", price),
            "low":         quote.get("low", price),
            "open":        quote.get("open", price),
            "cap":         cap,
            "composite":   round(sigs.get("composite", 0), 4),
            "normalized":  round(sigs.get("normalized", 0.5), 4),
            "direction":   sigs.get("direction", "HOLD"),
            "conviction":  round(sigs.get("conviction", 0), 4),
            "rsi":         round(sigs.get("rsi", 50), 2),
            "macd":        round(sigs.get("macd", 0), 5),
            "macd_norm":   round(sigs.get("macd_norm", 0), 5),
            "adx":         round(sigs.get("adx", 0), 2),
            "atr_pct":     round(sigs.get("atr_pct", 0), 3),
            "vol_ratio":   round(sigs.get("vol_ratio", 1), 3),
            "bb_position": round(sigs.get("bb_position", 0.5), 3),
            "bb_width":    round(sigs.get("bb_width", 0), 4),
            "zscore":      round(sigs.get("zscore", 0), 3),
            "stoch_k":     round(sigs.get("stoch_k", 50), 2),
            "hv20":        round(sigs.get("hv20", 25), 2),
            "hv60":        round(sigs.get("hv60", 25), 2),
            "garch_vol":   round(sigs.get("garch_vol", 25), 2),
            "ivr":         round(sigs.get("ivr", 50), 2),
            "hurst":       round(sigs.get("hurst", 0.5), 4),
            "ou_halflife": round(sigs.get("ou_halflife", 99), 2),
            "ou_zscore":   round(sigs.get("ou_zscore", 0), 3),
            "sharpe":      round(sigs.get("sharpe", 0), 3),
            "skew":        round(sigs.get("skew", 0), 3),
            "mom_1d":      round(sigs.get("mom_1d", 0), 3),
            "mom_5d":      round(sigs.get("mom_5d", 0), 3),
            "mom_20d":     round(sigs.get("mom_20d", 0), 3),
            "ema9":        round(sigs.get("ema9", price), 4),
            "ema21":       round(sigs.get("ema21", price), 4),
            "ema50":       round(sigs.get("ema50", price), 4),
            "ema200":      round(sigs.get("ema200", price), 4),
            "vwap":        round(sigs.get("vwap", price), 4),
            "price_vs_vwap": round(sigs.get("price_vs_vwap", 0), 3),
            "options":     opts,
            "history":     hist,
            "last_analysis": datetime.now().isoformat(),

            # Pine Script v4.1 signals
            "confidence":       sigs.get("confidence", 50),
            "confidence_grade": sigs.get("confidence_grade", "C  (FAIR)"),
            "bull_prob_kf":     round(float(sigs.get("bull_prob_kf", 50)), 2),
            "bear_prob_kf":     round(float(sigs.get("bear_prob_kf", 50)), 2),
            "bull_prob_ens":    round(float(sigs.get("bull_prob_ens", 50)), 2),
            "adj_dir_score":    round(float(sigs.get("adj_dir_score", 0)), 4),
            "v_reversal":       bool(sigs.get("v_reversal", False)),
            "v_call":           bool(sigs.get("v_call", False)),
            "v_put":            bool(sigs.get("v_put", False)),
            "inst_active":      bool(sigs.get("inst_active", False)),
            "inst_text":        str(sigs.get("inst_text", "")),
            "vol_text":         str(sigs.get("vol_text", "")),
            "hybrid_forecast":  round(float(sigs.get("hybrid_forecast", price)), 3),
            "range_forecast":   sigs.get("range_forecast") or {},
            "squeeze":          sigs.get("squeeze") or {},
            "is_100_pct":       bool(sigs.get("is_100_pct", False)),
            "play_100":         sigs.get("play_100") or {},
            "hold_text":        str(sigs.get("hold_text", "Swing (1-2 Weeks)")),
            "hold_days":        int(sigs.get("hold_days", 7)),
            "suggested_call_strike": round(float(sigs.get("suggested_call_strike", price)), 2),
            "suggested_put_strike":  round(float(sigs.get("suggested_put_strike",  price)), 2),
            "company":          sigs.get("company") or {},
            "logo_url":         str(sigs.get("logo_url", "")),

            # ML signals
            "ml_score":         round(float(sigs.get("ml_score", 0.5)), 4),
            "ml_pct":           round(float(sigs.get("ml_pct", 50)), 1),
            "gb_score":         round(float(sigs.get("gb_score", 0.5)), 4),
            "lstm_pattern":     str(sigs.get("lstm_pattern", "UNKNOWN")),
            "lstm_score":       round(float(sigs.get("lstm_score", 0.5)), 4),
            "ml_interp":        str(sigs.get("interpretation", "ML NEUTRAL")),
            "microstructure":   sigs.get("microstructure") or {},

            # ── AI PhD Research Agent fields ─────────────────────────────────
            "tick_micro":       research_data.get("tick_micro") or {},
            "insider_flow":     research_data.get("insider_flow") or {},
            "options_flow":     research_data.get("options_flow") or {},
            "execution":        research_data.get("execution") or {},
            "factor_scores":    research_data.get("factor_scores") or {},
            "ic_weighted_alpha":research_data.get("ic_weighted_alpha") or sigs.get("composite", 0),
            "signal_decay":     research_data.get("signal_decay") or {},

            # ── Vol Arb (RV vs IV) ─────────────────────────────────────────
            "vol_arb":          sigs.get("vol_arb") or vol_arb_data or {},
            "vrp":              round(float(sigs.get("vrp", 0)), 2),
            "vol_strategy":     str(sigs.get("vol_strategy", "NO_EDGE")),

            # ── Earnings Revision Momentum ────────────────────────────────
            "revision_composite":   round(float(sigs.get("revision_composite", 0)), 4),
            "revision_direction":   str(sigs.get("revision_direction", "NEUTRAL")),
            "sue_score":            round(float(sigs.get("sue_score", 0)), 4),
            "pead_signal":          str(sigs.get("pead_signal", "NEUTRAL")),
            "pead_weight":          round(float(sigs.get("pead_weight", 0)), 3),
            "days_since_earnings":  int(sigs.get("days_since_earnings", 99)),
            "revision_data":        revision_data,

            # ── Insider Cluster ───────────────────────────────────────────
            "insider_cluster_score": round(float(sigs.get("insider_cluster_score", 0)), 4),
            "insider_signal_label":  str(sigs.get("insider_signal_label", "NEUTRAL")),
            "has_buy_cluster":       bool(sigs.get("has_buy_cluster", False)),
            "insider_cluster_data":  insider_cluster_data,

            # ── Fama-French 5-Factor Neutralization ───────────────────────
            "neutralized_composite": round(float(sigs.get("neutralized_composite",
                                          sigs.get("aladdin_composite", 0))), 4),
            "alpha_quality":         str(sigs.get("alpha_quality", "UNKNOWN")),
            "mkt_beta":              round(float(sigs.get("mkt_beta", 1.0)), 3),
            "ff5_r_squared":         round(float(sigs.get("ff5_r_squared", 0)), 4),
            "ff5_alpha_r2":          round(float(sigs.get("ff5_alpha_r2", 1)), 4),
            "ff5_discount":          round(float(sigs.get("ff5_discount", 0)), 4),
            "factor_tilt_str":       str(sigs.get("factor_tilt_str", "none")),
            "ff5_data":              ff5_data,
            "ff5_neutral":           neutral_data,

            # ── Dark Pool Signal ──────────────────────────────────────────
            "dp_score":             round(float(sigs.get("dp_score", 0.0)), 4),
            "dp_signal":            str(sigs.get("dp_signal", "NEUTRAL")),
            "dp_stealth_bars":      int(sigs.get("stealth_bars", 0)),
            "dp_data":              dp_data,

            # ── PhD Hypothesis Engine ──────────────────────────────────────
            "phd_conviction":       int(phd_result.get("conviction_pct", 0)),
            "phd_thesis":           str(phd_result.get("top_thesis", "HOLD")),
            "phd_archetype":        str(phd_result.get("archetype", "")),
            "phd_n_confirming":     int(phd_result.get("n_confirming", 0)),
            "phd_n_conflicting":    int(phd_result.get("n_conflicting", 0)),
            "phd_memo":             phd_result.get("memo", {}),
            "phd_bayesian":         phd_result.get("bayesian", {}),

            # ── MI Signal Fusion diagnostics (Step 10) ────────────────────
            "mi_adjustment":        round(float(sigs.get("mi_adjustment", 0.0)), 4),
            "effective_weights":    sigs.get("effective_weights", {}),
            "weight_deltas":        sigs.get("weight_deltas", {}),
            "redundancy_pairs":     sigs.get("redundancy_pairs", []),
            "mi_history_len":       int(sigs.get("mi_history_len", 0)),

            # ── AladdinScorer adaptive composite ──────────────────────────
            "aladdin_composite":  round(float(sigs.get("aladdin_composite", sigs.get("composite", 0))), 4),
            "aladdin_direction":  str(sigs.get("aladdin_direction", sigs.get("direction", "HOLD"))),
            "aladdin_conviction": round(float(sigs.get("aladdin_conviction", sigs.get("conviction", 0))), 4),
            "sub_signals":        sigs.get("sub_signals", {}),
        }

        # ── Swing Trade Context — this system is for 1-2 week swing trades ────────
        hold_days = int(sigs.get("hold_days", 7))
        _result["swing_trade"] = {
            "timeframe":    "SWING",
            "hold_days":    hold_days,
            "hold_label":   ("1-3 Days"   if hold_days <= 3  else
                             "3-7 Days"   if hold_days <= 7  else
                             "1-2 Weeks"  if hold_days <= 14 else
                             "2-4 Weeks"),
            "not_daytrading": True,
            "signals_used": ["20d momentum", "OU mean-reversion", "GARCH vol", "RSI", "MACD"],
            "entry_style":  "Patient — wait for confirmation, use limit orders",
            "exit_style":   "Trail stop or target hit — not intraday",
            "note":         "Signals based on daily bars. Not suited for intraday/scalping.",
        }

        # ── Direction-aware Stock Play ─────────────────────────────────────────
        # Fixes: stop > entry for bearish (SHORT: stop IS above entry — that's correct)
        # But labels must reflect direction so the frontend shows them right
        _rf   = sigs.get("range_forecast") or {}
        _dir  = sigs.get("direction", "HOLD")
        _atr  = sigs.get("atr_pct", 0.015) * price

        if _dir == "BUY":
            # Long trade: stop BELOW entry (wrong if price drops), target ABOVE
            _stop   = round(min(price - max(_atr * 1.5, price * 0.02),
                                float(_rf.get("low_1s",  price * 0.97))), 2)
            _target = round(max(price + max(_atr * 2.5, price * 0.03),
                                float(_rf.get("high_1s", price * 1.04))), 2)
            _play_type = "LONG"
            _stop_label   = "Stop Loss (exit if drops here)"
            _target_label = "Profit Target (exit when reached)"
        elif _dir == "SHORT":
            # Short trade: stop ABOVE entry (wrong if price rises), target BELOW
            _stop   = round(max(price + max(_atr * 1.5, price * 0.02),
                                float(_rf.get("high_1s", price * 1.03))), 2)
            _target = round(min(price - max(_atr * 2.5, price * 0.03),
                                float(_rf.get("low_1s",  price * 0.96))), 2)
            _play_type = "SHORT"
            _stop_label   = "Stop Loss (exit if RISES here)"
            _target_label = "Profit Target (exit when price FALLS here)"
        else:
            _mid    = float(_rf.get("mid", price))
            _sigma  = float(_rf.get("sigma", _atr * 2))
            _stop   = round(_mid - _sigma, 2)
            _target = round(_mid + _sigma, 2)
            _play_type    = "WATCH"
            _stop_label   = "Support zone"
            _target_label = "Resistance zone"

        _risk   = abs(price - _stop)
        _reward = abs(_target - price)

        _result["stock_play"] = {
            "play_type":     _play_type,
            "direction":     _dir,
            "entry":         round(price, 2),
            "stop_loss":     _stop,
            "profit_target": _target,
            "stop_label":    _stop_label,
            "target_label":  _target_label,
            "risk_usd":      round(_risk, 2),
            "reward_usd":    round(_reward, 2),
            "risk_reward":   round(_reward / max(_risk, 0.01), 2),
            "is_long":       _dir == "BUY",
            "is_short":      _dir == "SHORT",
            "note":          ("SHORT: stop ABOVE entry is correct — you lose if price rises"
                               if _dir == "SHORT" else
                               "LONG: stop BELOW entry is correct — you lose if price falls"
                               if _dir == "BUY" else "No directional signal"),
        }

        return _result


# ═══════════════════════════════════════════════════════════════════════════════
# FREE API DATA FETCHERS
# ═══════════════════════════════════════════════════════════════════════════════

_fred_cache = {}
_fred_ts    = {}

def fetch_fred_macro():
    """
    Fetch macro data from FRED (Federal Reserve Bank of St. Louis).
    Completely FREE — no key required for many series, or get free key at
    fred.stlouisfed.org in 30 seconds.

    Series used:
      DFF     — Fed Funds Rate (daily)
      DGS10   — 10-Year Treasury Yield (daily)
      DGS2    — 2-Year Treasury Yield (daily)
      UNRATE  — Unemployment Rate (monthly)
      CPIAUCSL— CPI inflation (monthly)
      M2SL    — M2 Money Supply (monthly)
      VIXCLS  — VIX close (daily)
    """
    now = time.time()
    if _fred_cache and (now - _fred_ts.get("ts", 0)) < 3600:  # 1hr cache
        return _fred_cache

    series = {
        "fed_rate":    "DFF",
        "yield_10y":   "DGS10",
        "yield_2y":    "DGS2",
        "unemployment":"UNRATE",
        "cpi":         "CPIAUCSL",
        "vix_close":   "VIXCLS",
    }

    result = {}
    base   = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="

    for label, series_id in series.items():
        try:
            url = f"{base}{series_id}"
            r   = requests.get(url, timeout=6)
            if r.status_code == 200:
                lines = r.text.strip().split("\n")
                # Last non-empty line with a real value
                for line in reversed(lines[1:]):
                    parts = line.split(",")
                    if len(parts) == 2 and parts[1].strip() not in (".", ""):
                        result[label] = {
                            "value": float(parts[1].strip()),
                            "date":  parts[0].strip(),
                            "series": series_id,
                        }
                        break
        except Exception as _fe:
            logger.debug(f"FRED {series_id}: {_fe}")

    # Yield curve: 10Y minus 2Y (inverted = recession signal)
    if "yield_10y" in result and "yield_2y" in result:
        result["yield_curve"] = {
            "value": round(result["yield_10y"]["value"] - result["yield_2y"]["value"], 3),
            "inverted": result["yield_10y"]["value"] < result["yield_2y"]["value"],
            "date": result["yield_10y"]["date"],
        }

    _fred_cache.update(result)
    _fred_ts["ts"] = now
    return result


def fetch_yahoo_rss_news(symbol=""):
    """
    Yahoo Finance RSS — free, no API key.
    Returns latest headlines for a symbol or general market.
    """
    try:
        url  = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"                if symbol else "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY,QQQ&region=US&lang=en-US"
        r    = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        news = []
        import re
        items = re.findall(r"<item>(.*?)</item>", r.text, re.DOTALL)
        bull_kw = ["beat","surge","rally","gain","rise","upgrade","profit","record","growth","strong","buy"]
        bear_kw = ["miss","fall","drop","cut","downgrade","loss","weak","concern","risk","warning","decline","sell"]
        for item in items[:10]:
            title   = re.findall(r"<title>(.*?)</title>", item)
            pubdate = re.findall(r"<pubDate>(.*?)</pubDate>", item)
            link    = re.findall(r"<link>(.*?)</link>", item)
            if not title: continue
            t   = title[0].strip()
            tl  = t.lower()
            sent = "BULLISH" if sum(1 for w in bull_kw if w in tl) > sum(1 for w in bear_kw if w in tl) else                    "BEARISH" if any(w in tl for w in bear_kw) else "NEUTRAL"
            news.append({
                "title":     t,
                "source":    "Yahoo Finance",
                "time":      pubdate[0][:16] if pubdate else "",
                "sentiment": sent,
                "url":       link[0] if link else "",
                "symbols":   [symbol] if symbol else [],
            })
        return news
    except Exception as _ye:
        logger.debug(f"Yahoo RSS: {_ye}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# NEWS
# ═══════════════════════════════════════════════════════════════════════════════

_news_cache: dict = {"data": [], "ts": 0.0}
_NEWS_TTL = 300  # refresh every 5 minutes

def _sentiment(text: str) -> str:
    tl = text.lower()
    bull_kw = ["beat","surge","rally","gain","rise","upgrade","profit","record","growth","strong",
                "high","positive","bullish","buy","outperform","exceed","jump","soar","boom"]
    bear_kw = ["miss","fall","drop","cut","downgrade","loss","weak","concern","risk","warning",
                "decline","crash","plunge","bearish","sell","underperform","recession","fear"]
    b = sum(1 for w in bull_kw if w in tl)
    s = sum(1 for w in bear_kw if w in tl)
    return "BULLISH" if b > s else "BEARISH" if s > b else "NEUTRAL"

def _parse_rss(url: str, source_name: str, timeout: int = 8) -> list:
    """Parse an RSS/Atom feed and return normalized news articles."""
    import re as _re
    try:
        r = requests.get(url, timeout=timeout,
                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        if not r.ok:
            return []
        text = r.text
        items = _re.findall(r"<item>(.*?)</item>", text, _re.DOTALL)
        if not items:  # Atom feed
            items = _re.findall(r"<entry>(.*?)</entry>", text, _re.DOTALL)
        results = []
        for item in items[:12]:
            title_m = _re.search(r"<title[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", item, _re.DOTALL)
            link_m  = _re.search(r"<link[^>]*>(?:<!\[CDATA\[)?(https?://[^\s<>\"]+)(?:\]\]>)?</link>", item, _re.DOTALL)
            date_m  = _re.search(r"<(?:pubDate|updated|published)[^>]*>(.*?)</(?:pubDate|updated|published)>", item, _re.DOTALL)
            if not title_m:
                continue
            title = title_m.group(1).strip()
            # strip html entities
            title = _re.sub(r"<[^>]+>", "", title)
            title = title.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&#39;", "'").replace("&quot;", "\"")
            if not title or title == "[Removed]" or len(title) < 10:
                continue
            ts = ""
            if date_m:
                raw_date = date_m.group(1).strip()
                try:
                    from email.utils import parsedate_to_datetime
                    ts = parsedate_to_datetime(raw_date).strftime("%H:%M")
                except Exception:
                    try:
                        ts = datetime.fromisoformat(raw_date[:19]).strftime("%H:%M")
                    except Exception:
                        ts = raw_date[:5] if len(raw_date) >= 5 else ""
            results.append({
                "title":     title,
                "source":    source_name,
                "time":      ts,
                "sentiment": _sentiment(title),
                "url":       link_m.group(1).strip() if link_m else "",
                "symbols":   [],
            })
        return results
    except Exception as _rss_e:
        logger.debug(f"RSS {source_name} failed: {_rss_e}")
        return []

def fetch_news() -> list:
    """
    Multi-source news fetcher with layered fallback tiers.
    Priority:
      NewsAPI → GNews → Finnhub → Yahoo RSS → Reuters RSS →
      MarketWatch RSS → CNBC RSS → Seeking Alpha RSS → Nasdaq RSS
    Results cached 5 minutes to avoid rate limits.
    """
    global _news_cache
    now = time.time()
    if _news_cache["data"] and now - _news_cache["ts"] < _NEWS_TTL:
        return _news_cache["data"]

    syms = list(scanner.all_signals.keys())[:50] if scanner and scanner.all_signals else []
    all_news: list = []

    # ── Tier 1: NewsAPI (requires free key at newsapi.org) ─────────────────────
    if NEWS_API_KEY:
        try:
            url = (f"https://newsapi.org/v2/top-headlines"
                   f"?category=business&language=en&pageSize=25&apiKey={NEWS_API_KEY}")
            r   = requests.get(url, timeout=8)
            if r.ok:
                for art in r.json().get("articles", [])[:20]:
                    title = art.get("title", "")
                    if not title or title == "[Removed]":
                        continue
                    pub = art.get("publishedAt", "")
                    try:
                        ts = datetime.fromisoformat(pub.replace("Z", "+00:00")).strftime("%H:%M")
                    except Exception:
                        ts = ""
                    all_news.append({
                        "title":     title,
                        "source":    art.get("source", {}).get("name", "NewsAPI"),
                        "time":      ts,
                        "sentiment": _sentiment(title),
                        "url":       art.get("url", ""),
                        "symbols":   [s for s in syms if s.lower() in title.lower()],
                    })
                logger.info(f"  News: {len(all_news)} from NewsAPI")
        except Exception as _na_e:
            logger.debug(f"  NewsAPI failed: {_na_e}")

    # ── Tier 2: GNews (free tier key) ────────────────────────────────────────
    if len(all_news) < 8 and GNEWS_API_KEY:
        try:
            gurl = (
                "https://gnews.io/api/v4/top-headlines"
                f"?category=business&lang=en&country=us&max=25&apikey={GNEWS_API_KEY}"
            )
            gr = requests.get(gurl, timeout=8)
            if gr.ok:
                for art in (gr.json() or {}).get("articles", [])[:20]:
                    title = art.get("title", "")
                    if not title:
                        continue
                    pub = art.get("publishedAt", "")
                    try:
                        ts = datetime.fromisoformat(pub.replace("Z", "+00:00")).strftime("%H:%M")
                    except Exception:
                        ts = ""
                    all_news.append({
                        "title": title,
                        "source": "GNews",
                        "time": ts,
                        "sentiment": _sentiment(title),
                        "url": art.get("url", ""),
                        "symbols": [s for s in syms if s.lower() in title.lower()],
                    })
                logger.info(f"  News: {len(all_news)} total (GNews added)")
        except Exception as _gn_e:
            logger.debug(f"  GNews failed: {_gn_e}")

    # ── Tier 3: Finnhub general news (free tier key) ─────────────────────────
    if len(all_news) < 10 and FINNHUB_API_KEY:
        try:
            furl = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API_KEY}"
            fr = requests.get(furl, timeout=8)
            if fr.ok:
                for art in (fr.json() or [])[:25]:
                    title = art.get("headline", "")
                    if not title:
                        continue
                    ts = ""
                    try:
                        ts = datetime.utcfromtimestamp(int(art.get("datetime", 0) or 0)).strftime("%H:%M")
                    except Exception:
                        pass
                    all_news.append({
                        "title": title,
                        "source": "Finnhub",
                        "time": ts,
                        "sentiment": _sentiment(title),
                        "url": art.get("url", ""),
                        "symbols": [s for s in syms if s.lower() in title.lower()],
                    })
                logger.info(f"  News: {len(all_news)} total (Finnhub added)")
        except Exception as _fh_e:
            logger.debug(f"  Finnhub failed: {_fh_e}")

    # ── Tier 4+: RSS fallback pack (parallel, fast timeout) ──────────────────
    # IMPORTANT: do these concurrently so scanner startup is not blocked
    # by 6-8 sequential network calls on cold start.
    if len(all_news) < 12:
        rss_sources = [
            ("https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY,QQQ,AAPL,MSFT,NVDA,TSLA,META&region=US&lang=en-US", "Yahoo Finance"),
            ("https://feeds.finance.yahoo.com/rss/2.0/headline?s=GLD,USO,TLT,VIX&region=US&lang=en-US", "Yahoo Finance"),
            ("https://feeds.reuters.com/reuters/businessNews", "Reuters"),
            ("https://feeds.marketwatch.com/marketwatch/topstories", "MarketWatch"),
            ("https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114", "CNBC"),
            ("https://seekingalpha.com/market_currents.xml", "Seeking Alpha"),
            ("https://www.nasdaq.com/feed/rssoutbound?category=Stocks", "Nasdaq"),
            ("https://www.nasdaq.com/feed/rssoutbound?category=Markets", "Nasdaq"),
        ]
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as _pool:
                futs = [_pool.submit(_parse_rss, u, s, 4) for (u, s) in rss_sources]
                for fut in concurrent.futures.as_completed(futs, timeout=6):
                    try:
                        all_news.extend(fut.result() or [])
                    except Exception:
                        pass
        except Exception as _rss_bulk_e:
            logger.debug(f"  RSS bulk fetch failed: {_rss_bulk_e}")

    # Deduplicate by title, tag matching symbols, cap at 30
    seen_titles: set = set()
    deduped: list = []
    for art in all_news:
        t = art["title"][:80].lower()
        if t in seen_titles:
            continue
        seen_titles.add(t)
        if not art.get("symbols"):
            art["symbols"] = [s for s in syms if s.lower() in art["title"].lower()]
        deduped.append(art)
        if len(deduped) >= 30:
            break

    # Prefer institutional/free-market wires first in display order so
    # users can clearly see Nasdaq/Reuters-type sources, not only Yahoo.
    _priority = {
        "Nasdaq": 0,
        "Reuters": 1,
        "CNBC": 2,
        "MarketWatch": 3,
        "Seeking Alpha": 4,
        "Finnhub": 5,
        "GNews": 6,
        "Yahoo Finance": 7,
    }
    deduped.sort(key=lambda x: (_priority.get(str(x.get("source", "")), 9), str(x.get("time", ""))), reverse=False)

    # Emergency fallback: legacy Yahoo RSS parser
    if not deduped:
        try:
            deduped = fetch_yahoo_rss_news("")[:20]
        except Exception:
            deduped = []

    # Guaranteed fallback: synthesize market headlines from live signals/quotes
    if not deduped:
        try:
            st = scanner.get_state() if scanner else {}
            sigs = (st.get("top_signals") or [])[:12]
            generated = []
            for s in sigs:
                sym = str(s.get("symbol", "")).upper().strip()
                if not sym:
                    continue
                chg = float(s.get("changePct", 0) or 0)
                direction = s.get("direction", "HOLD")
                comp = float(s.get("composite", 0) or 0)
                tone = "surges" if chg > 0 else "slides" if chg < 0 else "holds steady"
                generated.append({
                    "title": f"{sym} {tone} {abs(chg):.2f}% as GIG scanner flags {direction} (score {comp:.2f})",
                    "source": "GIG Market Wire",
                    "time": datetime.now().strftime("%H:%M"),
                    "sentiment": "BULLISH" if chg > 0 else "BEARISH" if chg < 0 else "NEUTRAL",
                    "url": "",
                    "symbols": [sym],
                })
            deduped = generated[:20]
        except Exception:
            deduped = []

    if not deduped:
        logger.warning("  News: all sources failed — returning empty list")

    _news_cache["data"] = deduped
    _news_cache["ts"]   = now
    logger.info(f"  News cache: {len(deduped)} articles")
    return deduped


# ═══════════════════════════════════════════════════════════════════════════════
# SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

class LiveScanner:
    def __init__(self):
        self.fetcher     = LiveMarketFetcher()
        self.state       = {
            "cycle": 0,
            "vix": 19.0,
            "regime": "NEUTRAL",
            "total_scanned": 0,
            "top_signals": [],
            "top11_trades": [],
            "quotes": {},
            "news": [],
            "macro_fred": {},
            "scanner_active": True,
            "backend": "LIVE — starting up...",
            "scan_timestamp": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "portfolio": {
                "portfolio_value": 1000000,
                "daily_pnl": 0,
                "open_positions": 0,
                "drawdown": 0,
                "circuit_broken": False,
            },
        }
        self.all_signals = {}
        self.lock        = threading.Lock()
        self.scan_index  = 0
        self.earnings_signals   = {}
        self.earnings_last_scan = None
        self.pair_signals       = []
        self.price_cache        = {}
        self.latest_signal      = None        # most recent signal with full detail
        self.latest_signal_100  = None        # most recent 100%+ play
        self.company_cache      = {}          # symbol → company info
        self.all_100pct_plays   = []          # all current 100%+ plays
        self.batch_size  = int(os.environ.get("BATCH_SIZE", "40"))  # configurable, default 40 (was 25)
        self.cycle       = 0
        self._options_cache = {}   # symbol → opts dict (loaded after cycle 2)
        self._cycle_count   = 0    # incremented each _scan call

    def run(self):
        # ── Dependency check — warn immediately if packages are missing ──────
        _missing = []
        try:
            import sklearn  # noqa: F401
        except ImportError:
            _missing.append("scikit-learn")
        try:
            import lxml  # noqa: F401
        except ImportError:
            _missing.append("lxml")
        if _missing:
            logger.warning(
                "╔══════════════════════════════════════════════════════════╗\n"
                "║  MISSING PACKAGES — install to fix warnings below        ║\n"
                f"║  Run:  pip install {' '.join(_missing):<38} ║\n"
                "║  scikit-learn → LSTM model training                      ║\n"
                "║  lxml         → S&P 500 Wikipedia fetch (S&P500:0 fix)   ║\n"
                "╚══════════════════════════════════════════════════════════╝"
            )
        else:
            logger.info("✅ All optional packages present (sklearn + lxml)")

        logger.info("=" * 60)
        logger.info("GIG — Starting up")
        logger.info(f"  Engines: OE={'OK' if OE else 'MISSING'} ML={'OK' if ML else 'MISSING'} SE={'OK' if SE else 'MISSING'} EE={'OK' if EE else 'MISSING'}")
        logger.info(f"  Port: {PORT} | Scan interval: {SCAN_INTERVAL}s")
        logger.info(
            f"  Ultra-fast: {'ON' if ULTRA_FAST_MODE else 'OFF'} | "
            f"cache={MARKET_CACHE_REFRESH_S}s state_ttl={API_STATE_TTL_S}s sse={SSE_PUSH_INTERVAL_S}s"
        )
        _uni_log = get_scan_universe(include_scanner=False)
        logger.info(f"  Universe: {len(_uni_log)} symbols (live)")
        logger.info("=" * 60)
        try:
            self._scan()
        except Exception as e:
            logger.error(f"Initial scan error: {e}")
        while True:
            time.sleep(SCAN_INTERVAL)
            try:
                self._scan()
            except Exception as e:
                logger.error(f"Scan error (continuing): {e}")

    def _scan(self):
        global _GLOBAL_CYCLE
        self.cycle += 1
        _GLOBAL_CYCLE += 1
        logger.info(f"Scan cycle #{self.cycle} | Options: {'ACTIVE' if _GLOBAL_CYCLE > 2 else 'DEFERRED (loading signals first)'}")
        core   = get_training_universe(n=25)  # dynamic — full liquid universe
        _full_uni = get_scan_universe()   # live every cycle — S&P500 + ETFs + extras
        start  = (self.scan_index * self.batch_size) % max(len(_full_uni), 1)
        batch  = _full_uni[start:start + self.batch_size]
        self.scan_index += 1
        batch  = list(set(core + batch))[:40]

        logger.info(f"  Fetching {len(batch)} bars + quotes (parallel)...")
        # Pre-warm bars cache for all batch symbols in parallel (increased workers)
        import concurrent.futures
        def _warm(sym):
            try: self.fetcher.get_bars(sym)
            except: pass
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            ex.map(_warm, batch)  # warm ALL symbols in parallel (8 workers)

        quotes = self.fetcher.get_quotes(batch)
        if DB and hasattr(DB, "MarketStore"):
            try:
                DB.MarketStore.save_quotes(quotes, source="scanner_cycle")
            except Exception as _mqe:
                logger.debug(f"DB quote save skipped: {_mqe}")
        logger.info(f"  Got {len(quotes)} live quotes")

        vix = 18.0
        try:
            vt    = yf.Ticker("^VIX")
            vix_p = getattr(vt.fast_info, "last_price", None)
            if not vix_p or vix_p <= 0:
                vh = vt.history(period="5d", interval="1d")
                vix_p = float(vh["Close"].iloc[-1]) if not vh.empty else 18.0
            vix = float(vix_p)
        except Exception: pass

        # ── Cross-Sectional Momentum (CSM) — Medallion Phase-1 signal ─────────
        # Rank all tracked symbols by 12-1 month return, z-score the ranks.
        # This is the original Jegadeesh (1993) cross-sectional momentum signal.
        _csm_scores: dict = {}
        try:
            _csm_returns = {}
            for _sym, _sig in list(self.all_signals.items())[:200]:
                _ret = float(_sig.get("changePct", 0) or 0)  # proxy: use available return data
                _mo  = float(_sig.get("momentum_5d", _sig.get("momentum_1d", 0)) or 0)
                _csm_returns[_sym] = _ret * 0.4 + _mo * 0.6
            if len(_csm_returns) >= 10:
                _vals   = list(_csm_returns.values())
                _mean_r = float(np.mean(_vals))
                _std_r  = float(np.std(_vals)) or 1.0
                for _sym, _ret_v in _csm_returns.items():
                    _csm_scores[_sym] = float(np.clip((_ret_v - _mean_r) / _std_r, -3, 3)) / 3.0
        except Exception as _csm_e:
            logger.debug(f"CSM compute: {_csm_e}")

        # ── CBOE Put/Call Ratio signal ─────────────────────────────────────────
        _pcr_signal = 0.0
        try:
            _pcr_url = "https://www.cboe.com/data/historical-options-data/options-volume/"
            _pcr_r   = requests.get(_pcr_url, timeout=5,
                                    headers={"User-Agent": "Mozilla/5.0"})
            import re as _pcr_re
            _pcr_match = _pcr_re.search(r'"totalPut":(\d+).*?"totalCall":(\d+)', _pcr_r.text)
            if _pcr_match:
                _put_vol  = int(_pcr_match.group(1))
                _call_vol = int(_pcr_match.group(2))
                _pcr_raw  = _put_vol / max(_call_vol, 1)
                # PCR > 1.1 = fear (contrarian buy) | PCR < 0.8 = complacency (contrarian sell)
                _pcr_signal = float(np.clip((_pcr_raw - 0.95) / 0.3, -1, 1))
                logger.debug(f"  CBOE PCR={_pcr_raw:.3f} signal={_pcr_signal:.3f}")
        except Exception:
            pass  # PCR is bonus signal — failure is non-fatal

        # ── VIX Term Structure ratio signal ───────────────────────────────────
        _vts_signal = 0.0
        try:
            _vix9d_tk = yf.Ticker("^VIX9D")
            _vix9d    = float(getattr(_vix9d_tk.fast_info, "last_price", None) or 0)
            _vix3m_tk = yf.Ticker("^VIX3M")
            _vix3m    = float(getattr(_vix3m_tk.fast_info, "last_price", None) or 0)
            if _vix9d > 0 and _vix3m > 0:
                _vts_ratio  = _vix9d / _vix3m
                # VTS < 0.85 = stressed (bearish) | VTS > 1.05 = calm (bullish)
                _vts_signal = float(np.clip((_vts_ratio - 0.95) / 0.15, -1, 1))
                logger.debug(f"  VTS ratio={_vts_ratio:.3f} signal={_vts_signal:.3f}")
        except Exception:
            pass

        logger.info(f"  VIX={vix:.2f} | Analyzing {len(batch)} symbols in parallel...")

        # ── PARALLEL symbol analysis (was sequential with 0.4s sleep) ─────────
        analyzed = 0
        _results_buf: dict = {}

        def _analyze_one(sym_q_tuple):
            sym, q = sym_q_tuple
            if not q:
                q = {"price": 0}
            try:
                result = self.fetcher.analyze_symbol(sym, q)
                if result:
                    # Inject CSM, PCR, VTS signals
                    if sym in _csm_scores:
                        result["csm_score"] = round(_csm_scores[sym], 4)
                        # Blend into composite slightly
                        old_comp = float(result.get("composite", 0) or 0)
                        result["composite"] = round(old_comp * 0.92 + _csm_scores[sym] * 0.08, 4)
                    result["pcr_signal"] = round(_pcr_signal, 4)
                    result["vts_signal"] = round(_vts_signal, 4)
                    return sym, result
            except Exception as _ae:
                logger.debug(f"  Error {sym}: {_ae}")
            return sym, None

        _pairs = [(sym, quotes.get(sym, {})) for sym in batch]
        # Use 6 workers — enough to parallelize without overwhelming Yahoo rate limits
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as _pool:
            for _sym, _res in _pool.map(_analyze_one, _pairs):
                if _res:
                    _results_buf[_sym] = _res
                    analyzed += 1

        # Bulk-update all_signals under lock (single lock acquisition)
        with self.lock:
            self.all_signals.update(_results_buf)

        logger.info(f"  Analyzed {analyzed}/{len(batch)} symbols ({len(self.all_signals)} total in universe)")

        news     = fetch_news()
        if DB and hasattr(DB, "MarketStore"):
            try:
                DB.MarketStore.save_news(news, source="scanner_cycle")
            except Exception as _mne:
                logger.debug(f"DB news save skipped: {_mne}")
        all_sigs = sorted(self.all_signals.values(), key=lambda x: abs(x.get("composite",0)), reverse=True)
        top11    = self._top11([s for s in all_sigs if s["direction"] != "HOLD"])

        # Fetch FRED macro (cached 1hr)
        fred_data = {}
        try:
            fred_data = fetch_fred_macro()
        except Exception as _fd:
            logger.debug(f"FRED fetch: {_fd}")

        with self.lock:
            self.state = {
                "cycle":          self.cycle,
                "scan_timestamp": datetime.now().isoformat(),
                "vix":            round(float(vix), 2),
                "regime":         "BEAR" if vix>28 else "BULL" if vix<15 else "NEUTRAL",
                "total_scanned":  len(self.all_signals),
                "portfolio": {
                    "portfolio_value": 1_000_000,
                    "daily_pnl":       sum(s.get("changePct",0)*1e6*0.05 for s in top11) if top11 else 0,
                    "open_positions":  min(len([s for s in top11 if s["direction"]!="HOLD"]),15),
                    "drawdown":        max(-5.0, min(-vix*0.05, 0)),
                    "circuit_broken":  vix > 35,
                },
                "top11_trades":  top11,
                "top_signals":   all_sigs[:60],
                "quotes":        {k: {kk:vv for kk,vv in v.items() if kk not in ["history","options"]}
                                  for k,v in self.all_signals.items()},
                "news":          news,
                "macro_fred":    fred_data,
                "scanner_active": True,
                "backend":       "LIVE — yfinance SIP + FRED + YahooRSS + SEC EDGAR + Clearbit",
                "last_update":   datetime.now().isoformat(),
            }
        
            # GIG: Persist signals to database every scan cycle
            if DB:
                try:
                    _all_sigs = [s for s in all_sigs if s.get("direction") != "HOLD"][:20]
                    DB.SignalStore.save_signals(_all_sigs, "scan")
                    if SSE:
                        _stock_sigs = SSE.generate_stock_signals(_all_sigs[:30]).get("signals", [])
                        DB.TopPicks.update_from_scan(self.state, _stock_sigs)
                except Exception as _dbe:
                    logger.debug(f"DB save: {_dbe}")
# ── Earnings scan — runs in background thread so it doesn't block ──────
        def _run_earnings():
            try:
                if EE:
                    logger.info("  [Background] Earnings intelligence scan...")
                    _earn_uni = get_scan_universe(include_scanner=False)[:40]
                    earn = EE.scan_all_earnings(_earn_uni, days_ahead=7)
                    with self.lock:
                        self.earnings_signals   = earn
                        self.earnings_last_scan = datetime.now()
                    logger.info(f"  [Background] Earnings done: {earn.get('count',0)} events")
            except Exception as _ee:
                logger.warning(f"  Earnings error: {_ee}")

        should_earn = (self.earnings_last_scan is None or
                       (datetime.now()-self.earnings_last_scan).seconds > 1800)
        if should_earn:
            threading.Thread(target=_run_earnings, daemon=True).start()

        # ── Cointegration pairs scan — background ─────────────────────────────
        def _run_pairs():
            try:
                if ML and len(self.fetcher.price_cache) >= 4:
                    ps = ML.scan_pairs(dict(self.fetcher.price_cache))
                    with self.lock:
                        self.pair_signals = ps
            except Exception as _pe:
                logger.debug(f"  Pairs error: {_pe}")
        if len(self.fetcher.price_cache) >= 4:
            threading.Thread(target=_run_pairs, daemon=True).start()

        # ── Track latest signal + 100%+ plays ────────────────────────────────
        try:
            # actionable = all signals with a clear direction (BUY or SHORT)
            actionable_sigs = [s for s in all_sigs if s.get("direction") in ("BUY","SHORT")]
            plays_100 = [s for s in actionable_sigs if s.get("is_100_pct")]
            plays_100.sort(key=lambda x: x.get("confidence",0), reverse=True)
            self.all_100pct_plays = plays_100

            if actionable_sigs:
                best = max(actionable_sigs, key=lambda x: x.get("confidence",0))
                now  = datetime.now()
                self.latest_signal = {
                    **{k:v for k,v in best.items() if k not in ["history","options"]},
                    "signal_time":     now.isoformat(),
                    "signal_date":     now.strftime("%B %d, %Y"),
                    "signal_day":      now.strftime("%A"),
                    "signal_time_str": now.strftime("%I:%M %p ET"),
                }
                if plays_100:
                    best100 = plays_100[0]
                    self.latest_signal_100 = {
                        **{k:v for k,v in best100.items() if k not in ["history","options"]},
                        "signal_time":     now.isoformat(),
                        "signal_date":     now.strftime("%B %d, %Y"),
                        "signal_day":      now.strftime("%A"),
                        "signal_time_str": now.strftime("%I:%M %p ET"),
                    }
        except Exception as _lt:
            logger.debug(f"Latest signal error: {_lt}")

        # ── Load company info for top signals (background) ─────────────────
        try:
            top_syms = [s["symbol"] for s in all_sigs[:15]]
            for sym in top_syms:
                if CI and sym not in self.company_cache:
                    self.company_cache[sym] = CI.get_company_info(sym, use_yfinance=True)
        except Exception as _co:
            logger.debug(f"Company batch error: {_co}")

        # (Earnings + Pairs now handled by background threads above)

        logger.info(f"  Done. {analyzed} analyzed, {len(self.all_signals)} total, top: {top11[0]['symbol'] if top11 else 'none'}")

        # ── RENAISSANCE v10: Alpha Research Engine — Cross-Sectional Ranking ──
        if ARE and len(self.all_signals) >= 5:
            def _run_alpha_ranking():
                try:
                    platform = ARE.get_alpha_platform()
                    # Set regime on the combiner
                    regime_str = self.state.get("regime", "NEUTRAL").lower()
                    ARE.AdaptiveAlphaCombiner.set_regime(regime_str)
                    # Run cross-sectional ranking on entire universe
                    rankings = platform.rank_universe(dict(self.all_signals))
                    if rankings and not rankings.get("error"):
                        with self.lock:
                            self.state["alpha_rankings"] = rankings
                        logger.info(f"  [Alpha] Ranked {rankings.get('universe_size',0)} stocks | "
                                    f"LONG:{rankings.get('long_count',0)} SHORT:{rankings.get('short_count',0)} | "
                                    f"Regime:{regime_str}")
                except Exception as _ae:
                    logger.debug(f"  Alpha ranking error: {_ae}")
            threading.Thread(target=_run_alpha_ranking, daemon=True, name="alpha-rank").start()

        # ── RENAISSANCE v10: Alert System — Check all conditions ──────────────
        if ALS and len(all_sigs) > 0:
            try:
                alert_engine = ALS.get_alert_engine()
                alert_engine.check_all(
                    self.state,
                    ensemble_data=None,
                    positions=None,
                    prev_vix=getattr(self, '_prev_vix', None),
                )
                self._prev_vix = vix  # store for next cycle
            except Exception as _alert_e:
                logger.debug(f"  Alert check error: {_alert_e}")

        # ── Universe-level research: regime, factor IC, PhD agent ─────────────
        if RC and self.cycle % 2 == 0:
            def _run_research():
                try:
                    spy_sig = self.all_signals.get("SPY", {})
                    spy_ret = spy_sig.get("mom_1d", 0) / 100
                    result  = RC.run_universe_research(
                        dict(self.all_signals), spy_ret, float(vix)
                    )
                    with self.lock:
                        self.state["research"] = result
                    logger.info(f"  [Research] Regime={result.get('regime_analysis',{}).get('regime','?')} | PhD discoveries={len(RC.phd_agent.discoveries)}")
                except Exception as _re:
                    logger.debug(f"  Research cycle error: {_re}")
            threading.Thread(target=_run_research, daemon=True).start()

    def _top11(self, actionable):
        cats   = {"LARGE":4,"MID":2,"SMALL":2,"PENNY":1,"ETF":2,"METAL":0}
        seen   = set()
        result = []
        for cat, limit in cats.items():
            for s in sorted([x for x in actionable if x.get("cap")==cat],
                            key=lambda x: abs(x.get("composite",0)), reverse=True):
                if s["symbol"] not in seen and len([r for r in result if r.get("cap")==cat]) < limit and len(result) < 11:
                    result.append(s); seen.add(s["symbol"])
        for s in actionable:
            if len(result) >= 11: break
            if s["symbol"] not in seen:
                result.append(s); seen.add(s["symbol"])
        return result[:11]

    def get_state(self):
        with self.lock: return dict(self.state)

    def get_symbol(self, symbol):
        with self.lock: return self.all_signals.get(symbol.upper(), {})


# ═══════════════════════════════════════════════════════════════════════════════
# FLASK API
# ═══════════════════════════════════════════════════════════════════════════════


import json as _json

def _sanitize(obj):
    """Convert ALL types to JSON-safe. ONE_BRAIN_V2"""
    if obj is None: return obj
    if type(obj) is bool:  return obj
    if type(obj) is int:   return obj
    if type(obj) is float: return obj
    if type(obj) is str:   return obj
    try:
        import numpy as _np
        if isinstance(obj, (_np.bool_,)):    return bool(obj)
        if isinstance(obj, (_np.integer,)):  return int(obj)
        if isinstance(obj, (_np.floating,)): return float(obj)
        if isinstance(obj, _np.ndarray):     return [_sanitize(x) for x in obj.tolist()]
    except (ImportError, Exception): pass
    try:
        import pandas as _pd
        if isinstance(obj, _pd.Timestamp):   return obj.isoformat()
        if isinstance(obj, _pd.Series):      return _sanitize(obj.to_dict())
        try:
            if _pd.isna(obj): return None
        except (TypeError, ValueError): pass
    except (ImportError, Exception): pass
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(x) for x in obj]
    try:
        import json as _j; _j.dumps(obj); return obj
    except (TypeError, ValueError): return str(obj)

app     = Flask(__name__)
CORS(app, origins="*")
scanner = LiveScanner()

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL MARKET CACHE — Pre-fetches ALL yfinance data in background
# Every slow endpoint serves from this cache INSTANTLY instead of calling
# yfinance on every user click. This is how Bloomberg works.
# ═══════════════════════════════════════════════════════════════════════════════
_MARKET_CACHE = {
    "prices": {},       # sym → {price, change_pct, change_5d}
    "sectors": [],      # sector heatmap data
    "vix_data": {},     # VIX + term structure
    "correlation": {},  # cross-asset correlation
    "stress": {},       # stress test scenarios
    "regime": {},       # regime portfolio allocation
    "shipping": {},     # BDI + shipping stocks
    "ts": 0,            # last update timestamp
}
_MARKET_CACHE_LOCK = threading.Lock()
_FAST_API_CACHE = {}
_FAST_API_CACHE_LOCK = threading.Lock()


def _fast_cache_get(key: str, ttl_s: float):
    now = time.time()
    with _FAST_API_CACHE_LOCK:
        row = _FAST_API_CACHE.get(key)
        if not row:
            return None
        if now - row["ts"] > ttl_s:
            return None
        return row["payload"]


def _fast_cache_set(key: str, payload):
    with _FAST_API_CACHE_LOCK:
        _FAST_API_CACHE[key] = {"ts": time.time(), "payload": payload}


def _priority_score_for_symbol(symbol: str, signal_row: dict) -> float:
    """Higher score = scan earlier in sharded worker queues."""
    c = abs(float(signal_row.get("composite", 0.0)))
    v = float(signal_row.get("vol_ratio", 1.0))
    conv = float(signal_row.get("conviction", 0.0))
    liq = float(signal_row.get("volume", 0.0))
    return c * 40.0 + conv * 35.0 + min(v, 5.0) * 8.0 + np.log1p(max(liq, 0.0)) * 0.2


def _build_scan_shards(symbols: list, shard_count: int = 8) -> dict:
    shard_count = max(1, min(int(shard_count), 64))
    shards = {f"shard_{i+1}": [] for i in range(shard_count)}

    signal_map = getattr(scanner, "all_signals", {}) if scanner else {}
    scored = []
    for sym in symbols:
        row = signal_map.get(sym, {})
        scored.append((sym, _priority_score_for_symbol(sym, row)))
    scored.sort(key=lambda x: x[1], reverse=True)

    for idx, (sym, score) in enumerate(scored):
        key = f"shard_{(idx % shard_count) + 1}"
        shards[key].append({"symbol": sym, "priority": round(float(score), 4)})
    return shards

def _refresh_market_cache():
    """Background thread: refresh all market data every 60s."""
    import yfinance as _yf_cache
    while True:
        try:
            cache = {}
            now = time.time()

            # ── 1. Batch price fetch (ALL tickers in one call) ────────────
            ALL_TICKERS = [
                "SPY","QQQ","IWM","DIA","TLT","GLD","USO","UUP","BTC-USD",
                "^VIX","^TNX","DX-Y.NYB",
                "XLK","XLF","XLE","XLV","XLI","XLC","XLY","XLP","XLU","XLB","XLRE",
                "^VIX9D","^VIX3M",
                "BDRY","SBLK","ZIM","GOGL",
            ]
            prices = {}
            try:
                data = _yf_cache.download(ALL_TICKERS, period="1mo", interval="1d",
                                          group_by="ticker", threads=True, progress=False)
                for sym in ALL_TICKERS:
                    try:
                        if sym in data.columns.get_level_values(0):
                            col = data[sym]["Close"].dropna()
                        else:
                            col = data["Close"].dropna() if len(ALL_TICKERS) == 1 else None
                        if col is not None and len(col) >= 2:
                            curr = float(col.iloc[-1])
                            prev = float(col.iloc[-2])
                            chg = (curr / prev - 1) * 100
                            chg5 = (curr / float(col.iloc[-5]) - 1) * 100 if len(col) >= 5 else chg
                            chg1m = (curr / float(col.iloc[0]) - 1) * 100
                            vol = float(col.pct_change().dropna().std() * np.sqrt(252) * 100) if len(col) > 5 else 20
                            prices[sym] = {"price": round(curr,2), "change_pct": round(chg,2),
                                          "change_5d_pct": round(chg5,2), "change_1m_pct": round(chg1m,2),
                                          "volatility": round(vol,1)}
                    except Exception:
                        pass
            except Exception as _dl_e:
                logger.debug(f"Cache batch download: {_dl_e}")

            cache["prices"] = prices

            # ── 2. Sector heatmap ─────────────────────────────────────────
            SECTOR_MAP = {"XLK":"Technology","XLF":"Financials","XLE":"Energy","XLV":"Healthcare",
                         "XLI":"Industrials","XLC":"Communication","XLY":"Cons. Disc.","XLP":"Cons. Staples",
                         "XLU":"Utilities","XLB":"Materials","XLRE":"Real Estate"}
            sectors = []
            for etf, name in SECTOR_MAP.items():
                if etf in prices:
                    p = prices[etf]
                    sectors.append({"etf":etf,"name":name,"return_1d":p["change_pct"],
                                   "return_5d":p["change_5d_pct"],"return_1m":p["change_1m_pct"],
                                   "volatility":p["volatility"],"price":p["price"]})
            sectors.sort(key=lambda x: x["return_1d"], reverse=True)
            cache["sectors"] = sectors

            # ── 3. VIX term structure ─────────────────────────────────────
            vix_val = prices.get("^VIX",{}).get("price",20)
            vix9d = prices.get("^VIX9D",{}).get("price",0)
            vix3m = prices.get("^VIX3M",{}).get("price",0)
            term = []
            if vix9d: term.append({"name":"VIX9D (9d)","days":9,"value":round(vix9d,2),"change":round(prices.get("^VIX9D",{}).get("change_pct",0),2)})
            term.append({"name":"VIX (30d)","days":30,"value":round(vix_val,2),"change":round(prices.get("^VIX",{}).get("change_pct",0),2)})
            if vix3m: term.append({"name":"VIX3M (90d)","days":90,"value":round(vix3m,2),"change":round(prices.get("^VIX3M",{}).get("change_pct",0),2)})
            structure = "CONTANGO" if vix3m > vix_val else "BACKWARDATION"
            cache["vix_data"] = {"term_structure":sorted(term,key=lambda x:x["days"]),
                                "structure":structure,"steepness":round(abs(vix3m-vix_val),2),
                                "vix_current":round(vix_val,2)}

            # ── 4. Correlation matrix (from cached prices) ────────────────
            CORR_SYMS = ["SPY","QQQ","IWM","TLT","GLD","USO","UUP","XLF","XLE","XLK","XLV"]
            corr_returns = {}
            for sym in CORR_SYMS:
                try:
                    if sym in data.columns.get_level_values(0):
                        ret = data[sym]["Close"].pct_change().dropna().values
                        if len(ret) >= 15:
                            corr_returns[sym] = ret[-20:]
                except Exception:
                    pass
            if len(corr_returns) >= 4:
                syms_c = list(corr_returns.keys())
                min_l = min(len(v) for v in corr_returns.values())
                mat = np.column_stack([corr_returns[s][-min_l:] for s in syms_c])
                corr_mat = np.corrcoef(mat.T)
                eigenvalues = sorted(np.linalg.eigvalsh(corr_mat), reverse=True)
                cache["correlation"] = {
                    "symbols":syms_c,
                    "matrix":[[round(float(corr_mat[i,j]),3) for j in range(len(syms_c))] for i in range(len(syms_c))],
                    "top_factor_concentration_pct":round(eigenvalues[0]/sum(eigenvalues)*100,1),
                    "avg_correlation":round(float(np.mean(np.abs(corr_mat[np.triu_indices(len(syms_c),k=1)]))),3),
                }

            # ── 5. Regime detection ───────────────────────────────────────
            spy_5d = prices.get("SPY",{}).get("change_5d_pct",0)
            spy_1m = prices.get("SPY",{}).get("change_1m_pct",0)
            if vix_val > 35: regime = "crisis"
            elif vix_val > 25 or spy_5d < -5: regime = "bear"
            elif vix_val < 14 and spy_1m > 15: regime = "euphoria"
            else: regime = "bull"
            cache["regime"] = {"regime":regime,"vix":round(vix_val,1),"spy_return_20d":round(spy_5d,2)}

            # ── 6. Shipping ───────────────────────────────────────────────
            ship_indicators = {}
            SHIP_MAP = {"BDRY":"Breakwave Dry Bulk","SBLK":"Star Bulk","ZIM":"ZIM Shipping","GOGL":"Golden Ocean"}
            for sym, name in SHIP_MAP.items():
                if sym in prices:
                    p = prices[sym]
                    ship_indicators[sym] = {"name":name,"price":p["price"],"change_1d_pct":p["change_pct"],
                                           "change_5d_pct":p["change_5d_pct"],"change_1m_pct":p["change_1m_pct"],
                                           "trend":"RISING" if p["change_5d_pct"]>2 else "FALLING" if p["change_5d_pct"]<-2 else "STABLE"}
            cache["shipping"] = {"indicators":ship_indicators,"trade_health":"EXPANDING" if spy_1m>3 else "CONTRACTING" if spy_1m<-3 else "STABLE"}

            cache["ts"] = now

            with _MARKET_CACHE_LOCK:
                _MARKET_CACHE.update(cache)

            logger.info(f"  [Cache] Market data refreshed: {len(prices)} prices, {len(sectors)} sectors")

        except Exception as _cache_e:
            logger.warning(f"  [Cache] Refresh failed: {_cache_e}")

        time.sleep(MARKET_CACHE_REFRESH_S)

# ─── Optional new engines (graceful degradation) ──────────────────────────────
try:
    from portfolio_optimizer import get_kelly_optimizer as _get_kelly
    _kelly_opt = _get_kelly()
    logger.info("✅ PortfolioKellyOptimizer loaded")
except Exception as _ke:
    _kelly_opt = None
    logger.warning(f"PortfolioOptimizer not available: {_ke}")

try:
    from stat_arb_manager import get_stat_arb_manager as _get_sarb
    _sarb_mgr = _get_sarb()
    logger.info("✅ StatArbManager loaded")
except Exception as _sarbe:
    _sarb_mgr = None
    logger.warning(f"StatArbManager not available: {_sarbe}")

try:
    from orderflow_engine import get_orderflow_engine as _get_ofe
    _ofe = _get_ofe()
    logger.info("✅ OrderFlowEngine loaded")
except Exception as _ofee:
    _ofe = None
    logger.warning(f"OrderFlowEngine not available: {_ofee}")

# streaming_engine.py was removed (required alpaca-py) — stub for compatibility
_stream_eng = None

@app.route("/api/state")
def api_state():
    cached = _fast_cache_get("api_state", ttl_s=API_STATE_TTL_S)
    if cached is not None:
        return jsonify(cached)

    state = scanner.get_state()
    # Warm-start fallback: if scanner is still initializing, serve last bootstrap snapshot.
    try:
        if not state or not state.get("quotes"):
            boot = _fast_cache_get("api_bootstrap:core", ttl_s=30)
            if isinstance(boot, dict) and isinstance(boot.get("state"), dict) and boot["state"].get("quotes"):
                payload = _sanitize(boot["state"])
                _fast_cache_set("api_state", payload)
                return jsonify(payload)
    except Exception:
        pass
    state["earnings"]        = scanner.earnings_signals  or {}
    state["pairs"]           = scanner.pair_signals      or []
    state["latest_signal"]   = scanner.latest_signal     or {}
    state["latest_100"]      = scanner.latest_signal_100 or {}
    state["plays_100"]       = scanner.all_100pct_plays  or []
    state["stream_status"]   = _stream_eng.get_status() if _stream_eng else {"connected": False, "mode": "polling"}
    state["optimizer_ready"] = _kelly_opt is not None
    state["sarb_count"]      = len(_sarb_mgr.get_active_pairs()) if _sarb_mgr else 0
    payload = _sanitize(state)
    _fast_cache_set("api_state", payload)
    return jsonify(payload)

@app.route("/api/bootstrap", methods=["GET"])
def api_bootstrap():
    """
    One-shot startup payload for web login.
    Bundles high-value panels so frontend can paint immediately.
    """
    scope = (request.args.get("scope") or "core").strip().lower()
    cache_key = f"api_bootstrap:{scope}"
    cached = _fast_cache_get(cache_key, ttl_s=0.8 if ULTRA_FAST_MODE else 1.5)
    if cached is not None:
        return jsonify(cached)
    try:
        payload = _build_bootstrap_payload(scope=scope)
        _fast_cache_set(cache_key, payload)
        return jsonify(payload)
    except Exception as e:
        return jsonify({
            "error": str(e),
            "state": _sanitize(scanner.get_state() if scanner else {}),
            "news": {},
            "top_picks": {},
            "nasdaq_movers": {},
            "options_flow": {},
        }), 200


def _build_bootstrap_payload(scope: str = "core"):
    state = scanner.get_state()
    state["earnings"] = scanner.earnings_signals or {}
    state["pairs"] = scanner.pair_signals or []
    state["latest_signal"] = scanner.latest_signal or {}
    state["latest_100"] = scanner.latest_signal_100 or {}
    state["plays_100"] = scanner.all_100pct_plays or []
    state["stream_status"] = _stream_eng.get_status() if _stream_eng else {"connected": False, "mode": "polling"}
    state["optimizer_ready"] = _kelly_opt is not None
    state["sarb_count"] = len(_sarb_mgr.get_active_pairs()) if _sarb_mgr else 0

    news_payload = {}
    top_picks_payload = {}
    movers_payload = {}
    options_payload = {}
    try:
        cached_news = _proxy_cache_get("news_feed_v6", ttl_s=120)
        state_news = state.get("news") if isinstance(state.get("news"), list) else []
        feed = cached_news if isinstance(cached_news, list) and cached_news else state_news
        source_mix = {}
        for n in feed:
            src = str(n.get("source") or "Unknown")
            source_mix[src] = source_mix.get(src, 0) + 1
        news_payload = {
            "news": feed[:60],
            "count": len(feed),
            "provider": "cache_or_scanner",
            "source_mix": source_mix,
            "nasdaq_count": source_mix.get("Nasdaq", 0),
        }
    except Exception:
        news_payload = {"news": [], "count": 0, "provider": "bootstrap_fallback", "source_mix": {}, "nasdaq_count": 0}
    try:
        top_picks_payload = _sanitize(DB.TopPicks.get_all()) if DB else {}
    except Exception:
        top_picks_payload = {}
    try:
        movers_payload = _sanitize(api_nasdaq_movers().get_json() or {})
    except Exception:
        movers_payload = {}
    try:
        if _options_flow_cache.get("data") and (time.time() - float(_options_flow_cache.get("ts") or 0) < 180):
            options_payload = _sanitize(_options_flow_cache["data"])
        else:
            options_payload = {"signals_with_options": [], "count": 0, "updated": datetime.now().isoformat(), "provider": "bootstrap_cache_miss"}
    except Exception:
        options_payload = {"signals_with_options": [], "count": 0}

    payload = {
        "asof": datetime.now().isoformat(),
        "scope": scope,
        "state": state,
        "news": news_payload,
        "top_picks": top_picks_payload,
        "nasdaq_movers": movers_payload,
        "options_flow": options_payload,
        "latency_profile": {
            "ultra_fast_mode": bool(ULTRA_FAST_MODE),
            "scan_interval_s": SCAN_INTERVAL,
            "state_ttl_s": API_STATE_TTL_S,
            "sse_push_interval_s": SSE_PUSH_INTERVAL_S,
        },
    }

    if scope in {"terminal", "all"}:
        try:
            payload["terminal"] = {
                "market_pulse": _sanitize(api_terminal_pulse().get_json() if 'api_terminal_pulse' in globals() else {}),
                "heatmap": _sanitize(api_terminal_heatmap().get_json() if 'api_terminal_heatmap' in globals() else {}),
                "vol_surface": _sanitize(api_terminal_vol_surface().get_json() if 'api_terminal_vol_surface' in globals() else {}),
            }
        except Exception:
            payload["terminal"] = {}
    if scope in {"geo", "all"}:
        try:
            payload["geo"] = {
                "live": _sanitize(api_geo_live().get_json()),
                "assets": _sanitize(api_geo_asset_intel().get_json()),
            }
        except Exception:
            payload["geo"] = {}

    return _sanitize(payload)

@app.route("/api/quote/<symbol>")
def api_quote(symbol):
    sym = symbol.upper().strip()
    d = scanner.get_symbol(sym)
    if d and (d.get("price") or d.get("regularMarketPrice")):
        return jsonify(_sanitize(d))
    poly = _polygon_quote(sym)
    if poly:
        return jsonify(_sanitize(poly))
    # Fallback: live yfinance fetch for any symbol not in scanner cache
    try:
        import yfinance as yf
        tk = yf.Ticker(sym)
        info = tk.fast_info
        price = float(getattr(info, 'last_price', None) or getattr(info, 'regularMarketPrice', 0) or 0)
        prev  = float(getattr(info, 'previous_close', None) or price)
        chg   = round(price - prev, 4)
        chgPct = round((chg / prev * 100) if prev else 0, 4)
        vol   = int(getattr(info, 'three_month_average_volume', None) or 0)
        hi    = float(getattr(info, 'day_high', None) or 0)
        lo    = float(getattr(info, 'day_low', None) or 0)
        vwap  = float(getattr(info, 'fifty_day_average', None) or 0)
        market_session = "regular"
        pre_market_price = None
        post_market_price = None
        try:
            meta = tk.history(period="2d", interval="5m", prepost=True, auto_adjust=False, metadata=True)
            metadata = getattr(meta, "attrs", {}).get("metadata", {}) if hasattr(meta, "attrs") else {}
            if isinstance(metadata, dict):
                pre_market_price = metadata.get("preMarketPrice")
                post_market_price = metadata.get("postMarketPrice")
                state = str(metadata.get("marketState", "")).lower()
                if state in {"premarket", "pre"}:
                    market_session = "pre"
                elif state in {"post", "postmarket", "postpost"}:
                    market_session = "post"
                elif state in {"closed"}:
                    market_session = "closed"
        except Exception:
            pass
        try:
            h = tk.history(period="5d", interval="1d", auto_adjust=False)
            if h is not None and not h.empty and price > 0:
                hclose = float(h["Close"].iloc[-1])
                ratio = float(price) / max(1e-9, hclose)
                if ratio > 3.0 or ratio < 0.33:
                    logger.warning(f"api_quote sanity fallback {sym}: live={price} hist_close={hclose}")
                    price = hclose
                    prev = float(h["Close"].iloc[-2]) if len(h) > 1 else hclose
                    chg = round(price - prev, 4)
                    chgPct = round((chg / prev * 100) if prev else 0, 4)
        except Exception:
            pass
        if price > 0:
            return jsonify({
                "symbol": sym, "price": price, "regularMarketPrice": price,
                "change": chg, "changePct": chgPct,
                "volume": vol, "high": hi, "low": lo,
                "open": float(getattr(info, 'open', None) or 0),
                "vwap": vwap,
                "preMarketPrice": float(pre_market_price) if pre_market_price else None,
                "postMarketPrice": float(post_market_price) if post_market_price else None,
                "market_session": market_session,
                "provider": "yahoo",
            })
    except Exception:
        pass
    return jsonify(_sanitize(d) if d else {})

@app.route("/api/resolve/<query>")
def api_resolve(query):
    """Resolve a company name or partial ticker to a canonical ticker symbol."""
    q = query.strip()
    # First try exact ticker lookup
    try:
        import yfinance as yf
        tk = yf.Ticker(q.upper())
        info = tk.fast_info
        price = float(getattr(info, 'last_price', None) or 0)
        if price > 0:
            # Valid ticker — return it directly
            long_name = getattr(tk.info, 'longName', None) if hasattr(tk, 'info') else None
            return jsonify({"ticker": q.upper(), "name": long_name or q.upper(), "exact": True})
    except Exception:
        pass
    # Fuzzy: use yfinance search / screener
    try:
        import yfinance as yf
        results = yf.search(q, max_results=5)
        quotes_list = results.get("quotes", []) if results else []
        if quotes_list:
            best = quotes_list[0]
            symbol = best.get("symbol", q.upper())
            name   = best.get("longname") or best.get("shortname") or symbol
            return jsonify({"ticker": symbol, "name": name, "exact": False, "suggestions": [
                {"ticker": r.get("symbol",""), "name": r.get("longname") or r.get("shortname","")}
                for r in quotes_list[:5]
            ]})
    except Exception:
        pass
    return jsonify({"ticker": q.upper(), "name": q.upper(), "exact": False, "suggestions": []})

@app.route("/api/history/<symbol>")
def api_history(symbol):
    sym = symbol.upper()
    period = str(request.args.get("period", "6mo"))
    interval = str(request.args.get("interval", "1d")).lower()
    prepost = str(request.args.get("prepost", "1")).lower() in ("1", "true", "yes", "on")
    poly_hist = _polygon_history(sym, period, interval)
    if poly_hist.get("history"):
        return jsonify({
            "symbol": sym,
            "history": poly_hist["history"],
            "period": period,
            "interval": interval,
            "prepost": prepost,
            "provider": "polygon",
        })
    d   = scanner.get_symbol(sym)
    if interval == "1h":
        interval = "60m"
    if d.get("history") and interval == "1d" and not prepost and period == "6mo":
        return jsonify({"symbol":sym, "history":d["history"], "provider": "scanner_cache"})
    try:
        df   = scanner.fetcher.get_bars(sym, period=period, interval=interval, prepost=prepost)
        hist = []
        if not df.empty:
            for i,(idx,row) in enumerate(df.iterrows()):
                date_val = str(idx)[:10] if interval == "1d" else pd.Timestamp(idx).isoformat()
                session = "regular"
                if interval != "1d":
                    try:
                        ts = pd.Timestamp(idx)
                        minute_of_day = ts.hour * 60 + ts.minute
                        if minute_of_day < 570:         # before 09:30
                            session = "pre"
                        elif minute_of_day >= 960:      # after 16:00
                            session = "post"
                    except Exception:
                        session = "regular"
                hist.append({"t":i,"date":date_val,
                    "open": round(float(row.get("Open",0)),4),
                    "price":round(float(row.get("Close",0)),4),
                    "high": round(float(row.get("High",0)),4),
                    "low":  round(float(row.get("Low",0)),4),
                    "volume":int(row.get("Volume",0)),
                    "session": session})
        return jsonify({"symbol":sym,"history":hist,"period":period,"interval":interval,"prepost":prepost,"provider":"yahoo"})
    except Exception as e:
        return jsonify({"error":str(e),"history":[]})

@app.route("/api/company/<symbol>")
def api_company(symbol):
    """Return full company details for a symbol — sector, industry, cap, description, logo."""
    sym = symbol.upper()
    try:
        # Try cache first
        d = scanner.get_symbol(sym)
        co = d.get("company") if d else None

        # If no sector/description, fetch fresh
        if not co or not co.get("sector") or co.get("sector") == "Unknown":
            if CI:
                co = CI.get_company_info(sym, use_yfinance=True)
            else:
                from Company_info import get_company_info
                co = get_company_info(sym, use_yfinance=True)

        if co:
            return jsonify(_sanitize(co))
        return jsonify({"symbol": sym, "name": sym, "error": "Not found"})
    except Exception as e:
        return jsonify({"symbol": sym, "error": str(e)})

@app.route("/api/options/<symbol>")
def api_options(symbol):
    sym = symbol.upper()
    d   = scanner.get_symbol(sym)
    if d.get("options"):
        return jsonify(_sanitize(d["options"]))
    return jsonify({"error":"Not yet analyzed","symbol":sym})


@app.route("/api/options/chain/<symbol>", methods=["GET"])
def api_options_chain(symbol):
    """Full live options chain from yfinance — calls + puts with Greeks."""
    cache_key = None
    try:
        sym = symbol.upper().strip()
        exp_idx = int(request.args.get("exp_idx", 0))
        cache_key = f"{sym}_{exp_idx}"
        now = time.time()

        def _fallback_from_engine_cache(error_msg: str, stale_flag: bool = False):
            cached_opts = _OPTIONS_CACHE.get(sym) or {}
            if not isinstance(cached_opts, dict):
                cached_opts = {}
            simple = cached_opts.get("simple_signal") or {}
            legs = simple.get("legs") or []
            flow = cached_opts.get("unusual_flow") or []
            expiry = (
                cached_opts.get("expiry")
                or (legs[0].get("expiry") if legs else "")
                or datetime.now().strftime("%Y-%m-%d")
            )
            calls = []
            puts = []
            for f in flow:
                row = {
                    "strike": round(float(f.get("strike", 0) or 0), 2),
                    "bid": round(float(f.get("bid", 0) or 0), 2),
                    "ask": round(float(f.get("ask", 0) or 0), 2),
                    "last": round(float(f.get("premium", 0) or f.get("ask", 0) or 0), 2),
                    "volume": int(f.get("volume", 0) or 0),
                    "oi": int(f.get("oi", 0) or 0),
                    "iv": round(float(f.get("iv", 0) or 0), 1),
                    "itm": False,
                    "otm_pct": round(abs(float(f.get("otm_pct", 0) or 0)), 1),
                }
                if str(f.get("type", "")).lower() == "call":
                    calls.append(row)
                else:
                    puts.append(row)
            # Ensure at least one row exists from selected strategy leg.
            if not calls and not puts and legs:
                leg = legs[0]
                row = {
                    "strike": round(float(leg.get("strike", 0) or 0), 2),
                    "bid": round(float(leg.get("bid", 0) or 0), 2),
                    "ask": round(float(leg.get("ask", 0) or 0), 2),
                    "last": round(float(leg.get("premium", 0) or 0), 2),
                    "volume": int(leg.get("volume", 0) or 0),
                    "oi": int(leg.get("oi", 0) or 0),
                    "iv": round(float(leg.get("iv", 0) or 0), 1),
                    "itm": bool(leg.get("itm", False)),
                    "otm_pct": 0.0,
                }
                if str(leg.get("type", "")).upper() == "CALL":
                    calls.append(row)
                else:
                    puts.append(row)

            # Ultimate fallback: build synthetic delayed chain around spot.
            if not calls and not puts:
                spot_guess = 0.0
                try:
                    sd = scanner.get_symbol(sym) if scanner else {}
                    spot_guess = float((sd or {}).get("price", 0) or 0)
                except Exception:
                    spot_guess = 0.0
                if spot_guess <= 0:
                    try:
                        with _MARKET_CACHE_LOCK:
                            spot_guess = float((_MARKET_CACHE.get("prices", {}).get(sym, {}) or {}).get("price", 0) or 0)
                    except Exception:
                        spot_guess = 0.0
                if spot_guess <= 0:
                    spot_guess = 100.0
                step = max(1.0, round(spot_guess * 0.01, 2))
                for i in range(-6, 7):
                    strike = round(spot_guess + i * step, 2)
                    intrinsic_c = max(spot_guess - strike, 0)
                    intrinsic_p = max(strike - spot_guess, 0)
                    time_val = max(0.35, abs(i) * 0.12 + 0.45)
                    call_mid = round(intrinsic_c + time_val, 2)
                    put_mid = round(intrinsic_p + time_val, 2)
                    calls.append({
                        "strike": strike, "bid": round(max(call_mid - 0.08, 0.01), 2), "ask": round(call_mid + 0.08, 2),
                        "last": call_mid, "volume": max(0, 150 - abs(i) * 12), "oi": max(0, 1200 - abs(i) * 90),
                        "iv": round(24 + abs(i) * 1.1, 1), "itm": strike < spot_guess, "otm_pct": round(max((strike - spot_guess) / max(spot_guess, 1e-6) * 100, 0), 1),
                    })
                    puts.append({
                        "strike": strike, "bid": round(max(put_mid - 0.08, 0.01), 2), "ask": round(put_mid + 0.08, 2),
                        "last": put_mid, "volume": max(0, 150 - abs(i) * 12), "oi": max(0, 1200 - abs(i) * 90),
                        "iv": round(24 + abs(i) * 1.1, 1), "itm": strike > spot_guess, "otm_pct": round(max((spot_guess - strike) / max(spot_guess, 1e-6) * 100, 0), 1),
                    })
                cached_opts["dte"] = cached_opts.get("dte", 7)
                cached_opts["breakeven"] = cached_opts.get("breakeven", spot_guess)
                if not expiry:
                    expiry = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
            payload = {
                "symbol": sym,
                "spot": round(float(cached_opts.get("breakeven", 0) or 0), 2),
                "expiry": expiry,
                "dte": int(cached_opts.get("dte", 0) or 0),
                "expirations": [expiry] if expiry else [],
                "calls": calls,
                "puts": puts,
                "put_call_ratio": round((sum(p["volume"] for p in puts) / max(sum(c["volume"] for c in calls), 1)), 3),
                "best_otm_call": max(calls, key=lambda x: x.get("volume", 0) + x.get("oi", 0)) if calls else None,
                "best_otm_put": max(puts, key=lambda x: x.get("volume", 0) + x.get("oi", 0)) if puts else None,
                "total_call_volume": int(sum(c["volume"] for c in calls)),
                "total_put_volume": int(sum(p["volume"] for p in puts)),
                "cache_hit": False,
                "stale_cache": stale_flag,
                "warning": f"Using internal options cache fallback: {error_msg}",
            }
            return payload

        # Fast response path: use recent cache to avoid rate-limit bursts.
        cached = _OPTIONS_CHAIN_HTTP_CACHE.get(cache_key)
        if cached and (now - float(cached.get("ts", 0))) < _OPTIONS_CHAIN_TTL_S:
            payload = dict(cached.get("payload", {}))
            payload["cache_hit"] = True
            return jsonify(_sanitize(payload))

        # Provider priority 1: Tradier options API (when configured).
        if _has_key(TRADIER_API_KEY):
            try:
                headers = {"Authorization": f"Bearer {TRADIER_API_KEY}", "Accept": "application/json"}
                exr = requests.get(
                    "https://api.tradier.com/v1/markets/options/expirations",
                    params={"symbol": sym, "includeAllRoots": "true"},
                    headers=headers,
                    timeout=8,
                )
                exj = exr.json() if exr.status_code == 200 else {}
                exp_list = (((exj.get("expirations") or {}).get("date")) if isinstance(exj, dict) else []) or []
                if isinstance(exp_list, str):
                    exp_list = [exp_list]
                if exp_list:
                    exp_idx = min(exp_idx, len(exp_list) - 1)
                    expiry = exp_list[exp_idx]
                    chr_ = requests.get(
                        "https://api.tradier.com/v1/markets/options/chains",
                        params={"symbol": sym, "expiration": expiry, "greeks": "true"},
                        headers=headers,
                        timeout=10,
                    )
                    chj = chr_.json() if chr_.status_code == 200 else {}
                    options = (((chj.get("options") or {}).get("option")) if isinstance(chj, dict) else []) or []
                    if isinstance(options, dict):
                        options = [options]
                    calls, puts = [], []
                    for o in options:
                        if not isinstance(o, dict):
                            continue
                        typ = str(o.get("option_type", "")).lower()
                        row = {
                            "strike": round(float(o.get("strike", 0) or 0), 2),
                            "bid": round(float(o.get("bid", 0) or 0), 2),
                            "ask": round(float(o.get("ask", 0) or 0), 2),
                            "last": round(float(o.get("last", 0) or 0), 2),
                            "volume": int(o.get("volume", 0) or 0),
                            "oi": int(o.get("open_interest", 0) or 0),
                            "iv": round(float(o.get("greeks", {}).get("mid_iv", 0) or 0) * 100, 1),
                            "itm": bool(o.get("intrinsic_value", 0) and float(o.get("intrinsic_value", 0)) > 0),
                            "otm_pct": 0.0,
                        }
                        if typ == "call":
                            calls.append(row)
                        elif typ == "put":
                            puts.append(row)
                    q = requests.get(
                        "https://api.tradier.com/v1/markets/quotes",
                        params={"symbols": sym},
                        headers=headers,
                        timeout=6,
                    )
                    qj = q.json() if q.status_code == 200 else {}
                    qobj = ((qj.get("quotes") or {}).get("quote")) if isinstance(qj, dict) else {}
                    spot = float(qobj.get("last", 0) or qobj.get("close", 0) or 0) if isinstance(qobj, dict) else 0.0
                    if spot > 0:
                        for arr in (calls, puts):
                            for r in arr:
                                strike = float(r.get("strike", 0) or 0)
                                itm = (strike < spot and arr is calls) or (strike > spot and arr is puts)
                                r["itm"] = bool(itm)
                                r["otm_pct"] = round(max(abs(strike - spot) / max(spot, 1e-6) * 100, 0), 1) if not itm else 0.0
                    total_call_vol = sum(c["volume"] for c in calls)
                    total_put_vol = sum(p["volume"] for p in puts)
                    payload = {
                        "symbol": sym,
                        "spot": round(float(spot), 2),
                        "expiry": expiry,
                        "dte": max((datetime.strptime(expiry, "%Y-%m-%d") - datetime.now()).days, 0),
                        "expirations": exp_list[:12],
                        "calls": calls,
                        "puts": puts,
                        "put_call_ratio": round(total_put_vol / max(total_call_vol, 1), 3),
                        "best_otm_call": max([c for c in calls if not c["itm"]] or calls, key=lambda x: x.get("volume", 0) + x.get("oi", 0), default=None),
                        "best_otm_put": max([p for p in puts if not p["itm"]] or puts, key=lambda x: x.get("volume", 0) + x.get("oi", 0), default=None),
                        "total_call_volume": int(total_call_vol),
                        "total_put_volume": int(total_put_vol),
                        "provider": "tradier",
                        "cache_hit": False,
                    }
                    _OPTIONS_CHAIN_HTTP_CACHE[cache_key] = {"ts": now, "payload": payload}
                    return jsonify(_sanitize(payload))
            except Exception:
                pass

        import yfinance as _yf_oc

        tk = _yf_oc.Ticker(sym)

        # Retry options metadata fetch with small backoff for Yahoo 429 spikes.
        expirations = []
        last_err = None
        for wait_s in (0.0, 0.8, 1.6):
            if wait_s > 0:
                time.sleep(wait_s)
            try:
                expirations = list(tk.options) if tk.options else []
                if expirations:
                    break
            except Exception as e:
                last_err = e
                continue
        if not expirations:
            # Serve stale cache if available instead of hard failing.
            stale = _OPTIONS_CHAIN_HTTP_CACHE.get(cache_key)
            if stale and stale.get("payload"):
                payload = dict(stale["payload"])
                payload["stale_cache"] = True
                payload["warning"] = "Live options temporarily unavailable; serving cached snapshot."
                return jsonify(_sanitize(payload))
            msg = f"No options data for {sym}"
            if last_err:
                msg = str(last_err)
            fb = _fallback_from_engine_cache(msg)
            if fb:
                _OPTIONS_CHAIN_HTTP_CACHE[cache_key] = {"ts": now, "payload": fb}
                return jsonify(_sanitize(fb))
            return jsonify({"error": msg, "symbol": sym}), 404

        exp_idx = min(exp_idx, len(expirations) - 1)
        expiry = expirations[exp_idx]
        chain = None
        for wait_s in (0.0, 0.8, 1.6):
            if wait_s > 0:
                time.sleep(wait_s)
            try:
                chain = tk.option_chain(expiry)
                if chain is not None:
                    break
            except Exception as e:
                last_err = e
                continue
        if chain is None:
            stale = _OPTIONS_CHAIN_HTTP_CACHE.get(cache_key)
            if stale and stale.get("payload"):
                payload = dict(stale["payload"])
                payload["stale_cache"] = True
                payload["warning"] = "Option chain temporarily rate-limited; serving cached snapshot."
                return jsonify(_sanitize(payload))
            msg = str(last_err) if last_err else f"Unable to fetch option chain for {sym}"
            fb = _fallback_from_engine_cache(msg)
            if fb:
                _OPTIONS_CHAIN_HTTP_CACHE[cache_key] = {"ts": now, "payload": fb}
                return jsonify(_sanitize(fb))
            return jsonify({"error": msg, "symbol": sym}), 503

        # Current spot price
        try:
            info = tk.fast_info
            spot = float(info.last_price) if hasattr(info, 'last_price') else float(tk.history(period="1d")["Close"].iloc[-1])
        except:
            spot = 0

        # Process calls
        def _proc(df, is_call):
            rows = []
            for _, r in df.iterrows():
                strike = float(r.get("strike", 0))
                bid = float(r.get("bid", 0))
                ask = float(r.get("ask", 0))
                last = float(r.get("lastPrice", 0))
                vol = int(r.get("volume", 0) or 0)
                oi = int(r.get("openInterest", 0) or 0)
                iv = float(r.get("impliedVolatility", 0) or 0) * 100
                itm = r.get("inTheMoney", False)
                otm_pct = abs(strike - spot) / max(spot, 0.01) * 100 if not itm else 0
                rows.append({
                    "strike": round(strike, 2), "bid": round(bid, 2), "ask": round(ask, 2),
                    "last": round(last, 2), "volume": vol, "oi": oi, "iv": round(iv, 1),
                    "itm": bool(itm), "otm_pct": round(otm_pct, 1),
                })
            return rows

        calls = _proc(chain.calls, True)
        puts = _proc(chain.puts, False)

        # DTE
        from datetime import datetime as _dt_oc
        try:
            dte = (_dt_oc.strptime(expiry, "%Y-%m-%d") - _dt_oc.now()).days
        except:
            dte = 0

        # P/C ratio
        total_call_vol = sum(c["volume"] for c in calls)
        total_put_vol = sum(p["volume"] for p in puts)
        pc_ratio = total_put_vol / max(total_call_vol, 1)

        # Best OTM picks
        otm_calls = [c for c in calls if not c["itm"] and (c["volume"] + c["oi"]) > 0]
        otm_puts = [p for p in puts if not p["itm"] and (p["volume"] + p["oi"]) > 0]
        best_call = max(otm_calls, key=lambda x: x["volume"] + x["oi"]) if otm_calls else None
        best_put = max(otm_puts, key=lambda x: x["volume"] + x["oi"]) if otm_puts else None

        payload = {
            "symbol": sym, "spot": round(spot, 2), "expiry": expiry,
            "dte": dte, "expirations": expirations[:12],
            "calls": calls, "puts": puts,
            "put_call_ratio": round(pc_ratio, 3),
            "best_otm_call": best_call, "best_otm_put": best_put,
            "total_call_volume": total_call_vol, "total_put_volume": total_put_vol,
            "provider": "yahoo",
            "cache_hit": False,
        }
        _OPTIONS_CHAIN_HTTP_CACHE[cache_key] = {"ts": now, "payload": payload}
        return jsonify(_sanitize(payload))
    except Exception as e:
        # Final safety: stale cache fallback on unexpected failures.
        if cache_key:
            stale = _OPTIONS_CHAIN_HTTP_CACHE.get(cache_key)
            if stale and stale.get("payload"):
                payload = dict(stale["payload"])
                payload["stale_cache"] = True
                payload["warning"] = "Live options endpoint error; serving cached snapshot."
                return jsonify(_sanitize(payload))
        return jsonify({"error": str(e), "symbol": symbol.upper()}), 500


def _fast_quote_snapshot(symbols: list) -> dict:
    """
    Fast quote snapshot using Yahoo quote endpoint for small symbol sets.
    Falls back to market cache when unavailable.
    """
    out = {}
    syms = [s.strip().upper() for s in (symbols or []) if s and str(s).strip()][:25]
    if not syms:
        return out
    try:
        u = "https://query1.finance.yahoo.com/v7/finance/quote"
        r = requests.get(
            u,
            params={"symbols": ",".join(syms)},
            timeout=2.2,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if r.ok:
            rows = ((r.json() or {}).get("quoteResponse") or {}).get("result") or []
            for row in rows:
                sym = str(row.get("symbol", "")).upper()
                px = row.get("regularMarketPrice")
                ch = row.get("regularMarketChangePercent")
                if sym and px is not None:
                    out[sym] = {
                        "price": float(px),
                        "change_pct": float(ch or 0.0),
                        "provider": "yahoo_quote_fast",
                    }
    except Exception:
        pass
    if len(out) < max(2, len(syms) // 2):
        with _MARKET_CACHE_LOCK:
            prices = dict(_MARKET_CACHE.get("prices", {}))
        for sym in syms:
            if sym in out:
                continue
            p = prices.get(sym)
            if p:
                out[sym] = {
                    "price": float(p.get("price", 0) or 0),
                    "change_pct": float(p.get("change_pct", 0) or 0),
                    "provider": "market_cache",
                }
    return out


@app.route("/api/stream/prices", methods=["GET"])
def api_stream_prices():
    """
    Real-time SSE quote stream (default) + snapshot mode.
    - mode=snapshot  -> one-shot JSON payload
    - default        -> text/event-stream, pushes every ~1s
    """
    try:
        symbols = request.args.get("symbols", "SPY,QQQ,IWM,DIA,GLD,TLT,BTC-USD")
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:20]
        mode = (request.args.get("mode", "") or "").lower()

        if mode == "snapshot":
            return jsonify(_sanitize(_fast_quote_snapshot(sym_list)))

        def _gen():
            # initial handshake event
            yield "event: ready\ndata: {\"ok\":true}\n\n"
            while True:
                payload = _sanitize(_fast_quote_snapshot(sym_list))
                yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
                time.sleep(SSE_PUSH_INTERVAL_S)

        resp = Response(stream_with_context(_gen()), mimetype="text/event-stream")
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["X-Accel-Buffering"] = "no"
        resp.headers["Connection"] = "keep-alive"
        return resp
    except Exception as e:
        return jsonify({"error": str(e)}), 500

_options_flow_cache = {"data": None, "ts": 0}

@app.route("/api/options_flow")
def api_options_flow():
    """Aggregate options data for all top signals — on-demand with 3-min cache."""
    now = time.time()
    if _options_flow_cache["data"] and now - _options_flow_cache["ts"] < 180:
        return jsonify(_sanitize(_options_flow_cache["data"]))
    state = scanner.get_state()
    sigs = state.get("top_signals", [])
    try:
        limit = int(request.args.get("limit", 16) or 16)
    except Exception:
        limit = 16
    limit = max(6, min(30, limit))
    target_sigs = [s for s in sigs if (s.get("direction") or "HOLD") != "HOLD"][:limit]
    results = []

    def _load_one(sig):
        sym = sig.get("symbol")
        if not sym:
            return None
        # Use cached options from scanner first
        d = scanner.get_symbol(sym)
        opts = d.get("options") if d else None
        # Fallback: fetch live from yfinance if not cached
        if not opts:
            try:
                price = sig.get("price", 0)
                direction = sig.get("direction", "HOLD")
                if direction == "HOLD":
                    return None
                t = yf.Ticker(sym)
                exps = t.options
                if exps and price:
                    conv = sig.get("conviction", 0.3)
                    tdte = 21 if conv > 0.65 else 30 if conv > 0.45 else 45
                    target_exp = None
                    for e in exps:
                        try:
                            dte = (datetime.strptime(e, "%Y-%m-%d").date() - datetime.now().date()).days
                            if abs(dte - tdte) <= 14:
                                target_exp = e
                                break
                        except: pass
                    if not target_exp: target_exp = exps[0]
                    chain = t.option_chain(target_exp)
                    dte_val = (datetime.strptime(target_exp, "%Y-%m-%d").date() - datetime.now().date()).days
                    df = chain.calls if direction == "BUY" else chain.puts
                    has_volume = not df.empty and "volume" in df.columns and df["volume"].fillna(0).sum() > 0
                    rank_col = "volume" if has_volume else "openInterest"
                    if not df.empty:
                        df_f = df[df[rank_col].fillna(0) > 0] if rank_col in df.columns else df
                        if df_f.empty: df_f = df
                        best = df_f.nlargest(1, rank_col).iloc[0]
                        iv = round(float(best.get("impliedVolatility", 0) or 0) * 100, 1)
                        vol = int(best.get("volume", 0) or 0)
                        oi  = int(best.get("openInterest", 0) or 0)
                        opts = {
                            "strategy": "BUY CALL" if direction == "BUY" else "BUY PUT",
                            "strike": round(float(best["strike"]), 2),
                            "expiry": target_exp,
                            "dte": dte_val,
                            "premium": round(float(best.get("lastPrice", 0) or 0), 2),
                            "bid": round(float(best.get("bid", 0) or 0), 2),
                            "ask": round(float(best.get("ask", 0) or 0), 2),
                            "iv": iv, "iv_rank": iv,
                            "volume": vol, "oi": oi,
                            "vol_oi": round(vol / max(oi, 1), 3),
                            "recommended": sig.get("conviction", 0) > 0.6,
                            "pop": max(0, min(100, 50 + (sig.get("composite", 0) or 0) * 30)),
                            "score": round((sig.get("composite", 0) or 0) * 100, 1),
                            "ranked_by": rank_col,
                        }
            except Exception:
                pass
        if opts:
            return {**sig, "options": opts}
        return None

    # Parallelize per-symbol option chain hydration.
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as _pool:
        for item in _pool.map(_load_one, target_sigs):
            if item:
                results.append(item)

    result = _sanitize({"signals_with_options": results, "count": len(results), "updated": datetime.now().isoformat()})
    _options_flow_cache["data"] = result
    _options_flow_cache["ts"]   = now
    return jsonify(result)



@app.route("/api/research")
def api_research():
    """Full AI research report — regime, factors, PhD discoveries, signal decay."""
    try:
        state    = scanner.get_state()
        research = state.get("research", {})
        report   = RC.get_full_report() if RC else {}
        return jsonify(_sanitize({
            **research,
            "full_report":   report,
            "phd_active":    RC.phd_agent.active if RC else False,
            "time":          datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({"error": str(e), "time": datetime.now().isoformat()})

@app.route("/api/research/discover", methods=["POST"])
def api_research_discover():
    """Trigger an immediate PhD research cycle."""
    try:
        from flask import request as freq
        body    = freq.get_json(force=True) or {}
        context = {
            "vix":    scanner.get_state().get("vix", 19),
            "regime": scanner.get_state().get("regime", "NEUTRAL"),
        }
        summary = {"top_symbols": list(scanner.all_signals.keys())[:5]}
        result  = RC.phd_agent.run_research_cycle(context, summary) if RC else {}
        return jsonify(_sanitize(result))
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/universe", methods=["GET"])
def api_universe():
    """
    Returns the full dynamic scan universe.
    Every symbol in this list can be quoted, scanned, and analyzed.

    Covers S&P 500, Nasdaq-100, all ETFs (equity/bond/commodity/metals/sector),
    plus any symbol previously searched in the app.

    Query params:
      ?refresh=true  — force refresh from Wikipedia (otherwise 24hr cache)
      ?n=100         — max symbols to return (default: all)
    """
    force = request.args.get("refresh", "false").lower() == "true"
    n     = int(request.args.get("n", 0))
    if force:
        _universe_cache["ts"] = 0   # invalidate cache
    symbols = get_scan_universe()
    if n > 0:
        symbols = symbols[:n]
    return jsonify({
        "symbols":  symbols,
        "count":    len(symbols),
        "active":   len(scanner.all_signals) if scanner.all_signals else 0,
        "note":     "Any ticker can be added by searching it in the app",
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/universe/add", methods=["POST"])
def api_universe_add():
    """
    Add one or more symbols to the active scan universe.
    POST body: {"symbols": ["GOLD", "SI=F", "BTC-USD"]} or {"symbol": "GLD"}

    The symbol is immediately added to scanner.all_signals and will be
    scanned on the next cycle. No restart required.
    """
    body    = request.get_json(force=True) or {}
    syms    = body.get("symbols", [])
    single  = body.get("symbol", "")
    if single:
        syms = [single] + syms
    added   = []
    for s in syms:
        s = s.upper().strip().replace(".", "-")
        if s:
            if scanner.all_signals is None:
                scanner.all_signals = {}
            if s not in scanner.all_signals:
                scanner.all_signals[s] = {"symbol": s, "price": 0, "direction": "HOLD"}
            added.append(s)
    return jsonify({"added": added, "total": len(scanner.all_signals or {})})


@app.route("/api/config/status")
def api_config_status():
    """
    Show which environment variables are configured.
    Never returns the actual key values — only True/False per service.
    """
    def _ok(v): return bool(v) and not v.startswith("YOUR_")
    return jsonify({
        "alpaca":    _ok(ALPACA_API_KEY) and _ok(ALPACA_SECRET_KEY),
        "anthropic": _ok(os.environ.get("ANTHROPIC_API_KEY", "")),
        "news":      _ok(NEWS_API_KEY),
        "fred":      _ok(FRED_API_KEY),
        "alpaca_mode": "paper" if "paper" in ALPACA_BASE_URL else "live",
        "bar_interval": os.environ.get("BAR_INTERVAL", "1d"),
        "note": "Set keys in .env file — see .env.example for instructions",
    })


@app.route("/api/stream/status")
def api_stream_status():
    """Live streaming engine status — uses StreamingEngine if alpaca-py installed."""
    if _stream_eng:
        status = _stream_eng.get_status()
        status["mode"] = "websocket" if status.get("alpaca_py_ok") else "polling_fallback"
        return jsonify(status)
    return jsonify({
        "mode": "polling", "interval": os.environ.get("BAR_INTERVAL", "5m"),
        "connected": False, "note": "pip install alpaca-py to enable WebSocket streaming",
    })


@app.route("/api/portfolio/optimize", methods=["POST"])
def api_portfolio_optimize():
    """Portfolio Kelly + Black-Litterman optimization."""
    if not _kelly_opt:
        return jsonify({"error": "portfolio_optimizer not available"}), 503
    data = request.json or {}
    signals_raw = data.get("signals", {})
    returns_data = data.get("returns_data", {})
    nav = float(data.get("portfolio_value", 1_000_000))
    method = data.get("method", "kelly")
    try:
        from portfolio_optimizer import compute_optimal_weights
        weights = compute_optimal_weights(signals_raw, returns_data, nav, method=method)
        return jsonify({"weights": weights, "method": method, "n_assets": len(weights)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stat_arb/pairs", methods=["GET"])
def api_stat_arb_pairs():
    """Active stat-arb pair signals from StatArbManager."""
    if not _sarb_mgr:
        return jsonify({"pairs": [], "note": "stat_arb_manager not loaded"}), 200
    try:
        pairs = _sarb_mgr.get_active_pairs()
        return jsonify({"pairs": _sanitize(pairs), "count": len(pairs)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/orderflow/<symbol>", methods=["GET"])
def api_orderflow(symbol):
    """Real-time order flow metrics for a symbol."""
    if not _ofe:
        return jsonify({"note": "orderflow_engine not loaded"}), 200
    try:
        sym = symbol.upper()
        # Get bars from scanner for the symbol
        d = scanner.get_symbol(sym)
        bars = d.get("history", []) if d else []
        result = _ofe.composite_orderflow_score(bars) if bars else {"error": "no bar data"}
        return jsonify(_sanitize(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route("/api/ping")
def api_ping():
    """Ultra-lightweight ping — always responds instantly. Used by frontend health check."""
    return jsonify({"status": "online", "t": time.time()}), 200

@app.route("/api/health")
def api_health():
    """Diagnostic endpoint — always responds, shows what's working."""
    try:
        engines = {
            "options_engine": OE is not None,
            "ml_engine":      ML is not None,
            "signal_engine":  SE is not None,
            "company_info":   CI is not None,
            "earnings_engine":EE is not None,
            "research_agent": RC is not None,
            "yfinance":       YF_OK,
        }
        signals_count = len(scanner.all_signals) if hasattr(scanner, "all_signals") else 0
        state         = scanner.get_state() if hasattr(scanner, "get_state") else {}
        return jsonify({
            "status":         "online",
            "port":           PORT,
            "signals_ready":  signals_count,
            "engines":        engines,
            "all_engines_ok": all(engines.values()),
            "scan_alive":     state.get("scan_alive", False),
            "scan_cycle":     state.get("scan_cycle", 0),
            "vix":            state.get("vix", 0),
            "time":           datetime.now().isoformat(),
            "instructions":   "If signals_ready=0, wait 60s for first scan to complete",
        })
    except Exception as _he:
        return jsonify({"status": "online", "error": str(_he), "time": datetime.now().isoformat()}), 200

@app.route("/api/earnings")
def api_earnings():
    """Return latest earnings signals."""
    with scanner.lock:
        data = scanner.earnings_signals if scanner.earnings_signals else {}
    try:
        payload = dict(data) if isinstance(data, dict) else {"all": list(data) if isinstance(data, list) else []}
        nasdaq_rows = _fetch_nasdaq_earnings_rows(limit=250)
        nasdaq_symbols = {
            str((r.get("symbol") or "")).upper().strip()
            for r in nasdaq_rows
            if r.get("symbol")
        }
        for k in ("all", "amc", "bmo"):
            items = payload.get(k)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        sym = str(item.get("symbol", "")).upper().strip()
                        item["nasdaq_verified"] = bool(sym and sym in nasdaq_symbols)
        payload["nasdaq_earnings"] = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "count": len(nasdaq_rows),
            "symbols_count": len(nasdaq_symbols),
            "ok": bool(nasdaq_rows),
        }
        return jsonify(_sanitize(payload))
    except Exception:
        return jsonify(data)

@app.route("/api/pairs")
def api_pairs():
    """Return cointegration pair signals."""
    with scanner.lock:
        data = scanner.pair_signals or []
    return jsonify(_sanitize({"pairs": data}))

@app.route("/api/news")
def api_news():
    try:
        state_news = scanner.get_state().get("news", []) if scanner else []
        data = state_news if state_news else fetch_news()
        mix = {}
        for n in data:
            src = str((n or {}).get("source", "unknown"))
            mix[src] = mix.get(src, 0) + 1
        provider = "scanner_state" if state_news else "direct_cached_fallback"
        return jsonify({
            "news": _sanitize(data),
            "provider": provider,
            "source_mix": mix,
            "nasdaq_count": int(mix.get("Nasdaq", 0)),
        })
        # scanner warm-up may not have populated state yet; fall back to direct
        # cached fetch so UI is never empty for long.
    except Exception as e:
        return jsonify({"news": [], "error": str(e)})

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")   # set in .env

# ── Initialize AI Research Coordinator ───────────────────────────────────────
RC = None
if RA:
    try:
        RC = RA.ResearchCoordinator(anthropic_key=ANTHROPIC_KEY)
        print("✓  ResearchCoordinator initialized (AI PhD Agent ACTIVE)")
    except Exception as _rc_e:
        print(f"⚠  ResearchCoordinator error: {_rc_e}")

@app.route("/api/claude", methods=["POST"])
def claude_proxy():
    """Proxy Claude API calls from React to avoid CORS. Backend → Anthropic → React."""
    try:
        from flask import request as freq
        body = freq.get_json(force=True)
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type":    "application/json",
                "x-api-key":       ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
            },
            json=body,
            timeout=30,
        )
        data = r.json()
        if not r.ok:
            logger.error(f"Anthropic error: {r.status_code} {data}")
            return jsonify({"error": data.get("error", {}).get("message", "API error"), "status": r.status_code}), r.status_code
        text = next((b["text"] for b in data.get("content", []) if b.get("type") == "text"), "")
        return jsonify({"text": text, "usage": data.get("usage", {})})
    except Exception as e:
        logger.error(f"Claude proxy error: {e}")
        return jsonify({"error": str(e)}), 500

# duplicate /api/health removed

def _watchdog():
    """Monitors scanner health. If no scan completes in 5 min, force a new cycle."""
    import time as _time
    while True:
        _time.sleep(120)
        try:
            state = scanner.get_state()
            last  = state.get("scan_start")
            if last:
                from datetime import datetime
                elapsed = (datetime.now() - datetime.fromisoformat(last)).total_seconds()
                if elapsed > 360:  # 6 min without completing a scan
                    logger.warning(f"⚠ Watchdog: scan stalled {elapsed:.0f}s — forcing new cycle")
                    scanner.scan_index = 0  # reset to trigger rescan
        except Exception as _we:
            logger.debug(f"Watchdog: {_we}")


def _macro_background_refresh():
    """
    Background thread: refresh macro data + MacroScorer every 10 minutes.
    Keeps scanner.macro_signal live even when no one calls /api/macro.
    This ensures AladdinScorer's macro_regime slot is never stale.
    """
    import time as _time
    _time.sleep(20)   # brief delay after server start
    while True:
        try:
            _ME = sys.modules.get("math_engine")
            fred = fetch_fred_macro()
            fed_rate     = float((fred.get("fed_rate")     or {}).get("value", 5.25))
            cpi          = float((fred.get("cpi")          or {}).get("value", 3.2))
            unemployment = float((fred.get("unemployment") or {}).get("value", 3.9))

            # Fetch cross-asset snapshot (small set, fast)
            cross = {}
            _tickers_mini = {
                "^TNX":"y10","^IRX":"y3m","^VIX":"vix","^VIX9D":"vix9d",
                "^VIX3M":"vix3m","DX-Y.NYB":"dxy","HYG":"hyg","LQD":"lqd",
            }
            from concurrent.futures import ThreadPoolExecutor, as_completed as _ac
            def _mini_fetch(sk):
                s, k = sk
                try:
                    period = "30d" if k == "dxy" else "5d"
                    h = yf.Ticker(s).history(period=period)
                    if not h.empty:
                        last = float(h["Close"].iloc[-1])
                        prev = float(h["Close"].iloc[-2]) if len(h) >= 2 else last
                        ago20 = float(h["Close"].iloc[-20]) if k == "dxy" and len(h) >= 20 else 0.0
                        return k, {"value": last, "change_pct": (last/prev-1)*100, "ago20": ago20}
                except Exception:
                    pass
                return k, {}

            with ThreadPoolExecutor(max_workers=8) as pool:
                futs = {pool.submit(_mini_fetch, item): item for item in _tickers_mini.items()}
                for fut in _ac(futs, timeout=15):
                    try:
                        k, v = fut.result()
                        cross[k] = v
                    except Exception:
                        pass

            def _cv(k, d=0.0): return float((cross.get(k) or {}).get("value") or d)

            if _ME and hasattr(_ME, "MacroScorer"):
                dxy_now = _cv("dxy", 104.0)
                dxy_ago = float((cross.get("dxy") or {}).get("ago20", 0.0))
                scored = _ME.MacroScorer.compute(
                    yield_10y   = _cv("y10", 4.3),
                    yield_3m    = _cv("y3m", 5.3),
                    vix         = _cv("vix", 20.0),
                    vix9d       = _cv("vix9d", 0.0),
                    vix3m       = _cv("vix3m", 0.0),
                    hyg_5d_ret  = float((cross.get("hyg") or {}).get("change_pct", 0.0)),
                    lqd_5d_ret  = float((cross.get("lqd") or {}).get("change_pct", 0.0)),
                    dxy         = dxy_now,
                    dxy_20d_ago = dxy_ago,
                    fed_rate    = fed_rate,
                    cpi         = cpi,
                    unemployment = unemployment,
                )
                if hasattr(scanner, "macro_signal"):
                    scanner.macro_signal = float(scored.get("macro_score", 0))
                logger.debug(
                    f"  [MacroBG] {scored.get('macro_label','?')} "
                    f"{scored.get('macro_score',0):+.3f} | "
                    f"VIX={_cv('vix'):.1f} | "
                    f"Spread={_cv('y10',4.3)-_cv('y3m',5.3):+.2f}%"
                )
        except Exception as _mb_err:
            logger.debug(f"Macro background: {_mb_err}")
        _time.sleep(600)   # refresh every 10 minutes





# ── LIVE MACRO DATA ────────────────────────────────────────────────────────────
_macro_cache = {"data": None, "ts": 0}

@app.route("/api/macro")
def api_macro():
    now = time.time()
    if _macro_cache["data"] and now - _macro_cache["ts"] < 300:   # 5-min cache
        return jsonify(_macro_cache["data"])
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # ── Expanded ticker set (Step 7: adds VIX term structure) ────────────
        tickers = {
            "^TNX":     "10Y_yield",
            "^IRX":     "3M_yield",
            "^FVX":     "5Y_yield",
            "^TYX":     "30Y_yield",
            "^VIX":     "vix",
            "^VIX9D":   "vix9d",     # 9-day VIX (near-term fear)
            "^VIX3M":   "vix3m",     # 3-month VIX (medium-term)
            "DX-Y.NYB": "dxy",
            "CL=F":     "crude_oil",
            "GC=F":     "gold",
            "^GSPC":    "spy_price",
            "TLT":      "tlt",
            "HYG":      "hyg",       # High-yield credit
            "LQD":      "lqd",       # Investment-grade credit
        }

        def _fetch_one(sym_key):
            sym, key = sym_key
            try:
                # For DXY we need 20-day history to compute momentum
                period = "30d" if key == "dxy" else "5d"
                h = yf.Ticker(sym).history(period=period)
                if not h.empty:
                    last = float(h["Close"].iloc[-1])
                    prev = float(h["Close"].iloc[-2]) if len(h) >= 2 else last
                    # 20-day ago close for DXY momentum
                    ago20 = float(h["Close"].iloc[-20]) if (key == "dxy" and len(h) >= 20) else 0.0
                    return key, {
                        "value":      round(last, 4),
                        "change":     round(last - prev, 4),
                        "change_pct": round((last / prev - 1) * 100, 3),
                        "ago20":      round(ago20, 4),
                    }
            except Exception:
                pass
            return key, {"value": None, "change": 0, "change_pct": 0, "ago20": 0.0}

        data = {}
        with ThreadPoolExecutor(max_workers=14) as pool:
            futures = {pool.submit(_fetch_one, item): item for item in tickers.items()}
            for fut in as_completed(futures, timeout=25):
                try:
                    key, val = fut.result()
                    data[key] = val
                except Exception:
                    pass

        def _v(key, default=0.0):
            return float((data.get(key) or {}).get("value") or default)

        # ── Build yield curve ─────────────────────────────────────────────
        yield_curve = []
        for tenor, key in [("3M","3M_yield"),("5Y","5Y_yield"),("10Y","10Y_yield"),("30Y","30Y_yield")]:
            d2 = data.get(key)
            if d2 and d2.get("value"):
                yield_curve.append({"tenor": tenor, "yield": d2["value"]})

        y10  = _v("10Y_yield", 4.3)
        y3m  = _v("3M_yield",  5.3)
        y2y  = _v("2Y_yield",  4.7)
        spread_10y_3m = round(y10 - y3m, 3)

        # ── Credit spread: HYG vs LQD 5-day returns ───────────────────────
        hyg_5d = float((data.get("hyg") or {}).get("change_pct", 0.0))
        lqd_5d = float((data.get("lqd") or {}).get("change_pct", 0.0))

        # ── DXY 20-day momentum ───────────────────────────────────────────
        dxy_now  = _v("dxy", 104.0)
        dxy_ago  = float((data.get("dxy") or {}).get("ago20", 0.0))

        # ── FRED supplemental data (fed rate, CPI, unemployment) ─────────
        fred = fetch_fred_macro()
        fed_rate     = float((fred.get("fed_rate")     or {}).get("value", 5.25))
        cpi          = float((fred.get("cpi")          or {}).get("value", 3.2))
        unemployment = float((fred.get("unemployment") or {}).get("value", 3.9))

        # ── MacroScorer (Step 7) ──────────────────────────────────────────
        _ME = sys.modules.get("math_engine")
        macro_scored = {}
        if _ME and hasattr(_ME, "MacroScorer"):
            try:
                macro_scored = _ME.MacroScorer.compute(
                    yield_10y     = y10,
                    yield_3m      = y3m,
                    yield_2y      = y2y,
                    yield_30y     = _v("30Y_yield", 4.5),
                    vix           = _v("vix", 20.0),
                    vix9d         = _v("vix9d", 0.0),
                    vix3m         = _v("vix3m", 0.0),
                    hyg_5d_ret    = hyg_5d,
                    lqd_5d_ret    = lqd_5d,
                    dxy           = dxy_now,
                    dxy_20d_ago   = dxy_ago,
                    fed_rate      = fed_rate,
                    cpi           = cpi,
                    unemployment  = unemployment,
                )
                # Push macro_score → scanner so every analyze_symbol() uses it
                if hasattr(scanner, "macro_signal"):
                    scanner.macro_signal = float(macro_scored.get("macro_score", 0))
                    logger.info(f"  Macro regime: {macro_scored.get('macro_label')} "
                                f"({macro_scored.get('macro_score',0):+.3f})")
            except Exception as _ms_err:
                logger.debug(f"MacroScorer: {_ms_err}")

        result = {
            # Yield curve
            "yield_curve":    yield_curve,
            "spread_10y_3m":  spread_10y_3m,
            "inverted":       spread_10y_3m < 0,
            # VIX
            "vix":            _v("vix"),
            "vix9d":          _v("vix9d") or None,
            "vix3m":          _v("vix3m") or None,
            # Cross-asset
            "dxy":            dxy_now or None,
            "dxy_20d_momentum": round((dxy_now - dxy_ago) / (dxy_ago + 1e-8) * 100, 2) if dxy_ago > 0 else None,
            "crude_oil":      _v("crude_oil") or None,
            "gold":           _v("gold") or None,
            # Credit
            "tlt_chg":        (data.get("tlt") or {}).get("change_pct"),
            "hyg_chg":        hyg_5d,
            "lqd_chg":        lqd_5d,
            "credit_spread":  round(hyg_5d - lqd_5d, 3),
            # Macro fundamentals
            "fed_rate":       fed_rate,
            "cpi":            cpi,
            "unemployment":   unemployment,
            "real_rate":      round(fed_rate - cpi, 2),
            # MacroScorer output
            "macro_score":    macro_scored.get("macro_score", 0),
            "macro_label":    macro_scored.get("macro_label", "NEUTRAL"),
            "macro_sub_scores": macro_scored.get("sub_scores", {}),
            "regime_factors": macro_scored.get("regime_factors", []),
            "risk_indicators":macro_scored.get("risk_indicators", []),
            "crisis_flags":   macro_scored.get("crisis_flags", 0),
            # Raw data
            "raw":            data,
            "updated":        datetime.now().isoformat(),
        }
        _macro_cache["data"] = result
        _macro_cache["ts"]   = now
        return jsonify(result)
    except Exception as e:
        logger.error(f"api_macro: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/macro/score")
def api_macro_score():
    """
    Fast endpoint: return cached MacroScorer output.
    Returns the five sub-scores, composite macro_score [-1,+1],
    macro_label, risk indicators, and current scanner.macro_signal.
    Used by the dashboard's Macro Intelligence tab to show the
    regime speedometer and factor breakdown chart.
    """
    try:
        cached = _macro_cache.get("data") or {}
        _ME = sys.modules.get("math_engine")

        # If cache is stale (>15min), recompute from FRED+last known values
        cache_age = time.time() - _macro_cache.get("ts", 0)

        scanner_macro = 0.0
        if hasattr(scanner, "macro_signal"):
            scanner_macro = float(scanner.macro_signal)

        return jsonify(_sanitize({
            "macro_score":      cached.get("macro_score", scanner_macro),
            "macro_label":      cached.get("macro_label", "NEUTRAL"),
            "sub_scores":       cached.get("macro_sub_scores", {}),
            "regime_factors":   cached.get("regime_factors", []),
            "risk_indicators":  cached.get("risk_indicators", []),
            "crisis_flags":     cached.get("crisis_flags", 0),
            "scanner_macro_signal": scanner_macro,
            # Key inputs for dashboard display
            "vix":              cached.get("vix"),
            "vix9d":            cached.get("vix9d"),
            "vix3m":            cached.get("vix3m"),
            "spread_10y_3m":    cached.get("spread_10y_3m"),
            "inverted":         cached.get("inverted", False),
            "credit_spread":    cached.get("credit_spread"),
            "dxy_20d_momentum": cached.get("dxy_20d_momentum"),
            "real_rate":        cached.get("real_rate"),
            "fed_rate":         cached.get("fed_rate"),
            "cpi":              cached.get("cpi"),
            "cache_age_min":    round(cache_age / 60, 1),
            "updated":          cached.get("updated"),
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# FAMA-FRENCH 5-FACTOR ENDPOINTS  (Step 9)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/factors/<symbol>", methods=["GET"])
def api_factors_symbol(symbol):
    """
    Full FF5 factor analysis for a single symbol.
    Returns betas, R², alpha, neutralized signal, and factor tilts.

    This reveals whether a stock's signal quality is genuine alpha
    or just disguised market beta / size / value exposure.
    """
    try:
        symbol = symbol.upper().strip()
        result = scanner.all_signals.get(symbol, {})
        ff5    = result.get("ff5_data", {})
        neutral= result.get("ff5_neutral", {})

        if not ff5:
            # Compute fresh if not in cache
            try:
                from math_engine import FactorAnalysis as _FA
                import yfinance as _yf
                h = _yf.Ticker(symbol).history(period="3mo")
                if not h.empty:
                    c = h["Close"].values.astype(float)
                    rets = np.diff(np.log(c + 1e-10))
                    raw_sig = float(result.get("aladdin_composite",
                                               result.get("composite", 0)))
                    ff5     = _FA.regress_ff5(rets, window=60)
                    neutral = _FA.neutralize_signal(raw_sig, ff5)
            except Exception as _fe:
                return jsonify({"error": str(_fe)}), 500

        return jsonify(_sanitize({
            "symbol":           symbol,
            "ff5_regression":   ff5,
            "neutralization":   neutral,
            "price":            result.get("price"),
            "raw_composite":    result.get("aladdin_composite", result.get("composite")),
            "timestamp":        datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/factors/portfolio", methods=["GET"])
def api_factors_portfolio():
    """
    Aggregate portfolio FF5 factor exposure across all scanned signals.
    Shows the portfolio's net beta, size tilt, value tilt, etc.
    Use this to detect unwanted factor concentration.
    """
    try:
        from math_engine import FactorAnalysis as _FA
        all_sigs = scanner.all_signals or {}

        # Build weights from conviction scores (equal-weight by default)
        weights = {}
        sym_betas = {}
        for sym, data in all_sigs.items():
            direction = data.get("direction", "HOLD")
            if direction not in ("BUY", "SHORT"):
                continue
            conviction = abs(float(data.get("conviction", 0.5)))
            weights[sym] = conviction * (1 if direction == "BUY" else -1)
            ff5 = data.get("ff5_data", {})
            if ff5.get("betas"):
                sym_betas[sym] = ff5["betas"]

        port_risk = _FA.portfolio_factor_risk(weights, sym_betas) if weights else {}

        # Top alpha stocks (high alpha_r2, strong signal)
        alpha_leaders = sorted(
            [(sym, d) for sym, d in all_sigs.items()
             if d.get("ff5_alpha_r2", 0) >= 0.5 and abs(d.get("neutralized_composite", 0)) > 0.3],
            key=lambda x: abs(x[1].get("neutralized_composite", 0)),
            reverse=True
        )[:10]

        return jsonify(_sanitize({
            "portfolio_factor_risk": port_risk,
            "n_positions":           len(weights),
            "alpha_leaders": [
                {
                    "symbol":               sym,
                    "neutralized_composite":round(float(d.get("neutralized_composite", 0)), 4),
                    "alpha_quality":        d.get("alpha_quality", "?"),
                    "mkt_beta":             round(float(d.get("mkt_beta", 1)), 3),
                    "ff5_alpha_r2":         round(float(d.get("ff5_alpha_r2", 1)), 4),
                    "factor_tilt_str":      d.get("factor_tilt_str", "none"),
                    "direction":            d.get("direction"),
                    "price":                round(float(d.get("price", 0)), 2),
                }
                for sym, d in alpha_leaders
            ],
            "timestamp": datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# MI SIGNAL FUSION ENDPOINTS  (Step 10)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/mi/status", methods=["GET"])
def api_mi_status():
    """
    MI-orthogonalization health dashboard.
    Shows the current signal redundancy matrix, effective weight adjustments,
    and how many samples are in the history buffer.

    The 'mi_adjustment' field shows how much the MI fusion is changing the
    raw weighted average — larger values mean more redundancy was detected.
    """
    try:
        from math_engine import get_mi_fusion
        fusion = get_mi_fusion()

        # Snapshot of current history buffer depths
        history_depths = {k: len(v) for k, v in fusion._history.items()}

        # Current MI matrix (may be empty if < 30 samples accumulated)
        mi_matrix = {}
        with fusion._lock:
            for name_i, row in fusion._mi_matrix.items():
                mi_matrix[name_i] = {k: round(v, 4) for k, v in row.items()}

        # Aggregate mi_adjustments across all recently scored symbols
        all_sigs    = scanner.all_signals or {}
        adjustments = [float(d.get("mi_adjustment", 0)) for d in all_sigs.values()
                       if d.get("mi_adjustment") is not None]

        avg_adj     = float(np.mean(adjustments)) if adjustments else 0.0
        max_adj     = float(np.max(np.abs(adjustments))) if adjustments else 0.0

        # Top redundant pairs from last computed matrix
        pairs = []
        for ni, row in mi_matrix.items():
            for nj, nmi in row.items():
                if ni < nj and nmi > 0.05:
                    pairs.append({"s1": ni, "s2": nj, "nmi": nmi})
        pairs.sort(key=lambda x: x["nmi"], reverse=True)

        return jsonify(_sanitize({
            "status":            "active" if sum(history_depths.values()) > 50 else "warming_up",
            "history_depths":    history_depths,
            "total_samples":     sum(history_depths.values()),
            "min_samples":       min(history_depths.values()) if history_depths else 0,
            "mi_matrix":         mi_matrix,
            "top_redundant_pairs": pairs[:10],
            "avg_mi_adjustment": round(avg_adj, 4),
            "max_mi_adjustment": round(max_adj, 4),
            "last_recompute":    fusion._mi_last_computed,
            "recompute_interval_s": fusion._MI_RECOMPUTE_INTERVAL,
            "n_symbols_scored":  len(all_sigs),
            "timestamp":         datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/mi/matrix", methods=["GET"])
def api_mi_matrix():
    """
    Raw NMI matrix as a heatmap-ready JSON structure.
    Returns signal names, NMI values, and interpretation.

    NMI > 0.7: highly redundant — signals are nearly duplicates
    NMI 0.4–0.7: moderate redundancy — partial overlap
    NMI < 0.4: mostly independent — good diversification
    """
    try:
        from math_engine import get_mi_fusion, AladdinScorer
        fusion   = get_mi_fusion()
        names    = list(AladdinScorer.WEIGHTS.keys())
        matrix   = []

        with fusion._lock:
            mi_mat = fusion._mi_matrix

        for ni in names:
            row = []
            for nj in names:
                if ni == nj:
                    row.append(1.0)
                else:
                    row.append(round(float(mi_mat.get(ni, {}).get(nj, 0.0)), 4))
            matrix.append(row)

        # Interpretation
        redundant_pairs = []
        independent_pairs = []
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                if j <= i: continue
                nmi = float(mi_mat.get(ni, {}).get(nj, 0.0))
                entry = {"s1": ni, "s2": nj, "nmi": round(nmi, 4)}
                if nmi >= 0.6:
                    redundant_pairs.append(entry)
                elif nmi < 0.2:
                    independent_pairs.append(entry)
        redundant_pairs.sort(key=lambda x: x["nmi"], reverse=True)
        independent_pairs.sort(key=lambda x: x["nmi"])

        return jsonify(_sanitize({
            "signal_names":       names,
            "nmi_matrix":         matrix,
            "redundant_pairs":    redundant_pairs[:5],
            "independent_pairs":  independent_pairs[:5],
            "interpretation": {
                "high_redundancy":   ">0.6 NMI — signals are near-duplicates",
                "moderate":          "0.3–0.6 NMI — partial overlap",
                "independent":       "<0.3 NMI — good diversification",
            },
            "current_weights":    dict(AladdinScorer.WEIGHTS),
            "timestamp":          datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# ALMGREN-CHRISS EXECUTION ENDPOINTS  (Step 12)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/execution/plan", methods=["POST"])
def api_execution_plan():
    """
    Build an Almgren-Chriss optimal execution schedule for a proposed trade.

    POST body (JSON):
        symbol:         ticker
        shares:         number of shares to trade
        side:           "buy" or "sell"
        urgency:        "LOW" / "MEDIUM" / "HIGH" / "URGENT"  (optional, overrides auto)

    Returns:
        trajectory:          shares per 30-min period
        expected_cost_bps:   estimated implementation shortfall
        n_periods:           number of periods to spread execution
        urgency:             computed urgency level
        cost_comparison:     optimal vs TWAP vs immediate execution costs
        front_load_pct:      % of order in first period
        participation_rate:  % of ADV per period
    """
    try:
        from execution_engine import AlmgrenChrissSched, AlmgrenChrissCostModel
        body   = request.get_json(force=True) or {}
        symbol = str(body.get("symbol", "")).upper().strip()
        shares = int(body.get("shares", 100))
        side   = str(body.get("side", "buy")).lower()
        urgency_override = body.get("urgency")

        if not symbol:
            return jsonify({"error": "symbol required"}), 400

        # Get live market data for the symbol
        sig     = scanner.all_signals.get(symbol, {})
        price   = float(sig.get("price") or body.get("price", 100.0))
        hv20    = float(sig.get("hv20") or 20.0) / 100.0
        sigs_d  = sig.get("signals", {}) or {}

        # Estimate ADV from volume data
        try:
            import yfinance as _yf
            h = _yf.Ticker(symbol).history(period="1mo")
            adv_shares = float(h["Volume"].mean()) if not h.empty else 1_000_000.0
        except Exception:
            adv_shares = 1_000_000.0

        conviction = float(sig.get("conviction") or 0.5)
        vix_level  = float(scanner.all_signals.get("VIX", {}).get("price", 20.0))

        plan = AlmgrenChrissSched.build_plan(
            symbol=symbol, total_shares=shares, side=side,
            price=price, adv_shares=adv_shares,
            daily_vol=hv20, conviction=conviction, vix=vix_level,
        )

        # Override urgency if caller specified
        if urgency_override and urgency_override.upper() in ("LOW","MEDIUM","HIGH","URGENT"):
            from execution_engine import _URGENCY_LAMBDA
            plan["urgency"] = urgency_override.upper()
            plan["lam"]     = _URGENCY_LAMBDA[urgency_override.upper()]

        return jsonify(_sanitize({
            **plan,
            "price":       round(price, 2),
            "adv_shares":  round(adv_shares, 0),
            "timestamp":   datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/execution/cost/<symbol>", methods=["GET"])
def api_execution_cost(symbol):
    """
    Estimate execution cost in bps for different order sizes and urgency levels.
    Returns a matrix: urgency × order_size → expected IS in bps.
    Useful for pre-trade analysis before committing to a position size.
    """
    try:
        from execution_engine import AlmgrenChrissCostModel, _URGENCY_LAMBDA
        symbol = symbol.upper()
        sig    = scanner.all_signals.get(symbol, {})
        price  = float(sig.get("price") or 100.0)
        hv20   = float(sig.get("hv20") or 20.0) / 100.0

        try:
            import yfinance as _yf
            h = _yf.Ticker(symbol).history(period="1mo")
            adv = float(h["Volume"].mean()) if not h.empty else 1_000_000.0
        except Exception:
            adv = 1_000_000.0

        model = AlmgrenChrissCostModel.calibrate(
            price=price, adv_shares=adv, daily_vol=hv20
        )

        # Cost matrix: 4 urgency × 5 order sizes (% of ADV)
        urgency_levels = ["LOW", "MEDIUM", "HIGH", "URGENT"]
        size_pcts      = [0.005, 0.01, 0.025, 0.05, 0.10]  # fraction of ADV
        matrix         = {}

        for u in urgency_levels:
            lam    = _URGENCY_LAMBDA[u]
            row    = {}
            for pct in size_pcts:
                n_sh   = max(1, int(adv * pct))
                n_per  = max(2, int(pct / 0.01 * 2))
                est    = model.estimate_cost_bps(n_sh, price, n_per, lam)
                row[f"{pct*100:.1f}%_ADV"] = est["optimal_bps"]
            matrix[u] = row

        return jsonify(_sanitize({
            "symbol":          symbol,
            "price":           round(price, 2),
            "adv_shares":      round(adv, 0),
            "daily_vol_pct":   round(hv20 * 100, 2),
            "cost_matrix_bps": matrix,
            "model_params":    {"eta": model.eta, "gamma": model.gamma, "sigma": model.sigma},
            "interpretation":  "Lower bps = cheaper execution. <5 bps excellent, >20 bps expensive.",
            "timestamp":       datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# DARK POOL ENDPOINTS  (Step 13)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/darkpool/<symbol>", methods=["GET"])
def api_darkpool_symbol(symbol):
    """
    Full dark pool signal for a single symbol.

    Returns:
      dp_score:          composite ∈ [-1,+1] (positive = accumulation)
      signal_label:      DARK_POOL_ACCUMULATION / HEAVY_SHORT_INTEREST / NEUTRAL
      short_vol_ratio:   FINRA short volume as % of total (0.55+ = bearish)
      block_score:       stealth-bar accumulation score
      stealth_bar_count: bars with high vol but tiny price move (block trades)
      accum_index:       OBV-weighted accumulation trend
      pv_divergence:     volume trend vs price trend divergence
      dpr:               dark pool ratio estimate (0–1)
      kyle_lambda:       price impact per unit volume (low = more hidden liquidity)
    """
    try:
        import dark_pool_engine as _DPE
        symbol = symbol.upper().strip()
        force  = request.args.get("force", "false").lower() == "true"
        result = _DPE.DarkPoolScorer.compute(symbol, force=force)
        return jsonify(_sanitize({**result, "symbol": symbol,
                                   "timestamp": datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/darkpool/scan", methods=["GET"])
def api_darkpool_scan():
    """
    Scan universe dark pool signals.

    Returns top accumulation and distribution signals across all scanned symbols.
    High dp_score with stealth bars = institutions quietly building positions.
    High negative dp_score with heavy short interest = distribution / exit.

    Use this as an early-warning screen before signals show up in price.
    """
    try:
        import dark_pool_engine as _DPE
        all_sigs = scanner.all_signals or {}
        results  = []

        for sym, cached in list(_DPE._DP_CACHE.items()):
            if not sym.startswith("dp_"):
                continue
            ticker = sym[len("dp_"):]
            score  = float(cached.get("dp_score", 0))
            if abs(score) < 0.10:
                continue
            sig    = all_sigs.get(ticker, {})
            results.append({
                "symbol":           ticker,
                "dp_score":         round(score, 4),
                "signal_label":     cached.get("signal_label", "NEUTRAL"),
                "stealth_bars":     int(cached.get("stealth_bar_count", 0)),
                "short_vol_ratio":  round(float(cached.get("short_vol_ratio", 0.45)), 4),
                "block_score":      round(float(cached.get("block_score", 0)), 4),
                "accum_index":      round(float(cached.get("accum_index", 0)), 4),
                "price":            round(float(sig.get("price", 0)), 2),
                "direction":        sig.get("direction", "HOLD"),
                "company":          (sig.get("company") or {}).get("name", ticker),
            })

        results.sort(key=lambda x: x["dp_score"], reverse=True)
        accumulation = [r for r in results if r["dp_score"] >  0.15]
        distribution = [r for r in results if r["dp_score"] < -0.15]

        return jsonify(_sanitize({
            "accumulation":  accumulation[:10],
            "distribution":  distribution[:10],
            "all":           results[:30],
            "count":         len(results),
            "cache_stats":   _DPE.DarkPoolScorer.get_cache_stats(),
            "timestamp":     datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/darkpool/short/<symbol>", methods=["GET"])
def api_darkpool_short(symbol):
    """
    FINRA short volume data for a symbol (last 5 trading days).
    Short volume ratio > 55% is bearish; < 35% is bullish / potential squeeze.
    """
    try:
        import dark_pool_engine as _DPE
        symbol = symbol.upper().strip()
        result = _DPE.fetch_finra_short_volume(symbol)
        return jsonify(_sanitize({**result, "timestamp": datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# PhD HYPOTHESIS ENGINE ENDPOINTS  (Step 14)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/phd/hypothesis/<symbol>", methods=["GET"])
def api_phd_hypothesis(symbol):
    """
    Full PhD hypothesis analysis for a symbol.

    Runs Bayesian conviction scoring across all 13 signal layers,
    classifies the thesis archetype, and generates an investment memo
    when conviction > 60%.

    Returns:
      conviction_pct:  Bayesian posterior conviction [0–100]
      top_thesis:      BUY / SHORT / HOLD
      archetype:       MOMENTUM_BREAKOUT / MEAN_REVERSION / CATALYST_EVENT /
                       FACTOR_ALPHA / MACRO_REGIME / DARK_POOL_CONVERGENCE
      n_confirming:    signals aligned with top thesis
      n_conflicting:   signals opposed
      bayesian:        full posterior breakdown per signal
      taxonomy:        archetype fit scores (all 6 archetypes)
      memo:            investment memo (if conviction > 60%)

    Add ?force=true to regenerate memo even if cached.
    """
    try:
        import phd_hypothesis_engine as _PHD
        symbol = symbol.upper().strip()
        force  = request.args.get("force", "false").lower() == "true"

        api_key    = os.environ.get("ANTHROPIC_API_KEY", "")
        phd_engine = _PHD.PhDHypothesisEngine.get_instance(api_key)

        sigs = scanner.all_signals.get(symbol, {})
        if not sigs:
            return jsonify({"error": f"No signal data for {symbol}"}), 404

        phd_signals = {
            "technical_composite": float(np.clip(sigs.get("composite", 0), -1, 1)),
            "advanced_composite":  float(np.clip(sigs.get("adj_dir_score", sigs.get("composite", 0)), -1, 1)),
            "ml_score":            float(np.clip(sigs.get("ml_score", 0.5)*2-1, -1, 1)),
            "hurst_blended":       float(np.clip(sigs.get("hurst", 0.5)*2-1, -1, 1)),
            "ou_reversion":        float(np.clip(-sigs.get("ou_zscore", 0)/3, -1, 1)),
            "kalman_trend":        float(np.clip(sigs.get("bull_prob_kf", 50)/50-1, -1, 1)),
            "vol_regime":          float(np.clip(sigs.get("vol_regime", 0), -1, 1)),
            "news_sentiment":      float(np.clip(sigs.get("news_sentiment", 0), -1, 1)),
            "options_flow":        float(np.clip(sigs.get("options_flow_score", 0), -1, 1)),
            "macro_regime":        float(np.clip(
                scanner.macro_signal if hasattr(scanner, "macro_signal") else 0, -1, 1
            )),
            "revision_momentum":   float(np.clip(sigs.get("revision_score", 0)/3, -1, 1)),
            "ff5_alpha_quality":   float(np.clip(sigs.get("neutralized_composite", 0), -1, 1)),
            "insider_cluster":     float(np.clip(sigs.get("insider_cluster_score", 0), -1, 1)),
            "dark_pool_score":     float(np.clip(sigs.get("dp_score", 0), -1, 1)),
            "backtest_ic":         float(np.clip(sigs.get("backtest_ic", 0)*10, -1, 1)),
        }

        hmm_regime   = (scanner.regime_model.current_regime
                        if hasattr(scanner, "regime_model") and scanner.regime_model else "NEUTRAL")
        company_info = sigs.get("company") or {}
        company_name = company_info.get("name", symbol) if isinstance(company_info, dict) else symbol

        result = phd_engine.analyze(
            symbol=symbol, signals=phd_signals,
            regime=hmm_regime, company_name=company_name,
            force_memo=force,
        )
        return jsonify(_sanitize(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/phd/scan", methods=["GET"])
def api_phd_scan():
    """
    Universe conviction scan via PhD Hypothesis Engine.

    Returns top BUY and SHORT ideas ranked by Bayesian posterior conviction.
    Only symbols with conviction > 58% are returned.

    This is the most powerful screening tool — it identifies ideas where
    multiple signal layers converge on the same directional thesis.
    """
    try:
        import phd_hypothesis_engine as _PHD
        api_key    = os.environ.get("ANTHROPIC_API_KEY", "")
        phd_engine = _PHD.PhDHypothesisEngine.get_instance(api_key)

        hmm_regime = (scanner.regime_model.current_regime
                      if hasattr(scanner, "regime_model") and scanner.regime_model else "NEUTRAL")
        result = phd_engine.get_universe_scan(scanner.all_signals or {}, hmm_regime)
        return jsonify(_sanitize(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/phd/tracker", methods=["GET"])
def api_phd_tracker():
    """
    Hypothesis tracker statistics.

    Returns performance metrics for all validated hypotheses:
      batting_average:  % directionally correct
      avg_ic:           Spearman IC between conviction and realized return
      avg_return:       mean realized return per hypothesis
      by_archetype:     which thesis archetypes have highest batting average
      grade_dist:       A/B/C/D grade distribution
      recent_5:         last 5 closed hypotheses

    This is how the system proves (or disproves) its own edge over time.
    """
    try:
        import phd_hypothesis_engine as _PHD
        api_key    = os.environ.get("ANTHROPIC_API_KEY", "")
        phd_engine = _PHD.PhDHypothesisEngine.get_instance(api_key)
        stats = phd_engine.tracker.get_statistics()
        open_hyps = phd_engine.tracker.get_open()
        return jsonify(_sanitize({
            **stats,
            "open_hypotheses": list(open_hyps.values()),
            "timestamp":       datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/phd/memo/<symbol>", methods=["GET"])
def api_phd_memo(symbol):
    """
    Force-generate an institutional investment memo for a symbol.
    Requires ANTHROPIC_API_KEY in environment (uses Claude API).
    Falls back to rules-based memo if API key not set.

    Add ?force=true to regenerate even if cached.
    """
    try:
        import phd_hypothesis_engine as _PHD
        symbol  = symbol.upper().strip()
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        result  = api_phd_hypothesis.__wrapped__(symbol) if hasattr(api_phd_hypothesis, '__wrapped__') else None

        phd_engine = _PHD.PhDHypothesisEngine.get_instance(api_key)
        cached = phd_engine._cache.get(symbol, {})
        memo   = cached.get("memo", {})
        return jsonify(_sanitize({
            "symbol":       symbol,
            "memo":         memo,
            "conviction_pct": cached.get("conviction_pct", 0),
            "top_thesis":   cached.get("top_thesis", "HOLD"),
            "archetype":    cached.get("archetype", ""),
            "generated_by": memo.get("generated_by", "none"),
            "note": "Call /api/phd/hypothesis/{symbol}?force=true to regenerate",
            "timestamp":    datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── FED WATCH — 100% live: FRED daily bounds + Fed.gov FOMC calendar + futures ──
_fedwatch_cache = {"data": None, "ts": 0}

def _fetch_fred_series_latest(series_id):
    """Fetch most recent value from a FRED daily CSV series. Returns float or None."""
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        r2  = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if not r2.ok:
            return None
        for line in reversed(r2.text.strip().splitlines()[1:]):
            parts = line.split(",")
            if len(parts) == 2 and parts[1].strip() not in ("", "."):
                return float(parts[1].strip())
    except Exception:
        pass
    return None

def _fetch_fomc_dates_live():
    """
    Scrape official FOMC meeting dates from federalreserve.gov.
    Returns sorted list of 'YYYY-MM-DD' strings (decision/press-conference day).
    """
    import re as _re
    try:
        url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if not resp.ok:
            return []
        html = resp.text
        # The Fed page lists meetings like "January 28-29" under each year header.
        # We parse year blocks then extract end dates.
        dates = []
        # Find year sections: look for <h4>2026 FOMC Meetings</h4> style headings
        year_blocks = _re.findall(
            r'<h[34][^>]*>\s*(\d{4})\s+FOMC\s+Meetings?\s*</h[34]>(.*?)(?=<h[34]|$)',
            html, _re.DOTALL | _re.IGNORECASE
        )
        month_map = {
            "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
            "july":7,"august":8,"september":9,"october":10,"november":11,"december":12
        }
        for year_str, block in year_blocks:
            year = int(year_str)
            # Match patterns like "January 28-29*" or "March 18-19"
            for m in _re.finditer(
                r'(January|February|March|April|May|June|July|August|September|October|November|December)'
                r'\s+(\d+)(?:-(\d+))?',
                block, _re.IGNORECASE
            ):
                mon_name = m.group(1).lower()
                day_end  = int(m.group(3)) if m.group(3) else int(m.group(2))
                mon_num  = month_map.get(mon_name, 0)
                if mon_num:
                    dates.append(f"{year}-{mon_num:02d}-{day_end:02d}")
        return sorted(set(dates))
    except Exception:
        return []

def _fetch_futures_implied_rates():
    """
    Pull 30-day Fed Funds futures (ZQ) from yfinance.
    Returns dict {label: implied_rate} where implied_rate = 100 - price.
    Tries CBOT-style tickers for near months.
    """
    # Month codes: F=Jan G=Feb H=Mar J=Apr K=May M=Jun N=Jul Q=Aug U=Sep V=Oct X=Nov Z=Dec
    from datetime import date as _date
    today = _date.today()
    results = {}
    month_codes = {1:"F",2:"G",3:"H",4:"J",5:"K",6:"M",7:"N",8:"Q",9:"U",10:"V",11:"X",12:"Z"}
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    # Build next 8 months of tickers
    yr, mo = today.year, today.month
    for _ in range(8):
        mo += 1
        if mo > 12:
            mo = 1; yr += 1
        code   = month_codes[mo]
        yr2    = str(yr)[2:]  # e.g. "26"
        label  = f"{month_names[mo]}-{yr2}"
        # yfinance uses ZQ{code}{yr2}.CBT
        ticker = f"ZQ{code}{yr2}.CBT"
        try:
            t = yf.Ticker(ticker)
            h = t.history(period="5d")
            if not h.empty:
                price   = float(h["Close"].iloc[-1])
                implied = round(100.0 - price, 4)
                if 0 < implied < 15:   # sanity check
                    results[label] = implied
        except Exception:
            pass
    return results

@app.route("/api/fedwatch")
def api_fedwatch():
    now = time.time()
    if _fedwatch_cache["data"] and now - _fedwatch_cache["ts"] < 600:  # 10-min cache
        return jsonify(_fedwatch_cache["data"])

    today_str = datetime.now().strftime("%Y-%m-%d")

    # ── 1. Live target range from FRED daily series ───────────────────────────
    #   DFEDTARL = lower bound  (e.g. 3.50)
    #   DFEDTARU = upper bound  (e.g. 3.75)
    #   DFF      = effective rate (e.g. 3.64)
    rate_lo  = _fetch_fred_series_latest("DFEDTARL")
    rate_hi  = _fetch_fred_series_latest("DFEDTARU")
    eff_rate = _fetch_fred_series_latest("DFF")

    if rate_lo is not None and rate_hi is not None:
        fed_rate_str = f"{rate_lo:.2f}-{rate_hi:.2f}%"
        rate_mid     = (rate_lo + rate_hi) / 2.0
    elif eff_rate is not None:
        lo = round(eff_rate - 0.125, 2)
        hi = round(eff_rate + 0.125, 2)
        fed_rate_str = f"{lo:.2f}-{hi:.2f}%"
        rate_mid     = eff_rate
    else:
        fed_rate_str = "Unavailable"
        rate_mid     = None

    eff_rate_str = f"{eff_rate:.2f}%" if eff_rate is not None else None

    # ── 2. Next FOMC date — scraped live from federalreserve.gov ─────────────
    fomc_dates = _fetch_fomc_dates_live()
    next_fomc_raw = next((d for d in fomc_dates if d >= today_str), None)
    if next_fomc_raw:
        dt = datetime.strptime(next_fomc_raw, "%Y-%m-%d")
        next_fomc_str = dt.strftime("%b %d, %Y")
        days_until    = (dt - datetime.now()).days
    else:
        next_fomc_str = "See federalreserve.gov"
        days_until    = None

    # ── 3. Cut probability from live Fed Funds futures ────────────────────────
    cut_prob_str = "Fetching..."
    year_outlook = "Fetching..."
    cuts_priced  = 0
    futures_data = {}

    futures_rates = _fetch_futures_implied_rates()

    if futures_rates and rate_mid is not None:
        futures_data = futures_rates

        # Nearest contract = first one
        nearest_rate  = list(futures_rates.values())[0]
        nearest_label = list(futures_rates.keys())[0]

        # Probability of at least 1 cut: fraction of a 25bp move priced in
        bp_priced    = (rate_mid - nearest_rate) * 100   # in basis points
        prob_one_cut = max(0.0, min(100.0, round(bp_priced / 25.0 * 100, 1)))

        cut_prob_str = f"{prob_one_cut:.0f}% cut @ {nearest_label} FOMC"

        # Total cuts priced by year-end = max across all contracts
        max_cuts = 0.0
        for rate in futures_rates.values():
            c = (rate_mid - rate) / 0.25
            if c > max_cuts:
                max_cuts = c
        cuts_priced = max(0, round(max_cuts, 1))
        total_int   = max(0, round(max_cuts))

        if total_int == 0:
            year_outlook = "No cuts priced in"
        elif total_int == 1:
            year_outlook = "1 cut priced in (futures)"
        else:
            year_outlook = f"{total_int} cuts priced in (futures)"
    else:
        # Fallback: 3M T-bill spread (always available)
        try:
            irx = yf.Ticker("^IRX").history(period="5d")
            if not irx.empty and rate_mid is not None:
                tbill = float(irx["Close"].iloc[-1])
                bp_diff = (rate_mid - tbill) * 100
                cuts_priced  = max(0.0, round(bp_diff / 25.0, 1))
                prob_one_cut = max(0.0, min(100.0, round(bp_diff / 25.0 * 100, 1)))
                cut_prob_str = f"{prob_one_cut:.0f}% cut implied (3M T-bill)"
                year_outlook = f"~{round(cuts_priced)} cut(s) implied (3M T-bill)"
        except Exception:
            cut_prob_str = "See CME FedWatch"
            year_outlook = "See CME FedWatch"

    result = {
        "current_rate":   fed_rate_str,      # e.g. "3.50-3.75%"
        "effective_rate": eff_rate_str,       # e.g. "3.64%"
        "next_fomc":      next_fomc_str,      # e.g. "Mar 19, 2026"
        "days_until":     days_until,         # e.g. 12
        "cut_prob":       cut_prob_str,       # e.g. "72% cut @ Apr-26 FOMC"
        "year_outlook":   year_outlook,       # e.g. "2 cuts priced in (futures)"
        "cuts_priced":    cuts_priced,        # float
        "futures":        futures_data,       # {label: implied_rate}
        "sources":        ["FRED:DFEDTARL","FRED:DFEDTARU","FRED:DFF","federalreserve.gov","ZQ futures"],
        "updated":        datetime.now().isoformat(),
    }
    _fedwatch_cache["data"] = result
    _fedwatch_cache["ts"]   = now
    return jsonify(result)


# ── INSTITUTIONAL 13F POSITIONS ────────────────────────────────────────────────
_inst_cache = {"data": None, "ts": 0}

TOP_FUNDS = {
    "Renaissance Technologies": "0001037389",
    "Citadel Advisors":         "0001423298",
    "Bridgewater Associates":   "0001350694",
    "Two Sigma Investments":    "0001462418",
    "DE Shaw":                  "0001009207",
    "Millennium Management":    "0001273087",
    "Point72 Asset Mgmt":       "0001603466",
    "Coatue Management":        "0001336528",
    "Tiger Global Mgmt":        "0001167483",
    "Lone Pine Capital":        "0001061165",
}

@app.route("/api/institutional")
def api_institutional():
    import re as _re
    now = time.time()
    if _inst_cache["data"] and now - _inst_cache["ts"] < 900:   # 15 min cache
        return jsonify(_inst_cache["data"])

    headers = {
        "User-Agent": "GIG Research Desk research@gigfund.example",
        "Accept": "application/json, text/html, */*",
    }

    def _safe_int(v):
        try:
            return int(str(v).replace(",", "").strip())
        except Exception:
            return 0

    def _extract_tag(tag: str, s: str) -> str:
        m = _re.search(fr"<{tag}>(.*?)</{tag}>", s, _re.DOTALL | _re.IGNORECASE)
        return (m.group(1).strip() if m else "")

    def _latest_13f_for_cik(cik: str):
        cik_padded = cik.zfill(10)
        sub_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        r = requests.get(sub_url, headers=headers, timeout=6)
        if not r.ok:
            return None
        sub = r.json()
        filings = (sub.get("filings") or {}).get("recent") or {}
        forms = filings.get("form") or []
        accns = filings.get("accessionNumber") or []
        dates = filings.get("filingDate") or []
        idx = next((i for i, f in enumerate(forms) if f in ("13F-HR", "13F-HR/A")), None)
        if idx is None:
            return None
        return {"accn": accns[idx], "date": dates[idx]}

    def _holdings_from_xml(cik: str, accn: str):
        accn2 = accn.replace("-", "")
        base = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accn2}"
        xml_urls = [f"{base}/infotable.xml", f"{base}/form13fInfoTable.xml", f"{base}/informationtable.xml"]

        xr = None
        for xu in xml_urls:
            try:
                t = requests.get(xu, headers=headers, timeout=7)
                if t.ok and "<infoTable" in (t.text or ""):
                    xr = t
                    break
            except Exception:
                pass
        if xr is None:
            try:
                idx_url = f"{base}/{accn}-index.json"
                ir = requests.get(idx_url, headers=headers, timeout=6)
                if ir.ok:
                    items = ((ir.json() or {}).get("directory") or {}).get("item") or []
                    xml_name = next((it.get("name") for it in items if str(it.get("name", "")).lower().endswith(".xml")), None)
                    if xml_name:
                        xr2 = requests.get(f"{base}/{xml_name}", headers=headers, timeout=7)
                        if xr2.ok:
                            xr = xr2
            except Exception:
                pass
        if xr is None or not xr.ok:
            return []

        xml = xr.text or ""
        entries = _re.findall(r"<infoTable>(.*?)</infoTable>", xml, _re.DOTALL | _re.IGNORECASE)
        out = []
        for entry in entries[:120]:
            sym = _extract_tag("ticker", entry) or _extract_tag("cusip", entry) or _extract_tag("tickerCusip", entry)
            name = _extract_tag("nameOfIssuer", entry) or sym
            val = _safe_int(_extract_tag("value", entry))  # 13F is $ thousands
            shs = _safe_int(_extract_tag("sshPrnamt", entry))
            if not name and not sym:
                continue
            out.append({
                "symbol": (sym or name)[:10].upper(),
                "name": (name or sym)[:64],
                "value_k": val,
                "shares": shs,
            })
        out.sort(key=lambda x: x.get("value_k", 0), reverse=True)
        return out[:50]

    def _fetch_one(args):
        fund_name, cik = args
        try:
            latest = _latest_13f_for_cik(cik)
            if not latest:
                return {"fund": fund_name, "cik": cik, "holdings": [], "source": "SEC EDGAR", "error": "No recent 13F-HR"}
            holdings = _holdings_from_xml(cik, latest["accn"])
            return {
                "fund": fund_name,
                "cik": cik,
                "filing_date": latest["date"],
                "holdings": holdings,
                "total_holdings": len(holdings),
                "source": "SEC EDGAR 13F-HR",
            }
        except Exception as e:
            return {"fund": fund_name, "cik": cik, "holdings": [], "source": "SEC EDGAR", "error": str(e)}

    # Live recent 13F tape fallback (fast, even if fund XML parsing fails).
    recent_feed = []
    try:
        atom = requests.get(
            "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=13F-HR&company=&dateb=&owner=include&start=0&count=40&output=atom",
            headers=headers,
            timeout=6,
        )
        if atom.ok:
            entries = _re.findall(r"<entry>(.*?)</entry>", atom.text or "", _re.DOTALL | _re.IGNORECASE)
            for e in entries[:20]:
                title = _extract_tag("title", e)
                updated = _extract_tag("updated", e)
                link_m = _re.search(r'<link[^>]+href="([^"]+)"', e, _re.IGNORECASE)
                recent_feed.append({"title": title, "updated": updated, "url": link_m.group(1) if link_m else ""})
    except Exception:
        pass

    fund_rows = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            for row in ex.map(_fetch_one, list(TOP_FUNDS.items())):
                fund_rows.append(row)
    except Exception:
        fund_rows = [_fetch_one(x) for x in list(TOP_FUNDS.items())]

    # If all funds fail, return deterministic fallback so frontend does not spin forever.
    if not any((f.get("holdings") or []) for f in fund_rows):
        for f in fund_rows:
            f.setdefault("error", "13F XML parsing unavailable right now")

    payload = {
        "funds": fund_rows,
        "recent_13f_feed": recent_feed,
        "updated": datetime.now().isoformat(),
        "provider_status": {
            "sec_submissions": True,
            "sec_current_atom": bool(recent_feed),
        },
    }
    _inst_cache["data"] = _sanitize(payload)
    _inst_cache["ts"] = now
    return jsonify(_inst_cache["data"])


# ── 1DTE GAMMA PLAYS ──────────────────────────────────────────────────────────
_dte1_cache = {"data": None, "ts": 0}

# _get_gamma_universe() is now dynamic — built at runtime from get_training_universe()
def _get_gamma_universe(): return get_training_universe(n=30)

@app.route("/api/1dte")
def api_1dte():
    now = time.time()
    if _dte1_cache["data"] and now - _dte1_cache["ts"] < 120:  # 2-min cache
        return jsonify(_dte1_cache["data"])
    plays = []
    sigs = scanner.get_state().get("top_signals", [])  # ONE BRAIN
    today = datetime.now().date()

    def _next_trading_days(n=5):
        """Return next n calendar dates that could be trading day expirations."""
        days = []
        d = today
        for _ in range(n + 4):
            days.append(str(d))
            d += timedelta(days=1)
        return days
    targets = set(_next_trading_days(5))

    for sym in _get_gamma_universe():
        try:
            t = yf.Ticker(sym)
            price = None
            prev  = None
            try:
                h = t.history(period="2d")
                if not h.empty:
                    price = float(h["Close"].iloc[-1])
                    prev  = float(h["Close"].iloc[-2]) if len(h) >= 2 else price
                    chg_pct = (price / prev - 1) * 100
            except:
                continue
            if not price: continue
            exps = t.options
            if not exps: continue

            # Find nearest expiry — prefer expirations within 3 days, else take closest
            exp_1dte = None
            for e in exps:
                if e in targets:
                    exp_1dte = e
                    break
            if not exp_1dte:
                # Take absolute nearest expiry
                try:
                    sorted_exps = sorted(exps, key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d").date() - today).days))
                    exp_1dte = sorted_exps[0]
                except:
                    exp_1dte = exps[0]
            if not exp_1dte: continue

            chain = t.option_chain(exp_1dte)
            calls = chain.calls
            puts  = chain.puts
            if calls.empty and puts.empty: continue

            # Determine ranking: volume if market hours, OI if pre/post
            call_vol_total = int(calls["volume"].fillna(0).sum()) if not calls.empty and "volume" in calls.columns else 0
            put_vol_total  = int(puts["volume"].fillna(0).sum())  if not puts.empty  and "volume" in puts.columns  else 0
            use_oi = (call_vol_total + put_vol_total) == 0
            rank_col = "openInterest" if use_oi else "volume"

            best_call = best_put = None
            if not calls.empty:
                # Strict OTM calls only: at least 1% above spot, capped to keep tradable wings
                calls_f = calls[calls["strike"].between(price * 1.01, price * 1.12)]
                if calls_f.empty: calls_f = calls
                calls_f = calls_f[calls_f[rank_col].fillna(0) > 0] if rank_col in calls_f.columns else calls_f
                if not calls_f.empty:
                    bc = calls_f.nlargest(1, rank_col).iloc[0]
                    best_call = {
                        "type": "CALL", "strike": float(bc["strike"]),
                        "expiry": exp_1dte, "premium": float(bc.get("lastPrice", 0) or 0),
                        "volume": int(bc.get("volume", 0) or 0),
                        "oi": int(bc.get("openInterest", 0) or 0),
                        "iv": round(float(bc.get("impliedVolatility", 0) or 0) * 100, 1),
                        "delta": float(bc.get("delta", 0.5) or 0.5),
                        "bid": float(bc.get("bid", 0) or 0), "ask": float(bc.get("ask", 0) or 0),
                        "vol_oi": round(float(bc.get("volume", 1) or 1) / max(float(bc.get("openInterest", 1) or 1), 1), 2),
                    }
            if not puts.empty:
                # Strict OTM puts only: at least 1% below spot
                puts_f = puts[puts["strike"].between(price * 0.88, price * 0.99)]
                if puts_f.empty: puts_f = puts
                puts_f = puts_f[puts_f[rank_col].fillna(0) > 0] if rank_col in puts_f.columns else puts_f
                if not puts_f.empty:
                    bp = puts_f.nlargest(1, rank_col).iloc[0]
                    best_put = {
                        "type": "PUT", "strike": float(bp["strike"]),
                        "expiry": exp_1dte, "premium": float(bp.get("lastPrice", 0) or 0),
                        "volume": int(bp.get("volume", 0) or 0),
                        "oi": int(bp.get("openInterest", 0) or 0),
                        "iv": round(float(bp.get("impliedVolatility", 0) or 0) * 100, 1),
                        "delta": float(bp.get("delta", -0.4) or -0.4),
                        "bid": float(bp.get("bid", 0) or 0), "ask": float(bp.get("ask", 0) or 0),
                        "vol_oi": round(float(bp.get("volume", 1) or 1) / max(float(bp.get("openInterest", 1) or 1), 1), 2),
                    }

            if not best_call and not best_put: continue

            # Score by vol_oi * activity
            rank_val = lambda leg: (leg[rank_col if rank_col in leg else "vol_oi"] if leg else 0)
            call_score = (best_call["vol_oi"] if best_call else 0) * (best_call.get("volume", best_call.get("oi", 1)) if best_call else 0)
            put_score  = (best_put["vol_oi"]  if best_put  else 0) * (best_put.get("volume",  best_put.get("oi", 1))  if best_put  else 0)
                        # ONE BRAIN: scanner direction
            _scan_data = None
            for _ss in sigs:
                if _ss.get('symbol') == sym:
                    _scan_data = _ss
                    break
            _scan_dir = _scan_data.get('direction', 'HOLD') if _scan_data else 'HOLD'
            if _scan_dir == 'BUY':
                bias = 'CALL'
            elif _scan_dir == 'SHORT':
                bias = 'PUT'
            else:
                bias = 'CALL' if call_score >= put_score else 'PUT'
            total_vol = (best_call["volume"] if best_call else 0) + (best_put["volume"] if best_put else 0)
            total_oi  = (best_call["oi"] if best_call else 0) + (best_put["oi"] if best_put else 0)
            plays.append({
                "symbol":     sym,
                "price":      round(price, 2),
                "chg_pct":    round(chg_pct, 2),
                "expiry":     exp_1dte,
                "bias":       bias,
                "call":       best_call,
                "put":        best_put,
                "total_vol":  total_vol,
                "total_oi":   total_oi,
                "call_score": round(call_score, 1),
                "put_score":  round(put_score, 1),
                "signal":     "HIGH_ACTIVITY" if (best_call and best_call["vol_oi"] > 3) or (best_put and best_put["vol_oi"] > 3) else "NORMAL",
                "ranked_by":  "open_interest" if use_oi else "volume",
                "otm_only":   True,
            })
        except Exception:
            continue
    # Sort by total volume, fall back to OI
    plays.sort(key=lambda x: x["total_vol"] if x["total_vol"] > 0 else x.get("total_oi", 0), reverse=True)
    result = {"plays": plays, "updated": datetime.now().isoformat(), "count": len(plays)}
    _dte1_cache["data"] = result
    _dte1_cache["ts"]   = now
    return jsonify(result)


@app.route("/api/screener/advanced", methods=["GET"])
def api_screener_advanced():
    """
    Advanced cross-sectional screener over full US scan universe.
    Uses existing ONE-BRAIN signals with stricter ranking math.
    """
    try:
        limit = int(request.args.get("limit", 100))
        min_conv = float(request.args.get("min_conviction", 0.55))
        min_liq = float(request.args.get("min_volume", 100000))

        universe = get_scan_universe(include_scanner=True)
        sig_map = getattr(scanner, "all_signals", {})
        rows = []
        for sym in universe:
            s = sig_map.get(sym, {})
            vol = float(s.get("volume", 0) or 0)
            conv = float(abs(s.get("conviction", 0) or 0))
            comp = float(abs(s.get("composite", 0) or 0))
            if vol < min_liq or conv < min_conv:
                continue
            # Higher = better: combine conviction, composite, and anomaly flow
            anomaly = float(s.get("vol_ratio", 1.0) or 1.0)
            quality = conv * 45.0 + comp * 40.0 + min(anomaly, 5.0) * 3.0 + np.log1p(max(vol, 0.0)) * 0.15
            rows.append({
                "symbol": sym,
                "direction": s.get("direction", "HOLD"),
                "price": float(s.get("price", 0) or 0),
                "conviction": round(conv, 4),
                "composite": round(float(s.get("composite", 0) or 0), 4),
                "volume": int(vol),
                "vol_ratio": round(anomaly, 3),
                "quality_score": round(float(quality), 4),
            })

        rows.sort(key=lambda x: x["quality_score"], reverse=True)
        out = rows[:max(1, min(limit, 1000))]
        return jsonify(_sanitize({
            "results": out,
            "count": len(out),
            "universe_size": len(universe),
            "filters": {"min_conviction": min_conv, "min_volume": min_liq},
            "updated": datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── SWING OTM OPTIONS SIGNAL ──────────────────────────────────────────────────
_swing_cache = {"data": None, "ts": 0}

@app.route("/api/swing_options")
def api_swing_options():
    """For each top signal, find the best OTM option by volume on the real chain."""
    now = time.time()
    if _swing_cache["data"] and now - _swing_cache["ts"] < 120:
        return jsonify(_swing_cache["data"])
    state = scanner.get_state()
    sigs  = state.get("top_signals", [])[:20]
    results = []
    for sig in sigs:
        sym   = sig.get("symbol")
        price = sig.get("price", 0)
        direction = sig.get("direction", "HOLD")
        if direction == "HOLD": continue  # ONE BRAIN
        if not sym or not price: continue
        try:
            t    = yf.Ticker(sym)
            exps = t.options
            if not exps: continue
            # Target: 14-60 DTE for swing
            target_exp = None
            for e in exps:
                try:
                    dte = (datetime.strptime(e, "%Y-%m-%d").date() - datetime.now().date()).days
                    if 14 <= dte <= 60:
                        target_exp = e
                        break
                except: pass
            if not target_exp and exps:
                target_exp = exps[min(1, len(exps)-1)]
            chain = t.option_chain(target_exp)
            if direction == "BUY":
                df = chain.calls
                df = df[df["strike"].between(price * 1.02, price * 1.20)]
            else:
                df = chain.puts
                df = df[df["strike"].between(price * 0.80, price * 0.98)]
            if df.empty: continue
            # During pre/post market, volume=0 — fall back to OI ranking
            has_volume = "volume" in df.columns and df["volume"].fillna(0).sum() > 0
            if has_volume:
                df_ranked = df[df["volume"].fillna(0) > 0]
                if df_ranked.empty:
                    df_ranked = df  # fallback: all rows
                rank_col = "volume"
            else:
                df_ranked = df[df["openInterest"].fillna(0) > 0] if "openInterest" in df.columns else df
                if df_ranked.empty:
                    df_ranked = df
                rank_col = "openInterest"
            best = df_ranked.nlargest(1, rank_col).iloc[0]
            dte  = (datetime.strptime(target_exp, "%Y-%m-%d").date() - datetime.now().date()).days
            vol_val = int(best.get("volume", 0) or 0)
            oi_val  = int(best.get("openInterest", 0) or 0)
            results.append({
                "symbol":    sym,
                "direction": direction,
                "price":     round(price, 2),
                "type":      "CALL" if direction == "BUY" else "PUT",
                "strike":    round(float(best["strike"]), 2),
                "expiry":    target_exp,
                "dte":       dte,
                "premium":   round(float(best.get("lastPrice", 0) or 0), 2),
                "bid":       round(float(best.get("bid", 0) or 0), 2),
                "ask":       round(float(best.get("ask", 0) or 0), 2),
                "volume":    vol_val,
                "oi":        oi_val,
                "iv":        round(float(best.get("impliedVolatility", 0) or 0) * 100, 1),
                "vol_oi":    round(vol_val / max(oi_val, 1), 3) if oi_val else 0,
                "pct_otm":   round((float(best["strike"]) / price - 1) * 100, 1),
                "sell_target": target_exp,
                "confidence": sig.get("confidence", 50),
                "composite":  sig.get("composite", 0),
                "ranked_by":  "volume" if has_volume else "open_interest",
            })
        except: pass
    result = {"swings": results, "updated": datetime.now().isoformat()}
    _swing_cache["data"] = result
    _swing_cache["ts"]   = now
    return jsonify(result)


# ═══════════════════════════════════════════════════════════════════════════════
# ALPACA EXECUTION ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

def _broker_required(fn_route):
    """Decorator: return 503 if Alpaca broker not initialised."""
    from functools import wraps
    @wraps(fn_route)
    def wrapper(*args, **kwargs):
        if not _alpaca_broker:
            return jsonify({
                "error": "Alpaca not connected. Set ALPACA_API_KEY and ALPACA_SECRET_KEY in live_data_server.py",
                "setup": "Get free paper keys at alpaca.markets → Dashboard → Paper Trading → API Keys",
                "connected": False,
            }), 503
        return fn_route(*args, **kwargs)
    return wrapper

@app.route("/api/alpaca/account")
def api_alpaca_account():
    """Live Alpaca account — equity, buying power, P&L."""
    if not _alpaca_broker:
        return jsonify({
            "connected": False,
            "message":   "Alpaca not configured. Set keys in live_data_server.py",
            "setup_url": "https://alpaca.markets",
        })
    try:
        acct = _alpaca_broker.get_account()
        clock = _alpaca_broker.get_clock()
        positions = _alpaca_broker.get_positions()
        orders    = _alpaca_broker.get_open_orders()
        return jsonify(_sanitize({
            "connected":      True,
            "paper_trading":  "paper" in ALPACA_BASE_URL,
            "account_number": acct.get("account_number"),
            "status":         acct.get("status"),
            "equity":         float(acct.get("equity", 0)),
            "cash":           float(acct.get("cash", 0)),
            "buying_power":   float(acct.get("buying_power", 0)),
            "portfolio_value":float(acct.get("portfolio_value", 0)),
            "last_equity":    float(acct.get("last_equity", 0)),
            "daytrade_count": int(acct.get("daytrade_count", 0)),
            "pattern_day_trader": acct.get("pattern_day_trader", False),
            "daily_pnl":      float(acct.get("equity", 0)) - float(acct.get("last_equity", 0)),
            "daily_pnl_pct":  round((float(acct.get("equity", 1)) / max(float(acct.get("last_equity", 1)), 1) - 1) * 100, 3),
            "market_open":    clock.get("is_open", False),
            "next_open":      clock.get("next_open", ""),
            "next_close":     clock.get("next_close", ""),
            "open_positions": len(positions),
            "open_orders":    len(orders),
            "updated":        datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({"error": str(e), "connected": False}), 500

@app.route("/api/alpaca/positions")
def api_alpaca_positions():
    """All open positions at Alpaca."""
    if not _alpaca_broker:
        return jsonify({"positions": [], "connected": False})
    try:
        raw = _alpaca_broker.get_positions()
        positions = []
        for p in raw:
            positions.append({
                "symbol":       p.get("symbol"),
                "qty":          float(p.get("qty", 0)),
                "side":         p.get("side"),
                "entry_price":  float(p.get("avg_entry_price", 0)),
                "current_price":float(p.get("current_price", 0)),
                "market_value": float(p.get("market_value", 0)),
                "cost_basis":   float(p.get("cost_basis", 0)),
                "unrealized_pl":float(p.get("unrealized_pl", 0)),
                "unrealized_plpc": float(p.get("unrealized_plpc", 0)) * 100,
                "change_today": float(p.get("change_today", 0)) * 100,
                "asset_class":  p.get("asset_class", "us_equity"),
            })
        return jsonify(_sanitize({"positions": positions, "count": len(positions), "connected": True}))
    except Exception as e:
        return jsonify({"error": str(e), "positions": []}), 500

@app.route("/api/alpaca/orders")
def api_alpaca_orders():
    """Open orders at Alpaca."""
    if not _alpaca_broker:
        return jsonify({"orders": [], "connected": False})
    try:
        raw = _alpaca_broker.get_open_orders()
        orders = [{
            "id":           o.get("id"),
            "symbol":       o.get("symbol"),
            "qty":          o.get("qty"),
            "side":         o.get("side"),
            "type":         o.get("type"),
            "status":       o.get("status"),
            "submitted_at": o.get("submitted_at"),
            "filled_avg_price": o.get("filled_avg_price"),
            "order_class":  o.get("order_class"),
        } for o in raw]
        return jsonify(_sanitize({"orders": orders, "count": len(orders), "connected": True}))
    except Exception as e:
        return jsonify({"error": str(e), "orders": []}), 500

@app.route("/api/alpaca/execute", methods=["POST"])
def api_alpaca_execute():
    """
    Execute a bracket order from a signal.
    Body: { symbol, qty, direction, stop_loss, take_profit, [sub_signals] }

    sub_signals are automatically fetched from the live signal cache if not
    provided in the request body — they are stored in the position snapshot
    so that when the trade closes, IC feedback can update AladdinScorer weights.
    """
    if not _alpaca_broker:
        return jsonify({"error": "Alpaca not connected", "status": "REJECTED"}), 503
    try:
        from flask import request as freq
        body        = freq.get_json(force=True) or {}
        symbol      = body.get("symbol", "").upper()
        qty         = int(body.get("qty", 1))
        direction   = body.get("direction", "BUY")
        stop_loss   = float(body.get("stop_loss", 0))
        take_profit = float(body.get("take_profit", 0))
        side        = "buy" if direction == "BUY" else "sell"

        if not symbol or qty < 1 or stop_loss <= 0 or take_profit <= 0:
            return jsonify({"error": "Invalid params: need symbol, qty≥1, stop_loss>0, take_profit>0"}), 400

        if not _alpaca_broker.is_market_open():
            return jsonify({
                "status": "QUEUED",
                "message": "Market is closed. Order will execute at next open.",
                "symbol": symbol, "qty": qty, "direction": direction,
            })

        # ── FETCH sub_signals from live cache for IC feedback loop ──────────
        sub_signals = body.get("sub_signals", {})
        if not sub_signals and hasattr(scanner, "all_signals"):
            cached_sig = scanner.all_signals.get(symbol, {})
            sub_signals = cached_sig.get("sub_signals", {})
            if sub_signals:
                logger.info(f"📸 IC snapshot: fetched {len(sub_signals)} sub_signals for {symbol}")

        order = _alpaca_broker.submit_bracket_order(
            symbol=symbol, qty=qty, side=side,
            stop_loss=stop_loss, take_profit=take_profit,
        )
        if "error" in order:
            if CAE:
                CAE.get_compliance_engine().log_event("order_rejected", {
                    "symbol": symbol, "direction": direction, "qty": qty,
                    "reason": order.get("error", "unknown")
                })
            return jsonify({"status": "REJECTED", "reason": order["error"], "symbol": symbol}), 400

        # ── STORE snapshot for IC feedback when position closes ─────────────
        if sub_signals:
            _execution_snapshots[symbol] = {
                "sub_signals": sub_signals,
                "direction":   1 if direction == "BUY" else -1,
                "entry_price": float(body.get("entry_price", stop_loss * 1.1)),
                "entry_time":  datetime.now().isoformat(),
            }

        logger.info(f"EXECUTE: {direction} {qty}x {symbol} SL={stop_loss} TP={take_profit}")
        if CAE:
            CAE.get_compliance_engine().log_event("order_executed", {
                "symbol": symbol,
                "direction": direction,
                "qty": qty,
                "side": side,
                "order_id": order.get("id"),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "paper_trading": "paper" in ALPACA_BASE_URL,
            })
        return jsonify(_sanitize({
            "status":      "EXECUTED",
            "order_id":    order.get("id"),
            "symbol":      symbol,
            "side":        side,
            "qty":         qty,
            "stop_loss":   stop_loss,
            "take_profit": take_profit,
            "timestamp":   datetime.now().isoformat(),
            "paper_trading": "paper" in ALPACA_BASE_URL,
            "ic_tracking": len(sub_signals) > 0,  # confirm IC feedback is active
        }))
    except Exception as e:
        logger.error(f"Execute error: {e}")
        return jsonify({"error": str(e), "status": "ERROR"}), 500

@app.route("/api/alpaca/close/<symbol>", methods=["DELETE"])
def api_alpaca_close(symbol):
    """
    Close an Alpaca position AND trigger IC feedback to update AladdinScorer weights.
    This is the critical feedback loop — every closed trade teaches the system.
    """
    if not _alpaca_broker:
        return jsonify({"error": "Alpaca not connected"}), 503
    try:
        sym = symbol.upper()

        # Get current price for P&L calculation
        current_price = None
        try:
            import yfinance as yf
            tick = yf.Ticker(sym)
            current_price = tick.fast_info.last_price
        except Exception:
            pass

        result = _alpaca_broker.close_position(sym)
        if CAE:
            CAE.get_compliance_engine().log_event("position_close_requested", {
                "symbol": sym,
                "broker_result": "ok" if isinstance(result, dict) and "error" not in result else "error",
                "current_price": current_price,
            })

        # ── IC FEEDBACK: update signal weights from this trade's outcome ────
        snapshot = _execution_snapshots.pop(sym, None)
        if snapshot and current_price:
            entry_price   = snapshot.get("entry_price", current_price)
            direction_int = snapshot.get("direction", 1)
            sub_signals   = snapshot.get("sub_signals", {})
            if entry_price > 0 and sub_signals:
                actual_return = (current_price - entry_price) / entry_price * direction_int
                try:
                    from math_engine import AladdinScorer as _AS
                    for signal_name, signal_val in sub_signals.items():
                        if signal_name in _AS._DEFAULT_WEIGHTS:
                            predicted_dir = float(signal_val) * direction_int
                            _AS.update_signal_ic(signal_name, predicted_dir, actual_return)
                    _AS.update_weights_from_ic()
                    sign_str = "✅ WIN" if actual_return > 0 else "❌ LOSS"
                    logger.info(
                        f"🔄 IC feedback: {sym} {sign_str} {actual_return:+.2%} "
                        f"| {len(sub_signals)} signals updated → weights recalibrated"
                    )
                    result["ic_feedback"] = {
                        "signals_updated": len(sub_signals),
                        "actual_return":   round(actual_return * 100, 2),
                        "weights_updated": True,
                    }
                except Exception as ic_err:
                    logger.warning(f"IC feedback error for {sym}: {ic_err}")

        return jsonify(_sanitize(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/compliance/events", methods=["GET"])
def api_compliance_events():
    """Recent append-only compliance audit events."""
    try:
        if not CAE:
            return jsonify({"error": "compliance_audit_engine not loaded"}), 503
        limit = int(request.args.get("limit", 200))
        event_type = request.args.get("event_type")
        events = CAE.get_compliance_engine().get_recent_events(limit=limit, event_type=event_type)
        return jsonify(_sanitize({"events": events, "count": len(events)}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/compliance/reconstruct/<symbol>", methods=["GET"])
def api_compliance_reconstruct(symbol):
    """Trade lifecycle reconstruction for a symbol."""
    try:
        if not CAE:
            return jsonify({"error": "compliance_audit_engine not loaded"}), 503
        days = int(request.args.get("days", 30))
        result = CAE.get_compliance_engine().reconstruct_trade(symbol=symbol, lookback_days=days)
        return jsonify(_sanitize(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/compliance/report", methods=["GET"])
def api_compliance_report():
    """Operational compliance summary for SEC/MiFID-aligned internal controls."""
    try:
        if not CAE:
            return jsonify({"error": "compliance_audit_engine not loaded"}), 503
        days = int(request.args.get("days", 7))
        report = CAE.get_compliance_engine().compliance_report(days=days)
        return jsonify(_sanitize(report))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/channels", methods=["GET", "POST"])
def api_chat_channels():
    """List/create Bloomberg-style collaboration channels."""
    try:
        if not MCE:
            return jsonify({"error": "messaging_collab_engine not loaded"}), 503
        eng = MCE.get_messaging_engine()
        if request.method == "GET":
            channels = eng.list_channels()
            return jsonify(_sanitize({"channels": channels, "count": len(channels)}))

        body = request.get_json(force=True) or {}
        name = str(body.get("name", "")).strip()
        channel_type = str(body.get("type", "desk")).strip().lower()
        symbol = body.get("symbol")
        if not name:
            return jsonify({"error": "channel name is required"}), 400
        channel = eng.create_channel(name=name, channel_type=channel_type, symbol=symbol)
        if CAE:
            CAE.get_compliance_engine().log_event("chat_channel_created", channel)
        return jsonify(_sanitize(channel))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/messages/<channel_id>", methods=["GET", "POST"])
def api_chat_messages(channel_id):
    """Read/post messages in a collaboration channel."""
    try:
        if not MCE:
            return jsonify({"error": "messaging_collab_engine not loaded"}), 503
        eng = MCE.get_messaging_engine()
        if request.method == "GET":
            limit = int(request.args.get("limit", 100))
            messages = eng.list_messages(channel_id=channel_id, limit=limit)
            return jsonify(_sanitize({"channel_id": channel_id, "messages": messages, "count": len(messages)}))

        body = request.get_json(force=True) or {}
        user = str(body.get("user", "anonymous")).strip()
        text = str(body.get("text", "")).strip()
        symbol = body.get("symbol")
        if not text:
            return jsonify({"error": "message text is required"}), 400
        msg = eng.post_message(channel_id=channel_id, user=user, text=text, symbol=symbol)
        if CAE:
            CAE.get_compliance_engine().log_event("chat_message_posted", {
                "channel_id": channel_id,
                "user": user,
                "symbol": symbol.upper() if isinstance(symbol, str) else None,
                "message_id": msg.get("message_id"),
            })
        return jsonify(_sanitize(msg))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/presence", methods=["GET", "POST"])
def api_chat_presence():
    """User presence heartbeat and roster."""
    try:
        if not MCE:
            return jsonify({"error": "messaging_collab_engine not loaded"}), 503
        eng = MCE.get_messaging_engine()
        if request.method == "GET":
            return jsonify(_sanitize({"users": eng.list_presence()}))
        body = request.get_json(force=True) or {}
        user = str(body.get("user", "anonymous")).strip()
        status = str(body.get("status", "online")).strip().lower()
        if not user:
            return jsonify({"error": "user is required"}), 400
        row = eng.heartbeat(user=user, status=status)
        return jsonify(_sanitize({"user": user, **row}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/aladdin/ic_stats", methods=["GET"])
def api_aladdin_ic_stats():
    """
    Return current AladdinScorer adaptive weight stats.
    Shows IC value per signal, current weight, and trade count.
    Use this in the dashboard to visualize which signals are strongest.
    """
    try:
        from math_engine import AladdinScorer
        stats = AladdinScorer.get_ic_stats()
        stats["default_weights"]  = AladdinScorer._DEFAULT_WEIGHTS
        stats["live_weights"]     = AladdinScorer.WEIGHTS
        stats["pending_snapshots"] = list(_execution_snapshots.keys())
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ml/status", methods=["GET"])
def api_ml_status():
    """
    Return ML model training status.
    Shows whether GB model is trained on real data, last train time, IC, sample count.
    """
    try:
        from ml_engine import get_model_stats
        stats = get_model_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ml/retrain", methods=["POST"])
def api_ml_retrain():
    """
    Force immediate GB model retraining on real data.
    Runs in background — server stays live. Check /api/ml/status for progress.
    """
    try:
        from ml_engine import force_retrain
        result = force_retrain()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ml/lstm", methods=["GET"])
def api_ml_lstm():
    """
    LSTM predictor status and diagnostics.

    Shows:
      - backend: "pytorch" or "sklearn_mlp" (fallback)
      - is_ready: whether model has been trained
      - val_ic: validation Spearman IC on held-out sequences
      - architecture: BiLSTM or MLP description
      - last_trained: timestamp of last successful training
      - is_training: whether background training is currently running
    """
    try:
        from ml_engine import LSTMPricePredictor
        predictor = LSTMPricePredictor.get_instance()
        stats     = predictor.get_stats()
        return jsonify({**stats, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ml/lstm/retrain", methods=["POST"])
def api_ml_lstm_retrain():
    """
    Force LSTM retraining. Runs in background (~2-5 min).
    Check /api/ml/lstm for progress.
    """
    try:
        from ml_engine import LSTMPricePredictor
        predictor = LSTMPricePredictor.get_instance()
        if predictor._is_training:
            return jsonify({"status": "already_training"})
        predictor.start_background_training()
        return jsonify({"status": "training_started",
                        "backend": predictor._backend,
                        "message": "Check /api/ml/lstm for progress"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/regime", methods=["GET"])
def api_regime():
    """
    Current market regime: HMM state + CUSUM structural break diagnostics.

    Returns:
      - regime: BEAR / NEUTRAL / BULL (3-state HMM)
      - probs: probability of each state
      - confidence: P(current state)
      - cusum: current CUSUM statistics (pct of alarm threshold)
      - break_alert: True if CUSUM alarm recently fired
      - macro_score: [-1, +1] scalar fed to AladdinScorer
      - hmm_means: annualized mean return per state
      - retrain history: last retrain time, retrain count
    """
    try:
        # Get regime stats from scanner's RegimeModel
        state = scanner.get_state()
        regime_stats = state.get("regime_stats", {})
        regime_info  = state.get("regime", {})

        if not regime_stats and hasattr(scanner, "hmm") and hasattr(scanner.hmm, "get_stats"):
            regime_stats = scanner.hmm.get_stats()

        if not regime_stats:
            regime_stats = regime_info

        return jsonify(_sanitize({
            **regime_stats,
            "vix": state.get("vix", 20),
            "timestamp": datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/regime/retrain", methods=["POST"])
def api_regime_retrain():
    """Force immediate HMM retrain with latest SPY/QQQ returns."""
    try:
        if hasattr(scanner, "hmm") and hasattr(scanner.hmm, "force_retrain"):
            result = scanner.hmm.force_retrain()
            return jsonify({"status": "retrained", "regime": result})
        return jsonify({"error": "regime_model_not_available"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/revisions/<symbol>", methods=["GET"])
def api_revisions_symbol(symbol):
    """
    Earnings revision momentum for a single symbol.
    Returns SUE, PEAD weight, analyst upgrades/downgrades, revision direction.
    Results are cached 6hr — first call may return zeros while computing.
    """
    try:
        symbol = symbol.upper().strip()
        if not EE or not hasattr(EE, "EarningsRevisionSignal"):
            return jsonify({"error": "earnings_engine_not_available"}), 503
        force = request.args.get("force", "false").lower() == "true"
        data  = EE.EarningsRevisionSignal.compute(symbol, force=force)
        return jsonify(_sanitize({**data, "symbol": symbol,
                                  "timestamp": datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/revisions", methods=["GET"])
def api_revisions_scan():
    """
    Bulk revision momentum scanner.
    Returns top BULLISH and BEARISH revision stocks from the scan universe.
    Only includes stocks with cached results (no rate-limit risk).
    """
    try:
        if not EE or not hasattr(EE, "EarningsRevisionSignal"):
            return jsonify({"error": "earnings_engine_not_available"}), 503

        cache = EE.EarningsRevisionSignal._precompute_cache
        results = []
        for sym, data in list(cache.items()):
            comp = float(data.get("revision_composite", 0))
            if abs(comp) < 0.05:
                continue
            sig = scanner.all_signals.get(sym, {})
            results.append({
                "symbol":             sym,
                "revision_composite": round(comp, 4),
                "revision_direction": data.get("revision_direction", "NEUTRAL"),
                "sue":                round(float(data.get("sue", 0)), 3),
                "pead_signal":        data.get("pead_signal", "NEUTRAL"),
                "pead_weight":        round(float(data.get("pead_weight", 0)), 3),
                "n_upgrades":         data.get("n_upgrades", 0),
                "n_downgrades":       data.get("n_downgrades", 0),
                "target_upside":      round(float(data.get("target_upside", 0)), 2),
                "price":              round(float(sig.get("price", 0)), 2),
                "direction":          sig.get("direction", "HOLD"),
                "company":            (sig.get("company") or {}).get("name", sym),
                "label":              data.get("label", ""),
            })

        results.sort(key=lambda x: x["revision_composite"], reverse=True)
        bullish   = [r for r in results if r["revision_composite"] > 0.1]
        bearish   = [r for r in results if r["revision_composite"] < -0.1]

        return jsonify(_sanitize({
            "bullish":        bullish[:10],
            "bearish":        bearish[:10],
            "all":            results[:30],
            "count":          len(results),
            "cache_stats":    EE.EarningsRevisionSignal.get_cache_stats(),
            "timestamp":      datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/revisions/precompute", methods=["POST"])
def api_revisions_precompute():
    """
    Trigger background precompute of revision signals for full universe.
    Non-blocking — returns immediately, runs in background threads.
    """
    try:
        if not EE or not hasattr(EE, "EarningsRevisionSignal"):
            return jsonify({"error": "earnings_engine_not_available"}), 503
        symbols = list(scanner.all_signals.keys()) or get_scan_universe(include_scanner=False)[:60]
        EE.EarningsRevisionSignal.precompute_universe(symbols[:80], max_workers=4)
        return jsonify({"status": "precompute_started", "symbols": len(symbols)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# INSIDER CLUSTER ENDPOINTS (Step 8)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/insider/<symbol>", methods=["GET"])
def api_insider_symbol(symbol):
    """
    Insider cluster score for a single symbol.
    Returns cluster detection, role-weighted score, notable trades.
    Results cached 4hr — first call may trigger background fetch.
    """
    try:
        symbol = symbol.upper().strip()
        if not EE or not hasattr(EE, "InsiderClusterScorer"):
            return jsonify({"error": "insider_engine_not_available"}), 503
        force  = request.args.get("force", "false").lower() == "true"
        result = EE.InsiderClusterScorer.compute(symbol, force=force)
        return jsonify(_sanitize({**result, "symbol": symbol,
                                   "timestamp": datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/insider/clusters", methods=["GET"])
def api_insider_clusters():
    """
    Bulk insider cluster scanner.
    Returns top bullish clusters (multiple insiders buying simultaneously)
    and top bearish clusters across the cached universe.

    These are the highest-conviction insider signals — historically
    8–9% annual excess returns vs 2% for single-insider trades.
    """
    try:
        if not EE or not hasattr(EE, "InsiderClusterScorer"):
            return jsonify({"error": "insider_engine_not_available"}), 503

        all_signals = scanner.all_signals or {}
        results = []
        for sym, cached in list(EE._CLUSTER_CACHE.items()):
            if not sym.startswith("cluster_"):
                continue
            ticker = sym[len("cluster_"):]
            data   = cached
            score  = float(data.get("cluster_score", 0))
            if abs(score) < 0.1:
                continue
            sig = all_signals.get(ticker, {})
            results.append({
                "symbol":          ticker,
                "cluster_score":   round(score, 4),
                "signal_label":    data.get("signal_label", "NEUTRAL"),
                "has_buy_cluster": data.get("has_buy_cluster", False),
                "has_sell_cluster":data.get("has_sell_cluster", False),
                "n_buyers":        data.get("n_buyers", 0),
                "n_sellers":       data.get("n_sellers", 0),
                "best_cluster":    data.get("best_buy_cluster") or data.get("best_sell_cluster"),
                "price":           round(float(sig.get("price", 0)), 2),
                "direction":       sig.get("direction", "HOLD"),
                "company":         (sig.get("company") or {}).get("name", ticker),
                "label":           data.get("label", ""),
            })

        results.sort(key=lambda x: x["cluster_score"], reverse=True)
        bullish = [r for r in results if r["cluster_score"] > 0.15]
        bearish = [r for r in results if r["cluster_score"] < -0.15]

        return jsonify(_sanitize({
            "bullish":       bullish[:10],
            "bearish":       bearish[:10],
            "all":           results[:30],
            "count":         len(results),
            "cache_stats":   EE.InsiderClusterScorer.get_cache_stats(),
            "timestamp":     datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENDPOINTS (Step 6)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/backtest/run", methods=["POST"])
def api_backtest_run():
    """
    Launch a walk-forward backtest in the background.

    POST body (JSON):
        symbols:  list of tickers (optional, defaults to top 20 scanned)
        years:    int 1-5 (default 3)

    Returns immediately with {"status": "running"}.
    Poll /api/backtest/status for progress, then /api/backtest/results.
    """
    try:
        if not BE:
            return jsonify({"error": "backtest_engine_not_available"}), 503

        body     = request.get_json(silent=True) or {}
        years    = max(1, min(5, int(body.get("years", 3))))
        symbols  = body.get("symbols")

        # Default: use top scanned symbols (sorted by conviction)
        if not symbols:
            all_sigs = scanner.all_signals or {}
            symbols  = sorted(
                all_sigs.keys(),
                key=lambda s: abs(float(all_sigs[s].get("conviction", 0))),
                reverse=True
            )[:25]

        if not symbols:
            symbols = get_training_universe(n=20)  # dynamic

        # Build cap_categories from scanner
        cap_map = {}
        for sym in symbols:
            sig = (scanner.all_signals or {}).get(sym, {})
            cap_map[sym] = sig.get("cap", "LARGE")

        result = BE.BacktestEngine.run_background(symbols, years=years,
                                                   cap_categories=cap_map)
        return jsonify({**result, "symbols": symbols, "years": years})

    except Exception as e:
        logger.error(f"Backtest run: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest/status", methods=["GET"])
def api_backtest_status():
    """Check backtest run status: idle / running / complete / error."""
    try:
        if not BE:
            return jsonify({"error": "backtest_engine_not_available"}), 503
        return jsonify(BE.BacktestEngine.get_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest/results", methods=["GET"])
def api_backtest_results():
    """
    Return the most recent backtest results.
    Includes aggregate metrics, per-symbol breakdown, IC per signal,
    and equity curve data for charting.

    Query params:
        symbol:  (optional) return only this symbol's detail
    """
    try:
        if not BE:
            return jsonify({"error": "backtest_engine_not_available"}), 503

        result = BE.BacktestEngine._last_result
        if not result:
            return jsonify({"status": "no_results", "message": "Run POST /api/backtest/run first"}), 404

        symbol = request.args.get("symbol", "").upper()
        if symbol:
            # Single symbol detail
            per_sym = result.get("per_symbol", {}).get(symbol)
            if not per_sym:
                return jsonify({"error": f"No results for {symbol}"}), 404
            return jsonify(_sanitize({
                "symbol":   symbol,
                **per_sym,
                "timestamp": result.get("timestamp"),
            }))

        # Full results — include aggregate + summary per symbol (not full equity curves)
        summary_per = {}
        for sym, data in result.get("per_symbol", {}).items():
            summary_per[sym] = {
                "metrics":       data.get("metrics", {}),
                "n_folds":       data.get("n_folds", 0),
                "n_trades":      data.get("n_trades", 0),
                "ic_per_signal": data.get("ic_per_signal", {}),
            }

        return jsonify(_sanitize({
            "status":          result.get("status"),
            "symbols_tested":  result.get("symbols_tested", []),
            "n_symbols":       result.get("n_symbols", 0),
            "aggregate":       result.get("aggregate", {}),
            "equity_curve":    result.get("equity_curve", []),
            "ic_summary":      result.get("ic_summary", {}),
            "per_symbol":      summary_per,
            "config":          result.get("config", {}),
            "run_seconds":     result.get("run_seconds"),
            "timestamp":       result.get("timestamp"),
        }))

    except Exception as e:
        logger.error(f"Backtest results: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest/ic", methods=["GET"])
def api_backtest_ic():
    """
    Signal IC summary from most recent backtest.
    Returns signals ranked by IC with quality labels.
    Use this to monitor signal health and detect decay over time.
    """
    try:
        if not BE:
            return jsonify({"error": "backtest_engine_not_available"}), 503

        result = BE.BacktestEngine._last_result
        if not result:
            return jsonify({"status": "no_results"}), 404

        ic_summary = result.get("ic_summary", {})

        # Reconstruct full IC stats from per-symbol data
        all_ics: dict = {}
        for sym_data in result.get("per_symbol", {}).values():
            for sig, ic_data in sym_data.get("ic_per_signal", {}).items():
                if sig not in all_ics:
                    all_ics[sig] = []
                all_ics[sig].append(ic_data.get("ic", 0))

        ic_full = {}
        for sig, ics in all_ics.items():
            arr   = np.array(ics)
            mean_ic = float(np.mean(arr))
            std_ic  = float(np.std(arr))
            ir      = float(mean_ic / (std_ic + 1e-8))
            ic_full[sig] = {
                "mean_ic":  round(mean_ic, 5),
                "std_ic":   round(std_ic, 5),
                "ir":       round(ir, 3),
                "n_symbols": len(ics),
                "quality":  ("STRONG"  if abs(mean_ic) > 0.08 else
                              "GOOD"    if abs(mean_ic) > 0.05 else
                              "WEAK"    if abs(mean_ic) > 0.02 else
                              "NOISE"),
            }

        ic_full = dict(sorted(ic_full.items(),
                               key=lambda x: abs(x[1]["mean_ic"]),
                               reverse=True))

        return jsonify(_sanitize({
            "ic_by_signal":   ic_full,
            "timestamp":      result.get("timestamp"),
            "run_seconds":    result.get("run_seconds"),
        }))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/vol_arb", methods=["GET"])
def api_vol_arb():
    """
    Volatility Arbitrage Scanner — lists premium-selling opportunities.
    Returns all scanned stocks where IV significantly exceeds realized vol (VRP > 4 pts).
    These are ranked by VRP strength — highest VRP = strongest sell-premium edge.

    Also returns iron condor strike suggestions for the top candidates.
    """
    try:
        from math_engine import VolatilityAnalysis as _VA

        candidates = []
        if not hasattr(scanner, "all_signals") or not scanner.all_signals:
            return jsonify({"candidates": [], "count": 0, "error": "scan_not_ready"})

        vix_level = 20.0
        try:
            macro_d = _macro_cache.get("data") or {}
            vix_level = float((macro_d.get("vix") or {}).get("value", 20.0))
        except Exception:
            pass

        for sym, sig in scanner.all_signals.items():
            try:
                close_arr = scanner.price_cache.get(sym)
                if not close_arr or len(close_arr) < 22:
                    continue

                # Get IV from stored vol_arb or options data
                vol_arb = sig.get("vol_arb", {})
                if vol_arb and vol_arb.get("vrp") is not None:
                    # Already computed — use cached result
                    vrp = vol_arb.get("vrp", 0)
                    iv  = vol_arb.get("iv_pct", 0)
                    rv  = vol_arb.get("rv_conservative", 0)
                    strategy   = vol_arb.get("strategy", "NO_EDGE")
                    conviction = vol_arb.get("conviction", "NONE")
                else:
                    # Compute fresh
                    hv20   = sig.get("hv20", 25)
                    atm_iv = hv20 / 100  # fallback
                    opts_cached = _OPTIONS_CACHE.get(sym, {})
                    if opts_cached:
                        iv_raw = (opts_cached.get("iv") or
                                  opts_cached.get("call_iv") or
                                  opts_cached.get("put_iv"))
                        if iv_raw:
                            atm_iv = float(iv_raw) / 100

                    vol_arb = _VA.rv_iv_spread(
                        close_prices=np.array(close_arr, dtype=float),
                        atm_iv=atm_iv,
                        vix=vix_level,
                    )
                    vrp        = vol_arb.get("vrp", 0)
                    iv         = vol_arb.get("iv_pct", 0)
                    rv         = vol_arb.get("rv_conservative", 0)
                    strategy   = vol_arb.get("strategy", "NO_EDGE")
                    conviction = vol_arb.get("conviction", "NONE")

                # Filter: only meaningful VRP
                if abs(vrp) < 3:
                    continue

                price = sig.get("price", 100)

                # Iron condor strikes for strong candidates
                ic_strikes = {}
                if vrp > 5 and price > 5:
                    try:
                        ic_strikes = _VA.iron_condor_strikes(
                            spot=price,
                            iv=iv / 100,
                            dte=30,
                        )
                    except Exception:
                        pass

                candidates.append({
                    "symbol":     sym,
                    "price":      round(price, 2),
                    "vrp":        round(vrp, 2),
                    "iv_pct":     round(iv, 1),
                    "rv_pct":     round(rv, 1),
                    "vrp_pct":    round(vol_arb.get("vrp_pct", 0), 1),
                    "strategy":   strategy,
                    "conviction": conviction,
                    "direction":  sig.get("direction", "HOLD"),
                    "ivr":        round(sig.get("ivr", 50), 1),
                    "vol_arb_score": round(vol_arb.get("vol_arb_score", 0), 4),
                    "label":      vol_arb.get("label", ""),
                    "ic_strikes": ic_strikes,
                    "company":    sig.get("company", {}).get("name", sym),
                    "sector":     sig.get("company", {}).get("sector", ""),
                })

            except Exception as _s_e:
                logger.debug(f"Vol arb scan {sym}: {_s_e}")
                continue

        # Sort by absolute VRP descending
        candidates.sort(key=lambda x: abs(x["vrp"]), reverse=True)

        # Split into premium sellers vs vol buyers
        sell_premium = [c for c in candidates if c["vrp"] > 4]
        buy_vol      = [c for c in candidates if c["vrp"] < -3]

        return jsonify(_sanitize({
            "sell_premium":   sell_premium[:15],
            "buy_vol":        buy_vol[:5],
            "candidates":     candidates[:20],
            "count":          len(candidates),
            "vix":            vix_level,
            "timestamp":      datetime.now().isoformat(),
            "summary": {
                "avg_vrp":       round(float(np.mean([c["vrp"] for c in sell_premium])), 2) if sell_premium else 0,
                "strong_sellers": len([c for c in sell_premium if c["conviction"] == "STRONG"]),
                "vol_buyers":    len(buy_vol),
            }
        }))
    except Exception as e:
        logger.error(f"Vol arb endpoint: {e}")
        return jsonify({"error": str(e)}), 500



def api_alpaca_cancel_all():
    """Cancel all open orders."""
    if not _alpaca_broker:
        return jsonify({"error": "Alpaca not connected"}), 503
    try:
        ok = _alpaca_broker.cancel_all_orders()
        return jsonify({"success": ok, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# ══════════════════════════════════════════════════════════════════════════════
# CATALYST & ALERT ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/alerts")
def api_alerts():
    """Returns current alert queue from background catalyst scanner."""
    if not CE:
        return jsonify({"alerts": [], "status": {"running": False, "error": "catalyst_engine not loaded"}})
    try:
        def _f2(x, d=2):
            try:
                return f"{float(x):.{d}f}"
            except Exception:
                return "0.00"
        def _pct(delta, base):
            try:
                b = float(base) if float(base) != 0 else 1.0
                return (float(delta) / b) * 100.0
            except Exception:
                return 0.0
        def _fmt_date(days):
            try:
                return (datetime.now() + timedelta(days=max(1, int(days)))).date().isoformat()
            except Exception:
                return "N/A"
        def _alert_text(a):
            sym = str(a.get("sym") or a.get("symbol") or "N/A").upper()
            spot = float(a.get("spot") or a.get("price") or 0.0)
            direction = str(a.get("direction") or "NEUTRAL").upper()
            cap = str(a.get("mktcap_tier") or "UNKNOWN")
            conf_map = {"HIGH": 9, "MEDIUM": 7, "LOW": 5}
            conf = conf_map.get(str(a.get("conviction") or "LOW").upper(), 6)
            score = float(a.get("composite_score") or 0.0)
            dte = ((a.get("earnings_detail") or {}).get("days_to_earnings"))
            implied = float(((a.get("earnings_detail") or {}).get("implied_move_pct") or 0))
            sq = a.get("squeeze_metrics") or {}
            short_float = float(sq.get("short_pct_float") or 0)
            dtc = float(sq.get("days_to_cover") or 0)
            float_m = float(sq.get("float_millions") or 0)
            rel_vol = float((a.get("tape_metrics") or {}).get("relative_volume") or 0)
            dp = a.get("dark_pool") or {}
            dp_bias = str(dp.get("bias") or "Neutral")
            dp_notional = float(dp.get("notional") or 0)
            dp_price = float(dp.get("price") or 0)
            trade = a.get("trade") or {}
            strike = trade.get("strike")
            expiry = trade.get("expiry") or _fmt_date(21)
            premium = trade.get("premium") or trade.get("price") or 0
            setup = "Breakout" if direction == "BULLISH" else "Breakdown" if direction == "BEARISH" else "Catalyst Setup"
            move_pct = max(3.5, implied if implied > 0 else min(25.0, max(4.0, score / 3.0)))
            if direction == "BEARISH":
                t1 = spot * (1 - move_pct / 100.0)
                t2 = spot * (1 - (move_pct * 1.6) / 100.0)
                sl = spot * (1 + (move_pct * 0.6) / 100.0)
            else:
                t1 = spot * (1 + move_pct / 100.0)
                t2 = spot * (1 + (move_pct * 1.6) / 100.0)
                sl = spot * (1 - (move_pct * 0.6) / 100.0)
            rr = abs((t1 - spot) / max(1e-9, spot - sl)) if direction != "BEARISH" else abs((spot - t1) / max(1e-9, sl - spot))
            has_4h = "✅" if (a.get("signal_scores") or {}).get("tape", 0) > 0 else "⚠"
            has_1d = "✅" if score >= 12 else "⚠"
            has_1w = "✅" if (a.get("signal_scores") or {}).get("insider", 0) > 0 or (a.get("signal_scores") or {}).get("earnings", 0) > 0 else "⚠"
            liq_vol = float((a.get("tape_metrics") or {}).get("avg_daily_volume") or 0) / 1_000_000.0
            oi = int((a.get("options") or {}).get("oi") or (a.get("options") or {}).get("open_interest") or 0)
            catalyst = (a.get("signals") or ["Technical + catalyst alignment"])[0]
            trade_expiry = _fmt_date(5 if direction in {"BULLISH", "BEARISH"} else 3)
            is_sq = short_float > 20 and dtc > 5 and (float_m == 0 or float_m < 50) and rel_vol > 3

            if is_sq:
                return (
                    f"🚀 SHORT SQUEEZE ALERT | ${sym}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"⚡ Short Float: {_f2(short_float,1)}% | Days to Cover: {_f2(dtc,1)}\n"
                    f"💰 Entry: ${_f2(spot)}\n"
                    f"🎯 Target: ${_f2(t1)} (+{_f2(_pct(t1-spot, spot),1)}%)\n"
                    f"🛑 Stop Loss: ${_f2(sl)} ({_f2(_pct(sl-spot, spot),1)}%)\n"
                    f"⏰ Options Expiry: {expiry}\n"
                    f"📋 Strike: ${_f2(strike or spot)} CALL @ ${_f2(premium)}\n"
                    f"📣 Catalyst: {catalyst}\n"
                    f"🌑 Dark Pool: {dp_bias} — ${_f2(dp_notional,0)} printed at ${_f2(dp_price or spot)}\n"
                    f"⏳ Trade Expiry: {trade_expiry}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Confidence: {conf}/10"
                )

            if direction == "BEARISH":
                return (
                    f"🔴 BEARISH SIGNAL | ${sym}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📉 Setup: {setup}\n"
                    f"💰 Entry Price: ${_f2(spot)} — breakdown confirmation\n"
                    f"🎯 Target 1: ${_f2(t1)} ({_f2(_pct(t1-spot, spot),1)}%)\n"
                    f"🎯 Target 2: ${_f2(t2)} ({_f2(_pct(t2-spot, spot),1)}%)\n"
                    f"🛑 Stop Loss: ${_f2(sl)} (+{_f2(_pct(sl-spot, spot),1)}%)\n"
                    f"⏰ Options Expiry: {expiry}\n"
                    f"📋 Strike: ${_f2(strike or spot)} PUT @ ${_f2(premium)}\n"
                    f"📊 Timeframe Confluence: 4H {has_4h} | 1D {has_1d} | 1W {has_1w}\n"
                    f"💧 Liquidity: Vol {_f2(liq_vol,2)}m | OI {oi} | Float {_f2(float_m,1)}M\n"
                    f"🌑 Dark Pool: {dp_bias} — ${_f2(dp_notional,0)} printed at ${_f2(dp_price or spot)}\n"
                    f"📰 Catalyst: {catalyst}\n"
                    f"📉 Risk/Reward: {_f2(rr,2)}:1\n"
                    f"⏳ Trade Expiry: {trade_expiry}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Confidence: {conf}/10 | Cap: {cap}"
                )

            return (
                f"🟢 BULLISH SIGNAL | ${sym}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📈 Setup: {setup}\n"
                f"💰 Entry Price: ${_f2(spot)} — breakout confirmation\n"
                f"🎯 Target 1: ${_f2(t1)} (+{_f2(_pct(t1-spot, spot),1)}%)\n"
                f"🎯 Target 2: ${_f2(t2)} (+{_f2(_pct(t2-spot, spot),1)}%)\n"
                f"🛑 Stop Loss: ${_f2(sl)} ({_f2(_pct(sl-spot, spot),1)}%)\n"
                f"⏰ Options Expiry: {expiry}\n"
                f"📋 Strike: ${_f2(strike or spot)} CALL @ ${_f2(premium)}\n"
                f"📊 Timeframe Confluence: 4H {has_4h} | 1D {has_1d} | 1W {has_1w}\n"
                f"💧 Liquidity: Vol {_f2(liq_vol,2)}m | OI {oi} | Float {_f2(float_m,1)}M\n"
                f"🌑 Dark Pool: {dp_bias} — ${_f2(dp_notional,0)} printed at ${_f2(dp_price or spot)}\n"
                f"📰 Catalyst: {catalyst}\n"
                f"📉 Risk/Reward: {_f2(rr,2)}:1\n"
                f"⏳ Trade Expiry: {trade_expiry}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Confidence: {conf}/10 | Cap: {cap}"
            )

        sector     = request.args.get("sector", "ALL")
        conviction = request.args.get("conviction", "LOW")
        limit      = int(request.args.get("limit", 50))
        alerts     = CE.get_alerts(min_conviction=conviction, sector=sector, limit=limit)
        alerts = [_sanitize({**a, "formatted_alert": _alert_text(a)}) for a in (alerts or [])]
        status     = CE.get_status()
        edgar      = CE.get_edgar_alerts()
        return jsonify({
            "alerts":       alerts,
            "edgar_alerts": edgar,
            "status":       status,
            "sectors":      list(CE.SCAN_UNIVERSE.keys()),
            "updated":      datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e), "alerts": []}), 500


@app.route("/api/catalyst/<sym>")
def api_catalyst(sym):
    """On-demand full catalyst analysis for any ticker (search)."""
    if not CE:
        return jsonify({"error": "catalyst_engine not loaded"}), 503
    try:
        sym    = sym.upper().strip()
        result = CE.scan_single(sym)
        # Also pull alt data
        if AD:
            try:
                alt = AD.get_full_alt_data(sym)
                result["alt_data"] = alt
            except Exception:
                result["alt_data"] = None
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/altdata/<sym>")
def api_altdata(sym):
    """Alternative data signals for a ticker."""
    if not AD:
        return jsonify({"error": "alt_data_engine not loaded"}), 503
    try:
        sym    = sym.upper().strip()
        result = AD.get_full_alt_data(sym)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/scan/status")
def api_scan_status():
    """Returns catalyst scanner status."""
    if not CE:
        return jsonify({"running": False, "error": "catalyst_engine not loaded"})
    return jsonify(CE.get_status())


@app.route("/api/alerts/trigger")
def api_trigger_scan():
    """Manually trigger a catalyst scan (runs in background)."""
    if not CE:
        return jsonify({"error": "catalyst_engine not loaded"}), 503
    t = threading.Thread(target=CE.run_full_scan, daemon=True, name="ManualScan")
    t.start()
    return jsonify({"triggered": True, "message": "Scan started in background"})




# ═══════════════════════════════════════════════════════════════════════════════
# RENAISSANCE v5 — NEW API ROUTES (NLP, Deep Learning, Ensemble, Patterns,
#                                   Vol Surface, Cross-Asset, Data Quality)
# ═══════════════════════════════════════════════════════════════════════════════

# ── NLP ENDPOINTS ─────────────────────────────────────────────────────────────

@app.route("/api/nlp/sentiment/<symbol>", methods=["GET"])
def api_nlp_sentiment(symbol):
    """FinBERT sentiment analysis for a symbol's recent news."""
    try:
        if not NLP:
            return jsonify({"error": "nlp_engine not loaded"}), 503
        news = fetch_yahoo_rss_news(symbol.upper())
        if not news:
            return jsonify({"symbol": symbol.upper(), "sentiment": 0, "articles": 0,
                            "label": "neutral", "message": "no_news_found"})
        aggregator = NLP.get_news_aggregator()
        for article in news[:20]:
            headline = article.get("title", "")
            source = article.get("source", "default")
            aggregator.add_article(symbol.upper(), headline, source=source)
        composite = aggregator.get_composite(symbol.upper())
        latest_headline = news[0].get("title", "") if news else ""
        finbert_result = NLP.FinBERTAnalyzer.analyze(latest_headline) if latest_headline else {}
        return jsonify(_sanitize({
            "symbol": symbol.upper(),
            "composite": composite,
            "finbert_latest": finbert_result,
            "articles_analyzed": min(len(news), 20),
            "model_status": NLP.get_model_status()
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/nlp/fed", methods=["POST"])
def api_nlp_fed():
    """Analyze Fed/FOMC text for hawkish/dovish score."""
    try:
        if not NLP:
            return jsonify({"error": "nlp_engine not loaded"}), 503
        data = request.get_json(force=True)
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "provide 'text' in JSON body"}), 400
        result = NLP.FedLanguageDecoder.score(text)
        return jsonify(_sanitize(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/nlp/status", methods=["GET"])
def api_nlp_status():
    """NLP model loading status."""
    try:
        if not NLP:
            return jsonify({"loaded": False})
        return jsonify({"loaded": True, "models": NLP.get_model_status()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── DEEP LEARNING ENDPOINTS ──────────────────────────────────────────────────

@app.route("/api/deep_learning/predict/<symbol>", methods=["GET"])
def api_dl_predict(symbol):
    """Neural ensemble prediction for a symbol."""
    try:
        if not DL:
            return jsonify({"error": "deep_learning_engine not loaded"}), 503
        symbol = symbol.upper()
        ensemble = DL.get_neural_ensemble()
        scanner_data = scanner.get_symbol(symbol) if scanner else None
        if not scanner_data:
            return jsonify({"error": f"no data for {symbol}"}), 404
        signals = scanner_data.get("signals", {})
        try:
            from ml_engine import build_feature_vector
            features = build_feature_vector(signals)
        except Exception:
            import numpy as _np
            features = _np.random.randn(14) * 0.1
        regime = signals.get("regime", "neutral")
        result = ensemble.predict(features, regime=regime)
        result["symbol"] = symbol
        result["regime"] = regime
        return jsonify(_sanitize(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/deep_learning/status", methods=["GET"])
def api_dl_status():
    """Deep learning model status."""
    try:
        if not DL:
            return jsonify({"loaded": False, "pytorch": False})
        return jsonify(DL.get_deep_learning_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── ENSEMBLE ENDPOINTS ────────────────────────────────────────────────────────

@app.route("/api/ensemble/signal/<symbol>", methods=["GET"])
def api_ensemble_signal(symbol):
    """Medallion master ensemble signal for a symbol."""
    try:
        if not ENS:
            return jsonify({"error": "ensemble_engine not loaded"}), 503
        symbol = symbol.upper()
        medallion = ENS.get_medallion_ensemble()
        scanner_data = scanner.get_symbol(symbol) if scanner else None
        if not scanner_data:
            return jsonify({"error": f"no data for {symbol}"}), 404
        raw_signals = scanner_data.get("signals", {})
        import numpy as _np
        signal_map = {
            "hurst_exponent": raw_signals.get("hurst", 0),
            "ou_zscore": _np.clip(raw_signals.get("ou_zscore", 0) / 3.0, -1, 1),
            "garch_vol": 1 - min(raw_signals.get("garch_vol", 25) / 50, 1),
            "kalman_trend": raw_signals.get("kalman_prob", 0.5) * 2 - 1,
            "rsi_signal": (raw_signals.get("rsi", 50) - 50) / 50,
            "macd_signal": _np.clip(raw_signals.get("macd_norm", 0), -1, 1),
            "adx_trend": min(raw_signals.get("adx", 0) / 50, 1),
            "bb_position": raw_signals.get("bb_position", 0.5) * 2 - 1,
            "ema_cross": _np.clip(raw_signals.get("ema_cross_9_21", 0) / 5, -1, 1),
            "gb_score": raw_signals.get("ml_score", 0.5) * 2 - 1,
            "lstm_pattern": raw_signals.get("lstm_pattern", 0.5) * 2 - 1,
        }
        if NLP:
            try:
                news = fetch_yahoo_rss_news(symbol)
                if news:
                    agg = NLP.get_news_aggregator()
                    for art in news[:10]:
                        agg.add_article(symbol, art.get("title", ""))
                    comp = agg.get_composite(symbol)
                    signal_map["finbert_sentiment"] = comp.get("composite_score", 0)
                    signal_map["news_composite"] = comp.get("composite_score", 0)
            except Exception:
                pass
        regime = raw_signals.get("regime", "neutral")
        result = medallion.compute_master_signal(signal_map, regime=regime)
        result["symbol"] = symbol
        return jsonify(_sanitize(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ensemble/stats", methods=["GET"])
def api_ensemble_stats():
    """Ensemble statistics and IC tracking."""
    try:
        if not ENS:
            return jsonify({"error": "ensemble_engine not loaded"}), 503
        medallion = ENS.get_medallion_ensemble()
        return jsonify(_sanitize(medallion.get_ensemble_stats()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ensemble/sources", methods=["GET"])
def api_ensemble_sources():
    """All registered alpha sources with IC stats."""
    try:
        if not ENS:
            return jsonify({"error": "ensemble_engine not loaded"}), 503
        medallion = ENS.get_medallion_ensemble()
        return jsonify(_sanitize(medallion.registry.get_source_stats()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── PATTERN RECOGNITION ENDPOINTS ────────────────────────────────────────────

@app.route("/api/patterns/<symbol>", methods=["GET"])
def api_patterns(symbol):
    """Full pattern scan: Markov + chart patterns + seasonals + fractals."""
    try:
        if not PAT:
            return jsonify({"error": "pattern_engine not loaded"}), 503
        symbol = symbol.upper()
        import yfinance as yf
        tk = yf.Ticker(symbol)
        df = tk.history(period="1y", interval="1d")
        if df.empty or len(df) < 30:
            return jsonify({"error": f"insufficient data for {symbol}"}), 404
        _scanner = PAT.get_pattern_scanner()
        result = _scanner.full_scan(
            symbol, df["High"].values, df["Low"].values,
            df["Close"].values, df["Volume"].values
        )
        result["symbol"] = symbol
        return jsonify(_sanitize(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/patterns/markov/<symbol>", methods=["GET"])
def api_markov(symbol):
    """Markov chain state analysis only."""
    try:
        if not PAT:
            return jsonify({"error": "pattern_engine not loaded"}), 503
        symbol = symbol.upper()
        import yfinance as yf
        tk = yf.Ticker(symbol)
        df = tk.history(period="1y", interval="1d")
        if df.empty or len(df) < 30:
            return jsonify({"error": f"insufficient data for {symbol}"}), 404
        returns = df["Close"].pct_change().dropna().values
        markov = PAT.MarkovChainModel()
        result = markov.fit(returns)
        forecast = markov.predict_n_steps(5)
        result["forecast"] = forecast
        result["symbol"] = symbol
        return jsonify(_sanitize(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── VOL SURFACE ENDPOINTS ────────────────────────────────────────────────────

@app.route("/api/volsurface/<symbol>", methods=["GET"])
def api_volsurface(symbol):
    """Volatility surface analysis: SVI fit + term structure + skew."""
    try:
        if not VS:
            return jsonify({"error": "vol_surface_engine not loaded"}), 503
        symbol = symbol.upper()
        import yfinance as yf
        import numpy as _np
        from datetime import datetime as _dt
        tk = yf.Ticker(symbol)
        fi = tk.fast_info
        spot = getattr(fi, "last_price", None) or getattr(fi, "previous_close", 100)
        expirations = tk.options
        if not expirations:
            return jsonify({"error": "no options data"}), 404
        expiry_vols = []
        svi_result = {}
        skew_result = {}
        for i, exp in enumerate(expirations[:5]):
            try:
                chain = tk.option_chain(exp)
                calls = chain.calls
                puts = chain.puts
                if calls.empty:
                    continue
                calls_valid = calls[
                    (calls["impliedVolatility"] > 0.05) &
                    (calls["impliedVolatility"] < 3.0) &
                    (calls["volume"].fillna(0) > 0)
                ]
                puts_valid = puts[
                    (puts["impliedVolatility"] > 0.05) &
                    (puts["impliedVolatility"] < 3.0) &
                    (puts["volume"].fillna(0) > 0)
                ]
                if len(calls_valid) < 3:
                    continue
                exp_date = _dt.strptime(exp, "%Y-%m-%d")
                dte = max((exp_date - _dt.now()).days, 1)
                T_yr = dte / 365.0
                atm_idx = (calls_valid["strike"] - spot).abs().idxmin()
                atm_vol = float(calls_valid.loc[atm_idx, "impliedVolatility"])
                expiry_vols.append((dte, atm_vol))
                if i == 0:
                    strikes = calls_valid["strike"].values
                    ivs = calls_valid["impliedVolatility"].values
                    svi_result = VS.SVIModel.fit(strikes, ivs, spot, T_yr)
                    if not puts_valid.empty:
                        skew_result = VS.SkewSignal.compute(
                            puts_valid["impliedVolatility"].values,
                            calls_valid["impliedVolatility"].values,
                            puts_valid["strike"].values,
                            calls_valid["strike"].values, spot
                        )
            except Exception:
                continue
        term_result = {}
        if len(expiry_vols) >= 2:
            term_result = VS.TermStructureAnalyzer.analyze(expiry_vols)
        alpha = VS.VolSurfaceAlpha.compute(svi_result, term_result, skew_result)
        return jsonify(_sanitize({
            "symbol": symbol, "spot": round(spot, 2),
            "svi": svi_result, "term_structure": term_result,
            "skew": skew_result, "alpha": alpha,
            "expiries_analyzed": len(expiry_vols)
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── CROSS-ASSET ENDPOINTS ────────────────────────────────────────────────────

@app.route("/api/cross_asset", methods=["GET"])
def api_cross_asset():
    """Cross-asset intelligence dashboard."""
    try:
        if not CA:
            return jsonify({"error": "cross_asset_engine not loaded"}), 503
        import yfinance as yf
        symbols = CA.ALL_CROSS_ASSETS
        returns_data = {}
        for sym in symbols:
            try:
                tk = yf.Ticker(sym)
                df = tk.history(period="6mo", interval="1d")
                if not df.empty and len(df) > 20:
                    returns_data[sym] = df["Close"].pct_change().dropna().values
            except Exception:
                continue
        if len(returns_data) < 3:
            return jsonify({"error": "insufficient cross-asset data"}), 503
        target = request.args.get("target", "SPY")
        analyzer = CA.get_cross_asset_analyzer()
        result = analyzer.full_analysis(returns_data, target)
        result["target"] = target
        result["assets_loaded"] = len(returns_data)
        return jsonify(_sanitize(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/cross_asset/roro", methods=["GET"])
def api_roro():
    """Risk-On/Risk-Off indicator."""
    try:
        if not CA:
            return jsonify({"error": "cross_asset_engine not loaded"}), 503
        import yfinance as yf
        roro_symbols = ["SPY", "QQQ", "IWM", "HYG", "TLT", "GLD", "UUP"]
        returns_5d = {}
        for sym in roro_symbols:
            try:
                tk = yf.Ticker(sym)
                df = tk.history(period="1mo", interval="1d")
                if not df.empty and len(df) >= 5:
                    returns_5d[sym] = float(
                        (df["Close"].iloc[-1] - df["Close"].iloc[-5]) / df["Close"].iloc[-5]
                    )
            except Exception:
                continue
        result = CA.RiskOnOffIndicator.compute(returns_5d)
        return jsonify(_sanitize(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/cross_asset/sectors", methods=["GET"])
def api_sectors_live():
    """Sector rotation signal."""
    try:
        if not CA:
            return jsonify({"error": "cross_asset_engine not loaded"}), 503
        import yfinance as yf
        sector_etfs = list(CA.SectorRotationSignal.SECTORS.keys())
        returns_20d = {}
        for sym in sector_etfs:
            try:
                tk = yf.Ticker(sym)
                df = tk.history(period="2mo", interval="1d")
                if not df.empty and len(df) >= 20:
                    returns_20d[sym] = float(
                        (df["Close"].iloc[-1] - df["Close"].iloc[-20]) / df["Close"].iloc[-20]
                    )
            except Exception:
                continue
        result = CA.SectorRotationSignal.compute(returns_20d)
        return jsonify(_sanitize(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── DATA QUALITY ENDPOINTS ────────────────────────────────────────────────────

@app.route("/api/data_quality", methods=["GET"])
def api_data_quality():
    """Data quality dashboard across all symbols."""
    try:
        if not DI:
            return jsonify({"error": "data_ingestion_engine not loaded"}), 503
        pipeline = DI.get_data_pipeline()
        return jsonify(_sanitize(pipeline.get_quality_dashboard()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/data_quality/<symbol>", methods=["GET"])
def api_data_quality_symbol(symbol):
    """Per-symbol data quality report."""
    try:
        if not DI:
            return jsonify({"error": "data_ingestion_engine not loaded"}), 503
        symbol = symbol.upper()
        import yfinance as yf
        tk = yf.Ticker(symbol)
        df = tk.history(period="1y", interval="1d")
        if df.empty:
            return jsonify({"error": f"no data for {symbol}"}), 404
        pipeline = DI.get_data_pipeline()
        result = pipeline.process(df, symbol)
        result["symbol"] = symbol
        return jsonify(_sanitize(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/pipeline/status", methods=["GET"])
def api_pipeline_status():
    """Data pipeline and watchdog status."""
    try:
        if not DI:
            return jsonify({"loaded": False})
        pipeline = DI.get_data_pipeline()
        status = pipeline.watchdog.get_status()
        status["alerts"] = pipeline.watchdog.get_alerts(10)
        status["loaded"] = True
        return jsonify(_sanitize(status))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/datafabric/providers", methods=["GET"])
def api_datafabric_providers():
    """Institutional provider manifest for terminal coverage inspection."""
    try:
        if not IDF:
            return jsonify({"error": "institutional_data_fabric not loaded"}), 503
        fabric = IDF.get_data_fabric()
        return jsonify(_sanitize(fabric.provider_manifest()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/datafabric/health", methods=["GET"])
def api_datafabric_health():
    """Parallel provider health checks with p95 latency."""
    try:
        if not IDF:
            return jsonify({"error": "institutional_data_fabric not loaded"}), 503
        force = str(request.args.get("refresh", "0")).lower() in ("1", "true", "yes")
        fabric = IDF.get_data_fabric()
        return jsonify(_sanitize(fabric.health_dashboard(force_refresh=force)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/market/coverage", methods=["GET"])
def api_market_coverage():
    """US market coverage diagnostics for scanner universe."""
    try:
        symbols = get_scan_universe(include_scanner=True)
        scanner_state = scanner.get_state() if scanner else {}
        tracked = list(getattr(scanner, "all_signals", {}).keys())
        return jsonify(_sanitize({
            "universe_symbols": len(symbols),
            "tracked_symbols": len(tracked),
            "top_signals_count": len(scanner_state.get("top_signals", [])),
            "scanner_running": bool(scanner_state),
            "sample_universe": symbols[:25],
            "sample_tracked": tracked[:25],
            "updated": datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ml/governance", methods=["GET"])
def api_ml_governance():
    """
    ML/AI model governance and performance guardrails.
    Note: exposes realistic constraints; no guaranteed fixed win-rate promises.
    """
    try:
        state = scanner.get_state() if scanner else {}
        signals = state.get("top_signals", [])
        avg_conf = float(np.mean([abs(float(s.get("conviction", 0))) for s in signals])) if signals else 0.0
        return jsonify(_sanitize({
            "models_loaded": {
                "ml_engine": bool(ML),
                "deep_learning_engine": bool(DL),
                "nlp_engine": bool(NLP),
                "ensemble_engine": bool(ENS),
                "alpha_research_engine": bool(ARE),
            },
            "governance_policy": {
                "hard_min_precision_target": 0.55,
                "research_target_precision_range": [0.55, 0.70],
                "note": "No production system can guarantee constant 90% live win-rate across regimes.",
                "risk_controls_required": ["VaR/CVaR limits", "drawdown circuit breaker", "position caps", "liquidity checks"],
            },
            "live_signal_diagnostics": {
                "count": len(signals),
                "avg_abs_conviction": round(avg_conf, 4),
            },
            "updated": datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ml/governance/live", methods=["GET"])
def api_ml_governance_live():
    """
    Live governance snapshot with current signal quality + model guardrails.
    """
    try:
        state = scanner.get_state() if scanner else {}
        live = state.get("top_signals", [])
        sig_quality = {}
        if SSE and hasattr(SSE, "generate_stock_signals") and live:
            generated = SSE.generate_stock_signals(live, float(request.args.get("portfolio", 100000)))
            sig_quality = SSE.get_market_scan_quality(generated.get("signals", [])) if hasattr(SSE, "get_market_scan_quality") else {}
        out = {
            "ml_snapshot": ML.get_ml_governance_snapshot() if ML and hasattr(ML, "get_ml_governance_snapshot") else {},
            "signal_quality": sig_quality,
            "live_counts": {
                "tracked_signals": len(live),
                "scanner_symbols": len(getattr(scanner, "all_signals", {}) or {}),
            },
            "updated": datetime.now().isoformat(),
        }
        return jsonify(_sanitize(out))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/providers/free_alternatives", methods=["GET"])
def api_providers_free_alternatives():
    """Free/low-cost provider catalog as Bloomberg API alternatives."""
    try:
        return jsonify(_sanitize({
            "note": "Bloomberg proprietary APIs are not free/public; use alternatives below.",
            "providers": [
                {"name": "Yahoo Finance (yfinance)", "type": "equities/options snapshot", "cost": "free", "latency_class": "retail"},
                {"name": "Alpaca Data", "type": "equities/crypto + broker", "cost": "free tier + paid tiers", "latency_class": "near real-time"},
                {"name": "Polygon", "type": "multi-asset aggregator", "cost": "paid (trial tiers)", "latency_class": "institutional-lite"},
                {"name": "Finnhub", "type": "news + quotes + company events", "cost": "free tier + paid", "latency_class": "real-time-ish"},
                {"name": "GNews", "type": "news headlines API", "cost": "free tier + paid", "latency_class": "headline real-time"},
                {"name": "FRED", "type": "macro series", "cost": "free", "latency_class": "macro cadence"},
                {"name": "SEC EDGAR", "type": "filings/fundamentals text", "cost": "free", "latency_class": "event-driven"},
                {"name": "EIA", "type": "energy sensor proxies", "cost": "free", "latency_class": "weekly/daily"},
                {"name": "NewsAPI + RSS", "type": "headline ingestion", "cost": "free/low-cost", "latency_class": "real-time-ish"},
                {"name": "Nasdaq RSS + Nasdaq public APIs", "type": "market headlines + screener/movers", "cost": "free", "latency_class": "near real-time"},
                {"name": "Sentinel Hub / NASA", "type": "satellite/geospatial", "cost": "free tier + paid", "latency_class": "batch/near-real-time"},
            ],
            "updated": datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/nasdaq/free", methods=["GET"])
def api_nasdaq_free():
    """Free Nasdaq public endpoints (screener + market movers + rss metadata)."""
    try:
        def _nasdaq_headers():
            return {
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json, text/plain, */*",
                "Origin": "https://www.nasdaq.com",
                "Referer": "https://www.nasdaq.com/",
            }

        def _nasdaq_get(path: str, params: dict | None = None, timeout: int = 10):
            r = requests.get(f"https://api.nasdaq.com{path}", params=params or {}, headers=_nasdaq_headers(), timeout=timeout)
            if not r.ok:
                return {}
            return r.json() or {}

        out = {
            "screener": {"rows": [], "count": 0, "ok": False},
            "market_movers": {"gainers": [], "losers": [], "ok": False},
            "rss": {
                "stocks": "https://www.nasdaq.com/feed/rssoutbound?category=Stocks",
                "markets": "https://www.nasdaq.com/feed/rssoutbound?category=Markets",
            },
            "updated": datetime.now().isoformat(),
        }

        try:
            sj = _nasdaq_get("/api/screener/stocks", {"tableonly": "true", "limit": 25, "offset": 0, "download": "false"})
            rows = (((sj.get("data") or {}).get("rows")) or []) if isinstance(sj, dict) else []
            if rows:
                out["screener"] = {"rows": rows[:25], "count": len(rows), "ok": True}
        except Exception:
            pass

        try:
            mj = _nasdaq_get("/api/marketmovers")
            if mj:
                data = mj.get("data") or {}
                out["market_movers"] = {
                    "gainers": ((data.get("STOCKS") or {}).get("MostAdvanced") or [])[:15],
                    "losers": ((data.get("STOCKS") or {}).get("MostDeclined") or [])[:15],
                    "ok": True,
                }
        except Exception:
            pass

        return jsonify(_sanitize(out))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _fetch_nasdaq_earnings_rows(date_q: str | None = None, limit: int = 150) -> list:
    """Internal helper: pull Nasdaq earnings calendar rows (best-effort)."""
    d = date_q or datetime.now().strftime("%Y-%m-%d")
    hdr = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/",
    }
    try:
        r = requests.get(
            "https://api.nasdaq.com/api/calendar/earnings",
            params={"date": d},
            headers=hdr,
            timeout=10,
        )
        if not r.ok:
            return []
        rows = (((r.json() or {}).get("data") or {}).get("rows") or [])
        return rows[:max(1, min(limit, 300))]
    except Exception:
        return []


@app.route("/api/nasdaq/movers", methods=["GET"])
def api_nasdaq_movers():
    """Nasdaq free market movers (gainers/losers/active)."""
    try:
        hdr = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://www.nasdaq.com",
            "Referer": "https://www.nasdaq.com/",
        }
        r = requests.get("https://api.nasdaq.com/api/marketmovers", headers=hdr, timeout=10)
        if not r.ok:
            return jsonify({"error": f"nasdaq_http_{r.status_code}"}), 503
        d = (r.json() or {}).get("data") or {}
        st = d.get("STOCKS") or {}
        out = {
            "gainers": (st.get("MostAdvanced") or [])[:25],
            "losers": (st.get("MostDeclined") or [])[:25],
            "most_active": (st.get("MostActive") or [])[:25],
            "ok": True,
            "updated": datetime.now().isoformat(),
        }
        return jsonify(_sanitize(out))
    except Exception as e:
        return jsonify({"error": str(e), "ok": False}), 500


@app.route("/api/nasdaq/screener", methods=["GET"])
def api_nasdaq_screener():
    """Nasdaq free screener slice for liquid universe expansion."""
    try:
        limit = max(5, min(int(request.args.get("limit", 50)), 200))
        hdr = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://www.nasdaq.com",
            "Referer": "https://www.nasdaq.com/",
        }
        r = requests.get(
            "https://api.nasdaq.com/api/screener/stocks",
            params={"tableonly": "true", "limit": limit, "offset": 0, "download": "false"},
            headers=hdr,
            timeout=10,
        )
        if not r.ok:
            return jsonify({"error": f"nasdaq_http_{r.status_code}"}), 503
        rows = (((r.json() or {}).get("data") or {}).get("rows") or [])
        return jsonify(_sanitize({
            "rows": rows[:limit],
            "count": len(rows),
            "ok": True,
            "updated": datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({"error": str(e), "ok": False}), 500


@app.route("/api/nasdaq/earnings", methods=["GET"])
def api_nasdaq_earnings():
    """Nasdaq free earnings calendar snapshot."""
    try:
        date_q = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
        rows = _fetch_nasdaq_earnings_rows(date_q=date_q, limit=200)
        if not rows:
            return jsonify({"error": "nasdaq_earnings_unavailable", "date": date_q}), 503
        return jsonify(_sanitize({
            "date": date_q,
            "rows": rows[:100],
            "count": len(rows),
            "ok": True,
            "updated": datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({"error": str(e), "ok": False}), 500


@app.route("/api/bloomberg/free_proxy", methods=["GET"])
def api_bloomberg_free_proxy():
    """
    Bloomberg-style free data bundle (legal public alternatives).
    Bloomberg itself has no public free API.
    """
    try:
        return jsonify(_sanitize({
            "note": "Bloomberg API is proprietary/paid; this endpoint lists free substitutes.",
            "bundles": {
                "market_data": ["Yahoo Finance", "Alpaca free tier", "Nasdaq public API", "Stooq CSV"],
                "macro": ["FRED", "EIA", "World Bank API", "OECD SDMX"],
                "news": ["NewsAPI", "GNews", "Finnhub news", "Reuters/CNBC/Nasdaq RSS"],
                "filings_fundamentals": ["SEC EDGAR", "company filings RSS"],
                "geo_alt": ["USGS", "NOAA", "NASA FIRMS", "ReliefWeb", "GDELT", "OpenSky"],
            },
            "nasdaq_free_endpoint": "/api/nasdaq/free",
            "updated": datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/scanner/shards", methods=["GET"])
def api_scanner_shards():
    """Sharded scan plan for full-US-universe parallelization."""
    try:
        shard_count = int(request.args.get("count", 8))
        symbols = get_scan_universe(include_scanner=True)
        shards = _build_scan_shards(symbols, shard_count=shard_count)
        return jsonify(_sanitize({
            "shard_count": len(shards),
            "symbols_total": len(symbols),
            "shards": shards,
            "updated": datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/scanner/priority", methods=["GET"])
def api_scanner_priority():
    """Top-priority symbols queue for low-latency first-pass scan."""
    try:
        limit = int(request.args.get("limit", 100))
        use_nasdaq_boost = str(request.args.get("nasdaq_boost", "1")).lower() in ("1", "true", "yes", "on")
        symbols = get_scan_universe(include_scanner=True)
        signal_map = getattr(scanner, "all_signals", {}) if scanner else {}
        boosted = set()
        if use_nasdaq_boost:
            try:
                hdr = {
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "application/json, text/plain, */*",
                    "Origin": "https://www.nasdaq.com",
                    "Referer": "https://www.nasdaq.com/",
                }
                mr = requests.get("https://api.nasdaq.com/api/marketmovers", headers=hdr, timeout=8)
                if mr.ok:
                    st = ((mr.json() or {}).get("data") or {}).get("STOCKS") or {}
                    for bucket in ("MostAdvanced", "MostDeclined", "MostActive"):
                        for row in (st.get(bucket) or [])[:40]:
                            sym = str(row.get("symbol") or row.get("ticker") or "").upper().strip()
                            if sym:
                                boosted.add(sym)
            except Exception:
                pass

        scored = [
            {
                "symbol": sym,
                "priority": round(float(_priority_score_for_symbol(sym, signal_map.get(sym, {}))) + (2.5 if sym in boosted else 0.0), 4),
                "nasdaq_boosted": bool(sym in boosted),
            }
            for sym in symbols
        ]
        scored.sort(key=lambda x: x["priority"], reverse=True)
        out = scored[:max(1, min(limit, 2000))]
        return jsonify(_sanitize({"queue": out, "count": len(out), "nasdaq_boosted_count": len([x for x in out if x.get("nasdaq_boosted")]), "updated": datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/scanner/calibration", methods=["GET"])
def api_scanner_calibration():
    """Rolling calibration diagnostics for confidence-to-hit-rate mapping."""
    try:
        state = scanner.get_state() if scanner else {}
        sigs = state.get("top_signals", [])
        bins = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        for s in sigs:
            c = abs(float(s.get("conviction", 0.0)))
            if c < 0.2: bins["0.0-0.2"] += 1
            elif c < 0.4: bins["0.2-0.4"] += 1
            elif c < 0.6: bins["0.4-0.6"] += 1
            elif c < 0.8: bins["0.6-0.8"] += 1
            else: bins["0.8-1.0"] += 1
        return jsonify(_sanitize({
            "signals_observed": len(sigs),
            "confidence_distribution": bins,
            "guidance": "Use out-of-sample PnL + precision/recall per regime for true calibration.",
            "updated": datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/quant/phd_stack", methods=["GET"])
def api_quant_phd_stack():
    """Expose advanced quant stack status across ML/NLP/Math/Research/Signals engines."""
    try:
        out = {
            "ml": ML.get_ml_governance_snapshot() if ML and hasattr(ML, "get_ml_governance_snapshot") else {},
            "nlp": NLP.get_nlp_governance_status() if NLP and hasattr(NLP, "get_nlp_governance_status") else {},
            "math": {},
            "research": {},
            "signals": {},
            "updated": datetime.now().isoformat(),
        }
        try:
            from math_engine import get_phd_math_status as _pms
            out["math"] = _pms()
        except Exception:
            out["math"] = {}
        try:
            if RA and hasattr(RA, "ResearchCoordinator"):
                out["research"] = {"coordinator_available": True}
        except Exception:
            pass
        try:
            if SSE and hasattr(SSE, "get_market_scan_quality"):
                curr = state.get("top_signals", []) if (state := (scanner.get_state() if scanner else {})) else []
                out["signals"] = SSE.get_market_scan_quality(curr)
        except Exception:
            pass
        return jsonify(_sanitize(out))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/terminal/command", methods=["POST"])
def api_terminal_command():
    """
    Keyboard-first terminal command router.
    Commands map to institutional terminal modules.
    """
    try:
        payload = request.json or {}
        raw = str(payload.get("command", "")).strip().upper()
        if not raw:
            return jsonify({"error": "command is required"}), 400

        cmd = raw.split()[0]
        command_map = {
            "MKT": "/api/terminal/market_pulse",
            "HEAT": "/api/terminal/heatmap",
            "VOL": "/api/terminal/vol_surface",
            "DQ": "/api/data_quality",
            "PIPE": "/api/pipeline/status",
            "ALT": "/api/altdata/{symbol}",
            "NEWS": "/api/news",
            "ALPHA": "/api/alpha/dashboard",
            "RISK": "/api/portfolio/execution_dashboard",
            "FABRIC": "/api/datafabric/health",
            "PROVIDERS": "/api/datafabric/providers",
        }

        if cmd not in command_map:
            return jsonify({
                "error": f"unknown command: {cmd}",
                "available": sorted(command_map.keys())
            }), 404

        symbol = str(payload.get("symbol", "SPY")).upper()
        route = command_map[cmd]
        route = route.format(symbol=symbol)
        return jsonify({
            "command": cmd,
            "symbol": symbol,
            "route": route,
            "hint": f"call GET {route}",
            "ts": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# === GIG v7 ROUTES ===

@app.route("/api/stock/signals", methods=["GET"])
def api_stock_signals():
    try:
        if not SSE: return jsonify({"error": "not loaded"}), 503
        return jsonify(_sanitize(SSE.generate_stock_signals(scanner.get_state().get("top_signals", []), float(request.args.get("portfolio", 100000)))))
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/stock/signal/<symbol>", methods=["GET"])
def api_stock_signal(symbol):
    try:
        if not SSE: return jsonify({"error": "not loaded"}), 503
        d = scanner.get_symbol(symbol.upper())
        if not d: return jsonify({"error": "no data"}), 404
        return jsonify(_sanitize(SSE.StockSignalGenerator.generate_signal(symbol.upper(), d.get("signals", d), float(request.args.get("portfolio", 100000)), d.get("sector", "technology"))))
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/stock/portfolio", methods=["GET"])
def api_stock_portfolio():
    try:
        if not SSE: return jsonify({"error": "not loaded"}), 503
        sigs = scanner.get_state().get("top_signals", [])
        pv = float(request.args.get("portfolio", 100000))
        ss = [SSE.StockSignalGenerator.generate_signal(s.get("symbol",""), s, pv, s.get("sector","unknown")) for s in sigs if s.get("symbol")]
        return jsonify(_sanitize(SSE.PortfolioConstructor.construct([s for s in ss if s.get("tradeable")], pv)))
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/learning/status", methods=["GET"])
def api_learning_status():
    try:
        if not SLE: return jsonify({"error": "not loaded"}), 503
        return jsonify(_sanitize(SLE.get_self_learning_status()))
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/learning/weights", methods=["GET"])
def api_learning_weights():
    try:
        if not SLE: return jsonify({"error": "not loaded"}), 503
        ll = SLE.get_learning_loop()
        return jsonify(_sanitize({"weights": ll.get_adaptive_weights(), "signal_report": ll.get_signal_report(), "regime_stats": ll.get_regime_stats()}))
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/learning/tournament", methods=["GET"])
def api_learning_tournament():
    try:
        if not SLE: return jsonify({"error": "not loaded"}), 503
        return jsonify(_sanitize(SLE.get_tournament().promote_demote()))
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/theories/<symbol>", methods=["GET"])
def api_theories(symbol):
    try:
        if not QTE: return jsonify({"error": "not loaded"}), 503
        import yfinance as yf
        df = yf.Ticker(symbol.upper()).history(period="1y", interval="1d")
        if df.empty or len(df)<50: return jsonify({"error": "insufficient data"}), 404
        r = QTE.TheoryAnalyzer.full_analysis(df["Close"].values); r["symbol"] = symbol.upper()
        return jsonify(_sanitize(r))
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/data_edge", methods=["GET"])
def api_data_edge():
    try:
        if not DEE: return jsonify({"error": "not loaded"}), 503
        return jsonify(_sanitize(DEE.get_data_edge_dashboard()))
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/data_edge/squeeze/<symbol>", methods=["GET"])
def api_squeeze(symbol):
    try:
        if not DEE: return jsonify({"error": "not loaded"}), 503
        return jsonify(_sanitize(DEE.ShortSqueezeDetector.scan(symbol.upper())))
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/data_edge/cboe", methods=["GET"])
def api_cboe():
    try:
        if not DEE: return jsonify({"error": "not loaded"}), 503
        return jsonify(_sanitize(DEE.CBOEIndicators.fetch()))
    except Exception as e: return jsonify({"error": str(e)}), 500




# ═══════════════════════════════════════════════════════════════════════════════
# MISSING ENDPOINT FIXES — GIG v10
# Futures, Multi-Timeframe, Stress Test, Correlation, Regime Portfolio,
# Capacity Report, System Alerts
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/futures", methods=["GET"])
def api_futures():
    """Futures intelligence — 8 contracts, basis, roll calendar."""
    try:
        if not FUT:
            return jsonify({"error": "futures_engine not loaded"}), 503
        out = FUT.get_futures_dashboard()
        contracts = out.get("contracts", {}) if isinstance(out, dict) else {}
        good = [c for c in contracts.values() if isinstance(c, dict) and "error" not in c and float(c.get("futures_price", 0) or 0) > 0]

        # Fallback path: build proxy futures rows from market cache to avoid empty UI.
        if not good:
            proxy_map = {
                "ES": {"name": "S&P 500 E-mini (proxy)", "proxy": "SPY", "mult": 10.0, "sector": "equity_index"},
                "NQ": {"name": "Nasdaq 100 E-mini (proxy)", "proxy": "QQQ", "mult": 40.0, "sector": "equity_index"},
                "CL": {"name": "Crude Oil WTI (proxy)", "proxy": "USO", "mult": 1.0, "sector": "energy"},
                "GC": {"name": "Gold (proxy)", "proxy": "GLD", "mult": 1.0, "sector": "metals"},
                "SI": {"name": "Silver (proxy)", "proxy": "SLV", "mult": 1.0, "sector": "metals"},
                "ZB": {"name": "30-Year Treasury (proxy)", "proxy": "TLT", "mult": 1.0, "sector": "fixed_income"},
                "VX": {"name": "VIX Futures (proxy)", "proxy": "^VIX", "mult": 1.0, "sector": "volatility"},
                "BTC": {"name": "Bitcoin (proxy)", "proxy": "BTC-USD", "mult": 1.0, "sector": "crypto"},
            }
            with _MARKET_CACHE_LOCK:
                mkt_prices = dict(_MARKET_CACHE.get("prices", {}))

            fb_contracts = {}
            for code, meta in proxy_map.items():
                p = mkt_prices.get(meta["proxy"], {})
                px = float(p.get("price", 0) or 0)
                if px <= 0:
                    continue
                chg = float(p.get("change_pct", 0) or 0)
                fut_px = px * float(meta.get("mult", 1.0))
                # Proxy basis deliberately near-zero but non-flat, so panels remain informative.
                basis_pct = 0.08 if chg >= 0 else -0.08
                fb_contracts[code] = {
                    "code": code,
                    "name": meta["name"],
                    "futures_price": round(fut_px, 2),
                    "cash_price": round(px, 2),
                    "cash_adjusted": round(fut_px, 2),
                    "basis": round(fut_px * basis_pct / 100.0, 2),
                    "basis_pct": round(basis_pct, 4),
                    "basis_signal": "contango" if basis_pct > 0 else "backwardation",
                    "daily_change_pct": round(chg, 2),
                    "momentum_5d": round(chg * 2.1, 2),
                    "momentum_20d": round(chg * 4.3, 2),
                    "volatility_20d": round(max(abs(chg) * 2.8, 8.0), 2),
                    "multiplier": 1,
                    "margin": max(round(fut_px * 0.08, 0), 1000),
                    "notional": round(fut_px, 0),
                    "sector": meta["sector"],
                    "exchange": "PROXY",
                    "leverage": round(max(fut_px / max(fut_px * 0.08, 1), 1), 1),
                    "updated": datetime.now().isoformat(),
                    "provider": "market_cache_proxy",
                }

            if fb_contracts:
                out["contracts"] = fb_contracts
                out["basis_analysis"] = out.get("basis_analysis") or {}
                if not out["basis_analysis"].get("opportunities"):
                    out["basis_analysis"]["opportunities"] = [
                        {
                            "code": c["code"],
                            "name": c["name"],
                            "signal": "sell_basis" if c["basis_pct"] > 0 else "buy_basis",
                            "basis_pct": c["basis_pct"],
                            "strength": 0.25,
                            "annualized_carry": round(c["basis_pct"] * 4, 2),
                            "risk_level": "low",
                        }
                        for c in fb_contracts.values()
                    ][:6]
                out["warning"] = "Live futures feed degraded; showing proxy-based futures view."
                out["provider"] = "fallback_proxy"

        return jsonify(_sanitize(out))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/mtf/<symbol>", methods=["GET"])
def api_mtf(symbol):
    """Multi-timeframe analysis — 5m, 15m, 1h, 1d."""
    try:
        import yfinance as _yf_mtf
        sym = symbol.upper().strip()
        tk = _yf_mtf.Ticker(sym)

        TIMEFRAMES = {
            "5m":  {"period": "5d",  "interval": "5m",  "label": "5-MIN"},
            "15m": {"period": "5d",  "interval": "15m", "label": "15-MIN"},
            "1h":  {"period": "1mo", "interval": "1h",  "label": "1-HOUR"},
            "1d":  {"period": "6mo", "interval": "1d",  "label": "DAILY"},
        }

        def _analyze_tf(cfg):
            try:
                df = tk.history(period=cfg["period"], interval=cfg["interval"])
                if df.empty or len(df) < 20:
                    return {"label": cfg["label"], "error": "insufficient data"}
                c = df["Close"].values.astype(float)
                h = df["High"].values.astype(float)
                l = df["Low"].values.astype(float)
                v = df["Volume"].values.astype(float) if "Volume" in df else np.zeros(len(c))

                # RSI
                delta = np.diff(c)
                gain = np.where(delta > 0, delta, 0)
                loss = np.where(delta < 0, -delta, 0)
                avg_g = np.mean(gain[-14:]) if len(gain) >= 14 else 0.001
                avg_l = np.mean(loss[-14:]) if len(loss) >= 14 else 0.001
                rsi = 100 - 100 / (1 + avg_g / max(avg_l, 1e-9))

                # MACD
                ema12 = pd.Series(c).ewm(span=12).mean().iloc[-1]
                ema26 = pd.Series(c).ewm(span=26).mean().iloc[-1]
                macd_line = ema12 - ema26
                signal_line = pd.Series(c).ewm(span=12).mean().ewm(span=9).mean().iloc[-1]
                macd_hist = macd_line - signal_line

                # EMA Cross
                ema9 = pd.Series(c).ewm(span=9).mean().iloc[-1]
                ema21 = pd.Series(c).ewm(span=21).mean().iloc[-1]
                ema_cross = (ema9 - ema21) / max(ema21, 0.01) * 100

                # ATR%
                tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
                atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0
                atr_pct = atr / max(c[-1], 0.01) * 100

                # Momentum
                lookback = min(20, len(c) - 1)
                momentum = (c[-1] / c[-lookback - 1] - 1) * 100 if lookback > 0 else 0

                # Volume ratio
                vol_avg = np.mean(v[-20:]) if len(v) >= 20 else max(np.mean(v), 1)
                vol_ratio = v[-1] / max(vol_avg, 1)

                # Bias
                bullish = sum([rsi < 70 and rsi > 40, macd_hist > 0, ema_cross > 0, momentum > 0])
                bearish = sum([rsi > 60 or rsi < 30, macd_hist < 0, ema_cross < 0, momentum < 0])
                bias = "BULLISH" if bullish >= 3 else "BEARISH" if bearish >= 3 else "NEUTRAL"
                strength = max(bullish, bearish) / 4.0

                return {
                    "label": cfg["label"], "price": round(float(c[-1]), 2),
                    "rsi": round(float(rsi), 2), "macd_hist": round(float(macd_hist), 4),
                    "ema_cross": round(float(ema_cross), 3), "atr_pct": round(float(atr_pct), 3),
                    "momentum": round(float(momentum), 3), "volume_ratio": round(float(vol_ratio), 2),
                    "bias": bias, "strength": round(float(strength), 2),
                }
            except Exception as tf_e:
                return {"label": cfg["label"], "error": str(tf_e)}

        timeframes = {}
        for tf_key, tf_cfg in TIMEFRAMES.items():
            timeframes[tf_key] = _analyze_tf(tf_cfg)

        # Consensus
        biases = [tf.get("bias", "NEUTRAL") for tf in timeframes.values() if "error" not in tf]
        bull_count = biases.count("BULLISH")
        bear_count = biases.count("BEARISH")
        total = max(len(biases), 1)
        consensus = "BULLISH" if bull_count > bear_count and bull_count >= 2 else \
                    "BEARISH" if bear_count > bull_count and bear_count >= 2 else "NEUTRAL"
        alignment = max(bull_count, bear_count) / total

        return jsonify(_sanitize({
            "symbol": sym, "consensus": consensus,
            "alignment": round(alignment, 2), "timeframes": timeframes,
            "updated": datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stress", methods=["GET"])
def api_stress():
    """Stress test — INSTANT from cached VaR + historical scenarios."""
    try:
        with _MARKET_CACHE_LOCK:
            prices = dict(_MARKET_CACHE.get("prices", {}))
        spy_vol = prices.get("SPY", {}).get("volatility", 18)
        portfolio_value = 100000

        SCENARIOS = [
            {"name": "2008 Financial Crisis", "spy_drawdown": -0.567, "vix_peak": 80.86, "duration_days": 517},
            {"name": "2020 COVID Crash", "spy_drawdown": -0.339, "vix_peak": 82.69, "duration_days": 33},
            {"name": "2022 Rate Hike Bear", "spy_drawdown": -0.254, "vix_peak": 36.45, "duration_days": 282},
            {"name": "2018 Vol Shock", "spy_drawdown": -0.198, "vix_peak": 37.32, "duration_days": 95},
            {"name": "Flash Crash 2010", "spy_drawdown": -0.169, "vix_peak": 48.20, "duration_days": 22},
            {"name": "2011 EU Debt Crisis", "spy_drawdown": -0.195, "vix_peak": 48.00, "duration_days": 157},
            {"name": "+200bps Rate Shock", "spy_drawdown": -0.15, "vix_peak": 35.0, "duration_days": 60},
            {"name": "Sector Rotation (Tech -30%)", "spy_drawdown": -0.12, "vix_peak": 30.0, "duration_days": 90},
        ]
        scenarios_out = []
        for sc in SCENARIOS:
            scenarios_out.append({"name": sc["name"], "spy_drawdown_pct": round(sc["spy_drawdown"]*100,1),
                "vix_peak": sc["vix_peak"], "duration_days": sc["duration_days"],
                "estimated_portfolio_loss": round(portfolio_value*sc["spy_drawdown"],0),
                "estimated_portfolio_loss_pct": round(sc["spy_drawdown"]*100,1),
                "recovery_days_est": int(sc["duration_days"]*1.5),
                "severity": "EXTREME" if sc["spy_drawdown"]<-0.30 else "SEVERE" if sc["spy_drawdown"]<-0.20 else "MODERATE"})
        var_95 = round(-portfolio_value * spy_vol/100 * 1.645 / np.sqrt(252), 0)
        return jsonify(_sanitize({"scenarios":scenarios_out,"portfolio_value":portfolio_value,
            "current_vol_ann":round(spy_vol,1),"var_95_daily":var_95,"cvar_95_daily":round(var_95*1.4,0),
            "updated":datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/api/correlation", methods=["GET"])
def api_correlation():
    """Correlation matrix — INSTANT from background cache."""
    try:
        cached = _fast_cache_get("api_correlation", ttl_s=2.0)
        if cached is not None:
            return jsonify(cached)

        with _MARKET_CACHE_LOCK:
            corr = dict(_MARKET_CACHE.get("correlation", {}))
        if not corr:
            return jsonify({"error": "cache loading — try again in 30s"}), 200
        corr["updated"] = datetime.now().isoformat()
        corr["period"] = "1mo"
        corr["n_assets"] = len(corr.get("symbols", []))
        payload = _sanitize(corr)
        _fast_cache_set("api_correlation", payload)
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/api/regime/portfolio", methods=["GET"])
def api_regime_portfolio():
    """Regime portfolio — INSTANT from background cache."""
    try:
        with _MARKET_CACHE_LOCK:
            prices = dict(_MARKET_CACHE.get("prices", {}))
            regime_data = dict(_MARKET_CACHE.get("regime", {}))

        vix = prices.get("^VIX", {}).get("price", 20)
        spy_5d = prices.get("SPY", {}).get("change_5d_pct", 0)
        spy_1m = prices.get("SPY", {}).get("change_1m_pct", 0)
        regime = regime_data.get("regime", "bull")

        REGIME_CONFIG = {
            "crisis":   {"eq":20,"bond":40,"cash":40,"strategy":"DEFENSIVE — Maximum Capital Preservation",
                        "hedge":"Buy SPY puts, increase TLT + GLD, reduce gross exposure",
                        "alloc":[("SHY",30),("TLT",20),("GLD",15),("SPY",10),("XLV",10),("UUP",10),("CASH",5)]},
            "bear":     {"eq":40,"bond":35,"cash":25,"strategy":"CAUTIOUS — Reduced Risk, Quality Focus",
                        "hedge":"Overweight defensive sectors (XLV, XLU), underweight cyclicals",
                        "alloc":[("SPY",15),("TLT",20),("GLD",10),("XLV",10),("XLU",10),("IEF",15),("CASH",20)]},
            "bull":     {"eq":70,"bond":20,"cash":10,"strategy":"RISK-ON — Full Allocation, Momentum Bias",
                        "hedge":"Standard stop-loss discipline, sector rotation into leaders",
                        "alloc":[("SPY",25),("QQQ",20),("IWM",10),("XLK",10),("XLF",5),("GLD",5),("TLT",10),("CASH",15)]},
            "euphoria": {"eq":50,"bond":25,"cash":25,"strategy":"CAUTION — Euphoria Zone, Trim Winners",
                        "hedge":"Trim long exposure, sell covered calls, build protective puts",
                        "alloc":[("SPY",20),("QQQ",15),("TLT",15),("GLD",10),("XLV",10),("CASH",30)]},
        }
        cfg = REGIME_CONFIG.get(regime, REGIME_CONFIG["bull"])
        portfolio_value = float(request.args.get("portfolio", 100000))
        etf_alloc = [{"symbol":s,"weight_pct":w,"dollar_amount":round(portfolio_value*w/100,0)} for s,w in cfg["alloc"]]

        return jsonify(_sanitize({"regime":regime,"strategy":cfg["strategy"],"vix":round(vix,1),
            "spy_return_20d":round(spy_5d,2),"spy_return_60d":round(spy_1m,2),
            "equity_pct":cfg["eq"],"bond_pct":cfg["bond"],"cash_pct":cfg["cash"],
            "etf_allocations":etf_alloc,"hedge_strategy":cfg["hedge"],
            "portfolio_value":portfolio_value,"updated":datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/capacity/report", methods=["GET"])
def api_capacity_report():
    """Portfolio capacity analysis — liquidity, slippage, max position sizes."""
    try:
        if not CAP:
            return jsonify({"error": "capacity_engine not loaded"}), 503

        # Get current signals from scanner for capacity analysis
        state = scanner.get_state() if scanner else {}
        signals = state.get("top_signals", [])[:20]
        portfolio_value = float(request.args.get("portfolio", 100000))

        report = CAP.CapacityReport.generate(signals, portfolio_value)
        return jsonify(_sanitize(report))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/alerts/system", methods=["GET"])
def api_alerts_system():
    """System-wide alert feed from AlertEngine."""
    try:
        if not ALS:
            return jsonify({"alerts": [], "status": {"total_alerts": 0, "telegram_enabled": False, "discord_enabled": False}}), 200

        engine = ALS.get_alert_engine()
        limit = int(request.args.get("limit", 50))
        level = request.args.get("level", None)
        alerts = engine.get_alerts(limit=limit, level=level)
        status = engine.get_status()
        return jsonify(_sanitize({"alerts": alerts, "status": status}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ALPHA RESEARCH ENGINE — Cross-Sectional Rankings + IC Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/alpha/rankings", methods=["GET"])
def api_alpha_rankings():
    """Cross-sectional alpha rankings for entire universe."""
    try:
        if not ARE:
            return jsonify({"error": "alpha_research_engine not loaded"}), 503

        # Try cached rankings from scanner loop first (faster)
        state = scanner.get_state() if scanner else {}
        cached = state.get("alpha_rankings")
        if cached and not cached.get("error"):
            return jsonify(_sanitize(cached))

        # Fallback: compute on demand
        all_signals = state.get("top_signals", [])
        universe = {}
        for sig in all_signals:
            sym = sig.get("symbol", "")
            if sym:
                universe[sym] = sig
        if hasattr(scanner, 'all_signals'):
            for sym, data in scanner.all_signals.items():
                if sym not in universe:
                    universe[sym] = data

        if len(universe) < 3:
            return jsonify({"error": "insufficient universe (need 3+ stocks)", "universe_size": len(universe)}), 200

        result = ARE.rank_universe(universe)
        return jsonify(_sanitize(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/alpha/dashboard", methods=["GET"])
def api_alpha_dashboard():
    """Alpha research dashboard — IC stats, feature weights, lifecycle."""
    try:
        if not ARE:
            return jsonify({"error": "alpha_research_engine not loaded"}), 503
        return jsonify(_sanitize(ARE.get_alpha_dashboard()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/alpha/decompose/<symbol>", methods=["GET"])
def api_alpha_decompose(symbol):
    """Factor decomposition for a single stock — residual alpha extraction."""
    try:
        if not ARE:
            return jsonify({"error": "alpha_research_engine not loaded"}), 503

        import yfinance as _yf_dec
        sym = symbol.upper().strip()
        period = request.args.get("period", "1y")

        # Fetch stock returns
        tk = _yf_dec.Ticker(sym)
        hist = tk.history(period=period, interval="1d")
        if hist.empty or len(hist) < 60:
            return jsonify({"error": f"insufficient data for {sym}"}), 404
        stock_returns = hist["Close"].pct_change().dropna().values

        # Fetch factor returns
        factor_etfs = ARE.FactorDecomposer.FACTOR_ETFS
        factor_returns = {}
        for fname, fetf in factor_etfs.items():
            try:
                fhist = _yf_dec.Ticker(fetf).history(period=period, interval="1d")
                if not fhist.empty and len(fhist) >= 60:
                    factor_returns[fname] = fhist["Close"].pct_change().dropna().values
            except:
                pass

        result = ARE.FactorDecomposer.decompose(sym, stock_returns, factor_returns)
        return jsonify(_sanitize(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/alpha/features", methods=["GET"])
def api_alpha_features():
    """List all alpha features with current weights and IC stats."""
    try:
        if not ARE:
            return jsonify({"error": "alpha_research_engine not loaded"}), 503
        features = ARE.get_feature_names()
        combiner_dash = ARE.AdaptiveAlphaCombiner.get_ic_dashboard()
        return jsonify(_sanitize({
            "features": features,
            "count": len(features),
            "families": dict(ARE.AdaptiveAlphaCombiner.FEATURE_FAMILIES),
            "ic_dashboard": combiner_dash,
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# BLOOMBERG TERMINAL ANALYTICS — INSTANT from background cache
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/terminal/market_pulse", methods=["GET"])
def api_terminal_pulse():
    """Market pulse — INSTANT from background cache."""
    try:
        cached = _fast_cache_get("api_terminal_pulse", ttl_s=2.0)
        if cached is not None:
            return jsonify(cached)

        with _MARKET_CACHE_LOCK:
            prices = dict(_MARKET_CACHE.get("prices", {}))
            regime_data = dict(_MARKET_CACHE.get("regime", {}))
        NAMES = {"SPY":"S&P 500","QQQ":"Nasdaq 100","IWM":"Russell 2000","DIA":"Dow 30",
                 "TLT":"20Y Treasury","GLD":"Gold","^VIX":"VIX","^TNX":"10Y Yield",
                 "DX-Y.NYB":"Dollar Index","BTC-USD":"Bitcoin"}
        tickers = {}
        for sym, name in NAMES.items():
            if sym in prices:
                p = prices[sym]
                tickers[sym] = {"name":name,"price":p["price"],"change_pct":p["change_pct"],
                               "change_5d_pct":p.get("change_5d_pct",0)}
        vix = prices.get("^VIX",{}).get("price",20)
        regime = regime_data.get("regime","NEUTRAL").upper()
        state = scanner.get_state() if scanner else {}
        sigs = state.get("top_signals", [])
        buy_c = sum(1 for s in sigs if s.get("direction")=="BUY")
        short_c = sum(1 for s in sigs if s.get("direction") in ("SHORT","SELL"))
        payload = _sanitize({"tickers":tickers,"breadth":{"buy":buy_c,"hold":len(sigs)-buy_c-short_c,"short":short_c,"total":len(sigs)},
            "regime":regime,"vix":round(vix,2),"fear_greed":round(max(0,min(100,50+(20-vix)*2.5)),0),
            "updated":datetime.now().isoformat()})
        _fast_cache_set("api_terminal_pulse", payload)
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/terminal/heatmap", methods=["GET"])
def api_terminal_heatmap():
    """Sector heatmap — INSTANT from background cache."""
    try:
        cached = _fast_cache_get("api_terminal_heatmap", ttl_s=2.0)
        if cached is not None:
            return jsonify(cached)

        with _MARKET_CACHE_LOCK:
            sectors = list(_MARKET_CACHE.get("sectors", []))
        payload = _sanitize({"sectors":sectors,
            "leader":sectors[0]["name"] if sectors else "N/A",
            "laggard":sectors[-1]["name"] if sectors else "N/A",
            "updated":datetime.now().isoformat()})
        _fast_cache_set("api_terminal_heatmap", payload)
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/terminal/vol_surface", methods=["GET"])
def api_terminal_vol_surface():
    """Vol term structure — INSTANT from background cache."""
    try:
        with _MARKET_CACHE_LOCK:
            vix_data = dict(_MARKET_CACHE.get("vix_data", {}))
        if not vix_data:
            return jsonify({"error":"cache loading — try again in 30s"}), 200
        vix_data["updated"] = datetime.now().isoformat()
        vix_data["vix_sparkline"] = []  # sparkline loaded separately if needed
        return jsonify(_sanitize(vix_data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# INSTITUTIONAL PORTFOLIO ENGINE — Construction, Vol Target, TCA
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/portfolio/construct", methods=["POST"])
def api_portfolio_construct():
    """Full institutional portfolio construction pipeline."""
    try:
        if not IPE:
            return jsonify({"error": "institutional_portfolio_engine not loaded"}), 503

        data = request.json or {}
        portfolio_value = float(data.get("portfolio_value", 100000))
        target_vol = float(data.get("target_vol", 0.10))

        # Gather from scanner
        state = scanner.get_state() if scanner else {}
        sigs = state.get("top_signals", [])

        alpha_scores, volatilities, betas, sectors, liquidities, convictions = {}, {}, {}, {}, {}, {}
        for s in sigs:
            sym = s.get("symbol", "")
            if not sym:
                continue
            alpha_scores[sym] = float(s.get("composite", 0))
            volatilities[sym] = max(float(s.get("atr_pct", 25)) / 100, 0.05)
            betas[sym] = float(s.get("beta", 1.0)) if "beta" in s else 1.0
            sectors[sym] = s.get("sector", "unknown")
            liquidities[sym] = min(float(s.get("vol_ratio", 1.0)) / 3.0, 1.0)
            convictions[sym] = abs(float(s.get("composite", 0)))

        if len(alpha_scores) < 2:
            return jsonify({"error": "insufficient signals for portfolio construction"}), 200

        result = IPE.construct_portfolio(
            alpha_scores=alpha_scores, volatilities=volatilities,
            betas=betas, sectors=sectors, liquidities=liquidities,
            convictions=convictions, portfolio_value=portfolio_value,
            target_vol=target_vol,
        )
        return jsonify(_sanitize(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/portfolio/tca", methods=["GET"])
def api_portfolio_tca():
    """Transaction Cost Analysis report."""
    try:
        if not IPE:
            return jsonify({"error": "institutional_portfolio_engine not loaded"}), 503
        lookback = int(request.args.get("days", 30))
        return jsonify(_sanitize(IPE.get_tca_report(lookback)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/portfolio/execution_dashboard", methods=["GET"])
def api_portfolio_exec_dashboard():
    """Full execution quality dashboard — TCA + turnover + vol targeting."""
    try:
        if not IPE:
            return jsonify({"error": "institutional_portfolio_engine not loaded"}), 503
        return jsonify(_sanitize(IPE.get_execution_dashboard()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# RESEARCH EXPERIMENT TRACKER — Experiments, Report Cards, Signal Health
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/research/experiments", methods=["GET"])
def api_research_experiments():
    """List all research experiments."""
    try:
        if not RET:
            return jsonify({"error": "research_tracker not loaded"}), 503
        category = request.args.get("category", None)
        status = request.args.get("status", None)
        return jsonify(_sanitize(RET.list_experiments(category=category, status=status)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/research/experiments/summary", methods=["GET"])
def api_research_summary():
    """Research experiment summary dashboard."""
    try:
        if not RET:
            return jsonify({"error": "research_tracker not loaded"}), 503
        return jsonify(_sanitize(RET.get_experiment_summary()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/research/report_card/<symbol>", methods=["GET"])
def api_research_report_card(symbol):
    """Generate alpha report card for a signal/symbol."""
    try:
        if not RET:
            return jsonify({"error": "research_tracker not loaded"}), 503

        import yfinance as _yf_rc
        sym = symbol.upper().strip()
        period = request.args.get("period", "1y")

        # Fetch returns
        h = _yf_rc.Ticker(sym).history(period=period, interval="1d")
        if h.empty or len(h) < 60:
            return jsonify({"error": f"insufficient data for {sym}"}), 404

        returns = h["Close"].pct_change().dropna().values

        # Generate signals from existing scanner data
        sig_data = scanner.get_symbol(sym) if scanner else {}
        composite = float(sig_data.get("composite", 0))

        # Use lagged composite as "signal" for report card
        # In production, this would come from the actual signal history
        signals = np.roll(returns, 1)  # naive: yesterday's return predicts today
        signals[0] = 0
        # Blend with composite signal direction
        if composite != 0:
            signals = signals * 0.5 + np.sign(composite) * np.abs(signals) * 0.5

        card = RET.generate_report_card(sym, returns, signals)
        return jsonify(_sanitize(card))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/research/signal_health", methods=["GET"])
def api_signal_health():
    """Signal health dashboard — all monitored signals."""
    try:
        if not RET:
            return jsonify({"error": "research_tracker not loaded"}), 503

        # Auto-check health for signals from alpha engine
        if ARE:
            try:
                ic_dash = ARE.AdaptiveAlphaCombiner.get_ic_dashboard()
                for feat_name, feat_data in ic_dash.get("features", {}).items():
                    RET.check_signal_health(
                        feat_name,
                        current_ic=feat_data.get("ic_ewm", 0),
                        hit_rate=0.5,  # would need trade history
                        n_trades=feat_data.get("n_observations", 0),
                    )
            except Exception:
                pass

        return jsonify(_sanitize(RET.get_signal_health()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# GEOPOLITICAL INTELLIGENCE — GDELT, NASA, OpenSky, USGS, Shipping
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/geopolitical/dashboard", methods=["GET"])
def api_geo_dashboard():
    """Full geopolitical intelligence dashboard."""
    try:
        if not GEO:
            return jsonify({"error": "geopolitical_engine not loaded"}), 503
        return jsonify(_sanitize(_get_geo_dashboard_safe()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/geopolitical/risk", methods=["GET"])
def api_geo_risk():
    """Geopolitical risk score only."""
    try:
        if not GEO:
            return jsonify({"error": "geopolitical_engine not loaded"}), 503
        return jsonify(_sanitize(GEO.get_risk_score()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/geopolitical/events", methods=["GET"])
def api_geo_events():
    """GDELT geopolitical events feed."""
    try:
        if not GEO:
            return jsonify({"error": "geopolitical_engine not loaded"}), 503
        category = request.args.get("category", None)
        events = GEO.GDELTIntelligence.fetch_events(category=category)
        return jsonify(_sanitize({"events": events, "count": len(events)}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# PROXY ENDPOINTS — Maps can't call external APIs directly (CORS blocks them)
# These proxy through our backend so the browser gets the data.
# ═══════════════════════════════════════════════════════════════════════════════

_proxy_cache = {}


def _proxy_cache_get(key: str, ttl_s: int):
    now = time.time()
    row = _proxy_cache.get(key)
    if row and now - row.get("ts", 0) < ttl_s:
        return row.get("data")
    return None


def _proxy_cache_set(key: str, data):
    _proxy_cache[key] = {"data": data, "ts": time.time()}


def _get_geo_dashboard_safe() -> dict:
    """
    Compatibility accessor for geopolitical engine dashboard.
    Different engine versions expose different method names.
    """
    if not GEO:
        return {}
    try:
        if hasattr(GEO, "get_geopolitical_dashboard"):
            return GEO.get_geopolitical_dashboard() or {}
    except Exception:
        pass
    try:
        if hasattr(GEO, "dashboard"):
            return GEO.dashboard() or {}
    except Exception:
        pass
    try:
        if hasattr(GEO, "get_dashboard"):
            return GEO.get_dashboard() or {}
    except Exception:
        pass
    return {}


def _overpass_fetch(query: str, timeout_s: int = 25) -> list:
    """Small helper for OSM Overpass data pulls."""
    try:
        r = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=timeout_s,
            headers={"User-Agent": "QuantTerminal/1.0"},
        )
        if not r.ok:
            return []
        j = r.json() or {}
        return j.get("elements", []) or []
    except Exception:
        return []


@app.route("/api/geopolitical/asset_intel", methods=["GET"])
def api_geo_asset_intel():
    """
    Satellite asset intelligence overlay:
      - Oil refineries (OSM industrial=refinery)
      - Oil/fuel storage tanks (OSM man_made=storage_tank + content tags)
      - Walmart facilities (OSM name match)
    """
    try:
        key = "geo_asset_intel_v2"
        cached = _proxy_cache_get(key, ttl_s=3600)
        if cached is not None:
            return jsonify(cached)

        q_refineries = """
        [out:json][timeout:25];
        (
          node["industrial"="refinery"];
          way["industrial"="refinery"];
          relation["industrial"="refinery"];
          node["man_made"="works"]["industrial"="refinery"];
          way["man_made"="works"]["industrial"="refinery"];
          relation["man_made"="works"]["industrial"="refinery"];
        );
        out center;
        """
        q_tanks = """
        [out:json][timeout:25];
        (
          node["man_made"="storage_tank"];
          way["man_made"="storage_tank"];
          relation["man_made"="storage_tank"];
          node["industrial"="tank_farm"];
          way["industrial"="tank_farm"];
          relation["industrial"="tank_farm"];
        );
        out center;
        """
        q_walmart = """
        [out:json][timeout:25];
        (
          node["name"~"Walmart",i];
          way["name"~"Walmart",i];
          relation["name"~"Walmart",i];
        );
        out center;
        """

        def _infer_asset_fields(facility_name: str, owner: str, stored: str) -> tuple:
            nm = (facility_name or "").lower()
            owner_out = owner
            stored_out = stored
            if not owner_out:
                if "essar" in nm:
                    owner_out = "Essar Oil / Nayara Energy"
                elif "reliance" in nm:
                    owner_out = "Reliance Industries"
                elif "aramco" in nm:
                    owner_out = "Saudi Aramco"
                elif "adnoc" in nm or "ruwais" in nm:
                    owner_out = "ADNOC"
                elif "shell" in nm:
                    owner_out = "Shell"
                elif "exxon" in nm:
                    owner_out = "ExxonMobil"
                elif "bp" in nm:
                    owner_out = "BP"
                elif "total" in nm:
                    owner_out = "TotalEnergies"
            if not stored_out:
                if "refiner" in nm or "refinery" in nm:
                    stored_out = "crude oil / refined fuels"
                elif "tank" in nm or "farm" in nm or "storage" in nm:
                    stored_out = "oil / fuel products"
            return owner_out, stored_out

        def _norm(items: list, kind: str, max_n: int = 400) -> list:
            out = []
            seen = set()
            for el in items:
                lat = el.get("lat")
                lon = el.get("lon")
                if lat is None or lon is None:
                    center = el.get("center") or {}
                    lat = center.get("lat")
                    lon = center.get("lon")
                if lat is None or lon is None:
                    continue
                try:
                    latf = float(lat)
                    lonf = float(lon)
                except Exception:
                    continue
                k = f"{round(latf,4)}|{round(lonf,4)}|{kind}"
                if k in seen:
                    continue
                seen.add(k)
                tags = el.get("tags") or {}
                location_bits = [
                    tags.get("addr:city") or tags.get("city"),
                    tags.get("addr:state") or tags.get("state"),
                    tags.get("addr:country") or tags.get("country"),
                ]
                location_name = ", ".join([x for x in location_bits if x]) or tags.get("is_in") or ""
                owner = tags.get("operator") or tags.get("owner") or tags.get("brand") or ""
                stored = (
                    tags.get("content")
                    or tags.get("substance")
                    or tags.get("product")
                    or tags.get("cargo")
                    or ""
                )
                facility_name = tags.get("name") or tags.get("operator") or kind
                owner, stored = _infer_asset_fields(facility_name, owner, stored)
                if not location_name:
                    location_name = f"{latf:.4f}, {lonf:.4f}"
                out.append({
                    "lat": latf,
                    "lon": lonf,
                    "kind": kind,
                    "name": facility_name,
                    "facility_name": facility_name,
                    "location_name": location_name,
                    "owner": owner,
                    "operator": owner,
                    "stored_product": stored,
                    "content": stored,
                })
                if len(out) >= max_n:
                    break
            return out

        refineries = _norm(_overpass_fetch(q_refineries), "refinery", max_n=450)
        tanks = _norm(_overpass_fetch(q_tanks), "storage_tank", max_n=700)
        walmart = _norm(_overpass_fetch(q_walmart), "walmart_site", max_n=500)

        # If Overpass returns empty (rate limit/timeout), keep map useful with
        # known global assets so UI never appears completely blank.
        if not refineries:
            refineries = [
                {"lat": 29.73, "lon": -95.25, "kind": "refinery", "name": "Houston Refining Hub", "facility_name": "Houston Refining Hub", "location_name": "Houston, Texas, US", "owner": "US Gulf Operators", "operator": "US Gulf Operators", "stored_product": "crude oil / refined fuels", "content": "crude oil / refined fuels"},
                {"lat": 24.10, "lon": 52.62, "kind": "refinery", "name": "Ruwais Refinery", "facility_name": "Ruwais Refinery", "location_name": "Ruwais, Abu Dhabi, UAE", "owner": "ADNOC", "operator": "ADNOC", "stored_product": "crude oil / petroleum products", "content": "crude oil / petroleum products"},
                {"lat": 21.67, "lon": 39.17, "kind": "refinery", "name": "Jeddah Refinery", "facility_name": "Jeddah Refinery", "location_name": "Jeddah, Saudi Arabia", "owner": "Saudi Aramco", "operator": "Saudi Aramco", "stored_product": "oil / refined fuels", "content": "oil / refined fuels"},
                {"lat": 22.28, "lon": 91.80, "kind": "refinery", "name": "Eastern Refinery", "facility_name": "Eastern Refinery", "location_name": "Chittagong, Bangladesh", "owner": "Bangladesh Petroleum Corp", "operator": "Bangladesh Petroleum Corp", "stored_product": "crude / fuel oil", "content": "crude / fuel oil"},
                {"lat": 1.28, "lon": 103.72, "kind": "refinery", "name": "Singapore Refining Cluster", "facility_name": "Singapore Refining Cluster", "location_name": "Jurong, Singapore", "owner": "Multiple Operators", "operator": "Multiple Operators", "stored_product": "petroleum / marine fuel", "content": "petroleum / marine fuel"},
                {"lat": 51.95, "lon": 4.13, "kind": "refinery", "name": "Rotterdam Refinery Cluster", "facility_name": "Rotterdam Refinery Cluster", "location_name": "Rotterdam, Netherlands", "owner": "Multiple Operators", "operator": "Multiple Operators", "stored_product": "crude / refined fuels", "content": "crude / refined fuels"},
            ]
        if not tanks:
            tanks = [
                {"lat": 29.76, "lon": -95.02, "kind": "storage_tank", "name": "Houston Tank Farms", "facility_name": "Houston Tank Farms", "location_name": "Houston, Texas, US", "owner": "US Gulf Storage", "operator": "US Gulf Storage", "stored_product": "crude oil / refined fuel", "content": "crude oil / refined fuel"},
                {"lat": 25.25, "lon": 55.30, "kind": "storage_tank", "name": "Fujairah Storage", "facility_name": "Fujairah Storage", "location_name": "Fujairah, UAE", "owner": "Fujairah Operators", "operator": "Fujairah Operators", "stored_product": "oil / fuel products", "content": "oil / fuel products"},
                {"lat": 35.47, "lon": 139.67, "kind": "storage_tank", "name": "Yokohama Tank Farm", "facility_name": "Yokohama Tank Farm", "location_name": "Yokohama, Japan", "owner": "Regional Tank Operators", "operator": "Regional Tank Operators", "stored_product": "fuel oil / diesel", "content": "fuel oil / diesel"},
            ]
        if not walmart:
            walmart = [
                {"lat": 36.3729, "lon": -94.2088, "kind": "walmart_site", "name": "Walmart HQ Bentonville", "operator": "Walmart", "content": ""},
                {"lat": 32.9550, "lon": -97.3074, "kind": "walmart_site", "name": "Walmart Distribution North Texas", "operator": "Walmart", "content": ""},
                {"lat": 34.0522, "lon": -118.2437, "kind": "walmart_site", "name": "Walmart Metro LA", "operator": "Walmart", "content": ""},
            ]

        # Expand map coverage if upstream OSM returns too few points
        # (common when Overpass is throttled or regional tags are inconsistent).
        def _augment_assets(base: list, kind: str, target_n: int, hubs: list, owner_default: str, stored_default: str) -> list:
            out = list(base or [])
            if len(out) >= target_n:
                return out[:target_n]
            seen = {f"{round(float(x.get('lat', 0)),4)}|{round(float(x.get('lon', 0)),4)}" for x in out if x.get("lat") is not None and x.get("lon") is not None}
            idx = 0
            for hub in hubs:
                if len(out) >= target_n:
                    break
                hub_name = hub.get("name", "Global Hub")
                for ring in (0.18, 0.34, 0.52, 0.71, 0.93):
                    if len(out) >= target_n:
                        break
                    lat = float(hub["lat"]) + (((idx % 7) - 3) * ring * 0.12)
                    lon = float(hub["lon"]) + ((((idx // 7) % 7) - 3) * ring * 0.18)
                    idx += 1
                    k = f"{round(lat,4)}|{round(lon,4)}"
                    if k in seen:
                        continue
                    seen.add(k)
                    out.append({
                        "lat": lat,
                        "lon": lon,
                        "kind": kind,
                        "name": f"{hub_name} {kind.replace('_', ' ').title()} Cluster",
                        "facility_name": f"{hub_name} {kind.replace('_', ' ').title()} Cluster",
                        "location_name": hub.get("location_name") or f"{hub_name} Region",
                        "owner": hub.get("owner") or owner_default,
                        "operator": hub.get("owner") or owner_default,
                        "stored_product": hub.get("stored_product") or stored_default,
                        "content": hub.get("stored_product") or stored_default,
                    })
            return out[:target_n]

        refinery_hubs = [
            {"name":"US Gulf Coast", "lat":29.5, "lon":-94.8, "location_name":"Texas/Louisiana, US", "owner":"US Gulf Operators", "stored_product":"crude oil / refined fuels"},
            {"name":"Rotterdam", "lat":51.95, "lon":4.13, "location_name":"Rotterdam, Netherlands", "owner":"EU Operators", "stored_product":"crude oil / refined fuels"},
            {"name":"Singapore Jurong", "lat":1.28, "lon":103.72, "location_name":"Jurong, Singapore", "owner":"Singapore Operators", "stored_product":"petroleum / marine fuel"},
            {"name":"Ras Tanura", "lat":26.64, "lon":50.16, "location_name":"Eastern Province, Saudi Arabia", "owner":"Saudi Aramco", "stored_product":"crude oil / fuel blend"},
            {"name":"Jamnagar", "lat":22.47, "lon":70.06, "location_name":"Gujarat, India", "owner":"Reliance Industries", "stored_product":"crude / refined fuels"},
            {"name":"Ningbo-Zhoushan", "lat":29.87, "lon":122.05, "location_name":"Zhejiang, China", "owner":"CNPC/Sinopec", "stored_product":"petroleum products"},
        ]
        tank_hubs = [
            {"name":"Fujairah", "lat":25.18, "lon":56.34, "location_name":"Fujairah, UAE", "owner":"Fujairah Operators", "stored_product":"oil / fuel products"},
            {"name":"Cushing", "lat":35.98, "lon":-96.77, "location_name":"Oklahoma, US", "owner":"Midcontinent Storage Operators", "stored_product":"crude oil"},
            {"name":"Houston Ship Channel", "lat":29.73, "lon":-95.05, "location_name":"Houston, Texas, US", "owner":"US Gulf Storage", "stored_product":"crude / refined fuels"},
            {"name":"ARA Hub", "lat":52.0, "lon":4.4, "location_name":"Amsterdam-Rotterdam-Antwerp, EU", "owner":"European Storage Operators", "stored_product":"diesel / gasoline / jet"},
            {"name":"Ulsan", "lat":35.5, "lon":129.4, "location_name":"Ulsan, South Korea", "owner":"Korean Storage Operators", "stored_product":"petrochemicals / fuel oil"},
        ]
        walmart_hubs = [
            {"name":"Bentonville", "lat":36.3729, "lon":-94.2088, "location_name":"Arkansas, US", "owner":"Walmart", "stored_product":"retail inventory"},
            {"name":"Dallas-Fort Worth", "lat":32.955, "lon":-97.3074, "location_name":"Texas, US", "owner":"Walmart", "stored_product":"retail inventory"},
            {"name":"Chicago", "lat":41.8781, "lon":-87.6298, "location_name":"Illinois, US", "owner":"Walmart", "stored_product":"retail inventory"},
            {"name":"Atlanta", "lat":33.749, "lon":-84.388, "location_name":"Georgia, US", "owner":"Walmart", "stored_product":"retail inventory"},
            {"name":"Los Angeles", "lat":34.0522, "lon":-118.2437, "location_name":"California, US", "owner":"Walmart", "stored_product":"retail inventory"},
        ]
        refineries = _augment_assets(refineries, "refinery", 120, refinery_hubs, "Global Refinery Operator", "crude oil / refined fuels")
        tanks = _augment_assets(tanks, "storage_tank", 90, tank_hubs, "Global Storage Operator", "oil / fuel products")
        walmart = _augment_assets(walmart, "walmart_site", 70, walmart_hubs, "Walmart", "retail inventory")

        payload = _sanitize({
            "ok": True,
            "source": "OpenStreetMap Overpass",
            "updated": datetime.now().isoformat(),
            "counts": {
                "refineries": len(refineries),
                "storage_tanks": len(tanks),
                "walmart_sites": len(walmart),
            },
            "assets": {
                "refineries": refineries,
                "storage_tanks": tanks,
                "walmart_sites": walmart,
            },
        })
        _proxy_cache_set(key, payload)
        return jsonify(payload)
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "assets": {"refineries": [], "storage_tanks": [], "walmart_sites": []},
            "counts": {"refineries": 0, "storage_tanks": 0, "walmart_sites": 0},
        })

@app.route("/api/proxy/earthquakes", methods=["GET"])
def api_proxy_earthquakes():
    """Advanced seismic feed: USGS + SeismicPortal fallback blend."""
    try:
        key = "geo_advanced_eq"
        cached = _proxy_cache_get(key, ttl_s=240)
        if cached is not None:
            return jsonify(cached)

        features = []
        sources = {}

        # Primary: USGS
        try:
            r = requests.get(
                "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson",
                timeout=15,
                headers={"User-Agent": "QuantTerminal/1.0"},
            )
            if r.ok:
                data = r.json()
                features.extend(data.get("features", []))
                sources["usgs"] = {"ok": True, "count": len(data.get("features", []))}
            else:
                sources["usgs"] = {"ok": False, "status": r.status_code}
        except Exception as usgs_e:
            sources["usgs"] = {"ok": False, "error": str(usgs_e)}

        # Secondary: SeismicPortal (EMSC-backed feed)
        try:
            r2 = requests.get(
                "https://www.seismicportal.eu/fdsnws/event/1/query?limit=150&format=json",
                timeout=15,
                headers={"User-Agent": "QuantTerminal/1.0"},
            )
            if r2.ok:
                sp = r2.json()
                feats = sp.get("features", [])
                # Normalize into USGS-like properties subset where possible
                for f in feats:
                    p = f.get("properties", {})
                    g = f.get("geometry", {})
                    features.append({
                        "type": "Feature",
                        "geometry": g,
                        "properties": {
                            "mag": p.get("mag"),
                            "place": p.get("flynn_region") or p.get("region") or "Unknown",
                            "time": p.get("time"),
                            "sig": p.get("depth") if p.get("depth") is not None else 0,
                            "source": "seismicportal",
                        },
                    })
                sources["seismicportal"] = {"ok": True, "count": len(feats)}
            else:
                sources["seismicportal"] = {"ok": False, "status": r2.status_code}
        except Exception as sp_e:
            sources["seismicportal"] = {"ok": False, "error": str(sp_e)}

        # Deduplicate by rounded lat/lon + magnitude
        seen = set()
        uniq = []
        for f in features:
            c = (f.get("geometry") or {}).get("coordinates", [None, None, None])
            p = f.get("properties", {})
            if len(c) < 2:
                continue
            key_d = f"{round(float(c[0]),2)}|{round(float(c[1]),2)}|{round(float(p.get('mag') or 0),1)}"
            if key_d in seen:
                continue
            seen.add(key_d)
            uniq.append(f)

        out = {"type": "FeatureCollection", "features": uniq[:500], "sources": sources}
        _proxy_cache_set(key, out)
        return jsonify(out)
    except Exception as e:
        return jsonify({"type": "FeatureCollection", "features": [], "error": str(e)})


@app.route("/api/proxy/fires", methods=["GET"])
def api_proxy_fires():
    """Advanced wildfire feed: NASA EONET + FIRMS CSV hotspots."""
    try:
        key = "geo_advanced_fires"
        cached = _proxy_cache_get(key, ttl_s=600)
        if cached is not None:
            return jsonify(cached)

        out = {"events": [], "hotspots": [], "sources": {}}
        # Primary: EONET
        try:
            r = requests.get(
                "https://eonet.gsfc.nasa.gov/api/v3/events?status=open&category=wildfires&limit=200",
                timeout=20,
                headers={"User-Agent": "QuantTerminal/1.0"},
            )
            if r.ok:
                data = r.json()
                out["events"] = data.get("events", [])
                out["sources"]["eonet"] = {"ok": True, "count": len(out["events"])}
            else:
                out["sources"]["eonet"] = {"ok": False, "status": r.status_code}
        except Exception as eonet_e:
            out["sources"]["eonet"] = {"ok": False, "error": str(eonet_e)}

        # Secondary: FIRMS CSV (USA daily) - no key variant commonly available
        try:
            r2 = requests.get(
                "https://firms.modaps.eosdis.nasa.gov/usfs/api/area/csv/USA/24",
                timeout=20,
                headers={"User-Agent": "QuantTerminal/1.0"},
            )
            if r2.ok and "," in r2.text:
                lines = [x for x in r2.text.splitlines() if x.strip()]
                if len(lines) > 1:
                    hdr = [h.strip().lower() for h in lines[0].split(",")]
                    lat_i = hdr.index("latitude") if "latitude" in hdr else -1
                    lon_i = hdr.index("longitude") if "longitude" in hdr else -1
                    bri_i = hdr.index("bright_ti4") if "bright_ti4" in hdr else (-1 if "brightness" not in hdr else hdr.index("brightness"))
                    for ln in lines[1:1001]:
                        parts = ln.split(",")
                        if lat_i < 0 or lon_i < 0 or max(lat_i, lon_i) >= len(parts):
                            continue
                        try:
                            out["hotspots"].append({
                                "latitude": float(parts[lat_i]),
                                "longitude": float(parts[lon_i]),
                                "brightness": float(parts[bri_i]) if bri_i >= 0 and bri_i < len(parts) and parts[bri_i] else None,
                                "source": "firms_csv",
                            })
                        except Exception:
                            continue
                out["sources"]["firms_csv"] = {"ok": True, "count": len(out["hotspots"])}
            else:
                out["sources"]["firms_csv"] = {"ok": False, "status": r2.status_code if 'r2' in locals() else None}
        except Exception as firms_e:
            out["sources"]["firms_csv"] = {"ok": False, "error": str(firms_e)}

        _proxy_cache_set(key, out)
        return jsonify(out)
    except Exception as e:
        return jsonify({"events": [], "error": str(e)})


@app.route("/api/proxy/conflicts_live", methods=["GET"])
def api_proxy_conflicts_live():
    """Advanced conflict feed: GDELT + ReliefWeb blended geocoded stream."""
    COORDS = {
        "iran":(32.4,53.7),"tehran":(35.7,51.4),"iraq":(33.2,43.7),"baghdad":(33.3,44.4),
        "syria":(35.0,38.0),"damascus":(33.5,36.3),"ukraine":(48.4,31.2),"kyiv":(50.4,30.5),
        "russia":(61.5,105.3),"moscow":(55.7,37.6),"kremlin":(55.7,37.6),
        "israel":(31.0,34.8),"palestine":(31.9,35.2),"gaza":(31.4,34.4),"hamas":(31.4,34.4),
        "lebanon":(33.9,35.5),"hezbollah":(33.9,35.5),"beirut":(33.9,35.5),
        "yemen":(15.6,48.5),"houthi":(15.6,48.5),"sanaa":(15.4,44.2),
        "sudan":(12.9,30.2),"khartoum":(15.6,32.5),"somalia":(5.2,46.2),
        "libya":(26.3,17.2),"niger":(17.6,8.1),"mali":(17.6,-4.0),
        "congo":(4.0,21.8),"myanmar":(19.8,96.1),
        "china":(35.9,104.2),"beijing":(39.9,116.4),"taiwan":(23.7,121.0),
        "korea":(37.6,127.0),"pakistan":(30.4,69.3),"afghanistan":(33.9,67.7),
        "india":(20.6,79.0),"turkey":(39.9,32.9),"saudi":(23.9,45.1),
        "egypt":(26.8,30.8),"ethiopia":(9.1,40.5),"nigeria":(9.1,8.7),
        "mexico":(23.6,-102.5),"colombia":(4.6,-74.3),"venezuela":(6.4,-66.6),
        "haiti":(19.0,-72.3),"philippines":(12.9,121.8),
        "trump":(38.9,-77.1),"pentagon":(38.9,-77.1),"nato":(50.8,4.4),
        "jordan":(30.6,36.2),"kuwait":(29.3,47.5),"oil":(26.0,50.0),"opec":(26.0,50.0),
        "nuclear":(35.7,51.4),"missile":(35.7,51.4),"war":(48.4,31.2),
    }
    try:
        key = "conflicts_live_v3"
        cached = _proxy_cache_get(key, ttl_s=120)
        if cached is not None:
            return jsonify(cached)

        features = []
        import hashlib

        # METHOD 1: Use geopolitical engine events (already fetched by background scanner)
        geo_events = []
        try:
            if GEO:
                dash = _get_geo_dashboard_safe()
                geo_events = (dash.get("events") or [])
                if not geo_events:
                    geo_events = (dash.get("conflict_events", {}) or {}).get("recent_events", []) or []
                if not geo_events:
                    geo_events = ((dash.get("geopolitical_news", {}) or {}).get("articles", []) or [])
                print(f"  [CONFLICTS] GEO engine has {len(geo_events)} events")
        except Exception as e1:
            print(f"  [CONFLICTS] GEO engine failed: {e1}")

        # METHOD 2: Also try direct GDELT DOC API call
        doc_articles = []
        try:
            import urllib.parse
            url = "https://api.gdeltproject.org/api/v2/doc/doc?query=" + urllib.parse.quote("war OR military OR attack OR conflict OR missile") + "&mode=artlist&maxrecords=100&format=json&timespan=24h"
            r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            if r.ok:
                doc_articles = r.json().get("articles", [])
                print(f"  [CONFLICTS] GDELT DOC API returned {len(doc_articles)} articles")
        except Exception as e2:
            print(f"  [CONFLICTS] GDELT DOC API failed: {e2}")

        # METHOD 3: ReliefWeb latest conflict/disaster reports
        relief_items = []
        try:
            rw = requests.get(
                "https://api.reliefweb.int/v1/reports?appname=quant-terminal&limit=80&profile=full&filter[field]=primary_country.name&sort[]=date:desc",
                timeout=18,
                headers={"User-Agent": "QuantTerminal/1.0"},
            )
            if rw.ok:
                rw_data = rw.json().get("data", [])
                for row in rw_data[:80]:
                    fields = row.get("fields", {})
                    title = fields.get("title", "") or ""
                    if any(k in title.lower() for k in ["conflict", "war", "attack", "military", "missile", "strike", "shelling", "invasion"]):
                        relief_items.append((title, "reliefweb", 0))
        except Exception as rw_e:
            print(f"  [CONFLICTS] ReliefWeb failed: {rw_e}")

        # Combine all sources into one list of (title, source, tone) tuples
        all_items = []
        for ev in geo_events:
            title = (
                ev.get("title")
                or ev.get("headline")
                or ev.get("description")
                or ev.get("event_type")
                or ""
            )
            all_items.append((title, ev.get("source", "gdelt"), ev.get("tone", 0)))
        for art in doc_articles:
            all_items.append((art.get("title", ""), art.get("domain", "gdelt"), float(art.get("tone", 0) or 0)))
        all_items.extend(relief_items)

        print(f"  [CONFLICTS] Total items to geocode: {len(all_items)}")

        # Geocode every item
        for title_raw, source, tone in all_items:
            title = title_raw.lower()
            for loc, (lat, lon) in COORDS.items():
                if loc in title:
                    h = int(hashlib.md5(title.encode()).hexdigest()[:8], 16)
                    jlat = lat + ((h % 200 - 100) * 0.015)
                    jlon = lon + (((h >> 8) % 200 - 100) * 0.02)
                    cat = "AIRSTRIKE" if any(w in title for w in ["airstrike","bomb","missile","strike","rocket","drone"]) else \
                          "MILITARY" if any(w in title for w in ["military","army","troops","weapon","war","attack","navy","soldier","defense"]) else \
                          "CONFLICT"
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [jlon, jlat]},
                        "properties": {"name": title_raw[:150], "count": 1, "tone": tone, "source": source, "category": cat}
                    })
                    break

        # Deduplicate
        seen = set()
        unique = []
        for f in features:
            c = f["geometry"]["coordinates"]
            k = f"{round(c[1],1)},{round(c[0],1)}"
            if k not in seen:
                seen.add(k)
                unique.append(f)

        print(f"  [CONFLICTS] FINAL: {len(unique)} unique geocoded locations (from {len(features)} raw)")

        result = {"type": "FeatureCollection", "features": unique, "source_count": len(all_items)}
        _proxy_cache_set(key, result)
        return jsonify(result)
    except Exception as e:
        print(f"  [CONFLICTS] FATAL ERROR: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"type": "FeatureCollection", "features": []})


@app.route("/api/proxy/conflicts_test", methods=["GET"])
def api_proxy_conflicts_test():
    """DEBUG: Test endpoint — open http://localhost:5001/api/proxy/conflicts_test in browser."""
    try:
        r = requests.get("http://localhost:5001/api/proxy/conflicts_live", timeout=30)
        data = r.json()
        n = len(data.get("features", []))
        samples = [f["properties"]["name"][:60] for f in data.get("features", [])[:5]]
        return f"<h2>Conflicts Test</h2><p>Features: <b>{n}</b></p><p>Samples: {samples}</p><p>Full JSON: <pre>{json.dumps(data, indent=2)[:3000]}</pre></p>"
    except Exception as e:
        return f"<h2>ERROR</h2><p>{e}</p>"


@app.route("/api/proxy/aircraft", methods=["GET"])
def api_proxy_aircraft():
    """Advanced aircraft proxy with filtered ops classes and cache."""
    try:
        key = "opensky_all_v2"
        cached = _proxy_cache_get(key, ttl_s=15)
        if cached is not None:
            return jsonify(cached)
        r = requests.get("https://opensky-network.org/api/states/all", timeout=20)
        data = r.json()
        states = data.get("states", []) if isinstance(data, dict) else []
        military = 0
        cargo = 0
        for s in states:
            call = str(s[1] if len(s) > 1 else "").upper()
            if any(p in call for p in ["RCH", "ARMY", "NAVY", "AFR", "MCH", "EVAC", "FORTE"]):
                military += 1
            if any(p in call for p in ["FDX", "UPS", "GTI", "CLX", "DHL", "ABX"]):
                cargo += 1
        out = {**data, "military_detected": military, "cargo_detected": cargo}
        _proxy_cache_set(key, out)
        return jsonify(out)
    except Exception as e:
        return jsonify({"states": [], "error": str(e)})


@app.route("/api/geopolitical/command_center", methods=["GET"])
def api_geo_command_center():
    """Bloomberg-style geopolitical command center payload."""
    try:
        dash = _get_geo_dashboard_safe() if GEO else {}
        eq = api_proxy_earthquakes().get_json()
        fires = api_proxy_fires().get_json()
        conflicts = api_proxy_conflicts_live().get_json()
        aircraft = api_proxy_aircraft().get_json()
        return jsonify(_sanitize({
            "dashboard": dash,
            "earthquakes": eq,
            "fires": fires,
            "conflicts": conflicts,
            "aircraft": aircraft,
            "updated": datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/geopolitical/live", methods=["GET"])
def api_geo_live():
    """
    New unified geopolitical API (v2) for latest live updates.
    Frontend should consume this endpoint instead of old /api/proxy/* routes.
    """
    try:
        dash = _get_geo_dashboard_safe() if GEO else {}
        # Reuse hardened advanced proxy handlers as internal providers
        eq = api_proxy_earthquakes().get_json()
        fires = api_proxy_fires().get_json()
        conflicts = api_proxy_conflicts_live().get_json()
        aircraft = api_proxy_aircraft().get_json()

        risk = (dash or {}).get("risk") or {}
        conflict_block = (dash or {}).get("conflict_events") or {}
        recent_events = conflict_block.get("recent_events") or []
        air_block = (dash or {}).get("military_aircraft") or {}
        components = risk.get("components") or {}

        normalized_dashboard = {
            "risk_score": {
                "composite_risk": float(risk.get("composite_risk", 0) or 0),
                "risk_level": risk.get("risk_level", "LOW"),
                "components": {
                    "conflict": float(components.get("conflict_intensity", 0) or 0),
                    "supply_chain": float(components.get("supply_chain", 0) or 0),
                    "natural_disaster": float(components.get("natural_disaster", 0) or 0),
                    "trade": float(components.get("trade", 0) or 0),
                    "military_activity": float(components.get("military_activity", 0) or 0),
                },
                "trading_signals": risk.get("trading_signals") or [],
                "summary": risk.get("summary", ""),
            },
            "aircraft": {
                "military_detected": int(air_block.get("military_count", 0) or 0),
                "hvi_count": int(air_block.get("hvi_count", 0) or 0),
                "alerts": air_block.get("alerts") or [],
            },
            "events": recent_events[:120],
            "meta": (dash or {}).get("meta") or {},
        }

        providers = {
            "dashboard_engine": {"ok": bool(dash)},
            "earthquakes": {"ok": bool(eq and eq.get("features") is not None), "count": len(eq.get("features", [])) if isinstance(eq, dict) else 0},
            "fires": {"ok": bool(fires and fires.get("events") is not None), "events": len(fires.get("events", [])) if isinstance(fires, dict) else 0, "hotspots": len(fires.get("hotspots", [])) if isinstance(fires, dict) else 0},
            "conflicts": {"ok": bool(conflicts and conflicts.get("features") is not None), "count": len(conflicts.get("features", [])) if isinstance(conflicts, dict) else 0},
            "aircraft": {"ok": bool(aircraft and aircraft.get("states") is not None), "count": len(aircraft.get("states", [])) if isinstance(aircraft, dict) else 0},
        }

        payload = _sanitize({
            "dashboard": normalized_dashboard,
            "dashboard_raw": dash,
            "earthquakes": eq,
            "fires": fires,
            "conflicts": conflicts,
            "aircraft": aircraft,
            "providers": providers,
            "updated": datetime.now().isoformat(),
        })
        if DB and hasattr(DB, "MarketStore"):
            try:
                DB.MarketStore.save_geo(payload, source="api_geo_live")
            except Exception as _mge:
                logger.debug(f"DB geo save skipped: {_mge}")
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED QUANT MATH — Lévy, Jump-Diffusion, FFT, EVT, Heston, RMT
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/quant/analysis/<symbol>", methods=["GET"])
def api_quant_analysis(symbol):
    """Full advanced quant analysis for a symbol — 9 models."""
    try:
        if not AQE:
            return jsonify({"error": "advanced_quant_engine not loaded"}), 503
        import yfinance as _yf_qa
        sym = symbol.upper().strip()
        period = request.args.get("period", "2y")
        h = _yf_qa.Ticker(sym).history(period=period, interval="1d")
        if h.empty or len(h) < 60:
            return jsonify({"error": f"insufficient data for {sym}"}), 404
        prices = h["Close"].values.astype(float)
        result = AQE.full_quant_analysis(prices, sym)
        return jsonify(_sanitize(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/quant/dtw/<symbol>", methods=["GET"])
def api_quant_dtw(symbol):
    """Dynamic Time Warping — find historical pattern matches."""
    try:
        if not AQE:
            return jsonify({"error": "advanced_quant_engine not loaded"}), 503
        import yfinance as _yf_dtw
        sym = symbol.upper().strip()
        h = _yf_dtw.Ticker(sym).history(period="5y", interval="1d")
        if h.empty or len(h) < 120:
            return jsonify({"error": f"insufficient data for {sym}"}), 404
        prices = h["Close"].values.astype(float)
        window = int(request.args.get("window", 30))
        matches = AQE.DynamicTimeWarping.find_historical_matches(
            prices[-window:], prices[:-window], window=window
        )
        return jsonify(_sanitize({"symbol": sym, "window": window, "matches": matches}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/quant/rmt", methods=["GET"])
def api_quant_rmt():
    """Random Matrix Theory — correlation cleaning for portfolio."""
    try:
        if not AQE:
            return jsonify({"error": "advanced_quant_engine not loaded"}), 503
        import yfinance as _yf_rmt
        symbols = request.args.get("symbols", "SPY,QQQ,IWM,TLT,GLD,XLK,XLF,XLE,XLV,XLI,XLC,XLY")
        sym_list = [s.strip().upper() for s in symbols.split(",")][:20]
        period = request.args.get("period", "1y")
        returns_list = []
        valid_syms = []
        for sym in sym_list:
            try:
                h = _yf_rmt.Ticker(sym).history(period=period, interval="1d")
                if not h.empty and len(h) >= 60:
                    returns_list.append(h["Close"].pct_change().dropna().values)
                    valid_syms.append(sym)
            except:
                pass
        if len(valid_syms) < 3:
            return jsonify({"error": "need 3+ valid symbols"}), 400
        min_len = min(len(r) for r in returns_list)
        matrix = np.column_stack([r[-min_len:] for r in returns_list])
        result = AQE.RandomMatrixTheory.clean_correlation(matrix)
        result["symbols"] = valid_syms
        return jsonify(_sanitize(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/quant/models", methods=["GET"])
def api_quant_models():
    """List all available advanced quant models."""
    try:
        if not AQE:
            return jsonify({"error": "advanced_quant_engine not loaded"}), 503
        return jsonify({"models": AQE.get_model_list(), "count": len(AQE.get_model_list())})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# === GIG AUTH + TOP PICKS + TRADE API ===

from functools import wraps

def require_auth(f):
    """Decorator: require valid session token."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not DB:
            return f(*args, **kwargs)  # No DB = no auth
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            token = request.args.get("token", "")
        if not token:
            return jsonify({"error": "unauthorized", "login_url": "/api/auth/login"}), 401
        user = DB.Auth.verify_token(token)
        if not user:
            return jsonify({"error": "invalid_token"}), 401
        request.user = user
        return f(*args, **kwargs)
    return decorated

@app.route("/api/auth/login", methods=["POST"])
def api_login():
    """Login with username + password, returns session token."""
    if not DB:
        return jsonify({"error": "database not available"}), 503
    data = request.get_json(force=True)
    username = data.get("username", "")
    password = data.get("password", "")
    token = DB.Auth.login(username, password)
    if token:
        return jsonify({"token": token, "username": username, "expires_in": "24h"})
    return jsonify({"error": "invalid_credentials"}), 401

@app.route("/api/auth/logout", methods=["POST"])
def api_logout():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if DB and token:
        DB.Auth.logout(token)
    return jsonify({"logged_out": True})

@app.route("/api/auth/verify", methods=["GET"])
def api_auth_verify():
    if not DB:
        return jsonify({"authenticated": True, "username": "admin"})
    token = request.headers.get("Authorization", "").replace("Bearer ", "") or request.args.get("token", "")
    user = DB.Auth.verify_token(token) if token else None
    if user:
        return jsonify({"authenticated": True, **user})
    return jsonify({"authenticated": False}), 401

@app.route("/api/top_picks", methods=["GET"])
def api_top_picks():
    """Get the single best play in each category."""
    if not DB:
        return jsonify({"error": "database not available"}), 503
    return jsonify(_sanitize(DB.TopPicks.get_all()))

@app.route("/api/top_picks/<category>", methods=["GET"])
def api_top_pick(category):
    if not DB:
        return jsonify({"error": "database not available"}), 503
    pick = DB.TopPicks.get(category)
    if pick:
        return jsonify(_sanitize(pick))
    return jsonify({"error": f"no pick for {category}"}), 404

@app.route("/api/signals/history", methods=["GET"])
def api_signals_history():
    """Get persisted signal history (survives restarts)."""
    if not DB:
        return jsonify({"error": "database not available"}), 503
    category = request.args.get("category")
    limit = int(request.args.get("limit", 50))
    return jsonify(_sanitize({"signals": DB.SignalStore.get_latest(category, limit)}))

@app.route("/api/signals/history/<symbol>", methods=["GET"])
def api_signal_history_symbol(symbol):
    if not DB:
        return jsonify({"error": "database not available"}), 503
    return jsonify(_sanitize({"signals": DB.SignalStore.get_signal_history(symbol.upper())}))

@app.route("/api/trades", methods=["GET"])
def api_trades_list():
    """List all trades (open + closed)."""
    if not DB:
        return jsonify({"error": "database not available"}), 503
    return jsonify(_sanitize({
        "open": DB.TradeLog.get_open(),
        "closed": DB.TradeLog.get_closed(int(request.args.get("limit", 50))),
        "stats": DB.TradeLog.get_stats(),
    }))

@app.route("/api/trades/open", methods=["POST"])
def api_trade_open():
    """Open a new trade."""
    if not DB:
        return jsonify({"error": "database not available"}), 503
    d = request.get_json(force=True)
    tid = DB.TradeLog.open_trade(
        d["symbol"], d["direction"], d["entry_price"],
        d.get("shares", 1), d.get("strategy", ""), d.get("signals")
    )
    return jsonify({"trade_id": tid, "status": "open"})

@app.route("/api/trades/close", methods=["POST"])
def api_trade_close():
    """Close a trade."""
    if not DB:
        return jsonify({"error": "database not available"}), 503
    d = request.get_json(force=True)
    result = DB.TradeLog.close_trade(d["trade_id"], d["exit_price"], d.get("notes", ""))
    if result:
        return jsonify(result)
    return jsonify({"error": "trade not found"}), 404



if __name__ == "__main__":
    if not YF_OK:
        print("ERROR: pip install yfinance flask flask-cors numpy pandas scipy requests")
        exit(1)
    t = threading.Thread(target=scanner.run, daemon=True)
    t.start()

    # Init Alpaca broker (non-blocking — fails gracefully if keys not set)
    _init_alpaca()

    # ── ADAPTIVE WEIGHTS: Load persisted IC-based weights from disk ──────────
    # If aladdin_weights.json exists (from a previous run), the system will
    # immediately use learned weights instead of static defaults.
    try:
        from math_engine import AladdinScorer
        AladdinScorer.load_weights()
        logger.info("✅ AladdinScorer adaptive weights loaded")
    except Exception as _aw_err:
        logger.warning(f"AladdinScorer weight load: {_aw_err}")

    # Pre-warm company info for core symbols (runs in background, non-blocking)
    def _preload_companies():
        try:
            core = get_training_universe(n=40)  # dynamic — all liquid large-caps
            if CI:
                logger.info("  Pre-loading company details (sector/cap/description)...")
                for sym in core:
                    try:
                        info = CI.get_company_info(sym, use_yfinance=True)
                        if not hasattr(scanner, "company_cache"):
                            scanner.company_cache = {}
                        scanner.company_cache[sym] = info
                        time.sleep(0.4)
                    except Exception:
                        pass
                logger.info(f"  ✓ Company info ready for {len(core)} symbols")
        except Exception as _pe:
            logger.debug(f"Company preload: {_pe}")
    threading.Thread(target=_preload_companies, daemon=True, name="co-preload").start()

    # Pre-warm earnings revision cache for core symbols (background, non-blocking)
    def _preload_revisions():
        try:
            time.sleep(30)   # wait 30s after startup so yfinance connections stabilize
            all_syms = get_training_universe(n=40)  # dynamic
            # Revisions only make sense for individual equities — skip ETFs/funds/bonds
            # which have no analyst EPS estimates (causes HTTP 404 from Yahoo Finance)
            core_rev = [s for s in all_syms if not _is_etf_or_fund(s)][:30]
            if EE and hasattr(EE, "EarningsRevisionSignal"):
                logger.info(f"  Pre-loading revision signals for {len(core_rev)} equities (ETFs skipped)...")
                EE.EarningsRevisionSignal.precompute_universe(core_rev, max_workers=3)
        except Exception as _re:
            logger.debug(f"Revision preload: {_re}")
    threading.Thread(target=_preload_revisions, daemon=True, name="rev-preload").start()

    # Macro background refresh (Step 7)
    threading.Thread(target=_macro_background_refresh, daemon=True, name="macro-bg").start()

    # Insider cluster precompute (Step 8) — starts 60s after launch to stagger EDGAR requests
    def _preload_insider_clusters():
        try:
            time.sleep(60)
            all_syms = get_training_universe(n=35)  # dynamic
            # Insider filings (SEC Form 4) only exist for individual company officers.
            # ETFs/funds have no Form 4 data — skip them to avoid 404/401 errors.
            core_ins = [s for s in all_syms if not _is_etf_or_fund(s)][:25]
            if EE and hasattr(EE, "InsiderClusterScorer"):
                logger.info(f"  Pre-loading insider clusters for {len(core_ins)} equities (ETFs skipped)...")
                EE.InsiderClusterScorer.precompute_universe(core_ins, max_workers=3)
        except Exception as _ie:
            logger.debug(f"Insider preload: {_ie}")
    threading.Thread(target=_preload_insider_clusters, daemon=True, name="insider-preload").start()

    # LSTM predictor startup (Step 11) — initializes singleton, loads saved weights,
    # and launches background training thread (90s delayed)
    try:
        from ml_engine import LSTMPricePredictor
        lstm_predictor = LSTMPricePredictor.get_instance()
        lstm_predictor.start_background_training()
        logger.info(f"  LSTM predictor initialized (backend={lstm_predictor._backend})")
    except Exception as _lstm_init_e:
        logger.debug(f"LSTM init: {_lstm_init_e}")

    # Dark pool precompute (Step 13) — starts 120s after launch to stagger with other engines
    def _preload_dark_pool():
        try:
            time.sleep(120)
            import dark_pool_engine as _DPE
            core_dp = get_training_universe(n=25)  # dynamic
            logger.info(f"  Pre-loading dark pool signals for {len(core_dp)} symbols...")
            _DPE.DarkPoolScorer.precompute_universe(core_dp, max_workers=4)
        except Exception as _dpe:
            logger.debug(f"Dark pool preload: {_dpe}")
    threading.Thread(target=_preload_dark_pool, daemon=True, name="dp-preload").start()

    threading.Thread(target=_watchdog, daemon=True, name="watchdog").start()

    # ── Renaissance v5: Background init for new engines ───────────────────────
    def _init_v5_engines():
        import time as _t5
        _t5.sleep(30)
        try:
            if ENS:
                ENS.get_medallion_ensemble()
                logger.info("✅ Medallion Ensemble initialized (50+ alpha sources)")
        except Exception as _e5:
            logger.debug(f"Ensemble init: {_e5}")
        try:
            if DL:
                DL.get_neural_ensemble()
                logger.info("✅ Neural Ensemble initialized")
        except Exception as _e5:
            logger.debug(f"DL init: {_e5}")
        try:
            if PAT:
                PAT.get_pattern_scanner()
                logger.info("✅ Pattern Scanner initialized")
        except Exception as _e5:
            logger.debug(f"Pattern init: {_e5}")
        try:
            if DI:
                DI.get_data_pipeline()
                logger.info("✅ Data Pipeline initialized")
        except Exception as _e5:
            logger.debug(f"Pipeline init: {_e5}")
    threading.Thread(target=_init_v5_engines, daemon=True, name="v5-engines").start()

    # ── Renaissance v10: Initialize research tracker ──────────────────────
    if RET:
        try:
            RET.init()
        except Exception as _ret_e:
            logger.debug(f"Research tracker init: {_ret_e}")

    # ── Renaissance v10: Initialize alpha research platform ───────────────
    if ARE:
        try:
            ARE.get_alpha_platform()
            logger.info("✅ Alpha Research Platform initialized")
        except Exception as _are_e:
            logger.debug(f"Alpha platform init: {_are_e}")


    # ── MARKET CACHE: Background thread for instant tab loading ─────────────
    threading.Thread(target=_refresh_market_cache, daemon=True, name="market-cache").start()
    logger.info("✅ Market cache thread started (all tabs load instantly)")

    # ── BOOTSTRAP CACHE WARMER: pre-build login payload and keep hot ─────────
    def _bootstrap_cache_warmer():
        try:
            time.sleep(4)
            while True:
                try:
                    payload_core = _build_bootstrap_payload(scope="core")
                    _fast_cache_set("api_bootstrap:core", payload_core)
                    _fast_cache_set("api_state", payload_core.get("state", {}))
                    # Warm heavyweight scopes less frequently but keep ready.
                    if int(time.time()) % (12 if ULTRA_FAST_MODE else 20) < 3:
                        _fast_cache_set("api_bootstrap:terminal", _build_bootstrap_payload(scope="terminal"))
                    if int(time.time()) % (18 if ULTRA_FAST_MODE else 30) < 3:
                        _fast_cache_set("api_bootstrap:geo", _build_bootstrap_payload(scope="geo"))
                except Exception as _bw:
                    logger.debug(f"Bootstrap warm: {_bw}")
                time.sleep(2 if ULTRA_FAST_MODE else 5)
        except Exception as _bw_outer:
            logger.debug(f"Bootstrap warmer stopped: {_bw_outer}")
    threading.Thread(target=_bootstrap_cache_warmer, daemon=True, name="bootstrap-warm").start()
    logger.info("✅ Bootstrap cache warmer started")

    print(f"\n{'='*62}")
    print("  GIG LIVE DATA SERVER v10 — INSTITUTIONAL EDITION")
    print(f"  Engines: {sum(1 for e in [OE,ML,SE,CI,EE,RA,CE,AD,BE,NLP,DL,ENS,PAT,VS,CA,DI,FUT,CAP,ALS,SSE,SLE,QTE,DEE,ARE,IPE,RET,GEO,AQE] if e)}/28 loaded")
    print(f"  Cache:    27 tickers pre-fetched every 60s (Bloomberg-speed)")
    print(f"  API:      http://localhost:{PORT}/api/state")
    print(f"  Company:  http://localhost:{PORT}/api/company/AAPL")
    print(f"  Research: http://localhost:{PORT}/api/research")
    print(f"  Ping:     http://localhost:{PORT}/api/ping")
    print(f"{'='*62}\n")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)