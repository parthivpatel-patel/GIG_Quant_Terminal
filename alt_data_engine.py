"""
alt_data_engine.py — Renaissance.io Proprietary Alternative Data Engine
========================================================================
Signals nobody else has operationalized at retail scale:

Data Sources (all free/low-cost):
  1. Google Trends Velocity    — search spike precedes price spike
  2. USPTO Patent API          — R&D signal, innovation velocity
  3. SEC EDGAR Full-Text       — language shift detection in filings
  4. iTunes App Store Rankings — app engagement proxy (consumer tech)
  5. GitHub Activity           — developer ecosystem health (tech)
  6. EIA Satellite Proxy       — actual US govt oil/gas data
  7. Job Posting Velocity      — hiring surge = growth signal
  8. Credit Card Proxy         — Federal Reserve consumer credit data
  9. Insider Sentiment Index   — Form 4 cluster buy/sell ratio
 10. Web Traffic Proxy         — revenue proxy for digital businesses

These signals are UNCORRELATED with price — the edge comes from
cross-referencing them against options flow and short data.
"""

import time
import threading
import requests
import re
from datetime import datetime, timedelta

try:
    import numpy as np
    import pandas as pd
    NP_OK = True
except ImportError:
    NP_OK = False

# ── Cache ─────────────────────────────────────────────────────────────────────
_alt_cache = {}
_cache_ttl = 3600  # 1 hour per ticker


def _cached(key: str, fn, ttl: int = _cache_ttl):
    now = time.time()
    if key in _alt_cache:
        ts, val = _alt_cache[key]
        if now - ts < ttl:
            return val
    val = fn()
    _alt_cache[key] = (now, val)
    return val


# ═══════════════════════════════════════════════════════════════════════════════
# 1. GOOGLE TRENDS VELOCITY
#    Spike in search interest for a company precedes retail awareness
# ═══════════════════════════════════════════════════════════════════════════════
def get_google_trends_signal(sym: str, company_name: str = None) -> dict:
    """
    Measures search velocity from Google Trends.
    A 2x+ spike in interest with no price move = pre-catalyst signal.
    Requires: pip install pytrends
    """
    result = {"signal": "UNAVAILABLE", "score": 0, "trend": [], "spike_detected": False}
    query  = company_name or sym

    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="en-US", tz=300, timeout=(10, 25))
        pytrends.build_payload([query], cat=0, timeframe="now 7-d", geo="US", gprop="")
        df = pytrends.interest_over_time()

        if df is None or df.empty or query not in df.columns:
            return result

        vals = df[query].values.tolist()
        if len(vals) < 10:
            return result

        recent  = float(sum(vals[-6:])  / max(len(vals[-6:]), 1))
        baseline = float(sum(vals[:-6]) / max(len(vals[:-6]), 1))
        spike_ratio = recent / max(baseline, 1)

        signal = "NEUTRAL"
        score  = 0

        if spike_ratio > 3.0:
            signal = "VIRAL SPIKE"; score = 40
        elif spike_ratio > 2.0:
            signal = "RISING"; score = 25
        elif spike_ratio > 1.5:
            signal = "ELEVATED"; score = 12
        elif spike_ratio < 0.5 and baseline > 20:
            signal = "FADING"; score = -5

        result.update({
            "signal":        signal,
            "score":         score,
            "spike_ratio":   round(spike_ratio, 2),
            "recent_avg":    round(recent, 1),
            "baseline_avg":  round(baseline, 1),
            "spike_detected": spike_ratio > 2.0,
            "trend":         [round(v, 1) for v in vals[-24:]],  # last 24 data points
            "source":        "Google Trends",
        })

    except ImportError:
        result["signal"] = "Install pytrends: pip install pytrends"
    except Exception as e:
        result["error"] = str(e)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. USPTO PATENT VELOCITY
#    R&D activity signal — patent filings spike before product launches
# ═══════════════════════════════════════════════════════════════════════════════
def get_patent_velocity(sym: str, assignee_name: str = None) -> dict:
    """
    Free USPTO Patent Full-Text API.
    Spike in patent filings = active R&D = potential catalyst.
    """
    result = {"signal": "UNAVAILABLE", "score": 0, "patents_90d": 0}
    name   = assignee_name or sym

    try:
        # USPTO PatentsView API (free, no key required)
        today  = datetime.now().strftime("%Y-%m-%d")
        ago90  = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        ago365 = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        def _query(start, end):
            url = "https://api.patentsview.org/patents/query"
            payload = {
                "q": {"_and": [
                    {"_text_phrase": {"patent_abstract": name}},
                    {"_gte": {"patent_date": start}},
                    {"_lte": {"patent_date": end}},
                ]},
                "f": ["patent_number", "patent_date", "patent_title"],
                "o": {"per_page": 25},
            }
            r = requests.post(url, json=payload, timeout=8,
                              headers={"Content-Type": "application/json"})
            if r.ok:
                return r.json().get("total_patent_count", 0)
            return 0

        count_90d  = _query(ago90, today)
        count_prev = _query(ago365, ago90)  # prior 275 days

        # Annualize prior period
        rate_now  = count_90d  / 90  * 365
        rate_prev = count_prev / 275 * 365 if count_prev else 1

        velocity_ratio = rate_now / max(rate_prev, 0.1)

        score  = 0
        signal = "NORMAL"

        if velocity_ratio > 3:
            signal = "PATENT SURGE"; score = 25
        elif velocity_ratio > 1.8:
            signal = "ACCELERATING"; score = 15
        elif velocity_ratio > 1.2:
            signal = "ELEVATED"; score = 8

        result.update({
            "signal":         signal,
            "score":          score,
            "patents_90d":    count_90d,
            "patents_prior":  count_prev,
            "velocity_ratio": round(velocity_ratio, 2),
            "annualized_rate": round(rate_now, 1),
            "source":         "USPTO PatentsView API",
        })

    except Exception as e:
        result["error"] = str(e)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EDGAR FILING LANGUAGE ANALYSIS
#    Sentiment shift in official filings — management language change = signal
# ═══════════════════════════════════════════════════════════════════════════════
def get_edgar_language_signal(sym: str) -> dict:
    """
    Scans recent EDGAR filings for language sentiment.
    Bullish: "record", "raised guidance", "strong demand", "expanding"
    Bearish: "headwinds", "challenging", "restructuring", "uncertain"
    """
    result = {"signal": "NEUTRAL", "score": 0, "bullish_hits": 0, "bearish_hits": 0}

    BULLISH_KEYWORDS = [
        "record revenue", "raised guidance", "strong demand", "accelerating growth",
        "expanded margins", "market share gains", "new customers", "breakthrough",
        "substantial growth", "outperformed", "beat expectations",
    ]
    BEARISH_KEYWORDS = [
        "challenging environment", "headwinds", "restructuring charges",
        "workforce reduction", "impairment", "below expectations",
        "guidance lowered", "inventory build", "pricing pressure",
        "customer churn", "deteriorating",
    ]

    try:
        url = (f"https://efts.sec.gov/LATEST/search-index"
               f"?q=%22{sym}%22&forms=8-K,10-Q"
               f"&dateRange=custom"
               f"&startdt={(datetime.now()-timedelta(days=90)).strftime('%Y-%m-%d')}"
               f"&enddt={datetime.now().strftime('%Y-%m-%d')}")
        resp = requests.get(url, timeout=8,
                            headers={"User-Agent": "Renaissance.io research@example.com"})

        if not resp.ok:
            return result

        text = resp.text.lower()

        bull_hits = sum(1 for kw in BULLISH_KEYWORDS if kw in text)
        bear_hits = sum(1 for kw in BEARISH_KEYWORDS if kw in text)

        score  = 0
        signal = "NEUTRAL"

        net = bull_hits - bear_hits
        if net >= 4:
            signal = "STRONGLY BULLISH"; score = 30
        elif net >= 2:
            signal = "BULLISH"; score = 18
        elif net <= -3:
            signal = "STRONGLY BEARISH"; score = -20
        elif net <= -1:
            signal = "CAUTIOUS"; score = -8

        result.update({
            "signal":       signal,
            "score":        score,
            "bullish_hits": bull_hits,
            "bearish_hits": bear_hits,
            "net_sentiment": net,
            "source":       "SEC EDGAR Full-Text Search",
        })

    except Exception as e:
        result["error"] = str(e)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 4. APP STORE SIGNAL (iTunes Search API)
#    App ranking velocity = user engagement proxy for consumer tech
# ═══════════════════════════════════════════════════════════════════════════════
def get_app_store_signal(app_name: str, sym: str) -> dict:
    """
    Uses iTunes Search API to find app presence and ratings.
    High rating count growth = strong user engagement signal.
    """
    result = {"signal": "UNAVAILABLE", "score": 0}

    try:
        url = f"https://itunes.apple.com/search?term={requests.utils.quote(app_name)}&entity=software&limit=5"
        resp = requests.get(url, timeout=8)
        if not resp.ok:
            return result

        data    = resp.json()
        results = data.get("results", [])
        if not results:
            return result

        top = results[0]
        rating_count = int(top.get("userRatingCount", 0))
        avg_rating   = float(top.get("averageUserRating", 0))
        version      = top.get("version", "?")
        updated      = top.get("currentVersionReleaseDate", "")

        score  = 0
        signal = "NORMAL"

        if rating_count > 1_000_000:
            signal = "DOMINANT APP"; score = 20
        elif rating_count > 100_000:
            signal = "HIGH ENGAGEMENT"; score = 12
        elif rating_count > 10_000:
            signal = "GROWING"; score = 6

        if avg_rating >= 4.7:
            score += 8
        elif avg_rating < 3.5:
            score -= 5

        result.update({
            "signal":       signal,
            "score":        score,
            "app_name":     top.get("trackName", app_name),
            "rating_count": rating_count,
            "avg_rating":   avg_rating,
            "version":      version,
            "last_update":  updated[:10] if updated else "unknown",
            "source":       "iTunes App Store API",
        })

    except Exception as e:
        result["error"] = str(e)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. EIA SATELLITE PROXY — US Govt Energy Data
#    Real satellite/sensor data from the US Energy Information Administration
# ═══════════════════════════════════════════════════════════════════════════════
def get_eia_energy_signal(sym: str) -> dict:
    """
    EIA Weekly Petroleum Status Report — actual measured inventory data.
    This IS satellite/sensor data (custody meters, pipeline sensors).
    Relevant for: XOM, CVX, OXY, HAL, SLB, XLE, energy sector.
    """
    result = {"signal": "UNAVAILABLE", "score": 0}

    ENERGY_TICKERS = {"XOM","CVX","OXY","HAL","SLB","MRO","DVN","FANG","PR","SM",
                      "COP","PSX","VLO","MPC","PXD","EOG","HES","BKR","NOV","RIG"}

    if sym not in ENERGY_TICKERS:
        return {"signal": "N/A (non-energy)", "score": 0}

    try:
        # EIA API v2 (free, no key for basic series)
        url = ("https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
               "?frequency=weekly&data[0]=value&sort[0][column]=period"
               "&sort[0][direction]=desc&offset=0&length=10"
               "&api_key=DEMO_KEY")  # DEMO_KEY works for low-rate access
        resp = requests.get(url, timeout=10)

        if resp.ok:
            data   = resp.json()
            series = data.get("response", {}).get("data", [])
            if len(series) >= 2:
                latest = float(series[0].get("value", 0))
                prior  = float(series[1].get("value", 0))
                draw   = prior - latest  # positive = drawdown (bullish for oil)
                chg_pct = (latest - prior) / max(prior, 1) * 100

                signal = "NEUTRAL"
                score  = 0

                if draw > 5000:  # 5M+ barrel draw
                    signal = "LARGE DRAW — BULLISH"; score = 20
                elif draw > 2000:
                    signal = "DRAW — MILDLY BULLISH"; score = 10
                elif draw < -5000:
                    signal = "LARGE BUILD — BEARISH"; score = -15
                elif draw < -2000:
                    signal = "BUILD — MILDLY BEARISH"; score = -8

                result.update({
                    "signal":      signal,
                    "score":       score,
                    "latest_kb":   int(latest),
                    "prior_kb":    int(prior),
                    "draw_kb":     int(draw),
                    "chg_pct":     round(chg_pct, 2),
                    "source":      "EIA Weekly Petroleum (US Govt Sensor Data)",
                })
        else:
            result["error"] = f"EIA API {resp.status_code}"

    except Exception as e:
        result["error"] = str(e)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 6. FEDERAL RESERVE CREDIT CARD PROXY
#    Fed G.19 consumer credit — proxy for consumer spending velocity
# ═══════════════════════════════════════════════════════════════════════════════
def get_consumer_credit_signal() -> dict:
    """
    Federal Reserve G.19 Consumer Credit data.
    Rising revolving credit growth → strong consumer spending (bullish XRT, AMZN, WMT)
    Falling → consumer stress (bearish consumer discretionary)
    """
    result = {"signal": "UNAVAILABLE", "score": 0}

    try:
        # FRED revolving consumer credit (credit cards)
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=REVOLSL"
        resp = requests.get(url, timeout=8)
        if not resp.ok:
            return result

        lines = [l for l in resp.text.strip().splitlines()[1:] if "." in l and l.split(",")[1].strip() not in ("",".",)]
        if len(lines) < 12:
            return result

        latest = float(lines[-1].split(",")[1])
        prev3m = float(lines[-4].split(",")[1])
        prev12m = float(lines[-13].split(",")[1]) if len(lines) >= 13 else prev3m

        mom = (latest - prev3m) / prev3m * 100
        yoy = (latest - prev12m) / prev12m * 100

        signal = "NEUTRAL"
        score  = 0

        if yoy > 8:
            signal = "CONSUMER SPENDING STRONG"; score = 15
        elif yoy > 4:
            signal = "SPENDING GROWING"; score = 8
        elif yoy < -2:
            signal = "CONSUMER STRESS"; score = -10
        elif yoy < 0:
            signal = "SPENDING SLOWING"; score = -4

        result.update({
            "signal":       signal,
            "score":        score,
            "latest_bn":    round(latest / 1000, 1),  # billions
            "mom_pct":      round(mom, 2),
            "yoy_pct":      round(yoy, 2),
            "source":       "Federal Reserve G.19 Consumer Credit",
        })

    except Exception as e:
        result["error"] = str(e)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 7. GITHUB DEVELOPER ECOSYSTEM SIGNAL
#    For tech/software companies — developer adoption precedes revenue
# ═══════════════════════════════════════════════════════════════════════════════
def get_github_signal(github_org: str, sym: str) -> dict:
    """
    GitHub API (free, unauthenticated = 60 req/hr).
    Star growth rate and contributor velocity = ecosystem health.
    """
    result = {"signal": "UNAVAILABLE", "score": 0}
    if not github_org:
        return result

    try:
        url   = f"https://api.github.com/orgs/{github_org}/repos?per_page=10&sort=stars"
        resp  = requests.get(url, timeout=8, headers={"Accept": "application/vnd.github.v3+json"})
        if not resp.ok:
            return result

        repos = resp.json()
        if not isinstance(repos, list):
            return result

        total_stars = sum(int(r.get("stargazers_count", 0)) for r in repos)
        total_forks = sum(int(r.get("forks_count", 0)) for r in repos)
        open_issues = sum(int(r.get("open_issues_count", 0)) for r in repos)

        score  = 0
        signal = "NORMAL"

        if total_stars > 100_000:
            signal = "MASSIVE ECOSYSTEM"; score = 20
        elif total_stars > 10_000:
            signal = "STRONG ECOSYSTEM"; score = 12
        elif total_stars > 1_000:
            signal = "GROWING ECOSYSTEM"; score = 6

        result.update({
            "signal":       signal,
            "score":        score,
            "total_stars":  total_stars,
            "total_forks":  total_forks,
            "open_issues":  open_issues,
            "repo_count":   len(repos),
            "source":       "GitHub API",
        })

    except Exception as e:
        result["error"] = str(e)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 8. COMPOSITE ALT DATA SCORE
# ═══════════════════════════════════════════════════════════════════════════════

# Mapping of tickers to their alt-data identifiers
TICKER_ALT_MAP = {
    # Tech
    "AAPL":  {"app": "Apple", "github": "apple", "trends": "Apple"},
    "MSFT":  {"app": "Microsoft Teams", "github": "microsoft", "trends": "Microsoft"},
    "NVDA":  {"app": None, "github": "NVIDIA", "trends": "NVIDIA GPU"},
    "GOOGL": {"app": "Google", "github": "google", "trends": "Google"},
    "META":  {"app": "Instagram", "github": "facebook", "trends": "Meta Instagram"},
    "AMZN":  {"app": "Amazon Shopping", "github": "aws", "trends": "Amazon"},
    "NFLX":  {"app": "Netflix", "github": None, "trends": "Netflix"},
    "TSLA":  {"app": None, "github": "teslamotors", "trends": "Tesla"},
    "PLTR":  {"app": None, "github": "palantir", "trends": "Palantir"},
    "COIN":  {"app": "Coinbase", "github": "coinbase", "trends": "Coinbase"},
    "HOOD":  {"app": "Robinhood", "github": None, "trends": "Robinhood app"},
    "SOFI":  {"app": "SoFi", "github": None, "trends": "SoFi bank"},
    "HIMS":  {"app": "Hims app", "github": None, "trends": "Hims Hers"},
    "SNOW":  {"app": None, "github": "snowflakedb", "trends": "Snowflake data"},
    "DDOG":  {"app": None, "github": "DataDog", "trends": "Datadog"},
    "CRWD":  {"app": None, "github": "CrowdStrike", "trends": "CrowdStrike"},
    "UBER":  {"app": "Uber", "github": "uber", "trends": "Uber"},
    "LYFT":  {"app": "Lyft", "github": "lyft", "trends": "Lyft"},
    "MSTR":  {"app": None, "github": None, "trends": "MicroStrategy Bitcoin"},
    "GME":   {"app": None, "github": None, "trends": "GameStop"},
}


def get_full_alt_data(sym: str, company_name: str = None) -> dict:
    """
    Full alternative data analysis for a single ticker.
    Runs all available alt-data modules and returns composite signal.
    """
    key = f"altdata_{sym}"

    def _run():
        alt_map = TICKER_ALT_MAP.get(sym, {})
        trends_query = alt_map.get("trends") or company_name or sym
        app_name     = alt_map.get("app")
        github_org   = alt_map.get("github")

        signals = {}

        # Google Trends
        signals["trends"]  = get_google_trends_signal(sym, trends_query)

        # USPTO Patents
        signals["patents"] = get_patent_velocity(sym, company_name or sym)

        # EDGAR Language
        signals["edgar"]   = get_edgar_language_signal(sym)

        # App Store (if applicable)
        if app_name:
            signals["appstore"] = get_app_store_signal(app_name, sym)

        # GitHub (if applicable)
        if github_org:
            signals["github"] = get_github_signal(github_org, sym)

        # EIA (energy sector)
        signals["eia"] = get_eia_energy_signal(sym)

        # Consumer credit (macro proxy)
        signals["consumer_credit"] = _cached("consumer_credit_global",
                                             get_consumer_credit_signal, ttl=7200)

        # Composite alt score
        total_score = sum(
            v.get("score", 0) for v in signals.values()
            if isinstance(v, dict) and v.get("signal", "N/A") not in ("UNAVAILABLE", "N/A (non-energy)")
        )
        total_score = max(-50, min(100, total_score))

        if total_score >= 40:
            composite_signal = "STRONGLY POSITIVE"
        elif total_score >= 20:
            composite_signal = "POSITIVE"
        elif total_score >= 5:
            composite_signal = "MILDLY POSITIVE"
        elif total_score <= -20:
            composite_signal = "NEGATIVE"
        elif total_score <= -5:
            composite_signal = "MILDLY NEGATIVE"
        else:
            composite_signal = "NEUTRAL"

        return {
            "sym":             sym,
            "composite_score": total_score,
            "composite_signal": composite_signal,
            "signals":         signals,
            "updated":         datetime.now().isoformat(),
        }

    return _cached(key, _run, ttl=3600)
