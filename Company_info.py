"""
Company info cache — logos, full names, sectors, industries, descriptions.
Uses yfinance with timeout + rich static fallback data.
Cached in-memory to avoid repeated API calls.
"""

import time, logging, threading
logger = logging.getLogger("COMPANY")

# ── Static domain map for fastest logo lookup ─────────────────────────────────
TICKER_DOMAINS = {
    "AAPL":"apple.com","MSFT":"microsoft.com","GOOGL":"google.com","AMZN":"amazon.com",
    "META":"meta.com","TSLA":"tesla.com","NVDA":"nvidia.com","AMD":"amd.com",
    "INTC":"intel.com","AVGO":"broadcom.com","QCOM":"qualcomm.com","TXN":"ti.com",
    "MU":"micron.com","AMAT":"appliedmaterials.com","LRCX":"lamresearch.com",
    "KLAC":"kla.com","ADBE":"adobe.com","CRM":"salesforce.com","NOW":"servicenow.com",
    "ORCL":"oracle.com","JPM":"jpmorganchase.com","BAC":"bankofamerica.com",
    "WFC":"wellsfargo.com","GS":"goldmansachs.com","MS":"morganstanley.com",
    "BLK":"blackrock.com","SCHW":"schwab.com","AXP":"americanexpress.com",
    "COF":"capitalone.com","USB":"usbank.com","PNC":"pnc.com",
    "UNH":"unitedhealthgroup.com","LLY":"lilly.com","JNJ":"jnj.com",
    "ABT":"abbott.com","MRK":"merck.com","PFE":"pfizer.com","ABBV":"abbvie.com",
    "AMGN":"amgen.com","GILD":"gilead.com","VRTX":"vrtx.com","TMO":"thermofisher.com",
    "XOM":"exxonmobil.com","CVX":"chevron.com","COP":"conocophillips.com",
    "PSX":"phillips66.com","VLO":"valero.com","EOG":"eogresources.com",
    "OXY":"oxy.com","SLB":"slb.com","CAT":"caterpillar.com","HON":"honeywell.com",
    "GE":"ge.com","RTX":"rtx.com","LMT":"lockheedmartin.com","BA":"boeing.com",
    "NOC":"northropgrumman.com","UPS":"ups.com","FDX":"fedex.com","DAL":"delta.com",
    "WMT":"walmart.com","COST":"costco.com","TGT":"target.com","HD":"homedepot.com",
    "LOW":"lowes.com","MCD":"mcdonalds.com","SBUX":"starbucks.com","NKE":"nike.com",
    "PG":"pg.com","KO":"coca-cola.com","PEP":"pepsico.com","SPY":"ssga.com",
    "QQQ":"invesco.com","GLD":"spdrs.com","GDX":"vaneck.com","GDXJ":"vaneck.com",
    "PLTR":"palantir.com","COIN":"coinbase.com","CRWD":"crowdstrike.com",
    "DDOG":"datadoghq.com","NET":"cloudflare.com","SNOW":"snowflake.com",
    "ZS":"zscaler.com","UBER":"uber.com","LYFT":"lyft.com","ABNB":"airbnb.com",
    "DASH":"doordash.com","RBLX":"roblox.com","SNAP":"snap.com","PINS":"pinterest.com",
    "ENPH":"enphase.com","MSTR":"microstrategy.com","ARM":"arm.com",
    "SOUN":"soundhound.com","IONQ":"ionq.com","NEM":"newmont.com","FCX":"fcx.com",
    "XLK":"ssga.com","XLF":"ssga.com","XLE":"ssga.com","XLV":"ssga.com",
    "XLB":"ssga.com","XLI":"ssga.com","XLU":"ssga.com","XLP":"ssga.com",
    "IWM":"ishares.com","TLT":"ishares.com","HYG":"ishares.com","IEF":"ishares.com",
    "BND":"vanguard.com","AGG":"ishares.com","VTI":"vanguard.com",
    "ORCL":"oracle.com","COP":"conocophillips.com","MPC":"marathonpetroleum.com",
    "HON":"honeywell.com","RTX":"rtx.com","GE":"ge.com","CAT":"caterpillar.com",
    "DE":"johndeere.com","MMM":"3m.com","EMR":"emerson.com","ETN":"eaton.com",
    "PLD":"prologis.com","AMT":"americantower.com","CCI":"crowncastle.com",
    "EQIX":"equinix.com","DLR":"digitalrealty.com","O":"realtyincome.com",
    "VNQ":"vanguard.com","AMC":"amctheatres.com","GME":"gamestop.com",
    "NFLX":"netflix.com","DIS":"disney.com","WMG":"wmg.com","PARA":"paramount.com",
    "PYPL":"paypal.com","SQ":"squareup.com","SHOP":"shopify.com","MELI":"mercadolibre.com",
    "SE":"sea.com","WPM":"wheatonpreciousmetals.com","AU":"anglogoldashanti.com",
    "GDXJ":"vaneck.com","BOTZ":"globalxetfs.com",
}

# ── Rich static fallback — sector, industry, cap tier, description ────────────
STATIC_INFO = {
    "AAPL": {"name":"Apple Inc.","sector":"Technology","industry":"Consumer Electronics",
              "market_cap":3_500_000_000_000,"employees":164000,"country":"US","exchange":"NASDAQ",
              "website":"https://apple.com",
              "description":"Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide. Known for iPhone, Mac, iPad, Apple Watch, and services."},
    "MSFT": {"name":"Microsoft Corp","sector":"Technology","industry":"Software—Infrastructure",
              "market_cap":3_100_000_000_000,"employees":228000,"country":"US","exchange":"NASDAQ",
              "website":"https://microsoft.com",
              "description":"Microsoft develops and supports software, services, devices, and solutions globally. Products include Windows, Azure cloud, Office 365, LinkedIn, and Xbox."},
    "GOOGL":{"name":"Alphabet Inc.","sector":"Technology","industry":"Internet Content & Information",
              "market_cap":2_200_000_000_000,"employees":182000,"country":"US","exchange":"NASDAQ",
              "website":"https://abc.xyz",
              "description":"Alphabet is the parent company of Google, the world's dominant search engine. Also operates YouTube, Google Cloud, Waymo, and other moonshot businesses."},
    "META": {"name":"Meta Platforms","sector":"Technology","industry":"Internet Content & Information",
              "market_cap":1_500_000_000_000,"employees":72000,"country":"US","exchange":"NASDAQ",
              "website":"https://meta.com",
              "description":"Meta owns Facebook, Instagram, WhatsApp, and Messenger. Building the metaverse via Reality Labs. Dominant in social advertising with 3B+ daily active users."},
    "TSLA": {"name":"Tesla Inc.","sector":"Consumer Cyclical","industry":"Auto Manufacturers",
              "market_cap":900_000_000_000,"employees":140000,"country":"US","exchange":"NASDAQ",
              "website":"https://tesla.com",
              "description":"Tesla designs and manufactures electric vehicles, energy storage systems, and solar products. Also developing full self-driving technology and the Optimus humanoid robot."},
    "NVDA": {"name":"NVIDIA Corp","sector":"Technology","industry":"Semiconductors",
              "market_cap":3_300_000_000_000,"employees":32000,"country":"US","exchange":"NASDAQ",
              "website":"https://nvidia.com",
              "description":"NVIDIA designs GPUs for gaming, AI training, and data centers. Dominates the AI accelerator market with H100/B200 chips. CUDA software ecosystem creates deep moat."},
    "AMD":  {"name":"Advanced Micro Devices","sector":"Technology","industry":"Semiconductors",
              "market_cap":270_000_000_000,"employees":26000,"country":"US","exchange":"NASDAQ",
              "website":"https://amd.com",
              "description":"AMD designs CPUs and GPUs for PCs, servers, and gaming consoles. Competes with Intel in processors and NVIDIA in AI accelerators with MI300 series."},
    "AMZN": {"name":"Amazon.com Inc.","sector":"Consumer Cyclical","industry":"Internet Retail",
              "market_cap":2_400_000_000_000,"employees":1_500_000,"country":"US","exchange":"NASDAQ",
              "website":"https://amazon.com",
              "description":"Amazon is the world's largest e-commerce company and a leading cloud provider via AWS. Also operates Prime Video, Alexa, Whole Foods, and advertising business."},
    "JPM":  {"name":"JPMorgan Chase","sector":"Financial Services","industry":"Banks—Diversified",
              "market_cap":720_000_000_000,"employees":310000,"country":"US","exchange":"NYSE",
              "website":"https://jpmorganchase.com",
              "description":"JPMorgan Chase is the largest US bank by assets, with consumer banking, investment banking, commercial banking, and asset management divisions."},
    "BAC":  {"name":"Bank of America","sector":"Financial Services","industry":"Banks—Diversified",
              "market_cap":340_000_000_000,"employees":213000,"country":"US","exchange":"NYSE",
              "website":"https://bankofamerica.com",
              "description":"Bank of America serves individual consumers, small businesses, and large corporations with banking, investing, asset management, and risk management products."},
    "GS":   {"name":"Goldman Sachs","sector":"Financial Services","industry":"Capital Markets",
              "market_cap":180_000_000_000,"employees":46000,"country":"US","exchange":"NYSE",
              "website":"https://goldmansachs.com",
              "description":"Goldman Sachs is a leading global investment bank, securities, and investment management firm. Known for M&A advisory, trading, and asset management."},
    "XOM":  {"name":"ExxonMobil Corp","sector":"Energy","industry":"Oil & Gas Integrated",
              "market_cap":490_000_000_000,"employees":62000,"country":"US","exchange":"NYSE",
              "website":"https://exxonmobil.com",
              "description":"ExxonMobil is one of the world's largest publicly traded energy companies, engaged in exploration, production, refining, and marketing of petroleum products."},
    "GLD":  {"name":"SPDR Gold ETF","sector":"Commodity","industry":"Gold ETF",
              "market_cap":75_000_000_000,"employees":0,"country":"US","exchange":"NYSE",
              "website":"https://spdrs.com",
              "description":"SPDR Gold Shares (GLD) tracks the price of gold bullion. One of the largest physically-backed gold ETFs globally, used as an inflation hedge and safe haven asset."},
    "GDX":  {"name":"VanEck Gold Miners ETF","sector":"Materials","industry":"Gold Miners ETF",
              "market_cap":15_000_000_000,"employees":0,"country":"US","exchange":"NYSE",
              "website":"https://vaneck.com",
              "description":"VanEck Gold Miners ETF tracks the NYSE Arca Gold Miners Index, providing exposure to gold and silver mining companies globally."},
    "GDXJ": {"name":"VanEck Junior Gold Miners","sector":"Materials","industry":"Junior Gold Miners ETF",
              "market_cap":5_000_000_000,"employees":0,"country":"US","exchange":"NYSE",
              "website":"https://vaneck.com",
              "description":"VanEck Junior Gold Miners ETF provides exposure to small-cap gold and silver mining companies, offering higher leverage to gold prices than senior miners."},
    "SPY":  {"name":"SPDR S&P 500 ETF","sector":"ETF","industry":"Large Blend ETF",
              "market_cap":600_000_000_000,"employees":0,"country":"US","exchange":"NYSE",
              "website":"https://ssga.com",
              "description":"The SPDR S&P 500 ETF Trust tracks the S&P 500 Index, representing 500 large-cap US companies. The world's largest and most liquid ETF with $600B+ in assets."},
    "QQQ":  {"name":"Invesco QQQ ETF","sector":"ETF","industry":"Technology ETF",
              "market_cap":280_000_000_000,"employees":0,"country":"US","exchange":"NASDAQ",
              "website":"https://invesco.com",
              "description":"Invesco QQQ ETF tracks the Nasdaq-100 Index of the 100 largest non-financial companies on Nasdaq. Heavy technology weighting makes it a proxy for tech sector performance."},
    "COIN": {"name":"Coinbase Global","sector":"Financial Services","industry":"Capital Markets",
              "market_cap":55_000_000_000,"employees":4900,"country":"US","exchange":"NASDAQ",
              "website":"https://coinbase.com",
              "description":"Coinbase is the largest US cryptocurrency exchange, providing trading, custody, and staking services for Bitcoin, Ethereum, and hundreds of other digital assets."},
    "PLTR": {"name":"Palantir Technologies","sector":"Technology","industry":"Software—Infrastructure",
              "market_cap":180_000_000_000,"employees":3900,"country":"US","exchange":"NYSE",
              "website":"https://palantir.com",
              "description":"Palantir builds data analytics platforms for government intelligence and commercial enterprises. Gotham serves defense; Foundry serves enterprises; AIP integrates AI."},
    "CRWD": {"name":"CrowdStrike Holdings","sector":"Technology","industry":"Software—Infrastructure",
              "market_cap":90_000_000_000,"employees":10000,"country":"US","exchange":"NASDAQ",
              "website":"https://crowdstrike.com",
              "description":"CrowdStrike provides cloud-delivered endpoint security via its Falcon platform. AI-native approach processes trillions of signals daily to detect and prevent cyberattacks."},
    "NFLX": {"name":"Netflix Inc.","sector":"Communication Services","industry":"Entertainment",
              "market_cap":420_000_000_000,"employees":14000,"country":"US","exchange":"NASDAQ",
              "website":"https://netflix.com",
              "description":"Netflix is the world's leading streaming service with 300M+ subscribers globally. Produces original content and licenses films and TV shows across 190+ countries."},
    "AVGO": {"name":"Broadcom Inc.","sector":"Technology","industry":"Semiconductors",
              "market_cap":900_000_000_000,"employees":40000,"country":"US","exchange":"NASDAQ",
              "website":"https://broadcom.com",
              "description":"Broadcom designs and supplies a broad range of semiconductor and infrastructure software solutions. Major AI networking chip supplier and VMware acquirer."},
    "UBER": {"name":"Uber Technologies","sector":"Technology","industry":"Software—Application",
              "market_cap":190_000_000_000,"employees":32000,"country":"US","exchange":"NYSE",
              "website":"https://uber.com",
              "description":"Uber operates the world's largest ride-sharing network and Uber Eats food delivery platform. Expanding into autonomous vehicles via partnerships with Waymo."},
    "ARM":  {"name":"Arm Holdings","sector":"Technology","industry":"Semiconductors",
              "market_cap":150_000_000_000,"employees":6400,"country":"UK","exchange":"NASDAQ",
              "website":"https://arm.com",
              "description":"Arm Holdings designs the processor architecture used in virtually all smartphones and increasingly in data centers, IoT devices, and AI chips. IP licensing model."},
    "NEM":  {"name":"Newmont Corp","sector":"Materials","industry":"Gold",
              "market_cap":50_000_000_000,"employees":26000,"country":"US","exchange":"NYSE",
              "website":"https://newmont.com",
              "description":"Newmont is the world's largest gold mining company by production, with operations in North America, South America, Australia, and Africa."},
    "FCX":  {"name":"Freeport-McMoRan","sector":"Materials","industry":"Copper",
              "market_cap":65_000_000_000,"employees":28000,"country":"US","exchange":"NYSE",
              "website":"https://fcx.com",
              "description":"Freeport-McMoRan is the world's largest publicly traded copper company, with major mines in Indonesia (Grasberg), North America, and South America. Also mines gold and molybdenum."},
    "COP":  {"name":"ConocoPhillips","sector":"Energy","industry":"Oil & Gas E&P",
              "market_cap":140_000_000_000,"employees":10000,"country":"US","exchange":"NYSE",
              "website":"https://conocophillips.com",
              "description":"ConocoPhillips is a leading global E&P company focused on low-cost oil and gas production. Operations span the US, Norway, Australia, Qatar, and other regions."},
    "WPM":  {"name":"Wheaton Precious Metals","sector":"Materials","industry":"Gold Streaming",
              "market_cap":25_000_000_000,"employees":50,"country":"Canada","exchange":"NYSE",
              "website":"https://wheatonpreciousmetals.com",
              "description":"Wheaton Precious Metals is the world's largest precious metals streaming company, purchasing gold and silver production from miners at fixed prices for royalty income."},
    "AU":   {"name":"AngloGold Ashanti","sector":"Materials","industry":"Gold Mining",
              "market_cap":12_000_000_000,"employees":35000,"country":"South Africa","exchange":"NYSE",
              "website":"https://anglogoldashanti.com",
              "description":"AngloGold Ashanti is a global gold mining company with operations across Africa, the Americas, and Australia. One of the world's top 5 gold producers."},
    "IONQ": {"name":"IonQ Inc.","sector":"Technology","industry":"Quantum Computing",
              "market_cap":8_000_000_000,"employees":400,"country":"US","exchange":"NYSE",
              "website":"https://ionq.com",
              "description":"IonQ develops trapped-ion quantum computers, offering cloud access via AWS, Azure, and Google Cloud. Targeting commercial quantum advantage in chemistry and optimization."},
    "SOUN": {"name":"SoundHound AI","sector":"Technology","industry":"AI Software",
              "market_cap":4_000_000_000,"employees":500,"country":"US","exchange":"NASDAQ",
              "website":"https://soundhound.com",
              "description":"SoundHound AI develops voice AI technology for restaurants, automotive, and IoT applications. Powers voice ordering systems for major fast-food chains."},
    "MPC":  {"name":"Marathon Petroleum","sector":"Energy","industry":"Oil Refining",
              "market_cap":55_000_000_000,"employees":20000,"country":"US","exchange":"NYSE",
              "website":"https://marathonpetroleum.com",
              "description":"Marathon Petroleum is the largest US petroleum refiner, operating 13 refineries. Also owns MPLX, a large pipeline MLP, providing midstream infrastructure."},
    "XLK":  {"name":"Technology SPDR ETF","sector":"ETF","industry":"Technology ETF",
              "market_cap":70_000_000_000,"employees":0,"country":"US","exchange":"NYSE",
              "website":"https://ssga.com",
              "description":"SPDR Technology Select Sector ETF tracks the Technology Select Sector Index, holding top tech companies including Apple, Microsoft, NVIDIA, and Broadcom."},
    "XLF":  {"name":"Financial SPDR ETF","sector":"ETF","industry":"Financial ETF",
              "market_cap":45_000_000_000,"employees":0,"country":"US","exchange":"NYSE",
              "website":"https://ssga.com",
              "description":"SPDR Financial Select Sector ETF tracks financial sector companies including Berkshire Hathaway, JPMorgan, Bank of America, Wells Fargo, and Goldman Sachs."},
    "XLE":  {"name":"Energy SPDR ETF","sector":"ETF","industry":"Energy ETF",
              "market_cap":35_000_000_000,"employees":0,"country":"US","exchange":"NYSE",
              "website":"https://ssga.com",
              "description":"SPDR Energy Select Sector ETF holds major oil & gas companies including ExxonMobil, Chevron, ConocoPhillips, and Pioneer Natural Resources."},
    "XLV":  {"name":"Healthcare SPDR ETF","sector":"ETF","industry":"Healthcare ETF",
              "market_cap":38_000_000_000,"employees":0,"country":"US","exchange":"NYSE",
              "website":"https://ssga.com",
              "description":"SPDR Healthcare Select Sector ETF provides exposure to pharmaceutical, biotech, and healthcare services companies including UnitedHealth, Johnson & Johnson, and AbbVie."},
    "BOTZ": {"name":"Global X Robotics & AI ETF","sector":"ETF","industry":"Robotics/AI ETF",
              "market_cap":2_500_000_000,"employees":0,"country":"US","exchange":"NASDAQ",
              "website":"https://globalxetfs.com",
              "description":"Global X Robotics & AI ETF (BOTZ) tracks companies involved in robotics, automation, and artificial intelligence, including NVIDIA, Intuitive Surgical, and Fanuc."},
    "IEF":  {"name":"iShares 7-10 Year Treasury ETF","sector":"ETF","industry":"Treasury Bond ETF",
              "market_cap":28_000_000_000,"employees":0,"country":"US","exchange":"NASDAQ",
              "website":"https://ishares.com",
              "description":"iShares 7-10 Year Treasury Bond ETF tracks intermediate-term US Treasury bonds. Used as a safe-haven allocation and interest rate hedge."},
    "TLT":  {"name":"iShares 20+ Year Treasury ETF","sector":"ETF","industry":"Long Treasury ETF",
              "market_cap":55_000_000_000,"employees":0,"country":"US","exchange":"NASDAQ",
              "website":"https://ishares.com",
              "description":"iShares 20+ Year Treasury Bond ETF tracks long-duration US Treasuries. Highly sensitive to interest rates; used for duration exposure and rate bet plays."},
    "BND":  {"name":"Vanguard Total Bond ETF","sector":"ETF","industry":"Bond ETF",
              "market_cap":110_000_000_000,"employees":0,"country":"US","exchange":"NASDAQ",
              "website":"https://vanguard.com",
              "description":"Vanguard Total Bond Market ETF tracks the Bloomberg US Aggregate Float Adjusted Index covering US investment-grade bonds including Treasuries and corporates."},
    "LYFT": {"name":"Lyft Inc.","sector":"Technology","industry":"Ride-Sharing",
              "market_cap":6_000_000_000,"employees":4000,"country":"US","exchange":"NASDAQ",
              "website":"https://lyft.com",
              "description":"Lyft operates a ride-hailing platform in the United States and Canada, competing with Uber. Also provides bike-sharing and scooter services in major cities."},
    "NET":  {"name":"Cloudflare Inc.","sector":"Technology","industry":"Cloud Security",
              "market_cap":45_000_000_000,"employees":3900,"country":"US","exchange":"NYSE",
              "website":"https://cloudflare.com",
              "description":"Cloudflare provides cloud-based security and performance services including CDN, DDoS protection, zero-trust security, and serverless computing infrastructure."},
    "ENPH": {"name":"Enphase Energy","sector":"Technology","industry":"Solar",
              "market_cap":12_000_000_000,"employees":3000,"country":"US","exchange":"NASDAQ",
              "website":"https://enphase.com",
              "description":"Enphase Energy designs and manufactures microinverter-based solar systems and battery storage. Leading supplier of residential solar energy management technology."},
}

# In-memory cache
_company_cache = {}
_enrich_in_progress = set()
_lock = threading.Lock()


def get_logo_url(symbol: str) -> str:
    """Returns Clearbit logo URL for a ticker symbol."""
    domain = TICKER_DOMAINS.get(symbol.upper())
    if domain:
        return f"https://logo.clearbit.com/{domain}"
    return f"https://logo.clearbit.com/{symbol.lower()}.com"


def _build_from_static(sym: str) -> dict:
    """Build info dict from static fallback data."""
    static = STATIC_INFO.get(sym, {})
    cap    = static.get("market_cap", 0)
    return {
        "symbol":          sym,
        "name":            static.get("name", sym),
        "logo_url":        get_logo_url(sym),
        "sector":          static.get("sector", "Unknown"),
        "industry":        static.get("industry", "Unknown"),
        "market_cap":      cap,
        "market_cap_fmt":  _fmt_cap(cap),
        "employees":       static.get("employees", 0),
        "country":         static.get("country", "US"),
        "exchange":        static.get("exchange", ""),
        "website":         static.get("website", ""),
        "description":     static.get("description", ""),
        "_source":         "static",
        "_ts":             time.time(),
    }


def get_company_info(symbol: str, use_yfinance: bool = True) -> dict:
    """
    Returns company info dict with logo, name, sector, industry, description.
    Strategy:
      1. Return cache if fresh (< 4 hours)
      2. Try static lookup first (instant)
      3. If use_yfinance=True, enrich with live yfinance data
      4. Always returns something useful
    """
    sym    = symbol.upper()
    cached = _company_cache.get(sym)
    cache_age = time.time() - cached.get("_ts", 0) if cached else 9999

    # Return cache if fresh enough
    if cached and cache_age < 14400:  # 4 hours
        return cached

    # Start with static data (always available, instant)
    info = _build_from_static(sym)

    # Enrich with yfinance in background if requested and not already running
    if use_yfinance and sym not in _enrich_in_progress:
        with _lock:
            _enrich_in_progress.add(sym)

        def _yf_enrich():
            try:
                import yfinance as yf
                import signal as _sig

                tk = yf.Ticker(sym)

                # Use fast_info first (much faster than .info)
                fi = tk.fast_info
                name = getattr(fi, "shortName",  None) or getattr(fi, "longName", None)
                cap  = getattr(fi, "marketCap",  None) or getattr(fi, "totalAssets", None)

                # Merge fast_info data
                enriched = dict(info)
                if name: enriched["name"] = name
                if cap:
                    enriched["market_cap"]     = int(cap)
                    enriched["market_cap_fmt"] = _fmt_cap(int(cap))

                # Try full .info for sector/industry/description (slower)
                try:
                    yi = tk.info
                    if yi:
                        enriched["name"]        = yi.get("longName") or yi.get("shortName") or enriched["name"]
                        enriched["sector"]      = yi.get("sector")       or enriched["sector"]
                        enriched["industry"]    = yi.get("industry")     or enriched["industry"]
                        enriched["website"]     = yi.get("website")      or enriched["website"]
                        enriched["market_cap"]  = yi.get("marketCap")    or enriched["market_cap"]
                        enriched["employees"]   = yi.get("fullTimeEmployees") or enriched["employees"]
                        enriched["country"]     = yi.get("country")      or enriched["country"]
                        enriched["exchange"]    = yi.get("exchange")     or enriched["exchange"]
                        desc = yi.get("longBusinessSummary", "")
                        if desc:
                            enriched["description"]  = desc[:400]
                        enriched["market_cap_fmt"] = _fmt_cap(enriched["market_cap"])
                        # Prefer yfinance website for logo
                        website = enriched.get("website", "")
                        if website:
                            domain = website.replace("https://","").replace("http://","").split("/")[0].lstrip("www.")
                            enriched["logo_url"] = f"https://logo.clearbit.com/{domain}"
                except Exception:
                    pass  # fast_info was enough

                enriched["_source"] = "yfinance"
                enriched["_ts"]     = time.time()
                _company_cache[sym] = enriched
                logger.debug(f"Company enriched: {sym} | {enriched.get('sector')} | {_fmt_cap(enriched.get('market_cap',0))}")

            except Exception as e:
                logger.debug(f"yfinance enrich {sym}: {e}")
                # Keep static data in cache
                info["_ts"] = time.time()
                _company_cache[sym] = info
            finally:
                with _lock:
                    _enrich_in_progress.discard(sym)

        threading.Thread(target=_yf_enrich, daemon=True, name=f"co-{sym}").start()

    # Store static version immediately so caller gets something right away
    if sym not in _company_cache:
        _company_cache[sym] = info

    return _company_cache[sym]


def get_batch_info(symbols: list) -> dict:
    """Fetch company info for multiple symbols (uses background enrichment)."""
    results = {}
    for sym in symbols[:50]:
        results[sym] = get_company_info(sym, use_yfinance=True)
    return results


def _fmt_cap(cap: float) -> str:
    if not cap or cap == 0:
        return "—"
    if cap >= 1_000_000_000_000:
        return f"${cap/1e12:.1f}T"
    if cap >= 1_000_000_000:
        return f"${cap/1e9:.1f}B"
    if cap >= 1_000_000:
        return f"${cap/1e6:.0f}M"
    return f"${cap:,.0f}"