"""End-to-end research runner used by the CLI and tests."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from gig.config import get_settings
from gig.data.providers.synthetic import SyntheticProvider
from gig.data.universe_spec import load_universe, resolve_universe, universe_mode
from gig.logging import configure_logging
from gig.research.experiment import ExperimentLog
from gig.research.multiple_testing import deflated_sharpe
from gig.strategies.equity_ls import EquityLongShort
from gig.types import BacktestResult, MarketPanel


def optimizer_config(settings, enabled: bool | None = None):
    """
    Risk policy for book construction, or None for the equal-weight baseline.

    Gross and name caps come from the same settings the risk limits check
    reads, so the optimizer cannot be configured to build a book that the
    limits module would immediately flag.
    """
    from gig.portfolio.optimize import OptimizerConfig

    on = settings.use_optimizer if enabled is None else enabled
    if not on:
        return None
    return OptimizerConfig(
        target_vol=settings.target_vol,
        max_gross=settings.max_gross_leverage,
        max_name=settings.max_name_weight,
    )


def run_equity_research(
    source: str | None = None,
    seed: int = 42,
    persist: bool = True,
    use_ml: bool = True,
    use_news: bool = True,
    use_optimizer: bool | None = None,
) -> BacktestResult:
    log = configure_logging(get_settings().log_level)
    settings = get_settings()
    settings.ensure_dirs()
    src = (source or settings.data_source).lower()
    opt = optimizer_config(settings, use_optimizer)
    risk_kwargs = {
        "optimizer": opt,
        "risk_lookback": settings.risk_lookback,
        "risk_refit_every": settings.risk_refit_every,
        "risk_factors": settings.risk_factors,
    }

    if src == "synthetic":
        provider = SyntheticProvider(n_names=80, n_days=756, seed=seed)
        panel = provider.load_panel(provider.start, provider.start.replace(year=provider.start.year + 4))
        strategy = EquityLongShort(use_ml=use_ml, use_news=False, **risk_kwargs)
    elif src == "yahoo":
        panel = _load_live_panel(settings, use_news=use_news)
        n_names = len(panel.symbols())
        wide = n_names > 80
        strategy = EquityLongShort(
            n_long=50 if wide else 10,
            n_short=50 if wide else 10,
            min_adv_usd=1_000_000.0 if wide else 0.0,
            use_ml=use_ml and n_names <= 800,
            use_news=use_news,
            **risk_kwargs,
        )
        if use_ml and n_names > 800:
            log.info("walk-forward GBDT skipped (%s names) — pass a smaller tape or wait; factors still run", n_names)
    else:
        raise ValueError(f"Unknown data source {src!r} (use yahoo or synthetic)")

    result = strategy.backtest(panel)
    n_trials = max(3, len(result.factor_ic) or 3)
    dsr = deflated_sharpe(
        result.metrics.get("sharpe", float("nan")),
        n_trials=n_trials,
        n_obs=int(result.metrics.get("n_obs", 0)),
        skew=result.metrics.get("skew", 0.0),
        kurtosis=result.metrics.get("kurtosis", 3.0),
    )
    result.metrics.update(dsr)

    log.info(
        "equity_ls source=%s construction=%s sharpe=%.2f vol=%.1f%% max_dd=%.2f%% IC_mom=%.3f",
        src,
        "risk_constrained" if opt else "quantile_equal_weight",
        result.metrics.get("sharpe", float("nan")),
        100 * result.metrics.get("ann_vol", float("nan")),
        100 * result.metrics.get("max_drawdown", float("nan")),
        result.factor_ic.get("momentum_12_1", float("nan")),
    )
    if result.diagnostics is not None and not result.diagnostics.empty:
        log.info(
            "risk model: ex-ante vol %.1f%% systematic %.0f%% of variance across %d rebalances",
            100 * result.diagnostics["ex_ante_vol"].mean(),
            100 * result.diagnostics["systematic_share"].mean(),
            len(result.diagnostics),
        )

    if persist:
        out = settings.results_dir / "equity_ls_last.json"
        payload = {
            "source": src,
            "construction": "risk_constrained" if opt else "quantile_equal_weight",
            "metrics": result.metrics,
            "factor_ic": result.factor_ic,
            "factor_ic_n": result.factor_ic_n,
            "config_hash": result.config_hash,
        }
        if result.diagnostics is not None and not result.diagnostics.empty:
            payload["risk_path"] = {
                col: float(result.diagnostics[col].mean())
                for col in ("ex_ante_vol", "systematic_share", "gross", "effective_names")
                if col in result.diagnostics.columns
            }
        out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        ExperimentLog(settings.results_dir / "experiments.jsonl").record(
            "equity_ls", strategy.config(), result.metrics
        )
        try:
            from gig.store import Store

            store = Store(settings.db_path)
            store.init()
            store.record_experiment("equity_ls", result.config_hash, result.metrics)
        except Exception:
            log.debug("experiment not written to DuckDB", exc_info=True)
    return result


def ingest_live(use_news: bool = True, *, all_us: bool | None = None) -> dict:
    """Pull Yahoo bars plus any configured free APIs into DuckDB."""
    log = configure_logging(get_settings().log_level)
    settings = get_settings()
    settings.ensure_dirs()
    from gig.data.listings import cap_buckets_from_adv
    from gig.store import Store

    store = Store(settings.db_path)
    store.init()
    use_all = universe_mode(settings.universe_file) == "us_listed" if all_us is None else all_us
    if use_all:
        from gig.data.listings import refresh_us_listings
        from gig.data.universe_spec import DEFAULT_SECTORS

        listings = refresh_us_listings(store)
        col = "yahoo_symbol" if "yahoo_symbol" in listings.columns else "symbol"
        sectors = {
            str(sym).upper(): DEFAULT_SECTORS.get(str(sym).upper(), "unknown")
            for sym in listings[col].tolist()
        }
    else:
        sectors = load_universe(settings.universe_file)
    log.info("ingest universe=%s names=%s", "us_listed" if use_all else "mega", len(sectors))
    panel = _fetch_yahoo(settings, sectors)
    n_bars = store.upsert_bars(panel)
    store.upsert_universe(panel.sectors, asof=date.today())
    if use_all and panel.dollar_volume is not None:
        adv = panel.dollar_volume.tail(21).mean()
        store.update_cap_buckets(cap_buckets_from_adv(adv))
    vendor_names = _top_adv_symbols(panel, 40 if use_all else len(panel.symbols()))
    filing_names = _top_adv_symbols(panel, 200 if use_all else len(panel.symbols()))
    quote_names = _top_adv_symbols(panel, 50)
    extras = _ingest_vendor_feeds(
        store,
        settings,
        vendor_names,
        use_news=use_news,
        log=log,
        filing_symbols=filing_names,
        quote_symbols=quote_names,
    )
    cap_counts = {}
    try:
        listings = store.load_listings()
        if listings is not None and not listings.empty and "cap_bucket" in listings.columns:
            cap_counts = listings["cap_bucket"].value_counts().to_dict()
    except Exception:
        pass
    return {
        "universe": "us_listed" if use_all else "mega",
        "names": len(panel.symbols()),
        "bars": n_bars,
        "cap_buckets": cap_counts,
        **extras,
        **store.stats(),
    }


def live_quotes(limit: int = 50) -> pd.DataFrame:
    """Snapshot from Alpaca IEX, else Finnhub. Caps at `limit` names (ADV order)."""
    settings = get_settings()
    from gig.data.lake import open_store

    store = open_store()
    sectors = resolve_universe(settings.universe_file, store, refresh=False)
    symbols = _quote_symbols(store, list(sectors), limit)
    if settings.alpaca_api_key and settings.alpaca_secret_key:
        from gig.data.providers.alpaca_iex import fetch_latest_quotes

        return fetch_latest_quotes(
            symbols, settings.alpaca_api_key, settings.alpaca_secret_key, settings.alpaca_data_url
        )
    if settings.finnhub_api_key:
        from gig.data.providers.finnhub import fetch_quotes

        return fetch_quotes(symbols[:15], settings.finnhub_api_key)
    return pd.DataFrame(columns=["ts", "symbol", "bid", "ask", "last", "source"])


def _ingest_vendor_feeds(
    store,
    settings,
    symbols: list[str],
    use_news: bool,
    log,
    filing_symbols: list[str] | None = None,
    quote_symbols: list[str] | None = None,
) -> dict[str, int]:
    out = {"news": 0, "macro": 0, "filings": 0, "quotes": 0, "calendar": 0, "fundamentals": 0}
    filing_symbols = filing_symbols or symbols
    quote_symbols = quote_symbols or symbols[:50]
    if use_news:
        try:
            from gig.nlp.news import fetch_yahoo_news

            out["news"] += store.upsert_news(fetch_yahoo_news(symbols))
        except Exception as exc:
            log.warning("yahoo news ingest skipped: %s", exc)
    if settings.fred_api_key:
        try:
            from gig.data.providers.fred import fetch_default_macro

            out["macro"] = store.upsert_macro(fetch_default_macro(settings.fred_api_key))
        except Exception as exc:
            log.warning("FRED ingest skipped: %s", exc)
    if settings.keys_status()["edgar"]:
        try:
            from gig.data.providers.edgar import fetch_cik_map, fetch_filings
            from gig.data.providers.fundamentals_sec import fetch_book_equity

            cik = fetch_cik_map(user_agent=settings.edgar_user_agent)
            out["filings"] = store.upsert_filings(
                fetch_filings(filing_symbols, cik, settings.edgar_user_agent)
            )
            # Value factor sleeve: liquid names only (companyfacts is one HTTP call each).
            fund_syms = list(dict.fromkeys([*(quote_symbols or []), *filing_symbols[:60]]))[:80]
            out["fundamentals"] = store.upsert_fundamentals(
                fetch_book_equity(fund_syms, cik, settings.edgar_user_agent)
            )
        except Exception as extra:
            log.warning("EDGAR ingest skipped: %s", extra)
    if settings.keys_status()["alpaca"]:
        try:
            from gig.data.providers.alpaca_iex import fetch_latest_quotes

            out["quotes"] += store.upsert_quotes(
                fetch_latest_quotes(
                    quote_symbols,
                    settings.alpaca_api_key,
                    settings.alpaca_secret_key,
                    settings.alpaca_data_url,
                )
            )
        except Exception as extra:
            log.warning("Alpaca IEX ingest skipped: %s", extra)
    if settings.finnhub_api_key:
        from gig.data.providers.finnhub import fetch_earnings, fetch_news, fetch_quotes

        token = settings.finnhub_api_key
        if out["quotes"] == 0:
            try:
                out["quotes"] = store.upsert_quotes(fetch_quotes(quote_symbols[:20], token))
            except Exception as extra:
                log.warning("Finnhub quotes skipped: %s", extra)
        if use_news:
            try:
                out["news"] += store.upsert_news(fetch_news(token))
            except Exception as extra:
                log.warning("Finnhub news skipped: %s", extra)
        try:
            end = date.today()
            start = end - timedelta(days=45)
            out["calendar"] = store.upsert_calendar(fetch_earnings(token, start, end))
        except Exception as extra:
            log.warning("Finnhub calendar skipped: %s", extra)
    return out


def _load_live_panel(settings, use_news: bool) -> MarketPanel:
    from gig.data.providers.cached import CachedYahooProvider
    from gig.store import Store

    store = Store(settings.db_path)
    store.init()
    sectors = resolve_universe(settings.universe_file, store, refresh=False)
    if universe_mode(settings.universe_file) == "us_listed" and len(sectors) < 100:
        import logging

        logging.getLogger("gig").warning(
            "US listings tape is empty — run: python -m gig universe refresh && python -m gig ingest"
        )
    end = date.today()
    start = end - timedelta(days=365 * 4)
    panel = CachedYahooProvider(store, sectors).load_panel(start, end, list(sectors))
    if use_news:
        news = store.con.execute("SELECT * FROM news").df()
        if news is not None and not news.empty:
            panel.news = news
    try:
        filings = store.load_filings()
        if filings is not None and not filings.empty:
            panel.filings = filings
        try:
            fundamentals = store.load_fundamentals()
            if fundamentals is not None and not fundamentals.empty:
                panel.fundamentals = fundamentals
        except Exception:
            pass
        macro = store.load_macro()
        if macro is not None and not macro.empty:
            panel.macro = macro
    except Exception:
        pass
    return panel


def _fetch_yahoo(settings, sectors: dict[str, str] | None = None) -> MarketPanel:
    from gig.data.providers.yahoo import YahooProvider

    sectors = sectors or load_universe(settings.universe_file)
    end = date.today()
    start = end - timedelta(days=365 * 4)
    return YahooProvider(sectors).load_panel(start, end, list(sectors))


def _top_adv_symbols(panel: MarketPanel, n: int) -> list[str]:
    if panel.dollar_volume is None or panel.dollar_volume.empty:
        return list(panel.symbols())[:n]
    adv = panel.dollar_volume.tail(21).mean().sort_values(ascending=False)
    return [str(s) for s in adv.index if pd.notna(adv[s])][:n]


def _quote_symbols(store, symbols: list[str], limit: int) -> list[str]:
    try:
        end = date.today()
        start = end - timedelta(days=45)
        panel = store.load_panel(start, end, symbols)
        if panel is not None:
            return _top_adv_symbols(panel, limit)
    except Exception:
        pass
    return symbols[:limit]


def load_yaml_config(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
