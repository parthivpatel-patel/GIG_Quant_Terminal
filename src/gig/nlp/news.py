"""Live news pull (Yahoo) aligned to a trading date. Headline time must be ≤ close t."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from gig.nlp.sentiment import headline_sentiment


def headlines_to_frame(rows: list[dict]) -> pd.DataFrame:
    """Normalize raw headline dicts into the news table schema."""
    recs = []
    for r in rows:
        published = r.get("published") or r.get("providerPublishTime")
        if isinstance(published, (int, float)):
            published = datetime.fromtimestamp(float(published), tz=UTC)
        elif isinstance(published, str):
            published = pd.Timestamp(published).to_pydatetime()
        if published is None:
            continue
        if published.tzinfo is None:
            published = published.replace(tzinfo=UTC)
        asof = published.date()
        title = str(r.get("title") or r.get("headline") or "")
        recs.append(
            {
                "published": published,
                "asof_date": asof,
                "symbol": str(r.get("symbol") or ""),
                "title": title,
                "source": str(r.get("publisher") or r.get("source") or "yahoo"),
                "sentiment": headline_sentiment(title),
                "url": str(r.get("link") or r.get("url") or ""),
            }
        )
    if not recs:
        return pd.DataFrame(
            columns=["published", "asof_date", "symbol", "title", "source", "sentiment", "url"]
        )
    return pd.DataFrame.from_records(recs)


def fetch_yahoo_news(symbols: list[str]) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance to fetch live news") from exc
    rows: list[dict] = []
    for sym in symbols:
        try:
            items = yf.Ticker(sym).news or []
        except Exception:
            continue
        for it in items:
            content = it.get("content") if isinstance(it, dict) else None
            if isinstance(content, dict):
                title = content.get("title") or ""
                pub = content.get("pubDate") or content.get("displayTime")
                link = (content.get("canonicalUrl") or {}).get("url") if isinstance(
                    content.get("canonicalUrl"), dict
                ) else content.get("canonicalUrl")
                rows.append(
                    {
                        "symbol": sym,
                        "title": title,
                        "published": pub,
                        "publisher": (content.get("provider") or {}).get("displayName"),
                        "link": link,
                    }
                )
            elif isinstance(it, dict):
                rows.append({"symbol": sym, **it})
    return headlines_to_frame(rows)


def news_factor(news: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """
    Daily mean headline sentiment, as-of the calendar date of publication.

    Weekend/holiday headlines map to the next session (known at that close).
    Scores on date t may only use headlines with asof_date <= t. The backtest
    already applies next-session returns, so we do not shift again here.
    """
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    if news is None or news.empty or "asof_date" not in news.columns:
        return out
    work = news.copy()
    work["symbol"] = work["symbol"].astype(str).str.upper()
    work = work[work["symbol"].isin(close.columns) & work["symbol"].ne("")]
    if work.empty:
        return out
    work["asof_date"] = pd.to_datetime(work["asof_date"]).dt.normalize()
    daily = work.groupby(["asof_date", "symbol"], as_index=False)["sentiment"].mean()
    pivot = daily.pivot(index="asof_date", columns="symbol", values="sentiment")
    return align_asof_to_close(pivot, close, ffill=True)


def naive_trading_days(idx) -> pd.DatetimeIndex:
    di = pd.DatetimeIndex(pd.to_datetime(idx))
    if di.tz is not None:
        di = di.tz_convert("UTC").tz_localize(None)
    return di.normalize()


def align_asof_to_close(
    panel: pd.DataFrame,
    close: pd.DataFrame,
    *,
    ffill: bool = True,
    fill_value: float | None = None,
) -> pd.DataFrame:
    """
    Map event dates onto the trading calendar.

    An event on a non-session day is assigned to the next close — never the
    previous one — so Friday's book cannot see Sunday's 8-K.
    """
    empty = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    if panel is None or panel.empty:
        return empty if fill_value is None else empty.fillna(fill_value)
    close_idx = naive_trading_days(close.index)
    p = panel.copy()
    p.index = naive_trading_days(p.index)
    p = p.groupby(level=0).mean()
    pos = close_idx.searchsorted(p.index)
    mask = pos < len(close_idx)
    if not mask.any():
        return empty if fill_value is None else empty.fillna(fill_value)
    mapped = p.iloc[mask.nonzero()[0]].copy()
    mapped.index = close_idx[pos[mask]]
    mapped = mapped.groupby(level=0).mean()
    aligned = mapped.reindex(close_idx).reindex(columns=list(close.columns))
    if ffill:
        aligned = aligned.sort_index().ffill()
    elif fill_value is not None:
        aligned = aligned.fillna(fill_value)
    aligned.index = close.index
    return aligned
