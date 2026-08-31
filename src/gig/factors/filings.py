"""8-K item scores as a cross-sectional event factor. As-of filing date only."""

from __future__ import annotations

import pandas as pd

from gig.nlp.news import align_asof_to_close

WINDOW = 21


def filings_factor(filings: pd.DataFrame, close: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """
    Rolling sum of 8-K item scores over `window` sessions.

    10-K / 10-Q are scheduled and dropped. Missing days are 0 (no event), not
    a stale lexicon score on the string "8-K AAPL".
    """
    empty = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    if filings is None or filings.empty:
        return empty
    work = filings.copy()
    work["symbol"] = work["symbol"].astype(str).str.upper()
    if "form" in work.columns:
        forms = work["form"].astype(str).str.upper()
        work = work[forms.str.startswith("8-K")]
    if work.empty or "sentiment" not in work.columns:
        return empty.fillna(0.0)
    work["asof_date"] = pd.to_datetime(work["asof_date"]).dt.normalize()
    daily = work.groupby(["asof_date", "symbol"], as_index=False)["sentiment"].sum()
    pivot = daily.pivot(index="asof_date", columns="symbol", values="sentiment")
    aligned = align_asof_to_close(pivot, close, ffill=False, fill_value=0.0)
    return aligned.rolling(window, min_periods=1).sum()
