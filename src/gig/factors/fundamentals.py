"""
Point-in-time fundamentals factor scaffolding.

Commercial desks pull book equity / earnings from Compustat or Norgate with an
explicit as-of timestamp. This module reads the same shape from DuckDB
``fundamentals`` and returns a cross-sectional score. When the table is empty,
the factor is all-NaN and drops out of IC combination — never invents values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gig.factors.base import Factor
from gig.types import MarketPanel


class BookToPrice(Factor):
    """
    Book / price using the latest fundamentals row with ``asof_date <= t``.

    Higher = cheaper (more long). Requires ``fundamentals`` rows with
    ``field='book_equity'`` and a price from the panel close.
    """

    name = "book_to_price"

    def compute(self, panel: MarketPanel) -> pd.DataFrame:
        fund = getattr(panel, "fundamentals", None)
        if fund is None or fund.empty:
            return pd.DataFrame(np.nan, index=panel.close.index, columns=panel.close.columns)

        need = {"asof_date", "symbol", "field", "value"}
        if not need.issubset(set(fund.columns)):
            return pd.DataFrame(np.nan, index=panel.close.index, columns=panel.close.columns)

        book = fund.loc[fund["field"].astype(str) == "book_equity"].copy()
        if book.empty:
            return pd.DataFrame(np.nan, index=panel.close.index, columns=panel.close.columns)

        book["asof_date"] = pd.to_datetime(book["asof_date"]).dt.normalize()
        book["symbol"] = book["symbol"].astype(str)
        book = book.sort_values("asof_date")

        out = pd.DataFrame(np.nan, index=panel.close.index, columns=panel.close.columns)
        # Point-in-time join per date: last book equity known by that close.
        for dt in panel.close.index:
            asof = pd.Timestamp(dt).normalize()
            snap = book.loc[book["asof_date"] <= asof].drop_duplicates("symbol", keep="last")
            if snap.empty:
                continue
            be = snap.set_index("symbol")["value"].astype(float)
            px = panel.close.loc[dt]
            btp = be.reindex(px.index) / px.replace(0.0, np.nan)
            out.loc[dt] = btp
        return out


def fundamentals_factor(panel: MarketPanel) -> pd.DataFrame:
    """Convenience wrapper used by EquityLongShort.scores."""
    return BookToPrice().compute(panel)
