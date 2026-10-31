"""
Book-to-market from point-in-time fundamentals.

Score = book_equity / (price × shares_outstanding) when both fields exist.
Falls back to nothing (all-NaN) rather than a mis-scaled book/price ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gig.factors.base import Factor
from gig.types import MarketPanel


class BookToPrice(Factor):
    """Cross-sectional value: higher = cheaper (more long)."""

    name = "book_to_price"

    def compute(self, panel: MarketPanel) -> pd.DataFrame:
        fund = getattr(panel, "fundamentals", None)
        empty = pd.DataFrame(np.nan, index=panel.close.index, columns=panel.close.columns)
        if fund is None or fund.empty:
            return empty

        need = {"asof_date", "symbol", "field", "value"}
        if not need.issubset(set(fund.columns)):
            return empty

        frame = fund.copy()
        frame["asof_date"] = pd.to_datetime(frame["asof_date"]).dt.normalize()
        frame["symbol"] = frame["symbol"].astype(str)
        frame["field"] = frame["field"].astype(str)

        book = frame.loc[frame["field"] == "book_equity"].sort_values("asof_date")
        shares = frame.loc[frame["field"] == "shares_outstanding"].sort_values("asof_date")
        if book.empty or shares.empty:
            return empty

        out = empty.copy()
        for dt in panel.close.index:
            asof = pd.Timestamp(dt).normalize()
            be = (
                book.loc[book["asof_date"] <= asof]
                .drop_duplicates("symbol", keep="last")
                .set_index("symbol")["value"]
                .astype(float)
            )
            sh = (
                shares.loc[shares["asof_date"] <= asof]
                .drop_duplicates("symbol", keep="last")
                .set_index("symbol")["value"]
                .astype(float)
            )
            if be.empty or sh.empty:
                continue
            px = panel.close.loc[dt]
            common = be.index.intersection(sh.index).intersection(px.index)
            if len(common) == 0:
                continue
            mkt = px.reindex(common).astype(float) * sh.reindex(common)
            btm = be.reindex(common) / mkt.replace(0.0, np.nan)
            # Drop non-finite / negative book (buybacks / deficits are rare noise here).
            btm = btm.replace([np.inf, -np.inf], np.nan)
            btm = btm.where(btm > 0)
            out.loc[dt, common] = btm
        return out


def fundamentals_factor(panel: MarketPanel) -> pd.DataFrame:
    return BookToPrice().compute(panel)
