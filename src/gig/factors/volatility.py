"""Low idiosyncratic volatility (Ang, Hodrick, Xing, Zhang). Long low-vol names."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gig.factors.base import Factor
from gig.types import MarketPanel


class IdiosyncraticVol(Factor):
    name = "ivol_63d"

    def __init__(self, window: int = 63) -> None:
        self.window = window

    def compute(self, panel: MarketPanel) -> pd.DataFrame:
        rets = panel.close.pct_change()
        market = rets.mean(axis=1)
        resid = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
        y = rets.to_numpy()
        m = market.to_numpy()
        t, n = y.shape
        w = self.window
        out = np.full((t, n), np.nan)
        for i in range(w, t):
            sl = slice(i - w, i)
            x = m[sl]
            yy = y[sl]
            x_dm = x - np.nanmean(x)
            var_x = np.nanmean(x_dm**2)
            if var_x <= 0:
                continue
            cov = np.nanmean(x_dm[:, None] * (yy - np.nanmean(yy, axis=0)), axis=0)
            beta = cov / var_x
            fitted = x[:, None] * beta
            e = yy - fitted
            out[i] = np.nanstd(e, axis=0, ddof=1)
        resid.iloc[:, :] = out
        return -resid
