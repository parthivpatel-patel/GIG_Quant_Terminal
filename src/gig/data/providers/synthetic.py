"""
Deterministic synthetic equity panel with a known factor structure.

Returns are generated as:
    r_i,t = β_i * r_m,t + γ_{s(i)} * r_s,t + λ * mom_signal_{i,t-1} + ε_i,t

so a 12-1 momentum factor has positive expected IC after market/sector
neutralization. Tests rely on this — do not randomize the seed in CI.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from gig.data.providers.base import DataProvider
from gig.types import MarketPanel

SECTORS = ("tech", "health", "financials", "industrials", "energy", "consumer")


class SyntheticProvider(DataProvider):
    def __init__(
        self,
        n_names: int = 80,
        n_days: int = 756,
        seed: int = 42,
        momentum_premium: float = 0.015,
        start: date | None = None,
    ) -> None:
        self.n_names = n_names
        self.n_days = n_days
        self.seed = seed
        self.momentum_premium = momentum_premium
        self.start = start or date(2019, 1, 2)

    def load_panel(
        self,
        start: date,
        end: date,
        symbols: list[str] | None = None,
    ) -> MarketPanel:
        rng = np.random.default_rng(self.seed)
        n = self.n_names
        dates = pd.bdate_range(self.start, periods=self.n_days)
        dates = dates[(dates.date >= start) & (dates.date <= end)]
        if len(dates) == 0:
            raise ValueError("No business days in requested window")

        names = symbols or [f"S{i:03d}" for i in range(n)]
        n = len(names)
        sectors = pd.Series(
            [SECTORS[i % len(SECTORS)] for i in range(n)], index=names, name="sector"
        )
        beta = pd.Series(0.6 + 0.8 * rng.random(n), index=names)
        idio_vol = 0.006 + 0.004 * rng.random(n)

        t = len(dates)
        market = rng.normal(0.0003, 0.01, t)
        sector_ret = {s: rng.normal(0.0, 0.006, t) for s in SECTORS}

        # Latent momentum: slow AR(1) in expected return
        latent = np.zeros((t, n))
        shock = rng.normal(0, 1, (t, n))
        for i in range(1, t):
            latent[i] = 0.95 * latent[i - 1] + 0.05 * shock[i]

        rets = np.zeros((t, n))
        for j, sym in enumerate(names):
            sec = sectors[sym]
            rets[:, j] = (
                beta[sym] * market
                + sector_ret[sec]
                + self.momentum_premium * latent[:, j]
                + idio_vol[j] * rng.normal(0, 1, t)
            )

        close = 100.0 * np.exp(np.cumsum(rets, axis=0))
        close_df = pd.DataFrame(close, index=dates, columns=names)
        volume = pd.DataFrame(
            rng.integers(200_000, 2_000_000, size=close_df.shape),
            index=dates,
            columns=names,
        ).astype(float)
        dollar_volume = close_df * volume
        return MarketPanel(
            close=close_df,
            volume=volume,
            sectors=sectors,
            dollar_volume=dollar_volume,
        )


def trading_days_between(start: date, end: date) -> int:
    return len(pd.bdate_range(start, end))


def default_window(n_days: int = 756) -> tuple[date, date]:
    end = date(2024, 12, 31)
    start = end - timedelta(days=int(n_days * 1.5))
    return start, end
