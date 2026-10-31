"""Hard risk limits. Breaches are data, not exceptions — callers decide whether to halt."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from gig.portfolio.construct import sector_net_exposure


@dataclass(frozen=True, slots=True)
class LimitBreach:
    name: str
    value: float
    limit: float


def check_limits(
    weights: pd.Series,
    sectors: pd.Series,
    *,
    max_gross: float = 2.0,
    max_net: float = 0.10,
    max_name: float = 0.05,
    max_sector_net: float = 0.10,
) -> list[LimitBreach]:
    breaches: list[LimitBreach] = []
    gross = float(weights.abs().sum())
    net = float(weights.sum())
    name_max = float(weights.abs().max()) if len(weights) else 0.0
    if gross > max_gross + 1e-9:
        breaches.append(LimitBreach("gross_leverage", gross, max_gross))
    if abs(net) > max_net + 1e-9:
        breaches.append(LimitBreach("net_exposure", net, max_net))
    if name_max > max_name + 1e-9:
        breaches.append(LimitBreach("name_weight", name_max, max_name))
    sector_net = sector_net_exposure(weights, sectors)
    worst = float(sector_net.abs().max()) if len(sector_net) else 0.0
    if worst > max_sector_net + 1e-9:
        breaches.append(LimitBreach("sector_net", worst, max_sector_net))
    return breaches
