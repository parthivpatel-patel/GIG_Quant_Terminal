"""
Almgren–Chriss remaining-inventory trajectory for one parent order.

    n(t) = X * sinh(kappa * (T - t)) / sinh(kappa * T)
    kappa = sqrt(lambda * sigma^2 / eta)

`lambda` is execution risk aversion, `eta` is temporary impact, `sigma` is
price volatility. This plans a schedule. It does not submit orders.

When risk aversion is zero the schedule is uniform (TWAP). A larger risk
aversion trades more in the first bucket. Permanent impact does not change
the trajectory in the original Almgren–Chriss linear model; it changes cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ExecutionSchedule:
    name: str
    trades: np.ndarray
    expected_variance_cost: float
    expected_impact_cost: float

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "trades": self.trades.tolist(),
            "expected_variance_cost": self.expected_variance_cost,
            "expected_impact_cost": self.expected_impact_cost,
            "submits_orders": False,
        }


def twap_schedule(shares: float, n_buckets: int) -> ExecutionSchedule:
    if n_buckets < 1:
        raise ValueError("n_buckets must be positive")
    if shares == 0:
        raise ValueError("shares must be non-zero")
    trades = np.full(n_buckets, shares / n_buckets)
    return ExecutionSchedule("TWAP", trades, 0.0, 0.0)


def almgren_chriss_schedule(
    shares: float,
    n_buckets: int,
    sigma: float,
    eta: float,
    risk_aversion: float,
) -> ExecutionSchedule:
    """
    `shares` is signed parent quantity. `sigma` is per-bucket price volatility.
    `eta` is temporary impact in price per share per bucket.
    """
    if n_buckets < 1:
        raise ValueError("n_buckets must be positive")
    if shares == 0:
        raise ValueError("shares must be non-zero")
    if sigma < 0 or eta <= 0 or risk_aversion < 0:
        raise ValueError("sigma, eta, and risk_aversion must be non-negative; eta must be positive")
    if risk_aversion == 0 or sigma == 0:
        schedule = twap_schedule(shares, n_buckets)
        return ExecutionSchedule("Almgren-Chriss", schedule.trades, 0.0, 0.0)

    kappa = math.sqrt(risk_aversion * sigma * sigma / eta)
    # Remaining inventory at the start of each bucket, including t=0.
    remaining = np.array(
        [
            shares * math.sinh(kappa * (n_buckets - t)) / math.sinh(kappa * n_buckets)
            for t in range(n_buckets + 1)
        ]
    )
    trades = remaining[:-1] - remaining[1:]
    # Temporary impact cost ≈ eta * sum(trade^2). Variance cost uses remaining inventory.
    impact = float(eta * np.sum(trades**2))
    variance_cost = float(0.5 * risk_aversion * sigma * sigma * np.sum(remaining[:-1] ** 2))
    return ExecutionSchedule("Almgren-Chriss", trades, variance_cost, impact)
