"""
Bounded exposure scale. This suggests a multiplier. It does not change limits
and it is not applied to orders unless a caller does so explicitly.

The map is a logistic function of drawdown and of volatility versus a target.
A floor stops the scale from hitting zero, and the function is smooth so a
one-day noise spike does not flip exposure on and off.
"""

from __future__ import annotations

import math


def exposure_scale(
    drawdown: float,
    realized_vol: float,
    vol_target: float,
    drawdown_limit: float = 0.10,
    floor: float = 0.25,
    steepness: float = 6.0,
) -> float:
    """
    Return a scale in [`floor`, 1].

    `drawdown` is a non-positive number (0 at the peak, -0.10 for a 10% loss).
    Positive drawdown is treated as no drawdown stress.
    """
    if not 0.0 < floor < 1.0:
        raise ValueError("floor must lie in (0, 1)")
    if drawdown_limit <= 0 or vol_target <= 0:
        raise ValueError("limits must be positive")
    if realized_vol < 0:
        raise ValueError("realized_vol must be non-negative")
    dd_stress = max(-drawdown, 0.0) / drawdown_limit
    vol_stress = max(realized_vol / vol_target - 1.0, 0.0)
    stress = 0.7 * dd_stress + 0.3 * vol_stress
    # Center the sigmoid so mild stress stays near 1 and a 1.0 stress reading
    # is already well down the slope.
    raw = 1.0 / (1.0 + math.exp(steepness * (stress - 0.65)))
    return float(floor + (1.0 - floor) * raw)
