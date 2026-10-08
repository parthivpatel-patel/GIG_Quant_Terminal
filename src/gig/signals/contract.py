"""Final signal contract used by research views. It does not submit orders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Signal:
    symbol: str
    timestamp: str
    direction: str
    horizon: str
    expected_return: float | None
    confidence: float | None
    confidence_kind: str
    uncertainty: str
    contributors: tuple[str, ...]
    regime: str | None
    volatility: float | None
    data_freshness: str
    evidence: str
    invalidation: str
    target_weight: float | None
    rejection: str | None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contributors"] = list(self.contributors)
        return payload


def signal_from_weight(
    symbol: str,
    timestamp: str,
    weight: float | None,
    score: float | None,
    freshness: str,
    band: float = 0.002,
    stale: bool = False,
) -> Signal:
    """
    Map a target weight to BUY / SELL / HOLD / NO_TRADE.

    `confidence` is the absolute score clipped to a relative rank-style reading.
    It is not a calibrated probability.
    """
    if stale or weight is None or freshness.upper().startswith("DATA UNAVAILABLE"):
        return Signal(
            symbol=symbol,
            timestamp=timestamp,
            direction="NO_TRADE",
            horizon="1d",
            expected_return=None,
            confidence=None,
            confidence_kind="unavailable",
            uncertainty="data",
            contributors=(),
            regime=None,
            volatility=None,
            data_freshness=freshness,
            evidence="no fresh target weight",
            invalidation="data becomes available",
            target_weight=None,
            rejection="stale or missing data",
        )
    if weight > band:
        direction = "BUY"
        rejection = None
    elif weight < -band:
        direction = "SELL"
        rejection = None
    else:
        direction = "HOLD"
        rejection = "inside no-trade band"
    confidence = None if score is None else float(max(min(abs(score), 1.0), 0.0))
    return Signal(
        symbol=symbol,
        timestamp=timestamp,
        direction=direction,
        horizon="1d",
        expected_return=None,
        confidence=confidence,
        confidence_kind="relative_score",
        uncertainty="not calibrated",
        contributors=("equity_ls.factors", "equity_ls.optimizer"),
        regime=None,
        volatility=None,
        data_freshness=freshness,
        evidence="target weight from the factor-neutral book",
        invalidation="weight crosses the no-trade band or the lake goes stale",
        target_weight=float(weight),
        rejection=rejection,
    )
