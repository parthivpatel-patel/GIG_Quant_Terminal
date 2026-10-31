"""Timezone-aware nanosecond timestamp admission without fractional truncation."""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

_FRACTION = re.compile(r"[Tt ]\d{2}:\d{2}:\d{2}\.(\d+)")


def aware_timestamp(value: Any) -> pd.Timestamp:
    """Normalize to UTC, rejecting naive, missing, or subnanosecond input.

    Pandas accepts >9 fractional digits by silently dropping trailing digits.
    Evidence admission rejects such strings, even when those digits are zero,
    so the declared precision bound is explicit and independent of parser quirks.
    """
    if isinstance(value, str):
        if len(value) > 128:
            raise ValueError("timestamp exceeds declared text bound")
        fraction = _FRACTION.search(value)
        if fraction is not None and len(fraction.group(1)) > 9:
            raise ValueError("timestamp precision exceeds nanoseconds")
    try:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware and nonmissing")
        return timestamp.tz_convert("UTC").as_unit("ns")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("invalid, naive, missing, or out-of-range timestamp") from exc
