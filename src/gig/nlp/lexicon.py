"""Financial lexicon. Deterministic, no downloads — used when transformers are absent."""

from __future__ import annotations

import re

POSITIVE = {
    "beat",
    "beats",
    "surge",
    "surges",
    "rally",
    "record",
    "upgrade",
    "upgraded",
    "outperform",
    "bullish",
    "growth",
    "profit",
    "profits",
    "strong",
    "raise",
    "raises",
    "raised",
    "buyback",
    "dividend",
    "expansion",
    "accelerate",
}

NEGATIVE = {
    "miss",
    "misses",
    "plunge",
    "plunges",
    "downgrade",
    "downgraded",
    "bearish",
    "loss",
    "losses",
    "weak",
    "cut",
    "cuts",
    "lawsuit",
    "fraud",
    "recall",
    "bankruptcy",
    "layoff",
    "layoffs",
    "probe",
    "investigation",
    "warning",
    "slowdown",
    "decline",
}


_TOKEN = re.compile(r"[a-z]+")


def lexicon_score(text: str) -> float:
    """Return sentiment in [-1, 1] from a headline. Empty text → 0."""
    tokens = _TOKEN.findall((text or "").lower())
    if not tokens:
        return 0.0
    pos = sum(1 for t in tokens if t in POSITIVE)
    neg = sum(1 for t in tokens if t in NEGATIVE)
    denom = pos + neg
    if denom == 0:
        return 0.0
    return (pos - neg) / denom
