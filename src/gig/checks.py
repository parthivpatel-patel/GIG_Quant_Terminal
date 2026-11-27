"""Strict admission of caller-supplied numbers for bounded research calculators.

JSON booleans, strings, nonfinite values and oversized inputs are rejected
instead of coerced, so a calculator never runs on a silently reinterpreted
request.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def number(value: Any, name: str, *, low: float | None = None, high: float | None = None,
           positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    if (low is not None and result < low) or (high is not None and result > high):
        raise ValueError(f"{name} is outside its supported range")
    return result


def integer(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return int(value)


def _has_boolean(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_has_boolean(item) for item in value)
    return isinstance(value, (bool, np.bool_))


def array(value: Any, name: str, *, dimensions: int, maximum: int, minimum: int = 1) -> np.ndarray:
    """A finite real array with exactly ``dimensions`` axes and a bounded size."""
    # NumPy silently promotes a boolean among numbers to 1 or 0.
    if _has_boolean(value):
        raise ValueError(f"{name} must hold numbers, not booleans")
    try:
        raw = np.asarray(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a rectangular numeric array") from exc
    if raw.dtype.kind not in "iuf" or raw.ndim != dimensions or not minimum <= raw.size <= maximum:
        raise ValueError(f"{name} must be a numeric array with {dimensions} axes and {minimum} to {maximum} values")
    result = raw.astype(float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def confidence(value: Any) -> float:
    return number(value, "confidence", low=.5, high=.9999)
