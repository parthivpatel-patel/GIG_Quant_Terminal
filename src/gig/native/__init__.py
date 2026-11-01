"""Explicit optional C++20 kernels; imports never compile or load external code."""

from .rolling import available, rolling_stats

__all__ = ["available", "rolling_stats"]
