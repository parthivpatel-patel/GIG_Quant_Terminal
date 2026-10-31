"""Research statistics: IC, Newey–West, walk-forward, multiple testing.

Imports are lazy so concurrent FastAPI workers do not deadlock on package init
when several routes pull `gig.research.*` at once.
"""

from __future__ import annotations

from typing import Any

__all__ = ["spearman_ic", "information_ratio", "deflated_sharpe", "WalkForwardSplit"]


def __getattr__(name: str) -> Any:
    if name in ("spearman_ic", "information_ratio"):
        from gig.research.ic import information_ratio, spearman_ic

        return spearman_ic if name == "spearman_ic" else information_ratio
    if name == "deflated_sharpe":
        from gig.research.multiple_testing import deflated_sharpe

        return deflated_sharpe
    if name == "WalkForwardSplit":
        from gig.research.walkforward import WalkForwardSplit

        return WalkForwardSplit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
