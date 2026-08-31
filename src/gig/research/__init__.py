"""Research statistics: IC, Newey–West, walk-forward, multiple testing."""

from gig.research.ic import information_ratio, spearman_ic
from gig.research.multiple_testing import deflated_sharpe
from gig.research.walkforward import WalkForwardSplit

__all__ = ["spearman_ic", "information_ratio", "deflated_sharpe", "WalkForwardSplit"]
