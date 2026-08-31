"""Research strategies. Each returns a score panel plus a runnable backtest config."""

from gig.strategies.equity_ls import EquityLongShort
from gig.strategies.futures_tsmom import FuturesTSMOM
from gig.strategies.options_vrp import VarianceRiskPremium

__all__ = ["EquityLongShort", "FuturesTSMOM", "VarianceRiskPremium"]
