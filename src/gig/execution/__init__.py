"""
Execution layer. Research never depends on a live broker.

`AlpacaBroker` is intentionally absent from this namespace: importing it pulls
in `alpaca-py`, which is an optional extra, so it is imported at the point of
use instead of making the whole package require a broker SDK to load.
"""

from gig.execution.algo import ExecutionConfig
from gig.execution.base import Broker
from gig.execution.paper import PaperBroker
from gig.execution.pretrade import PreTradeConfig, PreTradeReport
from gig.execution.reconcile import ReconcileConfig, TradePlan, build_trade_plan
from gig.execution.targets import TargetBook, build_target_book
from gig.execution.trader import TradeRun, account_status, flatten, run_once

__all__ = [
    "Broker",
    "ExecutionConfig",
    "PaperBroker",
    "PreTradeConfig",
    "PreTradeReport",
    "ReconcileConfig",
    "TargetBook",
    "TradePlan",
    "TradeRun",
    "account_status",
    "build_target_book",
    "build_trade_plan",
    "flatten",
    "run_once",
]
