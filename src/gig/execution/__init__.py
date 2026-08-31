"""Execution adapters. Research never depends on a live broker."""

from gig.execution.base import Broker
from gig.execution.paper import PaperBroker

__all__ = ["Broker", "PaperBroker"]
