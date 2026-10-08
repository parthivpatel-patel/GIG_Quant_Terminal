"""Traceable signal objects. Confidence here is a relative score, not a probability."""

from gig.signals.contract import Signal, signal_from_weight

__all__ = ["Signal", "signal_from_weight"]
