"""Shared process guard for model code temporarily changing CPU Torch settings.

This module imports no numerical runtime or model assets. Every model adapter
must restore the process settings it changes while holding this reentrant lock.
"""

from threading import RLock

TORCH_CPU_LOCK = RLock()
