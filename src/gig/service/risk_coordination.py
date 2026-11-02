"""Execution-linked risk checkpoints and kill switch share the OMS transaction.

Complete immutable evidence is synced before its authoritative index is admitted.
A crash may leave an unindexed immutable file. It cannot publish an error risk
checkpoint without the same transaction engaging the existing OMS kill switch.
Pure research-ledger calls remain independent and preserve nanosecond precision.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from gig.execution.oms import OfflineOMS
from gig.execution.outbox import ExecutionOutbox
from gig.execution.scheduler import DurableTWAPScheduler
from gig.service.contracts import Contract
from gig.service.execution_workspace import _stamp
from gig.service.portfolio_workspace import PortfolioWorkspace
from gig.service.workspace import TerminalWorkspace


def reconcile_with_risk_controls(base_dir: Path, payload: dict[str, Any], *,
                                 fault_hook: Callable[[str], None] | None = None) -> Contract:
    # Precision validation is unconditional and precedes even opening the OMS.
    at = _stamp(payload["now"])
    workspace = TerminalWorkspace(base_dir)
    path = workspace._path("execution-engine/oms.sqlite")
    if not path.is_file():
        return PortfolioWorkspace(base_dir).run(payload)
    hook = fault_hook or (lambda _: None)
    with OfflineOMS(path) as oms:
        # Existing outbox initialization uses executescript; initialize before
        # the transaction so schema admission cannot implicitly commit it.
        outbox = ExecutionOutbox(oms)
        with oms._transaction():
            DurableTWAPScheduler(oms, outbox, create=False).ensure_not_before(at)
            result = PortfolioWorkspace(base_dir, oms=oms).run(payload)
            hook("checkpoint_indexed")
            if result.to_dict()["state"] == "ERROR" and not oms.kill_switch:
                oms.set_kill_switch(True, now=at)
            hook("risk_controls_applied")
            # Successful/stale/unknown reads never release an engaged switch;
            # release remains an explicit operation on the existing OMS.
            return result
