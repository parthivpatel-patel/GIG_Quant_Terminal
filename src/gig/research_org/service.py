"""One local worker. It runs the approved queue, then stops.

Start is explicit. The thread is not created when the module is imported.
A stop waits for the current job. A dead process leaves a lease that expires;
the next start reclaims that job. The worker will not invent a second study
after the one approved historical candidate.
"""

from __future__ import annotations

import math
import resource
import threading
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from gig.research_org.backlog import queue_view
from gig.research_org.ledger import ResearchLedger
from gig.research_org.scheduler import propose_candidate, tick

MIN_CADENCE = 0.05
MAX_CADENCE = 3600.0
DEFAULT_JOB_BUDGET = 4
DEFAULT_LEASE_SECONDS = 30.0
DEFAULT_MAX_DB_BYTES = 8_000_000
DEFAULT_MAX_CPU_SECONDS = 120.0
DEFAULT_MAX_RSS_KB = 2_000_000


class ResearchWorker:
    def __init__(self, ledger: ResearchLedger, *, max_jobs: int = DEFAULT_JOB_BUDGET,
                 lease_seconds: float = DEFAULT_LEASE_SECONDS,
                 max_db_bytes: int = DEFAULT_MAX_DB_BYTES,
                 max_cpu_seconds: float = DEFAULT_MAX_CPU_SECONDS,
                 max_rss_kb: int = DEFAULT_MAX_RSS_KB):
        for name, value in (("max_jobs", max_jobs), ("max_db_bytes", max_db_bytes),
                            ("max_rss_kb", max_rss_kb)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("lease_seconds", lease_seconds), ("max_cpu_seconds", max_cpu_seconds)):
            if not _positive_finite(value) or (name == "lease_seconds" and value > 3600):
                raise ValueError(f"{name} must be positive and finite")
        self.ledger = ledger
        self.max_jobs = max_jobs
        self.lease_seconds = lease_seconds
        self.max_db_bytes = max_db_bytes
        self.max_cpu_seconds = max_cpu_seconds
        self.max_rss_kb = max_rss_kb
        self.owner = uuid.uuid4().hex
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # Charge process CPU only after this worker's first explicit start.
        # Keep the baseline across pause/resume and restarts of the same object;
        # unrelated work before start must not exhaust a new worker's allowance.
        self._cpu_started: float | None = None

    def start(self, cadence_seconds: float = 5.0) -> dict[str, Any]:
        cadence = _cadence(cadence_seconds)
        with self._lock:
            self.ledger.set_desired("RUNNING", cadence, reason=None)
            if self._thread is not None and self._thread.is_alive():
                return self.snapshot()
            if self._cpu_started is None:
                self._cpu_started = time.process_time()
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="gig-research-worker", daemon=True)
            self._thread.start()
        return self.snapshot()

    def pause(self) -> dict[str, Any]:
        self.ledger.set_desired("PAUSED", reason="paused by request")
        return self.snapshot()

    def resume(self, cadence_seconds: float | None = None) -> dict[str, Any]:
        current = float(self.ledger.scheduler_state()["cadence_seconds"])
        return self.start(current if cadence_seconds is None else cadence_seconds)

    def stop(self, timeout: float = 30.0) -> dict[str, Any]:
        self.ledger.set_desired("STOPPED", reason="stopped by request")
        self._join(timeout)
        return self.snapshot()

    def shutdown(self, timeout: float = 30.0) -> None:
        """Leave the desired state on disk. A later process can resume RUNNING."""
        self._join(timeout)

    def recover(self) -> dict[str, Any]:
        """Start only when the ledger still says RUNNING. A fresh ledger stays stopped."""
        state = self.ledger.scheduler_state()
        if state["desired"] == "RUNNING":
            return self.start(float(state["cadence_seconds"]))
        return self.snapshot()

    def _join(self, timeout: float) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        if self.ledger.path.is_file() and not (thread and thread.is_alive()):
            self.ledger.release_lease(self.owner)

    def wait(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def snapshot(self) -> dict[str, Any]:
        state = self.ledger.scheduler_state()
        state["alive"] = bool(self._thread and self._thread.is_alive())
        state["owner"] = self.owner
        state["rss_kb"] = _rss_kb()
        state["cpu_seconds"] = round(self._cpu_seconds(), 6)
        state["db_bytes"] = self.ledger.path.stat().st_size if self.ledger.path.is_file() else 0
        return state

    def _budget_reason(self) -> str | None:
        path = self.ledger.path
        if path.is_file() and path.stat().st_size > self.max_db_bytes:
            return "storage budget"
        if self._cpu_seconds() > self.max_cpu_seconds:
            return "cpu budget"
        if _rss_kb() > self.max_rss_kb:
            return "memory budget"
        return None

    def _cpu_seconds(self) -> float:
        """Conservative process-wide CPU since first start, including other threads."""
        if self._cpu_started is None:
            return 0.0
        return max(0.0, time.process_time() - self._cpu_started)

    def _loop(self) -> None:
        completed = 0
        while not self._stop.is_set():
            desired = self.ledger.scheduler_state()["desired"]
            if desired == "STOPPED":
                break
            if desired != "RUNNING":
                if self._stop.wait(0.05):
                    break
                continue
            reason = self._budget_reason()
            if reason is not None or completed >= self.max_jobs:
                self.ledger.set_desired("PAUSED", reason=reason or "process job budget")
                break
            if not self.ledger.acquire_lease(self.owner, self.lease_seconds):
                if self._wait():
                    break
                continue
            try:
                if not any(row["status"] in {"QUEUED", "RUNNING"} for row in queue_view(self.ledger)):
                    propose_candidate(self.ledger)
                result = tick(self.ledger, lease_owner=self.owner, lease_seconds=self.lease_seconds)
            finally:
                self.ledger.release_lease(self.owner)
            action = result["tick"]["action"]
            if action == "idle":
                self.ledger.set_desired("PAUSED", reason="approved backlog is empty")
                break
            if action == "failed":
                job_id = result["tick"]["job_id"]
                job = next((row for row in self.ledger.jobs() if row["job_id"] == job_id), None)
                if job is not None and job["status"] == "FAILED":
                    self.ledger.set_desired("PAUSED", reason="failure escalation")
                    break
            if action == "ran":
                completed += 1
            if self._wait():
                break

    def _wait(self) -> bool:
        cadence = float(self.ledger.scheduler_state()["cadence_seconds"])
        deadline = time.monotonic() + cadence
        while time.monotonic() < deadline:
            if self._stop.is_set():
                return True
            if self.ledger.scheduler_state()["desired"] != "RUNNING":
                return False
            time.sleep(0.02)
        return False


def _positive_finite(value: float) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value > 0
    except OverflowError:
        return False


def _cadence(value: float) -> float:
    if not _positive_finite(value):
        raise ValueError("cadence")
    number = float(value)
    if number < MIN_CADENCE or number > MAX_CADENCE:
        raise ValueError("cadence")
    return number


def _rss_kb() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes. Other platforms are not used here.
    return int(usage)


def scheduler_contract(worker: ResearchWorker, action: str, cadence_seconds: float | None = None) -> Any:
    from gig.research_org.cycle import status_contract
    control(worker, action, cadence_seconds)
    return status_contract(worker.ledger, public_scheduler(worker))


def control(worker: ResearchWorker, action: str, cadence_seconds: float | None = None) -> dict[str, Any]:
    if action == "start":
        return worker.start(5.0 if cadence_seconds is None else cadence_seconds)
    if action == "pause":
        return worker.pause()
    if action == "resume":
        return worker.resume(cadence_seconds)
    if action == "stop":
        return worker.stop()
    raise ValueError("scheduler action")


def public_scheduler(worker: ResearchWorker) -> dict[str, Any]:
    """Fields safe to show. The lease owner token stays out of the payload."""
    state = worker.snapshot()
    until = state.get("lease_until")
    held = bool(until) and str(until) > datetime.now(UTC).isoformat()
    return {
        "desired": state["desired"],
        "cadence_seconds": state["cadence_seconds"],
        "alive": state["alive"],
        "lease_held": held,
        "candidates": state["candidates"],
        "reason": state["reason"],
        "rss_kb": state["rss_kb"],
        "cpu_seconds": state["cpu_seconds"],
        "db_bytes": state["db_bytes"],
        "background_thread": state["alive"],
    }
