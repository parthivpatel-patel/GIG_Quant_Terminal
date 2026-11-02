"""Lazy local integrations; HTTP input cannot choose files, accounts, or URLs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from gig.service.contracts import Contract, State
from gig.service.serialization import public_error
from gig.service.workspace import TerminalWorkspace


class CycleServices:
    def __init__(self, workspace: TerminalWorkspace, *, finbert_dir: Path | None = None) -> None:
        self.workspace = workspace
        self.finbert_dir = finbert_dir

    def _provider(self):
        from gig.data.ingestion import ProviderWorkspace

        return ProviderWorkspace(self.workspace.base_dir)

    def _sequence(self):
        from gig.service.sequence_workspace import SequenceWorkspace

        return SequenceWorkspace(self.workspace._path("sequence"))

    def _sentiment(self):
        from gig.service.sentiment_workspace import SentimentWorkspace

        return SentimentWorkspace(self.workspace.base_dir, self.finbert_dir)

    def _execution(self):
        from gig.service.execution_workspace import ExecutionWorkspace

        return ExecutionWorkspace(self.workspace._path("execution-engine"))

    def ingest(self, payload: dict[str, Any]) -> Contract:
        return self._provider().ingest(payload)

    def universe(self, payload: dict[str, Any]) -> Contract:
        return self._provider().universe(payload)

    def train_sequence(self, payload: dict[str, Any]) -> Contract:
        return self._sequence().train(payload)

    def predict_sequence(self, payload: dict[str, Any]) -> Contract:
        return self._sequence().predict(payload)

    def analyze_finbert(self, payload: dict[str, Any]) -> Contract:
        if self.finbert_dir is None:
            raise ImportError("A verified local sentiment checkpoint has not been configured")
        return self._sentiment().analyze(payload)

    def execution(self, payload: dict[str, Any]) -> Contract:
        return self._execution().run(payload)

    def _datasets(self):
        from gig.service.dataset_workspace import PersistedDatasetWorkspace
        return PersistedDatasetWorkspace(self.workspace.base_dir)

    def _research(self):
        from gig.service.persisted_research import PersistedResearchWorkspace
        return PersistedResearchWorkspace(self.workspace.base_dir)

    def _allocation(self):
        from gig.service.allocation_workspace import ResearchAllocationWorkspace
        return ResearchAllocationWorkspace(self.workspace.base_dir)

    def _accounting(self):
        from gig.service.portfolio_workspace import PortfolioWorkspace
        return PortfolioWorkspace(self.workspace.base_dir)

    def research_dataset(self, payload: dict[str, Any]) -> Contract:
        return self._datasets().build(payload)

    def research_model(self, payload: dict[str, Any]) -> Contract:
        return self._research().run(payload)

    def research_predict(self, payload: dict[str, Any]) -> Contract:
        return self._research().predict(payload)

    def research_allocate(self, payload: dict[str, Any]) -> Contract:
        return self._allocation().build(payload)

    def research_prepare(self, payload: dict[str, Any]) -> Contract:
        return self._allocation().prepare(payload)

    def portfolio_accounting(self, payload: dict[str, Any]) -> Contract:
        result = self._accounting().run(payload)
        if result.to_dict()["state"] == "ERROR":
            from gig.execution.oms import OfflineOMS
            from gig.service.execution_workspace import _stamp
            path = self.workspace._path("execution-engine/oms.sqlite")
            if path.is_file():
                with OfflineOMS(path) as oms:
                    oms.set_kill_switch(True, now=_stamp(payload["now"]))
        return result

    def section(self, engine: str, section: str, as_of: Any, maximum: float | None,
                *, universe_id: str | None = None) -> Contract:
        result = None
        if engine == "sequence":
            result = self._sequence().section(section, as_of, maximum)
        elif engine == "finbert":
            if self.finbert_dir is not None:
                result = self._sentiment().section(section, as_of, maximum)
        elif engine == "provider":
            if section == "operations":
                result = self._provider().status()
            elif section == "markets":
                result = self._provider().rates(as_of, max_age_seconds=maximum)
            elif section == "research" and universe_id is not None:
                result = self._provider().universe_snapshot(universe_id, as_of)
        elif engine == "execution-engine":
            result = self._execution().section(section, as_of, maximum)
        elif engine == "persisted":
            if section in {"markets", "operations"}:
                result = self._datasets().section(section, as_of, maximum)
            else:
                result = self._research().section(section, as_of, maximum)
                if result is None and section == "research":
                    result = self._datasets().section(section, as_of, maximum)
        elif engine == "allocation":
            result = self._allocation().section(section, as_of, maximum)
        elif engine == "accounting":
            result = self._accounting().section(section, as_of, maximum)
        if result is not None:
            return result
        return Contract.admit(State.UNAVAILABLE, None, source=f"local-{engine}",
                              classification="UNAVAILABLE", scope=f"terminal.{section}",
                              error=public_error("DATA_UNAVAILABLE"))
