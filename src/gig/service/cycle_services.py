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
        from gig.service.risk_coordination import reconcile_with_risk_controls
        return reconcile_with_risk_controls(self.workspace.base_dir, payload)

    def _depth(self):
        from gig.service.depth_workspace import DepthWorkspace
        return DepthWorkspace(self.workspace.base_dir)

    def admit_depth(self, payload: dict[str, Any]) -> Contract:
        return self._depth().ingest(payload)

    def _pairs(self):
        from gig.service.pairs_workspace import PairsResearchWorkspace
        return PairsResearchWorkspace(self.workspace.base_dir)

    def pairs_research(self, payload: dict[str, Any]) -> Contract:
        return self._pairs().run(payload)

    def pairs_replay(self, payload: dict[str, Any]) -> Contract:
        return self._pairs().replay(payload)

    def _options(self):
        from gig.service.options_workspace import OptionsWorkspace
        return OptionsWorkspace(self.workspace.base_dir)

    def options_admit(self, payload: dict[str, Any]) -> Contract:
        return self._options().ingest(payload)

    def options_research(self, payload: dict[str, Any]) -> Contract:
        return self._options().run(payload)

    def options_replay(self, payload: dict[str, Any]) -> Contract:
        return self._options().replay(payload)

    def _signals(self):
        from gig.service.signal_workspace import SignalWorkspace
        return SignalWorkspace(self.workspace.base_dir)

    def research_signals(self, payload: dict[str, Any]) -> Contract:
        return self._signals().run(payload)

    def signals_replay(self, payload: dict[str, Any]) -> Contract:
        return self._signals().replay(payload)

    def _overlay(self):
        from gig.service.overlay_workspace import OptionsOverlayWorkspace
        return OptionsOverlayWorkspace(self.workspace.base_dir)

    def options_overlay(self, payload: dict[str, Any]) -> Contract:
        return self._overlay().run(payload)

    def overlay_replay(self, payload: dict[str, Any]) -> Contract:
        return self._overlay().replay(payload)

    def _funded(self):
        from gig.service.funded_workspace import FundedWorkspace
        return FundedWorkspace(self.workspace.base_dir)

    def funded_account(self, payload: dict[str, Any]) -> Contract:
        return self._funded().run(payload)

    def _economics(self):
        from gig.service.economic_workspace import EconomicResearchWorkspace
        return EconomicResearchWorkspace(self.workspace.base_dir)

    def economic_research(self, payload: dict[str, Any]) -> Contract:
        return self._economics().run(payload)

    def economic_replay(self, payload: dict[str, Any]) -> Contract:
        return self._economics().replay(payload)

    def _microstructure(self):
        from gig.service.microstructure_workspace import MicrostructureWorkspace
        return MicrostructureWorkspace(self.workspace.base_dir)

    def microstructure_admit(self, payload: dict[str, Any]) -> Contract:
        return self._microstructure().ingest(payload)

    def microstructure_research(self, payload: dict[str, Any]) -> Contract:
        return self._microstructure().run(payload)

    def microstructure_replay(self, payload: dict[str, Any]) -> Contract:
        return self._microstructure().replay(payload)

    def _funded_risk(self):
        from gig.service.funded_risk_workspace import FundedRiskWorkspace
        return FundedRiskWorkspace(self.workspace.base_dir)

    def funded_risk(self, payload: dict[str, Any]) -> Contract:
        return self._funded_risk().run(payload)

    def funded_risk_replay(self, payload: dict[str, Any]) -> Contract:
        return self._funded_risk().replay(payload)

    def _lifecycle(self):
        from gig.service.lifecycle_workspace import FundedLifecycleWorkspace
        return FundedLifecycleWorkspace(self.workspace.base_dir)

    def funded_lifecycle(self, payload: dict[str, Any]) -> Contract:
        return self._lifecycle().run(payload)

    def _statements(self):
        from gig.service.statement_workspace import StatementWorkspace
        return StatementWorkspace(self.workspace.base_dir)

    def statement_admit(self, payload: dict[str, Any]) -> Contract:
        return self._statements().ingest(payload)

    def statement_reconcile(self, payload: dict[str, Any]) -> Contract:
        return self._statements().run(payload)

    def statement_replay(self, payload: dict[str, Any]) -> Contract:
        return self._statements().replay(payload)

    def _observed(self):
        from gig.service.observed_workspace import ObservedWorkspace
        return ObservedWorkspace(self.workspace.base_dir)

    def observed_reference(self, payload: dict[str, Any]) -> Contract:
        return self._observed().run(payload)

    def _equity(self):
        from gig.service.equity_workspace import ObservedEquityWorkspace
        return ObservedEquityWorkspace(self.workspace.base_dir)

    def equity_reference(self, payload: dict[str, Any]) -> Contract:
        return self._equity().run(payload)

    def section(self, engine: str, section: str, as_of: Any, maximum: float | None,
                *, universe_id: str | None = None, instrument_id: str | None = None) -> Contract:
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
        elif engine == "depth":
            result = self._depth().section(section, as_of, maximum, instrument_id=instrument_id)
        elif engine == "pairs":
            result = self._pairs().section(section, as_of, maximum)
        elif engine == "options-research":
            result = self._options().section(section, as_of, maximum)
        elif engine == "signals-research":
            result = self._signals().section(section, as_of, maximum)
        elif engine == "options-overlay":
            result = self._overlay().section(section, as_of, maximum)
        elif engine == "funded-account":
            result = self._funded().section(section, as_of, maximum)
        elif engine == "economic-research":
            result = self._economics().section(section, as_of, maximum)
        elif engine == "microstructure-research":
            result = self._microstructure().section(section, as_of, maximum)
        elif engine == "funded-risk":
            result = self._funded_risk().section(section, as_of, maximum)
        elif engine == "funded-lifecycle":
            result = self._lifecycle().section(section, as_of, maximum)
        elif engine == "independent-statement":
            result = self._statements().section(section, as_of, maximum)
        elif engine == "observed-reference":
            result = self._observed().section(section, as_of, maximum)
        elif engine == "observed-equity":
            result = self._equity().section(section, as_of, maximum)
        if result is not None:
            return result
        return Contract.admit(State.UNAVAILABLE, None, source=f"local-{engine}",
                              classification="UNAVAILABLE", scope=f"terminal.{section}",
                              error=public_error("DATA_UNAVAILABLE"))
