"""Fixed-path financial document actions; no HTTP source URL or credential input."""
from __future__ import annotations

from pathlib import Path

from gig.financial_intelligence.reference import treasury_document
from gig.financial_intelligence.store import FinancialCorpus
from gig.service.contracts import Contract, State
from gig.service.serialization import public_error
from gig.service.workspace import TerminalWorkspace, _fields


class FinancialIntelligenceWorkspace:
    def __init__(self, directory: Path):
        self.workspace = TerminalWorkspace(directory)
        self.corpus = FinancialCorpus(self.workspace._path("financial-intelligence"))

    @staticmethod
    def _contract(data, *, state=State.AVAILABLE, as_of=None):
        return Contract.admit(state, data, source="gig-financial-document-research",
            classification=data.get("classification", "HISTORICAL") if data else "UNAVAILABLE",
            as_of=as_of, scope="workspace.financial-intelligence", readiness="RESEARCH_ONLY",
            evidence_state="HASH_VERIFIED_SOURCE_CITATIONS_NOT_FINANCIAL_ADVICE",
            error=public_error("DATA_UNAVAILABLE") if state == State.UNAVAILABLE else None)

    def run(self, payload):
        if not isinstance(payload, dict) or not isinstance(payload.get("operation"), str):
            raise ValueError("explicit financial operation required")
        operation = payload["operation"]
        if operation == "ingest-reference":
            _fields(payload, {"operation"}, {"operation"})
            metadata = self.corpus.admit(*treasury_document())
            return self._contract({"title": "Admitted permitted Treasury document", "metadata": metadata,
                "classification": metadata["classification"], "document_id": metadata["document_id"],
                "monitoring_state": "NOT_RUNNING", "trading_approved": False}, as_of=metadata["known_at"])
        if operation == "admit-authored":
            _fields(payload, {"operation", "source_id", "title", "text", "license"},
                    {"operation", "source_id", "title", "text", "license"})
            metadata = self.corpus.admit_authored({k: v for k, v in payload.items() if k != "operation"})
            return self._contract({"title": "Admitted authored document; source authenticity unverified",
                "metadata": metadata, "classification": "SYNTHETIC", "document_id": metadata["document_id"],
                "trading_approved": False}, as_of=metadata["known_at"])
        if operation in {"search", "answer"}:
            _fields(payload, {"operation", "query", "as_of", "limit", "expand", "retrieval_mode", "latent_dimensions", "extract_facts"}, {"operation", "query", "as_of"})
            data = self.corpus.search(payload["query"], payload["as_of"], limit=payload.get("limit", 5),
                expand=payload.get("expand", False), answer=operation == "answer",
                retrieval_mode=payload.get("retrieval_mode", "bm25"), latent_dimensions=payload.get("latent_dimensions", 16),
                extract_facts=payload.get("extract_facts", False))
            if data is None:
                return self._contract(None, state=State.UNAVAILABLE, as_of=payload["as_of"])
            return self._contract({"title": ("Cited financial-document lexical research" if data["implementation"] == "gig.lexical-bm25.v1"
                else "Cited financial-document retrieval and mention research"), **data}, as_of=data["as_of"])
        if operation == "replay":
            _fields(payload, {"operation", "snapshot_id"}, {"operation", "snapshot_id"})
            try:
                data = self.corpus.replay(payload["snapshot_id"])
            except FileNotFoundError:
                return self._contract(None, state=State.UNAVAILABLE)
            return self._contract({"title": ("Cited financial-document lexical research" if data["implementation"] == "gig.lexical-bm25.v1"
                else "Cited financial-document retrieval and mention research"), **data}, as_of=data["as_of"])
        raise ValueError("unsupported financial research operation")
