"""Fixed-path financial document actions; no HTTP source URL or credential input."""
from __future__ import annotations

from pathlib import Path

from gig.financial_intelligence import documents, research
from gig.financial_intelligence.citations import verify_citation
from gig.financial_intelligence.reference import treasury_document
from gig.financial_intelligence.store import FinancialCorpus
from gig.service.contracts import Contract, State
from gig.service.serialization import public_error
from gig.service.workspace import TerminalWorkspace, _fields

RESEARCH = {
    "statements": ({"document_id"}, set(), "Cited statement line items; parsed mentions, not verified facts"),
    "synthesize": ({"query", "as_of"}, {"per_document", "expand"}, "Extractive multi-document digest with citations"),
    "watch-create": ({"query", "as_of"}, {"limit", "expand", "retrieval_mode", "latent_dimensions"},
                     "Saved research watch; checked only on explicit request"),
    "watch-check": ({"watch_id", "as_of"}, set(), "Explicit watch check against its saved baseline"),
    "investigate": ({"title", "questions", "as_of"}, set(), "Cited investigation report; extractive findings only"),
    "transcript": ({"document_id"}, {"nlp_profile"}, "Cited transcript structure; speaker labels and cue words, not interpretation"),
    "compare": ({"base_document_id", "revised_document_id"}, set(),
                "Cited document comparison; text differences, not verified restatements"),
    "mention-diff": ({"base_document_id", "revised_document_id"}, {"nlp_profile"},
                     "Cited mention differences; parsed text, not verified restatements"),
    "research-replay": ({"record_id"}, set(), "Saved financial research record replay"),
}


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

    @staticmethod
    def _title(data):
        if data.get("retrieval_mode", "bm25") in {"latent", "hybrid"}:
            return "Cited financial-document corpus-local LSA research"
        if data.get("extract_facts", False):
            return "Cited financial-document lexical research and mention rules"
        return "Cited financial-document lexical research"

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
        if operation == "admit-html":
            fields = {"operation", "source_id", "title", "html", "license"}
            _fields(payload, fields, fields)
            metadata = self.corpus.admit_html({k: v for k, v in payload.items() if k != "operation"})
            return self._contract({"title": "Admitted authored HTML document; source authenticity unverified",
                "metadata": metadata, "classification": "SYNTHETIC", "document_id": metadata["document_id"],
                "passage_count": len(metadata["passage_hashes"]), "trading_approved": False}, as_of=metadata["known_at"])
        if operation in RESEARCH:
            return self._research(operation, payload)
        if operation in {"search", "answer"}:
            _fields(payload, {"operation", "query", "as_of", "limit", "expand", "retrieval_mode", "latent_dimensions", "extract_facts", "nlp_profile"}, {"operation", "query", "as_of"})
            data = self.corpus.search(payload["query"], payload["as_of"], limit=payload.get("limit", 5),
                expand=payload.get("expand", False), answer=operation == "answer",
                retrieval_mode=payload.get("retrieval_mode", "bm25"), latent_dimensions=payload.get("latent_dimensions", 16),
                extract_facts=payload.get("extract_facts", False), nlp_profile=payload.get("nlp_profile", "v1"))
            if data is None:
                return self._contract(None, state=State.UNAVAILABLE, as_of=payload["as_of"])
            return self._contract({"title": self._title(data), **data}, as_of=data["as_of"])
        if operation == "verify-citation":
            _fields(payload, {"operation", "snapshot_id", "passage_id"}, {"operation", "snapshot_id", "passage_id"})
            try:
                data = verify_citation(self.corpus, payload["snapshot_id"], payload["passage_id"])
            except FileNotFoundError:
                return self._contract(None, state=State.UNAVAILABLE)
            return self._contract({"title": "Saved citation byte verification; financial truth UNKNOWN", **data}, as_of=data["as_of"])
        if operation == "replay":
            _fields(payload, {"operation", "snapshot_id"}, {"operation", "snapshot_id"})
            try:
                data = self.corpus.replay(payload["snapshot_id"])
            except FileNotFoundError:
                return self._contract(None, state=State.UNAVAILABLE)
            return self._contract({"title": self._title(data), **data}, as_of=data["as_of"])
        raise ValueError("unsupported financial research operation")


    def _research(self, operation, payload):
        required, optional, title = RESEARCH[operation]
        _fields(payload, {"operation", *required, *optional}, {"operation", *required})
        options = {k: payload[k] for k in optional if k in payload}
        try:
            if operation == "statements":
                data = research.statements(self.corpus, payload["document_id"])
            elif operation == "transcript":
                data = documents.transcript(self.corpus, payload["document_id"], profile=options.get("nlp_profile", "v1"))
            elif operation == "compare":
                data = documents.compare(self.corpus, payload["base_document_id"], payload["revised_document_id"])
            elif operation == "mention-diff":
                data = documents.mention_diff(self.corpus, payload["base_document_id"], payload["revised_document_id"],
                                              profile=options.get("nlp_profile", "v1"))
            elif operation == "synthesize":
                data = research.multi_document_digest(self.corpus, payload["query"], payload["as_of"], **options)
            elif operation == "watch-create":
                data = research.watch_create(self.corpus, payload["query"], payload["as_of"], **options)
            elif operation == "watch-check":
                data = research.watch_check(self.corpus, research.load(self.corpus, payload["watch_id"]), payload["as_of"])
            elif operation == "investigate":
                data = research.investigate(self.corpus, payload["title"], payload["questions"], payload["as_of"])
            else:
                data = research.replay(self.corpus, payload["record_id"])
        except FileNotFoundError:
            return self._contract(None, state=State.UNAVAILABLE)
        if data is None:
            return self._contract(None, state=State.UNAVAILABLE, as_of=payload.get("as_of"))
        if operation != "research-replay":
            research.save(self.corpus, data)
        return self._contract({"title": title, **data}, as_of=data["as_of"])
