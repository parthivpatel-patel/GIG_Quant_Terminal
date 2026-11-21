"""Read-only saved-citation byte audit, never financial truth or certification."""
from __future__ import annotations

import hashlib

from gig.financial_intelligence.store import FinancialCorpus, _sha, digest

IMPLEMENTATION = "gig.financial-citation-verification.v1"


def verify_citation(corpus: FinancialCorpus, snapshot_id: str, passage_id: str):
    """Reuse full immutable document/query validation; create no saved state."""
    snapshot_id, passage_id = _sha(snapshot_id), _sha(passage_id)
    snapshot = corpus.replay(snapshot_id)
    matches = [p for p in snapshot["rows"] if p["passage_id"] == passage_id]
    if len(matches) != 1:
        raise ValueError("citation must belong to the selected saved query")
    selected = matches[0]
    document = corpus.document(selected["document_id"])
    original = next((p for p in document["passages"] if p["passage_id"] == passage_id), None)
    fields = ("source_start", "source_end", "raw_sha256", "source_excerpt", "text",
              "normalization", "known_at", "classification", "document_id", "source_url")
    if original is None or any(original[k] != selected[k] for k in fields):
        raise ValueError("saved citation differs from immutable source passage")
    raw_quote = original["source_excerpt"].encode("utf-8")
    if len(raw_quote) != original["source_end"]-original["source_start"]:
        raise ValueError("citation byte length differs from original source range")
    metadata = document["metadata"]
    body = {"implementation": IMPLEMENTATION, "snapshot_id": snapshot_id,
            "passage_id": passage_id, "document_id": selected["document_id"],
            "as_of": snapshot["as_of"], "classification": snapshot["classification"],
            "quote_classification": original["classification"],
            "verification_scope": "IMMUTABLE_BYTE_PROVENANCE_ONLY",
            "saved_query_replay_verified": True, "passage_bound_to_query": True,
            "document_integrity_verified": True, "source_bytes_sha256": metadata["raw_sha256"],
            "quoted_bytes_sha256": hashlib.sha256(raw_quote).hexdigest(), "quoted_byte_length": len(raw_quote),
            "source_start": original["source_start"], "source_end": original["source_end"],
            "source_excerpt": original["source_excerpt"], "normalized_passage": original["text"],
            "normalization": original["normalization"],
            "direct_text_byte_match": original["text"] == original["source_excerpt"],
            "source_url": metadata["source_url"], "source_license": metadata["license"],
            "source_license_note": metadata.get("license_note"), "known_at": metadata["known_at"],
            "source_acquired_at": metadata["source_acquired_at"],
            "publication_time": metadata["publication_time"], "publication_state": metadata["publication_state"],
            "annotation_profile": snapshot.get("nlp_profile", "v1") if snapshot.get("extract_facts") else "NOT_REQUESTED",
            "retrieval_model_id": snapshot.get("model_id"),
            "signature_state": "UNSIGNED_LOCAL_HASH_CHECK", "freshness_state": "NOT_ASSESSED",
            "content_truth_state": "UNKNOWN", "issuer_identity_state": "UNKNOWN",
            "confidence_state": "UNAVAILABLE", "trading_approved": False,
            "certified": False, "new_state_written": False,
            "limitations": ["Hashes verify local byte integrity and query binding, not source authenticity, rights or financial truth",
                            "Normalized XML rendering is separate from original source bytes; direct token mappings are not inferred",
                            "This deterministic unsigned result is not an external audit, signature, freshness check or trading approval"]}
    return {**body, "verification_id": digest(body)}
