# Read-only saved-citation byte verification

`verify-citation` binds one retrieved passage to an immutable saved query and
reuses the existing complete query/document integrity checks. It does not admit
documents, save snapshots, fit retrieval models or modify the ledger. The request
accepts only `operation`, `snapshot_id` and `passage_id`; identities are exact
lowercase SHA256 strings. Unknown evidence is UNAVAILABLE. A passage outside the
selected query is rejected, including another passage in the same document.

The original raw document hash, passage content/hash/byte ranges, selected query
cutoff, saved retrieval model and pinned annotation profile are verified before
returning the source excerpt. UTF-8 byte length and excerpt SHA256 are explicit.
The response includes source/query/passage identities, known/acquisition and
publication clocks, original license metadata, verifier code hash and a
deterministic `verification_id`. Caller mutations cannot change future reads.

`verification_id` hashes the deterministic response body. It is **not a stored
audit record, digital signature, external attestation or independently issued
certificate**. Repeating the request reconstructs the result from existing
immutable query/source state. No new table/schema or saved-query migration is
introduced. A changed verifier code hash produces a different proof identity;
no earlier query is rewritten. External runtime backup is a separate operation.

`verification_scope=IMMUTABLE_BYTE_PROVENANCE_ONLY` and
`signature_state=UNSIGNED_LOCAL_HASH_CHECK` describe the exact boundary.
`content_truth_state` and `issuer_identity_state` stay UNKNOWN; freshness is
NOT_ASSESSED, confidence UNAVAILABLE, certified=false and trading_approved=false.
Local hash integrity does not establish source authenticity, rights, financial
truth, currentness or institutional acceptance. Existing source license notes
are evidence metadata, not new legal authorization.

For GIG-authored UTF-8 lines, `direct_text_byte_match` can be true. For GIG's
normalized Treasury XML rendering, the readable passage and original XML bytes
remain separate and direct text/byte match is false. A successful audit does not
invent direct source mappings for rendered tokens or validate inferred facts.
Original publication time remains UNKNOWN where unsupplied; 2023 observations
do not establish 2023 knowledge. Treasury reference yields are not transactions.
Authored protocols stay SYNTHETIC and prove no observed-market coverage.

The Research workbench offers an explicit read-only action. Successful saved
queries hand off their real snapshot/passage IDs; a new empty query clears the
old passage reference. Selection never submits a request. Explicit IDs can be
restored after reload. Original bytes, normalized text, hashes and limitations
are readable; source markup is inert text. All existing terminal sections,
themes, motion/accessibility controls and model workflows remain intact.

Tests cover Unicode/original NaN text, raw XML/rendering distinction, exact hashes,
foreign citations and malformed identities, tampering, later source revisions,
read-only connections, concurrent readers, pinned hybrid replay without fitting,
HTTP closed fields, restart and empty-query handoff. Actual Chromium workflows
cover explicit actions, inert script-looking text, proof reconstruction after
revision/reload, dark/light axe and responsive widths. Automated checks do not
certify accessibility or cross-browser behavior.

No new source acquisition, provider, credential, model asset, dependency, paid
service, broker/order, production promotion or deployment. Source rights and
identity blocks remain. FinBERT noncommercial research-only;
fully_funded_certified=false;T4_NOT_CERTIFIED;§139UNSUPPLIED. Docker capacity and
physicalARM64/Ampere acceptance remain blocked/unverified. Strict$0,localCPU.
