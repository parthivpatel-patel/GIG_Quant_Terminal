# Financial Intelligence requirements-to-implementation matrix

Scope: the user's16-capability roadmap for an original financial-intelligence
engine. This is not a claim of AlphaSense product parity or access to proprietary
reports/transcripts. Existing NLP, EDGAR metadata, Global Intelligence and source
permissions were inspected before adding this slice. No external queue/vector
DB/service, credentials, new dependency or pretrained model is introduced.

| # | Required capability | Current disposition | Source / acceptance / next gate |
| --- | --- | --- | --- |
|1|Document ingestion/normalization|IMPLEMENTED bounded XML/plain-text slice|`financial_intelligence/reference.py` reuses canonical Treasury validation; `store.py` retains raw BLOB, byte ranges, source hashes, exact authored UTF-8 lines and immutable revisions. PDF/HTML/OCR remain planned.|
|2|EDGAR/permitted financial documents|PARTIAL / SEC BLOCKED|Existing `data/providers/edgar.py` provides metadata only and requires a genuine contact User-Agent. No SEC request/identity was used. First corpus is already-packaged CC0 Treasury XML with pinned source and permission metadata.|
|3|Entity extraction/linking|PLANNED|Existing ticker/CIK metadata and authored Global Intelligence themes are not a general document entity graph. Need instrument/issuer authority and ambiguity tests.|
|4|Financial terminology|IMPLEMENTED minimal / PARTIAL|Explicit optional GIG-authored tenor aliases, not a comprehensive ontology. Original text remains unchanged; expansions are saved with query evidence.|
|5|BM25 lexical retrieval|IMPLEMENTED / TESTED|Original bounded BM25, deterministic ties, 512-character/64-term queries,≤4,096passages. All TF/IDF/length statistics use only visible document revisions; independent formula reference.|
|6|Semantic embedding retrieval|UNAVAILABLE / PLANNED|No licensed embedding weights loaded or executed. No lexical result is presented as semantic similarity.|
|7|Hybrid retrieval/reranking|UNAVAILABLE / PLANNED|Requires separately verified semantic/reranking components and relevance evaluation; BM25 is explicitly lexical.|
|8|Source-grounded QA|PARTIAL EXTRACTIVE ONLY|Returns a retrieved normalized source passage, not generated reasoning/financial advice. No match produces null/unavailable answer. No generative model is invoked.|
|9|Passage citations/provenance|IMPLEMENTED / TESTED|Document/passages/snapshot identities, original raw SHA, exact source byte spans/excerpts, known time and normalization type. Treasury prose is clearly GIG rendering of XML, not publisher-authored narrative.|
|10|Financial sentiment/events|EXISTING / NOT DOCUMENT-INTEGRATED|Existing lexicon/VADER/FinBERT/event workflows retained; document sentiment/event mapping planned. FinBERT remains noncommercial research-only, unavailable without pinned local assets.|
|11|Earnings/statement analysis|PLANNED|No unlicensed transcripts or fabricated issuer facts. Need permitted filing texts/XBRL taxonomy/context and period/unit validation.|
|12|Multi-document synthesis|PARTIAL RETRIEVAL / SYNTHESIS PLANNED|Search spans latest visible revisions; does not claim comparative reasoning or generated synthesis.|
|13|KPI extraction/numeric verification|PARTIAL TREASURY ONLY|Reuses Decimal quoted-yield parser, retains actual units/date precision. General issuer KPI/XBRL calculation and reconciliation remain planned.|
|14|Saved searches/monitoring|PARTIAL|Immutable query snapshots/replay implemented. No monitoring daemon or background source acquisition runs; monitoring_state=NOT_RUNNING.|
|15|Research agents/investigations|EXISTING / NEW BINDING PLANNED|Existing approved Research Director/lease workflow retained. No new document objective is automatically approved; arbitrary retrieved instructions never execute.|
|16|Terminal/Director/Lab/Charts integration|PARTIAL|Existing Research workbench has explicit reference/search/answer/replay actions and readable cited output. All16sections remain. Automatic Director/Charts cross-linking is planned; documents never become tradable market observations.|

## Verified first vertical

Explicit local reference admission → immutable original XML →250normalized
Treasury date passages →point-in-time lexical retrieval →extractive quote with
original-byte citation →saved query identity →independent restart replay →existing
Research workbench. No remote provider request occurs at runtime.

The archive was acquired2026-11-05T03:01:08Z; its284,423bytes and pinned
SHA256`99f80b80cfeff0a93e339735d72672080d9209073b5fe814c1252e88d1428227`
are retained. Permission is the catalog's verified CC0 field recorded in the
existing adjacent provenance JSON; that permission metadata is pinned too.
2023observation dates do not imply2023knowledge. Original publication time is
UNKNOWN. These are interpolated Treasury reference yields, not transactions,
issuer earnings, executable liquidity or evidence of equity alpha.

Authored protocol documents require an explicit GIG-AUTHORED attestation and stay
SYNTHETIC; attestation does not verify third-party rights. No arbitrary URL/file
fetch, HTML execution, model promotion or trading action is available. Document
text is inert evidence, including prompt-injection-looking content.

Federal Reserve statement/disclaimer/copyright endpoint probes returnedHTTP403
from this environment. No protected source was bypassed. SEC acquisition remains
blocked pending a genuine operator-approved contact identity. Unlicensed vendor
transcripts, research reports, FINRA data and model weights are not admitted.

The16rows are a development matrix, not institutional acceptance or a replacement
for the existing219-section tracker. `fully_funded_certified=false`,
`T4_NOT_CERTIFIED`, historical§139UNSUPPLIED. Docker capacity and physicalARM64
acceptance remain blocked/unverified. Strict$0, localCPU and standardSQLite only.
