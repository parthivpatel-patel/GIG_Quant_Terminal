# Financial Intelligence requirements-to-implementation matrix

Scope: the user's16-capability roadmap for an original financial-intelligence
engine. This is not a claim of AlphaSense product parity or access to proprietary
reports/transcripts. Existing NLP, EDGAR metadata, Global Intelligence and source
permissions were inspected before adding this slice. No external queue/vector
DB/service, credentials, new dependency or pretrained model is introduced.

| # | Required capability | Current disposition | Source / acceptance / next gate |
| --- | --- | --- | --- |
|1|Document ingestion/normalization|IMPLEMENTED bounded XML/plain-text/HTML slice|`financial_intelligence/reference.py` reuses canonical Treasury validation; `store.py` retains raw BLOB, byte ranges, source hashes, exact authored UTF-8 lines and immutable revisions. `ingest.py` admits GIG-authored HTML only: scripts, styles and head never become text, each visible block (table rows keep `|`-separated cells) cites its original byte range, and every read re-derives the block text from those bytes. No URL or file is fetched. PDF/OCR remain planned.|
|2|EDGAR/permitted financial documents|PARTIAL / SEC BLOCKED|Existing `data/providers/edgar.py` provides metadata only and requires a genuine contact User-Agent. No SEC request/identity was used. First corpus is already-packaged CC0 Treasury XML with pinned source and permission metadata.|
|3|Entity extraction/linking|PARTIAL ORIGINAL RULE MENTIONS|`nlp.py` allowlists frozen v1/default and opt-in v2 annotations. v2 validates complete bounded NYSE/NASDAQ symbol source mentions including one-letter dot share classes; unsupported suffixes/Unicode lookalikes never become partial ticker mentions. Instrument identity, issuer linking, exchange membership and real existence remain UNKNOWN. No entity graph or authority claim. See `NLP_PROFILES.md`.|
|4|Financial terminology|IMPLEMENTED minimal / PARTIAL|Explicit optional GIG-authored tenor aliases, not a comprehensive ontology. Original text remains unchanged; expansions are saved with query evidence.|
|5|BM25 lexical retrieval|IMPLEMENTED / TESTED|Original bounded BM25, deterministic ties, 512-character/64-term queries,≤4,096passages. All TF/IDF/length statistics use only visible document revisions; independent formula reference.|
|6|Semantic embedding retrieval|PARTIAL CORPUS_LOCAL_LSA / TESTED|Actual visible-corpus TF-IDF/SVD fit, dense projection and cosine;≤256passages/512retained terms/16dimensions with reject bounds. Saved model/algorithm identity and exact restart replay. Explicitly not pretrained neural semantics/paraphrase understanding; pure numbers/dates excluded by saved feature policy. See `LATENT_AND_FINANCIAL_MENTIONS.md`.|
|7|Hybrid retrieval/reranking|PARTIAL BM25+LSA RRF / TESTED|Original reciprocal-rank fusion over complete bounded BM25/LSA candidate rankings, component scores/ranks and independent numerical reference. Four-topic authored evaluation documented with limits; no trained cross-encoder or production relevance certification.|
|8|Source-grounded QA|PARTIAL EXTRACTIVE ONLY|Returns a retrieved normalized source passage, not generated reasoning/financial advice. No match produces null/unavailable answer. No generative model is invoked.|
|9|Passage citations/provenance|IMPLEMENTED / TESTED|Document/passages/snapshot identities, original raw SHA, exact source byte spans/excerpts, known time and normalization type. Explicit read-only saved-citation audit verifies query binding, original/quoted byte hashes and verifier identity; unsigned hash checks do not establish authenticity, rights, financial truth or freshness. Treasury prose is clearly GIG rendering of XML, not publisher-authored narrative.|
|10|Financial sentiment/events|PARTIAL DOCUMENT EVENT MENTIONS|Original finite event-mention rules with exact source provenance and limited negation/conditional cue flags. Occurrence always UNKNOWN; no inference that a keyword event happened. Existing VADER/FinBERT workflows retained; no trained document sentiment invoked.|
|11|Earnings/statement analysis|PARTIAL CITED STATEMENT ROWS / TESTED|`research.statements` reads period-labelled table rows of one admitted document through a finite line-item alias list: exact Decimal values, parenthesized negatives, explicitly stated scale, per-share items left unscaled, margins, like-for-like year-over-year growth (never from a non-positive base), free cash flow, gross-profit and balance-sheet identity checks, and conflicting repeated rows. Independent Fraction references. Values are parsed mentions of GIG-authored SYNTHETIC text, not verified issuer facts. `documents.transcript` reads the structure of one admitted call text: speaker turns from line labels, operator/analyst/management roles from the label text only, prepared-remarks and question-and-answer sections, question/answer exchanges (including questions with no management answer), word counts by role and forward-looking sentences quoted with their cue word, hedge flag and passage citation. Speaker identity is LABEL_TEXT_ONLY_NOT_VERIFIED, statements are STATED_IN_TEXT_NOT_VERIFIED and tone is NOT_SCORED. The existing v1 mention rules run on the speaker passages, at most the first 20, and their percent, money and event matches stay cited to the original bytes. No XBRL taxonomy/context, issuer filing, licensed third-party transcript, consensus or surprise is available.|
|12|Multi-document synthesis|PARTIAL EXTRACTIVE DIGEST / TESTED|`research.multi_document_digest` quotes the top BM25 passages per visible document at an explicit cutoff and compares statement values across documents (AGREE/DIFFER/SINGLE_DOCUMENT) with citations. Disagreement is reported, never resolved. `documents.compare` aligns the passages of two admitted documents and lists ADDED, REMOVED, REWORDED and NUMERIC_ONLY passages with citations into both documents, plus statement values that differ with the exact Decimal difference; it reports text differences, not verified restatements, and does not rank their importance. No generated or comparative reasoning; generative_state=UNAVAILABLE.|
|13|KPI extraction/numeric verification|PARTIAL TREASURY + EXPLICIT UNIT MENTIONS / TESTED|Exact Decimal percent/bps/dollar and supported scale/tenor mentions (bare$currency UNKNOWN/UNSPECIFIED_DOLLAR; only explicit USD/US$ syntax identifies USD) with original UTF-8 byte or verified XML-element citations. Whole-token validation rejects exponents, ambiguous separators, fractions/identifiers, nonfinite and oversized values. Mention truth/issuer identity UNKNOWN; general issuer KPI/XBRL reconciliation planned.|
|14|Saved searches/monitoring|PARTIAL EXPLICIT WATCHES / TESTED|Immutable query snapshots/replay implemented; stored NLP profiles select their frozen parser, absent legacy profiles always select v1, and default v1 query IDs remain unchanged. `research.watch_create`/`watch_check` save a search baseline and, only on an explicit request at a later cutoff, list new/dropped cited passages and new/superseded documents. No daemon, schedule, notification or background source acquisition exists; monitoring_state=EXPLICIT_CHECK_ONLY_NO_DAEMON.|
|15|Research agents/investigations|PARTIAL EXPLICIT INVESTIGATIONS / DIRECTOR BINDING PLANNED|Existing approved Research Director/lease workflow retained. `research.investigate` runs one to eight explicit questions and files one immutable report of extractive cited findings, unanswered questions and statement checks; it is a single request, not an autonomous agent, and is not bound to a Director objective. Arbitrary retrieved instructions never execute.|
|16|Terminal/Director/Lab/Charts integration|PARTIAL|Existing Research workbench has explicit reference/search/answer/replay actions plus authored-HTML admission, statements, digest, watch, watch check, investigation, authored call text, transcript structure, document comparison and record replay, with readable cited summaries and identity handoff (`scripts/browser_financial_research.py`). All16sections remain. Automatic Director/Charts cross-linking is planned; documents never become tradable market observations.|

## Verified first and second verticals

Explicit local reference admission → immutable original XML →250normalized
Treasury date passages →point-in-time lexical retrieval →extractive quote with
original-byte citation →saved query identity →independent restart replay →existing
Research workbench. No remote provider request occurs at runtime.

Second vertical: permitted visible revisions → bounded original corpus-local
TF-IDF/SVD fit → dense cosine or BM25/LSA reciprocal-rank fusion → cited
extractive result and optional original financial mention rules → atomic saved
model/snapshot → exact restart replay without refitting. A declared12passage
authored4topic evaluation retrieves12/12relevant top-three slots at4dimensions
versus6/12possible lexical matches; unseen vocabulary intentionally fails. These
are unblinded synthetic development judgments, not financial acceptance.
See [algorithm, provenance, evaluation and limitations](LATENT_AND_FINANCIAL_MENTIONS.md).

Third vertical: explicit GIG-authored HTML → visible blocks cited to original bytes →
cited statement line items with exact arithmetic and identity checks → extractive
multi-document digest with cross-document agreement → explicitly checked watches →
bounded cited investigation report → content-addressed `research_records` with
replay from pinned documents and saved queries. Every input is SYNTHETIC authored
text; this demonstrates the mechanics, not analysis of any real issuer.

Fourth vertical: explicit GIG-authored call text → speaker turns, roles, sections and
question/answer exchanges cited to the original lines → forward-looking sentences
quoted with cue and hedge flags → a revised text admitted as a second document →
passage-level comparison cited into both documents → both results saved as
content-addressed records and replayed from the pinned documents
(`tests/test_financial_documents.py`). The call text is a SYNTHETIC fixture about a
company that does not exist; no real transcript has been admitted.

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
