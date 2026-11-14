# Bounded corpus-local LSA and financial mention research

This second vertical adds actual corpus-local dense retrieval and hybrid rank
fusion, plus original finite financial mention rules. It uses already declared
NumPy on local CPU. It does not download weights, add dependencies, call a paid
service, fetch arbitrary URLs, access SEC under an invented identity, or ingest
unlicensed financial prose. Permitted inputs remain the packaged CC0 Treasury
archive and explicitly attested caller-authored SYNTHETIC UTF-8 lines.

`CORPUS_LOCAL_LSA` is the explicit model kind. It is not a pretrained neural
embedding model, LLM, general financial understanding or paraphrase engine. This
is a retrieval component, not an additional quant-model inventory family.

## Numerical algorithm and limits

`gig.corpus-local-lsa.v1` constructs an original document-term matrix. Its fixed
`ALPHABETIC_BEARING_TOKENS_ONLY_NO_PURE_NUMBERS_OR_DATES.v1` policy keeps lexical
tokens containing an ASCII letter, including `BC_10YEAR`. Pure numeric values
and dates are excluded from LSA features. This policy is saved in the model and
returned metadata; it is not a top-k vocabulary filter. BM25 keeps all original
numeric/date tokens. LSA alone cannot distinguish Treasury rows by date or rate.

For each visible passage and retained term, weight is
`(1 + ln(term_count)) * (1 + ln((1 + passage_count)/(1 + document_frequency)))`
when present, otherwise zero. Each matrix row is L2-normalized. NumPy's dense
SVD is actually fitted. The first requested nonzero right singular vectors form
the projection; signs are canonicalized by the largest absolute loading.
Document and query projections are L2-normalized and ranked by cosine. The
query uses the same vocabulary and IDF. Optional pre-existing authored tenor
aliases remain explicitly requested lexical expansion, never a semantic model.

A fit accepts at most256visible passages,512retained vocabulary terms and16
requested dimensions (exact integer1–16, default16). Numerical rank uses a
relative singular-value threshold of1e-12; retained dimension count is the
smaller of requested dimensions and numerical rank. Bounds reject the entire
latent/hybrid operation without passage/term truncation; existing BM25 remains
available within its original4,096passage budget. A small full-rank corpus may
have no dimensional reduction at the requested dimension count. Empty retained
vocabulary is rejected. Pure-number/date queries or unseen vocabulary produce
`UNKNOWN_VOCABULARY`; a zero projection produces `ZERO_PROJECTION`; neither
fabricates an LSA answer. Only positive cosine above1e-12is a dense candidate.

`gig.bm25-lsa-rrf.v2` fuses complete bounded candidate rankings using
`1/(60 + BM25_rank) + 1/(60 + LSA_rank)`, omitting absent components. It does not
combine incomparable BM25/cosine score magnitudes. Each hybrid row retains both
component scores and ranks. Within each component, score ties receive the mean
of their occupied ranks. A sorted score group compares every member to its first
score, with saved absolute1e-12and relative1e-10tolerances, so near-equal groups
do not chain. Thus250equal Treasury vectors each receive latent rank125.5;
ordinal order cannot create artificial dense relevance evidence. Missing one
component is explicit. Final ties use
document identity and ordinal, as existing BM25 does. Scores never express
financial confidence, predicted alpha or event probability.

## Time, persistence and replay

The corpus selects the latest revision of each logical document whose exact
nanosecond `known_at` is at or before `as_of`. Every vocabulary, frequency, IDF,
SVD fit and vector derives only from those selected passages. Future documents
and superseded revisions cannot enter historical statistics. The model stores
passage-corpus and token-statistics hashes; source document/passages retain their
existing hashes, original bytes and byte citations. Observation dates do not
imply historical knowledge; Treasury original publication time stays UNKNOWN.

The immutable `latent_models` SQLite table stores vocabulary, IDF, components,
singular values, dense vectors, NumPy version, algorithm/feature identities,
limits and fit metadata. Its SHA256 is `model_id`. Only compact model metadata
is returned through HTTP. The model and query snapshot insert in one
`BEGIN IMMEDIATE` transaction; a failure rolls both back. The additive single
`CREATE TABLE IF NOT EXISTS` migration is atomic SQLite DDL and does not change
existing document or snapshot rows.

Restart replay loads saved arrays, verifies the model hash, recomputes the
visible matrix/statistics, checks dimensions/orthonormality/document projection,
and projects the saved query without refitting SVD. The reproduced snapshot
must match its saved SHA256 exactly. Missing/tampered models fail closed.
Pre-existing `gig.lexical-bm25.v1` snapshot bodies/hashes and default search
behavior are unchanged; they replay without a model table. Newly requested
modes/extraction use `gig.financial-research.v2`. Identical fresh fitting on a
different NumPy/BLAS version is not promised byte-identical; saved replay is the
supported exact-replay mechanism, and runtime NumPy identity is recorded.

## Financial mention rules and provenance

`gig.financial-mention-rules.v1` is original rule-based extraction, explicitly
not FinBERT, trained NLP or sentiment inference. It analyzes only retrieved
passages (at most20), never executes document instructions, and returns:

- Supported explicit percent, basis-point and USD numeric mentions. Values use
  exact Decimal strings, original quoted precision, optional thousand/million/
  billion scale and exact scaled value. Metric labels are nearby text mentions;
  truth and issuer verification remain UNKNOWN.
- Explicit month/year and Treasury-field tenor mentions with exact month values.
- Treasury/Federal Reserve or exchange-qualified ticker surface mentions;
  authority-based issuer linking stays UNKNOWN.
- Finite event **mentions**, with occurrence always UNKNOWN. Sentence-local
  negation/conditional cue flags are limited lexical diagnostics, not evidence
  that an event did or did not happen, syntactic scope, or causal reasoning.

The scanner consumes whole numeric tokens before validation. Exponent syntax,
nonfinite words, fractions, underscores/identifier suffixes, ambiguous grouping,
more than32digits and unsupported formats return unsupported mentions instead
of verified subfragments. It never expands huge exponents. Only conventional
comma thousands grouping and dot decimals are supported. Unitless/unmatched
numerals and unsupported currencies are outside these rules; the absence of a
fact is not proof that a passage contains none. Percent and basis-point values
are not silently converted into returns, rates or economic effects.

Authored text facts/tenors/entities/events cite exact original UTF-8 byte spans,
including when preceding text contains multibyte characters. Treasury yield
facts additionally verify Decimal equality against the corresponding raw XML
element and cite that exact element. Other normalized mention spans explicitly
cite the original raw source passage rather than inventing a publisher-authored
phrase. Every item identifies its source passage/document/hash. Source citations
establish provenance, not source truth, issuer identity or trading suitability.

## Declared authored evaluation and observed limits

`tests/test_financial_latent_nlp.py` includes12explicitly GIG-authored synthetic
passages in4hand-declared topics: Treasury duration/rates, revenue/earnings,
liquidity/deposits and credit/default. Four in-vocabulary queries (`coupon`,
`sales`, `reserves`, `downgrade`) use3declared relevant passages each. At4LSA
dimensions, all12retrieved top-three slots match the declared topic (12/12);
BM25 returns6relevant passages total (6/12possible), because it returns only
lexically matching passages. These are development fixtures and unblinded,
authored relevance judgments; they establish co-occurrence retrieval beyond
exact matches, not a production benchmark or financial-quality acceptance.

The deliberately unseen `insolvency` query returns no LSA match. No synonym or
paraphrase knowledge is inferred. Numeric/date-only queries cannot use LSA;
hybrid lexical retrieval correctly selects Treasury2023-01-03and cites the
10-year3.79percent XML value. A separate late-date mixed query (`10-year Treasury
yield 2023-12-29`) verifies that hybrid retains the lexical date winner despite
250tied dense scores. Homogeneous Treasury renderings can collapse to
rank1and tie. This limitation is expected, not called semantic understanding.

Independent verification uses a Gram-matrix eigendecomposition to reproduce
TF-IDF projected cosine, a separate RRF formula, future-vocabulary/revision
adversaries, saved no-refit restart replay, atomic insertion failure, model
corruption, bounds/malformed inputs, Unicode byte evidence, numeric attacks,
negated/conditional event mentions and HTTP restart integration. Existing12
financial-intelligence tests continue to pass.

## HTTP hooks

Existing `search` and `answer` accept optional `retrieval_mode` (`bm25`, `latent`,
`hybrid`), `latent_dimensions` and exact boolean `extract_facts`. Default omitted
settings preserve the original BM25 snapshot schema. `answer` remains an exact
retrieved passage, not generated reasoning. `replay` still accepts only a saved
`snapshot_id`. No-corpus requests use the existing UNAVAILABLE contract; no-match
results have empty rows and null/unavailable extractive answer.

New response metadata includes `retrieval_mode`, `retrieval_implementation`,
`retrieval_state`, `model_id`, `model` and `latent_query_state` where applicable.
The `model` object includes `model_kind`, `training_scope`, `feature_policy`,
`dimensions`, `requested_dimensions`, counts and limitations. Optional
`financial_nlp` includes `facts`, `tenor_mentions`, `entity_mentions`,
`event_mentions`, `unsupported_mentions`, implementation identity and limitations.

All results remain research-only; generative models and monitoring remain
unavailable/not running, `trading_approved=false`. Broker/promotion/certification
controls are unchanged: `fully_funded_certified=false`, `T4_NOT_CERTIFIED`,
historical§139UNSUPPLIED. Strict$0and localCPU; no claim of external provider or
institutional financial acceptance.
