# Offline news and financial NLP

`gig.nlp.workflow` analyzes caller-supplied articles with source evidence. It does
not fetch articles or model weights. Admission records a caller's permission
declaration; it is not a licensing or authenticity verification service.

## Source evidence and knowledge time

`admit_article(raw)` requires `source`, `article_id`, `url`, `license`, `title`,
`body`, `published_at`, and `available_at`; `symbols` is optional provider-declared
attribution. The URL may be explicitly empty; a nonempty URL requires HTTP(S).
Unknown fields, missing permission declarations, malformed Unicode, naive times,
knowledge times preceding publication, and text exceeding 100,000 combined
characters are rejected. Original title/body, including whitespace, Unicode,
control characters, `NaN`, and `Infinity` substrings, are preserved exactly.
Timestamps normalize to UTC with up to nine fractional digits; greater precision
is rejected rather than truncated. `ArticleEvidence` holds immutable canonical
bytes plus a SHA-256 digest; each `as_dict()` call returns a detached structure.

`analyze_articles(articles, as_of=..., aliases={"Company Name":"TICKER"})`
admits at most 1,000 inputs and two million combined text characters, and excludes
articles not knowable at the cutoff. Entity output is capped at 20,000 literal
matches per article across at most 500 caller aliases.
Identity is source plus provider article ID. Replaying an identical revision is
idempotent; conflicting text at the same knowledge timestamp is rejected.
Different known revisions remain visible with `superseded_by`; only the latest
known revision contributes to aggregates. Exact duplicate title/body from other
sources retains its attribution and receives `duplicate_of`, preventing double
counting. Fuzzy duplicates are not detected.

Entity matching uses explicit caller dictionaries and provider-declared symbols.
Boundary-aware literal matching prevents regex aliases or ticker substrings from
becoming false matches. Entity offsets refer to Python Unicode codepoints in
`title + "\n" + body`. Event classification is an explainable keyword heuristic;
UNCLASSIFIED is explicit when no keyword applies. Relevance is explicit entity
overlap, not a learned relevance probability.

The default sentiment method is labelled LEXICON with positive/negative word
matches and offsets. It ignores negation, sarcasm, and financial context. A zero
lexicon score means matched lexical balance; it is not unknown model sentiment.
`sentiment_timeline(analysis, symbol=None)` aggregates unique current revisions by
UTC knowledge date, retaining source evidence hashes. This timeline describes
the given snapshot; recompute each historical cutoff for a historical-as-known
timeline across corrections.

`NewsStore(path).ingest(articles)` stores immutable evidence transactionally in
SQLite with idempotent replay and conflicting-revision rejection. `snapshot(as_of,
aliases=...)` selects the latest knowable revision per article identity, verifies
the digest and identity/knowledge metadata, and analyzes it. Restart, revision
selection, rollback, and corruption detection have regression coverage.

## Explicit local FinBERT

`LocalFinBERT(directory)` loads only a caller-provided flat checkpoint directory.
It requires `finbert-provenance.json` declaring model ID `ProsusAI/finbert`, a
revision string, and SHA-256 digests for every local model/tokenizer file. The
manifest must include `config.json` and `model.safetensors`. Symlinks, directories,
unlisted files, mismatched hashes, a manifest over 128 KiB, more than 256 files,
and more than 2 GiB of checkpoint files are rejected. This identifies declared
local bytes; the publisher's identity/revision remains caller-declared. Keep the
checkpoint directory quiescent while loading.

Torch/Transformers are optional. Loading sets `local_files_only=True`,
`trust_remote_code=False`, and `use_safetensors=True`; the model runs on CPU in
evaluation/inference mode. Labels must be positive/negative/neutral, IDs 0/1/2,
on a BERT model. Score is P(positive) minus P(negative). The model's 512-token
truncation is disclosed with original token count; original article evidence
remains untouched. Failure never substitutes lexicon output.

`finbert_capability(directory=None)` reports dependency/file inventory and never
claims a verified load. The legacy `transformer_sentiment(text,
local_directory=None)` returns None without a selected checkpoint and performs
no imports/downloads; a supplied local model follows the same strict adapter.
Real FinBERT inference was not run here because dependencies and a verified
checkpoint are absent. Mock loader tests verify CPU/local-only flags, label
mapping, exact input forwarding, and truncation disclosure; they do not verify
publisher weights or model quality.

## Verification

`PYTHONPATH=src python -m unittest discover -s tests -p test_news_workflow.py -v`
covers source text, hashes, knowledge-time exclusion, identity/revision replay,
duplicate attribution, lexical/dictionary explanations, store recovery/corruption,
local-model admission, and mocked local-only loading. No external provider,
broker connection, or economically validated sentiment strategy is claimed.
