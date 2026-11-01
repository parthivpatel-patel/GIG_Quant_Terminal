# Verified local FinBERT research

This cycle executes the actual official `ProsusAI/finbert` BERT classifier on CPU,
without a substitute sentiment model. The selected checkpoint revision is
`0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031`. Its official safetensors file is
437,965,908 bytes; SHA-256:
`e5897858ff819aad7629b96ce521ae5477952d03634ef5dd30ed2d76357a9f00`.
The complete model/tokenizer manifest SHA-256 is
`2bb447dda8eb845e6454b35722dd3ccd3ac9098c79d24a1ab0f68e64d764f457`.
The safetensors and tokenizer files remain outside the Git repository. No
restricted weights are distributed by this implementation.

## License and scope

The model card declares **CC-BY-NC-SA-4.0**. Attribution is retained to ProsusAI
and Dogu Araci, *FinBERT: Financial Sentiment Analysis with Pre-trained Language
Models* (2019), with the pinned upstream URLs and license URL. Execution here is
**noncommercial research validation only**. Every model/score record carries
`commercial_use_permitted: false` and `trading_approved: false`.

Code execution and successful test results do not authorize commercial or
production use. The article's caller-supplied license is preserved separately
from the model license. Historical model availability and the training-data
cutoff are unverified; these outputs are retrospective annotations, not proof
that a model was usable at a historical trading decision. Envelope freshness
refers to original article knowledge time. The stored `analysis_as_of` records
the research data cutoff, not model publication or a live inference feed.

Two possible finance sentiment alternatives were inspected through public
model metadata: `yiyanghkust/finbert-tone` at revision
`4921590d3c0c3832c0efea24c8381ce0bda7844b`, and
`ahmedrachid/FinancialBERT-Sentiment-Analysis` at revision
`656931965473ec085d195680bd62687b140c038f`. Neither response supplied a model-card
license. They were not downloaded or executed, and no commercial permission is
inferred. Evidence is retained in `finbert-commercial-alternatives.json`.

## Loading and execution boundary

`prepare_finbert_assets` is the only explicit asset-download utility. It requires
a noncommercial-research acknowledgement, pins an immutable repository revision,
checks the upstream card license, and verifies every file against the upstream
Git-blob or LFS hash before atomically publishing it. Existing assets are
verified rather than overwritten. It accepts no arbitrary model ID or URL.

`LocalFinBERT(directory)` performs **local-only** loading, requires a flat
nonsymlink safetensors/config/tokenizer directory and an exact hash manifest,
sets `trust_remote_code=False` and `use_safetensors=True`, validates the actual
three sentiment labels, and uses CPU evaluation mode. It never downloads weights
or falls back to lexical sentiment. A manifest hash establishes content identity;
arbitrary caller-created manifests alone do not authenticate upstream authorship.

Inference uses a shared process lock with the sequence-model runtime. It restores
Torch RNG, thread count, deterministic/warn-only flags, and MKLDNN state after
loading and scoring. Model execution uses one CPU thread. Source text remains
verbatim outside the tokenizer. Inputs beyond 512 model tokens explicitly report
truncation, the original token count, and the model token bound.

## Durable terminal integration

`SentimentWorkspace(base_dir, model_dir)` accepts a fixed application-selected
asset directory. HTTP payloads cannot choose paths, download models, supply
broker credentials, or grant trading approval. `analyze` accepts exactly
`articles`, `as_of`, and optional alias mappings. It preflights the complete typed
response before writing original news evidence and an immutable score artifact.
The original cutoff is inside the hashed artifact and checked against the
index, preventing an index edit from backdating a future annotation. Reopening does not require loading the model to read a
previous verified result. Missing/stale/unknown policies remain explicit.

The parent terminal selects this stored research through `engine=finbert`; it
does not relabel ordinary unscored news as FinBERT output or hide it in the
default news view.

## Observed evidence

Actual Torch CPU 2.15.0+cpu and Transformers 4.46.3 loaded the official checkpoint.
Positive, negative, and neutral authored fixtures produced the corresponding
maximum-probability labels. Direct Transformers forward probabilities matched
the wrapper to absolute tolerance `1e-8`. The three illustrative one-thread
measurements were about 59–67 ms per article on this executor; these are measured
examples, not latency guarantees, ARM64 evidence, or economic validation.

The focused run passed **36 tests**, zero failures or skips: 11 new asset/actual
model/workspace tests and 25 existing news tests. It covers replay identity,
source text/truncation, original licensing, SHA tampering, ledger backdating,
CPU state restoration, filesystem guards, and inference with network functions
blocked. In environments without the separately provisioned licensed assets,
the actual-model class skips explicitly; it never fabricates an inference pass.

```sh
PYTHONPATH=/tmp/gig-ml-deps:/workspace/work/dependencies:/workspace/work/tooling:src \
HF_HOME=/tmp/gig-finbert-cache HF_HUB_DISABLE_TELEMETRY=1 \
python -m pytest tests/test_finbert_actual.py tests/test_news_workflow.py
```

Ruff, focused mypy, compilation, and diff checks also pass. Tests and metadata do
not certify the entire product, model economic value, or T4 readiness.
