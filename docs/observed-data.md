# Acquired ECB reference observations

`POST /api/v1/workspace/observed-reference` supports `fetch`, `admit`,
`research` and `replay`. The `observed-reference` engine connects saved evidence
to Markets, Models, Signals, Research and Operations. It never supplies an
approved trading signal or funded equity instrument.

The fixed, credential-free provider is the European Central Bank's
[historical reference-rate ZIP](https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip).
The [official disclaimer](https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html)
permits accurate reproduction with source citation, disclosure of modifications
and a free-origin notice if distributed for a fee. GIG records that attribution.
The [ECB reference-rate guidance](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html)
describes information-purpose rates and discourages transaction use. No vendor
entitlement, executable price or brokerage connection is implied.

An actual HTTPS acquisition on November 3, 2026 returned HTTP 200, was parsed and
persisted. The 2025 USD/GBP/JPY window contains 255 source dates and 765 reference
observations. The saved verification identifies the exact raw and parsed hashes,
actual acquisition timestamp and source URL. Symbols have the form
`ECB:EXR:USD:PER_EUR`: USD per EUR, not a stock or a dollar-funded position.

```json
{"operation":"fetch","provider":"ECB_REFERENCE_FX","currencies":["USD","GBP","JPY"],"start":"2025-01-01","end":"2025-12-31"}
```

Only this fixed official HTTPS source is fetched; redirects remain on the
approved host. Archive size, archive membership, CSV rows, currencies and window
length are bounded. Arbitrary URLs and credentials are rejected. Manual CSV
admission is always SYNTHETIC, including a claimed provider name. Missing quotes
remain missing and reject an affected model dataset. No OHLC or volume is
invented. Source dates are known; intraday observation/publication time and
historical revision knowledge are UNKNOWN. Actual acquisition is never backdated.

Research accepts the snapshot identity and one of `ridge`, `lightgbm`, `bilstm`,
`hmm`, `garch` or `kalman`, plus bounded training/test/embargo/seed settings.
Features and labels are chronological with purging and causal held-out
conditioning. Actual CPU library-backed fits, saved model identities and input
hashes are retained. Replay checks original source/model/report bytes and the
publication index without training again. Later acquisitions cannot alter the
saved report. The research mode is explicitly
RETROSPECTIVE_OBSERVATION_ORDER_RESEARCH; this is not evidence that those inputs
or models were published at a historical decision time.

All six actual CPU programs executed and replayed on the acquired window. One
HMM fold did not converge: its 20 predictions remain unavailable. LightGBM emits
rank scores; GARCH emits conditional variance, so their return MSE is null.
BiLSTM held-out MSE was approximately 0.0382, a poor result, not evidence of alpha.
Other estimates likewise establish execution and reproducibility, not economic
usefulness. All runs remain research only and ineligible for funded allocation.

`scripts/check_observed_reference.py` performs the explicitly requested public
acquisition and six-model check. Ordinary regression and browser tests use
authored fixtures and make no provider requests. Exact actual-source/model proof
is under `docs/evidence/sprint7/observed-verification.json`. Raw source and saved
model artifacts remain in the local research state directory, outside Git.

Genuine historical equity coverage, authoritative historical publication clocks,
observed trading capacity and independently validated economic performance remain
unavailable. Alpaca remains fixture-only. FinBERT is a separate noncommercial
research workflow; no restricted weights are included in source or recovery.

Saved reports can now be compared with training-only baselines; see
[reference-model evaluation](reference-model-evaluation.md) for the actual negative
findings, uncertainty assumptions and immutable HTTP/terminal workflow.
