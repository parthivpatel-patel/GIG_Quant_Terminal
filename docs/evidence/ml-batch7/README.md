# Family 14 additive cubic splines: frozen-source acceptance

Functional source `2965249d2a2f4ab66548d6e41d19aadc5a30c474` (PR #25 head) was
accepted unchanged. Later commits add only this evidence and documentation;
`validated-source-manifest.json` records all 407 functional source hashes and
Git modes at that commit.

- Full backend at the frozen source: **1,343 passed, 22 skipped, 0 failures**,
  518.41 s, **83.68% combined line/branch coverage**. Skips: optional pinned
  FinBERT assets/runtime, the optional standalone C++ library, and ECB
  acquisition evidence that is unavailable in this environment. The 22 skips
  are listed in `regression.txt`.
- Installed wheel `gig-0.1.0-cp312-cp312-linux_x86_64` built from the frozen
  source in an isolated target: **68 passed, 1 skipped** (additive, GLM,
  Bayesian, KNN, persisted data/flow, native operations). All **240** tracked
  package files byte-match. A separate installed process replays the saved
  study with spline and ridge fitting patched to fail; response SHA-256
  `44e03603255ac232ae6a76f3616883d1bcaa5d5f05a3282bf9e054cc50c405dd`.
- Ruff, Pyflakes, compilation and the three configured mypy invocations pass.
  180 Node smoke checks pass; Node is not browser QA.
- Chromium **159.0.8154.6** with pinned axe-core 4.10.3: **22/22** checks,
  0 JavaScript errors, 0 external requests, 0 orders. Dark and light
  screenshots are retained. No formal accessibility or cross-browser claim.
- GitHub CI on the same head passed 6/6 jobs (runs 44936833953, 44936839422).

The authored **SYNTHETIC** study fits 6 models and evaluates 72 out-of-sample
rows: MSE `7.437458544945842e-05` against the training-mean baseline
`7.25311755680361e-05`. The model underperforms the baseline overall and in
4 of 6 folds; every fold is retained in `research-result.json`. These are
engineering fixtures, not observed-market or economic validation.

Not rerun here: the x86 C++ rolling-kernel parity benchmark (no native source
changed since batch 6) and container/ARM64 acceptance. Uncertainty stays
UNAVAILABLE; allocation and promotion stay rejected. No new library, weights,
provider, paid service, broker/order or deployment. FinBERT noncommercial
research-only; fully_funded_certified=false; T4_NOT_CERTIFIED; §139 UNSUPPLIED.
