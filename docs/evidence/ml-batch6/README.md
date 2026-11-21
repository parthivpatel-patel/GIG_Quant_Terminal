# Family13 binomial GLM: frozen-source acceptance

Source `5322a17c796487e647d0b42ffdf618d1b56439ea` preserves the original
`36dd65584b457e49aaf52604a69b6a95c8d8030a` implementation, reconciles PR22
development merge096fb144 and repairs the reproduced Chromium159 mobile defect.
Later evidence commits preserve all398functional source hashes and Git modes.

- Full backend: **1,340 passed, 9 skipped, 0 failures**,590.79s;
  **83.65% combined coverage**. Nine skips are optional unavailable pinned
  FinBERT assets/runtime. Focused suites overlap this total.
- Independent current native wheel: **119 passed, 1 skipped**. The optional
  standalone C-ABI library is separate from the packaged pybind extension,
  which imports successfully. All237tracked packaged source/data/UI files
  byte-match. Separate installed-process replay performs no fitting and returns
  response SHA256 `58a8079eae34e935ee4c90bf6d0b76f8edb2909f0eb3d2d2bb89e164e2331189`.
- Ruff,Pyflakes,compilation,configured22-module mypy plus15current numerical/
  document modules pass.173Node smoke checks pass; Node is not browser QA.
- Actual **Chromium159.0.8154.6**, matching the failed CI version: **24/24**
  passes,0JSerrors/external requests. Original CI and local failure reports and
  mobile screenshots are retained. Document widths were461px at320/375px;
  after the fix they are320/375px exactly.768/1440px also pass. Long categorical
  states now wrap; numeric figures retain their original nonbreaking rule.
  Strict overflow assertions and all existing browser suites remain unchanged.
  Current dark/light/mobile screenshots and axe details are retained. Contrast
  incompletes require manual review; no formal accessibility/cross-browser claim.
- Existing x86 C++20 rolling parity is measured on200,000samples/11repeats.
  No current-container or physicalARM64/Ampere acceptance is inferred.

The actual authored **SYNTHETIC** study fits6models and evaluates72OOSrows.
Brier `0.2637410078515328` underperforms train-rate baseline
`0.25163116000858504`; log loss `0.7245326645205504` underperforms baseline
`0.6964443134268321`.3of6folds underperform on each score; all folds and fixed
calibration bins are retained. Return-MSE is null/UNAVAILABLE_DIFFERENT_OUTPUT_UNITS.
These are engineering fixtures, not observed-market or economic validation.

Actual sklearn regularized binomial-logit fitting, an independent SciPy likelihood
solution, safe JSON replay and strict chronological label-vintage purging are
verified. The fixed label is admitted forward return strictly positive; original
return targets remain traceable. Bounds:20..256rows,16features,16total fits,
512iterations,one CPU thread and2,000inference rows. Exact initial optima may
use zero iterations, but the likelihood-gradient check still applies. Missing
classes, nonconvergence, malformed states and rehashed coefficient attacks reject.

Probabilities remain uncalibrated, parameter uncertainty UNAVAILABLE, and fresh/
saved estimates cannot allocate or promote, even if wrapper fit-state labels
change. Conditional Bernoulli outcome variance is not parameter confidence.
Other GLM links/likelihoods are unavailable. Inventory:13implemented families,
27planned. No new library, weights, provider, paid service, broker/order or
deployment. FinBERT noncommercial research-only;fully_funded_certified=false;
T4_NOT_CERTIFIED;§139UNSUPPLIED. Docker capacity blocked;physicalARM64 unverified.

An interrupted temporary filesystem was recycled. Published source and original
failure diagnostics were recovered without modifying protected workspaces. Full
acceptance logs were regenerated on this frozen source. Exact-head CI/review
and independent downloaded-archive restoration are recorded in the release PR.
GitHub is the only verified external source recovery provider;/tmp is ephemeral.
Runtime databases and restricted model weights are not added to source.
