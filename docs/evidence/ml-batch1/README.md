# ML batch 1 acceptance evidence

PR #13 was merged first at `53405c066d5ef63573c0a570ad781d7f66cdbae2`;
all post-merge checks passed in run `43433467105`. The merged source archive was
independently restored and all 990 file hashes/modes verified before ML work.
Main and protected quantitative refs were not modified.

The new branch is `development/gig-ml-expansion-batch1`, PR #14 against
`development/gig-ui-redesign`. Inventory commit `a09b91e9374db00ad354ec11227aba9b9aacb476`
precedes the actual GP implementation at `07779009aea634dbe7d4e344baf640633ddc7df2`.
Commit `0c9f7c0da67cb1851409b06007e48cd327ef5d94` corrects three transitive typing
failures found with current dependencies: guard an optional DuckDB count row,
normalize LightGBM prediction output to the declared numeric ndarray, and admit
provenance through Pydantic's mapping validator. No check, type error suppression,
worker lease, stale-owner fence, risk gate or broker boundary was weakened.

The `validated-source-manifest.json` freezes source/tests/browser scripts/CI
hashes and modes. Later evidence/documentation commits leave those source bytes
unchanged. Exact final CI commit/check URLs are recorded in PR #14.

- Final frozen-source regression: **1,247 passed,9 skipped,0 failures**,
  482.02 seconds, **83.1% combined coverage**. All nine skips are
  absent optional pinned FinBERT assets/runtime. The earlier repeated full run
  is not added to this count. Full logs/XML and coverage summary are preserved.
- Installed native wheel: **172 passed,1 skipped**, independent target install,
  217 packaged source/data/UI files byte-compared. The skip is the optional
  standalone C-ABI library, intentionally not installed in the wheel. Native
  pybind imports from the installed wheel, not the source checkout.
- Current configured mypy: **22 modules passed**, plus the new typed GP kernel.
  Ruff, Pyflakes, compileall and **150 Node DOM checks** pass. DOM checks are not
  substitutes for browser rendering or accessibility certification.
- Genuine Chromium **151.0.7922.173**: **20 passed,0 failed**, including actual
  admitted synthetic data, fit, inference, explicit saved-identity replay after
  page reload, allocation rejection and Models rendering. Both dark/light axe
  scans and 320/375/768/1440px document-overflow checks pass. Zero JS errors or
  external browser requests. Screenshots are actual application captures;
  `models-gp-dark.png` / `models-gp-light.png` show the saved-model view. The older
  19-check captures remain preserved under their original filenames.
- Native rolling parity: 200,000 samples,11 repeats, max absolute error
  `6.938893903907228e-18`. x86_64 only; shared-host latency is not an SLA.
- Separate processes: source process fitted/persisted, exited; independent
  installed-wheel process reproduced prediction SHA-256
  `adb9c290244b6567b5fe7672734aedad09eb5363a206d50643be6d0acf2e1607`,
  with training_performed=false and zero order state.

`research-result.json` is SYNTHETIC functional evidence, not observed market
validation. Six states,72 OOS rows, measured1.1454s fit/publish. GP MSE
`9.315493972534936e-5` is worse than the training-only mean baseline
`7.25311755680361e-5`; descriptive nominal95% coverage is86.1111%. Negative and
uncalibrated findings are retained. No economic claim or trading approval.

This batch uses existing BSD-licensed NumPy/SciPy/scikit-learn CPU dependencies,
not downloaded pretrained weights. FinBERT remains noncommercial research-only.
The remaining32 inventory additions are planned. No observed source permission,
PIT cash-equity universe, corporate-action vintage or options-print entitlement
is invented. Docker remains BLOCKED_STORAGE; no Docker command was run after
the read-only gate rejected capacity. Physical ARM64/Ampere, cross-browser and
full manual accessibility acceptance remain unverified.

`fully_funded_certified=false`; `T4_NOT_CERTIFIED`; historical §139 UNSUPPLIED.
No brokerage connections/orders, paid services, deployments or automatic model
promotion occurred. Only GitHub is verified external recovery; local /tmp files
are session evidence, not an independent external backup.
