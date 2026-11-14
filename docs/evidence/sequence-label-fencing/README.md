# BiLSTM label-knowledge fencing

Reproduced against development commit33378c9ef55c87cf51bf4896524ba679325a1dbf:
when every training label revision was first knowable at the dataset research
cutoff, the sequence adapter nevertheless fitted6models and returned72
out-of-sample predictions. Its raw-target fallback bypassed revision knowledge.
The controlled dataset is a labeled SYNTHETIC fixture, not observed-market evidence.

The repair rejects unavailable supervised labels before actual fitting and before
any research persistence. Feature-only warm-up observations and label-end windows
already purged by the canonical sequence kernel retain their existing behavior.
Known valid runs preserve their fitted model identities and predictions.

Loading an older immutable BiLSTM run rechecks its original admitted dataset,
training prefix, embargo exclusions and label knowledge. Unsafe legacy runs fail
closed before inference, cached section publication or allocation. Their files
are preserved, not rewritten. Runtime budgets, lease fencing, broker restrictions
and model-promotion controls are unchanged.

Focused acceptance:53passed,0failed,0skipped (new regression tests plus persisted
models, numerical sequence, forests and financial retrieval). Tests cover rejection
before the actual fitter/persistence, irrelevant warm-up target changes, legacy
immutable artifact preservation, safe HTTP422 and unchanged prior revision tests.
Final local regression: **1,270passed,9skipped,0failed**,475.51s; combined
branch/line coverage **83.13%**. All9skips are unavailable optional pinned FinBERT
assets/runtime. The complete JUnit and coverage log are retained; focused suites
overlap this regression and are not added to its total.

Independent installed current wheel: **84passed,1skipped,0failed**;224packaged
files match source; Python/native imports resolve inside the isolated installation.
The explicit skip is the separate optional local C-ABI rolling library, absent
from this wheel; the packaged pybind extension loads and numerical parity passes.
An initial command referenced a nonexistent test module and collected no tests;
the corrected invocation produced these84passes.

Ruff,Pyflakes,compilation, configured mypy22modules plus7numeric/document modules
and157Node checks pass. Native rolling comparison:200,000samples/11repeats on
x86_64, maximum absolute error6.938893903907228e-18. No physicalARM64 claim.

Current Chromium151.0.7922.173: financial retrieval24/24. The first forest run
was24/25 because its readiness check could read the previous section before
hashchange. Its screenshot then showed correct source and SYNTHETIC INPUTS labels.
The test now waits for the actual Models response and active section; every
assertion is retained. Two subsequent forest runs pass25/25 each. The original
failure report is retained. Dark/light axe finds0violations, with incomplete
contrast items requiring manual review; responsive widths320/375/768/1440 pass.
No JS errors or external browser requests. This is not formal accessibility or
cross-browser certification. Genuine screenshots are retained.

Python source is frozen at592b2b0531700efea8ae2e159c8a8c99686b3fa0; the only
subsequent executable change is this independently rerun browser wait. The
source manifest records every validated source/test/config hash and Git mode.
Final-head CI and independent recovery results are recorded in the PR.

This is a research correctness repair, not profitability or certification evidence.
fully_funded_certified=false; T4_NOT_CERTIFIED; historical§139UNSUPPLIED.
