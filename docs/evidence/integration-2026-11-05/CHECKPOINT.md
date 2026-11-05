# Integration checkpoint

Verified on 4 November 2026 (America/New_York), before the strategy-experiment commits on this branch.

`development/gig-ui-redesign` was fast-forwarded from `beb959f29e6d3540ed3f048dbe6475333a61e04b` to `ef7294343d9e01e9dd8a090721272c658d5bf081`. The history is linear. `main` stayed at `f946b23790e92b048a8cbd0160f876509b43455e`.

| Pull request | Head | Result |
| --- | --- | --- |
| #5 | `65943c9` | Merged into `development/gig-ui-redesign` at 9:31 PM EST |
| #6 | `747ef49` | Fast-forwarded onto development. GitHub would not retarget the stacked base after the commits were already there, so the pull request was closed with that explanation. |
| #7 | `94f07e4` | Same as #6 |
| #8 | `ef72943` | Same as #6 |

Branches were not deleted. Nothing was force-pushed.

## Checks run on `ef72943`

- `python -m pytest -p no:cacheprovider --junitxml`: 1624 collected, 0 failures, 0 errors, 11 skipped. 143.9 seconds.
- `ruff check src tests scripts/build_native.py`: passed.
- `node scripts/check_terminal.mjs`: 149 checks passed.
- `scripts/browser_stage_a_regressions.py`: 37 passed, 0 failed. The nine November 4 visual defects stayed fixed.
- `scripts/browser_charts.py`: 33 passed, 0 failed.
- `scripts/browser_observatory.py`: 7 passed, 0 failed. That script later grew with the two-cycle check; this count is the pre-change script.

Prior screenshots under `docs/evidence/review-2026-11-04/`, `docs/evidence/stage-a-2026-11-04/`, and `docs/evidence/charts-2026-11-04/` were not replaced.
