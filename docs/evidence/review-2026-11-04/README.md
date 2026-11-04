# Visual review of merged `development/gig-ui-redesign` (`beb959f`)

Captured 2026-11-04 by `scripts/browser_review_capture.py` against the terminal
launched as documented in the README:

```bash
pip install -e ".[web]"
GIG_BROKER_CONNECTIVITY_ENABLED=false GIG_DATA_DIR=$HOME/gig-data python -m gig serve   # http://127.0.0.1:8000
```

Chromium 156.0.8078.4, 100% zoom. Section pages are 1920 px wide at device
scale 1; module and Global Intelligence panels at device scale 2; mobile at
390×844, device scale 2. Every non-local browser request was aborted and
counted (none were attempted). No broker connection; no orders.

Sections 1–12 use the terminal's own "Synthetic research demo" data
environment (labelled SYNTHETIC). With the default "Admitted local data"
environment they show UNAVAILABLE because no data has been admitted.
Global Intelligence shows one explicit USGS request (339 events) and one NASA
EONET request (44 events), both labelled DELAYED (`provider-refresh.json`).

Results: `report.json`, 115 checks passed, 1 failed (truth strip below).

## Defects found (not fixed; awaiting review)

1. Global Intelligence does not update the global truth strip; it keeps the
   previous section's provenance (for example Operations' `local-process-diagnostics`).
2. Risk terrain legend shows a symmetric colour scale (−73,346.67 to
   73,346.67) while the data range is −12,317.29 to 73,346.67.
3. Legacy factor-space and option-repricing 3D charts (Command Center,
   Portfolio, Risk) overlap tick labels at the origin; they do not use the
   newer label layout.
4. Collision avoidance drops the vertical-axis minimum tick in the volatility
   surface and order book.
5. Correlation network: SYN002 is mostly hidden behind SYN004 at the default
   camera; its label is shown but the point is not distinct.
6. Selecting an event does not turn the globe to it; the Alaska event in
   `04` is selected but on the far side.
7. Flat map leaves empty bands above and below and crops the Pacific edges.
8. Metric-card labels are machine-cased ("Historical Es 95 Currency",
   "Synthetic Nav"); a fifth metric card sits alone on a second row.
9. Mobile: the point-in-time cutoff placeholder is truncated.
