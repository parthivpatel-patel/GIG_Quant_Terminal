# Stage A: repairs for the nine visual review defects

Captured 2026-11-04 against the same locally launched terminal as
[`../review-2026-11-04/`](../review-2026-11-04/README.md), which is kept
unchanged as the before-state evidence:

```bash
GIG_BROKER_CONNECTIVITY_ENABLED=false GIG_DATA_DIR=$HOME/gig-data python -m gig serve   # http://127.0.0.1:8000
python scripts/browser_review_capture.py --evidence-dir docs/evidence/stage-a-2026-11-04
python scripts/browser_stage_a_regressions.py --url http://127.0.0.1:8000 --report docs/evidence/stage-a-2026-11-04/regressions.json
```

Chromium 156.0.8078.4. Every non-local browser request is aborted and counted
(none were attempted). No broker connection, no orders, and no provider request
was made for this capture: Global Intelligence shows the USGS (339) and NASA
EONET (21 within the 30-day window) records already acquired by the earlier
review, labelled CACHED/DELAYED.

- `report.json`: review capture, 116 checks passed, 0 failed (it previously
  failed the Global Intelligence truth strip check).
- `regressions.json`: 37 defect regression checks passed, 0 failed.
- `screenshots/`: the same 19 views as the review, from the running application.

## Defects and their regression checks

| # | Defect | Repair | Checked by |
|---|--------|--------|------------|
| 1 | Global Intelligence kept the previous section's truth strip | The section writes its own provenance (loading, available, unavailable, error), and the next section restores its own | Source label starts with `global-intelligence`, no LIVE, restored after leaving |
| 2 | Risk legend showed a symmetric span | Legend shows the actual model P&L range (−12,317.29 to 73,346.67) with a zero tick | Legend values equal the payload extremes; zero tick present |
| 3 | Legacy charts overlapped origin tick labels | Collision-avoiding placement with a halo; minimum ticks sit along each axis | No overlapping label boxes in 3D and 2D on Command Center and Risk; all titles and end ticks placed |
| 4 | Vertical-axis minimum tick dropped | Vertical ticks are placed first; floor ticks nudge or drop instead | Minimum vertical tick present on the volatility surface and order book |
| 5 | SYN002 hidden behind SYN004 | Overlapping markers separated with a leader line to the true position; labels avoid other labelled markers | No marker overlap (also after rotation); no label covers another marker |
| 6 | Globe did not turn to the selected event | Selection eases the globe to centre the event (immediate under reduced motion); flat map pans if the event is off-screen | Projected event at canvas centre; reduced-motion jump; edge event brought into view |
| 7 | Flat map had empty bands and cropped the Pacific | Map fills the canvas height, wraps horizontally, clamps pan and zoom, and centres by default on the longitude showing the most events | No band at default, after panning or zooming out; both edges drawn; Pacific contiguous at the antimeridian |
| 8 | Machine-cased metric labels and a lone fifth card | Known keys have readable labels (VaR, ES, NAV); acronym-aware casing elsewhere; grid columns follow the card count | Labels checked on Risk and Models; five cards on one row at 1920 px; no lone half-width card at 390 px |
| 9 | Mobile point-in-time placeholder truncated | Shorter placeholder ("Now, or ISO time with zone"); help text keeps the timezone rule | Placeholder width fits the input at 390 px |

The flat map's visible longitude span depends on the canvas aspect ratio. At
the desktop size about 290° are visible at once, so the default view centres
on the Pacific, where most acquired events are. The rest of the world is one
drag away, because the map wraps.
