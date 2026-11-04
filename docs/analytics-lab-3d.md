# 3D Analytics Lab

Six interactive 3D modules shown in the sections they belong to. They share one
canvas engine and one data contract.

| Module | Sections | Data (`GET /api/v1/viz/<module>`) |
| --- | --- | --- |
| Volatility surface | Options | Authored BSM prices inverted numerically to implied vol; current and prior snapshots |
| Order book landscape | Markets, Execution | 36 authored depth frames × 15 levels, displayed quantity only; authored print counts |
| Portfolio risk terrain | Portfolio, Risk | Full BSM repricing of long, fully paid European options over spot × vol shocks, today and +30 days, with loss limit and premium concentration |
| Correlation network | Command Center, Signals, Research | Pearson correlations over three windows, classical MDS layout, average-linkage clusters, standard error at ρ = 0 |
| Stat-arb landscape | Statistical Arbitrage | Causal Kalman hedge elasticity, standardized innovations, filtered 2-state HMM regimes |
| Model evaluation | Models | Walk-forward GARCH QLIKE, Kalman MSE, HMM out-of-sample log-likelihood and LightGBM IC, each against a training-only baseline |

Code: `src/gig/service/viz.py` (builders), `src/gig/service/viz_synthetic.json`
(stored outputs), `static/viz3d.js` (engine), `static/models3d.js` (modules).
Tests: `tests/test_viz_modules.py`, `scripts/browser_global_3d.py`.

## Data honesty

- The default lab data is the authored synthetic fixture, labeled SYNTHETIC in
  the envelope and footer. "Admitted local evidence" returns UNAVAILABLE for
  every module and renders nothing; no local inputs are inferred.
- Outputs are deterministic. `scripts/build_viz_fixtures.py` stores them so a
  request does not rerun the HMM and GARCH fits (about 12 s cold); the test
  suite recomputes all six modules and compares them with the stored file.
- Model evaluation does not train the BiLSTM here and says so; metrics are not
  trading profitability or economic validation. Correlation is described as
  co-movement, not causation.

## Interaction and accessibility

- Mouse: drag rotates, shift-drag or right-drag pans, wheel zooms, hover shows a
  tooltip, click selects.
- Keyboard on the focused canvas: arrows rotate, shift+arrows pan, + / − zoom,
  0 resets, N / P step through items, Esc clears; selections are announced in a
  polite live region. Module tabs use arrow-key roving focus.
- Every module has filters, a legend with units, axis labels, a selection
  details panel, a method note, a 2D orthographic alternative and an exact-value
  data table.
- Orbit animation is off by default, can be paused, and is refused when the
  user prefers reduced motion.
- One engine instance and one canvas are reused across modules; resizing
  follows a ResizeObserver.

## Palette

Deep-black layer at the end of `static/styles.css`: background #050608,
panels #0C1016, elevated #141A22, text #F2F5F9 / #A6B2C3 / #8292A5, gain
#16C784, loss #F05259, warning #F2BE4C, accent #408BFF, borders #26303D.
