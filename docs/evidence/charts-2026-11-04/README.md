# Market Charts workspace, 2026-11-04

Captured by `scripts/browser_charts.py` against an isolated local server
started from this checkout. Chromium 156.0.8078.4. Every non-local browser
request was aborted and counted (none were attempted). No broker connection
and no orders.

```bash
python scripts/browser_charts.py --evidence-dir docs/evidence/charts-2026-11-04 --axe /tmp/axe/package/axe.min.js
```

`charts-report.json`: 35 checks passed, 0 failed, including axe-core WCAG A/AA
on the charts workspace with the command palette open, in both themes.

The synthetic views use the deterministic fixture `deterministic-synthetic-ohlcv-v1`
(labelled SYNTHETIC; fictional instruments). The admitted view posts an
explicitly labelled SYNTHETIC fixture through `/api/v1/workspace/bars` and then
charts it as known at a cutoff. With no admitted bars the workspace stays
UNAVAILABLE and draws nothing.

| Screenshot | What it shows |
|---|---|
| `20-charts-candles-synthetic.png` | Candles, volume, SMA 20/50, RSI, watchlist, crosshair readout |
| `21-charts-compare-macd.png` | Bollinger, VWAP, MACD and a synchronized comparison pane |
| `22-charts-local-unavailable.png` | Local mode before any bars are admitted |
| `23-charts-local-admitted.png` | Admitted bars, labelled with their source, as of the cutoff |
| `24-command-palette.png` | Ctrl/Cmd K search over sections, commands and instruments |
| `25-nav-collapsed-charts.png` | Icon navigation with the chart workspace still usable |
| `26-mobile-charts.png` | 390 px width: horizontal watchlist, chart, RSI |
