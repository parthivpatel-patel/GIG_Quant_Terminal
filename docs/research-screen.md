# Research buy / hold / sell screen

`POST /api/v1/workspace/research-screen` labels instruments from a saved
IC-weighted research signal. The labels are `RESEARCH_BUY`, `RESEARCH_HOLD`
and `RESEARCH_SELL`. They are threshold cuts on an uncalibrated centered-rank
score, not orders, and never set `trading_approved`.

Caller fields:

- `signal_id` (required): a saved ensemble from `research-signals`
- `buy_threshold` / `sell_threshold` (optional, default `0.25` / `-0.25`)
- `previous_screen_id` (optional): an earlier screen on the same dataset whose
  decision time is strictly earlier; used only for entry / exit research events

`research_weight` is the relative share of absolute score among actionable
names, signed by stance. It is not a lot size. Timing events are
`ENTER_LONG`, `EXIT_LONG`, `ENTER_SHORT`, `EXIT_SHORT`, `CONTINUE`, `FLIP`
and `ENTER_HOLD`.

Replay (`screen-replay`) recomputes the labels from the pinned signal and
refuses the record if they differ. Offline allocation still consumes the
`signal_id`; the screen sits beside that path as an explicit research view.
Broker connectivity stays disabled.
