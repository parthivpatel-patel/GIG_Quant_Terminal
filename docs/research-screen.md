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

`operation: "from-runs"` builds the ensemble and the screen in one request from
saved model `run_ids` and `now`. The resulting artifact still pins the
`signal_id` for offline allocation.

Replay (`screen-replay`) recomputes the labels from the pinned signal and
refuses the record if they differ. Offline allocation accepts `screen_id` and
sizes long/short legs from nonzero `research_weight` values. The underlying
`signal_id` path remains available.
`operation: "monitor"` chains two to thirty-two saved `signal_ids` into successive screens and summarizes timing. It is not a live market subscription.

`operation: "link-options"` joins a saved screen to a saved European options
research run. When the options `underlying_id` matches a screened instrument,
`RESEARCH_BUY` becomes a `RESEARCH_CALL_CANDIDATE` and `RESEARCH_SELL` becomes
a `RESEARCH_PUT_CANDIDATE`. Holds and unmatched underlyings stay explicit.
The link is not an options order and does not approve trading.

Broker connectivity stays disabled.
