"""
A/B the two book-construction rules on the same data, scores, and limits.

The only thing that varies between the two runs is how alpha becomes weights,
so the difference in volatility and in systematic risk share is attributable to
construction and nothing else. Run from the repo root:

    python scripts/ab_construction.py --limit 300
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from gig.backtest.simulator import CrossSectionalBacktest
from gig.config import get_settings
from gig.data.lake import load_liquid_panel
from gig.portfolio.optimize import OptimizerConfig
from gig.strategies.equity_ls import EquityLongShort


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=300, help="Universe size by ADV rank")
    parser.add_argument("--lookback-days", type=int, default=1500)
    args = parser.parse_args()

    settings = get_settings()
    panel, _store = load_liquid_panel(limit=args.limit, lookback_days=args.lookback_days)
    if panel is None or panel.close.empty:
        print("No bars in DuckDB. Run: python -m gig universe refresh && python -m gig ingest")
        return 1

    n_names = len(panel.symbols())
    n_side = max(5, min(50, n_names // 6))
    strategy = EquityLongShort(
        n_long=n_side, n_short=n_side, min_adv_usd=1_000_000.0, use_ml=False, use_news=False
    )
    print(f"universe {n_names} names, {len(panel.close)} sessions, {n_side} per side")

    # Scores are computed once and reused, so the comparison cannot be
    # contaminated by a different signal on either side.
    combo, tradable, _ic = strategy.scores(panel)

    rows = []
    for label, optimizer in (
        ("quantile_equal_weight", None),
        (
            "risk_constrained",
            OptimizerConfig(
                target_vol=settings.target_vol,
                max_gross=settings.max_gross_leverage,
                max_name=settings.max_name_weight,
            ),
        ),
    ):
        engine = CrossSectionalBacktest(
            n_long=n_side,
            n_short=n_side,
            rebalance_every=strategy.rebalance_every,
            cost_model=strategy.cost_model,
            optimizer=optimizer,
            risk_lookback=settings.risk_lookback,
            risk_refit_every=settings.risk_refit_every,
            risk_factors=settings.risk_factors,
        )
        result = engine.run(
            combo,
            panel.close,
            tradable=tradable,
            dollar_volume=panel.dollar_volume,
            sectors=panel.sectors,
        )
        m = result.metrics
        row = {
            "construction": label,
            "sharpe": m.get("sharpe"),
            "ann_return": m.get("ann_return"),
            "ann_vol": m.get("ann_vol"),
            "max_drawdown": m.get("max_drawdown"),
            "ann_turnover": m.get("ann_turnover"),
            "ex_ante_vol": m.get("ex_ante_vol_mean"),
            "systematic_share": m.get("systematic_share_mean"),
        }
        if result.diagnostics is not None and not result.diagnostics.empty:
            row["effective_names"] = float(result.diagnostics["effective_names"].mean())
            row["gross"] = float(result.diagnostics["gross"].mean())
        else:
            held = result.weights.iloc[-1].abs()
            row["effective_names"] = (
                float(1.0 / ((held / held.sum()) ** 2).sum()) if held.sum() else 0.0
            )
            row["gross"] = float(result.weights.abs().sum(axis=1).mean())
        rows.append(row)

    frame = pd.DataFrame(rows).set_index("construction")
    print("\n" + frame.T.to_string(float_format=lambda v: f"{v:,.4f}"))
    print("\n" + json.dumps(rows, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
