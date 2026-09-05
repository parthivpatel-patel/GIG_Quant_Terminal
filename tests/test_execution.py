"""
The trading layer, tested offline.

Nothing here reaches a broker or the network. A fake broker implements the
protocol and can be told to fail, so the paths that matter most -- a rejected
order, an unpriceable name, a blocked run -- are exercised rather than assumed.

The properties under test are the ones whose failure costs money: the gate
blocks when it should, `dry_run` sends nothing, an exit always trades, and no
order exceeds its participation or size cap.
"""

import math
from datetime import date, timedelta

import pandas as pd
import pytest

from gig.execution.algo import ExecutionConfig, limit_price, slice_shares
from gig.execution.base import Broker
from gig.execution.pretrade import PreTradeConfig, apply_drops
from gig.execution.pretrade import check as pretrade_check
from gig.execution.reconcile import (
    ReconcileConfig,
    build_trade_plan,
    current_weights,
    drift_report,
)
from gig.execution.targets import TargetBook
from gig.execution.trader import run_once
from gig.types import Fill

PRICES = pd.Series({"AAA": 100.0, "BBB": 50.0, "CCC": 25.0, "DDD": 10.0})
SECTORS = pd.Series({"AAA": "tech", "BBB": "tech", "CCC": "energy", "DDD": "energy"})


class FakeBroker(Broker):
    """Records what it was asked to do. Optionally rejects named symbols."""

    def __init__(self, nav=1_000_000.0, positions=None, reject=(), market_open=True):
        self._nav = nav
        self._positions = dict(positions or {})
        self._reject = set(reject)
        self._market_open = market_open
        self.submitted: list[dict] = []
        self.cancelled = 0
        self.closed = False

    def submit(self, symbol, side, quantity, *, limit_price=None):
        if symbol in self._reject:
            raise RuntimeError("insufficient buying power")
        self.submitted.append(
            {"symbol": symbol, "side": side, "quantity": quantity, "limit_price": limit_price}
        )
        return Fill(
            timestamp=pd.Timestamp.utcnow().to_pydatetime(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=limit_price or 0.0,
            cost=0.0,
            order_id=f"FAKE-{len(self.submitted)}",
        )

    def positions(self):
        return dict(self._positions)

    def nav(self):
        return self._nav

    def is_open(self):
        return self._market_open

    def last_prices(self, symbols):
        return PRICES.reindex([s for s in symbols if s in PRICES.index]).dropna()

    def cancel_open_orders(self):
        self.cancelled += 1
        return 3


def _book(weights, asof=None, diagnostics=None):
    series = pd.Series(weights, dtype=float)
    return TargetBook(
        asof=asof or date.today(),
        weights=series,
        sectors=SECTORS.reindex(series.index).fillna("unknown"),
        prices=PRICES.reindex(series.index),
        adv=pd.Series(50_000_000.0, index=series.index),
        construction="risk_constrained",
        universe=len(series),
        diagnostics=diagnostics if diagnostics is not None else {"ex_ante_vol": 0.10},
    )


class Settings:
    """Just the limit fields the gate reads."""

    max_gross_leverage = 2.0
    max_net_exposure = 0.10
    max_name_weight = 0.05
    max_sector_net = 0.10
    max_drawdown = 0.12


# --- pricing and slicing ----------------------------------------------------


def test_limit_price_crosses_the_touch_in_the_right_direction():
    assert limit_price("buy", 100.0, 10.0) > 100.0
    assert limit_price("sell", 100.0, 10.0) < 100.0
    # 10 bps of 100 is 0.10, and rounding goes the aggressive way.
    assert limit_price("buy", 100.0, 10.0) == pytest.approx(100.10)
    assert limit_price("sell", 100.0, 10.0) == pytest.approx(99.90)


def test_limit_price_never_returns_a_non_positive_price():
    assert limit_price("sell", 0.05, 9_000.0) > 0
    with pytest.raises(ValueError):
        limit_price("buy", 0.0, 10.0)


def test_slices_respect_the_notional_bound_and_conserve_shares():
    children = slice_shares(1000, 100.0, max_slice_notional=25_000.0)
    assert sum(children) == 1000
    assert all(c * 100.0 <= 25_000.0 + 1e-9 for c in children)
    assert all(float(c).is_integer() for c in children)
    # No stub child: the remainder is spread over the first slices.
    assert min(children) >= max(children) - 1


def test_small_order_is_a_single_slice():
    assert slice_shares(3, 100.0, max_slice_notional=25_000.0) == [3.0]
    assert slice_shares(0, 100.0, 25_000.0) == []


def test_share_priced_above_the_slice_bound_still_goes_out():
    """One share of a very expensive name cannot be split, so it must not vanish."""
    assert slice_shares(1, 5_000.0, max_slice_notional=1_000.0) == [1.0]


# --- reconciliation ---------------------------------------------------------


def test_current_weights_are_signed_market_value_over_nav():
    weights = current_weights({"AAA": 100, "BBB": -200}, PRICES, nav=100_000.0)
    assert weights["AAA"] == pytest.approx(0.10)
    assert weights["BBB"] == pytest.approx(-0.10)


def test_held_but_unpriced_name_is_nan_not_zero():
    """
    Reporting an unpriceable holding as flat would invite buying it twice.
    """
    weights = current_weights({"ZZZ": 100}, PRICES, nav=100_000.0)
    assert math.isnan(weights["ZZZ"])

    plan = build_trade_plan(
        pd.Series({"ZZZ": 0.05}), {"ZZZ": 100}, PRICES, 100_000.0
    )
    assert plan.trades == []
    assert any("unpriced" in s["reason"] or s["reason"] == "no price" for s in plan.skipped)


def test_plan_moves_the_book_toward_the_target():
    target = pd.Series({"AAA": 0.05, "BBB": -0.05})
    plan = build_trade_plan(target, {}, PRICES, nav=1_000_000.0)

    assert {t.symbol for t in plan.trades} == {"AAA", "BBB"}
    by_symbol = {t.symbol: t for t in plan.trades}
    assert by_symbol["AAA"].side == "buy"
    assert by_symbol["BBB"].side == "sell"
    # 5% of 1,000,000 at 100 is 500 shares.
    assert by_symbol["AAA"].shares == 500
    assert plan.post_trade_weights()["AAA"] == pytest.approx(0.05)


def test_no_trade_band_leaves_small_drift_alone():
    target = pd.Series({"AAA": 0.051})
    plan = build_trade_plan(
        target,
        {"AAA": 500},
        PRICES,
        nav=1_000_000.0,
        config=ReconcileConfig(no_trade_band=0.005),
    )
    assert plan.trades == []
    assert plan.skipped[0]["reason"] == "within no-trade band"


def test_exit_always_trades_even_inside_the_band():
    """A name the strategy dropped is a position with no thesis."""
    plan = build_trade_plan(
        pd.Series(dtype=float),
        {"AAA": 5},
        PRICES,
        nav=1_000_000.0,
        config=ReconcileConfig(no_trade_band=0.10, min_trade_notional=10_000.0),
    )
    assert len(plan.trades) == 1
    assert plan.trades[0].side == "sell"
    assert plan.trades[0].note == "exit"


def test_participation_cap_truncates_a_large_order():
    target = pd.Series({"AAA": 0.05})
    thin = pd.Series({"AAA": 100_000.0})  # only $100k trades in a session
    plan = build_trade_plan(
        target,
        {},
        PRICES,
        nav=1_000_000.0,
        adv=thin,
        config=ReconcileConfig(max_participation=0.02),
    )
    assert len(plan.trades) == 1
    assert plan.trades[0].notional <= 0.02 * 100_000.0 + 100.0
    assert "participation" in plan.trades[0].note


def test_single_order_size_cap_applies_before_participation():
    target = pd.Series({"AAA": 0.50})
    plan = build_trade_plan(
        target, {}, PRICES, nav=1_000_000.0, config=ReconcileConfig(max_order_pct_nav=0.06)
    )
    assert plan.trades[0].notional <= 0.06 * 1_000_000.0 + 100.0
    assert "max order size" in plan.trades[0].note


def test_dust_and_zero_share_orders_are_skipped():
    plan = build_trade_plan(
        pd.Series({"AAA": 0.0001}),
        {},
        PRICES,
        nav=1_000_000.0,
        config=ReconcileConfig(no_trade_band=0.0, min_trade_notional=250.0),
    )
    assert plan.trades == []
    assert plan.skipped[0]["reason"] == "below minimum notional"


def test_turnover_uses_the_backtest_convention():
    """One-way: half the sum of absolute weight changes."""
    target = pd.Series({"AAA": 0.05, "BBB": -0.05})
    plan = build_trade_plan(target, {}, PRICES, nav=1_000_000.0)
    assert plan.turnover == pytest.approx(0.05, rel=1e-6)


def test_zero_nav_produces_no_trades():
    plan = build_trade_plan(pd.Series({"AAA": 0.05}), {}, PRICES, nav=0.0)
    assert plan.trades == []
    assert plan.skipped[0]["reason"].startswith("nav is zero")


def test_drift_report_ranks_by_absolute_gap():
    frame = drift_report(
        pd.Series({"AAA": 0.05, "BBB": -0.05}), {"AAA": 490}, PRICES, 1_000_000.0
    )
    assert list(frame["symbol"])[0] == "BBB"
    assert frame["drift"].abs().is_monotonic_decreasing


# --- pre-trade gate ---------------------------------------------------------


def test_clean_run_is_not_blocked():
    book = _book({"AAA": 0.05, "CCC": -0.05})
    plan = build_trade_plan(book.weights, {}, PRICES, 1_000_000.0)
    report = pretrade_check(plan, book, Settings(), market_open=True)
    assert not report.blocked


def test_stale_lake_blocks_the_run():
    book = _book({"AAA": 0.05, "CCC": -0.05}, asof=date.today() - timedelta(days=30))
    plan = build_trade_plan(book.weights, {}, PRICES, 1_000_000.0)
    report = pretrade_check(plan, book, Settings(), market_open=True)
    assert report.blocked
    assert [f.code for f in report.blocking] == ["stale_data"]
    assert "gig ingest" in report.blocking[0].detail


def test_closed_market_blocks_and_is_the_only_forceable_check():
    from gig.execution.trader import FORCEABLE

    book = _book({"AAA": 0.05, "CCC": -0.05})
    plan = build_trade_plan(book.weights, {}, PRICES, 1_000_000.0)
    report = pretrade_check(plan, book, Settings(), market_open=False)
    assert [f.code for f in report.blocking] == ["market_closed"]
    assert FORCEABLE == {"market_closed"}


def test_limit_breach_in_the_target_blocks_the_run():
    """The gate re-derives limits; it does not trust the optimizer's report."""
    book = _book({"AAA": 0.40, "CCC": -0.40})  # 40% in one name
    plan = build_trade_plan(book.weights, {}, PRICES, 1_000_000.0)
    report = pretrade_check(plan, book, Settings(), market_open=True)
    codes = [f.code for f in report.blocking]
    assert "limit_breach" in codes
    assert any("name_weight" in f.detail for f in report.blocking)


def test_excess_ex_ante_vol_blocks_the_run():
    book = _book({"AAA": 0.05, "CCC": -0.05}, diagnostics={"ex_ante_vol": 0.55})
    plan = build_trade_plan(book.weights, {}, PRICES, 1_000_000.0)
    report = pretrade_check(plan, book, Settings(), market_open=True)
    assert "ex_ante_vol" in [f.code for f in report.blocking]


def test_drawdown_past_the_halt_blocks_the_run():
    book = _book({"AAA": 0.05, "CCC": -0.05})
    plan = build_trade_plan(book.weights, {}, PRICES, 800_000.0)
    report = pretrade_check(
        plan, book, Settings(), market_open=True, nav_high_water=1_000_000.0
    )
    assert "drawdown_halt" in [f.code for f in report.blocking]


def test_runaway_turnover_blocks_the_run():
    # Already invested — a huge flip should still trip the cap.
    book = _book({"AAA": 0.05, "BBB": -0.05})
    plan = build_trade_plan(
        book.weights, {"AAA": -500, "BBB": 500}, PRICES, 1_000_000.0
    )
    report = pretrade_check(
        plan, book, Settings(), config=PreTradeConfig(max_turnover=0.01), market_open=True
    )
    assert "turnover_cap" in [f.code for f in report.blocking]


def test_empty_account_bootstrap_waives_turnover_cap():
    book = _book({"AAA": 0.05, "BBB": -0.05})
    plan = build_trade_plan(book.weights, {}, PRICES, 1_000_000.0)
    report = pretrade_check(
        plan, book, Settings(), config=PreTradeConfig(max_turnover=0.01), market_open=True
    )
    assert not report.blocked
    assert "turnover_bootstrap" in [f.code for f in report.warnings]


def test_unborrowable_short_is_dropped_without_blocking_the_run():
    book = _book({"AAA": 0.05, "CCC": -0.05})
    plan = build_trade_plan(book.weights, {}, PRICES, 1_000_000.0)
    report = pretrade_check(
        plan,
        book,
        Settings(),
        market_open=True,
        shortable={"CCC": False},
    )
    assert not report.blocked
    assert report.dropped == {"CCC": "not shortable at broker"}

    removed = apply_drops(plan, report)
    assert removed == 1
    assert {t.symbol for t in plan.trades} == {"AAA"}


def test_selling_down_a_long_is_not_a_borrow_question():
    """Reducing a long position needs no locate, so it must not be dropped."""
    book = _book({"AAA": 0.02})
    plan = build_trade_plan(book.weights, {"AAA": 500}, PRICES, 1_000_000.0)
    assert plan.trades[0].side == "sell"
    report = pretrade_check(plan, book, Settings(), market_open=True, shortable={"AAA": False})
    assert report.dropped == {}


def test_missing_target_book_blocks_immediately():
    empty = TargetBook(
        asof=None,
        weights=pd.Series(dtype=float),
        sectors=pd.Series(dtype=object),
        prices=pd.Series(dtype=float),
        adv=pd.Series(dtype=float),
        construction="none",
        universe=0,
        available=False,
        hint="run gig ingest",
    )
    plan = build_trade_plan(pd.Series(dtype=float), {}, PRICES, 1_000_000.0)
    report = pretrade_check(plan, empty, Settings())
    assert [f.code for f in report.blocking] == ["no_target"]
    assert report.blocking[0].detail == "run gig ingest"


# --- the loop ---------------------------------------------------------------


@pytest.fixture
def stub_target(monkeypatch):
    """Replace the target builder so the loop tests need no data lake."""
    book = _book({"AAA": 0.05, "CCC": -0.05})
    monkeypatch.setattr("gig.execution.trader.build_target_book", lambda **_kw: book)
    return book


def test_dry_run_sends_nothing(stub_target):
    broker = FakeBroker()
    run = run_once(dry_run=True, broker=broker, record=False)

    assert run.dry_run
    assert run.plan.trades
    assert broker.submitted == []
    assert broker.cancelled == 0
    assert "DRY RUN" in run.report()


def test_live_run_submits_limit_orders_and_cancels_first(stub_target):
    broker = FakeBroker()
    run = run_once(
        dry_run=False,
        broker=broker,
        record=False,
        execution=ExecutionConfig(max_slice_notional=1e9, slice_pause_s=0.0),
    )

    assert not run.blocked
    assert broker.cancelled == 1
    assert len(broker.submitted) == 2
    assert all(o["limit_price"] is not None for o in broker.submitted)
    assert run.submitted == 2
    assert all(o["status"] == "submitted" for o in run.orders)

    buy = next(o for o in broker.submitted if o["side"] == "buy")
    assert buy["symbol"] == "AAA"
    assert buy["limit_price"] > PRICES["AAA"]


def test_blocked_run_submits_nothing(stub_target):
    broker = FakeBroker(market_open=False)
    run = run_once(dry_run=False, broker=broker, record=False)

    assert run.blocked
    assert broker.submitted == []
    assert "BLOCKED" in run.report()


def test_force_overrides_only_the_market_hours_check(stub_target):
    broker = FakeBroker(market_open=False)
    run = run_once(
        dry_run=False,
        broker=broker,
        force=True,
        record=False,
        execution=ExecutionConfig(max_slice_notional=1e9, slice_pause_s=0.0),
    )
    assert not run.blocked
    assert len(broker.submitted) == 2
    assert any("forced" in f.detail for f in run.pretrade.warnings)


def test_force_does_not_override_a_limit_breach(monkeypatch):
    monkeypatch.setattr(
        "gig.execution.trader.build_target_book",
        lambda **_kw: _book({"AAA": 0.40, "CCC": -0.40}),
    )
    broker = FakeBroker()
    run = run_once(dry_run=False, broker=broker, force=True, record=False)

    assert run.blocked
    assert broker.submitted == []


def test_rejected_order_is_recorded_and_does_not_abort_the_run(stub_target):
    broker = FakeBroker(reject={"AAA"})
    run = run_once(
        dry_run=False,
        broker=broker,
        record=False,
        execution=ExecutionConfig(max_slice_notional=1e9, slice_pause_s=0.0),
    )

    statuses = {o["symbol"]: o["status"] for o in run.orders}
    assert statuses["AAA"] == "rejected"
    assert statuses["CCC"] == "submitted"
    assert run.submitted == 1
    assert any("AAA" in e for e in run.errors)


def test_orders_are_sliced_by_notional(stub_target):
    broker = FakeBroker()
    run_once(
        dry_run=False,
        broker=broker,
        record=False,
        execution=ExecutionConfig(max_slice_notional=10_000.0, slice_pause_s=0.0),
    )
    # 50,000 notional per leg in 10,000 chunks.
    aaa = [o for o in broker.submitted if o["symbol"] == "AAA"]
    assert len(aaa) == 5
    assert sum(o["quantity"] for o in aaa) == 500


def test_existing_positions_are_reconciled_not_rebought(stub_target):
    broker = FakeBroker(positions={"AAA": 500})  # already at the 5% target
    run = run_once(dry_run=True, broker=broker, record=False)

    traded = {t.symbol for t in run.plan.trades}
    assert "AAA" not in traded
    assert "CCC" in traded


def test_broker_failure_is_reported_not_raised(stub_target):
    class Broken(FakeBroker):
        def nav(self):
            raise RuntimeError("api down")

    run = run_once(dry_run=True, broker=Broken(), record=False)
    assert any("nav unavailable" in e for e in run.errors)
    assert run.plan.trades == []


def test_close_all_default_flattens_every_position():
    broker = FakeBroker(positions={"AAA": 100, "BBB": -200})
    fills = broker.close_all()

    assert {f.symbol for f in fills} == {"AAA", "BBB"}
    sides = {f.symbol: f.side for f in fills}
    assert sides == {"AAA": "sell", "BBB": "buy"}
    assert all(f.quantity > 0 for f in fills)


def test_flatten_dry_run_closes_nothing():
    from gig.execution.trader import flatten

    broker = FakeBroker(positions={"AAA": 100})
    out = flatten(broker=broker, dry_run=True)
    assert out["dry_run"] is True
    assert out["would_close"] == {"AAA": 100}
    assert broker.submitted == []
