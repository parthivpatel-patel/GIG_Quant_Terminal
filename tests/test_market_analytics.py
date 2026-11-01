"""Independent numerical/event references for new market research APIs."""
from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from gig.microstructure import (
    MatchedTrade,
    Quote,
    Trade,
    kyle_lambda,
    lee_ready,
    match_quotes,
    order_flow_imbalance,
    vpin,
)
from gig.options import (
    OptionQuote,
    SurfacePoint,
    black_scholes,
    diagnose_call_slice,
    implied_volatility,
    interpolate_total_variance,
    validate_surface,
)


def q(ts=10, instrument="X", bid=99., ask=101., bid_size=10., ask_size=10., **kw):
    return Quote(ts, instrument, bid, ask, bid_size, ask_size, **kw)


def t(ts=11, instrument="X", price=100., size=5., **kw):
    return Trade(ts, instrument, price, size, **kw)


class MatchingTests(unittest.TestCase):
    def test_no_future_quote_and_instrument_separation(self):
        quotes = [q(20), q(9, instrument="Y"), q(10)]
        matches = match_quotes([t(11), t(8), t(10, instrument="Y")], quotes)
        self.assertEqual(matches[0].quote.timestamp_ns, 10)
        self.assertIsNone(matches[1].quote)
        self.assertEqual(matches[1].reason, "NO_PRIOR_QUOTE")
        self.assertEqual(matches[2].quote.instrument, "Y")

    def test_lag_staleness_and_boundary(self):
        self.assertEqual(match_quotes([t(11)], [q(10)], max_age_ns=1)[0].reason, "MATCHED")
        self.assertEqual(match_quotes([t(12)], [q(10)], max_age_ns=1)[0].reason, "STALE_QUOTE")
        self.assertIsNone(match_quotes([t(11)], [q(10)], lag_ns=2)[0].quote)
        self.assertEqual(match_quotes([t(10)], [q(10)], lag_ns=0)[0].quote_age_ns, 0)
        self.assertIsNone(match_quotes([t(10)], [q(10)], lag_ns=1)[0].quote)

    def test_timestamp_nanosecond_and_large_integer_identity(self):
        now = 1_800_000_000_000_000_001
        matches = match_quotes([t(now+1)], [q(now), q(now+2)], max_age_ns=1)
        self.assertEqual(matches[0].quote.timestamp_ns, now)
        self.assertEqual(matches[0].quote_age_ns, 1)

    def test_duplicate_quote_rejected_and_sequence_tie_explicit(self):
        with self.assertRaises(ValueError):
            match_quotes([t()], [q(), q()])
        matched = match_quotes([t()], [q(sequence=2), q(sequence=1)])
        self.assertEqual(matched[0].quote.sequence, 2)

    def test_duplicate_trade_rejected(self):
        with self.assertRaises(ValueError):
            match_quotes([t(), t()], [q()])

    def test_negative_lag_or_fractional_timestamp_rejected(self):
        for kwargs in ({"lag_ns": -1}, {"max_age_ns": -1}, {"lag_ns": .5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                match_quotes([t()], [q()], **kwargs)
        for bad in (True, 1.5, float("nan")):
            with self.subTest(timestamp=bad), self.assertRaises(ValueError):
                q(bad)

    def test_event_admission_and_immutability(self):
        for kwargs in ({"bid": float("nan")}, {"ask": float("inf")}, {"bid": 102}, {"bid_size": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                q(**kwargs)
        with self.assertRaises(ValueError):
            t(size=0)
        with self.assertRaises(FrozenInstanceError):
            q().bid = 42
        self.assertEqual(q(source="vendor-A", data_kind="SYNTHETIC").data_kind, "SYNTHETIC")
        self.assertEqual(match_quotes([t()], [q(source="vendor-A")])[0].quote.source, "vendor-A")

    def test_lee_ready_quote_then_tick_and_unknown(self):
        trades = [t(11, price=100), t(12, price=101), t(13, price=100), t(14, price=100)]
        signed = lee_ready(match_quotes(trades, [q(10)]))
        self.assertEqual([r.sign for r in signed], [0, 1, -1, -1])
        self.assertEqual([r.method for r in signed], ["UNKNOWN", "QUOTE_ESTIMATE", "TICK_ESTIMATE", "TICK_ESTIMATE"])

    def test_lee_ready_missing_quote_tick_is_estimate(self):
        signed = lee_ready(match_quotes([t(11, price=99), t(12, price=101)], []))
        self.assertEqual([r.sign for r in signed], [0, 1])
        self.assertEqual(signed[1].method, "TICK_ESTIMATE")

    def test_lee_ready_does_not_cross_instruments(self):
        signed = lee_ready(match_quotes([t(11, "X", price=100), t(12, "Y", price=200)], []))
        self.assertEqual([r.sign for r in signed], [0, 0])

    def test_lee_ready_preserves_input_order(self):
        signed = lee_ready(match_quotes([t(12, price=101), t(11, price=99)], []))
        self.assertEqual([r.sign for r in signed], [1, 0])

    def test_invalid_manual_match_rejected(self):
        for quote in (q(20), q(instrument="Y")):
            with self.assertRaises(ValueError):
                lee_ready([MatchedTrade(t(), quote, 0, "MATCHED")])


class MicrostructureNumericalTests(unittest.TestCase):
    def test_ofi_reference_events(self):
        # Size-only change: +5 bid and -3 ask => +8; upward best bid
        # contributes its new queue +7; upward ask removes old ask queue +7.
        quotes = [q(1), q(2, bid_size=15, ask_size=7), q(3, bid=100, ask=102, bid_size=7, ask_size=20)]
        np.testing.assert_array_equal(order_flow_imbalance(quotes), [0, 8, 14])

    def test_ofi_downward_price_reference(self):
        np.testing.assert_array_equal(order_flow_imbalance([q(1), q(2, bid=98, ask=100, ask_size=4)]), [0, -14])

    def test_ofi_requires_ordered_single_instrument(self):
        for rows in ([q(2), q(1)], [q(1), q(2, instrument="Y")], [q(1), q(1)]):
            with self.assertRaises(ValueError):
                order_flow_imbalance(rows)
        self.assertEqual(order_flow_imbalance([]).size, 0)

    def test_kyle_known_linear_regression(self):
        x = np.array([-4., -2., 0., 3., 7.])
        result = kyle_lambda(.25 + .02*x, x)
        self.assertEqual(result.state, "AVAILABLE")
        self.assertAlmostEqual(result.coefficient, .02, places=14)
        self.assertAlmostEqual(result.intercept, .25, places=14)
        self.assertAlmostEqual(result.r_squared, 1.)

    def test_kyle_missing_information_is_not_zero(self):
        for y, x in (([1, 2], [1, 2]), ([1, 2, 3], [4, 4, 4])):
            result = kyle_lambda(y, x)
            self.assertEqual(result.state, "UNAVAILABLE")
            self.assertIsNone(result.coefficient)
        with self.assertRaises(ValueError):
            kyle_lambda([1, float("nan"), 3], [1, 2, 3])

    def test_vpin_exact_volume_splitting_reference(self):
        # 150 buy then 150 sell: buckets are 100B, 50B/50S, 100S.
        result = vpin([150, 150], [1, -1], bucket_volume=100, window_buckets=2)
        self.assertEqual(result.imbalances, (1., 0., 1.))
        self.assertEqual(result.values, (.5, .5))
        self.assertEqual(result.incomplete_volume, 0)

    def test_vpin_unknown_and_partial_are_explicit(self):
        result = vpin([20, 100, 25], [0, 1, -1], bucket_volume=100, window_buckets=2)
        self.assertEqual(result.excluded_unknown_volume, 20)
        self.assertEqual(result.incomplete_volume, 25)
        self.assertEqual(result.completed_buckets, 1)
        self.assertEqual(result.state, "UNAVAILABLE")
        self.assertEqual(result.values, ())

    def test_vpin_bounds_and_limits(self):
        for args in (([1], [2]), ([float("nan")], [1]), ([1, 2], [1])):
            with self.assertRaises(ValueError):
                vpin(*args, bucket_volume=1)
        with self.assertRaises(ValueError):
            vpin([1000], [1], bucket_volume=1, max_buckets=2)


class OptionsNumericalTests(unittest.TestCase):
    def test_standard_black_scholes_published_reference(self):
        result = black_scholes(100, 100, .05, 0, .2, 1)
        self.assertAlmostEqual(result["price"], 10.450583572185565, places=11)
        self.assertAlmostEqual(result["delta"], .6368306511756191, places=12)
        self.assertAlmostEqual(result["gamma"], .018762017345846895, places=12)
        self.assertEqual(result["classification"], "MODEL_ESTIMATE")

    def test_put_call_parity_with_dividends(self):
        for spot, strike, rate, div, vol, ttm in ((100, 110, .03, .01, .4, 2), (80, 100, -.01, .02, .3, .2)):
            call = black_scholes(spot, strike, rate, div, vol, ttm)
            put = black_scholes(spot, strike, rate, div, vol, ttm, False)
            self.assertAlmostEqual(call["price"] - put["price"], spot*math.exp(-div*ttm)-strike*math.exp(-rate*ttm), places=12)
            self.assertAlmostEqual(call["gamma"], put["gamma"])
            self.assertAlmostEqual(call["vega"], put["vega"])

    def test_greeks_finite_difference_reference(self):
        args = dict(spot=100., strike=105., rate=.03, div=.01, vol=.24, ttm=1.3)
        result = black_scholes(**args)
        def price(**changes):
            return black_scholes(**{**args, **changes})["price"]
        h = .001
        delta = (price(spot=100+h)-price(spot=100-h))/(2*h)
        gamma = (price(spot=100+h)-2*price()+price(spot=100-h))/(h*h)
        self.assertAlmostEqual(result["delta"], delta, places=8)
        self.assertAlmostEqual(result["gamma"], gamma, places=7)
        h = 1e-5
        self.assertAlmostEqual(result["vega"], (price(vol=.24+h)-price(vol=.24-h))/(2*h)/100, places=8)
        self.assertAlmostEqual(result["rho"], (price(rate=.03+h)-price(rate=.03-h))/(2*h)/100, places=8)
        self.assertAlmostEqual(result["theta"], -(price(ttm=1.3+h)-price(ttm=1.3-h))/(2*h)/365, places=8)

    def test_input_admission_rejects_invalid_or_undefined_boundaries(self):
        for kwargs in ({"spot": -1}, {"strike": 0}, {"vol": 0}, {"ttm": 0}, {"rate": float("nan")}, {"call": 1}, {"spot": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                black_scholes(**{**dict(spot=100, strike=100), **kwargs})

    def test_iv_reprices_calls_puts_and_negative_rates(self):
        for call in (True, False):
            for strike in (70., 100., 130.):
                for vol in (.1, .3, 1.2):
                    args = dict(spot=100, strike=strike, rate=-.01, div=.02, ttm=1.5, call=call)
                    price = black_scholes(vol=vol, **args)["price"]
                    iv = implied_volatility(price, **args)
                    self.assertAlmostEqual(iv, vol, places=7)

    def test_iv_bounds_and_zero_boundary(self):
        self.assertEqual(implied_volatility(10, 110, 100), 0.)
        for price in (-1., 100., 101., float("inf")):
            with self.subTest(price=price), self.assertRaises(ValueError):
                implied_volatility(price, 100, 100)
        with self.assertRaises(ValueError):
            implied_volatility(5, 110, 100)
        with self.assertRaises(ValueError):
            implied_volatility(black_scholes(100, 100, vol=1.)["price"], 100, 100, max_vol=.2)

    def test_valid_discrete_call_slice(self):
        rows = [OptionQuote(k, black_scholes(100, k)["price"], "model", "MODEL_ESTIMATE") for k in (80, 90, 100, 110, 120)]
        result = diagnose_call_slice(rows, spot=100, rate=0, div=0, ttm=1)
        self.assertEqual(result["state"], "AVAILABLE")
        self.assertFalse(result["continuous_arbitrage_certified"])
        self.assertEqual(result["violations"], [])

    def test_bad_call_slice_diagnosed_without_false_certification(self):
        rows = [OptionQuote(k, p, "input") for k, p in ((90, 15), (100, 14), (110, 1))]
        result = diagnose_call_slice(rows, spot=100, rate=0, div=0, ttm=1)
        self.assertEqual(result["state"], "ERROR")
        self.assertIn("DISCRETE_CONVEXITY", [r["code"] for r in result["violations"]])
        with self.assertRaises(ValueError):
            diagnose_call_slice([rows[0], rows[0]], spot=100, rate=0, div=0, ttm=1)

    def test_sparse_call_slice_is_unknown(self):
        self.assertEqual(diagnose_call_slice([], spot=100, rate=0, div=0, ttm=1)["state"], "UNKNOWN")

    def test_surface_calendar_diagnostics(self):
        good = [SurfacePoint(0, .5, .2, "model"), SurfacePoint(0, 1, .2, "model")]
        self.assertEqual(validate_surface(good).state, "AVAILABLE")
        bad = [SurfacePoint(0, .5, .4, "model"), SurfacePoint(0, 1, .2, "model")]
        self.assertEqual(validate_surface(bad).state, "ERROR")
        self.assertFalse(validate_surface(bad).butterfly_checked)
        self.assertEqual(validate_surface([good[0]]).state, "UNKNOWN")
        with self.assertRaises(ValueError):
            validate_surface([good[0], good[0]])

    def test_surface_total_variance_interpolation_and_no_extrapolation(self):
        rows = [SurfacePoint(k, tau, .2, "input") for k in (-.2, .2) for tau in (.5, 1.5)]
        value = interpolate_total_variance(rows, 0, 1)
        self.assertAlmostEqual(value["volatility"], .2)
        self.assertAlmostEqual(value["total_variance"], .04)
        self.assertEqual(value["classification"], "MODEL_ESTIMATE")
        self.assertFalse(value["arbitrage_certified"])
        self.assertEqual(value["sources"], ["input"])
        with self.assertRaises(ValueError):
            interpolate_total_variance(rows, 1, 1)
        with self.assertRaises(ValueError):
            interpolate_total_variance([SurfacePoint(0, tau, .2, "input") for tau in (.5, 1, 1.5)], 0, 1)


if __name__ == "__main__":
    unittest.main()


class CostAndPortfolioTests(unittest.TestCase):
    def test_vpin_long_window_independent_reference_and_resource_bound(self):
        from gig.microstructure import vpin
        signs = np.random.default_rng(441).choice([-1, 1], 5000)
        result = vpin(np.ones(5000), signs, bucket_volume=2, window_buckets=1100)
        imbalances = np.abs(signs.reshape(-1, 2).sum(axis=1))/2
        expected = np.convolve(imbalances, np.ones(1100)/1100, mode="valid")
        np.testing.assert_allclose(result.values, expected, atol=2e-15, rtol=2e-15)
        self.assertIsInstance(result.values, tuple)
        with self.assertRaises(ValueError):
            vpin([1], [1], bucket_volume=1, max_buckets=100001)

    def test_scenario_work_budget_combines_positions_and_grid(self):
        from gig.options import OptionPosition, scenario_grid
        positions = [OptionPosition(str(i), "SYN", 100, .5, .2, 1) for i in range(101)]
        with self.assertRaises(ValueError):
            scenario_grid(positions, spot=100, spot_shocks=[0]*100, volatility_shocks=[0]*10)

    def test_queue_imbalance_and_unknown_empty_depth(self):
        from gig.microstructure import spread_metrics
        metrics = spread_metrics(q(bid_size=30, ask_size=10))
        self.assertEqual(metrics["queue_imbalance"], .5)
        self.assertEqual(metrics["quoted_spread_bps"], 200.)
        self.assertIsNone(spread_metrics(q(bid_size=0, ask_size=0))["queue_imbalance"])

    def test_signed_execution_costs_and_fees(self):
        from gig.microstructure import execution_costs
        buy = execution_costs(t(price=101, size=100), arrival_mid=100, side=1, fees=5)
        sell = execution_costs(t(price=99, size=100), arrival_mid=100, side=-1, fees=5)
        self.assertEqual(buy["implementation_shortfall"], 105)
        self.assertEqual(sell["implementation_shortfall"], 105)
        self.assertAlmostEqual(buy["cost_bps"], 105)
        with self.assertRaises(ValueError):
            execution_costs(t(), arrival_mid=100, side=0)

    def test_portfolio_multipliers_signed_greeks(self):
        from gig.options import OptionPosition, portfolio_exposure
        long = OptionPosition("X-C", "X", 100, 1, .2, 2)
        short = OptionPosition("X-P", "X", 100, 1, .2, -1, call=False)
        exposure = portfolio_exposure([long, short], spot=100)
        call = black_scholes(100, 100)
        put = black_scholes(100, 100, call=False)
        self.assertAlmostEqual(exposure["price"], 200*call["price"]-100*put["price"])
        self.assertAlmostEqual(exposure["delta"], 200*call["delta"]-100*put["delta"])
        self.assertEqual(exposure["classification"], "MODEL_ESTIMATE")

    def test_portfolio_identity_and_empty_admission(self):
        from gig.options import OptionPosition, portfolio_exposure
        position = OptionPosition("X-C", "X", 100, 1, .2, 1)
        with self.assertRaises(ValueError):
            portfolio_exposure([position, position], spot=100)
        with self.assertRaises(ValueError):
            portfolio_exposure([position, OptionPosition("Y-C", "Y", 100, 1, .2, 1)], spot=100)
        self.assertEqual(portfolio_exposure([], spot=100)["state"], "UNAVAILABLE")

    def test_model_scenario_repricing_and_zero_shock(self):
        from gig.options import OptionPosition, scenario_grid
        position = OptionPosition("X-C", "X", 100, 1, .2, 1)
        result = scenario_grid([position], spot=100, spot_shocks=[0, .1], volatility_shocks=[0, .02])
        self.assertAlmostEqual(result["scenarios"][0]["model_pnl"], 0)
        target = 100*(black_scholes(110, 100, vol=.22)["price"]-black_scholes(100, 100)["price"])
        self.assertAlmostEqual(result["scenarios"][-1]["model_pnl"], target)
        self.assertEqual(result["classification"], "MODEL_ESTIMATE")
        for changes in ({"spot_shocks": [-1]}, {"volatility_shocks": [-.2]}, {"elapsed_years": 1}):
            with self.assertRaises(ValueError):
                scenario_grid([position], **{**dict(spot=100, spot_shocks=[0], volatility_shocks=[0]), **changes})
