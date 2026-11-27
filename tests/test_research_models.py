"""Heston and barrier prices, expiry profiles, stress, liquidation VaR and verified long-only optimizers."""
from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.special import zeta
from scipy.stats import chi2, norm

from gig.options import exotics
from gig.options.analytics import black_scholes
from gig.portfolio import rebalance
from gig.risk import stress


def _bridge_value(spot, strike, level, rate, div, vol, ttm, call, knock, *, paths, steps, seed):
    """Brownian-bridge Monte Carlo for one single-barrier contract. Seeded, so the estimate does not move."""
    rng = np.random.default_rng(seed)
    dt = ttm/steps
    draws = rng.standard_normal((paths, steps))
    log_paths = np.log(spot)+np.cumsum((rate-div-.5*vol*vol)*dt+vol*math.sqrt(dt)*draws, axis=1)
    barrier = math.log(level)
    previous = np.full(paths, math.log(spot))
    hit = np.zeros(paths, dtype=bool)
    uniforms = rng.random((paths, steps))
    variance = vol*vol*dt
    up = level > spot
    for step in range(steps):
        current = log_paths[:, step]
        if up:
            endpoint = (previous >= barrier) | (current >= barrier)
            gap = np.maximum(barrier-previous, 0.)*np.maximum(barrier-current, 0.)
        else:
            endpoint = (previous <= barrier) | (current <= barrier)
            gap = np.maximum(previous-barrier, 0.)*np.maximum(current-barrier, 0.)
        hit |= endpoint | (uniforms[:, step] < np.exp(-2*gap/variance))
        previous = current
    terminal = np.exp(log_paths[:, -1])
    intrinsic = np.maximum(terminal-strike, 0.) if call else np.maximum(strike-terminal, 0.)
    if knock == "IN":
        kept = np.where(hit, intrinsic, 0.)
    elif knock == "OUT":
        kept = np.where(hit, 0., intrinsic)
    else:
        raise ValueError("knock must be IN or OUT")
    return float(math.exp(-rate*ttm)*kept.mean())


class OptionModelTests(unittest.TestCase):
    def test_heston_recovers_black_scholes_and_a_seeded_simulation(self):
        reference = black_scholes(100, 110, .02, .01, .2, .5, True)["price"]
        flat = exotics.heston(100, 110, .02, .01, .5, True, initial_variance=.04, mean_reversion=3,
                               long_run_variance=.04, vol_of_variance=1e-4, correlation=0.)
        self.assertAlmostEqual(flat["price"], reference, places=5)
        self.assertAlmostEqual(flat["implied_volatility"], .2, places=5)
        priced = exotics.heston(100, 100, .02, .01, .5, True, initial_variance=.04, mean_reversion=2,
                                 long_run_variance=.04, vol_of_variance=.5, correlation=-.4)
        discount = math.exp(-.02*.5)
        self.assertAlmostEqual(priced["call"]-priced["put"], discount*(priced["forward"]-100), places=8)
        self.assertLessEqual(discount*max(priced["forward"]-100, 0.), priced["call"])
        self.assertLessEqual(priced["call"], discount*priced["forward"])
        self.assertFalse(priced["feller_condition_satisfied"])
        rng = np.random.default_rng(1)
        paths, steps = 40_000, 200
        dt = .5/steps
        first = rng.standard_normal((paths, steps))
        second = -.4*first+math.sqrt(1-.16)*rng.standard_normal((paths, steps))
        variance = np.full(paths, .04)
        log_spot = np.full(paths, math.log(100))
        for step in range(steps):
            current = np.maximum(variance, 0.)
            log_spot += (.02-.01-.5*current)*dt+np.sqrt(current*dt)*first[:, step]
            variance = variance+2*(.04-current)*dt+.5*np.sqrt(current*dt)*second[:, step]
        simulated = math.exp(-.02*.5)*np.maximum(np.exp(log_spot)-100, 0.).mean()
        self.assertAlmostEqual(priced["price"], simulated, delta=.12)
        base = {"initial_variance": .04, "mean_reversion": 2, "long_run_variance": .04, "vol_of_variance": .3, "correlation": -.4}
        for bad in ({"correlation": 1}, {"initial_variance": -1}, {"vol_of_variance": 0}):
            with self.assertRaises(ValueError):
                exotics.heston(100, 100, .02, 0, 1, True, **{**base, **bad})

    def test_every_barrier_branch_agrees_with_a_brownian_bridge(self):
        shift = -zeta(.5)/math.sqrt(2*math.pi)
        self.assertAlmostEqual(exotics.DISCRETE_MONITORING_SHIFT, shift, places=12)
        spot, rate, div, vol, ttm = 100., .03, .01, .25, 1.
        for call, direction, level, strike in (
                (True, "DOWN", 80., 100.), (True, "DOWN", 90., 80.), (True, "UP", 120., 100.), (True, "UP", 120., 130.),
                (False, "DOWN", 80., 100.), (False, "DOWN", 90., 110.), (False, "UP", 120., 100.), (False, "UP", 120., 140.)):
            for knock in ("IN", "OUT"):
                closed = exotics.barrier(spot, strike, level, rate, div, vol, ttm, call, direction=direction, knock=knock)
                simulated = _bridge_value(spot, strike, level, rate, div, vol, ttm, call, knock, paths=40_000, steps=120, seed=0)
                self.assertAlmostEqual(closed["price"], simulated, delta=.2)
                self.assertAlmostEqual(closed["knock_in"]+closed["knock_out"], closed["vanilla"], places=8)
        vanilla = black_scholes(spot, 100, rate, div, vol, ttm, True)["price"]
        far = exotics.barrier(spot, 100, 20, rate, div, vol, ttm, True, direction="DOWN", knock="OUT")
        self.assertAlmostEqual(far["price"], vanilla, places=4)
        continuous = exotics.barrier(spot, 100, 80, rate, div, vol, ttm, True, direction="DOWN", knock="OUT")
        discrete = exotics.barrier(spot, 100, 80, rate, div, vol, ttm, True, direction="DOWN", knock="OUT", monitoring_dates=52)
        self.assertLess(discrete["effective_barrier"], 80)
        self.assertGreater(discrete["price"], continuous["price"])
        with self.assertRaisesRegex(ValueError, "above spot"):
            exotics.barrier(spot, 100, 90, rate, div, vol, ttm, True, direction="UP", knock="OUT")

    def test_expiry_profile_matches_hand_computed_payoffs(self):
        spread = exotics.strategy([{"kind": "CALL", "strike": 100, "quantity": 1, "premium": 5},
                                   {"kind": "CALL", "strike": 110, "quantity": -1, "premium": 2}], multiplier=100)
        self.assertEqual((spread["net_premium"], spread["premium_direction"], spread["breakevens"]), (300., "DEBIT", [103.]))
        self.assertEqual((spread["max_profit"], spread["max_loss"], spread["max_profit_state"], spread["max_loss_state"]),
                         (700., 300., "BOUNDED", "BOUNDED"))
        covered = exotics.strategy([{"kind": "UNDERLYING", "quantity": 1, "premium": 100},
                                    {"kind": "CALL", "strike": 110, "quantity": -1, "premium": 5}])
        self.assertEqual((covered["breakevens"], covered["max_profit"], covered["max_loss"], covered["slope_above_highest_strike"]),
                         ([95.], 15., 95., 0.))
        call = exotics.strategy([{"kind": "CALL", "strike": 100, "quantity": 1, "premium": 5}])
        self.assertEqual((call["breakevens"], call["max_profit_state"], call["max_loss"]), ([105.], "UNBOUNDED", 5.))
        at_breakeven = next(row for row in spread["profile"] if row["underlying"] == 103)
        self.assertEqual(at_breakeven["profit"], 0.)
        for bad in ([{"kind": "CALL", "strike": 100, "quantity": 0, "premium": 1}],
                    [{"kind": "CALL", "quantity": 1, "premium": 1}],
                    [{"kind": "UNDERLYING", "strike": 100, "quantity": 1, "premium": 100}],
                    [{"kind": "CALL", "strike": 100, "quantity": 1, "premium": 1}]*17):
            with self.assertRaises(ValueError):
                exotics.strategy(bad)


class StressTests(unittest.TestCase):
    def test_a_shock_propagates_by_the_regression_formula(self):
        result = stress.conditional_stress([.6, .4], [[.04, .02], [.02, .09]], [0], [-.3])
        self.assertAlmostEqual(result["asset_returns"][1], -.15)
        self.assertAlmostEqual(result["portfolio_return"], -.24)
        self.assertAlmostEqual(result["direct_return"], -.18)
        self.assertAlmostEqual(sum(result["contributions"]), result["portfolio_return"])
        self.assertAlmostEqual(result["shock_mahalanobis_distance"], 1.5)
        self.assertAlmostEqual(result["shock_tail_probability"], float(chi2.sf(2.25, 1)))
        self.assertAlmostEqual(result["residual_volatility"], .4*math.sqrt(.08))
        with self.assertRaises(ValueError):
            stress.conditional_stress([.6, .4], [[.04, .02], [.02, .09]], [0, 1], [-.3, .1])
        with self.assertRaises(ValueError):
            stress.conditional_stress([.6, .4], [[.04, .02], [.02, .09]], [0, 0], [-.3, .1])

    def test_liquidation_var_matches_the_closed_horizon_factor(self):
        result = stress.liquidity_adjusted_var(1_000_000, .02, average_daily_value=1_000_000, max_participation=.25,
                                               spread=.002, spread_volatility=.001, spread_multiplier=3, confidence=.99)
        factor = math.sqrt(5*9/(6*4))
        one_day = float(norm.ppf(.99))*.02*1_000_000
        self.assertEqual(result["liquidation_days"], 4)
        self.assertAlmostEqual(result["horizon_factor"], factor)
        self.assertAlmostEqual(result["one_day_var"], one_day)
        self.assertAlmostEqual(result["spread_cost"], .5*1_000_000*(.002+3*.001))
        self.assertAlmostEqual(result["liquidity_adjusted_var"], one_day*factor+result["spread_cost"])
        one = stress.liquidity_adjusted_var(250_000, .02, average_daily_value=1_000_000, max_participation=.25, confidence=.99)
        self.assertEqual((one["liquidation_days"], one["horizon_factor"]), (1, 1.))
        capped = stress.liquidity_adjusted_var(1_000_000, .02, average_daily_value=1_000_000, max_participation=1,
                                               spread=.9, spread_volatility=.2, spread_multiplier=3)
        self.assertAlmostEqual(capped["spread_cost"], 500_000)
        with self.assertRaisesRegex(ValueError, "ten trading years"):
            stress.liquidity_adjusted_var(2521, .02, average_daily_value=1, max_participation=1)
        with self.assertRaises(ValueError):
            stress.liquidity_adjusted_var(True, .02, average_daily_value=1_000_000)


class OptimizerTests(unittest.TestCase):
    def test_rebalance_matches_a_scan_and_leaves_an_optimal_book_unchanged(self):
        covariance = np.array([[.04, .01], [.01, .09]])
        expected = np.array([.08, .12])
        aversion, cost, current = 3., .02, np.array([.2, .8])

        def utility(weights):
            return float(expected @ weights-.5*aversion*weights @ covariance @ weights-cost*np.abs(weights-current).sum())

        grid = [np.array([weight, 1-weight]) for weight in np.linspace(0, 1, 2001)]
        best = max(grid, key=utility)
        result = rebalance.rebalance_with_costs(current.tolist(), expected.tolist(), covariance.tolist(),
                                                risk_aversion=aversion, cost=cost)
        self.assertLess(np.max(np.abs(np.array(result["weights"])-best)), 5e-4)
        self.assertGreaterEqual(result["utility_after_cost"], utility(best)-1e-8)
        self.assertEqual(result["verification"], "KKT_CONDITIONS_CHECKED")
        optimal = rebalance.rebalance_with_costs([.5, .5], [.1, .1], np.eye(2).tolist(), risk_aversion=2, cost=.01)
        self.assertLess(optimal["turnover"], 1e-6)
        self.assertLessEqual(optimal["utility_gain"], 1e-8)
        capped = rebalance.rebalance_with_costs(current.tolist(), expected.tolist(), covariance.tolist(),
                                                risk_aversion=aversion, cost=0, max_turnover=.05)
        self.assertLessEqual(capped["turnover"], .05+1e-8)

        def mean_variance(weights):
            return float(expected @ weights-.5*aversion*weights @ covariance @ weights)

        allowed = [weights for weights in grid if np.abs(weights-current).sum()/2 <= .05+1e-12]
        self.assertGreaterEqual(capped["utility_after_cost"], max(mean_variance(weights) for weights in allowed)-1e-6)
        with self.assertRaisesRegex(ValueError, "below a current weight"):
            rebalance.rebalance_with_costs([.8, .2], expected.tolist(), covariance.tolist(), risk_aversion=3, cost=0, max_weight=.5)

    def test_maximum_diversification_is_inverse_volatility_and_respects_a_cap(self):
        volatility = np.array([.2, .1, .3])
        covariance = np.diag(volatility**2)
        result = rebalance.maximum_diversification(covariance.tolist())
        inverse = 1/volatility
        self.assertLess(np.max(np.abs(np.array(result["weights"])-inverse/inverse.sum())), 1e-8)
        self.assertAlmostEqual(result["diversification_ratio"], math.sqrt(3))
        capped = rebalance.maximum_diversification(covariance.tolist(), max_weight=.4)
        self.assertLessEqual(max(capped["weights"]), .4+1e-8)
        best = 0.
        for first in np.linspace(0, .4, 41):
            for second in np.linspace(0, .4, 41):
                third = 1-first-second
                if 0 <= third <= .4+1e-12:
                    weights = np.array([first, second, third])
                    best = max(best, float(volatility @ weights/math.sqrt(weights @ covariance @ weights)))
        self.assertGreaterEqual(capped["diversification_ratio"], best-1e-6)

    def test_tracking_portfolio_stays_at_a_feasible_benchmark_and_refuses_an_unreachable_floor(self):
        covariance = [[.04, .01, .005], [.01, .09, .004], [.005, .004, .02]]
        benchmark = [.5, .3, .2]
        held = rebalance.tracking_error_portfolio(benchmark, covariance)
        self.assertLess(np.max(np.abs(np.array(held["weights"])-benchmark)), 1e-8)
        self.assertAlmostEqual(held["tracking_error"], 0)
        self.assertAlmostEqual(held["beta_to_benchmark"], 1)
        excluded = rebalance.tracking_error_portfolio(benchmark, covariance, excluded=[0])
        self.assertEqual(excluded["weights"][0], 0.)
        self.assertGreaterEqual(excluded["active_share"], .5-1e-8)
        self.assertAlmostEqual(sum(excluded["tracking_contributions"]), excluded["tracking_error"], places=6)
        matrix = np.array(covariance)
        weights, bench = np.array(excluded["weights"]), np.array(benchmark)
        self.assertAlmostEqual(excluded["beta_to_benchmark"], float(weights @ matrix @ bench/(bench @ matrix @ bench)))
        with self.assertRaisesRegex(ValueError, "not attainable"):
            rebalance.tracking_error_portfolio(benchmark, covariance, expected_returns=[.01, .02, .03], minimum_active_return=1)
        with self.assertRaisesRegex(ValueError, "together"):
            rebalance.tracking_error_portfolio(benchmark, covariance, expected_returns=[.01, .02, .03])


if __name__ == "__main__":
    unittest.main()
