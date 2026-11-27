"""Independent references for the stateless quantitative research calculators."""
from __future__ import annotations

import asyncio
import json
import math
import unittest
from itertools import combinations

import httpx
import numpy as np
from scipy.integrate import quad
from scipy.stats import binom, chi2, kurtosis, norm, skew

from gig.data import bars
from gig.execution import schedule
from gig.options import models
from gig.options.analytics import black_scholes
from gig.portfolio import construction
from gig.research import cross_section, overfitting
from gig.research.multiple_testing import probabilistic_sharpe
from gig.risk import var_models
from gig.service.app import create_app
from gig.service.quant_analytics import ANALYSES, run


def covariance(n: int, seed: int) -> np.ndarray:
    factor = np.random.default_rng(seed).normal(size=(n, n))
    return factor @ factor.T/n+.05*np.eye(n)


def tail_loss(losses: np.ndarray, level: float) -> float:
    """Rockafellar-Uryasev definition, minimized over every sample loss."""
    return min(z+np.maximum(losses-z, 0).mean()/(1-level) for z in losses)


class PortfolioConstructionTests(unittest.TestCase):
    def test_shrinkage_matches_scikit_learn(self):
        from sklearn.covariance import ledoit_wolf

        for seed, shape in ((1, (60, 5)), (2, (12, 20)), (3, (400, 8))):
            x = np.random.default_rng(seed).normal(size=shape)*np.linspace(.5, 2, shape[1])
            expected, shrinkage = ledoit_wolf(x)
            result = construction.ledoit_wolf(x.tolist())
            self.assertAlmostEqual(result["shrinkage"], shrinkage, places=12)
            np.testing.assert_allclose(result["covariance"], expected, rtol=1e-12, atol=1e-14)

    def test_hierarchical_risk_parity_matches_the_published_recursion(self):
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import pdist

        def quasi_diagonal(link):
            # Lopez de Prado's getQuasiDiag: expand merged clusters in place.
            link = link.astype(int)
            order, total = [link[-1, 0], link[-1, 1]], link[-1, 3]
            while max(order) >= total:
                order = [leaf for item in order
                         for leaf in ([link[item-total, 0], link[item-total, 1]] if item >= total else [item])]
            return order

        def allocate(cov, items):
            if len(items) == 1:
                return {items[0]: 1.}
            left, right = items[:len(items)//2], items[len(items)//2:]

            def variance(members):
                ivp = 1/np.diag(cov[np.ix_(members, members)])
                ivp /= ivp.sum()
                return ivp @ cov[np.ix_(members, members)] @ ivp
            alpha = 1-variance(left)/(variance(left)+variance(right))
            return {**{k: v*alpha for k, v in allocate(cov, left).items()},
                    **{k: v*(1-alpha) for k, v in allocate(cov, right).items()}}

        for n, seed in ((2, 1), (5, 2), (9, 3), (16, 4)):
            cov = covariance(n, seed)
            sd = np.sqrt(np.diag(cov))
            distance = np.sqrt(np.clip(.5*(1-cov/np.outer(sd, sd)), 0, None))
            np.fill_diagonal(distance, 0)
            order = quasi_diagonal(linkage(pdist(distance), "single"))
            expected = allocate(cov, order)
            result = construction.hierarchical_risk_parity(cov.tolist())
            self.assertEqual(result["order"], order)
            np.testing.assert_allclose(result["weights"], [expected[i] for i in range(n)], rtol=1e-12)
            self.assertAlmostEqual(sum(result["weights"]), 1, places=12)
            self.assertAlmostEqual(sum(result["risk_contributions"]), 1, places=12)

    def test_uncorrelated_assets_get_inverse_variance_weights(self):
        variances = np.array([.04, .01, .09, .0225, .16])
        result = construction.hierarchical_risk_parity(np.diag(variances).tolist())
        np.testing.assert_allclose(result["weights"], (1/variances)/(1/variances).sum(), rtol=1e-12)

    def test_tail_weights_match_the_definition(self):
        rng = np.random.default_rng(5)
        for size, level in ((20, .9), (37, .95), (100, .99), (50, .995), (64, .75)):
            losses = rng.normal(size=size)
            weights, var = var_models.tail_weights(losses, level)
            self.assertAlmostEqual(weights.sum(), 1, places=12)
            self.assertAlmostEqual(float(weights @ losses), tail_loss(losses, level), places=12)
            self.assertIn(var, losses)

    def test_minimum_cvar_matches_a_two_asset_grid(self):
        scenarios = np.random.default_rng(7).normal([.001, .0005], [.02, .008], size=(250, 2))
        result = construction.minimum_cvar(scenarios.tolist(), confidence=.95)
        grid = [tail_loss(-(scenarios @ np.array([w, 1-w])), .95) for w in np.linspace(0, 1, 2001)]
        self.assertLessEqual(result["cvar"], min(grid)+1e-12)
        self.assertAlmostEqual(result["cvar"], min(grid), places=5)
        self.assertAlmostEqual(result["cvar"], tail_loss(-(scenarios @ np.array(result["weights"])), .95), places=10)
        self.assertLess(result["cvar"], result["equal_weight_cvar"])

    def test_minimum_cvar_honours_cap_and_return_floor_or_refuses(self):
        scenarios = np.random.default_rng(8).normal([.002, .001, .0002, .0], [.03, .02, .01, .004], size=(300, 4))
        free = construction.minimum_cvar(scenarios.tolist(), confidence=.9)
        floor = float(scenarios.mean(axis=0) @ free["weights"])+.0004
        bound = construction.minimum_cvar(scenarios.tolist(), confidence=.9, max_weight=.4, minimum_return=floor)
        self.assertLessEqual(max(bound["weights"]), .4+1e-9)
        self.assertGreaterEqual(bound["expected_return"], floor-1e-9)
        self.assertGreaterEqual(bound["cvar"], free["cvar"]-1e-12)
        with self.assertRaisesRegex(ValueError, "minimum_return"):
            construction.minimum_cvar(scenarios.tolist(), minimum_return=1.)
        with self.assertRaises(ValueError):
            construction.minimum_cvar(scenarios.tolist(), max_weight=.2)

    def test_frontier_matches_closed_form_and_an_independent_solver(self):
        mu, cov = np.array([.06, .08, .1]), np.array([[.04, .01, .0], [.01, .05, .012], [.0, .012, .09]])
        result = construction.efficient_frontier(mu.tolist(), cov.tolist(), points=5)
        inverse, ones = np.linalg.inv(cov), np.ones(3)
        interior = 0
        for point in result["points"][1:-1]:
            # The unconstrained two-fund solution is the answer wherever it is long-only.
            system = np.array([[ones @ inverse @ ones, ones @ inverse @ mu], [mu @ inverse @ ones, mu @ inverse @ mu]])
            a, b = np.linalg.solve(system, [1, point["expected_return"]])
            closed = inverse @ (a*ones+b*mu)
            if closed.min() > 0:
                interior += 1
                np.testing.assert_allclose(point["weights"], closed, atol=2e-6)
        self.assertGreaterEqual(interior, 2)
        capped = construction.efficient_frontier(mu.tolist(), cov.tolist(), points=6, max_weight=.5)
        volatility = [row["volatility"] for row in capped["points"]]
        self.assertEqual(volatility, sorted(volatility))
        self.assertAlmostEqual(capped["maximum_return"]["expected_return"], .5*.1+.5*.08)
        for point in capped["points"]:
            self.assertLessEqual(max(point["weights"]), .5+1e-9)
            # With the sum and return fixed, three weights leave one free direction to scan.
            best = math.inf
            for third in np.linspace(0, .5, 50001):
                first, second = np.linalg.solve([[1, 1], mu[:2]], [1-third, point["expected_return"]-mu[2]*third])
                if min(first, second) >= -1e-12 and max(first, second) <= .5+1e-12:
                    w = np.array([first, second, third])
                    best = min(best, math.sqrt(w @ cov @ w))
            self.assertAlmostEqual(point["volatility"], best, places=7)


class RiskTests(unittest.TestCase):
    def test_parametric_components_sum_and_match_quadrature(self):
        cov, w = covariance(4, 11)/252, np.array([3e5, -1e5, 2e5, 5e4])
        mu = np.array([2e-4, 1e-4, 3e-4, 0])
        result = var_models.parametric(w.tolist(), cov.tolist(), mean=mu.tolist(), confidence=.975, horizon=10)
        sigma, drift = math.sqrt(10*w @ cov @ w), 10*mu @ w
        self.assertAlmostEqual(result["var"], -drift+norm.ppf(.975)*sigma, places=6)
        shortfall, _ = quad(lambda u: -drift+norm.ppf(u)*sigma, .975, 1)
        self.assertAlmostEqual(result["expected_shortfall"], shortfall/.025, places=4)
        self.assertAlmostEqual(sum(result["component_var"]), result["var"], places=6)
        self.assertAlmostEqual(sum(result["component_expected_shortfall"]), result["expected_shortfall"], places=6)
        step = 1e-3
        bumped = var_models.parametric((w+np.eye(4)[1]*step).tolist(), cov.tolist(), mean=mu.tolist(),
                                       confidence=.975, horizon=10)
        self.assertAlmostEqual((bumped["var"]-result["var"])/step, result["marginal_var"][1], places=5)

    def test_cornish_fisher_matches_scipy_moments(self):
        r = np.random.default_rng(12).standard_t(5, size=500)*.01
        result = var_models.cornish_fisher(r.tolist(), confidence=.99)
        s, k, z = skew(r), kurtosis(r), norm.ppf(.01)
        adjusted = z+(z*z-1)*s/6+(z**3-3*z)*k/24-(2*z**3-5*z)*s*s/36
        self.assertAlmostEqual(result["skewness"], s, places=12)
        self.assertAlmostEqual(result["excess_kurtosis"], k, places=12)
        self.assertAlmostEqual(result["var"], -(r.mean()+r.std(ddof=1)*adjusted), places=12)
        self.assertTrue(result["quantile_function_monotone"])
        lopsided = np.concatenate([np.full(95, .001), np.linspace(-.2, -.1, 5)])
        self.assertFalse(var_models.cornish_fisher(lopsided.tolist())["quantile_function_monotone"])

    def test_scenario_components_sum_to_the_tail_loss(self):
        r, w = np.random.default_rng(13).normal(size=(173, 5))*.01, np.array([.3, .2, -.1, .4, .2])
        result = var_models.historical_components(r.tolist(), w.tolist(), confidence=.95)
        self.assertAlmostEqual(result["expected_shortfall"], tail_loss(-(r @ w), .95), places=12)
        self.assertAlmostEqual(sum(result["component_expected_shortfall"]), result["expected_shortfall"], places=12)
        self.assertLessEqual(result["var"], result["expected_shortfall"])
        self.assertLessEqual(result["expected_shortfall"], result["worst_loss"])

    def test_backtest_matches_binomial_likelihoods_and_the_basel_table(self):
        for exceptions, zone in ((0, "GREEN"), (4, "GREEN"), (5, "YELLOW"), (9, "YELLOW"), (10, "RED")):
            returns = np.full(250, .001)
            returns[np.linspace(3, 240, exceptions).astype(int)] = -.05
            result = var_models.backtest(returns.tolist(), [.02]*250, confidence=.99)
            self.assertEqual(result["exceptions"], exceptions)
            self.assertEqual(result["basel_style_zone"], zone)
            expected = -2*(binom.logpmf(exceptions, 250, .01)-binom.logpmf(exceptions, 250, exceptions/250))
            self.assertAlmostEqual(result["kupiec_statistic"], expected, places=9)
            self.assertAlmostEqual(result["kupiec_p_value"], chi2.sf(expected, 1), places=9)

    def test_clustered_exceptions_fail_independence(self):
        hits = np.zeros(200, dtype=bool)
        hits[[20, 21, 22, 90, 91, 150, 151, 152]] = True
        result = var_models.backtest(np.where(hits, -.05, .001).tolist(), [.02]*200, confidence=.95)
        n00, n01, n10, n11 = (result["transitions"][key] for key in ("n00", "n01", "n10", "n11"))
        self.assertEqual((n00, n01, n10, n11), (188, 3, 3, 5))
        pi, pi01, pi11 = 8/199, 3/191, 5/8
        expected = -2*((n00+n10)*math.log(1-pi)+(n01+n11)*math.log(pi)-n00*math.log(1-pi01)-n01*math.log(pi01)
                       - n10*math.log(1-pi11)-n11*math.log(pi11))
        self.assertAlmostEqual(result["independence_statistic"], expected, places=9)
        self.assertLess(result["independence_p_value"], .001)
        spread = np.zeros(200, dtype=bool)
        spread[::25] = True
        self.assertGreater(var_models.backtest(np.where(spread, -.05, .001).tolist(), [.02]*200,
                                               confidence=.95)["independence_p_value"], .3)

    def test_monte_carlo_is_seeded_and_agrees_with_the_normal_formula(self):
        cov, w = covariance(3, 14), [.5, .3, .2]
        first = var_models.monte_carlo(w, cov.tolist(), confidence=.99, paths=200000, seed=3)
        self.assertEqual(first, var_models.monte_carlo(w, cov.tolist(), confidence=.99, paths=200000, seed=3))
        self.assertAlmostEqual(first["var"]/first["analytic_var"], 1, delta=.02)
        self.assertAlmostEqual(first["expected_shortfall"]/first["analytic_expected_shortfall"], 1, delta=.02)
        self.assertAlmostEqual(sum(first["component_expected_shortfall"]), first["expected_shortfall"], places=10)
        heavy = var_models.monte_carlo(w, cov.tolist(), confidence=.99, paths=200000, seed=3, degrees_of_freedom=4)
        self.assertGreater(heavy["expected_shortfall"], 1.1*first["expected_shortfall"])

    def test_inputs_are_refused_not_coerced(self):
        cov = covariance(3, 15).tolist()
        for arguments in (([1, 2], cov), ([0, 0, 0], cov), ([True, 1, 1], cov), ([1, 1, float("nan")], cov),
                          ([1, 1, 1], [[1, 2, 0], [2, 1, 0], [0, 0, 1]]), ([1, 1, 1], [[1, .1, 0], [.2, 1, 0], [0, 0, 1]])):
            with self.assertRaises(ValueError):
                var_models.parametric(*arguments)
        with self.assertRaises(ValueError):
            var_models.parametric([1, 1, 1], cov, confidence=1.)
        with self.assertRaises(ValueError):
            var_models.backtest([.01]*30, [.02]*29)
        with self.assertRaises(ValueError):
            var_models.backtest([.01]*30, [-.02]*30)


class ExecutionScheduleTests(unittest.TestCase):
    ARGS = dict(volatility=.95, temporary_impact=2.5e-6, permanent_impact=2.5e-7, fixed_cost=.0625)

    def test_trajectory_solves_the_first_order_conditions(self):
        for aversion, intervals in ((2e-6, 5), (1e-5, 20), (1e-7, 50), (5e-3, 30)):
            result = schedule.almgren_chriss(1e6, 5, intervals, risk_aversion=aversion, **self.ARGS)
            tau = 5/intervals
            adjusted = 2.5e-6-.5*2.5e-7*tau
            system = np.zeros((intervals-1, intervals-1))
            np.fill_diagonal(system, 2+aversion*.95**2/adjusted*tau**2)
            system[np.arange(intervals-2), np.arange(1, intervals-1)] = -1
            system[np.arange(1, intervals-1), np.arange(intervals-2)] = -1
            right = np.zeros(intervals-1)
            right[0] = 1e6
            np.testing.assert_allclose(result["holdings"][1:-1], np.linalg.solve(system, right), rtol=1e-9, atol=1e-6)
            self.assertAlmostEqual(sum(result["trades"]), 1e6, places=3)
            self.assertEqual(result["holdings"][-1], 0)

    def test_no_schedule_beats_the_optimum_and_risk_neutral_is_uniform(self):
        result = schedule.almgren_chriss(1e6, 5, 10, risk_aversion=3e-6, **self.ARGS)
        rng = np.random.default_rng(21)
        for _ in range(200):
            trades = np.array(result["trades"])+rng.normal(scale=5e3, size=10)
            trades += (1e6-trades.sum())/10
            expected, variance = schedule._moments(trades, .5, .95, 2.5e-7, .0625, 2.5e-6-.5*2.5e-7*.5)
            if trades.min() > 0:
                self.assertGreaterEqual(expected+3e-6*variance, result["objective"]-1e-6)
        neutral = schedule.almgren_chriss(1e6, 5, 10, **self.ARGS)
        np.testing.assert_allclose(neutral["trades"], [1e5]*10)
        self.assertAlmostEqual(neutral["expected_cost"], neutral["uniform_expected_cost"])
        self.assertGreater(result["expected_cost"], neutral["expected_cost"])
        self.assertLess(result["cost_variance"], neutral["cost_variance"])

    def test_frontier_trades_cost_for_variance(self):
        rows = schedule.almgren_chriss(1e6, 5, 20, frontier=[0, 1e-7, 1e-6, 1e-5, 1e-4], **self.ARGS)["frontier"]
        self.assertEqual([row["expected_cost"] for row in rows], sorted(row["expected_cost"] for row in rows))
        self.assertEqual([row["cost_variance"] for row in rows], sorted((row["cost_variance"] for row in rows), reverse=True))

    def test_simulation_reproduces_analytic_moments(self):
        trades = schedule.almgren_chriss(1e6, 5, 10, risk_aversion=3e-6, **self.ARGS)["trades"]
        quiet = schedule.simulate_shortfall(trades, .5, **{**self.ARGS, "volatility": 0}, paths=100)
        self.assertAlmostEqual(quiet["mean"], quiet["analytic_expected_cost"], places=4)
        self.assertAlmostEqual(quiet["standard_deviation"], 0, places=6)
        noisy = schedule.simulate_shortfall(trades, .5, paths=200000, seed=9, **self.ARGS)
        self.assertLess(abs(noisy["mean"]-noisy["analytic_expected_cost"]), 4*noisy["standard_error"])
        self.assertAlmostEqual(noisy["standard_deviation"]/noisy["analytic_standard_deviation"], 1, delta=.01)
        self.assertEqual(noisy, schedule.simulate_shortfall(trades, .5, paths=200000, seed=9, **self.ARGS))

    def test_participation_slices_are_whole_capped_and_complete(self):
        volumes = [120000, 80000, 0, 60000, 70001, 150000]
        result = schedule.participation_schedule(30000, volumes, max_participation=.1)
        self.assertEqual(sum(result["slices"]), 30000)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["slices"][2], 0)
        for size, volume in zip(result["slices"], volumes, strict=True):
            self.assertLessEqual(size, .1*volume)
            self.assertLessEqual(abs(size-30000*volume/sum(volumes)), 1)
        short = schedule.participation_schedule(60000, volumes, max_participation=.1)
        self.assertEqual(short["scheduled_shares"], sum(int(.1*v) for v in volumes))
        self.assertEqual(short["unfilled_shares"], 60000-short["scheduled_shares"])
        self.assertFalse(short["feasible"])

    def test_unsupported_impact_is_refused(self):
        with self.assertRaisesRegex(ValueError, "half the permanent impact"):
            schedule.almgren_chriss(1e6, 5, 5, volatility=1, temporary_impact=1e-7, permanent_impact=1e-6)
        with self.assertRaises(ValueError):
            schedule.almgren_chriss(-1, 5, 5, volatility=1, temporary_impact=1e-6)
        with self.assertRaises(ValueError):
            schedule.participation_schedule(10.5, [100, 100])


class OptionModelTests(unittest.TestCase):
    def test_american_put_matches_a_plain_recursion_and_a_textbook_tree(self):
        def recurse(s, k, r, q, vol, t, call, n):
            dt = t/n
            u = math.exp(vol*math.sqrt(dt))
            p = (math.exp((r-q)*dt)-1/u)/(u-1/u)

            def value(step, ups):
                price = s*u**(2*ups-step)
                exercise = max(price-k if call else k-price, 0)
                if step == n:
                    return exercise
                return max(exercise, math.exp(-r*dt)*(p*value(step+1, ups+1)+(1-p)*value(step+1, ups)))
            return value(0, 0)

        for arguments in ((50, 50, .1, 0, .4, 5/12, False, 5), (100, 95, .03, .06, .3, 1, True, 9),
                          (80, 100, .08, .01, .25, .5, False, 12)):
            result = models.american_binomial(*arguments[:7], steps=arguments[7])
            self.assertAlmostEqual(result["price"], recurse(*arguments), places=10)
        # Hull, Options, Futures, and Other Derivatives: five-step American put, 4.49.
        self.assertAlmostEqual(models.american_binomial(50, 50, .1, 0, .4, 5/12, False, steps=5)["price"], 4.49, places=2)

    def test_early_exercise_is_worthless_for_a_call_without_dividends_and_valuable_for_a_put(self):
        call = models.american_binomial(100, 100, .05, 0, .2, 1, True, steps=500)
        self.assertAlmostEqual(call["early_exercise_premium"], 0, places=12)
        self.assertEqual(call["exercise_boundary_nodes"], 0)
        self.assertAlmostEqual(call["price"], call["black_scholes_price"], delta=.01)
        self.assertAlmostEqual(call["delta"], black_scholes(100, 100, .05, 0, .2, 1, True)["delta"], delta=.005)
        put = models.american_binomial(100, 100, .05, 0, .2, 1, False, steps=500)
        self.assertGreater(put["early_exercise_premium"], .3)
        boundary = [row["critical_price"] for row in put["exercise_boundary"]]
        self.assertTrue(all(price < 100 for price in boundary))
        self.assertGreater(boundary[-1], boundary[0])

    def test_put_call_symmetry_holds_on_the_tree(self):
        call = models.american_binomial(100, 120, .07, .03, .3, 1.5, True, steps=300)["price"]
        put = models.american_binomial(120, 100, .03, .07, .3, 1.5, False, steps=300)["price"]
        self.assertAlmostEqual(call, put, places=8)

    def test_higher_order_greeks_match_finite_differences(self):
        for call in (True, False):
            base = dict(spot=100., strike=108., rate=.03, div=.012, vol=.24, ttm=.6, call=call)
            result = models.higher_order_greeks(**base)

            def bump(field, key, step, base=base):
                up, down = black_scholes(**{**base, field: base[field]+step}), black_scholes(**{**base, field: base[field]-step})
                return (up[key]-down[key])/(2*step)
            self.assertAlmostEqual(result["vanna"], bump("vol", "delta", 1e-5), places=6)
            self.assertAlmostEqual(result["volga"], 100*bump("vol", "vega", 1e-5), places=4)
            self.assertAlmostEqual(result["charm"], -bump("ttm", "delta", 1e-5), places=6)
            self.assertAlmostEqual(result["speed"], bump("spot", "gamma", 1e-3), places=8)
            self.assertAlmostEqual(result["zomma"], bump("vol", "gamma", 1e-5), places=6)
            self.assertAlmostEqual(result["vega"], 100*black_scholes(**base)["vega"], places=9)

    def test_svi_fit_recovers_generating_parameters(self):
        truth = dict(a=.02, b=.1, rho=-.4, m=.05, sigma=.15)
        k = np.linspace(-.5, .5, 21)
        result = models.fit_svi(k.tolist(), models.svi_total_variance(k, **truth).tolist())
        self.assertLess(result["maximum_absolute_error"], 1e-9)
        for key, value in truth.items():
            self.assertAlmostEqual(result["parameters"][key], value, places=5)
        self.assertTrue(result["diagnostics"]["butterfly_arbitrage_free_on_grid"])
        self.assertTrue(result["diagnostics"]["wing_slope_within_lee_bound"])

    def test_svi_diagnostics_separate_a_valid_density_from_butterfly_arbitrage(self):
        flat = models.svi_diagnostics(dict(a=.04, b=0., rho=0., m=0., sigma=.1))
        self.assertAlmostEqual(flat["density_mass_on_grid"], 1, places=9)
        self.assertAlmostEqual(flat["minimum_butterfly_function"], 0, delta=1)
        self.assertTrue(flat["butterfly_arbitrage_free_on_grid"])
        peak = max(flat["density"], key=lambda row: row["density"])
        self.assertAlmostEqual(peak["log_moneyness"], -.02, delta=.08)
        # Gatheral and Jacquier's published example of a slice with butterfly arbitrage.
        bad = models.svi_diagnostics(dict(a=-.0410, b=.1331, rho=.3060, m=.3586, sigma=.4153))
        self.assertLess(bad["minimum_butterfly_function"], 0)
        self.assertFalse(bad["butterfly_arbitrage_free_on_grid"])
        self.assertFalse(models.svi_diagnostics(dict(a=-.05, b=.1, rho=0., m=0., sigma=.1))["butterfly_arbitrage_free_on_grid"])

    def test_variance_swap_recovers_a_flat_volatility(self):
        strikes = np.arange(20., 400.5, 1.)
        forward = 100*math.exp(.03*.5)
        puts = [black_scholes(100, k, .03, 0, .2, .5, False)["price"] for k in strikes]
        calls = [black_scholes(100, k, .03, 0, .2, .5, True)["price"] for k in strikes]
        result = models.variance_swap_strike(strikes.tolist(), puts, calls, forward, .03, .5)
        self.assertAlmostEqual(result["fair_volatility"], .2, delta=2e-4)
        self.assertLess(result["lowest_strike_share"]+result["highest_strike_share"], 1e-6)
        narrow = models.variance_swap_strike(strikes[70:95].tolist(), puts[70:95], calls[70:95], forward, .03, .5)
        self.assertLess(narrow["fair_volatility"], .2)
        self.assertGreater(narrow["lowest_strike_share"]+narrow["highest_strike_share"], .01)
        with self.assertRaisesRegex(ValueError, "bracket the forward"):
            models.variance_swap_strike(strikes[:50].tolist(), puts[:50], calls[:50], forward, .03, .5)


class ResearchStatisticsTests(unittest.TestCase):
    def test_overfitting_probability_matches_a_direct_enumeration(self):
        m = np.random.default_rng(31).normal(.0005, .01, size=(120, 7))
        result = overfitting.probability_of_backtest_overfitting(m.tolist(), partitions=6)
        blocks, logits = np.array_split(np.arange(120), 6), []
        for inside in combinations(range(6), 3):
            train = np.concatenate([blocks[i] for i in inside])
            test = np.concatenate([blocks[i] for i in range(6) if i not in inside])
            sharpe = lambda rows: m[rows].mean(axis=0)/m[rows].std(axis=0, ddof=1)  # noqa: E731
            winner = int(np.argmax(sharpe(train)))
            rank = 1+int(np.sum(sharpe(test) < sharpe(test)[winner]))
            logits.append(math.log(rank/8/(1-rank/8)))
        self.assertEqual(result["splits"], 20)
        self.assertAlmostEqual(result["probability_of_backtest_overfitting"], float(np.mean(np.array(logits) <= 0)))
        self.assertAlmostEqual(result["median_logit"], float(np.median(logits)), places=12)

    def test_a_dominant_trial_is_not_flagged_and_noise_is(self):
        rng = np.random.default_rng(32)
        noise = rng.normal(0, .01, size=(400, 30))
        self.assertGreater(overfitting.probability_of_backtest_overfitting(noise.tolist(), partitions=10)
                           ["probability_of_backtest_overfitting"], .25)
        noise[:, 4] += .01
        result = overfitting.probability_of_backtest_overfitting(noise.tolist(), partitions=10)
        self.assertEqual(result["probability_of_backtest_overfitting"], 0)
        self.assertEqual(result["out_of_sample_loss_probability"], 0)
        with self.assertRaisesRegex(ValueError, "undefined"):
            overfitting.probability_of_backtest_overfitting(np.zeros((40, 3)).tolist(), partitions=4)

    def test_track_record_length_inverts_the_probabilistic_sharpe_ratio(self):
        for sharpe, benchmark, g3, g4, level in ((.1, 0, -.5, 5, .95), (.25, .1, 0, 3, .99), (.05, 0, .3, 9, .9)):
            result = overfitting.minimum_track_record_length(sharpe, benchmark=benchmark, skew=g3, kurtosis=g4,
                                                             confidence=level)
            n = result["observations"]
            sr_error = math.sqrt((1-g3*sharpe+(g4-1)/4*sharpe**2)/(n-1))
            self.assertAlmostEqual(norm.cdf((sharpe-benchmark)/sr_error), level, places=10)
            self.assertGreaterEqual(probabilistic_sharpe(sharpe, result["whole_observations"], g3, g4, benchmark), level)
        with self.assertRaisesRegex(ValueError, "exceed the benchmark"):
            overfitting.minimum_track_record_length(.1, benchmark=.1)

    def test_fama_macbeth_matches_statsmodels(self):
        try:
            import statsmodels.api as sm
        except ImportError:
            self.skipTest("statsmodels is not installed")
        rng = np.random.default_rng(33)
        x = rng.normal(size=(60, 25, 2))
        premia = np.array([.002, -.001])+.3*np.cumsum(rng.normal(scale=.001, size=(60, 2)), axis=0)
        r = .001+np.einsum("tnk,tk->tn", x, premia)+rng.normal(scale=.02, size=(60, 25))
        result = cross_section.fama_macbeth(r.tolist(), x.tolist(), lags=3)
        slopes = np.array([sm.OLS(r[t], sm.add_constant(x[t])).fit().params for t in range(60)])
        np.testing.assert_allclose(result["period_premia"], slopes, rtol=1e-9, atol=1e-12)
        for index, row in enumerate(result["premia"]):
            fit = sm.OLS(slopes[:, index], np.ones(60)).fit(cov_type="HAC", cov_kwds={"maxlags": 3, "use_correction": False})
            self.assertAlmostEqual(row["premium"], fit.params[0], places=12)
            self.assertAlmostEqual(row["newey_west_standard_error"], fit.bse[0], places=12)
            self.assertAlmostEqual(row["fama_macbeth_standard_error"], sm.OLS(slopes[:, index], np.ones(60)).fit().bse[0], places=12)
        self.assertEqual(cross_section.fama_macbeth(r.tolist(), x.tolist())["newey_west_lags"], 3)
        x[5, :, 1] = 2*x[5, :, 0]
        with self.assertRaisesRegex(ValueError, "collinear"):
            cross_section.fama_macbeth(r.tolist(), x.tolist())


class BarTests(unittest.TestCase):
    def trades(self):
        rng = np.random.default_rng(41)
        stamps = np.cumsum(rng.integers(1, 40, size=300))*1_000_000_000
        return [{"timestamp_ns": int(t), "price": float(p), "size": float(s)}
                for t, p, s in zip(stamps, 100+np.cumsum(rng.normal(0, .05, 300)), rng.integers(1, 500, 300), strict=True)]

    def test_time_bars_match_pandas(self):
        import pandas as pd

        trades = self.trades()
        result = bars.aggregate_trades(trades, kind="TIME", threshold=300_000_000_000)
        frame = pd.DataFrame(trades)
        frame["bucket"] = frame["timestamp_ns"]//300_000_000_000
        frame["notional"] = frame["price"]*frame["size"]
        expected = frame.groupby("bucket").agg(open=("price", "first"), high=("price", "max"), low=("price", "min"),
                                               close=("price", "last"), volume=("size", "sum"),
                                               notional=("notional", "sum"), trades=("price", "size"))
        self.assertEqual(len(result["bars"]), len(expected))
        for bar, (bucket, row) in zip(result["bars"], expected.iterrows(), strict=True):
            self.assertEqual(bar["interval_start_ns"], bucket*300_000_000_000)
            for key in ("open", "high", "low", "close", "volume", "notional", "trades"):
                self.assertAlmostEqual(bar[key], row[key], places=6)
            self.assertAlmostEqual(bar["vwap"], row["notional"]/row["volume"], places=9)
        self.assertEqual([bar["complete"] for bar in result["bars"]], [True]*(len(expected)-1)+[False])
        self.assertEqual(result["empty_intervals"], expected.index[-1]-expected.index[0]+1-len(expected))

    def test_information_bars_conserve_prints_and_close_at_their_threshold(self):
        trades = self.trades()
        total = sum(row["size"] for row in trades)
        for kind, threshold, key in (("TICK", 25, "trades"), ("VOLUME", 4000., "volume"), ("DOLLAR", 400000., "notional")):
            result = bars.aggregate_trades(trades, kind=kind, threshold=threshold)
            self.assertEqual(sum(bar["trades"] for bar in result["bars"]), 300)
            self.assertAlmostEqual(sum(bar["volume"] for bar in result["bars"]), total)
            for bar in result["bars"]:
                self.assertEqual(bar["complete"], bar[key] >= threshold)
                self.assertLessEqual(bar["low"], bar["vwap"])
                self.assertLessEqual(bar["vwap"], bar["high"])
            self.assertTrue(all(bar["complete"] for bar in result["bars"][:-1]))
        self.assertEqual([bar["trades"] for bar in bars.aggregate_trades(trades, kind="TICK", threshold=25)["bars"]], [25]*12)

    def test_malformed_prints_are_refused(self):
        trades = self.trades()
        for broken in (trades[::-1], [{**trades[0], "price": 0}], [{**trades[0], "size": True}],
                       [{"timestamp_ns": 1.5, "price": 1, "size": 1}], [{**trades[0], "venue": "X"}], []):
            with self.assertRaises(ValueError):
                bars.aggregate_trades(broken, kind="TICK", threshold=5)
        with self.assertRaises(ValueError):
            bars.aggregate_trades(trades, kind="RANGE", threshold=5)
        with self.assertRaises(ValueError):
            bars.aggregate_trades(trades, kind="TICK", threshold=2.5)

    def test_quality_report_names_each_defect_once(self):
        day = 86_400_000_000_000
        rng = np.random.default_rng(42)
        close = 100*np.exp(np.cumsum(rng.normal(0, .01, 60)))
        rows = [{"timestamp_ns": day*(i+1), "open": float(c), "high": float(c*1.01), "low": float(c*.99),
                 "close": float(c), "volume": 1000.} for i, c in enumerate(close)]
        clean = bars.quality_report(rows)
        self.assertEqual((clean["errors"], clean["warnings"], clean["structurally_usable"]), (0, 0, True))
        rows[5]["high"] = rows[5]["low"]*.9
        rows[9]["timestamp_ns"] = rows[8]["timestamp_ns"]
        rows[14]["timestamp_ns"] = rows[12]["timestamp_ns"]-1
        rows[20]["volume"] = 0.
        rows[21]["volume"] = -5.
        rows[25]["low"] = 0.
        for index in range(30, 37):
            rows[index].update(open=rows[30]["close"], high=rows[30]["close"]*1.01, low=rows[30]["close"]*.99,
                               close=rows[30]["close"])
        for index in range(45, 60):
            rows[index]["timestamp_ns"] += 9*day
        for key in ("open", "high", "low", "close"):
            rows[50][key] *= 1.6
        report = bars.quality_report(rows)
        found = {(row["index"], row["code"]) for row in report["findings"]}
        for expected in ((5, "OHLC_INCONSISTENT"), (9, "DUPLICATE_TIMESTAMP"), (14, "OUT_OF_ORDER"), (20, "ZERO_VOLUME"),
                         (21, "NEGATIVE_VOLUME"), (25, "NONPOSITIVE_PRICE"), (30, "STALE_CLOSE_RUN"), (45, "GAP"),
                         (50, "RETURN_OUTLIER"), (51, "RETURN_OUTLIER")):
            self.assertIn(expected, found)
        self.assertFalse(report["structurally_usable"])
        self.assertEqual(report["counts"]["STALE_CLOSE_RUN"], 1)
        self.assertEqual(report["errors"], sum(row["severity"] == "ERROR" for row in report["findings"]))


class RegistryHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async def clock():
            while True:
                await asyncio.sleep(.01)
        self.clock = asyncio.create_task(clock())
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.clock.cancel()

    async def post(self, kind, payload):
        return await self.client.post(f"/api/v1/analytics/{kind}", content=json.dumps(payload),
                                      headers={"content-type": "application/json"})

    async def test_every_registered_example_is_served_as_research_only(self):
        self.assertGreaterEqual(len(ANALYSES), 21)
        for key, analysis in ANALYSES.items():
            with self.subTest(key):
                response = await self.post(key, analysis.example)
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertEqual(body["state"], "AVAILABLE")
                self.assertEqual(body["provenance"]["classification"], "MODEL_ESTIMATE")
                self.assertEqual(body["provenance"]["source"], "user-supplied-research-inputs")
                self.assertEqual(body["provenance"]["readiness"], "RESEARCH_ONLY")
                self.assertEqual([row["label"] for row in body["data"]["headline"]], [label for label, _ in analysis.headline])
                self.assertEqual(body["data"], json.loads(json.dumps(run(key, analysis.example))))
                self.assertIn("no order, allocation or trading approval", analysis.scope)

    async def test_requests_outside_the_contract_are_rejected(self):
        for key, analysis in ANALYSES.items():
            with self.subTest(key):
                self.assertEqual((await self.post(key, {**analysis.example, "broker": "live"})).status_code, 422)
                missing = dict(analysis.example)
                missing.pop(analysis.required[0])
                self.assertEqual((await self.post(key, missing)).status_code, 422)
                self.assertEqual((await self.post(key, {**analysis.example, analysis.required[0]: True})).status_code, 422)
                self.assertEqual((await self.post(key, {**analysis.example, analysis.required[0]: "1"})).status_code, 422)
        self.assertEqual((await self.post("not-an-analysis", {})).status_code, 422)
        oversized = {"returns": [[.01]*65]*3}
        self.assertEqual((await self.post("ledoit-wolf", oversized)).status_code, 422)


if __name__ == "__main__":
    unittest.main()
