from __future__ import annotations

import pandas as pd

from gig.assets.options import black_scholes, implied_vol, rv_iv_signal
from gig.execution.paper import PaperBroker
from gig.risk.limits import check_limits
from gig.risk.var import historical_var_es


def test_put_call_parity():
    s, k, r, q, v, t = 100.0, 100.0, 0.03, 0.01, 0.2, 0.5
    c = black_scholes(s, k, r, q, v, t, call=True)["price"]
    p = black_scholes(s, k, r, q, v, t, call=False)["price"]
    lhs = c - p
    rhs = s * __import__("math").exp(-q * t) - k * __import__("math").exp(-r * t)
    assert abs(lhs - rhs) < 1e-6


def test_implied_vol_roundtrip():
    price = black_scholes(100, 100, 0.02, 0.0, 0.25, 1.0, call=True)["price"]
    iv = implied_vol(price, 100, 100, 0.02, 0.0, 1.0, call=True)
    assert abs(iv - 0.25) < 1e-4


def test_vrp_sign():
    assert rv_iv_signal(0.15, 0.20) > 0
    assert rv_iv_signal(0.25, 0.20) < 0


def test_paper_broker_nav_conserved_on_round_trip():
    prices = {"AAA": 10.0}
    broker = PaperBroker(lambda s: prices[s], nav0=1000.0)
    broker.submit("AAA", "buy", 10)
    assert abs(broker.nav() - 1000.0) < 1e-8
    broker.submit("AAA", "sell", 10)
    assert abs(broker.nav() - 1000.0) < 1e-8
    assert broker.positions()["AAA"] == 0.0


def test_limits_fire_on_gross():
    w = pd.Series({"A": 0.8, "B": -0.8, "C": 0.6})
    sectors = pd.Series({"A": "tech", "B": "tech", "C": "health"})
    breaches = check_limits(w, sectors, max_gross=1.5, max_net=0.5, max_name=0.5, max_sector_net=1.0)
    names = {b.name for b in breaches}
    assert "gross_leverage" in names


def test_var_es_ordering():
    r = pd.Series(__import__("numpy").random.default_rng(0).normal(0, 0.01, 500))
    out = historical_var_es(r, 0.99)
    assert out["es"] <= out["var"]
