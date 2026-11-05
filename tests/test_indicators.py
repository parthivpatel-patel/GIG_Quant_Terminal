import numpy as np
import pandas as pd
import pytest

from gig.research import indicators as ind

CLOSE = np.array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03,
                  45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64, 46.21, 46.25, 45.71, 46.45,
                  45.78, 45.35, 44.03, 44.18, 44.22, 44.57, 43.42, 42.66, 43.13])


def _bars(n=120, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .01, n)))
    open_ = np.concatenate(([100.0], close[:-1])) * np.exp(rng.normal(0, .002, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, .004, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, .004, n)))
    volume = rng.integers(1_000, 50_000, n).astype(float)
    return open_, high, low, close, volume


def test_sma_matches_pandas_and_has_no_lookahead():
    x = CLOSE
    np.testing.assert_allclose(ind.sma(x, 5), pd.Series(x).rolling(5).mean().to_numpy(), equal_nan=True)
    assert np.isnan(ind.sma(x, 5)[:4]).all()
    changed = x.copy()
    changed[20:] *= 3
    np.testing.assert_array_equal(ind.sma(changed, 5)[:20], ind.sma(x, 5)[:20])


def test_ema_is_seeded_with_simple_mean():
    out = ind.ema(CLOSE, 10)
    assert np.isnan(out[:9]).all()
    assert out[9] == pytest.approx(CLOSE[:10].mean())
    alpha = 2 / 11
    assert out[10] == pytest.approx(out[9] + alpha * (CLOSE[10] - out[9]))


def test_rsi_matches_an_independent_wilder_loop():
    # The often-quoted 70.53 for this series uses averages rounded to two
    # decimals; unrounded Wilder smoothing gives 70.464.
    out = ind.rsi(CLOSE, 14)
    change = np.diff(CLOSE)
    gain, loss = np.clip(change[:14], 0, None).mean(), np.clip(-change[:14], 0, None).mean()
    expected = [100 - 100 / (1 + gain / loss)]
    for c in change[14:]:
        gain = (gain * 13 + max(c, 0)) / 14
        loss = (loss * 13 + max(-c, 0)) / 14
        expected.append(100 - 100 / (1 + gain / loss))
    assert np.isnan(out[:14]).all()
    np.testing.assert_allclose(out[14:], expected)
    assert out[14] == pytest.approx(70.464, abs=0.001)


def test_rsi_flat_and_monotone_edges():
    assert ind.rsi(np.full(20, 5.0), 14)[-1] == 50
    assert ind.rsi(np.arange(1.0, 21.0), 14)[-1] == 100
    assert ind.rsi(np.arange(20.0, 0.0, -1), 14)[-1] == 0


def test_macd_lines_and_warmup():
    _, _, _, close, _ = _bars()
    out = ind.macd(close)
    np.testing.assert_allclose(out["macd"], ind.ema(close, 12) - ind.ema(close, 26), equal_nan=True)
    assert np.isnan(out["macd"][:25]).all() and not np.isnan(out["macd"][25])
    assert np.isnan(out["signal"][:33]).all() and out["signal"][33] == pytest.approx(out["macd"][25:34].mean())
    np.testing.assert_allclose(out["histogram"], out["macd"] - out["signal"], equal_nan=True)
    with pytest.raises(ValueError):
        ind.macd(close, fast=26, slow=12)


def test_bollinger_uses_population_std():
    out = ind.bollinger(CLOSE, 20, 2)
    window = CLOSE[:20]
    assert out["middle"][19] == pytest.approx(window.mean())
    assert out["upper"][19] == pytest.approx(window.mean() + 2 * window.std(ddof=0))
    assert out["lower"][19] == pytest.approx(window.mean() - 2 * window.std(ddof=0))
    assert np.isnan(out["upper"][:19]).all()


def test_true_range_and_atr():
    high = np.array([10.0, 11.0, 12.0, 11.5, 13.0])
    low = np.array([9.0, 10.0, 10.5, 10.0, 11.0])
    close = np.array([9.5, 10.5, 11.8, 10.2, 12.9])
    tr = ind.true_range(high, low, close)
    assert np.isnan(tr[0])
    np.testing.assert_allclose(tr[1:], [1.5, 1.5, 1.8, 2.8])
    out = ind.atr(high, low, close, 2)
    assert np.isnan(out[:2]).all()
    assert out[2] == pytest.approx(1.5)
    assert out[3] == pytest.approx(1.5 + (1.8 - 1.5) / 2)
    assert out[4] == pytest.approx(out[3] + (2.8 - out[3]) / 2)
    with pytest.raises(ValueError):
        ind.true_range([1.0], [2.0], [1.5])


def test_rolling_vwap_weights_typical_price_by_volume():
    _, high, low, close, volume = _bars()
    out = ind.vwap(high, low, close, volume, 20)
    typical = (high + low + close) / 3
    i = 50
    expected = (typical[i - 19:i + 1] * volume[i - 19:i + 1]).sum() / volume[i - 19:i + 1].sum()
    assert out[i] == pytest.approx(expected)
    assert np.isnan(out[:19]).all()
    assert np.isnan(ind.vwap(high[:20], low[:20], close[:20], np.zeros(20), 20)[-1])


def test_realized_volatility_annualized_sample_std():
    _, _, _, close, _ = _bars()
    out = ind.realized_volatility(close, 20)
    returns = np.diff(np.log(close))
    assert np.isnan(out[:20]).all()
    assert out[20] == pytest.approx(returns[:20].std(ddof=1) * np.sqrt(252))
    assert out[-1] == pytest.approx(returns[-20:].std(ddof=1) * np.sqrt(252))


@pytest.mark.parametrize("call", [
    lambda: ind.sma([1.0, np.nan, 3.0], 2),
    lambda: ind.sma([1.0, 2.0], 0),
    lambda: ind.sma([[1.0, 2.0]], 1),
    lambda: ind.bollinger([1.0, 2.0], 2, -1),
    lambda: ind.realized_volatility([1.0, -2.0, 3.0], 2),
    lambda: ind.vwap([1.0], [1.0], [1.0], [-1.0], 1),
])
def test_invalid_inputs_fail_closed(call):
    with pytest.raises(ValueError):
        call()


def test_short_series_return_all_nan():
    for out in (ind.sma([1.0, 2.0], 5), ind.ema([1.0], 3), ind.rsi([1.0, 2.0], 14),
                ind.atr([2.0, 3.0], [1.0, 2.0], [1.5, 2.5], 14), ind.realized_volatility([1.0, 2.0], 5)):
        assert np.isnan(out).all()
