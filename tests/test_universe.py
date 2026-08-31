from __future__ import annotations

import numpy as np
import pandas as pd

from gig.data.universe import point_in_time_mask


def test_listing_start_is_point_in_time():
    dates = pd.bdate_range("2020-01-02", periods=300)
    close = pd.DataFrame({"AAA": np.linspace(10, 20, 300), "BBB": np.linspace(10, 20, 300)}, index=dates)
    listing = pd.Series({"BBB": dates[200]})
    mask = point_in_time_mask(close, listing_start=listing, min_history=5, min_price=1.0)
    assert not mask.loc[dates[199], "BBB"]
    assert mask.loc[dates[210], "BBB"]
    assert mask.loc[dates[10], "AAA"]


def test_min_history_blocks_short_names():
    dates = pd.bdate_range("2020-01-02", periods=20)
    close = pd.DataFrame({"AAA": np.arange(20) + 5.0}, index=dates)
    mask = point_in_time_mask(close, min_history=50)
    assert not mask.any().any()
