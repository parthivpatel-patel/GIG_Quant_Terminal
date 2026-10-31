"""Data adapters. Research always consumes a MarketPanel — never a live socket."""

from gig.data.providers.base import DataProvider
from gig.data.providers.synthetic import SyntheticProvider
from gig.data.providers.yahoo import YahooProvider
from gig.data.universe import point_in_time_mask

__all__ = ["DataProvider", "SyntheticProvider", "YahooProvider", "point_in_time_mask"]
