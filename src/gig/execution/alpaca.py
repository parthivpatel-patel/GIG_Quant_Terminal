"""
Alpaca adapter. Imported only when gig[broker] is installed.

Paper and live share one API surface at Alpaca, distinguished only by the base
URL, which makes an accidental live connection a one-character mistake. So the
constructor refuses to build unless the URL says ``paper``, and ``allow_live``
has to be passed explicitly to get anything else. Guarding at construction
rather than at submit means the wrong environment fails before a target book is
even computed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from gig.config import get_settings
from gig.execution.base import Broker
from gig.types import Fill, Side


class AlpacaBroker(Broker):
    def __init__(self, allow_live: bool = False) -> None:
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:
            raise ImportError("Install gig[broker] to use AlpacaBroker") from exc
        cfg = get_settings()
        if not cfg.alpaca_api_key or not cfg.alpaca_secret_key:
            raise RuntimeError("ALPACA keys missing — set GIG_ALPACA_API_KEY / SECRET or ALPACA_* in .env")
        paper = "paper" in cfg.alpaca_base_url
        if not paper and not allow_live:
            raise RuntimeError(
                f"ALPACA_BASE_URL is {cfg.alpaca_base_url!r}, which is not a paper endpoint. "
                "Point it at https://paper-api.alpaca.markets, or pass allow_live=True "
                "if you genuinely intend to trade real money with this."
            )
        self.paper = paper
        self._settings = cfg
        self._client = TradingClient(cfg.alpaca_api_key, cfg.alpaca_secret_key, paper=paper)
        self._asset_cache: dict[str, dict[str, bool]] = {}

    def submit(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        *,
        limit_price: float | None = None,
    ) -> Fill:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        side_enum = OrderSide.BUY if side == "buy" else OrderSide.SELL
        if limit_price is not None:
            req = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=side_enum,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
            )
        else:
            req = MarketOrderRequest(
                symbol=symbol, qty=quantity, side=side_enum, time_in_force=TimeInForce.DAY
            )
        order = self._client.submit_order(req)
        # Alpaca acknowledges asynchronously, so this records the *intent* with
        # the broker's id. Realized price comes from the fills endpoint later.
        return Fill(
            timestamp=datetime.now(UTC),
            symbol=symbol,
            side=side,
            quantity=float(quantity),
            price=float(limit_price or 0.0),
            cost=0.0,
            order_id=str(order.id),
        )

    def positions(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for p in self._client.get_all_positions():
            out[str(p.symbol)] = float(p.qty)
        return out

    def nav(self) -> float:
        acct = self._client.get_account()
        return float(acct.equity)

    def account_summary(self) -> dict[str, float | bool | str]:
        acct = self._client.get_account()
        return {
            "equity": float(acct.equity),
            "last_equity": float(acct.last_equity or 0.0),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "long_market_value": float(acct.long_market_value or 0.0),
            "short_market_value": float(acct.short_market_value or 0.0),
            "multiplier": float(acct.multiplier or 1.0),
            "shorting_enabled": bool(getattr(acct, "shorting_enabled", False)),
            "trading_blocked": bool(getattr(acct, "trading_blocked", False)),
            "pattern_day_trader": bool(getattr(acct, "pattern_day_trader", False)),
            "status": str(getattr(acct, "status", "")),
            "paper": self.paper,
        }

    def is_open(self) -> bool | None:
        try:
            return bool(self._client.get_clock().is_open)
        except Exception:
            return None

    def _assets(self, symbols: list[str]) -> dict[str, dict[str, bool]]:
        missing = [s for s in symbols if s not in self._asset_cache]
        for symbol in missing:
            try:
                asset = self._client.get_asset(symbol)
                self._asset_cache[symbol] = {
                    "tradable": bool(asset.tradable),
                    "shortable": bool(asset.shortable),
                    "easy_to_borrow": bool(asset.easy_to_borrow),
                    "fractionable": bool(asset.fractionable),
                }
            except Exception:
                # An unknown asset is treated as untradable: refusing to trade
                # something the broker will not describe is the safe default.
                self._asset_cache[symbol] = {
                    "tradable": False,
                    "shortable": False,
                    "easy_to_borrow": False,
                    "fractionable": False,
                }
        return {s: self._asset_cache[s] for s in symbols}

    def shortable(self, symbols: list[str]) -> dict[str, bool]:
        """
        Borrow availability, requiring both flags.

        ``shortable`` alone can be true for a hard-to-borrow name where the
        locate fails at submit time, so this also requires ``easy_to_borrow``.
        """
        info = self._assets([str(s) for s in symbols])
        return {s: bool(v["shortable"] and v["easy_to_borrow"]) for s, v in info.items()}

    def tradable(self, symbols: list[str]) -> dict[str, bool]:
        return {s: bool(v["tradable"]) for s, v in self._assets([str(s) for s in symbols]).items()}

    def last_prices(self, symbols: list[str]) -> pd.Series:
        """Latest IEX trade price per symbol. Empty on any failure."""
        cfg = self._settings
        try:
            from gig.data.providers.alpaca_iex import fetch_latest_quotes

            quotes = fetch_latest_quotes(
                [str(s) for s in symbols],
                cfg.alpaca_api_key,
                cfg.alpaca_secret_key,
                cfg.alpaca_data_url,
            )
        except Exception:
            return pd.Series(dtype=float)
        if quotes is None or quotes.empty or "last" not in quotes.columns:
            return pd.Series(dtype=float)
        out = quotes.dropna(subset=["last"]).set_index("symbol")["last"].astype(float)
        return out[~out.index.duplicated(keep="last")]

    def cancel_open_orders(self) -> int:
        try:
            responses = self._client.cancel_orders()
            return len(responses or [])
        except Exception:
            return 0

    def close_all(self, limit_offset_bps: float = 25.0) -> list[Fill]:
        """Use Alpaca's own flatten endpoint, which handles shorts and odd lots."""
        held = self.positions()
        try:
            self._client.close_all_positions(cancel_orders=True)
        except Exception as exc:
            raise RuntimeError(f"close_all_positions failed: {exc}") from exc
        now = datetime.now(UTC)
        return [
            Fill(
                timestamp=now,
                symbol=symbol,
                side="sell" if qty > 0 else "buy",
                quantity=abs(float(qty)),
                price=0.0,
                cost=0.0,
                order_id="CLOSE_ALL",
            )
            for symbol, qty in held.items()
            if qty
        ]
