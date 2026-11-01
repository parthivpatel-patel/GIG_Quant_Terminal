"""
Alpaca paper adapter. Connectivity is disabled until explicitly configured.

The live-money endpoint is never supported. An exact endpoint check, performed
before loading the SDK or constructing a client, prevents substring spoofing.
"""

from __future__ import annotations

from datetime import UTC, datetime
import math
from urllib.parse import urlsplit
from uuid import uuid4

import pandas as pd

from gig.config import get_settings
from gig.execution.base import Broker
from gig.types import OrderAcknowledgement, Side


class AlpacaBroker(Broker):
    def __init__(self, allow_live: bool = False) -> None:
        cfg = get_settings()
        if allow_live:
            raise RuntimeError("live-money trading is disabled")
        try:
            endpoint = urlsplit(cfg.alpaca_base_url)
            port = endpoint.port
        except ValueError:
            raise RuntimeError("only the approved Alpaca paper endpoint is supported") from None
        if (
            endpoint.scheme != "https"
            or endpoint.hostname != "paper-api.alpaca.markets"
            or port not in (None, 443)
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.path not in ("", "/")
            or endpoint.query
            or endpoint.fragment
        ):
            raise RuntimeError("only the approved Alpaca paper endpoint is supported")
        if not cfg.broker_connectivity_enabled:
            raise RuntimeError("broker connectivity is disabled; explicit paper approval is required")
        try:
            data_endpoint = urlsplit(cfg.alpaca_data_url)
            data_valid = (data_endpoint.scheme == "https" and data_endpoint.hostname == "data.alpaca.markets"
                          and data_endpoint.port in (None, 443) and data_endpoint.username is None
                          and data_endpoint.password is None and data_endpoint.path in ("", "/")
                          and not data_endpoint.query and not data_endpoint.fragment)
        except ValueError:
            data_valid = False
        if not data_valid:
            raise RuntimeError("only the approved Alpaca data endpoint is supported")
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:
            raise ImportError("Install gig[broker] to use AlpacaBroker") from exc
        if not cfg.alpaca_api_key or not cfg.alpaca_secret_key:
            raise RuntimeError("ALPACA keys missing — set GIG_ALPACA_API_KEY / SECRET or ALPACA_* in .env")
        paper = True
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
        client_order_id: str | None = None,
    ) -> OrderAcknowledgement:
        if (not isinstance(symbol, str) or not symbol or len(symbol) > 128
            or any(ord(char) < 32 for char in symbol)):
            raise ValueError("valid instrument identity required")
        if side not in {"buy", "sell"} or isinstance(quantity, bool) or not math.isfinite(quantity) or quantity <= 0:
            raise ValueError("valid side and positive finite quantity required")
        if limit_price is not None and (isinstance(limit_price, bool) or not math.isfinite(limit_price) or limit_price <= 0):
            raise ValueError("limit price must be finite and positive")
        client_order_id = f"gig-{uuid4().hex}" if client_order_id is None else client_order_id
        if (not isinstance(client_order_id, str) or not 1 <= len(client_order_id) <= 48
            or not client_order_id.isascii() or not all(c.isalnum() or c in "-_" for c in client_order_id)):
            raise ValueError("invalid client order id")
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
                client_order_id=client_order_id,
            )
        else:
            req = MarketOrderRequest(
                symbol=symbol, qty=quantity, side=side_enum, time_in_force=TimeInForce.DAY,
                client_order_id=client_order_id,
            )
        order = self._client.submit_order(req)
        if not getattr(order, "id", None):
            raise RuntimeError("paper submission outcome is unknown; reconcile client order id")
        # An acknowledgement supplies neither executed quantity nor price.
        # There is no automatic retry after a network ambiguity: reconcile the
        # same client_order_id before deciding whether to submit again.
        return OrderAcknowledgement(
            timestamp=datetime.now(UTC),
            symbol=symbol,
            side=side,
            requested_quantity=float(quantity),
            order_id=str(order.id),
            client_order_id=client_order_id,
            status=str(getattr(order, "status", "accepted")),
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
            state = self._client.get_clock().is_open
            return state if isinstance(state, bool) else None
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
            if not isinstance(responses, list):
                raise RuntimeError("cancellation outcome unknown")
            if any(not 200 <= int(getattr(response, "status", 0)) < 300 for response in responses):
                raise RuntimeError("cancellation response contains failures")
            return len(responses or [])
        except Exception:
            raise RuntimeError("paper broker cancellation failed") from None

    def close_all(self, limit_offset_bps: float = 25.0) -> list[OrderAcknowledgement]:
        """Use Alpaca's own flatten endpoint, which handles shorts and odd lots."""
        held = self.positions()
        try:
            responses = self._client.close_all_positions(cancel_orders=True)
        except Exception:
            raise RuntimeError("paper broker flatten request failed") from None
        now = datetime.now(UTC)
        acknowledgements = []
        seen = set()
        for response in responses or []:
            symbol = str(getattr(response, "symbol", ""))
            if symbol not in held:
                continue
            qty = held[symbol]
            seen.add(symbol)
            try:
                status = int(getattr(response, "status", 0))
            except (ValueError, TypeError):
                status = 0
            body = getattr(response, "body", None)
            order_id = str(body.get("id", "") if isinstance(body, dict) else getattr(body, "id", ""))
            acknowledgements.append(OrderAcknowledgement(
                timestamp=now,
                symbol=symbol,
                side="sell" if qty > 0 else "buy",
                requested_quantity=abs(float(qty)),
                order_id=order_id,
                client_order_id="",
                status="close_acknowledged" if 200 <= status < 300 else ("close_rejected" if status >= 400 else "close_unknown"),
                classification="BROKER_ACKNOWLEDGED" if 200 <= status < 300 else ("BROKER_REJECTED" if status >= 400 else "UNKNOWN"),
            ))
        for symbol in sorted(held.keys() - seen):
            quantity = held[symbol]
            if quantity:
                acknowledgements.append(OrderAcknowledgement(
                    timestamp=now, symbol=symbol, side="sell" if quantity > 0 else "buy",
                    requested_quantity=abs(float(quantity)), order_id="", client_order_id="",
                    status="close_unknown", classification="UNKNOWN",
                ))
        return acknowledgements
