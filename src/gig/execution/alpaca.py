"""Optional Alpaca paper adapter. Imported only when gig[broker] is installed."""

from __future__ import annotations

from datetime import UTC, datetime

from gig.config import get_settings
from gig.execution.base import Broker
from gig.types import Fill, Side


class AlpacaBroker(Broker):
    def __init__(self) -> None:
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:
            raise ImportError("Install gig[broker] to use AlpacaBroker") from exc
        cfg = get_settings()
        if not cfg.alpaca_api_key or not cfg.alpaca_secret_key:
            raise RuntimeError("ALPACA keys missing — set GIG_ALPACA_API_KEY / SECRET or ALPACA_* in .env")
        paper = "paper" in cfg.alpaca_base_url
        self._client = TradingClient(cfg.alpaca_api_key, cfg.alpaca_secret_key, paper=paper)

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
