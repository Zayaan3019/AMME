"""api-gateway-svc: REST + WebSocket surface over the live pipeline.

Read-only by design. The gateway tails the event streams and serves the current
state; it cannot place a quote, amend risk limits, or trip the breaker. A
dashboard that can mutate trading state is a dashboard that can be used to
bypass risk, and the whole point of the risk boundary is that there is no way
around it.

It maintains its own view by consuming the same streams everything else does,
in a dedicated consumer group so it never competes with the trading path for
messages.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from amme_shared.config import Settings
from amme_shared.events import (
    BaseEvent,
    Fill,
    OrderBookUpdate,
    PortfolioState,
    QuoteIntent,
    RiskDecision,
)
from amme_shared.service import StreamService
from amme_shared.streams import RedisEventBus, Streams

STATIC_DIR = Path(__file__).parent / "static"
HISTORY = 600  # ~10 minutes of 1Hz samples


class LiveState:
    """Rolling snapshot of the pipeline, fed by the stream tail."""

    def __init__(self) -> None:
        self.mid_price = 0.0
        self.spread = 0.0
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.inventory = 0.0
        self.cash = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_pnl = 0.0
        self.n_fills = 0
        self.n_quotes = 0
        self.quote_bid = 0.0
        self.quote_ask = 0.0
        self.reservation_price = 0.0
        self.half_spread = 0.0
        self.toxicity_multiplier = 1.0
        self.drift_adjustment = 0.0
        self.sigma = 0.0
        self.vpin = 0.0
        self.ofi_zscore = 0.0
        self.risk_action = "none"
        self.risk_reason = ""
        self.breaker_tripped = False
        self.inventory_utilisation = 0.0
        self.last_update_ns = 0
        self.pnl_history: deque[dict[str, float]] = deque(maxlen=HISTORY)
        self.recent_fills: deque[dict[str, Any]] = deque(maxlen=50)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mid_price": self.mid_price,
            "spread": self.spread,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "inventory": self.inventory,
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_pnl,
            "n_fills": self.n_fills,
            "n_quotes": self.n_quotes,
            "quote_bid": self.quote_bid,
            "quote_ask": self.quote_ask,
            "quoted_spread": max(0.0, self.quote_ask - self.quote_bid),
            "reservation_price": self.reservation_price,
            "reservation_offset": (
                self.reservation_price - self.mid_price if self.mid_price else 0.0
            ),
            "half_spread": self.half_spread,
            "toxicity_multiplier": self.toxicity_multiplier,
            "drift_adjustment": self.drift_adjustment,
            "sigma": self.sigma,
            "vpin": self.vpin,
            "ofi_zscore": self.ofi_zscore,
            "risk_action": self.risk_action,
            "risk_reason": self.risk_reason,
            "breaker_tripped": self.breaker_tripped,
            "inventory_utilisation": self.inventory_utilisation,
            "last_update_ns": self.last_update_ns,
        }


class ApiGatewayService(StreamService):
    name = "api-gateway-svc"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        self.state = LiveState()
        self.bus = RedisEventBus(
            self.settings.infra.redis_url,
            maxlen=self.settings.infra.stream_maxlen,
            block_ms=self.settings.infra.block_ms,
        )
        self._clients: set[WebSocket] = set()
        self._broadcast_task: asyncio.Task[None] | None = None
        self.add_health_check("redis", self._check_redis)
        self.add_health_check("data", self._check_data)

    async def _check_redis(self) -> tuple[bool, str]:
        try:
            return bool(await self.bus.ping()), "connected"
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"

    async def _check_data(self) -> tuple[bool, str]:
        return True, (
            f"mid={self.state.mid_price:.2f} pnl={self.state.total_pnl:.2f} "
            f"fills={self.state.n_fills} clients={len(self._clients)}"
        )

    async def startup(self) -> None:
        group = f"{self.settings.infra.consumer_group}-gateway"
        for stream in (
            Streams.LOB_RAW,
            Streams.QUOTE_INTENT,
            Streams.RISK_DECISION,
            Streams.EXEC_FILL,
            Streams.EXEC_PORTFOLIO,
            Streams.FEATURES_READY,
        ):
            await self.bus.ensure_group(stream, group)
        self._broadcast_task = asyncio.create_task(self._broadcast(), name="gateway-broadcast")

    async def shutdown(self) -> None:
        if self._broadcast_task is not None:
            self._broadcast_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._broadcast_task
        await self.bus.close()

    async def _broadcast(self) -> None:
        """Push state to WebSocket clients at 2Hz and sample P&L at 1Hz."""
        tick = 0
        while not self.stopping:
            if await self.wait_for_stop(0.5):
                return
            tick += 1
            payload = self.state.as_dict()

            if tick % 2 == 0 and self.state.mid_price > 0:
                self.state.pnl_history.append(
                    {
                        "t": self.state.last_update_ns / 1e9,
                        "pnl": self.state.total_pnl,
                        "inventory": self.state.inventory,
                        "mid": self.state.mid_price,
                        "vpin": self.state.vpin,
                    }
                )

            if not self._clients:
                continue
            message = json.dumps({"type": "state", "data": payload})
            dead: set[WebSocket] = set()
            for client in self._clients:
                try:
                    await client.send_text(message)
                except Exception:  # noqa: BLE001 -- client vanished
                    dead.add(client)
            self._clients -= dead

    def build_app(self) -> FastAPI:
        app = super().build_app()

        @app.get("/", response_class=HTMLResponse)
        async def index() -> str:
            page = STATIC_DIR / "index.html"
            if page.exists():
                return page.read_text(encoding="utf-8")
            return "<h1>AMME</h1><p>Dashboard asset missing.</p>"

        @app.get("/api/state")
        async def state() -> dict[str, Any]:
            return self.state.as_dict()

        @app.get("/api/history")
        async def history() -> dict[str, Any]:
            return {"points": list(self.state.pnl_history)}

        @app.get("/api/fills")
        async def fills() -> dict[str, Any]:
            return {"fills": list(self.state.recent_fills)}

        @app.websocket("/ws")
        async def ws(socket: WebSocket) -> None:
            await socket.accept()
            self._clients.add(socket)
            try:
                await socket.send_text(
                    json.dumps({"type": "state", "data": self.state.as_dict()})
                )
                while True:
                    # The client never sends commands; this read exists purely
                    # to observe disconnects.
                    await socket.receive_text()
            except (WebSocketDisconnect, RuntimeError):
                pass
            finally:
                self._clients.discard(socket)

        return app

    async def run_loop(self) -> None:
        group = f"{self.settings.infra.consumer_group}-gateway"
        consumer = f"{self.name}-1"
        streams = [
            Streams.LOB_RAW,
            Streams.QUOTE_INTENT,
            Streams.RISK_DECISION,
            Streams.EXEC_FILL,
            Streams.EXEC_PORTFOLIO,
            Streams.FEATURES_READY,
        ]

        async for stream, message_id, event in self.bus.consume(
            streams, group, consumer, BaseEvent
        ):
            if self.stopping:
                break
            try:
                self._apply(stream, event)
            except Exception as exc:  # noqa: BLE001 -- the dashboard never breaks trading
                self.log.warning("gateway_apply_failed", stream=stream, error=str(exc))
            finally:
                await self.bus.ack(stream, group, message_id)

    def _apply(self, stream: str, event: BaseEvent) -> None:
        s = self.state
        payload = event.model_dump()
        s.last_update_ns = event.ts_event

        if stream == Streams.LOB_RAW:
            book = OrderBookUpdate.model_validate(payload)
            if book.mid_price is not None:
                s.mid_price = book.mid_price
                s.spread = book.spread or 0.0
                s.best_bid = book.bids[0].price if book.bids else 0.0
                s.best_ask = book.asks[0].price if book.asks else 0.0

        elif stream == Streams.FEATURES_READY:
            s.sigma = float(payload.get("realized_vol", 0.0))
            s.vpin = float(payload.get("vpin", 0.0))
            s.ofi_zscore = float(payload.get("ofi_zscore", 0.0))

        elif stream == Streams.QUOTE_INTENT:
            intent = QuoteIntent.model_validate(payload)
            s.quote_bid = intent.bid_price
            s.quote_ask = intent.ask_price
            s.reservation_price = intent.reservation_price
            s.half_spread = intent.half_spread
            s.toxicity_multiplier = intent.toxicity_multiplier
            s.drift_adjustment = intent.drift_adjustment
            s.n_quotes += 1

        elif stream == Streams.RISK_DECISION:
            decision = RiskDecision.model_validate(payload)
            s.risk_action = decision.action.value
            s.risk_reason = decision.reason
            s.breaker_tripped = decision.breaker_tripped
            s.inventory_utilisation = decision.inventory_utilisation

        elif stream == Streams.EXEC_FILL:
            fill = Fill.model_validate(payload)
            s.recent_fills.appendleft(
                {
                    "ts": fill.ts_event / 1e9,
                    "side": fill.side.value,
                    "price": fill.price,
                    "size": fill.size,
                    "inventory_after": fill.inventory_after,
                }
            )

        elif stream == Streams.EXEC_PORTFOLIO:
            portfolio = PortfolioState.model_validate(payload)
            s.inventory = portfolio.inventory
            s.cash = portfolio.cash
            s.realized_pnl = portfolio.realized_pnl
            s.unrealized_pnl = portfolio.unrealized_pnl
            s.total_pnl = portfolio.total_pnl
            s.n_fills = portfolio.n_fills


__all__ = ["ApiGatewayService", "LiveState"]
