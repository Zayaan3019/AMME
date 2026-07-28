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
import math
import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from amme_shared.config import Settings
from amme_shared.events import (
    BaseEvent,
    DriftSignal,
    FeatureVector,
    Fill,
    OrderBookUpdate,
    PortfolioState,
    QuoteIntent,
    RiskDecision,
    TradeTick,
)
from amme_shared.service import StreamService
from amme_shared.streams import RedisEventBus, Streams
from api_gateway_svc.peers import PIPELINE, probe_pipeline
from api_gateway_svc.research import load_research

STATIC_DIR = Path(__file__).parent / "static"

#: ~10 minutes of 1Hz samples. Bounded because this is an in-process projection,
#: not a time-series database -- Prometheus holds the long history.
HISTORY = 600
#: Staleness samples kept for the p50/p99 estimate shown on the System view.
LATENCY_WINDOW = 512
#: Seconds of stream-count history used to derive per-stream message rates.
RATE_WINDOW_S = 10.0


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile. Exact on small samples, no numpy dependency.

    Nearest-rank rather than linear interpolation because these are latency
    samples: reporting a p99 that is not an observed value invents a
    measurement, and at n=512 the interpolation buys nothing.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


class LiveState:
    """Rolling snapshot of the pipeline, fed by the stream tail.

    Every field is last-write-wins from its owning stream. There is deliberately
    no locking: the consumer loop is the only writer and it is single-threaded
    within the event loop, so a reader in a request handler sees a consistent
    field even though it may see fields from adjacent ticks. For a dashboard
    that is the right trade -- a lock on the hot consume path to make two
    numbers agree to within 100ms is not worth it.
    """

    def __init__(self, symbol: str = "", depth_levels: int = 10) -> None:
        self.symbol = symbol
        self.depth_levels = depth_levels
        self.started_at = time.time()

        # -- market -------------------------------------------------------
        self.mid_price = 0.0
        self.micro_price = 0.0
        self.spread = 0.0
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.bids: list[dict[str, float]] = []
        self.asks: list[dict[str, float]] = []
        self.last_update_id = 0
        self.last_trade_price = 0.0
        self.last_trade_side = ""

        # -- portfolio ----------------------------------------------------
        self.inventory = 0.0
        self.cash = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_pnl = 0.0
        self.fees_paid = 0.0
        self.n_fills = 0
        self.n_quotes = 0

        # -- quote --------------------------------------------------------
        self.quote_bid = 0.0
        self.quote_ask = 0.0
        self.quote_bid_size = 0.0
        self.quote_ask_size = 0.0
        self.reservation_price = 0.0
        self.half_spread = 0.0
        self.toxicity_multiplier = 1.0
        self.drift_adjustment = 0.0
        self.gamma = 0.0
        self.kappa = 0.0
        self.strategy = ""

        # -- features -----------------------------------------------------
        self.sigma = 0.0
        self.sigma_fast = 0.0
        self.sigma_baseline = 0.0
        self.vpin = 0.0
        self.vpin_warm = False
        self.ofi = 0.0
        self.ofi_zscore = 0.0
        self.ofi_warm = False
        self.vol_warm = False

        # -- drift signal -------------------------------------------------
        self.drift = 0.0
        self.drift_confidence = 0.0
        self.prob_down = 0.0
        self.prob_flat = 0.0
        self.prob_up = 0.0
        self.model_version = ""
        self.inference_latency_us = 0.0
        self.horizon_events = 0

        # -- risk ---------------------------------------------------------
        self.risk_action = "none"
        self.risk_reason = ""
        self.breaker_tripped = False
        self.inventory_utilisation = 0.0

        # -- observability ------------------------------------------------
        self.last_update_ns = 0
        self.stream_counts: dict[str, int] = dict.fromkeys(Streams.ALL, 0)
        self._rate_samples: deque[tuple[float, dict[str, int]]] = deque(maxlen=64)
        self._staleness_ms: deque[float] = deque(maxlen=LATENCY_WINDOW)
        self._inference_us: deque[float] = deque(maxlen=LATENCY_WINDOW)

        # -- history ------------------------------------------------------
        self.pnl_history: deque[dict[str, float]] = deque(maxlen=HISTORY)
        self.recent_fills: deque[dict[str, Any]] = deque(maxlen=50)
        self.recent_trades: deque[dict[str, Any]] = deque(maxlen=60)

    # -- derived ----------------------------------------------------------
    @property
    def book_imbalance(self) -> float:
        """Top-of-book size imbalance in [-1, 1]. Positive = bid-heavy.

        The single most-watched microstructure number on any trading screen,
        and free from data we already hold.
        """
        if not self.bids or not self.asks:
            return 0.0
        bid_size, ask_size = self.bids[0]["size"], self.asks[0]["size"]
        total = bid_size + ask_size
        return (bid_size - ask_size) / total if total > 0 else 0.0

    def stream_rates(self) -> dict[str, float]:
        """Messages/second per stream over the trailing rate window."""
        now = time.time()
        if not self._rate_samples:
            return dict.fromkeys(Streams.ALL, 0.0)
        # Oldest sample still inside the window; falls back to the oldest held.
        oldest_t, oldest_counts = self._rate_samples[0]
        for sample_t, counts in self._rate_samples:
            if now - sample_t <= RATE_WINDOW_S:
                oldest_t, oldest_counts = sample_t, counts
                break
        elapsed = now - oldest_t
        if elapsed <= 0.0:
            return dict.fromkeys(Streams.ALL, 0.0)
        return {
            name: max(0.0, (self.stream_counts[name] - oldest_counts.get(name, 0)) / elapsed)
            for name in Streams.ALL
        }

    def sample_rates(self) -> None:
        self._rate_samples.append((time.time(), dict(self.stream_counts)))

    def staleness_percentiles(self) -> tuple[float, float]:
        """(p50, p99) clock-corrected book staleness in milliseconds."""
        samples = list(self._staleness_ms)
        return _percentile(samples, 0.50), _percentile(samples, 0.99)

    def record_staleness_ms(self, value: float) -> None:
        self._staleness_ms.append(value)

    def record_inference_us(self, value: float) -> None:
        self._inference_us.append(value)

    def as_dict(self) -> dict[str, Any]:
        staleness_p50, staleness_p99 = self.staleness_percentiles()
        inference = list(self._inference_us)
        return {
            "symbol": self.symbol,
            "uptime_s": time.time() - self.started_at,
            # market
            "mid_price": self.mid_price,
            "micro_price": self.micro_price,
            "spread": self.spread,
            "spread_bps": (self.spread / self.mid_price * 1e4) if self.mid_price else 0.0,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "book_imbalance": self.book_imbalance,
            "last_update_id": self.last_update_id,
            "last_trade_price": self.last_trade_price,
            "last_trade_side": self.last_trade_side,
            # portfolio
            "inventory": self.inventory,
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_pnl,
            "fees_paid": self.fees_paid,
            "n_fills": self.n_fills,
            "n_quotes": self.n_quotes,
            "position_value": self.inventory * self.mid_price,
            # quote
            "quote_bid": self.quote_bid,
            "quote_ask": self.quote_ask,
            "quote_bid_size": self.quote_bid_size,
            "quote_ask_size": self.quote_ask_size,
            "quoted_spread": max(0.0, self.quote_ask - self.quote_bid),
            "reservation_price": self.reservation_price,
            "reservation_offset": (
                self.reservation_price - self.mid_price if self.mid_price else 0.0
            ),
            "half_spread": self.half_spread,
            "toxicity_multiplier": self.toxicity_multiplier,
            "drift_adjustment": self.drift_adjustment,
            "gamma": self.gamma,
            "kappa": self.kappa,
            "strategy": self.strategy,
            # features
            "sigma": self.sigma,
            "sigma_fast": self.sigma_fast,
            "sigma_baseline": self.sigma_baseline,
            "vpin": self.vpin,
            "vpin_warm": self.vpin_warm,
            "ofi": self.ofi,
            "ofi_zscore": self.ofi_zscore,
            "ofi_warm": self.ofi_warm,
            "vol_warm": self.vol_warm,
            # drift
            "drift": self.drift,
            "drift_confidence": self.drift_confidence,
            "prob_down": self.prob_down,
            "prob_flat": self.prob_flat,
            "prob_up": self.prob_up,
            "model_version": self.model_version,
            "inference_latency_us": self.inference_latency_us,
            "inference_p99_ms": _percentile(inference, 0.99) / 1000.0,
            "horizon_events": self.horizon_events,
            # risk
            "risk_action": self.risk_action,
            "risk_reason": self.risk_reason,
            "breaker_tripped": self.breaker_tripped,
            "inventory_utilisation": self.inventory_utilisation,
            # observability
            "last_update_ns": self.last_update_ns,
            "staleness_p50_ms": staleness_p50,
            "staleness_p99_ms": staleness_p99,
            "total_events": sum(self.stream_counts.values()),
        }


class ApiGatewayService(StreamService):
    name = "api-gateway-svc"

    #: Streams the gateway projects. This is every stream in the system: the
    #: dashboard's job is to show the whole pipeline, so a stream that exists
    #: but is not tailed here is a blind spot.
    STREAMS: tuple[str, ...] = (
        Streams.LOB_RAW,
        Streams.LOB_TRADES,
        Streams.FEATURES_READY,
        Streams.DRIFT_SIGNAL,
        Streams.QUOTE_INTENT,
        Streams.RISK_DECISION,
        Streams.EXEC_FILL,
        Streams.EXEC_PORTFOLIO,
    )

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        self.state = LiveState(
            symbol=self.settings.market.symbol,
            depth_levels=self.settings.market.depth_levels,
        )
        self.bus = RedisEventBus(
            self.settings.infra.redis_url,
            maxlen=self.settings.infra.stream_maxlen,
            block_ms=self.settings.infra.block_ms,
        )
        self._clients: set[WebSocket] = set()
        self._broadcast_task: asyncio.Task[None] | None = None
        self.add_health_check("redis", self._check_redis)
        self.add_health_check("data", self._check_data)

    # -- health -----------------------------------------------------------
    async def _check_redis(self) -> tuple[bool, str]:
        try:
            return bool(await self.bus.ping()), "connected"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    async def _check_data(self) -> tuple[bool, str]:
        return True, (
            f"mid={self.state.mid_price:.4f} pnl={self.state.total_pnl:.4f} "
            f"fills={self.state.n_fills} events={sum(self.state.stream_counts.values())} "
            f"clients={len(self._clients)}"
        )

    # -- lifecycle --------------------------------------------------------
    @property
    def _group(self) -> str:
        return f"{self.settings.infra.consumer_group}-gateway"

    async def startup(self) -> None:
        for stream in self.STREAMS:
            await self.bus.ensure_group(stream, self._group)
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

            if tick % 2 == 0:
                self.state.sample_rates()
                if self.state.mid_price > 0:
                    self.state.pnl_history.append(
                        {
                            "t": self.state.last_update_ns / 1e9,
                            "pnl": self.state.total_pnl,
                            "realized": self.state.realized_pnl,
                            "inventory": self.state.inventory,
                            "mid": self.state.mid_price,
                            "vpin": self.state.vpin,
                        }
                    )

            if not self._clients:
                continue
            message = json.dumps({"type": "state", "data": payload, "book": self._book_payload()})
            dead: set[WebSocket] = set()
            for client in self._clients:
                try:
                    await client.send_text(message)
                except Exception:
                    dead.add(client)
            self._clients -= dead

    # -- projections ------------------------------------------------------
    def _book_payload(self) -> dict[str, Any]:
        return {
            "bids": self.state.bids,
            "asks": self.state.asks,
            "mid_price": self.state.mid_price,
            "quote_bid": self.state.quote_bid,
            "quote_ask": self.state.quote_ask,
            "reservation_price": self.state.reservation_price,
            "imbalance": self.state.book_imbalance,
        }

    def _config_payload(self) -> dict[str, Any]:
        """The parameters the running system is actually using.

        Shown on the System view so the numbers on screen can be tied to the
        configuration that produced them -- a screenshot of a dashboard with no
        parameters is unfalsifiable.
        """
        s = self.settings
        return {
            "symbol": s.market.symbol,
            "depth_levels": s.market.depth_levels,
            "depth_update_ms": s.market.depth_update_ms,
            "tick_size": s.market.tick_size,
            "avellaneda_stoikov": {
                "gamma": s.avellaneda_stoikov.gamma,
                "kappa": s.avellaneda_stoikov.kappa,
                "horizon_mode": s.avellaneda_stoikov.horizon_mode,
                "horizon_seconds": s.avellaneda_stoikov.horizon_seconds,
                "quote_size": s.avellaneda_stoikov.quote_size,
                "min_half_spread_ticks": s.avellaneda_stoikov.min_half_spread_ticks,
                "drift_lambda": s.avellaneda_stoikov.drift_lambda,
                "drift_confidence_floor": s.avellaneda_stoikov.drift_confidence_floor,
            },
            "risk": {
                "q_max": s.risk.q_max,
                "q_soft_frac": s.risk.q_soft_frac,
                "unwind_frac": s.risk.unwind_frac,
                "max_position_value": s.risk.max_position_value,
                "breaker_vpin": s.risk.breaker_vpin,
                "breaker_vol_multiple": s.risk.breaker_vol_multiple,
                "breaker_drawdown": s.risk.breaker_drawdown,
                "enabled": s.risk.enabled,
            },
            "toxicity": {
                "vpin_buckets": s.toxicity.vpin_buckets,
                "vpin_bucket_volume": s.toxicity.vpin_bucket_volume,
                "vpin_threshold": s.toxicity.vpin_threshold,
                "vpin_beta": s.toxicity.vpin_beta,
                "ofi_threshold": s.toxicity.ofi_threshold,
                "ofi_beta": s.toxicity.ofi_beta,
            },
            "model": {
                "backend": s.model.backend,
                "lookback": s.model.lookback,
                "levels": s.model.levels,
                "horizon_events": s.model.horizon_events,
                "inference_p99_budget_ms": s.model.inference_p99_budget_ms,
            },
        }

    # -- HTTP surface -----------------------------------------------------
    def build_app(self) -> FastAPI:
        app = super().build_app()

        # Assets live under /static so the SPA can be split into modules
        # without a bundler. `html=False` keeps index.html served by the route
        # below, which is the only place the shell is resolved.
        if STATIC_DIR.is_dir():
            app.mount("/static", StaticFiles(directory=STATIC_DIR, html=False), name="static")

        @app.get("/", response_class=HTMLResponse)
        async def index() -> str:
            page = STATIC_DIR / "index.html"
            if page.exists():
                return page.read_text(encoding="utf-8")
            return "<h1>AMME</h1><p>Dashboard asset missing.</p>"

        @app.get("/api/state")
        async def state() -> dict[str, Any]:
            return self.state.as_dict()

        @app.get("/api/book")
        async def book() -> dict[str, Any]:
            return self._book_payload()

        @app.get("/api/history")
        async def history() -> dict[str, Any]:
            return {"points": list(self.state.pnl_history)}

        @app.get("/api/fills")
        async def fills() -> dict[str, Any]:
            return {"fills": list(self.state.recent_fills)}

        @app.get("/api/trades")
        async def trades() -> dict[str, Any]:
            return {"trades": list(self.state.recent_trades)}

        @app.get("/api/config")
        async def config() -> dict[str, Any]:
            return self._config_payload()

        @app.get("/api/system")
        async def system() -> dict[str, Any]:
            services = await probe_pipeline(self.settings.dashboard.peer_url_template)
            staleness_p50, staleness_p99 = self.state.staleness_percentiles()
            return {
                "services": services,
                "streams": [
                    {
                        "name": name,
                        "count": self.state.stream_counts[name],
                        "rate": round(rate, 2),
                    }
                    for name, rate in self.state.stream_rates().items()
                ],
                "staleness_p50_ms": staleness_p50,
                "staleness_p99_ms": staleness_p99,
                "uptime_s": time.time() - self.state.started_at,
                "config": self._config_payload(),
            }

        @app.get("/api/research")
        async def research() -> dict[str, Any]:
            return load_research(Path(self.settings.backtest.results_dir))

        @app.websocket("/ws")
        async def ws(socket: WebSocket) -> None:
            await socket.accept()
            self._clients.add(socket)
            try:
                await socket.send_text(
                    json.dumps(
                        {
                            "type": "state",
                            "data": self.state.as_dict(),
                            "book": self._book_payload(),
                        }
                    )
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

    # -- stream consumption -----------------------------------------------
    async def run_loop(self) -> None:
        consumer = f"{self.name}-1"
        async for stream, message_id, event in self.bus.consume(
            list(self.STREAMS), self._group, consumer, BaseEvent
        ):
            if self.stopping:
                break
            try:
                self._apply(stream, event)
            except Exception as exc:
                self.log.warning("gateway_apply_failed", stream=stream, error=str(exc))
            finally:
                await self.bus.ack(stream, self._group, message_id)

    def _apply(self, stream: str, event: BaseEvent) -> None:
        s = self.state
        payload = event.model_dump()
        s.last_update_ns = event.ts_event
        if stream in s.stream_counts:
            s.stream_counts[stream] += 1

        if stream == Streams.LOB_RAW:
            book = OrderBookUpdate.model_validate(payload)
            if book.mid_price is not None:
                s.mid_price = book.mid_price
                s.micro_price = book.micro_price or book.mid_price
                s.spread = book.spread or 0.0
                s.best_bid = book.bids[0].price
                s.best_ask = book.asks[0].price
            s.bids = [{"price": lv.price, "size": lv.size} for lv in book.bids]
            s.asks = [{"price": lv.price, "size": lv.size} for lv in book.asks]
            s.last_update_id = book.last_update_id
            # ts_ingest is stamped in *exchange* time by ingestion-svc's
            # ClockSync, so this difference is the clock-corrected one-way
            # latency, not the local clock's opinion of it.
            s.record_staleness_ms(book.staleness_ns / 1e6)

        elif stream == Streams.LOB_TRADES:
            trade = TradeTick.model_validate(payload)
            s.last_trade_price = trade.price
            s.last_trade_side = trade.aggressor.value
            s.recent_trades.appendleft(
                {
                    "ts": trade.ts_event / 1e9,
                    "price": trade.price,
                    "size": trade.size,
                    "aggressor": trade.aggressor.value,
                }
            )

        elif stream == Streams.FEATURES_READY:
            features = FeatureVector.model_validate(payload)
            s.sigma = features.realized_vol
            s.sigma_fast = features.realized_vol_fast
            s.sigma_baseline = features.realized_vol_baseline
            s.vpin = features.vpin
            s.vpin_warm = features.vpin_warm
            s.ofi = features.ofi
            s.ofi_zscore = features.ofi_zscore
            s.ofi_warm = features.ofi_warm
            s.vol_warm = features.vol_warm

        elif stream == Streams.DRIFT_SIGNAL:
            drift = DriftSignal.model_validate(payload)
            s.drift = drift.drift
            s.drift_confidence = drift.confidence
            s.prob_down = drift.prob_down
            s.prob_flat = drift.prob_flat
            s.prob_up = drift.prob_up
            s.model_version = drift.model_version
            s.inference_latency_us = drift.inference_latency_us
            s.horizon_events = drift.horizon_events
            s.record_inference_us(drift.inference_latency_us)

        elif stream == Streams.QUOTE_INTENT:
            intent = QuoteIntent.model_validate(payload)
            s.quote_bid = intent.bid_price
            s.quote_ask = intent.ask_price
            s.quote_bid_size = intent.bid_size
            s.quote_ask_size = intent.ask_size
            s.reservation_price = intent.reservation_price
            s.half_spread = intent.half_spread
            s.toxicity_multiplier = intent.toxicity_multiplier
            s.drift_adjustment = intent.drift_adjustment
            s.gamma = intent.gamma
            s.kappa = intent.kappa
            s.strategy = intent.strategy
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
                    "fee": fill.fee,
                    "queue_position": fill.queue_position_at_fill,
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
            s.fees_paid = portfolio.fees_paid
            s.n_fills = portfolio.n_fills


__all__ = ["PIPELINE", "ApiGatewayService", "LiveState"]
