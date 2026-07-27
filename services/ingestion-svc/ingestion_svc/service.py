"""ingestion-svc: live Binance market data -> Redis Streams + TimescaleDB.

The head of the pipeline. Everything downstream depends on this service being
*correct*, not merely alive, so its health check asserts book synchronisation
and data freshness rather than just process liveness -- a service that is
running but publishing a stale book is worse than one that is down, because
nothing alerts.

Persistence is batched and best-effort: a database hiccup must never stall the
live path. If TimescaleDB is unreachable, we log, drop the batch, and keep
quoting. The database is for replay and training, not for trading.
"""

from __future__ import annotations

import asyncio
import contextlib

from amme_core.book import ResyncReason
from amme_shared.config import Settings
from amme_shared.db import TickStore
from amme_shared.events import OrderBookUpdate, TradeTick
from amme_shared.metrics import (
    BOOK_RESYNCS,
    EVENTS_PUBLISHED,
    EVENT_STALENESS,
    MID_PRICE,
    WS_RECONNECTS,
)
from amme_shared.service import StreamService
from amme_shared.streams import RedisEventBus, Streams
from ingestion_svc.binance import BinanceConfig, BinanceDepthClient

#: Books are buffered this long before a database write. At 10 books/sec this
#: batches ~20 rows per round trip instead of paying one per book.
FLUSH_INTERVAL_SECONDS = 2.0
FLUSH_MAX_BATCH = 500

#: Health fails if no book has been published in this long. Tuned well above
#: the 100ms cadence so a quiet market does not page anyone, but low enough to
#: catch a genuinely wedged socket.
FRESHNESS_TIMEOUT_SECONDS = 15.0


class IngestionService(StreamService):
    name = "ingestion-svc"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        market = self.settings.market
        self.client = BinanceDepthClient(
            BinanceConfig(
                symbol=market.symbol,
                depth_levels=market.depth_levels,
                ws_base=market.ws_base,
                rest_base=market.rest_base,
                depth_update_ms=market.depth_update_ms,
            ),
            on_resync=self._on_resync,
        )
        self.bus = RedisEventBus(
            self.settings.infra.redis_url,
            maxlen=self.settings.infra.stream_maxlen,
            block_ms=self.settings.infra.block_ms,
        )
        self.store: TickStore | None = None
        self._book_buffer: list[OrderBookUpdate] = []
        self._trade_buffer: list[TradeTick] = []
        self._last_publish_monotonic = 0.0
        self._flush_task: asyncio.Task[None] | None = None

        self.add_health_check("redis", self._check_redis)
        self.add_health_check("book_synced", self._check_book)
        self.add_health_check("freshness", self._check_freshness)
        self.add_health_check("clock_sync", self._check_clock)

    # -- health ------------------------------------------------------------
    async def _check_redis(self) -> tuple[bool, str]:
        try:
            return bool(await self.bus.ping()), "connected"
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"

    async def _check_book(self) -> tuple[bool, str]:
        synced = self.client.sync.is_synced
        return synced, (
            f"state={self.client.sync.state} last_update_id={self.client.book.last_update_id} "
            f"resyncs={self.client.stats.resyncs}"
        )

    async def _check_freshness(self) -> tuple[bool, str]:
        if self._last_publish_monotonic == 0.0:
            return False, "no book published yet"
        age = asyncio.get_running_loop().time() - self._last_publish_monotonic
        return age < FRESHNESS_TIMEOUT_SECONDS, f"last book {age:.1f}s ago"

    async def _check_clock(self) -> tuple[bool, str]:
        clock = self.client.clock
        if not clock.synced:
            return False, "clock never synced with exchange"
        return True, f"offset={clock.offset_ms:+.1f}ms rtt={clock.rtt_ms:.1f}ms"

    def _on_resync(self, reason: ResyncReason) -> None:
        BOOK_RESYNCS.labels(self.settings.market.symbol, reason.value).inc()

    # -- lifecycle ---------------------------------------------------------
    async def startup(self) -> None:
        await self.client.prepare()
        if self.client.filters is not None:
            self.log.info(
                "instrument_ready",
                tick_size=self.client.filters.tick_size,
                lot_size=self.client.filters.lot_size,
                min_notional=self.client.filters.min_notional,
            )
        if self.settings.infra.persist_ticks:
            self.store = TickStore(self.settings.infra.postgres_dsn)
            try:
                await self.store.connect()
                self.add_health_check("timescaledb", self._check_db)
            except Exception as exc:  # noqa: BLE001 -- persistence is optional
                self.log.warning("timescaledb_unavailable", error=str(exc))
                self.store = None
        self._flush_task = asyncio.create_task(self._flush_loop(), name="ingestion-flush")

    async def _check_db(self) -> tuple[bool, str]:
        if self.store is None:
            return False, "not connected"
        try:
            return bool(await self.store.ping()), "connected"
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"

    async def shutdown(self) -> None:
        if self._flush_task is not None:
            self._flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._flush_task
        await self._flush(force=True)
        await self.client.close()
        await self.bus.close()
        if self.store is not None:
            await self.store.close()

    # -- persistence -------------------------------------------------------
    async def _flush_loop(self) -> None:
        while not self.stopping:
            if await self.wait_for_stop(FLUSH_INTERVAL_SECONDS):
                return
            await self._flush()

    async def _flush(self, force: bool = False) -> None:
        if self.store is None:
            self._book_buffer.clear()
            self._trade_buffer.clear()
            return
        books, self._book_buffer = self._book_buffer, []
        trades, self._trade_buffer = self._trade_buffer, []
        if not books and not trades:
            return
        try:
            if books:
                await self.store.write_books(books)
            if trades:
                await self.store.write_trades(trades)
        except Exception as exc:  # noqa: BLE001 -- never let storage stall the feed
            self.log.error(
                "persist_failed", books=len(books), trades=len(trades), error=str(exc)
            )

    # -- main loop ---------------------------------------------------------
    async def run_loop(self) -> None:
        symbol = self.settings.market.symbol
        loop = asyncio.get_running_loop()
        self.log.info("ingestion_started", symbol=symbol, stream=self.client.config.stream_url)

        prev_reconnects = 0
        async for event in self.client.stream():
            if self.stopping:
                break

            if self.client.stats.reconnects != prev_reconnects:
                WS_RECONNECTS.labels(symbol).inc(self.client.stats.reconnects - prev_reconnects)
                prev_reconnects = self.client.stats.reconnects

            if isinstance(event, OrderBookUpdate):
                await self.bus.publish(Streams.LOB_RAW, event)
                EVENTS_PUBLISHED.labels(self.name, Streams.LOB_RAW).inc()
                EVENT_STALENESS.labels(self.name).observe(event.staleness_ns / 1e6)
                if event.mid_price is not None:
                    MID_PRICE.labels(symbol).set(event.mid_price)
                self._last_publish_monotonic = loop.time()
                self._book_buffer.append(event)
            else:
                await self.bus.publish(Streams.LOB_TRADES, event)
                EVENTS_PUBLISHED.labels(self.name, Streams.LOB_TRADES).inc()
                self._trade_buffer.append(event)

            if len(self._book_buffer) + len(self._trade_buffer) >= FLUSH_MAX_BATCH:
                await self._flush()


__all__ = ["IngestionService"]
