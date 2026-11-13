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
import time

from amme_core.book import ResyncReason
from amme_shared.config import Settings
from amme_shared.db import TickStore
from amme_shared.events import OrderBookUpdate, TradeTick
from amme_shared.metrics import (
    BOOK_RESYNCS,
    CLOCK_OFFSET_MS,
    EVENT_STALENESS,
    EVENTS_FAILED,
    EVENTS_PUBLISHED,
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

#: While TimescaleDB is unreachable, connection attempts and write retries are
#: spaced this far apart, and unwritten rows wait in bounded buffers. At ~10
#: books/s these caps hold roughly half an hour of market data; past them the
#: oldest rows are dropped and counted rather than growing memory without bound.
PERSIST_RETRY_SECONDS = 15.0
PERSIST_MAX_BOOKS = 20_000
PERSIST_MAX_TRADES = 60_000

#: Health fails if no book has been published in this long. Tuned well above
#: the 100ms cadence so a quiet market does not page anyone, but low enough to
#: catch a genuinely wedged socket.
FRESHNESS_TIMEOUT_SECONDS = 15.0

#: How often to *consider* re-measuring the exchange clock. The decision to
#: actually re-measure is ClockSync's (`resync_interval_seconds`, 300s); this is
#: only the poll that lets that interval be honoured on a connection that never
#: reconnects. Kept well below the resync interval so the effective refresh
#: period is the interval itself rather than the interval plus a poll.
CLOCK_RESYNC_POLL_SECONDS = 30.0


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
        self._persist = self.settings.infra.persist_ticks
        self._persist_retry_at = 0.0  # loop time; no DB attempt before this
        self._persist_error = ""
        self._persist_written = 0
        self._persist_dropped = 0
        self._book_buffer: list[OrderBookUpdate] = []
        self._trade_buffer: list[TradeTick] = []
        self._last_publish_monotonic = 0.0
        self._publish_failures = 0
        self._flush_task: asyncio.Task[None] | None = None
        self._clock_task: asyncio.Task[None] | None = None

        self.add_health_check("redis", self._check_redis)
        self.add_health_check("book_synced", self._check_book)
        self.add_health_check("freshness", self._check_freshness)
        self.add_health_check("clock_sync", self._check_clock)

    # -- health ------------------------------------------------------------
    async def _check_redis(self) -> tuple[bool, str]:
        try:
            return bool(await self.bus.ping()), "connected"
        except Exception as exc:
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
        # Age is reported because a stale offset is the failure mode that hides:
        # the number stays plausible while the correction it feeds goes wrong,
        # and every latency metric silently inherits the error.
        age = time.monotonic() - clock.last_sync_monotonic
        return True, (
            f"offset={clock.offset_ms:+.1f}ms rtt={clock.rtt_ms:.1f}ms "
            f"measured {age:.0f}s ago (n={clock.sync_count})"
        )

    def _on_resync(self, reason: ResyncReason) -> None:
        BOOK_RESYNCS.labels(self.settings.market.symbol, reason.value).inc()

    # -- lifecycle ---------------------------------------------------------
    async def startup(self) -> None:
        await self.client.prepare()
        # Publish the offset from the very first measurement, not from the first
        # timer tick: a gauge that stays unset for 30s looks identical to a
        # service that never synced.
        CLOCK_OFFSET_MS.labels(self.settings.market.symbol).set(self.client.clock.offset_ms)
        if self.client.filters is not None:
            self.log.info(
                "instrument_ready",
                tick_size=self.client.filters.tick_size,
                lot_size=self.client.filters.lot_size,
                min_notional=self.client.filters.min_notional,
            )
        if self._persist:
            # Registered whether or not the first connection works. This used
            # to be added only on success: when TimescaleDB was down at
            # startup, recording switched off for the life of the process and
            # /health never mentioned it -- the stack ran "healthy" for months
            # while writing nothing. Non-critical: market data still flows to
            # Redis without the database, so the service is degraded, not down.
            self.add_health_check("timescaledb", self._check_db, critical=False)
            await self._ensure_store()
        self._flush_task = asyncio.create_task(self._flush_loop(), name="ingestion-flush")
        self._clock_task = asyncio.create_task(self._clock_loop(), name="ingestion-clock")

    async def _clock_loop(self) -> None:
        """Re-measure the exchange clock offset on a timer.

        This used to be driven solely by WebSocket reconnects, on the reasoning
        that a reconnect is a natural checkpoint. It is -- but it is not a
        *schedule*: on a connection that stays up for hours, the offset was
        measured exactly once and `resync_interval_seconds` never applied.

        That turned a transient measurement error into a permanent one. Observed
        in production: the host suspended, the Docker VM's clock resumed ~13s
        behind, the WebSocket reconnected, and ClockSync sampled while the VM
        clock was still catching up and latched offset=-13351ms. The VM clock
        corrected itself seconds later; the offset did not. Every latency metric
        in the system was wrong by 13.35 seconds for 45 minutes, and because the
        error lives in the correction rather than in the data, nothing else
        could detect it -- staleness simply read 13s and looked like a slow
        venue.

        A timer bounds the damage to one resync interval, and the offset is now
        also exported so a wrong correction is visible rather than inferred.
        """
        while not self.stopping:
            if await self.wait_for_stop(CLOCK_RESYNC_POLL_SECONDS):
                return
            try:
                await self.client.resync_clock()
            except Exception as exc:
                # A failed resync keeps the previous offset, which is the safe
                # outcome; the next tick retries. Never let it kill ingestion.
                self.log.warning("clock_resync_failed", error=f"{type(exc).__name__}: {exc}")
            CLOCK_OFFSET_MS.labels(self.settings.market.symbol).set(self.client.clock.offset_ms)

    async def _ensure_store(self) -> bool:
        """Connect to TimescaleDB if not connected, at most once per retry period."""
        if self.store is not None:
            return True
        now = asyncio.get_running_loop().time()
        if now < self._persist_retry_at:
            return False
        store = TickStore(self.settings.infra.postgres_dsn)
        try:
            await store.connect()
        except Exception as exc:
            self._persist_retry_at = now + PERSIST_RETRY_SECONDS
            self._persist_error = f"{type(exc).__name__}: {exc}"
            self.log.warning(
                "timescaledb_unavailable", error=self._persist_error,
                retry_in=PERSIST_RETRY_SECONDS, buffered=self._buffered(),
            )
            with contextlib.suppress(Exception):
                await store.close()
            return False
        self.store = store
        self._persist_error = ""
        self.log.info("timescaledb_connected", buffered=self._buffered())
        return True

    def _buffered(self) -> int:
        return len(self._book_buffer) + len(self._trade_buffer)

    async def _check_db(self) -> tuple[bool, str]:
        counts = (
            f"written={self._persist_written} buffered={self._buffered()} "
            f"dropped={self._persist_dropped}"
        )
        if self.store is None:
            return False, f"recording paused, not connected ({self._persist_error}); {counts}"
        if self._persist_error:
            return False, f"recording paused, last write failed ({self._persist_error}); {counts}"
        try:
            ok = bool(await self.store.ping())
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}; {counts}"
        return ok, f"recording; {counts}"

    async def shutdown(self) -> None:
        for task in (self._flush_task, self._clock_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
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

    def _trim_buffers(self) -> None:
        """Bound memory while the database is away: drop the oldest, and count it."""
        for buffer, cap in (
            (self._book_buffer, PERSIST_MAX_BOOKS),
            (self._trade_buffer, PERSIST_MAX_TRADES),
        ):
            excess = len(buffer) - cap
            if excess > 0:
                del buffer[:excess]
                self._persist_dropped += excess

    async def _flush(self, force: bool = False) -> None:
        """Write buffered ticks. A failed batch stays buffered and is retried.

        Books and trades are written as separate atomic batches (one
        ``executemany`` each), so a retry after a partial failure re-sends only
        the batch that did not commit -- nothing is written twice.
        """
        if not self._persist:
            self._book_buffer.clear()
            self._trade_buffer.clear()
            return
        self._trim_buffers()
        if not self._book_buffer and not self._trade_buffer:
            return
        if not await self._ensure_store() or self.store is None:
            return
        now = asyncio.get_running_loop().time()
        if not force and now < self._persist_retry_at:
            return
        try:
            # Snapshot the lengths: the read loop keeps appending while we await.
            n_books = len(self._book_buffer)
            if n_books:
                await self.store.write_books(self._book_buffer[:n_books])
                del self._book_buffer[:n_books]
                self._persist_written += n_books
            n_trades = len(self._trade_buffer)
            if n_trades:
                await self.store.write_trades(self._trade_buffer[:n_trades])
                del self._trade_buffer[:n_trades]
                self._persist_written += n_trades
        except Exception as exc:
            self._persist_error = f"{type(exc).__name__}: {exc}"
            self._persist_retry_at = now + PERSIST_RETRY_SECONDS
            self.log.error(
                "persist_failed", error=self._persist_error, buffered=self._buffered(),
                retry_in=PERSIST_RETRY_SECONDS,
            )
            return
        if self._persist_error:
            self.log.info("persist_recovered", written=self._persist_written)
        self._persist_error = ""

    async def _publish(self, stream: str, event: OrderBookUpdate | TradeTick) -> bool:
        """Publish one event. Returns False if it was dropped.

        A publish failure must never kill this service. It did once: Redis hit
        its memory ceiling under a mis-sized ``MAXLEN``, ``XADD`` raised
        ``OutOfMemoryError``, the exception escaped ``run_loop``, and the whole
        ingestion process died -- taking the WebSocket connection and the
        synchronised book with it. Recovering from that costs a full snapshot
        resync and seconds of blindness.

        Dropping one tick is strictly cheaper than losing the connection, so
        transient transport failures are counted and swallowed. The drop is
        visible in ``amme_events_failed_total``, and the freshness health check
        turns a *sustained* failure into an unhealthy signal within 15s -- so
        this degrades loudly without being fatal.
        """
        try:
            await self.bus.publish(stream, event)
        except Exception as exc:
            self._publish_failures += 1
            EVENTS_FAILED.labels(self.name, type(exc).__name__).inc()
            if self._publish_failures % 100 == 1:
                self.log.error(
                    "publish_failed_dropping_event",
                    stream=stream,
                    error=f"{type(exc).__name__}: {exc}",
                    consecutive=self._publish_failures,
                )
            return False
        self._publish_failures = 0
        EVENTS_PUBLISHED.labels(self.name, stream).inc()
        return True

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
                if await self._publish(Streams.LOB_RAW, event):
                    EVENT_STALENESS.labels(self.name).observe(event.staleness_ns / 1e6)
                    if event.mid_price is not None:
                        MID_PRICE.labels(symbol).set(event.mid_price)
                    self._last_publish_monotonic = loop.time()
                    self._book_buffer.append(event)
            elif await self._publish(Streams.LOB_TRADES, event):
                self._trade_buffer.append(event)

            if len(self._book_buffer) + len(self._trade_buffer) >= FLUSH_MAX_BATCH:
                await self._flush()


__all__ = ["IngestionService"]
