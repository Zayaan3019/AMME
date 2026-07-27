"""Live Binance market data client.

This is the real thing: a production WebSocket connection to
``stream.binance.com``, the REST depth snapshot, and the full documented
snapshot+diff resynchronisation handshake. No API key, no account, no cost.

Responsibilities are split deliberately:

* :class:`~amme_core.book.BinanceBookSynchronizer` owns the *protocol* -- pure,
  synchronous, fully unit-testable without a socket.
* This module owns the *transport* -- connecting, reconnecting, backoff,
  parsing, and deciding when to go fetch a snapshot.

That split is why the resync logic can be tested exhaustively against injected
gaps and disconnects (``tests/unit/test_book_resync.py``) rather than only
against whatever the live feed happens to do today.

Operational details that matter in production and are easy to miss:

* Binance closes any stream after **24 hours**. We proactively cycle the
  connection before that, so the reconnect happens on our schedule rather than
  mid-burst.
* The server sends a ping every ~3 minutes and expects a pong within 10; the
  ``websockets`` library answers automatically, but we set explicit intervals
  so a half-open TCP connection is detected in seconds rather than minutes.
* A depth diff can arrive *while* the REST snapshot is in flight. Those diffs
  must be buffered, not dropped -- that is exactly the case the synchroniser's
  buffer-and-replay path exists for.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import websockets
from websockets.asyncio.client import ClientConnection

from amme_core.book import (
    BinanceBookSynchronizer,
    DepthDiff,
    DepthSnapshot,
    L2Book,
    ResyncReason,
    levels_from_pairs,
)
from amme_shared.events import (
    AggressorSide,
    DataSource,
    OrderBookUpdate,
    TradeTick,
    new_id,
)
from amme_shared.logging import get_logger
from ingestion_svc.exchange import ClockSync, InstrumentFilters, fetch_instrument_filters

log = get_logger("ingestion.binance")

#: Binance caps REST depth at 5000; 1000 is the cheapest tier that reliably
#: covers more levels than we publish, so a snapshot is never truncated inside
#: the range we care about.
SNAPSHOT_LIMIT = 1000

#: Well inside Binance's 24h hard limit, with room for the reconnect itself.
CONNECTION_MAX_AGE_SECONDS = 20 * 3600


@dataclass(slots=True)
class BinanceConfig:
    symbol: str = "BTCUSDT"
    depth_levels: int = 10
    ws_base: str = "wss://stream.binance.com:9443"
    rest_base: str = "https://api.binance.com"
    depth_update_ms: int = 100
    subscribe_trades: bool = True

    # Reconnect backoff: exponential with jitter, capped.
    initial_backoff: float = 0.5
    max_backoff: float = 30.0
    backoff_multiplier: float = 2.0

    ws_ping_interval: float = 20.0
    ws_ping_timeout: float = 10.0
    ws_open_timeout: float = 15.0
    rest_timeout: float = 10.0
    max_snapshot_attempts: int = 5

    @property
    def stream_url(self) -> str:
        sym = self.symbol.lower()
        streams = [f"{sym}@depth@{self.depth_update_ms}ms"]
        if self.subscribe_trades:
            streams.append(f"{sym}@aggTrade")
        return f"{self.ws_base}/stream?streams={'/'.join(streams)}"

    @property
    def snapshot_url(self) -> str:
        return f"{self.rest_base}/api/v3/depth?symbol={self.symbol.upper()}&limit={SNAPSHOT_LIMIT}"


@dataclass(slots=True)
class IngestionStats:
    messages: int = 0
    depth_messages: int = 0
    trade_messages: int = 0
    books_published: int = 0
    trades_published: int = 0
    reconnects: int = 0
    resyncs: int = 0
    snapshot_fetches: int = 0
    snapshot_failures: int = 0
    decode_errors: int = 0
    last_event_ns: int = 0
    resync_reasons: dict[str, int] = field(default_factory=dict)

    def note_resync(self, reason: ResyncReason | None) -> None:
        self.resyncs += 1
        key = reason.value if reason is not None else "unknown"
        self.resync_reasons[key] = self.resync_reasons.get(key, 0) + 1


def parse_depth_diff(data: dict[str, Any]) -> DepthDiff:
    """`depthUpdate` payload -> :class:`DepthDiff`."""
    return DepthDiff(
        first_update_id=int(data["U"]),
        final_update_id=int(data["u"]),
        bids=levels_from_pairs(data.get("b", [])),
        asks=levels_from_pairs(data.get("a", [])),
        event_time_ms=int(data.get("E", 0)),
    )


def parse_agg_trade(
    data: dict[str, Any], symbol: str, clock: ClockSync | None = None
) -> TradeTick:
    """`aggTrade` payload -> :class:`TradeTick`.

    ``m`` is "buyer is the market maker". When it is true the *resting* order
    was the buy, so the aggressor was the **seller**. Getting this backwards
    inverts every VPIN reading, and it is a genuinely easy mistake to make
    because the field reads like it describes the aggressor.
    """
    trade_ms = int(data.get("T", data.get("E", 0)))
    buyer_is_maker = bool(data.get("m", False))
    return TradeTick(
        symbol=symbol,
        ts_event=trade_ms * 1_000_000,
        ts_ingest=clock.now_exchange_ns() if clock is not None else _now_ns(),
        source=DataSource.BINANCE_WS,
        correlation_id=new_id(),
        price=float(data["p"]),
        size=float(data["q"]),
        aggressor=AggressorSide.SELL if buyer_is_maker else AggressorSide.BUY,
        trade_id=int(data.get("a", 0)),
    )


def _now_ns() -> int:
    import time

    return time.time_ns()


class BinanceDepthClient:
    """Streams a correctly-sequenced L2 book and trade tape from Binance.

    Yields ``(OrderBookUpdate | TradeTick)`` events. The book events are
    *state*, not diffs: every one is a complete, validated top-N snapshot, so
    downstream consumers never reconstruct anything.
    """

    def __init__(
        self,
        config: BinanceConfig | None = None,
        session: aiohttp.ClientSession | None = None,
        on_resync: Callable[[ResyncReason], None] | None = None,
    ) -> None:
        self.config = config or BinanceConfig()
        self.book = L2Book(self.config.symbol)
        self.sync = BinanceBookSynchronizer(self.book)
        self.stats = IngestionStats()
        self._session = session
        self._owns_session = session is None
        self._on_resync = on_resync
        self._stopping: bool = False
        self._stop_event = asyncio.Event()
        self._ws: ClientConnection | None = None
        #: Corrects local-vs-exchange clock skew so staleness is honest.
        #: Measured +684ms on the dev host -- see ingestion_svc.exchange.
        self.clock = ClockSync(rest_base=self.config.rest_base)
        self.filters: InstrumentFilters | None = None

    # -- lifecycle ---------------------------------------------------------
    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.rest_timeout)
            )
            self._owns_session = True
        return self._session

    async def prepare(self) -> None:
        """Measure clock skew and load instrument filters. Call before streaming.

        Deliberately not folded into ``stream()``: a caller that skips it still
        works (timestamps just carry the skew), but a caller that wants honest
        latency numbers must opt in explicitly, and it is visible in the code
        that it did.
        """
        session = await self._ensure_session()
        await self.clock.sync(session)
        self.filters = await fetch_instrument_filters(
            session, self.config.symbol, self.config.rest_base
        )

    async def close(self) -> None:
        self._stopping = True
        self._stop_event.set()
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
        if self._owns_session and self._session is not None and not self._session.closed:
            await self._session.close()

    async def _wait_for_stop(self, timeout: float) -> bool:
        """Sleep up to ``timeout``. True if a stop was requested meanwhile."""
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout)
            return True
        except TimeoutError:
            return False

    @property
    def stopping(self) -> bool:
        """Read through a property, not the attribute.

        `stop()` is called from another task, so `_stopping` genuinely changes
        under the socket loop. mypy cannot model that: it narrows the attribute
        to False from `__init__` and marks every in-loop check unreachable.
        Going through a property is both the honest expression of "this is
        externally mutable" and what keeps the type checker correct.
        """
        return self._stopping

    def stop(self) -> None:
        self._stopping = True
        self._stop_event.set()

    # -- REST --------------------------------------------------------------
    async def fetch_snapshot(self) -> DepthSnapshot | None:
        """GET /api/v3/depth. Retries with backoff; ``None`` if it never lands."""
        session = await self._ensure_session()
        backoff = self.config.initial_backoff

        for attempt in range(1, self.config.max_snapshot_attempts + 1):
            if self.stopping:
                return None
            try:
                self.stats.snapshot_fetches += 1
                async with session.get(self.config.snapshot_url) as response:
                    if response.status == 429:
                        # Rate limited. Binance tells us how long to wait; obey
                        # it exactly rather than guessing, or the next attempt
                        # earns an IP ban rather than another 429.
                        retry_after = float(response.headers.get("Retry-After", backoff))
                        log.warning("snapshot_rate_limited", retry_after=retry_after)
                        await asyncio.sleep(retry_after)
                        continue
                    response.raise_for_status()
                    payload = await response.json()

                snapshot = DepthSnapshot(
                    last_update_id=int(payload["lastUpdateId"]),
                    bids=levels_from_pairs(payload["bids"]),
                    asks=levels_from_pairs(payload["asks"]),
                )
                log.info(
                    "snapshot_fetched",
                    last_update_id=snapshot.last_update_id,
                    bids=len(snapshot.bids),
                    asks=len(snapshot.asks),
                    attempt=attempt,
                )
                return snapshot
            except (TimeoutError, aiohttp.ClientError, KeyError, ValueError) as exc:
                self.stats.snapshot_failures += 1
                log.warning(
                    "snapshot_fetch_failed",
                    attempt=attempt,
                    error=f"{type(exc).__name__}: {exc}",
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * self.config.backoff_multiplier, self.config.max_backoff)

        log.error("snapshot_unavailable", attempts=self.config.max_snapshot_attempts)
        return None

    # -- book publication --------------------------------------------------
    def _build_book_event(self, correlation_id: str) -> OrderBookUpdate | None:
        bids, asks = self.book.top_n(self.config.depth_levels)
        if not bids or not asks:
            return None
        try:
            return OrderBookUpdate(
                symbol=self.config.symbol,
                ts_event=self.stats.last_event_ns or self.clock.now_exchange_ns(),
                ts_ingest=self.clock.now_exchange_ns(),
                source=DataSource.BINANCE_WS,
                correlation_id=correlation_id,
                last_update_id=self.book.last_update_id,
                is_snapshot=False,
                bids=bids,
                asks=asks,
            )
        except ValueError as exc:
            # The schema validator rejects crossed/unordered books. If that
            # fires, our local state is wrong and the honest response is to
            # resync, not to publish a book we know is invalid.
            log.error("book_validation_failed", error=str(exc))
            self.sync.request_resync(ResyncReason.CROSSED_BOOK)
            self.stats.note_resync(ResyncReason.CROSSED_BOOK)
            return None

    async def _resync(self) -> bool:
        """Fetch a snapshot and hand it to the synchroniser. True on success."""
        snapshot = await self.fetch_snapshot()
        if snapshot is None:
            return False
        result = self.sync.apply_snapshot(snapshot)
        if not result.applied:
            log.warning("resync_incomplete", reason=str(result.reason), detail=result.detail)
            return False
        log.info("book_synced", last_update_id=self.book.last_update_id, detail=result.detail)
        return True

    # -- main stream -------------------------------------------------------
    async def stream(self) -> AsyncIterator[OrderBookUpdate | TradeTick]:
        """Connect, resync, and yield events forever. Reconnects on its own."""
        backoff = self.config.initial_backoff

        while not self.stopping:
            try:
                async with websockets.connect(
                    self.config.stream_url,
                    ping_interval=self.config.ws_ping_interval,
                    ping_timeout=self.config.ws_ping_timeout,
                    open_timeout=self.config.ws_open_timeout,
                    max_queue=2048,
                ) as ws:
                    self._ws = ws
                    self.sync.on_connected()
                    backoff = self.config.initial_backoff
                    log.info("ws_connected", url=self.config.stream_url)
                    # Re-measure skew on every (re)connect: a long-lived process
                    # will drift, and a reconnect is the natural checkpoint.
                    await self.clock.maybe_resync(await self._ensure_session())

                    async for event in self._consume(ws):
                        yield event

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.stats.reconnects += 1
                self.sync.on_disconnected()
                # Full jitter: without it, a Binance-side blip reconnects every
                # client on the same schedule and we all retry in lockstep.
                sleep_for = random.uniform(0.0, backoff)
                log.warning(
                    "ws_disconnected",
                    error=f"{type(exc).__name__}: {exc}",
                    reconnect_in=round(sleep_for, 2),
                    reconnects=self.stats.reconnects,
                )
                # Wait on the stop event rather than the clock, so shutdown
                # during backoff is immediate. A plain `asyncio.sleep` here
                # meant SIGTERM could take the full 30s cap to be noticed --
                # long enough for an orchestrator to escalate to SIGKILL and
                # lose the buffered ticks. It also subsumes the redundant
                # `if self.stopping: break` that used to sit above.
                if await self._wait_for_stop(sleep_for):
                    break
                backoff = min(backoff * self.config.backoff_multiplier, self.config.max_backoff)
            finally:
                self._ws = None

    async def _consume(self, ws: ClientConnection) -> AsyncIterator[OrderBookUpdate | TradeTick]:
        """Read one connection's lifetime of messages."""
        loop = asyncio.get_running_loop()
        connected_at = loop.time()
        snapshot_task: asyncio.Task[bool] | None = None

        # try/finally, not a trailing cleanup block: this is an async
        # generator, so a consumer that `break`s out of it never reaches code
        # after the loop. Without this the in-flight snapshot task is orphaned
        # on every early exit -- and the aiohttp session it holds leaks with it.
        try:
            async for raw in ws:
                if self.stopping:
                    return

                # Cycle the connection on our own schedule, before Binance's 24h
                # limit forces it at an inconvenient moment.
                if loop.time() - connected_at > CONNECTION_MAX_AGE_SECONDS:
                    log.info("ws_cycling", age_hours=round((loop.time() - connected_at) / 3600, 1))
                    await ws.close()
                    return

                self.stats.messages += 1
                try:
                    message = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    self.stats.decode_errors += 1
                    continue

                data = message.get("data", message)
                event_type = data.get("e")

                if event_type == "aggTrade":
                    self.stats.trade_messages += 1
                    try:
                        trade = parse_agg_trade(data, self.config.symbol, self.clock)
                    except (KeyError, ValueError) as exc:
                        self.stats.decode_errors += 1
                        log.warning("trade_parse_failed", error=str(exc))
                        continue
                    self.stats.trades_published += 1
                    yield trade
                    continue

                if event_type != "depthUpdate":
                    continue

                self.stats.depth_messages += 1
                try:
                    diff = parse_depth_diff(data)
                except (KeyError, ValueError) as exc:
                    self.stats.decode_errors += 1
                    log.warning("depth_parse_failed", error=str(exc))
                    continue

                self.stats.last_event_ns = (
                    int(data.get("E", 0)) * 1_000_000 or self.clock.now_exchange_ns()
                )
                result = self.sync.submit_diff(diff)

                if result.reason is not None:
                    self.stats.note_resync(result.reason)
                    if self._on_resync is not None:
                        self._on_resync(result.reason)
                    log.warning("resync_required", reason=str(result.reason), detail=result.detail)

                # Fetch the snapshot concurrently: blocking the read loop on an HTTP
                # round trip would drop the very diffs we need to replay over it.
                if self.sync.needs_snapshot and (snapshot_task is None or snapshot_task.done()):
                    snapshot_task = asyncio.create_task(self._resync())

                if result.applied:
                    book_event = self._build_book_event(new_id())
                    if book_event is not None:
                        self.stats.books_published += 1
                        yield book_event

        finally:
            if snapshot_task is not None and not snapshot_task.done():
                snapshot_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await snapshot_task

__all__ = [
    "CONNECTION_MAX_AGE_SECONDS",
    "SNAPSHOT_LIMIT",
    "BinanceConfig",
    "BinanceDepthClient",
    "IngestionStats",
    "parse_agg_trade",
    "parse_depth_diff",
]
