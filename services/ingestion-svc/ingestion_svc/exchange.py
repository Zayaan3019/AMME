"""Exchange clock synchronisation and instrument filters.

Both of these exist because of things measured against the live venue, not
things assumed:

**Clock skew.** On the development machine the local clock ran **+684 ms** ahead
of ``api.binance.com``. Without correction, ``ts_ingest - ts_event`` reported
~734 ms of "staleness" when the true one-way latency was ~67 ms. Every latency
SLO in the system -- including the s.6.4 regression budgets -- would have been
measuring a constant clock offset and calling it network lag. NTP does not save
you here: the host clock can be perfectly correct and still disagree with the
exchange, and it is the *exchange's* clock that timestamps the data.

The correction uses the standard NTP-style offset estimate

.. math:: \\theta = \\frac{(t_1 - t_0) + (t_2 - t_3)}{2}

collapsed to ``(t_local_mid - t_server)`` because Binance returns a single
server timestamp rather than separate receive/transmit stamps. Samples are taken
repeatedly and the **median** is kept -- a mean would be dragged by any single
request that hit a slow path, and the whole point is robustness.

**Instrument filters.** Tick size, lot size and min-notional are fetched from
``/api/v3/exchangeInfo`` rather than hardcoded. Hardcoding them is how a quoting
engine ends up silently emitting orders the venue rejects after a filter change,
and it is free to just ask.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field

import aiohttp

from amme_shared.logging import get_logger

log = get_logger("ingestion.exchange")


@dataclass(slots=True)
class ClockSync:
    """Tracks the local-vs-exchange clock offset.

    ``offset_ns`` is *local minus exchange*: subtract it from a local timestamp
    to express that instant on the exchange's clock.
    """

    rest_base: str = "https://api.binance.com"
    samples: int = 7
    sample_gap_seconds: float = 0.2
    resync_interval_seconds: float = 300.0
    #: Warn above this; a drifting host clock is an operational problem.
    warn_threshold_ms: float = 250.0

    offset_ns: int = 0
    rtt_ms: float = 0.0
    last_sync_monotonic: float = 0.0
    synced: bool = False
    sync_count: int = 0
    history: list[float] = field(default_factory=list)

    @property
    def offset_ms(self) -> float:
        return self.offset_ns / 1e6

    def to_exchange_ns(self, local_ns: int) -> int:
        """Express a local timestamp on the exchange's clock."""
        return local_ns - self.offset_ns

    def now_exchange_ns(self) -> int:
        return time.time_ns() - self.offset_ns

    def staleness_ns(self, event_ns: int, ingest_local_ns: int) -> int:
        """True staleness of an event, with the skew removed."""
        return self.to_exchange_ns(ingest_local_ns) - event_ns

    async def sync(self, session: aiohttp.ClientSession) -> bool:
        """Measure the offset. Returns True if it succeeded."""
        offsets: list[float] = []
        rtts: list[float] = []
        url = f"{self.rest_base}/api/v3/time"

        for _ in range(self.samples):
            try:
                t0 = time.time_ns()
                async with session.get(url) as response:
                    response.raise_for_status()
                    payload = await response.json()
                t1 = time.time_ns()
            except (TimeoutError, aiohttp.ClientError, ValueError) as exc:
                log.warning("clock_sync_sample_failed", error=f"{type(exc).__name__}: {exc}")
                continue

            server_ns = int(payload["serverTime"]) * 1_000_000
            # Midpoint of the round trip is our best estimate of "now" locally
            # at the instant the server stamped its reply.
            local_mid_ns = (t0 + t1) // 2
            offsets.append(float(local_mid_ns - server_ns))
            rtts.append((t1 - t0) / 1e6)
            await asyncio.sleep(self.sample_gap_seconds)

        if not offsets:
            log.error("clock_sync_failed", samples_attempted=self.samples)
            return False

        self.offset_ns = int(statistics.median(offsets))
        self.rtt_ms = statistics.median(rtts)
        self.last_sync_monotonic = time.monotonic()
        self.synced = True
        self.sync_count += 1
        self.history.append(self.offset_ms)
        del self.history[:-64]

        level = log.warning if abs(self.offset_ms) > self.warn_threshold_ms else log.info
        level(
            "clock_synced",
            offset_ms=round(self.offset_ms, 1),
            rtt_ms=round(self.rtt_ms, 1),
            one_way_latency_ms=round(self.rtt_ms / 2, 1),
            samples=len(offsets),
        )
        return True

    def is_stale(self) -> bool:
        if not self.synced:
            return True
        return (time.monotonic() - self.last_sync_monotonic) > self.resync_interval_seconds

    async def maybe_resync(self, session: aiohttp.ClientSession) -> None:
        """Re-measure if the last measurement has aged out. Cheap to call often."""
        if self.is_stale():
            await self.sync(session)


@dataclass(frozen=True, slots=True)
class InstrumentFilters:
    """The venue's trading rules for one symbol."""

    symbol: str
    tick_size: float
    lot_size: float
    min_qty: float
    min_notional: float
    max_notional: float = 0.0
    status: str = "TRADING"

    @property
    def tradable(self) -> bool:
        return self.status == "TRADING"

    def validate_order(self, price: float, size: float) -> str | None:
        """``None`` if the venue would accept this order, else why it would not.

        Called before every quote goes out. The min-notional check in particular
        is not theoretical: BTCUSDT requires $5 notional, so a quote size chosen
        for a cheap symbol is silently rejected on an expensive one.
        """
        if size < self.min_qty:
            return f"size {size} below min_qty {self.min_qty}"
        notional = price * size
        if notional < self.min_notional:
            return f"notional {notional:.2f} below min_notional {self.min_notional}"
        if self.max_notional > 0 and notional > self.max_notional:
            return f"notional {notional:.2f} above max_notional {self.max_notional}"
        return None


async def fetch_instrument_filters(
    session: aiohttp.ClientSession, symbol: str, rest_base: str = "https://api.binance.com"
) -> InstrumentFilters | None:
    """GET /api/v3/exchangeInfo and extract the filters we care about."""
    url = f"{rest_base}/api/v3/exchangeInfo?symbol={symbol.upper()}"
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            payload = await response.json()
    except (TimeoutError, aiohttp.ClientError, ValueError) as exc:
        log.warning("exchange_info_failed", symbol=symbol, error=f"{type(exc).__name__}: {exc}")
        return None

    symbols = payload.get("symbols") or []
    if not symbols:
        log.warning("exchange_info_empty", symbol=symbol)
        return None

    entry = symbols[0]
    by_type = {f["filterType"]: f for f in entry.get("filters", [])}
    price_filter = by_type.get("PRICE_FILTER", {})
    lot_filter = by_type.get("LOT_SIZE", {})
    notional_filter = by_type.get("NOTIONAL") or by_type.get("MIN_NOTIONAL") or {}

    filters = InstrumentFilters(
        symbol=entry.get("symbol", symbol.upper()),
        tick_size=float(price_filter.get("tickSize", 0.01)),
        lot_size=float(lot_filter.get("stepSize", 0.00001)),
        min_qty=float(lot_filter.get("minQty", 0.00001)),
        min_notional=float(notional_filter.get("minNotional", 0.0)),
        max_notional=float(notional_filter.get("maxNotional", 0.0)),
        status=entry.get("status", "TRADING"),
    )
    log.info(
        "instrument_filters",
        symbol=filters.symbol,
        tick_size=filters.tick_size,
        lot_size=filters.lot_size,
        min_notional=filters.min_notional,
        status=filters.status,
    )
    return filters


__all__ = ["ClockSync", "InstrumentFilters", "fetch_instrument_filters"]
