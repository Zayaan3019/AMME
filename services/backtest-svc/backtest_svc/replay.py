"""Backtest on *recorded live market data*, not only on the simulator.

The regression harness is calibrated to measured live statistics, but until now
every published number came from synthetic data. That is a real limitation: a
simulator can only reproduce the dynamics you thought to model, so a strategy
tuned on it is partly tuned to your own assumptions. Replaying the ticks
``ingestion-svc`` already persists closes that gap.

**What replay does and does not prove.**

It removes the generative model from the loop: the book dynamics, the trade
arrivals, the queue behaviour and the volatility clustering are whatever the
exchange actually did. What it cannot remove is the *counterfactual* — our
orders were never in that book, so fills are still simulated by the same
queue-position model. Replay upgrades the market data from synthetic to real;
it does not turn a paper backtest into a live P&L.

**Causal ordering is the whole job here.** Books and trades arrive on separate
streams and are stored in separate tables. Merging them wrongly is the easiest
way to build a replay that leaks the future: attach a trade to the book
*after* it executed against and the strategy sees the consequence of flow it
should not yet know about. :func:`merge_causally` is a pure function with that
one responsibility, so it can be tested exhaustively.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from amme_shared.db import TickStore
from amme_shared.events import OrderBookUpdate, TradeTick
from amme_shared.logging import get_logger

log = get_logger("backtest.replay")


def merge_causally(
    books: Sequence[OrderBookUpdate], trades: Sequence[TradeTick]
) -> Iterator[tuple[OrderBookUpdate, list[TradeTick]]]:
    """Pair each book with the trades that happened *before or at* it.

    Contract, matching :class:`~amme_core.simulator.MarketSimulator`: step ``t``
    yields ``(book_t, trades_in (t-1, t])``. The engine then matches those
    trades against quotes resting from ``t-1`` and only afterwards lets the
    strategy see ``book_t`` — so no trade can influence a quote that was placed
    before the trade occurred.

    Trades outside the book window (before the first book, after the last) are
    dropped rather than bunched onto the nearest one: attaching a burst of
    unrelated flow to a single book would fabricate a toxicity spike that never
    happened.
    """
    if not books:
        return

    ordered_books = sorted(books, key=lambda b: (b.ts_event, b.last_update_id))
    ordered_trades = sorted(trades, key=lambda t: (t.ts_event, t.trade_id))
    trade_times = [t.ts_event for t in ordered_trades]

    previous_ts = ordered_books[0].ts_event - 1
    for book in ordered_books:
        # (previous_ts, book.ts_event] -- half-open below, closed above.
        lo = bisect.bisect_right(trade_times, previous_ts)
        hi = bisect.bisect_right(trade_times, book.ts_event)
        yield book, list(ordered_trades[lo:hi])
        previous_ts = book.ts_event


@dataclass(frozen=True, slots=True)
class RecordedSegment:
    """One contiguous stretch of the recording (see ``segments``)."""

    window: ReplayWindow
    n_books: int

    @property
    def end_exclusive_ns(self) -> int:
        """Exclusive upper bound that still includes the last book.

        Stored timestamps have microsecond resolution, so ``end + 1 ns``
        rounds back to ``end`` and ``ts_event < end`` would drop the final row.
        """
        return self.window.end_ns + 1_000


@dataclass(frozen=True, slots=True)
class ReplayWindow:
    """A contiguous slice of recorded data."""

    symbol: str
    start_ns: int
    end_ns: int

    @property
    def duration_seconds(self) -> float:
        return (self.end_ns - self.start_ns) / 1e9


class TimescaleReplaySource:
    """Streams recorded ticks out of TimescaleDB.

    Loads a window into memory rather than cursoring: an ablation runs the same
    window once per arm, and re-querying per arm would make the comparison
    depend on database timing. Loading once also guarantees every arm sees
    *byte-identical* input, which is the property the paired comparison rests
    on.
    """

    def __init__(self, dsn: str) -> None:
        self._store = TickStore(dsn)
        self._connected = False

    async def connect(self) -> None:
        if not self._connected:
            await self._store.connect()
            self._connected = True

    async def close(self) -> None:
        if self._connected:
            await self._store.close()
            self._connected = False

    async def available(self, symbol: str) -> ReplayWindow | None:
        """The full recorded range for ``symbol``, or ``None`` if empty."""
        await self.connect()
        async with self._store.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT min(ts_event) AS lo, max(ts_event) AS hi, count(*) AS n"
                " FROM book_snapshots WHERE symbol = $1",
                symbol,
            )
        if row is None or row["n"] == 0 or row["lo"] is None:
            return None
        return ReplayWindow(
            symbol=symbol,
            start_ns=int(row["lo"].timestamp() * 1e9),
            end_ns=int(row["hi"].timestamp() * 1e9),
        )

    async def segments(
        self, symbol: str, max_gap_seconds: float = 5.0
    ) -> list[RecordedSegment]:
        """Contiguous stretches of the recording, found in SQL from timestamps.

        Same rule as :func:`split_contiguous` (a gap longer than
        ``max_gap_seconds`` starts a new segment), but computed by the
        database with a window function, so finding the segment to train or
        replay on never pulls the whole recording into memory. That used to
        be the first step, and after months of recording it hit the pool's
        30 s command timeout.
        """
        await self.connect()
        async with self._store.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH g AS (
                    SELECT ts_event,
                           ts_event - lag(ts_event) OVER (ORDER BY ts_event) AS gap
                    FROM book_snapshots WHERE symbol = $1
                ), s AS (
                    SELECT ts_event,
                           sum(CASE WHEN gap IS NULL OR gap > make_interval(secs => $2)
                                    THEN 1 ELSE 0 END) OVER (ORDER BY ts_event) AS seg
                    FROM g
                )
                SELECT seg, min(ts_event) AS lo, max(ts_event) AS hi, count(*) AS n
                FROM s GROUP BY seg ORDER BY seg
                """,
                symbol,
                float(max_gap_seconds),
            )
        return [
            RecordedSegment(
                window=ReplayWindow(
                    symbol=symbol,
                    start_ns=int(r["lo"].timestamp() * 1e9),
                    end_ns=int(r["hi"].timestamp() * 1e9),
                ),
                n_books=int(r["n"]),
            )
            for r in rows
        ]

    async def load(
        self,
        symbol: str,
        start_ns: int,
        end_ns: int,
        limit: int = 2_000_000,
        chunk_seconds: float | None = None,
    ) -> list[tuple[OrderBookUpdate, list[TradeTick]]]:
        """Materialise a causally-merged window.

        ``chunk_seconds`` reads the window in slices so that no single query
        runs into the pool's command timeout on a long recording. Books and
        trades are gathered across all slices and merged once at the end, so
        a trade near a slice boundary still attaches to the right book.
        """
        await self.connect()
        if chunk_seconds and chunk_seconds > 0:
            step = int(chunk_seconds * 1e9)
            bounds = [(a, min(a + step, end_ns)) for a in range(start_ns, end_ns, step)]
        else:
            bounds = [(start_ns, end_ns)]

        books: list[OrderBookUpdate] = []
        records: list[Any] = []
        from datetime import UTC, datetime

        for a, b in bounds:
            books.extend(await self._store.read_books(symbol, a, b, limit))
            async with self._store.pool.acquire() as conn:
                records.extend(
                    await conn.fetch(
                        "SELECT * FROM trades WHERE symbol = $1 AND ts_event >= $2"
                        " AND ts_event < $3 ORDER BY ts_event ASC, trade_id ASC LIMIT $4",
                        symbol,
                        datetime.fromtimestamp(a / 1e9, tz=UTC),
                        datetime.fromtimestamp(b / 1e9, tz=UTC),
                        limit,
                    )
                )

        from amme_shared.events import AggressorSide, DataSource

        trades = [
            TradeTick(
                symbol=r["symbol"],
                ts_event=int(r["ts_event"].timestamp() * 1e9),
                ts_ingest=int(r["ts_ingest"].timestamp() * 1e9),
                source=DataSource(r["source"]),
                price=float(r["price"]),
                size=float(r["size"]),
                aggressor=AggressorSide(r["aggressor"]),
                trade_id=int(r["trade_id"]),
            )
            for r in records
        ]

        merged = list(merge_causally(books, trades))
        attached = sum(len(t) for _, t in merged)
        log.info(
            "replay_loaded",
            symbol=symbol,
            books=len(books),
            trades=len(trades),
            attached=attached,
            dropped=len(trades) - attached,
            seconds=round((end_ns - start_ns) / 1e9, 1),
        )
        return merged


def split_contiguous(
    window: Sequence[tuple[OrderBookUpdate, list[TradeTick]]],
    max_gap_seconds: float = 5.0,
) -> list[list[tuple[OrderBookUpdate, list[TradeTick]]]]:
    """Split a replay window at recording gaps.

    A capture accumulated over days contains the stack's own downtime: the
    first real recording here spanned 8,820 seconds but hid an **81-minute
    gap** where the containers were being rebuilt. Backtesting straight across
    such a gap is not merely noisy, it is wrong -- the strategy carries
    inventory across a discontinuity, every rolling statistic (sigma, VPIN, the
    OFI z-score) treats two unrelated market states as consecutive, and the
    resulting P&L is attributed to a period that never happened.

    ``max_gap_seconds`` defaults to 5s: fifty missed updates at the 100ms
    cadence, comfortably beyond ordinary jitter and far short of a restart.
    """
    if not window:
        return []

    segments: list[list[tuple[OrderBookUpdate, list[TradeTick]]]] = [[window[0]]]
    for previous, current in pairwise(window):
        gap = (current[0].ts_event - previous[0].ts_event) / 1e9
        if gap > max_gap_seconds:
            segments.append([current])
        else:
            segments[-1].append(current)
    return segments


def largest_contiguous(
    window: Sequence[tuple[OrderBookUpdate, list[TradeTick]]],
    max_gap_seconds: float = 5.0,
) -> list[tuple[OrderBookUpdate, list[TradeTick]]]:
    """The longest uninterrupted stretch of the recording.

    Preferred over stitching segments together: concatenation would reintroduce
    exactly the discontinuity :func:`split_contiguous` exists to remove, just
    with the timestamps hidden.
    """
    segments = split_contiguous(window, max_gap_seconds)
    if not segments:
        return []
    return max(segments, key=len)


def replay_quality(
    window: Sequence[tuple[OrderBookUpdate, list[TradeTick]]],
) -> dict[str, float]:
    """Sanity statistics for a replay window.

    A replay is only worth running if the recording is intact. Gaps, crossed
    books or a collapsed update rate mean the capture was degraded, and a
    backtest over it would produce numbers that look real and are not.
    """
    if not window:
        return {"books": 0.0}

    books = [b for b, _ in window]
    span = (books[-1].ts_event - books[0].ts_event) / 1e9
    gaps = [(b.ts_event - a.ts_event) / 1e9 for a, b in pairwise(books)]
    largest_gap = max(gaps) if gaps else 0.0
    crossed = sum(
        1 for b in books if b.bids and b.asks and b.bids[0].price >= b.asks[0].price
    )
    return {
        "books": float(len(books)),
        "trades": float(sum(len(t) for _, t in window)),
        "seconds": span,
        "update_rate_hz": len(books) / span if span > 0 else 0.0,
        "largest_gap_seconds": largest_gap,
        "crossed_books": float(crossed),
    }


__all__ = [
    "ReplayWindow",
    "TimescaleReplaySource",
    "largest_contiguous",
    "merge_causally",
    "replay_quality",
    "split_contiguous",
]
