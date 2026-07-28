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
from itertools import pairwise
from dataclasses import dataclass

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

    async def load(
        self, symbol: str, start_ns: int, end_ns: int, limit: int = 2_000_000
    ) -> list[tuple[OrderBookUpdate, list[TradeTick]]]:
        """Materialise a causally-merged window."""
        await self.connect()
        books = await self._store.read_books(symbol, start_ns, end_ns, limit)

        async with self._store.pool.acquire() as conn:
            from datetime import UTC, datetime

            records = await conn.fetch(
                "SELECT * FROM trades WHERE symbol = $1 AND ts_event >= $2"
                " AND ts_event < $3 ORDER BY ts_event ASC, trade_id ASC LIMIT $4",
                symbol,
                datetime.fromtimestamp(start_ns / 1e9, tz=UTC),
                datetime.fromtimestamp(end_ns / 1e9, tz=UTC),
                limit,
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
    "merge_causally",
    "replay_quality",
]
