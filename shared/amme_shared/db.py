"""TimescaleDB access.

Tick storage is append-only and write-heavy, which is exactly the workload
hypertables are for: Timescale partitions by time automatically, so the
insert path stays O(log n) on a small recent chunk instead of degrading against
one ever-growing B-tree.

Every write goes through ``executemany`` on a batch. Per-row inserts at 10
books/sec x 20 levels would spend more time in round-trips than in Postgres.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from types import TracebackType
from typing import Any

import asyncpg

from amme_shared.events import DataSource, OrderBookUpdate, PriceLevel, TradeTick
from amme_shared.logging import get_logger

log = get_logger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS book_snapshots (
    ts_event      TIMESTAMPTZ      NOT NULL,
    ts_ingest     TIMESTAMPTZ      NOT NULL,
    symbol        TEXT             NOT NULL,
    source        TEXT             NOT NULL,
    last_update_id BIGINT          NOT NULL,
    is_snapshot   BOOLEAN          NOT NULL DEFAULT FALSE,
    correlation_id TEXT            NOT NULL,
    bids          JSONB            NOT NULL,
    asks          JSONB            NOT NULL,
    mid_price     DOUBLE PRECISION,
    spread        DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS trades (
    ts_event      TIMESTAMPTZ      NOT NULL,
    ts_ingest     TIMESTAMPTZ      NOT NULL,
    symbol        TEXT             NOT NULL,
    source        TEXT             NOT NULL,
    trade_id      BIGINT           NOT NULL,
    price         DOUBLE PRECISION NOT NULL,
    size          DOUBLE PRECISION NOT NULL,
    aggressor     TEXT             NOT NULL
);

CREATE TABLE IF NOT EXISTS fills (
    ts_event      TIMESTAMPTZ      NOT NULL,
    symbol        TEXT             NOT NULL,
    order_id      TEXT             NOT NULL,
    side          TEXT             NOT NULL,
    price         DOUBLE PRECISION NOT NULL,
    size          DOUBLE PRECISION NOT NULL,
    fee           DOUBLE PRECISION NOT NULL,
    inventory_after DOUBLE PRECISION NOT NULL,
    cash_after    DOUBLE PRECISION NOT NULL,
    correlation_id TEXT            NOT NULL
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    ts_event       TIMESTAMPTZ     NOT NULL,
    symbol         TEXT            NOT NULL,
    strategy       TEXT            NOT NULL DEFAULT 'amme',
    inventory      DOUBLE PRECISION NOT NULL,
    cash           DOUBLE PRECISION NOT NULL,
    mid_price      DOUBLE PRECISION NOT NULL,
    realized_pnl   DOUBLE PRECISION NOT NULL,
    unrealized_pnl DOUBLE PRECISION NOT NULL,
    total_pnl      DOUBLE PRECISION NOT NULL,
    n_fills        INTEGER         NOT NULL DEFAULT 0
);
"""

# Run separately: hypertable conversion fails harmlessly if the extension is
# absent (e.g. plain postgres in a lightweight CI job), and we want the tables
# to exist regardless.
HYPERTABLE_SQL = """
SELECT create_hypertable('book_snapshots', 'ts_event', if_not_exists => TRUE);
SELECT create_hypertable('trades', 'ts_event', if_not_exists => TRUE);
SELECT create_hypertable('fills', 'ts_event', if_not_exists => TRUE);
SELECT create_hypertable('portfolio_snapshots', 'ts_event', if_not_exists => TRUE);
"""

INDEX_SQL = """
CREATE INDEX IF NOT EXISTS ix_book_symbol_ts ON book_snapshots (symbol, ts_event DESC);
CREATE INDEX IF NOT EXISTS ix_trades_symbol_ts ON trades (symbol, ts_event DESC);
CREATE INDEX IF NOT EXISTS ix_fills_symbol_ts ON fills (symbol, ts_event DESC);
CREATE INDEX IF NOT EXISTS ix_portfolio_symbol_ts ON portfolio_snapshots (symbol, ts_event DESC);
"""


def _levels_to_json(levels: Sequence[PriceLevel]) -> str:
    return json.dumps([[lv.price, lv.size] for lv in levels])


def _json_to_levels(payload: Any) -> list[PriceLevel]:
    raw = json.loads(payload) if isinstance(payload, str) else payload
    return [PriceLevel(price=float(p), size=float(s)) for p, s in raw]


class TickStore:
    """Async Postgres/TimescaleDB writer + replay reader."""

    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 8) -> None:
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._pool: asyncpg.Pool | None = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("TickStore.connect() has not been awaited")
        return self._pool

    async def connect(self) -> None:
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(
            self._dsn, min_size=self._min_size, max_size=self._max_size, command_timeout=30
        )
        await self.migrate()

    async def migrate(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)
            for statement in filter(None, (s.strip() for s in HYPERTABLE_SQL.split(";"))):
                try:
                    await conn.execute(statement)
                except asyncpg.PostgresError as exc:
                    log.warning("hypertable_skipped", statement=statement[:60], error=str(exc))
            await conn.execute(INDEX_SQL)
        log.info("db_migrated")

    async def ping(self) -> bool:
        async with self.pool.acquire() as conn:
            return bool(await conn.fetchval("SELECT 1"))

    # -- writes ------------------------------------------------------------
    async def write_books(self, books: Sequence[OrderBookUpdate]) -> int:
        if not books:
            return 0
        rows = [
            (
                _ts(b.ts_event),
                _ts(b.ts_ingest),
                b.symbol,
                b.source.value,
                b.last_update_id,
                b.is_snapshot,
                b.correlation_id,
                _levels_to_json(b.bids),
                _levels_to_json(b.asks),
                b.mid_price,
                b.spread,
            )
            for b in books
        ]
        async with self.pool.acquire() as conn:
            await conn.executemany(
                "INSERT INTO book_snapshots (ts_event, ts_ingest, symbol, source, last_update_id,"
                " is_snapshot, correlation_id, bids, asks, mid_price, spread)"
                " VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9::jsonb,$10,$11)",
                rows,
            )
        return len(rows)

    async def write_trades(self, trades: Sequence[TradeTick]) -> int:
        if not trades:
            return 0
        rows = [
            (
                _ts(t.ts_event), _ts(t.ts_ingest), t.symbol, t.source.value,
                t.trade_id, t.price, t.size, t.aggressor.value,
            )
            for t in trades
        ]
        async with self.pool.acquire() as conn:
            await conn.executemany(
                "INSERT INTO trades (ts_event, ts_ingest, symbol, source, trade_id, price, size,"
                " aggressor) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
                rows,
            )
        return len(rows)

    # -- replay ------------------------------------------------------------
    async def read_books(
        self, symbol: str, start_ns: int, end_ns: int, limit: int = 1_000_000
    ) -> list[OrderBookUpdate]:
        """Chronological replay of stored books. Feeds `backtest-svc --source=replay`."""
        async with self.pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT * FROM book_snapshots WHERE symbol = $1 AND ts_event >= $2"
                " AND ts_event < $3 ORDER BY ts_event ASC, last_update_id ASC LIMIT $4",
                symbol, _ts(start_ns), _ts(end_ns), limit,
            )
        return [
            OrderBookUpdate(
                symbol=r["symbol"],
                ts_event=_ns(r["ts_event"]),
                ts_ingest=_ns(r["ts_ingest"]),
                source=DataSource(r["source"]),
                correlation_id=r["correlation_id"],
                last_update_id=r["last_update_id"],
                is_snapshot=r["is_snapshot"],
                bids=_json_to_levels(r["bids"]),
                asks=_json_to_levels(r["asks"]),
            )
            for r in records
        ]

    async def count_books(self, symbol: str) -> int:
        async with self.pool.acquire() as conn:
            value = await conn.fetchval(
                "SELECT count(*) FROM book_snapshots WHERE symbol = $1", symbol
            )
        return int(value or 0)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def __aenter__(self) -> TickStore:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()


def _ts(ns: int) -> Any:
    """epoch-ns -> aware datetime, for asyncpg's TIMESTAMPTZ codec."""
    from datetime import UTC, datetime

    return datetime.fromtimestamp(ns / 1e9, tz=UTC)


def _ns(value: Any) -> int:
    return int(value.timestamp() * 1e9)


__all__ = ["HYPERTABLE_SQL", "INDEX_SQL", "SCHEMA_SQL", "TickStore"]
