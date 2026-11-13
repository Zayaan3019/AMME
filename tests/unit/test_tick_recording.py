"""Tick recording survives a TimescaleDB outage, and says so while it lasts.

The bug these pin: when TimescaleDB was down at ingestion start-up, recording
switched off for the life of the process, every buffered tick was discarded on
each flush, and ``/health`` had no database check at all -- so the stack stayed
"healthy" while recording nothing. Found on a stack that had run that way for
months after the database container lost its host port.
"""

from __future__ import annotations

import inspect
from typing import ClassVar

import pytest

from amme_shared.config import Settings
from amme_shared.events import DataSource, OrderBookUpdate, PriceLevel, TradeTick


class FakeStore:
    """Stands in for TickStore. Class-level switches drive the outage."""

    connect_ok = False
    write_ok = True
    books: ClassVar[list[OrderBookUpdate]] = []
    trades: ClassVar[list[TradeTick]] = []

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    async def connect(self) -> None:
        if not FakeStore.connect_ok:
            raise ConnectionRefusedError("timescaledb down")

    async def close(self) -> None:
        return None

    async def ping(self) -> bool:
        return True

    async def write_books(self, books):
        if not FakeStore.write_ok:
            raise ConnectionResetError("lost connection")
        FakeStore.books.extend(books)
        return len(books)

    async def write_trades(self, trades):
        if not FakeStore.write_ok:
            raise ConnectionResetError("lost connection")
        FakeStore.trades.extend(trades)
        return len(trades)


def _book(i: int) -> OrderBookUpdate:
    return OrderBookUpdate(
        symbol="BTCUSDT", ts_event=i, ts_ingest=i, source=DataSource.BINANCE_WS,
        bids=[PriceLevel(price=100.0, size=1.0)], asks=[PriceLevel(price=100.01, size=1.0)],
        last_update_id=i, is_snapshot=False,
    )


def _trade(i: int) -> TradeTick:
    return TradeTick(symbol="BTCUSDT", ts_event=i, ts_ingest=i, source=DataSource.BINANCE_WS,
                     price=100.0, size=0.1, trade_id=i)


@pytest.fixture
def ingestion(monkeypatch):
    import ingestion_svc.service as mod

    FakeStore.connect_ok, FakeStore.write_ok = False, True
    FakeStore.books, FakeStore.trades = [], []
    monkeypatch.setattr(mod, "TickStore", FakeStore)
    svc = mod.IngestionService(Settings())
    assert svc._persist, "these tests assume persist_ticks is on by default"
    return svc


def _retry_now(svc) -> None:
    svc._persist_retry_at = 0.0


class TestRecordingThroughAnOutage:
    async def test_ticks_are_kept_while_the_database_is_down(self, ingestion):
        ingestion._book_buffer.extend(_book(i) for i in range(5))
        ingestion._trade_buffer.extend(_trade(i) for i in range(3))
        await ingestion._flush()
        assert ingestion.store is None
        assert ingestion._buffered() == 8, "a flush with no database used to discard these"
        ok, detail = await ingestion._check_db()
        assert not ok and "recording paused" in detail and "buffered=8" in detail

    async def test_reconnects_and_writes_the_backlog(self, ingestion):
        ingestion._book_buffer.extend(_book(i) for i in range(5))
        await ingestion._flush()
        FakeStore.connect_ok = True
        _retry_now(ingestion)
        ingestion._book_buffer.append(_book(5))
        await ingestion._flush()
        assert [b.last_update_id for b in FakeStore.books] == [0, 1, 2, 3, 4, 5]
        assert ingestion._buffered() == 0
        ok, detail = await ingestion._check_db()
        assert ok and "written=6" in detail

    async def test_a_failed_write_is_retried_not_lost(self, ingestion):
        FakeStore.connect_ok = True
        ingestion._book_buffer.extend(_book(i) for i in range(3))
        FakeStore.write_ok = False
        await ingestion._flush()
        assert ingestion._buffered() == 3
        ok, _ = await ingestion._check_db()
        assert not ok
        FakeStore.write_ok = True
        await ingestion._flush()
        assert ingestion._buffered() == 3, "retries wait for the backoff period"
        _retry_now(ingestion)
        await ingestion._flush()
        assert len(FakeStore.books) == 3 and ingestion._buffered() == 0
        ok, _ = await ingestion._check_db()
        assert ok, "health recovers once a write succeeds"

    async def test_buffers_are_bounded_and_drops_are_counted(self, ingestion, monkeypatch):
        import ingestion_svc.service as mod

        monkeypatch.setattr(mod, "PERSIST_MAX_BOOKS", 4)
        ingestion._book_buffer.extend(_book(i) for i in range(10))
        await ingestion._flush()
        assert [b.last_update_id for b in ingestion._book_buffer] == [6, 7, 8, 9]
        assert ingestion._persist_dropped == 6

    async def test_connection_attempts_are_spaced(self, ingestion):
        calls = 0
        original = FakeStore.connect

        async def counting(self):
            nonlocal calls
            calls += 1
            await original(self)

        FakeStore.connect = counting
        try:
            for _ in range(5):
                await ingestion._ensure_store()
        finally:
            FakeStore.connect = original
        assert calls == 1, "a down database must not be hammered on every flush"

    def test_the_check_is_registered_even_when_the_first_connect_fails(self):
        from ingestion_svc.service import IngestionService

        startup = inspect.getsource(IngestionService.startup)
        registration = startup.index('add_health_check("timescaledb"')
        assert "critical=False" in startup[registration:registration + 120]
        # Not inside the try/except around the connection attempt.
        assert "await self._ensure_store()" in startup


class TestRedisStillLoading:
    """A restarted Redis replays its append-only log before serving commands.

    Services used to crash on their first command (``BusyLoadingError``) and
    lean on Docker's restart policy; compose then gave up on the services that
    depend on them. Seen on a stack restart after eight hours of streams.
    """

    class _LoadingRedis:
        def __init__(self, failures: int) -> None:
            self.failures = failures
            self.calls = 0

        async def ping(self) -> bool:
            from redis.exceptions import BusyLoadingError

            self.calls += 1
            if self.calls <= self.failures:
                raise BusyLoadingError("Redis is loading the dataset in memory")
            return True

    async def test_waits_until_loading_finishes(self):
        from amme_shared.streams import RedisEventBus

        bus = RedisEventBus("redis://localhost:6379/0")
        bus._redis = self._LoadingRedis(failures=3)
        await bus.wait_ready(timeout_s=5.0, interval_s=0.01)
        assert bus._redis.calls == 4

    async def test_gives_up_after_the_timeout(self):
        from redis.exceptions import BusyLoadingError

        from amme_shared.streams import RedisEventBus

        bus = RedisEventBus("redis://localhost:6379/0")
        bus._redis = self._LoadingRedis(failures=10_000)
        with pytest.raises(BusyLoadingError):
            await bus.wait_ready(timeout_s=0.05, interval_s=0.01)

    async def test_startup_waits_for_the_bus_first(self):
        from amme_shared.service import StreamService

        order: list[str] = []

        class Bus:
            async def wait_ready(self) -> None:
                order.append("wait_ready")

        class Svc(StreamService):
            name = "probe"

            def __init__(self) -> None:
                super().__init__(Settings())
                self.bus = Bus()

            async def startup(self) -> None:
                order.append("startup")

            async def run_loop(self) -> None:
                await self.wait_for_stop()

        app = Svc().build_app()
        async with app.router.lifespan_context(app):
            pass
        assert order == ["wait_ready", "startup"]


class TestDegradedHealth:
    async def test_a_failing_non_critical_check_degrades_instead_of_failing(self):
        from amme_shared.service import StreamService

        class Svc(StreamService):
            name = "probe"

            async def run_loop(self) -> None:
                return None

        svc = Svc(Settings())
        svc._ready = True

        async def ok():
            return True, "fine"

        async def bad():
            return False, "db down"

        svc.add_health_check("core", ok)
        svc.add_health_check("recorder", bad, critical=False)
        status, details = await svc._collect_health()
        assert status == "degraded"
        assert details["recorder"].startswith("DEGRADED:")

        svc.add_health_check("redis", bad)
        status, details = await svc._collect_health()
        assert status == "unhealthy"
        assert details["redis"].startswith("FAIL:")
