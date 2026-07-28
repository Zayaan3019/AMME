"""Regression tests for defects found by auditing the running system.

Each class pins one bug that was live in the codebase. The docstrings record
what the defect was and why it mattered, because the *reason* a test exists is
the part that stops someone deleting it later.

The common thread: every one of these failed silently. None raised, none logged
an error, and several left the system reporting healthy while producing wrong
numbers. That is what makes them worth permanent tests rather than a changelog
entry.
"""

from __future__ import annotations

import math

import pytest

from amme_core.matching import MatchingEngine, MatchingParams, Portfolio
from amme_core.performance import PnLPoint, compute_metrics
from amme_core.risk import (
    BreakerTrigger,
    ProposedQuote,
    RiskEngine,
    RiskInputs,
    RiskParams,
    RiskVerdict,
)
from amme_shared.events import AggressorSide, BaseEvent, Side, TradeTick
from amme_shared.streams import UnhandledEventTypeError, _decode, _encode


class TestVolumeConservation:
    """A trade could fill more of our size than the aggressor actually had.

    ``on_trade`` assigned ``order.remaining`` on a through-fill without drawing
    down the trade's volume. A 0.001 BTC print walking through two 5 BTC orders
    filled 10 BTC -- a 10,000x over-fill. Every backtest P&L, fill count, fill
    ratio and Sharpe in the repository was computed on top of this.
    """

    def _engine(self) -> MatchingEngine:
        return MatchingEngine(MatchingParams(latency_ns=0, maker_fee_bps=0.0))

    def test_through_fill_cannot_exceed_trade_size(self):
        e = self._engine()
        e.place(Side.BID, 100.0, 5.0, level_size=0.0, ts_ns=0)
        e.place(Side.BID, 99.0, 5.0, level_size=0.0, ts_ns=0)
        fills = e.on_trade(98.0, 0.001, AggressorSide.SELL, ts_ns=1)
        assert sum(f.size for f in fills) == pytest.approx(0.001)

    def test_queue_fill_cannot_exceed_trade_size(self):
        e = self._engine()
        e.place(Side.BID, 100.0, 5.0, level_size=0.0, ts_ns=0)
        fills = e.on_trade(100.0, 0.25, AggressorSide.SELL, ts_ns=1)
        assert sum(f.size for f in fills) == pytest.approx(0.25)

    @pytest.mark.parametrize("trade_size", [0.001, 0.5, 3.0, 11.0])
    def test_conservation_across_sizes(self, trade_size):
        e = self._engine()
        e.place(Side.BID, 100.0, 4.0, level_size=0.0, ts_ns=0)
        e.place(Side.BID, 99.5, 4.0, level_size=0.0, ts_ns=0)
        filled = sum(f.size for f in e.on_trade(99.0, trade_size, AggressorSide.SELL, 1))
        assert filled <= trade_size + 1e-12
        assert filled == pytest.approx(min(trade_size, 8.0))

    def test_price_priority_is_deterministic(self):
        """Fills must go to the most aggressive order first, regardless of the
        order quotes were placed in -- otherwise attribution depends on
        insertion order rather than exchange rules."""
        for order_of_placement in ((99.0, 100.0), (100.0, 99.0)):
            e = self._engine()
            for price in order_of_placement:
                e.place(Side.BID, price, 1.0, level_size=0.0, ts_ns=0)
            fills = e.on_trade(98.0, 1.0, AggressorSide.SELL, ts_ns=1)
            assert len(fills) == 1
            assert fills[0].price == 100.0, "best bid must fill first"

    def test_ask_side_conserves_too(self):
        e = self._engine()
        e.place(Side.ASK, 101.0, 5.0, level_size=0.0, ts_ns=0)
        e.place(Side.ASK, 102.0, 5.0, level_size=0.0, ts_ns=0)
        fills = e.on_trade(103.0, 0.002, AggressorSide.BUY, ts_ns=1)
        assert sum(f.size for f in fills) == pytest.approx(0.002)
        assert fills[0].price == 101.0, "best ask must fill first"

    def test_non_finite_trade_rejected(self):
        e = self._engine()
        with pytest.raises(ValueError, match="non-finite"):
            e.on_trade(float("nan"), 1.0, AggressorSide.SELL, ts_ns=1)


class TestRiskFailsClosedOnNaN:
    """NaN defeated every bound check in the risk engine.

    Every IEEE-754 comparison against NaN is False, so ``abs(q) >= q_unwind``,
    ``q + size > q_max`` and the notional cap all silently passed. A single NaN
    inventory turned the entire risk layer into a no-op *while it continued to
    return APPROVE* -- indistinguishable from healthy operation.
    """

    QUOTE = ProposedQuote(99.0, 101.0, 1.0, 1.0, half_spread=1.0)

    @pytest.mark.parametrize("field", ["inventory", "mid_price", "total_pnl", "vpin"])
    def test_non_finite_input_halts(self, field):
        engine = RiskEngine(RiskParams(q_max=0.05))
        kwargs = {"inventory": 0.0, "mid_price": 100.0, field: float("nan")}
        outcome = engine.evaluate(self.QUOTE, RiskInputs(**kwargs))
        assert outcome.verdict is RiskVerdict.HALT
        assert outcome.trigger is BreakerTrigger.INVALID_INPUT
        assert not outcome.actionable

    def test_infinite_input_halts(self):
        engine = RiskEngine(RiskParams(q_max=0.05))
        outcome = engine.evaluate(self.QUOTE, RiskInputs(float("inf"), 100.0))
        assert outcome.verdict is RiskVerdict.HALT

    def test_non_finite_quote_halts(self):
        engine = RiskEngine(RiskParams(q_max=0.05))
        bad = ProposedQuote(float("nan"), 101.0, 1.0, 1.0)
        assert engine.evaluate(bad, RiskInputs(0.0, 100.0)).verdict is RiskVerdict.HALT

    def test_utilisation_never_returns_nan(self):
        engine = RiskEngine(RiskParams(q_max=0.05))
        assert math.isfinite(engine.utilisation(float("nan")))

    def test_finite_inputs_still_work(self):
        """The guard must not break the normal path."""
        engine = RiskEngine(RiskParams(q_max=10.0, breaker_drawdown=1e9))
        outcome = engine.evaluate(self.QUOTE, RiskInputs(0.0, 100.0))
        assert outcome.actionable


class TestPortfolioRejectsCorruptFills:
    """A NaN price silently turned cash, P&L, Sharpe and drawdown into NaN.

    NaN then compares False against every threshold -- including the health
    check's own P&L identity test -- so a corrupted portfolio reported healthy
    and the run produced plausible-looking garbage.
    """

    @pytest.mark.parametrize(
        ("price", "size", "fee"),
        [
            (float("nan"), 1.0, 0.0),
            (float("inf"), 1.0, 0.0),
            (100.0, float("nan"), 0.0),
            (100.0, 1.0, float("nan")),
        ],
    )
    def test_non_finite_rejected(self, price, size, fee):
        with pytest.raises(ValueError, match="non-finite"):
            Portfolio().apply_fill(Side.BID, price, size, fee)

    def test_negative_size_rejected(self):
        with pytest.raises(ValueError, match="negative fill size"):
            Portfolio().apply_fill(Side.BID, 100.0, -1.0)

    def test_non_positive_price_rejected(self):
        with pytest.raises(ValueError, match="non-positive fill price"):
            Portfolio().apply_fill(Side.BID, 0.0, 1.0)

    def test_portfolio_stays_clean_after_rejections(self):
        p = Portfolio()
        for price, size in ((float("nan"), 1.0), (-5.0, 1.0), (100.0, -1.0)):
            with pytest.raises(ValueError):
                p.apply_fill(Side.BID, price, size)
        assert p.cash == 0.0
        assert p.inventory == 0.0
        assert p.check_identity(100.0)

    def test_normal_fills_unaffected(self):
        p = Portfolio()
        p.apply_fill(Side.BID, 100.0, 1.0)
        p.apply_fill(Side.ASK, 101.0, 1.0)
        assert p.realized_pnl == pytest.approx(1.0)


class TestMetricsRefuseCorruptTrajectories:
    """NaN propagated silently into Sharpe and max drawdown.

    A NaN sample makes every downstream statistic NaN, which compares False
    against the ablation's own pass/fail gates -- so a broken run was reported
    as merely unremarkable rather than as broken.
    """

    def _points(self, pnls):
        return [
            PnLPoint(ts_ns=i * 10**9, total_pnl=v, realized_pnl=0.0,
                     inventory=0.0, mid_price=1.0)
            for i, v in enumerate(pnls)
        ]

    def test_nan_pnl_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            compute_metrics("x", self._points([0.0, 1.0, float("nan"), 2.0]))

    def test_inf_pnl_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            compute_metrics("x", self._points([0.0, float("inf")]))

    def test_error_names_the_sample(self):
        with pytest.raises(ValueError, match="sample 2"):
            compute_metrics("x", self._points([0.0, 1.0, float("nan")]))

    def test_clean_trajectory_still_computes(self):
        m = compute_metrics("x", self._points([0.0, 1.0, 2.0, 3.0, 4.0]))
        assert math.isfinite(m.sharpe)
        assert math.isfinite(m.max_drawdown)


class TestForwardCompatibleDecode:
    """An unknown event tag was an unhandled ValidationError.

    During a rolling deploy a producer can be newer than its consumer. The old
    behaviour surfaced that as a wall of confusing "extra field not permitted"
    errors attributed to the *base* class, giving no hint that the real cause
    was a version skew.
    """

    def test_unknown_tag_raises_named_exception(self):
        encoded = _encode(TradeTick(symbol="S", ts_event=1, ts_ingest=2,
                                    price=1.0, size=1.0))
        encoded["type"] = "SomeFutureEvent"
        with pytest.raises(UnhandledEventTypeError) as excinfo:
            _decode(encoded, BaseEvent)
        assert excinfo.value.tag == "SomeFutureEvent"

    def test_known_but_unrequested_type_raises_named_exception(self):
        """Shared streams carry types a given consumer does not handle."""
        from amme_shared.events import OrderBookUpdate

        encoded = _encode(TradeTick(symbol="S", ts_event=1, ts_ingest=2,
                                    price=1.0, size=1.0))
        with pytest.raises(UnhandledEventTypeError):
            _decode(encoded, OrderBookUpdate)

    def test_valid_event_still_decodes(self):
        original = TradeTick(symbol="S", ts_event=1, ts_ingest=2, price=1.0, size=1.0)
        decoded = _decode(_encode(original), BaseEvent)
        assert isinstance(decoded, TradeTick)
        assert decoded.price == 1.0


class TestSimulatorDeterministicIds:
    """The simulator used uuid4 for correlation ids.

    P&L replayed identically because the digest ignores ids, but the s.6.1
    claim is byte-for-byte reproducibility *of the events*, and that was false.
    Anything keyed off correlation_id -- log correlation, dedup, golden-file
    comparison -- was non-reproducible.
    """

    def _ids(self, seed: int):
        from amme_core.simulator import MarketSimulator, SimulatorConfig

        sim = MarketSimulator(SimulatorConfig(seed=seed, tick_size=0.01))
        ids = []
        for book, trades in sim.run_steps(40):
            ids.append(book.correlation_id)
            ids.extend(t.correlation_id for t in trades)
        return ids

    def test_same_seed_gives_identical_ids(self):
        assert self._ids(7) == self._ids(7)

    def test_different_seeds_give_different_ids(self):
        assert self._ids(1) != self._ids(2)

    def test_ids_are_unique_within_a_run(self):
        ids = self._ids(11)
        assert len(ids) == len(set(ids)), "correlation ids collided"

    def test_ids_are_not_uuids(self):
        assert all(i.startswith("sim-") for i in self._ids(3))


class TestOfiWarmTravelsOnTheEvent:
    """quoting-engine-svc substituted ``vol_warm`` for ``ofi_warm``.

    The event contract had no ``ofi_warm`` field, so the service used the
    volatility warm-up flag instead. Those warm at different times (200 samples
    vs 50), so the OFI half of the toxicity gate was armed on the wrong
    condition -- late, and keyed to an unrelated statistic.
    """

    def test_feature_vector_carries_ofi_warm(self):
        from amme_shared.events import FeatureVector

        assert "ofi_warm" in FeatureVector.model_fields

    def test_ofi_warm_is_independent_of_vol_warm(self):
        from amme_shared.events import FeatureVector

        fv = FeatureVector(
            symbol="S", ts_event=1, ts_ingest=2, mid_price=1.0, micro_price=1.0,
            spread=0.1, vpin=0.0, ofi=0.0, realized_vol=0.0,
            ofi_warm=True, vol_warm=False,
        )
        assert fv.ofi_warm is True
        assert fv.vol_warm is False

    def test_pipeline_warms_ofi_before_volatility(self):
        """The two flags really do differ in practice, which is what made the
        substitution a live bug rather than a cosmetic one."""
        from amme_core.features import FeaturePipeline, FeaturePipelineConfig
        from amme_core.simulator import MarketSimulator, SimulatorConfig

        pipeline = FeaturePipeline(
            FeaturePipelineConfig(ofi_window=50, vol_window=200, emit_tensor=False)
        )
        sim = MarketSimulator(SimulatorConfig(seed=5, tick_size=0.01))
        ofi_first = vol_first = None
        for i, (book, trades) in enumerate(sim.run_steps(600)):
            for t in trades:
                pipeline.on_raw_trade(t.price, t.size, t.aggressor)
            snap = pipeline.on_book(book)
            if snap is None:
                continue
            if snap.ofi_warm and ofi_first is None:
                ofi_first = i
            if snap.vol_warm and vol_first is None:
                vol_first = i
        assert ofi_first is not None, "OFI never warmed"
        assert vol_first is None or ofi_first < vol_first, (
            f"expected OFI to warm first (ofi={ofi_first}, vol={vol_first})"
        )


class TestConsumerGroupRecovery:
    """A vanished consumer group must be recreated, not blocked on forever.

    Consumer groups disappear in practice: an operator ``FLUSHALL``, a Redis
    failover onto a replica that never replicated the group, or the stream key
    expiring. Redis then answers ``XREADGROUP`` with ``NOGROUP``. Unhandled,
    the consumer blocks forever on a group that no longer exists — and because
    every service's health check reports process liveness rather than
    throughput, the whole pipeline goes silent while all eight containers stay
    green. That is the same failure signature as the type-tag decode bug, from
    a different cause.
    """

    def test_nogroup_is_detected_by_substring(self):
        """Pin the discriminator: redis-py signals this as a ResponseError whose
        message contains NOGROUP, and we must not swallow other ResponseErrors."""
        from redis.exceptions import ResponseError

        nogroup = ResponseError(
            "NOGROUP No such key 'lob.raw' or consumer group 'amme' in XREADGROUP"
        )
        unrelated = ResponseError("WRONGTYPE Operation against a key holding the wrong kind")
        assert "NOGROUP" in str(nogroup)
        assert "NOGROUP" not in str(unrelated)

    def test_recovery_path_exists_in_consume(self):
        """Guard the guard: assert the handler is actually wired into consume."""
        import inspect

        from amme_shared.streams import RedisEventBus

        source = inspect.getsource(RedisEventBus.consume)
        assert "NOGROUP" in source, "consume() no longer handles a vanished group"
        assert "ensure_group" in source, "consume() no longer recreates the group"

    def test_known_groups_cache_is_invalidated(self):
        """The cache must be cleared or ensure_group would no-op on retry."""
        import inspect

        from amme_shared.streams import RedisEventBus

        source = inspect.getsource(RedisEventBus.consume)
        assert "_known_groups" in source, (
            "the group cache is not invalidated, so recreation would be skipped"
        )


class TestClockOffsetLatchedForever:
    """The exchange clock offset was measured once and never re-measured.

    `ClockSync.resync_interval_seconds` is 300s, but `maybe_resync` was called
    only from the WebSocket reconnect path. On a connection that stays up for
    hours the offset was therefore sampled exactly once, and the documented
    interval never applied.

    That converted a transient measurement error into a permanent one. Observed
    on a live stack: the host suspended, the Docker VM's clock resumed ~13s
    behind real time, the socket reconnected, and ClockSync sampled during the
    catch-up window and latched offset=-13351ms. The VM clock corrected itself
    seconds later; the offset did not. For the next 45 minutes every latency
    metric in the system was wrong by 13.35 seconds.

    The error is undetectable from staleness alone, which is what makes it
    dangerous: staleness is *computed with the offset applied*, so a wrong
    offset just makes the venue look slow. The fixes are a resync on a timer
    (bounding the damage to one interval) and exporting the offset itself.
    """

    def test_a_timer_drives_resync_not_only_reconnects(self):
        import inspect

        from ingestion_svc.service import IngestionService

        source = inspect.getsource(IngestionService)
        assert "_clock_loop" in source, (
            "clock resync is no longer on a timer; a long-lived connection will "
            "latch its first offset measurement forever"
        )
        loop = inspect.getsource(IngestionService._clock_loop)
        assert "resync_clock" in loop
        assert "wait_for_stop" in loop, "the resync loop must be interruptible on shutdown"

    def test_the_resync_task_is_started_and_cancelled(self):
        import inspect

        from ingestion_svc.service import IngestionService

        assert "_clock_loop" in inspect.getsource(IngestionService.startup), (
            "the resync task is never started"
        )
        assert "_clock_task" in inspect.getsource(IngestionService.shutdown), (
            "the resync task is never cancelled, so shutdown would hang or warn"
        )

    def test_poll_is_faster_than_the_resync_interval(self):
        # If the poll were slower, the effective refresh period would be the
        # poll, silently overriding the interval it exists to honour.
        from ingestion_svc.exchange import ClockSync
        from ingestion_svc.service import CLOCK_RESYNC_POLL_SECONDS

        # ClockSync is a slots dataclass, so the interval must be read off an
        # instance -- the class attribute is a descriptor, not the value.
        assert ClockSync().resync_interval_seconds > CLOCK_RESYNC_POLL_SECONDS

    def test_offset_is_exported_as_a_metric(self):
        # A wrong correction must be observable directly. Alerting on staleness
        # cannot catch it, because staleness is derived from the offset.
        from amme_shared.metrics import CLOCK_OFFSET_MS

        CLOCK_OFFSET_MS.labels("AVAXUSDT").set(-13351.0)
        assert CLOCK_OFFSET_MS.labels("AVAXUSDT")._value.get() == -13351.0

    def test_health_reports_the_age_of_the_measurement(self):
        import inspect

        from ingestion_svc.service import IngestionService

        source = inspect.getsource(IngestionService._check_clock)
        assert "last_sync_monotonic" in source, (
            "/health reports the offset but not its age, so a stale offset reads "
            "exactly like a fresh one"
        )

    def test_a_stale_offset_is_recognised_as_stale(self):
        import time as _time

        from ingestion_svc.exchange import ClockSync

        clock = ClockSync(synced=True, last_sync_monotonic=_time.monotonic())
        assert not clock.is_stale()
        clock.last_sync_monotonic = _time.monotonic() - clock.resync_interval_seconds - 1
        assert clock.is_stale(), "an aged-out measurement must be re-measured"
