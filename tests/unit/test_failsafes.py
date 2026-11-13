"""Fail-safes between the services, and the live-recording dataset builder.

Each test pins one way the pipeline could keep trading on information that is
no longer true:

* a maker quote that crosses the touch (would execute as a taker);
* an intent priced against an old market (a backlog after a restart);
* a decision that arrives long after it was made;
* risk-svc going silent while approved quotes keep resting (dead-man's switch);
* the same event delivered twice by an at-least-once bus.
"""

from __future__ import annotations

import asyncio
import math

import pytest

from amme_core.risk import ProposedQuote, RiskEngine, RiskInputs, RiskParams, RiskVerdict
from amme_shared.config import Settings
from amme_shared.events import (
    DataSource,
    FeatureVector,
    OrderBookUpdate,
    PriceLevel,
    QuoteIntent,
    RiskAction,
    RiskDecision,
    Side,
)
from amme_shared.streams import InMemoryEventBus, RecentIds, Streams

MID = 50_000.0
TICK = 0.01


def _quote(bid: float, ask: float, size: float = 0.001) -> ProposedQuote:
    return ProposedQuote(bid_price=bid, ask_price=ask, bid_size=size, ask_size=size,
                         half_spread=(ask - bid) / 2)


def _inputs(**kw: float | None) -> RiskInputs:
    base: dict = {"inventory": 0.0, "mid_price": MID, "ts_ns": 1}
    base.update(kw)
    return RiskInputs(**base)


class TestPostOnly:
    """A maker quote at or through the opposite touch is a taker order."""

    def test_quote_inside_the_touch_is_untouched(self):
        out = RiskEngine().evaluate(
            _quote(MID - TICK, MID + TICK), _inputs(best_bid=MID - TICK, best_ask=MID + TICK)
        )
        assert out.verdict is RiskVerdict.APPROVE
        assert out.bid_size == out.ask_size == 0.001

    def test_bid_at_the_best_ask_is_pulled(self):
        out = RiskEngine().evaluate(
            _quote(MID + TICK, MID + 3 * TICK), _inputs(best_bid=MID - TICK, best_ask=MID + TICK)
        )
        assert out.verdict is RiskVerdict.AMEND
        assert out.bid_size == 0.0 and out.ask_size == 0.001
        assert "post-only" in out.reason

    def test_ask_through_the_best_bid_is_pulled(self):
        out = RiskEngine().evaluate(
            _quote(MID - 5 * TICK, MID - 2 * TICK),
            _inputs(best_bid=MID - TICK, best_ask=MID + TICK),
        )
        assert out.ask_size == 0.0 and out.bid_size == 0.001

    def test_both_sides_crossing_is_a_reject(self):
        # Both prices outside the touch on the wrong side: a locked/crossed book
        # snapshot. Nothing can be posted, so nothing is authorised.
        out = RiskEngine().evaluate(
            _quote(MID + 2 * TICK, MID + 3 * TICK), _inputs(best_bid=MID + 4 * TICK, best_ask=MID)
        )
        assert out.verdict is RiskVerdict.REJECT

    def test_skew_is_applied_before_the_check(self):
        # Long near the limit: skew moves both prices down. The ask lands on the
        # best bid only after skewing, so the check must see the skewed price.
        engine = RiskEngine(RiskParams(q_max=0.05, q_soft_frac=0.1, skew_beta=50.0))
        out = engine.evaluate(
            _quote(MID - TICK, MID + TICK),
            _inputs(inventory=0.04, best_bid=MID - TICK, best_ask=MID + TICK),
        )
        assert out.skew < 0
        assert out.ask_price <= MID - TICK
        assert out.ask_size == 0.0

    def test_without_a_touch_the_check_is_off(self):
        out = RiskEngine().evaluate(_quote(MID + TICK, MID + 3 * TICK), _inputs())
        assert out.verdict is RiskVerdict.APPROVE

    def test_a_self_crossed_quote_is_rejected(self):
        out = RiskEngine().evaluate(_quote(MID + TICK, MID - TICK), _inputs())
        assert out.verdict is RiskVerdict.REJECT
        assert "self-crossed" in out.reason

    def test_a_nan_touch_fails_closed(self):
        out = RiskEngine().evaluate(_quote(MID - TICK, MID + TICK), _inputs(best_bid=math.nan))
        assert out.verdict is RiskVerdict.HALT


class TestRecentIds:
    def test_second_sighting_is_a_duplicate(self):
        seen = RecentIds(capacity=4)
        assert not seen.check_and_add("a")
        assert seen.check_and_add("a")

    def test_capacity_evicts_the_oldest(self):
        seen = RecentIds(capacity=2)
        for i in "abc":
            seen.check_and_add(i)
        assert len(seen) == 2
        assert not seen.check_and_add("a"), "'a' was evicted, so it reads as new"

    def test_rejects_a_useless_capacity(self):
        with pytest.raises(ValueError):
            RecentIds(capacity=0)


def _book(ts: int = 1) -> OrderBookUpdate:
    return OrderBookUpdate(
        symbol="BTCUSDT", ts_event=ts, ts_ingest=ts, source=DataSource.SIMULATOR,
        bids=[PriceLevel(price=MID - TICK, size=1.0)],
        asks=[PriceLevel(price=MID + TICK, size=1.0)],
        last_update_id=1, is_snapshot=True,
    )


class TestInMemoryBusDedup:
    async def test_the_same_event_is_delivered_once(self):
        bus = InMemoryEventBus()
        event = _book()
        await bus.publish(Streams.LOB_RAW, event)
        await bus.publish(Streams.LOB_RAW, event)  # e.g. a retried XADD
        await bus.publish(Streams.LOB_RAW, _book(ts=2))
        got = [e async for _, _, e in bus.consume([Streams.LOB_RAW], "g", "c", OrderBookUpdate)]
        assert [e.ts_event for e in got] == [1, 2]

    async def test_groups_dedup_independently(self):
        bus = InMemoryEventBus()
        await bus.publish(Streams.LOB_RAW, _book())
        for group in ("g1", "g2"):
            got = [e async for _, _, e in bus.consume([Streams.LOB_RAW], group, "c", OrderBookUpdate)]
            assert len(got) == 1


def _decision(now_ns: int, *, action: RiskAction = RiskAction.APPROVE, age_ms: float = 0.0
              ) -> RiskDecision:
    stamp = now_ns - int(age_ms * 1e6)
    return RiskDecision(
        symbol="BTCUSDT", ts_event=stamp, ts_ingest=stamp, source=DataSource.SIMULATOR,
        intent_event_id="i", action=action,
        bid_price=MID - TICK, ask_price=MID + TICK, bid_size=0.001, ask_size=0.001,
    )


@pytest.fixture
def execution():
    from execution_sim_svc.service import ExecutionSimService

    svc = ExecutionSimService(Settings())
    svc.bus = InMemoryEventBus()
    yield svc


def _resting(svc) -> int:
    return sum(svc.matching.active_order(s) is not None for s in (Side.BID, Side.ASK))


class TestExecutionFailsafes:
    async def _prime(self, svc) -> None:
        await svc._handle(Streams.LOB_RAW, _book(), "BTCUSDT")

    async def test_fresh_approval_places_quotes(self, execution):
        await self._prime(execution)
        await execution._handle(
            Streams.RISK_DECISION, _decision(execution.clock.now_ns()), "BTCUSDT"
        )
        assert _resting(execution) == 2

    async def test_stale_approval_cancels_instead_of_placing(self, execution):
        await self._prime(execution)
        await execution._handle(
            Streams.RISK_DECISION, _decision(execution.clock.now_ns()), "BTCUSDT"
        )
        assert _resting(execution) == 2
        old = _decision(execution.clock.now_ns(), age_ms=5_000)
        await execution._handle(Streams.RISK_DECISION, old, "BTCUSDT")
        assert _resting(execution) == 0
        assert execution._stale_decisions == 1

    async def test_dead_man_switch_pulls_quotes_when_risk_goes_silent(self, execution):
        await self._prime(execution)
        await execution._handle(
            Streams.RISK_DECISION, _decision(execution.clock.now_ns()), "BTCUSDT"
        )
        execution.on_risk_alive(now=100.0)
        limit = execution._dead_man_seconds
        assert not execution.check_dead_man(now=100.0 + limit * 0.5)
        assert _resting(execution) == 2
        assert execution.check_dead_man(now=100.0 + limit + 0.01)
        assert _resting(execution) == 0
        assert not execution.check_dead_man(now=100.0 + 10 * limit), "fires once, not repeatedly"
        execution.on_risk_alive(now=200.0)
        assert not execution._dead_man_tripped, "a fresh decision re-arms it"
        assert execution._dead_man_trips == 1

    async def test_nothing_fires_before_risk_has_ever_spoken(self, execution):
        assert not execution.check_dead_man(now=1e9)

    async def test_a_quiet_market_does_not_trip_it_while_risk_heartbeats(self, execution):
        # Found live: on a calm AVAXUSDT book no intents flow, so no decisions
        # flow, and the switch fired three times in a minute with risk healthy.
        from amme_shared.events import Heartbeat

        await self._prime(execution)
        await execution._handle(
            Streams.RISK_DECISION, _decision(execution.clock.now_ns()), "BTCUSDT"
        )
        limit = execution._dead_man_seconds
        start = execution._last_alive_at
        for k in range(1, 9):  # four dead-man periods with no decision at all
            now_ns = execution.clock.now_ns()
            beat = Heartbeat(symbol="BTCUSDT", ts_event=now_ns, ts_ingest=now_ns,
                             source=DataSource.SIMULATOR, service="risk-svc")
            await execution._handle(Streams.RISK_HEARTBEAT, beat, "BTCUSDT")
            execution._last_alive_at = start + k * limit / 2  # beats at 2x the needed rate
            assert not execution.check_dead_man(now=start + k * limit / 2 + limit * 0.6)
        assert _resting(execution) == 2
        assert execution._dead_man_trips == 0

    async def test_an_old_heartbeat_from_a_backlog_does_not_count(self, execution):
        from amme_shared.events import Heartbeat

        execution.on_risk_alive(now=100.0)
        old_ns = execution.clock.now_ns() - int((execution._dead_man_seconds + 5) * 1e9)
        beat = Heartbeat(symbol="BTCUSDT", ts_event=old_ns, ts_ingest=old_ns,
                         source=DataSource.SIMULATOR, service="risk-svc")
        await execution._handle(Streams.RISK_HEARTBEAT, beat, "BTCUSDT")
        assert execution._last_alive_at == 100.0

    def test_execution_listens_on_the_heartbeat_stream(self):
        import inspect

        from execution_sim_svc.service import ExecutionSimService

        assert "RISK_HEARTBEAT" in inspect.getsource(ExecutionSimService.run_loop)
        assert "RISK_HEARTBEAT" in inspect.getsource(ExecutionSimService.startup)

    def test_the_watchdog_task_is_started_and_stopped(self):
        import inspect

        from execution_sim_svc.service import ExecutionSimService

        assert "_watchdog" in inspect.getsource(ExecutionSimService.startup)
        assert "_watchdog_task" in inspect.getsource(ExecutionSimService.shutdown)


@pytest.fixture
def risk():
    from risk_svc.service import RiskService

    svc = RiskService(Settings())
    svc.bus = InMemoryEventBus()
    yield svc


def _intent(now_ns: int, *, age_ms: float = 0.0, bid: float = MID - TICK, ask: float = MID + TICK
            ) -> QuoteIntent:
    stamp = now_ns - int(age_ms * 1e6)
    return QuoteIntent(
        symbol="BTCUSDT", ts_event=stamp, ts_ingest=stamp, source=DataSource.SIMULATOR,
        bid_price=bid, ask_price=ask, bid_size=0.001, ask_size=0.001,
        reservation_price=MID, half_spread=TICK, inventory=0.0, sigma=1.0, gamma=0.1, kappa=1.0,
    )


class TestRiskServiceFailsafes:
    async def _decisions(self, svc) -> list[RiskDecision]:
        return svc.bus.drain(Streams.RISK_DECISION, RiskDecision)

    async def test_fresh_intent_is_approved(self, risk):
        await risk._evaluate(_intent(risk.clock.now_ns()), "BTCUSDT")
        (decision,) = await self._decisions(risk)
        assert decision.action is RiskAction.APPROVE

    async def test_stale_intent_is_rejected(self, risk):
        await risk._evaluate(_intent(risk.clock.now_ns(), age_ms=10_000), "BTCUSDT")
        (decision,) = await self._decisions(risk)
        assert decision.action is RiskAction.REJECT
        assert decision.reason.startswith("stale intent")
        assert not decision.breaker_tripped
        assert risk._stale_intents == 1

    async def test_risk_uses_its_own_view_of_the_touch(self, risk):
        # risk-svc learned the touch from features.ready; the intent's bid sits
        # on the best ask, so risk pulls it even though the intent says nothing
        # about the book.
        features = FeatureVector(
            symbol="BTCUSDT", ts_event=1, ts_ingest=1, source=DataSource.SIMULATOR,
            mid_price=MID, micro_price=MID, spread=2 * TICK, vpin=0.1, ofi=0.0,
            realized_vol=1.0,
        )
        await risk.bus.publish(Streams.FEATURES_READY, features)
        task = asyncio.create_task(risk._track_state())
        await asyncio.sleep(0)
        await task
        assert risk._best_ask == pytest.approx(MID + TICK)
        await risk._evaluate(
            _intent(risk.clock.now_ns(), bid=MID + TICK, ask=MID + 3 * TICK), "BTCUSDT"
        )
        (decision,) = await self._decisions(risk)
        assert decision.action is RiskAction.AMEND
        assert decision.bid_size == 0.0
        assert "post-only" in decision.reason


class TestRiskHeartbeat:
    async def test_risk_heartbeats_with_no_market_activity(self, risk):
        from amme_shared.events import Heartbeat

        risk.settings = risk.settings.model_copy(update={
            "risk": risk.settings.risk.model_copy(update={"heartbeat_seconds": 0.01})
        })
        task = asyncio.create_task(risk._heartbeat())
        await asyncio.sleep(0.08)
        risk.request_stop()
        await asyncio.wait_for(task, 1.0)
        beats = risk.bus.drain(Streams.RISK_HEARTBEAT, Heartbeat)
        assert len(beats) >= 3, "no intents, no features -- heartbeats still flow"
        assert all(b.service == risk.name for b in beats)
        assert risk.bus.drain(Streams.RISK_DECISION, RiskDecision) == []

    def test_heartbeat_must_be_well_inside_the_dead_man_window(self):
        from pydantic import ValidationError

        from amme_shared.config import RiskConfig

        RiskConfig(heartbeat_seconds=0.5, dead_man_seconds=2.0)
        with pytest.raises(ValidationError):
            RiskConfig(heartbeat_seconds=1.5, dead_man_seconds=2.0)


class TestLiveRecordingDataset:
    """``make record``: the live TimescaleDB capture becomes a training set."""

    @staticmethod
    def _window(seconds: float, seed: int = 7):
        from amme_core.simulator import MarketSimulator, SimulatorConfig

        sim = MarketSimulator(SimulatorConfig(seed=seed, tick_size=TICK))
        return list(sim.run(seconds))

    @staticmethod
    def _shift(window, by_seconds: float):
        shift = int(by_seconds * 1e9)
        return [
            (b.model_copy(update={"ts_event": b.ts_event + shift}),
             [t.model_copy(update={"ts_event": t.ts_event + shift}) for t in trades])
            for b, trades in window
        ]

    def test_builds_a_purged_time_split_from_the_recording(self):
        from amme_train.dataset import DatasetConfig
        from amme_train.record import build_live_dataset

        window = self._window(240.0)
        config = DatasetConfig(lookback=20, horizon=5)
        dataset, seconds = build_live_dataset(window, min_seconds=120.0, config=config)
        assert seconds >= 120.0
        assert len(dataset.train) > len(dataset.test) > 0
        assert dataset.train.x.shape[1:] == (20, config.levels * 4)

    def test_uses_the_largest_contiguous_segment_not_the_whole_range(self):
        from amme_train.dataset import DatasetConfig
        from amme_train.record import build_live_dataset, window_seconds

        window = self._window(300.0)
        cut = len(window) // 3
        # A 10-minute hole after the first third: the stack was down.
        gapped = window[:cut] + self._shift(window[cut:], 600.0)
        assert window_seconds(gapped) > 800
        _, seconds = build_live_dataset(
            gapped, min_seconds=60.0, config=DatasetConfig(lookback=20, horizon=5)
        )
        assert seconds < 300.0, "training windows must not straddle the outage"

    def test_too_little_recording_is_refused(self):
        from amme_train.record import InsufficientRecordingError, build_live_dataset

        with pytest.raises(InsufficientRecordingError) as err:
            build_live_dataset(self._window(60.0), min_seconds=600.0)
        assert err.value.need_seconds == 600.0
