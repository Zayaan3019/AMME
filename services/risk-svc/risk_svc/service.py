"""risk-svc: quote.intent -> risk.decision. Holds veto power over quoting.

This service exists as a separate process for one reason: **it must be able to
say no when the strategy is broken.** If risk lived inside the quoting engine,
every failure mode of the strategy -- a bad config, a NaN from the model, an
unbounded loop -- would take the safety layer down with it. Here, a quoting
engine that crashes simply stops producing intents, and nothing gets executed.
A quoting engine that goes *insane* has its output rejected by a process that
does not share its state.

That is the standard architecture on a real desk, and the reasoning transfers
directly: the component that can lose money must not be the component that
decides whether losing money is allowed.

State it maintains independently of quoting-engine-svc:

* inventory and P&L, from ``exec.portfolio`` -- never trusted from the intent
* the circuit breaker's latch and high-water mark
* the latest volatility/toxicity readings and the touch, from ``features.ready``

Two checks belong here rather than in the pure engine, because they are about
the *pipeline*, not the quote:

* **Stale intents are rejected.** An intent older than ``max_intent_age_ms``
  (a backlog replayed after a restart, a stalled consumer) was priced against a
  market that no longer exists. Approving it would put an old price on the book.
* **The touch comes from risk's own feature feed**, so the post-only check in
  :class:`~amme_core.risk.RiskEngine` does not trust the quoting engine's idea
  of where the market is.
"""

from __future__ import annotations

import asyncio
import contextlib

from amme_core.risk import (
    BreakerTrigger,
    ProposedQuote,
    RiskEngine,
    RiskInputs,
    RiskParams,
    RiskVerdict,
)
from amme_shared.config import Settings
from amme_shared.events import (
    BaseEvent,
    FeatureVector,
    Heartbeat,
    PortfolioState,
    QuoteIntent,
    RiskAction,
    RiskDecision,
)
from amme_shared.logging import correlation_scope
from amme_shared.metrics import (
    BREAKER_STATE,
    EVENTS_CONSUMED,
    EVENTS_FAILED,
    EVENTS_PUBLISHED,
    INVENTORY,
    INVENTORY_UTILISATION,
    PROCESSING_LATENCY,
    RISK_DECISIONS,
    SAFETY_EVENTS,
    observe_latency,
)
from amme_shared.service import StreamService
from amme_shared.streams import RedisEventBus, Streams

_VERDICT_TO_ACTION = {
    RiskVerdict.APPROVE: RiskAction.APPROVE,
    RiskVerdict.AMEND: RiskAction.AMEND,
    RiskVerdict.REJECT: RiskAction.REJECT,
    RiskVerdict.HALT: RiskAction.HALT,
}


class RiskService(StreamService):
    name = "risk-svc"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        r = self.settings.risk
        self.engine = RiskEngine(
            RiskParams(
                q_max=r.q_max,
                q_soft_frac=r.q_soft_frac,
                unwind_frac=r.unwind_frac,
                max_position_value=r.max_position_value,
                breaker_vol_multiple=r.breaker_vol_multiple,
                breaker_vpin=r.breaker_vpin,
                breaker_drawdown=r.breaker_drawdown,
                breaker_cooldown_seconds=r.breaker_cooldown_seconds,
                enabled=r.enabled,
            )
        )
        self.bus = RedisEventBus(
            self.settings.infra.redis_url,
            maxlen=self.settings.infra.stream_maxlen,
            block_ms=self.settings.infra.block_ms,
        )
        self._inventory = 0.0
        self._total_pnl = 0.0
        self._vol_fast = 0.0
        self._vol_baseline = 0.0
        self._vol_warm = False
        self._vpin = 0.0
        self._vpin_warm = False
        self._best_bid: float | None = None
        self._best_ask: float | None = None
        self._max_intent_age_ns = int(r.max_intent_age_ms * 1e6)
        self._stale_intents = 0
        self._decisions = 0
        self._halts = 0
        self._side_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._heartbeats = 0
        self.add_health_check("redis", self._check_redis)
        self.add_health_check("risk_state", self._check_state)

    async def _check_redis(self) -> tuple[bool, str]:
        try:
            return bool(await self.bus.ping()), "connected"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    async def _check_state(self) -> tuple[bool, str]:
        breaker = self.engine.breaker
        # A tripped breaker is not "unhealthy" -- it is the system working. It
        # is surfaced in the detail string and as a metric, not as a 503, or
        # an orchestrator would restart the very thing protecting us.
        return True, (
            f"inventory={self._inventory:.6g} util={self.engine.utilisation(self._inventory):.2f} "
            f"pnl={self._total_pnl:.2f} "
            f"breaker={'TRIPPED:' + breaker.trigger if breaker.tripped else 'ok'} "
            f"trips={breaker.trip_count} decisions={self._decisions} halts={self._halts} "
            f"stale_intents={self._stale_intents} heartbeats={self._heartbeats}"
        )

    async def startup(self) -> None:
        group = self.settings.infra.consumer_group
        for stream in (Streams.QUOTE_INTENT, Streams.FEATURES_READY, Streams.EXEC_PORTFOLIO):
            await self.bus.ensure_group(stream, group)
        self._side_task = asyncio.create_task(self._track_state(), name="risk-state")
        self._heartbeat_task = asyncio.create_task(self._heartbeat(), name="risk-heartbeat")

    async def shutdown(self) -> None:
        for task in (self._side_task, self._heartbeat_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        await self.bus.close()

    async def _heartbeat(self) -> None:
        """Tell execution "risk is alive" on a fixed cadence, market or not.

        Without this, the dead-man's switch could only listen for decisions,
        which stop whenever the book is quiet -- it fired on a calm market with
        risk perfectly healthy. A heartbeat that fails to publish is logged and
        retried; if it keeps failing, execution stops hearing it and pulls its
        quotes, which is the point.
        """
        interval = self.settings.risk.heartbeat_seconds
        symbol = self.settings.market.symbol
        while not self.stopping:
            now = self.clock.now_ns()
            try:
                await self.bus.publish(
                    Streams.RISK_HEARTBEAT,
                    Heartbeat(symbol=symbol, ts_event=now, ts_ingest=now, service=self.name),
                )
                self._heartbeats += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.log.warning("heartbeat_failed", error=f"{type(exc).__name__}: {exc}")
            await self.wait_for_stop(timeout=interval)

    async def _track_state(self) -> None:
        """Maintain risk's own view of portfolio and market state."""
        group = self.settings.infra.consumer_group
        consumer = f"{self.name}-state"
        symbol = self.settings.market.symbol
        try:
            async for stream, message_id, event in self.bus.consume(
                [Streams.FEATURES_READY, Streams.EXEC_PORTFOLIO], group, consumer, BaseEvent
            ):
                if self.stopping:
                    return
                try:
                    payload = event.model_dump()
                    if stream == Streams.EXEC_PORTFOLIO:
                        state = PortfolioState.model_validate(payload)
                        self._inventory = state.inventory
                        self._total_pnl = state.total_pnl
                        INVENTORY.labels(symbol).set(state.inventory)
                        INVENTORY_UTILISATION.labels(symbol).set(
                            self.engine.utilisation(state.inventory)
                        )
                    else:
                        features = FeatureVector.model_validate(payload)
                        self._vol_fast = features.realized_vol_fast
                        self._vol_baseline = features.realized_vol_baseline
                        self._vol_warm = features.vol_warm
                        self._vpin = features.vpin
                        self._vpin_warm = features.vpin_warm
                        half = features.spread / 2.0
                        self._best_bid = features.mid_price - half
                        self._best_ask = features.mid_price + half
                finally:
                    await self.bus.ack(stream, group, message_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log.exception("risk_state_stream_failed", error=str(exc))

    async def run_loop(self) -> None:
        group = self.settings.infra.consumer_group
        consumer = f"{self.name}-1"
        symbol = self.settings.market.symbol

        async for stream, message_id, intent in self.bus.consume(
            [Streams.QUOTE_INTENT], group, consumer, QuoteIntent
        ):
            if self.stopping:
                break
            EVENTS_CONSUMED.labels(self.name, stream).inc()
            try:
                with correlation_scope(intent.correlation_id):
                    await self._evaluate(intent, symbol)
            except Exception as exc:
                EVENTS_FAILED.labels(self.name, type(exc).__name__).inc()
                self.log.exception("risk_evaluation_failed", error=str(exc))
                # Failing closed: if risk cannot decide, nothing is authorised.
                await self._publish_stop(
                    intent, symbol, RiskAction.HALT, f"risk error: {type(exc).__name__}", "error"
                )
            finally:
                await self.bus.ack(stream, group, message_id)

    async def _evaluate(self, intent: QuoteIntent, symbol: str) -> None:
        age_ns = self.clock.now_ns() - intent.ts_ingest
        if age_ns > self._max_intent_age_ns:
            self._stale_intents += 1
            SAFETY_EVENTS.labels(self.name, "stale_intent").inc()
            await self._publish_stop(
                intent,
                symbol,
                RiskAction.REJECT,
                f"stale intent: {age_ns / 1e6:.0f} ms old > {self._max_intent_age_ns / 1e6:.0f} ms",
                "stale_intent",
            )
            return

        proposed = ProposedQuote(
            bid_price=intent.bid_price,
            ask_price=intent.ask_price,
            bid_size=intent.bid_size,
            ask_size=intent.ask_size,
            half_spread=intent.half_spread,
        )
        inputs = RiskInputs(
            inventory=self._inventory,
            mid_price=0.5 * (intent.bid_price + intent.ask_price),
            total_pnl=self._total_pnl,
            realized_vol=self._vol_fast,
            baseline_vol=self._vol_baseline if self._vol_warm else 0.0,
            vpin=self._vpin if self._vpin_warm else 0.0,
            ts_ns=intent.ts_event,
            best_bid=self._best_bid,
            best_ask=self._best_ask,
        )

        with observe_latency(PROCESSING_LATENCY, self.name, "evaluate"):
            outcome = self.engine.evaluate(proposed, inputs)

        breaker = self.engine.breaker
        BREAKER_STATE.labels(symbol).set(1 if breaker.tripped else 0)
        if outcome.verdict is RiskVerdict.HALT:
            self._halts += 1
            if breaker.trigger is not BreakerTrigger.NONE:
                self.log.warning(
                    "circuit_breaker",
                    trigger=str(breaker.trigger),
                    reason=outcome.reason,
                    inventory=self._inventory,
                    pnl=self._total_pnl,
                )

        decision = RiskDecision(
            symbol=symbol,
            ts_event=intent.ts_event,
            ts_ingest=self.clock.now_ns(),
            source=intent.source,
            correlation_id=intent.correlation_id,
            intent_event_id=intent.event_id,
            action=_VERDICT_TO_ACTION[outcome.verdict],
            reason=outcome.reason,
            bid_price=outcome.bid_price,
            ask_price=outcome.ask_price,
            bid_size=outcome.bid_size,
            ask_size=outcome.ask_size,
            inventory=self._inventory,
            inventory_utilisation=outcome.utilisation,
            breaker_tripped=outcome.breaker_tripped,
        )
        await self.bus.publish(Streams.RISK_DECISION, decision)
        EVENTS_PUBLISHED.labels(self.name, Streams.RISK_DECISION).inc()
        RISK_DECISIONS.labels(decision.action.value, outcome.trigger.value).inc()
        self._decisions += 1

    async def _publish_stop(
        self, intent: QuoteIntent, symbol: str, action: RiskAction, reason: str, label: str
    ) -> None:
        """Publish a decision that authorises nothing (REJECT or HALT)."""
        decision = RiskDecision(
            symbol=symbol,
            ts_event=intent.ts_event,
            ts_ingest=self.clock.now_ns(),
            source=intent.source,
            correlation_id=intent.correlation_id,
            intent_event_id=intent.event_id,
            action=action,
            reason=reason,
            inventory=self._inventory,
            breaker_tripped=action is RiskAction.HALT,
        )
        await self.bus.publish(Streams.RISK_DECISION, decision)
        RISK_DECISIONS.labels(action.value, label).inc()
        self._decisions += 1


__all__ = ["RiskService"]
