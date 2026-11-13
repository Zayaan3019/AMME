"""execution-sim-svc: risk.decision + live tape -> exec.fill / exec.portfolio.

The paper-trading matching engine. It rests our authorised quotes against the
*real* Binance order flow and simulates fills with a queue-position model
(:mod:`amme_core.matching`), so P&L reflects being genuinely behind 8 BTC of
resting size at the touch rather than assuming we are always first in line.

Only ``risk.decision`` is consumed -- never ``quote.intent`` directly. That is
the enforcement point for risk's veto: bypassing it would require changing
which stream this service subscribes to, which is a visible, reviewable change
rather than a missing `if`.

**Dead-man's switch.** Risk's veto only works while risk is alive. If risk-svc
dies, no new decisions arrive -- and without this switch the last approved
quotes would keep resting and keep filling. risk-svc sends a heartbeat every
``heartbeat_seconds`` on ``risk.heartbeat``, market or not; a watchdog pulls
every resting quote once nothing fresh (heartbeat or decision) has arrived for
``dead_man_seconds``; quoting resumes on the next fresh approval. Silence is
treated as "stop", which is the only safe reading of silence. Listening for
decisions alone was not enough: they stop whenever the book is quiet, and the
switch fired on a calm market with risk perfectly healthy.

**Stale decisions are not acted on.** A decision older than
``max_decision_age_ms`` (a backlog read after this service restarts) can still
*cancel* -- cancelling is always safe -- but never places a quote.
"""

from __future__ import annotations

import asyncio
import contextlib
import time

from amme_core.matching import MatchingEngine, MatchingParams
from amme_shared.config import Settings
from amme_shared.events import (
    BaseEvent,
    Fill,
    Heartbeat,
    OrderBookUpdate,
    PortfolioState,
    RiskAction,
    RiskDecision,
    Side,
    TradeTick,
)
from amme_shared.logging import correlation_scope
from amme_shared.metrics import (
    EVENTS_CONSUMED,
    EVENTS_FAILED,
    EVENTS_PUBLISHED,
    FILLS,
    PROCESSING_LATENCY,
    QUOTES_SENT,
    REALIZED_PNL,
    SAFETY_EVENTS,
    TOTAL_PNL,
    observe_latency,
)
from amme_shared.service import StreamService
from amme_shared.streams import RedisEventBus, Streams

#: Portfolio snapshots are published on this cadence rather than per fill.
#: Downstream consumers (risk, dashboard) want a heartbeat of state, and a
#: per-fill firehose would make the stream's volume depend on market activity.
PORTFOLIO_INTERVAL_SECONDS = 1.0

#: How often the dead-man watchdog looks. Well under ``dead_man_seconds``.
WATCHDOG_INTERVAL_SECONDS = 0.25


class ExecutionSimService(StreamService):
    name = "execution-sim-svc"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        self.matching = MatchingEngine(
            MatchingParams(
                maker_fee_bps=self.settings.market.maker_fee_bps,
                tick_size=self.settings.market.tick_size,
            )
        )
        self.bus = RedisEventBus(
            self.settings.infra.redis_url,
            maxlen=self.settings.infra.stream_maxlen,
            block_ms=self.settings.infra.block_ms,
        )
        self._last_book: OrderBookUpdate | None = None
        self._mid = 0.0
        self._quotes_placed = 0
        self._n_fills = 0
        self._publish_task: asyncio.Task[None] | None = None
        self._watchdog_task: asyncio.Task[None] | None = None
        risk = self.settings.risk
        self._dead_man_seconds = risk.dead_man_seconds
        self._max_decision_age_ns = int(risk.max_decision_age_ms * 1e6)
        #: Monotonic receipt time of the last risk decision; None until one arrives.
        self._last_alive_at: float | None = None
        self._dead_man_tripped = False
        self._dead_man_trips = 0
        self._stale_decisions = 0
        self.add_health_check("redis", self._check_redis)
        self.add_health_check("portfolio", self._check_portfolio)

    async def _check_redis(self) -> tuple[bool, str]:
        try:
            return bool(await self.bus.ping()), "connected"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    async def _check_portfolio(self) -> tuple[bool, str]:
        p = self.matching.portfolio
        identity_ok = p.check_identity(self._mid) if self._mid > 0 else True
        detail = (
            f"inventory={p.inventory:.6g} cash={p.cash:.2f} pnl={p.total_pnl(self._mid):.2f} "
            f"fills={p.n_fills} quotes={self._quotes_placed} "
            f"dead_man={'TRIPPED' if self._dead_man_tripped else 'armed'} "
            f"dead_man_trips={self._dead_man_trips} stale_decisions={self._stale_decisions}"
        )
        # A broken P&L identity means the accounting is wrong, which is a real
        # failure -- surface it as unhealthy rather than reporting bad numbers.
        return identity_ok, detail if identity_ok else f"P&L IDENTITY VIOLATED: {detail}"

    async def startup(self) -> None:
        group = self.settings.infra.consumer_group
        for stream in (
            Streams.RISK_DECISION, Streams.RISK_HEARTBEAT, Streams.LOB_RAW, Streams.LOB_TRADES
        ):
            await self.bus.ensure_group(stream, group)
        self._publish_task = asyncio.create_task(self._publish_portfolio(), name="exec-portfolio")
        self._watchdog_task = asyncio.create_task(self._watchdog(), name="exec-dead-man")

    async def shutdown(self) -> None:
        for task in (self._publish_task, self._watchdog_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        # Leave nothing resting behind us.
        self.matching.cancel_all()
        self.matching.prune()
        await self.bus.close()

    def check_dead_man(self, now: float | None = None) -> bool:
        """Pull every resting quote if risk has gone quiet. True when it fires.

        Separate from the loop so it can be driven by a test without sleeping.
        """
        now = time.monotonic() if now is None else now
        last = self._last_alive_at
        if last is None or self._dead_man_tripped:
            return False
        silent = now - last
        if silent < self._dead_man_seconds:
            return False
        self._dead_man_tripped = True
        self._dead_man_trips += 1
        self.matching.cancel_all()
        self.matching.prune()
        SAFETY_EVENTS.labels(self.name, "dead_man").inc()
        self.log.warning(
            "dead_man_switch",
            silent_seconds=round(silent, 2),
            limit_seconds=self._dead_man_seconds,
            action="all resting quotes cancelled",
        )
        return True

    async def _watchdog(self) -> None:
        while not self.stopping:
            if await self.wait_for_stop(WATCHDOG_INTERVAL_SECONDS):
                return
            self.check_dead_man()

    async def _publish_portfolio(self) -> None:
        symbol = self.settings.market.symbol
        while not self.stopping:
            if await self.wait_for_stop(PORTFOLIO_INTERVAL_SECONDS):
                return
            if self._mid <= 0:
                continue
            p = self.matching.portfolio
            state = PortfolioState(
                symbol=symbol,
                ts_event=self.clock.now_ns(),
                ts_ingest=self.clock.now_ns(),
                inventory=p.inventory,
                cash=p.cash,
                mid_price=self._mid,
                realized_pnl=p.realized_pnl,
                unrealized_pnl=p.unrealized_pnl(self._mid),
                total_pnl=p.total_pnl(self._mid),
                fees_paid=p.fees_paid,
                n_fills=p.n_fills,
                n_quotes=self._quotes_placed,
            )
            await self.bus.publish(Streams.EXEC_PORTFOLIO, state)
            EVENTS_PUBLISHED.labels(self.name, Streams.EXEC_PORTFOLIO).inc()
            TOTAL_PNL.labels(symbol, "amme").set(state.total_pnl)
            REALIZED_PNL.labels(symbol, "amme").set(state.realized_pnl)

    async def run_loop(self) -> None:
        group = self.settings.infra.consumer_group
        consumer = f"{self.name}-1"
        symbol = self.settings.market.symbol

        async for stream, message_id, event in self.bus.consume(
            [Streams.RISK_DECISION, Streams.RISK_HEARTBEAT, Streams.LOB_RAW, Streams.LOB_TRADES],
            group,
            consumer,
            BaseEvent,
        ):
            if self.stopping:
                break
            EVENTS_CONSUMED.labels(self.name, stream).inc()
            try:
                with correlation_scope(event.correlation_id):
                    await self._handle(stream, event, symbol)
            except Exception as exc:
                EVENTS_FAILED.labels(self.name, type(exc).__name__).inc()
                self.log.exception("execution_failed", stream=stream, error=str(exc))
            finally:
                await self.bus.ack(stream, group, message_id)

    async def _handle(self, stream: str, event: BaseEvent, symbol: str) -> None:
        payload = event.model_dump()

        if stream == Streams.RISK_HEARTBEAT:
            beat = Heartbeat.model_validate(payload)
            # A heartbeat read from a backlog (this service restarted) proves
            # nothing about now; only a fresh one counts as "risk is alive".
            if self.clock.now_ns() - beat.ts_ingest <= self._dead_man_seconds * 1e9:
                self.on_risk_alive()
            return

        if stream == Streams.LOB_RAW:
            book = OrderBookUpdate.model_validate(payload)
            self._last_book = book
            if book.mid_price is not None:
                self._mid = book.mid_price
            # Decay queue position for cancellations observed at our levels.
            self.matching.on_book_update(
                {lv.price: lv.size for lv in book.bids},
                {lv.price: lv.size for lv in book.asks},
            )
            return

        if stream == Streams.LOB_TRADES:
            trade = TradeTick.model_validate(payload)
            with observe_latency(PROCESSING_LATENCY, self.name, "match"):
                fills = self.matching.on_trade(
                    trade.price, trade.size, trade.aggressor, trade.ts_event
                )
            for sim_fill in fills:
                self._n_fills += 1
                portfolio = self.matching.portfolio
                fill = Fill(
                    symbol=symbol,
                    ts_event=sim_fill.ts_ns,
                    ts_ingest=self.clock.now_ns(),
                    source=trade.source,
                    correlation_id=trade.correlation_id,
                    order_id=sim_fill.order_id,
                    side=sim_fill.side,
                    price=sim_fill.price,
                    size=sim_fill.size,
                    fee=sim_fill.fee,
                    queue_position_at_fill=sim_fill.queue_position_at_fill,
                    inventory_after=portfolio.inventory,
                    cash_after=portfolio.cash,
                )
                await self.bus.publish(Streams.EXEC_FILL, fill)
                EVENTS_PUBLISHED.labels(self.name, Streams.EXEC_FILL).inc()
                FILLS.labels(symbol, sim_fill.side.value).inc()
                self.log.info(
                    "fill",
                    side=sim_fill.side.value,
                    price=sim_fill.price,
                    size=sim_fill.size,
                    inventory=portfolio.inventory,
                    pnl=portfolio.total_pnl(self._mid),
                )
            self.matching.prune()
            return

        # risk.decision
        decision = RiskDecision.model_validate(payload)
        self.on_risk_alive()
        if decision.action in (RiskAction.HALT, RiskAction.REJECT):
            self.matching.cancel_all()
            self.matching.prune()
            return

        age_ns = self.clock.now_ns() - decision.ts_ingest
        if age_ns > self._max_decision_age_ns:
            # Old approvals were priced against an old market. Cancel instead.
            self._stale_decisions += 1
            SAFETY_EVENTS.labels(self.name, "stale_decision").inc()
            self.matching.cancel_all()
            self.matching.prune()
            return

        last_book = self._last_book
        if last_book is None or decision.bid_price is None or decision.ask_price is None:
            return
        book = last_book

        bid_size = decision.bid_size or 0.0
        ask_size = decision.ask_size or 0.0
        placed = self.matching.replace_quotes(
            bid_price=decision.bid_price if bid_size > 0 else None,
            bid_size=bid_size,
            ask_price=decision.ask_price if ask_size > 0 else None,
            ask_size=ask_size,
            bid_level_size=self._level_size(book, Side.BID, decision.bid_price),
            ask_level_size=self._level_size(book, Side.ASK, decision.ask_price),
            ts_ns=decision.ts_event,
        )
        if placed:
            self._quotes_placed += placed
            QUOTES_SENT.labels(symbol).inc(placed)

    def on_risk_alive(self, now: float | None = None) -> None:
        """Re-arm the dead-man's switch: a fresh heartbeat or decision arrived."""
        self._last_alive_at = time.monotonic() if now is None else now
        if self._dead_man_tripped:
            self._dead_man_tripped = False
            self.log.info("dead_man_rearmed")

    @staticmethod
    def _level_size(book: OrderBookUpdate, side: Side, price: float) -> float:
        """Resting size at ``price`` -- our queue position when we join.

        Zero when the price is outside the published depth, which is the correct
        assumption: if we are quoting beyond the visible book we are alone at
        that level as far as we can observe.
        """
        for lv in book.bids if side is Side.BID else book.asks:
            if lv.price == price:
                return lv.size
        return 0.0


__all__ = ["ExecutionSimService"]
