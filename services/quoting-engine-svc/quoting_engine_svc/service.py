"""quoting-engine-svc: features.ready (+ drift.signal) -> quote.intent.

Emits *intents*, never orders. Only risk-svc can authorise something the
execution layer will act on -- see ``docs/ARCHITECTURE.md`` on why that
separation is a process boundary and not just a function call.

**Drift signal handling.** The drift stream is consumed asynchronously and
cached, rather than joined synchronously against the feature stream. Blocking a
quote on a model prediction would put a neural network on the critical path of
every quote, and couple quoting availability to inference availability. Instead
the most recent prediction is used if it is fresh, and ignored if it is not:
degrading to pure Avellaneda-Stoikov is a perfectly good failure mode, whereas
not quoting at all is not.
"""

from __future__ import annotations

import asyncio
import contextlib

from amme_core.avellaneda_stoikov import ASParams
from amme_core.strategies import AMMEStrategy, DecisionContext
from amme_core.toxicity import ToxicityParams
from amme_shared.config import Settings
from amme_shared.events import (
    BaseEvent,
    DriftSignal,
    FeatureVector,
    QuoteIntent,
)
from amme_shared.logging import correlation_scope
from amme_shared.metrics import (
    DRIFT_SIGNAL,
    EVENTS_CONSUMED,
    EVENTS_FAILED,
    EVENTS_PUBLISHED,
    PROCESSING_LATENCY,
    QUOTED_SPREAD,
    RESERVATION_OFFSET,
    TOXICITY_MULTIPLIER,
    observe_latency,
)
from amme_shared.service import StreamService
from amme_shared.streams import RedisEventBus, Streams

#: A prediction older than this is discarded. Roughly 20 book updates at the
#: 100ms depth cadence -- long enough to survive a GC pause in the inference
#: service, short enough that we never quote on a stale directional view.
DRIFT_MAX_AGE_NS = 2_000_000_000

#: Floor on sigma before the volatility estimator warms up. Without it the A-S
#: inventory term is ~0 at startup and the engine quotes one tick wide with no
#: inventory control at all -- precisely when it knows least.
SIGMA_FLOOR = 1e-4


class QuotingEngineService(StreamService):
    name = "quoting-engine-svc"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        s = self.settings
        self.strategy = AMMEStrategy(
            as_params=ASParams(
                gamma=s.avellaneda_stoikov.gamma,
                kappa=s.avellaneda_stoikov.kappa,
                horizon_mode=s.avellaneda_stoikov.horizon_mode,
                horizon_seconds=s.avellaneda_stoikov.horizon_seconds,
                session_seconds=s.avellaneda_stoikov.session_seconds,
                tick_size=s.market.tick_size,
                min_half_spread_ticks=s.avellaneda_stoikov.min_half_spread_ticks,
                max_half_spread_ticks=s.avellaneda_stoikov.max_half_spread_ticks,
                quote_size=s.avellaneda_stoikov.quote_size,
                # Inventory is normalised by the risk limit before entering the
                # A-S formulas; the quoter must therefore know q_max.
                inventory_scale=s.risk.q_max,
                drift_lambda=s.avellaneda_stoikov.drift_lambda,
                drift_confidence_floor=s.avellaneda_stoikov.drift_confidence_floor,
                drift_max_ticks=s.avellaneda_stoikov.drift_max_ticks,
            ),
            toxicity_params=ToxicityParams(
                vpin_threshold=s.toxicity.vpin_threshold,
                vpin_beta=s.toxicity.vpin_beta,
                ofi_threshold=s.toxicity.ofi_threshold,
                ofi_beta=s.toxicity.ofi_beta,
                ofi_skew_beta=s.toxicity.ofi_skew_beta,
                max_multiplier=s.toxicity.max_toxicity_multiplier,
                enabled=s.toxicity.enabled,
            ),
        )
        self.bus = RedisEventBus(
            s.infra.redis_url, maxlen=s.infra.stream_maxlen, block_ms=s.infra.block_ms
        )
        self._drift: DriftSignal | None = None
        self._inventory = 0.0
        self._start_ns = 0
        self._quotes = 0
        self._portfolio_task: asyncio.Task[None] | None = None
        self.add_health_check("redis", self._check_redis)
        self.add_health_check("quoting", self._check_quoting)

    async def _check_redis(self) -> tuple[bool, str]:
        try:
            return bool(await self.bus.ping()), "connected"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    async def _check_quoting(self) -> tuple[bool, str]:
        drift_age = "none"
        if self._drift is not None:
            drift_age = f"{(self._now_ns() - self._drift.ts_event) / 1e6:.0f}ms"
        return True, f"quotes={self._quotes} inventory={self._inventory:.6g} drift_age={drift_age}"

    def _now_ns(self) -> int:
        return self.clock.now_ns()

    async def startup(self) -> None:
        group = self.settings.infra.consumer_group
        for stream in (Streams.FEATURES_READY, Streams.DRIFT_SIGNAL, Streams.EXEC_PORTFOLIO):
            await self.bus.ensure_group(stream, group)
        # Drift and portfolio are ambient state, consumed on their own tasks so
        # neither can add latency to the quoting path.
        self._portfolio_task = asyncio.create_task(self._track_side_streams(), name="quote-side")

    async def shutdown(self) -> None:
        if self._portfolio_task is not None:
            self._portfolio_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._portfolio_task
        await self.bus.close()

    async def _track_side_streams(self) -> None:
        """Keep the latest drift signal and inventory without blocking quoting."""
        group = self.settings.infra.consumer_group
        consumer = f"{self.name}-side"
        symbol = self.settings.market.symbol
        try:
            async for stream, message_id, event in self.bus.consume(
                [Streams.DRIFT_SIGNAL, Streams.EXEC_PORTFOLIO], group, consumer, BaseEvent
            ):
                if self.stopping:
                    return
                try:
                    if stream == Streams.DRIFT_SIGNAL:
                        signal = DriftSignal.model_validate(event.model_dump())
                        self._drift = signal
                        DRIFT_SIGNAL.labels(symbol).set(signal.drift)
                    else:
                        self._inventory = float(event.model_dump().get("inventory", 0.0))
                finally:
                    await self.bus.ack(stream, group, message_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log.exception("side_stream_failed", error=str(exc))

    def _current_drift(self, now_ns: int) -> tuple[float, float]:
        signal = self._drift
        if signal is None:
            return 0.0, 0.0
        if now_ns - signal.ts_event > DRIFT_MAX_AGE_NS:
            return 0.0, 0.0
        return signal.drift, signal.confidence

    async def run_loop(self) -> None:
        group = self.settings.infra.consumer_group
        consumer = f"{self.name}-1"
        symbol = self.settings.market.symbol

        async for stream, message_id, event in self.bus.consume(
            [Streams.FEATURES_READY], group, consumer, FeatureVector
        ):
            if self.stopping:
                break
            EVENTS_CONSUMED.labels(self.name, stream).inc()
            try:
                with correlation_scope(event.correlation_id):
                    await self._on_features(event, symbol)
            except Exception as exc:
                EVENTS_FAILED.labels(self.name, type(exc).__name__).inc()
                self.log.exception("quote_failed", error=str(exc))
            finally:
                await self.bus.ack(stream, group, message_id)

    async def _on_features(self, features: FeatureVector, symbol: str) -> None:
        if self._start_ns == 0:
            self._start_ns = features.ts_event
        elapsed = (features.ts_event - self._start_ns) / 1e9
        drift, confidence = self._current_drift(features.ts_event)

        ctx = DecisionContext(
            mid=features.mid_price,
            micro_price=features.micro_price,
            sigma=max(features.realized_vol, SIGMA_FLOOR),
            inventory=self._inventory,
            elapsed_seconds=elapsed,
            best_bid=features.mid_price - features.spread / 2.0,
            best_ask=features.mid_price + features.spread / 2.0,
            vpin=features.vpin,
            vpin_warm=features.vpin_warm,
            ofi_zscore=features.ofi_zscore,
            ofi_warm=features.ofi_warm,
            drift=drift,
            drift_confidence=confidence,
        )

        with observe_latency(PROCESSING_LATENCY, self.name, "quote"):
            quote = self.strategy.decide(ctx)

        intent = QuoteIntent(
            symbol=symbol,
            ts_event=features.ts_event,
            ts_ingest=self._now_ns(),
            source=features.source,
            correlation_id=features.correlation_id,
            bid_price=quote.bid_price,
            ask_price=quote.ask_price,
            bid_size=quote.bid_size,
            ask_size=quote.ask_size,
            reservation_price=quote.reservation_price,
            half_spread=quote.half_spread,
            inventory=self._inventory,
            drift_adjustment=quote.drift_adjustment,
            toxicity_multiplier=quote.toxicity_multiplier,
            sigma=ctx.sigma,
            gamma=self.strategy.quoter.params.gamma,
            kappa=self.strategy.quoter.params.kappa,
            strategy=self.strategy.name,
        )
        await self.bus.publish(Streams.QUOTE_INTENT, intent)
        EVENTS_PUBLISHED.labels(self.name, Streams.QUOTE_INTENT).inc()
        self._quotes += 1

        QUOTED_SPREAD.labels(symbol).set(quote.ask_price - quote.bid_price)
        RESERVATION_OFFSET.labels(symbol).set(quote.reservation_price - features.mid_price)
        TOXICITY_MULTIPLIER.labels(symbol).set(quote.toxicity_multiplier)


__all__ = ["QuotingEngineService"]
