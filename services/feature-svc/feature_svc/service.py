"""feature-svc: lob.raw + lob.trades -> features.ready.

Consumes both the book and the tape, because the two toxicity measures need
different inputs: OFI is computed on top-of-book transitions, VPIN on the trade
tape. Merging them here rather than in two services keeps the causal ordering
of "trades that preceded this book state" trivially correct -- splitting them
would require a synchronisation protocol to reassemble something we get for
free by consuming in order.

The actual computation is :class:`~amme_core.features.FeaturePipeline`, the same
object the backtester uses. This service is a transport adapter around it.
"""

from __future__ import annotations

import asyncio
import contextlib

from amme_core.features import FeaturePipeline, FeaturePipelineConfig
from amme_core.vpin import ClassificationScheme
from amme_shared.config import Settings
from amme_shared.events import BaseEvent, FeatureVector, OrderBookUpdate, TradeTick
from amme_shared.logging import correlation_scope
from amme_shared.metrics import (
    EVENTS_CONSUMED,
    EVENTS_FAILED,
    EVENTS_PUBLISHED,
    OFI,
    PROCESSING_LATENCY,
    REALIZED_VOL,
    STREAM_LAG,
    VPIN,
    observe_latency,
)
from amme_shared.service import StreamService
from amme_shared.streams import RedisEventBus, Streams


class FeatureService(StreamService):
    name = "feature-svc"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        tox = self.settings.toxicity
        model = self.settings.model
        self.pipeline = FeaturePipeline(
            FeaturePipelineConfig(
                levels=min(model.levels, self.settings.market.depth_levels),
                lookback=model.lookback,
                vpin_bucket_volume=tox.vpin_bucket_volume,
                vpin_buckets=tox.vpin_buckets,
                vpin_scheme=ClassificationScheme.TRUE_AGGRESSOR,
                ofi_window=tox.ofi_window,
                ofi_zscore_window=tox.ofi_zscore_window,
                # The tensor is only needed when a real model will consume it.
                # Building a 100x40 array per update to then discard it is pure
                # waste on the hot path.
                emit_tensor=model.backend != "stub",
            )
        )
        self.bus = RedisEventBus(
            self.settings.infra.redis_url,
            maxlen=self.settings.infra.stream_maxlen,
            block_ms=self.settings.infra.block_ms,
        )
        self._books = 0
        self._trades = 0
        self._published = 0
        self._lag_task: asyncio.Task[None] | None = None
        self.add_health_check("redis", self._check_redis)
        self.add_health_check("progress", self._check_progress)

    async def _check_redis(self) -> tuple[bool, str]:
        try:
            return bool(await self.bus.ping()), "connected"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    async def _check_progress(self) -> tuple[bool, str]:
        return True, (
            f"books={self._books} trades={self._trades} published={self._published} "
            f"vpin={self.pipeline.vpin:.3f} sigma={self.pipeline.sigma:.5f}"
        )

    async def startup(self) -> None:
        for stream in (Streams.LOB_RAW, Streams.LOB_TRADES):
            await self.bus.ensure_group(stream, self.settings.infra.consumer_group)
        self._lag_task = asyncio.create_task(self._report_lag(), name="feature-lag")

    async def shutdown(self) -> None:
        if self._lag_task is not None:
            self._lag_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._lag_task
        await self.bus.close()

    async def _report_lag(self) -> None:
        group = self.settings.infra.consumer_group
        while not self.stopping:
            if await self.wait_for_stop(5.0):
                return
            for stream in (Streams.LOB_RAW, Streams.LOB_TRADES):
                lag = await self.bus.lag(stream, group)
                if lag >= 0:
                    STREAM_LAG.labels(self.name, stream).set(lag)

    async def run_loop(self) -> None:
        group = self.settings.infra.consumer_group
        symbol = self.settings.market.symbol
        consumer = f"{self.name}-1"

        async for stream, message_id, event in self.bus.consume(
            [Streams.LOB_RAW, Streams.LOB_TRADES], group, consumer, BaseEvent
        ):
            if self.stopping:
                break
            EVENTS_CONSUMED.labels(self.name, stream).inc()
            try:
                with correlation_scope(event.correlation_id):
                    await self._handle(stream, event, symbol)
            except Exception as exc:
                EVENTS_FAILED.labels(self.name, type(exc).__name__).inc()
                self.log.exception("feature_handler_failed", stream=stream, error=str(exc))
            finally:
                await self.bus.ack(stream, group, message_id)

    async def _handle(self, stream: str, event: BaseEvent, symbol: str) -> None:
        if isinstance(event, TradeTick) or stream == Streams.LOB_TRADES:
            trade = event if isinstance(event, TradeTick) else TradeTick.model_validate(
                event.model_dump()
            )
            self.pipeline.on_trade(trade)
            self._trades += 1
            VPIN.labels(symbol).set(self.pipeline.vpin)
            return

        book = event if isinstance(event, OrderBookUpdate) else OrderBookUpdate.model_validate(
            event.model_dump()
        )
        self._books += 1

        with observe_latency(PROCESSING_LATENCY, self.name, "compute"):
            snapshot = self.pipeline.on_book(book)
        if snapshot is None:
            return

        OFI.labels(symbol).set(snapshot.ofi)
        REALIZED_VOL.labels(symbol).set(snapshot.sigma)
        VPIN.labels(symbol).set(snapshot.vpin)

        tensor = snapshot.tensor
        vector = FeatureVector(
            symbol=book.symbol,
            ts_event=book.ts_event,
            ts_ingest=book.ts_ingest,
            source=book.source,
            # Propagate the causing book's id so a quote stays traceable back
            # to the exact tick that produced it.
            correlation_id=book.correlation_id,
            mid_price=snapshot.mid_price,
            micro_price=snapshot.micro_price,
            spread=snapshot.spread,
            vpin=snapshot.vpin,
            vpin_warm=snapshot.vpin_warm,
            ofi=snapshot.ofi,
            ofi_zscore=snapshot.ofi_zscore,
            ofi_warm=snapshot.ofi_warm,
            realized_vol=snapshot.sigma,
            realized_vol_fast=snapshot.sigma_fast,
            realized_vol_baseline=snapshot.sigma_baseline,
            vol_warm=snapshot.vol_warm,
            lob_tensor=tensor.ravel().tolist() if tensor is not None else [],
            tensor_shape=snapshot.tensor_shape,
        )
        await self.bus.publish(Streams.FEATURES_READY, vector)
        EVENTS_PUBLISHED.labels(self.name, Streams.FEATURES_READY).inc()
        self._published += 1


__all__ = ["FeatureService"]
