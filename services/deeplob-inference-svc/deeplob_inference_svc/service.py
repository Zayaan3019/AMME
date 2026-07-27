"""deeplob-inference-svc: features.ready -> drift.signal.

Serves the DeepLOB forward pass against a hard p99 budget (s.5.3). The budget is
not decoration: the quoting engine treats a prediction older than 2s as absent,
so an inference path that misses its budget does not slow the system down -- it
silently stops contributing, and the ablation would show the ML arm doing
nothing while looking healthy. The p99 is therefore tracked continuously and
surfaced on ``/health``.
"""

from __future__ import annotations

import numpy as np

from amme_shared.config import Settings
from amme_shared.events import DriftSignal, FeatureVector
from amme_shared.logging import correlation_scope
from amme_shared.metrics import (
    EVENTS_CONSUMED,
    EVENTS_FAILED,
    EVENTS_PUBLISHED,
    INFERENCE_LATENCY,
    PROCESSING_LATENCY,
    observe_latency,
)
from amme_shared.service import StreamService
from amme_shared.streams import RedisEventBus, Streams
from deeplob_inference_svc.predictor import build_predictor, probabilities_to_drift

#: Rolling window for the p99 estimate reported on /health.
LATENCY_WINDOW = 1000


class DeepLOBInferenceService(StreamService):
    name = "deeplob-inference-svc"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        m = self.settings.model
        self.predictor = build_predictor(
            m.backend, m.artifact_path, m.onnx_path, m.torch_threads
        )
        self.bus = RedisEventBus(
            self.settings.infra.redis_url,
            maxlen=self.settings.infra.stream_maxlen,
            block_ms=self.settings.infra.block_ms,
        )
        # The class threshold used to build the labels is the natural scale for
        # converting probabilities back into an expected return.
        self._alpha_scale = m.label_alpha
        self._latencies: list[float] = []
        self._predictions = 0
        self._skipped = 0
        self.add_health_check("redis", self._check_redis)
        self.add_health_check("inference", self._check_inference)

    async def _check_redis(self) -> tuple[bool, str]:
        try:
            return bool(await self.bus.ping()), "connected"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    async def _check_inference(self) -> tuple[bool, str]:
        p99 = self.p99_latency_ms()
        budget = self.settings.model.inference_p99_budget_ms
        detail = (
            f"backend={self.predictor.version} predictions={self._predictions} "
            f"skipped={self._skipped} p99={p99:.3f}ms budget={budget}ms"
        )
        if self.predictor.version == "stub":
            # Running on the stub is a valid state (no model trained yet), so
            # report healthy -- but say so loudly in the detail string.
            return True, f"DEGRADED (no model loaded): {detail}"
        return p99 <= budget, detail

    def p99_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        ordered = sorted(self._latencies)
        return ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))]

    async def startup(self) -> None:
        await self.bus.ensure_group(Streams.FEATURES_READY, self.settings.infra.consumer_group)
        self.log.info(
            "inference_ready",
            backend=self.predictor.version,
            budget_ms=self.settings.model.inference_p99_budget_ms,
        )

    async def shutdown(self) -> None:
        await self.bus.close()

    async def run_loop(self) -> None:
        group = self.settings.infra.consumer_group
        consumer = f"{self.name}-1"

        async for stream, message_id, features in self.bus.consume(
            [Streams.FEATURES_READY], group, consumer, FeatureVector
        ):
            if self.stopping:
                break
            EVENTS_CONSUMED.labels(self.name, stream).inc()
            try:
                with correlation_scope(features.correlation_id):
                    await self._infer(features)
            except Exception as exc:
                EVENTS_FAILED.labels(self.name, type(exc).__name__).inc()
                self.log.exception("inference_failed", error=str(exc))
            finally:
                await self.bus.ack(stream, group, message_id)

    async def _infer(self, features: FeatureVector) -> None:
        shape = features.tensor_shape
        if shape is None or not features.lob_tensor:
            # The window has not filled yet, or feature-svc is not emitting a
            # tensor because no model is loaded. Nothing to predict on.
            self._skipped += 1
            return

        tensor = np.asarray(features.lob_tensor, dtype=np.float32).reshape(shape)

        with observe_latency(PROCESSING_LATENCY, self.name, "infer"):
            probabilities, latency_us = self.predictor.predict(tensor)

        latency_ms = latency_us / 1000.0
        INFERENCE_LATENCY.labels(self.predictor.version).observe(latency_ms)
        self._latencies.append(latency_ms)
        del self._latencies[:-LATENCY_WINDOW]

        drift, confidence = probabilities_to_drift(probabilities, self._alpha_scale)
        p_down, p_flat, p_up = (float(v) for v in probabilities)

        signal = DriftSignal(
            symbol=features.symbol,
            ts_event=features.ts_event,
            ts_ingest=self.clock.now_ns(),
            source=features.source,
            correlation_id=features.correlation_id,
            drift=drift,
            confidence=confidence,
            prob_down=p_down,
            prob_flat=p_flat,
            prob_up=p_up,
            horizon_events=self.settings.model.horizon_events,
            model_version=self.predictor.version,
            inference_latency_us=latency_us,
        )
        await self.bus.publish(Streams.DRIFT_SIGNAL, signal)
        EVENTS_PUBLISHED.labels(self.name, Streams.DRIFT_SIGNAL).inc()
        self._predictions += 1


__all__ = ["DeepLOBInferenceService"]
