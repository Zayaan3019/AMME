"""Service scaffolding: every AMME process is a `StreamService`.

The contract from s.4 is that *even pure stream consumers* expose `/health` and
`/metrics`. This module makes that free: subclass :class:`StreamService`,
implement ``run_loop``, and you get the HTTP surface, structured logging,
correlation-ID propagation, graceful shutdown and metric wiring for nothing.
"""

from __future__ import annotations

import abc
import asyncio
import contextlib
import signal
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

from amme_shared.clock import Clock, SystemClock
from amme_shared.config import Settings, get_settings
from amme_shared.logging import configure_logging
from amme_shared.metrics import (
    CONTENT_TYPE_LATEST,
    SERVICE_UP,
    render_metrics,
)

HealthCheck = Callable[[], Awaitable[tuple[bool, str]]]


class StreamService(abc.ABC):
    """Base class for every AMME microservice."""

    #: Overridden by subclasses; used for logging and metric labels.
    name: str = "unnamed-svc"

    def __init__(self, settings: Settings | None = None, clock: Clock | None = None) -> None:
        self.settings = settings or get_settings()
        self.clock: Clock = clock or SystemClock()
        self.log = configure_logging(
            self.name, self.settings.infra.log_level, self.settings.infra.log_json
        )
        self._shutdown = asyncio.Event()
        self._ready = False
        self._health_checks: list[tuple[str, HealthCheck]] = []
        self._app: FastAPI | None = None

    # -- lifecycle ---------------------------------------------------------
    @abc.abstractmethod
    async def run_loop(self) -> None:
        """The service's main work. Must return promptly once :meth:`stopping` is set."""

    async def startup(self) -> None:
        """Acquire resources. Override as needed."""
        return None

    async def shutdown(self) -> None:
        """Release resources. Override as needed. Always runs, even on error."""
        return None

    @property
    def stopping(self) -> bool:
        return self._shutdown.is_set()

    def request_stop(self) -> None:
        self._shutdown.set()

    async def wait_for_stop(self, timeout: float | None = None) -> bool:
        """Sleep until shutdown is requested. Returns True if it was."""
        try:
            await asyncio.wait_for(self._shutdown.wait(), timeout)
            return True
        except TimeoutError:
            return False

    # -- health ------------------------------------------------------------
    def add_health_check(self, name: str, check: HealthCheck) -> None:
        self._health_checks.append((name, check))

    async def _collect_health(self) -> tuple[bool, dict[str, str]]:
        details: dict[str, str] = {}
        healthy = self._ready
        for name, check in self._health_checks:
            try:
                ok, detail = await check()
            except Exception as exc:
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            details[name] = ("ok: " if ok else "FAIL: ") + detail
            healthy = healthy and ok
        return healthy, details

    # -- HTTP surface ------------------------------------------------------
    def build_app(self) -> FastAPI:
        if self._app is not None:
            return self._app

        @contextlib.asynccontextmanager
        async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
            await self.startup()
            self._ready = True
            SERVICE_UP.labels(self.name).set(1)
            task = asyncio.create_task(self._supervised_loop(), name=f"{self.name}-loop")
            try:
                yield
            finally:
                self._ready = False
                SERVICE_UP.labels(self.name).set(0)
                self.request_stop()
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                await self.shutdown()

        app = FastAPI(title=self.name, version="0.1.0", lifespan=lifespan)

        @app.get("/health")
        async def health() -> Response:
            healthy, details = await self._collect_health()
            return JSONResponse(
                status_code=200 if healthy else 503,
                content={
                    "service": self.name,
                    "status": "healthy" if healthy else "unhealthy",
                    "checks": details,
                },
            )

        @app.get("/metrics")
        async def metrics() -> Response:
            return Response(content=render_metrics(), media_type=CONTENT_TYPE_LATEST)

        @app.get("/info")
        async def info() -> dict[str, Any]:
            return {
                "service": self.name,
                "symbol": self.settings.market.symbol,
                "depth_levels": self.settings.market.depth_levels,
            }

        self._app = app
        return app

    async def _supervised_loop(self) -> None:
        """Run ``run_loop``; log and re-raise anything that escapes it."""
        try:
            await self.run_loop()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log.exception("service_loop_crashed", error=str(exc))
            raise

    def serve(self, port: int | None = None) -> None:
        """Blocking entry point used by each service's ``__main__``."""
        bind_port = port or self.settings.infra.metrics_port
        self.log.info("service_starting", port=bind_port, symbol=self.settings.market.symbol)
        config = uvicorn.Config(
            self.build_app(),
            host="0.0.0.0",
            port=bind_port,
            log_config=None,
            access_log=False,
        )
        uvicorn.Server(config).run()


def install_signal_handlers(service: StreamService) -> None:
    """SIGTERM/SIGINT -> graceful stop. No-op where the platform lacks them."""
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, service.request_stop)


__all__ = ["HealthCheck", "StreamService", "install_signal_handlers"]
