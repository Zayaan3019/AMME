"""backtest-svc: on-demand ablation runs, plus the ``make regression-test`` CLI.

Runs as a service so the ablation can be triggered from the dashboard against
the same configuration the live stack is using, and as a CLI so CI can run it
without bringing up Redis. Both paths call the same
:class:`~backtest_svc.runner.BacktestRunner`.

Runs execute on a thread rather than the event loop: an ablation is seconds of
solid CPU, and blocking the loop would stall the ``/health`` endpoint, which
would get the container killed mid-run.
"""

from __future__ import annotations

import asyncio
from typing import Any

from amme_shared.config import Settings
from amme_shared.service import StreamService
from backtest_svc.runner import ABLATION_ARMS, BacktestConfig, BacktestRunner


def config_from_settings(settings: Settings) -> BacktestConfig:
    """Build a run config from the same settings the live stack uses."""
    return BacktestConfig(
        seed=settings.backtest.seed,
        duration_seconds=settings.backtest.duration_seconds,
        warmup_seconds=settings.backtest.warmup_seconds,
        tick_size=settings.market.tick_size,
        q_max=settings.risk.q_max,
        quote_size=settings.avellaneda_stoikov.quote_size,
        maker_fee_bps=settings.market.maker_fee_bps,
        gamma=settings.avellaneda_stoikov.gamma,
        kappa=settings.avellaneda_stoikov.kappa,
        horizon_seconds=settings.avellaneda_stoikov.horizon_seconds,
        min_half_spread_ticks=settings.avellaneda_stoikov.min_half_spread_ticks,
        drift_lambda=settings.avellaneda_stoikov.drift_lambda,
        vpin_bucket_volume=settings.toxicity.vpin_bucket_volume,
        vpin_buckets=settings.toxicity.vpin_buckets,
        vpin_threshold=settings.toxicity.vpin_threshold,
        vpin_beta=settings.toxicity.vpin_beta,
        ofi_threshold=settings.toxicity.ofi_threshold,
        ofi_beta=settings.toxicity.ofi_beta,
        ofi_skew_beta=settings.toxicity.ofi_skew_beta,
        results_dir=settings.backtest.results_dir,
    )


class BacktestService(StreamService):
    name = "backtest-svc"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        self.runner = BacktestRunner(config_from_settings(self.settings))
        self._running = False
        self._last_summary: list[dict[str, Any]] = []
        self.add_health_check("backtest", self._check)

    async def _check(self) -> tuple[bool, str]:
        state = "running" if self._running else "idle"
        return True, f"{state}, last run produced {len(self._last_summary)} arms"

    def build_app(self):  # type: ignore[no-untyped-def] -- FastAPI app, typed by base
        app = super().build_app()

        @app.post("/run")
        async def run() -> dict[str, Any]:
            if self._running:
                return {"status": "already_running"}
            self._running = True
            try:
                results = await asyncio.to_thread(self.runner.run_all, ABLATION_ARMS)
                await asyncio.to_thread(self.runner.write_report, results)
                self._last_summary = [r.summary() for r in results]
                return {"status": "complete", "arms": self._last_summary}
            finally:
                self._running = False

        @app.get("/results")
        async def results() -> dict[str, Any]:
            return {"arms": self._last_summary}

        return app

    async def run_loop(self) -> None:
        # Purely request-driven; nothing to poll. Idle until shutdown.
        await self.wait_for_stop()


__all__ = ["BacktestService", "config_from_settings"]
