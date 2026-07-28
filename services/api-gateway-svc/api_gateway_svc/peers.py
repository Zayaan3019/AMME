"""Fan-out health probe across the other seven services.

The System view shows whether the pipeline is actually whole, which means
asking each service rather than inferring it from stream traffic. Inference
would be wrong in the one case that matters: a service that is up, consuming,
and failing its own health assertion (book desynced, P&L identity broken) still
moves messages.

Every probe is independent and bounded. One unreachable service must degrade to
a red card, never to a hung request -- so the whole fan-out runs concurrently
under a single deadline and unreachable peers resolve to a status rather than
an exception.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx

#: Probe deadline per service. Generous next to a local `/health` (single-digit
#: ms) but short enough that a wedged peer cannot stall the System view.
PROBE_TIMEOUT_S = 2.0


@dataclass(frozen=True)
class Peer:
    """One service in the pipeline, in dataflow order."""

    name: str
    port: int
    role: str
    #: Streams this service reads and writes. Drives the topology diagram, so it
    #: is data rather than something the front-end hard-codes a second time.
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    extra: dict[str, str] = field(default_factory=dict)


#: Dataflow order, which is also the order the System view renders them in.
PIPELINE: tuple[Peer, ...] = (
    Peer(
        name="ingestion-svc",
        port=8001,
        role="Reconstructs the L2 book from Binance snapshot + diff, corrects for "
        "exchange clock skew, and persists ticks.",
        produces=("lob.raw", "lob.trades"),
    ),
    Peer(
        name="feature-svc",
        port=8002,
        role="Microstructure features: VPIN toxicity, order-flow imbalance, "
        "realized volatility, and the normalised DeepLOB input tensor.",
        consumes=("lob.raw", "lob.trades"),
        produces=("features.ready",),
    ),
    Peer(
        name="deeplob-inference-svc",
        port=8003,
        role="Serves the trained DeepLOB CNN-LSTM as a 3-class drift signal "
        "under a 5ms p99 budget.",
        consumes=("features.ready",),
        produces=("drift.signal",),
    ),
    Peer(
        name="quoting-engine-svc",
        port=8004,
        role="Avellaneda-Stoikov optimal quoting: reservation price, optimal "
        "spread, toxicity widening, drift injection.",
        consumes=("features.ready", "drift.signal"),
        produces=("quote.intent",),
    ),
    Peer(
        name="risk-svc",
        port=8005,
        role="The hard boundary. Inventory limits, notional caps, and circuit "
        "breakers. Nothing reaches execution without passing here.",
        consumes=("quote.intent",),
        produces=("risk.decision",),
    ),
    Peer(
        name="execution-sim-svc",
        port=8006,
        role="Queue-position-aware paper matching against the real tape.",
        consumes=("risk.decision", "lob.trades"),
        produces=("exec.fill", "exec.portfolio"),
    ),
    Peer(
        name="backtest-svc",
        port=8007,
        role="Deterministic replay, parameter calibration, and the multi-seed ablation harness.",
    ),
    Peer(
        name="api-gateway-svc",
        port=8000,
        role="This service. Read-only projection of every stream, plus the "
        "dashboard you are looking at.",
        consumes=(
            "lob.raw",
            "lob.trades",
            "features.ready",
            "drift.signal",
            "quote.intent",
            "risk.decision",
            "exec.fill",
            "exec.portfolio",
        ),
    ),
)


def _peer_url(peer: Peer, host_template: str) -> str:
    return host_template.format(name=peer.name, port=peer.port)


async def _probe_one(client: httpx.AsyncClient, peer: Peer, host_template: str) -> dict[str, Any]:
    """Probe one peer. Never raises -- an unreachable peer *is* the result."""
    base = {
        "name": peer.name,
        "port": peer.port,
        "role": peer.role,
        "consumes": list(peer.consumes),
        "produces": list(peer.produces),
    }
    started = asyncio.get_running_loop().time()
    try:
        response = await client.get(f"{_peer_url(peer, host_template)}/health")
        elapsed_ms = (asyncio.get_running_loop().time() - started) * 1000.0
        # /health answers 503 with a full body when a check fails; that body is
        # the most useful thing on the page, so parse it either way.
        payload = response.json()
        checks = payload.get("checks", {})
        return {
            **base,
            "status": payload.get(
                "status", "healthy" if response.status_code == 200 else "unhealthy"
            ),
            "http_status": response.status_code,
            "latency_ms": round(elapsed_ms, 2),
            "checks": checks if isinstance(checks, dict) else {},
        }
    except Exception as exc:
        elapsed_ms = (asyncio.get_running_loop().time() - started) * 1000.0
        return {
            **base,
            "status": "unreachable",
            "http_status": None,
            "latency_ms": round(elapsed_ms, 2),
            "checks": {"transport": f"FAIL: {type(exc).__name__}"},
        }


async def probe_pipeline(host_template: str) -> list[dict[str, Any]]:
    """Probe every service concurrently. Returns results in pipeline order."""
    timeout = httpx.Timeout(PROBE_TIMEOUT_S)
    async with httpx.AsyncClient(timeout=timeout) as client:
        return list(
            await asyncio.gather(*(_probe_one(client, peer, host_template) for peer in PIPELINE))
        )


__all__ = ["PIPELINE", "PROBE_TIMEOUT_S", "Peer", "probe_pipeline"]
