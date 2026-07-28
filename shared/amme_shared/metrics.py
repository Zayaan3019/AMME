"""Prometheus metric definitions shared across services.

Latency histograms use explicit microsecond-scale buckets. The default
prometheus_client buckets start at 5ms, which is *the entire p99 budget* for
DeepLOB inference -- with defaults every inference lands in one bucket and the
latency regression test (s.6.4) could never detect anything.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter_ns

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

# 50us .. 1s, geometric-ish. Covers both the sub-ms quant path and the ms ML path.
LATENCY_BUCKETS_MS: tuple[float, ...] = (
    0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 25.0, 50.0, 100.0, 250.0, 1000.0,
)

REGISTRY = CollectorRegistry(auto_describe=True)

# --- Pipeline throughput ---------------------------------------------------
EVENTS_CONSUMED = Counter(
    "amme_events_consumed_total", "Events read from a stream", ["service", "stream"],
    registry=REGISTRY,
)
EVENTS_PUBLISHED = Counter(
    "amme_events_published_total", "Events written to a stream", ["service", "stream"],
    registry=REGISTRY,
)
EVENTS_FAILED = Counter(
    "amme_events_failed_total", "Events that raised during handling", ["service", "reason"],
    registry=REGISTRY,
)
STREAM_LAG = Gauge(
    "amme_stream_lag_messages", "Undelivered messages for this consumer group",
    ["service", "stream"], registry=REGISTRY,
)

# --- Latency ---------------------------------------------------------------
PROCESSING_LATENCY = Histogram(
    "amme_processing_latency_ms", "Per-event handler wall time",
    ["service", "stage"], buckets=LATENCY_BUCKETS_MS, registry=REGISTRY,
)
EVENT_STALENESS = Histogram(
    "amme_event_staleness_ms", "ts_ingest - ts_event, i.e. how far behind source we are",
    ["service"], buckets=LATENCY_BUCKETS_MS, registry=REGISTRY,
)
INFERENCE_LATENCY = Histogram(
    "amme_inference_latency_ms", "DeepLOB forward pass only",
    ["model_version"], buckets=LATENCY_BUCKETS_MS, registry=REGISTRY,
)

# --- Market / strategy state ----------------------------------------------
MID_PRICE = Gauge("amme_mid_price", "Current mid price", ["symbol"], registry=REGISTRY)
QUOTED_SPREAD = Gauge(
    "amme_quoted_spread", "Our ask - our bid, in quote currency", ["symbol"], registry=REGISTRY
)
RESERVATION_OFFSET = Gauge(
    "amme_reservation_offset", "reservation_price - mid_price", ["symbol"], registry=REGISTRY
)
INVENTORY = Gauge("amme_inventory", "Position in base currency", ["symbol"], registry=REGISTRY)
INVENTORY_UTILISATION = Gauge(
    "amme_inventory_utilisation", "|q| / q_max", ["symbol"], registry=REGISTRY
)
VPIN = Gauge("amme_vpin", "Volume-synchronised prob. of informed trading", ["symbol"],
             registry=REGISTRY)
OFI = Gauge("amme_ofi", "Order flow imbalance over window", ["symbol"], registry=REGISTRY)
REALIZED_VOL = Gauge("amme_realized_vol", "Rolling realized volatility of mid", ["symbol"],
                     registry=REGISTRY)
TOXICITY_MULTIPLIER = Gauge(
    "amme_toxicity_multiplier", "Spread widening factor currently applied", ["symbol"],
    registry=REGISTRY,
)
DRIFT_SIGNAL = Gauge("amme_drift_signal", "DeepLOB expected return", ["symbol"], registry=REGISTRY)

# --- P&L -------------------------------------------------------------------
TOTAL_PNL = Gauge("amme_total_pnl", "Realized + unrealized", ["symbol", "strategy"],
                  registry=REGISTRY)
REALIZED_PNL = Gauge("amme_realized_pnl", "Realized only", ["symbol", "strategy"],
                     registry=REGISTRY)
FILLS = Counter("amme_fills_total", "Passive fills", ["symbol", "side"], registry=REGISTRY)
QUOTES_SENT = Counter("amme_quotes_total", "Quotes accepted by execution", ["symbol"],
                      registry=REGISTRY)

# --- Risk ------------------------------------------------------------------
RISK_DECISIONS = Counter(
    "amme_risk_decisions_total", "risk-svc verdicts", ["action", "reason"], registry=REGISTRY
)
BREAKER_STATE = Gauge(
    "amme_circuit_breaker_tripped", "1 = quoting halted", ["symbol"], registry=REGISTRY
)

# --- Connection health -----------------------------------------------------
WS_RECONNECTS = Counter("amme_ws_reconnects_total", "WebSocket reconnections", ["symbol"],
                        registry=REGISTRY)
BOOK_RESYNCS = Counter(
    "amme_book_resyncs_total", "REST snapshot resyncs", ["symbol", "reason"], registry=REGISTRY
)
SERVICE_UP = Gauge("amme_service_up", "1 while the service loop is running", ["service"],
                   registry=REGISTRY)
# Exported because a *wrong* correction is otherwise undetectable: staleness is
# computed with this offset applied, so a bad offset makes the tape merely look
# slow. Alert on a step change here, not on staleness.
CLOCK_OFFSET_MS = Gauge(
    "amme_clock_offset_ms", "Local minus exchange clock, as last measured", ["symbol"],
    registry=REGISTRY
)


@contextmanager
def observe_latency(histogram: Histogram, *labels: str) -> Iterator[None]:
    """Time a block and record it in milliseconds.

    Uses ``perf_counter_ns`` rather than prometheus_client's built-in
    ``.time()`` because the built-in records *seconds*, and our buckets are in
    milliseconds.
    """
    start = perf_counter_ns()
    try:
        yield
    finally:
        elapsed_ms = (perf_counter_ns() - start) / 1_000_000.0
        (histogram.labels(*labels) if labels else histogram).observe(elapsed_ms)


def render_metrics() -> bytes:
    """Prometheus text exposition of the shared registry."""
    payload: bytes = generate_latest(REGISTRY)
    return payload


CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

__all__ = [
    "BOOK_RESYNCS",
    "BREAKER_STATE",
    "CLOCK_OFFSET_MS",
    "CONTENT_TYPE_LATEST",
    "DRIFT_SIGNAL",
    "EVENTS_CONSUMED",
    "EVENTS_FAILED",
    "EVENTS_PUBLISHED",
    "EVENT_STALENESS",
    "FILLS",
    "INFERENCE_LATENCY",
    "INVENTORY",
    "INVENTORY_UTILISATION",
    "LATENCY_BUCKETS_MS",
    "MID_PRICE",
    "OFI",
    "PROCESSING_LATENCY",
    "QUOTED_SPREAD",
    "QUOTES_SENT",
    "REALIZED_PNL",
    "REALIZED_VOL",
    "REGISTRY",
    "RESERVATION_OFFSET",
    "RISK_DECISIONS",
    "SERVICE_UP",
    "STREAM_LAG",
    "TOTAL_PNL",
    "TOXICITY_MULTIPLIER",
    "VPIN",
    "WS_RECONNECTS",
    "observe_latency",
    "render_metrics",
]
