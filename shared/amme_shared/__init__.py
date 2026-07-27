"""Shared contracts and infrastructure for every AMME service.

Split of responsibility:

* ``amme_shared`` -- I/O, transport, configuration, observability. Anything that
  touches the network, the clock, or the environment.
* ``amme_core``   -- pure deterministic domain logic. No I/O, no wall clock, no
  globals. This is what the backtester and the live services both import, and
  it is why a replay run reproduces bit-for-bit.

Services depend on both; services never depend on each other.
"""

from __future__ import annotations

from amme_shared.clock import Clock, SimulatedClock, SystemClock
from amme_shared.config import Settings, get_settings
from amme_shared.events import (
    SCHEMA_VERSION,
    AggressorSide,
    BaseEvent,
    DataSource,
    DriftSignal,
    FeatureVector,
    Fill,
    OrderBookUpdate,
    PortfolioState,
    PriceLevel,
    QuoteIntent,
    RiskAction,
    RiskDecision,
    Side,
    TradeTick,
)
from amme_shared.logging import configure_logging, correlation_scope, get_logger
from amme_shared.streams import EventBus, InMemoryEventBus, RedisEventBus, Streams

__version__ = "0.1.0"

__all__ = [
    "SCHEMA_VERSION",
    "AggressorSide",
    "BaseEvent",
    "Clock",
    "DataSource",
    "DriftSignal",
    "EventBus",
    "FeatureVector",
    "Fill",
    "InMemoryEventBus",
    "OrderBookUpdate",
    "PortfolioState",
    "PriceLevel",
    "QuoteIntent",
    "RedisEventBus",
    "RiskAction",
    "RiskDecision",
    "Settings",
    "Side",
    "SimulatedClock",
    "Streams",
    "SystemClock",
    "TradeTick",
    "__version__",
    "configure_logging",
    "correlation_scope",
    "get_logger",
    "get_settings",
]
