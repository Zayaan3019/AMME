"""Injectable clocks.

Determinism (regression requirement s.6.1) is impossible if strategy code calls
``time.time()`` directly, so nothing in ``amme_core`` is allowed to. Every
component takes a :class:`Clock`; production wires :class:`SystemClock`, the
backtester wires :class:`SimulatedClock` driven off event timestamps.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

NS_PER_SECOND = 1_000_000_000


@runtime_checkable
class Clock(Protocol):
    def now_ns(self) -> int:
        """Current time in epoch nanoseconds."""

    def monotonic_ns(self) -> int:
        """Monotonic nanoseconds; only differences are meaningful."""


class SystemClock:
    """Wall clock. ``time.time_ns`` for epoch, ``perf_counter_ns`` for durations."""

    __slots__ = ()

    def now_ns(self) -> int:
        return time.time_ns()

    def monotonic_ns(self) -> int:
        return time.perf_counter_ns()


class SimulatedClock:
    """Deterministic clock advanced explicitly by the simulation driver."""

    __slots__ = ("_now_ns",)

    def __init__(self, start_ns: int = 0) -> None:
        self._now_ns = start_ns

    def now_ns(self) -> int:
        return self._now_ns

    def monotonic_ns(self) -> int:
        return self._now_ns

    def advance(self, delta_ns: int) -> int:
        if delta_ns < 0:
            raise ValueError(f"clock cannot move backwards (delta={delta_ns}ns)")
        self._now_ns += delta_ns
        return self._now_ns

    def set_to(self, ns: int) -> int:
        """Jump to an absolute time. Refuses to go backwards -- that would break
        the monotonicity every downstream windowed statistic assumes."""
        if ns < self._now_ns:
            raise ValueError(f"clock cannot move backwards ({self._now_ns} -> {ns})")
        self._now_ns = ns
        return self._now_ns


def seconds_to_ns(seconds: float) -> int:
    return int(seconds * NS_PER_SECOND)


def ns_to_seconds(ns: int) -> float:
    return ns / NS_PER_SECOND


__all__ = [
    "NS_PER_SECOND",
    "Clock",
    "SimulatedClock",
    "SystemClock",
    "ns_to_seconds",
    "seconds_to_ns",
]
