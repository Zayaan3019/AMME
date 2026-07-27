"""Exchange-grid rounding.

Kept separate and explicit because rounding direction is a *risk* decision, not
a formatting detail: rounding a bid up or an ask down makes the quote more
aggressive than the model intended, which converts a tick of granularity into a
systematic adverse-selection leak. Everything that reaches an exchange goes
through here with an explicit mode.
"""

from __future__ import annotations

import math
from typing import Literal

RoundMode = Literal["nearest", "up", "down"]


def clamp(value: float, low: float, high: float) -> float:
    if low > high:
        raise ValueError(f"clamp bounds inverted: [{low}, {high}]")
    return max(low, min(high, value))


def round_to_tick(price: float, tick_size: float, mode: RoundMode = "nearest") -> float:
    """Snap ``price`` onto the exchange's price grid.

    The final ``round(..., 10)`` removes float representation noise: with
    ``tick_size = 0.01`` the naive result is often ``68000.010000000002``, which
    some exchanges reject outright.
    """
    if tick_size <= 0.0:
        raise ValueError(f"tick_size must be positive, got {tick_size}")
    ticks = price / tick_size
    if mode == "up":
        snapped = math.ceil(ticks - 1e-9)
    elif mode == "down":
        snapped = math.floor(ticks + 1e-9)
    else:
        snapped = math.floor(ticks + 0.5)
    return round(snapped * tick_size, 10)


def round_to_lot(size: float, lot_size: float, mode: RoundMode = "down") -> float:
    """Snap ``size`` onto the size grid. Defaults to rounding *down* -- never
    send more risk than intended because of lot granularity."""
    if lot_size <= 0.0:
        raise ValueError(f"lot_size must be positive, got {lot_size}")
    lots = size / lot_size
    if mode == "up":
        snapped = math.ceil(lots - 1e-9)
    elif mode == "nearest":
        snapped = math.floor(lots + 0.5)
    else:
        snapped = math.floor(lots + 1e-9)
    return round(max(0.0, snapped * lot_size), 12)


def ticks_between(a: float, b: float, tick_size: float) -> float:
    if tick_size <= 0.0:
        raise ValueError(f"tick_size must be positive, got {tick_size}")
    return (b - a) / tick_size


__all__ = ["RoundMode", "clamp", "round_to_lot", "round_to_tick", "ticks_between"]
