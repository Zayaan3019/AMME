"""Performance analytics for the regression harness (s.6.2).

Every metric here is computed from a single time series of portfolio snapshots,
so all strategies in an ablation are measured by identical code on identically
sampled data. That is the point -- an ablation where the baseline and the
treatment are scored by different code proves nothing.

Sharpe deserves a note, because it is the number most easily fudged in a
market-making backtest. Three choices are made explicitly:

1. **P&L changes, not returns.** A market maker's capital base is the risk limit,
   not the position, and the position crosses zero constantly -- percentage
   returns are undefined there. We use the standard deviation of P&L increments
   as the risk denominator.
2. **Annualisation by wall-clock seconds** (31,536,000/year), because crypto
   trades continuously. Using 252 trading days would inflate the number ~2.4x.
3. **Fixed-interval resampling** before computing moments. Sampling per-event
   would let a strategy that quotes more often look less volatile purely
   because of its event count.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from itertools import pairwise

import numpy as np

SECONDS_PER_YEAR = 31_536_000.0


@dataclass(frozen=True, slots=True)
class PnLPoint:
    """One sample of the strategy's state."""

    ts_ns: int
    total_pnl: float
    realized_pnl: float
    inventory: float
    mid_price: float
    vpin: float = 0.0
    spread_quoted: float = 0.0
    n_fills: int = 0
    n_quotes: int = 0


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """The ablation report's row type."""

    strategy: str
    duration_seconds: float
    n_samples: int

    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    fees_paid: float

    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_pct: float
    pnl_per_second: float
    pnl_volatility: float

    inventory_mean: float
    inventory_std: float
    inventory_variance: float
    inventory_max_abs: float
    time_at_limit_pct: float

    n_fills: int
    n_quotes: int
    fill_ratio: float
    avg_spread_quoted: float
    spread_captured: float
    volume_traded: float

    # Toxicity split -- the headline number for the s.6.3 toxicity ablation.
    pnl_high_vpin: float
    pnl_low_vpin: float
    vpin_split_threshold: float
    high_vpin_seconds: float
    toxicity_adjusted_pnl: float

    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _resample(points: Sequence[PnLPoint], interval_seconds: float) -> np.ndarray:
    """Last-observation-carried-forward P&L on a fixed grid."""
    if len(points) < 2:
        return np.array([p.total_pnl for p in points], dtype=float)

    start, end = points[0].ts_ns, points[-1].ts_ns
    step_ns = max(1, int(interval_seconds * 1e9))
    if end <= start:
        return np.array([points[-1].total_pnl], dtype=float)

    grid = np.arange(start, end + step_ns, step_ns, dtype=np.int64)
    ts = np.fromiter((p.ts_ns for p in points), dtype=np.int64, count=len(points))
    pnl = np.fromiter((p.total_pnl for p in points), dtype=float, count=len(points))
    idx = np.searchsorted(ts, grid, side="right") - 1
    idx = np.clip(idx, 0, len(pnl) - 1)
    return pnl[idx]


def sharpe_ratio(
    pnl_series: np.ndarray, interval_seconds: float, risk_free_rate: float = 0.0
) -> float:
    """Annualised Sharpe of P&L increments. 0.0 when undefined."""
    if pnl_series.size < 3:
        return 0.0
    increments = np.diff(pnl_series)
    if increments.size < 2:
        return 0.0
    std = float(np.std(increments, ddof=1))
    if std < 1e-12:
        return 0.0
    mean = float(np.mean(increments)) - risk_free_rate * interval_seconds
    periods_per_year = SECONDS_PER_YEAR / interval_seconds
    return (mean / std) * math.sqrt(periods_per_year)


def sortino_ratio(pnl_series: np.ndarray, interval_seconds: float) -> float:
    """Like Sharpe but penalising only downside deviation."""
    if pnl_series.size < 3:
        return 0.0
    increments = np.diff(pnl_series)
    downside = increments[increments < 0.0]
    if downside.size < 2:
        return 0.0
    downside_std = float(np.sqrt(np.mean(downside**2)))
    if downside_std < 1e-12:
        return 0.0
    periods_per_year = SECONDS_PER_YEAR / interval_seconds
    return (float(np.mean(increments)) / downside_std) * math.sqrt(periods_per_year)


def max_drawdown(pnl_series: np.ndarray) -> tuple[float, float]:
    """``(absolute, fractional)`` peak-to-trough decline.

    The fractional figure is relative to the running peak *level*, which is only
    meaningful once the strategy is in profit; it is reported as 0.0 while the
    peak is non-positive rather than producing a nonsense percentage.
    """
    if pnl_series.size == 0:
        return 0.0, 0.0
    running_peak = np.maximum.accumulate(pnl_series)
    drawdowns = running_peak - pnl_series
    max_dd = float(np.max(drawdowns))
    peak_at_max = float(running_peak[int(np.argmax(drawdowns))])
    pct = max_dd / peak_at_max if peak_at_max > 1e-12 else 0.0
    return max_dd, pct


def compute_metrics(
    strategy: str,
    points: Sequence[PnLPoint],
    *,
    interval_seconds: float = 1.0,
    q_max: float = 0.05,
    fees_paid: float = 0.0,
    volume_traded: float = 0.0,
    vpin_split: float | None = None,
    risk_free_rate: float = 0.0,
    extra: dict[str, float] | None = None,
) -> PerformanceMetrics:
    """Reduce a trajectory to the ablation report's row."""
    if not points:
        return _empty_metrics(strategy)

    duration = (points[-1].ts_ns - points[0].ts_ns) / 1e9
    pnl_grid = _resample(points, interval_seconds)

    inventories = np.fromiter((p.inventory for p in points), dtype=float, count=len(points))
    spreads = np.fromiter((p.spread_quoted for p in points), dtype=float, count=len(points))
    quoted = spreads[spreads > 0.0]
    vpins = np.fromiter((p.vpin for p in points), dtype=float, count=len(points))

    last = points[-1]
    unrealized = last.total_pnl - last.realized_pnl
    dd_abs, dd_pct = max_drawdown(pnl_grid)
    increments = np.diff(pnl_grid) if pnl_grid.size > 1 else np.array([0.0])

    # Split P&L by toxicity regime. The threshold is the median VPIN of the run
    # unless pinned, so "high" and "low" always have comparable sample counts
    # instead of one bucket being empty on a calm day.
    threshold = float(np.median(vpins)) if vpin_split is None else vpin_split
    pnl_high, pnl_low, high_seconds = _split_pnl_by_vpin(points, threshold)

    n_fills = last.n_fills
    n_quotes = last.n_quotes

    return PerformanceMetrics(
        strategy=strategy,
        duration_seconds=duration,
        n_samples=len(points),
        total_pnl=last.total_pnl,
        realized_pnl=last.realized_pnl,
        unrealized_pnl=unrealized,
        fees_paid=fees_paid,
        sharpe=sharpe_ratio(pnl_grid, interval_seconds, risk_free_rate),
        sortino=sortino_ratio(pnl_grid, interval_seconds),
        max_drawdown=dd_abs,
        max_drawdown_pct=dd_pct,
        pnl_per_second=last.total_pnl / duration if duration > 0 else 0.0,
        pnl_volatility=float(np.std(increments, ddof=1)) if increments.size > 1 else 0.0,
        inventory_mean=float(np.mean(inventories)),
        inventory_std=float(np.std(inventories)),
        inventory_variance=float(np.var(inventories)),
        inventory_max_abs=float(np.max(np.abs(inventories))),
        time_at_limit_pct=float(np.mean(np.abs(inventories) >= 0.99 * q_max) * 100.0)
        if q_max > 0
        else 0.0,
        n_fills=n_fills,
        n_quotes=n_quotes,
        fill_ratio=n_fills / n_quotes if n_quotes > 0 else 0.0,
        avg_spread_quoted=float(np.mean(quoted)) if quoted.size else 0.0,
        spread_captured=last.realized_pnl / n_fills if n_fills > 0 else 0.0,
        volume_traded=volume_traded,
        pnl_high_vpin=pnl_high,
        pnl_low_vpin=pnl_low,
        vpin_split_threshold=threshold,
        high_vpin_seconds=high_seconds,
        toxicity_adjusted_pnl=pnl_high,
        extra=extra or {},
    )


def _split_pnl_by_vpin(
    points: Sequence[PnLPoint], threshold: float
) -> tuple[float, float, float]:
    """P&L *accrued* during high- vs low-VPIN periods, and time spent high.

    Attribution is on the increment between consecutive samples, tagged by the
    VPIN at the *start* of the interval. Tagging by the end would attribute a
    loss to the toxicity reading that the loss itself helped cause.
    """
    high = low = high_seconds = 0.0
    for prev, cur in pairwise(points):
        delta = cur.total_pnl - prev.total_pnl
        dt = (cur.ts_ns - prev.ts_ns) / 1e9
        if prev.vpin >= threshold:
            high += delta
            high_seconds += dt
        else:
            low += delta
    return high, low, high_seconds


def _empty_metrics(strategy: str) -> PerformanceMetrics:
    return PerformanceMetrics(
        strategy=strategy, duration_seconds=0.0, n_samples=0, total_pnl=0.0, realized_pnl=0.0,
        unrealized_pnl=0.0, fees_paid=0.0, sharpe=0.0, sortino=0.0, max_drawdown=0.0,
        max_drawdown_pct=0.0, pnl_per_second=0.0, pnl_volatility=0.0, inventory_mean=0.0,
        inventory_std=0.0, inventory_variance=0.0, inventory_max_abs=0.0, time_at_limit_pct=0.0,
        n_fills=0, n_quotes=0, fill_ratio=0.0, avg_spread_quoted=0.0, spread_captured=0.0,
        volume_traded=0.0, pnl_high_vpin=0.0, pnl_low_vpin=0.0, vpin_split_threshold=0.0,
        high_vpin_seconds=0.0, toxicity_adjusted_pnl=0.0,
    )


__all__ = [
    "SECONDS_PER_YEAR",
    "PerformanceMetrics",
    "PnLPoint",
    "compute_metrics",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
]
