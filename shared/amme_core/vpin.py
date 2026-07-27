"""VPIN -- Volume-Synchronised Probability of Informed Trading.

Easley, Lopez de Prado & O'Hara (2012), *"Flow Toxicity and Liquidity in a High
Frequency World"*, Review of Financial Studies 25(5), 1457-1493.

The idea: sample in **volume time**, not clock time. Cut the tape into buckets
of exactly ``V`` units of volume, split each bucket into buy- and sell-initiated
volume, and average the normalised imbalance over the last ``n`` buckets:

.. math::

    \\mathrm{VPIN} = \\frac{\\sum_{\\tau=1}^{n} |V^B_\\tau - V^S_\\tau|}{n V}

Volume-clock sampling is the whole point -- it makes the series roughly
i.i.d. and removes the intraday volume seasonality that wrecks clock-time
toxicity measures.

Two classification schemes are supported:

``TRUE_AGGRESSOR``
    Use the exchange's own aggressor flag. Binance's ``aggTrade`` carries ``m``
    ("buyer is the market maker"), so ``m == True`` means the **seller** was
    aggressive. This is exact, and it is the default for the crypto path --
    there is no reason to estimate a quantity the exchange hands you.

``BVC`` (Bulk Volume Classification)
    The paper's own estimator, for feeds without an aggressor flag (e.g. the
    LOBSTER equities adapter):

    .. math:: V^B_\\tau = V \\cdot Z\\!\\left(\\frac{P_\\tau - P_{\\tau-1}}{\\sigma_{\\Delta P}}\\right)

    with :math:`Z` the standard normal CDF. Note the consequence
    :math:`|V^B - V^S| = V|2Z(x) - 1|`, so VPIN under BVC is bounded in [0, 1]
    by construction.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from amme_shared.events import AggressorSide


class ClassificationScheme(StrEnum):
    TRUE_AGGRESSOR = "true_aggressor"
    BVC = "bvc"
    TICK_RULE = "tick_rule"


@dataclass(frozen=True, slots=True)
class VolumeBucket:
    """A completed equal-volume bucket."""

    buy_volume: float
    sell_volume: float
    close_price: float
    n_trades: int

    @property
    def total_volume(self) -> float:
        return self.buy_volume + self.sell_volume

    @property
    def imbalance(self) -> float:
        return abs(self.buy_volume - self.sell_volume)


def normal_cdf(x: float) -> float:
    """Standard normal CDF via ``erf``. Avoids a scipy dependency in the core."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class VPINCalculator:
    """Streaming VPIN.

    Handles the detail that trades routinely *straddle* a bucket boundary: a
    trade larger than the remaining bucket capacity is split proportionally
    across buckets rather than assigned wholesale to one. Skipping that makes
    VPIN sensitive to bucket size in a way the estimator is not supposed to be.
    """

    __slots__ = (
        "_bucket_buy",
        "_bucket_n",
        "_bucket_sell",
        "_buckets",
        "_bucket_volume",
        "_imbalance_sum",
        "_last_price",
        "_last_bucket_close",
        "_price_change_sq_sum",
        "_price_change_n",
        "_scheme",
        "_tick_sign",
    )

    def __init__(
        self,
        bucket_volume: float = 1.0,
        n_buckets: int = 50,
        scheme: ClassificationScheme = ClassificationScheme.TRUE_AGGRESSOR,
    ) -> None:
        if bucket_volume <= 0:
            raise ValueError("bucket_volume must be positive")
        if n_buckets < 1:
            raise ValueError("n_buckets must be >= 1")
        self._bucket_volume = bucket_volume
        self._buckets: deque[VolumeBucket] = deque(maxlen=n_buckets)
        self._scheme = scheme

        self._bucket_buy = 0.0
        self._bucket_sell = 0.0
        self._bucket_n = 0
        self._imbalance_sum = 0.0

        self._last_price: float | None = None
        self._last_bucket_close: float | None = None
        self._price_change_sq_sum = 0.0
        self._price_change_n = 0
        self._tick_sign = 1.0

    @property
    def n_complete_buckets(self) -> int:
        return len(self._buckets)

    @property
    def warm(self) -> bool:
        """VPIN is meaningless until ``n_buckets`` have filled -- s.5.2 exposes
        this as ``vpin_warm`` so downstream gating can stay neutral until then."""
        return len(self._buckets) == self._buckets.maxlen

    @property
    def value(self) -> float:
        """Current VPIN in [0, 1]. 0.0 before any bucket completes."""
        if not self._buckets:
            return 0.0
        denominator = len(self._buckets) * self._bucket_volume
        return min(1.0, self._imbalance_sum / denominator) if denominator > 0 else 0.0

    @property
    def bucket_fill_fraction(self) -> float:
        return (self._bucket_buy + self._bucket_sell) / self._bucket_volume

    def _classify(self, price: float, size: float, aggressor: AggressorSide) -> tuple[float, float]:
        """Split ``size`` into (buy_initiated, sell_initiated)."""
        if self._scheme is ClassificationScheme.TRUE_AGGRESSOR and aggressor in (
            AggressorSide.BUY,
            AggressorSide.SELL,
        ):
            return (size, 0.0) if aggressor is AggressorSide.BUY else (0.0, size)

        if self._scheme is ClassificationScheme.BVC:
            # Probabilistic split driven by the standardised price change.
            if self._last_bucket_close is None or self._price_change_n < 2:
                return 0.5 * size, 0.5 * size
            sigma = math.sqrt(self._price_change_sq_sum / self._price_change_n)
            if sigma < 1e-12:
                return 0.5 * size, 0.5 * size
            z = normal_cdf((price - self._last_bucket_close) / sigma)
            return size * z, size * (1.0 - z)

        # Tick rule: uptick => buy-initiated, downtick => sell-initiated, repeat
        # the last sign on a zero tick. This is also where TRUE_AGGRESSOR lands
        # when the feed hands us AggressorSide.UNKNOWN.
        if self._last_price is not None:
            if price > self._last_price:
                self._tick_sign = 1.0
            elif price < self._last_price:
                self._tick_sign = -1.0
        return (size, 0.0) if self._tick_sign > 0 else (0.0, size)

    def _close_bucket(self, close_price: float) -> VolumeBucket:
        bucket = VolumeBucket(
            buy_volume=self._bucket_buy,
            sell_volume=self._bucket_sell,
            close_price=close_price,
            n_trades=self._bucket_n,
        )
        if len(self._buckets) == self._buckets.maxlen and self._buckets.maxlen:
            self._imbalance_sum -= self._buckets[0].imbalance
        self._buckets.append(bucket)
        self._imbalance_sum += bucket.imbalance

        if self._last_bucket_close is not None:
            delta = close_price - self._last_bucket_close
            self._price_change_sq_sum += delta * delta
            self._price_change_n += 1
        self._last_bucket_close = close_price

        self._bucket_buy = 0.0
        self._bucket_sell = 0.0
        self._bucket_n = 0
        return bucket

    def update(
        self, price: float, size: float, aggressor: AggressorSide = AggressorSide.UNKNOWN
    ) -> list[VolumeBucket]:
        """Absorb one trade. Returns any buckets completed by it (usually none)."""
        if size <= 0:
            return []

        completed: list[VolumeBucket] = []
        remaining = size

        while remaining > 0:
            capacity = self._bucket_volume - (self._bucket_buy + self._bucket_sell)
            chunk = min(remaining, capacity)
            buy, sell = self._classify(price, chunk, aggressor)
            self._bucket_buy += buy
            self._bucket_sell += sell
            self._bucket_n += 1
            remaining -= chunk

            if (self._bucket_buy + self._bucket_sell) >= self._bucket_volume - 1e-12:
                completed.append(self._close_bucket(price))

        self._last_price = price
        return completed

    def reset(self) -> None:
        self._buckets.clear()
        self._bucket_buy = self._bucket_sell = 0.0
        self._bucket_n = 0
        self._imbalance_sum = 0.0
        self._last_price = None
        self._last_bucket_close = None
        self._price_change_sq_sum = 0.0
        self._price_change_n = 0
        self._tick_sign = 1.0


__all__ = [
    "ClassificationScheme",
    "VPINCalculator",
    "VolumeBucket",
    "normal_cdf",
]
