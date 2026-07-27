"""Realized volatility of the mid price, in absolute quote-currency units.

Units matter here and are easy to get wrong. Avellaneda-Stoikov models the mid
as arithmetic Brownian motion, :math:`dS_t = \\sigma dW_t`, so the ``sigma`` it
wants is in **price units per sqrt(second)** -- not a return volatility, not an
annualised percentage. The estimator below is therefore the realized-variance
*rate*:

.. math:: \\hat\\sigma^2 = \\frac{\\sum_i (\\Delta S_i)^2}{\\sum_i \\Delta t_i}

which is consistent for the diffusion coefficient of arithmetic BM under
irregular sampling -- and irregular sampling is the normal case here, since
Binance emits depth updates only when the book actually changes.

Two estimators are provided. Both are exposed because the circuit breaker wants
a *fast* reaction to a vol spike while the quoter wants a *stable* sigma, and
using one series for both forces a bad compromise.
"""

from __future__ import annotations

import math
from collections import deque


class RealizedVolatility:
    """Rolling realized-variance-rate estimator over a fixed number of samples."""

    __slots__ = ("_dt_sum", "_last_price", "_last_ts_ns", "_samples", "_sq_sum")

    def __init__(self, window: int = 200) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self._samples: deque[tuple[float, float]] = deque(maxlen=window)  # (dS^2, dt)
        self._sq_sum = 0.0
        self._dt_sum = 0.0
        self._last_price: float | None = None
        self._last_ts_ns: int | None = None

    @property
    def warm(self) -> bool:
        return len(self._samples) == self._samples.maxlen

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    def update(self, price: float, ts_ns: int) -> float:
        """Absorb a mid observation; return sigma in price units per sqrt(second)."""
        if self._last_price is None or self._last_ts_ns is None:
            self._last_price, self._last_ts_ns = price, ts_ns
            return self.value

        dt = (ts_ns - self._last_ts_ns) / 1e9
        if dt <= 0.0:
            # Same-nanosecond or out-of-order update: keep the newer price but
            # contribute no variance -- dividing by dt=0 would blow up.
            self._last_price = price
            return self.value

        d_s = price - self._last_price
        self._last_price, self._last_ts_ns = price, ts_ns

        if len(self._samples) == self._samples.maxlen and self._samples.maxlen:
            old_sq, old_dt = self._samples[0]
            self._sq_sum -= old_sq
            self._dt_sum -= old_dt
        self._samples.append((d_s * d_s, dt))
        self._sq_sum += d_s * d_s
        self._dt_sum += dt

        return self.value

    @property
    def value(self) -> float:
        """sigma, price units per sqrt(second). 0.0 before any sample lands."""
        if self._dt_sum <= 0.0:
            return 0.0
        return math.sqrt(max(0.0, self._sq_sum / self._dt_sum))

    @property
    def variance_rate(self) -> float:
        return 0.0 if self._dt_sum <= 0.0 else max(0.0, self._sq_sum / self._dt_sum)

    def reset(self) -> None:
        self._samples.clear()
        self._sq_sum = self._dt_sum = 0.0
        self._last_price = self._last_ts_ns = None


class VolatilityBaseline:
    """A slow reference level for "what is normal volatility right now".

    The circuit breaker needs a ratio, ``fast / baseline``, and the baseline has
    to be genuinely slow or the ratio is just noise over noise. Comparing a fast
    EWMA against the *rolling-window* sigma -- the obvious shortcut -- trips
    constantly, because both track the same recent data and their ratio wanders
    around 1 with high variance.

    This tracks an EWMA of the sigma series with **asymmetric** rates: it rises
    slowly and falls faster. Rising slowly means a 5-second spike barely moves
    it, so the breaker stays armed through the event. Rising *eventually* means
    a genuinely sustained regime change is absorbed over minutes rather than
    leaving the breaker latched forever.

    An earlier version froze the baseline outright whenever
    ``sigma > 2 * baseline``. That was wrong in a way worth recording: during
    warm-up the first non-zero sigma is tiny, the baseline latched onto it, and
    every subsequent reading then exceeded the freeze threshold -- so it froze
    permanently at a near-zero value and the fast/baseline ratio sat at a median
    of **3.48** instead of ~1.0, tripping the breaker 33% of the time. A hard
    freeze has no way back once it starts from a bad value; asymmetric decay
    always converges.
    """

    __slots__ = ("_alpha_down", "_alpha_up", "_n", "_value", "_warm_samples")

    def __init__(
        self,
        halflife: float = 3000.0,
        fall_halflife: float = 600.0,
        warm_samples: int = 200,
    ) -> None:
        if halflife <= 0.0 or fall_halflife <= 0.0:
            raise ValueError("halflives must be positive")
        self._alpha_up = 1.0 - math.exp(-math.log(2.0) / halflife)
        self._alpha_down = 1.0 - math.exp(-math.log(2.0) / fall_halflife)
        self._warm_samples = warm_samples
        self._value = 0.0
        self._n = 0

    @property
    def warm(self) -> bool:
        """False until enough samples have landed. The breaker must not arm
        before this -- an unconverged baseline produces meaningless ratios."""
        return self._n >= self._warm_samples

    @property
    def value(self) -> float:
        return self._value

    def update(self, sigma: float) -> float:
        if sigma <= 0.0:
            return self._value
        self._n += 1
        if self._value <= 0.0:
            self._value = sigma
            return self._value
        # During warm-up, track hard in both directions so the baseline reaches
        # a sensible level before any asymmetry is applied.
        if self._n < self._warm_samples:
            self._value += 0.05 * (sigma - self._value)
            return self._value
        alpha = self._alpha_up if sigma > self._value else self._alpha_down
        self._value += alpha * (sigma - self._value)
        return self._value

    def reset(self) -> None:
        self._value = 0.0
        self._n = 0


class EWMAVolatility:
    """Exponentially-weighted variant, for the circuit breaker.

    Reacts to a regime change in ``~halflife`` samples instead of waiting for a
    fixed window to roll over, which is the behaviour you want from a kill
    switch and *not* the behaviour you want from the sigma feeding the spread.
    """

    __slots__ = ("_alpha", "_last_price", "_last_ts_ns", "_n", "_var_rate")

    def __init__(self, halflife: float = 50.0) -> None:
        if halflife <= 0.0:
            raise ValueError("halflife must be positive")
        self._alpha = 1.0 - math.exp(-math.log(2.0) / halflife)
        self._var_rate = 0.0
        self._last_price: float | None = None
        self._last_ts_ns: int | None = None
        self._n = 0

    @property
    def warm(self) -> bool:
        return self._n >= 30

    def update(self, price: float, ts_ns: int) -> float:
        if self._last_price is None or self._last_ts_ns is None:
            self._last_price, self._last_ts_ns = price, ts_ns
            return 0.0

        dt = (ts_ns - self._last_ts_ns) / 1e9
        if dt <= 0.0:
            self._last_price = price
            return self.value

        d_s = price - self._last_price
        self._last_price, self._last_ts_ns = price, ts_ns

        instantaneous = (d_s * d_s) / dt
        self._var_rate += self._alpha * (instantaneous - self._var_rate)
        self._n += 1
        return self.value

    @property
    def value(self) -> float:
        return math.sqrt(max(0.0, self._var_rate))

    def reset(self) -> None:
        self._var_rate = 0.0
        self._last_price = self._last_ts_ns = None
        self._n = 0


__all__ = ["EWMAVolatility", "RealizedVolatility", "VolatilityBaseline"]
