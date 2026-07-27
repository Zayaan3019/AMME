"""Order Flow Imbalance (Cont, Kukanov & Stoikov 2014).

*"The Price Impact of Order Book Events"*, Journal of Financial Econometrics
12(1), 47-88. Their central empirical result is that mid-price changes are
close to linear in OFI, with an R^2 far above what trade imbalance alone
achieves -- which is exactly why OFI is worth computing on the book rather than
on the tape.

For consecutive best-quote observations n-1, n:

.. math::

    e_n = \\mathbb{1}_{P^b_n \\ge P^b_{n-1}} q^b_n
        - \\mathbb{1}_{P^b_n \\le P^b_{n-1}} q^b_{n-1}
        - \\mathbb{1}_{P^a_n \\le P^a_{n-1}} q^a_n
        + \\mathbb{1}_{P^a_n \\ge P^a_{n-1}} q^a_{n-1}

Reading the three cases on the bid side makes the sign convention obvious:

* bid price **up**   -> ``+q^b_n``            (aggressive buy interest arrived)
* bid price **down** -> ``-q^b_{n-1}``        (the old bid was pulled or eaten)
* bid price **flat** -> ``q^b_n - q^b_{n-1}`` (net size added at the same price)

and symmetrically, with opposite sign, on the ask. Note that when the price is
unchanged *both* indicators fire, which is what produces the size-difference
term -- a detail that is easy to get wrong by writing the conditions as an
if/elif chain.

``OFI_k`` is then the sum of ``e_n`` over the last ``k`` observations.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BestQuote:
    """Top of book at one instant."""

    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float


def ofi_increment(prev: BestQuote, cur: BestQuote) -> float:
    """One ``e_n`` term. Pure, and the direct target of the hand-worked unit test."""
    bid_term = 0.0
    if cur.bid_price >= prev.bid_price:
        bid_term += cur.bid_size
    if cur.bid_price <= prev.bid_price:
        bid_term -= prev.bid_size

    ask_term = 0.0
    if cur.ask_price <= prev.ask_price:
        ask_term -= cur.ask_size
    if cur.ask_price >= prev.ask_price:
        ask_term += prev.ask_size

    return bid_term + ask_term


class OFICalculator:
    """Rolling OFI over a fixed number of book updates, plus its z-score.

    The z-score matters more than the raw level for gating decisions: OFI's
    scale is proportional to typical resting size, which varies by symbol, by
    time of day, and after any exchange tick-size change. Standardising makes
    one threshold (``|z| > 2``) portable.
    """

    __slots__ = ("_prev", "_sum", "_sum_sq", "_window", "_zsum", "_zwindow")

    def __init__(self, window: int = 50, zscore_window: int = 500) -> None:
        if window < 1 or zscore_window < 2:
            raise ValueError("window >= 1 and zscore_window >= 2 required")
        self._window: deque[float] = deque(maxlen=window)
        self._zwindow: deque[float] = deque(maxlen=zscore_window)
        self._prev: BestQuote | None = None
        self._sum = 0.0
        self._zsum = 0.0
        self._sum_sq = 0.0

    @property
    def warm(self) -> bool:
        """True once the rolling window is fully populated."""
        return len(self._window) == self._window.maxlen

    @property
    def value(self) -> float:
        """Current ``OFI_k``."""
        return self._sum

    def update(self, quote: BestQuote) -> float:
        """Absorb a new top-of-book observation and return the updated OFI."""
        if self._prev is None:
            self._prev = quote
            return 0.0

        e = ofi_increment(self._prev, quote)
        self._prev = quote

        # Incremental sum: subtract whatever the bounded deque is about to evict.
        if len(self._window) == self._window.maxlen and self._window.maxlen:
            self._sum -= self._window[0]
        self._window.append(e)
        self._sum += e

        ofi = self._sum
        if len(self._zwindow) == self._zwindow.maxlen and self._zwindow.maxlen:
            evicted = self._zwindow[0]
            self._zsum -= evicted
            self._sum_sq -= evicted * evicted
        self._zwindow.append(ofi)
        self._zsum += ofi
        self._sum_sq += ofi * ofi

        return ofi

    def zscore(self) -> float:
        """OFI standardised by its own rolling mean/stdev. 0.0 until meaningful."""
        n = len(self._zwindow)
        if n < 2:
            return 0.0
        mean = self._zsum / n
        # Guard against catastrophic cancellation making the variance negative.
        variance = max(0.0, self._sum_sq / n - mean * mean)
        std = math.sqrt(variance)
        if std < 1e-12:
            return 0.0
        return (self._sum - mean) / std

    def reset(self) -> None:
        self._window.clear()
        self._zwindow.clear()
        self._prev = None
        self._sum = self._zsum = self._sum_sq = 0.0


__all__ = ["BestQuote", "OFICalculator", "ofi_increment"]
