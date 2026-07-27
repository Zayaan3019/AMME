"""Feature extraction: raw book and tape -> the tensors and scalars downstream needs.

Two consumers with different requirements share this pipeline:

* **quoting-engine-svc** wants scalars -- sigma, VPIN, OFI z-score.
* **deeplob-inference-svc** wants a normalised ``(T, 4L)`` tensor.

**Normalisation** deliberately departs from the DeepLOB paper, and the reason is
worth stating because it is the difference between a model that works in a
backtest and one that works next month. Zhang et al. z-score raw prices and
volumes using the previous five days' statistics. That is fine on FI-2010, a
fixed historical dataset, but raw price levels are *non-stationary*: BTC at
50,000 and BTC at 90,000 produce completely different feature distributions, so
a model trained at one level silently degrades at another.

Instead:

* **Price columns** become distance from the current mid, in basis points:
  ``(p_i - mid) / mid * 10000``. Stationary by construction and scale-free
  across symbols and price regimes.
* **Volume columns** are divided by the rolling mean total depth, then
  log1p-compressed. Depth is heavy-tailed, so raw sizes let one large iceberg
  dominate the input; log1p bounds that without discarding the ordering.

Both transforms are causal -- they use only information available at time ``t``.
That matters as much as the choice itself: a normalisation fitted over the whole
dataset leaks future information into every training sample, and is one of the
most common reasons a published LOB model fails to reproduce.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from amme_core.ofi import BestQuote, OFICalculator
from amme_core.volatility import EWMAVolatility, RealizedVolatility, VolatilityBaseline
from amme_core.vpin import ClassificationScheme, VPINCalculator
from amme_shared.events import AggressorSide, OrderBookUpdate, TradeTick

#: Column layout per level, matching DeepLOB's input convention.
FEATURES_PER_LEVEL = 4  # ask_price, ask_volume, bid_price, bid_volume


@dataclass(frozen=True, slots=True)
class FeaturePipelineConfig:
    levels: int = 10
    lookback: int = 100
    vpin_bucket_volume: float = 1.0
    vpin_buckets: int = 50
    vpin_scheme: ClassificationScheme = ClassificationScheme.TRUE_AGGRESSOR
    ofi_window: int = 50
    ofi_zscore_window: int = 500
    vol_window: int = 200
    vol_halflife: float = 50.0
    #: Slow reference for the circuit breaker's vol-spike ratio. Long by design;
    #: see :class:`~amme_core.volatility.VolatilityBaseline`.
    vol_baseline_halflife: float = 2000.0
    depth_norm_window: int = 500
    emit_tensor: bool = True


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    """Everything derived from one book update."""

    ts_ns: int
    mid_price: float
    micro_price: float
    spread: float
    sigma: float
    sigma_fast: float
    sigma_baseline: float
    vol_warm: bool
    vpin: float
    vpin_warm: bool
    ofi: float
    ofi_zscore: float
    ofi_warm: bool
    tensor: np.ndarray | None = None

    @property
    def tensor_shape(self) -> tuple[int, int] | None:
        if self.tensor is None:
            return None
        rows, cols = self.tensor.shape
        return int(rows), int(cols)


class FeaturePipeline:
    """Stateful streaming feature extractor.

    Pure in the ``amme_core`` sense: no I/O, no wall clock, no hidden
    randomness. The same instance type runs inside ``feature-svc`` and inside
    the backtester, so features can never silently diverge between the two.
    """

    __slots__ = (
        "_depth_sum",
        "_depth_window",
        "_ewma_vol",
        "_last_book",
        "_ofi",
        "_rows",
        "_vol",
        "_vol_baseline",
        "_vpin",
        "config",
    )

    def __init__(self, config: FeaturePipelineConfig | None = None) -> None:
        self.config = config or FeaturePipelineConfig()
        c = self.config
        self._ofi = OFICalculator(window=c.ofi_window, zscore_window=c.ofi_zscore_window)
        self._vpin = VPINCalculator(
            bucket_volume=c.vpin_bucket_volume, n_buckets=c.vpin_buckets, scheme=c.vpin_scheme
        )
        self._vol = RealizedVolatility(window=c.vol_window)
        self._ewma_vol = EWMAVolatility(halflife=c.vol_halflife)
        self._vol_baseline = VolatilityBaseline(halflife=c.vol_baseline_halflife)
        self._rows: deque[np.ndarray] = deque(maxlen=c.lookback)
        self._depth_window: deque[float] = deque(maxlen=c.depth_norm_window)
        self._depth_sum = 0.0
        self._last_book: OrderBookUpdate | None = None

    # -- accessors used by the services ------------------------------------
    @property
    def vpin(self) -> float:
        return self._vpin.value

    @property
    def sigma(self) -> float:
        return self._vol.value

    @property
    def tensor_ready(self) -> bool:
        return len(self._rows) == self.config.lookback

    # -- ingestion ---------------------------------------------------------
    def on_trade(self, trade: TradeTick) -> None:
        self._vpin.update(trade.price, trade.size, trade.aggressor)

    def on_trades(self, trades: list[TradeTick]) -> None:
        for trade in trades:
            self.on_trade(trade)

    def on_raw_trade(
        self, price: float, size: float, aggressor: AggressorSide = AggressorSide.UNKNOWN
    ) -> None:
        """Trade ingestion without constructing an event. Used by the backtester
        inner loop, where pydantic validation would dominate the profile."""
        self._vpin.update(price, size, aggressor)

    def _normalise_row(self, book: OrderBookUpdate, mid: float) -> np.ndarray:
        c = self.config
        row = np.zeros(c.levels * FEATURES_PER_LEVEL, dtype=np.float32)

        total_depth = sum(lv.size for lv in book.bids) + sum(lv.size for lv in book.asks)
        if len(self._depth_window) == self._depth_window.maxlen and self._depth_window.maxlen:
            self._depth_sum -= self._depth_window[0]
        self._depth_window.append(total_depth)
        self._depth_sum += total_depth
        mean_depth = self._depth_sum / len(self._depth_window) if self._depth_window else 1.0
        depth_scale = mean_depth / (2.0 * c.levels) if mean_depth > 0 else 1.0
        if depth_scale <= 0:
            depth_scale = 1.0

        for i in range(c.levels):
            base = i * FEATURES_PER_LEVEL
            if i < len(book.asks):
                ask = book.asks[i]
                row[base] = (ask.price - mid) / mid * 10_000.0
                row[base + 1] = np.log1p(ask.size / depth_scale)
            if i < len(book.bids):
                bid = book.bids[i]
                row[base + 2] = (bid.price - mid) / mid * 10_000.0
                row[base + 3] = np.log1p(bid.size / depth_scale)
        return row

    def on_book(self, book: OrderBookUpdate) -> FeatureSnapshot | None:
        """Absorb a book update. Returns ``None`` if the book is not quotable."""
        mid = book.mid_price
        micro = book.micro_price
        if mid is None or micro is None or not book.bids or not book.asks:
            return None

        quote = BestQuote(
            bid_price=book.bids[0].price,
            bid_size=book.bids[0].size,
            ask_price=book.asks[0].price,
            ask_size=book.asks[0].size,
        )
        ofi = self._ofi.update(quote)
        sigma = self._vol.update(mid, book.ts_event)
        sigma_fast = self._ewma_vol.update(mid, book.ts_event)
        sigma_baseline = self._vol_baseline.update(sigma)

        tensor: np.ndarray | None = None
        if self.config.emit_tensor:
            self._rows.append(self._normalise_row(book, mid))
            if self.tensor_ready:
                tensor = np.stack(self._rows, axis=0)

        self._last_book = book
        return FeatureSnapshot(
            ts_ns=book.ts_event,
            mid_price=mid,
            micro_price=micro,
            spread=book.spread or 0.0,
            sigma=sigma,
            sigma_fast=sigma_fast,
            sigma_baseline=sigma_baseline,
            vol_warm=self._vol.warm and self._vol_baseline.warm,
            vpin=self._vpin.value,
            vpin_warm=self._vpin.warm,
            ofi=ofi,
            ofi_zscore=self._ofi.zscore(),
            ofi_warm=self._ofi.warm,
            tensor=tensor,
        )

    def reset(self) -> None:
        self._ofi.reset()
        self._vpin.reset()
        self._vol.reset()
        self._ewma_vol.reset()
        self._vol_baseline.reset()
        self._rows.clear()
        self._depth_window.clear()
        self._depth_sum = 0.0
        self._last_book = None


def build_label(
    mids: np.ndarray, index: int, horizon: int, alpha: float
) -> int:
    """DeepLOB's smoothed 3-class label. 0 = down, 1 = flat, 2 = up.

    Zhang et al. define, with ``k`` the horizon,

    .. math::
        m_-(t) = \\frac1k \\sum_{i=0}^{k-1} p_{t-i}, \\quad
        m_+(t) = \\frac1k \\sum_{i=1}^{k} p_{t+i}, \\quad
        \\ell_t = \\frac{m_+(t) - m_-(t)}{m_-(t)}

    and threshold ``l`` at ``+/- alpha``. Averaging *both* windows is the part
    that matters: labelling on a single future tick makes the target almost
    pure microstructure noise, and the model learns the bid-ask bounce instead
    of the drift.
    """
    if index < horizon or index + horizon >= len(mids):
        return 1
    m_minus = float(np.mean(mids[index - horizon + 1 : index + 1]))
    m_plus = float(np.mean(mids[index + 1 : index + horizon + 1]))
    if m_minus <= 0:
        return 1
    l = (m_plus - m_minus) / m_minus
    if l > alpha:
        return 2
    if l < -alpha:
        return 0
    return 1


def label_series(mids: np.ndarray, horizon: int, alpha: float) -> np.ndarray:
    """Vectorised :func:`build_label` over a whole series."""
    n = len(mids)
    labels = np.ones(n, dtype=np.int64)
    if n < 2 * horizon + 1:
        return labels

    kernel = np.ones(horizon, dtype=np.float64) / horizon
    rolling = np.convolve(mids.astype(np.float64), kernel, mode="valid")  # len n-horizon+1

    idx = np.arange(horizon - 1, n - horizon)
    m_minus = rolling[idx - horizon + 1]
    m_plus = rolling[idx + 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        l = np.where(m_minus > 0, (m_plus - m_minus) / m_minus, 0.0)

    window = np.ones(len(idx), dtype=np.int64)
    window[l > alpha] = 2
    window[l < -alpha] = 0
    labels[idx] = window
    return labels


__all__ = [
    "FEATURES_PER_LEVEL",
    "FeaturePipeline",
    "FeaturePipelineConfig",
    "FeatureSnapshot",
    "build_label",
    "label_series",
]
