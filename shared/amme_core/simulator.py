"""Synthetic limit-order-book simulator.

Real market data is not a controlled experiment: you cannot ask BTC/USDT to
please become toxic for thirty seconds so you can check that the VPIN gate
fires. This module provides that control, and it is what the regression harness
(s.6) runs against -- real data is used to validate that the simulator is not
lying, not to gate the build.

Model (a standard reduced-form LOB, in the lineage of Cont-Stoikov-Talreja 2010
with an informed-flow overlay):

* **Fundamental value** :math:`dS_t = \\mu dt + \\sigma dW_t` -- arithmetic
  Brownian motion, matching the process Avellaneda-Stoikov assumes, so the
  quoter is being tested against its own generative assumptions in the calm
  regime and against violations of them in the others.
* **Limit orders** arrive at each level with intensity decaying in distance from
  the mid, :math:`\\lambda_{limit}(i) = A e^{-k i}`. This is the same
  exponential form the A-S ``kappa`` parameter describes, which lets the
  simulator produce data with a *known* ``kappa`` -- so the calibration code can
  be tested against ground truth.
* **Cancellations** remove resting size at a rate proportional to that size.
* **Market orders** arrive as a Poisson process. Their direction is biased by
  the gap between the fundamental and the current mid, which is the mechanism
  that makes order flow *informative*: aggressors buy when the book is cheap
  relative to fair value. This is what gives OFI and DeepLOB something real to
  predict, rather than fitting noise.

Regimes let a single run sweep through market conditions:

``CALM``      baseline.
``VOLATILE``  high sigma, wider arrivals -- tests spread response to sigma.
``TOXIC``     informed flow amplified and persistent -- the VPIN gate's target.
``TRENDING``  non-zero drift -- inventory management under a directional market.

Determinism: every random draw comes from an explicitly-seeded
``numpy.random.Generator``. No module-level ``np.random`` calls exist here, which
is what makes the s.6.1 byte-for-byte replay assertion possible.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from amme_shared.clock import NS_PER_SECOND
from amme_shared.events import (
    AggressorSide,
    DataSource,
    OrderBookUpdate,
    PriceLevel,
    TradeTick,
)


class Regime(StrEnum):
    CALM = "calm"
    VOLATILE = "volatile"
    TOXIC = "toxic"
    TRENDING = "trending"


@dataclass(frozen=True, slots=True)
class RegimeParams:
    """Multipliers applied on top of the base configuration."""

    sigma_multiple: float = 1.0
    drift: float = 0.0
    informed_intensity: float = 0.0
    market_order_multiple: float = 1.0
    cancel_multiple: float = 1.0


REGIME_TABLE: dict[Regime, RegimeParams] = {
    Regime.CALM: RegimeParams(),
    Regime.VOLATILE: RegimeParams(sigma_multiple=3.0, market_order_multiple=1.5,
                                  cancel_multiple=1.6),
    Regime.TOXIC: RegimeParams(sigma_multiple=1.8, informed_intensity=0.85,
                               market_order_multiple=2.5, cancel_multiple=2.0),
    Regime.TRENDING: RegimeParams(sigma_multiple=1.2, drift=0.15, informed_intensity=0.35,
                                  market_order_multiple=1.2),
}


@dataclass(frozen=True, slots=True)
class SimulatorConfig:
    symbol: str = "BTCUSDT"
    initial_price: float = 50_000.0
    #: Real BTCUSDT tick. Coarser grids make the A-S prescription unresolvable:
    #: at tick=0.1 the optimal half-spread rounds to <1 tick and inventory
    #: control has no price resolution to express itself through.
    tick_size: float = 0.01
    lot_size: float = 0.00001

    #: Volatility of the fundamental, quote-currency units per sqrt(second).
    sigma: float = 0.35
    #: Drift of the fundamental, quote-currency units per second.
    mu: float = 0.0

    depth_levels: int = 10
    #: Levels tracked internally beyond `depth_levels`, so the book does not
    #: run out of liquidity the instant a large aggressor arrives.
    book_levels: int = 40

    limit_order_rate: float = 40.0  # A, orders/sec at the touch
    limit_order_decay: float = 0.35  # k in A*exp(-k*i)
    mean_limit_size: float = 0.35
    cancel_rate: float = 0.55  # per unit of resting size per second
    market_order_rate: float = 2.2  # aggressive orders/sec
    mean_market_size: float = 0.22

    #: How strongly aggressors lean toward the fundamental. 0 = uninformed flow.
    informed_intensity: float = 0.25
    #: Sensitivity of that lean to (fundamental - mid), in ticks.
    informed_scale_ticks: float = 4.0

    #: Simulated makers widen their touch by this many ticks per tick of
    #: per-step volatility. Gives the book a regime-dependent spread.
    spread_vol_coeff: float = 0.8
    #: Extra cancellation hazard per tick a level sits on the wrong side of
    #: fair value. This is what lets the book migrate instead of going stale.
    stale_cancel_penalty: float = 1.5

    step_ms: float = 100.0  # matches Binance's @depth@100ms cadence
    seed: int = 42

    def steps_for(self, seconds: float) -> int:
        return max(1, int(seconds * 1000.0 / self.step_ms))


@dataclass(slots=True)
class SimulatorState:
    fundamental: float
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    t_ns: int = 0
    update_id: int = 0
    regime: Regime = Regime.CALM


class MarketSimulator:
    """Generates a self-consistent stream of book updates and trades.

    Yields ``(OrderBookUpdate, list[TradeTick])`` per step so a consumer sees
    the book state *and* the flow that produced it, in causal order.
    """

    __slots__ = ("_regime_schedule", "config", "rng", "state")

    def __init__(
        self,
        config: SimulatorConfig | None = None,
        regime: Regime = Regime.CALM,
        regime_schedule: list[tuple[float, Regime]] | None = None,
    ) -> None:
        self.config = config or SimulatorConfig()
        self.rng = np.random.Generator(np.random.PCG64(self.config.seed))
        self.state = SimulatorState(fundamental=self.config.initial_price, regime=regime)
        #: ``[(start_seconds, regime), ...]`` -- lets one run sweep regimes,
        #: which is how the toxicity ablation gets both clean and toxic periods
        #: inside a single deterministic seed.
        self._regime_schedule = sorted(regime_schedule or [], key=lambda x: x[0])
        self._initialise_book()

    # -- setup -------------------------------------------------------------
    def _event_id(self, kind: str, index: int = 0) -> str:
        """Deterministic, collision-free id derived from seed and sequence.

        The simulator previously used ``new_id()`` (uuid4), which made every
        emitted event carry a random correlation id. P&L still replayed
        identically because the digest ignores ids, but the s.6.1 claim is
        *byte-for-byte reproducibility of the events* -- and that was false.
        Anything that keyed off correlation_id (log correlation, dedup,
        golden-file comparison) was non-reproducible.
        """
        return f"sim-{self.config.seed}-{kind}-{self.state.update_id}-{index}"

    def _round(self, price: float) -> float:
        return round(round(price / self.config.tick_size) * self.config.tick_size, 8)

    def _initialise_book(self) -> None:
        c = self.config
        mid = c.initial_price
        half_spread = c.tick_size
        for i in range(c.book_levels):
            bid_price = self._round(mid - half_spread - i * c.tick_size)
            ask_price = self._round(mid + half_spread + i * c.tick_size)
            # Depth profile grows away from the touch then flattens -- the
            # familiar hump-shaped LOB. Seeded deterministically.
            base = c.mean_limit_size * (1.0 + 2.5 * (1.0 - np.exp(-0.25 * i)))
            self.state.bids[bid_price] = float(base * self.rng.uniform(0.6, 1.4))
            self.state.asks[ask_price] = float(base * self.rng.uniform(0.6, 1.4))

    # -- regime ------------------------------------------------------------
    def _regime_at(self, t_seconds: float) -> Regime:
        current = self.state.regime
        for start, regime in self._regime_schedule:
            if t_seconds >= start:
                current = regime
            else:
                break
        return current

    def _effective(self, regime: Regime) -> tuple[float, float, float, float, float]:
        p = REGIME_TABLE[regime]
        c = self.config
        return (
            c.sigma * p.sigma_multiple,
            c.mu + p.drift,
            max(c.informed_intensity, p.informed_intensity),
            c.market_order_rate * p.market_order_multiple,
            c.cancel_rate * p.cancel_multiple,
        )

    # -- book helpers ------------------------------------------------------
    @property
    def best_bid(self) -> float | None:
        return max(self.state.bids) if self.state.bids else None

    @property
    def best_ask(self) -> float | None:
        return min(self.state.asks) if self.state.asks else None

    @property
    def mid(self) -> float:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return self.state.fundamental
        return 0.5 * (bid + ask)

    def _prune(self) -> None:
        c = self.config
        self.state.bids = {p: s for p, s in self.state.bids.items() if s > 1e-9}
        self.state.asks = {p: s for p, s in self.state.asks.items() if s > 1e-9}
        # Keep memory bounded: only the levels near the touch matter.
        if len(self.state.bids) > c.book_levels * 2:
            keep = sorted(self.state.bids, reverse=True)[: c.book_levels]
            self.state.bids = {p: self.state.bids[p] for p in keep}
        if len(self.state.asks) > c.book_levels * 2:
            keep = sorted(self.state.asks)[: c.book_levels]
            self.state.asks = {p: self.state.asks[p] for p in keep}

    def _touch_offset_ticks(self, sigma: float) -> float:
        """How far from fair value the simulated makers post their best quote.

        Scaled by volatility, because that is what real makers do -- and it is
        what gives the simulated spread a regime dependence for our own quoter
        to have to compete against. Without this the book's spread would be a
        constant and the VOLATILE regime would only change the mid's variance.
        """
        c = self.config
        vol_ticks = sigma * np.sqrt(c.step_ms / 1000.0) / c.tick_size
        return float(max(1.0, 1.0 + c.spread_vol_coeff * vol_ticks))

    def _replenish(self, dt: float, side: str, sigma: float) -> None:
        """Poisson limit-order arrivals, anchored to the *fundamental*.

        Anchoring to fair value rather than to the current best quote is the
        mechanism that lets the book track the fundamental: when value moves up,
        new bids arrive higher and stale ones are cancelled, so the touch
        migrates. Anchoring to the current best instead produces a book that
        refills exactly where it was and a mid that never moves -- which is
        both wrong and, worse, silently wrong.
        """
        c = self.config
        book = self.state.bids if side == "bid" else self.state.asks
        sign = -1.0 if side == "bid" else 1.0
        base_offset = self._touch_offset_ticks(sigma)

        for i in range(c.book_levels):
            intensity = c.limit_order_rate * np.exp(-c.limit_order_decay * i) * dt
            n_orders = int(self.rng.poisson(intensity))
            if n_orders == 0:
                continue
            price = self._round(
                self.state.fundamental + sign * (base_offset + i) * c.tick_size
            )
            if price <= 0:
                continue
            size = float(self.rng.exponential(c.mean_limit_size, size=n_orders).sum())
            book[price] = book.get(price, 0.0) + size

    def _cancel(self, dt: float, rate: float) -> None:
        """Size-proportional cancellations with a staleness penalty.

        A resting bid *above* fair value is a losing quote, and real makers pull
        those fastest. Making the hazard rate grow with how far a level is on
        the wrong side of the fundamental is what stops stale liquidity walls
        from accumulating behind a moving price.
        """
        c = self.config
        fair = self.state.fundamental
        for is_bid, book in ((True, self.state.bids), (False, self.state.asks)):
            for price, size in list(book.items()):
                if size <= 0:
                    continue
                wrong_side_ticks = (
                    (price - fair) / c.tick_size if is_bid else (fair - price) / c.tick_size
                )
                hazard = rate * (1.0 + c.stale_cancel_penalty * max(0.0, wrong_side_ticks))
                survival = float(np.exp(-hazard * dt))
                # Multiplicative noise keeps levels from decaying in lockstep,
                # which would make OFI degenerately smooth.
                noise = float(self.rng.uniform(0.85, 1.15))
                book[price] = size * min(1.0, survival * noise)

    def _consume(self, side: str, volume: float) -> tuple[float, float]:
        """Walk the book. Returns ``(filled_volume, last_traded_price)``."""
        book = self.state.asks if side == "buy" else self.state.bids
        prices = sorted(book) if side == "buy" else sorted(book, reverse=True)
        remaining = volume
        last_price = self.mid
        for price in prices:
            if remaining <= 1e-12:
                break
            available = book[price]
            taken = min(available, remaining)
            book[price] = available - taken
            remaining -= taken
            last_price = price
        return volume - remaining, last_price

    # -- main step ---------------------------------------------------------
    def step(self) -> tuple[OrderBookUpdate, list[TradeTick]]:
        c = self.config
        dt = c.step_ms / 1000.0
        t_seconds = self.state.t_ns / NS_PER_SECOND
        regime = self._regime_at(t_seconds)
        self.state.regime = regime
        sigma, mu, informed, mo_rate, cancel_rate = self._effective(regime)

        # 1. Fundamental evolves.
        shock = float(self.rng.normal(0.0, 1.0))
        self.state.fundamental += mu * dt + sigma * np.sqrt(dt) * shock

        # 2. Passive flow: cancellations then arrivals.
        self._cancel(dt, cancel_rate)
        self._replenish(dt, "bid", sigma)
        self._replenish(dt, "ask", sigma)
        self._prune()

        # 3. Aggressive flow.
        trades: list[TradeTick] = []
        n_market = int(self.rng.poisson(mo_rate * dt))
        for _ in range(n_market):
            mid = self.mid
            gap_ticks = (self.state.fundamental - mid) / c.tick_size
            # Informed lean: logistic in the mispricing, blended with a coin
            # flip by `informed`. informed=0 -> pure noise trader.
            lean = 1.0 / (1.0 + np.exp(-gap_ticks / max(c.informed_scale_ticks, 1e-9)))
            p_buy = float(0.5 * (1.0 - informed) + informed * lean)
            is_buy = bool(self.rng.random() < p_buy)
            size = float(self.rng.exponential(c.mean_market_size))
            if size <= 0:
                continue
            filled, price = self._consume("buy" if is_buy else "sell", size)
            if filled <= 1e-12:
                continue
            trades.append(
                TradeTick(
                    symbol=c.symbol,
                    ts_event=self.state.t_ns,
                    ts_ingest=self.state.t_ns,
                    source=DataSource.SIMULATOR,
                    correlation_id=self._event_id("trade", len(trades)),
                    price=price,
                    size=filled,
                    aggressor=AggressorSide.BUY if is_buy else AggressorSide.SELL,
                    trade_id=self.state.update_id * 1000 + len(trades),
                )
            )
        self._prune()

        # 4. Guard against a fully one-sided or crossed book after a big sweep.
        self._ensure_two_sided()

        # 5. Publish.
        self.state.update_id += 1
        self.state.t_ns += int(c.step_ms * 1_000_000)
        book_event = self._snapshot_event()
        return book_event, trades

    def _ensure_two_sided(self) -> None:
        """Re-seed a side that got swept out, so the book is always quotable."""
        c = self.config
        if not self.state.bids:
            anchor = (self.best_ask or self.state.fundamental) - c.tick_size
            for i in range(5):
                self.state.bids[self._round(anchor - i * c.tick_size)] = float(
                    self.rng.exponential(c.mean_limit_size)
                ) + 1e-6
        if not self.state.asks:
            anchor = (self.best_bid or self.state.fundamental) + c.tick_size
            for i in range(5):
                self.state.asks[self._round(anchor + i * c.tick_size)] = float(
                    self.rng.exponential(c.mean_limit_size)
                ) + 1e-6

        # A crossed book is unphysical; clear the overlapping levels.
        guard = 0
        while self.best_bid is not None and self.best_ask is not None and (
            self.best_bid >= self.best_ask
        ):
            guard += 1
            if guard > 200:  # pragma: no cover -- structural safety net
                break
            if self.rng.random() < 0.5:
                del self.state.bids[max(self.state.bids)]
            else:
                del self.state.asks[min(self.state.asks)]
            if not self.state.bids or not self.state.asks:
                self._ensure_two_sided()
                return

    def _snapshot_event(self) -> OrderBookUpdate:
        c = self.config
        bid_prices = sorted(self.state.bids, reverse=True)[: c.depth_levels]
        ask_prices = sorted(self.state.asks)[: c.depth_levels]
        return OrderBookUpdate(
            symbol=c.symbol,
            ts_event=self.state.t_ns,
            ts_ingest=self.state.t_ns,
            source=DataSource.SIMULATOR,
            correlation_id=self._event_id("book"),
            last_update_id=self.state.update_id,
            is_snapshot=False,
            bids=[PriceLevel(price=p, size=self.state.bids[p]) for p in bid_prices],
            asks=[PriceLevel(price=p, size=self.state.asks[p]) for p in ask_prices],
        )

    # -- driving -----------------------------------------------------------
    def run(self, seconds: float) -> Iterator[tuple[OrderBookUpdate, list[TradeTick]]]:
        for _ in range(self.config.steps_for(seconds)):
            yield self.step()

    def run_steps(self, n: int) -> Iterator[tuple[OrderBookUpdate, list[TradeTick]]]:
        for _ in range(n):
            yield self.step()

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.Generator(np.random.PCG64(seed or self.config.seed))
        self.state = SimulatorState(
            fundamental=self.config.initial_price, regime=self.state.regime
        )
        self._initialise_book()


def default_regime_schedule(total_seconds: float) -> list[tuple[float, Regime]]:
    """Calm -> toxic -> calm -> volatile -> calm -> trending.

    Deliberately returns to calm between stressed regimes: the toxicity-adjusted
    P&L metric in s.6.2 needs both stressed and unstressed periods inside one
    seed to be comparable.

    Suitable for *backtesting*, where the whole run is scored as one sample.
    Use :func:`cycling_regime_schedule` for ML training data -- see there.
    """
    q = total_seconds / 8.0
    return [
        (0.0, Regime.CALM),
        (2 * q, Regime.TOXIC),
        (3 * q, Regime.CALM),
        (5 * q, Regime.VOLATILE),
        (6 * q, Regime.CALM),
        (7 * q, Regime.TRENDING),
    ]


def cycling_regime_schedule(
    total_seconds: float, cycles: int = 8
) -> list[tuple[float, Regime]]:
    """Repeat the full regime cycle ``cycles`` times across the run.

    This exists because of a concrete failure. Training data generated with
    :func:`default_regime_schedule` puts TRENDING in the final eighth of the
    run -- which, under the chronological train/val/test split that ML
    correctness *requires*, lands entirely inside the test set. The result was a
    test split that was **80.4% "up"** against a training split that was
    balanced 31/38/31.

    That combination is worthless for evaluation: a model that ignores its
    input and always predicts "up" scores 80.4%, and no honest accuracy figure
    can be quoted. The split itself is not the problem -- a random split would
    be far worse, since overlapping windows leak. The problem is that one
    contiguous slice of a single-pass schedule is not *representative*.

    Cycling the regimes means every chronological slice contains calm, toxic,
    volatile and trending periods, so the splits are comparable while remaining
    strictly time-ordered and leak-free.
    """
    if cycles < 1:
        raise ValueError("cycles must be >= 1")
    per_cycle = total_seconds / cycles
    schedule: list[tuple[float, Regime]] = []
    order = (Regime.CALM, Regime.TOXIC, Regime.CALM, Regime.VOLATILE,
             Regime.CALM, Regime.TRENDING)
    step = per_cycle / len(order)
    for c in range(cycles):
        base = c * per_cycle
        schedule.extend((base + i * step, regime) for i, regime in enumerate(order))
    return schedule


__all__ = [
    "REGIME_TABLE",
    "MarketSimulator",
    "Regime",
    "RegimeParams",
    "SimulatorConfig",
    "SimulatorState",
    "cycling_regime_schedule",
    "default_regime_schedule",
]
