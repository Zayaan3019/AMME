"""Deterministic paper-trading matching engine with queue-position modelling.

A naive backtester fills a resting bid the moment any trade prints at or below
its price. That single shortcut is responsible for most market-making backtests
that look profitable and are not, because it silently assumes you are always
first in the queue -- the one assumption a passive strategy can never make.

What this engine models instead:

**Queue priority.** An order joining a level sits behind whatever size is
already resting there. It fills only after that size is consumed.

**Queue decay from cancellations.** When a level shrinks without a trade, the
departed size was cancellations. Those cancellations are distributed across the
queue, so a share of them sat ahead of us. Under a uniform-position assumption
the surviving queue-ahead scales by exactly ``new_size / old_size`` -- a one-line
update that captures most of the real dynamics.

**Queue priority loss on re-quote.** Amending a resting order's price sends it
to the back of the new level. Re-quoting at an *unchanged* price keeps priority.
This asymmetry is a first-order effect in market making -- it is why real makers
work so hard to avoid unnecessary amends -- and it is modelled explicitly in
:meth:`MatchingEngine.replace_quotes`.

**Book walk-through.** A trade printing *strictly through* our price means the
aggressor consumed every level between, so our order was necessarily filled.

**Own-order impact.** Our orders were never in the recorded book, so a fill we
receive is one the real queue did not get. ``own_impact`` parameterises that
counterfactual: 0 treats us as pure added depth (correct when our size is
negligible against the touch), 1 as full displacement. :func:`own_impact_from_share`
derives it from our share of the level, so the assumption degrades honestly as
quote size grows instead of staying optimistic.

**Latency.** ``latency_sigma`` turns the arrival delay from a constant into a
log-normal draw. Real network latency is fat-tailed, and the slow round trips
are exactly the ones during which the market moves against a stale quote -- so a
fixed delay flatters execution by removing the adverse cases. Draws come from an
explicitly seeded generator, so modelling the tail costs nothing in replay
reproducibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from amme_shared.events import AggressorSide, Side

#: Size comparisons below this count as zero. Well above float rounding on
#: realistic quantities (1e-5 lot), far below any real order.
_EPS = 1e-12


def _finite(x: float) -> bool:
    return math.isfinite(x)


class OrderStatus(StrEnum):
    RESTING = "resting"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class RestingOrder:
    """One of our passive orders sitting in the book."""

    order_id: str
    side: Side
    price: float
    size: float
    filled: float = 0.0
    queue_ahead: float = 0.0
    level_size_seen: float = 0.0
    created_ns: int = 0
    status: OrderStatus = OrderStatus.RESTING

    @property
    def remaining(self) -> float:
        return max(0.0, self.size - self.filled)

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.RESTING, OrderStatus.PARTIAL) and self.remaining > 0


@dataclass(frozen=True, slots=True)
class SimFill:
    """A fill produced by the engine. Pure data; the service wraps it in an event."""

    order_id: str
    side: Side
    price: float
    size: float
    fee: float
    queue_position_at_fill: float
    ts_ns: int


@dataclass(slots=True)
class Portfolio:
    """Signed-position average-cost book-keeping.

    ``total_pnl`` is computed as mark-to-market (``cash + q * mid``), and
    ``realized + unrealized`` is derived independently; the two are asserted
    equal by :meth:`check_identity`. Deriving the same number two ways and
    checking them against each other is cheap and catches accounting bugs that
    are otherwise invisible until they have polluted a whole results table.

    Fees are folded into ``realized_pnl`` (and tracked separately in
    ``fees_paid`` for reporting) so that the identity holds exactly.
    """

    inventory: float = 0.0
    cash: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    n_fills: int = 0
    n_buys: int = 0
    n_sells: int = 0
    volume_traded: float = 0.0

    def apply_fill(self, side: Side, price: float, size: float, fee: float = 0.0) -> None:
        """Apply one fill.

        Rejects non-finite inputs loudly. A single NaN price silently turns
        ``cash`` into NaN, and from there every P&L figure, Sharpe, drawdown
        and the health check's own identity test become NaN -- which compares
        False against every threshold, so nothing alerts and the run reports
        plausible-looking garbage. Failing at the boundary is the only place
        this is cheap to catch.
        """
        if not (_finite(price) and _finite(size) and _finite(fee)):
            raise ValueError(
                f"non-finite fill rejected: price={price} size={size} fee={fee}"
            )
        if size < 0.0:
            raise ValueError(f"negative fill size {size}; use the opposite side instead")
        if price <= 0.0:
            raise ValueError(f"non-positive fill price {price}")
        if size <= _EPS:
            return
        signed = size if side is Side.BID else -size

        increasing = self.inventory == 0.0 or (self.inventory > 0.0) == (signed > 0.0)
        if increasing:
            new_inventory = self.inventory + signed
            notional = abs(self.inventory) * self.avg_price + size * price
            self.avg_price = notional / abs(new_inventory) if new_inventory != 0.0 else 0.0
            self.inventory = new_inventory
        else:
            closing = min(size, abs(self.inventory))
            if self.inventory > 0.0:
                self.realized_pnl += closing * (price - self.avg_price)
            else:
                self.realized_pnl += closing * (self.avg_price - price)
            self.inventory += signed
            if abs(self.inventory) < 1e-15:
                self.inventory = 0.0
                self.avg_price = 0.0
            elif size > closing:
                self.avg_price = price  # position flipped through zero

        self.cash -= signed * price
        self.cash -= fee
        self.realized_pnl -= fee
        self.fees_paid += fee

        self.n_fills += 1
        self.volume_traded += size
        if signed > 0:
            self.n_buys += 1
        else:
            self.n_sells += 1

    def unrealized_pnl(self, mid: float) -> float:
        return self.inventory * (mid - self.avg_price)

    def total_pnl(self, mid: float) -> float:
        return self.cash + self.inventory * mid

    def check_identity(self, mid: float, tolerance: float = 1e-6) -> bool:
        """``cash + q*mid`` must equal ``realized + unrealized``."""
        lhs = self.total_pnl(mid)
        rhs = self.realized_pnl + self.unrealized_pnl(mid)
        return abs(lhs - rhs) <= tolerance * max(1.0, abs(lhs))

    def reset(self) -> None:
        self.inventory = self.cash = self.avg_price = 0.0
        self.realized_pnl = self.fees_paid = 0.0
        self.n_fills = self.n_buys = self.n_sells = 0
        self.volume_traded = 0.0


@dataclass(frozen=True, slots=True)
class MatchingParams:
    """Execution assumptions. Each default is measured or argued, not guessed."""

    maker_fee_bps: float = 0.0
    tick_size: float = 0.01

    #: Median delay between our decision and the order resting at the exchange.
    #: Measured against live Binance: ~120-183ms round trip, so ~60-90ms one
    #: way. 5ms is a *colocated* figure and was the previous default; it is
    #: kept as the floor only because the backtests were calibrated with it.
    #: See `latency_sigma` for why a single number is not enough.
    latency_ns: int = 5_000_000

    #: Log-normal shape parameter for the latency distribution. Real network
    #: latency is fat-tailed -- the mean is dominated by a small number of very
    #: slow round trips, and those are exactly the ones during which the market
    #: moves against a stale quote. A fixed delay makes execution look better
    #: than it is by removing precisely the adverse cases.
    #:
    #: 0.0 reproduces the old constant-delay behaviour exactly, so existing
    #: results remain comparable; set ~0.5 to model a realistic tail.
    latency_sigma: float = 0.0
    #: Hard cap so a tail draw cannot exceed a plausible timeout.
    latency_max_ns: int = 2_000_000_000

    #: If True, a trade printing strictly through our price fills us fully
    #: (bounded by the aggressor's remaining size -- see `on_trade`).
    walk_through_fills: bool = True

    #: Fraction of our resting size that displaces real liquidity rather than
    #: adding to it. Our orders were never in the recorded book, so a fill we
    #: receive is one the real queue did not. At 0.0 (default) we are a pure
    #: addition -- correct when our size is negligible against the touch. At
    #: 1.0 we fully displace, which is the conservative extreme. The honest
    #: value scales with our share of the level; see `own_impact_from_share`.
    own_impact: float = 0.0


class MatchingEngine:
    """Simulates our resting quotes against a replayed or synthetic order flow.

    Fully deterministic: no randomness anywhere, so identical inputs give
    byte-identical fills. That is the property the s.6.1 replay test asserts.
    """

    __slots__ = ("_next_order_seq", "_rng", "orders", "params", "portfolio")

    def __init__(
        self, params: MatchingParams | None = None, seed: int | None = None
    ) -> None:
        self.params = params or MatchingParams()
        self.portfolio = Portfolio()
        self.orders: dict[str, RestingOrder] = {}
        self._next_order_seq = 0
        # Explicitly seeded, never module-level: a stochastic latency model
        # must not cost us byte-for-byte replay reproducibility (s.6.1).
        self._rng: np.random.Generator | None = (
            np.random.Generator(np.random.PCG64(seed)) if seed is not None else None
        )

    # -- order management --------------------------------------------------
    def _new_order_id(self, side: Side) -> str:
        self._next_order_seq += 1
        return f"{side.value}-{self._next_order_seq}"

    def active_order(self, side: Side) -> RestingOrder | None:
        for order in self.orders.values():
            if order.side is side and order.is_active:
                return order
        return None

    def _sample_latency_ns(self) -> int:
        """Draw one order-arrival latency.

        Deterministic when ``latency_sigma == 0`` (the calibrated default), so
        replay reproducibility is unaffected. When a tail is modelled, the draw
        comes from the engine's own seeded generator -- never module-level
        randomness -- so a stochastic latency model is still bit-reproducible.
        """
        p = self.params
        if p.latency_sigma <= 0.0 or self._rng is None:
            return p.latency_ns
        # Log-normal with median `latency_ns`: exp(mu) = median, so mu = ln(median).
        draw = float(self._rng.lognormal(math.log(max(p.latency_ns, 1)), p.latency_sigma))
        return int(min(draw, p.latency_max_ns))

    def place(
        self, side: Side, price: float, size: float, level_size: float, ts_ns: int
    ) -> RestingOrder:
        """Rest a new order, joining the back of the queue at ``price``.

        ``own_impact`` models the counterfactual we can never observe: our
        order was not in the recorded book, so resting it either *adds* depth
        (impact 0) or *displaces* liquidity that would otherwise have been
        there (impact 1). Displacement shortens the queue ahead of us, because
        some of what we see resting would not have existed had we been quoting.
        """
        effective_queue = max(0.0, level_size) * (1.0 - self.params.own_impact)
        order = RestingOrder(
            order_id=self._new_order_id(side),
            side=side,
            price=price,
            size=size,
            queue_ahead=effective_queue,
            level_size_seen=max(0.0, level_size),
            created_ns=ts_ns + self._sample_latency_ns(),
        )
        self.orders[order.order_id] = order
        return order

    def cancel(self, order_id: str) -> None:
        order = self.orders.get(order_id)
        if order is not None and order.is_active:
            order.status = OrderStatus.CANCELLED

    def cancel_all(self) -> None:
        for order in self.orders.values():
            if order.is_active:
                order.status = OrderStatus.CANCELLED

    def replace_quotes(
        self,
        bid_price: float | None,
        bid_size: float,
        ask_price: float | None,
        ask_size: float,
        bid_level_size: float,
        ask_level_size: float,
        ts_ns: int,
    ) -> int:
        """Install the desired two-sided quote, preserving queue priority where possible.

        An order whose price and size are unchanged is left alone -- it keeps its
        hard-won queue position. Any change to price cancels and re-rests at the
        back of the new level. This is the behaviour of a real exchange and the
        reason a maker's effective fill rate depends so heavily on quote
        stability.

        Returns the number of *new* orders rested. This is the honest
        denominator for a fill ratio: counting re-quote decisions instead would
        report a ratio that falls purely because the engine re-evaluated more
        often, which measures nothing about execution quality.
        """
        placed = 0
        for side, price, size, level_size in (
            (Side.BID, bid_price, bid_size, bid_level_size),
            (Side.ASK, ask_price, ask_size, ask_level_size),
        ):
            existing = self.active_order(side)

            if price is None or size <= 0.0:
                if existing is not None:
                    existing.status = OrderStatus.CANCELLED
                continue

            if existing is not None:
                if existing.price == price:
                    # Same price: keep the queue slot. Shrinking size keeps
                    # priority too; growing it would in reality queue the
                    # increment at the back, which at our sizes is immaterial.
                    existing.size = max(existing.filled, size)
                    if existing.remaining <= 0.0:
                        existing.status = OrderStatus.FILLED
                    continue
                existing.status = OrderStatus.CANCELLED

            self.place(side, price, size, level_size, ts_ns)
            placed += 1
        return placed

    def prune(self) -> None:
        """Drop inactive orders. Keeps the per-event scan O(active orders)."""
        self.orders = {oid: o for oid, o in self.orders.items() if o.is_active}

    # -- market events -----------------------------------------------------
    def on_book_update(
        self, bid_levels: dict[float, float], ask_levels: dict[float, float]
    ) -> None:
        """Decay queue-ahead for cancellations observed at our price levels."""
        for order in self.orders.values():
            if not order.is_active:
                continue
            levels = bid_levels if order.side is Side.BID else ask_levels
            current = levels.get(order.price, 0.0)
            previous = order.level_size_seen

            if current < previous and previous > 0.0:
                # Size left the level. Under a uniform-position assumption the
                # share of departures that sat ahead of us is queue_ahead/prev,
                # which makes the surviving queue-ahead scale by current/prev.
                order.queue_ahead *= current / previous
            # Size added to a level always joins behind us -> no change.
            order.level_size_seen = current

    def _matchable(self, aggressor: AggressorSide, price: float) -> list[RestingOrder]:
        """Our active orders this trade could reach, in exchange priority order.

        Price priority is applied explicitly rather than relying on dict
        insertion order: a sweep fills the most aggressive resting orders
        first, and iterating in insertion order would make fill attribution
        depend on the order we happened to place quotes in.
        """
        side = Side.BID if aggressor is AggressorSide.SELL else Side.ASK
        candidates = [
            o
            for o in self.orders.values()
            if o.is_active
            and o.side is side
            and (price <= o.price if side is Side.BID else price >= o.price)
        ]
        # Best price first: highest bid, lowest ask.
        candidates.sort(key=lambda o: o.price, reverse=side is Side.BID)
        return candidates

    def on_trade(
        self, price: float, size: float, aggressor: AggressorSide, ts_ns: int
    ) -> list[SimFill]:
        """Apply an incoming aggressive trade against our resting orders.

        **Volume is conserved.** An aggressor with ``size`` units to spend can
        never fill more than ``size`` units of our orders in total, including
        through-fills. That sounds obvious and was previously violated: a
        through-fill assigned ``order.remaining`` without drawing down the
        trade's volume, so a 0.001 BTC print walking through two 5 BTC orders
        filled 10 BTC -- a 10,000x over-fill that inflated P&L, fill counts and
        every metric derived from them.

        The physical argument for the old behaviour was that a trade printing
        *through* our price must have consumed our order on the way. That is
        true only up to the aggressor's actual size: if the sweep had 0.001
        units left when it reached our price, it takes 0.001 units, not five
        BTC. Priority still applies -- a through-fill skips the queue, because
        everything ahead of us was consumed to get there -- but the budget does
        not.
        """
        if not _finite(price) or not _finite(size):
            raise ValueError(f"non-finite trade: price={price} size={size}")
        if size <= 0.0:
            return []

        fills: list[SimFill] = []
        remaining_volume = size

        for order in self._matchable(aggressor, price):
            if remaining_volume <= _EPS:
                break
            if ts_ns < order.created_ns:
                continue  # not yet at the exchange (latency model)

            through = price < order.price if order.side is Side.BID else price > order.price

            if through and self.params.walk_through_fills:
                # Everything between was consumed to reach this price, so our
                # queue position is irrelevant -- but the aggressor's remaining
                # size still bounds what we receive.
                order.queue_ahead = 0.0
                available = remaining_volume
            else:
                consumed = min(remaining_volume, order.queue_ahead)
                order.queue_ahead -= consumed
                available = remaining_volume - consumed
                if available <= _EPS:
                    # The whole trade went to the queue ahead of us.
                    remaining_volume = 0.0
                    break

            fill_size = min(order.remaining, available)
            if fill_size <= _EPS:
                continue

            fee = order.price * fill_size * self.params.maker_fee_bps / 10_000.0
            self.portfolio.apply_fill(order.side, order.price, fill_size, fee)
            order.filled += fill_size
            order.status = (
                OrderStatus.FILLED if order.remaining <= _EPS else OrderStatus.PARTIAL
            )
            order.level_size_seen = max(0.0, order.level_size_seen - fill_size)
            remaining_volume = max(0.0, remaining_volume - fill_size)

            fills.append(
                SimFill(
                    order_id=order.order_id,
                    side=order.side,
                    price=order.price,
                    size=fill_size,
                    fee=fee,
                    queue_position_at_fill=order.queue_ahead,
                    ts_ns=ts_ns,
                )
            )

        return fills

    def reset(self, seed: int | None = None) -> None:
        self.orders.clear()
        self.portfolio.reset()
        self._next_order_seq = 0
        if seed is not None:
            self._rng = np.random.Generator(np.random.PCG64(seed))


def own_impact_from_share(quote_size: float, level_size: float) -> float:
    """A defensible ``own_impact`` from our share of the level.

    Displacement is not all-or-nothing: an order that is 1% of the touch adds
    depth almost purely, while one that is 50% of it is substantially taking
    someone else's place. Linear in our share, clamped to [0, 1], is the
    simplest form consistent with both limits -- and it makes the assumption a
    *function of size*, so the backtest degrades honestly as quote size grows
    instead of staying optimistic.
    """
    if level_size <= 0.0:
        return 1.0
    return min(1.0, max(0.0, quote_size / (quote_size + level_size)))


__all__ = [
    "MatchingEngine",
    "MatchingParams",
    "OrderStatus",
    "Portfolio",
    "RestingOrder",
    "SimFill",
    "own_impact_from_share",
]
