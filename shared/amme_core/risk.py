"""Inventory limits, quote skewing, unwind logic and the circuit breaker.

This is deliberately a *separate* layer from the quoter, mirroring how a real
desk is wired: the strategy proposes, risk disposes. The architectural argument
for making it a separate process is in ``docs/ARCHITECTURE.md``; the argument
for making it separate *code* is simpler -- risk must stay correct even when the
strategy is wrong, so it may not share state with it.

Three escalating layers, each strictly stronger than the last:

1. **Soft skew** beyond ``q_soft_frac * q_max``. Quotes lean toward flattening.
   A-S already tilts the reservation price with inventory; this adds a term that
   grows *quadratically* in utilisation, so it is negligible in normal operation
   and dominant near the limit.
2. **One-sided unwind** beyond ``unwind_frac * q_max``. The side that would add
   risk is pulled entirely. We stop making a market and start reducing.
3. **Circuit breaker.** All quotes pulled. Trips on a volatility spike, a VPIN
   spike, or a drawdown from the P&L high-water mark; clears only after a
   cooldown *and* only once the triggering condition has actually normalised.

The breaker is intentionally hard to un-trip: a flapping kill switch is worse
than one that stays latched a few seconds too long.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum


class RiskVerdict(StrEnum):
    APPROVE = "approve"
    AMEND = "amend"
    REJECT = "reject"
    HALT = "halt"


class BreakerTrigger(StrEnum):
    NONE = "none"
    VOLATILITY = "volatility"
    TOXICITY = "toxicity"
    DRAWDOWN = "drawdown"
    NOTIONAL = "notional"
    MANUAL = "manual"
    #: A NaN/inf reached the risk layer. Distinct from the market triggers
    #: because it indicates a defect upstream, not a market condition.
    INVALID_INPUT = "invalid_input"


@dataclass(frozen=True, slots=True)
class RiskParams:
    q_max: float = 0.05
    q_soft_frac: float = 0.6
    unwind_frac: float = 0.95
    max_position_value: float = 10_000.0
    breaker_vol_multiple: float = 4.0
    breaker_vpin: float = 0.80
    breaker_drawdown: float = 250.0
    breaker_cooldown_seconds: float = 30.0
    skew_beta: float = 1.0
    enabled: bool = True

    @property
    def q_soft(self) -> float:
        return self.q_soft_frac * self.q_max

    @property
    def q_unwind(self) -> float:
        return self.unwind_frac * self.q_max


@dataclass(frozen=True, slots=True)
class RiskInputs:
    """Everything risk needs to judge one quote."""

    inventory: float
    mid_price: float
    total_pnl: float = 0.0
    realized_vol: float = 0.0
    baseline_vol: float = 0.0
    vpin: float = 0.0
    ts_ns: int = 0


@dataclass(frozen=True, slots=True)
class ProposedQuote:
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    half_spread: float = 0.0


@dataclass(frozen=True, slots=True)
class RiskOutcome:
    verdict: RiskVerdict
    reason: str = ""
    bid_price: float | None = None
    ask_price: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    skew: float = 0.0
    utilisation: float = 0.0
    breaker_tripped: bool = False
    trigger: BreakerTrigger = BreakerTrigger.NONE

    @property
    def actionable(self) -> bool:
        return self.verdict in (RiskVerdict.APPROVE, RiskVerdict.AMEND)


@dataclass(slots=True)
class CircuitBreaker:
    """Latching kill switch with a cooldown."""

    params: RiskParams
    tripped: bool = False
    trigger: BreakerTrigger = BreakerTrigger.NONE
    tripped_at_ns: int = 0
    high_water_pnl: float = 0.0
    trip_count: int = 0
    _initialised: bool = field(default=False, repr=False)

    def _evaluate_conditions(self, inputs: RiskInputs) -> tuple[BreakerTrigger, str]:
        p = self.params

        if (
            p.breaker_vol_multiple > 0.0
            and inputs.baseline_vol > 0.0
            and inputs.realized_vol > p.breaker_vol_multiple * inputs.baseline_vol
        ):
            return (
                BreakerTrigger.VOLATILITY,
                f"realized_vol {inputs.realized_vol:.6g} > {p.breaker_vol_multiple}x "
                f"baseline {inputs.baseline_vol:.6g}",
            )

        if inputs.vpin >= p.breaker_vpin:
            return BreakerTrigger.TOXICITY, f"vpin {inputs.vpin:.3f} >= {p.breaker_vpin}"

        drawdown = self.high_water_pnl - inputs.total_pnl
        if drawdown >= p.breaker_drawdown:
            return (
                BreakerTrigger.DRAWDOWN,
                f"drawdown {drawdown:.2f} >= {p.breaker_drawdown} "
                f"(hwm {self.high_water_pnl:.2f})",
            )

        notional = abs(inputs.inventory) * inputs.mid_price
        if notional > p.max_position_value:
            return (
                BreakerTrigger.NOTIONAL,
                f"position notional {notional:.2f} > {p.max_position_value}",
            )

        return BreakerTrigger.NONE, ""

    def update(self, inputs: RiskInputs) -> tuple[bool, BreakerTrigger, str]:
        """Advance breaker state. Returns ``(tripped, trigger, reason)``."""
        if not self._initialised:
            self.high_water_pnl = inputs.total_pnl
            self._initialised = True
        self.high_water_pnl = max(self.high_water_pnl, inputs.total_pnl)

        trigger, reason = self._evaluate_conditions(inputs)

        if self.tripped:
            elapsed = (inputs.ts_ns - self.tripped_at_ns) / 1e9
            cooled = elapsed >= self.params.breaker_cooldown_seconds
            # Reset only when the cooldown has elapsed AND nothing is still
            # firing. Resetting on the timer alone produces a breaker that
            # flaps in and out of a genuine vol event, which is the worst of
            # both worlds.
            if cooled and trigger is BreakerTrigger.NONE:
                self.tripped = False
                self.trigger = BreakerTrigger.NONE
                # Re-base the high-water mark so a drawdown trip does not
                # instantly re-trip against a mark we can no longer reach.
                self.high_water_pnl = inputs.total_pnl
                return False, BreakerTrigger.NONE, "breaker reset"
            return True, self.trigger, f"breaker latched ({self.trigger})"

        if trigger is not BreakerTrigger.NONE:
            self.tripped = True
            self.trigger = trigger
            self.tripped_at_ns = inputs.ts_ns
            self.trip_count += 1
            return True, trigger, reason

        return False, BreakerTrigger.NONE, ""

    def manual_trip(self, ts_ns: int) -> None:
        self.tripped = True
        self.trigger = BreakerTrigger.MANUAL
        self.tripped_at_ns = ts_ns
        self.trip_count += 1

    def reset(self) -> None:
        self.tripped = False
        self.trigger = BreakerTrigger.NONE
        self.tripped_at_ns = 0
        self._initialised = False
        self.high_water_pnl = 0.0


class RiskEngine:
    """Pure decision function over (proposed quote, market/portfolio state)."""

    __slots__ = ("breaker", "params")

    def __init__(self, params: RiskParams | None = None) -> None:
        self.params = params or RiskParams()
        self.breaker = CircuitBreaker(self.params)

    @staticmethod
    def _non_finite(quote: ProposedQuote, inputs: RiskInputs) -> str:
        """Name the first non-finite field, or '' if all are sound."""
        checks = (
            ("inventory", inputs.inventory),
            ("mid_price", inputs.mid_price),
            ("total_pnl", inputs.total_pnl),
            ("realized_vol", inputs.realized_vol),
            ("baseline_vol", inputs.baseline_vol),
            ("vpin", inputs.vpin),
            ("bid_price", quote.bid_price),
            ("ask_price", quote.ask_price),
            ("bid_size", quote.bid_size),
            ("ask_size", quote.ask_size),
            ("half_spread", quote.half_spread),
        )
        return next((name for name, value in checks if not math.isfinite(value)), "")

    def utilisation(self, inventory: float) -> float:
        if not math.isfinite(inventory) or self.params.q_max <= 0:
            return 0.0
        return abs(inventory) / self.params.q_max

    def skew_for(self, inventory: float, half_spread: float) -> float:
        """Extra centre shift, in price units, beyond what A-S already applies.

        Quadratic in the utilisation *above the soft limit*: zero until
        ``q_soft``, then growing fast. That shape is chosen so this layer never
        interferes with normal quoting -- it exists only for the tail.
        """
        p = self.params
        if p.q_max <= 0.0:
            return 0.0
        excess = (abs(inventory) - p.q_soft) / p.q_max
        if excess <= 0.0:
            return 0.0
        magnitude = p.skew_beta * excess * excess * half_spread
        # Long inventory => skew down => our bid gets less attractive and our
        # ask gets more attractive, which is the direction that flattens us.
        return -magnitude if inventory > 0 else magnitude

    def evaluate(self, quote: ProposedQuote, inputs: RiskInputs) -> RiskOutcome:
        p = self.params

        # --- NaN is the risk layer's worst adversary --------------------------
        # Every IEEE-754 comparison against NaN is False, so `abs(q) >= q_unwind`,
        # `q + size > q_max` and every other bound below silently pass. A single
        # NaN inventory therefore turns the entire risk engine into a no-op while
        # it continues to report APPROVE -- the most dangerous failure mode in the
        # system, because it looks exactly like normal operation.
        #
        # Fail closed: refuse to authorise anything we cannot bound.
        bad = self._non_finite(quote, inputs)
        if bad:
            return RiskOutcome(
                verdict=RiskVerdict.HALT,
                reason=f"non-finite input: {bad}",
                utilisation=0.0,
                breaker_tripped=True,
                trigger=BreakerTrigger.INVALID_INPUT,
            )

        utilisation = self.utilisation(inputs.inventory)

        if not p.enabled:
            return RiskOutcome(
                verdict=RiskVerdict.APPROVE,
                reason="risk disabled",
                bid_price=quote.bid_price,
                ask_price=quote.ask_price,
                bid_size=quote.bid_size,
                ask_size=quote.ask_size,
                utilisation=utilisation,
            )

        tripped, trigger, breaker_reason = self.breaker.update(inputs)
        if tripped:
            return RiskOutcome(
                verdict=RiskVerdict.HALT,
                reason=breaker_reason,
                utilisation=utilisation,
                breaker_tripped=True,
                trigger=trigger,
            )

        skew = self.skew_for(inputs.inventory, quote.half_spread)
        bid_price = quote.bid_price + skew
        ask_price = quote.ask_price + skew
        bid_size = quote.bid_size
        ask_size = quote.ask_size
        reasons: list[str] = []

        # Layer 2: one-sided unwind. Pull the side that would add risk.
        if abs(inputs.inventory) >= p.q_unwind:
            if inputs.inventory > 0:
                bid_size = 0.0
                reasons.append(
                    f"unwind: long {inputs.inventory:.6g} >= {p.q_unwind:.6g}, bid pulled"
                )
            else:
                ask_size = 0.0
                reasons.append(
                    f"unwind: short {inputs.inventory:.6g} <= -{p.q_unwind:.6g}, ask pulled"
                )

        # Hard limit: never let a fill push us past q_max.
        if inputs.inventory + bid_size > p.q_max:
            bid_size = max(0.0, p.q_max - inputs.inventory)
            reasons.append("bid sized down to q_max")
        if inputs.inventory - ask_size < -p.q_max:
            ask_size = max(0.0, p.q_max + inputs.inventory)
            reasons.append("ask sized down to q_max")

        # Notional cap, applied on the resulting position, not the current one.
        if inputs.mid_price > 0.0:
            max_qty = p.max_position_value / inputs.mid_price
            if inputs.inventory + bid_size > max_qty:
                bid_size = max(0.0, max_qty - inputs.inventory)
                reasons.append("bid capped by max_position_value")
            if inputs.inventory - ask_size < -max_qty:
                ask_size = max(0.0, max_qty + inputs.inventory)
                reasons.append("ask capped by max_position_value")

        if bid_size <= 0.0 and ask_size <= 0.0:
            return RiskOutcome(
                verdict=RiskVerdict.REJECT,
                reason="; ".join(reasons) or "both sides sized to zero",
                utilisation=utilisation,
            )

        amended = bool(reasons) or skew != 0.0
        if skew != 0.0:
            reasons.append(f"inventory skew {skew:+.6g}")

        return RiskOutcome(
            verdict=RiskVerdict.AMEND if amended else RiskVerdict.APPROVE,
            reason="; ".join(reasons),
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            skew=skew,
            utilisation=utilisation,
        )

    def reset(self) -> None:
        self.breaker.reset()


__all__ = [
    "BreakerTrigger",
    "CircuitBreaker",
    "ProposedQuote",
    "RiskEngine",
    "RiskInputs",
    "RiskOutcome",
    "RiskParams",
    "RiskVerdict",
]
