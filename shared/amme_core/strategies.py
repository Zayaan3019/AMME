"""The three strategies the ablation compares (s.6.2).

They are built as a strict nesting, which is the whole point of the design:

.. code-block:: text

    NaiveSymmetric  : fixed spread around the mid. No inventory term at all.
    PureAvellanedaStoikov : + reservation price, + optimal spread
    AMME            : + DeepLOB drift injection, + VPIN/OFI toxicity gate

Each adds exactly one mechanism to the one above it, and they share the *same*
risk layer, the same matching engine and the same metric code. That is what
makes a difference in the results table attributable to the mechanism rather
than to an incidental implementation difference -- and it is why ``AMME`` with
``drift_lambda=0`` and ``toxicity.enabled=False`` must reproduce
``PureAvellanedaStoikov`` exactly. ``tests/regression`` asserts that equivalence
directly, so the nesting cannot silently rot.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from amme_core.avellaneda_stoikov import ASParams, AvellanedaStoikovQuoter, MarketState, Quote
from amme_core.quantize import round_to_tick
from amme_core.toxicity import ToxicityGate, ToxicityParams, ToxicityState


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """Everything a strategy may look at. Deliberately a closed set."""

    mid: float
    micro_price: float
    sigma: float
    inventory: float
    elapsed_seconds: float
    best_bid: float
    best_ask: float
    vpin: float = 0.0
    vpin_warm: bool = False
    ofi_zscore: float = 0.0
    ofi_warm: bool = False
    drift: float = 0.0
    drift_confidence: float = 0.0


class Strategy(Protocol):
    name: str

    def decide(self, ctx: DecisionContext) -> Quote: ...

    def reset(self) -> None: ...


@dataclass(frozen=True, slots=True)
class NaiveParams:
    half_spread_ticks: float = 2.0
    tick_size: float = 0.01
    quote_size: float = 0.001
    #: Anchor on the micro-price rather than the mid. Off by default so the
    #: baseline is genuinely naive -- turning it on is already a microstructure
    #: edge, and would muddy what the A-S comparison is measuring.
    use_micro_price: bool = False


class NaiveSymmetricStrategy:
    """Fixed-width two-sided quote. The floor the other strategies must beat.

    This is not a straw man -- a constant spread around the mid is a real and
    frequently-deployed configuration, and it wins whenever inventory risk is
    negligible. What it cannot do is manage inventory or step back from
    informed flow, which is exactly what the other two rows measure.
    """

    name = "naive_symmetric"

    __slots__ = ("params",)

    def __init__(self, params: NaiveParams | None = None) -> None:
        self.params = params or NaiveParams()

    def decide(self, ctx: DecisionContext) -> Quote:
        p = self.params
        anchor = ctx.micro_price if p.use_micro_price else ctx.mid
        half = p.half_spread_ticks * p.tick_size
        bid = round_to_tick(anchor - half, p.tick_size, mode="down")
        ask = round_to_tick(anchor + half, p.tick_size, mode="up")
        return Quote(
            bid_price=bid,
            ask_price=ask,
            bid_size=p.quote_size,
            ask_size=p.quote_size,
            reservation_price=anchor,
            half_spread=half,
        )

    def reset(self) -> None:
        return None


class PureAvellanedaStoikovStrategy:
    """A-S reservation price and optimal spread. No ML, no toxicity gating."""

    name = "pure_avellaneda_stoikov"

    __slots__ = ("quoter",)

    def __init__(self, params: ASParams | None = None) -> None:
        # Force the drift term off regardless of what was passed in: this
        # strategy's identity is "A-S and nothing else", and letting a stray
        # config value leak a drift term in would corrupt the baseline the
        # entire ablation is measured against.
        base = params or ASParams()
        # dataclasses.replace, not a dict splat: it is type-checked, so a
        # renamed or retyped field becomes a static error instead of silently
        # falling back to a default and corrupting the ablation baseline.
        self.quoter = AvellanedaStoikovQuoter(replace(base, drift_lambda=0.0))

    def decide(self, ctx: DecisionContext) -> Quote:
        state = MarketState(
            mid=ctx.mid,
            sigma=ctx.sigma,
            inventory=ctx.inventory,
            elapsed_seconds=ctx.elapsed_seconds,
            best_bid=ctx.best_bid,
            best_ask=ctx.best_ask,
        )
        return self.quoter.quote(state)

    def reset(self) -> None:
        return None


class AMMEStrategy:
    """The full engine: A-S + DeepLOB drift + VPIN/OFI toxicity gating."""

    name = "amme"

    __slots__ = ("gate", "quoter")

    def __init__(
        self,
        as_params: ASParams | None = None,
        toxicity_params: ToxicityParams | None = None,
    ) -> None:
        self.quoter = AvellanedaStoikovQuoter(as_params)
        self.gate = ToxicityGate(toxicity_params)

    def decide(self, ctx: DecisionContext) -> Quote:
        response = self.gate.evaluate(
            ToxicityState(
                vpin=ctx.vpin,
                vpin_warm=ctx.vpin_warm,
                ofi_zscore=ctx.ofi_zscore,
                ofi_warm=ctx.ofi_warm,
            )
        )
        state = MarketState(
            mid=ctx.mid,
            sigma=ctx.sigma,
            inventory=ctx.inventory,
            elapsed_seconds=ctx.elapsed_seconds,
            drift=ctx.drift,
            drift_confidence=ctx.drift_confidence,
            best_bid=ctx.best_bid,
            best_ask=ctx.best_ask,
        )
        return self.quoter.quote(
            state,
            toxicity_multiplier=response.multiplier,
            skew=response.skew_ticks * self.quoter.params.tick_size,
        )

    def reset(self) -> None:
        self.gate.reset()


def build_strategy(
    name: str,
    as_params: ASParams | None = None,
    toxicity_params: ToxicityParams | None = None,
    naive_params: NaiveParams | None = None,
) -> Strategy:
    """Factory used by the regression runner and the CLI."""
    key = name.lower().replace("-", "_")
    if key in ("naive", "naive_symmetric"):
        return NaiveSymmetricStrategy(naive_params)
    if key in ("pure_as", "pure_avellaneda_stoikov", "as"):
        return PureAvellanedaStoikovStrategy(as_params)
    if key == "amme":
        return AMMEStrategy(as_params, toxicity_params)
    raise ValueError(
        f"unknown strategy {name!r}; expected one of "
        "'naive_symmetric', 'pure_avellaneda_stoikov', 'amme'"
    )


STRATEGY_NAMES: tuple[str, ...] = ("naive_symmetric", "pure_avellaneda_stoikov", "amme")

__all__ = [
    "STRATEGY_NAMES",
    "AMMEStrategy",
    "DecisionContext",
    "NaiveParams",
    "NaiveSymmetricStrategy",
    "PureAvellanedaStoikovStrategy",
    "Strategy",
    "build_strategy",
]
