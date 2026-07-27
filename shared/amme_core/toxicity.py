"""Adverse-selection gating: how much to widen when the flow looks informed.

Market making loses money in exactly one way that matters -- being on the wrong
side of someone who knows more than you. VPIN and OFI are two orthogonal reads
on that risk:

* **VPIN** is a *magnitude* signal from the tape. High VPIN means order flow is
  persistently one-sided, i.e. someone is working a large informed order. It
  says "be careful", not "which way".
* **OFI** is a *directional* signal from the book. Its sign predicts the next
  mid move (Cont et al. 2014), so it says which side is about to get run over.

They therefore map onto two different controls, and keeping that mapping strict
is what makes the ablation interpretable:

``multiplier``
    Symmetric widening. Driven by VPIN magnitude and |OFI| magnitude. Pure
    "charge more for liquidity when liquidity is dangerous".

``skew``
    Asymmetric shift in price units, driven by the *signed* OFI. Widening the
    at-risk side more than the safe side is exactly a symmetric widening plus a
    centre shift, so it decomposes onto the existing quote API with no special
    case in the quoter.

Both degrade to the identity (1.0, 0.0) when ``enabled=False``, which is how
the Phase 3 pure-A-S baseline is produced from the same code path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToxicityParams:
    vpin_threshold: float = 0.35
    vpin_beta: float = 2.0
    ofi_threshold: float = 2.0
    ofi_beta: float = 0.25
    ofi_skew_beta: float = 0.5
    max_multiplier: float = 5.0
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class ToxicityState:
    vpin: float = 0.0
    vpin_warm: bool = False
    ofi_zscore: float = 0.0
    ofi_warm: bool = False


@dataclass(frozen=True, slots=True)
class ToxicityResponse:
    multiplier: float
    skew_ticks: float
    vpin_excess: float = 0.0
    ofi_excess: float = 0.0
    triggered: bool = False


class ToxicityGate:
    """Maps (VPIN, OFI) onto a spread multiplier and a centre skew."""

    __slots__ = ("params",)

    def __init__(self, params: ToxicityParams | None = None) -> None:
        self.params = params or ToxicityParams()

    def evaluate(self, state: ToxicityState) -> ToxicityResponse:
        p = self.params
        if not p.enabled:
            return ToxicityResponse(multiplier=1.0, skew_ticks=0.0)

        # VPIN excess, renormalised to [0, 1] over the range above the
        # threshold. Without renormalising, `vpin_beta` would mean something
        # different for every choice of threshold, which makes the parameter
        # impossible to reason about or transfer between symbols.
        vpin_excess = 0.0
        if state.vpin_warm and p.vpin_threshold < 1.0:
            vpin_excess = max(0.0, (state.vpin - p.vpin_threshold) / (1.0 - p.vpin_threshold))

        # OFI excess in units of "standard deviations beyond the threshold".
        ofi_excess = 0.0
        if state.ofi_warm:
            ofi_excess = max(0.0, abs(state.ofi_zscore) - p.ofi_threshold)

        multiplier = 1.0 + p.vpin_beta * vpin_excess + p.ofi_beta * ofi_excess
        multiplier = min(multiplier, p.max_multiplier)

        # Signed lean, away from the side about to be adversely selected.
        # Positive OFI => buy pressure => mid likely up => shift quotes up so we
        # are not the one selling into the move.
        skew_ticks = 0.0
        if state.ofi_warm and ofi_excess > 0.0:
            direction = 1.0 if state.ofi_zscore > 0 else -1.0
            skew_ticks = p.ofi_skew_beta * ofi_excess * direction

        return ToxicityResponse(
            multiplier=multiplier,
            skew_ticks=skew_ticks,
            vpin_excess=vpin_excess,
            ofi_excess=ofi_excess,
            triggered=multiplier > 1.0 + 1e-12 or skew_ticks != 0.0,
        )


__all__ = ["ToxicityGate", "ToxicityParams", "ToxicityResponse", "ToxicityState"]
