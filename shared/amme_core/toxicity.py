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

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToxicityParams:
    """Gate parameters.

    ``vpin_threshold`` is the one number here that must not be a magic
    constant. VPIN's *absolute* level depends on bucket size, trade-size
    distribution and venue: measured on the calibrated simulator regime its
    median is 0.52 and its 10th percentile is 0.43, so the textbook-looking
    threshold of 0.35 fired on **99.3% of updates**. The gate was not detecting
    toxicity, it was permanently widening -- which cut fills from 151 to 16 and
    made the "toxicity gate" arm strictly worse than pure A-S.

    The fix is to define the threshold as a *quantile* of the observed VPIN
    distribution rather than a level. With ``adaptive=True`` the gate tracks
    that quantile itself from a rolling window, so it self-calibrates to
    whatever market it is pointed at and the parameter means the same thing
    everywhere: "react to the most toxic N% of the tape".
    """

    #: Used when ``adaptive`` is False. Empirically the p90 of VPIN in the
    #: reference regime -- re-derive it for any other market.
    vpin_threshold: float = 0.67
    vpin_beta: float = 2.0
    #: Track the threshold as this quantile of recent VPIN instead of fixing it.
    adaptive: bool = True
    vpin_quantile: float = 0.90
    #: Rolling window for the quantile estimate, in book updates.
    vpin_window: int = 2000
    #: Recompute the quantile every N updates. Sorting 2000 floats at 10Hz
    #: would be wasteful; the distribution does not move meaningfully faster
    #: than this.
    vpin_recompute_every: int = 100
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
    threshold_used: float = 0.0
    triggered: bool = False


class ToxicityGate:
    """Maps (VPIN, OFI) onto a spread multiplier and a centre skew.

    Stateful only through the adaptive threshold's rolling window. That window
    is updated deterministically from the VPIN series, so the gate remains
    reproducible under replay -- no wall clock, no RNG.
    """

    __slots__ = ("_cached_threshold", "_since_recompute", "_vpin_history", "params")

    def __init__(self, params: ToxicityParams | None = None) -> None:
        self.params = params or ToxicityParams()
        self._vpin_history: deque[float] = deque(maxlen=self.params.vpin_window)
        self._cached_threshold = self.params.vpin_threshold
        self._since_recompute = 0

    @property
    def threshold(self) -> float:
        """The VPIN threshold currently in force."""
        return self._cached_threshold if self.params.adaptive else self.params.vpin_threshold

    def _update_threshold(self, vpin: float) -> float:
        """Track the configured quantile of recent VPIN."""
        p = self.params
        if not p.adaptive:
            return p.vpin_threshold

        self._vpin_history.append(vpin)
        self._since_recompute += 1

        # Hold the static default until the window has enough mass for a
        # quantile to mean anything; a p90 over 20 samples is noise.
        if len(self._vpin_history) < max(50, p.vpin_window // 10):
            return p.vpin_threshold

        if self._since_recompute >= p.vpin_recompute_every or self._cached_threshold <= 0.0:
            self._since_recompute = 0
            ordered = sorted(self._vpin_history)
            index = min(len(ordered) - 1, int(p.vpin_quantile * len(ordered)))
            self._cached_threshold = ordered[index]
        return self._cached_threshold

    def evaluate(self, state: ToxicityState) -> ToxicityResponse:
        p = self.params
        if not p.enabled:
            return ToxicityResponse(multiplier=1.0, skew_ticks=0.0)

        threshold = self._update_threshold(state.vpin) if state.vpin_warm else self.threshold

        # VPIN excess, renormalised to [0, 1] over the range above the
        # threshold. Without renormalising, `vpin_beta` would mean something
        # different for every choice of threshold, which makes the parameter
        # impossible to reason about or transfer between symbols.
        vpin_excess = 0.0
        if state.vpin_warm and threshold < 1.0:
            vpin_excess = max(0.0, (state.vpin - threshold) / (1.0 - threshold))

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
            threshold_used=threshold,
            triggered=multiplier > 1.0 + 1e-12 or skew_ticks != 0.0,
        )

    def reset(self) -> None:
        self._vpin_history.clear()
        self._cached_threshold = self.params.vpin_threshold
        self._since_recompute = 0


__all__ = ["ToxicityGate", "ToxicityParams", "ToxicityResponse", "ToxicityState"]
