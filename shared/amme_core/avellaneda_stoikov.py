"""Avellaneda-Stoikov optimal market making.

Avellaneda & Stoikov (2008), *"High-frequency trading in a limit order book"*,
Quantitative Finance 8(3), 217-224. The full HJB derivation is written out in
``docs/QUANT_DERIVATIONS.md``; this module is only the closed-form result and
the engineering around it.

**Reservation price** -- the indifference price of a maker holding inventory
``q``. It is *not* the mid: an inventory-long maker marks their own book down,
so they get hit less on the bid and lifted more on the ask.

.. math:: r(s, q, t) = s - q \\gamma \\sigma^2 (T - t)

**Optimal total spread** -- the sum of the two half-spreads:

.. math:: \\delta^a + \\delta^b = \\gamma \\sigma^2 (T-t) + \\frac{2}{\\gamma}\\ln\\!\\left(1 + \\frac{\\gamma}{\\kappa}\\right)

Term one is inventory-risk compensation; term two is the monopolistic markup
implied by the exponential arrival intensity :math:`\\lambda(\\delta) = A e^{-\\kappa\\delta}`.
Note the second term does **not** depend on ``q`` or ``t`` -- the whole
inventory story lives in the reservation price and in term one.

**Units** (pinned here because getting them wrong fails silently):

* ``sigma`` -- volatility of the mid in **quote-currency units per sqrt(second)**,
  matching the paper's arithmetic Brownian motion :math:`dS_t = \\sigma dW_t`.
  Passing a *return* volatility here is a units error that produces spreads
  wrong by a factor of the price level.
* ``gamma`` -- absolute risk aversion, units of 1/quote-currency.
* ``kappa`` -- arrival decay, units of 1/quote-currency.
* ``q``     -- a dimensionless position *count*. Checking the reservation-price
  formula: ``q * gamma * sigma^2 * tau`` = ``1 * (1/$) * ($^2/s) * s`` = ``$``. ✓

**Inventory normalisation.** That last point is the practical trap. A-S assumes
``q`` is a share count of order 1. Our inventory is in BTC with a limit of
0.05, so a literal reading makes the reservation-price tilt ~20x smaller than
the spread's own inventory term, and inventory control silently does nothing.

The fix is a change of units, not a change of model: we measure inventory in
units of the risk limit, :math:`\\tilde q = q / q_{max}`. Since ``q`` enters the
solution only through the product :math:`q\\gamma`, rescaling ``q`` by
:math:`q_{max}` and ``gamma`` by :math:`1/q_{max}` leaves the reservation price
*exactly* invariant -- it is a reparametrisation, not an approximation. What it
buys is an interpretable ``gamma``: risk aversion per full-limit position, on
the same scale as the spread's inventory term. ``ASParams.inventory_scale``
carries :math:`q_{max}`; the pure functions below still take the raw textbook
``q`` so the hand-verified unit tests check the literal published formula.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from amme_core.quantize import clamp, round_to_tick

# ---------------------------------------------------------------------------
# Closed-form components (pure functions -- these are the hand-verified ones)
# ---------------------------------------------------------------------------


def reservation_price(mid: float, inventory: float, gamma: float, sigma: float, tau: float) -> float:
    """:math:`r = s - q\\gamma\\sigma^2\\tau` where ``tau = T - t`` in seconds."""
    if gamma <= 0.0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if tau < 0.0:
        raise ValueError(f"tau must be non-negative, got {tau}")
    return mid - inventory * gamma * sigma * sigma * tau


def optimal_total_spread(gamma: float, sigma: float, tau: float, kappa: float) -> float:
    """:math:`\\gamma\\sigma^2\\tau + (2/\\gamma)\\ln(1 + \\gamma/\\kappa)`."""
    if gamma <= 0.0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    if kappa <= 0.0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    if sigma < 0.0 or tau < 0.0:
        raise ValueError("sigma and tau must be non-negative")
    inventory_term = gamma * sigma * sigma * tau
    markup_term = (2.0 / gamma) * math.log1p(gamma / kappa)
    return inventory_term + markup_term


def inventory_half_life(gamma: float, sigma: float, kappa: float) -> float:
    """Rough time-scale over which the reservation-price tilt closes 1 unit of
    inventory. Diagnostic only -- used to sanity-check that ``gamma`` is in a
    sane range for the symbol's volatility, not by the quoting path."""
    if sigma <= 0.0:
        return math.inf
    return 1.0 / (gamma * sigma * sigma * max(kappa, 1e-12))


# ---------------------------------------------------------------------------
# Quoting
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ASParams:
    """Everything the quoter needs that is not market state.

    Default ``gamma``/``kappa``/``horizon_seconds`` are jointly calibrated, not
    picked independently -- see ``docs/ARCHITECTURE.md`` "Calibration defaults".
    They are solved, not guessed, against measured conditions: sigma ~ 0.20
    $/sqrt(s) and a median market spread of 2 ticks at tick=0.01. Requiring
    (a) a 1-tick half-spread when flat, so we are competitive with that market
    spread, and (b) a 1-tick reservation tilt at full inventory, so inventory
    pressure is on the same scale as spread compensation, gives

        gamma*sigma^2*tau = 1 tick   and   (2/gamma)*ln(1 + gamma/kappa) = 1 tick

    which solves to gamma ~ 0.017, kappa ~ 200 at tau = 15s.

    The target matters as much as the arithmetic: a maker quoting several ticks
    wider than the market is not being prudent, it is not participating -- it
    will simply never be filled, and an ablation arm that never trades measures
    nothing. Phase 3 re-derives all three by grid search.
    """

    gamma: float = 0.017
    kappa: float = 200.0
    horizon_mode: str = "infinite"
    horizon_seconds: float = 15.0
    session_seconds: float = 3600.0
    tick_size: float = 0.01
    min_half_spread_ticks: float = 1.0
    max_half_spread_ticks: float = 500.0
    quote_size: float = 0.001
    #: q_max, in base currency. Inventory is divided by this before entering the
    #: A-S formulas -- see the module docstring on inventory normalisation.
    inventory_scale: float = 0.05
    drift_lambda: float = 0.0
    drift_confidence_floor: float = 0.40
    drift_max_ticks: float = 20.0

    def normalise_inventory(self, q: float) -> float:
        """Raw inventory -> units of the risk limit."""
        return q / self.inventory_scale if self.inventory_scale > 0.0 else q

    def tau(self, elapsed_seconds: float) -> float:
        """Time-to-horizon, in seconds.

        ``finite`` reproduces the paper literally: ``tau`` decays to zero at the
        session end, so the maker's urgency to flatten grows without bound.
        ``infinite`` pins ``tau`` to a constant, which is the honest model for a
        desk that quotes 24/7 and has no terminal liquidation. Default is
        ``infinite``; see docs/ARCHITECTURE.md "Calibration defaults".
        """
        if self.horizon_mode == "finite":
            return max(0.0, self.session_seconds - elapsed_seconds)
        return self.horizon_seconds


@dataclass(frozen=True, slots=True)
class MarketState:
    """Market inputs to one quoting decision."""

    mid: float
    sigma: float
    inventory: float
    elapsed_seconds: float = 0.0
    drift: float = 0.0
    drift_confidence: float = 0.0
    best_bid: float | None = None
    best_ask: float | None = None


@dataclass(frozen=True, slots=True)
class Quote:
    """A two-sided quote plus the decomposition needed for the ablation report."""

    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    reservation_price: float
    half_spread: float
    drift_adjustment: float = 0.0
    toxicity_multiplier: float = 1.0
    skew: float = 0.0

    @property
    def total_spread(self) -> float:
        return self.ask_price - self.bid_price

    @property
    def is_crossed(self) -> bool:
        return self.bid_price >= self.ask_price


class AvellanedaStoikovQuoter:
    """Turns market state into a two-sided quote.

    Deliberately stateless: every input arrives in :class:`MarketState`. That is
    what lets the backtester call ``quote()`` a million times in a loop and get
    identical results to the live service, and what makes the property-based
    tests in s.6.6 straightforward to write.
    """

    __slots__ = ("params",)

    def __init__(self, params: ASParams | None = None) -> None:
        self.params = params or ASParams()

    # -- drift injection ---------------------------------------------------
    def drift_adjustment(self, state: MarketState) -> float:
        """``lambda * drift * mid``, gated on confidence and hard-clamped.

        The functional form is additive on the reservation price:

            ``r_adj = r + lambda * E[return] * mid``

        chosen over a multiplicative or spread-based alternative for three
        reasons. (1) It is dimensionally clean: ``drift`` is a dimensionless
        expected return, so multiplying by ``mid`` yields quote-currency units
        that add directly to ``r``. (2) It keeps the *spread* -- the compensation
        for inventory and adverse selection -- completely independent of the
        directional view, so the ablation cleanly separates "where we quote"
        from "how wide we quote". (3) It degrades to exactly pure A-S at
        ``lambda = 0``, which is what makes the Phase 3 baseline an exact
        special case of the Phase 4 system rather than a different code path.

        Two safety rails, both non-negotiable for anything model-driven:
        a confidence floor (ignore low-conviction predictions outright) and an
        absolute clamp in ticks (a model bug can never walk the quote away from
        the market).
        """
        p = self.params
        if p.drift_lambda <= 0.0 or state.drift == 0.0:
            return 0.0
        if state.drift_confidence < p.drift_confidence_floor:
            return 0.0
        raw = p.drift_lambda * state.drift * state.mid
        limit = p.drift_max_ticks * p.tick_size
        return clamp(raw, -limit, limit)

    # -- main entry point --------------------------------------------------
    def quote(
        self,
        state: MarketState,
        toxicity_multiplier: float = 1.0,
        skew: float = 0.0,
        size_override: tuple[float, float] | None = None,
    ) -> Quote:
        """Compute the two-sided quote.

        ``toxicity_multiplier`` scales the half-spread (>= 1 widens).
        ``skew`` shifts both quotes in price units (negative leans down, i.e.
        away from accumulating more inventory when already long).
        """
        p = self.params
        tau = p.tau(state.elapsed_seconds)
        q_tilde = p.normalise_inventory(state.inventory)

        r = reservation_price(state.mid, q_tilde, p.gamma, state.sigma, tau)
        drift_adj = self.drift_adjustment(state)
        r_adj = r + drift_adj + skew

        total_spread = optimal_total_spread(p.gamma, state.sigma, tau, p.kappa)
        half_spread = 0.5 * total_spread * max(0.0, toxicity_multiplier)

        # Bound the half-spread. The floor keeps us competitive enough to ever
        # get filled; the ceiling stops a volatility spike from producing a
        # nominally-valid quote that is really just "not quoting".
        half_spread = clamp(
            half_spread,
            p.min_half_spread_ticks * p.tick_size,
            p.max_half_spread_ticks * p.tick_size,
        )

        bid = r_adj - half_spread
        ask = r_adj + half_spread

        # Round *outward* -- never round a quote to a more aggressive price than
        # the model asked for. Rounding the bid up would mean paying more than
        # the model's indifference price purely because of tick granularity.
        bid = round_to_tick(bid, p.tick_size, mode="down")
        ask = round_to_tick(ask, p.tick_size, mode="up")

        if bid >= ask:  # degenerate at sub-tick spreads; force one tick apart
            centre = round_to_tick(r_adj, p.tick_size, mode="nearest")
            bid = centre - p.tick_size
            ask = centre + p.tick_size

        bid_size, ask_size = size_override or (p.quote_size, p.quote_size)

        return Quote(
            bid_price=bid,
            ask_price=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            reservation_price=r_adj,
            half_spread=half_spread,
            drift_adjustment=drift_adj,
            toxicity_multiplier=toxicity_multiplier,
            skew=skew,
        )

    def with_params(self, **overrides: float | str) -> AvellanedaStoikovQuoter:
        """A copy with some parameters changed. Used by the calibration grid search."""
        return AvellanedaStoikovQuoter(replace(self.params, **overrides))  # type: ignore[arg-type]


__all__ = [
    "ASParams",
    "AvellanedaStoikovQuoter",
    "MarketState",
    "Quote",
    "inventory_half_life",
    "optimal_total_spread",
    "reservation_price",
]
