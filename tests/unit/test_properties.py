"""Property-based tests over the quant formulas (spec s.6.6).

Example-based tests check the cases you thought of. These check invariants that
must hold across the whole parameter space -- which is where a market maker
actually operates, since sigma and inventory are whatever the market hands you.

The properties are chosen to be *structural*: each one, if violated, indicates a
specific real bug (negative spread, inverted inventory response, crossed quote)
rather than a numerical wobble.
"""

from __future__ import annotations

import math

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from amme_core.avellaneda_stoikov import (
    ASParams,
    AvellanedaStoikovQuoter,
    MarketState,
    optimal_total_spread,
    reservation_price,
)
from amme_core.ofi import BestQuote, ofi_increment
from amme_core.quantize import round_to_lot, round_to_tick
from amme_core.risk import ProposedQuote, RiskEngine, RiskInputs, RiskParams
from amme_core.toxicity import ToxicityGate, ToxicityParams, ToxicityState
from amme_core.vpin import VPINCalculator, normal_cdf
from amme_shared.events import AggressorSide, Side

positive = st.floats(min_value=1e-4, max_value=1e3, allow_nan=False, allow_infinity=False)
nonneg = st.floats(min_value=0.0, max_value=1e3, allow_nan=False, allow_infinity=False)
prices = st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False)
inventories = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)


class TestSpreadProperties:
    @given(gamma=positive, sigma=nonneg, tau=nonneg, kappa=positive)
    def test_spread_is_always_positive(self, gamma, sigma, tau, kappa):
        """A negative spread would mean quoting a crossed market -- offering to
        buy above where we offer to sell. It must be impossible by construction."""
        assert optimal_total_spread(gamma, sigma, tau, kappa) > 0.0

    @given(gamma=positive, sigma=nonneg, tau=nonneg, kappa=positive)
    def test_spread_at_least_the_markup_term(self, gamma, sigma, tau, kappa):
        """Even at zero volatility the monopolistic markup survives."""
        spread = optimal_total_spread(gamma, sigma, tau, kappa)
        markup = (2.0 / gamma) * math.log1p(gamma / kappa)
        assert spread >= markup - 1e-9

    @given(gamma=positive, sigma=positive, tau=positive, kappa=positive,
           extra=st.floats(min_value=0.01, max_value=100.0))
    def test_spread_increases_with_volatility(self, gamma, sigma, tau, kappa, extra):
        a = optimal_total_spread(gamma, sigma, tau, kappa)
        b = optimal_total_spread(gamma, sigma + extra, tau, kappa)
        assert b >= a - 1e-9

    @given(gamma=positive, sigma=positive, tau=positive,
           kappa=positive, extra=st.floats(min_value=0.01, max_value=1e3))
    def test_spread_decreases_with_kappa(self, gamma, sigma, tau, kappa, extra):
        """Higher kappa = fills decay faster with distance = quote tighter."""
        wide = optimal_total_spread(gamma, sigma, tau, kappa)
        tight = optimal_total_spread(gamma, sigma, tau, kappa + extra)
        assert tight <= wide + 1e-9


class TestReservationPriceProperties:
    @given(mid=prices, gamma=positive, sigma=nonneg, tau=nonneg)
    def test_flat_inventory_is_the_mid(self, mid, gamma, sigma, tau):
        assert reservation_price(mid, 0.0, gamma, sigma, tau) == mid

    @given(mid=prices, q=st.floats(min_value=0.01, max_value=1e3),
           gamma=positive, sigma=positive, tau=positive)
    def test_long_marks_down_short_marks_up(self, mid, q, gamma, sigma, tau):
        """The core economic content of the reservation price.

        The ``assume`` is a float-representation guard, not a weakened
        assertion: hypothesis will happily pick an adjustment of ~1e-18 against
        a mid of 1.0, where ``mid - adjustment == mid`` in IEEE-754. That is
        double precision behaving correctly, not the formula being wrong, so
        the case is excluded rather than asserted away with a tolerance.
        """
        adjustment = q * gamma * sigma * sigma * tau
        assume(adjustment > abs(mid) * 1e-15)
        assert reservation_price(mid, q, gamma, sigma, tau) < mid
        assert reservation_price(mid, -q, gamma, sigma, tau) > mid

    @given(mid=prices, q=st.floats(min_value=0.01, max_value=100.0),
           gamma=st.floats(min_value=0.01, max_value=10.0),
           sigma=st.floats(min_value=0.01, max_value=10.0),
           tau=st.floats(min_value=0.01, max_value=100.0),
           bump=st.floats(min_value=0.01, max_value=10.0))
    def test_higher_gamma_pushes_harder_toward_flat(self, mid, q, gamma, sigma, tau, bump):
        """Spec s.6.6 names this one: reservation price must move toward
        reducing |inventory| as gamma increases."""
        base = reservation_price(mid, q, gamma, sigma, tau)
        more = reservation_price(mid, q, gamma + bump, sigma, tau)
        assert more <= base + 1e-9   # long -> marked down further

    @given(mid=prices, q=inventories, gamma=positive, sigma=nonneg, tau=nonneg)
    def test_antisymmetric_in_inventory(self, mid, q, gamma, sigma, tau):
        up = reservation_price(mid, q, gamma, sigma, tau) - mid
        down = reservation_price(mid, -q, gamma, sigma, tau) - mid
        assert math.isclose(up, -down, rel_tol=1e-9, abs_tol=1e-9)


class TestQuoteProperties:
    @settings(max_examples=200)
    @given(
        mid=st.floats(min_value=10.0, max_value=1e5),
        sigma=st.floats(min_value=0.0, max_value=50.0),
        inventory=st.floats(min_value=-5.0, max_value=5.0),
        gamma=st.floats(min_value=0.001, max_value=5.0),
        kappa=st.floats(min_value=0.1, max_value=500.0),
    )
    def test_quote_is_never_crossed(self, mid, sigma, inventory, gamma, kappa):
        """The single most important invariant: bid < ask, always, for every
        reachable combination of market state and parameters."""
        quoter = AvellanedaStoikovQuoter(
            ASParams(gamma=gamma, kappa=kappa, horizon_seconds=10.0, tick_size=0.01,
                     inventory_scale=1.0)
        )
        q = quoter.quote(MarketState(mid=mid, sigma=sigma, inventory=inventory))
        assert q.bid_price < q.ask_price
        assert not q.is_crossed

    @settings(max_examples=100)
    @given(
        mid=st.floats(min_value=10.0, max_value=1e5),
        sigma=st.floats(min_value=0.0, max_value=20.0),
        inventory=st.floats(min_value=0.1, max_value=5.0),
    )
    def test_long_inventory_lowers_both_quotes(self, mid, sigma, inventory):
        quoter = AvellanedaStoikovQuoter(
            ASParams(gamma=0.5, kappa=10.0, horizon_seconds=10.0, tick_size=0.01,
                     inventory_scale=1.0, min_half_spread_ticks=0.0,
                     max_half_spread_ticks=1e9)
        )
        flat = quoter.quote(MarketState(mid=mid, sigma=sigma, inventory=0.0))
        long = quoter.quote(MarketState(mid=mid, sigma=sigma, inventory=inventory))
        assert long.bid_price <= flat.bid_price + 1e-9
        assert long.ask_price <= flat.ask_price + 1e-9

    @settings(max_examples=100)
    @given(
        mid=st.floats(min_value=10.0, max_value=1e5),
        sigma=st.floats(min_value=0.0, max_value=20.0),
        multiplier=st.floats(min_value=1.0, max_value=5.0),
    )
    def test_toxicity_multiplier_never_narrows(self, mid, sigma, multiplier):
        quoter = AvellanedaStoikovQuoter(
            ASParams(gamma=0.5, kappa=10.0, horizon_seconds=10.0, tick_size=0.01,
                     inventory_scale=1.0, min_half_spread_ticks=0.0,
                     max_half_spread_ticks=1e9)
        )
        state = MarketState(mid=mid, sigma=sigma, inventory=0.0)
        base = quoter.quote(state, toxicity_multiplier=1.0)
        widened = quoter.quote(state, toxicity_multiplier=multiplier)
        assert widened.total_spread >= base.total_spread - 1e-9


class TestToxicityProperties:
    @given(
        vpin=st.floats(min_value=0.0, max_value=1.0),
        ofi_z=st.floats(min_value=-10.0, max_value=10.0),
    )
    def test_multiplier_is_at_least_one_and_bounded(self, vpin, ofi_z):
        """Toxicity may only widen, never tighten -- and never without bound."""
        gate = ToxicityGate(ToxicityParams(max_multiplier=5.0))
        r = gate.evaluate(ToxicityState(vpin=vpin, vpin_warm=True,
                                        ofi_zscore=ofi_z, ofi_warm=True))
        assert 1.0 <= r.multiplier <= 5.0

    @given(vpin=st.floats(min_value=0.0, max_value=1.0),
           ofi_z=st.floats(min_value=-10.0, max_value=10.0))
    def test_disabled_gate_is_the_identity(self, vpin, ofi_z):
        """This is what makes the pure-A-S ablation arm an exact special case
        rather than a separate code path."""
        gate = ToxicityGate(ToxicityParams(enabled=False))
        r = gate.evaluate(ToxicityState(vpin=vpin, vpin_warm=True,
                                        ofi_zscore=ofi_z, ofi_warm=True))
        assert r.multiplier == 1.0
        assert r.skew_ticks == 0.0

    @given(vpin=st.floats(min_value=0.0, max_value=1.0))
    def test_cold_vpin_does_not_widen(self, vpin):
        gate = ToxicityGate(ToxicityParams())
        r = gate.evaluate(ToxicityState(vpin=vpin, vpin_warm=False, ofi_warm=False))
        assert r.multiplier == 1.0

    @given(ofi_z=st.floats(min_value=2.5, max_value=10.0))
    def test_skew_follows_ofi_sign(self, ofi_z):
        gate = ToxicityGate(ToxicityParams(ofi_threshold=2.0, ofi_skew_beta=1.0))
        up = gate.evaluate(ToxicityState(ofi_zscore=ofi_z, ofi_warm=True))
        down = gate.evaluate(ToxicityState(ofi_zscore=-ofi_z, ofi_warm=True))
        assert up.skew_ticks > 0    # buy pressure -> lean up
        assert down.skew_ticks < 0
        assert math.isclose(up.skew_ticks, -down.skew_ticks, rel_tol=1e-9)


class TestVPINProperties:
    @settings(max_examples=50)
    @given(
        trades=st.lists(
            st.tuples(
                st.floats(min_value=1.0, max_value=1000.0),
                st.floats(min_value=0.01, max_value=10.0),
                st.sampled_from([AggressorSide.BUY, AggressorSide.SELL]),
            ),
            min_size=1, max_size=120,
        )
    )
    def test_vpin_stays_in_unit_interval(self, trades):
        calc = VPINCalculator(bucket_volume=5.0, n_buckets=10)
        for price, size, side in trades:
            calc.update(price, size, side)
            assert 0.0 <= calc.value <= 1.0

    @given(x=st.floats(min_value=-8.0, max_value=8.0))
    def test_normal_cdf_is_a_cdf(self, x):
        v = normal_cdf(x)
        assert 0.0 <= v <= 1.0
        assert math.isclose(v + normal_cdf(-x), 1.0, abs_tol=1e-12)

    @given(x=st.floats(min_value=-5.0, max_value=5.0),
           d=st.floats(min_value=0.01, max_value=5.0))
    def test_normal_cdf_is_monotone(self, x, d):
        assert normal_cdf(x + d) >= normal_cdf(x)


class TestOFIProperties:
    @given(
        pb=st.floats(min_value=1.0, max_value=1000.0),
        qb=st.floats(min_value=0.0, max_value=100.0),
        pa=st.floats(min_value=1.0, max_value=1000.0),
        qa=st.floats(min_value=0.0, max_value=100.0),
    )
    def test_unchanged_book_gives_zero(self, pb, qb, pa, qa):
        q = BestQuote(pb, qb, pa, qa)
        assert ofi_increment(q, q) == 0.0

    @given(
        pb=st.floats(min_value=10.0, max_value=100.0),
        qb=st.floats(min_value=0.1, max_value=50.0),
        qa=st.floats(min_value=0.1, max_value=50.0),
        up=st.floats(min_value=0.01, max_value=10.0),
    )
    def test_both_sides_up_is_buy_pressure(self, pb, qb, qa, up):
        """OFI's sign convention: the book shifting up must read positive."""
        pa = pb + 1.0
        prev = BestQuote(pb, qb, pa, qa)
        cur = BestQuote(pb + up, qb, pa + up, qa)
        assert ofi_increment(prev, cur) > 0.0


class TestQuantizeProperties:
    @given(price=st.floats(min_value=0.01, max_value=1e6),
           tick=st.sampled_from([0.01, 0.1, 0.5, 1.0, 5.0]))
    def test_rounding_lands_on_the_grid(self, price, tick):
        for mode in ("nearest", "up", "down"):
            snapped = round_to_tick(price, tick, mode)
            assert math.isclose(snapped / tick, round(snapped / tick), abs_tol=1e-6)

    @given(price=st.floats(min_value=0.01, max_value=1e6),
           tick=st.sampled_from([0.01, 0.1, 0.5, 1.0]))
    def test_directional_rounding_respects_direction(self, price, tick):
        assert round_to_tick(price, tick, "down") <= price + 1e-9
        assert round_to_tick(price, tick, "up") >= price - 1e-9

    @given(price=st.floats(min_value=0.01, max_value=1e6),
           tick=st.sampled_from([0.01, 0.1, 1.0]))
    def test_rounding_error_bounded_by_one_tick(self, price, tick):
        assert abs(round_to_tick(price, tick, "nearest") - price) <= tick

    @given(size=st.floats(min_value=0.0, max_value=1e4),
           lot=st.sampled_from([1e-5, 1e-3, 0.1]))
    def test_lot_rounding_never_increases_risk(self, size, lot):
        """Default rounds down: never send more size than intended."""
        assert round_to_lot(size, lot) <= size + 1e-9
        assert round_to_lot(size, lot) >= 0.0


class TestRiskProperties:
    @settings(max_examples=200)
    @given(
        inventory=st.floats(min_value=-0.2, max_value=0.2),
        mid=st.floats(min_value=100.0, max_value=1e5),
        half_spread=st.floats(min_value=0.01, max_value=10.0),
    )
    def test_approved_size_never_breaches_the_limit(self, inventory, mid, half_spread):
        """The hard guarantee risk-svc exists to provide: no approved quote can
        take the position past q_max on either side."""
        q_max = 0.05
        eng = RiskEngine(RiskParams(q_max=q_max, breaker_drawdown=1e9,
                                    max_position_value=1e12))
        assume(abs(inventory) <= q_max)
        out = eng.evaluate(
            ProposedQuote(mid - half_spread, mid + half_spread, 0.01, 0.01, half_spread),
            RiskInputs(inventory=inventory, mid_price=mid),
        )
        if out.actionable:
            assert inventory + (out.bid_size or 0.0) <= q_max + 1e-12
            assert inventory - (out.ask_size or 0.0) >= -q_max - 1e-12

    @settings(max_examples=100)
    @given(inventory=st.floats(min_value=-0.2, max_value=0.2),
           mid=st.floats(min_value=100.0, max_value=1e5))
    def test_approved_quote_is_never_crossed(self, inventory, mid):
        eng = RiskEngine(RiskParams(q_max=0.05, breaker_drawdown=1e9))
        out = eng.evaluate(
            ProposedQuote(mid - 1.0, mid + 1.0, 0.01, 0.01, 1.0),
            RiskInputs(inventory=inventory, mid_price=mid),
        )
        if out.actionable and out.bid_price is not None and out.ask_price is not None:
            assert out.bid_price < out.ask_price

    @given(size=st.floats(min_value=0.0, max_value=1.0))
    def test_portfolio_identity_under_arbitrary_fills(self, size):
        from amme_core.matching import Portfolio

        p = Portfolio()
        p.apply_fill(Side.BID, 100.0, size)
        p.apply_fill(Side.ASK, 101.0, size * 0.5)
        assert p.check_identity(100.5)
