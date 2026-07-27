"""Avellaneda-Stoikov: hand-computed verification of the closed-form solution.

Every expected value below is worked out by hand in the docstring or comment
that precedes it, using numbers chosen so the arithmetic is checkable without a
calculator. That is the point of spec s.6.5 -- an assertion against a value the
code itself produced proves only that the code is deterministic.
"""

from __future__ import annotations

import math

import pytest

from amme_core.avellaneda_stoikov import (
    ASParams,
    AvellanedaStoikovQuoter,
    MarketState,
    optimal_total_spread,
    reservation_price,
)


class TestReservationPrice:
    """r(s, q, t) = s - q * gamma * sigma^2 * (T - t)"""

    def test_flat_inventory_equals_mid(self):
        # q = 0 kills the whole adjustment term, whatever the other parameters.
        assert reservation_price(100.0, 0.0, 0.5, 2.0, 1.0) == 100.0

    def test_hand_computed_long(self):
        # s=100, q=2, gamma=0.5, sigma=2, tau=1
        #   adjustment = 2 * 0.5 * 2^2 * 1 = 2 * 0.5 * 4 = 4
        #   r = 100 - 4 = 96
        assert reservation_price(100.0, 2.0, 0.5, 2.0, 1.0) == pytest.approx(96.0)

    def test_hand_computed_short(self):
        # Same magnitudes, q = -2 -> r = 100 + 4 = 104.
        # A short maker marks their book UP: they want to be lifted less on the
        # ask and hit more on the bid.
        assert reservation_price(100.0, -2.0, 0.5, 2.0, 1.0) == pytest.approx(104.0)

    def test_second_worked_example(self):
        # s=100, q=3, gamma=0.1, sigma=2, tau=600
        #   adjustment = 3 * 0.1 * 4 * 600 = 720   ->  r = -620
        assert reservation_price(100.0, 3.0, 0.1, 2.0, 600.0) == pytest.approx(-620.0)

    def test_scales_linearly_in_inventory(self):
        a = 100.0 - reservation_price(100.0, 1.0, 0.3, 1.5, 2.0)
        b = 100.0 - reservation_price(100.0, 4.0, 0.3, 1.5, 2.0)
        assert b == pytest.approx(4.0 * a)

    def test_quadratic_in_sigma(self):
        # sigma appears squared, so doubling it quadruples the tilt.
        a = 100.0 - reservation_price(100.0, 1.0, 0.3, 1.0, 2.0)
        b = 100.0 - reservation_price(100.0, 1.0, 0.3, 2.0, 2.0)
        assert b == pytest.approx(4.0 * a)

    @pytest.mark.parametrize(
        ("gamma", "sigma", "tau"),
        [(0.0, 1.0, 1.0), (-0.1, 1.0, 1.0), (0.1, -1.0, 1.0), (0.1, 1.0, -1.0)],
    )
    def test_rejects_invalid_parameters(self, gamma, sigma, tau):
        with pytest.raises(ValueError):
            reservation_price(100.0, 1.0, gamma, sigma, tau)


class TestOptimalSpread:
    """delta_a + delta_b = gamma*sigma^2*tau + (2/gamma)*ln(1 + gamma/kappa)"""

    def test_hand_computed(self):
        # gamma=0.5, sigma=2, tau=1, kappa=0.5
        #   inventory term = 0.5 * 4 * 1                = 2.0
        #   markup         = (2/0.5) * ln(1 + 0.5/0.5)
        #                  = 4 * ln(2) = 4 * 0.693147...= 2.7725887
        #   total          =                              4.7725887
        expected = 2.0 + 4.0 * math.log(2.0)
        assert optimal_total_spread(0.5, 2.0, 1.0, 0.5) == pytest.approx(expected)
        assert expected == pytest.approx(4.772588722239781)

    def test_second_worked_example(self):
        # gamma=0.1, sigma=2, tau=600, kappa=1.5
        #   inventory = 0.1 * 4 * 600 = 240
        #   markup    = 20 * ln(1 + 0.0666...) = 20 * 0.0645385 = 1.2907704
        expected = 240.0 + 20.0 * math.log1p(0.1 / 1.5)
        assert optimal_total_spread(0.1, 2.0, 600.0, 1.5) == pytest.approx(expected)
        assert expected == pytest.approx(241.29077042275142)

    def test_markup_term_survives_zero_volatility(self):
        # With sigma=0 the inventory term vanishes but the monopolistic markup
        # does not -- a maker still charges for providing the option, even in a
        # market that never moves. This is the term that is easy to lose.
        spread = optimal_total_spread(0.5, 0.0, 1.0, 0.5)
        assert spread == pytest.approx(4.0 * math.log(2.0))
        assert spread > 0.0

    def test_independent_of_inventory(self):
        # The published spread has no q in it at all; inventory acts only
        # through the reservation price. Guards against "fixing" the formula
        # by sneaking an inventory term into the spread.
        import inspect

        source = inspect.getsource(optimal_total_spread)
        assert "inventory" not in source.split('"""')[-1].replace("inventory_term", "")

    def test_decreasing_in_kappa(self):
        # Larger kappa = fill intensity decays faster with distance = quoting
        # far out is punished harder = tighter optimal spread.
        wide = optimal_total_spread(0.5, 1.0, 1.0, 0.5)
        tight = optimal_total_spread(0.5, 1.0, 1.0, 50.0)
        assert tight < wide

    @pytest.mark.parametrize(
        ("gamma", "kappa"), [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)]
    )
    def test_rejects_invalid_parameters(self, gamma, kappa):
        with pytest.raises(ValueError):
            optimal_total_spread(gamma, 1.0, 1.0, kappa)


class TestQuoter:
    def _quoter(self, **kw):
        params = ASParams(
            gamma=0.5, kappa=0.5, horizon_seconds=1.0, tick_size=0.01,
            min_half_spread_ticks=0.0, max_half_spread_ticks=1e9,
            inventory_scale=1.0, quote_size=1.0, **kw,
        )
        return AvellanedaStoikovQuoter(params)

    def test_flat_quote_is_symmetric_about_mid(self):
        q = self._quoter().quote(MarketState(mid=100.0, sigma=2.0, inventory=0.0))
        assert q.reservation_price == pytest.approx(100.0)
        # Half-spread = 4.7725887 / 2 = 2.3862944; outward rounding to 0.01.
        assert q.bid_price == pytest.approx(97.61, abs=0.011)
        assert q.ask_price == pytest.approx(102.39, abs=0.011)
        assert (q.ask_price - 100.0) == pytest.approx(100.0 - q.bid_price, abs=0.02)

    def test_long_inventory_shifts_quotes_down(self):
        flat = self._quoter().quote(MarketState(mid=100.0, sigma=2.0, inventory=0.0))
        long = self._quoter().quote(MarketState(mid=100.0, sigma=2.0, inventory=2.0))
        assert long.reservation_price == pytest.approx(96.0)
        assert long.bid_price < flat.bid_price
        assert long.ask_price < flat.ask_price
        # The spread itself is unchanged by inventory -- only the centre moves.
        assert long.total_spread == pytest.approx(flat.total_spread, abs=0.02)

    def test_short_inventory_shifts_quotes_up(self):
        flat = self._quoter().quote(MarketState(mid=100.0, sigma=2.0, inventory=0.0))
        short = self._quoter().quote(MarketState(mid=100.0, sigma=2.0, inventory=-2.0))
        assert short.reservation_price == pytest.approx(104.0)
        assert short.bid_price > flat.bid_price
        assert short.ask_price > flat.ask_price

    def test_inventory_normalisation_is_exact_reparametrisation(self):
        # q/q_max with gamma is identical to q with gamma/q_max. If this ever
        # stops holding, the "it is a change of units, not of model" claim in
        # the module docstring is false.
        raw = AvellanedaStoikovQuoter(
            ASParams(gamma=0.5, kappa=0.5, horizon_seconds=1.0, inventory_scale=1.0,
                     min_half_spread_ticks=0.0, max_half_spread_ticks=1e9)
        ).quote(MarketState(mid=100.0, sigma=2.0, inventory=0.5))
        scaled = AvellanedaStoikovQuoter(
            ASParams(gamma=0.5, kappa=0.5, horizon_seconds=1.0, inventory_scale=0.25,
                     min_half_spread_ticks=0.0, max_half_spread_ticks=1e9)
        ).quote(MarketState(mid=100.0, sigma=2.0, inventory=0.125))
        assert raw.reservation_price == pytest.approx(scaled.reservation_price)

    def test_never_crosses(self):
        q = self._quoter().quote(MarketState(mid=100.0, sigma=0.0, inventory=0.0))
        assert q.bid_price < q.ask_price

    def test_outward_rounding_never_improves_on_the_model(self):
        # The bid must never round UP and the ask never DOWN: that would mean
        # paying more (or selling cheaper) than the model's indifference price
        # purely because of tick granularity.
        params = ASParams(gamma=0.5, kappa=0.5, horizon_seconds=1.0, tick_size=0.5,
                          min_half_spread_ticks=0.0, max_half_spread_ticks=1e9,
                          inventory_scale=1.0)
        quoter = AvellanedaStoikovQuoter(params)
        for mid in (100.0, 100.13, 100.37, 100.49, 100.76):
            q = quoter.quote(MarketState(mid=mid, sigma=1.0, inventory=0.0))
            assert q.bid_price <= q.reservation_price - q.half_spread + 1e-9
            assert q.ask_price >= q.reservation_price + q.half_spread - 1e-9

    def test_floor_is_enforced(self):
        # kappa=1e6 makes the markup term negligible ((2/g)*ln(1+g/k) ~ 1e-6)
        # and sigma=0 kills the inventory term, so the unclamped half-spread is
        # far below the 100-tick floor.
        quoter = AvellanedaStoikovQuoter(
            ASParams(gamma=0.5, kappa=1e6, horizon_seconds=1.0, tick_size=0.01,
                     min_half_spread_ticks=100.0, max_half_spread_ticks=200.0,
                     inventory_scale=1.0)
        )
        q = quoter.quote(MarketState(mid=100.0, sigma=0.0, inventory=0.0))
        assert q.half_spread == pytest.approx(1.0)  # 100 ticks * 0.01

    def test_ceiling_is_enforced(self):
        # A volatility spike must not produce a nominally-valid quote that is
        # really just "not quoting".
        quoter = AvellanedaStoikovQuoter(
            ASParams(gamma=0.5, kappa=0.5, horizon_seconds=1.0, tick_size=0.01,
                     min_half_spread_ticks=0.0, max_half_spread_ticks=200.0,
                     inventory_scale=1.0)
        )
        q = quoter.quote(MarketState(mid=100.0, sigma=50.0, inventory=0.0))
        assert q.half_spread == pytest.approx(2.0)  # capped at 200 ticks * 0.01


class TestDriftInjection:
    def _quoter(self, **kw):
        return AvellanedaStoikovQuoter(
            ASParams(gamma=0.5, kappa=0.5, horizon_seconds=1.0, tick_size=0.01,
                     min_half_spread_ticks=0.0, max_half_spread_ticks=1e9,
                     inventory_scale=1.0, **kw)
        )

    def test_lambda_zero_disables_drift_entirely(self):
        q = self._quoter(drift_lambda=0.0).quote(
            MarketState(mid=100.0, sigma=1.0, inventory=0.0, drift=0.01, drift_confidence=1.0)
        )
        assert q.drift_adjustment == 0.0

    def test_hand_computed_adjustment(self):
        # lambda=2, drift=0.001, mid=100  ->  2 * 0.001 * 100 = 0.2
        q = self._quoter(drift_lambda=2.0, drift_max_ticks=1e9).quote(
            MarketState(mid=100.0, sigma=1.0, inventory=0.0, drift=0.001, drift_confidence=1.0)
        )
        assert q.drift_adjustment == pytest.approx(0.2)
        assert q.reservation_price == pytest.approx(100.2)

    def test_confidence_floor_suppresses_low_conviction(self):
        q = self._quoter(drift_lambda=2.0, drift_confidence_floor=0.6).quote(
            MarketState(mid=100.0, sigma=1.0, inventory=0.0, drift=0.001, drift_confidence=0.5)
        )
        assert q.drift_adjustment == 0.0

    def test_clamped_to_max_ticks(self):
        # A runaway model prediction must not be able to walk the quote away
        # from the market. 5 ticks * 0.01 = 0.05 is the hard ceiling here.
        q = self._quoter(drift_lambda=1000.0, drift_max_ticks=5.0).quote(
            MarketState(mid=100.0, sigma=1.0, inventory=0.0, drift=0.5, drift_confidence=1.0)
        )
        assert q.drift_adjustment == pytest.approx(0.05)

    def test_negative_drift_shifts_down(self):
        q = self._quoter(drift_lambda=2.0, drift_max_ticks=1e9).quote(
            MarketState(mid=100.0, sigma=1.0, inventory=0.0, drift=-0.001, drift_confidence=1.0)
        )
        assert q.drift_adjustment == pytest.approx(-0.2)
        assert q.reservation_price == pytest.approx(99.8)
