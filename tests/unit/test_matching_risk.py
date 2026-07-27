"""Matching engine (queue position) and risk engine.

The queue-position tests matter more than they look: a naive fill model is the
single most common reason a market-making backtest is optimistic, and the whole
credibility of ``docs/RESULTS.md`` rests on these being right.
"""

from __future__ import annotations

import pytest

from amme_core.matching import (
    MatchingEngine,
    MatchingParams,
    OrderStatus,
    Portfolio,
)
from amme_core.risk import (
    BreakerTrigger,
    ProposedQuote,
    RiskEngine,
    RiskInputs,
    RiskParams,
    RiskVerdict,
)
from amme_shared.events import AggressorSide, Side


@pytest.fixture
def engine() -> MatchingEngine:
    return MatchingEngine(MatchingParams(maker_fee_bps=0.0, tick_size=0.01, latency_ns=0))


class TestPortfolioAccounting:
    def test_buy_then_sell_realizes_spread(self):
        p = Portfolio()
        p.apply_fill(Side.BID, 100.0, 1.0)
        assert p.inventory == 1.0
        assert p.cash == pytest.approx(-100.0)
        assert p.avg_price == pytest.approx(100.0)
        p.apply_fill(Side.ASK, 101.0, 1.0)
        assert p.inventory == 0.0
        assert p.realized_pnl == pytest.approx(1.0)
        assert p.cash == pytest.approx(1.0)

    def test_pnl_identity_holds(self):
        # cash + q*mid must equal realized + unrealized. Deriving the same
        # number two ways and checking them is what catches accounting drift.
        p = Portfolio()
        p.apply_fill(Side.BID, 100.0, 2.0)
        p.apply_fill(Side.ASK, 102.0, 1.0)
        for mid in (95.0, 100.0, 110.0):
            assert p.check_identity(mid), f"identity broken at mid={mid}"
            assert p.total_pnl(mid) == pytest.approx(p.realized_pnl + p.unrealized_pnl(mid))

    def test_average_cost_on_scaling_in(self):
        p = Portfolio()
        p.apply_fill(Side.BID, 100.0, 1.0)
        p.apply_fill(Side.BID, 102.0, 1.0)
        assert p.avg_price == pytest.approx(101.0)
        assert p.unrealized_pnl(103.0) == pytest.approx(2.0 * 2.0)

    def test_short_position_accounting(self):
        p = Portfolio()
        p.apply_fill(Side.ASK, 100.0, 1.0)
        assert p.inventory == -1.0
        assert p.cash == pytest.approx(100.0)
        # Covering lower is a profit for a short.
        p.apply_fill(Side.BID, 98.0, 1.0)
        assert p.realized_pnl == pytest.approx(2.0)

    def test_flip_through_zero(self):
        p = Portfolio()
        p.apply_fill(Side.BID, 100.0, 1.0)
        p.apply_fill(Side.ASK, 105.0, 3.0)   # close 1 long (+5), open 2 short
        assert p.inventory == pytest.approx(-2.0)
        assert p.realized_pnl == pytest.approx(5.0)
        assert p.avg_price == pytest.approx(105.0)
        assert p.check_identity(105.0)

    def test_fees_are_folded_into_realized(self):
        p = Portfolio()
        p.apply_fill(Side.BID, 100.0, 1.0, fee=0.5)
        assert p.fees_paid == pytest.approx(0.5)
        assert p.realized_pnl == pytest.approx(-0.5)
        assert p.check_identity(100.0)


class TestQueuePosition:
    def test_order_behind_queue_does_not_fill(self, engine):
        # 5.0 resting ahead of us; a 2.0 trade cannot reach us.
        engine.place(Side.BID, 100.0, 1.0, level_size=5.0, ts_ns=0)
        fills = engine.on_trade(100.0, 2.0, AggressorSide.SELL, ts_ns=1)
        assert fills == []
        assert engine.orders["bid-1"].queue_ahead == pytest.approx(3.0)

    def test_fills_once_queue_is_consumed(self, engine):
        engine.place(Side.BID, 100.0, 1.0, level_size=5.0, ts_ns=0)
        engine.on_trade(100.0, 5.0, AggressorSide.SELL, ts_ns=1)   # clears queue exactly
        fills = engine.on_trade(100.0, 1.0, AggressorSide.SELL, ts_ns=2)
        assert len(fills) == 1
        assert fills[0].size == pytest.approx(1.0)
        assert engine.portfolio.inventory == pytest.approx(1.0)

    def test_front_of_queue_fills_immediately(self, engine):
        engine.place(Side.BID, 100.0, 1.0, level_size=0.0, ts_ns=0)
        fills = engine.on_trade(100.0, 1.0, AggressorSide.SELL, ts_ns=1)
        assert len(fills) == 1

    def test_partial_fill(self, engine):
        engine.place(Side.BID, 100.0, 2.0, level_size=0.0, ts_ns=0)
        fills = engine.on_trade(100.0, 0.5, AggressorSide.SELL, ts_ns=1)
        assert fills[0].size == pytest.approx(0.5)
        order = engine.orders["bid-1"]
        assert order.status is OrderStatus.PARTIAL
        assert order.remaining == pytest.approx(1.5)

    def test_wrong_aggressor_side_never_fills(self, engine):
        # A BUY aggressor lifts asks; it cannot fill our bid.
        engine.place(Side.BID, 100.0, 1.0, level_size=0.0, ts_ns=0)
        assert engine.on_trade(100.0, 5.0, AggressorSide.BUY, ts_ns=1) == []

    def test_trade_above_our_bid_does_not_fill_us(self, engine):
        engine.place(Side.BID, 100.0, 1.0, level_size=0.0, ts_ns=0)
        assert engine.on_trade(100.5, 5.0, AggressorSide.SELL, ts_ns=1) == []

    def test_walk_through_skips_the_queue(self, engine):
        # A print strictly BELOW our bid means the aggressor consumed every
        # level in between, so queue position no longer protects us: we fill
        # despite 99 units resting ahead.
        engine.place(Side.BID, 100.0, 1.0, level_size=99.0, ts_ns=0)
        fills = engine.on_trade(99.0, 1.0, AggressorSide.SELL, ts_ns=1)
        assert len(fills) == 1
        assert fills[0].size == pytest.approx(1.0)

    def test_walk_through_is_still_bounded_by_trade_size(self, engine):
        """Skipping the queue does not mean unlimited size.

        This assertion previously read ``== 1.0``, encoding the over-fill bug
        as intended behaviour: a 0.1-unit aggressor was credited with filling a
        full 1.0-unit order. An aggressor can only ever hand over the size it
        actually brought, however deep it swept.
        """
        engine.place(Side.BID, 100.0, 1.0, level_size=99.0, ts_ns=0)
        fills = engine.on_trade(99.0, 0.1, AggressorSide.SELL, ts_ns=1)
        assert len(fills) == 1
        assert fills[0].size == pytest.approx(0.1)
        assert engine.active_order(Side.BID).remaining == pytest.approx(0.9)

    def test_cancellations_ahead_improve_queue_position(self, engine):
        # Level shrinks 10 -> 5 with no trade: those were cancellations, and
        # under a uniform-position assumption half of them sat ahead of us.
        engine.place(Side.BID, 100.0, 1.0, level_size=10.0, ts_ns=0)
        assert engine.orders["bid-1"].queue_ahead == pytest.approx(10.0)
        engine.on_book_update({100.0: 5.0}, {101.0: 5.0})
        assert engine.orders["bid-1"].queue_ahead == pytest.approx(5.0)

    def test_additions_join_behind_us(self, engine):
        engine.place(Side.BID, 100.0, 1.0, level_size=10.0, ts_ns=0)
        engine.on_book_update({100.0: 25.0}, {101.0: 5.0})
        assert engine.orders["bid-1"].queue_ahead == pytest.approx(10.0)

    def test_ask_side_symmetry(self, engine):
        engine.place(Side.ASK, 101.0, 1.0, level_size=2.0, ts_ns=0)
        assert engine.on_trade(101.0, 1.0, AggressorSide.BUY, ts_ns=1) == []
        engine.on_trade(101.0, 1.0, AggressorSide.BUY, ts_ns=2)
        fills = engine.on_trade(101.0, 1.0, AggressorSide.BUY, ts_ns=3)
        assert len(fills) == 1
        assert engine.portfolio.inventory == pytest.approx(-1.0)

    def test_latency_delays_eligibility(self):
        eng = MatchingEngine(MatchingParams(latency_ns=5_000_000))
        eng.place(Side.BID, 100.0, 1.0, level_size=0.0, ts_ns=0)
        assert eng.on_trade(100.0, 1.0, AggressorSide.SELL, ts_ns=1_000_000) == []
        assert len(eng.on_trade(100.0, 1.0, AggressorSide.SELL, ts_ns=6_000_000)) == 1


class TestQueuePriorityOnRequote:
    def test_same_price_keeps_priority(self, engine):
        engine.replace_quotes(100.0, 1.0, 101.0, 1.0, 8.0, 8.0, ts_ns=0)
        engine.on_trade(100.0, 4.0, AggressorSide.SELL, ts_ns=1)
        before = engine.active_order(Side.BID).queue_ahead
        assert before == pytest.approx(4.0)
        engine.replace_quotes(100.0, 1.0, 101.0, 1.0, 8.0, 8.0, ts_ns=2)
        assert engine.active_order(Side.BID).queue_ahead == pytest.approx(4.0)

    def test_price_change_loses_priority(self, engine):
        engine.replace_quotes(100.0, 1.0, 101.0, 1.0, 8.0, 8.0, ts_ns=0)
        engine.on_trade(100.0, 6.0, AggressorSide.SELL, ts_ns=1)
        assert engine.active_order(Side.BID).queue_ahead == pytest.approx(2.0)
        # Amending the price sends us to the back of the new level.
        engine.replace_quotes(99.99, 1.0, 101.0, 1.0, 12.0, 8.0, ts_ns=2)
        assert engine.active_order(Side.BID).queue_ahead == pytest.approx(12.0)

    def test_none_price_cancels_side(self, engine):
        engine.replace_quotes(100.0, 1.0, 101.0, 1.0, 1.0, 1.0, ts_ns=0)
        engine.replace_quotes(None, 0.0, 101.0, 1.0, 0.0, 1.0, ts_ns=1)
        assert engine.active_order(Side.BID) is None
        assert engine.active_order(Side.ASK) is not None

    def test_returns_number_of_new_placements(self, engine):
        assert engine.replace_quotes(100.0, 1.0, 101.0, 1.0, 1.0, 1.0, ts_ns=0) == 2
        # Unchanged quote rests as-is: no new order, so nothing to count.
        assert engine.replace_quotes(100.0, 1.0, 101.0, 1.0, 1.0, 1.0, ts_ns=1) == 0
        assert engine.replace_quotes(99.0, 1.0, 101.0, 1.0, 1.0, 1.0, ts_ns=2) == 1


class TestRiskEngine:
    def _inputs(self, inventory=0.0, **kw):
        kw.setdefault("mid_price", 100.0)
        return RiskInputs(inventory=inventory, **kw)

    def _quote(self):
        return ProposedQuote(99.0, 101.0, 1.0, 1.0, half_spread=1.0)

    def test_flat_inventory_approves(self):
        eng = RiskEngine(RiskParams(q_max=10.0, breaker_drawdown=1e9))
        out = eng.evaluate(ProposedQuote(99.0, 101.0, 1.0, 1.0), self._inputs())
        assert out.verdict is RiskVerdict.APPROVE
        assert out.bid_size == pytest.approx(1.0)

    def test_skew_is_zero_below_soft_limit(self):
        eng = RiskEngine(RiskParams(q_max=10.0, q_soft_frac=0.6))
        assert eng.skew_for(3.0, 1.0) == 0.0

    def test_skew_leans_against_inventory(self):
        eng = RiskEngine(RiskParams(q_max=10.0, q_soft_frac=0.6, skew_beta=1.0))
        # Long -> skew DOWN so our bid is less attractive and ask more so.
        assert eng.skew_for(9.0, 1.0) < 0
        assert eng.skew_for(-9.0, 1.0) > 0

    def test_skew_grows_quadratically(self):
        eng = RiskEngine(RiskParams(q_max=10.0, q_soft_frac=0.6, skew_beta=1.0))
        # excess at q=7 is 0.1, at q=8 is 0.2 -> quadratic gives a 4x ratio.
        assert abs(eng.skew_for(8.0, 1.0)) == pytest.approx(4.0 * abs(eng.skew_for(7.0, 1.0)))

    def test_one_sided_unwind_when_long(self):
        eng = RiskEngine(RiskParams(q_max=10.0, unwind_frac=0.9, breaker_drawdown=1e9))
        out = eng.evaluate(self._quote(), self._inputs(inventory=9.5))
        assert out.verdict is RiskVerdict.AMEND
        assert out.bid_size == 0.0      # stop adding risk
        assert out.ask_size > 0.0       # keep reducing
        assert "unwind" in out.reason

    def test_one_sided_unwind_when_short(self):
        eng = RiskEngine(RiskParams(q_max=10.0, unwind_frac=0.9, breaker_drawdown=1e9))
        out = eng.evaluate(self._quote(), self._inputs(inventory=-9.5))
        assert out.ask_size == 0.0
        assert out.bid_size > 0.0

    def test_hard_limit_sizes_down_the_bid(self):
        eng = RiskEngine(RiskParams(q_max=10.0, unwind_frac=1.5, breaker_drawdown=1e9))
        out = eng.evaluate(
            ProposedQuote(99.0, 101.0, 5.0, 5.0, 1.0), self._inputs(inventory=9.7)
        )
        assert out.bid_size == pytest.approx(0.3)   # never exceed q_max

    def test_notional_cap(self):
        eng = RiskEngine(
            RiskParams(q_max=1e9, max_position_value=1000.0, breaker_drawdown=1e9,
                       unwind_frac=1e9)
        )
        out = eng.evaluate(
            ProposedQuote(99.0, 101.0, 100.0, 100.0, 1.0),
            self._inputs(inventory=0.0, mid_price=100.0),
        )
        assert out.bid_size == pytest.approx(10.0)  # 1000 / 100

    def test_disabled_risk_passes_through(self):
        eng = RiskEngine(RiskParams(enabled=False))
        out = eng.evaluate(self._quote(), self._inputs(inventory=1e6))
        assert out.verdict is RiskVerdict.APPROVE


class TestCircuitBreaker:
    def test_vpin_spike_trips(self):
        eng = RiskEngine(RiskParams(breaker_vpin=0.8, breaker_drawdown=1e9))
        out = eng.evaluate(
            ProposedQuote(99.0, 101.0, 1.0, 1.0), RiskInputs(0.0, 100.0, vpin=0.9)
        )
        assert out.verdict is RiskVerdict.HALT
        assert eng.breaker.trigger is BreakerTrigger.TOXICITY

    def test_vol_spike_trips(self):
        eng = RiskEngine(RiskParams(breaker_vol_multiple=4.0, breaker_drawdown=1e9))
        out = eng.evaluate(
            ProposedQuote(99.0, 101.0, 1.0, 1.0),
            RiskInputs(0.0, 100.0, realized_vol=5.0, baseline_vol=1.0),
        )
        assert out.verdict is RiskVerdict.HALT
        assert eng.breaker.trigger is BreakerTrigger.VOLATILITY

    def test_vol_test_disarmed_without_baseline(self):
        # baseline 0 would make every reading an infinite multiple.
        eng = RiskEngine(RiskParams(breaker_vol_multiple=4.0, breaker_drawdown=1e9))
        out = eng.evaluate(
            ProposedQuote(99.0, 101.0, 1.0, 1.0),
            RiskInputs(0.0, 100.0, realized_vol=999.0, baseline_vol=0.0),
        )
        assert out.verdict is not RiskVerdict.HALT

    def test_drawdown_trips_from_high_water_mark(self):
        eng = RiskEngine(RiskParams(breaker_drawdown=50.0))
        q = ProposedQuote(99.0, 101.0, 1.0, 1.0)
        eng.evaluate(q, RiskInputs(0.0, 100.0, total_pnl=100.0, ts_ns=0))
        out = eng.evaluate(q, RiskInputs(0.0, 100.0, total_pnl=40.0, ts_ns=1_000_000_000))
        assert out.verdict is RiskVerdict.HALT
        assert eng.breaker.trigger is BreakerTrigger.DRAWDOWN

    def test_breaker_latches_through_cooldown(self):
        eng = RiskEngine(RiskParams(breaker_vpin=0.8, breaker_cooldown_seconds=30.0,
                                    breaker_drawdown=1e9))
        q = ProposedQuote(99.0, 101.0, 1.0, 1.0)
        eng.evaluate(q, RiskInputs(0.0, 100.0, vpin=0.9, ts_ns=0))
        # Condition cleared, but the cooldown has not elapsed.
        out = eng.evaluate(q, RiskInputs(0.0, 100.0, vpin=0.0, ts_ns=10_000_000_000))
        assert out.verdict is RiskVerdict.HALT

    def test_breaker_resets_only_when_cooled_and_clear(self):
        eng = RiskEngine(RiskParams(breaker_vpin=0.8, breaker_cooldown_seconds=30.0,
                                    breaker_drawdown=1e9))
        q = ProposedQuote(99.0, 101.0, 1.0, 1.0)
        eng.evaluate(q, RiskInputs(0.0, 100.0, vpin=0.9, ts_ns=0))
        out = eng.evaluate(q, RiskInputs(0.0, 100.0, vpin=0.0, ts_ns=40_000_000_000))
        assert out.verdict is not RiskVerdict.HALT
        assert not eng.breaker.tripped

    def test_still_tripped_after_cooldown_if_condition_persists(self):
        # A breaker that un-trips on the timer alone flaps in and out of a real
        # volatility event, which is the worst of both behaviours.
        eng = RiskEngine(RiskParams(breaker_vpin=0.8, breaker_cooldown_seconds=30.0,
                                    breaker_drawdown=1e9))
        q = ProposedQuote(99.0, 101.0, 1.0, 1.0)
        eng.evaluate(q, RiskInputs(0.0, 100.0, vpin=0.9, ts_ns=0))
        out = eng.evaluate(q, RiskInputs(0.0, 100.0, vpin=0.95, ts_ns=40_000_000_000))
        assert out.verdict is RiskVerdict.HALT

    def test_trip_count_increments_once_per_event(self):
        eng = RiskEngine(RiskParams(breaker_vpin=0.8, breaker_drawdown=1e9))
        q = ProposedQuote(99.0, 101.0, 1.0, 1.0)
        for i in range(5):
            eng.evaluate(q, RiskInputs(0.0, 100.0, vpin=0.9, ts_ns=i))
        assert eng.breaker.trip_count == 1
