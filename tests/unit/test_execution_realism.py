"""Execution-realism models: own-order impact, latency tails, replay merging.

Each closes a documented limitation. They share a design rule: the *default*
reproduces the previous behaviour exactly, so existing calibrations and results
stay comparable, and the added realism is opt-in and measurable rather than a
silent change to every number in the repository.
"""

from __future__ import annotations

import numpy as np
import pytest

from amme_core.matching import MatchingEngine, MatchingParams, own_impact_from_share
from amme_shared.events import AggressorSide, OrderBookUpdate, PriceLevel, Side, TradeTick


class TestOwnOrderImpact:
    """Our orders were never in the recorded book.

    A fill we receive is one the real queue did not get. Treating ourselves as
    pure added depth is correct only while our size is negligible against the
    touch; at size it makes the backtest optimistic in a way that grows with
    the very parameter a user is most likely to increase.
    """

    def test_zero_impact_reproduces_previous_behaviour(self):
        """The default must not move any existing result."""
        engine = MatchingEngine(MatchingParams(latency_ns=0, own_impact=0.0))
        order = engine.place(Side.BID, 100.0, 1.0, level_size=10.0, ts_ns=0)
        assert order.queue_ahead == pytest.approx(10.0)

    def test_full_impact_clears_the_queue(self):
        engine = MatchingEngine(MatchingParams(latency_ns=0, own_impact=1.0))
        order = engine.place(Side.BID, 100.0, 1.0, level_size=10.0, ts_ns=0)
        assert order.queue_ahead == pytest.approx(0.0)

    def test_partial_impact_scales_the_queue(self):
        engine = MatchingEngine(MatchingParams(latency_ns=0, own_impact=0.25))
        order = engine.place(Side.BID, 100.0, 1.0, level_size=10.0, ts_ns=0)
        assert order.queue_ahead == pytest.approx(7.5)

    @pytest.mark.parametrize("impact", [0.0, 0.3, 0.7, 1.0])
    def test_impact_only_ever_shortens_the_queue(self, impact):
        """Displacement must never invent liquidity ahead of us."""
        engine = MatchingEngine(MatchingParams(latency_ns=0, own_impact=impact))
        order = engine.place(Side.BID, 100.0, 1.0, level_size=10.0, ts_ns=0)
        assert 0.0 <= order.queue_ahead <= 10.0

    @pytest.mark.parametrize(
        ("quote", "level", "expected"),
        [
            (0.001, 10.0, 0.0001),
            (1.0, 1.0, 0.5),
            (10.0, 0.0, 1.0),
        ],
    )
    def test_share_derivation(self, quote, level, expected):
        assert own_impact_from_share(quote, level) == pytest.approx(expected, abs=1e-3)

    def test_share_is_monotone_in_quote_size(self):
        """The assumption must degrade as size grows, not stay flat."""
        shares = [own_impact_from_share(q, 5.0) for q in (0.01, 0.1, 1.0, 5.0, 50.0)]
        assert shares == sorted(shares)
        assert shares[0] < 0.01
        assert shares[-1] < 1.0 + 1e-9


class TestLatencyDistribution:
    """A fixed delay removes exactly the adverse cases.

    Real network latency is fat-tailed, and the slow round trips are the ones
    during which the market moves against a stale quote. Modelling only the
    median is therefore not conservative, it is optimistic.
    """

    def test_sigma_zero_is_constant(self):
        engine = MatchingEngine(MatchingParams(latency_ns=5_000_000, latency_sigma=0.0))
        assert {engine._sample_latency_ns() for _ in range(50)} == {5_000_000}

    def test_tail_is_heavier_than_the_median(self):
        engine = MatchingEngine(
            MatchingParams(latency_ns=5_000_000, latency_sigma=0.5), seed=7
        )
        draws = np.array([engine._sample_latency_ns() for _ in range(4000)])
        p50, p99 = np.percentile(draws, [50, 99])
        assert p50 == pytest.approx(5_000_000, rel=0.1), "median must stay put"
        assert p99 > 2.0 * p50, "a log-normal tail must exceed 2x the median"

    def test_capped(self):
        engine = MatchingEngine(
            MatchingParams(
                latency_ns=5_000_000, latency_sigma=3.0, latency_max_ns=50_000_000
            ),
            seed=3,
        )
        assert max(engine._sample_latency_ns() for _ in range(2000)) <= 50_000_000

    def test_stochastic_latency_is_still_reproducible(self):
        """Modelling the tail must not cost byte-for-byte replay (spec s.6.1)."""

        def draws(seed: int) -> list[int]:
            engine = MatchingEngine(
                MatchingParams(latency_ns=5_000_000, latency_sigma=0.6), seed=seed
            )
            return [engine._sample_latency_ns() for _ in range(200)]

        assert draws(11) == draws(11)
        assert draws(11) != draws(12)

    def test_latency_delays_eligibility_end_to_end(self):
        engine = MatchingEngine(MatchingParams(latency_ns=10_000_000, latency_sigma=0.0))
        engine.place(Side.BID, 100.0, 1.0, level_size=0.0, ts_ns=0)
        assert engine.on_trade(100.0, 1.0, AggressorSide.SELL, ts_ns=5_000_000) == []
        assert len(engine.on_trade(100.0, 1.0, AggressorSide.SELL, ts_ns=11_000_000)) == 1


def _book(ts: int, uid: int) -> OrderBookUpdate:
    return OrderBookUpdate(
        symbol="X",
        ts_event=ts,
        ts_ingest=ts,
        last_update_id=uid,
        bids=[PriceLevel(price=100.0, size=1.0)],
        asks=[PriceLevel(price=101.0, size=1.0)],
    )


def _trade(ts: int, tid: int) -> TradeTick:
    return TradeTick(
        symbol="X", ts_event=ts, ts_ingest=ts, price=100.5, size=1.0, trade_id=tid
    )


class TestReplayCausalMerge:
    """Replaying recorded data must not leak the future.

    Books and trades arrive on separate streams and live in separate tables.
    Attaching a trade to the book it executed *against* would let the strategy
    see the consequence of flow it should not yet know about -- a backtest that
    is both profitable and worthless.
    """

    def test_trades_attach_to_the_book_at_or_after_them(self):
        from backtest_svc.replay import merge_causally

        books = [_book(100, 1), _book(200, 2), _book(300, 3)]
        trades = [_trade(150, 1), _trade(250, 2)]
        merged = list(merge_causally(books, trades))
        assert [m[0].ts_event for m in merged] == [100, 200, 300]
        assert [len(m[1]) for m in merged] == [0, 1, 1]
        assert merged[1][1][0].trade_id == 1

    def test_trade_on_a_book_timestamp_belongs_to_that_book(self):
        from backtest_svc.replay import merge_causally

        merged = list(merge_causally([_book(100, 1), _book(200, 2)], [_trade(200, 1)]))
        assert len(merged[0][1]) == 0
        assert len(merged[1][1]) == 1, "a trade at t belongs to book t, not t+1"

    def test_trades_outside_the_window_are_dropped_not_bunched(self):
        """Bunching stray flow onto the nearest book would fabricate a
        toxicity spike that never happened."""
        from backtest_svc.replay import merge_causally

        books = [_book(100, 1), _book(200, 2)]
        trades = [_trade(10, 1), _trade(150, 2), _trade(900, 3)]
        merged = list(merge_causally(books, trades))
        assert sum(len(m[1]) for m in merged) == 1

    def test_unsorted_input_is_ordered(self):
        from backtest_svc.replay import merge_causally

        books = [_book(300, 3), _book(100, 1), _book(200, 2)]
        merged = list(merge_causally(books, [_trade(250, 1), _trade(150, 2)]))
        assert [m[0].ts_event for m in merged] == [100, 200, 300]

    def test_empty_inputs(self):
        from backtest_svc.replay import merge_causally

        assert list(merge_causally([], [_trade(1, 1)])) == []
        assert [len(m[1]) for m in merge_causally([_book(1, 1)], [])] == [0]

    def test_every_trade_used_at_most_once(self):
        from backtest_svc.replay import merge_causally

        books = [_book(t, t) for t in range(100, 1001, 100)]
        trades = [_trade(t, t) for t in range(150, 951, 50)]
        merged = list(merge_causally(books, trades))
        ids = [t.trade_id for _, batch in merged for t in batch]
        assert len(ids) == len(set(ids)), "a trade was double-counted"

    def test_quality_report_flags_gaps(self):
        from backtest_svc.replay import replay_quality

        window = [(_book(0, 1), []), (_book(10_000_000_000, 2), [])]
        quality = replay_quality(window)
        assert quality["largest_gap_seconds"] == pytest.approx(10.0)
        assert quality["crossed_books"] == 0.0
