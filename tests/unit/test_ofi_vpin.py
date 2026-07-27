"""OFI and VPIN: hand-computed verification.

These two are the toxicity inputs, so a sign error here would silently invert
the entire adverse-selection defence -- and the system would still run, quote,
and produce a plausible-looking P&L. That failure mode is exactly why the
expected values below are derived by hand from the published definitions rather
than captured from the implementation.
"""

from __future__ import annotations

import pytest

from amme_core.ofi import BestQuote, OFICalculator, ofi_increment
from amme_core.vpin import ClassificationScheme, VPINCalculator, normal_cdf
from amme_shared.events import AggressorSide


class TestOFIIncrement:
    """e_n = 1{Pb_n >= Pb_{n-1}} qb_n - 1{Pb_n <= Pb_{n-1}} qb_{n-1}
             - 1{Pa_n <= Pa_{n-1}} qa_n + 1{Pa_n >= Pa_{n-1}} qa_{n-1}
    """

    def test_bid_price_up_adds_new_bid_size(self):
        # Bid 100@5 -> 101@7, ask unchanged 102@3 -> 102@3.
        #   bid: 101>=100 -> +7 ; 101<=100 false -> 0
        #   ask: 102<=102 -> -3 ; 102>=102       -> +3   (cancels)
        #   e = +7
        prev = BestQuote(100.0, 5.0, 102.0, 3.0)
        cur = BestQuote(101.0, 7.0, 102.0, 3.0)
        assert ofi_increment(prev, cur) == pytest.approx(7.0)

    def test_bid_price_down_removes_old_bid_size(self):
        # Bid 100@5 -> 99@7 : the old bid was pulled or eaten.
        #   bid: 99>=100 false -> 0 ; 99<=100 -> -5
        #   ask unchanged -> 0
        prev = BestQuote(100.0, 5.0, 102.0, 3.0)
        cur = BestQuote(99.0, 7.0, 102.0, 3.0)
        assert ofi_increment(prev, cur) == pytest.approx(-5.0)

    def test_bid_price_flat_gives_size_difference(self):
        # Both indicators fire when the price is unchanged: +q_n - q_{n-1}.
        # This is the case an if/elif chain gets wrong.
        prev = BestQuote(100.0, 5.0, 102.0, 3.0)
        cur = BestQuote(100.0, 9.0, 102.0, 3.0)
        assert ofi_increment(prev, cur) == pytest.approx(4.0)

    def test_ask_price_down_is_negative(self):
        # Ask 102@3 -> 101@4 : aggressive sell interest arrived.
        #   ask: 101<=102 -> -4 ; 101>=102 false -> 0
        prev = BestQuote(100.0, 5.0, 102.0, 3.0)
        cur = BestQuote(100.0, 5.0, 101.0, 4.0)
        assert ofi_increment(prev, cur) == pytest.approx(-4.0)

    def test_ask_price_up_is_positive(self):
        # Ask 102@3 -> 103@4 : the old ask was lifted or pulled -> buy pressure.
        prev = BestQuote(100.0, 5.0, 102.0, 3.0)
        cur = BestQuote(100.0, 5.0, 103.0, 4.0)
        assert ofi_increment(prev, cur) == pytest.approx(3.0)

    def test_combined_hand_example(self):
        # bid 100@5 -> 101@7 (up), ask 102@3 -> 102@4 (flat, grew by 1)
        #   bid: +7 ; ask: -4 + 3 = -1   ->  e = 6
        prev = BestQuote(100.0, 5.0, 102.0, 3.0)
        cur = BestQuote(101.0, 7.0, 102.0, 4.0)
        assert ofi_increment(prev, cur) == pytest.approx(6.0)

    def test_completely_unchanged_book_is_zero(self):
        q = BestQuote(100.0, 5.0, 102.0, 3.0)
        assert ofi_increment(q, q) == 0.0

    def test_sign_convention_buy_pressure_is_positive(self):
        # The whole point of OFI: positive means the mid is being pushed up.
        prev = BestQuote(100.0, 5.0, 102.0, 5.0)
        buy_pressure = BestQuote(101.0, 5.0, 103.0, 5.0)   # both sides moved up
        sell_pressure = BestQuote(99.0, 5.0, 101.0, 5.0)   # both moved down
        assert ofi_increment(prev, buy_pressure) > 0
        assert ofi_increment(prev, sell_pressure) < 0


class TestOFICalculator:
    def test_first_observation_yields_zero(self):
        calc = OFICalculator(window=5)
        assert calc.update(BestQuote(100.0, 5.0, 102.0, 3.0)) == 0.0

    def test_running_sum_over_window(self):
        calc = OFICalculator(window=10)
        calc.update(BestQuote(100.0, 5.0, 102.0, 3.0))
        calc.update(BestQuote(100.0, 8.0, 102.0, 3.0))  # +3
        assert calc.value == pytest.approx(3.0)
        calc.update(BestQuote(100.0, 10.0, 102.0, 3.0))  # +2
        assert calc.value == pytest.approx(5.0)

    def test_window_evicts_oldest(self):
        calc = OFICalculator(window=2)
        calc.update(BestQuote(100.0, 5.0, 102.0, 3.0))
        calc.update(BestQuote(100.0, 6.0, 102.0, 3.0))   # +1
        calc.update(BestQuote(100.0, 8.0, 102.0, 3.0))   # +2   -> sum 3
        assert calc.value == pytest.approx(3.0)
        calc.update(BestQuote(100.0, 9.0, 102.0, 3.0))   # +1, evicts the +1
        assert calc.value == pytest.approx(3.0)
        assert calc.warm

    def test_zscore_zero_when_degenerate(self):
        calc = OFICalculator(window=3, zscore_window=10)
        q = BestQuote(100.0, 5.0, 102.0, 3.0)
        for _ in range(5):
            calc.update(q)   # every increment is 0 -> no dispersion
        assert calc.zscore() == 0.0


class TestNormalCDF:
    @pytest.mark.parametrize(
        ("x", "expected"),
        [(0.0, 0.5), (1.0, 0.8413447461), (-1.0, 0.1586552539), (1.96, 0.9750021049)],
    )
    def test_known_values(self, x, expected):
        assert normal_cdf(x) == pytest.approx(expected, abs=1e-9)

    def test_symmetry(self):
        for x in (0.3, 1.1, 2.4):
            assert normal_cdf(x) + normal_cdf(-x) == pytest.approx(1.0)


class TestVPIN:
    """VPIN = sum |Vbuy - Vsell| / (n * V)"""

    def _calc(self, **kw):
        kw.setdefault("scheme", ClassificationScheme.TRUE_AGGRESSOR)
        return VPINCalculator(**kw)

    def test_maximally_toxic_is_one(self):
        # V=10, n=2. Bucket 1 all buy (imbalance 10), bucket 2 all sell (10).
        #   VPIN = (10 + 10) / (2 * 10) = 1.0
        calc = self._calc(bucket_volume=10.0, n_buckets=2)
        calc.update(100.0, 10.0, AggressorSide.BUY)
        calc.update(100.0, 10.0, AggressorSide.SELL)
        assert calc.value == pytest.approx(1.0)
        assert calc.warm

    def test_perfectly_balanced_is_zero(self):
        # One bucket, 5 buy + 5 sell -> |5-5| = 0 -> VPIN = 0.
        calc = self._calc(bucket_volume=10.0, n_buckets=1)
        calc.update(100.0, 5.0, AggressorSide.BUY)
        calc.update(100.0, 5.0, AggressorSide.SELL)
        assert calc.value == pytest.approx(0.0)

    def test_partial_imbalance_hand_computed(self):
        # V=10, n=1, 7 buy + 3 sell -> |7-3| = 4 -> VPIN = 4/10 = 0.4
        calc = self._calc(bucket_volume=10.0, n_buckets=1)
        calc.update(100.0, 7.0, AggressorSide.BUY)
        calc.update(100.0, 3.0, AggressorSide.SELL)
        assert calc.value == pytest.approx(0.4)

    def test_incomplete_bucket_contributes_nothing(self):
        calc = self._calc(bucket_volume=10.0, n_buckets=2)
        calc.update(100.0, 3.0, AggressorSide.BUY)
        assert calc.n_complete_buckets == 0
        assert calc.value == 0.0
        assert not calc.warm
        assert calc.bucket_fill_fraction == pytest.approx(0.3)

    def test_large_trade_splits_across_buckets(self):
        # A single 25-unit buy with V=10 fills two whole buckets and leaves 5
        # in the third. Assigning it wholesale to one bucket would make VPIN
        # depend on bucket size, which the estimator is not supposed to do.
        calc = self._calc(bucket_volume=10.0, n_buckets=2)
        completed = calc.update(100.0, 25.0, AggressorSide.BUY)
        assert len(completed) == 2
        assert all(b.buy_volume == pytest.approx(10.0) for b in completed)
        assert calc.value == pytest.approx(1.0)
        assert calc.bucket_fill_fraction == pytest.approx(0.5)

    def test_rolling_window_evicts(self):
        calc = self._calc(bucket_volume=10.0, n_buckets=1)
        calc.update(100.0, 10.0, AggressorSide.BUY)      # toxic bucket
        assert calc.value == pytest.approx(1.0)
        calc.update(100.0, 5.0, AggressorSide.BUY)
        calc.update(100.0, 5.0, AggressorSide.SELL)      # balanced, evicts
        assert calc.value == pytest.approx(0.0)

    def test_bounded_in_unit_interval(self):
        calc = self._calc(bucket_volume=1.0, n_buckets=5)
        for i in range(50):
            calc.update(100.0 + i * 0.1, 0.7,
                        AggressorSide.BUY if i % 3 else AggressorSide.SELL)
            assert 0.0 <= calc.value <= 1.0

    def test_tick_rule_fallback_when_aggressor_unknown(self):
        # No exchange flag -> uptick means buy-initiated.
        calc = self._calc(bucket_volume=10.0, n_buckets=1)
        calc.update(100.0, 5.0, AggressorSide.UNKNOWN)
        calc.update(101.0, 5.0, AggressorSide.UNKNOWN)   # uptick -> buy
        assert calc.value == pytest.approx(1.0)

    def test_bvc_splits_evenly_with_no_price_history(self):
        # BVC before sigma is estimable falls back to a 50/50 split, which
        # yields zero imbalance rather than a spurious toxicity reading.
        calc = VPINCalculator(bucket_volume=10.0, n_buckets=1,
                              scheme=ClassificationScheme.BVC)
        calc.update(100.0, 10.0, AggressorSide.BUY)
        assert calc.value == pytest.approx(0.0)

    def test_true_aggressor_beats_tick_rule_when_available(self):
        # Downtick, but the exchange says the BUYER was aggressive. The true
        # flag must win -- this is the Binance `m` field's whole value.
        calc = self._calc(bucket_volume=10.0, n_buckets=1)
        calc.update(100.0, 5.0, AggressorSide.BUY)
        calc.update(99.0, 5.0, AggressorSide.BUY)
        assert calc.value == pytest.approx(1.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_rejects_invalid_bucket_volume(self, bad):
        with pytest.raises(ValueError):
            VPINCalculator(bucket_volume=bad)
