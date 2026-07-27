"""Binance snapshot+diff resynchronisation -- the chaos tests (spec s.5.1).

The spec calls this out specifically: *"handle Binance's documented
snapshot+diff resync protocol precisely -- this is a common correctness bug,
test it explicitly"* and *"test this with chaos-testing -- kill the connection
mid-stream in a test and assert correct resync"*.

Because the synchroniser is a pure state machine with no socket in it, every
failure mode is directly constructible: sequence gaps, stale snapshots,
mid-stream disconnects, duplicate replays, and the boundary case where a single
diff straddles the snapshot. Doing this against a live socket would only test
whichever of these Binance happened to do today.
"""

from __future__ import annotations

import pytest

from amme_core.book import (
    BinanceBookSynchronizer,
    DepthDiff,
    DepthSnapshot,
    L2Book,
    ResyncReason,
    SyncState,
    levels_from_pairs,
)


def diff(u_first: int, u_final: int, bids=None, asks=None) -> DepthDiff:
    return DepthDiff(
        first_update_id=u_first,
        final_update_id=u_final,
        bids=bids or [],
        asks=asks or [],
    )


def snapshot(last_id: int, bid: float = 100.0, ask: float = 101.0) -> DepthSnapshot:
    return DepthSnapshot(
        last_update_id=last_id,
        bids=[(bid, 5.0), (bid - 1, 8.0)],
        asks=[(ask, 4.0), (ask + 1, 9.0)],
    )


@pytest.fixture
def sync() -> BinanceBookSynchronizer:
    return BinanceBookSynchronizer(L2Book("BTCUSDT"))


class TestL2Book:
    def test_zero_size_removes_level(self):
        book = L2Book("X")
        book.load_snapshot(snapshot(1))
        assert book.n_bid_levels == 2
        book.apply_diff(diff(2, 2, bids=[(100.0, 0.0)]))
        assert book.n_bid_levels == 1
        assert book.best_bid == 99.0

    def test_removing_absent_level_is_not_an_error(self):
        # Binance documents this as normal, not exceptional (rule 7).
        book = L2Book("X")
        book.load_snapshot(snapshot(1))
        book.apply_diff(diff(2, 2, bids=[(12.34, 0.0)]))
        assert book.n_bid_levels == 2

    def test_top_n_ordering(self):
        book = L2Book("X")
        book.load_snapshot(
            DepthSnapshot(1, bids=[(99.0, 1), (100.0, 2), (98.0, 3)],
                          asks=[(102.0, 1), (101.0, 2), (103.0, 3)])
        )
        bids, asks = book.top_n(3)
        assert [b.price for b in bids] == [100.0, 99.0, 98.0]  # descending
        assert [a.price for a in asks] == [101.0, 102.0, 103.0]  # ascending

    def test_mid_and_validity(self):
        book = L2Book("X")
        assert book.mid_price is None
        assert not book.is_valid()
        book.load_snapshot(snapshot(1, bid=100.0, ask=101.0))
        assert book.mid_price == pytest.approx(100.5)
        assert book.is_valid()

    def test_levels_from_pairs_parses_binance_strings(self):
        assert levels_from_pairs([["68000.10", "0.35"]]) == [(68000.10, 0.35)]


class TestHappyPath:
    def test_snapshot_then_contiguous_diffs(self, sync):
        sync.on_connected()
        assert sync.state is SyncState.BUFFERING
        assert sync.needs_snapshot

        assert sync.apply_snapshot(snapshot(100)).applied
        assert sync.is_synced

        for i in range(101, 106):
            result = sync.submit_diff(diff(i, i, bids=[(100.0, float(i))]))
            assert result.applied, f"diff {i} should apply"
        assert sync.book.last_update_id == 105
        assert sync.book.size_at("bid", 100.0) == 105.0
        assert sync.resync_count == 0

    def test_multi_id_diffs_are_contiguous_on_u_plus_one(self):
        # Binance batches: U..u can span many ids. The contract is
        # U == prev_u + 1, not u == prev_u + 1.
        s = BinanceBookSynchronizer(L2Book("X"))
        s.on_connected()
        s.apply_snapshot(snapshot(100))
        assert s.submit_diff(diff(101, 110)).applied
        assert s.submit_diff(diff(111, 120)).applied
        assert s.book.last_update_id == 120


class TestBufferReplay:
    def test_diffs_arriving_during_snapshot_fetch_are_replayed(self, sync):
        # The realistic race: the REST call is in flight while the socket keeps
        # delivering. Those diffs must be buffered and replayed, not dropped.
        sync.on_connected()
        for i in range(98, 104):
            result = sync.submit_diff(diff(i, i, bids=[(100.0, float(i))]))
            assert not result.applied  # buffered, not applied
        assert sync.buffered == 6

        result = sync.apply_snapshot(snapshot(100))
        assert result.applied
        assert sync.is_synced
        # Diffs with u <= 100 are dropped (rule 3); 101..103 replay.
        assert sync.book.last_update_id == 103
        assert sync.book.size_at("bid", 100.0) == 103.0

    def test_snapshot_boundary_straddling_diff_is_accepted(self, sync):
        # Rule 4 is an interval containment check: U <= lastUpdateId+1 <= u.
        # A single batched diff can legitimately span the boundary, and an
        # equality check would reject it and loop forever.
        sync.on_connected()
        sync.submit_diff(diff(95, 110, bids=[(100.0, 7.0)]))
        result = sync.apply_snapshot(snapshot(100))
        assert result.applied, result.detail
        assert sync.is_synced
        assert sync.book.size_at("bid", 100.0) == 7.0

    def test_snapshot_with_empty_buffer_still_syncs(self, sync):
        sync.on_connected()
        result = sync.apply_snapshot(snapshot(100))
        assert result.applied
        assert sync.is_synced

    def test_buffer_is_bounded(self, sync):
        sync.on_connected()
        sync.max_buffer = 10
        for i in range(200, 260):
            sync.submit_diff(diff(i, i))
        assert sync.buffered <= 10


class TestSequenceGap:
    def test_gap_triggers_resync(self, sync):
        sync.on_connected()
        sync.apply_snapshot(snapshot(100))
        assert sync.submit_diff(diff(101, 101)).applied

        result = sync.submit_diff(diff(105, 105))  # 102-104 missing
        assert not result.applied
        assert result.reason is ResyncReason.SEQUENCE_GAP
        assert sync.state is SyncState.RESYNC_REQUIRED
        assert sync.needs_snapshot
        assert sync.resync_count == 1

    def test_book_is_discarded_on_gap(self, sync):
        # Carrying a stale book forward is the dangerous failure: it is
        # silently wrong rather than loudly absent.
        sync.on_connected()
        sync.apply_snapshot(snapshot(100))
        assert sync.book.n_bid_levels > 0
        sync.submit_diff(diff(999, 999))
        assert sync.book.n_bid_levels == 0
        assert sync.book.last_update_id == 0

    def test_recovers_after_gap(self, sync):
        sync.on_connected()
        sync.apply_snapshot(snapshot(100))
        sync.submit_diff(diff(500, 500))          # gap -> resync required
        assert sync.needs_snapshot
        # The gap-causing diff is retained so a fresh snapshot can bridge to it.
        assert sync.apply_snapshot(snapshot(499)).applied
        assert sync.is_synced
        assert sync.book.last_update_id == 500

    def test_stale_duplicate_diff_is_ignored_without_resync(self, sync):
        sync.on_connected()
        sync.apply_snapshot(snapshot(100))
        sync.submit_diff(diff(101, 105))
        result = sync.submit_diff(diff(101, 105))  # replay
        assert not result.applied
        assert result.reason is None       # not an error
        assert sync.is_synced              # still healthy
        assert sync.resync_count == 0


class TestStaleSnapshot:
    def test_snapshot_older_than_buffer_is_rejected(self, sync):
        sync.on_connected()
        sync.submit_diff(diff(500, 510))
        result = sync.apply_snapshot(snapshot(100))  # far too old to bridge
        assert not result.applied
        assert result.reason is ResyncReason.STALE_SNAPSHOT
        assert not sync.is_synced
        # The buffer is kept so a retry with a fresher snapshot can succeed.
        assert sync.buffered == 1

    def test_retry_with_fresh_snapshot_succeeds(self, sync):
        sync.on_connected()
        sync.submit_diff(diff(500, 510))
        assert not sync.apply_snapshot(snapshot(100)).applied
        assert sync.apply_snapshot(snapshot(505)).applied
        assert sync.is_synced


class TestDisconnect:
    def test_disconnect_discards_book_and_requires_resync(self, sync):
        """The chaos test: kill the connection mid-stream."""
        sync.on_connected()
        sync.apply_snapshot(snapshot(100))
        for i in range(101, 111):
            sync.submit_diff(diff(i, i))
        assert sync.is_synced
        assert sync.book.n_bid_levels > 0

        sync.on_disconnected()

        assert sync.state is SyncState.DISCONNECTED
        assert sync.book.n_bid_levels == 0
        assert sync.book.last_update_id == 0
        assert sync.last_resync_reason is ResyncReason.DISCONNECT

    def test_diffs_while_disconnected_are_refused(self, sync):
        sync.on_connected()
        sync.apply_snapshot(snapshot(100))
        sync.on_disconnected()
        result = sync.submit_diff(diff(101, 101))
        assert not result.applied
        assert "not connected" in result.detail

    def test_snapshot_while_disconnected_is_refused(self, sync):
        sync.on_disconnected()
        assert not sync.apply_snapshot(snapshot(100)).applied

    def test_full_reconnect_cycle_restores_correctness(self, sync):
        sync.on_connected()
        sync.apply_snapshot(snapshot(100))
        sync.submit_diff(diff(101, 101, bids=[(100.0, 42.0)]))
        assert sync.book.size_at("bid", 100.0) == 42.0

        sync.on_disconnected()
        # Reconnect: buffer, snapshot, replay -- exactly the cold-start path.
        sync.on_connected()
        assert sync.state is SyncState.BUFFERING
        sync.submit_diff(diff(201, 201, bids=[(100.0, 7.0)]))
        assert sync.apply_snapshot(snapshot(200)).applied
        assert sync.is_synced
        assert sync.book.size_at("bid", 100.0) == 7.0
        assert sync.book.last_update_id == 201

    def test_repeated_disconnects_do_not_corrupt_state(self, sync):
        for cycle in range(5):
            sync.on_connected()
            base = 1000 * (cycle + 1)
            assert sync.apply_snapshot(snapshot(base)).applied
            for i in range(base + 1, base + 6):
                assert sync.submit_diff(diff(i, i)).applied
            assert sync.book.last_update_id == base + 5
            sync.on_disconnected()
        assert sync.state is SyncState.DISCONNECTED


class TestCrossedBook:
    def test_crossed_book_forces_resync(self, sync):
        sync.on_connected()
        sync.apply_snapshot(snapshot(100, bid=100.0, ask=101.0))
        # Push a bid above the best ask -- physically impossible, so our local
        # state must be wrong; the honest response is to resync, not publish.
        result = sync.submit_diff(diff(101, 101, bids=[(105.0, 1.0)]))
        assert not result.applied
        assert result.reason is ResyncReason.CROSSED_BOOK
        assert sync.needs_snapshot
        assert sync.book.n_bid_levels == 0
