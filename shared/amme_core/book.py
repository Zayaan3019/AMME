"""L2 order book state and Binance's snapshot+diff resynchronisation protocol.

The resync state machine below is the single most bug-prone part of any
exchange connector, so it is implemented here as a *pure* state machine with no
network calls at all -- :class:`BinanceBookSynchronizer` consumes already-parsed
diffs and tells the caller when it needs a snapshot. That inversion is what lets
``tests/unit/test_book_resync.py`` drive every branch (gap, stale, premature,
mid-stream disconnect) deterministically without a socket.

Binance's documented procedure (spot API, /api/v3/depth + <symbol>@depth):

  1. Open the stream and buffer events.
  2. Fetch a REST snapshot; note its ``lastUpdateId``.
  3. Drop any buffered event with ``u <= lastUpdateId``.
  4. The first event applied must satisfy ``U <= lastUpdateId + 1 <= u``.
  5. Every subsequent event must satisfy ``U == prev_u + 1``; otherwise the
     stream has a gap and the book must be resynced from step 2.
  6. Quantities are absolute, not deltas. Quantity 0 removes the level.
  7. Deleting a level that is not present locally is normal, not an error.

Step 4 is the one implementations usually get wrong -- it is an interval
containment check, not an equality check, because a single diff can span the
snapshot boundary.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum, auto

from amme_shared.events import PriceLevel

# Binance sends prices/sizes as decimal strings. Level values below this are
# treated as "level removed" -- guards against float noise like 1e-17.
_ZERO_EPS = 1e-12


@dataclass(slots=True)
class DepthDiff:
    """One `<symbol>@depth` message, already parsed out of JSON.

    ``first_update_id``/``final_update_id`` are Binance's ``U``/``u``.
    """

    first_update_id: int
    final_update_id: int
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    event_time_ms: int = 0


@dataclass(slots=True)
class DepthSnapshot:
    """One `/api/v3/depth` REST response."""

    last_update_id: int
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]


class L2Book:
    """Price-keyed L2 book.

    Storage is two plain dicts. Top-N extraction sorts on demand: at 10 updates
    per second against a 1000-level book that is ~50us of sorting per update,
    which is three orders of magnitude inside our budget and avoids taking on a
    sorted-container dependency for no measurable gain. If depth ever grows to
    where this matters, the fix is an incrementally-maintained heap behind the
    same interface -- ``top_n`` is the seam.
    """

    __slots__ = ("_asks", "_bids", "last_update_id", "symbol")

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self.last_update_id: int = 0

    def clear(self) -> None:
        self._bids.clear()
        self._asks.clear()
        self.last_update_id = 0

    @property
    def n_bid_levels(self) -> int:
        return len(self._bids)

    @property
    def n_ask_levels(self) -> int:
        return len(self._asks)

    def apply_levels(self, side: dict[float, float], levels: Iterable[tuple[float, float]]) -> None:
        for price, size in levels:
            if size <= _ZERO_EPS:
                side.pop(price, None)  # deleting an absent level is normal (rule 7)
            else:
                side[price] = size

    def apply_diff(self, diff: DepthDiff) -> None:
        self.apply_levels(self._bids, diff.bids)
        self.apply_levels(self._asks, diff.asks)
        self.last_update_id = diff.final_update_id

    def load_snapshot(self, snapshot: DepthSnapshot) -> None:
        self._bids = {p: s for p, s in snapshot.bids if s > _ZERO_EPS}
        self._asks = {p: s for p, s in snapshot.asks if s > _ZERO_EPS}
        self.last_update_id = snapshot.last_update_id

    def top_n(self, n: int) -> tuple[list[PriceLevel], list[PriceLevel]]:
        """Best ``n`` levels per side: bids descending, asks ascending."""
        bid_prices = sorted(self._bids, reverse=True)[:n]
        ask_prices = sorted(self._asks)[:n]
        return (
            [PriceLevel(price=p, size=self._bids[p]) for p in bid_prices],
            [PriceLevel(price=p, size=self._asks[p]) for p in ask_prices],
        )

    @property
    def best_bid(self) -> float | None:
        return max(self._bids) if self._bids else None

    @property
    def best_ask(self) -> float | None:
        return min(self._asks) if self._asks else None

    @property
    def mid_price(self) -> float | None:
        bid, ask = self.best_bid, self.best_ask
        return None if bid is None or ask is None else 0.5 * (bid + ask)

    def size_at(self, side: str, price: float) -> float:
        book = self._bids if side == "bid" else self._asks
        return book.get(price, 0.0)

    def is_crossed(self) -> bool:
        bid, ask = self.best_bid, self.best_ask
        return bid is not None and ask is not None and bid >= ask

    def is_valid(self) -> bool:
        return bool(self._bids) and bool(self._asks) and not self.is_crossed()


class SyncState(StrEnum):
    """Where the synchroniser is in the protocol."""

    DISCONNECTED = auto()
    BUFFERING = auto()  # streaming, waiting for a REST snapshot
    SYNCED = auto()  # book is authoritative
    RESYNC_REQUIRED = auto()  # a gap was detected; drop the book and start over


class ResyncReason(StrEnum):
    INITIAL = auto()
    SEQUENCE_GAP = auto()
    STALE_SNAPSHOT = auto()
    DISCONNECT = auto()
    CROSSED_BOOK = auto()


@dataclass(slots=True)
class SyncResult:
    """What the caller should do with the diff it just submitted."""

    applied: bool
    state: SyncState
    reason: ResyncReason | None = None
    detail: str = ""


@dataclass(slots=True)
class BinanceBookSynchronizer:
    """Pure state machine implementing the snapshot+diff protocol.

    Usage::

        sync = BinanceBookSynchronizer(book)
        sync.on_connected()                 # -> BUFFERING
        for diff in ws_stream:
            result = sync.submit_diff(diff)
            if sync.needs_snapshot:
                sync.apply_snapshot(await fetch_rest_snapshot())
            if result.applied:
                publish(book.top_n(depth))
    """

    book: L2Book
    state: SyncState = SyncState.DISCONNECTED
    _buffer: list[DepthDiff] = field(default_factory=list)
    _prev_final_id: int = 0
    resync_count: int = 0
    last_resync_reason: ResyncReason | None = None
    #: Bounded so a snapshot fetch that never lands cannot exhaust memory.
    max_buffer: int = 5_000

    @property
    def needs_snapshot(self) -> bool:
        return self.state in (SyncState.BUFFERING, SyncState.RESYNC_REQUIRED)

    @property
    def is_synced(self) -> bool:
        return self.state is SyncState.SYNCED

    @property
    def buffered(self) -> int:
        return len(self._buffer)

    def on_connected(self) -> None:
        """Socket opened. Buffer diffs until a snapshot arrives."""
        self.state = SyncState.BUFFERING
        self._buffer.clear()
        self._prev_final_id = 0
        self.book.clear()

    def on_disconnected(self) -> None:
        """Socket dropped. The book is no longer trustworthy -- discard it.

        Keeping a stale book across a disconnect is the subtle failure mode this
        guards against: the diffs we missed are gone forever, so any book we
        carry forward is silently wrong rather than loudly absent.
        """
        self.state = SyncState.DISCONNECTED
        self._buffer.clear()
        self._prev_final_id = 0
        self.book.clear()
        self.last_resync_reason = ResyncReason.DISCONNECT

    def request_resync(self, reason: ResyncReason) -> None:
        self.state = SyncState.RESYNC_REQUIRED
        self.last_resync_reason = reason
        self._buffer.clear()
        self._prev_final_id = 0
        self.book.clear()

    def submit_diff(self, diff: DepthDiff) -> SyncResult:
        """Feed one stream event. Returns whether the book advanced."""
        if self.state is SyncState.DISCONNECTED:
            return SyncResult(False, self.state, detail="not connected")

        if self.state in (SyncState.BUFFERING, SyncState.RESYNC_REQUIRED):
            if self.state is SyncState.RESYNC_REQUIRED:
                self.state = SyncState.BUFFERING
            self._buffer.append(diff)
            if len(self._buffer) > self.max_buffer:
                # Oldest diffs are worthless once we exceed the window: any
                # snapshot we eventually get will be newer than them anyway.
                del self._buffer[: len(self._buffer) - self.max_buffer]
            return SyncResult(False, self.state, detail=f"buffered ({len(self._buffer)})")

        # SYNCED: enforce contiguity (rule 5).
        if diff.final_update_id <= self._prev_final_id:
            # Duplicate/stale replay. Harmless -- drop it, stay synced.
            return SyncResult(False, self.state, detail="stale diff ignored")

        if diff.first_update_id != self._prev_final_id + 1:
            detail = (
                f"gap: expected U={self._prev_final_id + 1}, got U={diff.first_update_id} "
                f"u={diff.final_update_id}"
            )
            self.resync_count += 1
            self.request_resync(ResyncReason.SEQUENCE_GAP)
            self._buffer.append(diff)
            return SyncResult(False, self.state, ResyncReason.SEQUENCE_GAP, detail)

        self.book.apply_diff(diff)
        self._prev_final_id = diff.final_update_id

        if self.book.is_crossed():
            self.resync_count += 1
            self.request_resync(ResyncReason.CROSSED_BOOK)
            return SyncResult(
                False, self.state, ResyncReason.CROSSED_BOOK, "book crossed after diff"
            )

        return SyncResult(True, self.state)

    def apply_snapshot(self, snapshot: DepthSnapshot) -> SyncResult:
        """Install a REST snapshot and replay the buffer over it (rules 3-4)."""
        if self.state is SyncState.DISCONNECTED:
            return SyncResult(False, self.state, detail="not connected")

        # Rule 3: everything fully superseded by the snapshot is dead weight.
        pending = [d for d in self._buffer if d.final_update_id > snapshot.last_update_id]
        self._buffer.clear()

        if not pending:
            # No diff yet bridges the snapshot. Stay in BUFFERING and wait --
            # the next diff off the socket will. Load the snapshot so we have
            # *something* consistent to serve if asked.
            self.book.load_snapshot(snapshot)
            self._prev_final_id = snapshot.last_update_id
            self.state = SyncState.SYNCED
            return SyncResult(True, self.state, detail="snapshot with empty buffer")

        # Rule 4: the bridging diff must straddle lastUpdateId + 1.
        head = pending[0]
        target = snapshot.last_update_id + 1
        if not (head.first_update_id <= target <= head.final_update_id):
            # Snapshot is older than the buffer we hold -> it can never bridge.
            # Ask for a fresher one, keeping the buffer so the retry can succeed.
            self.resync_count += 1
            self.last_resync_reason = ResyncReason.STALE_SNAPSHOT
            self.state = SyncState.BUFFERING
            self._buffer = pending
            self.book.clear()
            return SyncResult(
                False,
                self.state,
                ResyncReason.STALE_SNAPSHOT,
                f"snapshot lastUpdateId={snapshot.last_update_id} does not bridge "
                f"[U={head.first_update_id}, u={head.final_update_id}]",
            )

        self.book.load_snapshot(snapshot)
        self._prev_final_id = snapshot.last_update_id

        for diff in pending:
            # Contiguity within the replayed buffer still has to hold.
            if diff.first_update_id > self._prev_final_id + 1:
                self.resync_count += 1
                self.request_resync(ResyncReason.SEQUENCE_GAP)
                return SyncResult(
                    False,
                    self.state,
                    ResyncReason.SEQUENCE_GAP,
                    f"gap while replaying buffer at U={diff.first_update_id}",
                )
            self.book.apply_diff(diff)
            self._prev_final_id = diff.final_update_id

        self.state = SyncState.SYNCED
        return SyncResult(True, self.state, detail=f"replayed {len(pending)} buffered diffs")


def levels_from_pairs(pairs: Sequence[Sequence[str | float]]) -> list[tuple[float, float]]:
    """Binance sends ``[["68000.10", "0.35"], ...]``. Normalise to floats."""
    return [(float(p), float(q)) for p, q in pairs]


__all__ = [
    "BinanceBookSynchronizer",
    "DepthDiff",
    "DepthSnapshot",
    "L2Book",
    "ResyncReason",
    "SyncResult",
    "SyncState",
    "levels_from_pairs",
]
