"""``python -m amme_train.record`` -- turn the live recording into a training set.

The live stack already records: ``ingestion-svc`` writes every book update and
trade it sees to TimescaleDB (``persist_ticks``). This command waits until that
recording holds enough *uninterrupted* market time, then runs it through the
same path the simulator dataset takes:

* the **largest contiguous segment**, split at gaps longer than ``--max-gap``
  seconds. A capture that spans a restart contains the stack's own downtime,
  and training windows that straddle the gap would teach the model that the
  market jumps between two unrelated states. Same rule as the replay backtest
  (:func:`backtest_svc.replay.largest_contiguous`);
* the **production feature pipeline** (:func:`amme_train.dataset.build_dataset`),
  so there is no train/serve skew;
* the **purged chronological split** with an embargo at each boundary.

The result is an ``.npz`` that ``python -m amme_train.train --dataset`` reads.
``make train-live`` writes the model and metrics under ``*_live`` names, so a
live-data model never overwrites the shipped artefacts by accident.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from amme_shared.events import OrderBookUpdate, TradeTick
from amme_train.dataset import Dataset, DatasetConfig, build_dataset, save_dataset

Window = Sequence[tuple[OrderBookUpdate, list[TradeTick]]]


class InsufficientRecordingError(Exception):
    """The recording does not yet hold enough contiguous market time."""

    def __init__(self, have_seconds: float, need_seconds: float) -> None:
        self.have_seconds = have_seconds
        self.need_seconds = need_seconds
        super().__init__(
            f"largest contiguous segment is {have_seconds / 60:.1f} min; "
            f"need {need_seconds / 60:.1f} min"
        )


def window_seconds(window: Window) -> float:
    if len(window) < 2:
        return 0.0
    return (window[-1][0].ts_event - window[0][0].ts_event) / 1e9


def build_live_dataset(
    window: Window,
    *,
    min_seconds: float,
    max_gap_seconds: float = 5.0,
    config: DatasetConfig | None = None,
) -> tuple[Dataset, float]:
    """Dataset from the largest contiguous segment of a recorded window.

    Returns the dataset and the number of seconds of market time it covers.
    Raises :class:`InsufficientRecordingError` when that segment is too short.
    """
    # Imported here so the training package does not depend on backtest-svc
    # unless this command is actually used.
    from backtest_svc.replay import largest_contiguous

    segment = largest_contiguous(window, max_gap_seconds)
    seconds = window_seconds(segment)
    if seconds < min_seconds:
        raise InsufficientRecordingError(seconds, min_seconds)
    return build_dataset(iter(segment), config or DatasetConfig()), seconds


#: Rows are read in slices of this much market time, so no single query runs
#: into the connection pool's 30 s command timeout.
LOAD_CHUNK_SECONDS = 1800.0


async def _load(
    dsn: str, symbol: str, max_gap_seconds: float = 5.0
) -> list[tuple[OrderBookUpdate, list[TradeTick]]]:
    """Load the largest contiguous segment only.

    The segments are found in SQL from timestamps alone. Loading the whole
    recording first -- months of it, most of which is not the segment we
    want -- timed out once the stack had been recording for a while.
    """
    from backtest_svc.replay import TimescaleReplaySource

    source = TimescaleReplaySource(dsn)
    try:
        segments = await source.segments(symbol, max_gap_seconds)
        if not segments:
            return []
        best = max(segments, key=lambda s: s.n_books)
        loaded: list[tuple[OrderBookUpdate, list[TradeTick]]] = await source.load(
            symbol, best.window.start_ns, best.end_exclusive_ns,
            chunk_seconds=LOAD_CHUNK_SECONDS,
        )
        return loaded
    finally:
        await source.close()


async def run(args: argparse.Namespace) -> int:
    from amme_shared.config import get_settings

    settings = get_settings()
    symbol = args.symbol or settings.market.symbol
    dsn = args.dsn or settings.infra.postgres_dsn
    need = args.minutes * 60.0
    deadline = time.monotonic() + args.wait * 60.0
    config = DatasetConfig(levels=args.levels, lookback=args.lookback, horizon=args.horizon)

    while True:
        try:
            recorded = await _load(dsn, symbol, args.max_gap)
        except TimeoutError as exc:  # a subclass of OSError: check it first
            print(
                f"TimescaleDB query timed out ({exc!r}). The database is up but a read "
                "took longer than the pool's command timeout.",
                file=sys.stderr,
            )
            return 2
        except OSError as exc:  # refused / unreachable / DNS
            print(
                f"cannot reach TimescaleDB ({type(exc).__name__}: {exc}). "
                "Start the stack with `make up`; it records while it runs.",
                file=sys.stderr,
            )
            return 2
        try:
            dataset, seconds = build_live_dataset(
                recorded, min_seconds=need, max_gap_seconds=args.max_gap, config=config
            )
            break
        except InsufficientRecordingError as exc:
            if time.monotonic() >= deadline:
                print(
                    f"{exc}. Keep the stack running (`make up`) and retry, or pass "
                    f"--wait to block until enough is recorded.",
                    file=sys.stderr,
                )
                return 1
            print(f"waiting: {exc}", file=sys.stderr)
            await asyncio.sleep(min(60.0, max(5.0, deadline - time.monotonic())))

    path = save_dataset(dataset, args.out)
    print(f"{symbol}: {seconds / 60:.1f} min of contiguous live data -> {path}")
    print(dataset.summary())
    print("\ntrain on it with:  make train-live")
    print(f"  (or: python -m amme_train.train --dataset {path})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--minutes", type=float, default=60.0,
                        help="contiguous live market time required (default 60)")
    parser.add_argument("--wait", type=float, default=0.0,
                        help="minutes to keep polling until enough is recorded (default 0)")
    parser.add_argument("--max-gap", type=float, default=5.0, dest="max_gap",
                        help="seconds without a book update that split the recording")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--dsn", default=None, help="TimescaleDB DSN (default: settings)")
    parser.add_argument("--out", default="data/live_dataset.npz")
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=20)
    args = parser.parse_args(argv)
    if args.minutes <= 0:
        parser.error("--minutes must be positive")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
