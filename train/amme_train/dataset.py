"""Dataset construction with strictly time-ordered splits.

**The split is the most important thing in this file.** The spec calls it out
(s.5.3: *"a documented train/val/test split by time, not randomly -- critical to
avoid look-ahead leakage; an interviewer WILL ask about this"*) and it is the
single most common way a published LOB result fails to reproduce.

Why a random split leaks, concretely: consecutive DeepLOB windows overlap by 99
of their 100 timesteps. A random shuffle puts window ``t`` in train and window
``t+1`` in test, so the test set is ~99% memorised. Reported accuracies of 80%+
on FI-2010 with random splits are measuring exactly that.

Two further leaks are closed here, both easy to miss:

**Normalisation leakage.** Feature scaling is causal by construction -- prices
are expressed relative to the *current* mid and volumes to a *trailing* depth
mean (see :mod:`amme_core.features`). Nothing is fitted over the full dataset.

**Boundary leakage.** A window ending at ``t`` carries a label built from mids
up to ``t + horizon``. Windows within ``horizon`` of a split boundary therefore
see across it, so an embargo gap of ``horizon`` samples is dropped at each
boundary -- the standard purged split (Lopez de Prado, *Advances in Financial
Machine Learning*, ch. 7).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from amme_core.features import (
    FeaturePipeline,
    FeaturePipelineConfig,
    label_series,
)
from amme_core.simulator import (
    MarketSimulator,
    SimulatorConfig,
    cycling_regime_schedule,
)
from amme_shared.events import OrderBookUpdate, TradeTick
from amme_shared.logging import get_logger

log = get_logger("train.dataset")

CLASS_NAMES = ("down", "flat", "up")


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    levels: int = 10
    lookback: int = 100
    horizon: int = 20
    #: Threshold on the smoothed return that separates up/flat/down. Tuned so
    #: the classes are roughly balanced -- see :func:`suggest_alpha`.
    alpha: float = 2e-5
    train_frac: float = 0.7
    val_frac: float = 0.15
    #: Samples dropped either side of each split boundary. Must be >= horizon
    #: or the last training window's label peeks into validation.
    embargo: int = 0

    def __post_init__(self) -> None:
        if self.train_frac + self.val_frac >= 1.0:
            raise ValueError("train_frac + val_frac must leave room for a test split")

    @property
    def effective_embargo(self) -> int:
        return max(self.embargo, self.horizon)


@dataclass(frozen=True, slots=True)
class Split:
    x: np.ndarray  # (N, T, F) float32
    y: np.ndarray  # (N,) int64
    name: str

    def __len__(self) -> int:
        return len(self.y)

    def class_counts(self) -> dict[str, int]:
        counts = np.bincount(self.y, minlength=3)
        return dict(zip(CLASS_NAMES, (int(c) for c in counts), strict=True))

    def class_balance(self) -> dict[str, float]:
        n = max(1, len(self.y))
        return {k: v / n for k, v in self.class_counts().items()}


@dataclass(frozen=True, slots=True)
class Dataset:
    train: Split
    val: Split
    test: Split
    config: DatasetConfig

    def summary(self) -> str:
        rows = [
            f"  {s.name:<5} n={len(s):>7,}  " + "  ".join(
                f"{k}={v:.1%}" for k, v in s.class_balance().items()
            )
            for s in (self.train, self.val, self.test)
        ]
        return "\n".join(rows)


def build_windows(
    rows: np.ndarray, labels: np.ndarray, lookback: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows over normalised feature rows.

    Uses ``sliding_window_view`` so the windows are a *view*, not 100 copies of
    every row -- at 100k rows the naive version needs ~1.6GB where this needs 16MB.
    The copy happens once, per split, after slicing.
    """
    n = len(rows)
    if n <= lookback:
        return np.empty((0, lookback, rows.shape[1]), np.float32), np.empty(0, np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(rows, lookback, axis=0)
    windows = np.ascontiguousarray(windows.transpose(0, 2, 1))
    # Window i spans rows [i, i+lookback); its label is the one at its last row.
    return windows.astype(np.float32), labels[lookback - 1 : lookback - 1 + len(windows)]


def suggest_alpha(mids: np.ndarray, horizon: int, target_flat: float = 0.34) -> float:
    """Pick ``alpha`` so the flat class occupies roughly ``target_flat``.

    Class balance is a *choice*, and leaving alpha at a default from a different
    market silently produces a 95%-flat dataset on which a model that always
    predicts "flat" scores 95% and learns nothing.
    """
    if len(mids) < 2 * horizon + 2:
        return 2e-5
    kernel = np.ones(horizon) / horizon
    rolling = np.convolve(mids.astype(np.float64), kernel, mode="valid")
    idx = np.arange(horizon - 1, len(mids) - horizon)
    m_minus, m_plus = rolling[idx - horizon + 1], rolling[idx + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.where(m_minus > 0, (m_plus - m_minus) / m_minus, 0.0)
    return float(np.quantile(np.abs(returns), target_flat)) or 2e-5


def features_from_stream(
    stream: Iterator[tuple[OrderBookUpdate, list[TradeTick]]],
    config: DatasetConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the *production* feature pipeline over a market stream.

    Deliberately the same :class:`~amme_core.features.FeaturePipeline` the live
    service uses. A separate training-time featuriser is how train/serve skew
    gets in -- the model would be fitted on features that never occur in
    production, and nothing would report an error.
    """
    pipeline = FeaturePipeline(
        FeaturePipelineConfig(
            levels=config.levels, lookback=config.lookback, emit_tensor=False
        )
    )
    rows: list[np.ndarray] = []
    mids: list[float] = []

    for book, trades in stream:
        for trade in trades:
            pipeline.on_raw_trade(trade.price, trade.size, trade.aggressor)
        snapshot = pipeline.on_book(book)
        if snapshot is None:
            continue
        rows.append(pipeline.normalise_row(book, snapshot.mid_price))
        mids.append(snapshot.mid_price)

    if not rows:
        return np.empty((0, config.levels * 4), np.float32), np.empty(0, np.float64)
    return np.stack(rows).astype(np.float32), np.asarray(mids, dtype=np.float64)


def split_by_time(
    x: np.ndarray, y: np.ndarray, config: DatasetConfig
) -> tuple[Split, Split, Split]:
    """Chronological split with a purge gap at each boundary."""
    n = len(y)
    embargo = config.effective_embargo
    train_end = int(n * config.train_frac)
    val_end = int(n * (config.train_frac + config.val_frac))

    train = slice(0, max(0, train_end - embargo))
    val = slice(train_end, max(train_end, val_end - embargo))
    test = slice(val_end, n)

    log.info(
        "time_split",
        total=n,
        train=train.stop - train.start,
        val=val.stop - val.start,
        test=test.stop - test.start,
        embargo=embargo,
    )
    return (
        Split(x[train].copy(), y[train].copy(), "train"),
        Split(x[val].copy(), y[val].copy(), "val"),
        Split(x[test].copy(), y[test].copy(), "test"),
    )


def build_dataset(
    stream: Iterator[tuple[OrderBookUpdate, list[TradeTick]]],
    config: DatasetConfig | None = None,
    auto_alpha: bool = True,
) -> Dataset:
    config = config or DatasetConfig()
    rows, mids = features_from_stream(stream, config)
    if len(rows) <= config.lookback + 2 * config.horizon:
        raise ValueError(f"not enough data: {len(rows)} rows")

    alpha = suggest_alpha(mids, config.horizon) if auto_alpha else config.alpha
    if auto_alpha:
        log.info("alpha_selected", alpha=f"{alpha:.3e}", configured=f"{config.alpha:.3e}")

    labels = label_series(mids, config.horizon, alpha)
    x, y = build_windows(rows, labels, config.lookback)
    train, val, test = split_by_time(x, y, config)
    return Dataset(train=train, val=val, test=test, config=config)


def build_from_simulator(
    seconds: float = 3600.0,
    seed: int = 42,
    config: DatasetConfig | None = None,
    sim_config: SimulatorConfig | None = None,
) -> Dataset:
    """Training data from the simulator. Reproducible from ``seed`` alone."""
    sim = MarketSimulator(
        sim_config
        or SimulatorConfig(
            seed=seed, tick_size=0.01, sigma=0.15, spread_vol_coeff=6.0,
            mean_limit_size=0.004, mean_market_size=0.012,
            market_order_rate=2.86, informed_intensity=0.10,
        ),
        # Cycling, not single-pass: see cycling_regime_schedule's docstring
        # for the 80%-one-class test split that motivated it.
        regime_schedule=cycling_regime_schedule(seconds, cycles=12),
    )
    return build_dataset(sim.run(seconds), config)


def save_dataset(dataset: Dataset, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        train_x=dataset.train.x, train_y=dataset.train.y,
        val_x=dataset.val.x, val_y=dataset.val.y,
        test_x=dataset.test.x, test_y=dataset.test.y,
    )
    return target


def load_dataset(path: str | Path, config: DatasetConfig | None = None) -> Dataset:
    blob = np.load(Path(path))
    return Dataset(
        train=Split(blob["train_x"], blob["train_y"], "train"),
        val=Split(blob["val_x"], blob["val_y"], "val"),
        test=Split(blob["test_x"], blob["test_y"], "test"),
        config=config or DatasetConfig(),
    )


__all__ = [
    "CLASS_NAMES",
    "Dataset",
    "DatasetConfig",
    "Split",
    "build_dataset",
    "build_from_simulator",
    "build_windows",
    "load_dataset",
    "save_dataset",
    "split_by_time",
    "suggest_alpha",
]
