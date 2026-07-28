"""``python -m backtest_svc.replay_cli`` -- backtest on recorded live ticks.

Defaults to the largest *contiguous* segment of the recording rather than the
whole range, because a capture accumulated across restarts contains the stack's
own downtime and backtesting over a gap silently attributes P&L to a period
that never happened.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from amme_core.avellaneda_stoikov import ASParams
from amme_core.engine import EngineConfig, TradingEngine
from amme_core.features import FeaturePipelineConfig
from amme_core.matching import MatchingParams
from amme_core.performance import compute_metrics
from amme_core.risk import RiskParams
from amme_core.strategies import build_strategy
from amme_core.toxicity import ToxicityParams
from amme_shared.config import get_settings
from amme_shared.logging import configure_logging
from backtest_svc.replay import (
    TimescaleReplaySource,
    largest_contiguous,
    replay_quality,
    split_contiguous,
)


async def run(args: argparse.Namespace) -> int:
    log = configure_logging("replay-cli", "INFO", json_output=False)
    settings = get_settings()
    symbol = args.symbol or settings.market.symbol

    source = TimescaleReplaySource(args.dsn or settings.infra.postgres_dsn)
    try:
        window = await source.available(symbol)
        if window is None:
            print(f"no recorded data for {symbol}; run the stack first", file=sys.stderr)
            return 1

        full = await source.load(symbol, window.start_ns, window.end_ns)
        segments = split_contiguous(full, args.max_gap)
        data = full if args.whole_range else largest_contiguous(full, args.max_gap)
    finally:
        await source.close()

    quality = replay_quality(data)
    log.info("replay_window", segments=len(segments), **{
        k: round(v, 2) for k, v in quality.items()
    })
    if quality.get("books", 0) < 100:
        print("recording too short to backtest", file=sys.stderr)
        return 1

    tick = settings.market.tick_size
    q_max = settings.risk.q_max
    quote_size = settings.avellaneda_stoikov.quote_size

    print(f"\n=== REPLAY: {symbol} ===")
    print(
        f"  {quality['books']:.0f} books, {quality['trades']:.0f} trades, "
        f"{quality['seconds']:.0f}s @ {quality['update_rate_hz']:.2f}Hz, "
        f"max gap {quality['largest_gap_seconds']:.2f}s"
    )
    print(f"  {len(segments)} contiguous segment(s) in the recording\n")

    for name in ("naive_symmetric", "pure_avellaneda_stoikov", "amme"):
        strategy = build_strategy(
            name,
            as_params=ASParams(
                gamma=settings.avellaneda_stoikov.gamma,
                kappa=settings.avellaneda_stoikov.kappa,
                horizon_seconds=settings.avellaneda_stoikov.horizon_seconds,
                tick_size=tick,
                inventory_scale=q_max,
                quote_size=quote_size,
                min_half_spread_ticks=settings.avellaneda_stoikov.min_half_spread_ticks,
            ),
            toxicity_params=ToxicityParams(),
        )
        engine = TradingEngine(
            strategy,
            config=EngineConfig(q_max=q_max, warmup_seconds=args.warmup),
            feature_config=FeaturePipelineConfig(
                vpin_bucket_volume=settings.toxicity.vpin_bucket_volume,
                vpin_buckets=settings.toxicity.vpin_buckets,
                emit_tensor=False,
            ),
            risk_params=RiskParams(q_max=q_max),
            matching_params=MatchingParams(
                tick_size=tick, maker_fee_bps=settings.market.maker_fee_bps
            ),
        )
        result = engine.run(data)
        m = compute_metrics(name, result.points, q_max=q_max)
        print(
            f"  {name:26s} pnl={m.total_pnl:+9.4f} sharpe={m.sharpe:9.1f} "
            f"fills={m.n_fills:5d} invStd={m.inventory_std:.5f} "
            f"maxDD={m.max_drawdown:.4f} identity={result.pnl_identity_ok}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--dsn", default=None)
    parser.add_argument("--warmup", type=float, default=30.0)
    parser.add_argument(
        "--max-gap", type=float, default=5.0, dest="max_gap",
        help="seconds without an update that counts as a recording break",
    )
    parser.add_argument(
        "--whole-range", action="store_true", dest="whole_range",
        help="backtest the entire capture including gaps (not recommended)",
    )
    return asyncio.run(run(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
