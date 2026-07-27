"""``python -m backtest_svc.cli`` -- the regression harness entry point.

Runs without Redis, Postgres or Docker, which is what lets CI gate a PR on the
ablation without standing up the whole stack.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from amme_shared.logging import configure_logging
from backtest_svc.calibrate import DEFAULT_AS_GRID, Calibrator
from backtest_svc.runner import (
    ABLATION_ARMS,
    BacktestConfig,
    BacktestRunner,
    render_multi_seed_markdown,
)

#: Fixed seed set. Pinned rather than random so the ablation is comparable
#: across commits -- a results table that silently changes its own sample is
#: not a regression test.
DEFAULT_SEEDS = (11, 23, 42, 77, 101, 404, 900, 1234, 5150, 8675)


def _config(args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(
        duration_seconds=args.duration,
        warmup_seconds=args.warmup,
        drift_lambda=args.drift_lambda,
    )


def cmd_ablation(args: argparse.Namespace) -> int:
    log = configure_logging("backtest-cli", "INFO", json_output=False)
    config = _config(args)
    runner = BacktestRunner(config)
    arms = tuple(
        a for a in ABLATION_ARMS
        # Drift arms need a model; skip them until one exists so the harness
        # stays runnable in CI rather than failing on a missing artifact.
        if a.drift_lambda >= 0 or config.drift_lambda > 0
    )
    seeds = args.seeds or list(DEFAULT_SEEDS)
    log.info("ablation_start", seeds=len(seeds), arms=[a.key for a in arms],
             duration=config.duration_seconds)

    report = runner.run_multi_seed(seeds, arms)
    markdown = render_multi_seed_markdown(report, config)

    out = Path(config.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation_multiseed.md").write_text(markdown, encoding="utf-8")
    (out / "ablation_multiseed.json").write_text(
        json.dumps(
            {
                "seeds": report.seeds,
                "config": dataclasses.asdict(config),
                "aggregates": [dataclasses.asdict(a) for a in report.aggregate()],
                "deltas": {
                    f"{t}_vs_{c}": report.paired_deltas(t, c)
                    for t, c in (
                        ("pure_as", "naive_symmetric"),
                        ("as_toxicity", "pure_as"),
                        ("as_drift", "pure_as"),
                        ("amme", "pure_as"),
                    )
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(markdown)

    aggregates = {a.key: a for a in report.aggregate()}
    identity_ok = all(
        r.engine.pnl_identity_ok for runs in report.by_arm.values() for r in runs
    )
    if not identity_ok:
        print("FAIL: P&L identity violated in at least one run", file=sys.stderr)
        return 1

    # The headline claim the DoD asks us to defend, checked rather than asserted
    # in prose: A-S must beat naive on RISK-ADJUSTED terms. Raw P&L is not the
    # criterion -- a maker that takes more directional risk can out-earn a
    # better one over a finite sample, which is the whole reason this harness
    # runs many seeds and reports dispersion.
    if args.gate and "pure_as" in aggregates and "naive_symmetric" in aggregates:
        a_s, naive = aggregates["pure_as"], aggregates["naive_symmetric"]
        checks = {
            "sharpe": a_s.sharpe_mean > naive.sharpe_mean,
            "inventory_variance": a_s.inventory_variance_mean < naive.inventory_variance_mean,
            "max_drawdown": a_s.max_drawdown_mean < naive.max_drawdown_mean,
        }
        for name, passed in checks.items():
            print(f"  gate {name}: {'PASS' if passed else 'FAIL'}", file=sys.stderr)
        if not all(checks.values()):
            return 1
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    configure_logging("backtest-cli", "INFO", json_output=False)
    config = _config(args)
    seeds = args.seeds or list(DEFAULT_SEEDS[:3])
    calibrator = Calibrator(config, seeds=seeds, min_fills=args.min_fills)
    result = calibrator.sweep(DEFAULT_AS_GRID)

    out = Path(config.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    markdown = result.to_markdown()
    (out / "calibration.md").write_text(markdown, encoding="utf-8")
    print(markdown)

    best = result.best()
    if best is None:
        print("no eligible configuration found", file=sys.stderr)
        return 1
    print(f"\nBEST: {best.label()}  score={best.score:+.4f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="backtest_svc.cli", description=__doc__)
    parser.add_argument("--duration", type=float, default=900.0, help="seconds per run")
    parser.add_argument("--warmup", type=float, default=90.0, help="seconds excluded from metrics")
    parser.add_argument("--drift-lambda", type=float, default=0.0, dest="drift_lambda")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)

    sub = parser.add_subparsers(dest="command", required=True)

    ablation = sub.add_parser("ablation", help="run the multi-seed ablation grid")
    ablation.add_argument(
        "--gate", action="store_true", default=True,
        help="fail the process if A-S does not beat naive on risk-adjusted terms",
    )
    ablation.add_argument("--no-gate", action="store_false", dest="gate")
    ablation.set_defaults(func=cmd_ablation)

    calibrate = sub.add_parser("calibrate", help="grid-search gamma/kappa")
    calibrate.add_argument("--min-fills", type=int, default=30, dest="min_fills")
    calibrate.set_defaults(func=cmd_calibrate)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
