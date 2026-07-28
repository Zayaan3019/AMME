"""Grid-search calibration for gamma, kappa and the drift lambda (spec s.5.4).

The spec asks for lambda to be "calibrated via backtest grid search". The same
machinery calibrates the A-S parameters, and it should: gamma and kappa are not
free aesthetic choices, they are the market-dependent quantities that decide
whether the strategy participates at all.

**Selection criterion.** Not raw P&L. Ranking a market maker by realised P&L
over a handful of paths selects for whichever configuration took the most
directional risk and got lucky -- the exact failure the multi-seed harness
exists to expose. The objective here is instead

    score = mean(P&L across seeds) - risk_penalty * stdev(P&L across seeds)

a mean-variance criterion over *seeds*, which prefers configurations that are
consistently mediocre to ones that are wildly bimodal. ``risk_penalty`` defaults
to 1.0, i.e. one unit of dispersion is charged at one unit of return.

**Guard against degenerate winners.** A configuration that quotes so wide it
never fills scores a perfect 0.0 +/- 0.0 and would win outright. Candidates with
fewer than ``min_fills`` fills are therefore disqualified rather than ranked --
not trading is not a strategy.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from amme_core.engine import DriftPredictor
from amme_shared.logging import get_logger
from backtest_svc.runner import (
    ABLATION_ARMS,
    ArmSpec,
    BacktestConfig,
    BacktestRunner,
    _mean,
    _stdev,
)

log = get_logger("backtest.calibrate")


@dataclass(frozen=True, slots=True)
class CalibrationPoint:
    params: dict[str, float]
    pnl_mean: float
    pnl_std: float
    sharpe_mean: float
    inventory_std_mean: float
    fills_mean: float
    score: float
    disqualified: bool = False
    reason: str = ""

    def label(self) -> str:
        return " ".join(f"{k}={v:g}" for k, v in self.params.items())


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    points: list[CalibrationPoint]

    @property
    def eligible(self) -> list[CalibrationPoint]:
        return [p for p in self.points if not p.disqualified]

    def best(self) -> CalibrationPoint | None:
        ranked = sorted(self.eligible, key=lambda p: p.score, reverse=True)
        return ranked[0] if ranked else None

    def to_markdown(self, top: int = 12) -> str:
        ranked = sorted(self.eligible, key=lambda p: p.score, reverse=True)[:top]
        lines = [
            "| Rank | Parameters | Score | P&L (mean ± sd) | Sharpe | Inventory σ | Fills |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
        for i, p in enumerate(ranked, 1):
            lines.append(
                f"| {i} | `{p.label()}` | {p.score:+.4f} "
                f"| {p.pnl_mean:+.3f} ± {p.pnl_std:.3f} | {p.sharpe_mean:.0f} "
                f"| {p.inventory_std_mean:.5f} | {p.fills_mean:.0f} |"
            )
        dq = [p for p in self.points if p.disqualified]
        if dq:
            lines.append("")
            lines.append(
                f"_{len(dq)} candidate(s) disqualified_ — "
                + "; ".join(f"`{p.label()}` ({p.reason})" for p in dq[:6])
                + ("…" if len(dq) > 6 else "")
            )
        return "\n".join(lines) + "\n"


class Calibrator:
    """Sweeps a parameter grid over several seeds and ranks the results."""

    def __init__(
        self,
        base: BacktestConfig,
        seeds: Sequence[int],
        arm: ArmSpec | None = None,
        risk_penalty: float = 1.0,
        min_fills: int = 30,
        predictor_factory: Callable[[], DriftPredictor | None] | None = None,
    ) -> None:
        self.base = base
        self.seeds = list(seeds)
        # Calibrate on pure A-S by default: gamma and kappa belong to the A-S
        # solution, and tuning them with the toxicity gate active would confound
        # the two mechanisms the ablation is supposed to separate.
        self.arm = arm or next(a for a in ABLATION_ARMS if a.key == "pure_as")
        self.risk_penalty = risk_penalty
        self.min_fills = min_fills
        self.predictor_factory = predictor_factory

    def _evaluate(self, overrides: dict[str, float]) -> CalibrationPoint:
        pnls: list[float] = []
        sharpes: list[float] = []
        inv_stds: list[float] = []
        fills: list[float] = []

        for seed in self.seeds:
            # replace() cannot type-check a **dict splat, so build the
            # config in two typed steps instead of one untyped one.
            config = replace(self.base, seed=seed)
            for field_name, value in overrides.items():
                config = replace(config, **{field_name: value})  # type: ignore[arg-type]
            runner = BacktestRunner(config, predictor_factory=self.predictor_factory)
            result = runner.run_arm(self.arm)
            m = result.metrics
            pnls.append(m.total_pnl)
            sharpes.append(m.sharpe)
            inv_stds.append(m.inventory_std)
            fills.append(float(m.n_fills))

        pnl_mean, pnl_std = _mean(pnls), _stdev(pnls)
        fills_mean = _mean(fills)
        disqualified = fills_mean < self.min_fills
        return CalibrationPoint(
            params=overrides,
            pnl_mean=pnl_mean,
            pnl_std=pnl_std,
            sharpe_mean=_mean(sharpes),
            inventory_std_mean=_mean(inv_stds),
            fills_mean=fills_mean,
            score=pnl_mean - self.risk_penalty * pnl_std,
            disqualified=disqualified,
            reason=f"only {fills_mean:.0f} fills < {self.min_fills}" if disqualified else "",
        )

    def break_even(self, points: list[CalibrationPoint], key: str) -> float | None:
        """Largest value of ``key`` at which mean P&L is still positive.

        For the fee grid this is *the* deployability number: the maximum maker
        fee the strategy survives. Reported separately from `best()` because
        the ranking criterion (mean minus dispersion) answers a different
        question -- "which configuration would I choose" rather than "where
        does the edge vanish".
        """
        eligible = sorted(
            (p for p in points if key in p.params and p.pnl_mean > 0),
            key=lambda p: p.params[key],
        )
        return eligible[-1].params[key] if eligible else None

    def sweep(self, grid: dict[str, Sequence[float]]) -> CalibrationResult:
        """Full factorial sweep over ``grid``."""
        keys = list(grid)
        combos = list(itertools.product(*(grid[k] for k in keys)))
        log.info("calibration_start", combinations=len(combos), seeds=len(self.seeds))

        points: list[CalibrationPoint] = []
        for i, combo in enumerate(combos, 1):
            overrides = dict(zip(keys, combo, strict=True))
            point = self._evaluate(overrides)
            points.append(point)
            log.info(
                "calibration_point",
                index=f"{i}/{len(combos)}",
                params=point.label(),
                score=round(point.score, 4),
                fills=round(point.fills_mean),
            )
        return CalibrationResult(points=points)


#: Default grids. Ranges bracket the analytically-derived defaults by roughly
#: an order of magnitude either side, which is the useful scale for parameters
#: that enter the solution multiplicatively.
DEFAULT_AS_GRID: dict[str, Sequence[float]] = {
    "gamma": (0.005, 0.017, 0.05, 0.15),
    "kappa": (50.0, 200.0, 800.0),
}

DEFAULT_LAMBDA_GRID: dict[str, Sequence[float]] = {
    "drift_lambda": (0.0, 0.5, 1.0, 2.0, 5.0, 10.0),
}

#: Maker fees, in bps, spanning the realistic range. Negative values are
#: rebates (several venues pay makers); 0 is rebate-neutral; +10 is Binance
#: spot VIP0. Sweeping this answers the question that decides deployability:
#: *at what fee does the edge disappear?* A strategy whose break-even fee is
#: below the fee you actually pay is not a strategy.
DEFAULT_FEE_GRID: dict[str, Sequence[float]] = {
    "maker_fee_bps": (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0),
}

#: Execution-realism sweep. Isolates how much of the measured edge survives a
#: fat-tailed latency tail and own-order queue displacement -- the two
#: assumptions most likely to be flattering a paper backtest.
DEFAULT_REALISM_GRID: dict[str, Sequence[float]] = {
    "latency_sigma": (0.0, 0.5, 1.0),
    "own_impact": (0.0, 0.5, 1.0),
}


__all__ = [
    "DEFAULT_AS_GRID",
    "DEFAULT_FEE_GRID",
    "DEFAULT_LAMBDA_GRID",
    "DEFAULT_REALISM_GRID",
    "CalibrationPoint",
    "CalibrationResult",
    "Calibrator",
]
