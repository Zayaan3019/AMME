"""The regression and ablation harness (spec s.6).

This produces the single most important artifact in the project: a table that
shows what each component is actually worth. Its credibility rests on three
properties, each enforced here rather than assumed:

**Identical inputs.** Every arm runs over the *same* market stream, generated
from the same seed. Not statistically similar -- byte-identical. Any difference
in the results is therefore caused by the strategy, not by luck of the draw.

**Identical scoring.** Every arm is measured by the same
:func:`~amme_core.performance.compute_metrics` call. There is no per-strategy
metric code where a favourable convention could hide.

**Strict nesting.** The arms differ by exactly one mechanism at a time:

.. code-block:: text

    naive_symmetric   fixed spread, no inventory term
    pure_as           + A-S reservation price and optimal spread
    as_toxicity       + VPIN/OFI toxicity gate
    as_drift          + DeepLOB drift injection
    amme              + both

``as_toxicity`` and ``as_drift`` are what make the report an ablation rather
than a leaderboard: with only the endpoints you can say AMME beats the
baseline, but not *which half* earned it -- and that is the question an
interviewer will actually ask.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from amme_core.avellaneda_stoikov import ASParams
from amme_core.engine import DriftPredictor, EngineConfig, EngineResult, TradingEngine
from amme_core.features import FeaturePipelineConfig, FeatureSnapshot
from amme_core.matching import MatchingParams
from amme_core.performance import PerformanceMetrics, compute_metrics
from amme_core.risk import RiskParams
from amme_core.simulator import (
    MarketSimulator,
    Regime,
    SimulatorConfig,
    default_regime_schedule,
)
from amme_core.strategies import (
    AMMEStrategy,
    NaiveParams,
    NaiveSymmetricStrategy,
    PureAvellanedaStoikovStrategy,
    Strategy,
)
from amme_core.toxicity import ToxicityParams
from amme_shared.events import OrderBookUpdate, TradeTick
from amme_shared.logging import get_logger

log = get_logger("backtest.runner")

MarketStreamFactory = Callable[[], Iterator[tuple[OrderBookUpdate, list[TradeTick]]]]


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """One row of the ablation table."""

    key: str
    label: str
    description: str
    toxicity_enabled: bool
    drift_lambda: float
    naive: bool = False


#: The ablation grid. Order is meaningful -- it reads as a build-up.
ABLATION_ARMS: tuple[ArmSpec, ...] = (
    ArmSpec(
        "naive_symmetric",
        "Naive symmetric",
        "Fixed spread around the mid. No inventory term, no gating.",
        toxicity_enabled=False,
        drift_lambda=0.0,
        naive=True,
    ),
    ArmSpec(
        "pure_as",
        "Pure Avellaneda-Stoikov",
        "Reservation price + optimal spread. No ML, no toxicity gate.",
        toxicity_enabled=False,
        drift_lambda=0.0,
    ),
    ArmSpec(
        "as_toxicity",
        "A-S + toxicity gate",
        "Adds VPIN/OFI spread widening. Isolates the adverse-selection defence.",
        toxicity_enabled=True,
        drift_lambda=0.0,
    ),
    ArmSpec(
        "as_drift",
        "A-S + DeepLOB drift",
        "Adds drift injection into the reservation price. Isolates the ML edge.",
        toxicity_enabled=False,
        drift_lambda=-1.0,  # sentinel: replaced with the configured lambda
    ),
    ArmSpec(
        "amme",
        "Full AMME",
        "A-S + toxicity gate + DeepLOB drift.",
        toxicity_enabled=True,
        drift_lambda=-1.0,
    ),
)


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    seed: int = 42
    duration_seconds: float = 1800.0
    warmup_seconds: float = 120.0
    tick_size: float = 0.01
    q_max: float = 0.05
    quote_size: float = 0.001
    initial_price: float = 50_000.0
    maker_fee_bps: float = 0.0

    # Grid-searched, not guessed: backtest_svc.calibrate swept gamma x kappa
    # over 3 seeds and ranked by mean(P&L) - stdev(P&L) across seeds, with
    # candidates under 30 fills disqualified so "never trade" cannot win.
    #
    # RE-DERIVED after the matching-engine over-fill fix. The previous winner
    # (gamma=1.5, kappa=3) traded only 87 times per run and its P&L was
    # inflated by through-fills that ignored the aggressor's size. On the
    # corrected engine it drops to rank 2, and the optimum moves to a
    # configuration that actually participates: 397 fills, tighter dispersion
    # (+0.022 +/- 0.014 vs +0.043 +/- 0.038), Sharpe 668. See docs/RESULTS.md.
    gamma: float = 4.0
    kappa: float = 40.0
    horizon_seconds: float = 15.0
    min_half_spread_ticks: float = 1.0
    drift_lambda: float = 0.0

    # Matched to the recalibrated A-S half-spread so the baseline competes at
    # the same width rather than being a straw man.
    naive_half_spread_ticks: float = 8.0

    vpin_bucket_volume: float = 0.05
    vpin_buckets: int = 30
    vpin_threshold: float = 0.35
    vpin_beta: float = 2.0
    ofi_threshold: float = 2.0
    ofi_beta: float = 0.25
    ofi_skew_beta: float = 0.5

    sample_interval_seconds: float = 1.0
    use_regime_schedule: bool = True
    results_dir: str = "artifacts"

    # Must match the trained artifact's input shape. `model_predictor` reads
    # the real shape off the ONNX model and raises if these disagree, rather
    # than letting onnxruntime fail 200 frames deep mid-run.
    model_lookback: int = 50
    model_levels: int = 10

    # --- simulated market regime ------------------------------------------
    # Defaults describe a *makeable* venue, not BTCUSDT spot. That distinction
    # is deliberate and measured: 180s of live BTCUSDT (artifacts/
    # binance_calibration.json) shows a spread pinned at exactly 1 tick at both
    # p50 and p90, against sigma of 17 ticks per 100ms step and ~5 BTC resting
    # at the touch. The tick is binding there, adverse selection dwarfs spread
    # capture by more than an order of magnitude, and no passive quoter with a
    # 1-tick spread has positive expectancy -- which is why real BTCUSDT makers
    # live on fee tiers and speed, not on the quoted spread.
    #
    # Running the ablation there would only rank strategies by which loses
    # least. These defaults instead describe a wider-spread, thinner-queue
    # instrument where classical A-S economics apply, so the ablation measures
    # each component's contribution rather than a common floor. The live
    # pipeline still runs against real BTCUSDT.
    sim_sigma: float = 0.15
    sim_spread_vol_coeff: float = 6.0
    sim_mean_limit_size: float = 0.004
    sim_mean_market_size: float = 0.012
    sim_market_order_rate: float = 2.86
    sim_informed_intensity: float = 0.10


@dataclass
class ArmResult:
    spec: ArmSpec
    metrics: PerformanceMetrics
    engine: EngineResult = field(repr=False)

    def summary(self) -> dict[str, Any]:
        m = self.metrics
        return {
            "arm": self.spec.key,
            "label": self.spec.label,
            "total_pnl": round(m.total_pnl, 4),
            "sharpe": round(m.sharpe, 2),
            "max_drawdown": round(m.max_drawdown, 4),
            "inventory_variance": round(m.inventory_variance, 10),
            "inventory_std": round(m.inventory_std, 6),
            "n_fills": m.n_fills,
            "fill_ratio": round(m.fill_ratio, 4),
            "avg_spread_quoted": round(m.avg_spread_quoted, 5),
            "spread_captured": round(m.spread_captured, 6),
            "pnl_high_vpin": round(m.pnl_high_vpin, 4),
            "pnl_low_vpin": round(m.pnl_low_vpin, 4),
            "toxicity_adjusted_pnl": round(m.toxicity_adjusted_pnl, 4),
            "time_at_limit_pct": round(m.time_at_limit_pct, 2),
            "breaker_trips": self.engine.n_breaker_trips,
            "risk_halts": self.engine.n_risk_halts,
            "pnl_identity_ok": self.engine.pnl_identity_ok,
        }


class BacktestRunner:
    """Runs the ablation grid and writes the report."""

    def __init__(
        self,
        config: BacktestConfig | None = None,
        predictor_factory: Callable[[], DriftPredictor | None] | None = None,
    ) -> None:
        self.config = config or BacktestConfig()
        self._predictor_factory = predictor_factory

    # -- market stream -----------------------------------------------------
    def _make_stream(self) -> Iterator[tuple[OrderBookUpdate, list[TradeTick]]]:
        """A fresh, identically-seeded simulator run.

        Reconstructed per arm rather than materialised once and replayed. Both
        are deterministic; rebuilding keeps peak memory flat for long runs, and
        the seeding guarantees the streams are identical.
        """
        c = self.config
        sim = MarketSimulator(
            SimulatorConfig(
                seed=c.seed,
                tick_size=c.tick_size,
                initial_price=c.initial_price,
                sigma=c.sim_sigma,
                spread_vol_coeff=c.sim_spread_vol_coeff,
                mean_limit_size=c.sim_mean_limit_size,
                mean_market_size=c.sim_mean_market_size,
                market_order_rate=c.sim_market_order_rate,
                informed_intensity=c.sim_informed_intensity,
            ),
            regime_schedule=(
                default_regime_schedule(c.duration_seconds) if c.use_regime_schedule else None
            ),
            regime=Regime.CALM,
        )
        return sim.run(c.duration_seconds)

    # -- arm construction --------------------------------------------------
    def _as_params(self, drift_lambda: float) -> ASParams:
        c = self.config
        return ASParams(
            gamma=c.gamma,
            kappa=c.kappa,
            horizon_mode="infinite",
            horizon_seconds=c.horizon_seconds,
            tick_size=c.tick_size,
            min_half_spread_ticks=c.min_half_spread_ticks,
            quote_size=c.quote_size,
            inventory_scale=c.q_max,
            drift_lambda=drift_lambda,
        )

    def _toxicity_params(self, enabled: bool) -> ToxicityParams:
        c = self.config
        return ToxicityParams(
            vpin_threshold=c.vpin_threshold,
            vpin_beta=c.vpin_beta,
            ofi_threshold=c.ofi_threshold,
            ofi_beta=c.ofi_beta,
            ofi_skew_beta=c.ofi_skew_beta,
            enabled=enabled,
        )

    def _build_strategy(self, spec: ArmSpec) -> Strategy:
        c = self.config
        if spec.naive:
            return NaiveSymmetricStrategy(
                NaiveParams(
                    half_spread_ticks=c.naive_half_spread_ticks,
                    tick_size=c.tick_size,
                    quote_size=c.quote_size,
                )
            )
        drift_lambda = c.drift_lambda if spec.drift_lambda < 0 else spec.drift_lambda
        if not spec.toxicity_enabled and drift_lambda == 0.0:
            # Exactly the pure-A-S arm. Uses the dedicated class so the baseline
            # cannot be perturbed by a stray toxicity or drift setting.
            return PureAvellanedaStoikovStrategy(self._as_params(0.0))
        return AMMEStrategy(
            as_params=self._as_params(drift_lambda),
            toxicity_params=self._toxicity_params(spec.toxicity_enabled),
        )

    def _needs_predictor(self, spec: ArmSpec) -> bool:
        drift_lambda = self.config.drift_lambda if spec.drift_lambda < 0 else spec.drift_lambda
        return drift_lambda > 0.0

    # -- execution ---------------------------------------------------------
    def run_arm(self, spec: ArmSpec) -> ArmResult:
        c = self.config
        strategy = self._build_strategy(spec)

        predictor: DriftPredictor | None = None
        if self._needs_predictor(spec) and self._predictor_factory is not None:
            predictor = self._predictor_factory()

        engine = TradingEngine(
            strategy,
            config=EngineConfig(
                q_max=c.q_max,
                warmup_seconds=c.warmup_seconds,
                sample_interval_seconds=c.sample_interval_seconds,
            ),
            feature_config=FeaturePipelineConfig(
                vpin_bucket_volume=c.vpin_bucket_volume,
                vpin_buckets=c.vpin_buckets,
                lookback=c.model_lookback,
                levels=c.model_levels,
                # Only build the tensor when a model will consume it: it is a
                # (T x 4L) array per update, pure waste for the non-drift arms.
                emit_tensor=predictor is not None,
            ),
            risk_params=RiskParams(q_max=c.q_max),
            matching_params=MatchingParams(
                maker_fee_bps=c.maker_fee_bps, tick_size=c.tick_size
            ),
            predictor=predictor,
        )

        result = engine.run(self._make_stream())
        metrics = compute_metrics(
            spec.key,
            result.points,
            interval_seconds=c.sample_interval_seconds,
            q_max=c.q_max,
            fees_paid=result.fees_paid,
            volume_traded=result.volume_traded,
        )
        log.info(
            "arm_complete",
            arm=spec.key,
            pnl=round(metrics.total_pnl, 4),
            sharpe=round(metrics.sharpe, 2),
            fills=metrics.n_fills,
        )
        return ArmResult(spec=spec, metrics=metrics, engine=result)

    def run_all(self, arms: tuple[ArmSpec, ...] = ABLATION_ARMS) -> list[ArmResult]:
        results = []
        for spec in arms:
            if self._needs_predictor(spec) and self._predictor_factory is None:
                log.warning("arm_skipped", arm=spec.key, reason="no drift predictor available")
                continue
            results.append(self.run_arm(spec))
        return results

    def run_multi_seed(
        self, seeds: Sequence[int], arms: tuple[ArmSpec, ...] = ABLATION_ARMS
    ) -> MultiSeedReport:
        """Run the full grid across several seeds and aggregate.

        This is not optional rigour, it is the difference between a result and
        an anecdote. A market maker's P&L over a single 20-minute path is
        dominated by whether its residual inventory happened to drift the right
        way. On one seed we measured the naive baseline *beating* pure
        Avellaneda-Stoikov -- not because a fixed spread is better, but because
        that path rewarded the directional risk A-S exists to suppress.

        Reporting mean +/- stdev across seeds, and a per-seed win rate, makes
        that visible instead of letting one lucky path become the headline.
        Every arm still sees byte-identical market data *within* each seed, so
        the paired comparison stays valid.
        """
        by_arm: dict[str, list[ArmResult]] = {}
        base_seed = self.config.seed
        try:
            for seed in seeds:
                self.config = replace(self.config, seed=seed)
                log.info("seed_start", seed=seed)
                for result in self.run_all(arms):
                    by_arm.setdefault(result.spec.key, []).append(result)
        finally:
            self.config = replace(self.config, seed=base_seed)
        return MultiSeedReport(seeds=list(seeds), by_arm=by_arm)

    # -- reporting ---------------------------------------------------------
    def write_report(
        self, results: list[ArmResult], results_dir: str | None = None
    ) -> tuple[Path, Path]:
        target = Path(results_dir or self.config.results_dir)
        target.mkdir(parents=True, exist_ok=True)

        payload = {
            "config": asdict(self.config),
            "arms": [r.summary() for r in results],
            "full_metrics": {r.spec.key: r.metrics.as_dict() for r in results},
        }
        json_path = target / "ablation.json"
        json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

        md_path = target / "ablation.md"
        md_path.write_text(render_markdown(results, self.config), encoding="utf-8")
        log.info("report_written", json=str(json_path), markdown=str(md_path))
        return json_path, md_path


@dataclass(frozen=True, slots=True)
class ArmAggregate:
    """One arm's results summarised across seeds."""

    key: str
    label: str
    n_seeds: int
    pnl_mean: float
    pnl_std: float
    pnl_min: float
    pnl_max: float
    sharpe_mean: float
    sharpe_std: float
    max_drawdown_mean: float
    inventory_std_mean: float
    inventory_variance_mean: float
    fills_mean: float
    fill_ratio_mean: float
    avg_spread_mean: float
    pnl_high_vpin_mean: float
    pnl_low_vpin_mean: float
    seeds_profitable: int

    @property
    def profitable_rate(self) -> float:
        return self.seeds_profitable / self.n_seeds if self.n_seeds else 0.0


@dataclass(frozen=True, slots=True)
class MultiSeedReport:
    seeds: list[int]
    by_arm: dict[str, list[ArmResult]]

    def aggregate(self) -> list[ArmAggregate]:
        out: list[ArmAggregate] = []
        for key, runs in self.by_arm.items():
            pnls = [r.metrics.total_pnl for r in runs]
            sharpes = [r.metrics.sharpe for r in runs]
            out.append(
                ArmAggregate(
                    key=key,
                    label=runs[0].spec.label,
                    n_seeds=len(runs),
                    pnl_mean=_mean(pnls),
                    pnl_std=_stdev(pnls),
                    pnl_min=min(pnls),
                    pnl_max=max(pnls),
                    sharpe_mean=_mean(sharpes),
                    sharpe_std=_stdev(sharpes),
                    max_drawdown_mean=_mean([r.metrics.max_drawdown for r in runs]),
                    inventory_std_mean=_mean([r.metrics.inventory_std for r in runs]),
                    inventory_variance_mean=_mean(
                        [r.metrics.inventory_variance for r in runs]
                    ),
                    fills_mean=_mean([float(r.metrics.n_fills) for r in runs]),
                    fill_ratio_mean=_mean([r.metrics.fill_ratio for r in runs]),
                    avg_spread_mean=_mean([r.metrics.avg_spread_quoted for r in runs]),
                    pnl_high_vpin_mean=_mean([r.metrics.pnl_high_vpin for r in runs]),
                    pnl_low_vpin_mean=_mean([r.metrics.pnl_low_vpin for r in runs]),
                    seeds_profitable=sum(1 for p in pnls if p > 0),
                )
            )
        order = {spec.key: i for i, spec in enumerate(ABLATION_ARMS)}
        out.sort(key=lambda a: order.get(a.key, 99))
        return out

    def paired_deltas(self, treatment: str, control: str) -> dict[str, float]:
        """Per-seed paired difference ``treatment - control``.

        Pairing is what makes a 5-seed comparison informative at all: the two
        arms saw byte-identical market data on each seed, so the path noise
        that dominates the raw P&L cancels in the difference.
        """
        t_runs = {id(r): r for r in self.by_arm.get(treatment, [])}
        c_runs = self.by_arm.get(control, [])
        t_list = self.by_arm.get(treatment, [])
        if not t_list or not c_runs or len(t_list) != len(c_runs):
            return {}
        del t_runs
        deltas = [t.metrics.total_pnl - c.metrics.total_pnl
                  for t, c in zip(t_list, c_runs, strict=True)]
        sharpe_deltas = [t.metrics.sharpe - c.metrics.sharpe
                         for t, c in zip(t_list, c_runs, strict=True)]
        inv_deltas = [t.metrics.inventory_std - c.metrics.inventory_std
                      for t, c in zip(t_list, c_runs, strict=True)]
        return {
            "pnl_delta_mean": _mean(deltas),
            "pnl_delta_std": _stdev(deltas),
            "sharpe_delta_mean": _mean(sharpe_deltas),
            "inventory_std_delta_mean": _mean(inv_deltas),
            "seeds_won": float(sum(1 for d in deltas if d > 0)),
            "n_seeds": float(len(deltas)),
        }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (n - 1))


def render_multi_seed_markdown(report: MultiSeedReport, config: BacktestConfig) -> str:
    """The headline results table: mean +/- stdev across seeds."""
    aggregates = report.aggregate()
    if not aggregates:
        return "_No ablation arms produced results._\n"

    lines: list[str] = []
    lines.append(
        f"Across **{len(report.seeds)} independent seeds** "
        f"({', '.join(str(s) for s in report.seeds)}), "
        f"{config.duration_seconds:.0f}s each. Every arm sees byte-identical market "
        "data within a seed, so comparisons are paired."
    )
    lines.append("")
    lines.append(
        "| Arm | P&L (mean ± sd) | Sharpe (mean ± sd) | Max DD | Inventory σ "
        "| Fills | Avg spread | Profitable seeds |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for a in aggregates:
        lines.append(
            f"| **{a.label}** | {a.pnl_mean:+.3f} ± {a.pnl_std:.3f} "
            f"| {a.sharpe_mean:.1f} ± {a.sharpe_std:.1f} | {a.max_drawdown_mean:.3f} "
            f"| {a.inventory_std_mean:.5f} | {a.fills_mean:.0f} "
            f"| {a.avg_spread_mean:.4f} | {a.seeds_profitable}/{a.n_seeds} |"
        )

    # Toxicity-adjusted P&L (spec s.6.2). Splitting P&L by the VPIN regime it
    # accrued in is the only way to see whether the toxicity gate does its job:
    # a gate that widens during informed flow should show up as *less negative*
    # high-VPIN P&L, even if the overall figure is unchanged or slightly worse
    # from the fills it forgoes.
    lines.append("")
    lines.append("### Toxicity-adjusted P&L")
    lines.append("")
    lines.append(
        "P&L attributed to the regime it accrued in, split at each run's median "
        "VPIN. Increments are tagged by the VPIN at the *start* of the interval, "
        "so a loss is never attributed to the toxicity reading it helped cause."
    )
    lines.append("")
    lines.append("| Arm | P&L in high-VPIN | P&L in low-VPIN | High-VPIN share of total |")
    lines.append("|---|---:|---:|---:|")
    for a in aggregates:
        total = a.pnl_high_vpin_mean + a.pnl_low_vpin_mean
        share = (a.pnl_high_vpin_mean / total * 100.0) if abs(total) > 1e-12 else 0.0
        lines.append(
            f"| **{a.label}** | {a.pnl_high_vpin_mean:+.4f} "
            f"| {a.pnl_low_vpin_mean:+.4f} | {share:.0f}% |"
        )

    lines.append("")
    lines.append("### Paired component deltas")
    lines.append("")
    lines.append(
        "Each row isolates one mechanism by differencing it against the arm "
        "immediately below it in the build-up, seed by seed."
    )
    lines.append("")
    lines.append("| Comparison | Δ P&L (mean ± sd) | Δ Sharpe | Δ Inventory σ | Seeds won |")
    lines.append("|---|---:|---:|---:|---:|")
    comparisons = [
        ("pure_as", "naive_symmetric", "A-S vs naive"),
        ("as_toxicity", "pure_as", "Toxicity gate (vs pure A-S)"),
        ("as_drift", "pure_as", "DeepLOB drift (vs pure A-S)"),
        ("amme", "pure_as", "Full AMME (vs pure A-S)"),
        ("amme", "as_toxicity", "Drift on top of toxicity"),
    ]
    for treatment, control, label in comparisons:
        d = report.paired_deltas(treatment, control)
        if not d:
            continue
        lines.append(
            f"| {label} | {d['pnl_delta_mean']:+.3f} ± {d['pnl_delta_std']:.3f} "
            f"| {d['sharpe_delta_mean']:+.1f} | {d['inventory_std_delta_mean']:+.5f} "
            f"| {d['seeds_won']:.0f}/{d['n_seeds']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def render_markdown(results: list[ArmResult], config: BacktestConfig) -> str:
    """The results table, as it appears in ``docs/RESULTS.md``."""
    if not results:
        return "_No ablation arms produced results._\n"

    lines: list[str] = []
    lines.append("| Arm | Total P&L | Sharpe | Max DD | Inv. σ | Fills | Fill ratio "
                 "| Avg spread | P&L high-VPIN | P&L low-VPIN |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        m = r.metrics
        lines.append(
            f"| **{r.spec.label}** | {m.total_pnl:+.3f} | {m.sharpe:.1f} | {m.max_drawdown:.3f} "
            f"| {m.inventory_std:.5f} | {m.n_fills} | {m.fill_ratio:.3f} "
            f"| {m.avg_spread_quoted:.4f} | {m.pnl_high_vpin:+.3f} | {m.pnl_low_vpin:+.3f} |"
        )

    lines.append("")
    lines.append("**Arm definitions**")
    lines.append("")
    for r in results:
        lines.append(f"- `{r.spec.key}` — {r.spec.description}")

    lines.append("")
    lines.append("**Run parameters**")
    lines.append("")
    lines.append(
        f"- seed `{config.seed}`, duration `{config.duration_seconds:.0f}s`, "
        f"warmup `{config.warmup_seconds:.0f}s` (excluded from metrics)"
    )
    lines.append(
        f"- γ=`{config.gamma}`, κ=`{config.kappa}`, τ=`{config.horizon_seconds}s`, "
        f"q_max=`{config.q_max}`, quote size=`{config.quote_size}`"
    )
    lines.append(
        f"- maker fee `{config.maker_fee_bps}bps`, tick `{config.tick_size}`, "
        f"VPIN bucket `{config.vpin_bucket_volume}` × `{config.vpin_buckets}`"
    )

    identity_failures = [r.spec.key for r in results if not r.engine.pnl_identity_ok]
    if identity_failures:
        lines.append("")
        lines.append(
            f"> ⚠️ **P&L identity violated** for: {', '.join(identity_failures)}. "
            "Results are not trustworthy."
        )
    return "\n".join(lines) + "\n"


def stub_predictor() -> DriftPredictor:
    """Zero-drift predictor, so drift arms are runnable before a model exists."""

    def predict(_snapshot: FeatureSnapshot) -> tuple[float, float]:
        return 0.0, 0.0

    return predict


def model_predictor(
    onnx_path: str = "models/deeplob.onnx",
    torchscript_path: str = "models/deeplob.ts",
    alpha_scale: float = 2e-5,
    threads: int = 1,
) -> DriftPredictor | None:
    """Bridge the trained artifact into the backtester's drift arms.

    This is what makes ``as_drift`` and ``amme`` runnable at all: the harness
    needs a ``FeatureSnapshot -> (drift, confidence)`` callable, and the model
    speaks ``(T, F) tensor -> 3 class probabilities``.

    It deliberately reuses ``deeplob_inference_svc``'s own predictor and
    conversion rather than reimplementing them. Any divergence between how the
    ablation scores the model and how the live service uses it would make the
    reported edge unattributable -- the classic train/serve skew, moved one
    layer up into evaluation.

    Returns ``None`` when no artifact exists, so the runner can skip the drift
    arms rather than silently score a stub as if it were a model.
    """
    import sys

    service_dir = Path(__file__).resolve().parents[2] / "deeplob-inference-svc"
    if str(service_dir) not in sys.path:
        sys.path.insert(0, str(service_dir))

    try:
        # Cross-service import resolved via sys.path at runtime, so mypy cannot
        # see it. Deliberate: services do not depend on each other in the
        # deployed topology, but the research harness reaches for the *same*
        # inference code the live service runs, precisely so evaluation cannot
        # drift from serving.
        from deeplob_inference_svc.predictor import (
            build_predictor,
            probabilities_to_drift,
        )
    except ImportError as exc:
        log.warning("predictor_import_failed", error=str(exc))
        return None

    backend = "onnx" if Path(onnx_path).exists() else "torchscript"
    if not Path(onnx_path).exists() and not Path(torchscript_path).exists():
        log.warning("no_model_artifact", onnx=onnx_path, torchscript=torchscript_path)
        return None

    predictor = build_predictor(backend, torchscript_path, onnx_path, threads)
    if predictor.version == "stub":
        log.warning("model_load_degraded_to_stub", backend=backend)
        return None

    expected = model_input_shape(onnx_path) if backend == "onnx" else None
    log.info(
        "drift_predictor_ready",
        model=predictor.version,
        alpha_scale=alpha_scale,
        expects=expected,
    )

    def predict(snapshot: FeatureSnapshot) -> tuple[float, float]:
        tensor = snapshot.tensor
        if tensor is None:
            # Window not yet full. Zero drift is the correct neutral value --
            # the confidence floor in the quoter would reject it anyway.
            return 0.0, 0.0
        if expected is not None and tensor.shape != expected:
            # Fail loudly rather than at the ONNX layer 200 frames deep. This
            # exact mismatch (a 100-step feature window against a 50-step
            # model) already bit the live stack once; the artifact knows its
            # own shape, so the harness must be configured to match it.
            raise ValueError(
                f"feature tensor {tensor.shape} does not match what "
                f"{predictor.version} expects {expected}. Set "
                f"BacktestConfig.model_lookback/model_levels to "
                f"{expected[0]}/{expected[1] // 4}."
            )
        probabilities, _latency_us = predictor.predict(tensor)
        drift, confidence = probabilities_to_drift(probabilities, alpha_scale)
        return drift, confidence

    return predict


def model_input_shape(onnx_path: str) -> tuple[int, int] | None:
    """``(timesteps, features)`` the ONNX artifact declares it accepts.

    Reading the shape off the model rather than assuming it is what turns a
    configuration mismatch from a runtime crash deep inside onnxruntime into a
    check the harness can make before it starts.
    """
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        dims = session.get_inputs()[0].shape  # (batch, channel, T, F)
    except Exception as exc:
        log.warning("model_shape_probe_failed", error=str(exc))
        return None
    if len(dims) != 4 or not all(isinstance(d, int) for d in dims[2:]):
        return None
    return int(dims[2]), int(dims[3])


__all__ = [
    "ABLATION_ARMS",
    "ArmAggregate",
    "ArmResult",
    "ArmSpec",
    "BacktestConfig",
    "BacktestRunner",
    "MultiSeedReport",
    "render_markdown",
    "render_multi_seed_markdown",
    "stub_predictor",
]
