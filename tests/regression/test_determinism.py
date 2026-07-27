"""Determinism and latency regression (spec s.6.1 and s.6.4).

s.6.1 asks for byte-for-byte reproducibility of P&L and inventory trajectories
from a fixed seed. That is not a nicety -- it is the property that makes every
other number in ``docs/RESULTS.md`` meaningful. If a run is not reproducible,
you cannot tell a real improvement from scheduler noise, and an ablation table
becomes decoration.

The usual sources of hidden non-determinism, and where each is prevented:

* module-level ``np.random``  -> every stochastic component takes an explicit
  ``Generator``; ``test_no_global_random_in_core`` greps for violations
* wall-clock reads           -> ``amme_core`` takes an injected ``Clock``
* dict/set iteration order   -> no float-keyed set iteration on the hot path
* thread scheduling          -> the core is single-threaded by construction
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import time

import pytest

from amme_core.engine import EngineConfig, TradingEngine
from amme_core.features import FeaturePipelineConfig
from amme_core.matching import MatchingParams
from amme_core.performance import compute_metrics
from amme_core.risk import RiskParams
from amme_core.simulator import (
    MarketSimulator,
    Regime,
    SimulatorConfig,
    default_regime_schedule,
)
from amme_core.strategies import build_strategy
from amme_core.avellaneda_stoikov import ASParams
from amme_core.toxicity import ToxicityParams

SEED = 20260727
DURATION = 120.0


def make_stream(seed: int = SEED, duration: float = DURATION):
    sim = MarketSimulator(
        SimulatorConfig(seed=seed, tick_size=0.01, sigma=0.15, spread_vol_coeff=6.0,
                        mean_limit_size=0.004, mean_market_size=0.012,
                        market_order_rate=2.86, informed_intensity=0.10),
        regime_schedule=default_regime_schedule(duration),
        regime=Regime.CALM,
    )
    return sim.run(duration)


def run_engine(strategy_name: str = "amme", seed: int = SEED):
    strategy = build_strategy(
        strategy_name,
        as_params=ASParams(gamma=1.5, kappa=3.0, horizon_seconds=15.0, tick_size=0.01,
                           inventory_scale=0.05, quote_size=0.001),
        toxicity_params=ToxicityParams(),
    )
    engine = TradingEngine(
        strategy,
        config=EngineConfig(q_max=0.05, warmup_seconds=20.0),
        feature_config=FeaturePipelineConfig(vpin_bucket_volume=0.05, vpin_buckets=30,
                                             emit_tensor=False),
        risk_params=RiskParams(q_max=0.05),
        matching_params=MatchingParams(tick_size=0.01),
    )
    return engine.run(make_stream(seed))


def trajectory_digest(result) -> str:
    """A hash over the full P&L/inventory trajectory, not just the endpoint.

    Hashing only the final P&L would pass even if the path differed wildly --
    two runs can end in the same place by coincidence. The digest covers every
    sample and every fill.
    """
    payload = {
        "points": [
            [p.ts_ns, round(p.total_pnl, 12), round(p.inventory, 12),
             round(p.mid_price, 12), round(p.vpin, 12)]
            for p in result.points
        ],
        "fills": [
            [f.ts_ns, f.side.value, round(f.price, 12), round(f.size, 12)]
            for f in result.fills
        ],
        "counters": [result.n_quotes, result.n_risk_halts, result.n_risk_rejects,
                     result.n_breaker_trips],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


class TestSimulatorDeterminism:
    def test_same_seed_gives_identical_stream(self):
        a = [(b.last_update_id, b.mid_price, b.bids[0].size, len(t))
             for b, t in make_stream(seed=7, duration=30.0)]
        b = [(x.last_update_id, x.mid_price, x.bids[0].size, len(t))
             for x, t in make_stream(seed=7, duration=30.0)]
        assert a == b

    def test_different_seeds_give_different_streams(self):
        a = [b.mid_price for b, _ in make_stream(seed=1, duration=30.0)]
        b = [x.mid_price for x, _ in make_stream(seed=2, duration=30.0)]
        assert a != b

    def test_reset_restores_the_stream(self):
        sim = MarketSimulator(SimulatorConfig(seed=99, tick_size=0.01))
        first = [b.mid_price for b, _ in sim.run_steps(80)]
        sim.reset()
        second = [b.mid_price for b, _ in sim.run_steps(80)]
        assert first == second


class TestPipelineDeterminism:
    def test_full_pipeline_is_byte_identical(self):
        """The s.6.1 assertion: same seed, same trajectory, hash for hash."""
        first = run_engine()
        second = run_engine()
        assert trajectory_digest(first) == trajectory_digest(second)

    def test_repeated_runs_are_stable(self):
        digests = {trajectory_digest(run_engine()) for _ in range(3)}
        assert len(digests) == 1

    def test_every_strategy_is_deterministic(self):
        for name in ("naive_symmetric", "pure_avellaneda_stoikov", "amme"):
            a = trajectory_digest(run_engine(name))
            b = trajectory_digest(run_engine(name))
            assert a == b, f"{name} is not deterministic"

    def test_different_seeds_diverge(self):
        assert trajectory_digest(run_engine(seed=1)) != trajectory_digest(run_engine(seed=2))

    def test_pnl_identity_holds_end_to_end(self):
        result = run_engine()
        assert result.pnl_identity_ok, "cash + q*mid != realized + unrealized"

    def test_amme_with_features_disabled_equals_pure_as(self):
        """The nesting claim in ``strategies.py``: AMME with lambda=0 and the
        toxicity gate off must reproduce pure A-S *exactly*, not approximately.
        If this drifts, the ablation is comparing different code paths."""
        as_params = ASParams(gamma=1.5, kappa=3.0, horizon_seconds=15.0, tick_size=0.01,
                             inventory_scale=0.05, quote_size=0.001, drift_lambda=0.0)
        common = {
            "config": EngineConfig(q_max=0.05, warmup_seconds=20.0),
            "feature_config": FeaturePipelineConfig(vpin_bucket_volume=0.05,
                                                    vpin_buckets=30, emit_tensor=False),
            "risk_params": RiskParams(q_max=0.05),
            "matching_params": MatchingParams(tick_size=0.01),
        }
        amme = TradingEngine(
            build_strategy("amme", as_params=as_params,
                           toxicity_params=ToxicityParams(enabled=False)),
            **common,
        ).run(make_stream())
        pure = TradingEngine(
            build_strategy("pure_avellaneda_stoikov", as_params=as_params), **common
        ).run(make_stream())

        assert [round(p.total_pnl, 10) for p in amme.points] == \
               [round(p.total_pnl, 10) for p in pure.points]
        assert len(amme.fills) == len(pure.fills)


def _dotted_name(node: ast.AST) -> str:
    """``np.random.normal`` from an Attribute/Name chain; '' if not one."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _core_calls() -> list[tuple[str, int, str]]:
    """Every function call in ``amme_core``, as ``(file, line, dotted name)``.

    AST rather than regex: an earlier regex version matched the phrase
    ``time.time()`` inside a *docstring* that was explaining the rule, and
    failed the test it was documenting. Parsing means only real call
    expressions are inspected -- comments and prose are structurally invisible.
    """
    core = pathlib.Path(__file__).resolve().parents[2] / "shared" / "amme_core"
    found: list[tuple[str, int, str]] = []
    for path in sorted(core.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _dotted_name(node.func)
                if name:
                    found.append((path.name, node.lineno, name))
    return found


class TestNoHiddenNonDeterminism:
    #: The only sanctioned numpy RNG entry points.
    ALLOWED_RNG = {"np.random.Generator", "np.random.PCG64", "np.random.default_rng"}

    def test_no_global_random_in_core(self):
        """A single ``np.random.normal(...)`` in ``amme_core`` would make every
        replay test above pass locally and fail the moment anything else in the
        process touched the global seed. Forbid it outright."""
        offenders = [
            f"{f}:{line}: {name}"
            for f, line, name in _core_calls()
            if (name.startswith(("np.random.", "numpy.random.", "random."))
                and name not in self.ALLOWED_RNG)
        ]
        assert not offenders, "global RNG use in amme_core:\n" + "\n".join(offenders)

    def test_no_wall_clock_in_core(self):
        """``amme_core`` must take time as an argument, never read it."""
        banned = {"time.time", "time.time_ns", "time.monotonic", "time.monotonic_ns",
                  "time.perf_counter", "time.perf_counter_ns", "datetime.now",
                  "datetime.utcnow", "datetime.datetime.now"}
        offenders = [
            f"{f}:{line}: {name}" for f, line, name in _core_calls() if name in banned
        ]
        assert not offenders, "wall-clock read in amme_core:\n" + "\n".join(offenders)

    def test_detector_actually_detects(self):
        """Guard the guard: a static check that can never fail is worthless.

        The two tests above pass trivially if ``_core_calls`` returns nothing --
        a wrong path, a parse error swallowed somewhere. This asserts the
        scanner really is walking real code, and that it would flag a violation
        if one existed.
        """
        calls = _core_calls()
        assert len(calls) > 100, f"scanner only found {len(calls)} calls; is the path right?"
        assert any(name in self.ALLOWED_RNG for _, _, name in calls), (
            "expected the simulator's seeded Generator/PCG64 construction to be seen"
        )
        tree = ast.parse("import time\ndef f():\n    return time.time()\n")
        names = [
            _dotted_name(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)
        ]
        assert "time.time" in names


@pytest.mark.latency
class TestLatencyBudgets:
    """s.6.4: assert per-stage latency stays inside budget.

    Budgets are set well above measured medians because CI runners are shared
    and noisy. A latency test that flakes gets disabled, and a disabled test
    protects nothing -- so these are sized to catch an order-of-magnitude
    regression, which is the failure mode that actually matters.
    """

    def _time_us(self, fn, n=2000):
        fn()  # warm caches / JIT paths
        start = time.perf_counter_ns()
        for _ in range(n):
            fn()
        return (time.perf_counter_ns() - start) / n / 1000.0

    def test_quote_computation_under_budget(self):
        from amme_core.avellaneda_stoikov import AvellanedaStoikovQuoter, MarketState

        quoter = AvellanedaStoikovQuoter(ASParams(gamma=1.5, kappa=3.0, tick_size=0.01))
        state = MarketState(mid=50_000.0, sigma=0.15, inventory=0.01)
        per_call = self._time_us(lambda: quoter.quote(state))
        assert per_call < 50.0, f"A-S quote took {per_call:.1f}us (budget 50us)"

    def test_ofi_update_under_budget(self):
        from amme_core.ofi import BestQuote, OFICalculator

        calc = OFICalculator(window=50, zscore_window=500)
        quote = BestQuote(100.0, 5.0, 100.01, 4.0)
        per_call = self._time_us(lambda: calc.update(quote))
        assert per_call < 20.0, f"OFI update took {per_call:.1f}us (budget 20us)"

    def test_vpin_update_under_budget(self):
        from amme_core.vpin import VPINCalculator
        from amme_shared.events import AggressorSide

        calc = VPINCalculator(bucket_volume=0.05, n_buckets=30)
        per_call = self._time_us(lambda: calc.update(100.0, 0.001, AggressorSide.BUY))
        assert per_call < 20.0, f"VPIN update took {per_call:.1f}us (budget 20us)"

    def test_risk_evaluation_under_budget(self):
        from amme_core.risk import ProposedQuote, RiskEngine, RiskInputs, RiskParams

        eng = RiskEngine(RiskParams(q_max=0.05))
        quote = ProposedQuote(49_999.0, 50_001.0, 0.001, 0.001, 1.0)
        inputs = RiskInputs(inventory=0.01, mid_price=50_000.0)
        per_call = self._time_us(lambda: eng.evaluate(quote, inputs))
        assert per_call < 50.0, f"risk evaluation took {per_call:.1f}us (budget 50us)"

    def test_end_to_end_throughput(self):
        """The whole pipeline must keep well ahead of a 10Hz book feed."""
        start = time.perf_counter()
        result = run_engine()
        elapsed = time.perf_counter() - start
        updates = int(DURATION * 10)
        per_update_ms = elapsed / updates * 1000.0
        assert per_update_ms < 100.0, (
            f"{per_update_ms:.2f}ms per book update -- slower than the 100ms feed cadence"
        )
        assert len(result.points) > 0


class TestMetricsStability:
    def test_metrics_are_reproducible(self):
        a = compute_metrics("amme", run_engine().points, q_max=0.05)
        b = compute_metrics("amme", run_engine().points, q_max=0.05)
        assert a.total_pnl == b.total_pnl
        assert a.sharpe == b.sharpe
        assert a.inventory_variance == b.inventory_variance
