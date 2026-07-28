"""api-gateway-svc: the dashboard's read model and its HTTP surface.

Two things are being defended here.

*The projection is correct.* The gateway rebuilds the whole pipeline's state
from eight streams. Every field the dashboard shows comes from exactly one
event type, so each is fed a real event and the projection asserted -- a field
wired to the wrong stream is invisible in the browser (it just shows a stale
zero) and is exactly the class of bug that ships.

*The assets are actually shipped.* The dashboard is a tree of static files
inside an installed package. If a file is not declared in ``package-data`` the
container builds, starts, passes ``/health``, and serves a blank page. That
failure is silent everywhere except a browser, so it is pinned here instead.
"""

from __future__ import annotations

import json
import re
import time
import tomllib
from pathlib import Path

import pytest
from api_gateway_svc.peers import PIPELINE
from api_gateway_svc.research import _parse_markdown_table, load_research
from api_gateway_svc.service import STATIC_DIR, ApiGatewayService, LiveState, _percentile
from fastapi.testclient import TestClient

from amme_shared.events import (
    AggressorSide,
    DriftSignal,
    FeatureVector,
    Fill,
    OrderBookUpdate,
    PortfolioState,
    PriceLevel,
    QuoteIntent,
    RiskAction,
    RiskDecision,
    Side,
    TradeTick,
)
from amme_shared.streams import Streams

REPO_ROOT = Path(__file__).resolve().parents[2]
GATEWAY_ROOT = REPO_ROOT / "services" / "api-gateway-svc"


@pytest.fixture
def service() -> ApiGatewayService:
    return ApiGatewayService()


@pytest.fixture
def client(service: ApiGatewayService) -> TestClient:
    # Deliberately not used as a context manager: entering one runs the app's
    # lifespan, which opens Redis consumer groups. These tests exercise the
    # read model and the HTTP surface, neither of which needs a broker.
    return TestClient(service.build_app())


def _kw(**overrides: object) -> dict[str, object]:
    """Envelope fields every event requires."""
    now = time.time_ns()
    base: dict[str, object] = {"symbol": "AVAXUSDT", "ts_event": now, "ts_ingest": now}
    base.update(overrides)
    return base


# ---------------------------------------------------------------- percentile --
class TestPercentile:
    def test_nearest_rank_returns_an_observed_value(self) -> None:
        # Nearest-rank, not interpolated: a reported p99 must be a latency that
        # was actually measured, never a number between two of them.
        samples = [float(i) for i in range(1, 101)]
        assert _percentile(samples, 0.50) == 50.0
        assert _percentile(samples, 0.99) == 99.0
        assert _percentile(samples, 1.0) == 100.0

    def test_empty_is_zero_not_an_error(self) -> None:
        assert _percentile([], 0.99) == 0.0

    def test_single_sample(self) -> None:
        assert _percentile([7.5], 0.99) == 7.5

    def test_unsorted_input(self) -> None:
        assert _percentile([9.0, 1.0, 5.0], 0.5) == 5.0


# ---------------------------------------------------------------- projection --
class TestProjection:
    """Each stream feeds exactly the fields the dashboard reads from it."""

    def test_order_book_populates_depth_and_staleness(
        self, service: ApiGatewayService
    ) -> None:
        book = OrderBookUpdate(
            **_kw(ts_ingest=1_000_000_000 + 63_800_000, ts_event=1_000_000_000),
            bids=[PriceLevel(price=6.50, size=10.0), PriceLevel(price=6.49, size=5.0)],
            asks=[PriceLevel(price=6.52, size=2.0), PriceLevel(price=6.53, size=8.0)],
            last_update_id=42,
        )
        service._apply(Streams.LOB_RAW, book)
        state = service.state

        assert state.mid_price == pytest.approx(6.51)
        assert state.best_bid == 6.50
        assert state.best_ask == 6.52
        assert state.spread == pytest.approx(0.02)
        assert len(state.bids) == 2 and len(state.asks) == 2
        assert state.bids[0] == {"price": 6.50, "size": 10.0}
        assert state.last_update_id == 42
        # ts_ingest is stamped in exchange time upstream, so this is the
        # clock-corrected one-way latency in milliseconds.
        p50, _ = state.staleness_percentiles()
        assert p50 == pytest.approx(63.8)

    def test_book_imbalance_is_signed_toward_the_heavier_side(
        self, service: ApiGatewayService
    ) -> None:
        service._apply(
            Streams.LOB_RAW,
            OrderBookUpdate(
                **_kw(),
                bids=[PriceLevel(price=6.50, size=30.0)],
                asks=[PriceLevel(price=6.52, size=10.0)],
                last_update_id=1,
            ),
        )
        # Bid-heavy book reads positive: (30 - 10) / 40.
        assert service.state.book_imbalance == pytest.approx(0.5)

    def test_feature_vector_carries_every_warm_flag(
        self, service: ApiGatewayService
    ) -> None:
        # The warm flags matter as much as the values: a gate armed on a
        # feature that has not warmed is a gate firing on noise.
        service._apply(
            Streams.FEATURES_READY,
            FeatureVector(
                **_kw(),
                mid_price=6.51,
                micro_price=6.505,
                spread=0.02,
                vpin=0.42,
                vpin_warm=True,
                ofi=-3.5,
                ofi_zscore=-1.8,
                ofi_warm=True,
                realized_vol=0.004,
                realized_vol_fast=0.006,
                realized_vol_baseline=0.003,
                vol_warm=False,
            ),
        )
        s = service.state
        assert (s.vpin, s.vpin_warm) == (0.42, True)
        assert (s.ofi, s.ofi_zscore, s.ofi_warm) == (-3.5, -1.8, True)
        assert (s.sigma, s.sigma_fast, s.sigma_baseline) == (0.004, 0.006, 0.003)
        assert s.vol_warm is False

    def test_drift_signal_populates_class_probabilities(
        self, service: ApiGatewayService
    ) -> None:
        service._apply(
            Streams.DRIFT_SIGNAL,
            DriftSignal(
                **_kw(),
                drift=0.00012,
                confidence=0.61,
                prob_down=0.19,
                prob_flat=0.20,
                prob_up=0.61,
                horizon_events=20,
                model_version="deeplob-v1",
                inference_latency_us=2830.0,
            ),
        )
        s = service.state
        assert (s.prob_down, s.prob_flat, s.prob_up) == (0.19, 0.20, 0.61)
        assert s.model_version == "deeplob-v1"
        assert s.as_dict()["inference_p99_ms"] == pytest.approx(2.83)

    def test_quote_intent_decomposition(self, service: ApiGatewayService) -> None:
        service._apply(
            Streams.QUOTE_INTENT,
            QuoteIntent(
                **_kw(),
                bid_price=6.49,
                ask_price=6.53,
                bid_size=1.0,
                ask_size=1.0,
                reservation_price=6.505,
                half_spread=0.02,
                inventory=0.3,
                drift_adjustment=-0.001,
                toxicity_multiplier=1.7,
                sigma=0.004,
                gamma=0.017,
                kappa=200.0,
                strategy="amme",
            ),
        )
        s = service.state
        assert (s.quote_bid, s.quote_ask) == (6.49, 6.53)
        assert s.toxicity_multiplier == 1.7
        assert s.drift_adjustment == -0.001
        assert (s.gamma, s.kappa, s.strategy) == (0.017, 200.0, "amme")
        assert s.n_quotes == 1

    def test_risk_decision_surfaces_a_tripped_breaker(
        self, service: ApiGatewayService
    ) -> None:
        service._apply(
            Streams.RISK_DECISION,
            RiskDecision(
                **_kw(),
                intent_event_id="abc",
                action=RiskAction.HALT,
                reason="vpin 0.91 above breaker 0.80",
                inventory=0.4,
                inventory_utilisation=0.8,
                breaker_tripped=True,
            ),
        )
        s = service.state
        assert s.risk_action == "halt"
        assert s.breaker_tripped is True
        assert "vpin" in s.risk_reason

    def test_portfolio_is_the_only_source_of_pnl(
        self, service: ApiGatewayService
    ) -> None:
        service._apply(
            Streams.EXEC_PORTFOLIO,
            PortfolioState(
                **_kw(),
                inventory=0.02,
                cash=-0.13,
                mid_price=6.51,
                realized_pnl=0.011,
                unrealized_pnl=0.004,
                total_pnl=0.015,
                fees_paid=0.0002,
                n_fills=17,
            ),
        )
        s = service.state
        assert s.total_pnl == 0.015
        assert s.realized_pnl + s.unrealized_pnl == pytest.approx(s.total_pnl)
        assert s.n_fills == 17

    def test_fill_and_trade_tapes_are_newest_first(
        self, service: ApiGatewayService
    ) -> None:
        for price in (6.50, 6.51):
            service._apply(
                Streams.EXEC_FILL,
                Fill(
                    **_kw(),
                    order_id="o1",
                    side=Side.BID,
                    price=price,
                    size=1.0,
                    queue_position_at_fill=0.25,
                    inventory_after=1.0,
                ),
            )
            service._apply(
                Streams.LOB_TRADES,
                TradeTick(**_kw(), price=price, size=2.0, aggressor=AggressorSide.SELL),
            )

        assert service.state.recent_fills[0]["price"] == 6.51
        assert service.state.recent_fills[0]["queue_position"] == 0.25
        assert service.state.recent_trades[0]["price"] == 6.51
        assert service.state.recent_trades[0]["aggressor"] == "sell"
        assert service.state.last_trade_side == "sell"

    def test_every_stream_increments_its_own_counter(
        self, service: ApiGatewayService
    ) -> None:
        service._apply(
            Streams.LOB_TRADES,
            TradeTick(**_kw(), price=6.5, size=1.0, aggressor=AggressorSide.BUY),
        )
        assert service.state.stream_counts[Streams.LOB_TRADES] == 1
        assert service.state.stream_counts[Streams.LOB_RAW] == 0

    def test_gateway_tails_every_stream_in_the_system(self) -> None:
        # A stream that exists but is not projected is a blind spot on the
        # dashboard. If a new stream is added, this fails until it is wired in.
        assert set(ApiGatewayService.STREAMS) == set(Streams.ALL)


class TestStreamRates:
    def test_rate_is_derived_from_the_counter_delta(self) -> None:
        state = LiveState()
        state.sample_rates()
        state.stream_counts[Streams.LOB_RAW] = 100
        time.sleep(0.05)
        rates = state.stream_rates()
        # 100 messages over ~50ms is a large but finite rate; the point is that
        # it is positive and derived, not that it hits a specific value.
        assert rates[Streams.LOB_RAW] > 0
        assert rates[Streams.EXEC_FILL] == 0.0

    def test_no_samples_yet_reports_zero_rather_than_dividing_by_zero(self) -> None:
        assert LiveState().stream_rates()[Streams.LOB_RAW] == 0.0


# ------------------------------------------------------------------- routes --
class TestHttpSurface:
    def test_index_is_served(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "AMME" in response.text

    def test_state_exposes_what_the_dashboard_reads(self, client: TestClient) -> None:
        payload = client.get("/api/state").json()
        # Sampled across every stream, so a regression that drops a whole
        # stream's contribution shows up here.
        for key in (
            "mid_price", "spread_bps", "book_imbalance", "total_pnl", "realized_pnl",
            "inventory", "inventory_utilisation", "quote_bid", "half_spread",
            "toxicity_multiplier", "drift_adjustment", "vpin", "vpin_warm",
            "ofi_zscore", "sigma", "prob_up", "drift_confidence", "model_version",
            "risk_action", "breaker_tripped", "staleness_p50_ms", "n_fills", "symbol",
        ):
            assert key in payload, f"/api/state lost {key}"

    def test_book_endpoint_shape(self, client: TestClient) -> None:
        payload = client.get("/api/book").json()
        assert set(payload) >= {"bids", "asks", "mid_price", "quote_bid", "quote_ask"}

    @pytest.mark.parametrize("path", ["/api/history", "/api/fills", "/api/trades"])
    def test_list_endpoints_are_empty_not_absent(
        self, client: TestClient, path: str
    ) -> None:
        payload = client.get(path).json()
        assert isinstance(next(iter(payload.values())), list)

    def test_config_reports_the_running_parameters(
        self, client: TestClient, service: ApiGatewayService
    ) -> None:
        payload = client.get("/api/config").json()
        assert payload["symbol"] == service.settings.market.symbol
        assert payload["risk"]["q_max"] == service.settings.risk.q_max
        assert payload["avellaneda_stoikov"]["gamma"] == service.settings.avellaneda_stoikov.gamma
        assert payload["model"]["backend"] == service.settings.model.backend

    def test_research_endpoint_serves_the_repo_artifacts(
        self, client: TestClient
    ) -> None:
        payload = client.get("/api/research").json()
        assert set(payload) >= {"ablation", "model", "venues", "latency", "available"}
        assert payload["latency"], "latency budgets are static and must always be present"

    def test_gateway_cannot_mutate_trading_state(self, client: TestClient) -> None:
        # The read-only guarantee is architectural, not documentary: assert that
        # no route accepts a mutating method. A dashboard that can write is a
        # route around the risk layer.
        app = client.app
        for route in app.routes:
            methods = getattr(route, "methods", set()) or set()
            assert not (methods & {"POST", "PUT", "PATCH", "DELETE"}), (
                f"{getattr(route, 'path', route)} accepts {methods}"
            )


# ------------------------------------------------------------------- assets --
class TestStaticAssets:
    """The blank-page failure mode: assets that exist locally but never ship."""

    def test_every_asset_referenced_by_the_page_exists(self) -> None:
        index = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        referenced = re.findall(r'(?:href|src)="/static/([^"]+)"', index)
        assert referenced, "index.html references no local assets -- selector is wrong"
        for rel in referenced:
            assert (STATIC_DIR / rel).is_file(), f"index.html references missing {rel}"

    def test_every_module_import_resolves(self) -> None:
        # ES modules are resolved by the browser at runtime, so a typo in a
        # relative import is a blank page rather than a build error.
        for module in (STATIC_DIR / "js").rglob("*.js"):
            for spec in re.findall(r'^\s*(?:import|export)[^"\']*from\s+"([^"]+)"',
                                   module.read_text(encoding="utf-8"), re.MULTILINE):
                if spec.startswith("."):
                    assert (module.parent / spec).resolve().is_file(), (
                        f"{module.name} imports missing {spec}"
                    )

    def test_every_static_file_is_covered_by_package_data(self) -> None:
        config = tomllib.loads((GATEWAY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        patterns = config["tool"]["setuptools"]["package-data"]["api_gateway_svc"]
        package_root = GATEWAY_ROOT / "api_gateway_svc"

        declared: set[Path] = set()
        for pattern in patterns:
            declared.update(p.resolve() for p in package_root.glob(pattern))

        for asset in STATIC_DIR.rglob("*"):
            if asset.is_file():
                assert asset.resolve() in declared, (
                    f"{asset.relative_to(package_root)} is not covered by package-data; "
                    "it will be missing from the installed wheel and the container "
                    "will serve a broken page."
                )


# ----------------------------------------------------------------- research --
class TestResearchLoading:
    def test_markdown_table_is_parsed_into_columns_and_rows(self) -> None:
        parsed = _parse_markdown_table(
            "prose before\n"
            "| Symbol | Makeability | Verdict |\n"
            "|---|---:|---|\n"
            "| `AVAXUSDT` | **1.41** | workable |\n"
            "| `BTCUSDT` | **0.00** | hostile |\n"
            "\ntrailing prose that must not become a row\n"
        )
        assert parsed is not None
        assert parsed["columns"] == ["Symbol", "Makeability", "Verdict"]
        assert len(parsed["rows"]) == 2
        assert parsed["rows"][0][2] == "workable"

    def test_alignment_row_is_not_mistaken_for_data(self) -> None:
        parsed = _parse_markdown_table("| A | B |\n|:--|--:|\n| 1 | 2 |\n")
        assert parsed is not None
        assert parsed["rows"] == [["1", "2"]]

    def test_missing_artifacts_degrade_to_none_not_an_exception(
        self, tmp_path: Path
    ) -> None:
        # The normal state of a fresh clone: research has simply not been run.
        payload = load_research(tmp_path)
        assert payload["ablation"] is None
        assert payload["available"] == {"ablation": False, "model": False, "venues": False}
        assert payload["latency"], "static budgets survive a missing artifacts dir"

    def test_corrupt_json_degrades_to_none(self, tmp_path: Path) -> None:
        (tmp_path / "ablation_multiseed.json").write_text("{not json", encoding="utf-8")
        assert load_research(tmp_path)["ablation"] is None

    def test_repo_artifacts_have_the_shape_the_ui_renders(self) -> None:
        payload = load_research(REPO_ROOT / "artifacts")
        ablation = payload["ablation"]
        if ablation is None:
            pytest.skip("ablation artifact not generated in this checkout")

        # The Research view indexes arms by these exact keys and reads these
        # exact fields; a rename in the harness must fail here, not in a browser.
        keys = {arm["key"] for arm in ablation["aggregates"]}
        assert {"naive_symmetric", "pure_as", "as_toxicity", "as_drift", "amme"} <= keys
        for arm in ablation["aggregates"]:
            assert {
                "label", "pnl_mean", "pnl_std", "sharpe_mean", "max_drawdown_mean",
                "inventory_std_mean", "fills_mean", "fill_ratio_mean",
                "pnl_high_vpin_mean", "pnl_low_vpin_mean", "seeds_profitable", "n_seeds",
            } <= set(arm)
        for delta in ablation["deltas"].values():
            assert {"pnl_delta_mean", "sharpe_delta_mean", "seeds_won", "n_seeds"} <= set(delta)

    def test_repo_model_metrics_have_the_shape_the_ui_renders(self) -> None:
        model = load_research(REPO_ROOT / "artifacts")["model"]
        if model is None:
            pytest.skip("model metrics not generated in this checkout")
        metrics = model["metrics"]
        assert len(metrics["test_confusion"]) == 3
        assert all(len(row) == 3 for row in metrics["test_confusion"])
        for row in metrics["history"]:
            assert {"epoch", "train_loss", "val_macro_f1"} <= set(row)


# ------------------------------------------------------------------- peers --
class TestPipelineDescription:
    def test_covers_all_eight_services(self) -> None:
        assert len(PIPELINE) == 8
        assert len({p.port for p in PIPELINE}) == 8

    def test_ports_match_the_compose_file(self) -> None:
        compose = (REPO_ROOT / "infra" / "docker-compose.yml").read_text(encoding="utf-8")
        for peer in PIPELINE:
            assert f'"{peer.port}:{peer.port}"' in compose, (
                f"{peer.name} is described on :{peer.port}, which compose does not publish"
            )

    def test_declared_streams_exist(self) -> None:
        # The topology diagram is rendered from these, so a typo would draw a
        # stream that does not exist.
        for peer in PIPELINE:
            for stream in (*peer.consumes, *peer.produces):
                assert stream in Streams.ALL, f"{peer.name} declares unknown {stream}"

    def test_every_produced_stream_has_a_consumer(self) -> None:
        produced = {s for p in PIPELINE for s in p.produces}
        consumed = {s for p in PIPELINE for s in p.consumes}
        assert produced <= consumed, f"orphaned streams: {produced - consumed}"

    def test_peer_url_template_formats(self) -> None:
        settings = ApiGatewayService().settings
        url = settings.dashboard.peer_url_template.format(
            name=PIPELINE[0].name, port=PIPELINE[0].port
        )
        assert url.startswith("http")
        assert str(PIPELINE[0].port) in url


def test_research_json_is_serialisable(tmp_path: Path) -> None:
    """The endpoint returns it verbatim, so it must survive json.dumps."""
    json.dumps(load_research(REPO_ROOT / "artifacts"))
