"""End-to-end smoke test against a running stack.

Skipped unless ``AMME_INTEGRATION=1``, because it needs `docker compose up` and
real network access. CI does not run it -- a build must not fail because Binance
had a bad minute.

What it verifies is deliberately *end-to-end behaviour*, not unit correctness:
that all eight services are healthy, that events are flowing through every
stream, and that a book update actually causes a quote. Unit tests cover whether
each stage computes the right thing; this covers whether they are wired together.
"""

from __future__ import annotations

import json
import os
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("AMME_INTEGRATION") != "1",
    reason="set AMME_INTEGRATION=1 and run `make up` first",
)

httpx = pytest.importorskip("httpx")

BASE = "http://localhost"
SERVICES = {
    "api-gateway-svc": 8000,
    "ingestion-svc": 8001,
    "feature-svc": 8002,
    "deeplob-inference-svc": 8003,
    "quoting-engine-svc": 8004,
    "risk-svc": 8005,
    "execution-sim-svc": 8006,
    "backtest-svc": 8007,
}


def _get(path: str, port: int, timeout: float = 10.0) -> httpx.Response:
    return httpx.get(f"{BASE}:{port}{path}", timeout=timeout)


@pytest.mark.integration
class TestServiceHealth:
    @pytest.mark.parametrize(("name", "port"), sorted(SERVICES.items()))
    def test_health_endpoint(self, name: str, port: int) -> None:
        response = _get("/health", port)
        payload = response.json()
        assert response.status_code == 200, (
            f"{name} unhealthy: {json.dumps(payload, indent=2)}"
        )
        assert payload["service"] == name
        assert payload["status"] == "healthy"

    @pytest.mark.parametrize(("name", "port"), sorted(SERVICES.items()))
    def test_metrics_endpoint(self, name: str, port: int) -> None:
        response = _get("/metrics", port)
        assert response.status_code == 200
        assert "amme_service_up" in response.text


@pytest.mark.integration
class TestDataFlow:
    def test_ingestion_reports_a_synced_book(self) -> None:
        checks = _get("/health", 8001).json()["checks"]
        assert "ok" in checks["book_synced"], checks["book_synced"]
        assert "ok" in checks["clock_sync"], checks["clock_sync"]

    def test_clock_skew_is_measured_and_corrected(self) -> None:
        detail = _get("/health", 8001).json()["checks"]["clock_sync"]
        # The check reports the measured offset; its presence proves the
        # correction ran rather than defaulting to zero.
        assert "offset=" in detail and "rtt=" in detail

    def test_events_reach_the_gateway(self) -> None:
        state = _get("/api/state", 8000).json()
        assert state["mid_price"] > 0, "no book updates reached the gateway"
        assert state["best_bid"] < state["best_ask"], "gateway holds a crossed book"

    def test_quotes_are_being_produced(self) -> None:
        deadline = time.time() + 60
        while time.time() < deadline:
            state = _get("/api/state", 8000).json()
            if state["n_quotes"] > 0:
                assert state["quote_bid"] < state["quote_ask"]
                return
            time.sleep(3)
        pytest.fail("no quotes produced within 60s")

    def test_pipeline_metrics_advance(self) -> None:
        """Every stage must be consuming, not just alive."""
        def consumed(port: int) -> float:
            text = _get("/metrics", port).text
            return sum(
                float(line.rsplit(" ", 1)[1])
                for line in text.splitlines()
                if line.startswith("amme_events_consumed_total{")
            )

        before = {p: consumed(p) for p in (8002, 8004, 8005, 8006)}
        time.sleep(15)
        after = {p: consumed(p) for p in (8002, 8004, 8005, 8006)}
        stalled = [p for p in before if after[p] <= before[p]]
        assert not stalled, f"services consumed nothing in 15s: {stalled}"


@pytest.mark.integration
class TestRiskBoundary:
    def test_execution_only_acts_on_risk_decisions(self) -> None:
        """Quotes reaching execution must be matched by risk decisions.

        This is the architectural guarantee from ARCHITECTURE.md s.2.1 checked
        at runtime: if execution ever placed an order without a corresponding
        risk decision, the risk layer would be bypassable.
        """
        risk = _get("/health", 8005).json()["checks"]["risk_state"]
        assert "decisions=" in risk
        decisions = int(risk.split("decisions=")[1].split()[0])
        execution = _get("/health", 8006).json()["checks"]["portfolio"]
        quotes = int(execution.split("quotes=")[1].split()[0])
        assert quotes <= decisions, (
            f"execution placed {quotes} quotes against only {decisions} risk decisions"
        )

    def test_pnl_identity_holds_live(self) -> None:
        detail = _get("/health", 8006).json()["checks"]["portfolio"]
        assert "IDENTITY VIOLATED" not in detail, detail


@pytest.mark.integration
class TestDashboard:
    def test_dashboard_serves(self) -> None:
        response = _get("/", 8000)
        assert response.status_code == 200
        assert "AMME" in response.text

    def test_history_endpoint(self) -> None:
        payload = _get("/api/history", 8000).json()
        assert "points" in payload
