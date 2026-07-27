"""Configuration wiring — the bugs that only appear inside a container.

Both tests here exist because of failures that unit tests with in-process
Python objects could never catch, and that only surfaced on `docker compose up`.
"""

from __future__ import annotations

import pytest

from amme_shared.config import MarketConfig, Settings


class TestEnvironmentCoercion:
    """Environment variables are always strings.

    The original `depth_update_ms: Literal[100, 1000]` accepted the int 100 in
    tests and *rejected* the string "100" from a `.env` file, so every service
    crash-looped on startup with a validation error. Any config field must
    therefore be constructible from its string form.
    """

    def test_all_fields_parse_from_strings(self, monkeypatch):
        for key, value in {
            "AMME_MARKET_DEPTH_UPDATE_MS": "100",
            "AMME_MARKET_DEPTH_LEVELS": "10",
            "AMME_MARKET_TICK_SIZE": "0.01",
            "AMME_AS_GAMMA": "1.5",
            "AMME_AS_KAPPA": "3.0",
            "AMME_RISK_Q_MAX": "0.05",
            "AMME_RISK_ENABLED": "true",
            "AMME_TOX_ENABLED": "false",
            "AMME_MODEL_LOOKBACK": "50",
            "AMME_MODEL_BACKEND": "stub",
            "AMME_PERSIST_TICKS": "true",
            "AMME_BT_SEED": "42",
        }.items():
            monkeypatch.setenv(key, value)

        settings = Settings()
        assert settings.market.depth_update_ms == 100
        assert isinstance(settings.market.depth_update_ms, int)
        assert settings.avellaneda_stoikov.gamma == pytest.approx(1.5)
        assert settings.risk.q_max == pytest.approx(0.05)
        assert settings.risk.enabled is True
        assert settings.toxicity.enabled is False
        assert settings.model.lookback == 50

    @pytest.mark.parametrize("bad", ["250", "0", "-100"])
    def test_invalid_depth_cadence_rejected(self, monkeypatch, bad):
        monkeypatch.setenv("AMME_MARKET_DEPTH_UPDATE_MS", bad)
        with pytest.raises(Exception, match="100 or 1000"):
            MarketConfig()


class TestBinanceUrlConstruction:
    """`ws_base` must be the host root, not a stream path.

    `MarketConfig.ws_base` previously ended in `/ws` while the client appends
    `/stream?streams=...` for the combined depth+trade subscription. The result
    was `.../ws/stream?streams=...`, an HTTP 404 on every connect — which the
    reconnect loop dutifully retried forever, reporting it as a transient
    network problem rather than a permanent config error.
    """

    def test_settings_ws_base_has_no_path(self):
        base = MarketConfig().ws_base
        assert base.startswith("wss://")
        assert not base.rstrip("/").endswith("/ws"), (
            f"ws_base must be the host root; got {base!r}"
        )

    def test_client_builds_a_valid_combined_stream_url(self):
        pytest.importorskip("aiohttp")
        pytest.importorskip("websockets")
        import sys
        from pathlib import Path

        svc = Path(__file__).resolve().parents[2] / "services" / "ingestion-svc"
        sys.path.insert(0, str(svc))
        try:
            from ingestion_svc.binance import BinanceConfig
        finally:
            sys.path.remove(str(svc))

        market = MarketConfig()
        config = BinanceConfig(
            symbol=market.symbol,
            ws_base=market.ws_base,
            rest_base=market.rest_base,
            depth_update_ms=market.depth_update_ms,
        )
        url = config.stream_url
        assert url == (
            "wss://stream.binance.com:9443/stream"
            "?streams=btcusdt@depth@100ms/btcusdt@aggTrade"
        ), url
        assert "/ws/stream" not in url
        assert config.snapshot_url.startswith("https://api.binance.com/api/v3/depth?")


class TestPolymorphicDecode:
    """The bus must decode by the envelope's type tag, not the requested type.

    Services consuming several streams ask for ``BaseEvent``. Validating a
    concrete payload *as* ``BaseEvent`` fails, because the envelope forbids
    extra fields — and it fails *silently*: the consumer logs, acks, and moves
    on, so every service reports healthy while no data flows. This is the test
    that would have caught it before `docker compose up` did.
    """

    def test_concrete_type_recovered_from_envelope(self):
        from amme_shared.events import BaseEvent, OrderBookUpdate, PriceLevel
        from amme_shared.streams import _decode, _encode

        book = OrderBookUpdate(
            symbol="BTCUSDT", ts_event=1, ts_ingest=2, last_update_id=99,
            bids=[PriceLevel(price=100.0, size=1.0)],
            asks=[PriceLevel(price=101.0, size=2.0)],
        )
        decoded = _decode(_encode(book), BaseEvent)
        assert isinstance(decoded, OrderBookUpdate), type(decoded)
        assert decoded.last_update_id == 99
        assert decoded.mid_price == pytest.approx(100.5)

    @pytest.mark.parametrize("factory", [
        lambda: __import__("amme_shared.events", fromlist=["x"]).TradeTick(
            symbol="S", ts_event=1, ts_ingest=2, price=1.0, size=1.0),
        lambda: __import__("amme_shared.events", fromlist=["x"]).PortfolioState(
            symbol="S", ts_event=1, ts_ingest=2, inventory=0.0, cash=0.0,
            mid_price=1.0, realized_pnl=0.0, unrealized_pnl=0.0, total_pnl=0.0),
    ])
    def test_round_trip_for_every_event_type(self, factory):
        from amme_shared.events import BaseEvent
        from amme_shared.streams import _decode, _encode

        original = factory()
        decoded = _decode(_encode(original), BaseEvent)
        assert type(decoded) is type(original)
        assert decoded.model_dump() == original.model_dump()

    def test_every_registered_type_round_trips(self):
        """Guards the registry itself: a new event class added to `events.py`
        but not to EVENT_TYPES would decode as BaseEvent and fail in production."""
        import amme_shared.events as ev
        from amme_shared.events import EVENT_TYPES

        exported = {
            name for name, obj in vars(ev).items()
            if isinstance(obj, type) and issubclass(obj, ev.BaseEvent) and obj is not ev.BaseEvent
        }
        assert exported == set(EVENT_TYPES), (
            f"EVENT_TYPES out of sync: missing {exported - set(EVENT_TYPES)}"
        )
