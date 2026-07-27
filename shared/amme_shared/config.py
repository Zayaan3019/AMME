"""Central configuration. One `.env` file drives every service (DoD s.12).

Every knob has a defensible default so `docker compose up` works with an empty
`.env`. Values that materially change strategy behaviour are documented in
``docs/ARCHITECTURE.md`` under "Calibration defaults".
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self


def _env(prefix: str = "") -> SettingsConfigDict:
    """Shared settings behaviour: read `.env`, ignore unknown keys, stay immutable."""
    return SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        env_prefix=prefix,
    )


class MarketConfig(BaseSettings):
    model_config = _env("AMME_MARKET_")

    symbol: str = Field(default="BTCUSDT", description="Binance symbol, upper case")
    depth_levels: int = Field(default=10, gt=0, le=20, description="L2 levels tracked and published")
    ws_base: str = "wss://stream.binance.com:9443/ws"
    rest_base: str = "https://api.binance.com"
    depth_update_ms: Literal[100, 1000] = 100

    tick_size: float = Field(default=0.01, gt=0.0, description="Min price increment")
    lot_size: float = Field(default=0.00001, gt=0.0, description="Min size increment")

    maker_fee_bps: float = Field(
        default=0.0,
        description="Signed maker fee in bps. Binance VIP0 spot is +1.0; 0.0 models a "
        "rebate-neutral desk, which is the conservative default for the ablation.",
    )
    taker_fee_bps: float = Field(default=5.0, description="Signed taker fee in bps")


class AvellanedaStoikovConfig(BaseSettings):
    """Parameters of the A-S optimal quoting solution (see docs/QUANT_DERIVATIONS.md)."""

    model_config = _env("AMME_AS_")

    gamma: float = Field(
        default=0.017,
        gt=0.0,
        description="Risk aversion per full-limit position (inventory is "
        "normalised by risk.q_max before entering the A-S formulas)",
    )
    kappa: float = Field(default=200.0, gt=0.0, description="Order arrival decay: lambda = A e^{-k d}")
    horizon_mode: Literal["finite", "infinite"] = Field(
        default="infinite",
        description="'finite' uses (T-t) and terminal liquidation; 'infinite' pins "
        "(T-t) to `horizon_seconds`, which is the correct model for a desk that "
        "quotes continuously and never has a terminal time. Default: infinite.",
    )
    horizon_seconds: float = Field(default=15.0, gt=0.0)
    session_seconds: float = Field(
        default=3600.0, gt=0.0, description="Length of T in 'finite' mode"
    )

    min_half_spread_ticks: float = Field(default=1.0, ge=0.0)
    max_half_spread_ticks: float = Field(default=500.0, gt=0.0)
    quote_size: float = Field(default=0.001, gt=0.0, description="Base size per side")

    # DeepLOB drift injection: r_adj = r + lambda * drift * mid
    drift_lambda: float = Field(
        default=0.0,
        ge=0.0,
        description="0.0 disables drift injection (pure A-S baseline). Calibrated "
        "value lives in AMME_AS_DRIFT_LAMBDA; see docs/RESULTS.md grid search.",
    )
    drift_confidence_floor: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Ignore predictions below this max-softmax confidence.",
    )
    drift_max_ticks: float = Field(
        default=20.0,
        ge=0.0,
        description="Hard clamp on |drift adjustment| in ticks. A model bug must "
        "never be able to walk the reservation price away from the market.",
    )

    @model_validator(mode="after")
    def _check_spread_bounds(self) -> Self:
        if self.min_half_spread_ticks > self.max_half_spread_ticks:
            raise ValueError("min_half_spread_ticks must be <= max_half_spread_ticks")
        return self


class ToxicityConfig(BaseSettings):
    model_config = _env("AMME_TOX_")

    vpin_buckets: int = Field(default=50, gt=1, description="n in the VPIN moving average")
    vpin_bucket_volume: float = Field(
        default=1.0, gt=0.0, description="V, in base currency, per volume bucket"
    )
    vpin_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    vpin_beta: float = Field(default=2.0, ge=0.0, description="Spread widening sensitivity to VPIN")

    ofi_window: int = Field(default=50, gt=1, description="Book updates in the OFI sum")
    ofi_zscore_window: int = Field(default=500, gt=1)
    ofi_threshold: float = Field(default=2.0, ge=0.0, description="|z| beyond which OFI is 'toxic'")
    ofi_beta: float = Field(default=0.25, ge=0.0)
    ofi_skew_beta: float = Field(
        default=0.5, ge=0.0, description="How hard to lean away from the OFI direction"
    )

    max_toxicity_multiplier: float = Field(default=5.0, ge=1.0)
    enabled: bool = Field(default=True, description="False reproduces the pure-A-S baseline")


class RiskConfig(BaseSettings):
    model_config = _env("AMME_RISK_")

    q_max: float = Field(default=0.05, gt=0.0, description="Hard inventory limit, base currency")
    q_soft_frac: float = Field(
        default=0.6, gt=0.0, le=1.0, description="Skewing begins at q_soft_frac * q_max"
    )
    unwind_frac: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description="Beyond this, quote one-sided only (pure unwind)",
    )
    max_position_value: float = Field(default=10_000.0, gt=0.0, description="Notional cap, quote ccy")

    breaker_vol_multiple: float = Field(
        default=4.0, gt=0.0, description="Trip if realized vol > multiple * its rolling median"
    )
    breaker_vpin: float = Field(default=0.80, ge=0.0, le=1.0)
    breaker_drawdown: float = Field(
        default=250.0, gt=0.0, description="Trip if P&L drops this far below its high-water mark"
    )
    breaker_cooldown_seconds: float = Field(default=30.0, ge=0.0)
    enabled: bool = True


class ModelConfig(BaseSettings):
    model_config = _env("AMME_MODEL_")

    lookback: int = Field(default=100, gt=0, description="T in the (T x F) DeepLOB input")
    levels: int = Field(default=10, gt=0, description="Book levels; F = 4 * levels")
    horizon_events: int = Field(default=20, gt=0, description="k, the label smoothing horizon")
    label_alpha: float = Field(
        default=2e-5, gt=0.0, description="Threshold on smoothed return for up/down/flat"
    )
    artifact_path: str = Field(default="models/deeplob.ts", description="TorchScript archive")
    onnx_path: str = Field(default="models/deeplob.onnx")
    backend: Literal["torchscript", "onnx", "stub"] = Field(
        default="stub",
        description="'stub' emits a zero-drift, uniform-probability signal so the "
        "pipeline runs end-to-end before any model exists.",
    )
    inference_p99_budget_ms: float = Field(default=5.0, gt=0.0)
    torch_threads: int = Field(default=1, gt=0, description="Pinned: avoids p99 thread-pool jitter")


class InfraConfig(BaseSettings):
    model_config = _env("AMME_")

    redis_url: str = "redis://localhost:6379/0"
    postgres_dsn: str = "postgresql://amme:amme@localhost:5432/amme"
    log_level: str = "INFO"
    log_json: bool = True
    metrics_port: int = 8000
    consumer_group: str = "amme"
    stream_maxlen: int = Field(
        default=100_000, gt=0, description="Approximate cap per stream (XADD MAXLEN ~)"
    )
    block_ms: int = Field(default=1000, gt=0, description="XREADGROUP block timeout")
    batch_size: int = Field(default=64, gt=0)
    persist_ticks: bool = Field(default=True, description="Write lob.raw to TimescaleDB")


class BacktestConfig(BaseSettings):
    model_config = _env("AMME_BT_")

    seed: int = 42
    duration_seconds: float = Field(default=1800.0, gt=0.0)
    warmup_seconds: float = Field(default=120.0, ge=0.0, description="Excluded from metrics")
    results_dir: str = "artifacts"
    risk_free_rate: float = Field(default=0.0, description="For the Sharpe numerator")
    periods_per_year: float = Field(
        default=31_536_000.0,
        gt=0.0,
        description="Seconds per year: crypto trades 24/7, so annualisation uses "
        "wall-clock seconds, not 252 trading days.",
    )


class Settings(BaseSettings):
    """Aggregate. Instantiate once per process via :func:`get_settings`."""

    model_config = _env()

    market: MarketConfig = Field(default_factory=MarketConfig)
    avellaneda_stoikov: AvellanedaStoikovConfig = Field(default_factory=AvellanedaStoikovConfig)
    toxicity: ToxicityConfig = Field(default_factory=ToxicityConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    infra: InfraConfig = Field(default_factory=InfraConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


__all__ = [
    "AvellanedaStoikovConfig",
    "BacktestConfig",
    "InfraConfig",
    "MarketConfig",
    "ModelConfig",
    "RiskConfig",
    "Settings",
    "ToxicityConfig",
    "get_settings",
]
