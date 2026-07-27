"""Versioned event schemas -- the wire contract between every AMME service.

Design rules (enforced by review, and by ``tests/unit/test_event_schemas.py``):

* Every event carries ``schema_version``. Consumers MUST reject majors they do
  not understand rather than silently mis-parsing. Adding an optional field is a
  minor bump; removing/retyping a field is a major bump.
* Every event carries ``correlation_id``, propagated unchanged from the
  order-book update that ultimately caused it. That single field is what makes a
  quote traceable back to the exact tick that produced it, across eight
  processes -- see ``docs/ARCHITECTURE.md`` s."Observability".
* Timestamps are explicit about their origin: ``ts_event`` is exchange/simulator
  time, ``ts_ingest`` is when *we* first saw it. Their difference is the only
  honest measure of end-to-end staleness.

Numeric representation: prices and sizes are ``float`` (IEEE-754 double), not
``Decimal``. Rationale in ``docs/ARCHITECTURE.md`` -- doubles carry ~15-16
significant digits against a worst case of ~10 for crypto prices, the engine
never does repeated accumulation without re-basing, and ``Decimal`` costs ~50x
on the hot path. Exchange-side tick/lot rounding is applied explicitly at the
quote boundary (``amme_core.quantize``) rather than being implicitly relied upon.
"""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Current major schema version for the whole event family. Bump deliberately.
SCHEMA_VERSION: Literal[1] = 1

Price = Annotated[float, Field(gt=0.0, description="Price in quote currency")]
Size = Annotated[float, Field(ge=0.0, description="Size in base currency")]


def new_id() -> str:
    """Fresh opaque identifier. Hex (not UUID str) to keep the wire payload small."""
    return uuid.uuid4().hex


class Side(StrEnum):
    BID = "bid"
    ASK = "ask"


class AggressorSide(StrEnum):
    """Which side *initiated* a trade (i.e. crossed the spread)."""

    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"


class DataSource(StrEnum):
    BINANCE_WS = "binance_ws"
    SIMULATOR = "simulator"
    REPLAY = "replay"
    LOBSTER = "lobster"


class BaseEvent(BaseModel):
    """Common envelope. All AMME events inherit this."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)

    schema_version: int = Field(default=SCHEMA_VERSION)
    event_id: str = Field(default_factory=new_id)
    correlation_id: str = Field(
        default_factory=new_id,
        description="Propagated unchanged along a causal chain of events.",
    )
    symbol: str
    ts_event: int = Field(description="Source-of-truth timestamp, epoch nanoseconds")
    ts_ingest: int = Field(description="Local receipt timestamp, epoch nanoseconds")
    source: DataSource = DataSource.BINANCE_WS

    @property
    def staleness_ns(self) -> int:
        """How far behind the source we were when we first saw this event."""
        return self.ts_ingest - self.ts_event


class PriceLevel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    price: Price
    size: Size


# ---------------------------------------------------------------------------
# lob.raw -- produced by ingestion-svc
# ---------------------------------------------------------------------------
class OrderBookUpdate(BaseEvent):
    """A consistent L2 snapshot of the top-N book after applying a diff.

    ingestion-svc publishes *state*, not diffs. That is a deliberate choice: it
    makes every downstream consumer stateless with respect to book
    reconstruction, so feature-svc can be restarted mid-stream and be correct on
    its very first message. The cost is bandwidth, which is irrelevant at
    100ms x 10 levels.
    """

    bids: list[PriceLevel] = Field(description="Descending by price; index 0 is best bid")
    asks: list[PriceLevel] = Field(description="Ascending by price; index 0 is best ask")
    last_update_id: int = Field(description="Exchange sequence number (Binance: lastUpdateId)")
    is_snapshot: bool = Field(
        default=False, description="True when emitted directly after a REST resync"
    )

    @model_validator(mode="after")
    def _check_book_integrity(self) -> Self:
        if self.bids and self.asks and self.bids[0].price >= self.asks[0].price:
            raise ValueError(
                f"crossed book: bid {self.bids[0].price} >= ask {self.asks[0].price}"
            )
        for name, levels, descending in (
            ("bids", self.bids, True),
            ("asks", self.asks, False),
        ):
            for prev, cur in zip(levels, levels[1:], strict=False):
                ordered = prev.price > cur.price if descending else prev.price < cur.price
                if not ordered:
                    raise ValueError(f"{name} not strictly ordered at {prev.price}->{cur.price}")
        return self

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return 0.5 * (self.bids[0].price + self.asks[0].price)

    @property
    def micro_price(self) -> float | None:
        """Size-weighted mid. Leans toward the side with less resting size."""
        if not self.bids or not self.asks:
            return None
        bp, bs = self.bids[0].price, self.bids[0].size
        ap, as_ = self.asks[0].price, self.asks[0].size
        total = bs + as_
        if total <= 0.0:
            return 0.5 * (bp + ap)
        return (bp * as_ + ap * bs) / total

    @property
    def spread(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return self.asks[0].price - self.bids[0].price


# ---------------------------------------------------------------------------
# lob.trades -- produced by ingestion-svc
# ---------------------------------------------------------------------------
class TradeTick(BaseEvent):
    """A public trade print. Feeds VPIN bucketing and the fill simulator."""

    price: Price
    size: Size
    aggressor: AggressorSide = AggressorSide.UNKNOWN
    trade_id: int = 0


# ---------------------------------------------------------------------------
# features.ready -- produced by feature-svc
# ---------------------------------------------------------------------------
class FeatureVector(BaseEvent):
    """Derived microstructure state at one point in time."""

    mid_price: Price
    micro_price: Price
    spread: float = Field(ge=0.0)

    # Toxicity / flow
    vpin: float = Field(ge=0.0, le=1.0, description="Volume-synchronised prob. of informed trading")
    vpin_warm: bool = Field(
        default=False, description="False until enough volume buckets have filled"
    )
    ofi: float = Field(description="Order flow imbalance over the configured window (Cont et al.)")
    ofi_zscore: float = Field(default=0.0, description="OFI standardised by its rolling stdev")

    # Volatility, in absolute quote-currency units per sqrt(second) -- the
    # convention Avellaneda-Stoikov requires (arithmetic BM), NOT a return vol.
    realized_vol: float = Field(ge=0.0, description="Rolling-window sigma; feeds the quoter")
    realized_vol_fast: float = Field(
        default=0.0, ge=0.0, description="EWMA sigma; reacts fast, feeds the breaker"
    )
    realized_vol_baseline: float = Field(
        default=0.0,
        ge=0.0,
        description="Slow reference sigma. The breaker trips on fast/baseline, so "
        "both must travel together or the ratio is meaningless.",
    )
    vol_warm: bool = Field(default=False)

    # Raw normalised tensor for DeepLOB, flattened row-major (T x F).
    lob_tensor: list[float] = Field(default_factory=list)
    tensor_shape: tuple[int, int] | None = Field(
        default=None, description="(timesteps, features) of lob_tensor; None when not emitted"
    )

    @model_validator(mode="after")
    def _check_tensor(self) -> Self:
        if self.tensor_shape is not None:
            t, f = self.tensor_shape
            if t * f != len(self.lob_tensor):
                raise ValueError(
                    f"tensor_shape {self.tensor_shape} implies {t * f} values, "
                    f"got {len(self.lob_tensor)}"
                )
        return self


# ---------------------------------------------------------------------------
# drift.signal -- produced by deeplob-inference-svc
# ---------------------------------------------------------------------------
class DriftSignal(BaseEvent):
    """DeepLOB's view of short-horizon mid-price drift.

    ``drift`` is a *dimensionless expected return* over ``horizon_events``, so
    the quoting engine converts to price units by multiplying by mid. Keeping it
    dimensionless means the same lambda calibration transfers across symbols and
    price regimes.
    """

    drift: float = Field(description="E[(m_{t+h} - m_t) / m_t], dimensionless")
    confidence: float = Field(ge=0.0, le=1.0, description="max softmax prob over the 3 classes")
    prob_down: float = Field(ge=0.0, le=1.0)
    prob_flat: float = Field(ge=0.0, le=1.0)
    prob_up: float = Field(ge=0.0, le=1.0)
    horizon_events: int = Field(gt=0, description="Prediction horizon in book updates")
    model_version: str = Field(default="untrained")
    inference_latency_us: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _check_probs(self) -> Self:
        total = self.prob_down + self.prob_flat + self.prob_up
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"class probabilities must sum to 1, got {total}")
        return self


# ---------------------------------------------------------------------------
# quote.intent -- produced by quoting-engine-svc
# ---------------------------------------------------------------------------
class QuoteIntent(BaseEvent):
    """A desired two-sided quote, *before* risk approval.

    Naming matters here: the quoting engine expresses an *intent*. Only risk-svc
    can turn an intent into something the execution layer will act on. See
    ``docs/ARCHITECTURE.md`` s."Why risk-svc is a separate process".
    """

    bid_price: Price
    ask_price: Price
    bid_size: Size
    ask_size: Size

    # Decomposition, carried for observability and for the ablation report.
    reservation_price: float
    half_spread: float = Field(ge=0.0)
    inventory: float
    drift_adjustment: float = Field(default=0.0, description="lambda * drift * mid, in price units")
    toxicity_multiplier: float = Field(default=1.0, ge=0.0)
    sigma: float = Field(ge=0.0)
    gamma: float = Field(gt=0.0)
    kappa: float = Field(gt=0.0)
    strategy: str = Field(default="amme")

    @model_validator(mode="after")
    def _check_not_crossed(self) -> Self:
        if self.bid_price >= self.ask_price:
            raise ValueError(f"crossed quote: bid {self.bid_price} >= ask {self.ask_price}")
        return self


class RiskAction(StrEnum):
    APPROVE = "approve"
    AMEND = "amend"  # sizes/prices skewed to reduce inventory
    REJECT = "reject"  # this quote only
    HALT = "halt"  # circuit breaker: pull everything


# ---------------------------------------------------------------------------
# risk.decision -- produced by risk-svc
# ---------------------------------------------------------------------------
class RiskDecision(BaseEvent):
    """risk-svc's verdict on a QuoteIntent. Execution only acts on this."""

    intent_event_id: str
    action: RiskAction
    reason: str = ""

    # Populated for APPROVE/AMEND; None for REJECT/HALT.
    bid_price: float | None = None
    ask_price: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None

    inventory: float = 0.0
    inventory_utilisation: float = Field(default=0.0, description="|q| / q_max")
    breaker_tripped: bool = False

    @model_validator(mode="after")
    def _check_payload(self) -> Self:
        actionable = self.action in (RiskAction.APPROVE, RiskAction.AMEND)
        has_quote = None not in (self.bid_price, self.ask_price, self.bid_size, self.ask_size)
        if actionable and not has_quote:
            raise ValueError(f"action={self.action} requires a full quote payload")
        if actionable and self.bid_price is not None and self.ask_price is not None:
            if self.bid_price >= self.ask_price:
                raise ValueError("crossed quote after risk amendment")
        return self


# ---------------------------------------------------------------------------
# exec.fill / exec.portfolio -- produced by execution-sim-svc
# ---------------------------------------------------------------------------
class Fill(BaseEvent):
    """A simulated passive fill of one of our resting quotes."""

    order_id: str
    side: Side
    price: Price
    size: Size
    fee: float = Field(default=0.0, description="Signed: negative = rebate earned")
    queue_position_at_fill: float = Field(default=0.0, ge=0.0)
    inventory_after: float = 0.0
    cash_after: float = 0.0


class PortfolioState(BaseEvent):
    """Mark-to-market snapshot. The single source of truth for P&L reporting."""

    inventory: float
    cash: float
    mid_price: Price
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    fees_paid: float = 0.0
    n_fills: int = 0
    n_quotes: int = 0

    @model_validator(mode="after")
    def _check_pnl_identity(self) -> Self:
        expected = self.realized_pnl + self.unrealized_pnl
        if abs(expected - self.total_pnl) > 1e-6 * max(1.0, abs(self.total_pnl)):
            raise ValueError(
                f"P&L identity violated: realized {self.realized_pnl} + unrealized "
                f"{self.unrealized_pnl} != total {self.total_pnl}"
            )
        return self


EVENT_TYPES: dict[str, type[BaseEvent]] = {
    "OrderBookUpdate": OrderBookUpdate,
    "TradeTick": TradeTick,
    "FeatureVector": FeatureVector,
    "DriftSignal": DriftSignal,
    "QuoteIntent": QuoteIntent,
    "RiskDecision": RiskDecision,
    "Fill": Fill,
    "PortfolioState": PortfolioState,
}

__all__ = [
    "EVENT_TYPES",
    "SCHEMA_VERSION",
    "AggressorSide",
    "BaseEvent",
    "DataSource",
    "DriftSignal",
    "Fill",
    "FeatureVector",
    "OrderBookUpdate",
    "PortfolioState",
    "Price",
    "PriceLevel",
    "QuoteIntent",
    "RiskAction",
    "RiskDecision",
    "Side",
    "Size",
    "TradeTick",
    "new_id",
]
