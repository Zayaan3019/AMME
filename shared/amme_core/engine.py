"""The in-process trading loop: features -> drift -> strategy -> risk -> execution.

This is the same pipeline the eight services implement over Redis, collapsed
into one deterministic function call. Having both is not duplication -- the
*stages* are the identical objects (``FeaturePipeline``, ``AvellanedaStoikovQuoter``,
``RiskEngine``, ``MatchingEngine``); only the transport differs. The services
are thin adapters that move an event from one stage to the next; this module
calls them directly.

**Causal ordering** is the one thing that must not be got wrong, because
violating it produces a backtest that is both profitable and worthless. Each
step processes, in order:

1. Trades from step ``t`` match against quotes placed at step ``t-1``.
   We can never react to a trade before it happens.
2. Features absorb book ``t`` and trades ``t``.
3. The strategy quotes for step ``t+1`` using only information through ``t``.

The one-step lag between observing and quoting is not a modelling nicety; it is
the exchange round trip, and removing it is how backtests come to show edges
that evaporate live.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

from amme_core.features import FeaturePipeline, FeaturePipelineConfig, FeatureSnapshot
from amme_core.matching import MatchingEngine, MatchingParams, SimFill
from amme_core.performance import PnLPoint
from amme_core.risk import RiskEngine, RiskInputs, RiskOutcome, RiskParams, RiskVerdict
from amme_core.strategies import DecisionContext, Strategy
from amme_shared.events import OrderBookUpdate, Side, TradeTick

#: A drift predictor: features in, ``(expected_return, confidence)`` out.
DriftPredictor = Callable[[FeatureSnapshot], tuple[float, float]]

MarketStream = Iterable[tuple[OrderBookUpdate, list[TradeTick]]]


@dataclass(frozen=True, slots=True)
class EngineConfig:
    q_max: float = 0.05
    warmup_seconds: float = 120.0
    #: Emit a P&L sample at most this often. Decoupled from the book update rate
    #: so metrics are comparable across strategies that quote at different rates.
    sample_interval_seconds: float = 1.0
    #: Minimum sigma fed to the quoter. Before the volatility estimator warms up
    #: sigma is ~0, which collapses the A-S inventory term to nothing and would
    #: have the maker quoting at one tick with no inventory control at all.
    sigma_floor: float = 1e-4
    requote_every_n_updates: int = 1


@dataclass(slots=True)
class EngineResult:
    strategy: str
    points: list[PnLPoint] = field(default_factory=list)
    fills: list[SimFill] = field(default_factory=list)
    #: Orders actually rested at the exchange.
    n_quotes: int = 0
    #: Quoting decisions taken, whether or not they changed the resting orders.
    n_decisions: int = 0
    n_risk_rejects: int = 0
    n_risk_halts: int = 0
    n_risk_amends: int = 0
    n_breaker_trips: int = 0
    fees_paid: float = 0.0
    volume_traded: float = 0.0
    final_inventory: float = 0.0
    final_pnl: float = 0.0
    pnl_identity_ok: bool = True

    @property
    def n_fills(self) -> int:
        return len(self.fills)


class TradingEngine:
    """Composes the full pipeline for one strategy over one market stream."""

    __slots__ = (
        "_last_sample_ns",
        "_start_ns",
        "config",
        "features",
        "matching",
        "predictor",
        "result",
        "risk",
        "strategy",
    )

    def __init__(
        self,
        strategy: Strategy,
        *,
        config: EngineConfig | None = None,
        feature_config: FeaturePipelineConfig | None = None,
        risk_params: RiskParams | None = None,
        matching_params: MatchingParams | None = None,
        predictor: DriftPredictor | None = None,
    ) -> None:
        self.strategy = strategy
        self.config = config or EngineConfig()
        self.features = FeaturePipeline(feature_config)
        self.risk = RiskEngine(risk_params)
        self.matching = MatchingEngine(matching_params)
        self.predictor = predictor
        self.result = EngineResult(strategy=strategy.name)
        self._start_ns = 0
        self._last_sample_ns = 0

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _level_map(levels: Iterable[object]) -> dict[float, float]:
        return {lv.price: lv.size for lv in levels}  # type: ignore[attr-defined]

    def _level_size_at(self, book: OrderBookUpdate, side: Side, price: float) -> float:
        levels = book.bids if side is Side.BID else book.asks
        for lv in levels:
            if lv.price == price:
                return lv.size
        return 0.0

    # -- main loop ---------------------------------------------------------
    def run(self, stream: MarketStream) -> EngineResult:
        for _ in self.iter_run(stream):
            pass
        return self.result

    def iter_run(self, stream: MarketStream) -> Iterator[PnLPoint]:
        """Drive the pipeline, yielding each P&L sample as it is produced."""
        cfg = self.config
        update_count = 0

        for book, trades in stream:
            if self._start_ns == 0:
                self._start_ns = book.ts_event
                self._last_sample_ns = book.ts_event

            update_count += 1

            # --- 1. Match step-t trades against quotes resting from t-1 ----
            self.matching.on_book_update(
                self._level_map(book.bids), self._level_map(book.asks)
            )
            for trade in trades:
                fills = self.matching.on_trade(
                    trade.price, trade.size, trade.aggressor, trade.ts_event
                )
                for fill in fills:
                    self.result.fills.append(fill)
                    self.result.fees_paid += fill.fee
                    self.result.volume_traded += fill.size
            self.matching.prune()

            # --- 2. Features absorb book t and trades t --------------------
            for trade in trades:
                self.features.on_raw_trade(trade.price, trade.size, trade.aggressor)
            snapshot = self.features.on_book(book)
            if snapshot is None:
                continue

            mid = snapshot.mid_price
            elapsed = (book.ts_event - self._start_ns) / 1e9
            portfolio = self.matching.portfolio

            # --- 3. Quote for t+1 using information through t --------------
            if update_count % cfg.requote_every_n_updates == 0:
                self._quote(book, snapshot, elapsed, mid)

            # --- 4. Sample ------------------------------------------------
            if book.ts_event - self._last_sample_ns >= cfg.sample_interval_seconds * 1e9:
                self._last_sample_ns = book.ts_event
                point = PnLPoint(
                    ts_ns=book.ts_event,
                    total_pnl=portfolio.total_pnl(mid),
                    realized_pnl=portfolio.realized_pnl,
                    inventory=portfolio.inventory,
                    mid_price=mid,
                    vpin=snapshot.vpin,
                    spread_quoted=self._current_quoted_spread(),
                    n_fills=len(self.result.fills),
                    n_quotes=self.result.n_quotes,
                )
                if elapsed >= cfg.warmup_seconds:
                    self.result.points.append(point)
                yield point

        self._finalise()
        return

    def _quote(
        self, book: OrderBookUpdate, snapshot: FeatureSnapshot, elapsed: float, mid: float
    ) -> None:
        cfg = self.config
        portfolio = self.matching.portfolio

        drift, confidence = (0.0, 0.0)
        if self.predictor is not None:
            drift, confidence = self.predictor(snapshot)

        ctx = DecisionContext(
            mid=mid,
            micro_price=snapshot.micro_price,
            sigma=max(snapshot.sigma, cfg.sigma_floor),
            inventory=portfolio.inventory,
            elapsed_seconds=elapsed,
            best_bid=book.bids[0].price,
            best_ask=book.asks[0].price,
            vpin=snapshot.vpin,
            vpin_warm=snapshot.vpin_warm,
            ofi_zscore=snapshot.ofi_zscore,
            ofi_warm=snapshot.ofi_warm,
            drift=drift,
            drift_confidence=confidence,
        )
        quote = self.strategy.decide(ctx)

        outcome = self._apply_risk(quote, snapshot, portfolio.total_pnl(mid), mid, book.ts_event)

        if outcome.verdict is RiskVerdict.HALT:
            self.matching.cancel_all()
            self.result.n_risk_halts += 1
            return
        if outcome.verdict is RiskVerdict.REJECT:
            self.matching.cancel_all()
            self.result.n_risk_rejects += 1
            return
        if outcome.verdict is RiskVerdict.AMEND:
            self.result.n_risk_amends += 1

        bid_price = outcome.bid_price
        ask_price = outcome.ask_price
        bid_size = outcome.bid_size or 0.0
        ask_size = outcome.ask_size or 0.0

        self.result.n_quotes += self.matching.replace_quotes(
            bid_price=bid_price if bid_size > 0 else None,
            bid_size=bid_size,
            ask_price=ask_price if ask_size > 0 else None,
            ask_size=ask_size,
            bid_level_size=self._level_size_at(book, Side.BID, bid_price or 0.0),
            ask_level_size=self._level_size_at(book, Side.ASK, ask_price or 0.0),
            ts_ns=book.ts_event,
        )
        self.result.n_decisions += 1

    def _apply_risk(
        self,
        quote: object,
        snapshot: FeatureSnapshot,
        total_pnl: float,
        mid: float,
        ts_ns: int,
    ) -> RiskOutcome:
        from amme_core.risk import ProposedQuote

        proposed = ProposedQuote(
            bid_price=quote.bid_price,  # type: ignore[attr-defined]
            ask_price=quote.ask_price,  # type: ignore[attr-defined]
            bid_size=quote.bid_size,  # type: ignore[attr-defined]
            ask_size=quote.ask_size,  # type: ignore[attr-defined]
            half_spread=quote.half_spread,  # type: ignore[attr-defined]
        )
        inputs = RiskInputs(
            inventory=self.matching.portfolio.inventory,
            mid_price=mid,
            total_pnl=total_pnl,
            realized_vol=snapshot.sigma_fast,
            # Only arm the vol-spike test once the slow baseline is meaningful.
            # A baseline of 0 would make every reading an infinite multiple.
            baseline_vol=snapshot.sigma_baseline if snapshot.vol_warm else 0.0,
            vpin=snapshot.vpin if snapshot.vpin_warm else 0.0,
            ts_ns=ts_ns,
        )
        before = self.risk.breaker.trip_count
        outcome = self.risk.evaluate(proposed, inputs)
        self.result.n_breaker_trips += self.risk.breaker.trip_count - before
        return outcome

    def _current_quoted_spread(self) -> float:
        bid = self.matching.active_order(Side.BID)
        ask = self.matching.active_order(Side.ASK)
        if bid is None or ask is None:
            return 0.0
        return ask.price - bid.price

    def _finalise(self) -> None:
        portfolio = self.matching.portfolio
        last_mid = self.result.points[-1].mid_price if self.result.points else 0.0
        self.result.final_inventory = portfolio.inventory
        self.result.final_pnl = portfolio.total_pnl(last_mid) if last_mid else portfolio.cash
        self.result.fees_paid = portfolio.fees_paid
        self.result.volume_traded = portfolio.volume_traded
        if last_mid:
            self.result.pnl_identity_ok = portfolio.check_identity(last_mid)

    def reset(self) -> None:
        self.features.reset()
        self.risk.reset()
        self.matching.reset()
        self.strategy.reset()
        self.result = EngineResult(strategy=self.strategy.name)
        self._start_ns = 0
        self._last_sample_ns = 0


__all__ = ["DriftPredictor", "EngineConfig", "EngineResult", "MarketStream", "TradingEngine"]
