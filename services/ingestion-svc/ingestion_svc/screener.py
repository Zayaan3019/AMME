"""Venue screening: find instruments where passive market making is viable.

This exists because of a measurement, not a hunch. Live BTCUSDT has its spread
pinned at **exactly one tick at both p50 and p90** while sigma is ~17 ticks per
100 ms step. The tick is *binding* -- participants want a tighter spread than
the venue permits -- so adverse selection exceeds spread capture by more than an
order of magnitude and no passive quoter has positive expectancy there. Real
BTCUSDT makers live on fee tiers and latency, not on the quoted spread.

Documenting that is not a fix. The fix is to be able to *choose* an instrument,
which is what this module does.

**The screening ratio.** A maker earns roughly half the spread per round trip
and pays for the price drift accumulated while holding inventory. Over a holding
horizon :math:`h`:

.. math::

    \\text{edge} \\;\\propto\\; \\frac{\\delta/2}{\\sigma\\sqrt{h}}

so the quantity that decides viability is the spread measured in units of the
volatility over the holding period -- not the spread in ticks, and not the
spread in basis points. Both of those rank symbols wrongly: a wide spread on a
violently volatile asset is worse than a narrow one on a calm asset.

We report it as ``makeability = spread / (sigma * sqrt(holding_seconds))``.

* ``< 0.5``  hostile. Adverse selection dominates; do not quote passively.
* ``0.5-1.0`` marginal. Viable only with rebates or real speed.
* ``> 1.0``  workable. Classical A-S economics apply.

**Why REST and not the WebSocket.** Screening twenty symbols over a WebSocket
means twenty concurrent connections and minutes of warm-up. Depth, klines and
aggTrades give the same three quantities in three cheap calls per symbol, and
the ranking only needs to be approximately right -- the chosen symbol is then
validated properly by the live pipeline.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import aiohttp

from amme_shared.logging import get_logger

log = get_logger("ingestion.screener")

#: Binance-wide rate limit is 1200 weight/minute; each symbol costs ~7. Twenty
#: symbols is ~140 weight, comfortably inside it, but requests are still
#: serialised through a semaphore so a large candidate list cannot trip a ban.
_MAX_CONCURRENCY = 4

DEFAULT_CANDIDATES: tuple[str, ...] = (
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "LINKUSDT", "AVAXUSDT", "DOTUSDT",
    "MATICUSDT", "LTCUSDT", "ATOMUSDT", "UNIUSDT", "FILUSDT",
)


@dataclass(frozen=True, slots=True)
class VenueProfile:
    """Everything the screen measured for one symbol."""

    symbol: str
    mid_price: float
    tick_size: float
    spread: float
    spread_ticks: float
    spread_bps: float
    #: Quote-currency units per sqrt(second), from 1m klines.
    sigma: float
    sigma_bps_per_sqrt_s: float
    touch_depth: float
    trade_rate_hz: float
    mean_trade_size: float
    volume_per_second: float
    min_notional: float

    #: How long our quote size would take to clear the touch queue, in seconds.
    #: A proxy for the holding horizon, and the denominator's time scale.
    queue_clear_seconds: float
    makeability: float

    @property
    def verdict(self) -> str:
        if self.makeability >= 1.0:
            return "workable"
        if self.makeability >= 0.5:
            return "marginal"
        return "hostile"

    @property
    def tick_is_binding(self) -> bool:
        """A spread that cannot compress below one tick is a venue constraint,
        not a market equilibrium -- and it caps the maker's revenue."""
        return self.spread_ticks <= 1.0 + 1e-9


async def _get_json(
    session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore
) -> Any | None:
    async with semaphore:
        try:
            async with session.get(url) as response:
                if response.status == 429:
                    retry = float(response.headers.get("Retry-After", 2.0))
                    log.warning("screener_rate_limited", retry_after=retry)
                    await asyncio.sleep(retry)
                    return None
                response.raise_for_status()
                payload: Any = await response.json()
                return payload
        except (TimeoutError, aiohttp.ClientError, ValueError) as exc:
            log.warning("screener_request_failed", url=url[:70], error=str(exc))
            return None


def _sigma_from_klines(klines: list[Any], interval_seconds: float) -> float:
    """Realized volatility rate in price units per sqrt(second).

    Uses close-to-close differences over the bar series, which is the same
    arithmetic-BM diffusion estimate the quoter consumes -- so the number is
    directly comparable with the sigma feature-svc computes live.
    """
    closes = [float(k[4]) for k in klines if len(k) > 4]
    if len(closes) < 3:
        return 0.0
    diffs = [b - a for a, b in pairwise(closes)]
    mean_sq = sum(d * d for d in diffs) / len(diffs)
    return math.sqrt(mean_sq / interval_seconds)


async def profile_symbol(
    session: aiohttp.ClientSession,
    symbol: str,
    semaphore: asyncio.Semaphore,
    rest_base: str = "https://api.binance.com",
) -> VenueProfile | None:
    """Measure one symbol. ``None`` if any required call failed."""
    depth, klines, trades, info = await asyncio.gather(
        _get_json(session, f"{rest_base}/api/v3/depth?symbol={symbol}&limit=20", semaphore),
        _get_json(
            session, f"{rest_base}/api/v3/klines?symbol={symbol}&interval=1m&limit=120",
            semaphore,
        ),
        _get_json(session, f"{rest_base}/api/v3/aggTrades?symbol={symbol}&limit=500", semaphore),
        _get_json(session, f"{rest_base}/api/v3/exchangeInfo?symbol={symbol}", semaphore),
    )
    if not (depth and klines and trades and info):
        return None

    try:
        best_bid, bid_size = float(depth["bids"][0][0]), float(depth["bids"][0][1])
        best_ask, ask_size = float(depth["asks"][0][0]), float(depth["asks"][0][1])
    except (KeyError, IndexError, ValueError):
        return None

    mid = 0.5 * (best_bid + best_ask)
    spread = best_ask - best_bid
    if mid <= 0 or spread <= 0:
        return None

    entry = info["symbols"][0]
    filters = {f["filterType"]: f for f in entry.get("filters", [])}
    tick = float(filters.get("PRICE_FILTER", {}).get("tickSize", 0.01))
    min_notional = float(
        (filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {}).get("minNotional", 0.0)
    )

    sigma = _sigma_from_klines(klines, interval_seconds=60.0)

    sizes = [float(t["q"]) for t in trades]
    first_ms, last_ms = int(trades[0]["T"]), int(trades[-1]["T"])
    span = max(1e-3, (last_ms - first_ms) / 1000.0)
    trade_rate = len(trades) / span
    volume_rate = sum(sizes) / span

    touch_depth = min(bid_size, ask_size)
    # Half the flow hits each side, so our queue clears at ~volume_rate/2.
    queue_clear = touch_depth / (volume_rate / 2.0) if volume_rate > 0 else math.inf
    # Cap the horizon: beyond a couple of minutes a maker would amend rather
    # than wait, so a huge queue should not flatter the ratio via sqrt().
    holding = min(max(queue_clear, 1.0), 120.0)

    makeability = spread / (sigma * math.sqrt(holding)) if sigma > 0 else 0.0

    return VenueProfile(
        symbol=symbol,
        mid_price=mid,
        tick_size=tick,
        spread=spread,
        spread_ticks=spread / tick,
        spread_bps=spread / mid * 10_000.0,
        sigma=sigma,
        sigma_bps_per_sqrt_s=sigma / mid * 10_000.0,
        touch_depth=touch_depth,
        trade_rate_hz=trade_rate,
        mean_trade_size=sum(sizes) / len(sizes),
        volume_per_second=volume_rate,
        min_notional=min_notional,
        queue_clear_seconds=queue_clear,
        makeability=makeability,
    )


async def screen(
    symbols: tuple[str, ...] = DEFAULT_CANDIDATES,
    rest_base: str = "https://api.binance.com",
) -> list[VenueProfile]:
    """Profile every candidate, best-first by makeability."""
    semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=20)
    ) as session:
        results = await asyncio.gather(
            *(profile_symbol(session, s, semaphore, rest_base) for s in symbols)
        )
    profiles = [p for p in results if p is not None]
    profiles.sort(key=lambda p: p.makeability, reverse=True)
    return profiles


def render_table(profiles: list[VenueProfile]) -> str:
    lines = [
        "| Symbol | Mid | Spread (ticks) | Spread (bps) | σ (bps/√s) | Queue clear | "
        "Makeability | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for p in profiles:
        binding = " ⚠" if p.tick_is_binding else ""
        lines.append(
            f"| `{p.symbol}` | {p.mid_price:,.4f} | {p.spread_ticks:.1f}{binding} "
            f"| {p.spread_bps:.2f} | {p.sigma_bps_per_sqrt_s:.3f} "
            f"| {p.queue_clear_seconds:.1f}s | **{p.makeability:.2f}** | {p.verdict} |"
        )
    lines.append("")
    lines.append(
        "`makeability = spread / (σ·√holding)`. **> 1.0** workable, "
        "**0.5–1.0** marginal, **< 0.5** hostile. ⚠ marks a *binding tick*: "
        "the spread cannot compress below one tick, so the venue itself caps "
        "maker revenue."
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_CANDIDATES",
    "VenueProfile",
    "profile_symbol",
    "render_table",
    "screen",
]


def main() -> int:
    """``python -m ingestion_svc.screener`` -- rank venues, write the report."""
    import pathlib

    from amme_shared.logging import configure_logging

    configure_logging("screener", "WARNING", json_output=False)
    profiles = asyncio.run(screen())
    if not profiles:
        print("no symbols could be profiled (network?)")
        return 1

    table = render_table(profiles)
    out = pathlib.Path("artifacts")
    out.mkdir(parents=True, exist_ok=True)
    (out / "venue_screen.md").write_text(table, encoding="utf-8")
    print(table)

    workable = [p for p in profiles if p.makeability >= 0.5]
    print(f"{len(workable)}/{len(profiles)} symbols are at least marginal")
    if workable:
        best = workable[0]
        print(f"\nBest: {best.symbol}  makeability={best.makeability:.2f}  "
              f"tick={best.tick_size}  min_notional=${best.min_notional:g}")
        print("Set AMME_MARKET_SYMBOL / TICK_SIZE / LOT_SIZE accordingly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
