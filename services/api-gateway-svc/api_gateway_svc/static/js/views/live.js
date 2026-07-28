/* Live trading view.
 *
 * Reads the store; writes nothing. Rendering is split so the expensive parts
 * (charts, the ladder) only redraw when the view is actually visible — the
 * store keeps ticking at 2Hz whether or not anyone is looking at this tab.
 */

import { $, el, text, setClass, replace, empty } from "../dom.js";
import {
  num, signed, int, pct, price, clockTime, latency, isNum, DASH,
} from "../format.js";
import { mount, drawLine, drawDivergingStack } from "../charts.js";
import { store } from "../store.js";

let charts = null;
let visible = false;

/* Tick size drives price precision everywhere on this view. It comes from the
 * running config, so AVAXUSDT renders 4dp and BTCUSDT 2dp without a special
 * case anywhere. */
const tick = () => store.config?.tick_size ?? 0.01;
const qMax = () => store.config?.risk?.q_max ?? 0;

/* ---------------------------------------------------------------- tiles -- */

function renderTiles() {
  const s = store.state;

  const pnl = s.total_pnl;
  const pnlEl = $("#t-pnl");
  text(pnlEl, isNum(pnl) ? signed(pnl, 4) : DASH);
  setClass(pnlEl, "tile-value " + (!isNum(pnl) ? "dim" : pnl >= 0 ? "pos" : "neg"));
  text($("#t-pnl-note"),
    `realized ${signed(s.realized_pnl, 4)} · unrealized ${signed(s.unrealized_pnl, 4)}`);

  text($("#t-inv"), isNum(s.inventory) ? signed(s.inventory, 5) : DASH);
  const util = Math.min(1, Math.abs(s.inventory_utilisation ?? 0));
  text($("#t-inv-note"),
    `${pct(util, 0)} of limit · ${num(s.position_value, 2)} notional`);
  const invMeter = $("#t-inv-meter");
  setClass(invMeter, "meter" + (util >= 0.95 ? " is-crit" : util >= 0.6 ? " is-warn" : ""));
  invMeter.firstElementChild.style.width = `${(util * 100).toFixed(1)}%`;

  text($("#t-mid"), isNum(s.mid_price) && s.mid_price > 0 ? price(s.mid_price, tick()) : DASH);
  text($("#t-mid-note"),
    s.mid_price > 0
      ? `spread ${num(s.spread_bps, 2)} bps · imbalance ${signed(s.book_imbalance, 2)}`
      : "waiting for the book");

  text($("#t-quote"),
    s.quote_bid > 0 ? `${price(s.quote_bid, tick())} / ${price(s.quote_ask, tick())}` : DASH);
  text($("#t-quote-note"),
    s.half_spread > 0
      ? `half-spread ${num(s.half_spread, 5)} · reservation ${signed(s.reservation_offset, 5)} vs mid`
      : "not quoting");

  text($("#t-vpin"), isNum(s.vpin) ? num(s.vpin, 3) : DASH);
  text($("#t-vpin-note"),
    `spread ×${num(s.toxicity_multiplier, 2)} · OFI z ${signed(s.ofi_zscore, 1)}`
    + (s.vpin_warm === false ? " · warming up" : ""));
  const vpinMeter = $("#t-vpin-meter");
  const v = Math.min(1, s.vpin ?? 0);
  const breakerVpin = store.config?.risk?.breaker_vpin ?? 0.8;
  setClass(vpinMeter,
    "meter" + (v >= breakerVpin ? " is-crit" : v >= (store.config?.toxicity?.vpin_threshold ?? 0.35) ? " is-warn" : ""));
  vpinMeter.firstElementChild.style.width = `${(v * 100).toFixed(1)}%`;

  text($("#t-fills"), isNum(s.n_fills) ? int(s.n_fills) : DASH);
  text($("#t-fills-note"), `${int(s.n_quotes)} quotes sent`);
}

/* --------------------------------------------------------------- banner -- */

function renderBanner() {
  const s = store.state;
  const host = $("#risk-banner");
  const halted = s.breaker_tripped || s.risk_action === "halt";
  const rejected = s.risk_action === "reject";

  if (!halted && !rejected) { replace(host, []); return; }

  // Icon + word + colour, never colour alone.
  const node = el("div", { class: halted ? "banner crit" : "banner" }, [
    el("span", { class: "banner-icon", "aria-hidden": "true" }, halted ? "■" : "▲"),
    el("div", {}, [
      el("strong", { text: halted ? "Circuit breaker tripped — quoting halted" : "Quote rejected by risk" }),
      el("div", { class: "cap", text: s.risk_reason || "no reason reported" }),
    ]),
  ]);
  replace(host, node);
}

/* --------------------------------------------------------------- charts -- */

function historyTable(rows, columns) {
  if (!rows.length) return empty("No samples yet.");
  const recent = rows.slice(-40).reverse();
  return el("table", {}, [
    el("thead", {}, el("tr", {}, columns.map((c) => el("th", { class: c.num ? "num" : null, text: c.label })))),
    el("tbody", {}, recent.map((row) =>
      el("tr", {}, columns.map((c) => el("td", { class: c.num ? "num" : null, text: c.get(row) }))))),
  ]);
}

function renderCharts() {
  const points = store.history;

  charts.pnl.render(points, {
    series: [
      { key: "pnl", label: "total", colorVar: "--series-1" },
      { key: "realized", label: "realized", colorVar: "--series-3" },
    ],
    dp: 4,
    includeZero: true,
    xLabels: points.length > 1
      ? [clockTime(points[0].t), clockTime(points[points.length - 1].t)]
      : null,
    tipTitle: (p) => clockTime(p.t),
  });

  charts.inv.render(points, {
    series: [{ key: "inventory", label: "inventory", colorVar: "--series-3" }],
    dp: 5,
    symmetric: true,
    area: true,
    limit: qMax() || undefined,
    limitLabel: "q_max",
    xLabels: points.length > 1
      ? [clockTime(points[0].t), clockTime(points[points.length - 1].t)]
      : null,
    tipTitle: (p) => clockTime(p.t),
  });

  charts.drift.render({
    down: store.state.prob_down,
    flat: store.state.prob_flat,
    up: store.state.prob_up,
  });

  replace($("#pnl-table"), historyTable(points, [
    { label: "Time", get: (r) => clockTime(r.t) },
    { label: "Total P&L", num: true, get: (r) => signed(r.pnl, 5) },
    { label: "Realized", num: true, get: (r) => signed(r.realized, 5) },
    { label: "Mid", num: true, get: (r) => price(r.mid, tick()) },
  ]));
  replace($("#inv-table"), historyTable(points, [
    { label: "Time", get: (r) => clockTime(r.t) },
    { label: "Inventory", num: true, get: (r) => signed(r.inventory, 6) },
    { label: "VPIN", num: true, get: (r) => num(r.vpin, 3) },
  ]));
}

/* --------------------------------------------------------------- ladder -- */

function renderLadder() {
  const { bids = [], asks = [] } = store.book;
  const host = $("#ladder");
  if (!bids.length || !asks.length) { replace(host, empty("Waiting for the order book.")); return; }

  const depth = 7;
  const topAsks = asks.slice(0, depth).reverse();  // best ask nearest the mid
  const topBids = bids.slice(0, depth);
  const maxSize = Math.max(...[...topAsks, ...topBids].map((l) => l.size), 1e-12);
  const t = tick();
  const s = store.state;

  // Our quote sits at whichever visible level it rests on. Matching on the
  // quantised price is what makes the marker honest: if the quote is inside the
  // spread, no level is marked and the ladder says so by omission.
  const near = (a, b) => isNum(a) && isNum(b) && a > 0 && Math.abs(a - b) < t / 2;

  const row = (level, side) => {
    const ours = near(side === "ask" ? s.quote_ask : s.quote_bid, level.price);
    return el("div", { class: `ladder-row ${side}${ours ? " is-ours" : ""}` }, [
      el("i", { class: "ladder-bar", style: `width:${((level.size / maxSize) * 100).toFixed(1)}%` }),
      el("span", { class: "ladder-tag", text: ours ? (side === "ask" ? "our ask" : "our bid") : "" }),
      el("span", { class: "ladder-px", text: price(level.price, t) }),
      el("span", { class: "ladder-sz", text: num(level.size, 4) }),
    ]);
  };

  replace(host, [
    ...topAsks.map((l) => row(l, "ask")),
    el("div", { class: "ladder-mid" }, [
      el("span", { class: "ladder-tag", text: "mid" }),
      el("span", { class: "ladder-px", text: price(store.book.mid_price, t) }),
      el("span", { class: "ladder-sz", text: `${num(s.spread_bps, 2)} bps` }),
    ]),
    ...topBids.map((l) => row(l, "bid")),
  ]);
}

/* ------------------------------------------------------------ key/value -- */

function kv(pairs) {
  return pairs.flatMap(([key, value, cls]) => [
    el("dt", { text: key }),
    el("dd", { class: cls ?? null, text: value }),
  ]);
}

function renderPanels() {
  const s = store.state;
  const t = tick();

  replace($("#quote-kv"), kv([
    ["Mid price", s.mid_price > 0 ? price(s.mid_price, t) : DASH],
    ["Reservation price", s.reservation_price > 0 ? price(s.reservation_price, t) : DASH],
    ["Reservation offset", signed(s.reservation_offset, 6),
      isNum(s.reservation_offset) && s.reservation_offset !== 0
        ? (s.reservation_offset > 0 ? "pos" : "neg") : "dim"],
    ["Optimal half-spread", num(s.half_spread, 6)],
    ["Toxicity multiplier", `×${num(s.toxicity_multiplier, 3)}`],
    ["Drift adjustment", signed(s.drift_adjustment, 6)],
    ["γ (risk aversion)", num(s.gamma, 4)],
    ["κ (arrival decay)", num(s.kappa, 2)],
    ["Strategy", s.strategy || DASH],
  ]));

  text($("#drift-horizon"), isNum(s.horizon_events) && s.horizon_events > 0 ? s.horizon_events : DASH);
  replace($("#drift-kv"), kv([
    ["Expected drift", isNum(s.drift) ? `${signed(s.drift * 1e4, 3)} bps` : DASH],
    ["Confidence", pct(s.drift_confidence, 1)],
    ["Backend", store.config?.model?.backend ?? DASH],
    ["Model version", s.model_version || DASH],
    ["Inference p99", latency(s.inference_p99_ms)],
  ]));

  const warm = (ok) => (ok === true ? "warm" : ok === false ? "warming up" : DASH);
  replace($("#feature-kv"), kv([
    ["VPIN", num(s.vpin, 4)],
    ["VPIN state", warm(s.vpin_warm), s.vpin_warm ? null : "dim"],
    ["OFI", signed(s.ofi, 4)],
    ["OFI z-score", signed(s.ofi_zscore, 3)],
    ["OFI state", warm(s.ofi_warm), s.ofi_warm ? null : "dim"],
    ["σ (rolling)", num(s.sigma, 6)],
    ["σ (fast EWMA)", num(s.sigma_fast, 6)],
    ["σ (baseline)", num(s.sigma_baseline, 6)],
    ["Book imbalance", signed(s.book_imbalance, 3)],
    ["Book staleness p50", latency(s.staleness_p50_ms)],
    ["Book staleness p99", latency(s.staleness_p99_ms)],
  ]));
}

/* ---------------------------------------------------------------- lists -- */

function renderLists() {
  const t = tick();
  const fills = store.fills;
  replace($("#fills-table"), !fills.length ? empty("No fills yet.") : el("table", {}, [
    el("thead", {}, el("tr", {}, [
      el("th", { text: "Side" }), el("th", { class: "num", text: "Price" }),
      el("th", { class: "num", text: "Size" }), el("th", { class: "num", text: "Queue" }),
      el("th", { class: "num", text: "Inventory after" }), el("th", { class: "num", text: "Time" }),
    ])),
    el("tbody", {}, fills.slice(0, 20).map((f) => el("tr", {}, [
      // The side is a word first; the colour only reinforces it.
      el("td", {}, el("span", {
        class: f.side === "bid" ? "badge ok" : "badge warn",
        text: f.side === "bid" ? "BUY" : "SELL",
      })),
      el("td", { class: "num", text: price(f.price, t) }),
      el("td", { class: "num", text: num(f.size, 5) }),
      el("td", { class: "num", text: num(f.queue_position, 3) }),
      el("td", { class: "num", text: signed(f.inventory_after, 5) }),
      el("td", { class: "num", text: clockTime(f.ts) }),
    ]))),
  ]));

  const trades = store.trades;
  replace($("#trades-table"), !trades.length ? empty("No trades yet.") : el("table", {}, [
    el("thead", {}, el("tr", {}, [
      el("th", { text: "Aggressor" }), el("th", { class: "num", text: "Price" }),
      el("th", { class: "num", text: "Size" }), el("th", { class: "num", text: "Time" }),
    ])),
    el("tbody", {}, trades.slice(0, 20).map((tr) => el("tr", {}, [
      el("td", { class: tr.aggressor === "buy" ? "pos" : tr.aggressor === "sell" ? "neg" : "dim",
                 text: tr.aggressor }),
      el("td", { class: "num", text: price(tr.price, t) }),
      el("td", { class: "num", text: num(tr.size, 5) }),
      el("td", { class: "num", text: clockTime(tr.ts) }),
    ]))),
  ]));
}

/* ----------------------------------------------------------------- view -- */

export const liveView = {
  title: "Live",
  root: null,

  mount() {
    charts = {
      pnl: mount($("#chart-pnl"), 210, drawLine),
      inv: mount($("#chart-inv"), 210, drawLine),
      drift: mount($("#chart-drift"), 46, drawDivergingStack),
    };
    // A legend is required for two series; the chart also direct-labels both,
    // so identity survives losing either channel.
    replace($("#pnl-legend"), [
      el("li", {}, [el("span", { class: "swatch", style: "background:var(--series-1)" }), "total P&L"]),
      el("li", {}, [el("span", { class: "swatch", style: "background:var(--series-3)" }), "realized only"]),
    ]);
  },

  enter() { visible = true; this.update("all"); },
  leave() { visible = false; },

  update(event) {
    if (!visible) return;
    renderTiles();
    renderBanner();
    renderPanels();
    if (event === "lists" || event === "all" || event === "bootstrap") renderLists();
    renderLadder();
    renderCharts();
  },
};
