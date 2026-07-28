/* Research view: the measured evidence that each mechanism earns its place.
 *
 * Everything is read from artifacts/ via /api/research. Where an artifact is
 * missing the panel says which make target produces it, rather than rendering a
 * zero — an un-run experiment and a result of zero are different claims.
 *
 * Encoding note. The five ablation arms and the fourteen screened venues are
 * *magnitude* comparisons across named categories, so they get one hue with the
 * headline row emphasised — not five categorical colours. Colour here would
 * duplicate the category label and burn the only free channel. The one genuine
 * identity comparison (P&L in high- vs low-VPIN tape) is two series and gets
 * categorical slots plus a legend.
 */

import { $, el, replace, empty } from "../dom.js";
import { num, signed, int, pct, latency, isNum, DASH } from "../format.js";
import { mount, drawBars, drawGrouped, drawLine, drawHeatmap } from "../charts.js";
import { getJSON } from "../store.js";

let loaded = null;
const charts = {};

/* --------------------------------------------------------------- pieces -- */

function sectionHead(title, caption) {
  return el("div", { class: "section-head" }, [
    el("h2", { text: title }),
    caption ? el("p", { text: caption }) : null,
  ]);
}

function card(title, caption, body, extra) {
  return el("div", { class: "card" }, [
    el("header", {}, [
      el("h3", { text: title }),
      caption ? el("p", { class: "cap", text: caption }) : null,
    ]),
    body,
    extra ?? null,
  ]);
}

function missing(what, command) {
  return el("div", { class: "card" }, [
    el("p", { class: "empty" }, [
      `${what} has not been generated yet. Run `,
      el("code", { text: command }),
      ".",
    ]),
  ]);
}

/** A chart container plus the table-view twin every chart is required to have. */
function chartSlot(id, height, drawFn, tableId) {
  const holder = el("div", { id });
  const node = el("div", {}, [
    holder,
    el("details", { class: "table-view" }, [
      el("summary", { text: "Table view" }),
      el("div", { class: "table-wrap", id: tableId }),
    ]),
  ]);
  // Mounting is deferred to after insertion: ResizeObserver needs the node in
  // the document to report a non-zero width.
  queueMicrotask(() => { charts[id] = mount(holder, height, drawFn); });
  return node;
}

function table(columns, rows, opts = {}) {
  return el("div", { class: "table-wrap" }, el("table", {}, [
    opts.caption ? el("caption", { text: opts.caption }) : null,
    el("thead", {}, el("tr", {}, columns.map((c) =>
      el("th", { class: c.num ? "num" : null, text: c.label })))),
    el("tbody", {}, rows.map((row, i) =>
      el("tr", { class: opts.highlight?.(row, i) ? "is-highlight" : null },
        columns.map((c) => el("td", { class: c.num ? "num" : null }, [c.get(row, i)]))))),
  ]));
}

/* ------------------------------------------------------------- ablation -- */

const ARM_ORDER = ["naive_symmetric", "pure_as", "as_toxicity", "as_drift", "amme"];

/* Chart data and DOM are built by separate functions (`ablationChartData` here,
 * `ablationSectionDom` below) so a theme flip can re-paint every chart without
 * rebuilding the section — chart strokes read CSS custom properties at draw
 * time, so they must be redrawn when those properties change. */
function ablationChartData(ablation) {
  if (!ablation) return null;
  const byKey = new Map((ablation.aggregates ?? []).map((a) => [a.key, a]));
  const arms = ARM_ORDER.map((k) => byKey.get(k)).filter(Boolean);
  const isHeadline = (a) => a.key === "amme";

  const deltaLabels = {
    pure_as_vs_naive_symmetric: "A-S vs naive",
    as_toxicity_vs_pure_as: "+ toxicity gate",
    as_drift_vs_pure_as: "+ DeepLOB drift",
    amme_vs_pure_as: "Full AMME vs A-S",
  };

  return {
    pnl: arms.map((a) => ({
      label: a.label, value: a.pnl_mean, emphasis: isHeadline(a),
      note: `±${num(a.pnl_std, 3)} sd · ${a.seeds_profitable}/${a.n_seeds} profitable`,
    })),
    sharpe: arms.map((a) => ({
      label: a.label, value: a.sharpe_mean, emphasis: isHeadline(a),
      note: `max drawdown ${num(a.max_drawdown_mean, 3)}`,
    })),
    vpin: arms.map((a) => ({
      label: a.label, values: [a.pnl_high_vpin_mean, a.pnl_low_vpin_mean],
    })),
    delta: Object.entries(ablation.deltas ?? {}).map(([key, d]) => ({
      label: deltaLabels[key] ?? key,
      value: d.pnl_delta_mean,
      emphasis: true,
      note: `${int(d.seeds_won)}/${int(d.n_seeds)} seeds won · Δ Sharpe ${signed(d.sharpe_delta_mean, 0)}`,
    })),
    arms,
  };
}

/* ---------------------------------------------------------------- model -- */

function modelSection(model) {
  if (!model) return missing("Model metrics", "make train");

  const m = model.metrics ?? {};
  const splits = model.splits ?? {};
  const beats = m.beats_majority_baseline;
  const history = m.history ?? [];

  const summaryTiles = el("section", { class: "tiles" }, [
    el("div", { class: "tile" }, [
      el("div", { class: "tile-label", text: "Test macro F1" }),
      el("div", { class: "tile-value", text: num(m.test_macro_f1, 4) }),
      el("div", { class: "tile-note", text: `validation ${num(m.val_macro_f1, 4)}` }),
    ]),
    el("div", { class: "tile" }, [
      el("div", { class: "tile-label", text: "Test accuracy" }),
      el("div", { class: "tile-value", text: pct(m.test_accuracy, 2) }),
      el("div", { class: "tile-note", text: `majority-class baseline ${pct(m.majority_class_accuracy, 2)}` }),
    ]),
    el("div", { class: "tile" }, [
      el("div", { class: "tile-label", text: "Beats baseline" }),
      el("div", { class: `tile-value ${beats ? "pos" : "neg"}`, text: beats ? "Yes" : "No" }),
      el("div", { class: "tile-note", text: "accuracy alone is not the test — class balance is skewed" }),
    ]),
    el("div", { class: "tile" }, [
      el("div", { class: "tile-label", text: "Parameters" }),
      el("div", { class: "tile-value", text: int(m.n_parameters) }),
      el("div", { class: "tile-note", text: `T=${model.dataset?.lookback ?? DASH} × ${model.dataset?.levels ?? DASH} levels` }),
    ]),
  ]);

  return el("div", {}, [
    sectionHead(
      "DeepLOB — the drift signal",
      "A CNN-LSTM over the raw order book, trained on recorded live ticks to predict the sign of "
      + `mid-price movement ${model.dataset?.horizon ?? DASH} events ahead.`),
    summaryTiles,

    el("div", { class: "grid cols-3", style: "margin-top:10px" }, [
      // Two measures on different scales get two charts, never two y-axes.
      card("Training loss", "Cross-entropy per epoch.",
        chartSlot("chart-model-loss", 170, drawLine, "table-model-loss")),
      card("Validation macro F1", "Selection metric. The best epoch is kept, not the last — note it peaks at epoch 2 and then overfits.",
        chartSlot("chart-model-f1", 170, drawLine, "table-model-f1")),
      card("Confusion matrix", "Row-normalised: of the tape that actually moved this way, what did the model predict?",
        chartSlot("chart-model-confusion", 205, drawHeatmap, "table-model-confusion")),
    ]),

    el("div", { style: "margin-top:10px" },
      card("Splits and class balance",
        "Chronological, never shuffled — a shuffled split on order-book data leaks the future into training and inflates every metric.",
        table(
          [
            { label: "Split", get: (r) => r.name },
            { label: "Samples", num: true, get: (r) => int(r.n) },
            { label: "Down", num: true, get: (r) => pct(r.down, 1) },
            { label: "Flat", num: true, get: (r) => pct(r.flat, 1) },
            { label: "Up", num: true, get: (r) => pct(r.up, 1) },
          ],
          ["train", "val", "test"].filter((k) => splits[k]).map((k) => ({ name: k, ...splits[k] })))),
    ),
  ]);
}

/* --------------------------------------------------------------- venues -- */

function venueSection(venues) {
  if (!venues) return missing("The venue screen", "make screen");

  const cols = venues.columns;
  const symbolIdx = 0;
  const makeIdx = cols.findIndex((c) => /makeability/i.test(c));
  const verdictIdx = cols.findIndex((c) => /verdict/i.test(c));

  // The markdown carries `**1.41**` and backticked symbols; strip the emphasis
  // so the table renders as data rather than as source.
  const clean = (s) => String(s).replace(/[*`]/g, "").trim();
  const verdictClass = (v) => /workable/i.test(v) ? "ok" : /marginal/i.test(v) ? "warn" : "bad";

  const rows = venues.rows.map((r) => r.map(clean));
  const chartRows = rows.map((r) => ({
    label: r[symbolIdx],
    value: Number(r[makeIdx]),
    emphasis: /workable/i.test(r[verdictIdx] ?? ""),
    note: r[verdictIdx],
  })).filter((r) => isNum(r.value));

  return el("div", {}, [
    sectionHead(
      "Venue screen — where is market making even viable?",
      "makeability = spread ÷ (σ·√holding): the spread expressed in units of the volatility "
      + "accrued while holding inventory. Spread in ticks and spread in bps both rank symbols wrongly."),

    el("div", { class: "grid split" }, [
      card("Makeability by symbol",
        "Above 1.0 workable, 0.5–1.0 marginal, below 0.5 hostile. Only the workable symbols keep the accent; the rest are context.",
        chartSlot("chart-venues", Math.max(200, chartRows.length * 22 + 30), drawBars, "table-venues")),
      card("Full screen", "Every symbol ranked, as `make screen` wrote it.",
        table(
          cols.map((label, i) => ({
            label,
            num: i > 0 && i !== verdictIdx,
            get: (r) => i === verdictIdx
              ? el("span", { class: `badge ${verdictClass(r[i])}`, text: r[i] })
              : r[i],
          })),
          rows,
          {
            highlight: (r) => /workable/i.test(r[verdictIdx] ?? ""),
            caption: "Every Binance spot symbol screened has a binding tick — the spread sits at exactly one tick, so the venue itself caps maker revenue. What separates them is how much volatility that tick must cover.",
          })),
    ]),
  ]);
}

/* -------------------------------------------------------------- latency -- */

function latencySection(budgets) {
  if (!budgets?.length) return null;

  const verdict = (b) => {
    if (!isNum(b.budget_ms)) return el("span", { class: "badge neutral", text: "no budget" });
    const measured = isNum(b.p99_ms) ? b.p99_ms : b.p50_ms;
    const ok = measured <= b.budget_ms;
    return el("span", { class: `badge ${ok ? "ok" : "bad"}`, text: ok ? "within budget" : "over budget" });
  };

  return el("div", {}, [
    sectionHead(
      "Latency budgets",
      "Measured per stage and asserted in CI rather than reported. The budgets are sized to catch "
      + "an order-of-magnitude regression, not CI noise."),
    card("Per-stage latency", null,
      table(
        [
          { label: "Stage", get: (b) => el("div", {}, [
              el("div", { text: b.stage }),
              el("div", { class: "cap", text: b.detail }),
            ]) },
          { label: "p50", num: true, get: (b) => latency(b.p50_ms) },
          { label: "p99", num: true, get: (b) => isNum(b.p99_ms) ? latency(b.p99_ms) : DASH },
          { label: "Budget", num: true, get: (b) => isNum(b.budget_ms) ? latency(b.budget_ms) : DASH },
          { label: "", get: verdict },
          { label: "Note", get: (b) => el("span", { class: "cap", text: b.note }) },
        ],
        budgets)),
  ]);
}

/* ---------------------------------------------------------------- paint -- */

function paintCharts(payload) {
  const abl = ablationChartData(payload.ablation);
  if (abl) {
    charts["chart-abl-pnl"]?.render(abl.pnl, {
      diverging: true, dp: 3, valueLabel: "mean P&L",
      format: (v) => signed(v, 3),
    });
    charts["chart-abl-sharpe"]?.render(abl.sharpe, {
      diverging: true, dp: 0, valueLabel: "mean Sharpe",
      format: (v) => signed(v, 0),
    });
    charts["chart-abl-vpin"]?.render(abl.vpin, {
      series: [
        { label: "high-VPIN", colorVar: "--series-1" },
        { label: "low-VPIN", colorVar: "--series-2" },
      ],
      dp: 3,
    });
    charts["chart-abl-delta"]?.render(abl.delta, {
      diverging: true, dp: 3, valueLabel: "Δ P&L", format: (v) => signed(v, 3),
    });

    replace($("#table-abl-pnl"), table(
      [{ label: "Arm", get: (r) => r.label },
       { label: "Mean P&L", num: true, get: (r) => signed(r.value, 4) },
       { label: "Detail", get: (r) => r.note }], abl.pnl));
    replace($("#table-abl-sharpe"), table(
      [{ label: "Arm", get: (r) => r.label },
       { label: "Mean Sharpe", num: true, get: (r) => signed(r.value, 0) },
       { label: "Detail", get: (r) => r.note }], abl.sharpe));
    replace($("#table-abl-vpin"), table(
      [{ label: "Arm", get: (r) => r.label },
       { label: "High-VPIN P&L", num: true, get: (r) => signed(r.values[0], 4) },
       { label: "Low-VPIN P&L", num: true, get: (r) => signed(r.values[1], 4) }], abl.vpin));
    replace($("#table-abl-delta"), table(
      [{ label: "Comparison", get: (r) => r.label },
       { label: "Δ P&L", num: true, get: (r) => signed(r.value, 4) },
       { label: "Detail", get: (r) => r.note }], abl.delta));
  }

  const m = payload.model?.metrics;
  if (m?.history?.length) {
    const history = m.history;
    charts["chart-model-loss"]?.render(history, {
      series: [{ key: "train_loss", label: "loss", colorVar: "--series-1" }],
      dp: 3, area: true,
      xLabels: [`epoch ${history[0].epoch}`, `epoch ${history[history.length - 1].epoch}`],
      tipTitle: (p) => `epoch ${p.epoch}`,
    });
    charts["chart-model-f1"]?.render(history, {
      series: [{ key: "val_macro_f1", label: "macro F1", colorVar: "--series-3" }],
      dp: 3,
      xLabels: [`epoch ${history[0].epoch}`, `epoch ${history[history.length - 1].epoch}`],
      tipTitle: (p) => `epoch ${p.epoch}`,
    });

    const epochCols = [
      { label: "Epoch", num: true, get: (r) => r.epoch },
      { label: "Train loss", num: true, get: (r) => num(r.train_loss, 4) },
      { label: "Val macro F1", num: true, get: (r) => num(r.val_macro_f1, 4) },
      { label: "Val accuracy", num: true, get: (r) => pct(r.val_accuracy, 2) },
      { label: "Seconds", num: true, get: (r) => num(r.seconds, 0) },
    ];
    replace($("#table-model-loss"), table(epochCols, history));
    replace($("#table-model-f1"), table(epochCols, history));
  }

  if (m?.test_confusion) {
    const labels = ["down", "flat", "up"];
    charts["chart-model-confusion"]?.render({ matrix: m.test_confusion, labels });
    replace($("#table-model-confusion"), table(
      [{ label: "Actual", get: (r) => r.label },
       ...labels.map((l, j) => ({ label: `pred ${l}`, num: true, get: (r) => int(r.counts[j]) }))],
      m.test_confusion.map((counts, i) => ({ label: labels[i], counts }))));
  }

  if (payload.venues) {
    const clean = (s) => String(s).replace(/[*`]/g, "").trim();
    const cols = payload.venues.columns;
    const makeIdx = cols.findIndex((c) => /makeability/i.test(c));
    const verdictIdx = cols.findIndex((c) => /verdict/i.test(c));
    const rows = payload.venues.rows.map((r) => r.map(clean));
    const chartRows = rows.map((r) => ({
      label: r[0], value: Number(r[makeIdx]),
      emphasis: /workable/i.test(r[verdictIdx] ?? ""), note: r[verdictIdx],
    })).filter((r) => isNum(r.value));

    charts["chart-venues"]?.render(chartRows, {
      dp: 2, valueLabel: "makeability", labelWidth: 96,
      format: (v) => num(v, 2),
    });
    replace($("#table-venues"), table(
      [{ label: "Symbol", get: (r) => r.label },
       { label: "Makeability", num: true, get: (r) => num(r.value, 2) },
       { label: "Verdict", get: (r) => r.note }], chartRows));
  }
}

function render(payload) {
  const host = $("#research-body");
  if (!payload) { replace(host, empty("Could not reach /api/research.")); return; }

  replace(host, [
    ablationSectionDom(payload.ablation),
    venueSection(payload.venues),
    modelSection(payload.model),
    latencySection(payload.latency),
  ]);

  // Charts mount in a microtask (they need layout); paint on the next frame.
  requestAnimationFrame(() => paintCharts(payload));
}

/* The DOM half of the ablation section, kept separate from its chart data. */
function ablationSectionDom(ablation) {
  if (!ablation) return missing("The ablation", "make ablation");

  const byKey = new Map((ablation.aggregates ?? []).map((a) => [a.key, a]));
  const arms = ARM_ORDER.map((k) => byKey.get(k)).filter(Boolean);
  const seeds = ablation.seeds ?? [];
  const cfg = ablation.config ?? {};

  const deltaLabels = {
    pure_as_vs_naive_symmetric: "Avellaneda–Stoikov vs naive",
    as_toxicity_vs_pure_as: "+ toxicity gate",
    as_drift_vs_pure_as: "+ DeepLOB drift",
    amme_vs_pure_as: "Full AMME vs pure A-S",
  };
  const deltas = Object.entries(ablation.deltas ?? {}).map(([key, d]) => ({
    label: deltaLabels[key] ?? key, ...d,
  }));

  return el("div", {}, [
    sectionHead(
      "Ablation — five arms, paired across seeds",
      `${seeds.length} seeds × ${num(cfg.duration_seconds, 0)}s each. Every arm sees byte-identical `
      + "market data within each seed, so comparisons are paired and path noise cancels. "
      + "γ, κ and λ are grid-searched, not chosen."),

    el("div", { class: "grid cols-2" }, [
      card("Mean P&L by arm",
        "Quote currency, averaged over seeds. Negative and positive share one diverging scale about zero — only the naive baseline loses money.",
        chartSlot("chart-abl-pnl", 190, drawBars, "table-abl-pnl")),
      card("Mean Sharpe by arm",
        "Annualised on wall-clock seconds: crypto trades 24/7, so 252 trading days is the wrong convention.",
        chartSlot("chart-abl-sharpe", 190, drawBars, "table-abl-sharpe")),
    ]),

    el("div", { class: "grid cols-2", style: "margin-top:10px" }, [
      card("Where the money is made and lost",
        "P&L split by whether the tape was in its high- or low-VPIN regime. The naive quoter is not merely unprofitable — it is being adversely selected, exactly as theory predicts.",
        el("div", {}, [
          el("ul", { class: "legend" }, [
            el("li", {}, [el("span", { class: "swatch", style: "background:var(--series-1)" }), "high-VPIN tape"]),
            el("li", {}, [el("span", { class: "swatch", style: "background:var(--series-2)" }), "low-VPIN tape"]),
          ]),
          chartSlot("chart-abl-vpin", 210, drawGrouped, "table-abl-vpin"),
        ])),
      card("Paired mechanism deltas",
        "Each mechanism against the arm it was added to, seed by seed. A positive mean that wins on few seeds is noise; these win on 7–8 of 10.",
        el("div", {}, [
          chartSlot("chart-abl-delta", 150, drawBars, "table-abl-delta"),
          table(
            [
              { label: "Comparison", get: (d) => d.label },
              { label: "Δ P&L", num: true, get: (d) => signed(d.pnl_delta_mean, 3) },
              { label: "Δ Sharpe", num: true, get: (d) => signed(d.sharpe_delta_mean, 0) },
              { label: "Seeds won", num: true, get: (d) => `${int(d.seeds_won)}/${int(d.n_seeds)}` },
            ],
            deltas),
        ])),
    ]),

    el("div", { style: "margin-top:10px" },
      card("Full ablation table", "Every arm, every reported statistic.",
        table(
          [
            { label: "Arm", get: (a) => a.label },
            { label: "P&L", num: true, get: (a) => `${signed(a.pnl_mean, 3)} ± ${num(a.pnl_std, 3)}` },
            { label: "Sharpe", num: true, get: (a) => signed(a.sharpe_mean, 0) },
            { label: "Max DD", num: true, get: (a) => num(a.max_drawdown_mean, 3) },
            { label: "Inventory σ", num: true, get: (a) => num(a.inventory_std_mean, 5) },
            { label: "Fills", num: true, get: (a) => int(a.fills_mean) },
            { label: "Fill ratio", num: true, get: (a) => pct(a.fill_ratio_mean, 1) },
            { label: "Avg spread", num: true, get: (a) => num(a.avg_spread_mean, 4) },
            { label: "Profitable", num: true, get: (a) => `${a.seeds_profitable}/${a.n_seeds}` },
          ],
          arms,
          {
            highlight: (a) => a.key === "amme",
            caption: `Seeds: ${seeds.join(", ")}. Warm-up ${num(cfg.warmup_seconds, 0)}s excluded from every statistic.`,
          }))),
  ]);
}

/* ----------------------------------------------------------------- view -- */

export const researchView = {
  title: "Research",
  root: null,

  mount() { replace($("#research-body"), empty("Loading artifacts…")); },

  async enter() {
    // Artifacts change only when a make target is re-run; fetch once per
    // session and re-paint on a theme flip.
    if (!loaded) loaded = await getJSON("/api/research");
    render(loaded);
  },

  repaint() { if (loaded) paintCharts(loaded); },
};
