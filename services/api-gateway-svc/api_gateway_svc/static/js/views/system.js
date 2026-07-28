/* System view: is the pipeline actually whole, and what is it doing?
 *
 * Polls /api/system while visible. The gateway fans out to all eight /health
 * endpoints concurrently under one deadline, so this is a single request no
 * matter how many services are wedged.
 */

import { $, el, replace, empty } from "../dom.js";
import { num, int, latency, duration, isNum, DASH } from "../format.js";
import { store, getJSON } from "../store.js";

let timer = null;
let lastPayload = null;

/* --------------------------------------------------------------- pieces -- */

function tile(label, value, note, cls) {
  return el("div", { class: "tile" }, [
    el("div", { class: "tile-label", text: label }),
    el("div", { class: `tile-value ${cls ?? ""}`, text: value }),
    note ? el("div", { class: "tile-note", text: note }) : null,
  ]);
}

function summary(payload) {
  const services = payload.services ?? [];
  const healthy = services.filter((s) => s.status === "healthy").length;
  const allUp = healthy === services.length && services.length > 0;
  const totalRate = (payload.streams ?? []).reduce((a, s) => a + (s.rate ?? 0), 0);

  return el("section", { class: "tiles", "aria-label": "System summary" }, [
    tile("Services healthy", `${healthy}/${services.length}`,
      allUp ? "whole pipeline responding" : "one or more degraded",
      allUp ? "pos" : "neg"),
    tile("Gateway uptime", duration(payload.uptime_s), "since this container started"),
    tile("Event throughput", `${num(totalRate, 1)}/s`,
      `${int(store.state.total_events)} events projected`),
    tile("Book staleness p50", latency(payload.staleness_p50_ms),
      `p99 ${latency(payload.staleness_p99_ms)} · clock-corrected`),
  ]);
}

/* The topology is rendered from the same `consumes`/`produces` the gateway
 * declares, so the diagram cannot drift from the wiring it depicts. */
function topology(services) {
  const nodes = services.flatMap((svc, i) => {
    const cls = svc.status === "healthy" ? "ok" : svc.status === "unreachable" ? "bad" : "warn";
    const checks = Object.entries(svc.checks ?? {});

    const node = el("div", { class: `topo-node ${cls}` }, [
      el("span", { class: "status-dot", "aria-hidden": "true" }),
      el("div", {}, [
        el("div", { class: "topo-name" }, [
          svc.name,
          " ",
          el("span", {
            class: `badge ${cls === "ok" ? "ok" : cls === "bad" ? "bad" : "warn"}`,
            text: svc.status,
          }),
        ]),
        el("div", { class: "topo-role", text: svc.role }),
        el("div", { class: "topo-streams" }, [
          ...svc.consumes.map((s) => el("span", { class: "stream-chip", text: `← ${s}` })),
          ...svc.produces.map((s) => el("span", { class: "stream-chip out", text: `→ ${s}` })),
        ]),
        checks.length
          // Full detail, not a fixed-width slice: these strings carry the
          // assertion each service makes about itself, and the tail is usually
          // the part that matters (`measured 240s ago`, `resyncs=3`). Chips
          // wrap rather than truncate.
          ? el("div", { class: "topo-streams" },
              checks.map(([name, detail]) => {
                const failed = String(detail).startsWith("FAIL");
                return el("span", {
                  class: `stream-chip${failed ? " out" : ""}`,
                  text: `${name}: ${String(detail).replace(/^ok: /, "")}`,
                });
              }))
          : null,
      ]),
      el("div", { class: "topo-right" }, [
        el("div", { text: `:${svc.port}` }),
        el("div", { text: isNum(svc.latency_ms) ? latency(svc.latency_ms) : DASH }),
      ]),
    ]);

    // A connector between consecutive pipeline stages, but not before
    // backtest-svc or the gateway — neither sits on the trading path.
    const inPath = i > 0 && i < 6;
    return inPath ? [el("div", { class: "topo-arrow", text: "↓" }), node] : [node];
  });

  return el("div", { class: "topo" }, nodes);
}

function streamTable(streams) {
  if (!streams?.length) return empty("No streams observed yet.");
  return el("div", { class: "table-wrap" }, el("table", {}, [
    el("caption", { text: "Counted by this gateway's consumer group since it started. Rate is a 10-second trailing average." }),
    el("thead", {}, el("tr", {}, [
      el("th", { text: "Stream" }),
      el("th", { class: "num", text: "Messages" }),
      el("th", { class: "num", text: "Rate" }),
    ])),
    el("tbody", {}, streams.map((s) => el("tr", {}, [
      el("td", {}, el("span", { class: "mono", text: s.name })),
      el("td", { class: "num", text: int(s.count) }),
      el("td", { class: "num", text: s.count ? `${num(s.rate, 2)}/s` : DASH }),
    ]))),
  ]));
}

function configTable(config) {
  if (!config) return empty("Configuration unavailable.");

  const groups = [
    ["Market", [
      ["Symbol", config.symbol],
      ["Depth levels", config.depth_levels],
      ["Depth cadence", `${config.depth_update_ms} ms`],
      ["Tick size", config.tick_size],
    ]],
    ["Avellaneda–Stoikov", [
      ["γ risk aversion", config.avellaneda_stoikov.gamma],
      ["κ arrival decay", config.avellaneda_stoikov.kappa],
      ["Horizon mode", config.avellaneda_stoikov.horizon_mode],
      ["Horizon", `${config.avellaneda_stoikov.horizon_seconds} s`],
      ["Quote size", config.avellaneda_stoikov.quote_size],
      ["Min half-spread", `${config.avellaneda_stoikov.min_half_spread_ticks} ticks`],
      ["Drift λ", config.avellaneda_stoikov.drift_lambda],
      ["Drift confidence floor", config.avellaneda_stoikov.drift_confidence_floor],
    ]],
    ["Risk", [
      ["q_max", config.risk.q_max],
      ["Soft limit", `${(config.risk.q_soft_frac * 100).toFixed(0)}% of q_max`],
      ["Unwind-only beyond", `${(config.risk.unwind_frac * 100).toFixed(0)}% of q_max`],
      ["Max position value", config.risk.max_position_value],
      ["Breaker: VPIN", config.risk.breaker_vpin],
      ["Breaker: vol multiple", `${config.risk.breaker_vol_multiple}×`],
      ["Breaker: drawdown", config.risk.breaker_drawdown],
      ["Risk layer", config.risk.enabled ? "enabled" : "DISABLED"],
    ]],
    ["Toxicity", [
      ["VPIN buckets", config.toxicity.vpin_buckets],
      ["VPIN bucket volume", config.toxicity.vpin_bucket_volume],
      ["VPIN threshold", config.toxicity.vpin_threshold],
      ["VPIN β", config.toxicity.vpin_beta],
      ["OFI threshold", config.toxicity.ofi_threshold],
      ["OFI β", config.toxicity.ofi_beta],
    ]],
    ["Model", [
      ["Backend", config.model.backend],
      ["Lookback T", config.model.lookback],
      ["Levels", config.model.levels],
      ["Prediction horizon", `${config.model.horizon_events} events`],
      ["p99 budget", `${config.model.inference_p99_budget_ms} ms`],
    ]],
  ];

  return el("div", { class: "grid cols-3" }, groups.map(([name, rows]) =>
    el("div", { class: "card" }, [
      el("header", {}, el("h3", { text: name })),
      el("dl", { class: "kv" }, rows.flatMap(([k, v]) => [
        el("dt", { text: k }),
        el("dd", { text: v === undefined || v === null ? DASH : String(v) }),
      ])),
    ])));
}

/* ----------------------------------------------------------------- view -- */

function render(payload) {
  const host = $("#system-body");
  if (!payload) { replace(host, empty("Could not reach /api/system.")); return; }

  replace(host, [
    summary(payload),

    el("div", { class: "section-head" }, [
      el("h2", { text: "Pipeline" }),
      el("p", { text: "Dataflow order. Each service is probed directly; the streams shown are what it declares it reads and writes." }),
    ]),
    el("div", { class: "card" }, topology(payload.services ?? [])),

    el("div", { class: "section-head" }, [
      el("h2", { text: "Stream throughput" }),
      el("p", { text: "Redis Streams carry every message between services." }),
    ]),
    el("div", { class: "card" }, streamTable(payload.streams)),

    el("div", { class: "section-head" }, [
      el("h2", { text: "Active configuration" }),
      el("p", { text: "What the running system is actually using — a dashboard with no parameters beside it is unfalsifiable." }),
    ]),
    configTable(payload.config),
  ]);
}

async function refresh() {
  const payload = await getJSON("/api/system");
  // On a failed refresh, hold the previous render rather than flashing an
  // error — a single dropped poll is not news.
  if (payload) { lastPayload = payload; render(payload); }
  else if (!lastPayload) render(null);
}

export const systemView = {
  title: "System",
  root: null,

  mount() { replace($("#system-body"), empty("Probing services…")); },

  enter() {
    refresh();
    // Health is slow-moving; 5s is responsive without eight probes a second.
    timer = setInterval(refresh, 5000);
  },

  leave() { clearInterval(timer); timer = null; },
};
