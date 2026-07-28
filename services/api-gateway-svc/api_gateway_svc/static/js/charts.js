/* Hand-rolled SVG charts.
 *
 * Why not a charting library: the page must be fully self-contained inside the
 * container — no CDN is reachable from the compose network, and adding a
 * bundler to a Python repo to draw six charts costs more than it returns. The
 * forms here are deliberately few and each is used more than once.
 *
 * Sizing: every chart is rendered at the container's real pixel width via a
 * ResizeObserver rather than a scaled viewBox. A viewBox with
 * preserveAspectRatio="none" — the obvious shortcut — stretches strokes and
 * text non-uniformly, so axis labels come out horizontally smeared at wide
 * widths. Rendering at 1:1 keeps 2px lines 2px and text at its stated size.
 *
 * Encoding rules applied throughout (see the palette notes in tokens.css):
 *   - magnitude across categories -> one hue, emphasis on the row that matters
 *   - polarity around zero        -> the diverging pair, neutral gray midpoint
 *   - identity (2-3 series)       -> categorical slots, legend + direct labels
 * No chart here has two y-scales.
 */

import { cssVar, tickLabel } from "./format.js";
import { esc } from "./dom.js";

const SVG_NS = "http://www.w3.org/2000/svg";

/* ------------------------------------------------------------- plumbing -- */

/** Attach an auto-sizing SVG to `container`, re-rendering on resize.
 *
 *  Returns a handle whose `.render(data, opts)` can be called as often as the
 *  data changes; the observer re-invokes the same draw function with the last
 *  arguments when the box changes, so a resize never blanks a chart.
 */
export function mount(container, height, draw) {
  container.classList.add("chart");
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("role", "img");
  container.replaceChildren(svg);

  const tip = document.createElement("div");
  tip.className = "tooltip";
  tip.setAttribute("role", "status");
  container.append(tip);

  let last = null;
  let width = container.clientWidth || 640;

  const paint = () => {
    if (last === null) return;
    const h = typeof height === "function" ? height(last.data, last.opts, width) : height;
    svg.setAttribute("width", String(width));
    svg.setAttribute("height", String(h));
    svg.setAttribute("viewBox", `0 0 ${width} ${h}`);
    draw({ svg, tip, width, height: h, container, ...last });
  };

  // clientWidth is 0 while the view is hidden; ignore those so a chart drawn on
  // a background tab is not baked at zero width and left blank when shown.
  const observer = new ResizeObserver((entries) => {
    const next = Math.round(entries[0].contentRect.width);
    if (next > 0 && next !== width) { width = next; paint(); }
  });
  observer.observe(container);

  return {
    render(data, opts = {}) {
      last = { data, opts };
      const next = container.clientWidth;
      if (next > 0) width = next;
      paint();
    },
    destroy() { observer.disconnect(); },
  };
}

function showTip(tip, container, clientX, clientY, html) {
  tip.innerHTML = html;
  tip.dataset.show = "1";
  const box = container.getBoundingClientRect();
  const tipBox = tip.getBoundingClientRect();
  // Flip to the left of the cursor near the right edge so the tooltip never
  // forces the container to scroll.
  let x = clientX - box.left + 14;
  if (x + tipBox.width > box.width) x = clientX - box.left - tipBox.width - 14;
  let y = clientY - box.top - tipBox.height - 12;
  if (y < 0) y = clientY - box.top + 18;
  tip.style.left = `${Math.max(0, x)}px`;
  tip.style.top = `${Math.max(0, y)}px`;
}

function hideTip(tip) { tip.dataset.show = "0"; }

function tipRow(color, key, value) {
  const swatch = color ? `<span class="swatch" style="background:${esc(color)}"></span>` : "";
  return `<div class="tt-row">${swatch}<span class="tt-key">${esc(key)}</span><b>${esc(value)}</b></div>`;
}

function emptySvg(svg, width, height, message) {
  svg.innerHTML =
    `<text class="emptystate" x="${width / 2}" y="${height / 2}" text-anchor="middle"
      dominant-baseline="middle">${esc(message)}</text>`;
}

/** Nice round axis bounds so ticks land on readable numbers. */
function niceBounds(lo, hi, ticks = 4) {
  if (!(hi > lo)) { const c = hi || 0; lo = c - 1; hi = c + 1; }
  const raw = (hi - lo) / ticks;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  return { lo: Math.floor(lo / step) * step, hi: Math.ceil(hi / step) * step, step };
}

/* ----------------------------------------------------------- line chart -- */
/**
 * Single- or dual-series time line with a crosshair tooltip.
 *
 * opts: { series: [{ key, label, colorVar }], dp, includeZero, symmetric,
 *         limit, limitLabel, yTitle, area }
 */
export function drawLine({ svg, tip, width, height, container, data, opts }) {
  const points = data ?? [];
  if (points.length < 2) return emptySvg(svg, width, height, opts.emptyText ?? "waiting for data…");

  const series = opts.series ?? [{ key: "value", label: "value", colorVar: "--series-1" }];
  const padL = 56, padR = 62, padT = 10, padB = 24;
  const plotW = Math.max(10, width - padL - padR);
  const plotH = Math.max(10, height - padT - padB);

  let lo = Infinity, hi = -Infinity;
  for (const p of points) {
    for (const s of series) {
      const v = p[s.key];
      if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    }
  }
  if (!Number.isFinite(lo)) return emptySvg(svg, width, height, "waiting for data…");
  if (opts.includeZero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  if (opts.symmetric) { const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1; lo = -m; hi = m; }
  // Expand to include the risk rails only when they are close enough to be
  // informative. q_max is sized to the venue, not to current activity, so on a
  // quiet book forcing them into scale flattens every real inventory movement
  // into a dead line at zero — the tile's utilisation meter carries the
  // "how close to the limit" story in that case, and this chart carries the
  // shape.
  if (Number.isFinite(opts.limit) && opts.limit > 0) {
    const reach = Math.max(Math.abs(lo), Math.abs(hi)) * 4;
    if (opts.limit <= reach) {
      lo = Math.min(lo, -opts.limit * 1.1);
      hi = Math.max(hi, opts.limit * 1.1);
    }
  }
  const span = hi - lo || 1;
  const bounds = niceBounds(lo - span * 0.06, hi + span * 0.06);

  const x = (i) => padL + (points.length === 1 ? plotW / 2 : (i / (points.length - 1)) * plotW);
  const y = (v) => padT + (1 - (v - bounds.lo) / (bounds.hi - bounds.lo)) * plotH;

  let out = "";

  // Recessive solid hairline grid — never dashed.
  for (let v = bounds.lo; v <= bounds.hi + 1e-9; v += bounds.step) {
    const yy = y(v);
    out += `<line class="gridline" x1="${padL}" y1="${yy.toFixed(1)}" x2="${padL + plotW}" y2="${yy.toFixed(1)}"/>`;
    out += `<text class="tick" x="${padL - 7}" y="${(yy + 3).toFixed(1)}" text-anchor="end">${esc(tickLabel(v, opts.dp ?? 2))}</text>`;
  }
  if (bounds.lo < 0 && bounds.hi > 0) {
    out += `<line class="axisline" x1="${padL}" y1="${y(0).toFixed(1)}" x2="${padL + plotW}" y2="${y(0).toFixed(1)}"/>`;
  }

  // Risk limit rails. Dashed here is meaningful — it *is* a threshold, which is
  // exactly the case the "no dashed gridlines" rule carves out.
  if (Number.isFinite(opts.limit) && opts.limit > 0) {
    const crit = cssVar("--critical");
    for (const sign of [1, -1]) {
      const lv = sign * opts.limit;
      if (lv > bounds.lo && lv < bounds.hi) {
        out += `<line x1="${padL}" y1="${y(lv).toFixed(1)}" x2="${padL + plotW}" y2="${y(lv).toFixed(1)}"
          stroke="${crit}" stroke-width="1" stroke-dasharray="4 3" opacity="0.7"/>`;
      }
    }
    if (opts.limit < bounds.hi) {
      out += `<text class="tick" x="${padL + plotW + 5}" y="${(y(opts.limit) + 3).toFixed(1)}"
        fill="${crit}">${esc(opts.limitLabel ?? "limit")}</text>`;
    }
  }

  // Direct labels are placed after the paths so collisions can be resolved
  // against each other: two series that end at the same value (P&L and
  // realized P&L both at zero, which is the normal state before the first
  // fill) would otherwise overprint into unreadable glyph soup.
  const labels = [];

  for (const s of series) {
    const color = cssVar(s.colorVar);
    const valid = points.map((p, i) => [i, p[s.key]]).filter(([, v]) => Number.isFinite(v));
    if (valid.length < 2) continue;
    const path = valid.map(([i, v], n) => `${n ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");

    if (opts.area && series.length === 1) {
      const base = y(Math.max(bounds.lo, Math.min(bounds.hi, 0)));
      const [firstI] = valid[0], [lastI] = valid[valid.length - 1];
      out += `<path d="${path}L${x(lastI).toFixed(1)},${base.toFixed(1)}L${x(firstI).toFixed(1)},${base.toFixed(1)}Z"
        fill="${color}" opacity="0.10"/>`;
    }
    out += `<path d="${path}" fill="none" stroke="${color}" stroke-width="2"
      stroke-linejoin="round" stroke-linecap="round"/>`;

    // Direct label at the series end. This is the relief for the sub-3:1
    // contrast slot and the reason a 1-2 series chart needs no legend box.
    const [lastIdx, lastVal] = valid[valid.length - 1];
    out += `<circle cx="${x(lastIdx).toFixed(1)}" cy="${y(lastVal).toFixed(1)}" r="3.5"
      fill="${color}" stroke="${cssVar("--surface-1")}" stroke-width="2"/>`;
    labels.push({ label: s.label, y: y(lastVal), color });
  }

  // Nudge overlapping labels apart, nearest-first, keeping them inside the plot.
  const LABEL_H = 13;
  labels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) {
    if (labels[i].y - labels[i - 1].y < LABEL_H) labels[i].y = labels[i - 1].y + LABEL_H;
  }
  const overflow = labels.length ? labels[labels.length - 1].y - (height - padB - 2) : 0;
  if (overflow > 0) for (const l of labels) l.y -= overflow;
  for (const l of labels) {
    const ly = Math.max(padT + 9, Math.min(height - padB - 2, l.y));
    out += `<text class="serieslabel" x="${padL + plotW + 6}" y="${(ly + 4).toFixed(1)}"
      fill="${cssVar("--text-secondary")}">${esc(l.label)}</text>`;
  }

  // Time axis: first and last sample only. Intermediate ticks on a rolling
  // 1Hz window are noise — the shape carries the story, not the timestamps.
  if (opts.xLabels) {
    out += `<text class="tick" x="${padL}" y="${height - 7}">${esc(opts.xLabels[0])}</text>`;
    out += `<text class="tick" x="${padL + plotW}" y="${height - 7}" text-anchor="end">${esc(opts.xLabels[1])}</text>`;
  }
  if (opts.yTitle) {
    out += `<text class="axis-title" x="${padL}" y="${padT - 1}" text-anchor="start">${esc(opts.yTitle)}</text>`;
  }

  out += `<g class="crosshair" style="display:none">
    <line class="axisline" y1="${padT}" y2="${padT + plotH}" stroke-dasharray="3 3"/>
    ${series.map((s) => `<circle r="4" fill="${cssVar(s.colorVar)}" stroke="${cssVar("--surface-1")}" stroke-width="2"/>`).join("")}
  </g>`;

  svg.innerHTML = out;

  const cross = svg.querySelector(".crosshair");
  const rule = cross.querySelector("line");
  const dots = Array.from(cross.querySelectorAll("circle"));

  const onMove = (event) => {
    const rect = svg.getBoundingClientRect();
    const px = event.clientX - rect.left;
    if (px < padL || px > padL + plotW) { cross.style.display = "none"; return hideTip(tip); }
    const idx = Math.max(0, Math.min(points.length - 1,
      Math.round(((px - padL) / plotW) * (points.length - 1))));
    const p = points[idx];
    cross.style.display = "";
    rule.setAttribute("x1", x(idx).toFixed(1));
    rule.setAttribute("x2", x(idx).toFixed(1));
    series.forEach((s, n) => {
      const v = p[s.key];
      const visible = Number.isFinite(v);
      dots[n].setAttribute("opacity", visible ? "1" : "0");
      if (visible) {
        dots[n].setAttribute("cx", x(idx).toFixed(1));
        dots[n].setAttribute("cy", y(v).toFixed(1));
      }
    });
    const rows = series
      .map((s) => tipRow(cssVar(s.colorVar), s.label, tickLabel(p[s.key], opts.dp ?? 2)))
      .join("");
    const head = opts.tipTitle ? `<b>${esc(opts.tipTitle(p, idx))}</b>` : "";
    showTip(tip, container, event.clientX, event.clientY, head + rows);
  };

  svg.onpointermove = onMove;
  svg.onpointerleave = () => { cross.style.display = "none"; hideTip(tip); };
}

/* ------------------------------------------------------ horizontal bars -- */
/**
 * Magnitude across named categories. One hue by default; `diverging: true`
 * splits polarity across the blue/red pair around a zero baseline.
 *
 * data: [{ label, value, note, emphasis }]
 * opts: { dp, diverging, unit, labelWidth, emptyText, format }
 */
export function drawBars({ svg, tip, width, height, container, data, opts }) {
  const rows = data ?? [];
  if (!rows.length) return emptySvg(svg, width, height, opts.emptyText ?? "no data");

  const labelW = opts.labelWidth ?? Math.min(190, Math.max(96, Math.round(width * 0.26)));
  const padR = 66, padT = 4, padB = 20;
  const plotW = Math.max(24, width - labelW - padR);
  const rowH = (height - padT - padB) / rows.length;
  const barH = Math.min(22, Math.max(9, rowH - 9));
  const fmt = opts.format ?? ((v) => tickLabel(v, opts.dp ?? 2));

  const values = rows.map((r) => r.value).filter(Number.isFinite);
  const maxAbs = Math.max(...values.map(Math.abs), 1e-12);
  const hasNeg = values.some((v) => v < 0);
  const useDiverging = opts.diverging && hasNeg;
  // With negatives in play the zero line sits inside the plot; without them it
  // sits at the left edge and the full width encodes magnitude.
  //
  // The gutter reserves room for the value label of a full-length *negative*
  // bar. Without it the longest negative bar runs to the edge of the plot and
  // its label overprints the category name beside it — the positive side gets
  // the same relief for free from padR.
  const GUTTER = 54;
  const half = plotW / 2;
  const zeroX = useDiverging ? labelW + half : labelW;
  const scale = useDiverging
    ? Math.max(12, half - GUTTER) / maxAbs
    : plotW / maxAbs;

  const posColor = cssVar(useDiverging ? "--div-pos" : "--seq-450");
  const negColor = cssVar("--div-neg");
  const contextColor = cssVar("--context");

  let out = "";
  if (useDiverging) {
    out += `<line class="axisline" x1="${zeroX.toFixed(1)}" y1="${padT}" x2="${zeroX.toFixed(1)}" y2="${(height - padB).toFixed(1)}"/>`;
  }

  rows.forEach((row, i) => {
    const cy = padT + i * rowH + rowH / 2;
    const v = Number.isFinite(row.value) ? row.value : 0;
    const len = Math.abs(v) * scale;
    const x0 = v >= 0 ? zeroX : zeroX - len;
    // Emphasis: the row that carries the story keeps the hue, the rest recede.
    // Colour never encodes rank here — it encodes "this is the one to read".
    const color = useDiverging ? (v >= 0 ? posColor : negColor)
      : row.emphasis === false ? contextColor : posColor;

    out += `<text class="tick" x="${labelW - 10}" y="${(cy + 3.5).toFixed(1)}" text-anchor="end"
      fill="${cssVar(row.emphasis ? "--text-primary" : "--text-secondary")}"
      style="font-size:11.5px;font-weight:${row.emphasis ? 660 : 400}">${esc(row.label)}</text>`;

    // 4px rounded data-end, square against the baseline.
    const r = Math.min(4, barH / 2);
    out += `<rect x="${x0.toFixed(1)}" y="${(cy - barH / 2).toFixed(1)}"
      width="${Math.max(1.5, len).toFixed(1)}" height="${barH.toFixed(1)}"
      rx="${r}" fill="${color}" data-i="${i}"/>`;

    const endX = v >= 0 ? x0 + len + 7 : x0 - 7;
    out += `<text class="marklabel" x="${endX.toFixed(1)}" y="${(cy + 3.5).toFixed(1)}"
      text-anchor="${v >= 0 ? "start" : "end"}"
      fill="${cssVar(row.emphasis ? "--text-primary" : "--text-secondary")}"
      >${esc(fmt(row.value))}${esc(opts.unit ?? "")}</text>`;
  });

  svg.innerHTML = out;

  svg.onpointermove = (event) => {
    const rect = svg.getBoundingClientRect();
    const y = event.clientY - rect.top - padT;
    const i = Math.floor(y / rowH);
    if (i < 0 || i >= rows.length || y < 0) return hideTip(tip);
    const row = rows[i];
    const body = tipRow(null, opts.valueLabel ?? "value", fmt(row.value) + (opts.unit ?? ""))
      + (row.note ? `<div class="tt-row"><span class="tt-key">${esc(row.note)}</span></div>` : "");
    showTip(tip, container, event.clientX, event.clientY, `<b>${esc(row.label)}</b>${body}`);
  };
  svg.onpointerleave = () => hideTip(tip);
}

/* --------------------------------------------------------- grouped bars -- */
/**
 * Two named series across categories — identity, so categorical slots plus a
 * legend rendered by the caller. Capped at 3 series by the palette's all-pairs
 * ceiling; nothing here needs more.
 *
 * data: [{ label, values: [a, b] }]
 * opts: { series: [{ label, colorVar }], dp, unit }
 */
export function drawGrouped({ svg, tip, width, height, container, data, opts }) {
  const rows = data ?? [];
  const series = opts.series ?? [];
  if (!rows.length || !series.length) return emptySvg(svg, width, height, "no data");

  const labelW = opts.labelWidth ?? Math.min(190, Math.max(96, Math.round(width * 0.26)));
  const padR = 62, padT = 4, padB = 20;
  const plotW = Math.max(24, width - labelW - padR);
  const rowH = (height - padT - padB) / rows.length;
  const fmt = (v) => tickLabel(v, opts.dp ?? 2);

  const flat = rows.flatMap((r) => r.values).filter(Number.isFinite);
  const maxAbs = Math.max(...flat.map(Math.abs), 1e-12);
  const hasNeg = flat.some((v) => v < 0);
  // Same label gutter as drawBars: a full-length negative bar must leave room
  // for its own value label without colliding with the category name.
  const GUTTER = 50;
  const half = plotW / 2;
  const zeroX = hasNeg ? labelW + half : labelW;
  const scale = hasNeg ? Math.max(12, half - GUTTER) / maxAbs : plotW / maxAbs;

  // 2px surface gap between adjacent bars — a gap, never a stroked border.
  const GAP = 2;
  const barH = Math.max(6, (rowH - 12 - GAP * (series.length - 1)) / series.length);

  let out = "";
  if (hasNeg) {
    out += `<line class="axisline" x1="${zeroX.toFixed(1)}" y1="${padT}" x2="${zeroX.toFixed(1)}" y2="${(height - padB).toFixed(1)}"/>`;
  }

  rows.forEach((row, i) => {
    const top = padT + i * rowH + (rowH - (barH * series.length + GAP * (series.length - 1))) / 2;
    out += `<text class="tick" x="${labelW - 10}" y="${(padT + i * rowH + rowH / 2 + 3.5).toFixed(1)}"
      text-anchor="end" fill="${cssVar("--text-secondary")}" style="font-size:11.5px">${esc(row.label)}</text>`;

    series.forEach((s, n) => {
      const v = Number.isFinite(row.values[n]) ? row.values[n] : 0;
      const len = Math.abs(v) * scale;
      const x0 = v >= 0 ? zeroX : zeroX - len;
      const y0 = top + n * (barH + GAP);
      out += `<rect x="${x0.toFixed(1)}" y="${y0.toFixed(1)}"
        width="${Math.max(1.5, len).toFixed(1)}" height="${barH.toFixed(1)}"
        rx="${Math.min(4, barH / 2)}" fill="${cssVar(s.colorVar)}"/>`;
      const endX = v >= 0 ? x0 + len + 6 : x0 - 6;
      out += `<text class="marklabel" x="${endX.toFixed(1)}" y="${(y0 + barH / 2 + 3.5).toFixed(1)}"
        text-anchor="${v >= 0 ? "start" : "end"}">${esc(fmt(v))}</text>`;
    });
  });

  svg.innerHTML = out;

  svg.onpointermove = (event) => {
    const rect = svg.getBoundingClientRect();
    const i = Math.floor((event.clientY - rect.top - padT) / rowH);
    if (i < 0 || i >= rows.length) return hideTip(tip);
    const row = rows[i];
    const body = series
      .map((s, n) => tipRow(cssVar(s.colorVar), s.label, fmt(row.values[n]) + (opts.unit ?? "")))
      .join("");
    showTip(tip, container, event.clientX, event.clientY, `<b>${esc(row.label)}</b>${body}`);
  };
  svg.onpointerleave = () => hideTip(tip);
}

/* ---------------------------------------------------- diverging stack ---- */
/**
 * An ordered three-class share centred on its neutral class — the correct form
 * for DeepLOB's down/flat/up softmax. Down and up are the diverging poles, flat
 * is the neutral gray midpoint straddling the centre line, so "no signal"
 * reads as *nothing* rather than as a third opinion.
 *
 * data: { down, flat, up }
 */
export function drawDivergingStack({ svg, width, height, data, tip, container }) {
  const d = data ?? {};
  const down = Number.isFinite(d.down) ? d.down : 0;
  const flat = Number.isFinite(d.flat) ? d.flat : 0;
  const up = Number.isFinite(d.up) ? d.up : 0;
  const total = down + flat + up;
  if (total <= 0) return emptySvg(svg, width, height, "no model signal yet");

  const padX = 8, padB = 17;
  const plotW = width - padX * 2;
  const barH = Math.max(14, height - padB - 6);
  const top = 3;
  const centre = padX + plotW / 2;
  const GAP = 2;

  // Centre the neutral class on the axis: the split point is at half the flat
  // share, so equal down/up renders symmetric regardless of how flat is sized.
  const unit = plotW / total;
  const flatL = centre - (flat / 2) * unit;
  const flatR = centre + (flat / 2) * unit;
  const downX = flatL - down * unit;
  const upW = up * unit;

  const cDown = cssVar("--div-neg");
  const cUp = cssVar("--div-pos");
  const cFlat = cssVar("--div-mid");

  const seg = (x, w, fill, rx) =>
    w <= 0.5 ? "" : `<rect x="${x.toFixed(1)}" y="${top}" width="${w.toFixed(1)}"
      height="${barH.toFixed(1)}" rx="${rx}" fill="${fill}"/>`;

  let out = "";
  out += seg(downX, Math.max(0, flatL - downX - GAP), cDown, 3);
  out += seg(flatL, Math.max(0, flatR - flatL), cFlat, 0);
  out += seg(flatR + GAP, Math.max(0, upW - GAP), cUp, 3);
  out += `<line class="axisline" x1="${centre}" y1="${top - 2}" x2="${centre}" y2="${(top + barH + 2).toFixed(1)}"/>`;
  out += `<text class="tick" x="${padX}" y="${height - 4}">down ${(down * 100).toFixed(0)}%</text>`;
  out += `<text class="tick" x="${centre}" y="${height - 4}" text-anchor="middle">flat ${(flat * 100).toFixed(0)}%</text>`;
  out += `<text class="tick" x="${padX + plotW}" y="${height - 4}" text-anchor="end">up ${(up * 100).toFixed(0)}%</text>`;

  svg.innerHTML = out;

  svg.onpointermove = (event) => {
    showTip(tip, container, event.clientX, event.clientY,
      `<b>DeepLOB class probabilities</b>` +
      tipRow(cDown, "down", (down * 100).toFixed(1) + "%") +
      tipRow(cFlat, "flat", (flat * 100).toFixed(1) + "%") +
      tipRow(cUp, "up", (up * 100).toFixed(1) + "%"));
  };
  svg.onpointerleave = () => hideTip(tip);
}

/* --------------------------------------------------------------- heatmap -- */
/**
 * Confusion matrix. Sequential — one hue, light to dark — because the encoded
 * quantity is a magnitude (row-normalised share), not an identity.
 *
 * data: { matrix: number[][], labels: string[] }
 */
export function drawHeatmap({ svg, tip, width, height, container, data }) {
  const matrix = data?.matrix ?? [];
  const labels = data?.labels ?? [];
  if (!matrix.length) return emptySvg(svg, width, height, "model metrics not available");

  const n = matrix.length;
  // padT carries two stacked rows of chrome: the "predicted" axis title and the
  // column labels beneath it. Sized for both, or the title overprints them.
  const padL = 74, padT = 34, padR = 8, padB = 26;
  const cell = Math.min((width - padL - padR) / n, (height - padT - padB) / n);
  const gridW = cell * n;
  const GAP = 2;

  // Row-normalised: each row is "of the actual X, what did we predict?", which
  // is the question a confusion matrix is read to answer, and it keeps class
  // imbalance from making one row dominate the colour scale.
  const rowTotals = matrix.map((row) => row.reduce((a, b) => a + b, 0) || 1);
  const ramp = ["--seq-100", "--seq-250", "--seq-350", "--seq-450", "--seq-550", "--seq-650"]
    .map(cssVar);

  let out = "";
  out += `<text class="axis-title" x="${padL}" y="11">predicted →</text>`;
  labels.forEach((label, j) => {
    out += `<text class="tick" x="${(padL + j * cell + cell / 2).toFixed(1)}" y="${padT - 8}"
      text-anchor="middle">${esc(label)}</text>`;
  });

  matrix.forEach((row, i) => {
    out += `<text class="tick" x="${padL - 9}" y="${(padT + i * cell + cell / 2 + 3.5).toFixed(1)}"
      text-anchor="end">${esc(labels[i] ?? i)}</text>`;
    row.forEach((value, j) => {
      const share = value / rowTotals[i];
      const step = ramp[Math.min(ramp.length - 1, Math.floor(share * ramp.length))];
      const x = padL + j * cell, y = padT + i * cell;
      out += `<rect x="${(x + GAP / 2).toFixed(1)}" y="${(y + GAP / 2).toFixed(1)}"
        width="${(cell - GAP).toFixed(1)}" height="${(cell - GAP).toFixed(1)}"
        rx="3" fill="${step}" data-i="${i}" data-j="${j}"/>`;
      // Ink stays a text token; the cell fill carries the magnitude. White on
      // the darkest two steps is the contrast exception, not a series colour.
      const dark = share >= 0.5;
      out += `<text x="${(x + cell / 2).toFixed(1)}" y="${(y + cell / 2 + 4).toFixed(1)}"
        text-anchor="middle" style="font-size:11.5px;font-variant-numeric:tabular-nums"
        fill="${dark ? "#fff" : cssVar("--text-primary")}">${(share * 100).toFixed(0)}%</text>`;
    });
  });
  out += `<text class="axis-title" x="4" y="${height - 6}">actual ↓</text>`;

  svg.innerHTML = out;

  svg.onpointermove = (event) => {
    const rect = svg.getBoundingClientRect();
    const j = Math.floor((event.clientX - rect.left - padL) / cell);
    const i = Math.floor((event.clientY - rect.top - padT) / cell);
    if (i < 0 || j < 0 || i >= n || j >= n) return hideTip(tip);
    const count = matrix[i][j];
    showTip(tip, container, event.clientX, event.clientY,
      `<b>actual ${esc(labels[i])} → predicted ${esc(labels[j])}</b>` +
      tipRow(null, "share of actual", ((count / rowTotals[i]) * 100).toFixed(1) + "%") +
      tipRow(null, "samples", count.toLocaleString("en-US")));
  };
  svg.onpointerleave = () => hideTip(tip);
}

/* ------------------------------------------------------------ sparkline -- */
/** Trend context inside a stat tile. No axes, no labels — the tile's value is
 *  the number; this is only the shape. */
export function drawSparkline({ svg, width, height, data, opts }) {
  const values = (data ?? []).filter(Number.isFinite);
  if (values.length < 2) { svg.innerHTML = ""; return; }
  const lo = Math.min(...values, 0), hi = Math.max(...values, 0);
  const span = (hi - lo) || 1;
  const x = (i) => (i / (values.length - 1)) * width;
  const y = (v) => 2 + (1 - (v - lo) / span) * (height - 4);
  const color = cssVar(opts?.colorVar ?? "--series-1");
  const path = values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const base = y(Math.max(lo, Math.min(hi, 0)));
  svg.innerHTML =
    `<path d="${path}L${width},${base.toFixed(1)}L0,${base.toFixed(1)}Z" fill="${color}" opacity="0.10"/>` +
    `<path d="${path}" fill="none" stroke="${color}" stroke-width="1.5"
      stroke-linejoin="round" stroke-linecap="round"/>`;
}
