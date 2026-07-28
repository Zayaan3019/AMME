/* Number and time formatting.
 *
 * One rule throughout: a value that does not exist renders as an em-dash, never
 * as 0. On a trading screen those mean opposite things — "no data yet" and "a
 * position of exactly zero" — and a dashboard that conflates them is worse than
 * one that shows nothing.
 */

const DASH = "—";

export const isNum = (v) => typeof v === "number" && Number.isFinite(v);

export function num(v, dp = 2) {
  return isNum(v) ? v.toFixed(dp) : DASH;
}

export function signed(v, dp = 2) {
  if (!isNum(v)) return DASH;
  return (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(dp);
}

/** Compact integer with thin thousands separators. */
export function int(v) {
  return isNum(v) ? Math.round(v).toLocaleString("en-US") : DASH;
}

export function pct(v, dp = 1) {
  return isNum(v) ? (v * 100).toFixed(dp) + "%" : DASH;
}

/** Price with a symbol-appropriate number of decimals.
 *  AVAXUSDT trades near 6.5 and BTCUSDT near 65000; a fixed 2dp is either
 *  noise or a loss of every meaningful digit depending on which is loaded. */
export function price(v, tick = 0.01) {
  if (!isNum(v)) return DASH;
  const dp = tick > 0 ? Math.max(0, Math.min(8, Math.ceil(-Math.log10(tick)))) : 2;
  return v.toFixed(dp);
}

/** Duration as the largest two units that fit: "2h 14m", "45s". */
export function duration(seconds) {
  if (!isNum(seconds) || seconds < 0) return DASH;
  const s = Math.floor(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60), rs = s % 60;
  if (m < 60) return `${m}m ${rs}s`;
  const h = Math.floor(m / 60), rm = m % 60;
  if (h < 24) return `${h}h ${rm}m`;
  return `${Math.floor(h / 24)}d ${h % 24}h`;
}

export function clockTime(epochSeconds) {
  if (!isNum(epochSeconds) || epochSeconds <= 0) return DASH;
  return new Date(epochSeconds * 1000).toLocaleTimeString("en-GB", { hour12: false });
}

/** Latency, auto-scaled: microseconds below 1ms, else milliseconds. */
export function latency(ms) {
  if (!isNum(ms)) return DASH;
  if (ms < 1) return `${(ms * 1000).toFixed(0)}µs`;
  if (ms < 1000) return `${ms.toFixed(ms < 10 ? 2 : 1)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

/** Axis ticks: drop trailing noise so an axis reads 0.05 rather than 0.050000. */
export function tickLabel(v, dp = 2) {
  if (!isNum(v)) return "";
  const abs = Math.abs(v);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e6)) return v.toExponential(1);
  return Number(v.toFixed(dp)).toString();
}

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export { DASH };
