/* Entry point: wire the store, the router and the three views.
 *
 * Load order matters only here. Everything else is an ES module with explicit
 * imports, so the browser resolves the graph itself — no bundler, no global
 * namespace, and no script tag ordering to get wrong.
 */

import { $, text, setClass } from "./dom.js";
import { store } from "./store.js";
import { createRouter } from "./router.js";
import { liveView } from "./views/live.js";
import { systemView } from "./views/system.js";
import { researchView } from "./views/research.js";

/* ---------------------------------------------------------------- theme -- */

const THEME_KEY = "amme-theme";

function applyTheme(theme) {
  if (theme === "light" || theme === "dark") {
    document.documentElement.dataset.theme = theme;
  } else {
    delete document.documentElement.dataset.theme;
  }
}

function currentTheme() {
  const stored = localStorage.getItem(THEME_KEY);
  if (stored === "light" || stored === "dark") return stored;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function initTheme() {
  const stored = localStorage.getItem(THEME_KEY);
  if (stored) applyTheme(stored);

  $("#theme-toggle").addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    localStorage.setItem(THEME_KEY, next);
    applyTheme(next);
    repaintAll();
  });

  // Charts read their colours from CSS custom properties at draw time, so an
  // OS-level theme change has to trigger a redraw or every stroke keeps the
  // previous mode's hue.
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (!localStorage.getItem(THEME_KEY)) repaintAll();
  });
}

function repaintAll() {
  liveView.update?.("all");
  researchView.repaint?.();
}

/* --------------------------------------------------------------- status -- */

function renderStatusPills() {
  const s = store.state;

  const conn = $("#pill-conn");
  const live = store.connection === "live";
  setClass(conn, "pill " + (live ? "is-live" : "is-down"));
  text($("#conn-text"), store.connection);

  text($("#symbol-text"), s.symbol || store.config?.symbol || "—");

  const risk = $("#pill-risk");
  const halted = s.breaker_tripped || s.risk_action === "halt";
  const action = s.risk_action ?? "—";
  setClass(risk, "pill " + (halted ? "is-halt" : action === "approve" ? "is-live" : action === "reject" ? "is-warn" : ""));
  text($("#risk-text"), halted ? "HALTED" : `risk ${action}`);
}

/* ---------------------------------------------------------------- boot --- */

async function boot() {
  initTheme();

  liveView.root = $("#view-live");
  systemView.root = $("#view-system");
  researchView.root = $("#view-research");

  const router = createRouter({
    live: liveView,
    system: systemView,
    research: researchView,
  });

  store.subscribe((event) => {
    renderStatusPills();
    if (router.current === "live") liveView.update(event);
  });

  // Bootstrap first so the live view has history and, crucially, the tick size
  // before the first render — price precision depends on it.
  await store.bootstrap();

  if (!location.hash) location.hash = "#/live";
  router.start();

  store.connect();
  store.startPolling();
  renderStatusPills();
}

boot().catch((err) => {
  console.error("AMME dashboard failed to start", err);
  const main = $("#main");
  if (main) {
    main.insertAdjacentHTML("afterbegin",
      '<div class="banner crit"><span class="banner-icon">■</span><div>' +
      "<strong>The dashboard failed to start.</strong>" +
      '<div class="cap">Check the browser console and that api-gateway-svc is reachable.</div>' +
      "</div></div>");
  }
});
