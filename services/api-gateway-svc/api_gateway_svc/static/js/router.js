/* Hash router.
 *
 * Hash rather than History API on purpose: the gateway serves a single page and
 * has no catch-all route, so a real path like /research would 404 on refresh or
 * on a shared link. A hash keeps every URL bookmarkable without teaching the
 * server about the front-end's routes.
 *
 * Views are objects: { mount(root), enter(), leave() }. `mount` runs once and
 * builds the DOM; `enter`/`leave` bracket visibility so a background view can
 * stop doing work it cannot show.
 */

export function createRouter(routes, { onChange } = {}) {
  const names = Object.keys(routes);
  const mounted = new Set();
  let current = null;

  const resolve = () => {
    const raw = location.hash.replace(/^#\/?/, "").split("?")[0];
    return names.includes(raw) ? raw : names[0];
  };

  const go = () => {
    const next = resolve();
    if (next === current) return;

    if (current) routes[current].leave?.();
    current = next;

    for (const name of names) {
      const view = routes[name];
      view.root.hidden = name !== next;
    }

    const view = routes[next];
    if (!mounted.has(next)) { view.mount?.(); mounted.add(next); }
    view.enter?.();

    for (const tab of document.querySelectorAll("[data-route]")) {
      const active = tab.dataset.route === next;
      // aria-current is the accessible signal; the styling hangs off it too, so
      // the two can never disagree.
      if (active) tab.setAttribute("aria-current", "page");
      else tab.removeAttribute("aria-current");
    }

    document.title = `AMME — ${routes[next].title ?? next}`;
    onChange?.(next);
  };

  window.addEventListener("hashchange", go);
  return { start: go, get current() { return current; } };
}
