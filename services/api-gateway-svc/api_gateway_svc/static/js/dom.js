/* Minimal DOM helpers.
 *
 * `el` builds elements from a tag/props/children triple and `text` sets
 * textContent. Everything that renders user-visible strings goes through one of
 * them, which means no innerHTML on any path that touches server data — the
 * risk reason, a service's health detail and a venue symbol all originate
 * outside this file.
 *
 * SVG is the one exception: chart bodies are assembled as markup strings and
 * assigned to innerHTML, so every value interpolated into them is either a
 * number this module produced or passed through `esc`.
 */

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

/** Build an element. Children may be nodes, strings, or nested arrays. */
export function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = String(value);
    else if (key === "html") node.innerHTML = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of [children].flat(4)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

/** Escape a string for safe interpolation into SVG/HTML markup. */
export function esc(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

/** Set textContent only when it changed.
 *  The live view rewrites ~40 fields twice a second; skipping no-op writes
 *  keeps the browser from invalidating layout for values that did not move,
 *  and stops text selection being dropped mid-read. */
export function text(node, value) {
  if (!node) return;
  const next = String(value);
  if (node.textContent !== next) node.textContent = next;
}

export function setClass(node, value) {
  if (node && node.className !== value) node.className = value;
}

/** Replace a container's children in one shot. */
export function replace(node, children) {
  if (!node) return;
  node.replaceChildren(...[children].flat(4).filter(Boolean));
}

export function empty(message) {
  return el("p", { class: "empty", text: message });
}
