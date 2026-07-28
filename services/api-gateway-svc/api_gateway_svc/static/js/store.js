/* Live data layer: one WebSocket, one rolling history, a pub/sub for views.
 *
 * The gateway pushes the whole state twice a second rather than diffs. At ~60
 * scalar fields that is a few KB/s on a loopback socket, and it makes the
 * client stateless with respect to reconnection — a client that has just
 * reconnected is correct on its first message rather than after the first
 * change to every field.
 */

const MAX_POINTS = 900;

/** Fetch JSON, returning null instead of throwing.
 *  A panel that cannot load renders its own empty state; it must never take the
 *  page down with it. */
export async function getJSON(path) {
  try {
    const response = await fetch(path, { headers: { accept: "application/json" } });
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

class Store {
  constructor() {
    this.state = {};
    this.book = { bids: [], asks: [] };
    this.history = [];
    this.fills = [];
    this.trades = [];
    this.config = null;
    this.connection = "connecting";
    this._listeners = new Set();
    this._socket = null;
    this._retry = 0;
    this._retryTimer = null;
  }

  subscribe(fn) {
    this._listeners.add(fn);
    return () => this._listeners.delete(fn);
  }

  _emit(event) {
    for (const fn of this._listeners) {
      try { fn(event, this); } catch (err) { console.error("store listener failed", err); }
    }
  }

  async bootstrap() {
    const [history, fills, trades, config] = await Promise.all([
      getJSON("/api/history"),
      getJSON("/api/fills"),
      getJSON("/api/trades"),
      getJSON("/api/config"),
    ]);
    if (history?.points) this.history = history.points.slice(-MAX_POINTS);
    if (fills?.fills) this.fills = fills.fills;
    if (trades?.trades) this.trades = trades.trades;
    if (config) this.config = config;
    this._emit("bootstrap");
  }

  connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    let socket;
    try {
      socket = new WebSocket(`${proto}//${location.host}/ws`);
    } catch {
      return this._scheduleReconnect();
    }
    this._socket = socket;

    socket.onopen = () => {
      this._retry = 0;
      this.connection = "live";
      this._emit("connection");
    };

    socket.onmessage = (event) => {
      let message;
      try { message = JSON.parse(event.data); } catch { return; }
      if (message.type !== "state") return;
      this.state = message.data ?? {};
      if (message.book) this.book = message.book;
      // Only sample once the book is real: a leading run of zeros would
      // otherwise anchor every chart's y-axis at the origin.
      if (this.state.mid_price > 0) {
        this.history.push({
          t: this.state.last_update_ns / 1e9,
          pnl: this.state.total_pnl,
          realized: this.state.realized_pnl,
          inventory: this.state.inventory,
          mid: this.state.mid_price,
          vpin: this.state.vpin,
        });
        if (this.history.length > MAX_POINTS) this.history.shift();
      }
      this._emit("state");
    };

    socket.onclose = () => {
      this.connection = "reconnecting";
      this._emit("connection");
      this._scheduleReconnect();
    };

    socket.onerror = () => socket.close();
  }

  /** Exponential backoff, capped. The gateway is usually local, so the first
   *  retry is fast; the cap keeps a genuinely-down stack from being hammered
   *  by a tab left open overnight. */
  _scheduleReconnect() {
    clearTimeout(this._retryTimer);
    const delay = Math.min(15000, 500 * 2 ** this._retry);
    this._retry = Math.min(this._retry + 1, 5);
    this._retryTimer = setTimeout(() => this.connect(), delay);
  }

  /** Poll the endpoints that are not on the WebSocket. Fills and trades change
   *  far less often than state and are lists rather than scalars. */
  startPolling() {
    const tick = async () => {
      if (document.hidden) return;
      const [fills, trades] = await Promise.all([
        getJSON("/api/fills"),
        getJSON("/api/trades"),
      ]);
      if (fills?.fills) this.fills = fills.fills;
      if (trades?.trades) this.trades = trades.trades;
      this._emit("lists");
    };
    setInterval(tick, 2500);
  }
}

export const store = new Store();
export { MAX_POINTS };
