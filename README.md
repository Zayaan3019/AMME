# AMME — Adaptive Market Making Engine

A production-grade, microservices market-making system: live Level-2 order book
ingestion from Binance, microstructure feature extraction (VPIN / OFI), optimal
quoting via Avellaneda–Stoikov stochastic control, a DeepLOB drift signal, a
hard risk layer with circuit breakers, and a queue-position-aware paper matching
engine — with a deterministic regression harness that measures what each
component is actually worth.

Runs entirely on free infrastructure. No API keys, no paid data, no broker.

```bash
git clone <repo> && cd amme
cp .env.example .env
docker compose -f infra/docker-compose.yml up -d
open http://localhost:8000     # live dashboard
```

---

## What it does

```
Binance WebSocket ─▶ ingestion ─▶ features ─▶ DeepLOB ─▶ quoting ─▶ risk ─▶ execution ─▶ dashboard
   (real L2 book)    (resync,     (VPIN, OFI,  (drift)    (A-S)    (veto)   (paper fills)
                      clock skew)   sigma)
```

Eight independently deployable services over Redis Streams, each with `/health`
and `/metrics`, scraped by Prometheus and rendered in Grafana.

## The dashboard

`http://localhost:8000` — three views over the same read-only gateway.

| View | What it answers |
|---|---|
| **Live** | What is the strategy doing right now, and why? L2 depth with our own quotes marked, the A-S quote decomposed term by term, DeepLOB's three-class drift, VPIN/OFI/σ with warm-up state, fills and the public tape. |
| **System** | Is the pipeline whole? All eight services probed directly (not inferred from stream traffic — a service can move messages while failing its own health assertion), stream throughput, and the exact parameters the running system is using. |
| **Research** | Does any of it work? The five-arm ablation, paired mechanism deltas, the high-/low-VPIN toxicity split, the venue screen, DeepLOB's training curve and confusion matrix, and the latency budgets. |

Every figure on **Research** is read from `artifacts/` at request time — the files
`make ablation`, `make screen` and `make train` actually wrote. Nothing is
transcribed, so the page cannot drift from the measurements.

The gateway is read-only by construction: it has no route that accepts a
mutating HTTP method, and a test asserts it. A dashboard that can write is a
route around the risk layer.

No build step, no bundler, no CDN — the page is ES modules and CSS served
straight from the service, so it works inside the container with no network
egress. Charts are hand-rolled SVG against a palette validated for
colour-vision deficiency in both light and dark mode; every chart has a
table-view twin.

**What is real:** the market data, the book reconstruction, the features, the
model, the quoting maths, the risk layer, and the services.
**What is simulated:** order execution. Fills are matched against the *real*
tape with a queue-position model. That is the design (spec §5.6) — real fills
need a broker and capital, and paper matching is more rigorous for evaluation
because it is deterministic and reproducible.

---

## Headline results

### Ablation — 10 seeds, 900s each, paired, all five arms

Every arm sees byte-identical market data within each seed, so comparisons are
paired and path noise cancels. γ, κ and λ are all grid-searched, not chosen.

| Arm | P&L (mean ± sd) | Sharpe | Max DD | Inventory σ | Profitable seeds |
|---|---:|---:|---:|---:|---:|
| Naive symmetric | −0.158 ± 0.329 | −392 | 0.394 | 0.01449 | 2/10 |
| Pure Avellaneda–Stoikov | +0.027 ± 0.069 | 243 | 0.073 | 0.00405 | 8/10 |
| A-S + toxicity gate | +0.041 ± 0.072 | 385 | 0.062 | **0.00394** | 8/10 |
| A-S + DeepLOB drift | +0.128 ± 0.128 | **596** | 0.054 | 0.00828 | 8/10 |
| **Full AMME** | **+0.129 ± 0.127** | 590 | **0.051** | 0.00803 | 8/10 |

**Every mechanism earns its place.** Paired, seed by seed:

| Comparison | Δ P&L | Δ Sharpe | Seeds won |
|---|---:|---:|---:|
| A-S vs naive | **+0.185** | **+635** | 8/10 |
| Toxicity gate | +0.014 | +142 | 7/10 |
| DeepLOB drift | **+0.101** | **+353** | 8/10 |

The build-up is monotone in both P&L and Sharpe — AMME > A-S+toxicity >
pure A-S > naive — which is what the spec's Definition of Done requires.

**The sharpest result is the toxicity split.** 86% of the naive baseline's loss
is incurred during high-VPIN periods: it is being adversely selected, exactly as
theory predicts. A-S neutralises that, the toxicity gate triples high-VPIN P&L,
and the drift signal turns those periods into the strategy's main earner.

| Arm | P&L in high-VPIN | P&L in low-VPIN |
|---|---:|---:|
| Naive symmetric | **−0.1446** | −0.0241 |
| Pure Avellaneda–Stoikov | +0.0046 | +0.0121 |
| **Full AMME** | **+0.0949** | +0.0335 |

Full tables, including the earlier *incorrect* version and why it was wrong, in
[docs/RESULTS.md](docs/RESULTS.md).

### Measured latency

| Stage | p50 | p99 | Budget |
|---|---:|---:|---:|
| Binance ingest staleness (clock-corrected) | 63.8 ms | 422 ms | — |
| A-S quote computation | < 50 µs | — | 50 µs |
| DeepLOB inference (ONNX, T=50) | 2.83 ms | **4.04 ms** | 5 ms ✅ |
| DeepLOB inference (ONNX, T=100) | 5.42 ms | 7.06 ms | 5 ms ❌ |
| DeepLOB inference (TorchScript, T=100) | 11.44 ms | 15.51 ms | 5 ms ❌ |

The paper's T=100 does not fit a 5ms CPU budget on this hardware. Halving the
lookback and serving via ONNX Runtime gets p99 to 4.04ms — a 2.8× improvement
over TorchScript.

---

## Four findings worth reading

These came out of measurement, not design, and each changed the system.

### 1. The local clock was 677ms ahead of Binance

Reported staleness was ~734ms against a true one-way latency of ~63ms — an 11×
error. NTP does not help: the host clock can be perfectly correct and still
disagree with the exchange, and it is the *exchange's* clock that timestamps the
data. `ClockSync` measures the offset with an NTP-style estimator and every
latency metric in the system is corrected by it.

### 2. BTCUSDT is the *worst* venue for this strategy — so the system picks another

`make screen` ranks live symbols by

```
makeability = spread / (σ · √holding)
```

the spread expressed in units of the volatility accrued while holding
inventory. That is the quantity that decides whether a passive quoter has
positive expectancy; spread in ticks and spread in bps both rank symbols wrongly.

| Symbol | Spread | Makeability | Verdict |
|---|---:|---:|---|
| AVAXUSDT | 1.52 bps | **1.41** | workable |
| ATOMUSDT | 7.42 bps | 0.84 | marginal |
| ETHUSDT | 0.05 bps | 0.04 | hostile |
| BTCUSDT | 0.0015 bps | **0.00** | hostile |

**Every Binance spot symbol screened has a binding tick** — the spread sits at
exactly one tick, so the venue itself caps maker revenue. What separates them is
how much volatility that one tick must cover. On BTCUSDT it is hopeless; on
AVAXUSDT one tick is 1.52 bps against a ~1s queue, which works.

The live default is now AVAXUSDT, chosen by measurement rather than assumption.

### 3. A textbook VPIN threshold fired 99.3% of the time

VPIN's absolute level depends on bucket size, trade-size distribution and venue.
The literature-standard 0.35 sat below the *10th percentile* of observed VPIN in
this regime, so the toxicity gate widened permanently rather than reacting —
fills fell from 151 to 16 and the "toxicity" arm became strictly worse than the
baseline it was supposed to improve.

The gate now tracks a **quantile** (p90) of recent VPIN, so the parameter means
the same thing on any venue: *react to the most toxic 10% of the tape*.

### 4. A clock correction that was measured once and then trusted forever

`ClockSync` re-measures the exchange offset every `resync_interval_seconds`
(300s) — except `maybe_resync` was only ever called from the WebSocket reconnect
path. On a socket that stays up for hours, the offset was measured exactly once
and the documented interval never applied.

That turns a transient measurement error into a permanent one. Caught on a live
stack while building the dashboard: the host suspended, the Docker VM's clock
resumed ~13s behind, the socket reconnected, and `ClockSync` sampled *during the
catch-up window* and latched `offset=-13351ms`. The VM clock corrected itself
seconds later. The offset did not.

For the next 45 minutes every latency metric in the system was wrong by 13.35
seconds — and nothing could detect it, because staleness is *computed with the
offset applied*. A wrong correction does not look like an error; it looks like a
slow venue.

Three changes: the resync runs on its own timer, so the damage is bounded by one
interval; `/health` reports the *age* of the measurement, not just its value;
and the offset is exported as `amme_clock_offset_ms`, because the only way to
catch a bad correction is to watch the correction itself.

---

## Quick start

### Full stack (Docker)

```bash
cp .env.example .env
make up            # or: docker compose -f infra/docker-compose.yml up -d
make health        # probe all 8 /health endpoints
make logs
```

| Service | Port | |
|---|---|---|
| Dashboard | 8000 | Live / System / Research |
| Grafana | 3000 | `admin` / `admin` |
| Prometheus | 9090 | |
| ingestion / feature / deeplob / quoting / risk / execution / backtest | 8001–8007 | `/health`, `/metrics` |

### Local development (no Docker)

```bash
make install
make test                # 268 tests
make regression-test     # lint + types + tests + ablation report
```

### Research

```bash
make calibrate    # grid-search gamma/kappa -> artifacts/calibration.md
make ablation     # multi-seed ablation    -> artifacts/ablation_multiseed.md
make train        # train DeepLOB          -> models/deeplob.{ts,onnx}
```

Then set `AMME_MODEL_BACKEND=onnx` and `AMME_AS_DRIFT_LAMBDA=<calibrated>` to
activate the drift signal.

---

## Testing

268 tests. The ones that carry weight:

**Hand-computed quant verification** — every formula checked against a value
worked out by hand, not captured from the implementation. E.g. A-S with
γ=0.5, σ=2, τ=1, κ=0.5 must give spread = 2 + 4·ln2 = 4.7725887.

**Resync chaos suite** — the Binance snapshot+diff protocol as a *pure* state
machine, so every failure mode is directly constructible: sequence gaps, stale
snapshots, mid-stream disconnects, duplicate replays, crossed books, and the
boundary case where one diff straddles the snapshot.

**Byte-for-byte determinism** — the full pipeline hashed over its entire P&L and
inventory trajectory, not just the endpoint. Plus AST-walking tests that forbid
global RNG and wall-clock reads anywhere in `amme_core`.

**Property-based invariants** (hypothesis) — spread is always positive, quotes
never cross, long inventory always marks down, an approved quote can never
breach `q_max`.

**Latency budgets** — per-stage p99 assertions, sized to catch an
order-of-magnitude regression rather than CI noise.

**Bug regressions** — every defect found by auditing the *running* system, each
pinned with the reason it mattered. Volume conservation in the matching engine,
NaN failing the risk layer closed, corrupt-fill rejection, forward-compatible
event decoding. All of these passed the unit suite and failed only once real
containers ran.

---

## Documentation

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Every service boundary, the alternative considered, why it lost |
| [docs/QUANT_DERIVATIONS.md](docs/QUANT_DERIVATIONS.md) | Full HJB derivation of A-S; OFI, VPIN, queue model, Sharpe conventions |
| [docs/RESULTS.md](docs/RESULTS.md) | Ablation tables, calibration sweeps, model metrics |

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Transport | Redis Streams | Consumer groups + replayable log in a 6MB container. Kafka's ops cost buys nothing at 10 msg/s; `EventBus` is the seam if that changes. |
| Storage | TimescaleDB | Tick storage is append-heavy; hypertables keep inserts on a small recent chunk instead of one ever-growing B-tree. |
| Contracts | Pydantic v2 | Versioned schemas, validation at the boundary. A crossed book is rejected at parse time. |
| ML | PyTorch → ONNX Runtime | 2.8× faster CPU inference; the only backend that meets the 5ms budget. |
| Observability | Prometheus + Grafana + structlog | Correlation IDs trace one tick across all 8 processes. |
| Services | FastAPI + uvicorn | Uniform `/health` + `/metrics` even for pure stream consumers. |

---

## Cost

**$0.** Binance public market data requires no key. Every dependency is
open-source and self-hosted. GPU training is optional and free on Colab/Kaggle;
the CPU path trains the reference model in minutes.

---

## Status

| Phase | State |
|---|---|
| 0 — Scaffolding, contracts, quant core | ✅ |
| 1 — Live Binance ingestion, resync, persistence | ✅ verified against production |
| 2 — Feature / quoting / risk / execution services | ✅ |
| 3 — Calibration + multi-seed ablation harness | ✅ |
| 4 — DeepLOB training, export, inference service | ✅ trained, exported, serving; drift arms in the ablation |
| 5 — Prometheus, Grafana, dashboard | ✅ |
| 6 — Hardening, latency profiling, docs | ✅ |

Known limitations are listed in
[ARCHITECTURE.md §8](docs/ARCHITECTURE.md#8-open-decisions-and-known-limitations)
rather than omitted.
