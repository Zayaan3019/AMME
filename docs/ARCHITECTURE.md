# Architecture

Why the system is shaped the way it is. Every boundary below was a decision with
alternatives; this document records the alternative and the reason it lost.

---

## 1. The pipeline

```
Binance WS ──▶ ingestion-svc ──▶ lob.raw ─────┬──▶ feature-svc ──▶ features.ready ──┬──▶ deeplob-inference-svc
              (book resync,          lob.trades │   (VPIN, OFI,                      │        │
               clock skew)                      │    sigma, tensor)                  │        ▼
                    │                           │                                    │   drift.signal
                    ▼                           │                                    │        │
              TimescaleDB                       │                                    ▼        │
              (replay/training)                 │                            quoting-engine-svc ◀┘
                                                │                                    │
                                                │                              quote.intent
                                                │                                    │
                                                │                                    ▼
                                                │                               risk-svc  ◀── veto authority
                                                │                                    │
                                                │                             risk.decision
                                                │                                    │
                                                └───────────────────────────▶ execution-sim-svc
                                                     (real tape for matching)         │
                                                                          exec.fill / exec.portfolio
                                                                                      │
                                                                    ┌─────────────────┴──────────────┐
                                                                    ▼                                ▼
                                                            api-gateway-svc                    backtest-svc
                                                            (dashboard, WS)                  (ablation, calibration)
```

Cross-cutting: Prometheus scrapes `/metrics` on every service; Grafana reads
Prometheus; every event carries a `correlation_id` propagated unchanged from the
book update that caused it.

---

## 2. Service boundaries

### 2.1 Why `risk-svc` is a separate process

This is the boundary that matters most, and the one most likely to be
challenged.

**The decision.** `quoting-engine-svc` emits `quote.intent`. It cannot reach the
execution layer. `risk-svc` consumes intents and emits `risk.decision`.
`execution-sim-svc` subscribes **only** to `risk.decision`.

**Why not a function call inside the quoting engine?** Because risk must remain
correct when the strategy is *wrong*. Consider the failure modes:

| Failure | In-process risk | Separate process |
|---|---|---|
| Strategy crashes | Risk dies with it | Risk survives; no intents arrive; nothing executes |
| Strategy hangs / infinite loop | Risk never runs | Risk still processes; stale intents age out |
| Strategy emits NaN or absurd quotes | Shared memory may be corrupt | Risk validates from *its own* state |
| Bad config deployed to strategy | Both mis-configured together | Risk limits deploy separately |

The last row is the practical one. Risk limits and strategy parameters have
different change-control requirements: you tune $\gamma$ often and $q_{\max}$
almost never. Separate processes make that separation enforceable rather than
conventional.

**The enforcement point.** `execution-sim-svc` subscribes to `risk.decision` and
not to `quote.intent`. Bypassing risk would require changing which stream that
service reads — a visible, reviewable diff, not a missing `if`.

**Risk keeps its own state.** Inventory and P&L come from `exec.portfolio`, never
from the intent. If the strategy believes it is flat and is not, risk is not
fooled.

**The cost, stated honestly.** One extra hop: ~0.5–2ms of Redis round trip, and
a window where an intent is authorised against inventory a few milliseconds
stale. At a 100ms book cadence that is acceptable. At microsecond latencies it
would not be, and risk would have to be an in-process library with a separate
review gate — a different design for a different regime.

### 2.2 Why `feature-svc` is separate from `ingestion-svc`

Ingestion is I/O-bound and must never block; feature computation is CPU-bound
and bursty. Fusing them means a slow feature calculation applies backpressure to
the socket, and Binance disconnects a slow consumer. Splitting lets the stream
buffer absorb a stall.

It also means feature logic can be redeployed without dropping the WebSocket
connection and re-running the snapshot handshake — which takes seconds and
during which we are blind.

### 2.3 Why `feature-svc` consumes both book and trades

OFI needs the book; VPIN needs the tape. Splitting them into two services would
require a synchronisation protocol to re-establish "which trades preceded this
book state" — ordering we get for free by consuming both in one service.

### 2.4 Why `deeplob-inference-svc` is off the critical path

The quoting engine consumes `drift.signal` **asynchronously** and caches the
latest value; it never blocks on a prediction.

Blocking would put a neural network on the critical path of every quote and
couple quoting availability to inference availability. Instead, a prediction
older than 2s is ignored and the engine degrades to pure Avellaneda–Stoikov.
Degrading to a known-good strategy is a good failure mode; not quoting is not.

This is also why `build_predictor` falls back to the stub on *any* load failure
rather than raising. A service that refuses to start takes the drift signal
offline **and** leaves its stream unconsumed.

### 2.5 Why `execution-sim-svc` is simulated

The $0 constraint. Real fills need a broker and capital. Paper matching against
the *real* tape is also strictly better for evaluation: fills are deterministic
and reproducible, which is what makes the byte-for-byte replay test possible.

The realism lives in the queue model (§7 of `QUANT_DERIVATIONS.md`), not in
having real money at risk.

---

## 3. Transport: Redis Streams, not Kafka

**Chosen:** Redis Streams. **Rejected:** Kafka, NATS, raw sockets.

At ~10 events/sec/stream, Kafka's operational cost — KRaft/ZooKeeper, partition
management, a JVM per broker, ~1GB RAM before a single message — buys nothing.
Redis Streams provides the two properties actually needed:

1. **Consumer groups** with at-least-once delivery and per-consumer acks.
2. **A replayable log**, so a restarted consumer resumes rather than losing data.

in one ~6MB container.

**The migration path is designed in, not hypothetical.** `EventBus` is a
`Protocol`; `RedisEventBus` and `InMemoryEventBus` implement it. A
`KafkaEventBus` would implement the same three methods and no service logic
would change. That seam is also what lets the backtester run the identical
pipeline in-process with zero infrastructure.

**Where Redis Streams would stop being right:** sustained throughput above
~50k msg/sec, cross-datacentre replication, or multi-day retention with
compaction. None apply here.

**Delivery is at-least-once**, so consumers must tolerate duplicates.
`event_id` is the dedup key.

---

## 4. Events carry state, not diffs

`ingestion-svc` publishes a complete top-N book snapshot on every update rather
than the diff it received.

The cost is bandwidth — irrelevant at 10 levels and 10Hz. The benefit is that
every downstream consumer is **stateless with respect to book reconstruction**.
`feature-svc` can be restarted mid-stream and be correct on its very first
message. With diffs, every consumer would need its own copy of the resync state
machine, and every one would be a place to get it wrong.

---

## 5. The `amme_core` / `amme_shared` split

```
shared/amme_shared/   I/O, transport, config, observability, service scaffolding
shared/amme_core/     pure domain logic — no I/O, no wall clock, no globals
```

`amme_core` obeys four rules, enforced by tests:

1. **No I/O.** No network, disk, or database.
2. **No wall clock.** Time arrives as an argument.
3. **No hidden randomness.** Every stochastic component takes an explicit
   `numpy.random.Generator`.
4. **No global mutable state.**

Rules 2 and 3 are checked by AST-walking tests
(`test_no_wall_clock_in_core`, `test_no_global_random_in_core`) that inspect real
call expressions — an earlier regex version matched the phrase `time.time()`
inside a docstring explaining the rule and failed the test it documented.

**Why this matters:** it is what makes the backtester and the live services run
*the same code*. There is exactly one implementation of the A-S maths in this
repo. A result in `RESULTS.md` is a statement about the code that runs in
production, not about a research copy of it.

**Deviation from spec §9:** the spec lists only `shared/` for "shared pydantic
event schemas". Two packages live there instead. `amme_core` is deliberately
dependency-light (numpy only), so the backtester and training pipeline import
the strategy code without dragging in redis/fastapi.

---

## 6. Calibration defaults

Every number below is derived or measured. None is a taste judgement.

### 6.1 Avellaneda–Stoikov: $\gamma = 1.5$, $\kappa = 3.0$, $\tau = 15\text{s}$

**Method.** Analytically solved against two targets (§2.2 of
`QUANT_DERIVATIONS.md`), then **verified by grid search** over
$\gamma \times \kappa$ across 3 seeds, ranked by
$\text{mean}(\text{P\&L}) - \text{sd}(\text{P\&L})$.

**Why that objective and not P&L.** Ranking a market maker by realised P&L over
a handful of paths selects for whichever configuration took the most directional
risk and got lucky. Mean-minus-dispersion prefers consistency.

**Why a minimum-fills floor.** A configuration that quotes so wide it never
fills scores a perfect $0.0 \pm 0.0$ and wins outright. Candidates under 30 fills
are disqualified — not trading is not a strategy.

### 6.2 Toxicity threshold: a **quantile**, not a level

VPIN's absolute level depends on bucket size, trade-size distribution and venue.
Measured on the reference regime its median is 0.52, so a textbook 0.35 threshold
fired on **99.3% of updates** — the gate widened permanently instead of reacting,
cutting fills from 151 to 16.

The gate now tracks the p90 of recent VPIN from a rolling window, so the
parameter means the same thing anywhere: *react to the most toxic 10% of tape*.

### 6.3 `horizon_mode = "infinite"`

The published solution has a terminal time $T$, so urgency to flatten grows
without bound as $t \to T$. A crypto desk quotes 24/7 and has no terminal time,
so $(T-t)$ is pinned to a constant.

### 6.4 Volatility baseline: asymmetric decay

Rises slowly (a spike barely moves it), falls faster (a sustained regime change
is absorbed). A hard freeze — the first attempt — latched onto a near-zero
warm-up value and never recovered, holding the breaker on 63% of the time.

### 6.5 The simulator regime is *not* BTCUSDT

Measured over 180s of live BTCUSDT (`artifacts/binance_calibration.json`):

| Quantity | Measured |
|---|---|
| Spread | **1 tick at both p50 and p90** |
| $\sigma$ | 0.553 $/√s = **17 ticks per 100ms step** |
| Depth at touch | 4.97 BTC |
| Trade rate | 2.86/s, mean size 0.0117 BTC |

The tick is *binding* there — the market wants a tighter spread than the tick
allows — and $\sigma$ per step is 17× the entire spread. Adverse selection
dominates spread capture by more than an order of magnitude, and no passive
quoter with a 1-tick spread has positive expectancy. Real BTCUSDT makers live on
fee tiers and speed, not on the quoted spread.

Running the ablation there would only rank strategies by which loses least. The
regression harness therefore uses a wider-spread, thinner-queue instrument where
classical A-S economics apply. **The live pipeline still runs against real
BTCUSDT** — the two are separate on purpose, and both are reported.

---

## 7. Observability

### 7.1 Correlation IDs

Every event carries a `correlation_id` propagated unchanged from the book update
that caused it. Grab it off a suspicious quote in the dashboard and one grep
returns every log line from ingestion through fill, across eight processes.

### 7.2 Latency histogram buckets

Default `prometheus_client` buckets start at 5ms — the *entire* p99 budget for
DeepLOB inference. With defaults every inference lands in one bucket and the
latency regression test could never detect anything. Buckets are explicit and
start at 50µs.

### 7.3 Clock skew correction

Measured **+677ms** local-vs-exchange offset on the development host. Uncorrected,
`ts_ingest - ts_event` reported ~734ms of staleness against a true one-way
latency of ~63ms — an 11× error that would have made every latency SLO
meaningless. NTP does not help: the host clock can be correct and still disagree
with the exchange, and it is the exchange's clock that stamps the data.

### 7.4 Health checks assert correctness, not liveness

`ingestion-svc` reports unhealthy if the book is not synced or no update has
arrived in 15s. `execution-sim-svc` reports unhealthy if the P&L identity is
violated. A service that is running but publishing a stale book is worse than
one that is down, because nothing alerts.

**Deliberate exception:** a tripped circuit breaker is *not* unhealthy. It is the
system working. It surfaces as a metric and in the health detail string — if it
returned 503, an orchestrator would restart the very thing protecting us.

---

## 8. Open decisions and known limitations

### Resolved

| Area | Resolution |
|---|---|
| Own-order impact | Modelled. `own_impact` parameterises queue displacement; `own_impact_from_share` derives it from quote size. **Measured cost: 32% of P&L, ~half the Sharpe** at full displacement. |
| Latency model | Log-normal draw with a seeded generator, so the tail costs no reproducibility. **Measured: no effect at a 100ms cadence** (Sharpe 668 → 669) — the tail lands inside one book update. Implemented for when cadence tightens. |
| Backtest fees | Swept. **Break-even maker fee is 0.0 bps**; at Binance VIP0 (+10bps) P&L is −18.98. Reported in RESULTS.md §3b. |
| Drift arms | In the ablation, with a grid-searched λ. |
| Simulator-only results | `backtest_svc.replay` backtests on ticks recorded from the live feed, causally merged. |
| Inference budget under load | CPU reservation for `deeplob-inference-svc` in compose; contention, not the model, was the 4.0 → 7.6ms difference. |

### Genuinely open

| Area | Current | Why it stays open |
|---|---|---|
| Single symbol | One instrument at a time | Cross-asset hedging and basis trading are a scope expansion, not a defect. The screener ranks venues; running several concurrently needs a portfolio-level risk layer that does not exist. |
| Counterfactual fills | Queue model over recorded flow | Replay removes the *generative model*, not the counterfactual: our orders were never in that book. Only live trading closes this, which the $0 constraint precludes. |
| Fee dependence | Break-even at 0.0 bps | Not fixable in code. The strategy needs a rebate or a fee tier; that is a commercial fact about passive market making, and it is now measured rather than assumed. |

---

## 9. Repository layout

```
amme/
├── services/               8 independently deployable services
│   └── <name>/             package + Dockerfile + pyproject + tests
├── shared/
│   ├── amme_shared/        transport, config, observability
│   └── amme_core/          pure quant/ML domain logic
├── train/amme_train/       offline DeepLOB training (Colab/Kaggle-ready)
├── infra/                  docker-compose, prometheus, grafana, postgres
├── tests/{unit,integration,regression}/
├── docs/                   this file, QUANT_DERIVATIONS.md, RESULTS.md
└── artifacts/              generated reports (ablation, calibration, metrics)
```

Services never import each other. They share `amme_shared` (contracts) and
`amme_core` (domain logic), and communicate only over the stream bus.
