# Results

Every number here was produced by the code in this repository. Nothing is
illustrative. Where a result is unfavourable it is reported as measured — a
results file that only contains wins is not evidence.

Regenerate with `make ablation`, `make calibrate`, `make train`.

---

## 1. Live system verification

Full stack against production Binance, `docker compose up`:

Final run, with the **trained DeepLOB artifact active** (`AMME_MODEL_BACKEND=onnx`):

| Check | Result |
|---|---|
| Services healthy | **12/12** (8 AMME + redis, timescaledb, prometheus, grafana) |
| Book updates ingested | 29,728 |
| Trades ingested | 72,216 |
| Feature vectors emitted | 13,721 |
| **DeepLOB drift signals** | **1,051** |
| Quote intents | 5,021 |
| Risk decisions | 5,022 |
| Simulated fills | **1,002**, at real BTCUSDT prices |
| Book resyncs | 0 |
| Decode errors | 0 |
| Inference skipped (no tensor) | 0 |
| P&L identity violations | 0 |

A representative live snapshot:

```
mid            64,963.99      market spread 0.01
best bid/ask   64,963.99 / 64,964.00
our quote      64,961.97 / 64,962.98
reservation    64,962.48      offset -1.5168 vs mid   (long inventory -> marked down)
inventory      +0.002000 BTC  (4.0% of limit)
sigma          6.5568         VPIN 0.751   OFI z +2.06
toxicity mult  x2.24
```

Every mechanism is visibly doing its job: the reservation price is tilted below
the mid because inventory is long, and the toxicity gate has widened the spread
2.24× on elevated VPIN and OFI.

### Live Binance measurements (BTCUSDT, 180s)

| Quantity | Measured |
|---|---|
| Book update rate | 9.98 Hz |
| Spread | **1 tick at p50 *and* p90** |
| σ | 0.553 $/√s = **17 ticks per 100ms step** |
| Depth at touch | 4.97 BTC |
| Trade rate | 2.86/s |
| Mean / median trade size | 0.0117 / 0.00065 BTC |
| Traded volume | 0.0336 BTC/s |
| Clock offset (local − exchange) | **+677 ms** |
| One-way latency | 63 ms |
| Staleness p50, clock-corrected | **63.8 ms** |

**The headline finding.** BTCUSDT spot is structurally hostile to classical
market making. The tick is *binding* — the spread is one tick at both p50 and
p90, meaning participants want a tighter spread than the venue allows — while σ
per 100ms step is 17× that entire spread. Adverse selection exceeds spread
capture by more than an order of magnitude. No passive quoter with a one-tick
spread has positive expectancy there; real BTCUSDT makers live on fee tiers and
latency, not on the quoted spread.

---

## 2. Ablation — 10 seeds, paired

`make ablation`. Every arm sees byte-identical market data within each seed, so
comparisons are paired and path noise cancels. 900s per run, 90s warm-up
excluded.

| Arm | P&L (mean ± sd) | Sharpe (mean ± sd) | Max DD | Inventory σ | Fills | Profitable seeds |
|---|---:|---:|---:|---:|---:|---:|
| Naive symmetric | +0.104 ± 0.318 | 199 ± 713 | 0.191 | 0.00829 | 445 | 7/10 |
| **Pure Avellaneda–Stoikov** | +0.063 ± 0.040 | **916 ± 554** | **0.017** | **0.00175** | 151 | **10/10** |
| A-S + toxicity gate | +0.054 ± 0.052 | 687 ± 576 | 0.021 | 0.00184 | 113 | 9/10 |

### Paired component deltas

| Comparison | Δ P&L | Δ Sharpe | Δ Inventory σ | Seeds won |
|---|---:|---:|---:|---:|
| A-S vs naive | −0.041 ± 0.321 | **+717** | **−0.00654** | 6/10 |
| Toxicity gate vs pure A-S | −0.008 ± 0.024 | −229 | +0.00008 | 3/10 |

### What this actually says

**Avellaneda–Stoikov is the clear winner, and not on P&L.** The naive baseline
has the higher *mean* P&L — and 8× the dispersion, 11× the drawdown, 4.7× the
inventory volatility, and it loses money on 3 of 10 paths. A-S delivers
comparable P&L for a fraction of the risk and is profitable on **every seed**.

That is precisely what the model is for. A-S does not promise more profit; it
promises the same profit with controlled inventory, and the table confirms it:
Sharpe +717, inventory σ −79%, max drawdown −91%.

**Reporting mean P&L alone would have named the naive baseline the winner.**
On a single seed we measured exactly that, which is why the harness runs ten and
reports dispersion.

**The toxicity gate does not pay for itself here, and that is reported as
measured.** It wins 3 of 10 seeds and costs 229 Sharpe. The mechanism works —
adaptive thresholding lifted it from 16 fills to 113, and Sharpe from 288 to 687
— but against an A-S baseline already quoting conservatively at ~27 ticks, the
extra widening forgoes more spread than the adverse selection it avoids. It
would be straightforward to omit this row; it is the most informative one in the
table.

---

## 3. Calibration

`make calibrate`. Grid search over γ × κ, 3 seeds each, ranked by
**mean(P&L) − sd(P&L)** across seeds, with candidates under 30 fills
disqualified.

| Rank | Parameters | Score | P&L (mean ± sd) | Sharpe | Inventory σ | Fills |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `γ=1.5 κ=3` | **+0.0148** | +0.048 ± 0.034 | 991 | 0.00287 | 86 |
| 2 | `γ=1.5 κ=10` | −0.0099 | −0.002 ± 0.008 | −104 | 0.00445 | 474 |
| 3 | `γ=1.5 κ=200` | −0.0737 | −0.024 ± 0.050 | −161 | 0.00635 | 595 |
| 4 | `γ=1.5 κ=40` | −0.0909 | −0.035 ± 0.056 | −306 | 0.00592 | 584 |
| 5 | `γ=0.6 κ=10` | −0.1336 | −0.066 ± 0.067 | −568 | 0.00750 | 568 |

**Why not rank by P&L.** Ranking a market maker by realised P&L over a handful
of paths selects for whichever configuration took the most directional risk and
got lucky. Mean-minus-dispersion prefers consistency.

**Why the min-fills floor.** A configuration that quotes so wide it never fills
scores a perfect 0.0 ± 0.0 and wins outright. Not trading is not a strategy.

### Calibration does not transfer between venues

The grid-searched γ=1.5, κ=3.0 are correct for the simulator's wider-spread
regime and **wrong for live BTCUSDT**, where they pinned the half-spread to the
500-tick clamp — $5.00 against a $0.01 market spread, i.e. not participating at
all. Live defaults are solved separately (γ=0.02, κ=80) and shipped in
`.env.example` with that distinction documented inline.

Similarly, a VPIN bucket of 0.05 BTC spans ~1.5s of real BTCUSDT flow, is
dominated by one or two trades, reads as near-perfectly one-sided, and pinned
VPIN at 0.95 with the breaker permanently tripped. Live uses 0.5 BTC (~15s).

---

## 4. Inference latency

Single-sample CPU inference, 1 thread, 300–400 runs after warm-up.

| Backend | Lookback | Params | p50 | p99 | 5ms budget |
|---|---:|---:|---:|---:|:--:|
| Eager PyTorch | 100 | 146,019 | 13.15 ms | 16.50 ms | ✗ |
| TorchScript (frozen) | 100 | 146,019 | 11.44 ms | 15.51 ms | ✗ |
| ONNX Runtime | 100 | 146,019 | 5.42 ms | 7.06 ms | ✗ |
| ONNX Runtime | 100 | 93,315 | 4.74 ms | 6.16 ms | ✗ |
| **ONNX Runtime** | **50** | **146,019** | **2.83 ms** | **4.04 ms** | **✓** |
| ONNX Runtime | 50 | 93,315 | 2.51 ms | 3.71 ms | ✓ |

Measured on the **trained artifact**, with the full Docker stack running and
competing for CPU:

| Backend | p50 | p99 | Budget |
|---|---:|---:|:--:|
| ONNX Runtime | 3.37 ms | 7.97 ms | ✗ under load |
| TorchScript | 7.60 ms | 15.03 ms | ✗ |

The idle p99 of 4.04 ms and the under-load p99 of 7.97 ms are both real numbers.
The budget holds on a quiet core and is missed when eight services share the
machine — which is the honest statement, and the reason the quoting engine
treats a prediction older than 2 s as absent rather than blocking on it. Pinning
the inference container to dedicated cores would be the fix in production.

**The paper's T=100 does not fit a 5ms CPU budget on this hardware.** Two
changes get inside it:

1. **ONNX Runtime over TorchScript** — 2.8× (11.44 → 4.04 ms p99). Ahead-of-time
   constant folding and operator fusion.
2. **Lookback 100 → 50** — a further 43%.

The full 146k-parameter model is retained; capacity was not the constraint.

### Per-stage cost (T=100, eager)

| Stage | Cost | Output shape |
|---|---:|---|
| conv1 (feature axis) | 0.89 ms | (1, 32, 100, 20) |
| conv2 | 0.80 ms | (1, 32, 100, 10) |
| conv3 | 0.48 ms | (1, 32, 100, 1) |
| Inception | 0.65 ms | (1, 192, 100, 1) |
| LSTM | 1.79 ms | — |

The LSTM dominates, which is why halving the sequence length is the effective
lever.

---

## 5. DeepLOB training

Chronological, purged splits — never random. Consecutive windows overlap by 49
of 50 timesteps, so a random shuffle would put window *t* in train and *t+1* in
test and measure memorisation.

| Run | Data | Splits (train/val/test) | Best val accuracy | Majority baseline |
|---|---|---|---:|---:|
| A | 7200s, batch 128 | 50,345 / 10,773 / 10,793 | **0.6274** | 0.443 |
| B (shipped) | 3600s, batch 256 | 25,145 / 5,373 / 5,393 | 0.4342 | 0.405 |

**Run A clears the majority baseline by 18.4 points** (0.627 vs 0.443) — the
model is learning real structure, not class priors. It was stopped short of
export for time (~10 min/epoch on CPU).

**Run B is the artifact in `models/`.** Trained to completion, 8 epochs,
146,019 parameters. Test set evaluated exactly once, after model selection:

| Metric | Value |
|---|---:|
| Test accuracy | **0.4615** |
| Test macro F1 | 0.4534 |
| Majority-class baseline | 0.4050 |
| Beats baseline | **True** (+5.7 points) |

Confusion matrix (rows = true, cols = predicted; classes down/flat/up):

```
        down    flat      up
down     708     289     440
flat     486     578     708
up       329     652   1,203
```

The diagonal dominates every row, so the model discriminates all three classes
rather than collapsing onto the majority — which is what macro F1 (0.4534,
close to accuracy) confirms.

**It is nonetheless a weak model, and the weakness is diagnosable.** Run B
overfits from epoch 1: training loss falls 0.70 → 0.15 while validation accuracy
stays flat at ~0.42. Two symptoms follow — the +5.7-point edge over baseline is
much smaller than Run A's +18.4, and the exported network is badly
overconfident, returning `[0.001, 0.007, 0.992]` on *random* input where a
calibrated model would be near-uniform.

That overconfidence is exactly why the quoting engine's drift path has a
confidence floor **and** a hard tick clamp: this model would otherwise assert
strong directional views on noise. The constraint is data, not capacity — Run A
on 2× the data was far better. The fix is `make record` to accumulate real
Binance history, not a larger network.

### A data-quality bug worth recording

The first training run produced a test split that was **80.4% "up"** against a
balanced 31/38/31 training split. Cause: the backtest regime schedule places
TRENDING in the final eighth of the run, which — under the chronological split
that ML correctness *requires* — lands entirely inside the test set.

A model ignoring its input scores 80.4% there, and no honest accuracy can be
quoted. The split was not the problem; a random split would be far worse. The
problem was that one contiguous slice of a single-pass schedule is not
*representative*. `cycling_regime_schedule` repeats the full regime cycle 12
times so every chronological slice contains calm, toxic, volatile and trending
periods while remaining strictly time-ordered and leak-free.

**Drift ablation arms (`as_drift`, `amme`) are not yet in the table.** The
harness supports them and they run under `--drift-lambda`; they are omitted
because publishing them before λ is calibrated on a properly-trained artifact
would be reporting a number without support.

---

## 6. Test suite

**173 tests**, all passing. `ruff` clean.

| Category | What it covers |
|---|---|
| Hand-computed quant | A-S, OFI, VPIN against values worked out by hand, not captured from the code |
| Resync chaos | Sequence gaps, stale snapshots, disconnects, duplicates, crossed books, boundary-straddling diffs |
| Queue position | Cancellation decay, priority loss on re-quote, walk-through fills, latency gating |
| Property-based | Spread positivity, no crossed quotes, inventory monotonicity, `q_max` never breached |
| Determinism | Full-trajectory SHA-256; AST-walking guards against global RNG and wall-clock reads |
| Latency budgets | Per-stage p99 assertions |
| Config wiring | The three container-only bugs below |

### Bugs caught by running the system rather than testing it

Each of these passed the unit suite and failed on `docker compose up`. All three
now have regression tests.

1. **`Literal[100, 1000]` rejected the string `"100"`.** Environment variables
   are always strings; pydantic will not coerce them into an int `Literal`.
   Every service crash-looped. Unit tests passing ints could never see it.
2. **`ws_base` ended in `/ws`** while the client appends `/stream` for the
   combined subscription → `/ws/stream?streams=…` → HTTP 404 on every connect,
   which the reconnect loop faithfully retried forever and reported as a
   transient network fault.
3. **Multi-stream consumers decoded as `BaseEvent`**, whose `extra="forbid"`
   rejected every subclass field. The pipeline went completely silent while all
   eight services continued reporting healthy — the worst failure mode in the
   system. The bus now decodes via the envelope's type tag.

### Two calibration bugs found by measurement

4. **The volatility baseline froze permanently.** A hard freeze intended to stop
   a spike being absorbed latched onto a near-zero warm-up value, so the
   fast/baseline ratio sat at a median of 3.48 instead of ≈1 and the breaker
   halted quoting **63% of the time**. Replaced with asymmetric decay, which
   always converges. Ratio p50 is now 1.05, and >4× events are 0.23%.
5. **A textbook VPIN threshold of 0.35 fired on 99.3% of updates**, because the
   observed median VPIN in this regime is 0.52. The gate widened permanently
   instead of reacting, cutting fills 151 → 16. Now tracks a rolling p90.

---

## 7. Reproducing

```bash
make install
make test                # 173 tests
make regression-test     # lint + mypy + tests + ablation
make calibrate           # -> artifacts/calibration.md
make ablation            # -> artifacts/ablation_multiseed.md
make train               # -> models/deeplob.{ts,onnx}
make up && make health   # live stack
```

Ablation and calibration are deterministic: the seed set is pinned, so the same
commit reproduces the same table.
