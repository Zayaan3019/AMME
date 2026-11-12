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

**Final run on AVAXUSDT** (the venue the screener selected), with the retrained
model and drift injection active:

| Check | Result |
|---|---|
| Services healthy | 12/12 |
| Pipeline | 2,972 books -> 649 features -> 227 drift signals -> 235 quotes -> 235 risk decisions |
| Live quote | 6.6020 / 6.6280 around a 6.6145 mid |
| Drift adjustment | +0.000003 (model actively moving the reservation price) |
| DeepLOB p99 | **4.005 ms — inside the 5 ms budget** |
| Book resyncs / decode errors | 0 / 0 |

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

## 2. Ablation — 10 seeds, paired, all five arms

`make ablation`. Every arm sees byte-identical market data within each seed, so
comparisons are paired. 900s per run, 90s warm-up excluded, γ=4.0, κ=40
grid-searched; drift weight λ=1.0 (the λ grid ranked λ=0 first on its
consistency score — see the note at the end of §5).

| Arm | P&L (mean ± sd) | Sharpe (mean ± sd) | Max DD | Inventory σ | Fills | Profitable seeds |
|---|---:|---:|---:|---:|---:|---:|
| Naive symmetric | −0.158 ± 0.329 | −392 ± 645 | 0.394 | 0.01449 | 1577 | 2/10 |
| Pure Avellaneda–Stoikov | +0.027 ± 0.069 | 243 ± 642 | 0.073 | 0.00405 | 902 | 8/10 |
| A-S + toxicity gate | +0.041 ± 0.072 | 385 ± 677 | 0.062 | **0.00394** | 848 | 8/10 |
| A-S + DeepLOB drift | +0.128 ± 0.128 | **596 ± 560** | 0.054 | 0.00828 | 510 | 8/10 |
| **Full AMME** | **+0.129 ± 0.127** | 590 ± 545 | **0.051** | 0.00803 | 474 | 8/10 |

### Paired component deltas — every mechanism earns its place

| Comparison | Δ P&L (mean ± sd) | Δ Sharpe | Δ Inventory σ | Seeds won |
|---|---:|---:|---:|---:|
| A-S vs naive | **+0.185 ± 0.375** | **+635** | **−0.01043** | 8/10 |
| Toxicity gate (vs pure A-S) | +0.014 ± 0.022 | +142 | −0.00012 | 7/10 |
| DeepLOB drift (vs pure A-S) | **+0.101 ± 0.121** | **+353** | +0.00422 | 8/10 |
| Full AMME (vs pure A-S) | +0.102 ± 0.124 | +347 | +0.00398 | 7/10 |
| Drift on top of toxicity | +0.089 ± 0.123 | +205 | +0.00410 | 7/10 |

**The build-up is monotone in P&L** — naive < pure A-S < A-S+toxicity <
A-S+drift < AMME — and every paired delta is positive, with every mechanism
winning the majority of seeds. **It is not monotone in Sharpe:** full AMME
(590) sits just below A-S + drift alone (596). Adding the toxicity gate on top
of drift buys a lower max drawdown (0.054 → 0.051) and slightly lower inventory
σ, not a higher Sharpe; the 6-point gap is far inside the ±545 seed-to-seed
spread, so the honest reading is "no Sharpe difference" between those two arms.

Sharpe here is annualised from 900-second runs (`performance.sharpe_ratio`
scales by √(seconds per year / interval)), so the levels are not comparable to
a fund's annual Sharpe. They are valid for ranking arms against each other on
identical data, and that is the only use made of them.

### Toxicity-adjusted P&L

P&L attributed to the regime it accrued in, split at each run's median VPIN.
Increments are tagged by the VPIN at the *start* of the interval, so a loss is
never attributed to the toxicity reading it helped cause.

| Arm | P&L in high-VPIN | P&L in low-VPIN | High-VPIN share |
|---|---:|---:|---:|
| Naive symmetric | **−0.1446** | −0.0241 | 86% |
| Pure Avellaneda–Stoikov | +0.0046 | +0.0121 | 28% |
| A-S + toxicity gate | +0.0143 | +0.0177 | 45% |
| A-S + DeepLOB drift | +0.0938 | +0.0340 | 73% |
| **Full AMME** | **+0.0949** | +0.0335 | 74% |

This is the clearest single result in the report. **86% of the naive baseline's
loss is incurred during high-VPIN periods** — it is being adversely selected,
exactly as theory predicts. Pure A-S neutralises that (+0.0046 instead of
−0.1446). The toxicity gate then triples high-VPIN P&L (+0.0143), and the drift
signal turns those periods into the strategy's main earner (+0.0949).

### Correction: these numbers supersede an earlier, wrong table

An earlier version of this report showed the toxicity gate *losing* 229 Sharpe
and A-S losing to naive on mean P&L. Both were artefacts of a matching-engine
defect: a through-fill assigned `order.remaining` without drawing down the
aggressor's volume, so a 0.001-unit trade could fill 10 units — a 10,000×
over-fill that corrupted every fill count and P&L in the harness.

The bug flattered configurations that traded rarely (the old calibration
optimum filled 87 times per run; the corrected optimum fills 397) and destroyed
the toxicity gate's economics, since its whole value is in *which* fills it
avoids. With volume conserved, both anomalies disappear and the ordering the
theory predicts is recovered. `tests/unit/test_bug_regressions.py` pins it.

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

---

## 3b. Fee sensitivity — the deployability question

`backtest_svc.calibrate` sweeping `maker_fee_bps`, 3 seeds, 400s each. Fills are
identical across the sweep (397 every row), because a fee change does not alter
*whether* you trade — only what each trade is worth. That makes this a clean
read on economics alone.

| Maker fee | P&L (mean ± sd) | Sharpe | Verdict |
|---:|---:|---:|---|
| −2.0 bps (rebate) | +3.822 ± 0.603 | 4464 | strongly profitable |
| −1.0 bps (rebate) | +1.922 ± 0.305 | 4401 | profitable |
| −0.5 bps (rebate) | +0.972 ± 0.156 | 4267 | profitable |
| **0.0 bps** | **+0.022 ± 0.014** | **668** | **break-even** |
| +0.5 bps | −0.928 ± 0.143 | −4707 | loses |
| +1.0 bps | −1.878 ± 0.292 | −4628 | loses |
| +10.0 bps (Binance VIP0) | **−18.980 ± 2.975** | −4534 | catastrophic |

**Break-even maker fee: 0.0 bps.** The strategy has positive expectancy only at
zero or negative fees.

That is the single most important number for anyone thinking about deploying
this, and it is not a defect — it is the economics of passive market making.
Spread capture at these widths is measured in fractions of a basis point, so a
fee of even half a bp per side dominates it. This is precisely why maker
rebates exist, why VIP tiers are worth so much to real desks, and why the
earlier note that "Binance VIP0 spot is +10bps, which no passive strategy at
this spread survives" is now a measured result rather than an aside.

The backtest default remains 0.0 bps (rebate-neutral) and is stated wherever
results are quoted, so no number in this report is silently assuming a rebate.

---

## 3c. Execution realism — how much of the edge is an artefact?

Sweeping the two assumptions most likely to flatter a paper backtest.

| latency_sigma | own_impact | P&L | Sharpe | Fills |
|---:|---:|---:|---:|---:|
| 0.0 | 0.0 | +0.022 ± 0.014 | 668 | 397 |
| 1.0 | 0.0 | +0.022 ± 0.014 | 669 | 397 |
| 0.0 | 0.5 | +0.015 ± 0.008 | 326 | 403 |
| 0.0 | **1.0** | **+0.015 ± 0.008** | **341** | 409 |

**Own-order impact is the binding assumption.** Modelling our resting size as
fully *displacing* real liquidity rather than adding to it costs **32% of P&L
and roughly half the Sharpe**. The edge survives — it does not evaporate — but a
third of it was an artefact of assuming we were free extra depth. The default
stays at 0.0 (correct while quote size is negligible against the touch) and
`own_impact_from_share` derives it from actual size, so the assumption now
degrades honestly as quote size grows.

**Latency tails do not bind at this cadence, and that is worth stating.**
A log-normal tail on a 5ms median changes Sharpe by one point (668 → 669),
because even a heavy tail on 5ms lands well inside a 100ms book update. The
model is implemented and seeded, so it costs nothing and will matter if the
feed cadence or the strategy's reaction time ever tightens — but claiming it
"fixed" anything here would be false. At 100ms, latency is not the constraint.

---

---

## 3d. Replay on real recorded data

`backtest_svc.replay` runs the harness over ticks `ingestion-svc` persisted from
the live feed. This is the only section of this report whose market data is not
synthetic.

**The recording had to be segmented first.** The raw capture spanned 8,820
seconds and hid an **81-minute gap** where containers were being rebuilt.
Backtesting across that is not merely noisy, it is wrong: the strategy carries
inventory over a discontinuity and every rolling statistic treats two unrelated
market states as consecutive. `split_contiguous` cut the capture into 6
segments; the largest is **16,625 books over 1,663s at exactly 10.00 Hz, max gap
0.20s** — a clean recording.

BTCUSDT, largest contiguous segment, `make replay` using the live `.env`
parameters:

| Arm | P&L | Sharpe | Fills | Inventory σ | Max DD |
|---|---:|---:|---:|---:|---:|
| Naive symmetric | −24.815 | −141.6 | 6986 | 0.04424 | 41.725 |
| **Pure Avellaneda–Stoikov** | **−3.634** | **+4.1** | 1537 | 0.05093 | 58.524 |

**A-S loses 6.8× less than naive and is the only arm with a positive Sharpe** —
on real market data, with no simulator in the loop. Both still lose, which is
the expected result and validates two earlier claims independently.

First, the venue screener scored BTCUSDT **0.00 makeability** — the worst of 14
symbols — and predicted no passive quoter has positive expectancy there. Real
data agrees. This is the screener being right, measured on data it never saw.

Second, **calibration does not transfer between venues**, which was asserted
from the live stack pinning its half-spread to the clamp. Here it is quantified:
γ=4/κ=40 were grid-searched on a simulator regime with a ~29-tick spread, and
applied to a 1-tick market they quote far too wide — 960 fills against naive's
3024, each one losing more.

**A-S trades far more selectively**: 1,537 fills against naive's 6,986. On a
venue where every fill is adversely selected, quoting less is the correct
response, and it is the difference between losing 3.6 and losing 24.8.

With simulator-calibrated parameters (γ=4, κ=40) applied to this 1-tick market
instead, A-S quotes far too wide — 960 fills, P&L −11.4 — but holds inventory σ
at 0.00047 against naive's 0.02570, **55× tighter**. The inventory mechanism
does exactly what the theory says under either parameterisation; what changes
is whether the spread is wide enough to earn against.

What replay does not fix: our orders were never in that book, so fills remain
counterfactual under the same queue model. Replay upgrades the *market data*
from synthetic to real. It does not turn a paper backtest into live P&L.

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

1. **ONNX Runtime over TorchScript** — **2.2×** on p99 at the same lookback
   (T=100: 15.51 → 7.06 ms; 2.1× on p50, 11.44 → 5.42 ms). Ahead-of-time
   constant folding and operator fusion.
2. **Lookback 100 → 50** — a further 43% (ONNX p99 7.06 → 4.04 ms).

**Shipped configuration vs. the TorchScript starting point** (ONNX, T=50 vs.
TorchScript, T=100), three ways of stating the same measurements:

| Comparison | TorchScript T=100 | ONNX T=50 | Speed-up |
|---|---:|---:|---:|
| Tail vs. median: shipped **p99** against TorchScript's **p50** | 11.44 ms | 4.04 ms | **2.8×** |
| p99 vs. p99 | 15.51 ms | 4.04 ms | 3.8× |
| p50 vs. p50 | 11.44 ms | 2.83 ms | 4.0× |

The headline figure is the conservative first row: the shipped model's 99th
percentile is 2.8× below TorchScript's *median* — its worst case beats
TorchScript's typical case. Of that gain, the backend alone accounts for 2.2×
(p99 at the same lookback, above) and the shorter lookback for the rest.

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

**Training data is simulator output**, not recorded Binance ticks:
`build_from_simulator` over 7,200 simulated seconds (seed 42), lookback 50,
10 levels, 20-step horizon, 70/15/15 chronological split. The drift arms in §2
run on the same simulator family, so the drift gains there are measured
in-distribution; §3d is the only real-data evaluation, and it does not include
the drift arm.

**The artifact in `models/`** was exported in the same run that wrote
`artifacts/deeplob_metrics.json` — those are the numbers below. 146,019
parameters, batch 128, splits 50,345 / 10,773 / 10,793. Training ran 8 epochs
and early-stopped; the exported weights are the **epoch-2 checkpoint** (best
validation macro F1, 0.552). Test set evaluated exactly once, after model
selection:

| Metric | Value |
|---|---:|
| Metric | DeepLOB | Majority class ("always up") | Margin |
|---|---:|---:|---:|
| Micro F1 (= accuracy for single-label 3-class) | **0.5214** | 0.4429 | **+7.9 pts** |
| Macro F1 | **0.5045** | 0.2046 | +30.0 pts |
| Weighted F1 | 0.5249 | 0.2719 | +25.3 pts |

The model beats the majority baseline on every F1 averaging. Micro F1 is the
conservative headline (+7.9 points); macro F1, which a constant predictor
cannot game, shows the larger gap.

Test class balance is down 23% / flat 33% / up 44%, against 25 / 35 / 40 in
train.

Confusion matrix (rows = true, cols = predicted; classes down/flat/up):

```
        down    flat      up
down   1,072     814     582
flat     840   1,793     912
up       690   1,328   2,762
```

The diagonal dominates every row, so the model discriminates all three classes
rather than collapsing onto the majority — which is what macro F1 (0.5045,
against 0.2046 for a constant "up" predictor) confirms.

**It is nonetheless a weak model, and the weakness is diagnosable.** It
overfits after epoch 2: training loss falls 0.69 → 0.14 over 8 epochs while
validation macro F1 peaks at 0.552 and drifts down to ~0.50. The exported
network is also overconfident on out-of-distribution input: on standard-normal
*random* tensors it returns roughly `[0.001, 0.01, 0.99]` (98–99% "up") where a
calibrated model would be near-uniform.

That overconfidence is exactly why the quoting engine's drift path has a
confidence floor **and** a hard tick clamp: this model would otherwise assert
strong directional views on noise. The constraint is data, not capacity. The
fix is training on recorded Binance history (the `ingestion-svc` TimescaleDB
capture used by §3d), not a larger network.

*An earlier draft of this section described a 3,600 s run (test accuracy
0.4615) as the shipped artifact. That run was superseded before export; the
table above matches the files in `models/`.*

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

**λ choice.** The drift arms in §2 use λ=1. The λ grid
(`artifacts/calibration_lambda.md`, ranked by mean − sd of P&L) put λ=0 first
(+0.012 ± 0.019), because the drift arms carry more P&L *and* more dispersion;
λ=1 has the higher mean (+0.041) and Sharpe (365 vs 237). Shipping λ=1 is a
choice for mean P&L over that consistency score, and it is stated here rather
than presented as the grid's pick.

---

## 6. Test suite

**314 tests.** 289 run offline and pass; the other 25 are live-stack
integration tests that skip unless `AMME_INTEGRATION=1` and `make up` is
running. `ruff` clean; `mypy --strict` clean across shared and all 8 services.

| Category | What it covers |
|---|---|
| Hand-computed quant | A-S, OFI, VPIN against values worked out by hand, not captured from the code |
| Resync chaos | Sequence gaps, stale snapshots, disconnects, duplicates, crossed books, boundary-straddling diffs |
| Queue position | Cancellation decay, priority loss on re-quote, walk-through fills, latency gating |
| Property-based | Spread positivity, no crossed quotes, inventory monotonicity, `q_max` never breached |
| Determinism | Full-trajectory SHA-256; AST-walking guards against global RNG and wall-clock reads |
| Latency budgets | Per-stage p99 assertions |
| Config wiring | The three container-only bugs below |
| Bug regressions | Volume conservation, NaN fail-closed, corrupt-fill rejection, forward-compatible decode |

### Two capacity/robustness defects found on the second live run

6. **Redis exhausted its memory and killed ingestion.** `stream_maxlen` was
   100,000 across 8 streams; a 10-level `OrderBookUpdate` serialises to 907
   bytes, so the worst case was ~730 MB against a 512 MB Redis with
   `noeviction`. `XADD` raised `OutOfMemoryError`, the exception escaped
   `run_loop`, and the ingestion process died — taking the WebSocket and the
   synchronised book with it, which costs a full snapshot resync to recover.

   Three fixes, because one would not have been enough: `stream_maxlen` sized
   from the actual byte cost (20,000 x 8 ~= 145 MB), Redis raised to 1 GB so
   MAXLEN trimming rather than the ceiling bounds growth, and the publish path
   made non-fatal — a dropped tick is strictly cheaper than a lost connection,
   and the drop is visible in `amme_events_failed_total` while the freshness
   check still turns a *sustained* failure unhealthy within 15s.

7. **A vanished consumer group wedged every consumer forever.** After a
   `FLUSHALL`, Redis answers `XREADGROUP` with `NOGROUP`; unhandled, the
   consumer blocks on a group that no longer exists. All eight services stayed
   green while the pipeline was silent — the same signature as the type-tag
   bug, from a different cause. `consume()` now detects `NOGROUP`,
   invalidates its group cache and recreates the group. Groups also disappear
   on failover and key expiry, so this is not only an operator-error path.

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
make test                # 314 tests (25 need the live stack)
make regression-test     # lint + mypy + tests + ablation
make calibrate           # -> artifacts/calibration.md
make ablation            # -> artifacts/ablation_multiseed.md
make train               # -> models/deeplob.{ts,onnx}
make up && make health   # live stack
```

Ablation and calibration are deterministic: the seed set is pinned, so the same
commit reproduces the same table.
