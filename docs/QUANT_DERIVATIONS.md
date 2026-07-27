# Quantitative derivations

Full derivations for every formula the engine uses, from first principles.
Each section ends with the exact place the result is implemented and the test
that verifies it against a hand-computed value.

**Contents**

1. [Avellaneda–Stoikov: the HJB derivation](#1-avellanedastoikov)
2. [Units, and the inventory-normalisation trap](#2-units)
3. [Order Flow Imbalance](#3-order-flow-imbalance-ofi)
4. [VPIN](#4-vpin)
5. [Realized volatility under irregular sampling](#5-realized-volatility)
6. [Drift injection](#6-drift-injection)
7. [Queue-position fill model](#7-queue-position-fill-model)
8. [Performance statistics](#8-performance-statistics)

---

## 1. Avellaneda–Stoikov

Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order
book.* Quantitative Finance 8(3), 217–224.

### 1.1 Setup

The mid price is arithmetic Brownian motion — **not** geometric:

$$dS_t = \sigma\, dW_t$$

Arithmetic is the right choice over a market-making horizon of seconds to
minutes: over such intervals $S_t$ cannot plausibly reach zero, the
distinction from geometric BM is second-order, and arithmetic gives a
closed-form solution where geometric does not.

The maker quotes a bid at $S_t - \delta^b$ and an ask at $S_t + \delta^a$.
Orders arrive as Poisson processes whose intensity decays exponentially in
distance from the mid:

$$\lambda^{b}(\delta^b) = A e^{-\kappa \delta^b}, \qquad
  \lambda^{a}(\delta^a) = A e^{-\kappa \delta^a}$$

This form is not arbitrary. It follows from two empirical regularities: market
order sizes are power-law distributed, and the price impact of an order of size
$Q$ grows roughly like $\ln Q$. Composing them gives an exponential
fill probability in $\delta$. $\kappa$ is therefore a *market-measurable*
quantity — the decay rate of fill intensity with distance — which is why the
implementation calibrates it rather than assuming it.

State variables: cash $X_t$, inventory $q_t$, and

$$dX_t = (S_t + \delta^a)\,dN^a_t - (S_t - \delta^b)\,dN^b_t, \qquad
  q_t = N^b_t - N^a_t$$

The maker maximises expected exponential (CARA) utility of terminal wealth:

$$u(x, q, s, t) = \max_{\delta^a, \delta^b}\;
   \mathbb{E}\!\left[-\exp\big(-\gamma (X_T + q_T S_T)\big)\right]$$

CARA is what makes the problem tractable: it removes wealth from the optimal
control, so the solution depends on inventory but not on how rich the maker is.

### 1.2 The HJB equation

By dynamic programming, $u$ satisfies

$$\partial_t u + \tfrac{1}{2}\sigma^2 \partial_{ss} u
  + \max_{\delta^b}\lambda^b(\delta^b)\big[u(x - s + \delta^b,\, q+1,\, s,\, t) - u\big]
  + \max_{\delta^a}\lambda^a(\delta^a)\big[u(x + s + \delta^a,\, q-1,\, s,\, t) - u\big] = 0$$

with terminal condition $u(x,q,s,T) = -\exp(-\gamma(x + qs))$.

### 1.3 The reduction that makes it solvable

Try the ansatz

$$u(x,q,s,t) = -\exp(-\gamma x)\exp(-\gamma \theta(q,s,t))$$

Substituting, the $x$-dependence factors out completely — this is the CARA
payoff — and $\theta$ satisfies a system indexed by $q$. Writing

$$\theta(q,s,t) = q s + \theta_q(s,t)$$

and expanding to first order in the (small) inventory yields the
**indifference price**: the price at which the maker is indifferent between
holding $q$ and holding $q \pm 1$:

$$\boxed{\;r(s,q,t) = s - q\,\gamma\,\sigma^2\,(T-t)\;}$$

**Reading it.** The maker does not centre quotes on the mid. Holding $q > 0$,
they mark their own book *down* by $q\gamma\sigma^2(T-t)$, which makes their bid
less attractive and their ask more attractive — both push inventory toward zero.
The magnitude scales with risk aversion $\gamma$, with variance $\sigma^2$, and
with the time $(T-t)$ over which that variance can hurt.

### 1.4 The optimal spread

Substituting the reservation price and maximising over $\delta^a, \delta^b$
gives first-order conditions. With $\delta = \delta^a + \delta^b$:

$$\frac{\partial}{\partial \delta^b}\left[
  A e^{-\kappa\delta^b}\left(1 - e^{-\gamma(\delta^b - \text{shift})}\right)\right] = 0$$

Solving and summing the two sides:

$$\boxed{\;\delta^a + \delta^b = \gamma\sigma^2(T-t) + \frac{2}{\gamma}\ln\!\left(1 + \frac{\gamma}{\kappa}\right)\;}$$

**Two terms, two different jobs.**

- $\gamma\sigma^2(T-t)$ — *inventory-risk compensation*. Grows with volatility
  and horizon. This is the maker charging for the risk of being caught with a
  position.
- $\frac{2}{\gamma}\ln(1 + \gamma/\kappa)$ — *monopolistic markup*. The
  profit-maximising markup a liquidity supplier extracts given arrival
  elasticity $\kappa$. Note carefully: it does **not** depend on $q$ or on $t$.

That second point is worth stating explicitly because it is a common
misunderstanding: **the optimal spread is independent of inventory.** All
inventory management happens through the *reservation price*, which shifts the
quote pair without widening it. A test asserts this
(`test_long_inventory_shifts_quotes_down` checks the spread is unchanged).

Limiting behaviour, both worth being able to recite:

- $\kappa \to \infty$ (fills fall off instantly with distance): markup $\to 0$.
  Perfect competition, quote at the reservation price.
- $\gamma \to 0$ (risk-neutral): using $\ln(1+x) \approx x$, markup
  $\to 2/\kappa$, and the inventory term vanishes. A risk-neutral maker still
  charges a spread — the markup is a monopoly rent, not a risk premium.

Final quotes:

$$\delta^b_{\text{bid}} = r - \tfrac{1}{2}\delta, \qquad
  \delta^a_{\text{ask}} = r + \tfrac{1}{2}\delta$$

> **Implementation** `shared/amme_core/avellaneda_stoikov.py`
> **Verified** `tests/unit/test_avellaneda_stoikov.py` — e.g. $\gamma=0.5$,
> $\sigma=2$, $\tau=1$, $\kappa=0.5$ gives spread $= 2 + 4\ln 2 = 4.7725887$,
> checked against the closed form by hand.

### 1.5 Finite vs infinite horizon

The published solution has a terminal time $T$, so $(T-t) \to 0$ and the maker
becomes infinitely urgent to flatten. That models a desk that must liquidate by
the close. A crypto desk quoting 24/7 has no such time, so
`horizon_mode="infinite"` pins $(T-t)$ to a constant $\tau$ — the effective
inventory-holding horizon. This is the standard practitioner adaptation and is
the default here.

---

## 2. Units

Getting units wrong here fails *silently* — the engine still runs and still
quotes, just at systematically wrong widths. Pinning them down:

| Symbol | Units | Note |
|---|---|---|
| $S$, $\delta$, $r$ | quote currency | |
| $\sigma$ | quote currency $\cdot$ s$^{-1/2}$ | **absolute**, not a return vol |
| $\gamma$ | (quote currency)$^{-1}$ | absolute risk aversion |
| $\kappa$ | (quote currency)$^{-1}$ | so $\gamma/\kappa$ is dimensionless |
| $q$ | dimensionless | a share *count* |

Check the reservation price:
$q \cdot \gamma \cdot \sigma^2 \cdot \tau = 1 \cdot \$^{-1} \cdot \$^2 s^{-1} \cdot s = \$$ ✓

### 2.1 The inventory-normalisation trap

A-S assumes $q$ is a count of order 1. Our inventory is in BTC with a limit of
$q_{\max} = 0.05$. Take the ratio of the reservation tilt at the limit to the
spread's own inventory term:

$$\frac{q_{\max}\gamma\sigma^2\tau}{\gamma\sigma^2\tau} = q_{\max} = 0.05$$

So a literal implementation makes the inventory tilt **20× smaller** than the
spread term — inventory control is present in the code and absent in effect.
Nothing errors; the strategy simply does not manage inventory.

The fix is a change of units, not of model. Measure inventory in units of the
risk limit, $\tilde q = q/q_{\max}$, and rescale $\tilde\gamma = \gamma\,q_{\max}$.
Since $q$ enters only through the product $q\gamma$, the reservation price is
**exactly invariant**:

$$q\gamma = \left(\frac{q}{q_{\max}}\right)\left(\gamma\, q_{\max}\right) = \tilde q \tilde\gamma$$

What it buys is an interpretable parameter: $\gamma$ is now *risk aversion per
full-limit position*, on the same scale as the spread term.

> **Verified** `test_inventory_normalisation_is_exact_reparametrisation` asserts
> the two parametrisations produce identical reservation prices.

### 2.2 Calibrating $\gamma$, $\kappa$, $\tau$

They are solved jointly against two targets, not chosen independently:

1. **Half-spread when flat ≈ the competitive market half-spread.** A maker
   quoting materially wider than the market is not being prudent — it is not
   participating, and will never be filled.
2. **Reservation tilt at full inventory ≈ the half-spread.** So inventory
   pressure and spread compensation live on the same scale.

$$\gamma\sigma^2\tau = h, \qquad \frac{2}{\gamma}\ln\!\left(1+\frac{\gamma}{\kappa}\right) = h$$

Two equations, three unknowns: fix $\tau$ at the intended holding horizon and
solve for $\gamma, \kappa$. The result is then **verified by grid search**
(`backtest_svc/calibrate.py`), which ranks candidates by
$\text{mean}(\text{P\&L}) - \text{sd}(\text{P\&L})$ across seeds. See
`docs/RESULTS.md` for the sweep.

---

## 3. Order Flow Imbalance (OFI)

Cont, R., Kukanov, A. & Stoikov, S. (2014). *The price impact of order book
events.* Journal of Financial Econometrics 12(1), 47–88.

### 3.1 Definition

For consecutive best-quote observations $n-1, n$:

$$e_n = \mathbb{1}_{\{P^b_n \ge P^b_{n-1}\}} q^b_n
      - \mathbb{1}_{\{P^b_n \le P^b_{n-1}\}} q^b_{n-1}
      - \mathbb{1}_{\{P^a_n \le P^a_{n-1}\}} q^a_n
      + \mathbb{1}_{\{P^a_n \ge P^a_{n-1}\}} q^a_{n-1}$$

$$\text{OFI}_k = \sum_{n \in \text{window}} e_n$$

### 3.2 Why the indicator form

Reading the bid side case by case makes the construction obvious:

| Case | Indicators | Contribution | Interpretation |
|---|---|---|---|
| $P^b_n > P^b_{n-1}$ | first only | $+q^b_n$ | bid improved — buy interest arrived |
| $P^b_n < P^b_{n-1}$ | second only | $-q^b_{n-1}$ | bid pulled or consumed |
| $P^b_n = P^b_{n-1}$ | **both** | $q^b_n - q^b_{n-1}$ | net size change at the same price |

The ask side is identical with the sign flipped.

The flat case is the one implementations get wrong. Writing the definition as an
`if/elif/else` chain gives each case one branch, and the "unchanged price" case
silently loses one of its two terms. Both indicators must be allowed to fire.

> **Verified** `test_bid_price_flat_gives_size_difference` targets exactly this.

### 3.3 Why it works

Cont et al.'s empirical result is that mid-price changes are approximately
*linear* in OFI:

$$\Delta P_n \approx \beta\, \text{OFI}_n + \varepsilon_n$$

with $R^2$ far above what trade imbalance alone achieves. The intuition: OFI
counts *all* book events — limit orders, cancellations, and trades — whereas
trade imbalance sees only the last. Most information arrives as quote revision,
not as executions.

### 3.4 Standardisation

The raw level of OFI scales with typical resting size, which varies by symbol,
by time of day, and after any tick-size change. The gate therefore uses a
rolling z-score, so a single threshold ($|z| > 2$) is portable.

> **Implementation** `shared/amme_core/ofi.py` · **Verified** `tests/unit/test_ofi_vpin.py`

---

## 4. VPIN

Easley, D., López de Prado, M. & O'Hara, M. (2012). *Flow toxicity and
liquidity in a high-frequency world.* Review of Financial Studies 25(5),
1457–1493.

### 4.1 Volume clock

The central idea is **not** the formula but the sampling: cut the tape into
buckets of exactly $V$ units of volume rather than fixed time intervals.
Volume-clock sampling removes intraday volume seasonality and makes the
resulting series much closer to i.i.d., which is what makes a threshold on it
meaningful.

### 4.2 Definition

$$\text{VPIN} = \frac{\sum_{\tau=1}^{n} \left|V^B_\tau - V^S_\tau\right|}{n V}$$

Since $V^B_\tau + V^S_\tau = V$ by construction, VPIN $\in [0,1]$: 0 is perfectly
balanced flow, 1 is every bucket entirely one-sided.

### 4.3 Classification

**True aggressor (default).** Binance's `aggTrade` carries `m` = "buyer is the
market maker". If `m` is true the resting order was the buy, so the aggressor
was the **seller**. This is exact — there is no reason to estimate a quantity
the exchange hands you. Getting this inversion wrong flips every VPIN reading
while leaving the system apparently healthy.

**Bulk Volume Classification** (the paper's own estimator, for feeds without an
aggressor flag):

$$V^B_\tau = V \cdot Z\!\left(\frac{P_\tau - P_{\tau-1}}{\sigma_{\Delta P}}\right), \qquad V^S_\tau = V - V^B_\tau$$

with $Z$ the standard normal CDF. Then $|V^B - V^S| = V|2Z(x) - 1|$, so the
bound in $[0,1]$ holds automatically.

**Tick rule.** Uptick → buy-initiated, downtick → sell, repeat on a zero tick.
Used as the fallback when the aggressor flag is `UNKNOWN`.

### 4.4 Bucket-straddling trades

A trade larger than the remaining bucket capacity is split *proportionally*
across buckets. Assigning it wholesale to one bucket makes VPIN depend on
bucket size, which the estimator is specifically designed not to do.

> **Verified** `test_large_trade_splits_across_buckets`

### 4.5 The threshold is a quantile, not a level

VPIN's absolute level depends on bucket size, trade-size distribution and venue.
Measured on the reference regime its median is 0.52 — so a textbook-looking
threshold of 0.35 fires on **99.3% of updates**. The gate then widens
permanently instead of reacting to toxicity, which in testing cut fills from 151
to 16 and made the toxicity arm strictly worse than pure A-S.

The gate therefore tracks a **quantile** of recent VPIN (default p90), so the
parameter means the same thing on any venue: *react to the most toxic 10% of the
tape*.

> **Implementation** `shared/amme_core/vpin.py`, `toxicity.py`

---

## 5. Realized volatility

A-S needs $\sigma$ in absolute price units per $\sqrt{\text{second}}$. Binance
emits depth updates only when the book changes, so sampling is irregular and the
naive "standard deviation of price changes" is wrong — it ignores that the gaps
differ.

The consistent estimator for the diffusion coefficient of arithmetic BM under
irregular sampling is the realized-variance **rate**:

$$\hat\sigma^2 = \frac{\sum_i (\Delta S_i)^2}{\sum_i \Delta t_i}$$

i.e. a ratio of sums, not a mean of ratios. (By Jensen these differ, and the
mean-of-ratios form is dominated by the smallest $\Delta t$.)

### 5.1 Two estimators, deliberately

- **Rolling window** — stable, feeds the quoter's spread.
- **EWMA** — fast, feeds the circuit breaker.

A kill switch and a spread calculation want opposite things from a volatility
estimate, and forcing one series to serve both is a bad compromise.

### 5.2 The baseline, and a bug worth recording

The breaker trips on $\sigma_{\text{fast}} / \sigma_{\text{baseline}} > 4$, so
the baseline must be genuinely slow. An earlier version *froze* the baseline
whenever $\sigma > 2\sigma_{\text{base}}$, intending to stop a spike being
absorbed. During warm-up the first non-zero $\sigma$ is tiny, the baseline
latched onto it, every later reading then exceeded the freeze threshold — so it
froze permanently at a near-zero value. The measured ratio sat at a **median of
3.48** instead of ≈1, and the breaker halted quoting **63% of the time**.

The fix is asymmetric decay: rise slowly (a spike barely moves it), fall faster
(a sustained regime change is eventually absorbed). A hard freeze has no way back
from a bad initial value; asymmetric decay always converges.

> **Implementation** `shared/amme_core/volatility.py`

---

## 6. Drift injection

The DeepLOB classifier outputs $[P(\text{down}), P(\text{flat}), P(\text{up})]$.
Convert to an expected return using the label threshold $\alpha$ that *defined*
the classes:

$$\mathbb{E}[r] = \big(P(\text{up}) - P(\text{down})\big)\cdot\alpha$$

so the signal carries the same units as the training labels and $\lambda$ stays
interpretable across retrainings. Then:

$$\boxed{\;r_{\text{adj}} = r + \lambda \cdot \mathbb{E}[r] \cdot S\;}$$

**Why additive on the reservation price**, rather than multiplicative or
spread-based:

1. **Dimensionally clean.** $\mathbb{E}[r]$ is a dimensionless return, so
   multiplying by $S$ gives quote-currency units that add directly to $r$.
2. **Separates concerns.** The *spread* compensates for inventory and adverse
   selection; the *centre* expresses the directional view. Keeping them
   independent is what lets the ablation attribute a result to one or the other.
3. **Exactly nests.** At $\lambda = 0$ the term vanishes and the system *is*
   pure A-S — the same code path, not a parallel one. The regression suite
   asserts this equality.

Two safety rails, both non-negotiable for a model-driven term:

- **Confidence floor** — ignore predictions below a max-softmax threshold.
- **Hard clamp in ticks** — a model bug can never walk the quote away from the
  market. $\left|\lambda \mathbb{E}[r] S\right| \le \delta_{\max}$.

$\lambda$ is calibrated by grid search, never assumed.

---

## 7. Queue-position fill model

A naive backtest fills a resting bid whenever a trade prints at or below its
price. That assumes we are always first in the queue — the one assumption a
passive strategy can never make, and the single biggest reason market-making
backtests are optimistic.

### 7.1 Queue decay from cancellations

An order joins a level behind resting size $Q_{\text{ahead}}$. When the level
shrinks from $Q_{\text{old}}$ to $Q_{\text{new}}$ with no trade, the departed
size was cancellations. Under a uniform-position assumption, the fraction of
cancellations sitting ahead of us is $Q_{\text{ahead}}/Q_{\text{old}}$, so:

$$Q_{\text{ahead}}^{\text{new}} = Q_{\text{ahead}}
  - (Q_{\text{old}} - Q_{\text{new}})\frac{Q_{\text{ahead}}}{Q_{\text{old}}}
  = Q_{\text{ahead}} \cdot \frac{Q_{\text{new}}}{Q_{\text{old}}}$$

A one-line update that captures most of the real dynamics. Size *added* to a
level always joins behind us, so it leaves $Q_{\text{ahead}}$ unchanged.

### 7.2 Priority loss on re-quote

Amending a resting order's price sends it to the back of the new level;
re-quoting at an unchanged price keeps priority. This asymmetry is first-order
in market making — it is why real makers work hard to avoid unnecessary amends —
and it is modelled explicitly.

### 7.3 Walk-through

A print *strictly through* our price means the aggressor consumed every level
between, so our order was necessarily filled regardless of queue position.

### 7.4 Stated simplifications

- Our orders do not affect the replayed flow. At 0.001 BTC against ~5 BTC at the
  touch this is far below the noise floor; it would not be acceptable at size.
- Latency is a fixed delay, not a distribution.

> **Implementation** `shared/amme_core/matching.py` · **Verified** `tests/unit/test_matching_risk.py`

---

## 8. Performance statistics

### 8.1 P&L accounting

Signed-position average cost. On a fill of size $s$ at price $p$:

- **Increasing** the position: $\bar p \leftarrow \dfrac{|q|\bar p + s p}{|q + s|}$
- **Reducing**: realize $s_{\text{closed}}(p - \bar p)$ if long, $s_{\text{closed}}(\bar p - p)$ if short

Mark-to-market total P&L is $X_t + q_t S_t$. The identity

$$\underbrace{X_t + q_t S_t}_{\text{mark-to-market}} = \underbrace{\text{realized}}_{\text{closed trades}} + \underbrace{q_t(S_t - \bar p)}_{\text{unrealized}}$$

is computed **two independent ways and asserted equal** on every snapshot. Fees
are folded into realized P&L so the identity holds exactly. A violation means the
accounting is wrong, and every downstream number is meaningless — so it fails the
health check rather than being reported.

### 8.2 Sharpe

Three choices, each of which changes the number materially:

1. **P&L increments, not returns.** A maker's capital base is the risk limit,
   not the position, and the position crosses zero constantly — percentage
   returns are undefined there.
2. **Fixed-interval resampling** before computing moments. Sampling per-event
   would let a strategy that quotes more often look less volatile purely from
   its event count.
3. **Annualisation by wall-clock seconds** ($31{,}536{,}000$/yr), because crypto
   trades continuously. Using 252 trading days would inflate Sharpe by ≈2.4×.

$$\text{Sharpe} = \frac{\overline{\Delta \text{P\&L}}}{\text{sd}(\Delta \text{P\&L})}\sqrt{\frac{31{,}536{,}000}{\Delta t}}$$

### 8.3 Toxicity-adjusted P&L

P&L increments are tagged by the VPIN at the **start** of each interval, then
split at the run's median. Tagging by the end would attribute a loss to the
toxicity reading that the loss itself helped cause.

### 8.4 Why multi-seed, paired

A market maker's P&L over one finite path is dominated by whether residual
inventory happened to drift the right way. On a single seed we measured the
naive baseline *beating* pure A-S — not because a fixed spread is better, but
because that path rewarded the directional risk A-S exists to suppress.

Every arm therefore sees **byte-identical** market data within a seed, so path
noise cancels in the paired difference, and results are reported as mean ± sd
across seeds with a per-seed win count.

> **Implementation** `shared/amme_core/performance.py`, `services/backtest-svc/`

---

## References

1. Avellaneda & Stoikov (2008), *High-frequency trading in a limit order book*, Quantitative Finance 8(3).
2. Cont, Kukanov & Stoikov (2014), *The price impact of order book events*, J. Financial Econometrics 12(1).
3. Easley, López de Prado & O'Hara (2012), *Flow toxicity and liquidity in a high-frequency world*, RFS 25(5).
4. Zhang, Zohren & Roberts (2019), *DeepLOB: Deep convolutional neural networks for limit order books*, IEEE TSP 67(11).
5. Guéant, Lehalle & Fernandez-Tapia (2013), *Dealing with the inventory risk*, Mathematics and Financial Economics 7(4).
6. Cont, Stoikov & Talreja (2010), *A stochastic model for order book dynamics*, Operations Research 58(3).
7. López de Prado (2018), *Advances in Financial Machine Learning*, Wiley — ch. 7 on purged splits.
