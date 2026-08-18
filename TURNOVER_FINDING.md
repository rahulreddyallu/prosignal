# Turnover: the variable the evidence indicts — 2026-08-18

338 tests passing. Every number measured.

## Why this and not another factor

Three independent measurements pointed at the same variable:

| Evidence | Value |
|---|---|
| Nifty200 Momentum 30, **live** since 2020 launch, semi-annual rebalance | Sharpe **0.36** |
| Nifty 200 benchmark, same window | Sharpe 0.23 |
| This engine, walk-forward, **18-session** holds | DSR **0.2%** |
| Measured cost drag per trade | **0.38%** |
| Measured mean net return per trade | **+0.42%** |

Costs consumed ~90% of the gross edge. The premium looked real; the
implementation destroyed it. So the addition is **not another factor** — it is
a portfolio layer that trades less.

## What was built

`portfolio.py` — fixed-cadence rebalancing with **buffer bands**. A name enters
only inside `entry_rank` but is not sold until it falls outside a wider
`exit_rank`. This is standard index construction (NSE's own factor indices use
it) and the reason is arithmetic: without hysteresis, a name oscillating around
the rank boundary is bought and sold repeatedly, paying a full round trip each
time for no change in view.

**Nothing about scoring changed.** Only how often the book may change.

## Measured result — quarterly rebalance, equal-weight, 11 rebalances

| Config | Turnover/yr | Quarterly mean | Ann. vol | Sharpe | Cost paid |
|---|---|---|---|---|---|
| **Quarterly + buffer** (enter ≤15, exit >30) | **122%** | +6.38% | 26.4% | **0.82** | ₹33,820 (3.4%) |
| Quarterly, no buffer (enter ≤15, exit >15) | 146% | +5.36% | 25.7% | 0.65 | ₹45,207 (4.5%) |
| Engine, 18-session per-signal | — | — | — | DSR 0.2% | — |

**The buffer cut turnover 146% → 122%, cut costs by a quarter, and raised
Sharpe 0.65 → 0.82.** That is the mechanism working exactly as the literature
predicts.

## And now the part that matters more

**None of this clears the statistical bar, and I am not going to pretend it
does.**

| Config | Quarters | PSR vs 0 | **DSR (842 trials)** | Passes |
|---|---|---|---|---|
| Quarterly + buffer | 9 | 90.5% | **3.9%** | **NO** |
| Quarterly, no buffer | 9 | 88.7% | **1.8%** | **NO** |

**Nine observations.** A Sharpe of 0.82 from nine quarterly numbers is an
anecdote with error bars wider than the estimate. Even granting the
indefensible assumption of a single trial, PSR reaches only 90.5% — still under
95%.

The trial ledger now stands at **842**, including the two configurations tested
here. I counted them against myself rather than quietly excluding them.

## What this does and does not establish

**Establishes:** the turnover mechanism is real and measurable. Lower turnover →
lower cost → higher Sharpe, in the predicted direction and magnitude, on this
data.

**Does not establish:** that the strategy has an edge. Nine quarters cannot.

## The one experiment that would settle it

Not more factors. Not more parameter search. **More quarters.**

At quarterly rebalance you need roughly 40 observations for a Sharpe estimate
to have usable error bars — that is **ten years** of history. The store holds
four. Every remaining question is now blocked on data, not on modelling:

1. Ingest 10 years of bhavcopy (the ingest is incremental and already works).
2. Re-run this exact comparison — **pre-registered**, no new configurations.
3. Only then, if it clears, run the charter's horizon grid h ∈ {21, 42, 63, 126}
   as a single counted experiment.

## What I deliberately did not do

Tune `entry_rank`, `exit_rank`, or the rebalance cadence to improve the number.
Two configurations were tested, both pre-specified, both reported. Searching
band widths until the Sharpe looked good is precisely the behaviour the DSR
discipline exists to catch, and at 842 trials the bar is already high enough to
be unclearable by luck.
