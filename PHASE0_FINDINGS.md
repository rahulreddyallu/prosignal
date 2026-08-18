# Phase 0 empirical findings — 2026-08-18

Run against data already in the store. No new ingestion, no cost, ~10 minutes.
Every number is reproducible from `data/curated/indices`.

---

## 1. The headline: momentum DOES have a live edge in India — at index turnover

NSE publishes investable factor indices. Nifty200 Momentum 30 launched
**25 August 2020**, so everything in our store (2022-08 to 2026-08) is
**live, out-of-sample, real money** — not a backfilled construction.

Risk-adjusted, using a 6.5% Indian risk-free rate:

| Index | CAGR | vol | **Sharpe** | max DD |
|---|---|---|---|---|
| Nifty200 Alpha 30 | 21.1% | 22.4% | **0.65** | −29.9% |
| **Nifty200 Momentum 30** | **13.8%** | **20.2%** | **0.36** | −31.8% |
| **Nifty 200 (benchmark)** | 11.0% | 19.3% | **0.23** | −18.1% |
| NIFTY200 Quality 30 | 9.0% | 19.5% | **0.13** | −23.0% |
| Nifty 50 | 8.4% | 16.5% | 0.12 | −15.8% |

**Momentum beat the benchmark on a risk-adjusted basis: 0.36 vs 0.23**, over
four live years. That is real out-of-sample evidence, and it is *better*
evidence than any backtest in this repository because nobody constructed it
knowing what came next.

**Quality UNDERPERFORMED the benchmark** (0.13 vs 0.23).

### This corrects RESEARCH.md

`RESEARCH.md` concluded momentum had the weakest India evidence and value/
quality the strongest, based on a literature survey. **The live index data says
the opposite for quality**, and supports momentum. Direct measurement beats a
literature survey about a different market's factors. RESEARCH.md §1.3 should
be read with this correction attached.

---

## 2. A data-integrity catch that would have produced a fabricated finding

My first pass reported **Nifty200 Value 30 at +17.9%/yr excess with 75.9%
volatility** — which would have made value look like the standout factor.

It is corrupt. One day, **2024-06-13, shows +107.4%** — an index rebase or bad
print, not a market move. Excluding it drops volatility from 75.9% to 18.8%.

**Value 30 is excluded from the table above.** Had I not checked an implausible
number, this document would have recommended a factor tilt on the strength of a
single bad tick. This is the second time in this project that checking an
anomaly rather than reporting it changed the conclusion.

---

## 3. Why the engine fails despite momentum working

The index earns its 0.36 Sharpe at **semi-annual rebalance**, ~30 names, and
index-level turnover costs near zero.

This engine trades an **18-session median hold** with a measured **0.38%
per-trade cost drag** against a mean net return of **+0.42%**. Costs consume
roughly 90% of the gross edge.

> **The momentum premium in India appears real and the implementation destroys
> it.** That single sentence reconciles the live index evidence with both failed
> walk-forwards (DSR 0.7% and 0.2%).

This also matches the Phase 0 charter's own reasoning: it proposes h = 63
trading days precisely because cost and 20% short-term capital gains tax
dominate at shorter horizons.

---

## 4. What this implies — and what it does NOT license

**Implies:**
1. Lengthen the horizon. The charter's pre-registered grid h ∈ {21, 42, 63, 126}
   is the right test; the engine currently sits at the bottom of it.
2. Reconsider the quality factor. It underperformed the benchmark live.
3. Alpha 30 (0.65 Sharpe) deserves investigation as a construction.

**Does NOT license:** re-running the backtest at h=63 and reporting the best
result. That is one more trial against a ledger already at 796. If the horizon
is changed it must be as a **pre-registered** test across the full grid with all
four outcomes reported.

---

## 5. Tasks not completed, and why

| Task | Status |
|---|---|
| Nifty200 Momentum 30 pre/post-launch split | **PARTIAL.** Store holds 2022-08 onward — all post-launch. Live stats computed; the pre-launch backfill comparison needs history from niftyindices, whose CSV endpoint returned an HTML page rather than data. |
| IIMA monthly factor file — Sharpe + drawdown path | **NOT DONE.** Requires fetching `faculty.iima.ac.in/iffm`. Next session. |
| IIMA size breakpoints | **NOT DONE.** Same source. |
| Regress Winners/Big on the four factors | **NOT DONE.** Blocked on the IIMA file above. |

Three of four remain open. What was completed is the one that used data already
held, and it produced the most decision-relevant result so far.
