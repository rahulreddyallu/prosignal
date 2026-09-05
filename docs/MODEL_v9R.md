> **RETIRED 2026-09-05. `features/v9r.py` is deleted and `v9r_core` is no longer
> selectable — `ranking.source` itself no longer exists.**
>
> This document is kept as the record of a model that was built, sealed,
> measured and **failed its own pre-registered gate**: +9.50% net active on the
> sealed 2012-2017 window at Newey-West t +1.87 against a bar of 2.0. Positive
> and underpowered is a failed ship gate, not a passed one, and it never became
> the default.
>
> Two further reasons it was deleted rather than kept for shadow running. It
> weighted `mom_2_0` at **13.80%** — a factor whose standalone rank IC is
> **+0.0001 at t 0.02**, measured over 380 dates in the 2026-09-05 audit. And a
> scorer nobody runs on a schedule is not a control, it is a second answer to
> the question `features/engine.py` exists to answer; the repository carried six
> of those and could not say in one sentence what it recommended.
>
> The sealed 2012-2017 window is spent and cannot be reused. Nothing here
> transfers to the shipped engine.

# v9R CORE — the model card

`src/prosignal/features/v9r.py`. Selectable as
`stage4_core_score.ranking.source: v9r_core`. **Not the default.**

This file exists because the research tree that produced the model was deleted in
the 2026-09-03 cleanup. The numbers below are the whole of what was measured; the
scripts that produced them are gone, so treat this as the record rather than as a
pointer to one.

## What it is

Nine factors from `features/v3_factors.py`, blended at weights that equalise each
factor's contribution to composite variance, ranked **unneutralised**, with a
missing factor scoring the cross-sectional mean rather than redistributing its
weight.

| factor | weight |
|---|---|
| `ret_kurt_126` | 0.1816 |
| `mom_2_0` | 0.1380 |
| `mom_accel` | 0.1126 |
| `mom_consist_126` | 0.1108 |
| `voladj_mom_12_1` | 0.0964 |
| `intraday_mom_126` | 0.0941 |
| `prox_52w_now` | 0.0934 |
| `prox_52w` | 0.0885 |
| `voladj_mom_6_1` | 0.0848 |

Book: long only, top **20** by composite, equal weighted, rebalanced every **42**
sessions, filled at the next session's close. Coverage floor 0.70.

## How it was selected

A long-side gate on the **training window 2018-01-01 to 2026-08-28 only**: form the
top-quintile equal-weighted book on each factor alone, delete the bottom quintile,
require positive annualised excess over the equal-weighted eligible universe at
Newey-West t > 2.0 with 4 of 5 time folds positive. Twelve of twenty-two factors
survived; nine of those are computable from OHLCV alone and form CORE.

Weights are equal-risk-contribution on the survivors' rank correlation matrix,
capped at each factor's coverage. Equal *nominal* weight fails a
risk-contribution gate at 0.078 because seven of the nine are momentum and
correlate 0.50-0.75; ERC brings that to 0.008.

## The sealed window

**2012-01-01 to 2017-12-31**, a window no generation v3 through v9 had seen. NSE
bhavcopy for it was pulled from a GitHub mirror pinned at commit `6e13b7b0` and
reconciled against the existing store over 30 random overlapping 2017 sessions:
worst deviation **0.0000 bp** across 43,359 compared closes.

Opened **once**, against criteria hashed
`d2dfba4f9a1e4ee1ed24d5cb6429307fdcf53bddb531979b25c2dbb78100f877` before the
window was computed.

| | sealed 2012-2017 | train 2018-2026 |
|---|---|---|
| book, annualised | +25.44% | — |
| benchmark, annualised | +15.94% | — |
| **net active, base costs** | **+9.50%** | +12.86% |
| **Newey-West t** | **+1.87** | +3.17 |
| composite rank IC, h=42 | +0.0806 (t +8.39) | +0.0739 |
| under pessimistic costs | +8.79% | — |
| (N,H) cells positive | 76% | 100% |
| factors with positive excess | 8 of 9 | 9 of 9 |

**Pre-registered primary criteria: net active > 0 PASS, NW t > 2.0 FAIL, composite
rank IC > 0 PASS. SHIP GATE FAILED.**

The pre-registration declared in advance that positive-but-not-significant would be
reported as *"positive, underpowered"* and not as a pass. It is.

## Why it is not the default

Five of ten gates. Besides the sealed t, the nested CPCV p5 is −0.65% (bar > 0),
PBO is 0.373 over 42 configurations (bar < 0.10), and the Deflated Sharpe is 0.877
deflated by 140 trials (bar > 0.95).

**The forecast generalised; the significance did not.** Composite rank IC is higher
out of sample than in, on a window with half the breadth — 293 eligible names per
date against 631, because a fixed ₹5 crore ADTV floor is far stricter in real terms
in 2012. A t of 1.87 on 1,433 book days at a 42-session hold is a power problem.

## Two things not to do

**Do not re-specify against 2012-2017.** It is burnt. In particular
`ret_kurt_126` carries the largest weight and is the only one of the nine negative
on that window (−2.50%, t −0.93). Removing it and re-testing there is exactly what
the pre-registration forbids; it is a hypothesis for a window that does not exist
yet.

**Do not expect a paper trade to settle this.** At the sealed window's information
ratio of 0.78, expected `t = IR x sqrt(T)` reaches 2.0 after **6.5 years**. Any
forward test scheduled to return a verdict on significance in 6 or 18 months is
registered to fail. Use it to falsify — sign, IC decay, cost realism, fill rate —
not to ship.

## Known limits

- The sealed window has **no delivery data, no VWAP and no fundamentals**, so
  `deliv_z_21`, `net_margin` and `margin_stability` — all three LS-GATE survivors,
  and two of them the composite's only real diversifiers — got no out-of-sample
  test at all.
- Effective bets: **4.46 from 9 factors**, PC1 explaining 54.8%.
- Impact uses traded value as the volume term; there is no free-float data. The
  half-spread is a swept scenario (3/6/10/15, 5/10/15/25, 10/20/30/50 bps per side
  by ADTV), not an estimate — Corwin-Schultz and Abdi-Ranaldo were both tried on
  this data and rejected, correlating +0.069 with each other.
