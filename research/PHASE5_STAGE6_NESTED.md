# Phase 5/7 — Stage 6 under nested validation

**Question (Gap 7, §8).** Stage 6's band widths were chosen by convention —
`entry_rank` = book size, `exit_rank` = 2 × entry, the rule NSE, MSCI and FTSE
use for factor-index construction. A convention is not a validation. Does
selecting the bands properly beat fixing them?

**Answer. No. And the reason is more useful than the answer.**

## Construction

Outer loop: 45 CPCV splits over the selection period, purged 63 sessions,
embargoed 21. Inner loop: within each outer *training* set the dates are cut
again — the earlier part fits the model, the later 30% selects the band from a
grid fixed in advance. The chosen band is then applied to the outer test block,
which nothing in the selection has touched.

Grid, fixed before any result was seen:
`(5,10) (5,15) (8,16) (8,24) (10,20) (10,30) (12,24)`

## Result

| | median Sharpe | min | max | sd | splits negative |
|---|---|---|---|---|---|
| nested selection | **+0.80** | −1.44 | +2.31 | 0.83 | 20% |
| fixed convention (8/16) | **+0.77** | −1.21 | +2.31 | 0.85 | 22% |

Selecting the bands properly is worth **+0.03 Sharpe** against fixing them by
convention, on a distribution whose standard deviation is 0.83. That is noise.

## The scatter is the finding

How often each band won the inner loop across 45 splits:

| entry_rank | wins | | exit_rank | wins |
|---|---|---|---|---|
| 10 | 21 | | 30 | 19 |
| 8 | 18 | | 16 | 9 |
| 5 | 6 | | 24 | 9 |
| | | | 15 | 6 |
| | | | 20 | 2 |

**If one band were genuinely better, the inner loop would keep picking it.** It
does not — the winner scatters across the grid, and `exit_rank` scatters across
all five values. The inner loop is reading noise, which is direct evidence that
band width does not matter anywhere in this range.

## Selection optimism is near zero, for the same reason

| | mean Sharpe |
|---|---|
| inner block (where the choice was made) | +0.76 |
| outer block (where it is reported) | +0.74 |
| **optimism** | **+0.02** |

Nested validation usually exposes large optimism, because the inner loop finds
something and the outer loop discovers it was noise. Here there is almost none —
not because the selection is skilful but because **there is nothing to select.**
All seven grid points perform about the same.

## Decision

**Keep `entry_rank` = 8, `exit_rank` = 16. Stop treating band width as a
parameter worth tuning.**

The convention is as good as anything the data can select, and searching it
harder would add researcher degrees of freedom — the PBO 44.3% problem — in
exchange for +0.03 Sharpe of noise. The right response to a parameter that does
not matter is to fix it on a defensible external rule and spend the degrees of
freedom nowhere.

## What this does and does not settle

It settles the band widths. It does **not** revisit whether admission should be
by rank at all — that was measured separately and decisively: the old
price-trigger gate returned holdout Sharpe +0.46 against +1.56 for rank
admission with hysteresis, and Grinold/Clarke–de Silva–Thorley explain why
(breadth accounts for one sixth of the gap; the rest is transfer coefficient).

Both results point the same way: **Stage 6's value is in admitting the model's
own ranking, not in any parameter of the admission.**
