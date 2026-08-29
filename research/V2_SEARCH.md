# v2 signal engine — what was searched, what was cut, what shipped

Assembled 2026-08-29. Every number below is from the training window
(2018-11-27 → 2024-10-25) unless it is labelled **holdout**, in which case it is
from the sealed window that was opened exactly once.

The search code is in `work/v2/`; the result tables are in `research/v2/`.

---

## 1. The holdout, sealed first

`work/v2/seal.py` split the panel before any model was fitted:

| | |
|---|---|
| Holdout | **2025-03-06 → 2026-08-17**, 72 signal dates, 54,000 rows |
| Training | 2018-11-27 → 2024-10-25, 293 signal dates, 142,298 rows |
| Purged between | 17 signal dates (label horizon 42 + embargo 21 = 84 sessions) |
| sha256 of the sealed file | `c4e4386d…4cdac118`, recorded in `research/v2/SEAL.json` |

The seal was re-cut once, before any model existed, when a data defect was
found: `prices.deliv_pct` is empty for the whole store and delivery has to be
read from its own table. The boundary date is a constant in `seal.py` and did
not move. No holdout row was read until `work/v2/holdout.py` ran.

## 2. Execution model

A signal computed on the close of session *t* is filled at the **VWAP of
session t+1** and exited at the VWAP of *t+1+h*. Nothing in a feature reads past
*t*. That gap is the manual next-session execution the product asks of its user,
and it costs real return relative to a close-to-close label.

Costs are the engine's own `CostModel`: delivery-segment STT, stamp duty,
exchange and SEBI fees, GST on the right base, DP charge, and square-root impact
— **87 bps round trip** on a ₹1.25 lakh position at ₹20 cr ADTV, 137 bps at ₹5 cr.

## 3. Factors: 69 built, 10 shipped

Built across momentum (multiple horizons, residual, vol-adjusted,
intraday/overnight decomposition, frog-in-the-pan), reversal, volatility and
drawdown, liquidity and size, NSE delivery, trend, and Heston–Sadka seasonality.

**The screen is a placebo alignment, not a shuffle.** The factor cross-section
from date *t* is scored against the label cross-section from *t+k* for every
|k| ≥ 60 signal dates. Those placebo series carry the same cross-sectional
structure and the same overlap-induced autocorrelation as the real one and none
of the signal, so their t-statistics are the honest critical value. An analytic
N(0,1) is not: overlapping 42-session labels sampled every 5 inflate a naive t
by roughly √(h/step).

That null is strict, and it is the single most useful thing in this search:

| Factor | naive t (h=42) | placebo \|t\| 95th | verdict |
|---|---|---|---|
| `ret_kurt_126` | −9.51 | 6.86 | keep |
| `voladj_mom_12_1` | +7.18 | 6.99 | keep |
| `intraday_mom_126` | +6.82 | 4.71 | keep |
| `prox_52w` | +6.70 | 4.33 | keep |
| **`resid_mom_252_21`** | **+7.03** | **16.97** | **cut** |
| **`mom_12_1`** | **+6.85** | **8.82** | **cut** |
| **`dist_low_52w`** | **+6.57** | **7.66** | **cut** |
| **`log_adtv_60`** | **−4.55** | **9.11** | **cut** |

The cut column is the point. `resid_mom` — the incumbent engine's flagship — has
a large naive t and a placebo distribution more than twice as large, because a
factor that persistent produces an IC series that persistent, and a t computed
on it measures persistence rather than prediction. Also cut on the same grounds:
`beta_120`, `beta_252`, `amihud_60`, `idio_vol_126`, `ma_50_200`,
`max_dd_252`, `ulcer_252`, `trend_r2_120`, `seasonal_same_month`, and 26 others.

**Shipped (10):** `ret_kurt_126` (−), `voladj_mom_12_1`, `mom_consist_126`,
`intraday_mom_126`, `prox_52w`, `voladj_mom_6_1`, `deliv_z_21`, `prox_52w_now`,
`mom_3_1`, `volume_shock_5`.

### Cut for want of data, measured rather than assumed

| Family | Why |
|---|---|
| Value, quality (earnings yield, B/P, S/P, EV/EBITDA, FCF yield, ROE, accruals) | The PIT fundamentals table covers **22–28%** of the panel and its newest `filing_date` is **2025-03-11**. By the end of the holdout the median name's most recent filing is **453 days old**. A frozen snapshot is not a point-in-time factor. |
| Ownership structure (promoter holding, pledge, FII/DII flows) | `config/reference/promoter_pledging.csv` is a header row. The NSE JSON API is unreachable from this environment. Nothing to build from. |

The one ownership-shaped signal that *is* available is delivery — the share of
traded volume that actually settles — and it survives the placebo null at every
horizon tested. It is the India-specific factor in the shipped set.

## 4. Combination methods: complexity did not pay

Every method ran through the same purged walk-forward. Mean over all feature
sets and horizons, universes 500 and 750:

| Method | rank IC | top-10 excess / period |
|---|---|---|
| **sign-oriented equal weight** | **0.0465** | **+0.36%** |
| PCR (8 components) | 0.0444 | +0.19% |
| IC-weighted | 0.0436 | +0.34% |
| Lasso | 0.0427 | +0.38% |
| Ridge (α 20 / 200) | 0.0415 | +0.40% |
| XGBoost, depth 2 | 0.0340 | **−0.10%** |
| XGBoost, depth 3 | 0.0319 | **−0.11%** |
| XGBoost, depth 4 | 0.0256 | **−0.19%** |

Gradient boosting has a defensible pooled IC and a **negative top-decile
excess** at every depth: it ranks the middle of the cross-section and gets the
tail wrong, which is the only part of the cross-section a concentrated long book
lives in. Cut.

Ridge collapsed once the validation window was extended back to include the 2020
crash — the incumbent factor set under ridge scored top-10 excess **+0.026
(t 5.3)** on a 2021-onward window and **+0.0003 (t 0.12)** on a window starting
2020-01. That is the momentum-crash exposure the earlier audit documented,
showing up as a selection artefact.

The shipped composite fits **one parameter per factor: its sign.** That sign was
identical in all eight walk-forward folds for all ten factors.

## 5. Universe

Mean excess and information ratio across the whole book-construction sweep
(48,384 configurations, `research/v2/r3.csv.gz`):

| Top-N by 60-session median ADTV | mean excess | mean IR |
|---|---|---|
| 100 | −12.7% | −1.10 |
| 200 | −16.0% | −1.03 |
| **500** | **−1.9%** | **+0.04** |
| **750** | **−1.8%** | **+0.05** |

NIFTY-100- and NIFTY-200-sized universes are clearly worse. The data supports a
broad liquid universe; 750 was shipped. (Every mean here is negative because
most of a 48k grid is bad — the marginals compare dimensions, they do not
describe the shipped book.)

## 6. Regime gating: measured, and it cost money

| Entry gate | open % of sessions | excess | max drawdown | 2020 excess |
|---|---|---|---|---|
| **none** | 100% | **+28.8%** | −32.5% | **+36.4%** |
| market > 200 DMA | 63% | +16.4% | −30.7% | +14.3% |
| calm volatility | 68% | +16.1% | −32.6% | +12.2% |
| name > 50 DMA (bottom-up) | — | +15.8% | −37.2% | +16.9% |

Every gate cost 8–13 points of annual excess **and none of them meaningfully
reduced the drawdown they exist to avoid** — including through 2020, because the
fall was faster than any of these signals and the rebound faster still.

So the gate is **computed and reported on every run, and not applied**, and the
drawdown circuit breaker **flags** rather than disabling anything
(`v2_monitor.review_drawdown`). NO TRADE remains reachable structurally: a slot
with no admissible name holds cash at 0%, and cash is never credited with a
return, so sitting out is never rewarded by an assumption.

## 7. The power test — the most important result here

Before the holdout was opened, the **entire pipeline** — screen, sign
orientation, composite, book — was re-run 40 times on labels permuted within
each cross-section (`work/v2/run_r8.py`). If the pipeline is clean and the
signal is real, the real result should sit outside that null.

| Statistic | real | null mean | null p95 | p | z |
|---|---|---|---|---|---|
| Quintile spread / 42 sessions | **+1.85%** | +0.003% | +0.41% | **0.00** | **6.37** |
| Quintile-spread t | **6.05** | 0.01 | 2.54 | **0.00** | 3.31 |
| 10-name book, annual excess | +28.8% | −0.6% | +14.4% | 0.00 | 2.52 |
| 25-name book, annual excess | +14.2% | +1.4% | +15.1% | **0.10** | 1.38 |
| 40-name book, annual excess | +8.3% | +2.3% | +13.7% | **0.25** | 0.76 |

Read the null's standard deviation on the book rows: **11.7 percentage points**
for a ten-name book over five years, under a null with no signal at all. The
ranking separates from noise by six sigma. **A concentrated book's annual excess
barely separates at all**, and this was known *before* the holdout was opened.

## 8. The sealed holdout — one run, one configuration

`work/v2/holdout.py`, executed once against
`FROZEN_CONFIG.json` (sha256 `9717e1ba…0854a1b7`, written before the file ran).

| | validation 2020-01 → 2024-10 | **holdout 2025-03 → 2026-08** |
|---|---|---|
| Net annualised | 52.2% | **17.7%** |
| Benchmark (equal-weight eligible) | 29.7% | **15.3%** |
| **Excess, net of costs** | +22.5% | **+2.4%** |
| Information ratio | 1.15 | **−0.19** |
| Sharpe | 1.71 | **1.07** |
| Max drawdown | −38.0% | **−14.0%** |
| Cost drag | 6.0%/yr | **5.4%/yr** |
| Median hold | 20 sessions | **20 sessions** |
| Rank IC (t) | 0.049 (6.0) | **0.045 (2.59)** |
| Quintile spread / period (t) | +1.85% (6.05) | **+1.65% (2.56)** |
| Top-10 excess / period (t) | +2.05% (3.82) | **−1.39% (−2.16)** |

Against a 200-draw shuffled-score null on the holdout itself: quintile spread
**p = 0.00**, book excess **p = 0.00**.

### What that means, in both directions

**The ranking generalised.** Rank IC and quintile spread held roughly their
training magnitude on data the model had never seen, and both sit far outside
their own null. The scorer orders the cross-section.

**The ten-name book did not.** +2.4% a year net is inside noise — the
information ratio is *negative*, so the positive compounded excess is a
volatility artefact rather than a mean — and the top ten names underperformed
the cross-section they were drawn from at the 42-session horizon. Section 7
predicted exactly this: the +22.5% the validation window showed was mostly luck
sitting on a real but much smaller edge.

**Nothing was re-tuned after this number was seen.** No parameter moved, no
window was re-cut, no second configuration was tried. `work/v2/holdout.py`
refuses to run twice.

## 9. What is NOT established

- That the shortlist ordering is meaningful between #1 and #10. On the holdout
  it was not. The card says so.
- That a ten-name book earns an excess return. The point estimate is +2.4% a
  year with a null standard deviation of roughly ±12 points.
- That any of this survives a regime the 2018–2026 window does not contain.
- That the value and quality families are dead — only that this store cannot
  test them. A point-in-time fundamentals feed with real filing dates would
  reopen the question.

## 10. What would change the answer

1. **Accumulate the quarterly re-check** (`prosignal research v2 --recheck`).
   It withholds a verdict until it has the 8 independent windows the deploy was
   judged on — about 68 signal dates, or roughly six quarters.
2. **A point-in-time fundamentals feed.** The value/quality block is cut for
   coverage, not for evidence.
3. **A book-sizing study with power.** The estimator, not the signal, is what
   failed on the holdout. Widening the book lowers the point estimate and lowers
   the noise; which wins is measurable and has not been measured on fresh data.
