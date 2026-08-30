# v3 signal engine — the two-level thematic composite

Assembled 2026-08-30. Every number is from the training window
(2018-11-27 → 2024-10-25) or its validation slice (2020-01-31 → 2024-10-25)
unless it is labelled **holdout**, in which case it comes from a sealed window
opened exactly once per configuration.

Search code in `work/v3/`; result tables in `research/v3/`. The shipped scorer
is `src/prosignal/features/v3.py` and `v3_factors.py`; nothing in this document
is re-derived at runtime.

---

## 0. What the brief asked for, and what it got

| Asked | Delivered | Where |
|---|---|---|
| Point-in-time fundamentals, lagged to filing | measured p99 lag per quarter-end month | §1 |
| Survivorship bias removed | measured, quantified, **not fixed** — see the caveat | §2 |
| Factor redundancy resolved | 2 of the brief's 3 suspicions refuted, the real duplicate found | §4 |
| Two-level thematic composite | 22 factors → 5 theme sub-scores → capped blend | §5–6 |
| No theme swamps the others | cap 0.40, floor 0.06, **coverage cap** | §6 |
| Absolute quality floor, NO TRADE below it | 200-DMA **and** ≥3 themes above median | §7 |
| One blind holdout per configuration | two windows, and one defect that cost a re-run | §8 |

---

## 1. Point-in-time fundamentals

`fundamentals.parquet` carries a real `filing_date` and is used as given.
`statements.parquet` carries only `period_end`, and a factor keyed on period end
reads a number weeks or months before anybody could see it.

**The lag is measured, not asserted.** The 3,504 rows that do carry a filing
date give the disclosure-lag distribution directly:

| quarter end | n | p95 | p99 | max |
|---|---|---|---|---|
| March (audited) | 871 | 91 | **112** | 391 |
| June | 832 | 64 | **104** | 132 |
| September | 880 | 45 | **57** | 170 |
| December | 921 | 45 | **54** | 77 |

The statutory SEBI LODR deadline is 45 days (60 for the audited annual). Only
**84.1%** of real filings were out by day 45 — a 45-day assumption leaks on one
row in six. Shipped lag is the measured p99 per quarter-end month floored at 60
days: `{3: 112, 6: 104, 9: 60, 12: 60}`. It is deliberately on the late side. A
lag that is too long makes a factor staler and weaker; a lag that is too short
manufactures alpha.

A per-field staleness gate drops any value older than **420 days** (one annual
cycle + the audited lag + a quarter of slack), because a number that old is not
a description of the company any more.

**Market cap is made split-invariant** rather than recomputed from a current
share count: `mcap_t = adj_close_t × shares_at_filing / adj_factor_at_filing`.
Dividing today's price by a share count that has since split is a factor that
jumps on corporate actions and looks like information.

## 2. Survivorship — measured, and only partly fixable here

The store holds no historical index-membership snapshots, so a
point-in-time **liquidity** screen stands in for point-in-time index membership.
That is a real limitation and it is stated rather than papered over. What was
measured:

- **141 of 1,425** panel symbols stop printing before the panel ends.
- **6.2%** of panel rows belong to them.
- **26–38%** of an old cross-section is absent from today's symbol list.

So the delisted names ARE in the panel — the panel is built from the full price
history, not from today's list applied backward, which is the bias the brief
names. What is missing is index membership: a name that was in NIFTY 500 in 2019
and is not now is included if it met the liquidity bar on that date, which is
the right answer for a liquidity-defined universe and an approximation for an
index-defined one.

## 3. The universe defect that cost a holdout re-run

NSE publishes ETFs, gold funds, silver funds, liquid funds and bond funds in the
**same EQ-series bhavcopy** as equities. They pass every liquidity screen. In
the first evaluation of holdout window A they took **26.25% of the top-ten
slots** — the model's best ideas were substantially a basket of gold ETFs.

`data/instruments.py` excludes them by two conditions, and only for symbols
**absent from the equity master**: a scheme-name pattern, and a volatility
backstop (annualised vol < 10% over 504 sessions, ≥250 observations). The
backstop was calibrated: on a 60-session slice it over-excluded 199 names
against the pattern's 61, so the window was widened and a minimum observation
count added. Verified against real gold-**jewellery** equities — SKYGOLD,
GOLDIAM and SILVERTUC are kept; 183 instruments are excluded.

## 4. Factors: 93 built, 33 cleared, 22 shipped

Eight themes were built in full before any was judged: momentum (24), risk (19),
quality (15), liquidity (9), value (8), reversal (10), ownership (6),
seasonality (2).

### The screen is a placebo alignment, not a shuffle

The factor cross-section at date *t* is scored against the label cross-section
at *t+k* for every |k| ≥ 60 signal steps. Those placebo series carry the same
cross-sectional structure and the same overlap-induced autocorrelation as the
real one and none of the signal, so their |t| distribution is the honest
critical value. An analytic N(0,1) is not: overlapping labels sampled every 5
sessions inflate a naive t by roughly √(h/step).

Cleared at some horizon: **33 of 93**.

| theme | built | cleared (h=10 / 21 / 42 / 63) |
|---|---|---|
| momentum | 24 | 8 / 6 / 7 / 13 |
| reversal | 10 | 6 / 2 / 0 / 1 |
| risk | 19 | 4 / 3 / 2 / 1 |
| ownership | 6 | 4 / 2 / 1 / 1 |
| quality | 15 | 1 / 2 / 2 / 1 |
| **value** | 8 | **0 / 0 / 0 / 0** |
| **liquidity** | 9 | 0 / 0 / 1 / 0 |
| **seasonality** | 2 | **0 / 0 / 0 / 0** |

### Three themes ship with nothing, and each was built first

- **value** (0 of 8). Built PIT-correct against the measured filing lags:
  `sales_to_price`, `fcf_yield`, `book_to_price`, `ebitda_to_ev`,
  `earnings_yield` and three more. Balance-sheet coverage in this store begins
  **2023**, and the median training date has **zero** names with a book value.
  This is a data limitation, not a verdict on value.
- **liquidity** (0 of 9). `volume_shock_5` clears at h=42 and its sign flips
  between the halves of its own life (−0.0005 → +0.0108), which is the
  both-halves test doing its job.
- **seasonality** (0 of 2). Placebo |t| threshold **9.9** against a real t of
  −1.2. Indian month-of-year seasonality is not detectable at this sample size.

A theme with no validated factor is **excluded, not carried at zero weight** —
carrying it would put an empty column into the renormalisation and change other
names' weights for no reason.

### The both-halves test

Every surviving factor's IC is recomputed on the first and second halves of its
own life. Two failed on a sign flip and were cut: `volume_shock_5`,
`dist_50dma`. All 31 others held their sign.

Sample of the survivors (full table in `research/v3/stability.csv`):

| theme | factor | n dates | IC | t | t (H1) | t (H2) |
|---|---|---|---|---|---|---|
| risk | `ret_kurt_126` | 293 | −0.0293 | −8.26 | −5.69 | −6.20 |
| ownership | `deliv_z_21` | 222 | +0.0276 | +6.24 | +3.73 | +5.26 |
| quality | `margin_stability` | 142 | −0.0351 | −5.52 | −2.83 | −5.30 |
| momentum | `mom_consist_126` | 293 | +0.0330 | +5.35 | +4.73 | +2.64 |
| momentum | `prox_52w` | 293 | +0.0454 | +4.86 | +2.79 | +5.99 |
| risk | `ulcer_120` | 293 | −0.0436 | −4.48 | −2.68 | −5.39 |
| momentum | `mom_accel` | 293 | **−0.0242** | −3.75 | −2.67 | −2.67 |

`mom_accel` is negative inside momentum, consistently, in both halves. It ships
at sign −1 and is pinned by a test so nobody "corrects" it.

## 5. Redundancy — the brief's suspicions, measured

The brief named two suspected duplicate groups. Measurement refuted both and
found a third the brief had not named.

| pair | \|rho\| | verdict |
|---|---|---|
| `mom_6_1` vs `rev_1m_scaled` | **+0.028** | not duplicates |
| `resid_mom_252_21` vs `rev_1m_scaled` | **+0.005** | not duplicates |
| `deliv_pct_60` vs `deliv_trend` | **+0.095** | not duplicates |
| `resid_mom_252_21` vs `mom_6_1` | +0.552 | related, not redundant |
| **`deliv_z_21` vs `deliv_trend`** | **+0.848** | **the real duplicate** |
| `mom_6_1` vs `voladj_mom_6_1` | +0.961 | the strongest pair in the panel |
| `max_dd_120` vs `ulcer_120` | −0.907 | the same statistic twice |

Resolution is an **iterative, order-independent** survivor pass: within a
cluster at |rho| ≥ 0.80, keep the factor with the stronger stability |t| and
drop the other, repeating until no pair remains. 11 factors cut:

| cut | reason |
|---|---|
| `mom_6_1` | \|rho\| 0.96 with `voladj_mom_6_1`; weaker (\|t\| 3.67 vs 4.34) |
| `max_dd_120` | \|rho\| 0.91 with `ulcer_120`; weaker (3.96 vs 4.48) |
| `rev_2w` | \|rho\| 0.88 with `price_vs_vwap_20`; weaker (1.88 vs 2.26) |
| `rsi_14` | \|rho\| 0.87 with `price_vs_vwap_20`; weaker (1.84 vs 2.26) |
| `rev_1m_scaled` | \|rho\| 0.86 with `resid_rev_21`; weaker (1.80 vs 3.32) |
| `deliv_trend` | \|rho\| 0.85 with `deliv_z_21`; weaker (3.18 vs 6.24) |
| `dist_200dma` | \|rho\| 0.85 with `trend_slope_120`; weaker (3.60 vs 3.99) |
| `trend_slope_120` | \|rho\| 0.84 with `voladj_mom_6_1`; weaker (3.99 vs 4.34) |
| `dist_50dma`, `volume_shock_5` | sign flips between halves |
| `gross_profitability` | only 63 signal dates, under the 120 floor |

**22 factors** survive into 5 themes.

## 6. The composite, in two levels

### Level 1 — within theme

Each factor is a **sector-neutral cross-sectional rank in [−1, 1]** (a single
residual group; within-sector and universe ranks are never mixed in one column).
A theme's sub-score is the sign-oriented mean of its factor ranks, then
**re-ranked within the date**. Without the re-rank, a theme whose factors happen
to be more dispersed dominates the blend for a reason that has nothing to do
with information.

**Each theme is oriented at the horizon it works at.** Forced onto a single
42-session label, the reversal sub-score came out **anti-predictive at t −3.96**
— reversal is a two-week effect and at 42 sessions its sign has turned over.

| theme | horizon | factors | coverage |
|---|---|---|---|
| momentum | 42 | 10 | 99.88% |
| quality | 21 | 2 | **18.99%** |
| ownership | 10 | 3 | 89.85% |
| risk | 21 | 3 | 99.83% |
| reversal | 10 | 4 | 99.93% |

Four combination methods were fitted per theme and compared on the same folds:
equal-weight, IC-weighted, ridge, XGBoost.

| method | quintile | quintile t | top-k | top-k t | excess | IR | Sharpe |
|---|---|---|---|---|---|---|---|
| **equal** | **0.0049** | **2.02** | +0.0009 | +0.60 | −0.087 | −0.52 | 0.88 |
| icw | 0.0021 | 0.79 | −0.0010 | −0.62 | −0.139 | −0.85 | 0.73 |
| xgb | 0.0004 | 0.17 | −0.0008 | −0.54 | −0.142 | −1.04 | 0.74 |
| ridge | −0.0004 | −0.14 | −0.0033 | −2.05 | −0.179 | −1.16 | 0.62 |

Sign-oriented equal weight wins on every column above. The learned methods do
buy a shallower drawdown — median maxDD −0.34 (xgb) and −0.36 (icw) against
equal weight's −0.40 — but they buy it by ranking worse, and ridge is
significantly *anti*-predictive on top-k at t −2.05. Two to ten factors per
theme, a few hundred dates and overlapping labels is not a regime where a
gradient-boosted tree has anything to learn. Equal weight ships.

### Level 2 — across themes

Raw theme weights are each theme's validated top-decile excess, then:

| constraint | value | evidence |
|---|---|---|
| cap | 0.40 | uncapped, momentum + quality took 74% between them |
| floor | 0.06 | moved max drawdown −38.5% → −34.9% at no cost in excess |
| **coverage cap** | per theme | see below |

**The coverage cap is the constraint the brief did not ask for and the search
needed.** Weights renormalise over the themes a *name* actually has. Fitted
without a coverage constraint, `quality` took the cap — 40%+ of the composite —
while only **19% of names have fundamentals at all**. That ranks the 19% and the
81% by two different models and calls the result one score. Capping each theme
at the share of names it can speak about:

| | IC | IC t | quintile | quintile t | excess | IR | Sharpe |
|---|---|---|---|---|---|---|---|
| coverage cap **off** | 0.0400 | 4.84 | 0.0073 | 2.91 | +0.055 | 0.30 | 1.35 |
| coverage cap **on** | **0.0444** | **5.80** | **0.0082** | **3.42** | +0.019 | 0.10 | 1.23 |

The ranking improves on both statistics with power and the ten-name book's
excess falls. **The ranking numbers are the ones that decide**: a permuted-label
test put the book's excess almost entirely inside its own null while the
quintile spread sat 6.4 standard deviations outside it (§9). The cap ships on.

Shipped weights, post-cap, post-floor, post-coverage:

| theme | weight | | theme | weight |
|---|---|---|---|---|
| momentum | 0.40000 | | risk | 0.11088 |
| quality | 0.18991 | | reversal | 0.10982 |
| ownership | 0.18939 | | | |

Momentum sits *at* the cap, which is the cap doing its job. Without the two
levels a flat 22-factor sum would have been 10/22 momentum by construction and
"holistic" only in the sense that it contained other columns.

## 7. The absolute quality floor

The brief: *"a stock doesn't make the list just for being the best of a weak
universe on a given day. NO TRADE is what you get when nothing clears that
floor."*

**A floor on a cross-sectional rank cannot fire** — somebody is top of the list
every day. The first attempt, "at least 3 of the name's available themes above
the cross-sectional median", left **at least 87 names on every one of 235
validation dates**. It was not a floor, it was decoration.

The shipped floor adds an absolute condition measured against the stock itself:

> **close > 200-session MA** *and* **≥3 themes above the cross-sectional median**

| gate | pass rate | dates with no name | dates short of a full book | excess | Sharpe | maxDD |
|---|---|---|---|---|---|---|
| none | 100% | 0 | 0 | −0.004 | 1.155 | −0.372 |
| `npos3` only | 37.2% | 0 | 0 | −0.004 | 1.152 | −0.349 |
| **`trend_npos3`** | **29.6%** | 0 | 0 | −0.029 | 1.102 | **−0.327** |
| `trend_npos4` | 10.8% | 0 | 8 | −0.239 | 0.345 | −0.332 |
| `strict` | 6.5% | **1** | 32 | −0.392 | −0.644 | −0.478 |

Verified it can actually empty the list: **11 names** clear at the COVID trough
(2020-03-24), against a book of 10 slots; 8–9 in the 2022 drawdown. On the
sealed windows the minimum was **47 names (A)** and **9 names (B)** — window B
had one date short of a full book and none with no name at all.

**The floor applies to ENTRIES only, not to holdings.** Found by mechanism, not
by search: a floor applied to the whole population forces exits every time a
held name dips below its 200-DMA, and forced exits are turnover. Measured on
training data only, over the same grid:

| | cost | gross | excess | IR | turnover |
|---|---|---|---|---|---|
| population filter | 8.8% | 0.381 | +0.001 | −0.045 | 434 |
| **entry-only** | **5.6%** | 0.373 | **+0.018** | **+0.083** | 400 |

## 8. The sealed holdouts

`work/v3/seal2.py` cut two windows before any v3 model was fitted:

| | window A | window B |
|---|---|---|
| dates | 2025-03-06 → 2026-08-17 | 2021-07-01 → 2022-12-27 |
| signal dates | 72 | 75 |
| rows | 54,000 | 40,027 |
| sha256 | `b0e7abf9…6c4f8f83` | `2b4b5cb7…1c62060e8` |

Training: 2018-11-27 → 2024-10-25 (293 dates). For **window B the entire
pipeline was re-run on data ending 2021-02-17** (111 dates) — screen,
stability, redundancy, admission, theme weights — and only then evaluated on the
eighteen months that followed. That is a test of the *method*, not of a fitted
model, and it is the cleanest read available here.

### What they said

| | **A** 2025-03→2026-08 | **B** 2021-07→2022-12 |
|---|---|---|
| rank IC, h=21 (t) | **+0.0493 (3.69)** | **+0.0357 (3.83)** |
| quintile spread (t) | **+1.07% (2.89)** | **+0.86% (3.05)** |
| top-ten excess (t) | +0.38% (0.81) | +1.37% (2.50) |
| themes with positive OOS IC | **5 of 5** | **3 of 3** |
| shuffled-score null, p | 0.02 | 0.01 |
| ten-name book, net excess | **−2.8%/yr** | **+2.0%/yr** |
| modelled cost drag | 9.7%/yr | 13.7%/yr |
| max drawdown | −23.9% | −16.4% |
| names clearing the floor (median / min) | 169.5 / 47 | 48 / 9 |

### And the run these supersede

The **first** evaluation of both windows ran on the universe with ETFs still in
it (§3). Those numbers stay on the record because they happened:

| | A | B |
|---|---|---|
| rank IC (t) | +0.0586 (3.66) | +0.0380 (4.08) |
| quintile spread (t) | +1.12% (2.65) | +0.90% (3.18) |
| top-ten excess (t) | −0.11% (−0.25) | +1.50% (2.72) |
| ten-name book, net excess | −7.2%/yr | +1.3%/yr |

The **universe** was defective, not the configuration. Both windows were re-run
with every parameter untouched — nothing was tuned after a holdout number was
seen, which is the rule the brief set. It still costs something: **window A has
now been evaluated three times** counting the pre-seal dry run, and its
t-statistics carry that multiplicity. Window B — once before the fix, once
after, positive both times — is the cleaner of the two.

### What generalised and what did not

**The ranking generalised on both windows**, and every theme carried positive
information out of sample on both — which is the two-level structure doing its
job rather than momentum doing all the work while wearing a theme costume.

**The concentrated ten-name book did not.** On window B it earned ~15.7% gross
and paid **13.7%** of it away in transaction costs. That is not a data surprise:
the same book showed a **9.9%** cost drag in validation, before either seal was
opened. It was visible and it was selected past. The cost curve across the whole
book sweep:

| modelled cost/yr | gross | excess | IR | Sharpe | median hold |
|---|---|---|---|---|---|
| 0–2% | 0.329 | +0.015 | 0.03 | 1.47 | 80 |
| 2–3% | 0.340 | +0.017 | 0.05 | 1.40 | 60 |
| **3–4%** | 0.355 | **+0.023** | **0.07** | 1.46 | 40 |
| 4–6% | 0.358 | +0.013 | −0.03 | 1.58 | 40 |
| 6–9% | 0.369 | −0.001 | −0.08 | 1.33 | 30 |
| 9%+ | 0.414 | +0.025 | 0.08 | 1.41 | 20 |

Costs are the engine's own `CostModel`: delivery-segment STT (0.1% both legs),
stamp 0.015% buy, exchange 0.00297%, SEBI 0.0001%, GST 18% on the right base,
₹20 brokerage, ₹15.93 DP sell, plus square-root impact — **87 bps round trip**
on a ₹1.25 lakh position at ₹20 cr ADTV.

### Three books exist, and only one of them trades

Keeping them apart matters more than any of their contents: a number measured on
one book and quoted about another is how a backtest becomes a claim it never
made.

| | slots | entry | exit | cadence | evaluated? |
|---|---|---|---|---|---|
| the holdout book | 10 | 20 | 30 | weekly | **yes, both windows** |
| the research book (`v3.RESEARCH_BOOK`) | 12 | 24 | 48 | 10 sessions | no — chosen on training against a cost target |
| **the LIVE book** (`parameters.yaml`) | **6** | **6** | **18** | **21 sessions** | **no** |

The live book is both **slower** and **much more concentrated** than anything
the sealed windows measured. Slower is the safe direction: it cuts the cost drag
that sank the tested book, and turnover is arithmetic on a fee schedule — no
labels needed to verify it.

Concentration cuts the other way, and it leans on the statistic that generalised
**least**. On window A the top-ten excess was **+0.38% at t 0.81** —
indistinguishable from zero — while the quintile spread held at t 2.89. The
ordering *within* the top few names is precisely the part of this model the
holdouts did not support, and a six-name book is a bet on exactly that.

**So the ranking is holdout-validated and no book is.** The concentration is an
operator's risk choice, not a validated one. `features/v3.py::BOOK_NOTE` says
this in the code, `LIVE_BOOK` mirrors the config with a test that fails when it
drifts, and turnover and cost drag are reported on every run.

## 9. Power — is any of this distinguishable from luck?

A permuted-label test re-runs the **entire** pipeline — screen, orientation,
composite, book — on labels permuted within each cross-section, 40 draws:

| statistic | real | null mean | null p95 | z | p |
|---|---|---|---|---|---|
| **quintile spread** | **0.0185** | 0.00003 | 0.0041 | **+6.37** | 0.00 |
| quintile t | 6.05 | 0.01 | 2.54 | +3.31 | 0.00 |
| ten-name excess | 0.288 | +0.034 | **0.266** | — | ≈0.10 |

**The quintile spread is six standard deviations outside its own null. The
ten-name book's five-year excess is barely outside the 95th percentile of a
pipeline run on random labels.** That single comparison is why the ranking is
the headline everywhere in this document and the book is reported with its cost
drag attached.

## 10. Lookahead guard

`mae5`, `mfe5`, `entry_px` and every `y*`/`b*` column leaked into an early
screen as factors with IC 0.22–0.25 at t > 30. They are forward-looking by
construction — the screen had picked them because it enumerated columns by
exclusion.

Fixed by inverting the rule: factor columns are sourced from **declared themes
only**, never by excluding known-bad names, and an explicit forbidden set raises
`LookaheadError` if one ever appears. A test in `tests/test_v3_score.py`
truncates the panel and asserts the last row is bit-identical, so no factor can
read a session after the decision row.

## 11. Execution

Signal on the close of *t* → fill at the **VWAP of t+1** → exit at the VWAP of
*t+1+h*. That gap is the manual next-session execution the product asks of its
user, and it costs real return relative to a close-to-close label. Nothing in
any feature reads past *t*.

## 12. What is monitored live

`v3_monitor.py`, and it **flags, never disables** — a monitor that switches a
theme off changes the model without a decision being taken, and the next person
to read the config sees a model that is not the one running.

- rolling IC **per factor**, oriented by the shipped sign
- rolling IC **per theme** — a composite can hold up in aggregate while one
  theme has inverted, and the composite's own IC will not say which
- each theme's **share of realised cross-sectional spread**, flagged past 55%
  or 15 points above its own declared weight
- book drawdown against a −25% flag (deepest across both sealed windows: −23.9%)

The influence share is a **dispersion** share, not a variance share. Variance is
quadratic in the weight: measured as variance, momentum reads 62% at its
declared 40%, and a 55% alarm set in the units of the weight cap would have
fired on a perfectly healthy book every day from the first run. A monitor that
always fires is a monitor that gets turned off.

## 13. Known limitations

1. **No ownership-structure data.** Promoter holding, pledge percentage and
   FII/DII flows are unbuildable from this store — no network route to them was
   reachable. The `ownership` theme is NSE **delivery percentage** only, which
   is a different (and weaker) thing than the brief's ownership structure.
2. **Value is empty because the data is.** Balance-sheet coverage starts 2023.
   When it deepens, the value theme should be re-screened — it was built
   PIT-correct and is ready to run.
3. **Index membership is not point-in-time**, only liquidity is (§2).
4. **The shipped book is not holdout-validated.** Only the ranking is.
5. **Window A carries a multiplicity charge of 3.** Read its t-statistics with
   that in mind; window B is the cleaner read.
