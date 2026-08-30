# CHANGELOG

## v3 signal engine — the two-level thematic composite — 2026-08-30

Full search record in `research/V3_SEARCH.md`; search code in `work/v3/`;
sealed artefacts and result tables in `research/v3/`.

v3 replaces the v2 **combination step**, not the engine. The universe screen,
execution model, cost model, holdout machinery and UI contract are unchanged.
What changed is how factors become a score: v2 summed ten factor ranks flat,
which is whichever theme brought the most columns wearing a disguise. v3 groups
22 factors into 5 themes, combines within each theme first, and blends the
themes under a cap.

### The holdouts that earned the deploy

**Two** windows, sealed before any v3 model was fitted, opened once each per
configuration:

| | **A** 2025-03-06→2026-08-17 | **B** 2021-07-01→2022-12-27 |
|---|---|---|
| signal dates / rows | 72 / 54,000 | 75 / 40,027 |
| rank IC, h=21 (t) | **+0.0493 (3.69)** | **+0.0357 (3.83)** |
| quintile spread (t) | **+1.07% (2.89)** | **+0.86% (3.05)** |
| top-ten excess (t) | +0.38% (0.81) | +1.37% (2.50) |
| themes with positive OOS IC | **5 of 5** | **3 of 3** |
| shuffled-score null, p | 0.02 | 0.01 |
| ten-name book, net excess | −2.8%/yr | +2.0%/yr |
| modelled cost drag | 9.7%/yr | 13.7%/yr |
| max drawdown | −23.9% | −16.4% |

For **window B the entire pipeline** — screen, stability, redundancy, admission,
theme weights — was re-run on data ending 2021-02-17 and evaluated once on the
eighteen months that followed. That tests the *method*, not a fitted model.

**The ranking is holdout-validated. No book is.** Every theme carried positive
information out of sample on both windows, which is the two-level structure
doing its job rather than momentum doing all the work. The ten-name book the
windows evaluated, though, earned ~15.7% gross on window B and paid **13.7%** of
it away in costs — and that was visible before either seal was opened, at 9.9%
in validation.

**And that is not the book that trades.** Production runs **six** positions on a
21-session cadence with a 3× exit band, from `config/parameters.yaml` — slower
*and* far more concentrated than anything either window measured. Slower cuts
the cost drag that sank the tested book, and turnover needs no labels to verify.
More concentrated cuts the other way, and it leans on the statistic that
generalised least: top-ten excess on window A was **+0.38% at t 0.81**,
indistinguishable from zero, while the quintile spread held at t 2.89. Ordering
within the top few names is the part of this model the holdouts did not support,
and a six-name book is a bet on exactly that. The concentration is an operator's
risk choice, not a validated one. `features/v3.py::BOOK_NOTE` now says all of
this, and a test fails if the mirror of the live book drifts from the config.

### A defect found, and what it cost

The **first** evaluation of both windows ran on a universe that still contained
ETFs, gold funds and liquid funds — NSE publishes them in the same EQ-series
bhavcopy as equities, and they took **26.25% of window A's top-ten slots**. That
run read A +0.0586 (3.66) / +1.12% (2.65) / −7.2% book and B +0.0380 (4.08) /
+0.90% (3.18) / +1.3% book; both are kept on the record in `parameters.yaml`.

The **universe** was defective, not the configuration, so both windows were
re-run with every parameter untouched — nothing was tuned after a holdout number
was seen. It still costs: window A has now been evaluated three times counting
the pre-seal dry run, and its t-statistics carry that multiplicity. Window B,
positive before and after the fix, is the cleaner read.

### The shipped configuration

- **22 factors in 5 themes.** momentum (10, oriented at 42 sessions), quality
  (2, 21), ownership (3, 10), risk (3, 21), reversal (4, 10). Each theme is
  oriented at the horizon it works at: forced onto one 42-session label the
  reversal sub-score came out **anti-predictive at t −3.96**.
- **Level 1**: sign-oriented equal-weight mean of sector-neutral factor ranks,
  re-ranked within the date so themes are commensurable. Equal weight beat
  IC-weighted, ridge and XGBoost on every ranking column of the fold comparison
  (the learned methods bought a ~4-point shallower drawdown by ranking worse;
  ridge was significantly *anti*-predictive on top-k at t −2.05).
- **Level 2**: weights from validated contribution, then **cap 0.40, floor 0.06,
  and a coverage cap**. Shipped: momentum 0.400, quality 0.190, ownership 0.189,
  risk 0.111, reversal 0.110.
- **Absolute floor**, entries only: close > 200-session MA **and** ≥3 themes
  above the cross-sectional median. Below it, **NO TRADE**.

### Three things the search found that the brief did not ask for

- **The coverage cap.** Weights renormalise over the themes a *name* has. Fitted
  without a coverage constraint, `quality` took the 40% cap while only **19% of
  names have fundamentals** — ranking the 19% and the 81% by two different
  models and calling it one score. Capping each theme at the share of names it
  can speak about: IC t 4.84 → **5.80**, quintile t 2.91 → **3.42**.
- **A rank floor cannot fire.** "≥3 themes above median" alone left at least
  **87 names on every one of 235 validation dates**. Adding the 200-DMA
  condition made it a floor: 11 names at the COVID trough against a 10-slot
  book, 8–9 in the 2022 drawdown, 47 (A) and 9 (B) minimum on the sealed windows.
- **The floor belongs on entries, not holdings.** Applied to the population it
  forces an exit every time a held name dips below its 200-DMA, and forced exits
  are turnover: cost drag **8.8% → 5.6%**, excess +0.1% → +1.8% (training only).

### The brief's redundancy suspicions, measured

Two of the three were refuted and the real duplicate was elsewhere:
`mom_6_1` vs `rev_1m_scaled` **+0.028**; `deliv_pct_60` vs `deliv_trend`
**+0.095**; but `deliv_z_21` vs `deliv_trend` **+0.848**. 11 factors were cut by
an order-independent survivor pass at |rho| ≥ 0.80, keeping the stronger of each
pair by stability t.

### Three themes ship with nothing, and each was built in full first

- **value** — 0 of 8 clear at any horizon. Built PIT-correct against measured
  filing lags; balance-sheet data in this store begins **2023** and the median
  training date has **zero** names with a book value. A data limitation, not a
  verdict on value.
- **liquidity** — 0 of 9. `volume_shock_5` clears at h=42 and flips sign between
  the halves of its own life.
- **seasonality** — 0 of 2. Placebo |t| threshold **9.9** against a real 1.2.

An empty theme is **excluded, not carried at zero** — carrying it would put a
dead column into the per-name renormalisation.

### Point-in-time integrity

Every fundamental factor is as-of joined on **disclosure** dates, never period
ends. Where a real `filing_date` exists it is used; where only `period_end`
exists the lag is the **measured p99** of the real filing lag by quarter-end
month — `{Mar: 112, Jun: 104, Sep: 60, Dec: 60}` days, from 3,504 real filings.
The statutory 45-day deadline would have leaked on **15.9%** of them. A per-field
staleness gate drops anything older than 420 days, and market cap is made
split-invariant rather than recomputed from a current share count.

**Survivorship, measured rather than assumed:** 141 of 1,425 panel symbols stop
printing before the panel ends, 6.2% of rows belong to them, and 26–38% of an
old cross-section is absent from today's list. The panel keeps them. What is
still missing is point-in-time *index membership* — the store has no membership
snapshots, so a point-in-time liquidity screen stands in. Stated, not papered over.

### Code changes

- `features/v3.py` — the shipped scorer: themes, `theme_subscore`,
  `cap_weights` (cap / floor / coverage), `score_frame`, `absolute_floor`,
  `attribution` returning **FACTOR / THEME / VALUE / Z / WEIGHT / CONTRIB /
  LEVEL**, and `BOOK_NOTE` stating plainly that the book is not holdout-tested.
- `features/v3_factors.py` — all 22 factors, computed exactly as researched.
- `features/pit_fundamentals.py` — the disclosure-date as-of join.
- `data/instruments.py` — non-equity exclusion by scheme pattern plus a
  volatility backstop, applied only to symbols absent from the equity master.
  183 excluded; the real gold-**jewellery** equities SKYGOLD, GOLDIAM and
  SILVERTUC are kept, which is what the backstop's 504-session window and
  250-observation minimum were calibrated to protect.
- `stage4_core_score.py` — ranking source `v3_composite` and `build_v3_block`.
  It raises `RankingUnavailable` rather than silently falling back to v2: a run
  that quietly scores with a different model than the config names is worse than
  a run that fails.
- `stage8_final_signal.py` — the card now shows the **theme** line and the
  factors beneath it, and gates entries on the absolute floor with a
  `floor_blocked` count in the funnel.
- `v3_monitor.py` — rolling IC **per factor and per theme**, each theme's share
  of realised cross-sectional spread, and a −25% drawdown flag. All **flag**;
  none disable. **Split by what each one needs to know.** Rolling IC needs
  forward outcomes, so it cannot say anything about today and stays quarterly.
  Theme dominance needs none — whether one theme is doing most of the
  separating between names is a property of today's scores — so it runs on
  **every** run and its flag lands in the run notes. The drawdown flag reads
  closed trades, which lags an open book: it is a *floor* on the drawdown, not
  an estimate, and every line it prints says so. It stays silent below 20
  closed trades, because "0%, inside the flag" about an untested book is a
  reassurance rather than a measurement.
- `validation/v3_panel.py` + `prosignal research v3 --monitor --recheck` — the
  quarterly re-check, same discipline, and it **withholds a verdict** until the
  window holds the 8 independent 21-session windows the deploy was judged on.

### One monitoring defect fixed before it ever ran

The theme-dominance alarm compared a **variance** share against a threshold set
in the units of the linear 40% weight cap. Variance is quadratic in the weight,
so at the shipped configuration momentum reads w²/Σw² = **62%** while carrying
40% — the alarm would have fired on a perfectly healthy book every day from the
first run, and a monitor that always fires is a monitor that gets turned off.
Now measured as a **dispersion** share, which reads back the declared weight
when themes are equally dispersed, with a second rule that catches a *small*
theme over-running (quality could double its influence without approaching an
absolute 55%). Pinned by `tests/test_v3_monitor.py`.

### Tried after the deploy: one rejected, one shipped as a disclosure

Both sealed windows are spent, so neither could be validated. One is a screen
under the same bar the other 93 factors faced; the other predicts no return and
so spends no evidence. Full tables in `research/V3_SEARCH.md` SS13.

- **Dividend yield: built, screened, REJECTED.** The value theme is empty
  because balance-sheet history starts in 2023, but a dividend needs only a
  price and a payment record and that runs dense from 2017. Built PIT-correct,
  split-invariant, 29.5% coverage — and it **fails the placebo screen at every
  horizon**. At h=63 the naive t is −5.45, which reads as a strong factor, but a
  payout policy barely moves quarter to quarter so persistence alone produces
  |t| above **13.69** in five percent of placebo alignments. Value stays empty
  for a measured reason now rather than a data one.
- **Earnings-gap risk: measured and now on the card.** The engine sizes off an
  ATR stop and prints the result as the risk. A stop is a level, not a fill.
  Measured on 179 names with a real calendar over 246,437 sessions, each name
  against **itself** outside its earnings windows: an earnings window carries
  **1.79× the daily volatility** and **4.94× the chance of an overnight gap
  worse than −5%** (0.96% against 0.20%). The card now says when a name reports
  and that the printed risk is a floor on the loss, not a cap. A **disclosure,
  not a gate** — gating would change a traded number.
  - The obvious comparison is wrong and flattered the answer threefold:
    earnings sessions against *all* sessions in the store gives 1.6×, because
    the names with calendars are large caps calmer than the universe around
    them. `work/v3/earnings_gap.py` controls for it.

### What the engine is actually running on, measured

| input | coverage of the live 750 |
|---|---|
| prices, delivery | ~100% |
| **sector map** | **61.2%** |
| PIT fundamentals | 25.6% |
| dividend history | 24.1% |
| earnings calendar — history / forward | 99.7% / 23.1% |

**The sector-neutral rank is sector-neutral for barely half the book.** 291 names
have no sector, and with the six sectors below the 12-name minimum folded in,
**329 — 43.9% of the universe — rank inside one mixed residual bucket**. Not a
regression: the research used the same map, so the holdout numbers include it.
But it bounds what the sector neutrality buys, and it had never been measured.
`equity_master` has no industry column, so this store cannot improve it.

### Read this before trusting the shortlist

Same caveat as v2, and it has not gone away: the ordering *within* the top ten
was not better than the ranking as a whole. Read the output as a shortlist drawn
from an evidenced ranking, not as an ordering — and note that the ranking is
what the holdouts validated, while the ten-name book's excess sits barely
outside a permuted-label null whose quintile spread sits **6.4 standard
deviations** outside it.

---

## v2 signal engine — 2026-08-29

Full search record in `research/V2_SEARCH.md`; search code in `work/v2/`;
result tables and the sealed artefacts in `research/v2/`.

### The holdout that earned the deploy

Sealed **before any model was fitted**: 2025-03-06 → 2026-08-17, 72 signal
dates, 54,000 rows, sha256 recorded in `research/v2/SEAL.json`, with an 84-session
purge and embargo between it and training. Opened **once**, against a frozen
configuration whose sha256 was written before the evaluation ran.

| | validation | **sealed holdout** |
|---|---|---|
| Net annualised / benchmark | 52.2% / 29.7% | **17.7% / 15.3%** |
| Excess, net of realistic costs | +22.5% | **+2.4%** |
| Information ratio / Sharpe | 1.15 / 1.71 | **−0.19 / 1.07** |
| Max drawdown | −38.0% | **−14.0%** |
| Rank IC (t) | 0.049 (6.0) | **0.045 (2.59)** |
| Quintile spread per 42 sessions (t) | +1.85% (6.05) | **+1.65% (2.56)** |
| Top-10 excess per period (t) | +2.05% (3.82) | **−1.39% (−2.16)** |

**The ranking held; the ten-name book did not.** Rank IC and quintile spread
kept their training magnitude on unseen data and sit outside a 200-draw
shuffled null at p = 0.00. The book's +2.4% is inside noise — its information
ratio is negative — and the top ten names underperformed the cross-section they
came from. A permuted-label power test run *before* the holdout predicted this:
a ten-name book's five-year excess has a null standard deviation of ±11.7
points, so the validation figure was mostly luck on a real but smaller edge.
Nothing was re-tuned after the number was seen, and `work/v2/holdout.py`
refuses to run twice.

### The shipped configuration

- Universe: point-in-time liquidity screen, top **750** by 60-session median
  ADTV, ≥₹5cr ADTV, ≥₹20 quoted price, ≥300 sessions listed.
- Score: ten sector-neutral cross-sectional factor ranks, sign-oriented, equal
  weight. `src/prosignal/features/v2.py`.
- Book: 10 slots, buy at rank ≤15, hold while ≤25, weekly refresh, max 3 per
  sector, equal weight, filled at the next session's VWAP. Median hold 20
  sessions.
- Entry gate: computed, reported, **not applied**. Drawdown breaker **flags**.

### What got tried, and what got cut

- **69 candidate factors built** across momentum (multiple horizons, residual,
  vol-adjusted, intraday/overnight, frog-in-the-pan), reversal, volatility,
  drawdown, liquidity, size, NSE delivery, trend and Heston–Sadka seasonality.
- **Screen is a placebo alignment, not a shuffle** — factor cross-section at *t*
  against labels at *t+k* for |k| ≥ 1 year, which reproduces the overlap
  inflation a naive t hides. 35 of 69 never cleared it at any horizon.
- **Cut: `resid_mom`, `mom_12_1`, `beta_120`, `beta_252`, `amihud`, `log_adtv`,
  `idio_vol`, `ma_50_200`, `ulcer_252`, `max_dd_252`, `trend_r2`, seasonality**
  — every one has a large naive t and a placebo distribution as large or larger.
  Persistence, not prediction.
- **Cut: value and quality (5+6 factors).** PIT fundamentals cover 22–28% of the
  panel and the newest filing date in the store is 2025-03-11; by the end of the
  holdout the median name's filing is 453 days old. Data wall, measured.
- **Cut: ownership structure (promoter holding, pledge, FII/DII).** The
  reference tables are header rows and the NSE JSON API is unreachable here.
  Delivery — `deliv_z_21` — is the surviving India-specific ownership proxy and
  clears the null at every horizon.
- **Cut: gradient boosting** (XGBoost, depths 2/3/4). Defensible pooled IC,
  **negative** top-decile excess at every depth — it ranks the middle and gets
  the tail wrong, and the tail is the whole book.
- **Cut: ridge / lasso / elastic net / PCR as the combiner.** All beaten by a
  sign-oriented equal-weight composite once the validation window was extended
  back through the 2020 crash. Ridge's apparent edge was a window artefact.
- **Cut: market-timing entry gates** (200 DMA, calm volatility, breadth,
  bottom-up 50 DMA). Each cost 8–13 points of annual excess and none reduced the
  drawdown it exists to avoid, including through 2020.
- **Cut: NIFTY-100- and NIFTY-200-sized universes.** Clearly worse than 500/750
  across the whole 48,384-configuration book sweep.
- **Kept, and it is the one fitted parameter:** each factor's sign, identical in
  all eight walk-forward folds.

### Code changes

- `features/v2.py` — the shipped scorer. Carries its own factor definitions
  rather than borrowing `crosssec`'s: several share a name and are not the same
  construction (`crosssec.mom_6_1` spans 126 sessions, v2's spans 105), and a
  scorer that earned a holdout number has to compute what it was measured
  computing. A parity harness checks all ten against the search code to machine
  precision on five dates spanning 2020–2026.
- `stage4_core_score.py` — new ranking source `v2_composite`; `build_v2_block`
  reads delivery from its own table because `prices.deliv_pct` is empty for the
  whole store, which would have made `deliv_z_21` silently neutral on every run.
- `contracts.py`, `rundetail.py` — `FactorScore.contribution` serialised, so the
  per-stock table is FACTOR / VALUE / Z / WEIGHT / CONTRIB end to end. UI
  contract unchanged: one button, `POST /analysis/run`, no computation client-side.
- `stage8_final_signal.py` — the card now prints each factor's **own** sector
  percentile and contribution. It printed the composite percentile on every
  line, so all ten factors read "98th" whatever they measured; and it formatted
  ratios as percentages, rendering a vol-adjusted return of 33.6 as "+3363.09%".
- `v2_monitor.py` — rolling per-factor IC against each shipped sign, and a
  drawdown circuit breaker at −15% (deepest holdout drawdown was −14.0%). Both
  **flag**; neither disables. A monitor that switches a factor off changes the
  model without a decision being taken.
- `validation/v2_panel.py` + `prosignal research v2 --recheck` +
  `scripts/quarterly_recheck.sh` — the quarterly re-check, same holdout
  discipline. It **withholds a verdict** until it has the 8 independent
  42-session windows the deploy itself was judged on (~68 signal dates, roughly
  six quarters), and says so rather than issuing a pass/fail on 1.5 windows.
- `tests/test_v2_score.py`, `tests/test_v2_monitor.py` — 16 new tests, including
  a lookahead test that appends a violently different future and requires today's
  values not to move.

### Pre-existing failures fixed along the way

- `exits.rules_from_config` — the exit-hierarchy test asserted `(True, True,
  True)`; the config deliberately switches the 3R target and thesis invalidation
  off with measured ablations beside them. Test corrected to the config's actual
  state, with the reason recorded.
- `harness.deflated` — the DSR guard asserted `n_observations` equalled the
  strict sub-sample. It uses every distinct date and deflates the *count* by the
  analytic overlap inflation instead. Assertion moved onto `effective_n`, which
  is the quantity that carries the correction.
- `portfolio_sim.phase_summary` — `max_drawdown` had been silently rebound from
  the mean across phase offsets to the pooled-path figure, changing what earlier
  write-ups refer to. The pooled figure now has its own key,
  `pooled_path_drawdown`; `max_drawdown` keeps its published meaning and
  `worst_schedule_drawdown` stays the headline.

### Read this before trusting the shortlist

The ordering between #1 and #10 was **not** better than the ranking as a whole
over the holdout. Read the output as a shortlist drawn from an evidenced
ranking, not as an ordering. The card says this on every run.
