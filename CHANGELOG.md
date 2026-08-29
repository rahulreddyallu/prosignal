# CHANGELOG

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
