# Forensic Code Review — 2026-08-18

Every conclusion cites the file, function or command that proves it. Where I
could not establish something, it says UNVERIFIED.

Reviewed at `v2-fundamentals-value-quality`. **327 tests passing, 1 skipped.**

---

## 1. ACTUAL EXECUTION FLOW

Traced by following imports, not documentation.

```
static/index.html  (button)
   -> POST /admin/bootstrap ......... api.py:_bootstrap_runner  (if store empty)
   -> POST /analysis/run ............ api.py:start_analysis
        -> jobs.py:JobManager.start(kind="analysis")   [single-flight]
             -> pipeline.py:run_analysis
                  Stage 0  _universe / _manifest_from_store
                  Stage 1  stage1_data_quality.run    -> DataQualityReport
                  Stage 2  stage2_regime.run          -> RegimeState
                  Stage 3  stage3_eligibility.run     -> EligibilityReport
                  Stage 4  stage4_core_score.run      -> CoreScoreReport
                  Stage 5  stage5_false_signal.run    -> FalseSignalReport
                  Stage 6  stage6_entry.run           -> EntryReport
                  Stage 7  stage7_risk.build_plan     -> RiskPlan (per name)
                  Stage 8  stage8_final_signal.run    -> BUY / WATCH / NO TRADE
                  ledger.py:Ledger.append             [fatal on failure]
   -> GET /analysis/{id} ............ poll
   -> GET /analysis/{id}/results .... render
```

Nothing in this path is mocked, stubbed or hardcoded. Verified by executing it.

---

## 2. THE EXACT BUY FORMULA

Extracted from `stage4_core_score.py` and `stage8_final_signal.py`, not paraphrased.

**Stage 4 — composite score**

For each factor *f* ∈ {momentum_12_1, value, quality, sector_relative_strength}:

```
x_f  = winsorise(raw_f, 2%, 98%)              # crosssection.py:winsorise
z_f  = standardise(x_f, method="rank")        # config: standardisation=rank
w_f  = base_weight_f x regime_multiplier_f    # stage2 multipliers
w_f  = w_f / sum(w)                           # renormalised over SURVIVING factors

composite_raw  = sum_f ( z_f.fillna(median(z_f)) x w_f )
composite_unit = rank_to_unit_interval(composite_raw)   # -> [0,1]
```

**Stage 8 — gates, in order.** A candidate must clear every one:

```
1. regime.allow_new_entries          else NO TRADE (market-wide)
2. not defense.market_halt           else NO TRADE (market-wide)
3. defense.final_status != REJECTED
4. composite_after_penalty >= 0.60   (scarcity.min_composite_score)
5. percentile >= 90                  (scarcity.min_universe_percentile)
6. entry trigger fired               else WATCHLIST, never BUY
7. sector count < 2                  (portfolio.max_signals_per_sector)
8. max pairwise corr <= 0.70         (portfolio.max_pairwise_correlation)
9. len(buys) < 5                     (portfolio.max_signals_per_run)
```

Current live weights, all four factors active: **0.25 each**.

---

## 3. CRITICAL DEFECTS

### P0-1 — The top-ranked name is scored on data it does not have

`stage4_core_score.py`, composite line:
```python
composite_raw = sum(frame[n].fillna(frame[n].median()) * w for n, w in effective.items())
```

Measured on the live universe today:

| Factor | Median-filled | Weight |
|---|---|---|
| momentum_12_1 | 0/19 (0%) | 0.25 |
| sector_relative_strength | 0/19 (0%) | 0.25 |
| **value** | **7/19 (37%)** | **0.25** |
| quality | 0/19 (0%) | 0.25 |

**The #1 ranked name, GVT&D at composite 1.000, carries an imputed value
score.** So do #4 PREMIERENE and #5 HYUNDAI. Those names are ranked partly on a
number that was never computed for them, and imputing the median makes a
missing factor look average rather than unknown.

The 60% coverage floor (`_MIN_FACTOR_COVERAGE`) gates the *factor*, not the
*name*: value cleared at 63% coverage, so it applies to all 19 names including
the 7 with no data.

**Fix:** rank each name only on factors it actually has, renormalising weights
per name; or exclude names below a per-name coverage floor. Do not median-fill
into a ranking that gets presented as a BUY.

### P0-2 — Undated filings crashed the pipeline (FIXED this review)

`features/fundamentals.py:point_in_time_snapshot` compared an all-NaT
`datetime64` column against a `datetime.date`, raising
`TypeError: Invalid comparison`. A provider returning rows without filing dates
would have taken the whole run down rather than degrading to "factor
unavailable". Found by a test written during this review; fixed and pinned by
`test_rows_without_a_filing_date_are_dropped`.

---

## 4. UPSTOX AUDIT

```
grep -rin "upstox" src config tests   ->  0
grep -rin "instrument_key" src config ->  0
```

**There is no Upstox integration.** Not outdated, not misconfigured — absent.
Every price comes from NSE archives (`data/providers/nse_archives.py`) and every
fundamental from NSE JSON + XBRL (`data/providers/nse_fundamentals.py`).

Sections of the brief covering instrument_key, WebSocket, quotes and rate-limit
risk have no subject matter here. Per `RESEARCH.md`, this is the right call for
an EOD system: NSE archives are authoritative, free, need no auth, and Upstox's
fundamentals carry no filing date.

---

## 5. INDICATOR AUDIT

| Indicator | Present? | Evidence |
|---|---|---|
| SMA, EMA, ATR, returns, realised volatility | **Yes** | verified against reference implementations: `true_range` 0.00e+00, `sma` 0.00e+00, `ema` 2.8e-14, `momentum_12_1` 0.00e+00, Wilder ATR tail 0.03% |
| RSI, MACD, Bollinger, ADX, Supertrend, Stochastic, VWAP | **ABSENT** | `dir(prosignal.indicators)` |

Their absence is deliberate (`RESEARCH.md` §2): no peer-reviewed evidence they
add information *beyond* momentum, and Stage 4 measures that redundancy rather
than assuming it.

**No duplicate implementations.** `sma`/`sma_atr` and `atr`/`atr_pct_of_price`
are distinct functions, not competing copies.

---

## 6. PROBABILITY AUDIT

**The system emits no probability, anywhere.** Enforced, not merely intended:
`test_pipeline_stages.py::test_engine_never_emits_a_probability` walks the
serialised output and fails on any field name implying a likelihood.

`composite_score` is a **cross-sectional rank in [0,1]** — `rank_to_unit_interval`
— and every card says so verbatim.

This is the correct call: nothing has been calibrated against realised outcomes,
so any percentage would be a weighted score in a statistical costume.

---

## 7. BACKTEST AUDIT — is the methodology trustworthy?

`backtest.py:_simulate`.

| Property | Implementation | Verdict |
|---|---|---|
| Execution price | **Next session's OPEN** | Correct. Filling at the signal close would grant a session of foresight |
| Stop + target on one bar | **Stop wins** | Correct pessimism; daily bars cannot reveal intraday order |
| Costs | Same `CostModel` as live | Correct |
| Look-ahead | Instrumented test with 278 sessions of future data in the store — **zero violations** | Verified |
| Survivorship | **NOT handled** | Current NIFTY 200 applied historically |
| Overlapping trades | Not modelled as a portfolio | Trade-level stats only; no capital constraint across concurrent positions |

**Trustworthy on execution realism; NOT trustworthy on survivorship or portfolio
accounting.** Both failure modes flatter results.

---

## 8. LEAKAGE / TIMESTAMP AUDIT

| Check | Result |
|---|---|
| Indicator lookahead | Verified absent — truncation-invariance tests |
| Pipeline lookahead | Verified absent — instrumented, 278 future sessions present |
| Fundamentals | Gated on `filing_date <= as_of`; measured lag 9–45 days |
| Forward-fill | None anywhere in `store.py` |

**This is the strongest area of the codebase.**

---

## 9. ERROR HANDLING

```
bare `except:`            0
`except Exception: pass`  0
```

All broad handlers record the error: `nse_fundamentals.py` (2), `api.py` (2),
`jobs.py` (1). `jobs.py:_execute` stores the full traceback and marks FAILED;
`api.py:job_results` returns **409** for a failed job — never a signal.
`ledger.py:append` is **fatal on failure**, because a run that is not recorded
must not be counted as evidence.

**Errors cannot become BUY signals.**

---

## 10. TEST AUDIT — what do they prove?

327 passing. The ones that carry weight:

| Test | Proves |
|---|---|
| `test_indicator_does_not_peek_into_the_future` | truncation invariance across 8 indicators |
| `test_engine_never_emits_a_probability` | no field can imply calibration |
| `test_double_click_does_not_launch_two_analyses` | single-flight |
| `test_stale_running_job_is_reaped` | a crash cannot block the button forever |
| `test_bar_touching_both_stop_and_target_resolves_as_the_stop` | backtest pessimism |
| `test_income_does_not_collide_with_other_income` | XBRL substring trap |

**Gap closed this review:** `nse_fundamentals.py` had **zero** coverage.
`tests/test_fundamentals.py` (22 tests) now covers namespace variance, Indian
comma grouping, negatives, malformed documents, and the PIT gate.

**Remaining untested:** `yfinance_provider.py`.

---

## 11. CONFIGURATION

**182 tunables, 23 unconsumed (13%)** — down from 57% before Stages 3–8 existed.
Residue is in `stage4_core_score` (6), `stage5_false_signal` (5),
`stage3_eligibility` (4), `validation` (4).

`extra="forbid"` throughout: a typo cannot fall back to a hidden default.

---

## 12. VERDICT BY CATEGORY

**VERIFIED** — indicator mathematics; no look-ahead at indicator or pipeline
level; no forward-fill; execution realism in the backtest; single-flight job
control; stale-job reaping; ledger persistence; error handling; the refusal to
emit probabilities; NSE data provenance.

**PARTIALLY VERIFIED** — value/quality factors (correct where data exists, 37%
median-filled where it does not); backtest (execution honest, survivorship and
portfolio accounting absent).

**UNSUPPORTED** — any claim of predictive edge. Two walk-forwards, DSR 0.7% and
0.2% against a 95% bar.

**DEAD** — nothing meaningful. `validation/` is unwired but is the harness the
next phase needs, and is exercised by 28 tests.

**MISSING** — point-in-time index membership; balance-sheet fundamentals (ROE,
book-to-price); PBO; bear-regime data; portfolio-level backtest accounting.

---

## 13. PRIORITISED CHANGES

**P0 — before trusting any output**
1. Stop median-filling factors into a presented ranking (§3, P0-1). The current
   #1 pick is affected.

**P1 — materially affects signal quality**
2. Deeper fundamentals — 8 quarters cannot support a 3-year backtest.
3. Point-in-time index membership, or quantify the survivorship bias.
4. Portfolio-level backtest accounting (concurrent positions, capital).

**P2 — reliability**
5. Tests for `yfinance_provider.py`.
6. Remove or justify the 23 unconsumed parameters.

**P3**
7. PBO across a configuration sweep — only after (2) and (3), or it measures noise.

---

## 14. WHAT SHOULD BE DELETED

Very little, because earlier audits already removed the derivatives feed, the
unobtainable factor and the unused dependencies.

**Candidate:** `sector_relative_strength`. Measured correlation to momentum is
**+0.375**, and it is a momentum transform carrying 25% weight. Contrast value
at **−0.545** — genuinely orthogonal. Dropping sector-RS would reduce the model
by one correlated dimension without losing independent information.

I am **not** doing it in this review: that is a factor-model change, and
changing the model in response to a correlation observed on one date is exactly
the search behaviour the DSR discipline forbids. It belongs in a deliberate
research pass with the trial counted.

## 15. WHAT SHOULD BE PRESERVED

The look-ahead protections and their tests. The point-in-time fundamentals gate.
The cost model. The ledger's fatal-on-failure semantics. The refusal to emit
probabilities. The NO-TRADE discipline. The `extra="forbid"` config layer.

These are the parts that make the negative result *credible* — which is
currently the most valuable thing this codebase produces.
