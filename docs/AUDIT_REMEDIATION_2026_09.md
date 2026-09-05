# Quant-audit remediation — 2026-09

> **Base:** rebased onto `main` after PR #94 ("consolidate v3-v9 / new epoch")
> landed. Verified #94 did **not** touch the v3 core — `features/v3.py` THEMES,
> signs and weights are unchanged — so every finding and result below still
> describes the shipped scorer. The experiment result JSONs were computed on the
> identical v3 code and remain valid.

End-to-end tracker for the adversarial quant review. Every item maps a finding to
a concrete action, a type, and a status. **The governing rule:** the shipped
model's signs and weights are frozen in `features/v3.py`, hashed into
`config_sha256`, matched by `tests/test_v3_score.py`, and tied to two sealed
holdouts. Fixes therefore split into:

- **SAFE (merge now):** presentation, docs, and experiment harnesses. Never
  changes what the engine recommends BUY.
- **EPOCH-GATED:** anything that moves a sign, weight, factor set, or the traded
  book. These do not ship until an experiment justifies them AND the frozen
  pipeline is re-run and re-sealed (a new epoch). Hand-editing a frozen number
  to "fix" it silently ships an unvalidated model — the exact failure the repo
  and `MEMORY.md` ("production is not the validated model") warn against.

## Priority ladder

| # | Finding (audit §) | Action | Type | Status |
|---|---|---|---|---|
| P0-1 | "quality" shown as *Business quality* but fitted anti-quality; "ownership" ≠ ownership (§D) | Honest card labels + config sign-warning | SAFE | ✅ merged (`fae96e2`) |
| P0-2 | Book selected on holdout-overlapping data; only clean OOS book test lost (§I, K-1) | `recheck_status.py` (forward-gated status) + document that clean K-1 is forward-only | SAFE harness | ✅ harness; ⏳ verdict forward-gated |
| P1-3 | Blend may not beat momentum-alone (§K-2) | `audit_2026_09.py` EXP-C (IC-level precursor) | SAFE harness | ✅ run — blend wins at IC level; book K-2 open |
| P1-4 | Perf-proportional weights overfit / anti-OOS (§H, #4) | EXP-D equal- vs frozen-weight, same sub-scores | SAFE harness | ✅ run — frozen wins; not supported |
| P1-5 | "quality" sign may be a 2020–21 artifact (§K-4) | EXP-A sign-stability across halves/thirds | SAFE harness | ✅ run — **stable, REFUTED**; keep+relabel |
| P1-6 | Uncalibrated impact coeff gates net-of-cost (§H, K-6) | `cost_sensitivity.py` — exact cost model on real selected names, swept | SAFE harness | ✅ cost side done; full book P&L still gated |
| P2-7 | Delivery may be a liquidity/vol proxy (§K-3) | EXP-B incremental IC after controls | SAFE harness | ✅ run — **not a proxy, REFUTED** |
| P2-8 | Residual panel survivorship inflates OOS IC (§E, K-5) | `survivorship_bound.py` — stress delisting-within-horizon rows | SAFE harness | ✅ run — inflation ~2.4% of IC, **negligible** |
| P2-11 | Book underperforms the universe net of cost (§I, K-1/K-2) | `book_sim.py` — repo's own simulator on v3 rankings | SAFE harness | ✅ run — book loses to EW universe everywhere (sign robust) |
| BUG | `research portfolio` crashes: `cfg.params.stage1_universe` | fix to `cfg.params.universe` in `cli._portfolio_inputs` | code fix | ✅ fixed |
| P2-9 | Momentum factor theatre — prox/voladj dupes (§D, F) | Prune experiment → new epoch if it holds | EPOCH-gated | ⏳ after EXP results |
| P3-10 | Doc contradictions / withdrawn-number consistency | Config note done; README/CHANGELOG sweep | SAFE | 🔄 partial |

## What ships in this round (safe, merged)

1. **Honest theme labels** (`presentation/viewmodel.py`): `quality → "Low-margin
   tilt"`, `ownership → "Delivery strength"`. Keys/weights/signs untouched;
   `test_v3_score.py` still green.
2. **Config sign-warning** on the `quality` factor block.
3. **Runnable experiment harnesses** under `research/v3/experiments/`:
   - `audit_2026_09.py` — EXP-A (sign stability), EXP-B (delivery incremental),
     EXP-C (blend vs single theme), EXP-D (equal vs frozen weights).
   - `recheck_status.py` — K-1 forward-gated ranking status via the shipped
     `recheck()`.

## Results — `audit_2026_09.py`  (run 2026-09-03)

Panel: 199,823 rows, **364 signal dates, 2019-03-19 .. 2026-07-30**, label `y21`.
**Caveat that governs every number here:** signs/weights were fit on 2018-2024,
so full-window IC is **partly in-sample**. Sign STABILITY across sub-periods and
the liquidity/vol control test are the trustworthy reads; IC *levels* are not
out-of-sample. The window does extend ~21 months past the 2024-10 fit end.

**These results OVERTURN three of the audit's own hypotheses. Recorded as such —
a reviewer updates on evidence, and the point of gating model edits behind
experiments is to catch exactly this before deleting a working factor.**

| Exp | Question | Result | Verdict vs audit |
|---|---|---|---|
| EXP-A | Is the inverted "quality" sign a 2020-21 artifact? | quality IC **+0.0374 (t 6.34)**, no sign flip across halves or thirds; net_margin +0.040/+0.029 (no flip), margin_stability +0.009/+0.021 (no flip) | **REFUTED.** Sign is inverted vs the namesake but empirically **stable**. Do NOT drop it. Relabel only (done). |
| EXP-B | Is delivery just a liquidity/vol proxy? | delivery IC raw **t 9.14**; residualised on ADTV+downside-vol **t 9.83** (unchanged) | **REFUTED.** Delivery is real and price-orthogonal, not a liquidity proxy. |
| EXP-B′ | Does delivery add in a JOINT all-theme FM? | partial slope **t 0.36** (momentum t 3.38, quality t 6.12 are the only jointly-significant themes; risk −0.55, reversal 0.80) | Nuance: univariately strong, but partly spanned by momentum+quality at the return level. Worth a cleaner incremental test; not a reason to cut it. |
| EXP-C | Does the 5-theme blend beat the best single theme? | composite spread **+0.0151 (t 6.85)** > best single (quality +0.0116; momentum +0.0112) | **REFUTED (IC level).** The blend adds ~35% over the best single theme. (Net-of-cost book K-2 still open.) |
| EXP-D | Does equal weight beat the frozen perf-weights? | frozen spread **+0.0151** vs equal **+0.0122**; equal has marginally higher IC (0.0609 vs 0.0580) | **Not supported.** Frozen weights win on the headline spread. Finding #4 does not hold at the ranking level. |

### What this means for the model-level items

- **"Quality" / low-margin theme:** downgraded from *INVALID UNTIL FIXED* to
  **KEEP + relabel + monitor**. The sign is stable; the honesty fix (label) is
  the correct and sufficient action. `v3_monitor.review_themes` already flags an
  inversion if it ever starts.
- **Delivery theme:** **KEEP.** Genuine, price-orthogonal, non-liquidity signal.
- **Equal-weight switch:** **do not make it.** No OOS-justified benefit.
- **Blend vs momentum-alone:** the ensemble earns its place at the ranking
  level; the remaining open question is net-of-cost at 6 names (K-2 book).

### K-1 status (`recheck_status.py`, run 2026-09-03)

Verdict **HOLDS** on the last 18 months (rank IC +0.0542 t 4.13, quintile spread
+1.25% t 3.56, null p 0.000), no theme inverted. **But the runner reports the
window is 94% inside sealed holdout A** — so this is a re-reading of the dates
the deploy was already judged on, not independent evidence. The clean K-1
(traded book on dates that never touched the 378-cell surface) is **forward-only**
and accrues as the live window moves past 2026-08. This is the honest status, not
a pass.

### K-6 cost sensitivity (`cost_sensitivity.py`, run 2026-09-03)

Prices the EXACT shipped cost model on the real names the v3 score ranks top-6
(median ADTV ₹34cr; ₹1.67L/position at ₹10L, 6 slots). Round-trip cost and the
annual drag it implies at 34 trades/yr:

| impact coeff | round-trip (median) | annual drag | vs window-A gross (+2.2%) | vs window-B gross (+15.6%) |
|---|---|---|---|---|
| 0.10 (shipped) | 80 bps | 4.5% | **net −2.3%** | net +11.1% |
| 0.25 (2.5×) | 146 bps | 8.3% | net −6.1% | net +7.3% |
| 0.50 (5×) | 256 bps | 14.5% | net −12.3% | net +1.1% |

**The cost question is really a gross-edge question.** At the shipped coefficient
the 6-name book's drag is ~4.5%/yr — below the 10-name/weekly holdout's 9.5%
(slower cadence, as `BOOK_NOTE` predicts), but still **more than window A's entire
gross edge**. Break-even vs window A needs a coefficient of ~0.006 (1/16th the
shipped 0.10); window B survives even at 5×. **The two sealed windows disagree ~7×
on the gross edge, so the net-of-cost verdict is undetermined until the forward
test settles which gross number is real.** (Caveat: uses the 10-name book's gross
as a proxy for the 6-name book's; it is a cost-side bound, not a re-simulated P&L.)

### Book P&L (`book_sim.py`, run 2026-09-03)

The v3 rankings (+ entries-only absolute floor) run through the repository's OWN
`simulate`/`phase_summary` with the shipped cost model, swept over book size,
window and impact coefficient:

| window | book | excess/yr | IR | gross/yr | maxDD |
|---|---|---|---|---|---|
| A 2025-26 | live 6 | −9.6% | −0.79 | −9.2% | −16.8% |
| A 2025-26 | holdout 10 | −8.5% | −0.80 | −7.9% | −24.3% |
| B 2021-22 | live 6 | −3.6% | −0.31 | −3.3% | −12.2% |
| full | live 6 | −17.6% | −0.83 | −17.3% | −27.2% |

And every single theme's 6-name book (full window) is negative too: momentum
−16.8%, quality −18.6%, ownership −19.9%, risk −18.3%, reversal −20.4%.

**The book underperforms the equal-weight eligible universe in every window, at
every book size, at every cost level, and for every single theme — GROSS as well
as net.** This corroborates the repo's own `DEPLOY_REFERENCE` (−2.83%) and
`HOLDOUT_A` (−7.25%) and the audit's central thesis: **the ranking has IC; the
concentrated long-only book does not beat the universe it selects from.** The
gap is selection + the 200-DMA floor + stops sitting in cash through the
2019-21 small-cap rally, not costs (cost drag here is 0.3-1.6%/yr at this low
cohort turnover). **Caveat:** repo cohort model (63-session hold), NOT the sealed
weekly book, so magnitudes are not comparable to the holdout; windows overlap the
378-cell selection surface (in-sample). Read the SIGN (robust, corroborated), not
the magnitude.

### K-5 survivorship bound (`survivorship_bound.py`, run 2026-09-03)

The panel INCLUDES 841 of 3,552 names (23.7%) that stopped printing — it is
survivorship-free for the collection period. Only 438 rows (0.22%) have a name
delisting within the 21-session horizon; stressing them at −30% moves composite
IC +0.0580 → +0.0566 (**2.4% inflation, negligible**). Unquantifiable residual:
names that delisted before data collection are absent from the store entirely.

### What remains genuinely unresolved

- **K-1 clean-window book** — forward-gated (`recheck_status.py`); only data past
  2026-08 clears the 378-cell-surface overlap.
- **DSR failure** (0.030 / 0.97) and **holdout-overlap book selection** — about
  the traded book and multiple testing, not the ranking; they stand.
- **Gross-edge instability across windows** — the single biggest open question;
  only forward data resolves it.
- **The book-vs-universe gap** — `book_sim` shows the concentrated book losing to
  the EW universe gross; whether that is the 200-DMA floor, the concentration, or
  the cohort-model artifact is worth isolating (the floor is the prime suspect).

## Epoch-gated decisions (do NOT hand-edit; require a re-fit + re-seal)

- **Drop / re-sign "quality"** — only if EXP-A shows the negative sign is a
  sub-period artifact (sign flips across thirds).
- **Move to equal theme weights** — only if EXP-D shows equal ≥ frozen OOS.
- **Prune momentum to ~4 representatives** — cosmetic to weight (two-level cap
  already contains it); do it in the same epoch as any of the above, never alone.
- **6-name book concentration** — do not re-baseline off the 378-cell surface;
  let the forward test / `recheck` grade it on non-overlapping data.

## Definition of done (per item)

- SAFE items: merged to `main`, tests green.
- EPOCH-gated items: experiment result recorded here → if it justifies a change,
  open a new epoch (re-run `freeze`, re-seal a holdout, re-register the forward
  test) → only then update `features/v3.py`.

---

# Round 2 — the 2026-09-05 factor audit

A second adversarial pass, this time factor-by-factor, run against the shipped
scorer on the same 364-date panel. Same governing rule: **SAFE ships, EPOCH-GATED
does not.** Everything below was recomputed from
`research/v3/experiments/panel_2026_09.parquet`, and every t-statistic is
**Newey-West lag 4** — the overlap a 21-session label sampled every 5 sessions
implies. Naive t-statistics run ~1.74× higher and are not used for any verdict.

## The central finding

**The ranking is credible. The concentrated book is not, and they had been
reported as one thing.**

| | |
|---|---|
| Composite rank IC | +0.0580, **NW t +4.76**, positive on 69.5% of dates |
| Decile spread D10−D1 | +1.84% per 21 sessions, NW t +4.14 |
| Rank IC, most liquid tercile | +0.041, NW t +2.73 — survives where it must |
| Top-6 gross excess | +6.2%/yr, **NW t +1.22** — not significant |
| Top-6, 2024–2026 only | **+0.13%/yr** |
| Top-6, liquid half only | **+0.5%/yr, NW t +0.11** |

The decile curve is flat D3–D8 and separates only at the ends (D1 +0.52%,
D10 +2.36%). Most of the information is *avoid the bottom two deciles* — a
screen. The six-name book bets on the tail ordering, which is the part the
sealed holdouts supported least (window-A top-ten excess was t 0.81).

## Theme layer — declared weights are not the applied ones

| theme | declared | **effective** | coverage | marginal ΔIC | NW t |
|---|---|---|---|---|---|
| ownership (delivery) | 18.94% | 21.5% | 95.7% | −0.0096 | **−4.07** |
| quality | 18.99% | **4.04%** | 21.2% | −0.0015 | −2.21 |
| risk | 11.09% | 13.3% | 99.7% | −0.0026 | −1.66 |
| reversal | 10.98% | 13.2% | 99.97% | −0.0021 | −1.63 |
| **momentum** | 40.00% | **48.0%** | 99.96% | −0.0030 | **−0.34** |

Momentum holds 48% of the effective weight and its marginal contribution is
statistically zero. Delivery holds 21% and carries three times the value, and it
is *growing*: −0.0056 in 2019-21, −0.0164 in 2024-26. This is consistent with
Chui, Ranganathan, Rohit & Veeraraghavan (2023), who find Indian momentum
concentrates in the most liquid names while reversals dominate illiquid ones —
and this universe has a median ADTV of ₹23 crore.

## Factor layer — three factors are actively harmful

Drop-one, paired by date, population held fixed. Removing these *raises*
composite IC: `ulcer_120` (+0.0012, t +2.81), `voladj_mom_6_1` (+0.0010, t +2.83),
`resid_rev_21` (+0.0006, t +2.65). `mom_2_0` has standalone IC **+0.0001 at
t 0.02** — absent, not weak; it is 2-month momentum with no skip, so the reversal
window sits inside it. It is also 13.80% of `v9r_core`'s weight vector.

Four factors cannot fund their own turnover at the shipped 89.6 bps round trip
(₹1.67L position, ₹23 cr ADTV, impact 60% of the total): `rev_1w` 24.2%/yr,
`deliv_chg_5` 13.9%, `price_vs_vwap_20` 9.7%, `resid_rev_21` 6.8%.

**Seven factors were nominated for removal by an independent split-half
selection in BOTH halves:** `mom_2_0`, `mom_3_1`, `mom_accel`, `voladj_mom_6_1`,
`ulcer_120`, `resid_rev_21`, `deliv_chg_5`.

## SAFE — merged in this round

| # | Finding | Action | Status |
|---|---|---|---|
| R2-1 | Card quoted holdout t-stats for a book with no holdout | Stage 8 `_card` now states what is evidenced (the ranking) and what is not (the shortlist), with the measured numbers | ✅ |
| R2-2 | **`_redundancy` fed `model_features`, which is `None` on every run since the fitted ranker was removed — no shipped v3 factor pair had ever been checked** | `_v3_redundancy` checks theme sub-scores AND the 22 oriented factor ranks; cross-theme pairs above cutoff are BREACHES because the per-theme cap cannot see them | ✅ |
| R2-3 | Card printed declared theme weights, which no name is scored at | `FactorScore.nominal_weight` added; `weight` is now the name's effective weight; `theme_effective_weights()` reports the population average per run | ✅ |
| R2-4 | Sector map replaced on every refresh, so a name dropping out of NIFTY 500 lost its sector permanently | Map now **accumulates**; four more index files added; residual share reported per run | ✅ |
| R2-5 | `store.read_sector_map` docstring claimed the map "never feeds the score" — untrue since v3 shipped | Corrected | ✅ |

`_v3_redundancy` on the 2026-07-30 cross-section returns three breaches, all
cross-theme, all involving `ulcer_120`: `prox_52w` +0.74, `prox_52w_now` +0.68,
`voladj_mom_12_1` +0.64. **Real momentum exposure is 48% plus most of risk's
13%.** Sector neutralisation covers 75.9% of ranked names; the rest sit in one
residual group ranked against itself. (The audit first reported 41.9% from a
stale panel column; the live map gives 24.1%.)

## EPOCH-GATED — measured, not shipped

`research/v3/experiments/epoch_2026_09.py`. Pre-registered specs (sha256
`4e5cc72b…`), evaluated on 45 purged and embargoed CPCV folds, purge = embargo =
5 signal dates, all specs scored on the same folds and the same population.
**The harness refuses to report unless its rebuild reproduces the shipped `score`
exactly** — a first version matched to 5e-3, two ties out of 750, and was
measuring a different model.

| spec | ΔIC | NW t | CPCV folds + | p5 | selected on this panel? |
|---|---|---|---|---|---|
| A/C prune the seven | +0.0057 | +1.99 | **91%** | −0.0012 | yes |
| **B move `ulcer_120` → momentum** | +0.0016 | **+2.39** | 87% | −0.0006 | **no** |
| D remove the four cost-infeasible | −0.0005 | −0.22 | 44% | −0.0055 | no |
| E equal theme weight | +0.0029 | +0.74 | 64% | −0.0087 | no |
| F prune + equal weight | +0.0045 | +0.69 | 60% | −0.0168 | partly |
| *G* move + cost removals *(round 2)* | +0.0006 | +0.24 | 51% | −0.0067 | no |
| *H* prune six + move `ulcer_120` *(round 2)* | +0.0054 | +1.69 | 84% | −0.0035 | partly |

**What CPCV does and does not buy here:** the specs have no fitted parameters, so
nothing is re-estimated per fold and CPCV cannot make a selection
out-of-sample. It measures **stability** — `folds_positive` and `p5` carry
information the full-window delta does not. A/C/F/H additionally inherit the
split-half selection that chose the seven from this panel; **B, D, E and G do
not, which makes B the cleanest read in the table.**

**H answers a question A could not:** pruning the other six and *moving*
`ulcer_120` gives +0.0054 against A's +0.0057. **The prune's gain is not about
`ulcer_120`** — it is the other six.

**Recommended epoch, when one is opened:** B first (highest t, mechanically
motivated, nothing deleted, tiny downside), then the four cost removals as an
explicit execution decision — D shows they cost IC, and the case for them is
turnover, not information. A/C should wait for forward data that did not select
them.

## New factors — the shipped set is not missing an axis

`research/v3/experiments/candidates_2026_09.py`. Six candidates with published
Indian evidence, all PIT, coverage 92–99%:

| candidate | own IC | NW t | ΔIC @10% | NW t |
|---|---|---|---|---|
| `idio_vol_120` | +0.0396 | +2.84 | +0.0036 | +1.71 |
| `beta_120` | +0.0321 | +1.67 | +0.0016 | +0.61 |
| `max_dd_120` | +0.0279 | +2.17 | +0.0001 | +0.04 |
| `idio_skew_120` | +0.0181 | +2.10 | +0.0014 | +0.97 |
| `deliv_pct_252` | +0.0146 | +1.36 | +0.0007 | +0.46 |
| `deliv_val_z_60` | +0.0073 | +0.70 | −0.0014 | −0.97 |

**None adds at conventional significance.** `beta_120` carries a genuine
standalone edge — the beta anomaly is confirmed on NSE 2001-2016 — and adds
nothing because it correlates 0.47 with `downside_vol_60` and 0.41 with
`deliv_pct_60`. `idio_vol_120` is the best of the six and correlates 0.56 with
`downside_vol_60`. Also tested and rejected earlier this round: promoter-holding
change (NW t +1.42 standalone, +0.66 added, coverage from 2022 only), free float,
promoter level, earnings recency.

**PEAD is unmeasurable, not refuted.** `statements` × `results_calendar` yields
1,325 rows across 664 symbols; a standardised unexpected earnings needs four
quarters of lag plus eight for scaling. It is the most promising missing signal
and it needs a real fundamentals feed.

## Verification of the round-2 SAFE changes

**The claim these changes make is that none of them can move a ranking.** That is
not a claim to be reasoned about, so it was measured: the pre-change tree was
extracted with `git archive HEAD src`, and both trees scored the same 600-name
cross-section at 2026-08-21 through `build_v3_block`.

```
HEAD (before):  n=574  sha 6f85515b286df30ccf75c671  sum 0.8807056993627289
working (after) n=574  sha 6f85515b286df30ccf75c671  sum 0.8807056993627289
top-10 identical: E2E, MTARTECH, CPPLUS, TECHM, ADANIENSOL, LIQUIDPLUS,
                  IDEAFORGE, NYKAA, ATHERENERG, LAURUSLABS
```

Byte-identical scores across all 574 scored names.

**The MODEL gate still trips, and it is right to.** `modelprint.MODEL_SOURCES`
hashes `stage4_core_score.py` and `stage8_final_signal.py` **whole**, so any edit
to either moves `model_sources_sha` (`da717d375fc1 -> 8493a537ab6b`) whether or
not it touches a coefficient. That is the conservative choice and should not be
weakened to accommodate this change; the score-equality above is the evidence
that belongs beside it when the epoch decision is taken.

### Test-suite state

`1,635 passed, 6 failed` on the full run, and every failure is accounted for:

| test | cause |
|---|---|
| `test_value_block_coverage` (×2) | **now pass** — failed only because the suite was running while `stage4` was mid-edit |
| `test_api_jobs_ledger::test_run_is_persisted_to_the_ledger` | **now passes** — same |
| `test_remediation_guards::test_adj_factor_without_a_price_column_is_refused` | environment: the test constructs `DataStore(Path("/nonexistent-curated"))` and macOS refuses `mkdir` on a read-only root. Pre-existing, unrelated |
| `test_data_manifest::test_the_shipped_store_is_manifested_and_verifies` | **the store drift of R2-6.** Pre-existing (2026-09-04/05 ingest) |
| `test_restart_gate::test_the_shipped_engine_is_ready_to_be_restarted` | the same store drift, plus the expected `code_sha` / `model_sources_sha` / `config_version` drift from these uncommitted changes. The gate is doing its job |

`tests/test_v3_redundancy_and_weights.py` was added with the fixes: 8 tests
covering orientation before correlation, cross-theme vs within-theme breach
classification, effective-vs-declared weight, and the residual-bucket count.

## R2-6 — the finding that supersedes several others: **the research panel was stale**

Found while verifying the round-2 fixes with a real pipeline run rather than
tests. The run reported the v3 `quality` theme covering **99%** of eligible
names. The panel says **25%**. Both were computed by the same code.

`panel_2026_09.parquet` was built **2026-09-03 20:47**. The fundamentals ingest
wrote `results_calendar.parquet` on **2026-09-04 21:10** and rewrote
`statements.parquet` on **2026-09-05 06:31**, taking it from 253 KB to 917 KB.
Those files feed `pit_fundamentals.build_records` — which returns 9,207 rows
across 758 symbols today — and `build_records` feeds the v3 `quality` theme.

Measured on the live store at 2026-08-21, restricted to the liquid names the
engine actually ranks:

| theme | panel (2026-09-03 store) | **live (current store)** |
|---|---|---|
| momentum | 48.0% eff / 99.96% cov | 41.4% eff / 99.6% cov |
| quality | **4.0% eff / 21.2% cov** | **16.3% eff / 85.4% cov** |
| ownership | 21.5% / 95.7% | 19.5% / 99.0% |
| risk | 13.3% / 99.7% | 11.5% / 99.2% |
| reversal | 13.2% / 99.97% | 11.3% / 99.0% |

The panel's quality coverage is flat at 23–26% from 2021 onward, so this is not
a time trend in filings — it is a store that moved.

**What this invalidates.** Every panel-derived statement about `quality`,
including round 1's EXP-A ("quality IC +0.0374 t 6.34, sign stable, keep it"),
which is *entirely* about that theme and was measured on a version of it that was
three-quarters absent. The composite IC of +0.0580 is also a four-and-a-bit-theme
composite, not the five-theme one that ships. **The audit's headline — ranking
credible, book not — rests on momentum, ownership, risk and the book statistics,
none of which touch fundamentals, so it stands. The quality and effective-weight
numbers do not.**

`prosignal data manifest --verify` detects this and reports 10 discrepancies
including `statements.parquet: changed (253,563 bytes recorded, 916,771 on
disk)`. **The check existed and nobody ran it**, because nothing connected "the
store moved" to "the JSON in this directory describes a model that no longer
exists."

### Fixed

- **`research/v3/experiments/_panel_guard.py`** — the panel now carries a
  `.provenance.json` sidecar recording size and mtime of every curated input it
  reads plus the store manifest digest. `require_fresh()` **refuses to run** a
  harness against a panel the store has moved under; `--allow-stale` is available
  and prints what is stale. Wired into `audit_2026_09.py`, `epoch_2026_09.py` and
  `candidates_2026_09.py`.
- **`research/v3/experiments/build_panel.py`** — rebuilds through
  `validation.v3_panel.build_v3_panel` (no second implementation), stamps
  provenance, and prints per-theme coverage so a coverage collapse is visible at
  build time rather than two audits later.
- **`audit_2026_09.py` import fixed** — it imported `prosignal.validation.v2_panel`,
  removed in the 2026-09-03 cleanup, so **the harness has been unrunnable since
  then** and round 1's EXP-A..D results could not be reproduced by running it.
  Both functions live in `validation.metrics`.

### Still open, and it is the user's call

`data manifest --verify` reports the store as DRIFTED and `research epoch status`
reports the open epoch `2026-09-03-5e0c98515d13e6e2` as drifted on `code_sha`,
`model_sources_sha` and `config_version`. Refreshing the manifest and deciding
whether the fundamentals expansion warrants a new epoch are operational
decisions, deliberately not taken here — `epoch.py` says it plainly: *"`drifted_from`
reports; a person opens."*

## What remains unresolved after round 2

- **The book, still.** Nothing here fixes a top-6 whose gross excess is +0.13%/yr
  since 2024 and +0.5%/yr among liquid names. The ranking improvements are worth
  ~10% relative IC; the book gap is an order of magnitude larger.
- **Delivery as a screen rather than a ranker.** Momentum ranked *within* the top
  half of the delivery distribution gives +19.9%/yr top-6 at NW t +3.66, against
  −2.8% (t −0.64) for delivery as the ranker. Largest effect in the review, fully
  in-sample and regime-loaded (momentum's tail edge was +35%/yr in 2019-21 and
  +4.9% in 2024-26). **Needs a sealed window, not a config edit.**
- **The residual sector bucket.** 24.1% of ranked names. Index-constituent files
  are exhausted at ~755 symbols; closing the rest needs a per-symbol NSE industry
  feed (`/api/equity-meta-info` now 404s, `/api/quote-equity` 403s behind the bot
  shield). Ranking the residual cross-sectionally instead of within itself would
  be a model change and is epoch-gated.

---

# Round 2, RE-RUN on the rebuilt panel (2026-09-05)

**Everything above the R2-6 section was computed on the stale panel.** The panel
was rebuilt through `build_panel.py` (204,425 rows, 380 dates,
2018-11-27..2026-08-03, quality coverage 21.2% -> **49.0%**) and every headline
number recomputed. The rebuild reproduces the stored `score` to `max|diff| 0`
across 100.0000% of rows.

**Three claims did not survive the rebuild and are withdrawn.** They are listed
first, because a correction buried under the results that agreed is not a
correction.

| claim (stale panel) | rebuilt panel | status |
|---|---|---|
| Top-6 excess is +0.13%/yr over 2024–26 — "flat since 2021" | **+7.19%/yr, NW t +1.15** | **WITHDRAWN.** The book is insignificant in every window, but it is not decaying |
| Top-6 "collapses to +0.5%/yr in the liquid half" | **+2.46%/yr, NW t +0.53** | **WITHDRAWN as stated.** Still insignificant, but "collapses" overstated it |
| `quality` effective weight is 4.04% | **9.33%** (coverage 49.0%) | corrected; live engine on the current store is ~16% at 85% coverage |

### What survives

| | stale | **rebuilt** |
|---|---|---|
| Composite rank IC | +0.0580, NW t +4.76 | **+0.0541, NW t +4.52**, hit 68.7% |
| Decile spread D10−D1 | +1.84%, t +4.14 | **+1.645%, t +3.55** |
| Top-6 gross excess, full window | +6.2%, t +1.22 | **+6.41%, t +1.32** — still indistinguishable from zero |
| momentum effective weight | 48.0% | **45.7%** |
| momentum marginal ΔIC | −0.0030, t −0.34 | **+0.00068, t +0.08** — still exactly nothing |
| ownership marginal ΔIC | −0.0096, t −4.07 | **−0.00893, t −4.10** — still the load-bearing theme |

**The central finding is unchanged and is now better supported:** the ranking has
a real edge (NW t +4.52 over 380 dates); the six-name book has no window in which
it is statistically distinguishable from the universe it selects from (every
|t| < 1.5); and momentum carries 45.7% of the effective weight for no marginal
contribution while delivery carries 19.8% and does all the work.

Two changes in the other direction: `reversal` is now the **second** most
valuable theme (ΔIC −0.00305, NW t −2.39, up from −1.63) and `risk` the least
(−0.00217, t −1.28). `ulcer_120` is no longer significantly harmful on its own
(t +1.43, was +2.81); `voladj_mom_6_1` (t +2.25) and `resid_rev_21` (t +2.14)
still are.

### The prune survives a full panel rebuild, unchanged

The split-half selection was re-derived from scratch on the rebuilt panel — a
different date range, 16 more dates, and a quality theme with more than twice the
coverage. **It returns exactly the same seven factors**, with no additions and no
removals:

```
deliv_chg_5, mom_2_0, mom_3_1, mom_accel, resid_rev_21, ulcer_120, voladj_mom_6_1
```

and the epoch harness is stronger on the rebuilt panel than on the stale one:

| spec | ΔIC | NW t | CPCV folds + | p5 |
|---|---|---|---|---|
| **A/C prune the seven** | **+0.00658** | **+2.37** | **96%** | **+0.00024** |
| B move `ulcer_120` → momentum | +0.00153 | +2.37 | 89% | −0.00038 |
| H prune six + move `ulcer_120` | +0.00636 | +2.05 | 89% | −0.00174 |
| E equal theme weight | +0.00509 | +1.41 | 82% | −0.00386 |
| F prune + equal weight | +0.00783 | +1.27 | 80% | −0.01009 |
| D / G cost-only removals | ~0 | ~0 | ~55% | −0.004 |

**A/C's fifth percentile is positive**: across 45 purged and embargoed folds, even
the worst 5% improved on the incumbent. Half-wise, +0.00498 (t +1.24) on
2018-11..2022-09 and +0.00817 (t +2.15) on 2022-09..2026-08.

**Revised recommendation.** On the stale panel B was the cleaner read because A
inherited its selection from that panel. That objection is now much weaker: the
seven were re-selected independently on a materially different panel and came
back identical. **A (prune the seven) is the epoch to open**, with B folded in or
not — H shows it makes almost no difference, so the prune's value is the other
six and `ulcer_120` can be moved or dropped on taste. The four cost-infeasible
removals remain an execution decision, not an IC one (D: ΔIC ≈ 0).

### Candidates, re-run — conclusion unchanged

None of the six adds at NW t ≥ 2.0 on the rebuilt panel. Best is `idio_vol_120`
(ΔIC +0.0037, t +1.71). Three (`idio_vol_120`, `idio_skew_120`, `max_dd_120`)
carry a real standalone edge and add nothing blended, which is what a factor
already spanned by the shipped twenty-two looks like. The harness's verdict is
now **computed from its own table** rather than typed, after the first version
was left quoting t-statistics the rebuild had already replaced.

### Item 6 (delivery-as-screen) re-run — the recommendation REVERSES

On the stale panel, ranking momentum within the top half of the delivery
distribution beat unscreened momentum (+19.9% vs +18.4%). **On the rebuilt panel
the screen subtracts, monotonically:**

| top-6, gross excess/yr | rebuilt panel | NW t |
|---|---|---|
| momentum alone, unscreened | **+19.27%** | **+3.72** |
| momentum, delivery top 75% | +18.03% | +3.34 |
| momentum, delivery top 50% | +12.95% | +2.59 |
| momentum, delivery top 25% | +8.62% | +1.65 |
| delivery alone as the ranker | +1.61% | +0.36 |

**Do not pursue delivery-as-a-screen.** It was a stale-panel artifact.

What the re-run does show is the resolution of the audit's central paradox.
Momentum has **zero cross-sectional marginal value** (ΔIC +0.0007, t +0.08) and
**strong right-tail value** (+19.3%/yr at t +3.72 in the top six). Those are
consistent: IC is a whole-cross-section statistic and a six-name book only reads
the extreme tail. But the tail effect has decayed to nothing:

| momentum-alone top-6 | excess/yr | NW t |
|---|---|---|
| 2019–2021 | +30.15% | +4.03 |
| 2022–2023 | +6.70% | +0.70 |
| 2024–2026 | +3.63% | +0.47 |

So the honest reading of momentum's 45.7% effective weight is not "it does
nothing" — it is **"it did one thing, in the tail, and that thing stopped
working after 2021."** Which is the same conclusion the book statistics reach by
a different route, and neither is settled by anything short of forward data.

---

# SHIPPED: v4 composite (epoch `2026-09-05-c1d9632e92105fb4`)

`ranking.source: v3_composite -> v4_composite`. **SCAN MARKET now runs the pruned
model.** This is a model change and it went through the epoch protocol rather
than around it.

**v4 is v3 minus seven factors and nothing else.** Same five themes, same frozen
weights, signs, horizons and coverages, same two-level blend, same sector-neutral
ranks. `features/v4.py` calls `v3.score_frame` with a different theme table, so
there is no second scorer to drift; `score_frame` gained an optional `themes`
argument whose default path is proven identical in `tests/test_v4_score.py`.

Removed: `mom_2_0`, `mom_3_1`, `mom_accel`, `voladj_mom_6_1`, `ulcer_120`,
`resid_rev_21`, `deliv_chg_5` — 22 factors to 15.

**Evidence.** dIC +0.0066 at Newey-West t +2.37 across 45 purged, embargoed CPCV
folds; 96% of folds improved; fifth percentile still positive. The seven were
selected by an independent split-half and **re-derived unchanged** after the
panel rebuild — a different date range, 16 more dates, quality coverage 21% to
49% — returning the same seven with no additions or removals.

**Limits, stated on the card and in the config.** Not a sealed-holdout result:
v3's two windows were earned by the 22-factor set, do not transfer, and are
spent. CPCV measures stability, not out-of-sample selection, and the seven were
chosen on this panel. **The forward test registered with this epoch (375
sessions / 18 months) is what grades it.**

**It does not fix the book.** At six names on a 21-session cadence the change is
inside the noise. The ranking has an edge; the concentrated book does not inherit
it. Unchanged, and still the largest open problem.

One immediate confirmation: with `ulcer_120` gone the live redundancy check
reports **zero breaches**. It was the source of all three.

### What was rejected, and why the biggest number lost

| candidate | full-window book | why not |
|---|---|---|
| **momentum-only** | **+21.4%/yr gross, t +2.51**, positive in all three sub-periods | Chosen by reading book results across **25 model × exit-band combinations**. Contradicts the IC evidence (momentum's marginal ΔIC is +0.0007, t +0.08). Sub-period t declines monotonically **2.88 → 1.94 → 0.93**. Discards the only out-of-sample evidence the engine has. Chui et al. (2023) put Indian momentum in the *most liquid* names; this universe's median ADTV is ₹23 cr |
| equal theme weights | ΔIC +0.0051 | t +1.41, p5 −0.0039 |
| cost-only prune | ΔIC ≈ 0 | t −0.05, 53% of folds. The case is execution, not information; two of the four are already in the seven |
| six new factors | — | none adds at NW t ≥ 2.0; three predict standalone and add nothing blended |

**The book turnover finding, not acted on and worth more than any of the above.**
At the live 6-slot / 21-session / exit-18 configuration the book replaces **61%
of itself per rebalance**, costing **~6.6%/yr** against a gross excess of +8.5%
(t +1.41) — a net of +1.9%. Widening the exit band cuts turnover monotonically
for every model tested (v3: 85% → 70% → 61% → 47% → 35% at exit 6/12/18/30/48).
The cost side of that is mechanical and certain; the gross side is noisy and
non-monotonic at n=95 rebalances, which is why **no exit-band change ships here**
— picking one off this sweep would be selecting a config on the same data the
audit already used. It is the highest-value remaining experiment and it belongs
in a pre-registered harness, not in a config edit.

### Epoch hygiene done alongside

- Old epoch `2026-09-03-5e0c98515d13e6e2` closed **SUPERSEDED** (scorer changed;
  store had drifted).
- Store re-manifested: `75152d895cd25607 -> 86f8b3d6e8865906`, 51 files,
  9,436,569 rows, `--verify` clean.
- First v4 epoch open (`…6a059ec0`) **VOIDed after 50 seconds** and re-cut as
  `2026-09-05-c1d9632e92105fb4`: it had recorded the pre-rebuild manifest digest,
  and an epoch drifted from birth is the 2026-09-03 dirty-tree mistake again. No
  outcome was ever resolved under it.
- Forward test registered against `baseline-v2@7b4f50fcf98d11ba`, commit
  `5daaff349593`.

