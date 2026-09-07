# PROSIGNAL REBUILD — PHASE 0, 1 AND 2

*Archaeology, research, target architecture and deletion plan. Written 2026-09-07. No code changed to produce this document.*

> **The one-sentence finding.** This repository contains a cross-sectional ranking with genuine statistical evidence behind it, wrapped in a portfolio layer that holds 15–30% of capital in equities and therefore throws that evidence away. The ranking is the asset. The book is the defect. Almost everything that should be deleted sits between them.

---

## PART 1 — CODEBASE MAP

### 1.1 Scale

| | |
|---|---|
| tracked files | 293 |
| Python LOC | 76,811 |
| `src/prosignal` | 110 modules, 46,388 LOC |
| tests | 119 files, 28,277 LOC |
| `config/parameters.yaml` | 3,200 lines, 199 tunables (123 `UNVALIDATED`) |
| `README.md` | 92 KB |
| branches | 30 local, 24 on origin |
| local data | 813 MB (410 M cache, 246 M curated, 147 M `jobs.sqlite3`) |

### 1.2 Package layout

```
src/prosignal/
  pipeline.py            955   orchestrates stages 1-8; the SCAN path
  api.py               1,861   FastAPI, 36 routes
  cli.py               3,144   data / analyse / research subcommands
  config/schema.py     2,355   every tunable, typed, with status tags
  core/                        contracts, calendar, clock, errors, logging
  data/                        store, ingest, universe, corporate_actions, providers/
  features/            6,000   v3, v3_factors, crosssec, crossmodel, famamacbeth,
                               fundamentals, fundamental_factors, pit_fundamentals,
                               labels, exits, families, linear, earnings, v9r
  indicators/                  pure price functions
  stages/              5,688   the eight-stage decision pipeline
  validation/          7,531   harness, results, metrics, portfolio_sim, cpcv,
                               forward, epoch, registry, significance, readiness
  presentation/        2,616   viewmodel, history, evidence, narrative, outcome
  static/index.html    3,487   the entire UI, one file
```

### 1.3 Reachability (measured, not assumed)

An AST import-graph walk over all 110 modules:

| root | modules reached | LOC |
|---|---|---|
| `pipeline` (what SCAN executes) | 68 | 25,496 |
| `api` | 95 | — |
| `cli` | 93 | — |
| reachable from none of the three | 5 | — |

Only two modules are genuinely orphaned: `data/providers/nse_shareholding.py` and `data/providers/nse_surveillance.py`. **There is very little dead code at module level.** The waste in this repository is not unreferenced files — it is live code that runs on every scan and does not change the answer. That distinction drives the whole deletion plan below.

---

## PART 2 — THE ACTUAL SIGNAL FLOW

Traced through runtime, not filenames.

```
POST /analysis/run
   └─ jobs.py enqueues → pipeline.run_analysis(config, as_of)
        │  [one store lock held for the whole run: prices at Friday and
        │   delivery at Tuesday describe no day that existed]
        │
        ├─ Stage 1  data quality + leakage gate      706 LOC
        │     market-wide failures raise MarketWideHalt (≠ NO TRADE)
        │     per-stock failures exclude that name
        │
        ├─ Stage 2  regime                            733 LOC
        │     trend / volatility (India VIX percentile) / breadth / dispersion
        │     SCALES factor weights; does not filter output
        │
        ├─ Stage 3  eligibility                       432 LOC
        │     hard binary gates; NOT_TESTABLE never becomes PASS
        │
        ├─ Stage 4  cross-sectional scoring         1,361 LOC   ◀ THE ASSET
        │     ├─ v3 block: 22 factors → 5 themes → capped weighted blend
        │     │    winsorise → standardise → sector-neutralise → weight → rank
        │     └─ family block: momentum_12_1, sector_relative_strength,
        │          value, quality — A SECOND, PARALLEL FACTOR SYSTEM that
        │          does not rank anything under `ranking.source: v3_composite`
        │
        ├─ Stage 5  false-signal defense              576 LOC
        ├─ Stage 6  entry: rank admission + hysteresis 400 LOC
        ├─ Stage 7  risk: ATR stop, R targets, SIZE   370 LOC   ◀ THE DEFECT
        └─ Stage 8  final gates + NO-TRADE funnel   1,041 LOC
              │
              ├─ ledger.py       immutable run row, versioned
              ├─ outcomes.py     forward returns resolved later
              └─ presentation/   viewmodel → static/index.html
```

---

## PART 3 — THE CENTRAL FINDING

### 3.1 The ranking has evidence

> [!IMPORTANT]
> **Superseded 2026-09-07.** These figures were computed on a panel that began at store session 301 against a factor block requiring 421 — see §8.4. They are kept here as the claim under test, not as the finding, and are being regenerated. The §3.2 and §3.3 conclusions do not depend on them.

From `docs/RESULTS_OF_RECORD.md` (generated 2026-09-04), panel 2018-11-27 → 2026-08-03, 204,425 rows, 380 signal dates, **31.1 independent 63-session windows**. Signal dates are 5 sessions apart against a 63-session label, so naive *t* is inflated by ≈√VIF and is not quoted:

**Regenerated 2026-09-07** on the rebuilt panel: 197,940 rows, 356 signal dates, **29.2 independent 63-session windows**. The shipped-panel figures are shown struck through beside them.

| horizon | rank IC | IC *t* (corr.) | *was* | quintile spread *t* | *was* | top-decile excess | decile *t* | *was* |
|---|---|---|---|---|---|---|---|---|
| 21 | +0.0545 | **+4.01** | ~~+3.87~~ | +2.60 | ~~+2.89~~ | +0.49% | **+1.88** | ~~+2.49~~ |
| 42 | +0.0582 | **+3.12** | ~~+3.39~~ | +2.46 | ~~+2.96~~ | +0.93% | **+1.75** | ~~+2.36~~ |
| 63 | +0.0618 | **+2.70** | ~~+3.11~~ | +2.20 | ~~+2.80~~ | +1.33% | **+1.61** | ~~+2.21~~ |

**The top-decile statistic no longer clears even a conventional *t* = 2.0 at any horizon.** It fell at every one — +2.49→+1.88, +2.36→+1.75, +2.21→+1.61 — because the 24 dates the rebuilt panel correctly refuses (§8.4) were a flattering momentum stretch. The quintile spread fell too. Only the h=21 rank IC rose, to +4.01, and even that misses the 4.13 bar of §3.1's honest trial count.

**Nothing in this engine clears its own honest multiple-testing bar.** That is the finding, and it is a finding about the prior evidence base rather than about this rebuild: the numbers that looked adequate were computed on a panel whose first 120 sessions were scored on truncated factor frames.

`top-decile excess` is computed per date as *mean(top 10% by score) − mean(the whole eligible universe that date)*, then corrected for window overlap (`validation/results.py:248`). It is measured **against the equal-weight eligible universe**, which is the correct alternative for a long-only book drawn from that universe.

~~This clears the Harvey–Liu–Zhu multiple-testing bar of *t* > 3.0 at h=21 and h=42, and sits marginally above it at h=63 — against 119 cumulative charged trials.~~

**That sentence was wrong, and 119 is the wrong number.** A prior audit (`docs/BUILD_PLAN_V10.md`, D-021) already reconciled three irreconcilable trial counts — 119 in the registry, 4,877 in `parameters.yaml`, 81 in the README — and reached the honest answer: **"at least 5,140 and not reconstructible."** The registry was created *after* most of the search happened, and the v3 search that chose these 22 factors and 5 themes had its code deleted on 2026-09-03, so **its arms cannot be recovered from this repository at all**.

That matters because the bar scales with the number of looks. The expected maximum of *N* standard normals is ≈ √(2 ln *N*):

| trials *N* | bar a best-of-*N* pick must clear | measured |
|---|---|---|
| 119 (registry) | *t* ≥ 3.09 | h=63 at +3.11 — **scrapes past** |
| 300 (HLZ's published-factor setting) | *t* ≥ 3.38 | h=42 at +3.39 — scrapes past |
| 4,877 | *t* ≥ 4.12 | **all three fail** |
| ≥5,140 (the honest floor) | *t* ≥ 4.13 | **all three fail** |

**On the honest trial count, none of the three horizons clears.** Only h=21 (+3.87) survives even the 1,000-trial bar. The correct conclusion is not that the ranking is disproven — it is that **the in-sample evidence for it cannot be rescued by any amount of re-measurement, because the denominator is unknown and unrecoverable.**

This changes what the rebuild is for. The v3 ranking must be treated as a **hypothesis with a plausible economic story and an unrecoverable selection history**, not as a validated result. The only thing that can settle it is evidence from data never used to select it — which makes the pre-registration in Part 8 the whole game, and makes it doubly important that both sealed windows are already spent and 2012–2017 is burnt.


### 3.2 The book destroys it

`LIVE_BOOK` = 6 slots, entry rank 6, exit rank 18, 21-session cadence. From the repository's own simulator (`research/v3/experiments/book_sim.json`):

| window | arm | excess p.a. | IR | **beta** | avg names | gross excess | cost drag |
|---|---|---|---|---|---|---|---|
| full (regenerated) | shipped 6-name | **−19.7%** | **−0.89** | **0.12** | 4.7 | −19.3% | 0.5% |
| full | live 6 @0.1 | −17.6% | −0.83 | 0.15 | 5.0 | −17.3% | 0.35% |
| A 2025-26 | live 6 @0.1 | −9.6% | −0.79 | **0.30** | 4.8 | −9.2% | 0.40% |
| B 2021-22 | live 6 @0.1 | −3.6% | −0.31 | **0.25** | 4.9 | −3.3% | 0.33% |
| full | holdout 10 @0.1 | −14.4% | −0.75 | **0.26** | 8.3 | −13.8% | 0.56% |

Alpha per period is **+0.09%** — indistinguishable from zero. Cost drag is 0.3–0.9% a year and is **not** the binding constraint. The binding constraint is beta 0.15.

### 3.3 Why: three lines of arithmetic

`stages/stage7_risk.py:270` sizes every position as `min(qty_risk, qty_slot, qty_liq)` where

```
qty_risk_value = capital × risk_per_trade_pct × category_fraction ÷ stop_distance
qty_slot_value = capital ÷ max_open_positions × category_fraction
```

Shipped parameters: `risk_per_trade_pct = 1.0`, `max_open_positions = 6`, stop = `8 × ATR` capped at `max_stop_distance_pct = 35`, `category_fraction ∈ {1.0, 0.6, 0.3}`.

An 8×ATR stop on a typical NSE mid-cap lands at 20–35% of entry, so:

| stop distance | risk-based position | capital-slot position | binding | 6 slots invested |
|---|---|---|---|---|
| 35% (the cap) | **2.9%** of capital | 16.7% | risk budget | **17%** |
| 20% | **5.0%** of capital | 16.7% | risk budget | **30%** |

The risk budget always binds; `category_fraction` only makes it smaller.

**Measured, driving the real `stage7_risk.build_plan` against the real config** (`tests/test_invested_capital.py`), six names spanning 1.2%–3.5% daily volatility with liquidity made abundant so it cannot bind:

| name | daily vol | stop | position | % of capital | binding |
|---|---|---|---|---|---|
| NAME0 | 1.2% | 21.6% | ₹46,089 | 4.6% | risk budget |
| NAME1 | 1.8% | 29.6% | ₹33,064 | 3.3% | risk budget |
| NAME2 | 2.2% | 35.0% | ₹27,847 | 2.8% | risk budget |
| NAME3 | 2.6% | 35.0% | ₹27,783 | 2.8% | risk budget |
| NAME4 | 3.0% | 35.0% | ₹27,314 | 2.7% | risk budget |
| NAME5 | 3.5% | 35.0% | ₹27,117 | 2.7% | risk budget |
| | | | | **18.9% invested** | **81.1% idle** |

The stop cap binds on four of six names, so the more volatile a name is the *smaller* its position — and beyond 2.2% daily vol the sizing stops responding to volatility at all, because every stop is pinned at the 35% cap.

**Forgone benchmark return on 81.1% idle cash: −18.6% annually** at the regenerated benchmark of +22.9%. **The regenerated net excess of the shipped book is −19.7%.** The cash drag accounts for **94% of the underperformance**, with cost drag at 0.5%. Nothing else needs to be invoked — not the factors, not the costs, not the regime layer.

This is the one conclusion in this document that the panel rebuild did not weaken. It rests on shipped config values and a benchmark return, not on any statistic with a *t* beside it.

`config/parameters.yaml:236` states the design intent in its own note: *"With a 1% risk budget and a 5% stop, the position is 20% of capital."* The shipped stop is not 5%. It is 8×ATR. **Nobody multiplied the two shipped numbers together.** This survived the entire test suite because no test asserts on aggregate invested capital.

### 3.3.1 Why a constant could not have fixed it

`stage7_risk.build_plan` is called **once per symbol**, in a loop (`pipeline.py:314`), with that name's frame, price, ADTV and score. It never sees the other holdings.

```python
for sym in defended:
    plans[sym] = stage7_risk.build_plan(ticker=sym, frame=frame, ...)
```

**"Equal weight across the book" is not expressible in that signature.** 1/N is a property of the set, and no function in this pipeline was given the set before the positions were already sized. The sizer's only book-level input is `max_open_positions`, which it uses to derive a *slot* — and the slot is the constraint that never binds.

So this was not a badly chosen constant. It was a **shape error in the interface**, and that is why lowering `atr_multiple` or raising `risk_per_trade_pct` would not have fixed it: any per-name sizing rule produces a book whose total exposure is an accident of the names that happened to qualify that day. The fix has to move sizing from per-name to per-book, which is what `prosignal/book.py` does and why the change is structural rather than a tuning.

It also explains the survival. A reviewer reading `build_plan` sees a defensible risk rule correctly implemented; the defect is only visible when you sum across a loop that lives in a different file. No test summed it. `tests/test_invested_capital.py` now does.

### 3.4 The engine already knew

`features/v3.py:465` `BOOK_NOTE`, in the repository's own words: the two sealed windows evaluated a *different* book (10 slots, weekly, exit 30) which lost 2.8%/yr on window A and beat by 2.0% on window B; top-ten excess on window A was **+0.38% at t 0.81**, while the quintile spread held at **t 2.89**. It concludes: *"Ordering within the top few names is the part of this model the holdouts did not support, and a six-name book is a bet on exactly that."*

That paragraph is correct and has been correct for some time. It was written into a docstring instead of into the architecture.

### 3.5 A second defect: the identity system recorded two different things under one name

Found while re-manifesting, and it invalidates §36 (signal versioning) rather than the returns.

`AppConfig.version` returned **two different kinds of string with the same shape**:

| caller bound a store? | `version` | covers |
|---|---|---|
| yes (`cli.main`, `pipeline.run_analysis`) | `baseline-v2@2496c3bcbf1dd996` | `XOR(params, store, train_window)` |
| no (**`create_app`**, every test) | `baseline-v2@6828f4d19d68a1ac` | parameters only |

Both print as `baseline-v2@<16 hex>`. Nothing distinguished them by eye, and `load_config()` is a process-wide singleton, so *which one a record carried depended on whether some unrelated earlier code path happened to bind a store first.*

**The API never bound a store at all.** So `/ready`'s config check, the **forward test's registration**, the performance cache key, the live/research parity comparison at `/measurement`, and every measurement record stamped a parameters-only identity — while the ledger rows written by the same process, through `pipeline`, carried the full one. The forward test's entire integrity check is *"did `config_version` change"*, and it was registered under an identity that by construction cannot see the store.

This also explains the fourth test failure. `test_restart_gate` compared a store-bound epoch against an unbound in-test reading and reported **"configuration changed since the epoch opened"** — a change that never happened. It could never have passed.

Fixed in this phase:

- `create_app` binds a store, as `cli.main` and `pipeline` already did.
- The degraded form is now self-describing: `baseline-v2@params-only:6828f4…`. It cannot be mistaken for, or compare equal to, a full identity.
- `epoch.current_identity` and `readiness.restart_refusals` resolve the full identity themselves, so the answer to "which engine is this" no longer depends on the caller's construction order.
- `Identity.differences` and the MODEL gate **refuse** to compare a parameters-only identity instead of reporting a phantom drift.

This is the same root pattern as §3.4 and as the family block: one concept, two implementations, and no guard that they agree. It is the pattern the rebuild is meant to eliminate, and it is why §48's "one canonical implementation for each concept" is the load-bearing rule in Part 6.

### 3.6 The other sealed result

`v9r_core` on a sealed 2012–2017 window: **+9.50% net active at Newey-West t +1.87 against a pre-registered bar of 2.0 — a FAILED ship gate.** Positive and underpowered. That window is spent and must not be re-specified against.

---

## PART 4 — CODEBASE VALUE AUDIT

| Module / layer | LOC | Purpose | Research justification | Decision |
|---|---|---|---|---|
| `data/store.py`, `data/ingest.py`, `providers/nse_archives.py` | 2,949 | PIT curated store, NSE bhavcopy/delivery ingest | Store lock, manifest digest, staleness guard all correct | **KEEP** |
| `data/corporate_actions.py` | 583 | Split/dividend/bonus reconciliation | Was broken (66% of pre-2018 splits missing), now fixed and load-bearing | **KEEP** |
| `data/universe.py` | 508 | PIT membership with explicit survivorship flag | Honest about the gap; `pre_snapshot_policy: halt` is right | **KEEP / REWORK** |
| `stages/stage1_data_quality.py` | 706 | Leakage + integrity gate; market-wide vs per-stock | Correct design; fail-closed | **KEEP** |
| `stages/stage3_eligibility.py` | 432 | Hard binary gates before scoring | Correct ordering; NOT_TESTABLE ≠ PASS | **KEEP** |
| `features/v3_factors.py` + `v3.py` | 926 | The 22 shipped factors and theme blend | The only object in the repo with a corrected *t* above 3 | **KEEP / PRUNE** |
| `validation/results.py`, `metrics.py`, `significance.py` | 1,972 | Per-date IC, quintile, decile, VIF correction | The measurement system is the best part of this codebase | **KEEP** |
| `validation/epoch.py`, `registry.py` | 827 | Trial budget, epoch discipline | Required by §19/§37; 119 trials charged | **KEEP** |
| `ledger.py`, `outcomes.py` | 1,384 | Immutable signal ledger + forward resolution | Required by §33 | **KEEP / REWORK** |
| **`stages/stage7_risk.py`** | **370** | **ATR stop, R targets, invalidation, risk-budget sizing** | **Produces beta 0.15. No OOS support for any of it** | **DELETE** |
| **`stages/stage5_false_signal.py`** | **576** | **Penalty/reject screen after ranking** | **No recorded ablation showing it adds OOS value; it can only subtract from a ranking that is already the evidence** | **DELETE pending ablation** |
| **`stages/stage8_final_signal.py`** | **1,041** | **Decision gates, bands, NO-TRADE funnel** | **Gating a validated ranking on unvalidated thresholds** | **REPLACE (≈150 LOC)** |
| `stages/stage6_entry.py` | 400 | Rank admission + hysteresis buffer | The buffer *is* measured (index-construction practice, turnover arithmetic) | **MERGE into book layer** |
| **`stage4` family block** (`features/fundamentals.py`, `fundamental_factors.py`, `crosssec.py`, `crossmodel.py`, `famamacbeth.py`, `linear.py`, `families.py`) | **~4,000** | **A second, parallel factor system computed every run** | **Does not rank anything. `crosssec.py` (806 LOC) is imported for one function, `liquidity_mask`** | **DELETE** |
| `features/v9r.py` + `v9r_core` path | ~500 | Alternative model | Its only evidence is a FAILED ship gate | **DELETE** (archive the doc) |
| `stages/stage2_regime.py` | 733 | Regime scaling of factor weights | Daniel–Moskowitz (2016) supports the *concern*; no ablation here shows the *implementation* helps OOS | **UNKNOWN → ablate, then decide** |
| `v3_monitor.py` | 434 | Redundancy monitor | Inspects a frame that is always `None`; no shipped factor pair has ever been checked | **REPLACE** |
| `config/schema.py` + `parameters.yaml` | 5,555 | **199 tunables: 123 UNVALIDATED, 33 OPERATIONAL, 15 MEASURED** | §45 — 62% of the surface has no research basis, and much of it is inert | **REPLACE (~60 parameters)** |
| `cli.py` | 3,144 | 25+ subcommands | Research and ops in one file | **MERGE / SPLIT** |
| `static/index.html` | 3,487 | Entire UI | One file, one button; the shape is right | **REWORK** |

---

## PART 5 — RESEARCH REPORT

### 5.1 What the external evidence says

| Claim | Source | Bearing on ProSignal |
|---|---|---|
| A new factor needs **t > 3.0**, not 2.0, once data mining is accounted for | Harvey, Liu & Zhu (2016), *RFS* 29(1) | The engine's corrected IC *t* is +3.87 (h=21), +3.11 (h=63) against 119 trials. It passes — narrowly, and only because the corrected statistic was used |
| IR = **TC × IC × √breadth**; realised TC is 0.3–0.8 under real constraints | Grinold (1989); Clarke, de Silva & Thorley (2002) | A 6-name book from a 380-name universe has minimal breadth *and* a TC crushed by risk-budget sizing. This is the theoretical statement of §3.3 |
| Momentum is present and persistent in Indian equities | Multiple NSE/BSE studies, 1997–2022 | Supports the momentum theme (40% weight) |
| **Contradictory:** Indian value and momentum anomalies are *explained by risk models* over 2005–2016 and have faded; **decile corner portfolios beat quintiles**; **bivariate sorts do not beat univariate** | Sharma, Subramaniam & Sehgal (2021), *Global Business Review* 22(1) | Three consequences: (a) expect decay, (b) build at decile granularity, (c) do not add complexity to the sort |
| MAX (highest daily return over the past month) predicts returns **negatively**; decile spread > 1%/month; robust to size, B/M, momentum, reversal, liquidity, skewness | Bali, Cakici & Whitelaw (2011), *JFE* 99(2) | Independent, peer-reviewed support for `max5_21` at sign −1. One of the few shipped factors with outside corroboration |
| Sharpe must be **deflated** by number of trials, skew and kurtosis; PBO rises fast with trial count | Bailey & López de Prado (2014); Bailey et al. | The trial registry exists and is charged. Any new book config must be deflated against 119+ trials |
| **Delivery percentage**: no peer-reviewed cross-sectional evidence located. It is a practitioner/retail screen metric | NSE security-wise archives; practitioner sites only | The `ownership` theme carries **18.9% weight** on a factor family with **zero external corroboration**. It was found by internal search alone and must clear a Harvey–Liu–Zhu bar on its own |

### 5.2 Evidence matrix — the shipped factors

| Theme | Weight | Factors | External evidence | PIT safe | Coverage | Verdict |
|---|---|---|---|---|---|---|
| momentum | 0.400 | 10 (mom_2_0, mom_3_1, mom_12_6, mom_accel, voladj ×2, mom_consist_126, intraday_mom_126, prox_52w ×2) | ●●● strong, incl. India | yes | 99.9% | **KEEP — see §5.2.1; the theme average already absorbs the collinearity** |
| reversal | 0.110 | 4 (max5_21, rev_1w, price_vs_vwap_20, resid_rev_21) | ●●● (Bali et al. for MAX; short-term reversal well documented) | yes | 99.9% | **KEEP** |
| risk | 0.111 | 3 (downside_vol_60, ret_kurt_126, ulcer_120) | ●●○ low-vol / lottery literature | yes | 99.6% | **KEEP** |
| ownership | 0.189 | 3 (deliv_pct_60, deliv_chg_5, deliv_z_21) | ○○○ **none found** | yes | 93.7% | **DEMOTE — must re-clear a t>3 bar alone or be deleted** |
| quality | 0.190 | 2 (net_margin, margin_stability) | ●●● for quality — **but both ship at sign −1**, so the theme buys *low and unstable* margins | yes | **49% on panel, ~85% live, fit at 19%** | **DELETE — see 5.3** |
| *value* | — | 0 of 8 cleared | ●●● in literature | — | balance-sheet data starts 2023 | **excluded — correctly, it is a data gap not a finding** |
| *liquidity* | — | 0 of 9 cleared | ●●○ | — | — | excluded |
| *seasonality* | — | 0 of 2 cleared | ○○○ | — | — | excluded |

### 5.2.1 Redundancy, measured — and a claim of mine that did not survive it

Computed on the rebuilt panel, **per date across all 356 dates, then averaged** (pooling would mix cross-sectional structure with time variation and inflate *N* roughly 500-fold):

**Strongest factor pairs, mean per-date Spearman of the ranked factors**

| A | B | ρ | |
|---|---|---|---|
| `prox_52w` | `ulcer_120` | **−0.769** | momentum / risk |
| `prox_52w` | `prox_52w_now` | +0.740 | within momentum |
| `mom_accel` | `voladj_mom_6_1` | −0.728 | within momentum |
| `prox_52w_now` | `ulcer_120` | −0.707 | momentum / risk |
| `price_vs_vwap_20` | `resid_rev_21` | +0.702 | within reversal |
| `mom_12_6` | `voladj_mom_12_1` | +0.701 | within momentum |

**Effective number of independent bets** (entropy-effective rank of the per-date rank-correlation matrix):

| | measured | nominal |
|---|---|---|
| factors | **12.10** | 22 |
| themes | **4.64** | 5 |

**I was wrong in §5.2's first draft, and the measurement is why.** I wrote that ten near-collinear momentum columns "are not ten bets — prune to 3–4". At the *factor* level that is true (12.1 of 22). But the themes come out at **4.64 of 5**, which means the theme average is doing exactly the job `_redundancy`'s own docstring claims for it: *"their collinearity is what the aggregation exists to absorb."* Pruning within a theme that is then averaged would buy **simplicity, not statistical honesty** — the sub-score barely moves. That is still a reason to prune under §44, but it is not the reason I gave, and the ablation has to be judged on the composite, not on the pair correlations.

**What the measurement does indict is the `risk` theme.** Its strongest member `ulcer_120` correlates **−0.769** with momentum's `prox_52w` — both are "how far below the recent high" — and momentum/risk is the highest theme pair at **+0.345**. `risk` carries 11.1% weight as though it were independent evidence, and roughly a third of it is momentum with a sign flip. That is the ablation to run first, ahead of any momentum pruning.

**And the shipped monitor would not have told anyone.** `redundancy.max_abs_spearman` is **0.60**, status `UNVALIDATED`, and `on_breach: log`. At least **11 pairs** exceed it on the panel — six exceed 0.70. A cutoff that is breached eleven times and only writes a log line is not a control; it is a record nobody reads.

### 5.3 The quality theme is a data-mining artifact

It carries 19% weight. Both its factors ship at sign −1, which means the theme prefers **low** net margins and **unstable** ones — the repository's own comment concedes the key `quality` "is the wrong word for it" and relabels it *"Low-margin tilt"*. Its coverage was **19% at fit time**, is **49% on the research panel**, and **~85% live** — three different models wearing one name. Its underlying `statements.parquet` was rewritten twice during the last audit cycle, which is what invalidated the previous panel.

A theme with an inverted economic sign, no external corroboration for that inversion, and coverage that has quadrupled since it was fitted is not a factor. It is a fit to 19% of a panel. **Delete it and renormalise.**

### 5.4 Where the data actually stands

| dataset | span | depth | verdict |
|---|---|---|---|
| prices | 2017-09-08 → 2026-09-04 | 7,119 symbols, 5.04 M rows | sound |
| delivery | 2019-06-27 → 2026-09-04 | 1,712 sessions | sound, but **truncates any factor using it to 2019+** |
| indices | 2017-09-08 → 2026-09-04 | 177 | sound |
| corporate actions | — | 6,693 rows | fixed; previously missed 66% of pre-2018 splits |
| fundamentals | → 2025-03-11 (older stamp) | **3,504 rows total** | thin and stale |
| **PIT index membership** | **2026-08-14 → 2026-09-04** | **9 snapshots** | **three weeks of genuine PIT membership. Everything before it is reconstructed** |

Survivorship has been **bounded, not eliminated**: 3,552 names, 841 disappeared, 2.64%/yr disappearance; stressing every disappearing row to −30% moves rank IC from 0.0580 to 0.0566 — a **2.4% relative** haircut (`survivorship_bound.json`). Real, measured, and not the thing that is wrong with this system.

### 5.5 Costs are not the problem — *for the broken book, and that is the catch*

Round-trip cost at the shipped impact coefficient (0.1): **80 bps median**, 106 bps p75; annual drag 3.2–6.4% at 24–48 trades. At coefficient 0.5 it is 256 bps and 10–20% a year. The live book's measured cost drag is **0.5% a year** — because it barely trades and barely invests.

**Measured on the rebuilt book, that changes.** `research/v3/experiments/book_exposure.py` runs the actual construction over all 89 rebalances of the panel, with the real universe and the real ADTV distribution. It reports mechanical properties only — deployed fraction and turnover — so it charges no trial:

| | measured |
|---|---|
| deployed fraction | **99.96% mean, 99.64% worst, never below 95%** |
| names held | median 50 (eligible universe median **532**, range 230–750) |
| turnover | **35.3% per rebalance → ~445% a year** |
| cost at 80 bps | **~3.56% a year** |
| liquidity cap binding | **never**, on any name, on any date |

So the net arithmetic of the fix is: **recover ~18.6% a year of forgone benchmark return, pay ~3.1% a year more in transaction costs.** Strongly positive, and not close.

But "costs are not the problem" was true *of a book that did not trade*. Once it does, costs are the second-order question and no longer negligible — 3.56% a year is roughly a fifth of the benchmark's return. The impact coefficient, which nothing binds on today, starts to matter: at 0.25 the same turnover costs ~6.5% a year.

**Turnover is high because the signal is short-horizon.** 445% a year means the book rotates about every 11 weeks against a 63-session label. That is consistent with a model whose live fit prices reversal and delivery rather than momentum, and it is a property of the signal, not of the construction.

**The band was not tuned, and the curve is recorded so nobody has to re-derive it:**

| cadence | exit band | turnover p.a. | cost @80bps |
|---|---|---|---|
| 21 sessions | **20% (pre-specified)** | **445%** | **3.56%** |
| 21 sessions | 30% | 301% | 2.41% |
| 21 sessions | 50% | 176% | 1.41% |
| 63 sessions | 20% | 238% | 1.90% |

The 20% band is the one Part 6.2 declared *before* this was measured, and it stays. Widening it to 50% would cut costs by 2.2 points a year and would also hold names ranked in the bottom half of the top half — a returns trade this repository cannot currently price, because doing so means spending trials against an unknown denominator (§3.1). **It is registered as a pre-specified question for the out-of-sample surface, not decided here.**

### 5.6 Stopping rule (§55)

Research stops here. The material architecture decisions — ranking vs book, granularity, factor set, multiple-testing bar, cost model — now have evidence attached. The remaining uncertainty is explicit and is listed in Part 8.

---

## PART 6 — TARGET ARCHITECTURE

```
        NSE ARCHIVES ─────────────┐
        (bhavcopy, delivery,      │
         corp actions, indices)   ▼
                          ┌───────────────┐
                          │  PIT STORE    │  one lock, one manifest digest,
                          │  data/        │  staleness guard on every read
                          └───────┬───────┘
                                  ▼
                          ┌───────────────┐
                          │  UNIVERSE     │  PIT membership + listing dates
                          │               │  halt before first snapshot
                          └───────┬───────┘
                                  ▼
                          ┌───────────────┐
                          │  QUALITY GATE │  fail closed; market-wide vs per-stock
                          └───────┬───────┘
                                  ▼
                          ┌───────────────┐
                          │  ELIGIBILITY  │  hard binary gates only
                          └───────┬───────┘
                                  ▼
                          ┌───────────────┐
                          │  ONE RANKING  │  N factors → themes → sector-neutral
                          │               │  composite. ONE canonical scorer.
                          └───────┬───────┘
                                  ▼
                          ┌───────────────┐
                          │  BOOK         │  top decile, EQUAL WEIGHT,
                          │               │  FULLY INVESTED, buffer band,
                          │               │  liquidity cap only
                          └───────┬───────┘
                                  ▼
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
            ┌───────────────┐          ┌───────────────┐
            │ IMMUTABLE     │          │  UI           │
            │ LEDGER        │──────────▶  one button   │
            └───────┬───────┘          └───────────────┘
                    ▼
            ┌───────────────┐
            │ OUTCOMES      │  forward returns vs equal-weight universe
            └───────────────┘
```

### 6.1 The five decisions

| Decision | Alternatives considered | Evidence | Failure mode if wrong |
|---|---|---|---|
| **Ship the ranking at decile granularity, not a 6-name book** | keep 6 names; 10 names; top quintile | **Not the in-sample decile statistic** — on the rebuilt panel that is +1.33%/63d at *t* +1.61 and does not clear 2.0 (§3.1). The decile rests on evidence external to this panel: Grinold (1989) IR = IC×√breadth and Clarke–de Silva–Thorley (2002) IR = TC×IC×√breadth, under which a 6-name book from ~380 is small on breadth *and* on transfer coefficient; and Sharma et al. (2021), who find decile corners beat quintiles in Indian equities specifically | The ranking has no edge at any granularity. Breadth then costs nothing relative to concentration, and the pre-registered re-check is what finds out |
| **Equal weight, fully invested** | risk-budget sizing (current); vol-target; conviction tilt | §3.3 arithmetic; IR = TC×IC×√breadth. Conviction tilts were tested and "did not survive changing the cadence" (`parameters.yaml:222`) | Drawdowns get larger. This is a stated, quantified risk trade, not a hidden one |
| **Delete stops, targets, invalidation levels** | keep 8×ATR floor; keep as alerts only | They are the mechanism of the cash drag; none has an OOS number; the disaster floor is already `armed=False` for the target and invalidation | A true disaster name runs. Mitigated by eligibility gates + the liquidity cap, and bounded by 1/N ≈ 2.5–4% per name |
| **One factor system** | keep the family block for the UI cards | Two systems means two definitions of "quality" in one run record, which already produced a contradictory run note | UI loses some card detail; rebuild attribution from the one scorer |
| **Rebuild the book as the thing that was measured** | patch sizing inside stage 7 | §57: compatibility is not more important than correctness. Stage 7's sizing, stage 8's gates and stage 5's penalties all move the held portfolio away from the measured decile | — |

### 6.2 Signal specification

```
UNIVERSE     NSE equities, PIT membership, listed, price/liquidity floors
             from stage 3, ~380 eligible names per date

FACTORS      pruned v3 set (target 8-12 columns, from 22), in 3-4 themes:
             momentum, reversal, risk, and ownership ONLY IF it re-clears t>3

TRANSFORM    per date: winsorise → standardise → sector-neutralise
             (residual bucket for sectors under 12 names) → theme sub-score
             → capped weighted blend. Nothing reads a session after the
             decision row.

SCORE        cross-sectional composite; the ranking IS the output

BUY          top decile of the eligible universe on the decision date,
             entered at rank <= 10th percentile, held while inside the
             20th percentile (buffer band)

WEIGHTS      equal, 1/N, subject only to: position <= 1% of trailing ADTV

HORIZON      63 sessions (the label the evidence was measured at),
             rebalanced every 21 sessions with the buffer

EXIT         rank leaves the buffer band, OR eligibility is lost,
             OR delisting. No price stop. No profit target.

COSTS        square-root impact model, coefficient 0.1 shipped,
             stress-tested at 0.25 and 0.5 on every reported number

CONFIDENCE   NOT a probability. Reported as: corrected t of the ranking at
             this horizon, coverage of themes for this name, and whether
             this date's dispersion is inside the normal band
```

### 6.2.1 What the decile choice does and does not claim

After the panel rebuild this needs saying plainly, because the earlier draft of this document leaned on a statistic that no longer supports it.

**It does not claim** the top decile earns +1.33% per 63 sessions. That figure carries *t* +1.61 against a multiple-testing bar of 4.13, and is not evidence of anything.

**It does claim** three things that survive:

1. **The shipped book's loss is arithmetic, not statistical.** 18.9% invested against a benchmark compounding at 22.9% forfeits 18.6% a year before any factor is consulted. Fixing it requires no belief about the ranking at all.
2. **If the ranking has any edge, breadth extracts more of it than concentration does** — IR = TC × IC × √breadth, and both terms favour 38 equal-weighted names over 6 sized by stop distance. This is theory, and it holds whatever IC turns out to be.
3. **If the ranking has no edge, breadth costs less than concentration.** An equal-weighted decile of the eligible universe tracks that universe closely; a 6-name book does not. The downside of being wrong is far smaller.

The decile is therefore the choice that is **least sensitive to the question this repository cannot currently answer.** That is the argument for it — not a backtest.

### 6.2.2 Breadth and weighting must change in the same commit

A sequencing hazard, recorded because the tempting order is the dangerous one.

Stage 8 currently selects **6** names and stage 7 sizes each at ~3% of capital. There are two independent defects — too few names, and each too small — and fixing either alone is worse than fixing neither:

| | names | per name | invested | verdict |
|---|---|---|---|---|
| today | 6 | 3.2% | **18.9%** | ~0 alpha, 94% of the loss is cash drag |
| **sizing fixed alone** | 6 | 16.7% | 100% | **a 6-name concentrated book at full exposure** — the worst of the three |
| breadth fixed alone | 38 | 0.5% | 18.9% | still 81% cash; nothing improves |
| both | 38 | 2.6% | ~100% | the target |

Sizing alone converts a mostly-cash book into a fully-invested 6-name one, leaning the entire account on the statistic that generalises *least* — top-decile ordering, whose sealed-window *t* was 0.81 and whose rebuilt-panel *t* is +1.61. `book.build_book` therefore takes the ranking and returns both the membership and the weights: the two cannot be deployed separately because the function does not offer them separately.

### 6.3 What the user sees

One button. The result is a ranked list of ~25–40 names with equal target weights, each carrying its theme attribution, its eligibility record, and the honest statement that the evidence behind it is a decile-level statistic on 31 independent observations. Not six names presented as high conviction.

---

## PART 7 — DELETION PLAN

| # | Delete | LOC | Why |
|---|---|---|---|
| 1 | `stages/stage7_risk.py` sizing, stops, targets, invalidation, risk categories | 370 | Cause of beta 0.15; no OOS support |
| 2 | `stage4` family block + `features/fundamentals.py`, `fundamental_factors.py`, `crosssec.py`, `crossmodel.py`, `famamacbeth.py`, `linear.py`, `families.py` | ~4,000 | Second parallel factor system that ranks nothing |
| 3 | `features/v9r.py`, `ranking.source` options `v9r_core`, `v2_composite`, `fitted_composite`, `family_average`, `measured_factor` | ~800 | Five selectable scorers, one shipped; `measured_factor` misled a full generation of the README |
| 4 | `stages/stage5_false_signal.py` | 576 | Pending ablation; it can only subtract from the measured ranking |
| 5 | `stages/stage8_final_signal.py` gates and bands | ~900 of 1,041 | Unvalidated thresholds gating a validated ranking; keep the NO-TRADE funnel counter |
| 6 | `quality` theme (`net_margin`, `margin_stability`) | — | Inverted sign, no corroboration, coverage 19%→49%→85% |
| 7 | Within-momentum duplicates (`prox_52w_now` given `prox_52w` at ρ +0.74), and `ulcer_120` if the `risk` ablation confirms §5.2.1 | — | Simplicity under §44. The theme average already absorbs within-theme collinearity (4.64 of 5 effective themes), so this is not an accuracy claim |
| 8 | `config/parameters.yaml` — the 123 `UNVALIDATED` tunables, minus any that survive ablation | 123 of 199 parameters | §45 |
| 9 | `data/providers/nse_shareholding.py`, `nse_surveillance.py` | ~400 | Unreachable |
| 10 | 28 of 30 branches; `*.pre-dedupe` / `*.pre-lineage-repair` ledger backups (19 MB) | — | Repo hygiene |
| 11 | `README.md` → under 10 KB, pointing at `RESULTS_OF_RECORD.md` | 82 KB | It carried withdrawn numbers for a generation |

**Estimated: ~11,000 LOC of source and ~2,400 lines of config removed. Target `src/prosignal` ≈ 18–20k LOC.**

### 7.1 Kept, deliberately

The measurement layer (`validation/`), the PIT store, corporate actions, the trial registry and epoch discipline, the immutable ledger, the store lock, and the staleness guard. These are the parts of this repository that are genuinely institutional, and they are the reason the central finding above could be established at all.

---

## PART 8 — VALIDATION PLAN (PRE-REGISTERED)

Written **before** the new book is measured, and charged to the trial registry.

### 8.1 The bar

| gate | threshold | rationale |
|---|---|---|
| rank IC, corrected *t* | **≥ 4.15** | √(2 ln N) at the honest floor of ≥5,140 trials (§3.1). *Not* HLZ's 3.0: that is calibrated to ~300 published factors, and this engine's selection history is larger and unrecoverable |
| top-decile excess vs equal-weight universe, corrected *t* | **> 2.0** | The statistic the product is built from |
| decile monotonicity | ≥ 6 of 9 steps rising, peak in decile 10 | A spread whose peak sits at decile 6 does not describe the traded object |
| net excess after costs at impact coeff **0.25** | **> 0** | Shipped is 0.1; 0.25 is the stress case |
| beta to the equal-weight eligible universe | **0.85 – 1.05** | The failure being fixed. A book that is not invested is not a book |
| deflated Sharpe | positive after deflation by cumulative trials, skew, kurtosis | Bailey & López de Prado |

### 8.2 Method

- **Purged, embargoed walk-forward.** No random shuffling. Embargo ≥ the 63-session label.
- **CPCV** for the overfitting probability, using the existing `validation/cpcv.py`.
- **Every number reported per date, then averaged.** Never pooled. Pooled figures lie about *N* here — 204,425 rows are 31.1 independent observations.
- **Ablation, forward and reverse**, on: each theme, each surviving factor, the buffer band, the rebalance cadence, sector neutralisation, and the regime layer. A module that changes no OOS statistic is deleted under §44.
- **The 2012–2017 window is burnt** (opened once for v9r; failed at *t* 1.87). It may not be re-specified against.
- **The quarterly re-check is the out-of-sample surface**, once its window stops overlapping window A. The forward test is registered as *falsification*, not as a ship gate: at this IR a paper trade needs ~6.5 years to reach *t* = 2.0.

### 8.3 The repository is currently un-shippable, and its own guards say so

`pytest` on `main`, 2026-09-07: **4 failed, 1,835 passed, 4 skipped** (19m 28s). Three of the four failures are one event.

| failing test | says |
|---|---|
| `test_data_manifest::test_the_shipped_store_is_manifested_and_verifies` | 9 curated files differ from `MANIFEST.json` (`prices/year=2026`: 22,580,636 recorded vs 22,706,721 on disk, and 8 more) |
| `test_restart_gate::test_the_shipped_engine_is_ready_to_be_restarted_and_has_not_been` | DATA, MODEL and REPRODUCIBILITY gates all fail: `config_version` moved `baseline-v2@25d9176dacd25857 → baseline-v2@6828f4d19d68a1ac` |
| `test_no_dead_modules::test_every_module_is_imported_by_something` | the same 2 orphans this document's AST walk found, 384 lines |
| `test_remediation_guards::…does_not_claim_a_table_describes_what_ships` | `README.md` no longer carries the mandated "NEITHER TABLE IN THIS FILE DESCRIBES WHAT SHIPS" disclaimer |

The restart-gate failure was **not** what it said. Its "configuration changed" message was an artifact of the identity split in §3.5; no configuration had changed. That is why a guard's message matters as much as its trigger — this one sent the reader looking for a parameter edit that did not exist.

**One ingest on 2026-09-07 invalidated four things at once**: the store manifest, the model identity, the epoch, and the research panel. They are not four bugs — `config_version = label@XOR(params_hash, store_hash, train_hash)` (`config/identity.py:346`), so a routine data pull moves the model's identity by construction. No parameter was edited.

The guards all fired correctly. **The defect is that recovery is entirely manual**, and this is the second recurrence of the panel half of it. Under §32 and §46 an ingest must either re-manifest, re-stamp and open a new epoch atomically, or refuse to land.

### 8.4 The panel rebuild moved the evidence base

Rebuilt 2026-09-07 against the re-manifested store, through `validation.v3_panel.build_v3_panel`:

| | shipped panel | rebuilt panel |
|---|---|---|
| rows | 204,425 | **197,940** |
| signal dates | 380 | **356** |
| first date | 2018-11-27 | **2019-05-23** |
| quality coverage | 49.0% | 46.7% |

**The start moved forward by 120 sessions, and that is not a data problem — it is the shipped panel being wrong.** `2018-11-27` is session **301** of the store; `2019-05-23` is session **421**. `v3_factors.LOOKBACK_SESSIONS` is 420, and `FRAME_SESSIONS` is 436, because `resid_rev_21` is a six-stage rolling chain reaching ~375 sessions behind the decision row. The module's own measurement, recorded in its docstring: at 315 sessions `resid_rev_21` "was wrong by up to 4.5e-2 on the last row, and up to 0.507 on another date" — and because every stage relaxes on `min_periods`, a short frame does not produce `NaN`, it produces **a different number, silently**.

So the shipped panel scored its first ~24 signal dates on frames roughly 119 sessions too shallow, with a reversal factor that was wrong rather than missing. Every figure in Part 3.1 was computed on that panel.

It also cannot have been built by `research/v3/experiments/build_panel.py`, because that script has **never run** — it was committed on 2026-09-05 importing `build_panel` from a module that exports `build_v3_panel`, and fails at import. Repaired in this phase.

**And there are two different frames both called "the v3 panel".** `validation.results.build` — which generates `RESULTS_OF_RECORD.md` — calls `build_v3_panel(horizons=(21, 42, horizon))` where `horizon` is 63. `build_panel.py` called it with no horizons at all, taking the default `(LABEL_HORIZON_SESSIONS, 42)` = **(21, 42)**. So the experiments panel has **no column for the 63 sessions the book actually trades**, which is the horizon of every headline row in the results document.

`research results --panel-cache` accepts any parquet. Handing it the experiments panel produced a results document silently missing its own headline horizon, because an absent label column reads exactly like a horizon that scored no dates. Both are fixed in this phase: `build_panel.py` now takes its horizons from `stage4_core_score.model_horizon_sessions`, and `results.build` **refuses** a panel that lacks any horizon it reports.

(`research_panel.build_research_panel` is a genuinely different object — the *fitted* model's feature panel, used for CPCV of the fitted composite — and is not a duplicate of these.)

**Consequence: every number in Part 3.1 and in `docs/RESULTS_OF_RECORD.md` is superseded and must be regenerated.** They are retained above as the claim under test, not as the finding. The §3.3 arithmetic is unaffected — it reads shipped config values and needs no panel.

### 8.5 First actions before any measurement

1. **Re-manifest the store and open a new epoch**, or re-ingest to the manifested state. Until then no number this engine produces is attributable.
2. **Rebuild the research panel** — 4 of 24 inputs drifted since it was built (`prices/year=2026`, `delivery/year=2026`, `fundamentals.parquet`, `sector_map.parquet`). `_panel_guard.require_fresh()` refuses by default, correctly. **Every panel-derived number in this document inherits that staleness and must be re-run**, including Part 3.
3. **Make ingest → manifest → panel → epoch one transaction.** A guard that refuses is necessary and not sufficient.
4. **Add the missing test:** assert total invested capital across a full book is within the intended band. The defect in §3.3 survived 1,835 passing tests because nothing asserted on the aggregate.
5. **Restore the README disclaimer**, or finish the README rewrite the test is guarding.

---

## PART 9 — IMPLEMENTATION ORDER

| phase | work | gate to pass |
|---|---|---|
| **3** ✅ | Re-manifest + new epoch; fix the identity split (§3.5); make ingest settle manifest and epoch; repair `build_panel.py`; add the invested-capital test | MODEL gate green; identity is caller-independent; `test_invested_capital` pins 18.9% and xfails the acceptance gate |
| **4** ← *was 5* ✅ | **Replace the book**: breadth 6 → 50 names and equal-weight set-level sizing, in one change. `pipeline._size_the_book` calls `book.size_book` after selection | acceptance gate now measures the book, not the sizer, and passes |
| **5** ← *was 6* | Delete the family block and the four dead ranking sources; one canonical scorer | scan-path LOC down ~40%; parity test green |
| **6** ← *was 4* | Factor work: ablate `risk` (§5.2.1), delete `quality`, re-test `ownership` standalone, prune within momentum for simplicity | **blocked until a clean out-of-sample surface exists** — see the note below |
| **7** | Collapse config from 199 to ~60 parameters; every one with a research basis and a sensitivity number | `config show` fits on one screen |
| **8** | Ledger v2: record the held book, not just the shortlist; resolve outcomes against the equal-weight universe | a signal is reconstructible from its ID alone |
| **9** | UI: one button, ranked decile list, honest confidence statement | — |
| **10** | Adversarial audit (§42) against the rebuilt engine | — |

**What Phase 4 actually changed.** Four parameters and one new step:

| parameter | was | now |
|---|---|---|
| `capital.max_open_positions` | 6 | **50** |
| `stage6_entry.admission.entry_rank` | 6 | **50** |
| `stage6_entry.admission.exit_rank` | 18 | **106** |
| `stage8_final_signal.portfolio.max_signals_per_run` | 8 | **50** |

Each sat outside its own declared `search_range`, and the loader refused all four — "the engine will not guess". The ranges were widened deliberately, which is the intended way to make a change of this size visible in the diff rather than in a value.

**Selection was left exactly where it was.** `stage6_entry._admit` is a correct per-name hysteresis band and `stage8_final_signal` fills held names before fresh ones, so incumbents already had slot priority. Neither has the truncation defect that made this document's own first `build_book` buffer inert. Adding a second selector would have introduced the one-concept-two-implementations failure catalogued throughout this document, so `book.py` exposes `size_book` (sizing only, used by the pipeline) and keeps `build_book` (selection + sizing) for research that must run outside a pipeline.

**An honest observation that reduces this change's own importance.** At 50 slots the capital slot is 2% of capital and the risk-based size is 2.9–5.0%, so the *slot* now binds and stage 7's original sizer would leave only ~3% in cash. **The breadth change alone recovers most of the exposure.** `size_book` still earns its place — it makes the weighting equal by construction rather than by whichever constraint happens to be tighter this month, and it redistributes a binding liquidity cap instead of banking it — but the headline recovery belongs to breadth, not to the sizer.

**What the first real run found, that 1,907 unit tests did not.** Phase 4 was verified end-to-end with `prosignal analyse run`, and two things surfaced immediately:

1. **The engine cannot run live at all.** `LEDGER_AMBIGUOUS`: **2026-09-03 carries 18 `live` rows, all logged 2026-09-06**, disagreeing about the open book across four versions. The open book is the engine's entire position memory, so the guard refuses rather than guessing. This is pre-existing and **unresolved** — choosing among 18 contradictory position histories decides which trades the engine believes it holds, which is an operator's decision, not a code one. Verification used a replay on a clean date instead.

2. **`max_signals_per_sector` was 2, and would have capped a 50-name book at ~2 per sector** — the target size was unreachable. It is a *book-size* constraint expressed per sector, so it had to move with the book, and changing breadth without checking it was a §61 propagation failure. Grounded on the real 2026-08-03 cross-section: a top-50 spans **19 of 23 sectors** with at most **6** in any genuine one. Raised to 6, range widened to [1, 15].

   Recorded rather than worked around: the largest bucket in that top-50 was **`UNCLASSIFIED`, with 9 names**. The cap treats a hole in `sector_map.parquet` as if it were a sector, so a data gap becomes a concentration limit on names that are in different real sectors. That wants the sector map fixed, not the cap.

**The result, from the engine rather than from a test:**

```
Book: 40 names, equal weight, 97.4% of capital deployed.
40 BUY / 0 WATCH
```

against **6 names and 18.9% deployed** before. Forty rather than fifty because the sector cap and the eligibility gates bind — which is the honest number and is reported as such.

**Why the factor work moved from 4 to 6.** §3.1 establishes that the trial denominator is unknown and unrecoverable, so an in-sample ablation of a v3 theme spends a charged trial and settles nothing — it adds to *N* while measuring against a surface v3 was already selected on. The book replacement, by contrast, needs **no statistics at all**: §3.3 is arithmetic over shipped config values, and its acceptance gate (beta 0.85–1.05, a full book actually invested) is a property of the construction rather than a claim about returns.

**Do the work that needs no evidence first, then buy evidence with what remains.** The P3 allocation is 12 trials and the honest floor is already ≥5,140; spending any of it on questions the current data cannot answer is the most expensive mistake still available in this project.

---

## PART 10 — WHAT REMAINS UNCERTAIN

Stated explicitly rather than buried.

1. **31 independent observations.** No amount of engineering creates statistical power. Every conclusion here is bounded by that number, and the corrected *t* of +3.11 at h=63 is not a comfortable margin against 119 trials.
2. **PIT index membership begins 2026-08-14.** Everything earlier is reconstructed. The survivorship bound (2.4% relative IC haircut) is reassuring but it is a bound, not a measurement of the actual universe.
3. **The panel overlaps its own selection surfaces.** Neither the RESULTS_OF_RECORD arms nor anything in Part 3 is out-of-sample evidence. Both sealed windows are spent.
4. **`ownership` at 18.9% weight has no external corroboration.** If it fails a standalone t>3 test it takes a fifth of the model's weight with it, and the remaining model has not been measured.
5. **Indian anomalies may be fading** (Sharma et al. 2021). A model fitted 2018-11-27 → 2024-10-25 and not refitted since is exposed to exactly that.

---

*This document supersedes the architectural claims in README.md. Numerical claims are subordinate to `docs/RESULTS_OF_RECORD.md`, which is generated — and which must itself be regenerated after the panel rebuild in Phase 3.*
