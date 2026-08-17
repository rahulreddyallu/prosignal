# Build Log

Chunk-by-chunk record of what exists, what was verified, what was deliberately
not built, and exactly where to resume.

---

## CHUNK 1 — Foundation, Config System & Point-in-Time Data Layer

**Status: COMPLETE and running against live NSE data.**

### What you can do right now

```bash
.venv/bin/prosignal config validate
```

```bash
.venv/bin/prosignal config show --unvalidated-only
```

```bash
.venv/bin/prosignal data ingest --full
```

```bash
.venv/bin/prosignal data status
```

```bash
.venv/bin/prosignal data check
```

```bash
.venv/bin/python -m pytest tests/ -q
```

### Delivered

| Area | Files | Lines |
|---|---|---|
| Parameter file (the only file you edit) | `config/parameters.yaml` | 1,457 |
| Config schema + loader | `src/prosignal/config/` | 1,627 |
| Core primitives (contracts, calendar, enums, errors, logging, paths) | `src/prosignal/core/` | 1,564 |
| Data layer (store, universe, corporate actions, types, ingest) | `src/prosignal/data/` | 2,375 |
| Providers (NSE archives, yfinance, CSV import, HTTP) | `src/prosignal/data/providers/` | 1,485 |
| CLI | `src/prosignal/cli.py` | 501 |
| Tests | `tests/` | 1,130 |
| Docs | `README.md`, this file | ~500 |

**~10,400 lines. 91 tests passing. Zero known failures.**

### Proven working, end to end

A live ingest produced:

- decision date resolved to **2026-08-14** (Monday 17 Aug has no bhavcopy yet —
  the resolver snapped back to the last real session, which is the correct
  behaviour, not a bug);
- **200-name NIFTY 200 universe** with sector labels;
- **3,464 symbols** of OHLCV for the session, **41,725 rows** across 12 sessions
  on the first smoke run;
- **164 index series** including **India VIX** — one file per session covers
  Nifty 50, Nifty 200, every sector index, and VIX;
- **3,296 symbols** with delivery quantity and percentage;
- **208 underlyings** of F&O open interest;
- 5,356 corporate actions and 4,379 earnings dates;
- a re-run served **entirely from cache: 0 network requests**.

### Verified data-source facts (do not re-probe these)

| Endpoint | Result |
|---|---|
| `nsearchives.nseindia.com` | **200 OK**, no auth |
| `archives.nseindia.com` | **200 OK**, no auth |
| `www.nseindia.com` (JSON API) | **403 — bot-shielded.** Implemented as best-effort; no required feed depends on it |
| `ind_close_all_DDMMYYYY.csv` | one file per session, all indices **+ India VIX**. This single file is the whole Stage 2 input set |
| `sec_bhavdata_full_DDMMYYYY.csv` | adds `DELIV_QTY` / `DELIV_PER` and `AVG_PRICE` (an honest daily VWAP proxy) |
| F&O bhavcopy | `FinInstrmTp` values are `STF` (stock futures), `STO`, `IDF`, `IDO`. Filter `STF`, sum OI across expiries |
| `EQUITY_L.csv` | every symbol with `DATE OF LISTING` — the survivorship anchor |
| yfinance | `^NSEI`, `^CNX200`, `^INDIAVIX` all work; Indian **bonus issues appear in `Stock Splits`** (Reliance Oct-2024 1:1 → ratio 2.0); `get_earnings_dates` needs `lxml` |
| **Gotcha** | several NSE CSVs pad values with a **leading space** (`' 14-Aug-2026'`). Passing those to `pd.to_datetime(format=...)` yields NaT for every row and silently drops the entire feed. `_parse_date()` in `nse_archives.py` strips first — this bug cost the delivery feed on the first run |

### Two real defects the build caught

**1. Delivery feed silently empty.** Leading-space dates → every row NaT → feed
looked "missing" rather than "misparsed". Fixed with a defensive
`_parse_date()` used for every fixed-format NSE date column.

**2. Unexplained-jump detector was miscalibrated.** The candidate "clean split
ratio" set was generated from every `a/b` up to 12, making it so dense
(~1% average spacing) that a 3% tolerance matched almost any large move — the
check would have fired on ordinary volatility and been trained-out by the
operator. Replaced with a curated set of ratios Indian issuers actually declare,
and the shipped tolerance was tightened from 3% to **1.5%**, which must stay
below half the minimum candidate gap (~4.2%).
`tests/test_corporate_actions.py::test_shipped_tolerance_cannot_bridge_two_candidate_ratios`
now enforces that relationship against the live config, so the calibration
cannot silently rot.

After the fix, on real data the detector flags exactly one name —
`IVZINNIFTY`, 0.1007× overnight against a clean 10:1 split factor with no
corporate action on file. That is a true positive. Untreated it reads as a −90%
single-session return and poisons a 12-1 momentum score for a year.

### Deliberate decisions worth remembering

- **The trading calendar is discovered, never hardcoded.** A 404 on the daily
  index file means "no session". Only fixed-date holidays (26 Jan, 15 Aug,
  2 Oct, 25 Dec, 1 May) are hinted, purely to skip pointless probes. Lunar
  holidays are learned from data, because guessing them wrong is worse than not
  guessing.
- **Every lookback is in sessions, never calendar days**, so a Diwali week
  cannot quietly change a window length or trip the staleness gate.
- **No forward-fill anywhere.** Gaps stay NaN so Stage 1's continuity check can
  see them.
- **Missing data ≠ passing check.** Pledging and fundamentals have no reliable
  free India source, so they report `NOT_TESTABLE`.
- **`equal_weight` is the shipped default** for Stage 4, per the research
  program's own prediction that rank-IC weighting's in-sample gains are usually
  overfitting.
- **Store writes are atomic and idempotent.** `.tmp` + `os.replace`; duplicate
  `(symbol, date)` rows are impossible by construction.

### Config as shipped

**182 parameters. 132 `UNVALIDATED`, 28 `OPERATIONAL`, 15 `STRUCTURAL`,
7 `STATUTORY`.** The large unvalidated count is correct and intended: it is the
honest state of a system whose CPCV harness has not been run.

The loader rejects, at startup: unknown keys; values outside their declared
`search_range`; `VALIDATED` without a ledger trial id and date; history shorter
than the longest lookback any stage needs; CPCV purge shorter than the label
horizon; a non-zero weight on the estimate-revision factor; relaxing the
earnings hard-reject without enabling the PEAD flag; and any attempt to set
`api.allow_order_placement: true`.

---

## CHUNK 1b — Disk-exhaustion post-mortem, storage discipline, and the
## anti-overfitting framework

**Status: COMPLETE. 132 tests passing.**

The first full backfill filled the machine's disk. This section is the
post-mortem and the fixes, because the same failure would otherwise recur at a
larger scale during CPCV.

### What actually consumed the disk

Measured, not guessed:

| Source | Size | Whose |
|---|---|---|
| `~/Downloads` | **80 GB** | pre-existing |
| `~/Library` | 19 GB | pre-existing |
| `~/.npm` | 3.1 GB | pre-existing |
| `~/.git` orphaned objects | **536 MB** | **my error** |
| `data/cache` (raw NSE payloads) | 235 MB | this project |
| `.venv` | 242 MB | this project |
| `data/curated` (the actual data) | 24 MB | this project |

The volume was already at **191 GB used of 228 GB (99%)** before this project
existed. That is the dominant fact. But two things were genuinely wrong on our
side:

**1. `git add -A` was run in a repository rooted at the home directory.** The
`git status` available at the start showed paths like `../../.zshrc`, which
should have been read as "the repo root is `~`, not the project". It wrote
4,290 orphaned objects (536 MB) before dying on ENOSPC. Recovered in full:
0 commits and 0 staged entries existed, so every object was unreachable
garbage. `git gc --prune=now` took `~/.git` from 536 MB to 84 KB.

**2. The raw cache was 7.4× larger than the curated data it produced**, and
70% of it was a single feed:

| feed | raw MB/session | what we keep |
|---|---|---|
| **F&O bhavcopy** | **1.324** | 208 rows of aggregated OI (~5 KB) — a **280:1** waste ratio |
| delivery | 0.366 | ~3,300 rows |
| CM bhavcopy | 0.190 | ~3,464 rows |
| ind_close_all | 0.016 | 164 rows |
| **raw total** | **1.896** | |
| **curated total** | **0.255** | |

The F&O file contains every option strike for every underlying; the engine uses
only the stock-futures rows, for a check the research program rates ●○○
practitioner-grade.

### How much data is actually required

The binding lookback is **273 sessions** (12-1 momentum: 252 + 21 skip).
Configured minimum is 300, plus a 30-session warm-up buffer.

| purpose | sessions | curated |
|---|---|---|
| live signals | 330 | 61 MB |
| + 18-month sacred holdout | 708 | 131 MB |
| CPCV, 5 years | 1,250 | 232 MB |
| CPCV, 10 years | 2,500 | **464 MB** |

Ten years of everything fits in under half a gigabyte. The data was never the
problem; the caching policy was.

### Fixes shipped

- **Storage budget in config** (`storage:` section): `max_total_mb`, a raw-cache
  cap with LRU eviction, `warn_free_disk_mb` / `halt_free_disk_mb` interlocks.
- **Never-cache policy** for parse-once feeds. F&O payloads are now parsed and
  discarded. URL markers are *derived from the configured paths*, so renaming
  an endpoint cannot leave the sweep silently matching nothing.
- **Retroactive policy sweep.** Policy applies on write, so 128 F&O files
  cached under the old policy were still being served and never re-evaluated.
  `prosignal data gc` reclaimed **169.5 MB**; `data/` went 266.9 MB → 97.4 MB.
- **Batched parquet writes** (`storage.write_batch_sessions: 25`). Writing one
  session at a time forced a full read-modify-write of the year file on every
  iteration — O(n²) disk I/O across a backfill, and the main reason the first
  one crawled.
- **Disk preflight**, re-checked every batch. The ingest now refuses to start
  or continue below the floor rather than taking the machine down with it.
- **Explicit three-tier storage separation**: TRANSIENT (`data/cache`, LRU,
  disposable) / AUDIT (`data/raw`, optional rolling window) / DURABLE
  (`data/curated`, `data/snapshots`, `data/ledger` — the record).

New commands: `prosignal data budget`, `prosignal data gc`.

### Parameter classification — the anti-overfitting framework

All 182 parameters are now classified into an **optimisation tier**, which is
a different question from evidence status. "UNVALIDATED" says a value has not
been tested; it does *not* say it should be searched. Conflating the two is
how you end up with a 132-dimensional search space.

| tier | count | meaning |
|---|---|---|
| **A_SEARCH** | **5** | genuinely changes the edge; enters the CPCV grid; charged to the DSR trial count |
| **B_SENSITIVITY** | 128 | perturbed to prove the result is not knife-edge; **never selected on** |
| **C_FIXED** | 22 | evidence or convention (12-1 construction, statutory rates); never searched |
| **D_OPERATIONAL** | 27 | your business constraint, not a research parameter |

The five that earned Tier A, each traceable to a named experiment or an
explicit statement in the research program:

| parameter | pts | why |
|---|---|---|
| `capital.max_participation_of_adtv` | 4 | §9 — capacity vs market impact is the test the whole solo-founder thesis lives on |
| `stage7_risk.stop_loss.atr_multiple` | 4 | §7.1 — Kaminski & Lo establish *when* stops help, not a multiplier |
| `capital.max_open_positions` | 3 | Experiment #7 — concentration vs diversification |
| `stage7_risk.holding_period.max_holding_sessions` | 3 | Experiment #9 — calendar vs signal-deterioration exit |
| `stage8_final_signal.scarcity.min_composite_score` | 3 | selectivity vs opportunity |

**The number that matters: 4×4×3×3×3 = 432 configurations.** A 3-point sweep of
all 132 unvalidated parameters would be **9.55 × 10⁶²** — not expensive,
arithmetically impossible, with PBO ≈ 1.

Deliberately *not* promoted, and why: the 12-1 momentum construction is
STRUCTURAL — Experiment #11 tests alternative *constructions* as discrete
experiments, not as a grid sweep of a parameter, and the loader now refuses
`STRUCTURAL` + `A_SEARCH` as contradictory. Same for `weighting_mode`
(Experiment #4) and the regime multiplier table: these are discrete hypotheses,
not continuous knobs.

Enforcement is at config-load time, not by convention:

- more than `max_tier_a_parameters` (6) promoted → refuse to load;
- grid product over `max_grid_configurations` (512) → refuse to load;
- `A_SEARCH` without a `search_range` → refuse (cannot search an unbounded
  parameter);
- `search_points` outside 2–9 → refuse (a finer grid buys precision the data
  cannot support);
- `STATUTORY`/`STRUCTURAL` marked `A_SEARCH` → refuse.

New command: `prosignal config tiers`.

### Out-of-sample framework — `src/prosignal/validation/`

Real implementations, 28 tests, no scipy dependency.

- **`cpcv.py`** — Combinatorial Purged CV. `C(N,k)` splits, label-overlap
  purging, and post-block embargo. Tests assert what makes it *mean* something:
  train and test never intersect; every surviving pre-block training row ends
  its label window strictly before the test block begins; embargoed rows are
  genuinely gone; adjacent test groups merge into one contiguous block. With
  the shipped config: 45 splits, 9 backtest paths, 19,440 total model fits.
- **`metrics.py`** — PBO via combinatorially symmetric CV, and the Deflated
  Sharpe Ratio (PSR against the expected-maximum-Sharpe-under-null benchmark,
  corrected for skew and kurtosis). Tests confirm PBO > 0.35 when every
  configuration is noise, PBO < 0.1 when one genuinely dominates, and DSR
  collapsing as the trial count rises.

PBO's interpretation string states the rule plainly, because it is the one most
often broken in practice: a high PBO means **simplify the model**, not keep
searching until something scores better.

---

## START HERE TOMORROW — Chunk 2

**Goal: Stage 1 (Data Quality Gate) + Stage 2 (Market Regime Engine), plus the
indicator library both depend on.**

Everything needed already exists. No new data sources, no new providers.

### Step 0 — confirm the ground is solid (2 minutes)

```bash
.venv/bin/python -m pytest tests/ -q && .venv/bin/prosignal data status
```

If the full backfill was interrupted, resume it — it is incremental and
cache-backed, so re-running costs almost nothing:

```bash
.venv/bin/prosignal data ingest --full
```

### Step 1 — `src/prosignal/indicators/` (new package)

Pure functions over pandas, no config reads, no I/O. Everything below is needed
by Stage 2 and reused by Stages 4–7.

- `returns.py` — simple/log returns, cumulative return over a session window,
  the 12-1 skip-month construction, rolling realised volatility.
- `moving_averages.py` — SMA, EMA, and distance-from-MA in percent and in ATRs.
- `atr.py` — true range, Wilder ATR **and** SMA ATR (config selects; Wilder is
  the default and `STRUCTURAL`).
- `trend.py` — OLS log-price slope annualised over a session window.
- `crosssection.py` — winsorise, z-score, cross-sectional rank to `[0, 1]`,
  sector-neutral demeaning, Spearman correlation matrix (Stage 4's redundancy
  check needs this).
- `stats.py` — rolling percentile of a value within its own trailing
  distribution (the India VIX tercile needs it), rolling sigma of returns.

Write `tests/test_indicators.py` alongside — hand-computed expected values, not
"whatever the code returns".

### Step 2 — `src/prosignal/stages/stage1_data_quality.py`

Signature: `run(manifest, store, calendar, universe, config) -> DataQualityReport`
(contract already defined in `core/contracts.py`).

Checks, each binary — never blended into a score:

1. **Staleness** — every feed's `age_sessions` vs `feeds.<name>.max_age_sessions`.
   Any *required* feed stale → market-wide FAIL → raise `MarketWideHalt`.
2. **Universe-wide failure fraction** — if more than
   `max_universe_failure_fraction` of names fail stock-level checks, the *feed*
   is broken, not the stocks. Halt.
3. **Cross-source agreement** — NSE close vs the `prices_secondary` table
   (yfinance), in bps. Beyond tolerance → soft flag (or reject, per
   `source_disagreement_action`). Never silently pick a source.
4. **Outlier / bad tick** — return beyond `outlier_return_sigma` of the stock's
   own trailing distribution, *or* beyond `outlier_absolute_return_pct`, with no
   corroborating volume and no corporate action → hard-reject that stock.
5. **Corporate-action adjustment** — call `detect_unexplained_jumps()` (already
   written and calibrated). Any hit → hard-reject that stock.
6. **Continuity** — more than `max_consecutive_missing_sessions` gaps inside
   `continuity_window_sessions` → exclude.
7. **Point-in-time audit** — populate `pit_audit` / `pit_audit_failures` from
   `manifest.survivorship_risk` and the `stage1_data_quality.pit_audit` switches.

Note: `MarketWideHalt` already exists in `core/errors.py` and carries the
reason list the webapp's FR-9 needs.

### Step 3 — `src/prosignal/stages/stage2_regime.py`

Signature: `run(store, calendar, eligible_symbols, config) -> RegimeState`
(contract already defined).

All inputs come from `store.index_series(...)`, which is already populated:

- **Trend** — Nifty 200 vs 50-DMA and 200-DMA, plus annualised OLS slope; flat
  band → `Range-bound`.
- **Volatility** — India VIX tercile against its own trailing 252-session
  distribution. Then the **asymmetric split** (Thenmozhi & Chandra 2013):
  compute VIX rate-of-change and the index move over the same window, and map to
  `rising-in-decline` / `rising-in-rally` / `falling` / `stable`. Set
  `vol_signal_confidence` from `asymmetric_confidence` — a rising-VIX read is
  more reliable than a falling-VIX all-clear (G.C. & Kothari 2016).
- **Breadth** — % of the eligible universe above its own 200-DMA, plus the
  divergence flag (index makes a new N-session high, breadth does not).
  Market-level only. It must never reach a stock-level score.
- **Transition** — compare the current read against the trailing
  `lookback_sessions` read; if at least `min_components_disagreeing` of
  {trend, vol, breadth} disagree, set `transition_flag` and apply `dampener`
  **market-wide**, not just to the flagged name.
- **Bucket** — `f"{trend.bucket_key}_{vol.bucket_key}"`, with the special
  `uptrend_highvol_rebound` case when the Daniel & Moskowitz crash signature
  fires (prior decline + high vol + sharp rebound). Look the multipliers up in
  `stage2_regime.multipliers.table`; apply `weak_breadth_momentum_penalty` when
  breadth is weak; set `allow_new_entries=False` when the bucket is in
  `no_new_entry_buckets`.

`RegimeState.compatibility()` is already written and maps the multipliers onto
the card's Favorable / Neutral / Unfavorable line.

### Step 4 — wire up and verify

Add `prosignal analyse regime --date YYYY-MM-DD` to the CLI so Stage 2 can be
inspected on its own. That command is also what the webapp's `GET /regime`
endpoint (FR-1, the always-visible regime strip) will call later.

Sanity check against reality: with roughly a year of data loaded, print the
regime for the last ~20 sessions and confirm the buckets move sensibly rather
than flapping every session. Flapping means the transition detector or the
tercile lookback needs attention — and that is a finding to log, not a number
to tune until it looks nice.

---

## Chunk roadmap

| # | Scope | Status |
|---|---|---|
| 1 | Foundation, config, Stage 0 data layer | **done** |
| 2 | Indicator library, Stage 1 data-quality gate, Stage 2 regime engine | next |
| 3 | Stage 3 eligibility, Stage 4 core score + redundancy check | |
| 4 | Stage 5 false-signal defense matrix (the largest single stage) | |
| 5 | Stage 6 entry confirmation, Stage 7 risk/position engine | |
| 6 | Stage 8 final signal, recommendation formatter, research ledger | |
| 7 | FastAPI `/run-analysis`, `/regime`, `/ledger`, `/config`; pipeline orchestrator | |
| 8 | One-button webapp (regime strip, stage progress, cards, NO TRADE, history, config panel) | |
| 9 | CPCV / PBO / DSR validation harness + cost & market-impact model | |
| 10 | Backtest driver, monitoring/drift tracking, hardening | |

---

## Standing constraints — do not violate in any later chunk

1. No new hardcoded numbers in core modules. Everything goes in
   `config/parameters.yaml` with a `status` tag.
2. No fabricated precision in any output field. Bands and gates only, until CPCV
   produces something real.
3. `NO TRADE` is a first-class output with closest-candidate detail.
4. A check that cannot run reports `NOT_TESTABLE`. It never returns `PASS`.
5. Hard rejects fire before score penalties; eligibility runs before scoring.
6. Every run is written to the append-only ledger, signal or not.
7. No order-routing code. Ever.
