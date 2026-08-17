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

## CHUNK 2 — Indicator library, Stage 1 gate, Stage 2 regime engine

**Status: COMPLETE. 242 tests passing (110 new). Verified against 318 sessions
of live NSE data.**

### `src/prosignal/indicators/` — 6 modules, 58 tests

Pure functions over pandas. Nothing in this package imports config, contracts,
or the store, which keeps every function testable against hand-computed values
and means a stage cannot smuggle a threshold in here where it would escape the
parameter inventory.

Two invariants, both enforced by tests rather than by convention:

1. **Point-in-time safety.** The lookahead tests assert the property directly:
   truncating the series must not change any value that was already
   computable. Any indicator that peeks fails immediately. `as_of=` on the
   scalar helpers is asserted to behave exactly like truncation.
2. **Honest windows.** A window of `n` returns `NaN` until it has `n` real
   observations; scalar helpers return `None`, never `0.0`. A missing factor
   must be reported as missing so Stage 4 can renormalise — a zero would read
   as "no momentum", which is a completely different claim.

Details worth knowing, each of which is a real bug in some other codebase:

- `ema` uses `adjust=False`. Pandas' default is a *different estimator* from
  the recursive EMA every charting package means by "50 EMA"; they converge,
  but not before ~`span` observations.
- `wilder_ma(n)` ≡ `ema(span=2n-1)`, and that is asserted. Confusing Wilder
  smoothing with a same-period EMA is why two packages disagree about "the
  14-period ATR" — and it mis-sizes every stop.
- `true_range` returns `NaN` on the first bar, not `high - low`. Seeding a
  recursive average with an understated first value biases it for many
  sessions. Gap handling is tested both ways: on the NSE, results-day and
  block-deal gaps are routine, and a stop sized on the intraday range alone is
  far too tight on exactly the days that matter.
- `momentum_skip` implements 12-1 properly, and a test proves the recent month
  genuinely cannot influence the value. Dropping the skip does not "use more
  data" — short-term reversal dominates the last month, so including it mixes
  two opposing effects and cancels a real edge.
- `distance_from_ma_atr` exists because percent extension is not comparable
  across the universe. 5% means different things for an FMCG major and a
  smallcap that moves 4% a session; normalising by the stock's own ATR is what
  stops a naive screen filling with high-volatility names.
- `sector_neutralise` skips sectors below `min_sector_size`. Demeaning a
  two-stock sector forces one to `+x` and the other to `-x` by construction,
  manufacturing a signal out of arithmetic.

### Stage 1 — `stages/stage1_data_quality.py`, 23 tests

Every check binary and independent; nothing blended into a score, because a
score lets three moderate problems average into an acceptable-looking number.

The market-wide vs per-stock split is the core of the design. If a quarter of
the universe fails the same check on the same session, the feed changed format
— 50 companies did not simultaneously have bad ticks. Treating that as 50
individual exclusions would silently shrink the universe to whatever survived,
which is worse than halting *and invisible*. `MarketWideHalt` is explicitly not
NO TRADE: it means the engine refuses to form an opinion.

The bad-tick check requires a move to be extreme **and** unexplained. Volume is
what separates a real breakout from a fat-fingered print — a genuine 20% move
on results day arrives with a volume surge. Rejecting on size alone would throw
away exactly the moves the engine exists to find, and there is a test for each
direction.

Continuity counts **consecutive** missing sessions, not total. Eight scattered
gaps across a quarter is a data annoyance; eight in a row was a suspension,
which is a different fact about the company.

### Stage 2 — `stages/stage2_regime.py`, 27 tests

Four reads kept deliberately separate so one broken input cannot swing the
whole conclusion.

**Trend** requires the regression slope on log price and the 200-DMA position
to *agree* before making a directional call. This is what stops the flapping
that makes regime engines useless. The test that matters constructs a bear-
market bounce: slope strongly positive, price still below the 200-DMA. Slope
alone calls that an uptrend and gets you long into a bear rally; the engine
reports Range-bound and says why.

**Volatility** reads India VIX as a percentile of its own trailing year, never
an absolute level — it has printed from ~8 to ~87, so "high is above 20" would
call 2017 alarming and April 2020 a relief. Two tests assert the same VIX of 18
maps to HIGH in a calm year and LOW in a violent one. Then the asymmetric split
(Thenmozhi & Chandra): rising-into-decline, rising-into-rally, falling, stable
— with `vol_signal_confidence` carrying the asymmetry explicitly, because a
falling-VIX all-clear is the least reliable read there is.

**Breadth** is market-level only and never reaches a stock-level score. It is
the read that would have flagged 2021-22 on the NSE, when the index was carried
by a handful of heavyweights while participation narrowed underneath. A
cap-weighted index cannot show you that.

**Transition** compares today's read against the read 10 sessions ago. Where
the dampener applies is a deliberate call: momentum and sector-RS are cut,
**quality is left intact**, because quality's documented job is crash
stabilisation and dampening it during a turn removes the exposure most likely
to help. Tagged UNVALIDATED like the rest of the multiplier design.

**Momentum crash** — the Daniel & Moskowitz state gets its own bucket and a
hard entry block, not a trimmed multiplier. The crash does not happen in the
decline; it happens in the violent rebound off the bottom, when the most
beaten-down names rally hardest and everything a momentum screen owns
underperforms at once. March–June 2020 is the textbook instance, and the test
reconstructs that shape.

### Three real defects found by running against live data

1. **Index name case mismatch.** NSE publishes `Nifty 200`; the config says
   `NIFTY 200`. Exact-match lookup returned an empty series and Stage 2
   correctly refused to form a view — a silent production halt. Fixed with a
   case- and whitespace-insensitive resolver in the store, so NSE changing
   capitalisation cannot break callers.

2. **Flat-series percentile returned 100.** Under naive "count values ≤
   current" ranking, every element of a constant series ties, so the rank is
   100. A dead-flat India VIX would have read as *maximum volatility forever*,
   parking the engine in its most defensive bucket during the calmest possible
   market. Fixed by adopting the textbook midpoint convention
   `(below + 0.5·equal) / n`, which returns 50 — today is exactly typical of
   its own history. Regression test pins it.

3. **Empty index series had a `RangeIndex`.** Callers slice these by date, so
   `series.index <= Timestamp` raised an opaque `TypeError`. That turned a
   missing India VIX — a case Stage 2 is explicitly designed to survive with a
   reduced-confidence note — into a crash. Fixed by returning an empty
   `DatetimeIndex` so the type contract holds whether or not data exists.

### One finding logged, not tuned away

Over the last 25 sessions the bucket changed 7 times, with long persistent runs
(`range_midvol` → `uptrend_midvol` for 8 sessions → `uptrend_lowvol`). That is
healthy. But **2026-07-24 shows a single-session flip** to `range_highvol` and
straight back — a one-day VIX spike crossing the tercile boundary.

This is tercile-boundary sensitivity, and it is recorded here rather than
smoothed away. Adding hysteresis would be a parameter chosen to make the output
look tidy, which is precisely the behaviour the research ledger exists to
prevent. If CPCV shows the boundary matters, it gets fixed with evidence.

### New parameter

`stage1_data_quality.min_universe_for_failure_fraction: 20` (STRUCTURAL). The
failure-fraction rule is a claim about a population and needs one to mean
anything — on a 3-name watchlist a single bad tick is a 33% failure rate and
would halt the run. Below this many names the engine excludes individual
stocks and draws no conclusion about the feed.

### Storage discipline validated end to end

The full backfill — the exact operation that filled the disk two days ago — ran
to completion: **318 sessions, `data/` at 230 MB against a 3,072 MB budget,
3.4 GB free throughout.** Batched writes and the never-cache policy held.

### New command

```bash
.venv/bin/prosignal analyse regime --history 25
```

Single-date or N-session history. The history view exists specifically to check
that buckets persist rather than flap, and it prints a warning if the bucket
changes more than once every three sessions. This is also the query behind the
webapp's always-visible regime strip (FR-1).

---

## START HERE TOMORROW — Chunk 3

**Goal: Stage 3 (Eligibility) + Stage 4 (Core Score + redundancy check).**

Everything needed exists. No new data sources.

### Step 0 — confirm the ground (1 minute)

```bash
.venv/bin/python -m pytest tests/ -q && .venv/bin/prosignal analyse regime
```

### Step 1 — `stages/stage3_eligibility.py`

Signature: `run(universe, store, calendar, quality_report, config) -> EligibilityReport`
(contract already defined in `core/contracts.py`).

Hard gates only, all binary, all firing **before** any scoring. Order matters:
a stock excluded here must never reach a stage that could score it well enough
to overcome the exclusion.

1. **Data quality** — anything Stage 1 failed is out. Feed
   `quality_report.failed_tickers()` straight in.
2. **History** — `universe.min_history_sessions` (300). Insufficient history is
   `RejectionReason.INSUFFICIENT_HISTORY`, never a partial-window score.
3. **Liquidity** — ADTV in INR over the configured window, and the
   `max_participation_of_adtv` position-size check. This is Tier-A parameter
   #1; the ADTV figure computed here feeds it directly.
4. **Price floor**, **series allowed** (`EQ` only), **manual exclusions**.
5. **Pledging** — `NOT_TESTABLE` when the CSV is absent. It does NOT pass.
6. **Earnings conflict** — hard reject inside the blackout window; the config
   validator already forbids relaxing this without the PEAD flag.
7. **Regulatory cooldown** — from `regulatory_events`.

Populate `not_testable` verbatim for every check that could not run; it prints
on every card.

### Step 2 — `stages/stage4_core_score.py`

Signature: `run(eligibility, store, calendar, regime, config) -> CoreScoreReport`

Use the cross-section helpers already written and tested:
`winsorise → standardise → sector_neutralise → weight`. That order is not
negotiable and `tests/test_indicators.py` explains why.

- Factors from `stage4_core_score.factors`. **Quality is dropped** — no
  point-in-time fundamentals exist, Stage 1 already records
  `fundamentals_filing_date: False`, and the remaining weights renormalise.
  State it on every card.
- Apply the Stage 2 multipliers per factor: `momentum_multiplier`,
  `quality_multiplier`, `sector_rs_multiplier`.
- Map the composite to `[0, 1]` across the eligible universe with
  `rank_to_unit_interval` — every Stage 5/8 threshold operates on that scale.
- **Redundancy check** — `spearman_pairs()` is written and tested. Measure it,
  do not assume it. Log the technical-collapse diagnostic too: RSI / MACD /
  MA-crossover are *expected* to collapse into momentum, and expecting it is
  not the same as verifying it.

### Step 3 — wire and verify

Add `prosignal analyse score --date --top 20`. Then read the top 20 as a
trader: if it is 15 names from two sectors, sector-neutralisation is not doing
its job and that is a finding to log.

---

## Chunk roadmap

| # | Scope | Status |
|---|---|---|
| 1 | Foundation, config, Stage 0 data layer | **done** |
| 2 | Indicator library, Stage 1 data-quality gate, Stage 2 regime engine | **done** |
| 3 | Stage 3 eligibility, Stage 4 core score + redundancy check | next |
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
