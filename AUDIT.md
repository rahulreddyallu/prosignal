# Forensic Audit — 2026-08-17

Evidence-based audit of what this codebase actually does. Every claim below
cites a file, a command, or a measurement. Where I could not verify something,
it says UNVERIFIED.

---

## 1. EXECUTIVE VERDICT

### PARTIALLY VERIFIED — and the system is roughly 25% built

The single most important finding is structural, and it changes what the rest
of this audit can even mean:

**Only 2 of 8 pipeline stages exist.** `ls src/prosignal/stages/` returns
`stage1_data_quality.py` and `stage2_regime.py`. Nothing else.

There is therefore **no stock scoring, no signal generation, no entry price, no
stop loss, no target, no position sizing, no confidence score, no probability,
no BUY/SELL/HOLD decision, and no notification output** anywhere in this
repository. Not broken — *absent*.

This means several sections of the audit brief have no subject matter:

| Audit section | Status |
|---|---|
| Part 3 — trace values into the final signal | **No final signal exists** |
| Part 4 — fake sophistication in scoring | **No scoring code exists** |
| Part 8 — confidence/probability audit | **Nothing outputs a probability.** Nothing to mislabel — the honest outcome, by absence rather than by design |
| Part 7A — look-ahead in backtests | **No backtest driver exists.** Look-ahead verified at the *indicator* level only |
| Part 7C — survivorship in historical performance | **No performance claims exist** |

I will not manufacture findings about code that isn't there. What follows
audits what does exist.

---

## 2. WHAT ACTUALLY WORKS (verified)

| Claim | Evidence |
|---|---|
| NSE archive ingestion is real | 1,028,374 price rows, 46,453 index rows, 978,045 delivery rows in `data/curated/`. `prosignal data ingest` re-runs clean |
| Trading calendar is discovered, not hardcoded | 318 sessions derived from 404-vs-200 on daily index files (`nse_archives.py`) |
| Indicators are point-in-time safe | `tests/test_indicators.py::test_indicator_does_not_peek_into_the_future` asserts the property directly: truncating the series must not change any already-computable value. 8 indicators parameterised + ATR + `as_of` equivalence |
| Store is idempotent and atomic | `.tmp` + `os.replace`; dedup on `(symbol, date)` at write (`store.py:113-136`). `test_data_layer.py` covers double-write |
| No forward-fill anywhere | Verified by inspection of `store.py` and `_PartitionedTable.read`; Stage 1's continuity check depends on gaps staying gaps |
| Survivorship is detected and can halt the run | `universe.py:184` sets `survivorship_risk=True`; `pre_snapshot_policy: halt` refuses to run (`universe.py:101`) |
| Stage 1 gate genuinely fails things | 23 tests construct real defects (unadjusted split, bad tick, suspension, broken feed) and assert FAIL, not PASS |
| Stage 2 regime is coherent on live data | `prosignal analyse regime --history 25`: 7 bucket changes over 25 sessions with persistent runs, transitions at real turning points |
| No silent error swallowing | `grep "except:"` → zero hits. No `except Exception: pass` found |
| Config fails loudly on bad edits | `extra="forbid"` + 26 tests in `test_config.py` |

---

## 3. WHAT DOES NOT WORK / IS MISLEADING

### HIGH — the universe is not what the MVP specifies

The brief requires **Top 250 NSE by market capitalisation**. The system uses
the **NIFTY 200 index** (`universe.index_name: "NIFTY 200"`, snapshot = 200
names).

These are materially different populations, and the substitution is exactly
what the brief said not to do silently. Worse:

**Market cap is not computable from any stored data.** Evidence:

```
equity_master columns: symbol, company_name, series, listing_date,
                       paid_up_value, face_value, isin
prices columns:        date, symbol, series, isin, open, high, low, close,
                       prev_close, last, vwap, volume, turnover, trades, ...
```

No shares-outstanding field exists anywhere. Market cap = price × shares
outstanding, and the second term is absent. A point-in-time Top-250 also needs
*historical* shares outstanding, which changes with buybacks, splits and
issuance.

**Status: UNVERIFIED / BLOCKED.** The Top-250 universe as specified cannot be
built today. See §6 for options.

### HIGH — 57% of config describes a system that does not exist

101 of 178 tunables are never referenced by any runtime code. Measured by
scanning every `src/**/*.py` except `schema.py` (which only declares them):

| Section | Unconsumed |
|---|---|
| stage5_false_signal | 20 of 39 |
| stage7_risk | 18 of 20 |
| stage4_core_score | 15 of 19 |
| costs | 14 of 16 |
| stage3_eligibility | 9 of 10 |
| stage6_entry | 9 of 16 |
| stage8_final_signal | 8 of 9 |

This is not dead code in the usual sense — it is scaffolding for unbuilt
stages. But it is **misleading**: `parameters.yaml` reads as a complete
8-stage trading system, and a reader would reasonably conclude those stages
exist. Kept (they are the specification for chunks 3–6), but now stated
plainly.

### MEDIUM — one module has no tests

`yfinance_provider.py` — the secondary price source, used for cross-source
agreement, corporate actions and earnings dates. No test file references it.
**UNVERIFIED.**

### RESOLVED THIS SESSION — computed but never consumed

F&O open interest: 67,287 rows fetched, parsed, stored, and reported in
`data status` — and **read by no stage**. It could not influence any output. It
cost ~70% of the raw HTTP cache to produce data nothing consumed. Removed (§4).

---

## 4. WHAT WAS REMOVED

| Component | Reason | Dependency evidence | Effect on MVP |
|---|---|---|---|
| F&O open interest (provider method, store table, ingest wiring, feed, config) | Derivatives — outside the equities-only MVP. Computed but consumed by nothing | `grep open_interest src/prosignal/stages/` → no stage reads it | None. No output changes |
| `stage5_false_signal.short_covering` | Its only input was F&O OI | Config-only; Stage 5 does not exist | None |
| `fastapi`, `uvicorn` | Never imported | `grep -rn "fastapi\|uvicorn" src/ tests/` → only egg-info metadata | None. Re-add in chunk 7 when an API actually exists |
| `data/curated/open_interest/` (1.4 MB) | Orphaned data for a removed feed | — | None |
| `estimate_revision_momentum` (earlier this session) | No obtainable data, cannot be derived | `DATA_SOURCES.md` | Factor was already weight-locked at zero |

**Verification after deletion:** 246 tests pass; `data ingest --sessions 5`
runs clean with `fo_open_interest` absent from the feed table; `analyse regime`
produces an identical read (`uptrend_lowvol`, Uptrend, Low/stable, entries
allowed).

A new test, `test_no_derivatives_feed_remains_in_the_mvp_path`, asserts no F&O
can return to the config, provider, or feed list.

### A near-miss worth recording

The regex that removed `short_covering` from YAML also silently deleted
`beta_explained_move` — a pure-equity check (stock returns regressed on the
index; no derivatives). Caught by the config validator refusing to load, and
restored from git. Recorded because it is precisely the collateral damage this
kind of cleanup causes, and the loader's `extra="forbid"` strictness is what
caught it.

---

## 5. WHAT REMAINS — the actual architecture

Only what exists and executes:

```
NSE archives (nsearchives.nseindia.com, no auth)
    |
    +-- bhavcopy (OHLCV)      +-- ind_close_all (indices + India VIX)
    +-- sec_bhavdata (deliv)  +-- EQUITY_L (listing dates)
    v
Stage 0  data/ingest.py           -> RawDataManifest    [VERIFIED]
    v
         DataStore (parquet, atomic, idempotent, no ffill)
    v
Stage 1  stage1_data_quality.py   -> DataQualityReport  [VERIFIED]
         staleness | continuity | outlier | unadjusted action |
         cross-source | PIT audit
    v
Stage 2  stage2_regime.py         -> RegimeState        [VERIFIED]
         trend | volatility | breadth | transition | crash bucket
    v
    ============ PIPELINE ENDS HERE ============
    Stages 3-8 do not exist. No signal is produced.

Supporting, verified but not yet wired into a pipeline:
  indicators/   6 modules, lookahead-tested
  validation/   CPCV + PBO + DSR, 28 tests, no backtest driver calls them yet
```

---

## 6. TOP-250 UNIVERSE — options, since it is currently blocked

The MVP requires it and it cannot be built from stored data. Three honest paths:

1. **NIFTY 200 index (current).** Reproducible, point-in-time via dated
   snapshots, free, no market-cap needed. But it is an index, not a
   market-cap ranking — and it is 200, not 250.
2. **NIFTY Total Market / NIFTY 500 constituents, sliced.** Still index-based.
3. **Compute market cap properly.** Needs shares outstanding per symbol per
   date. NSE's `corporate-share-holdings-master` (now reachable, see
   `DATA_SOURCES.md`) plus `EQUITY_L` face value is a partial path; Upstox
   Company Profile returns current market cap but not historical.

**Recommendation: keep NIFTY 200 for now and state it plainly rather than
call it "Top 250".** Option 3 is the only one that satisfies the brief, and it
is a real piece of work with a genuine point-in-time hazard.

---

## 7. FINAL NUMBERS

| Metric | Before | After |
|---|---|---|
| Python files | 48 | 48 |
| Lines (src+tests) | 15,176 | 15,094 |
| Classes | — | 166 |
| Functions | — | 412 |
| Config tunables | 182 | 178 |
| Dependencies | 12 | 10 |
| Tests | 244 | 246 |
| pyflakes findings | 24 (start of session) | 3 |
| Unconsumed config | 103/182 (57%) | 101/178 (57%) |
| Stages built | 2 of 8 | 2 of 8 |

Line count barely moved. That is the correct outcome: this codebase's problem
was never bloat — it is that it is **incomplete**, and the parts that exist are
mostly real.

---

## 8. CRITICAL RISKS / UNRESOLVED

1. **The system produces no trading signal.** Six stages remain.
2. **Top-250-by-market-cap is not buildable** from current data. UNVERIFIED.
3. **`yfinance_provider` is untested.** UNVERIFIED.
4. **Nothing has been validated.** 127 of 178 parameters are UNVALIDATED. CPCV
   machinery exists but no backtest driver has ever called it. Any performance
   claim about this system today would be fabrication.
5. **NSE JSON API endpoints are undocumented** and bot-shielded; they must stay
   optional feeds (see `DATA_SOURCES.md`).
6. **Look-ahead is verified at indicator level only.** Pipeline-level and
   backtest-level look-ahead cannot be tested until a backtest driver exists.
