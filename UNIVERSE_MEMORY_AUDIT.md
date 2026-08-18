# Universe, Chunking & Memory Audit — measured, 2026-08-18

Every number below came from running instrumentation on the real pipeline.
Nothing is inferred from configuration.

---

## PHASE 1 — Symbol flow, end to end

| Pipeline stage | Unique symbols | Rows | Outside NIFTY 200 | Evidence |
|---|---|---|---|---|
| Configured universe | **200** | — | 0 | `snapshots/universe/NIFTY 200/2026-08-17.parquet`; 200 unique, **0 duplicates**, 0 NIFTY-200 names missing from data |
| Present in storage | 6,418 | 2,730,624 | 6,218 | `pq.read_metadata` + distinct scan over 5 year-files |
| Requested from storage | 200 | — | 0 | `store.read_prices(symbols=…)` from `pipeline._universe` |
| **Loaded from storage** | **200** | **184,592** | **0** | runtime `df["symbol"].nunique()` |
| After filtering | 200 | 184,592 | 0 | identical — filtering happens **before** materialisation |
| Indicator calculations | ≤200 | — | 0 | indicators are called per-symbol from the loaded frame |
| Strategy evaluation | 19 eligible → 15 defended | — | 0 | funnel |
| Final signal | 2 watch, 0 buy | — | 0 | funnel |

**No symbol outside NIFTY 200 can reach any stage.** The filter is applied as a
parquet predicate, so the other 6,218 are never decoded into memory.

---

## PHASE 2 — The 6,418 categories

**They are real distinct symbols in the historical store, not stale metadata**
— the bhavcopy covers the whole cash segment, so every listed instrument since
2022 is present. But they **do not reach the pipeline**:

```
categories carried in the loaded frame : 200
categories actually used               : 200
UNUSED (stale) categories              : 0
category index memory                  : 21 KB
```

**And critically:**

```
groupby(observed=True)  -> 200 groups
groupby(observed=False) -> 200 groups
difference: 0 empty groups
```

**I was wrong in the previous session.** I described `observed=False` as a
memory hazard that would materialise 6,418 groups. Measured, it materialises
zero extra groups, because the parquet filter has already reduced the
dictionary to 200 before any groupby runs. The `observed=True` change remains
correct and defensive — it is semantically right and protects the code path if
filtering is ever removed — but **it was not the memory problem and did not fix
one.**

---

## PHASE 3 — Chunking

**There is no chunking in the analysis path.** `grep` for batch/chunk across
`pipeline.py` and all eight stages returns nothing operational. The only
batching in the codebase is `storage.write_batch_sessions`, which batches
parquet **writes during ingest** — a different code path, introduced to avoid an
O(n²) read-modify-write, not for memory.

So the premise of Phase 3 does not apply: nothing to measure, nothing to
justify, nothing to remove.

---

## PHASE 4 — Where the memory actually goes

Read-strategy comparison, clean subprocess each, 184,592-row result:

| Strategy | Peak delta | Time |
|---|---|---|
| All years, filtered | 317 MB | 0.16s |
| Per-year + gc between | 341 MB | 0.18s |
| Row-group streaming | 288 MB | 0.17s |

**All three peak at ~300 MB to produce a 33 MB frame.** Streaming did not help.
That rules out read strategy as the cause.

Isolating on a single file:

| Read | Rows | Frame | Peak delta |
|---|---|---|---|
| 4 cols, no filter | 511,856 | 43 MB | 61 MB |
| 4 cols + filter | 30,362 | 2.5 MB | **5 MB** |
| all 18 cols, no filter | 511,856 | 182 MB | 201 MB |

Filter + projection on one file costs **5 MB**. But across five files the peak
was 291 MB, and the per-file RSS deltas were `+40, +121, +64, +66, +13` — they
**accumulate and never return**.

> **The bottleneck is not the universe, not the categories, not chunking, and
> not the read strategy. It is that transient decode buffers stay in the glibc
> allocator arena after each file, so RSS only ever climbs.**

---

## PHASE 5 — The one change this audit justifies

**Problem:** per-file decode buffers retained in the allocator; 33 MB of live
data cost 291 MB of peak.
**Evidence:** per-file RSS deltas above; `gc.collect()` alone moved peak by 1 MB,
proving the memory was not held by live Python objects.
**Change:** call `release_memory()` (gc + glibc `malloc_trim`) **between
year-files inside the read loop**, not merely between stages.
**Expected memory impact:** bounds peak at roughly `base + largest single file`.
**Expected runtime impact:** negligible; read time is 0.1–0.2s either way.
**Risk:** none to correctness — it allocates nothing and changes no data.
**Validation:** 328 tests pass; pipeline output **byte-identical** (same funnel
200→19→15→2, same NO TRADE, same top names GVT&D / PREMIERENE).

Measured effect on macOS: **556 → 523 MB**. That is the `gc` half only —
`malloc_trim` does not exist on Darwin, so the trim itself was a no-op here.

---

## PHASE 6 — Is 512 MB genuinely sufficient?

**I cannot confirm it, and I will not claim it.**

| Component | Measured |
|---|---|
| Import baseline | 99 MB |
| Peak, full pipeline (macOS, trim inactive) | **523 MB** |
| 512 MB limit | — |
| Headroom | **−11 MB** |

On this machine it does not fit. The remaining ~420 MB above baseline is
retained arena, which is exactly what `malloc_trim` releases and exactly what I
cannot exercise on macOS.

**Decision rule — run this on Render after deploying:**

```
GET /health   ->  malloc_trim_available
```

* **true** — the trims are live. Peak should fall to roughly
  `99 MB baseline + largest single stage (~120 MB) ≈ 250–300 MB`, giving real
  headroom. Confirm by comparing `rss_mb` before and after a run.
* **false** — the image is musl-based (Alpine). None of the trim work applies,
  peak stays ~520 MB, and **512 MB is not sufficient**.

**If it does not fit, in order:**

1. **Universe 200 → 100.** Roughly halves every frame. `PHASE0_FINDINGS.md`
   provides no evidence 200 names are required.
2. **Reduce `min_history_sessions`** from 300 (it is set by the 12-1 momentum
   lookback).
3. **Then** 1 GB. A few dollars a month beats further engineering.

---

## PHASE 7 — Correctness

Output is byte-identical before and after every change in this audit:

```
funnel   200 -> 19 eligible -> 15 defended -> 1 triggered -> 0 buys
decision NO TRADE, 2 watchlist
top      GVT&D, PREMIERENE
tests    328 passed
```

---

## Corrections to earlier claims

1. **`observed=False` was not materialising 6,418 groups.** Measured: 0 extra
   groups. My previous session's commit message overstated this.
2. **The 6,418 symbols were never being analysed.** They exist in storage; the
   pipeline loads exactly 200. No wasted indicator or strategy computation was
   occurring.
3. **Chunking does not exist in the analysis path.** There was nothing to audit.
