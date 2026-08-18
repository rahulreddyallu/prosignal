# Memory optimisation — measured, 2026-08-18

Your 512 MB Render instance was being restarted. Every number below came from
`psutil` on a real run, not from estimation.

## What was actually consuming memory

| Item | Measured |
|---|---|
| Import floor (python + pandas + pyarrow + prosignal) | **99 MB** |
| `read_prices()` unfiltered | 2.73M rows, **972 MB** deep |
| — of which: `symbol` / `series` / `isin` / `source` strings | **681 MB** |
| — `source` alone, with **one** unique value | **180 MB** |
| Peak RSS, full pipeline | **556 MB** |

Two independent problems:

**1. Frames were far larger than they needed to be.** Object-dtype strings
dominated, and every read pulled all 18 columns and every symbol on the
exchange when a stage needed 10 columns and 200 names.

**2. Freed memory was never returned to the OS.** RSS climbed monotonically
206 → 546 MB across six stages even though no stage needs the previous stage's
frames. `gc.collect()` changed it by 1 MB — proving the memory was held in the
glibc allocator arena, not by live Python objects. **RSS is the number Render
kills on.**

## What was changed

### Store reads (`data/store.py`)

| Step | Year-file memory |
|---|---|
| as loaded before | 182 MB |
| + categorical strings | 58 MB (−68%) |
| + only needed columns | 17 MB (−91%) |
| + symbol filter pushed into parquet | **1 MB (−99%)** |

Across all years: **972 MB → 178 MB** unfiltered; **11.8 MB** for a
universe-filtered read.

**Float columns were deliberately left at float64.** Downcasting to float32
bought only another 14 percentage points and would put a precision question
over every price, stop and target. Not worth it.

### A bug this introduced, and fixed

Categorical `groupby` defaults to `observed=False`, which materialises a group
for **every** category — 6,418 symbols — not just those present. Every
`groupby(SYMBOL)` and `pivot_table` now passes `observed=True`.

### Memory release (`core/memory.py`)

`release_memory()` = `gc.collect()` + glibc `malloc_trim(0)`, called between
every pipeline stage and after every job. `malloc_trim` exists only on glibc,
so it is a no-op on macOS and Alpine/musl — the guard is deliberate so the call
is always safe.

### Deployment (`render.yaml`)

- `--workers 1`. Each worker carries its own 99 MB import floor; two would cost
  ~200 MB of 512 before reading any data, and the analysis is single-flight
  anyway so a second adds no throughput.
- `MALLOC_ARENA_MAX=2` — bounds glibc arena fragmentation.
- `PYTHONMALLOC=malloc` — routes through the trimmable allocator.

### Observability

`GET /health` now reports `rss_mb` and `malloc_trim_available`.
`POST /admin/release-memory` trims on demand and reports bytes freed.

## Does it fit in 512 MB? — honest answer

**Probably, and you can now verify it rather than trust me.**

The frame-size work is verified: 972 → 178 MB, output byte-identical (same
funnel, same ranking, 328 tests pass).

The release work is **UNVERIFIED on your platform**. `malloc_trim` does not
exist on macOS, so I could not measure its effect here. The arithmetic says it
should work: with arenas returned between stages, peak becomes roughly
`baseline + largest single stage` ≈ 206 + 103 ≈ **310 MB**, comfortably inside
512. Without it, peak stays ~556 MB and the instance keeps restarting.

**Check it yourself after deploying:**

```
GET /health          -> malloc_trim_available must be true
                        rss_mb before and after a run
POST /analysis/run   -> then GET /health again
```

If `malloc_trim_available` is **false** on Render, the image is musl-based and
the arena fix will not apply — tell me and the answer changes to a bigger
instance.

## If it still does not fit

In order of preference:

1. **Reduce the universe.** 200 → 100 names roughly halves every frame. The
   live evidence (`PHASE0_FINDINGS.md`) does not require 200 names.
2. **Shorten `min_history_sessions`** from 300. It is set by the 12-1 momentum
   lookback; a shorter momentum horizon would cut the window materially.
3. **Then** upgrade the instance. 1 GB removes the question entirely for a few
   dollars a month — cheaper than more engineering time if the above are not
   attractive.

I would not go past step 1 before checking `/health` on the deployed instance.
