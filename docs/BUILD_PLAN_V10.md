# BUILD PLAN V10 — ProSignal to paper-trading readiness

Audit date **2026-09-04**. Store as of **2026-09-03** (2,219 sessions, 2017-09-08 → 2026-09-03).
Config `baseline-v2@74ad0b78ffe70cc0`. Code `1e9cefba57d4`. Tree clean at `afe217b`.

Every finding below is marked **VERIFIED** (something was executed and the result observed)
or **READ** (reasoned from source). Nothing in this file was copied from the README,
CHANGELOG, or a prior audit without re-deriving it. Where a seeded claim was wrong, it is
recorded as refuted, with the number that replaces it.

No source, config, or data file was modified during the audit. `git status` was clean
before and after. The test suite was executed (it writes only to `.pytest_cache`, which is
gitignored); everything else ran through read-only harnesses in `/tmp` that call the stage
functions directly rather than `run_analysis`, because `run_analysis` appends to the ledger.

---

## 0. Status — 2026-09-05

Phases 0, 1 and 2 have been implemented on branch `build-plan-v10-phase0-2`.

| defect | was | now |
|---|---|---|
| D-001 cap defeated by renormalisation | P0 | **CLOSED** — `v3.py:_weights_for_pattern`; momentum 48.55% → **40.00%** (mean and max), variance share 63.59% → 44.4% |
| D-002 quality theme sign-inverted | P0 | **REFUTED** — see below. Downgraded to P3 naming, closed |
| D-003 card weight ≠ contribution | P0 | **CLOSED** — `stage4:548`, `contracts.py:FactorScore.weight`; `z × w == contrib` exactly |
| D-004 screen certifies a deleted model | P0 | **CLOSED** — `viewmodel._scorer_used` keys on `evidence_tier`, not factor names |
| D-005 expectancy from another ranker | P0 | **CLOSED** — `expectancy.enabled: false` until re-measured on `v3_composite` |
| D-008 open book by file-append order | P0 | **CLOSED** — `Ledger.previous_run` is lineage-scoped and refuses ambiguity |
| D-010 `resid_rev_21` under-converged | P1 | **CLOSED** — `LOOKBACK_SESSIONS` 300 → 420, plus a per-name guard |
| D-018 `effective_weights` from dead path | P1 | **CLOSED** — `_reported_weights` |
| D-038 regime note that moves nothing | P2 | **CLOSED** — emitted only on the path where it is true |

### Round 2 — 2026-09-05, ledger contamination and the store's write key

| defect | was | now |
|---|---|---|
| D-008 five contaminated dates | PARTIAL | **CLOSED** — `Ledger.repair_lineage`; 5 conflicts → 0, 1,942 rows in and 1,942 out, trial count unchanged |
| D-013 T0 sessions lost | P1 | **CLOSED for recurrence** — the cause was the price table's write key, not the series filter. Recovering the 235 already-lost rows needs a re-ingest |
| D-032 store constructor mkdir | P2 | **CLOSED** — and `test_adj_factor_without_a_price_column_is_refused` passes for the right reason |

**The ledger was not contaminated in five places. It was mislabelled in 1,365.**
`lineage_audit` reads what the rows' own timestamps prove: of **253 market dates,
6 were ever recorded on the day** — 2026-08-17, 08-18, 08-21, 08-25, 08-28 and
09-02. The other 247 were backfilled later, and every one of them said
`mode: "live"`. The engine's "live record" is six sessions.

Nothing was deleted. Deleting the 1,365 contaminated rows was the obvious move
and the wrong one: the honest trial count feeds the Deflated Sharpe directly and
`Ledger.append` is fatal-on-failure precisely so that count cannot be corrupted.
Rows are relabelled from evidence already in them, backed up first, and
`prosignal data lineage [--repair]` makes it reviewable and re-runnable.

Two of the six live dates disagree with themselves — 2026-08-17 (2 books) and
2026-08-25 (8 books, 8 config versions). Those are **quarantined, not guessed
at**. Adding `config_version` to the lineage key makes it *worse* (6 conflicts,
not 5): the same date under one config produced up to five books, so the code
moved underneath, and `model_fingerprint` — the field that exists to catch
exactly that — is null on 1,733 of 1,942 rows. No recorded property
reconstructs which book was real.

**A third correction to this document.** D-013 said NSE's `T0` rows were
dropped by the store's equity-series *read* filter. The filter is right; the
defect was one level down. `_PartitionedTable` deduplicated the price table on
`(symbol, date)` while NSE publishes several series per symbol per session —
the EQ line, the T0 same-day line, and an issuer's ND/N7 debenture lines, which
`read_prices`' own docstring is about. So two lines for one symbol on one day
collapsed at WRITE time to whichever sorted last, and the read filter could
only choose among the survivors. 235 (symbol, date) pairs are held under T0
with no EQ row, across 59 symbols including SBIN, RELIANCE, HDFCBANK and INFY.
The key is now `(symbol, date, series)`.

The related claim that **61 of the last 305 sessions were partial ingests is
also withdrawn.** It compared each session against a 305-session median while
the listed universe grew from 2,025 names to 2,641. Against a trailing median
the worst session in nine years is 0.923 and none breaches a 0.90 floor. A
completeness check ships anyway, calibrated to that measurement and documented
as never having fired, because a truncated ingest remains a real failure mode.

### Round 3 — 2026-09-05, Phase 4 and a finding of mine that did not survive replay

| defect | was | now |
|---|---|---|
| D-006 NO TRADE unreachable | P0 | **RESTATED and closed.** See below — the claim was too strong |
| D-043 dispersion gate reads a deleted model's field | new, P1 | **CLOSED** — wired to the v3 composite with a measured constant |
| D-044 `_quality_from_features` crashes on any date with public fundamentals | new, P1 | **CLOSED** — would have turned into a daily crash the moment D-011 was fixed |
| D-045 a refusal to rank surfaces as an exception, not a decision | new, P2 | **CLOSED** — `RankingUnavailable` → `PipelineBlocked` |

**D-006 was overstated and the replay proved it.** The finding said the engine
has no state in which it refuses on the evidence. Replaying the engine on its
worst episodes:

| date | eligible | above 200-DMA | buys | NO TRADE | via |
|---|---|---|---|---|---|
| 2020-03-23 | 6 | — | — | — | refused to rank (6 names) |
| 2020-03-31 | 34 | 24 | 0 | yes | **Stage 2 regime gate**, `downtrend_highvol` |
| 2022-06-17 | 74 | 43 | 0 | yes | **Stage 2 regime gate**, `downtrend_highvol` |
| 2026-09-03 | 386 | 309 | 6 | no | — |

**NO TRADE was already reachable.** What is true, and survives, is narrower:
every *scarcity* control was a rank and could not bind (`composite_score` is
`(rank-1)/(n-1)`, so the top decile exists on the worst day in history —
measured live, the score gate rejected 0 of 37), and the one control written to
fix that read a dead field. What was missing was a bar on the **cross-section**
rather than on the index: the regime gate reads NIFTY trend and volatility, so
it says the market is bad, not that these candidates are weak, and the two
disagree exactly when an index is carried by a few large caps while breadth
collapses underneath it.

**Two gates, and only one of them can close a market-quality hole.**

`min_dispersion_ratio` is now wired to the v3 composite with
`TYPICAL_DISPERSION = 0.5732`, measured on 61 sampled training-window dates
(min 0.5010, median 0.5732, max 0.7358 — neither sealed window touched). But it
**cannot fire at the shipped 0.50 ratio and that is not a bar to lower**: a
blend of cross-sectional ranks has a spread bounded by construction, so the
worst training day is 0.874× the median and a simulation puts the day-to-day sd
near 0.027 inside a regime. It detects a degenerate *scorer*, not a bad market,
which is what the config always said it was for. The limit is pinned by a test
so nobody assumes it protects them.

The **book-level cash rule** counts names, not ranks: how many of the eligible
universe close above their 200-session average, which can collapse for the whole
market at once. Measured over 2,018 sessions: min 16, median 319, max 708;
below 18 on exactly one session, 2020-03-23. The threshold is not a new
parameter — it defaults to `exit_rank`, the band the engine already treats as
the set a position may live in.

*Its marginal contribution over the regime gate is unmeasured and probably
small*, because on every crisis date replayed the regime gate fires first. It is
kept as independent evidence — breadth and index disagree in narrow markets —
and its counter is now on every run whether or not it ever binds.

`absolute_floor` was **not** re-enabled to achieve this. It was disabled on a
measured −2.2% ATE (95% CI [−3.2%, −1.4%]) and that measurement was of a
*per-name entry gate*, a different question. `StockScore.absolute_bar_cleared`
records the bar always and gates never, so the book-level question can be asked
without reopening the per-name one.

**A crash that was waiting for the data to be fixed.** Replaying 2020-03-23
found `_quality_from_features` calling `_min_coverage(cfg)` with the *quality
factor's* config, which has no such attribute — every call raised
`AttributeError`. It never fired only because `fundamentals.parquet` has been
frozen since 2025-03-11, so no live date has had public fundamentals to reach
it. **Repairing D-011 would have converted a dead branch into a daily crash.**

### Round 3b — D-011 re-diagnosed: the feed is up and the archive is not advancing

A fourth correction to this document. D-011 said `fundamentals.parquet` is
frozen at 2025-03-11 because the ingest was mis-wired, and named two causes: the
symbol list came from a `NIFTY 200` snapshot rather than the live universe, and
`if not dates: return` exits silently. Both are real and both still want fixing.

Neither is why the data is stale. Probed live through the repo's own client,
`/api/corporates-financial-results?...&period=Quarterly` returns:

| symbol | rows | newest `toDate` |
|---|---|---|
| RELIANCE | 130 | 2024-12-31 |
| TCS | 162 | 2024-12-31 |
| HDFCBANK | 103 | 2024-12-31 |
| SBIN | 145 | 2024-12-31 |
| INFY | 144 | 2024-12-31 |
| LLOYDSME | 14 | 2024-12-31 |

**Uniform, and exactly equal to the store's own `period_end` maximum.** Tried
with `period=Annual`, with `type=Standalone`, and with no period parameter —
same ceiling. The ingest is not failing; it has nothing new to fetch. NSE's
archive on this path stopped advancing after Q3 FY25, and `results_calendar.
parquet` (91,585 rows, 1,288 symbols) shows the same ceiling, which is what a
feed that stopped rather than a parser that broke looks like.

*(An earlier probe in this document reported the ceiling as 2024-03-31. That
sorted `"31-Mar-2024"` as a string. The figure is 2024-12-31.)*

**So the fix is not a re-ingest.** The current quarters exist — they are on the
NSE site — but not behind this endpoint. Two routes, and the second is already
half-built:

1. Find the endpoint that does serve them. The corporate-announcements and XBRL
   index paths are the candidates; both return 200 and neither is tried.
2. **Lean on the secondary source, which already works.** `statements.parquet`
   carries `period_end` to 2026-06-30 and `pit_fundamentals` already consumes it
   as source B with a measured p99 disclosure lag per quarter-end month. It
   covers 200 symbols. Widening that toward the 750-name universe is a
   bounded, well-understood job with no new PIT reasoning required.

Route 2 is the recommendation: it closes the remaining P0 (D-007, two
populations in one ordering) by taking fundamental coverage from 8.8% toward the
share of the universe yfinance can serve, and it does not depend on
reverse-engineering an endpoint NSE may have retired deliberately.

### Round 4 — 2026-09-05, the store was deleting the data it had fetched

| defect | was | now |
|---|---|---|
| D-046 `write_statements` replaces instead of merging | new, **P1** | **CLOSED** — this is why coverage sat at 200 symbols |
| D-007 two populations in one ordering | **P0** | **being closed** — statements widening from 200 toward 750 |
| D-030 implausible corporate-action ratios | P2 | **CLOSED** — `MAX_PRICE_FACTOR` |
| D-031 two staleness constants | P2 | **CLOSED** — documented as family-block-only |
| D-033 unstable ranking sort | P2 | **CLOSED** — `kind="stable"` |
| D-022 `/ready` checks the wrong things | P1 | **CLOSED** — reports the inputs the scorer reads |
| D-040 `V2Recheck` dead class | P3 | **CLOSED** — removed, and the undefined `dt` with it |
| D-041 duplicate dict keys | P3 | **CLOSED** |
| D-026 no lock file, two interpreters | P1 | **PARTIAL** — `requirements.lock.txt` written, divergence named |
| D-017 two README book tables | P1 | **CLOSED** — both labelled with the model they measured |
| D-021 three trial counts | P1 | **CLOSED** — honest floor stated as "≥5,140 and not reconstructible" |

**The last P0 was a store defect, not a data-availability one.**
`write_statements` called `replace_table`, so every write discarded whatever the
previous one had fetched and kept only the current batch. `fetch_statements` is
per-symbol and skips whatever the provider refuses — correct, since the universe
is wider than any statement feed — so the table converged on whichever fetch ran
last rather than on the union of everything ever fetched.

Probed live: **yfinance returned statements for 12 of 12 sampled symbols the
store had nothing for**, current to 2026-06-30. Nothing was wrong with the feed.
The store was deleting it. That is the whole of why fundamental coverage was
8.8% of the live cross-section, which is the last P0.

**Two bugs of my own in the repair, both caught by watching the numbers rather
than by reasoning.**

1. The first widening run used the still-broken `replace_table` and drove
   coverage from 192 symbols down to 25. Restored from a backup taken before
   starting; the store is byte-identical to where it began.
2. The first *fix* keyed the merge on `(symbol, period_end, kind)` — which is
   wrong, because `fetch_statements` emits **three** annual rows per symbol-year
   (income statement, balance sheet, cash flow), each carrying only its own
   fields. Deduplicating kept the cash-flow row and discarded revenue and
   equity. Annual rows fell 2,406 → 1,259 while symbol coverage rose, which is
   what gave it away. The rows are now *folded* rather than deduplicated,
   taking the newest non-null value per field: measured on the real table,
   3,420 rows become 1,842 and the count of populated cells is **unchanged at
   28,904**. Rows fall because NaN padding goes, not because data does.

Both are pinned by `tests/test_store_merge_semantics.py`, including a test that
asserts the fold is lossless on the shipped store.

*`replace_table` is still correct for `equity_master`, `sector_map` and
`corporate_actions` — each is fetched as one complete file, so a name NSE drops
should leave the store. Its docstring now says which question to ask before
adding a caller.*

### Round 5 — 2026-09-05, self-audit against this document

Checked mechanically, not from memory. **27 of 46 defects closed; 19 open.**
Two of the "open" were my own predicate errors (D-032 was fixed; D-041 was
fixed) and **three more of my findings did not survive re-checking:**

**D-023 is largely refuted.** It said the 45-session earnings blackout
under-covers a 63-session hold and that the gate "only bites where the calendar
has data". Checking:

* the gap is **deliberate and measured** — `parameters.yaml:857-865` records
  that widening 45 → 63 "collapses the eligible universe from 145 names to 26",
  because Indian results are quarterly at ~63 sessions, so a window spanning the
  hold excludes an entire reporting cycle;
* `on_missing_data: "not_testable"` already reports rather than silently passes,
  and each card shows "Not testable: earnings_proximity";
* the calendar covers **748 of 750** universe names, not the sparse coverage I
  inferred from the 11% rejection rate.

What survives is narrower and was not what I claimed: only **173 of 750 (23%)
have a *forward* earnings date**, so the gate protects under a quarter of the
book and the rest proceed as NOT_TESTABLE. That is disclosed per name already.

**D-029 was real and is fixed.** `bench = close.mean(axis=1)` was the mean price
LEVEL — price-weighted, so a ₹30,000 name moved it a thousand times more than a
₹30 one — differenced into a "return". It is now the mean of per-name returns.
The two series correlate 0.90, so this is a materially different market leg for
`resid_rev_21`'s beta and residual.

**D-019 now reports itself.** Every run states how much of "sector-neutral" is
true: *"Sector-neutral for 236 of 386 names. The other 150 (39%) are ranked
inside one residual bucket — 79 with no sector in the map and 71 folded in from
13 sectors too small to rank within."*

**D-027 / D-028 surfaced.** A delivery or fundamentals feed failure removed a
theme and re-capped the blend with nothing but a log line to show for it; both
now write a `DEGRADED` note into the run record. A non-equity filter failure now
travels on the universe snapshot instead of only into a log.

**D-039 partially closed.** 97 pyflakes findings → 51. 68 unused imports were
removed programmatically, then **every `__init__.py` was restored**: pyflakes
cannot tell a dead import from a package re-export, and the first pass deleted
`compute_features` from `features/__init__.py`, breaking three modules. Verified
by importing every module in the package. The remaining 30 are in `api.py` and
`cli.py`, which the remover skipped rather than risk a syntax break, plus the
restored re-exports.

**Still open and honestly so:** D-009 (forward test), D-011/D-016/D-024/D-025/
D-034/D-035 (data and epoch operations), D-012 (deploy-host probe), D-014
(dividends), D-020 (`OUTSIDE_MODEL_DOMAIN` re-justification), D-036 (short leg),
D-037 (deleting the dead model files, which the plan sequences after Phase 6).

### Round 6 — 2026-09-05, the last P0 closed, and what closing it changed

**Fundamental coverage 8.8% → 100%.** `statements.parquet` went from 200 symbols
to 758, covering **750 of 750** universe names, newest period 2026-06-30. Nothing
about the feed changed — `write_statements` was replacing the table instead of
merging it (D-046), so every refresh discarded what the previous one fetched.

| defect | now |
|---|---|
| **D-007** two populations in one ordering | **CLOSED** — every one of 386 names is now scored on all five themes |
| **D-011** fundamentals stale | **CLOSED for the model** — via `statements`, not the dead NSE endpoint. `fundamentals.parquet` itself is still frozen and that is upstream, not ours |

**The effective weights are now exactly the declared weights.** With no name
missing a theme, D-001's renormaliser has nothing to renormalise:

| theme | declared | effective (mean) | max over names |
|---|---|---|---|
| momentum | 40.00% | **40.00%** | 40.00% |
| quality | 18.99% | **18.99%** | 18.99% |
| ownership | 18.94% | **18.94%** | 18.94% |
| risk | 11.09% | **11.09%** | 11.09% |
| reversal | 10.98% | **10.98%** | 10.98% |

`n_themes` is 5 for all 386 names. The engine is, for the first time, running
the model as specified.

**AND THAT IS THE THING TO BE CAREFUL ABOUT.** The `quality` theme ships at
signs (−1, −1): it buys low and unstable margins, which is what the search
measured (IC −0.0351, t −5.52) and not a defect (D-002). But it had been
reaching an effective weight of **1.67%** because 91% of names had no
fundamentals. It now reaches **18.99% on every name** — an eleven-fold increase
in the engine's exposure to a counterintuitive tilt that **has never run at full
weight in production**.

The book moved accordingly. Top six before → after:

```
before   SJS  GRWRHITECH  AVL  MBAPL  IKS  SKYGOLD
after    SKYGOLD  MBAPL  HFCL  ASTERDM  GRWRHITECH  SYRMA
```

Three of six names change; SJS falls from 1st to 8th, AVL from 3rd to 15th. This
is a data repair, not a code change, and it moved the book more than every code
fix in this plan combined. It is the *validated* configuration — but "validated"
here means the ranking carried two sealed-holdout evaluations at these weights,
on a panel where coverage was what the fit assumed. **Nobody has watched this
run.** Treat the first weeks of output as new, not as a continuation.

Momentum's share of realised cross-sectional variance is 60.79% against a 40%
weight. That is dispersion, not weight, and the weight constraint now holds
exactly — momentum's sub-score is simply more spread out than the others. It is
reported on every run by the dominance monitor.

### Round 7 — 2026-09-05, Phase 6/7 verified against their own exit gates, and a cleanup reverted

**Phase 6 exit gate: 0 of 3 pass. Phase 7: 1 of 3.** Run verbatim, not from memory.

| Phase 6 item | state | evidence |
|---|---|---|
| D-009 forward test registered | **NOT DONE** | `research forward` → "No forward test is registered" |
| D-016 outcomes partitioned by epoch | **NOT DONE** | 128 rows, all `epoch_id: pre-epoch`, all `baseline-v1@*` |
| D-017 README book tables replaced | **PARTIAL** | labelled with the model each measured; the gate wants the numbers gone — `grep -c` returns **11**, must be 0 |
| D-021 honest trial count | **PARTIAL** | README section written; the registry still reports 119 |
| D-022 `/ready` | **DONE** | six data checks verified |

**Phase 6 should not be done yet, and this plan says so:** *"Must come last. Every
phase above changes the signal."* The signal changed today — quality coverage
8.8% → 100% moved three of six book names. Registering a forward test now pins a
hash to an engine that changes again the moment dividends (D-014) land. And
D-017's gate is unmeetable as written: it asks for the tables to be *replaced*
by a measurement on `v3_composite` that does not exist and needs a backtest
after the data settles. Deleting the numbers without the replacement would leave
the README asserting nothing, which is worse than labelling them.

`research readiness` reports four FAILs — DATA (manifest), MODEL (scoring path
`da717d375fc1 → ac8ac60394a7`), REPRODUCIBILITY (tree drift), FORWARD (no
registration). All four are expected while the code is still moving.

**Phase 7:** D-032, D-033, D-040, D-041 done; D-026 partial; the §5 deletion list
correctly deferred (§5 itself sequences it after Phase 6).

**D-039 IS REVERTED, AND THAT IS THE RIGHT ANSWER.** Removing the 68 unused
imports broke working code **twice**:

1. it deleted `compute_features` from `features/__init__.py` — a package
   re-export, which pyflakes cannot distinguish from a dead import — breaking
   three modules;
2. it deleted `UNSCORED_CONTROLS` from `crossmodel.py`, which
   `test_factor_overhaul.py` reads as `cm.UNSCORED_CONTROLS`. That one survived
   a targeted re-test and was caught only by the full 18-minute suite.

A scripted restore then inserted imports inside multi-line import blocks and
broke six files at the syntax level. All of it is now reverted: the 18 files
whose only change was imports are back at HEAD, and the 12 files carrying real
fixes were de-duplicated by hand. Every module imports; all 22 behavioural
checks still pass.

pyflakes stands at **88** against an original 97 — the −9 is the `metrics.py`
duplicate keys, the two undefined names and `V2Recheck`, all of which were real.
The remaining 88 are cosmetic, and the lesson is worth more than clearing them:
**pyflakes-driven import removal is unsafe in this codebase** without a
reachability check that understands re-exports and aliased attribute access. A
P3 finding is not worth a defect that only an 18-minute suite can find.

### Round 8 — 2026-09-05, mutation testing, and two gaps in my own tests

Plan section 8 item 8 asked whether the suite asserts anything load-bearing, and
predicted that mutating the blend would not be caught. That was true of the
suite as it stood. Measured against the suite as it is now — each repair broken
deliberately, the relevant tests run, the mutation reverted:

| mutant | result |
|---|---|
| D-001 revert the cap to a bare renormaliser | **CAUGHT** |
| D-003 serve the frozen weight to the card | **SURVIVED → now CAUGHT** |
| D-004 key the scorer on factor names again | **CAUGHT** |
| D-008 drop the ambiguity raise | **CAUGHT** |
| D-008 order rows by file position | **CAUGHT** |
| D-008 drop the lineage filter | **SURVIVED → now CAUGHT** |
| D-010 shorten the window to 300 | **CAUGHT** |
| D-046 replace instead of merge | **CAUGHT** |
| D-002 flip the quality signs to +1 | **SURVIVED, correctly** |

**D-003 survived for a reason worth writing down.** Since the statements repair
took coverage from 8.8% to 100%, every live name carries all five themes — so
the per-name blend weight and the frozen `Theme.weight` are *numerically
identical*, and no test on today's cross-section can distinguish them. Reverting
Stage 4 to `th.weight` passed everything. That is not safety, it is luck:
`build_v3_block` swallows a delivery or fundamentals failure and continues with
the theme absent (D-027), and the moment that happens the two diverge again with
nothing watching. Closed by a test that forces the divergence — one theme NaN'd,
re-blended, fed through the stage's own card construction.

**D-008's lineage filter survived** because `previous_run` scans for the newest
date under one filter and fetches that date's rows under another; removing the
first only matters when a replay is dated later than the newest live run, which
no test covered. Closed, and a guard added: if the two filters ever disagree the
result is now a `LedgerError` naming the mismatch rather than an `IndexError` on
an empty list.

**D-002 survived and should.** The quality signs are a measurement (IC −0.0351,
t −5.52), not an intent, and nothing should pin them to −1 as though they were.
`research/V3_SEARCH.md` records the same convention for `mom_accel`.

**Remaining P0 count: 0.** D-006 closed and restated (Round 3); D-007 closed by
the coverage repair (Round 6). Every P0 in this register is either fixed or
refuted. That is not the same as safe to trade — see the checklist.

Four new test files pin the repairs: `test_ledger_lineage.py`,
`test_v3_blend_is_capped.py`, `test_factor_window_depth.py`,
`test_card_arithmetic.py`.

**Two corrections to this document, found while implementing it.** Both are
recorded rather than quietly edited, because a defect register that silently
loses entries is not evidence.

1. **D-002 is refuted.** See its entry below.
2. **D-001's fix produces momentum at exactly 40.00%, not 39.47%.** The 39.47%
   figure came from re-running `cap_weights` with its default `floor=0.06` on a
   weight vector that is already floored. That re-floors it and compresses every
   weight toward equal — momentum comes back as `0.06 + 0.70*0.40 = 0.34` even
   for a name carrying all five themes, which is a different model applied to
   the names the fit was correct for. The floor is a fit-time constraint;
   renormalisation only raises weights, so it has no work to do at scoring time.
   Caught by `test_a_full_coverage_name_keeps_the_fitted_weights`, which was
   written before the bug and failed on it.

---

## 1. Verdict

> **VERDICT UPDATED 2026-09-05, after Rounds 0-6.** The system is **still not
> safe to paper trade**, but for entirely different reasons than the ones below.
> Every P0 in the register is closed or refuted, and the engine now runs the
> model as specified — every name scored on all five themes, effective weights
> exactly the declared ones, the card's arithmetic reconciling, and the screen
> naming the scorer that actually ran.
>
> What blocks now is that **nobody has watched it run in this state**. There is
> no content-hashed daily snapshot, no slippage or factor attribution in the
> outcome record, and no IC-decay threshold — so the first weeks of output
> cannot be scored against the engine's own claim. Two data gaps remain real
> (dividends are not adjusted; delivery starts 2019 against prices from 2017),
> and the manifest and epoch gates are deliberately left failing until the code
> stops moving.
>
> **The single most consequential change was not a code fix.** Repairing
> `write_statements` took fundamental coverage from 8.8% to 100%, which took the
> `quality` theme's effective weight from 1.67% to 18.99% — an eleven-fold
> increase in exposure to a low-margin tilt that has never run at full weight in
> production. Three of the top six names changed. Read Round 6 before trusting
> continuity with anything this engine printed before today.
>
> The original verdict, as written on 2026-09-04, follows unedited.

**The system is not safe to paper trade.** The single thing most wrong with it is that the
40% dominance cap on the momentum theme — the constraint the shipped composite was fitted
under and the one thing its own module says protects it from being "a momentum bet with
decoration" — is silently removed at scoring time by the coverage renormaliser two functions
later, so momentum actually carries **48.55% of the blend and 63.59% of the realised
cross-sectional spread**, while a theme called "quality" whose signs are both inverted
applies to 8.8% of names and the screen tells the operator the ranking came from a
"validated cross-sectional model" that was deleted from the codebase on 2026-09-03.

The engine is closer to correct than the register's length suggests: **21 of 22 factors are
point-in-time clean under an independent probe**, Newey-West and PBO are correctly
implemented, and the eight-stage pipeline is coherent. The defects are concentrated in three
places — the blend arithmetic, what the screen claims about it, and a data layer everyone
believed was network-blocked and is not.

---

## 2. What actually runs today

### 2.1 The live call graph

```
UI  #scan click                    static/index.html:3264
 └─ POST /analysis/run             api.py:309-316
     └─ _start("analysis")         api.py:289-307   single-flight, thread-locked (jobs.py:263-299)
         └─ JobManager.start → _runner(progress)    api.py:79-81
             └─ pipeline.run_analysis(cfg)          pipeline.py:104-125
                 ├─ store_lock(curated, exclusive=False)      pipeline.py:120
                 ├─ DataStore(curated, snapshots)             pipeline.py:127
                 ├─ _universe → _universe_liquidity_pit       pipeline.py:642-673
                 │    └─ UniverseResolver.resolve_liquidity_pit   data/universe.py:120-262
                 │         └─ instruments.non_equity_symbols      data/instruments.py:55
                 ├─ _prefetch_prices                          pipeline.py:171
                 ├─ stage1_data_quality.run                   pipeline.py:176
                 ├─ stage2_regime.run                         pipeline.py:186
                 ├─ Ledger.previous_run(before=resolved)      pipeline.py:201-204   ← the entire position memory
                 ├─ stage3_eligibility.run(held=open_book)    pipeline.py:210
                 ├─ stage4_core_score.run                     pipeline.py:220
                 │    ├─ family factors → composite_raw       stage4:341-476  (computed, then DISCARDED)
                 │    ├─ build_v3_block                       stage4:229-306
                 │    │    ├─ store.read_prices (adjusted)    data/store.py:420-451
                 │    │    ├─ store.read_delivery             stage4:265-270
                 │    │    ├─ pit_fundamentals.build_records  stage4:277-291
                 │    │    ├─ v3_factors.factor_frame         features/v3_factors.py:31-149
                 │    │    └─ v3.score_frame                  features/v3.py:226-265   ← THE RANKING
                 │    └─ _apply_ranking_policy                stage4:85-227
                 ├─ stage5_false_signal.run                   pipeline.py:229
                 ├─ stage6_entry.run  (+ cadence.clock)       pipeline.py:251-256
                 ├─ stage7_risk.build_plan  (+ CostModel)     pipeline.py:262-278
                 ├─ stage8_final_signal.run                   pipeline.py:296-315
                 ├─ _review_open_positions                    pipeline.py:329
                 ├─ _build_slate → presentation.selection     pipeline.py:352-358
                 ├─ Ledger.append   (fails the run if it fails)   pipeline.py:415
                 └─ rundetail.save  (display cache, swallows) pipeline.py:453
     └─ GET /analysis/{id}/view → viewmodel.build_view        api.py:350-383
```

Other reachable entry points: **CLI** (`prosignal.cli:main`, 2,953 lines, 8 subcommand
families — `analyse run` is the same `run_analysis`), **cron** (`scripts/forward_run.sh`,
`scripts/quarterly_recheck.sh`, `scripts/open_production_epoch.sh`), and the FastAPI surface
(48 routes). No notebooks. `render.yaml` deploys the API.

### 2.2 The real shipped model

`stage4_core_score.ranking.source = "v3_composite"` — **VERIFIED** by loading the config.
The scorer is `features/v3.py`: 22 factors in 5 themes, sector-neutral ranks combined
within theme, themes re-ranked, then blended.

**The family-factor path is dead weight with one live side effect.** Stage 4 computes
`momentum_12_1` and `sector_relative_strength`, winsorises, standardises, applies regime
multipliers and renormalises (stage4:417-476) — and `_apply_ranking_policy` throws the
result away (stage4:139 `covered = ranked.reindex(composite_raw.index)`). Its *index*
survives as a population filter, so the family path decides **who may be ranked** while the
v3 composite decides **how**. On today's run it computed a two-factor composite over 386
names purely to supply that index.

### 2.3 The real effective weights — VERIFIED on 2026-09-03, 386 eligible names

| theme | declared | live sub-score coverage | **effective weight (mean)** | max over names | **share of realised score variance** |
|---|---|---|---|---|---|
| momentum | 40.00% | 100.0% | **48.55%** | 49.38% | **63.59%** |
| quality | 18.99% | **8.81%** | **1.67%** | 18.99% | 12.80% |
| ownership | 18.94% | 100.0% | **22.99%** | 23.38% | 17.40% |
| risk | 11.09% | 100.0% | **13.46%** | 13.69% | 12.75% |
| reversal | 10.98% | 100.0% | **13.33%** | 13.56% | 5.74% |

The seeded figure "momentum 51.6% effective against 40% nominal" is **confirmed in
direction, corrected in magnitude**: the mean renormalised weight is 48.55% and the share of
today's cross-sectional spread is 63.59%. Both breach the 40% cap.

Factor coverage on the live cross-section: 20 of 22 factors at 99.5–100%; `net_margin` and
`margin_stability` at **8.8%** (34 of 386 names).

`n_themes` distribution: 352 names scored on 4 themes, 34 on 5. **Not one name in the live
universe is scored by the model as specified.**

### 2.4 Where documentation contradicts code

| # | Document says | Code does | Address |
|---|---|---|---|
| 1 | "Ranking: **`mom_6_1_r`** — the sector-neutral rank of 6-1 momentum, one column" | `ranking.source: v3_composite`, 22 factors | `README.md:11,29` vs `config/parameters.yaml:1053` |
| 2 | "themes blended with weights capped at 40%" (run note, every run) | cap defeated by renormalisation; momentum 48.55% | `stage4:148-160` vs `v3.py:251-256` |
| 3 | expectancy = "Sector-neutral 6-1 momentum rank, six names…" | stamped on cards produced by v3_composite | `config/parameters.yaml:2732` vs `:1053` |
| 4 | "`data/ledger/forward_test.json` still names `baseline-v1@a30a8d4847080ddc`" | **the file does not exist**; `research forward` prints "No forward test is registered." | `README.md:46-49` vs `data/ledger/` |
| 5 | "the engine runs `baseline-v2@9ffe2b1b65e17832`" | `baseline-v2@74ad0b78ffe70cc0` | `README.md:47` vs `loader.load_config()` |
| 6 | README table A: book +42.6% / benchmark +18.9% / alpha +20.3% | measured on the 6-1 momentum ranker, current book geometry | `README.md:95-110` |
| 7 | README table B: "**this table is the live one**", excess −4.23%/period | measured on the *fitted Fama-MacBeth* model with a 2.5×ATR stop, 3R target and risk-budget sizing — none of which is live | `README.md:270-300` |
| 8 | "Deflated Sharpe 0.030 against **4,877** trials" / "charging **81** trials" | trial registry charges **119** | `config/parameters.yaml:2769` / `README.md:285` vs `data/curated/trial_registry.jsonl` |
| 9 | "Egress to NSE, BSE and AMFI is blocked by a refused gateway" | all three return **200** through the repo's own HTTP client | see D-012 |
| 10 | `stage4` comment: the fitted model "cost about 3,000 lines … all of it executing daily" and was removed | the *call* was removed; `crossmodel.py` (1,387 ln), `famamacbeth.py` (735), `linear.py` (105), `fundamental_factors.py` (429) are still imported by `api.py:876`, `viewmodel.py:518`, `data/coverage.py:104` | `stage4:483-497` |
| 11 | `v3_factors.py` docstring: "Nothing reads a session after the decision row" | true (VERIFIED), but `resid_rev_21` reads *before* the loaded window and silently returns a different number | `v3_factors.py:11` vs D-010 |
| 12 | `universe.py` docstring: `bench_ret` is "the equal-weight return of the ELIGIBLE universe" | `close.mean(axis=1)` — the mean of price **levels**, i.e. price-weighted | `v3_factors.py:41` vs `stage4:274` |

### 2.5 Silent fallbacks and swallowed failures on the live path

Every one of these logs a warning and continues with degraded input.

| Address | What is swallowed | Consequence |
|---|---|---|
| `stage4:265-270` | delivery table unreadable | ownership theme (19% nominal, 17.4% of live spread) vanishes; momentum renormalises to ~57% |
| `stage4:277-291` | PIT fundamentals unavailable | quality theme vanishes (already true for 91.2% of names) |
| `stage4:169-172` | theme-dominance monitor throws | the one live check on D-001 goes quiet |
| `universe.py:206-210` | non-equity filter throws | ETFs, gold/silver funds and bond funds re-enter an equity universe |
| `pipeline.py:294` | earnings-proximity lookup throws | risk disclosure disappears from cards |
| `pipeline.py:427-433` | drawdown monitor throws | realised-drawdown flag goes quiet |
| `ingest.py:899-901` | fundamentals refresh throws | store silently keeps 18-month-old filings |
| `rundetail.save` (`pipeline.py:453`) | display cache write fails | Today renders an empty evidence panel |

Config keys read but ignored: `universe.index_name` and `universe.pre_snapshot_policy`
(both dead under `source: liquidity_pit`, `pipeline.py:644`), `ranking.column: mom_6_1_r`
(only read under `source: measured_factor`, `stage4:196`), and
`stage4.max_fundamental_age_days: 450` — the v3 path uses the hardcoded
`pit_fundamentals.MAX_AGE_DAYS = 420` instead (`v3_factors.py:132`).

A schema sweep found only 6 of 1,692 leaves whose name appears nowhere in `src/`, all of
them dict-keyed provider data — so the config is genuinely live. The problem is not unread
keys; it is **18 leaf names declared in more than one place**, where a single grep hit hides
the rest (`enabled` × 34, `max_age_sessions` × 11).

---

## 3. Defect register

Sorted by severity, then by blast radius.

### P0 — produces a wrong signal or shows the user a false number

---

**D-001 · P0 · VERIFIED · `src/prosignal/features/v3.py:251-256`**
**The 40% dominance cap is removed at scoring time by the coverage renormaliser.**

`THEMES` holds weights that are already post-cap, post-floor and post-coverage-cap
(`v3.py:93-115`; quality's 0.18991 *is* its 0.1899 coverage cap). `score_frame` then divides
by the sum of the weights a name actually has:

```python
num = np.nansum(np.where(ok, M * W, 0.0), axis=1)     # v3.py:251
den = np.where(ok, W, 0.0).sum(axis=1)                # v3.py:252
out["score"] = num / den                              # v3.py:256
```

For the 352 names with no quality data `den = 0.81009`, so momentum's weight becomes
`0.40 / 0.81009 = 49.38%`. `cap_weights` (`v3.py:192-224`) exists and implements exactly the
right per-theme cap — and is **never called at scoring time**. The cap is applied at fit time
and undone at score time.

*Blast radius:* every score, every rank, every book, every date. Momentum runs at 48.55%
mean effective weight and **63.59% of realised cross-sectional variance** against a 40% cap.
The module docstring's own stated failure mode — "a momentum bet with decoration" — is what
ships.

*Found by:* decomposing the live scored frame; computing per-name renormalised weights and
`cov(contrib_i, score)/var(score)`.

*Fix, and its measured effect:* calling `cap_weights` per name over the themes it has
returns momentum to **39.47%** (inside the cap), raises risk to 17.10% and reversal to
17.00%. Spearman correlation to the current ranking is 0.977, but it **swaps 1 of the 6 book
names and 2 of the top 20**. This is a change to what gets bought, not a cosmetic one.

---

**D-002 · ~~P0~~ → P3 · REFUTED, then closed · `src/prosignal/features/v3.py:103`**
~~The "quality" theme is sign-inverted on both of its factors.~~
**The signs are correct as measured. The theme is misnamed, and only in code.**

The original finding claimed a double negative: `v3_factors` computes
`margin_stability = -sd(net_margin)` and `v3.py` then applies sign `-1`, so the
theme prefers unstable margins. The arithmetic is real and the measured
correlations stand (`corr(sd, signed contribution) = +0.3312`,
`corr(net_margin, signed) = -0.5032`). The **inference was wrong**, on three
counts, all checked in git history before any sign was touched:

1. **The search computed it the same way.** `work/v3/themes.py:92` at commit
   `636346f`: `out["margin_stability"] = -_roll(out["net_margin"], 504, "std",
   mp=250)` — identical to the shipped line. The double negative is not
   something that happened to a validated model afterwards; it is inside what
   was validated.
2. **The sign is a measurement.** `research/V3_SEARCH.md:150` records
   `margin_stability` at **IC −0.0351, t −5.52**, holding its sign across both
   halves of its own life (t −2.83 / −5.30). `research/v3/FROZEN_V3.json` freezes
   the theme at `sign: -1` with `validated_ic_t: 4.756` and the highest
   `validated_topk_contribution` of any theme (0.0137, against momentum's
   0.0088). The same document says of `mom_accel`, treated identically: "It
   ships at sign −1 and is pinned by a test **so nobody 'corrects' it**."
3. **The screen was already honest.** `viewmodel.py:578` has labelled the theme
   **"Low-margin tilt"** since commit `fae96e2` ("Honest theme labels + quality
   sign warning"). The audit missed that mitigation and asserted the operator
   is shown a theme called "quality". They are not.

Flipping the signs would have broken the one thing the sealed windows actually
measured, to fix a defect that did not exist. **What survived** is that the dict
KEY `quality` contradicts what the theme does, and the key — not the label —
was what reached the run notes, the ledger and the CLI: `"...(momentum 40%,
quality 19%, ...)"`.

*Closed by:* `Theme.label` (`v3.py:109`), carried into the Stage 4 run note and
into `FactorScore.citation`, with `viewmodel.V3_THEME_LABELS` now derived from
it instead of being a second copy. `test_card_arithmetic.py::
test_the_screen_and_the_model_agree_about_what_a_theme_is_called` pins that
there is one table.

*What this cost:* one P0 that was never a P0. The lesson is in the brief's own
instruction — the code is the truth — applied to the audit's own reasoning: a
sign that looks backwards against its name is a hypothesis, and the search
artefacts were sitting in git history the whole time.

---

**D-003 · P0 · VERIFIED · `src/prosignal/presentation/viewmodel.py:634-635` + `src/prosignal/static/index.html:3138-3150`**
**The per-stock theme table shows a weight that does not produce the contribution beside it.**

The panel renders, per theme: label, `(r.coefficient * 100)% weight`, a sector percentile,
and `Adds +X.XXX` — under the heading **"What ordered it — sums to the score"**. The
contributions do sum to the score. The weights do not produce them. Measured on the live BUY:

| theme | Z (sub) | **WEIGHT shown** | z × WEIGHT | **CONTRIB shown** | ratio |
|---|---|---|---|---|---|
| momentum | 0.72539 | 40.00% | 0.29016 | **0.35818** | 1.2344 |
| quality | — | **18.99%** | — | **(blank)** | — |
| ownership | 0.60622 | 18.94% | 0.11481 | 0.14173 | 1.2344 |
| risk | 0.94041 | 11.09% | 0.10427 | 0.12872 | 1.2344 |
| reversal | 0.82902 | 10.98% | 0.09104 | 0.11239 | 1.2344 |

Every contribution is exactly `1/0.81009 = 1.2344 ×` the arithmetic the card invites the
reader to check. The card asserts momentum contributed 40% of this name's score; it
contributed **49.38%**. It shows quality at 18.99% with a blank contribution, and never says
that the 18.99% was redistributed to the other four.

`_contributions`' own docstring (`viewmodel.py:588-597`) diagnoses this precisely — "WRONG
for the v3 composite, whose weights renormalise over the themes a name actually has … it is
now used" — and then fixes only `contribution`, leaving `coefficient` at the declared weight.
The mismatched pair ships.

*Blast radius:* every card, every run. This is the number the operator uses to decide whether
they believe the pick.

*Note:* `tests/test_v3_score.py:127` and `tests/test_model_attribution.py` both assert that
contributions sum to the score — which is true — and nothing asserts that the *displayed*
weight is the weight that was used. The defect passes 1,638 tests.

---

**D-004 · P0 · VERIFIED · `src/prosignal/presentation/viewmodel.py:121-122` + `src/prosignal/presentation/evidence.py:91-92`**
**The screen certifies every run as produced by a "validated cross-sectional model" that no longer exists.**

`_scorer_used` decides which scorer ran by intersecting the card's factor keys with
`MODEL_KEYS` (the deleted Fama-MacBeth families) minus `COMPOSITE_KEYS`:

```
MODEL_KEYS     = {beta, delivery, drawdown, lottery, mom, quality, reversal, skew, value}
COMPOSITE_KEYS = {momentum_12_1, quality, sector_relative_strength, value}
live v3 keys   = {momentum, ownership, quality, reversal, risk}
live & (MODEL - COMPOSITE) = {'reversal'}          ← collision
```

The v3 theme named `reversal` collides with a family named `reversal` from the removed model,
so `seen & (MODEL_KEYS - COMPOSITE_KEYS)` is truthy and the function returns
`{"model": "cross-sectional", "validated": True, "note": None}` — executed and observed.

`index.html:2182` renders its warning only when `scorer.validated === false`. **That branch is
unreachable.** The function's own docstring says presenting the wrong scorer as validated "is
the single most misleading thing this interface could do"; it does exactly that, on every run,
by accident.

*Blast radius:* the operator is told the ranking is validated. `v3.py:BOOK_NOTE` says the
opposite of the thing that actually ships: "THE COMPOSITE CARRIES TWO SEALED-HOLDOUT
EVALUATIONS. NO BOOK DOES … the concentration is an operator's risk choice, not a validated
one."

---

**D-005 · P0 · VERIFIED · `config/parameters.yaml:2732-2795` → `src/prosignal/tradeplan.py:88-104`**
**Every card and every ledger row is stamped with the historical frequencies of a different model.**

The `expectancy` block is a frozen table — p(profit) 0.578, mean net +7.09%, mean excess
+3.74%, 258 trades — attached to every trade plan the engine issues. The config states its own
provenance at line 2732: *"THE STUDY. **Sector-neutral 6-1 momentum rank**, six names, entries
every 21 sessions…"*. The engine ranks on `v3_composite`. Confirmed present on the live BUY's
`trade_plan` payload.

Three compounding problems: (a) the ranker is not the one measured; (b) the study period
2018-11-27 → 2026-08-25 **contains the whole sealed holdout 2025-01 → 2026-08**; (c) the
holdout figures those numbers descend from are withdrawn (`README.md:36-40`) because the panel
was survivorship-biased.

The web card no longer prints the figures (`index.html:2310-2317`, removed at the owner's
call), but the CLI does (`cli.py:893-911`), `viewmodel._expectancy` still serves them, and
they are written into the ledger on every trade.

---

**D-006 · P0 · VERIFIED · `src/prosignal/stages/stage5_false_signal.py:68-75` + `config/parameters.yaml:1088`**
**NO TRADE cannot fire on the quality of the evidence. Proof, not conjecture.**

`composite_score` is `rank_to_unit_interval(composite_raw)` = `(rank-1)/(n-1)`
(`indicators/crosssection.py:102-116`) — a pure cross-sectional rank. `percentile` is the same
number × 100. The Stage 8 scarcity gate is
`min_composite_score: 0.6` **and** `min_universe_percentile: 90.0` — two scales of one
quantity, binding at the 90th percentile. **The top 10% of any distribution always exists.**
On the live run the gate rejected 0 of 37: `passed_score_threshold == survived_defense == 37`.

The only absolute (non-rank) bar was `absolute_floor`, and it is **`enabled: false`**
(disabled 2026-09-02 on a measured −2.2% ATE). `config/parameters.yaml:1070-1072` states the
principle correctly — *"A floor on a cross-sectional RANK cannot fire: somebody is top of the
list every day however weak the day is"* — and then the mechanism written to satisfy it is
switched off.

What remains: (a) the entry cadence — a schedule, not a judgement, and the state on 20 of
every 21 sessions; (b) a market-wide regime or data halt; (c) all top-6 names hard-rejected
by Stage 5. There is **no state in which this engine says "nothing is good enough today"**,
which is precisely what its own config claims the floor exists to provide.

---

**D-007 · P0 · VERIFIED · `src/prosignal/features/v3.py:186-190` + `:251-256`**
**Two populations are ranked by two different models and merged into one ordering.**

34 of 386 names (8.81%) carry a quality sub-score. Those 34 are scored by a 5-theme model in
which quality holds 18.99%; the other 352 by a 4-theme model in which it holds 0% and
momentum is 9.4 percentage points heavier. The two scores are then sorted together as if
commensurable.

It is worse than a weighting difference. `theme_subscore` (`v3.py:190`) re-ranks each theme
*within the names that have it*, so the quality sub-score for those 34 names spans [−1, +1]
across a 34-name population while every other theme spans [−1, +1] across 386. A quality
sub-score of +1.0 means "best of 34"; a momentum sub-score of +1.0 means "best of 386". They
are blended as equals.

The declared coverage in `THEMES` is 0.1899; the live figure is **0.0881**. The frozen weight
was capped at a coverage that no longer holds.

---

**D-008 · P0 · VERIFIED · `src/prosignal/ledger.py:109-134`, called at `pipeline.py:201-204`**
**The engine's entire position memory is "whichever line was appended last".**

`run_analysis` holds no position state; `Ledger.previous_run` is the only source of the open
book, and it feeds Stage 3 (held names bypass filters), Stage 6 (hysteresis), Stage 8
(portfolio limits) and the orphan review. Its selection rule is
`if latest_date is None or when >= latest_date` (`ledger.py:131`) — **`>=`, so within a date
the last row in file order wins**.

The ledger it reads:

- 1,942 rows over **253 distinct market dates**
- **80 dates carry more than one recorded run; 2026-08-18 carries 676**, across **14 config
  versions**, recording **7 mutually inconsistent books** — `()` ×201,
  `(CIEINDIA, CONFIPET, DCBBANK, EPL, KTKBANK, MAYURUNIQ, …)` ×173, `(CESC, GICRE,
  INDIAGLYCO, PETRONET)` ×157, `(MANORAMA,)` ×66, and three more
- **27 config versions** overall; the running `baseline-v2@74ad0b78ffe70cc0` appears **zero**
  times
- `logged_at` is **not monotonic** — 3 out-of-order appends
- `model_fingerprint` is null on 1,733 of 1,942 rows
- `mode` is `"live"` on all 1,942, so backfills and replays are indistinguishable from real runs

Today's read happens to be well-defined (all 6 rows for 2026-09-02 agree on `['LLOYDSME']`).
That is luck, not design.

---

### P1 — invalidates a performance claim or a validation result

---

**D-009 · P1 · VERIFIED · `data/ledger/` (absent file), `src/prosignal/validation/forward.py:58`**
`prosignal research forward` prints **"No forward test is registered."** `forward_test.json`
does not exist. The README describes it as registered-but-INVALID on three counts; the truth
is that the pre-registration was never re-created after the config moved. The seeded claim
"the forward test reports INVALID on three counts" is **refuted — it is worse than that**.
For the record, `forward.progress` (`forward.py:511-566`) implements five invalidation
conditions correctly (hash mismatch, legacy scheme, no benchmark-relative hypothesis, config
drift, session coverage below 60%); none of them can run without a registration.

---

**D-010 · P1 · VERIFIED · `src/prosignal/features/v3_factors.py:25` and `:100-107`**
**`resid_rev_21` is computed on too little history and returns a wrong number instead of NaN.**

Independent probe on live data, holding everything else fixed:

- *Same window start, end moved from T to T+10*: **all 22 factors identical at T** to <1e-9.
  The engine has **no lookahead** — the seeded concern is refuted and the docstring claim
  at `v3_factors.py:11` verified.
- *Same end T, window start moved 10 sessions later*: **`resid_rev_21` alone differs, by up
  to 0.507 across 118 of 120 names.** Every other factor is bit-stable.

Convergence sweep against a 1,200-session reference:

| history depth | 315 (what Stage 4 reads) | 400 | 500 | 650 | 800 |
|---|---|---|---|---|---|
| max abs error in `resid_rev_21` | **4.53e-02** | 1.7e-14 | 1.7e-14 | 2.0e-14 | 1.6e-14 |

The chain is `resid_rev_21 ← idio(126) ← resid(126) ← beta(126) ← bvar(126) ← bc(126) ←
b(126)`, so it reaches ~375 sessions back. `LOOKBACK_SESSIONS = 300` (+15 in
`build_v3_block`) truncates it, and the `min_periods=90` relaxations let the partial window
return a number rather than NaN. `resid_rev_21` is the highest-fidelity factor in the
reversal theme (raw-to-signed-rank correlation −0.931) and reversal is 13.33% of the live
blend. **One-line fix; exit gate is a bit-equality assertion.**

---

**D-011 · P1 · VERIFIED · `data/curated/fundamentals.parquet`, `src/prosignal/data/ingest.py:854-901`**
**Fundamentals are frozen 18 months back, the feed is up, and nothing alarms.**

`fundamentals.parquet`: 3,504 rows, **186 symbols**, `filing_date` max **2025-03-11**,
`period_end` max **2024-12-31**. `results_calendar.parquet` (91,585 rows, 1,288 symbols) also
stops at `period_end` 2024-12-31. Zero symbols have a filing after 2025-03-11.

Root cause, run live against NSE: the endpoint the provider uses
(`/api/corporates-financial-results?index=equities&symbol={sym}&period=Quarterly`,
`nse_fundamentals.py:113`) **answers** — 130 rows for RELIANCE — but the newest `toDate` on
any of them is **31-Mar-2024**, and `provider.fetch_symbol("RELIANCE")` returns 12 clean rows
whose newest is `filing_date 2025-01-16 / period_end 2024-12-31`. The archive this query
shape reaches has stopped advancing. `last_error` is `None` — the provider reports success.

A second, independent defect in the same function: `ingest.py:890-894` takes its symbol list
from `read_universe_snapshot("NIFTY 200", …)` while the live universe is a 750-name liquidity
screen — which is why the file covers 186 symbols and not 750 — and `if not dates: return`
exits silently when no snapshot exists.

Meanwhile `statements.parquet` (yfinance, 200 symbols) **does** carry `period_end` to
2026-06-30. Current fundamentals are available; they are not being fetched.

Downstream: `net_margin` and `margin_stability` are the only two factors affected, gated by
`MAX_AGE_DAYS = 420` in `v3_factors.py:132`. At 541 days of age the source-A records are all
dead, and the 8.8% that survive come from the lagged `statements` path.

---

**D-012 · P1 · VERIFIED · live network probe through `src/prosignal/data/providers/http.py`**
**Egress is not blocked. The premise behind half the "cannot fix" items in this repo is false.**

Using the repository's own `HttpClient` with its own configured User-Agent
(`config/parameters.yaml:393-395`) and its own configured warmup path
(`nse_json_api.warmup_path`), on 2026-09-04:

| endpoint | result |
|---|---|
| `nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_20260903_F_0000.csv.zip` | **200, 203,909 bytes** |
| `archives.nseindia.com/content/equities/EQUITY_L.csv` | **200, 181,507 bytes** |
| `www.nseindia.com/api/marketStatus` | **200, 2,046 bytes** |
| `www.nseindia.com/api/corporates-financial-results?...&symbol=RELIANCE` | **200, 130 records** |
| `www.nseindia.com/api/corporates-corporateActions?...01-08-2026→04-09-2026` | **200, 427 records** |
| `www.nseindia.com/api/corporate-share-holdings-master?...&symbol=RELIANCE` | **200, 22 records** |
| `portal.amfiindia.com/spages/NAVAll.txt` | **200, 1,516,816 bytes** |
| `api.bseindia.com/BseIndiaAPI/api/DefaultData/w` | **200, 14,287 bytes** (with a UA) |

A bare `curl https://www.nseindia.com/` returns `000` — that is NSE's bot shield rejecting a
curl User-Agent, and it is very likely the observation the "refused gateway" belief was built
on. AMFI moved to `portal.amfiindia.com` behind a 302; `requests` follows it. There is **no
AMFI code in `src/` at all** — the monthly-holdings feed was never implemented, not blocked.

*Blast radius:* this is a P1 rather than P0 because it breaks no signal, but it changes the
plan's whole structure and several scorecard ceilings. Dividends, pre-2018 history, ASM/GSM,
current fundamentals and the T0 gap are all **reachable**.

*Caveat, stated plainly:* this was measured from the development machine. Whether
`prosignal.duckdns.org` is equally unblocked needs one `curl` on that host, and the plan
carries it as a checklist item rather than an assumption.

---

**D-013 · P1 · VERIFIED · `src/prosignal/data/store.py:308-311` + `:443`**
**Rows published under NSE's `T0` series are silently dropped, holing the largest names.**

`DataStore.__init__` hardcodes `equity_series = ("EQ",)` and `read_prices` filters on it. In
the last 305 sessions, **44 symbols lose 183 sessions** to rows that exist only under series
`T0`. Affected: **SBIN (19 missing), RELIANCE (3), HDFCBANK (1), INFY (2)** — and 38 of the
44 are in today's 750-name universe.

Consequence: **SBIN is excluded from the universe entirely** — 286 sessions of history
against a `min_history_sessions` floor of 300 — and it is one of 42 names rejected as
`INSUFFICIENT_HISTORY`. It also drives most of Stage 1's 26 `session_continuity` failures.
Rolling windows with `min_periods` compute over the holes and return numbers.

Note `universe.allowed_series: ['EQ']` *is* read, but at `stage3_eligibility.py:357` — the
store keeps its own independent hardcoded copy of the same rule.

---

**D-014 · P1 · VERIFIED · `data/curated/corporate_actions.parquet`**
**Prices are split- and bonus-adjusted but not dividend-adjusted.**

All **5,064 dividend rows carry `ratio = 1.0`** — no price rescaling. `apply_adjustments`
(`corporate_actions.py`) is called on the read path and does the right thing for splits (703),
bonuses (790) and combinations; dividends pass through unchanged. `build_dividend_factors`
and `dividend_amount` exist and are not wired into the price adjustment.

The price series is therefore a **price return**, not a total return. That is a systematic
downward drift of roughly 1–1.5%/yr in every momentum factor, and — because it scales with
dividend yield — it is **cross-sectionally correlated**: high-yield names are systematically
penalised on `mom_2_0`, `mom_3_1`, `mom_12_6`, `prox_52w`, `voladj_mom_*` and every risk and
reversal statistic. Momentum is 48.55% of the blend.

---

**D-015 · P1 · VERIFIED · `src/prosignal/costs.py:125-210` vs `config/parameters.yaml:2788`**
**The engine's own cost model says a round trip costs 2–4× what the card claims.**

Statutory rates in `config/parameters.yaml:2616-2662` are all correct for Indian delivery
equity (STT 0.1% both legs, stamp 0.015% buy-side, exchange txn 0.00297%, SEBI 0.0001%, GST
18% on brokerage + exchange + SEBI, DP ₹15.93 on sell). Fees are ~26 bps. Impact and spread
are the rest. Computed via `CostModel.round_trip` on a ₹10 lakh book, 6 slots:

| name | ADTV | fees | impact | spread | **round trip** | stressed |
|---|---|---|---|---|---|---|
| ₹1,850 | ₹30 cr | ₹433 | ₹785 | ₹167 | **83.1 bps** | 223 bps |
| ₹1,850 | ₹5 cr (the floor) | ₹433 | ₹1,922 | ₹167 | **151.4 bps** | 428 bps |

The expectancy block quotes the headline **net of 40 bps** and calls **120 bps "stressed"**.
120 bps is *below* the model's own figure for a name at the liquidity floor. At 120 bps the
block already reports `p_beat 0.496` — under a coin flip.

Capacity, same names, same 6 slots, at the ADTV floor:

| book | participation | round trip |
|---|---|---|
| ₹10 L | 0.33% | 151 bps |
| ₹50 L | 1.67% | 291 bps |
| **₹1 cr** | 3.33% | **398 bps** |
| ₹2.5 cr | 8.33% | 610 bps |
| ₹5 cr | 16.67% | 849 bps |

The claimed mean excess per trade is **+3.74% = 374 bps**. **The edge is fully consumed at
roughly a ₹1 crore book.** The impact coefficient (0.1, exponent 0.5) is documented
UNVALIDATED in `costs.py:16-17`; the capacity number inherits that uncertainty and should be
read as an order of magnitude.

---

**D-016 · P1 · VERIFIED · `data/ledger/outcomes.jsonl`**
**There are zero closed trades from the shipped configuration.**

128 outcomes, 72 tickers, entries 2023-11-10 → 2026-08-24. **All 128 carry `baseline-v1@*`
config versions** — none carries `baseline-v2`, let alone the running
`baseline-v2@74ad0b78ffe70cc0`. `epoch_id` is `pre-epoch` on all 128. Median
`sessions_held` is **3** (max 10) against a 63-session planned hold — these were produced by
an engine with entirely different exit geometry. Mean net return −2.21%, win rate 27.3%.

`api.py:1310-1316` correctly scopes `/performance` to the live config version, so the page
shows **nothing** with `excluded_closed: 128`. That is the honest behaviour and it means the
live record is empty. `price_basis_factor != 1` on 31 of 128 rows, confirming the price-basis
divergence.

---

**D-017 · P1 · VERIFIED · `README.md:95-110` and `README.md:270-300`**
**The two book tables are reconciled: they measure two different models, and neither is live.**

| | Table A (`README:95-110`) | Table B (`README:270-300`) |
|---|---|---|
| headline | +42.6% book vs +18.9% bench, **alpha +20.3%** | +1.04% vs +5.27%/period, **excess −4.23%** |
| ranker | **sector-neutral `mom_6_1_r`, single column** | **the fitted Fama–MacBeth composite** |
| exits | 8×ATR disaster floor only; no target, no invalidation | 2.5×ATR stop, 3R target, MA50−1.5ATR invalidation |
| sizing | equal weight | risk-budget |
| sample | 258 trades, 2018-11-27 → 2026-08-25 | 35,730 rows / 85 dates, 2019-02-18 → 2025-02-03 |
| producer | `horizon-map-2026-08` trade-level study | `validation/portfolio_sim.py` + CPCV harness |

They do not contradict each other; they answer different questions. What is false is Table B's
header — *"If this table and a figure elsewhere disagree, this table is the live one"* — since
Table B's stop, target, invalidation and sizing were all removed before the current config
shipped. **Neither table describes `v3_composite` with the live book geometry. The shipped
configuration has no book-level measurement at all**, which is exactly what
`v3.py:BOOK_NOTE` says and the README's summary tables obscure.

---

**D-018 · P1 · VERIFIED · `src/prosignal/stages/stage4_core_score.py:731`**
`CoreScoreReport.effective_weights` is populated from the *discarded* family path. On the
live run it reads `{'momentum_12_1': 0.4688, 'sector_relative_strength': 0.5312}` — two
factors that rank nothing, presented as the model's effective weights, carried into the
report contract (`core/contracts.py:388`) and available to any consumer. The v3 composite's
actual effective weights (§2.3) are computed nowhere and reported nowhere.

The same run appends the note *"Regime 'range_lowvol' multipliers applied (momentum x0.75,
sector-RS x0.85), then weights renormalised"* (`stage4:447-451`). Under `v3_composite` those
multipliers move **nothing** — they scale the family composite that is thrown away. The
`redundancy` report (`stage4:696-711`) likewise runs on `frame`, the two-factor legacy block,
because `model_features` is permanently `None` (`stage4:490`).

---

**D-019 · P1 · VERIFIED · `src/prosignal/features/v3.py:143`, `:156-160`**
**"Sector-neutral" is false for 38.9% of the universe.**

`sector_neutral_rank` maps missing/`Unknown` sectors to `__RESID__`, then folds every sector
with fewer than `MIN_SECTOR_NAMES = 12` members into the same bucket. On the live
cross-section of 386 eligible names:

```
__RESID__                       150   (38.9%)   ← 79 genuinely Unknown + 71 from 13 real sectors
Capital Goods                    55
Financial Services               45
Healthcare                       39
Automobile and Auto Components   25
Chemicals                        21
Consumer Services                20
Fast Moving Consumer Goods       17
Information Technology           14
```

Folded in: Construction, Construction Materials, Consumer Durables, Forest Materials, Media,
Metals & Mining, Oil Gas & Consumable Fuels, Power, Realty, Services, Telecommunication,
Textiles, Utilities. **A Power stock is neutralised against Realty, Telecom and 79
unclassified names.** Any sector tilt inside `__RESID__` is not neutralised at all.

*The seeded "42% UNCLASSIFIED" claim is refuted as stated.* `data/curated/sector_map.parquet`
(the live map, 754 rows) has **zero** UNCLASSIFIED and zero nulls. The 42% belongs to
`data/curated/v5_aux/sector_map_imputed.parquet` — a research artefact with 607 of 1,049 rows
UNCLASSIFIED and `coverage_after: 0.4214` in its metadata — which **nothing in `src/` reads**
(verified by grep). The live number is: the map covers 611 of the 750-name universe, 139
(18.5%) fall through to `"Unknown"`, and the bucket that actually matters is the 38.9%
`__RESID__` above. The treatment is the better of the two possible bugs — a residual group,
not a fake sector — but it is far larger than anyone has been told.

---

**D-020 · P1 · READ + VERIFIED count · `src/prosignal/stages/stage3_eligibility.py:221`**
**A trend filter justified by a deleted model removes 27% of the universe before ranking.**

`OUTSIDE_MODEL_DOMAIN` rejects **203 of 750 names (27.1%)** on the live run — the largest
single gate. Its reason text reads *"Rs 706.65 is below the thesis-invalidation level of Rs
738.87 (50-session average less 1.5 ATR). **The model's coefficients were fit…**"* — the
fitted cross-sectional model, removed 2026-09-03. HDFCBANK is among the rejected.

Two problems. First the justification is stale. Second, and worse: **this is a momentum/trend
screen applied to the population before a composite that already carries 48.55% momentum.**
It compounds the concentration D-001 creates, and it is invisible in the weight decomposition
because it acts on the universe rather than on the score.

Full funnel, VERIFIED, 2026-09-03:

```
universe (liquidity screen)   750     183 ETFs/funds excluded as non-equity
  OUTSIDE_MODEL_DOMAIN       -203     27.1%
  EARNINGS_CONFLICT           -84     11.2%
  INSUFFICIENT_HISTORY        -42      5.6%   (SBIN among them, see D-013)
  DATA_QUALITY                -35      4.7%   (26 session_continuity, 19 unexplained CA)
eligible                      386
scored                        386
defended (top 10% by rank)     39     the scarcity gate, D-006
survived defense               37
passed score threshold         37     0 rejected — the gate is a rank, D-006
triggered                       1     entry cadence closed: session 2 of 21
buys                            1     LLOYDSME, carried
```

---

**D-021 · P1 · VERIFIED · `data/curated/trial_registry.jsonl`, `config/parameters.yaml:2769`, `README.md:285`**
**Three irreconcilable trial counts, and the honest number is none of them.**

`prosignal research trials` reports **99 recorded + 20 carried = 119 CHARGED BY THE DSR**.
`config/parameters.yaml:2769` reports **DSR 0.030 against 4,877 trials — FAILS**.
`README.md:285` reports **DSR 0.346 charging 81 trials — FAIL**.

Counting honestly, as the brief asks — every configuration ever evaluated against
out-of-sample data, including abandoned ones — the floor is:

- 4,877 trade-level configurations (the `horizon-map-2026-08` sweep, `README:57-59`)
- 144 fitted-composite configurations (`README:72`)
- the v3 search that produced the 22 factors and 5 themes (search code deleted 2026-09-03; the
  registry does not carry its arms — **the count is unrecoverable from this repo**)
- v4 through v9: six model generations built and not shipped, each with its own sweep
- 119 arms in the registry, of which the DSR's own `MIN_TRIAL_SCORE_COVERAGE` guard
  (`metrics.py:MIN_TRIAL_SCORE_COVERAGE = 0.5`) notes only 18 carried a score, forcing the
  conservative unit variance

**The honest count is "at least 5,140 and not reconstructible."** That is itself the finding:
the registry was created after most of the search happened, so no deflated Sharpe computed
from it can be trusted, in either direction.

*What is well-implemented, and verified as such:* `newey_west_t` reproduces a reference
Bartlett-kernel HAC estimator to 1e-9 at lags 0, 2 and 5; `analytic_vif(63,21,n)` converges to
the correct asymptotic 3.0; `overlap_lag(63,21) = 2`; `probabilistic_sharpe_ratio` and
`expected_max_sharpe` match Bailey & López de Prado exactly (including non-excess kurtosis in
`_moments`); `compute_pbo` is textbook CSCV. **The statistical machinery is not the problem —
the inputs are.**

---

**D-022 · P1 · READ · `src/prosignal/api.py:188-283`**
`/ready` checks price depth, staleness against the Stage 1 tolerance, and the presence of a
**`NIFTY 200` universe snapshot** — an index the live `liquidity_pit` universe never consults
(`pipeline.py:644`). So it fails closed on an irrelevant condition while checking **nothing**
about: fundamentals freshness (18 months stale, D-011), sector-map coverage (18.5% Unknown,
D-019), delivery table currency, corporate-actions currency, or per-factor coverage (quality
at 8.8%, D-007). A store with dead fundamentals and a broken quality theme reports `ready:
true`.

---

**D-023 · P1 · VERIFIED · `src/prosignal/stages/stage3_eligibility.py:257-258`**
**The earnings blackout uses a 45-session window against a 63-session hold, and only bites where the calendar has data.**

84 of 750 names (11.2%) rejected as `EARNINGS_CONFLICT` — *"results due in ~32 sessions,
inside the 45-session holding window"*. The planned hold is **63 sessions**
(`trade_plan.planned_hold_sessions`), so the window under-covers the actual exposure by 18
sessions.

The selection bias is the larger problem. On a quarterly-reporting universe a genuine
45-session blackout would catch ~70% of names at any time; it catches 11%. The difference is
`earnings_calendar.parquet` coverage. **A name the calendar does not know is not excluded** —
so the gate systematically favours the less-covered, smaller, less-followed names, which are
exactly where the top of the ranking sits (SJS, GRWRHITECH, AVL, MBAPL, IKS).

---

**D-024 · P1 · VERIFIED · `data/curated/delivery/` vs `data/curated/prices/`**
**The ownership theme did not exist for the first 23% of the price history.**

Delivery data begins **2019-06-27**; prices begin **2017-09-08**. **508 of 2,219 price
sessions (22.9%) have no delivery data at all**, and `deliv_pct` is null on a further 9.69% of
the rows that exist. On every one of those dates all three ownership factors are NaN, the
theme drops out, and D-001's renormaliser silently pushes momentum from 48.55% to ~64%.

The v3 training window is 2018-11-27 → 2024-10-25. Its first ~145 sessions (about 10%) were
scored by a **4-theme model with momentum at 64%**, and the remainder by a 5-theme model. The
frozen weights were fitted across that structural break.

---

**D-025 · P1 · VERIFIED · `pytest` (3 failures of 1,641), `tests/test_restart_gate.py:290`, `tests/test_data_manifest.py`**
**The repository's own restart gate says the engine is not ready to be restarted.**

Full suite: **3 failed, 1,638 passed, 4 skipped, 873s**. Two of the three are real:

```
DATA gate not met: manifest 75152d895cd25607 does not describe the store:
  delivery/year=2026.parquet: 5,555,905 bytes recorded, 5,585,439 on disk;
  fo_lots.parquet: present on disk and absent from the manifest;
  indices/year=2026.parquet: 1,234,600 recorded, 1,241,919 on disk (+6 more)
REPRODUCIBILITY gate not met: the tree has drifted from epoch
  2026-09-03-5e0c98515d13e6e2: code_sha 677e96e9ccd7 -> 1e9cefba57d4
```

`prosignal research epoch status` confirms the same drift. Every result produced since is
attributed to an epoch whose code no longer exists.

---

**D-026 · P1 · VERIFIED · `.gitignore:38-50`, `pyproject.toml`, `.python-version`**
**A clean clone cannot reproduce anything.**

`data/curated/*` and `data/ledger/*` are gitignored except four provenance files
(`MANIFEST.json`, `trial_registry.jsonl`, `epochs.jsonl`, `data/shadow/2026-08-18.json`), so a
clone has **zero** of the 251 MB the engine needs. Rebuilding it requires
`prosignal data ingest --full` against NSE — which now works (D-012), but takes hours and has
no documented single command.

Three Python versions are declared in one tree: `.python-version` says **3.12**, `.venv` is
**3.9.6**, and `__pycache__` artefacts are **cpython-310**. Dependencies are declared as
ranges, not pins, and there is **no lock file** — `requirements.txt:7-9` explicitly tells the
reader to make one and nobody has. Two identical clones on two machines will resolve different
numpy and pandas.

---

### P2 — correctness risk that has not fired yet, or will cause a P0 later

---

**D-027 · P2 · READ · `stage4:265-270`, `stage4:277-291`**
Delivery and PIT-fundamental failures are caught, logged at WARNING and the run continues.
Losing delivery costs a 19%-nominal theme and pushes momentum to ~57% effective; losing
fundamentals costs the quality theme. Neither raises, neither reaches the screen, and the only
visible symptom is a slightly different ranking. Given D-001, a swallowed feed failure is a
silent 8-point shift in the model's factor exposure.

**D-028 · P2 · READ · `src/prosignal/data/universe.py:206-210`**
The non-equity filter is wrapped in `except Exception` that logs *"the universe may contain
ETFs and funds"* and continues with an **empty** exclusion set. It also depends on
`equity_master` (2,568 rows) for its "this is a real company" test: if that read returns empty,
`known` is empty and every real company matching `SCHEME_PATTERN` — GOLDIAM, SKYGOLD,
PNBGILTS, SILVERTUC — is dropped instead. Two silent failure modes in opposite directions.
*Currently healthy:* 183 instruments excluded on the live run, and across all 1,942 ledger
rows **no non-equity instrument ever reached a book, slate or watchlist** (0 of 1,405
book-dates). **The seeded "gold ETFs on 17.6% of book-dates" is refuted against the live
record**; that figure describes a research backtest run before this filter existed
(`v3.py:52-58` records the same event at 26.25% of top-ten slots on window A).

**D-029 · P2 · VERIFIED · `stage4:274` + `v3_factors.py:41`**
`bench = close.mean(axis=1)` is the mean of **price levels** across the names being scored,
documented as "the equal-weight return of the ELIGIBLE universe". It is price-weighted, and
on dates where the non-NaN column count changes its implied return is an artefact — measured
sd 0.0156 on composition-change dates against 0.0110 on stable ones, a 42% inflation. Only
`resid_rev_21` consumes it.

**D-030 · P2 · VERIFIED · `data/curated/corporate_actions.parquet`**
Three rows carry `ratio > 1.0` for a split/bonus, which is arithmetically impossible:
`PATANJALI 2019-11-14 = 100.0`, `JSWSTEEL 2005-02-21 = 22.857`, `LT 2004-05-19 = 2.0`. All
three are inert today (each symbol has **0** price rows before its ex-date) but there is no
range validation on ingest, so the next one that lands inside the window multiplies a name's
entire prior history by 100.

**D-031 · P2 · READ · `pit_fundamentals.py:36` vs `config/parameters.yaml`**
Two staleness limits for one concept: `MAX_AGE_DAYS = 420` hardcoded and used by the v3
quality factor (`v3_factors.py:132`), `stage4.max_fundamental_age_days = 450` in config and
used only by the discarded family path.

**D-032 · P2 · VERIFIED · `src/prosignal/data/store.py:315`**
`DataStore.__init__` calls `self.curated.mkdir(parents=True, exist_ok=True)`. Constructing a
read-only store has a filesystem side effect, and it is why
`tests/test_remediation_guards.py:413` fails on this machine (`OSError: Read-only file system:
'/nonexistent-curated'`). The guard it tests is intact; the constructor is the defect.

**D-033 · P2 · READ · `stage4:530`**
`composite_raw.sort_values(ascending=False)` uses pandas' default quicksort, which is not
stable. On tied scores the rank order — and therefore the book — is not reproducible.
*Currently latent:* 386 distinct score values over 386 names, zero ties.

**D-034 · P2 · VERIFIED · `data/ledger/runs-*.jsonl`, `data/ledger/epochs.jsonl`**
The running config `baseline-v2@74ad0b78ffe70cc0` appears in **zero** of 1,942 ledger rows.
The open epoch `2026-09-03-5e0c98515d13e6e2` registered `baseline-v2@74ad0b78ffe70cc0` and
code `677e96e9ccd7`; the tree is `1e9cefba57d4`. Every artefact in the repo is attributed to
an identity that no longer matches.

**D-035 · P2 · VERIFIED · `data/curated/security_list.parquet`**
`asm_flag` is **`False` on all 3,534 rows** — the ASM feed is not populated. GSM is thin (77
rows across 6 stages). `is_t2t` True on 278, `restricted` True on 918, `band_pct` null on 540.
23 of the 386 eligible names carry a GSM/T2T/restricted flag and **2 are in today's top 20**.
Given D-012 the ASM list is fetchable.

**D-036 · P2 · VERIFIED · `data/curated/fo_lots.parquet`**
**The wanted short leg is unbuildable on this universe.** Of 216 F&O symbols (210
single-stock, 6 index, snapshot 2026-09-04): **209 of 750 universe names (27.9%)** and **50 of
386 eligible names (13.0%)** have futures. Of the current **bottom 20 — the short candidates —
exactly 1 (KPITTECH)**. Of the top 20, 2. Lot sizes run 20 to 71,475 (median 625), so a ₹1.67
lakh slot cannot express most positions at all. F&O ban periods, margin and roll cost are not
modelled anywhere in `src/`.

**D-037 · P2 · VERIFIED · `sys.settrace` line coverage over one real scan**
**30.8% of statement lines (5,240 of 16,991) execute on a live run.** Module-level import
reachability is nearly total (94 of 106 modules reachable from the API/pipeline, 102 from
either entry point, 4 orphans and all of them `__init__.py`) — so the dead code is *inside*
live modules, not in unreachable files. Largest zero-coverage blocks on the signal path:
`features/crossmodel.py` (452 stmts, 0%), `features/famamacbeth.py` (248, 0%),
`features/fundamental_factors.py` (179, 0%), `features/v9r.py` (68, 0%),
`features/linear.py` (105, 13.3%), `features/crosssec.py` (254, 9.8%).

**D-038 · P2 · VERIFIED · `stage4:447-451`**
The regime multiplier note is appended to `notes` on every run and moves nothing under
`v3_composite` (see D-018). It is carried out of the run by `AnalysisRun.scoring_notes`.

---

### P3 — dead code, cleanup, cosmetics

**D-039 · P3 · VERIFIED · `pyflakes src/prosignal`** — 97 findings: 68 unused imports, 19
unused locals, 4 f-strings without placeholders, 4 duplicate dict keys, 2 undefined names.
Concentrated in `cli.py` (48).

**D-040 · P3 · VERIFIED · `src/prosignal/validation/metrics.py:701-716`** — `V2Recheck` is
defined once, referenced nowhere, and annotates `as_of: dt.date` where `dt` is never imported.
`from __future__ import annotations` makes it latent; `typing.get_type_hints(V2Recheck)` raises
`NameError: name 'dt' is not defined`.

**D-041 · P3 · VERIFIED · `src/prosignal/validation/metrics.py:277-284`** — `DsrResult.to_dict`
repeats `sr_variance` and `sr_variance_source`. Same values; harmless; a linter signal that the
method was edited twice.

**D-042 · P3 · VERIFIED · `tests/test_remediation_guards.py:413`** — the fixture path
`/nonexistent-curated` is at filesystem root, which macOS mounts read-only. Test fails for an
environment reason; see D-032 for the constructor that makes the path matter.

---

## 4. Phased plan

Ordered by dependency. Every phase closes numbered defects; every exit gate is a runnable
assertion. Estimates are agent-hours including the verification run.

---

### Phase 0 — Fence the record before changing anything · **4 h**
**Closes D-008, D-025, D-034.** Touches `src/prosignal/ledger.py`, `data/ledger/` (a new
partition, no deletion), `scripts/`.

Nothing downstream can be measured while the open book is decided by file-append order.

1. Add a `run_kind` field to the ledger row (`live` | `backfill` | `replay`) and make
   `Ledger.previous_run` select `run_kind == "live"` only, ordering by
   `(date, logged_at, run_id)` with an explicit deterministic tiebreak — not `>=` over file
   order (`ledger.py:131`).
2. Rewrite `previous_run` to **raise** when a single date carries conflicting
   `signals_generated`, rather than silently taking the last one.
3. Re-manifest the store (`data manifest`) and open a new epoch against
   code `1e9cefba57d4`, so `test_restart_gate` passes for a real reason.

**Exit gate**
```bash
.venv/bin/python -m pytest tests/test_restart_gate.py tests/test_data_manifest.py -q
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0,"src")
from prosignal.config.loader import load_config
from prosignal.ledger import Ledger
import datetime as dt, collections, json, glob
rows=[json.loads(l) for p in sorted(glob.glob("data/ledger/runs-*.jsonl"))
      for l in open(p) if l.strip()]
live=[r for r in rows if r.get("run_kind")=="live"]
d=collections.Counter(str(r["date"])[:10] for r in live)
assert max(d.values())==1, f"a live date still carries {max(d.values())} runs"
PY
```
*Scorecard:* Research Credibility 6.5 → 7.0.

---

### Phase 1 — Fix the scorer · **10 h**
**Closes D-001, D-002, D-007, D-010, D-024 (disclosure half).** Touches
`src/prosignal/features/v3.py`, `src/prosignal/features/v3_factors.py`.

This is the phase. Nothing else in the plan is worth doing first.

1. **D-001.** In `score_frame` (`v3.py:249-256`), replace the bare renormaliser with a
   per-name call to `cap_weights` over the themes that name actually has, at the same
   `cap=0.40, floor=0.06` the fit used. The function already exists and is already correct.
   *Measured effect: momentum 48.55% → 39.47%; risk 13.46% → 17.10%; reversal 13.33% →
   17.00%; Spearman 0.977 to the current ranking; 1 of 6 book names and 2 of 20 top names
   change.*
2. **D-002 — DONE, and the answer was "do not touch the signs".** The search artefacts
   (`work/v3/themes.py:92`, `research/V3_SEARCH.md:150`, `research/v3/FROZEN_V3.json`)
   settle it: the double negative is inside the validated model and the sign is a
   measurement at t −5.52. The theme is misnamed, not mis-signed, and only in code —
   the screen has said "Low-margin tilt" since `fae96e2`. Closed by `Theme.label`,
   carried into the run note so the record and the screen agree.
3. **D-007.** Set `MIN_THEMES` behaviour so a name is scored only on themes whose *live*
   coverage supports them, and emit `n_themes` and the per-name effective weights on the
   report contract so D-018 and D-003 have something true to display.
4. **D-010.** Raise `v3_factors.LOOKBACK_SESSIONS` from 300 to **420** (measured convergence
   at 400; 420 gives margin) and add an explicit assertion that the residual chain's window
   is inside the loaded frame rather than relying on `min_periods` to paper over it.
5. **D-024 disclosure.** Emit a per-theme coverage line on the report whenever a theme's live
   coverage is below its frozen `Theme.coverage`, so the 8.8%-vs-18.99% gap is visible.

**Exit gate**
```bash
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0,"src"); import numpy as np, pandas as pd
from prosignal.features import v3, v3_factors
assert v3_factors.LOOKBACK_SESSIONS >= 420
# no theme exceeds the cap for any name, on live data
scored = ...  # build_v3_block on the last session
eff = ...     # per-name effective weights now emitted by score_frame
assert eff.max().max() <= 0.4 + 1e-9, "a theme still exceeds the 40% cap"
# resid_rev_21 is window-invariant
a = factor_frame_at(depth=420); b = factor_frame_at(depth=1200)
assert np.nanmax(np.abs(a["resid_rev_21"]-b["resid_rev_21"])) < 1e-9
# the quality theme is oriented as named
assert np.corrcoef(sd_net_margin, quality_signed)[0,1] < 0
PY
.venv/bin/python -m pytest tests/test_v3_score.py tests/test_new_factors.py -q
```
*Scorecard:* Signal Credibility 2.5 → 5.0; Factor Diversification 3.0 → 6.0.

---

### Phase 2 — Make the screen tell the truth · **8 h**
**Closes D-003, D-004, D-005, D-018, D-038.** Touches
`src/prosignal/presentation/viewmodel.py`, `src/prosignal/presentation/evidence.py`,
`src/prosignal/static/index.html`, `src/prosignal/stages/stage4_core_score.py`,
`config/parameters.yaml`.

Depends on Phase 1 emitting per-name effective weights.

1. **D-003.** `viewmodel.py:634` must serve the **effective** weight, not `Theme.weight`, so
   `z × weight == contribution` holds on the card. Add the missing theme as an explicit row
   ("quality — no data, its 18.99% was redistributed") rather than a blank.
2. **D-004.** Replace the key-collision heuristic in `_scorer_used` with the
   `ranking_source` string Stage 4 already computes and carries
   (`stage4:510`, `_apply_ranking_policy` returns it). Namespace the v3 theme keys
   (`v3:reversal`) so no future collision can re-certify a deleted model. Then make
   `index.html:2182` render the honest state for `v3_composite`: **ranking evidenced on two
   sealed windows, book geometry not validated** — which is what `v3.BOOK_NOTE` says.
3. **D-005.** Set `expectancy.enabled: false` until a study measured on `v3_composite` with
   the live book geometry exists. The card fields become null, which is correct; a frequency
   from a different model is worse than no frequency.
4. **D-018 / D-038.** Populate `CoreScoreReport.effective_weights` from the v3 blend. Delete
   the regime-multiplier note and the redundancy report from the `v3_composite` path, or
   compute them on the columns that actually rank.

**Exit gate**
```bash
.venv/bin/python - <<'PY'
# on the live run, for every card and every theme row:
#   abs(z*weight - contribution) < 1e-9
#   and the weights on a card sum to 1.0
# and the scorer block reports v3_composite with validated=False
assert view["scorer"]["model"] == "v3_composite"
assert view["scorer"]["validated"] is False
for p in view["picks"]:
    tot = sum(t["coefficient"] for t in p["themes"])
    assert abs(tot - 1.0) < 1e-6
    for t in p["themes"]:
        assert abs(t["z"]*t["coefficient"] - t["contribution"]) < 1e-9
PY
```
*Scorecard:* Research Credibility 7.0 → 8.0.

---

### Phase 3 — Repair the data layer, now that it is known to be reachable · **16 h**
**Closes D-011, D-012, D-013, D-014, D-024, D-030, D-031, D-035.** Touches
`src/prosignal/data/ingest.py`, `src/prosignal/data/store.py`,
`src/prosignal/data/corporate_actions.py`, `src/prosignal/data/providers/`,
`config/parameters.yaml`.

Independent of Phases 1–2; runnable in parallel.

1. **D-012 first, as a checklist item, not a code change:** run the probe in §D-012 on
   `prosignal.duckdns.org`. Everything below assumes it passes there as it does here.
2. **D-013.** Accept `T0` alongside `EQ` in `DataStore.equity_series`, or preferably backfill
   the missing `EQ` closes; either way recover the 183 lost sessions across 44 names and get
   SBIN back into the universe. Make the store read `universe.allowed_series` instead of
   keeping a second hardcoded copy.
3. **D-011.** Fix the symbol source at `ingest.py:890-894` — the live 750-name liquidity
   universe, not a `NIFTY 200` snapshot. Then find the query shape that reaches quarters after
   2024-12-31 (`statements.parquet` already has them to 2026-06-30, so a cross-check target
   exists), and add a hard staleness assertion: **a feed that answers with nothing newer than
   what is stored is a failure, not a success.**
4. **D-014.** Wire `build_dividend_factors` into `apply_adjustments` so the price series is a
   total return. This changes every momentum and risk factor, so it must land *before* any
   re-measurement in Phase 6 and be recorded as a new epoch.
5. **D-024.** Backfill delivery to 2017-09-08 if NSE serves it; if not, restrict the training
   window to the delivery era rather than fitting across the break.
6. **D-030.** Validate corporate-action ratios on ingest: reject `ratio <= 0` or `ratio > 1`
   for `split`/`bonus`/`split_or_bonus`.
7. **D-031.** One staleness constant.
8. **D-035.** Populate the ASM list.

**Exit gate**
```bash
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0,"src"); import pandas as pd
f = pd.read_parquet("data/curated/fundamentals.parquet")
assert pd.to_datetime(f.filing_date).max() >= pd.Timestamp("2026-06-30"), "fundamentals still stale"
assert f.symbol.nunique() >= 600, f"only {f.symbol.nunique()} symbols covered"
ca = pd.read_parquet("data/curated/corporate_actions.parquet")
sp = ca[ca.action_type.isin(["split","bonus","split_or_bonus","bonus+split"])]
assert sp.ratio.between(0, 1, inclusive="neither").all(), "impossible split ratio present"
assert (ca[ca.action_type=="dividend"].ratio != 1.0).any(), "dividends still unadjusted"
sl = pd.read_parquet("data/curated/security_list.parquet")
assert sl.asm_flag.any(), "ASM feed still empty"
PY
# SBIN is back
.venv/bin/python -c "..."  # assert 'SBIN' in universe.symbols
```
*Scorecard:* Data Integrity 5.5 → 8.0; Indian-Market Fit 6.5 → 7.5.

---

### Phase 4 — Give the engine a way to say no · **8 h**
**Closes D-006, D-019, D-020, D-023.** Touches `config/parameters.yaml`,
`src/prosignal/features/v3.py`, `src/prosignal/stages/stage3_eligibility.py`,
`src/prosignal/stages/stage8_final_signal.py`.

Depends on Phase 1 — an absolute bar on a mis-weighted score is a bar on the wrong thing.

1. **D-006.** Replace the rank-based scarcity gate with a **book-level cash rule**, which is
   what `config/parameters.yaml:1112-1116` itself proposes: hold cash when fewer than N names
   clear a stated absolute bar. The measured evidence against the old `absolute_floor` (ATE
   −2.2%, 95% CI [−3.2%, −1.4%]) was against a *per-name entry* floor and does not transfer to
   a book-level rule; re-derive on the training window only. Delete the redundant
   `min_composite_score`/`min_universe_percentile` pair — two scales of one rank.
2. **D-019.** Either raise sector-map coverage above 90% (D-012 makes the NSE industry feed
   reachable) or lower `MIN_SECTOR_NAMES` and report `__RESID__` size on every run. Ranking
   38.9% of the universe inside a synthetic sector must be visible, whichever way it is fixed.
3. **D-020.** Re-justify or remove the `OUTSIDE_MODEL_DOMAIN` filter. It cites coefficients
   from a deleted model to remove 27% of the universe on a trend condition, in front of a
   composite that is already 40–49% momentum. Measure it as a treatment on the *current*
   scorer, on the training window.
4. **D-023.** Align the earnings window with the 63-session hold, and make an
   earnings-calendar **gap** report NOT_TESTABLE rather than pass — otherwise the gate
   selects for poor coverage.

**Exit gate**
```bash
# NO TRADE must be constructible. Replay the COVID trough through the current scorer
# and assert the book holds cash.
.venv/bin/python -m prosignal.cli analyse run --date 2020-03-24 --dry-run
# assert: no_trade is not None and reason cites the cash rule, not the cadence
# and assert the reverse: on a normal date it does NOT fire
```
*Scorecard:* Robustness 3.5 → 6.5.

---

### Phase 5 — Price the book honestly · **6 h**
**Closes D-015, D-036.** Touches `config/parameters.yaml`,
`src/prosignal/presentation/viewmodel.py`, `src/prosignal/static/index.html`.

1. **D-015.** The engine already computes 83–151 bps per round trip. Make the displayed and
   ledgered cost assumption **come from `CostModel` for the actual name and size**, not from
   a frozen 40 bps. Publish the capacity curve (§D-015) on the screen: at the ADTV floor the
   edge is gone by ~₹1 crore.
2. **D-036.** State the short-side finding as a constraint, not a backlog item: **1 of the
   bottom 20 ranked names has a single-stock future.** A short leg on this universe is not
   available. If the short side is still wanted, it needs a different universe (F&O-eligible
   only, ~209 names), which is a different model and belongs after Phase 6, not before it.

**Exit gate**
```bash
.venv/bin/python - <<'PY'
# every issued card's cost figure equals CostModel.round_trip for that name and size
for card in buys:
    cb = costs.round_trip(card.entry, qty(card), adtv_inr=adtv[card.ticker])
    assert abs(card.trade_plan.assumed_cost_bps - cb.total_bps_of_buy) < 1.0
PY
```
*Scorecard:* Indian-Market Fit 7.5 → 8.0.

---

### Phase 6 — Re-measure and re-register · **20 h**
**Closes D-009, D-016, D-017, D-021, D-022.** Touches `README.md`,
`src/prosignal/validation/`, `data/ledger/forward_test.json` (new), `src/prosignal/api.py`.

**Must come last.** Every phase above changes the signal; measuring before them measures
something that will not ship.

1. **D-021.** Write the honest trial count into the registry with its provenance and its
   irrecoverable component, and stop quoting three different numbers. Recompute the DSR with
   `sr_variance` supplied rather than falling back to unit — the guard at
   `metrics.MIN_TRIAL_SCORE_COVERAGE` is correct and should be respected, not worked around.
2. **D-017.** Delete both README book tables. Replace with one table measured on
   `v3_composite` at the live geometry, on the **training window only**, plus an explicit
   "the shipped configuration has no out-of-sample book measurement" line.
3. **D-016.** Partition `outcomes.jsonl` by epoch and stop pooling 128 `baseline-v1` trades
   with a median 3-session hold into anything.
4. **D-009.** Re-register the forward test against the post-Phase-5 config, with the
   benchmark-relative hypothesis `forward.py:521-528` requires. Register it as
   **falsification** — per the constraint already recorded in this project, a paper trade needs
   ~6.5 years to reach t = 2.0 at this book size, so it can refute the model but cannot ship it.
5. **D-022.** Extend `/ready` to check what the live path actually uses: fundamentals age,
   sector coverage, per-theme live coverage, delivery currency, corporate-actions currency.
   Drop the `NIFTY 200` snapshot check.

**Do not open the sealed window 2025-01 → 2026-08 for any of this.** It has been opened five
times; it is burnt for selection and its use here would be invalid.

**Exit gate**
```bash
.venv/bin/python -m prosignal.cli research forward     # must print a registered, VALID window
.venv/bin/python -m prosignal.cli research ready       # every dimension reports its own number
grep -c "20.3%\|+42.6%\|4.23%" README.md               # must be 0
```
*Scorecard:* Research Credibility 8.0 (held); Robustness 6.5 → 7.5.

---

### Phase 7 — Delete and pin · **6 h**
**Closes D-026, D-032, D-033, D-037, D-039, D-040, D-041, D-042, and the deletion list in §5.**

1. Execute §5 as one commit.
2. **D-026.** Generate `requirements.lock.txt`, reconcile `.python-version` / `.venv` /
   `pyproject.toml` on one interpreter, and write the single documented bootstrap command.
3. **D-032.** Remove the `mkdir` side effect from `DataStore.__init__`; move it to the write
   path. **D-042** then passes for the right reason.
4. **D-033.** `sort_values(..., kind="stable")` on the ranking sort.
5. **D-039/40/41.** Clear pyflakes to zero on `src/`.

**Exit gate**
```bash
.venv/bin/python -m pyflakes src/prosignal   # zero output
.venv/bin/python -m pytest -q                # 0 failed
git clean -xdn | grep -q . && echo "clean-clone bootstrap documented and tested"
```
*Scorecard:* no dimension moves; this is the phase that stops the next P0.

---

**Total: 78 agent-hours.** Phases 0 → 1 → 2 are the critical path and are 22 of them.

---

## 5. Deletion list

One commit, after Phase 6 (`research_panel` and `/admin/model` depend on some of it until then).

### Whole files — the removed fitted model, still imported and never executed

| file | lines | live-run coverage | evidence |
|---|---|---|---|
| `src/prosignal/features/crossmodel.py` | 1,387 | **0.0%** (452 stmts) | `stage4:483-497` records the fit as removed; imported only by `api.py:876` (`/admin/model`), `viewmodel.py:518` (one constant), `data/coverage.py:104`, `validation/research_panel.py:26`, and 3 CLI research commands |
| `src/prosignal/features/famamacbeth.py` | 735 | **0.0%** (248 stmts) | imported by `cli.py:1701,1811` and `validation/decay.py:39` only |
| `src/prosignal/features/fundamental_factors.py` | 429 | **0.0%** (179 stmts) | named in `modelprint.py:41` and nothing else |
| `src/prosignal/features/linear.py` | 105 stmts | 13.3% | named in `modelprint.py:40`; the 13.3% is import-time only |

**2,656 source lines.** Before deleting: `viewmodel.py:518` needs `FEATURE_COLUMNS` inlined,
`data/coverage.py:104` needs its guard expression restated, and `/admin/model` should be
deleted with them — it reports a model that does not choose anything.

### Functions and classes

| symbol | address | evidence |
|---|---|---|
| `V2Recheck` | `validation/metrics.py:701-716` | defined once, referenced nowhere (grep), and `get_type_hints` raises `NameError: dt` |
| `_scorer_used`'s composite branch | `viewmodel.py:135-146` | unreachable — the `MODEL_KEYS` collision (D-004) returns before it on every run |
| `Theme.coverage` field consumers | `features/v3.py:88` | the frozen coverage figures are never read at scoring time; either use them (Phase 1) or drop the field |
| `_LEVEL_RUNG["target_1"/"target_2"]` paths | `viewmodel.py:652-657` | correct today, but `target_achieved` is permanently disarmed; keep only if Phase 4 rearms it |

### Config

| key | address | evidence |
|---|---|---|
| `stage4_core_score.ranking.column: mom_6_1_r` | `config/parameters.yaml:1054` | read only under `source: measured_factor` (`stage4:196`), which is not the shipped source |
| `universe.index_name`, `universe.pre_snapshot_policy` | `config/parameters.yaml` | dead under `source: liquidity_pit` (`pipeline.py:644`) — but see D-011, `ingest.py:890` reads `index_name` and *should not*, so fix the reader before deleting the key |
| `stage8_final_signal.scarcity.min_composite_score` | | duplicate scale of `min_universe_percentile` (D-006) |
| `stage4_core_score.max_fundamental_age_days` | | superseded by `MAX_AGE_DAYS` (D-031) |

### Research artefacts confirmed unread by `src/`

`data/curated/v5_aux/sector_map_imputed.parquet` and `sec_map_meta.json` — grep for
`sector_map_imputed` across `src/` returns nothing. Keep as research provenance, but they must
stop being cited as live figures (they are the source of the "42% UNCLASSIFIED" claim, D-019).

### Do NOT delete

`features/v9r.py` (0% coverage) is reachable through `ranking.source: v9r_core` and is the only
model in the repo with a sealed out-of-sample number. `validation/portfolio_sim.py`,
`cpcv.py`, `significance.py`, `metrics.py` are all correct and are needed by Phase 6.

---

## 6. Scorecard trajectory

| dimension | now | after | moved by | ceiling |
|---|---|---|---|---|
| **Research Credibility** | 6.5 | **8.0** | P0 (ledger fenced), P2 (screen honest), P6 (one table, honest trial count) | 8.0 |
| **Factor Diversification** | 3.0 | **6.0** | P1 (cap re-applied: momentum 63.6% → ~40% of spread; risk and reversal up 3.6pp each) | **6.5** — see below |
| **Data Integrity** | 5.5 | **8.0** | P3 (fundamentals current, T0 recovered, dividends adjusted, ASM populated), P0 (ledger) | 8.0 |
| **Indian-Market Fit** | 6.5 | **8.0** | P3 (ASM/GSM, T0), P5 (real costs, capacity published) | 8.0 |
| **Signal Credibility** | 2.5 | **5.0** | P1 (cap, quality signs, `resid_rev_21`) | **5.0** — see below |
| **Robustness** | 3.5 | **7.5** | P4 (NO TRADE reachable, sector neutralisation honest), P6 (forward test registered) | **7.5** — see below |

### Where 8 is not reachable, and why

**Factor Diversification — ceiling 6.5.** The composite has 5 themes and only 4 have live
coverage above 90%. Value, liquidity and seasonality were each built in full and each failed
the placebo screen (`v3.py:120-131`) — value at 0 of 8 factors, because balance-sheet data
begins in 2023 and the median training date has zero names with a book value. That is not an
egress problem and Phase 3 does not fix it. Genuine diversification past 4 price-and-volume
themes needs a fundamentals history this store does not have and NSE does not publish free;
it needs a paid point-in-time feed. **Reaching 8 requires spending money, not agent-hours.**

**Signal Credibility — ceiling 5.0 without new measurement.** After Phase 1 the engine
computes what it says it computes. It still has: no book-level out-of-sample measurement of
the shipped configuration (D-017), two spent sealed windows, a burnt 2012–2017 window, and a
forward test that needs roughly 6.5 years to reach t = 2.0 at a 6-name book. **The ceiling
here is set by arithmetic, not by code.** Credibility above 5 requires either a materially
larger book (which D-015's capacity curve permits up to ~₹1 crore) or a wider universe with
more independent bets — both of which are new research and both of which the brief correctly
defers until correctness is closed.

**Robustness — ceiling 7.5.** Phase 4 gives the engine a cash state and Phase 6 registers a
falsification test. What it cannot have is a regime it has not seen: the price store begins
**2017-09-08**, so the model has never been measured across 2008, 2013 or 2011. Pre-2018 NSE
history is available (`archives.nseindia.com` returns 200, D-012), so this ceiling is
liftable — but only by an ingest campaign that is itself several days of wall-clock, and any
model re-fitted on the longer history is a new out-of-sample question requiring a new epoch.

**Everything blamed on blocked egress is not blocked.** The three dimensions previously
capped by it — Data Integrity, Indian-Market Fit, and half of Factor Diversification — all
reach their targets in Phase 3. That is the single largest correction in this audit.

---

## 7. Paper-trading go-live checklist

Binary. Every item passes or blocks.

| # | Item | Now |
|---|---|---|
| 1 | No theme exceeds its 40% cap for any name on the live cross-section | **PASS** — 40.00% mean and max after D-001 |
| 2 | Every factor's sign matches its stated economic intent, or its name is corrected to match the sign | **PASS** — D-002 refuted; signs are measurements, `Theme.label` now carries the honest name into the record |
| 3 | Every factor at date T is invariant to the loaded window's start and end | **PASS** — D-010; a too-short window now yields NaN, not a number |
| 4 | `z × weight == contribution` for every theme row on every card | **PASS** — exact to 1e-12 on the blend, display precision on the card |
| 5 | The screen names the scorer that ran and does not claim it is validated | **PASS** — reports `v3_composite`, `validated: false`, as a disclosure not an alarm |
| 6 | No card carries frequencies measured on a different model | **PASS** — `expectancy.enabled: false` until re-measured |
| 7 | NO TRADE is reachable, and a replay demonstrates it | **PASS** — verified on 2020-03-31 and 2022-06-17 via the regime gate; a cross-sectional breadth bar added beside it |
| 8 | One live run per market date in the ledger; open book is unambiguous | **PASS** — 0 conflicts after `data lineage --repair`; 2 irreconcilable dates quarantined, 0 rows deleted |
| 9 | A forward test is registered and reports VALID | **BLOCK** (none registered, D-009) |
| 10 | Fundamentals reaching the model are ≤ 120 days old | **PASS** — `statements` newest period 2026-06-30 over **758 symbols**, 750/750 of the universe. `fundamentals.parquet` itself is still 2025-03-11 and stays that way: NSE's results endpoint stopped advancing, uniformly, and no query shape reaches past it |
| 11 | Prices are total-return adjusted | **BLOCK** (all dividend ratios 1.0, D-014) |
| 12 | Cost shown on a card equals `CostModel` for that name and size | **PASS** — `cost_note` is built from `RiskPlan.estimated_round_trip_cost_bps`, which `stage7_risk` computes per name and size; the flat 40 bps went with `expectancy.enabled: false` |
| 13 | The store manifest describes the store; the tree matches the open epoch | **BLOCK** (D-025, repo's own test) |
| 14 | Dependencies pinned; one interpreter; clean-clone bootstrap documented | **PARTIAL** — `requirements.lock.txt` written and the 3.9.6-dev vs 3.12-deploy divergence named in its header; the two interpreters are not reconciled and the bootstrap is not documented |
| 15 | Egress verified from the deploy host, not just the dev machine | **UNVERIFIED** (D-012 measured here only) |
| 16 | Every price session has a delivery row, or the theme is disclosed as absent | **BLOCK** (508 sessions missing, D-024) |
| 17 | `/ready` checks fundamentals age, sector coverage and freshness | **PASS** — reports fundamentals age, statements, sector map, delivery, corporate actions and ledger conflicts; the irrelevant NIFTY-200 snapshot check is gone |
| 18 | Each day's signal snapshot is written immutably with a content hash | **BLOCK** — `rundetail.save` is a display cache that swallows its own errors (`pipeline.py:453`); the ledger row has no content hash |
| 19 | A paper-trading ledger records intended entry, realised next-open, slippage, exit, and factor attribution | **BLOCK** — `outcomes.jsonl` has entry/exit/MAE/MFE but no intended-vs-realised-open slippage and no factor attribution; and all 128 rows are from a retired config (D-016) |
| 20 | Rolling IC decay, coverage and freshness alarms with a stated stop-emitting threshold | **PARTIAL** — `/ready` now alarms on fundamentals age and delivery staleness, and every run reports theme coverage, residual-bucket size and feed degradation. No IC-decay threshold exists on the live path |
| 21 | Full test suite green | **PARTIAL** — the two remaining failures are the manifest and epoch gates (#13), deliberately deferred |
| 22 | Job double-fire is impossible | **PASS** — single-flight, thread-locked (`jobs.py:263-299`), idempotent for the same kind, 409 otherwise |
| 23 | No lookahead: factors at T unchanged by sessions after T | **PASS** — VERIFIED across all 22 factors |
| 24 | Statistical machinery correct (NW, PSR, DSR, PBO, CPCV) | **PASS** — VERIFIED against reference implementations |
| 25 | Prices split/bonus-adjusted on the read path, vwap adjusted with them | **PASS** (`store.py:463-551`) |
| 26 | Non-equity instruments excluded from the equity universe | **PASS** — 183 excluded on the live run; 0 of 1,405 book-dates ever contained one |
| 27 | Corporate-action adjustment failure raises rather than serving raw prices | **PASS** (`store.py:539-551`) |
| 28 | Ledger write failure fails the run | **PASS** (`pipeline.py:409-419`) |
| 29 | The sealed 2025-01 → 2026-08 window is not used for selection anywhere in this plan | **PASS** |

**Re-scored 2026-09-05 after Rounds 0-6: 15 pass, 4 partial, 9 block, 1 unverified.**

Every blocker that was about the *engine* — the blend arithmetic, what the screen
says about it, the scorer's identity, the ledger's memory, NO TRADE, fundamental
coverage — is closed. What blocks now is a different and smaller class:

* **operational** (#13 manifest/epoch, #14 interpreters) — deferred on purpose,
  because re-manifesting while the code is still moving just re-breaks the gate;
* **record-keeping** (#18 content-hashed snapshots, #19 slippage and attribution,
  #20 an IC-decay threshold) — the things that make a paper-trading record
  scoreable, none of which exist yet;
* **data still genuinely missing** (#11 dividends, #16 pre-2019 delivery);
* **one probe I cannot run** (#15, the deploy host).

The system is materially closer to safe than the audit found it. It is not
safe: it has never been watched running the model as specified, which is what
#18-#20 exist to make possible.

---

## 8. Unverified

Everything I could not confirm, why, and what would confirm it.

1. **Egress from the deploy host.** D-012 was measured from the M1 development machine on
   2026-09-04. `prosignal.duckdns.org` may sit behind a different network. *Confirm:* run the
   §D-012 probe script on that host. One command; it decides whether Phase 3 is a 16-hour job
   or a data-licensing conversation.

2. **Whether the NSE quarterly-results archive genuinely stops at 2024-12-31, or the query
   shape is wrong.** I confirmed the endpoint answers and that its newest `toDate` for
   RELIANCE is 31-Mar-2024, and that `results_calendar.parquet` (1,288 symbols) shows the same
   ceiling. I did not try alternate `period` values, the announcements endpoint, or the XBRL
   index. *Confirm:* sweep `period=` variants and compare against
   `statements.parquet`, which already carries `period_end` to 2026-06-30 for 200 symbols.

3. **Why 44 symbols have T0-only sessions.** I confirmed the rows exist under series `T0` and
   that the store's `EQ` filter drops them. I did not determine whether NSE published no `EQ`
   row that day or whether the ingest failed to capture one. The fix differs: accept `T0`, or
   re-fetch. *Confirm:* pull the raw bhavcopy for 2025-06-23 and look for an `SBIN`/`EQ` row.

4. **Whether the v3 factor signs were fitted or chosen.** The search that produced
   `THEMES` was deleted on 2026-09-03 (`v3_factors.py:5`). I cannot tell whether
   `quality: (-1, -1)` was an empirical finding the search made or a transcription error.
   D-002's two fix options hinge on this. *Confirm:* `git log -S'"quality": Theme' -- src/prosignal/features/v3.py`
   and read the commit that introduced it; if the search code is in history, re-run it.

5. **The effect of D-001's fix on realised performance.** I measured the effect on today's
   ranking (Spearman 0.977, 1 of 6 book names changes). I did not measure whether the capped
   blend is *better* — that requires a backtest, and running one before Phase 3's data fixes
   would measure the wrong thing. *Confirm:* after Phase 3, CPCV on the training window only.

6. **Whether the `OUTSIDE_MODEL_DOMAIN` filter helps or hurts.** It removes 27% of the
   universe and I have no measurement of it against the current scorer. *Confirm:* Phase 4's
   treatment-effect run on the training window.

7. **The impact coefficient.** `costs.py` documents it as UNVALIDATED (0.1, exponent 0.5).
   Every capacity number in D-015 inherits that. *Confirm:* fit against realised slippage once
   the paper-trading ledger has fills — which is circular until go-live, so the honest position
   is to publish the curve with its assumption named.

8. **Whether 1,638 passing tests assert anything load-bearing.** I checked structure (1,561
   test functions, 29 skips, 0 xfail, 21 with no visible `assert`) and hand-read the two tests
   closest to D-003, both of which pass while the defect ships. I did not mutation-test.
   *Confirm:* mutate `v3.py:252` (`den` → `1.0`) and `v3.py:103` (sign → `+1`) and count
   failures. My expectation is that neither is caught, which would put a number on how much
   the suite is worth.

9. **The 2012–2017 sealed window's current status.** Recorded as opened once, +9.50% at
   t +1.87 against a pre-registered bar of 2.0 — a failed ship gate. I did not re-derive it and
   deliberately did not touch it. Nothing in this plan uses it.

10. **`v9r_core` as a live alternative.** It is selectable, 0% covered on a `v3_composite`
    run, and is the only model here with a sealed out-of-sample number. I did not run it.
    *Confirm:* shadow-run it beside the incumbent after Phase 1 — but note that its number is
    a **failed** gate, and switching on it would be reading a failure as a pass.

11. **Whether `ranking.source: v3_composite` has been the live setting for the whole ledger.**
    The ledger holds 27 config versions and `model_fingerprint` is null on 1,733 of 1,942 rows,
    so I cannot reconstruct which scorer produced most historical runs. *Confirm:* nothing can;
    this is why Phase 0 exists.

12. **The UI on mobile, and its behaviour during a long pipeline run.** I read
    `index.html` (3,406 lines) but did not render it. Double-fire is handled server-side
    (checklist #22); the client's disabled-state handling, the progress poll, and the mobile
    layout are unverified. *Confirm:* load it in the browser pane against a running instance
    and drive a scan.

13. **`/performance`, `/outcomes`, `/history` response shapes.** I traced the code and
    computed the filter's effect by hand (0 rows survive the live-config scope, `excluded_closed
    = 128`). I did not call the endpoints, because `_resolved_rows` resolves outcomes on read
    and may append to `outcomes.jsonl` — a data write the audit was not permitted to make.
    *Confirm:* copy `data/` to a scratch directory, point `paths` at it, and exercise the API.

14. **Ledger contamination before 2026-08.** I characterised the 2026 rows in detail. The 2023
    (18 rows), 2024 (122) and 2025 (116) partitions were profiled only at the summary level.
    *Confirm:* the Phase 0 script, run over all four files.

15. **Whether the 183 non-equity exclusions are all correct.** The filter excluded 183
    instruments on the live run and I spot-checked that the pattern catches all 53 liquid ones
    without the volatility backstop contributing anything. I did not verify that no real
    company is among the 183. *Confirm:* list them and eyeball against `equity_master`.
