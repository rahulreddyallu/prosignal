# Re-audit of the remediation — findings, prices and guards

Two independent passes over the engine after the September readiness dossier.
Everything below was regenerated on the tree as it stands, not read from the
dossier. Where my numbers and the dossier's disagree, both are shown.

Sections A–E are the **first pass**, which found and priced the defects on
`baseline-v1@127d8a314ec49aa2` without changing a traded value. Section F is
the **second pass**, which closed everything the first left open — and, because
one of those was the population the model is fitted on, refitted the
coefficients. Every number in sections C, D and E was computed on the old
panel. They are kept as the record of what was believed, and section F says
what each became.

**Suite 1,550 passed / 0 failed / 9 skipped, from 1,376 on a clean checkout.
Mutation probe 43 of 43 caught — four of them only after the tests meant to
catch them were found incapable of it. Config `baseline-v2@189efe9f49cb39ce`;
data manifest `6b6737fc418864aa`; research epoch
`2026-08-29-6451d9181041cdb4`. Holdout unspent — and unspendable without a
config change.**

---

## A · The verdict, and what changed about it

The dossier's economic conclusion is correct and I reproduce it independently:
**the shipped book returns 3.9 points per 63-session period less than an
equal-weight hold of the universe it selects from** (−3.88% on 68 CPCV-woven
out-of-sample dates, −4.35% on 35 purged walk-forward dates, against the
dossier's −3.85%). The ranking is weakly positive; everything below it is
subtractive.

What the dossier got wrong is narrower and worse: **three of the repairs it
reports as complete are not in the code**, and the most important of them means
the engine currently tells its operator it has passed a test it has not.

| | dossier says | engine actually said, before this pass |
|---|---|---|
| Deflated Sharpe | 0.346 at 81 trials — **FAIL** | `Deflated Sharpe 1.000 charging 81 trials -- **PASS**` |
| Forward test | re-registered against `127d8a31…`, not started | registered against `a30a8d48…`, **INVALID** |
| Tertiary hypothesis | registered while the window is closed | `grep -ri tertiary` → **no matches anywhere** |

Those are not disagreements of judgement. The first two are what the engine
printed when asked; the third is a search over the whole repository.

**The verdict is unchanged — NOT READY — and it is now the engine's own verdict
rather than a document's.** `research cpcv` prints FAIL. `research forward`
prints INVALID and names three independent reasons. After the second pass
`research readiness` prints it as eight gates, seven of which now pass; the
eighth is the forward test, which has not been restarted (§F8).

---

## B · Defect register

`PRICED` means the cost is measured, in return per 63-session period, on both
ranking constructions. `GUARDED` means a test fails if the fix is reverted, and
that reversion was performed and observed.

| id | finding | severity | status | price |
|---|---|---|---|---|
| **R1** | Forward test registered against a config the engine no longer runs. The only stated route to READY has not started. | **critical** | **GATED, GUARDED** — a restart is now refused unless the engine is ready, and names every reason. Still not restarted: that is an operator's decision (§F8) | — |
| **R2** | The benchmark-relative hypothesis exists nowhere: no field, no text, not in the fingerprint. A window could be passed by an engine that loses to holding its own universe. | **critical** | FIXED, GUARDED | — |
| **R3** | `CpcvResult.deflated()` scored 612 duplicated `(split, date)` pairs with a fallback variance of `1/(n−1)` documented as "a conservative unit variance". Returned **1.0000 PASS at any trial count**. | **critical** | FIXED, GUARDED | 1.000 → 0.000 |
| **R4** | The trial registry counted trials and discarded what they scored, so `Var[SR]` — the input the DSR is most sensitive to — had to be guessed. | **high** | FIXED, GUARDED | 0.38 ↔ 0.91 |
| **R5** | Trial scores drawn from one sweep of near-identical arms would set the multiple-testing bar 24× too low. Found by running the fix. | **high** | FIXED, GUARDED | PASS → FAIL |
| **R6** | `portfolio_sim._hold` called `resolve_exits(high=None)`. The 3R target could fire only on a **close** while the stop fired on the intraday **low**. The label passes `high`. | **high** | FIXED, GUARDED, PRICED | +0.43% / +0.10% |
| **R7** | Re-entry after an early exit was free. Cohorts fully close before the next opens and 84% close early, yet only names absent from the previous book paid a round trip. | **high** | FIXED, GUARDED, PRICED | −0.07% / −0.06% |
| **R8** | `stage7_risk.exit_hierarchy` was read by the card and by nothing else. Turning the stop off in config would have changed no backtest, no label and no validation number. | **high** | FIXED, GUARDED | — |
| **R9** | The training panel is not the population the book can buy. 7.29 of 8 selected slots fill. The dossier lists this as fixed (F5) "in training and at the decision", but the training half lives inside `resolve_exits`, which `triple_barrier: false` routes around — on the very config the dossier is anchored to. | **high** | **FIXED, GUARDED, REFITTED** — second pass; also found the fix had never reached the LIVE refit (§F1) | −0.31% / −0.85% after refit |
| **R10** | `phase_summary` reported `max_drawdown` as the **mean across schedules** — an experience nobody had, and always shallower than the real one. The dossier lists this as fixed (C2). | **medium** | FIXED, GUARDED | −13.7% → **−19.1%** |
| **R11** | Cash drag arrives inside the number labelled "position sizing". The book runs 75% invested against a fully-invested benchmark and nothing said so. | **medium** | disclosed, GUARDED | see §D |
| **R12** | `validation.holdout.sacred: true` was read by no code. Eight commands each carried their own `--include-holdout` arithmetic. | **medium** | FIXED, GUARDED | — |
| **R13** | A name with no ADTV got the largest size the slot allows **and** the cheapest possible fill. Double optimism, concentrated in the thinnest names. | **high** | **FIXED, GUARDED** — second pass; four liquidity states, monotone execution model | +0.19% / +0.08% after refit |
| **R15** | A name selected but never filled was recorded as a held position, so when it finally filled it paid nothing. It had never been bought. Found by re-reading my own diff. | **low** | FIXED, GUARDED | inert on this sample |
| **R14** | `test_exit_agreement` compared the two exit paths correctly and its fixture could not produce the bar that distinguishes them, so it passed for months with R6 live. | **medium** | FIXED, GUARDED | — |

The second pass adds five, in the review's own classification:

| id | category | finding | status | consequence |
|---|---|---|---|---|
| **W2** | MODEL | Every traded coefficient is a survivor of a selection on its own t-statistic. | BUILT, REPORTED NOT TRADED | fails acceptance criterion 6, by design and in advance |
| **C3** | VALIDATION | The operating record pooled trades decided by two different engines. | FIXED, GUARDED | history moves; partitioned by epoch |
| **C4** | UI | The retired record was not surfaced anywhere a reader could see what it belonged to. | FIXED, GUARDED | labelled, not dropped |
| **D1** | REPRODUCIBILITY | `data/` is not in version control, so no result can name its own inputs. | FIXED, GUARDED | manifest `6b6737fc418864aa`; restart-blocking |
| **D2** | REPRODUCIBILITY | Nothing recorded which engine a result came from. | FIXED, GUARDED | epoch ledger; restart-blocking |

`prosignal research findings` prints this register from
`validation/findings.py`, where each entry carries its root cause, code
location, fix, regression test, before/after, and three consequence flags —
whether coefficients moved, whether published history moved, and whether the
forward test must restart. A finding claimed FIXED with no regression test
raises on import, and a test asserts every named test exists and collects. The
first draft named five that did not.

Three of these — R5, R14 and R15 — were found by testing my own fixes rather
than by reading the code, which is the only reason I trust the rest of the list.
R15 is inert on this data: a name refused at entry tends to stay refused or fall
out of the exit band before it could fill, so the case never arises here. It is
fixed and pinned by a constructed fixture, and reported as costing nothing
measurable rather than as a saving.

---

## C · The three that matter

### R1 · The forward test is void, and the engine says so

```
$ prosignal research forward
  The forward test is INVALID: the pre-registration predates the
  benchmark-relative hypothesis ...; the registration carries no
  benchmark-relative hypothesis ...; the engine is running
  baseline-v1@127d8a314ec49aa2 but the test was registered against
  baseline-v1@a30a8d4847080ddc -- the configuration changed after
  registration, so this window is not one experiment.
```

Three registrations exist and no two agree:

| where | config | started |
|---|---|---|
| dossier cover | `baseline-v1@127d8a314ec49aa2` | not started |
| `research/FORWARD_TEST_REGISTRATION.json` | `baseline-v1@3e463f324bfa1d67` | 2026-08-26 |
| `data/ledger/forward_test.json` — **what the engine reads** | `baseline-v1@a30a8d4847080ddc` | 2026-08-27 |

`research/REPAIR_LOG.md` predicted this exactly — "`config_version` moved
`a30a8d4847080ddc → 1b2f891704ae3bb6`, which by the forward test's own
invalidation rules voids the registration opened 2026-08-27" — and listed
re-registration under **Not yet done**. It was never done.

**The eighteen-month clock the dossier says is running has not started.** I have
not started it either: opening a window is a decision about when your one clean
test begins, and it belongs to whoever holds the capital. When you are ready:

```
prosignal research forward --restart
```

The detection code was already correct. Nothing was wrong with it; the state was
wrong, and nothing forced anyone to look.

### R2 · The hypothesis that asks the only question that matters

`primary` regresses the book on six factors. `secondary` scores the ranking's
IC. Neither asks whether running the engine beats holding the universe it
selects from — and on the selection period it does not, by 3.9 points.

`Registration.tertiary` now exists, carries that hypothesis, sits inside the
fingerprint so it cannot be added or softened after a result lands, and a window
carrying no benchmark-relative hypothesis is marked broken rather than graded.
The registration text states in advance that **the engine is expected to fail
it**, because a forward test whose outcome is not in doubt is not a test.

### R3 · The multiple-testing defence could not fail

Two errors compounding, exactly as the dossier describes — and still present.

`CpcvResult.deflated()` handed the DSR its pooled `excess` vector: with N=10,
k=2 every panel date is tested in nine splits, so 68 dates became 612
observations. The fallback variance was `1/(n−1)`, which at n=612 is 0.0016 —
documented in the same function as "a conservative unit variance". Together they
produce an expected-maximum bar of 0.099 against an observed Sharpe of 0.38.
Anything positive passes, at any trial count.

Reproduced by running all four constructions on one set of scores:

| construction | n | Sharpe | Var[SR] | E[max] | DSR | dossier |
|---|---|---|---|---|---|---|
| as shipped — pooled pairs, `1/(n−1)` | 612 | +0.371 | 0.0016 | 0.099 | **1.0000 PASS** | 1.0000 pass |
| distinct panel dates | 68 | +0.422 | 0.0149 | 0.300 | 0.8617 fail | 0.4649 fail |
| independent 63-session windows | 23 | +0.467 | 0.0455 | 0.524 | 0.3799 fail | 0.1477 fail |
| windows + Var[SR] from the woven paths | 23 | +0.467 | 0.0083 | 0.224 | 0.9060 fail | 0.3130 fail |

The shape matches the dossier's table and the independent-window count matches
exactly (23 = 23). The levels differ because this tree's edge is stronger
(pooled excess +1.42% against the dossier's +0.97%).

**One correction to the dossier's own arithmetic.** Its preferred construction —
row four, `Var[SR]` from the woven paths — uses the wrong population. The nine
paths are resamples of the *one selected* configuration, so their variance
measures sampling noise in this strategy, not dispersion across the alternatives
it was chosen from. It is the smaller number, so it *shrinks* the bar: 0.0083
against 0.0455, which moves the answer from 0.38 to 0.91. The dossier's most
permissive honest construction is more permissive than it should be. It still
fails, so the conclusion holds; the reasoning needed fixing.

`deflated()` now requires `horizon_sessions` and `step_sessions` as keyword
arguments — no defaults, because a default is how the old pooled answer survived
a call site that read `result.deflated(n_trials=trials)` and looked complete.

**The engine now prints:**

```
Deflated Sharpe 0.000 charging 117 trials, on 23 independent 63-session
windows -- FAIL
  Var[SR] 1.0000 (unit_undercovered_trials); expected max under the null
  +2.585 against an observed +0.382
```

`0.000` is a floor rather than a measurement, and the line says so: 48 of the
117 charged trials carry a score, which is 41% and below the 50% the DSR
requires before it will use their variance (R5). **The verdict does not depend
on that rule:**

| Var[SR] | E[max] under the null | observed SR | DSR | |
|---|---|---|---|---|
| 1.0000 — conservative, what the engine uses | +2.585 | +0.382 | 0.0000 | FAIL |
| 0.0902 — measured from the 48 scored trials | +0.777 | +0.382 | 0.0468 | FAIL |

It fails either way, and by a wide margin on both.

---

## D · Where the money goes

Section G regenerated on the repaired simulator, plus the two lines it did not
have. `vs bench` is against an equal-weight hold of the eligible universe on the
same windows.

| layer | weave | vs bench | Δ | forward | vs bench | Δ | invested |
|---|---|---|---|---|---|---|---|
| 0 equal-weight eligible universe | +5.41% | — | | +5.05% | — | | 100% |
| 1 top 8, equal weight, no exits | +7.41% | **+2.00%** | | +6.22% | **+1.16%** | | 100% |
| 1a + refuse the names it cannot open | +7.15% | +1.74% | −0.25% | +6.02% | +0.96% | −0.20% | 91% |
| 2 + risk-budget sizing | +5.65% | +0.24% | −1.50% | +5.21% | +0.16% | −0.81% | 75% |
| 3 + 2.5× ATR stop | +2.93% | −2.48% | **−2.72%** | +1.74% | −3.31% | **−3.47%** | 75% |
| 4 + 3R target | +2.17% | −3.24% | −0.75% | +1.34% | −3.71% | −0.40% | 75% |
| 5 + invalidation exit | +2.02% | −3.39% | −0.16% | +1.16% | −3.89% | −0.18% | 75% |
| 6 + costs — **shipped** | +1.53% | **−3.88%** | −0.49% | +0.71% | **−4.35%** | −0.46% | 75% |

**Layer 1a is new.** `tradeable_at_entry` is applied live by stage 3 and stage 6,
and in the label path inside `resolve_exits` — which `build_panel` reaches only
when `exit_rules` is not None, which under the shipped `triple_barrier: false`
it is not. So the model ranks a population the book refuses part of, and the
simulator finds out at fill time by leaving slots empty. 7.29 of 8 fill.

**The `invested` column is new, and it explains layer 2.** Position value is
`min(risk_budget/dist, slot, liquidity)`. With 1% risk over 8 slots the risk
term binds above an 8.0% stop distance, and the median stop distance is 10.0%,
so:

```
invested ≈ (risk_per_trade_pct/100) × max_open_positions / median stop distance
         = 0.01 × 8 / 0.10 = 0.80
```

Three parameters set independently decide the book's exposure and nothing said
so. **This is not a defect.** The cash buys a hard cap — `risk_per_trade_pct ×
max_open_positions` = 8% of capital if every position stops at once — and
removing it is leverage, not alpha: fully invested, Sharpe moves 0.54 → 0.55 on
weave and 0.24 → 0.17 on forward while the worst drawdown deepens from −19.1% to
−28.9%. Nothing is changed; `deployed_frac` is now reported so the number
arrives under its own name.

### The measurement corrections, priced against layer 6

| correction | weave | Δ | forward | Δ |
|---|---|---|---|---|
| A · target read on the HIGH (R6) | +1.96% | **+0.43%** | +0.81% | **+0.10%** |
| B · re-entry pays its round trip (R7) | +1.46% | −0.07% | +0.65% | −0.06% |
| D · unknown liquidity refused (R13) | +1.70% | +0.17% | +0.87% | +0.17% |
| A+B+D together | +2.03% | +0.50% | +0.85% | +0.14% |

**None of them changes the verdict.** All three together leave the book at
−3.38% and −4.21% against its benchmark. They are worth fixing because the
target layer's cost was overstated by roughly half and turnover was understated,
not because they rescue anything.

### And the number that does change things

| arm | weave | vs bench | Sharpe | forward | vs bench | Sharpe |
|---|---|---|---|---|---|---|
| shipped | +1.53% | −3.88% | 0.54 | +0.71% | −4.35% | 0.24 |
| **no stop, no target, no invalidation** | **+5.14%** | **−0.27%** | **1.11** | **+4.74%** | **−0.31%** | **1.07** |

Sizing, the rank band and full costs all still in place. Remove the three exits
and the book is level with the universe it selects from, at roughly twice the
Sharpe. **The engine's entire deficit is its exit layer**, and that now has a
number on it from two independent constructions.

`docs/CONFIG_V2.md` has the stop frontier, the flat-stop comparison and the
finding that a tighter stop makes the *equity* drawdown worse while improving
the worst single quarter — two different questions the dossier's section H
answers only one of.

---

## E · What I proposed, and it failed

I thought I could fix the label. I pre-registered the attempt in
`work/PREREGISTRATION.md` before running it, and it was rejected.

`REPAIR_LOG.md` Job 1 turned the engine-geometry label off because `|label| ∝
ATR/P` made it a volatility-predicting target. Dividing the same outcome by its
own stop distance removes that exactly — a winner is +3R and a loser −1R for
every name. Measured, the within-date correlation between `|label|` and ATR/price
goes from **+0.292** (rupee) to **−0.036** (R-multiple). The correction works.

The label does not. Fitted on each and scored by running the book:

| fitted on | weave vs bench | Sharpe | forward vs bench | Sharpe |
|---|---|---|---|---|
| horizon return (shipped) | **−3.87%** | **0.54** | **−4.34%** | **0.24** |
| engine geometry, rupee | −5.71% | −0.33 | −5.57% | −0.13 |
| engine geometry, R-multiple | −4.48% | 0.19 | −5.10% | 0.13 |

The shipped label wins on both constructions. The reason is identifiable: there
are two channels and the R-multiple closes only one. The second is the
invalidation exit, a trend filter living inside the target — under the
R-multiple label `mom_f` falls from t +3.61 to −0.89, exactly as Job 1
described. **`triple_barrier: false` survives an adversarial challenge and
should stay.**

---

## F · Second pass — the remaining findings, and what the refit cost

The five items section F left open are closed. They were taken in the
dependency order the review set: **data provenance → point-in-time universe →
fitted coefficients → execution assumptions → forward test**, because each one
invalidates the measurement of the next.

### F1 · R9 was fixed in the research panel and not in the engine

The correction reached `validation/research_panel.py`, which is what this audit
measures with. It never reached `features/crossmodel.fit_predict`, which is
what the engine refits with. So the audited numbers moved and the shipped
coefficients did not — a worse state than either, because the measurement and
the model disagreed and both looked correct.

`admission_rules` is now threaded through the live refit from
`universe.train_on_admissible_only`, and — this is the part that would have
bitten silently — the population is inside the **cache fingerprint**. Without
that, `load_cached` would have gone on serving wide-population coefficients for
up to `refit_every × 2` = 42 sessions after the switch flipped, with every run
looking normal. That is the same trap the label repair walked into, one field
along.

### F2 · The refit, and it costs money

Enabled, then refitted. The training panel loses 19.3% of its rows.

| | v1 wide panel | v2 admissible panel |
|---|---|---|
| training rows | 33,064 | 26,681 |
| slots filled per rebalance | 7.29 of 8 | **8.00 of 8** |
| shipped book vs benchmark, weave | −3.88% | **−4.19%** |
| shipped book vs benchmark, forward | −4.35% | **−5.20%** |
| ranking alone vs benchmark, forward | +1.16% | **+0.10%** |

Two things to read here. The first is that the fill gap is gone: layer 1a, the
line that used to cost 0.26 points by refusing names the ranking had chosen, is
now **+0.00%**, because there is nothing left to refuse. The correction is
applied at the source instead of discovered at fill time.

The second is the last row. On the purged walk-forward construction the
**ranking's edge over equal-weight falls from +1.16% to +0.10%** once the model
is fitted on the population it can actually trade. A meaningful part of the
edge this engine was credited with was earned on names the book cannot buy.

### F3 · The coefficients moved, and one of them changed the answer

All seven. The two the engine traded survive; a third crosses the floor.

| theme | v1 λ | v2 λ | v1 t | v2 t | |
|---|---|---|---|---|---|
| mom_f | +0.0748 | +0.0727 | +3.61 | +3.69 | traded in both |
| delivery_f | +0.0454 | +0.0445 | +4.12 | +3.56 | traded in both |
| **reversal_f** | +0.0181 | +0.0267 | +1.59 | **+2.71** | **enters** |
| lottery_f | −0.0309 | −0.0253 | −1.34 | −0.99 | zeroed in both |
| drawdown_f | +0.0174 | +0.0221 | +0.86 | +1.07 | zeroed in both |
| skew_f | −0.0122 | −0.0136 | −1.10 | −1.19 | zeroed in both |
| beta_f | +0.0152 | +0.0134 | +0.82 | +0.75 | zeroed in both |

The engine now trades three themes where it traded two. And W2 has something to
say about exactly the new one: corrected for the gate it passed, `reversal_f`
comes back at an implied true **t +1.74**, which does not clear the floor of
2.0 it was selected by. `mom_f` (+3.56) and `delivery_f` (+3.39) do.

That is a reported number and not a traded one, so it changes nothing
automatically. It is the single most useful thing on this page for deciding
whether to keep the third theme.

### F4 · The population audit, as a series rather than an average

"7.29 of 8 slots fill" is one number over nine years, and the average was
hiding the finding.

| | admissible fraction of the eligible panel |
|---|---|
| mean | 80.8% |
| range | 38.1% – 100% |
| standard deviation | 16.1 points |
| tightest dates | 2022-06-15, 2019-07-24, **2020-03-02**, 2019-02-18 |

It is not a level shift, which would bias a fit predictably. It is a drift, and
its worst dates are market stress — which is precisely when the difference
between "the screen would list it" and "the book could buy it" decides whether
a stop is a stop.

### F5 · R13, W2, C3/C4 in one line each

**R13** ships on. Four states — `KNOWN_VALID`, `KNOWN_STALE`, `MISSING`,
`INVALID` — and unknown means no position, no optimistic fill and no
imputation. `adtv_inr` is `None` whenever untradable, so a caller that ignores
the gate raises a `TypeError` rather than sizing against a plausible-looking
float. Unknown-liquidity impact 5.0 → **105.0 bps**. Even after R9 closes the
population gap, refusing unknown liquidity still costs 0.5 slots per rebalance,
so it is a live constraint and not a formality.

**W2** ships as a report and is unreachable from the scoring path by test. Its
sixth acceptance criterion was fixed in advance and is not met, and mutation
testing then found the guard on its one-sided conditioning **could not fail** —
the two tails cancel in a signed mean. Measured on the positive tail, where a
long-only engine actually reads its coefficients, a true effect of **zero**
comes back at **+0.59** under the correct estimator and **+1.00** under the
two-sided one. The +0.59 is not a bug: a clamp cannot subtract more than all of
an effect, and the survivors that clamp are the ones with the least evidence.
It is the sharpest single reason this number is not traded.

**C3/C4** — outcomes carry `epoch_id`, statistics partition on it as they
already did on `exit_model`, and the retired record is served **labelled**
beside the current epoch rather than dropped. `/outcomes` prints what pooling
would have claimed and by how much it would be wrong, because a partition whose
cost is invisible gets re-collapsed by the next person who finds the per-epoch
samples too small to be interesting.

### F6 · `data/` — a manifest, not the data

37 files, 0.25 GB, 9,270,123 rows, digest **`6b6737fc418864aa`**. Per file:
sha256, size, row count, date span, schema hash. The digest is over content and
never over mtime, so re-downloading an identical file does not invent a new
dataset; it is recomputed on load and never trusted from the file it was read
out of.

Two exclusions are judgement calls and are stated rather than buried:
`trial_registry.jsonl` and `crosssec_model.json` are a research ledger and a
model output, both of which change without the market data changing. Leaving
them in meant the readiness gate reported the data as drifted every time a
hypothesis was recorded, and a manifest that cries wolf is a manifest nobody
reads.

**The `.gitignore` would have swallowed the manifest.** `data/curated/` and
`data/ledger/` were excluded as directories, and git cannot re-include a file
whose parent directory is excluded — so `MANIFEST.json`, `epochs.jsonl` and
`trial_registry.jsonl` were all uncommittable, which defeats the entire point
of the exercise. They are now excluded by contents, with those three negated,
and `work/cache/` (134 MB of derived pickles, previously tracked) is excluded
instead.

### F7 · Research epochs, and READY as a gate

An append-only ledger binding **code sha, model-source sha, config version,
data manifest digest, feature schema hash, universe policy and execution
model** into one identity. v1 is archived `VOID` with a reason naming
R1/R2/R3/W2/R9/R13. v2 is open as `2026-08-29-6451d9181041cdb4`. Drift is
reported and never acted on — whether a change is material enough to open a new
epoch is a judgement, and one made in the open beats one made by a rule that
guessed.

READY now means eight gates, not a green suite. A green suite said so while the
model was fitted on a population the book cannot buy (R9), while an unmeasured
name got the cheapest fill in the model (R13), while no result could name its
data (D1), and while the window measuring all of it was void (R1).

```
DATA             PASS   37 files, 9,270,123 rows, digest 6b6737fc418864aa
UNIVERSE         PASS   training panel restricted to the admissible population
FEATURE          PASS   schema d71b38516bb244f1, matching the open epoch
MODEL            PASS   fit attributable to bbc806242212 under baseline-v2
EXECUTION        PASS   monotone; unknown liquidity 105.0 bps vs 12.7 deep
VALIDATION       PASS   20 findings registered, none open but R1
REPRODUCIBILITY  PASS   epoch 2026-08-29-6451d9181041cdb4
FORWARD          FAIL   registered against baseline-v1; the window is void
```

### F8 · R1 — the gate is open and the restart has not been taken

`research forward --start/--restart` now refuses while a restart-blocking
finding is open, the manifest is unverified, or no epoch describes the engine,
and it names **every** reason at once rather than the first. The refusal runs
before the overwrite check, so "the engine is not ready" cannot present itself
as "a test is already registered".

As of this commit every precondition holds and the gate permits a restart. **It
has not been taken.** Starting the window discards the observations collected
so far and starts an eighteen-month clock, and that is an operator's decision —
which is exactly why the gate reports and a person acts. R1 stays `OPEN` in the
register, and a test asserts both halves of that: that nothing refuses, and
that R1 has not been quietly marked resolved.

### F9 · What the statistics say after the refit

Unchanged in direction and stronger in degree. The Deflated Sharpe now fails on
**all four** constructions rather than three:

| construction | n | SR | Var[SR] | DSR | |
|---|---|---|---|---|---|
| as shipped — pooled pairs | 603 | +0.3384 | 0.00166 | 0.0000 | FAIL |
| distinct panel dates | 67 | +0.3740 | 0.01515 | 0.0000 | FAIL |
| independent 63-session windows | 23 | +0.1929 | 0.04545 | 0.0000 | FAIL |
| windows, Var[SR] from woven paths | 23 | +0.1929 | 0.00624 | 0.0000 | FAIL |

Every pre-registered hypothesis from the first pass survives the refit. H1 —
the stop frontier — still gives Spearman **+1.000** with the OFF arm best, on
both constructions. H2 still attributes 99% of the sizing layer to cash rather
than weighting. H3 still finds the label 2.0–2.4× longer than the median trade.
H5's cost is now **+0.00%**, because R9 closed it at the source.

---

## G · What is still not settled

- **R1.** The window may be restarted and has not been. That is a decision, not
  a finding.
- **The third theme.** `reversal_f` enters on the corrected panel at t +2.71 and
  corrects to +1.74 against a floor of 2.0. It ships because the fit says so and
  the correction is reported rather than traded; it is the first thing to look
  at if the forward test disappoints.
- **The exit layer.** Unchanged by any of this. `no exits at all` still lands
  within half a point of the benchmark while the shipped book loses four. The
  stop is where the money goes and no finding in this pass touched it.
- **The forward test is the only remaining source of independent evidence.**
  Nothing in these two passes changes that, and nothing in them can.

---

## H · Reproducing this

Everything is in `work/`, and the slow parts cache. `work/cache/` is
deliberately **not** in version control — `build_panel.py` rebuilds it from the
store in about seven minutes, and `panels.pkl` alone is 134 MB.

```
python work/build_panel.py        # panel + price panels -> work/cache  (~7 min)
python work/test_parity.py        # work/book.py == portfolio_sim, exactly
python work/decompose.py          # sections D and F2
python work/frontier.py           # the stop frontier, H1/H2/H3/H5
python work/refit_compare.py      # section F3 -- the coefficients, both panels
python work/dsr_reconstruct.py    # section F9 -- the four DSR constructions
python work/h4_label.py           # the R-multiple experiment (rejected)
python work/v2_candidates.py      # every v2 arm, return AND risk
python work/mutation_probe.py     # 43 reversions, each must go red
```

And from the engine itself, which now answers these questions directly:

```
prosignal data manifest --verify   # the store is the one the figures used
prosignal research epoch status    # which engine produced them
prosignal research findings        # the register, classified
prosignal research readiness       # the eight gates
prosignal research record          # the operating record, by epoch
```

`work/PREREGISTRATION.md` was written before any arm was run and is unedited.
This document departs from it in two places — the trial count is higher than the
16 estimated, and H4 was rejected — and both are stated rather than reconciled.

**Trials.** Every arm was looked at and could have been chosen. The registry
now records **99** and carries 20 from earlier campaigns, so the DSR charges
**119**. 50 carry a score — **42% coverage against the charged count**, below
the 50% the DSR requires before it will use their variance, so it still reports
the conservative unit bar and names it (`unit_undercovered_trials`) rather than
substituting one silently. The verdict does not depend on the choice: measured
variance and conservative variance both give **0.000, FAIL**.

**Mutation testing is the part worth repeating.** Of the 43 reversions, four
were caught only after the test that was supposed to catch them was found
incapable of it — a W2 guard whose statistic cancelled the defect it was
watching for, an R9 guard that read source text a mutation left intact, a
restart-gate guard that never had two kinds of blocker at once, and a
`test_pit_universe.py` name collision that silently replaced an existing file.
Every one was a weak test rather than a bad fix, and none would have been found
by reading.
