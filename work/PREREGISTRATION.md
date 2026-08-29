# Pre-registration — execution-layer and label experiments

Written **before** any of the arms below were run. Committed first so that no
result can be reinterpreted afterwards as the one that was expected.

- Base commit for this work: the tree as staged, config `baseline-v1@127d8a314ec49aa2`.
- Data: selection period only. `end` is the config's own holdout boundary
  (`sessions[-378]` = 2025-02-11). No command in this work is given
  `--include-holdout`, and the container it runs in was deliberately loaded
  without any post-2025 price file until the store had to match session count.
- Baseline: `work/book.py` at shipped settings, proven equal to
  `validation.portfolio_sim.simulate` to floating point by `work/test_parity.py`.
  Every arm below is one named departure from that baseline.
- Two ranking constructions are reported for every arm: CPCV weave path 0
  (68 dates) and purged expanding walk-forward (35 dates). **A result that does
  not hold on both is reported as not holding on both.**

---

## What is being asked

The dossier's section G finds that the ranking earns about +1% per period
against the equal-weight eligible universe and the machinery below it gives back
roughly 4.9. Its section M calls this "the most tractable finding in the dossier
and the one nothing in the repository was previously able to see."

The question is therefore **not** "can the stop be tuned to a better number".
It is: **is the exit machinery earning the return it costs, and is the model
being asked to predict the thing the machinery does?**

## H1 — The exit frontier is monotone and has no interior optimum

**Hypothesis.** Across stop multiples {none, 1.5, 2.0, 2.5, 3.0, 3.5, 5.0} at a
fixed 3R target, mean return per period is monotonically increasing in the stop
multiple over the range tested, and no interior setting beats switching the stop
off.

**Why it is registered as a hypothesis and not a search.** If it is TRUE, no
stop multiple can be selected on return grounds and the choice is a pure
risk-tolerance decision belonging to the capital owner — which is what section H
already argued and what this re-tests on an independent construction. If it is
FALSE — if some interior multiple wins — that is a local optimum found by
looking at the data, it is a trial, and it must be treated as one rather than
adopted.

**Pass condition.** Spearman rank correlation between stop multiple and mean
return `>= +0.9` on BOTH constructions, with the "none" arm ranked highest.

**Decision rule, fixed now.** Whatever the outcome, **no stop parameter in
`config/parameters.yaml` is changed by this work.** The frontier is reported as
a priced menu — return given up, worst-schedule drawdown bought — and the
selection is Rahul's. Adopting the best-returning arm would be exactly the
search this exercise forbids.

## H2 — Most of what "risk-budget sizing" costs is cash, not weighting

**Hypothesis.** Position value is `min(risk_budget/dist, slot, liquidity)`.
At 1% risk over 8 slots the risk term binds above an 8.0% stop distance, so the
book holds cash while the benchmark it is scored against is fully invested.

Holding the same relative weights but rescaling the book to full investment
recovers **at least half** of the sizing layer's measured cost.

**Pass condition.** `full_investment` recovers >= 50% of the layer-2 delta on
BOTH constructions.

**Consequence if true.** Section G's layer 2 is not one number. It is a
weighting decision and an exposure decision added together and labelled as the
first. They should be reported separately, whatever their sizes.

## H3 — The label does not describe the trade the book takes

**Hypothesis.** The shipped config sets `labels.triple_barrier: false`, so the
model is fitted on the plain 63-session forward return. The book exits on a
2.5x ATR stop, a 3R target and an invalidation level. If the median realised
hold is materially shorter than 63 sessions, the model is being asked to rank
names by an outcome the book does not experience.

**Pass condition.** Median realised hold `<= 40` sessions and the share of
positions reaching the 63-session timeout `<= 25%`, on both constructions.

## H4 — The R-multiple label removes the artefact that forced H3

**Background.** `research/REPAIR_LOG.md` Job 1 turned the barrier label off for
a correct reason: under the engine's geometry a winner realises
`+3 x (2.5·ATR/P)` and a loser `−1 x (2.5·ATR/P)`, so `|label| ∝ ATR/P` and any
volatility-correlated feature predicts it by construction. That diagnosis is
right and is not in question.

The fix chosen — revert to the horizon return — removes the artefact by
discarding the engine's geometry, which is what H3 measures the cost of.

**Hypothesis.** Dividing the engine-geometry outcome by its own stop distance —
the **R-multiple**, `realised_return / stop_fraction` — removes the
proportionality exactly (a winner is +3R and a loser −1R for every name,
whatever its volatility) while keeping the label equal to the trade the engine
takes.

Under an R-multiple label:

- **H4a.** The within-date correlation between `|label|` and ATR/P falls below
  `0.10` in absolute value, against the barrier label's, which is what Job 1
  identified.
- **H4b.** `lottery_f` and `reversal_f` — the two themes Job 1 showed were
  pricing the artefact — do not return to significance. Specifically neither
  reaches `|t| >= 2.0` with the sign it carried under the barrier label.
- **H4c.** Out-of-sample top-decile excess return over the equal-weight
  eligible universe is positive.

**Decision rule, fixed now.** H4 is a **research finding, not a config change.**
Even if all three pass, `labels.triple_barrier` is not flipped in
`config/parameters.yaml` by this work. It is written up as a candidate v2 with
its own hash and its own pre-committed forward hypotheses, so that the v1
registration and its clock are untouched.

## H5 — Ranking a population the book cannot buy costs measurable return

**Hypothesis.** `exits.tradeable_at_entry` is applied by stage 3, by stage 6 and
inside `resolve_exits` — but `build_panel` reaches `resolve_exits` only when
`exit_rules` is not None, which under `triple_barrier: false` it is not. So the
training panel and every research figure are computed on a population the book
refuses part of.

**Pass condition.** The book fills materially fewer than 8 of its 8 selected
slots, and refusing the unbuyable names at selection time (rather than
discovering them at fill time) is worth a non-zero return delta on both
constructions.

---

## Trial accounting

Every arm below is a configuration comparison and is charged to the Deflated
Sharpe as such. Counted in advance:

| group | arms |
|---|---|
| H1 stop frontier | 7 stop multiples x 2 constructions, counted as **7** |
| H1 exit-component grid (target on/off x invalidation on/off) | **4** |
| H2 full-investment | **1** |
| H3 | 0 — a measurement of the shipped book, no alternative fitted |
| H4 label | **3** (R-multiple, barrier, horizon — refitted and scored) |
| H5 | **1** |
| **total added** | **16** |

The registry currently charges 81. Any figure produced by this work must be
deflated at **97**, not 81. That number is recorded here before the arms are
run so it cannot be trimmed afterwards.

## What would make me wrong

- If the frontier has an interior optimum, H1 is false and the section H
  argument is weaker than the dossier states.
- If `full_investment` recovers less than half of layer 2, H2 is false and the
  sizing rule is genuinely destroying edge rather than merely holding cash.
- If the R-multiple label reproduces Job 1's artefact — H4a fails — then the
  proportionality is not the whole story and the horizon return is the better
  of the two available labels after all.
