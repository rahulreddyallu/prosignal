# config v2 — the candidate, its evidence, and what it does not fix

`config/parameters-v2.yaml`, `baseline-v2@4e89512b6c343e05`.

One traded value moves: `stage7_risk.stop_loss.atr_multiple`, **2.5 → 3.5**.
Everything else is identical. `baseline-v1@127d8a314ec49aa2` is untouched, still
loads, still reproduces coefficient for coefficient, and keeps whatever
registration it is given.

**v2 does not make this engine work.** It still returns 3.6–3.9 points per
63-session period less than an equal-weight hold of the universe it selects
from. It is a smaller loss, arrived at without changing anything else, and it is
offered as a registered candidate rather than as a fix.

---

## How everything below was measured

Selection period only, ending at the config's own holdout boundary
(`sessions[-378]` = 2025-02-11). No command was given `--include-holdout`, and
that flag is now refused outright while `validation.holdout.sacred` is true.

Every book figure comes from `work/book.py`, which `work/test_parity.py` proves
equal to `validation.portfolio_sim.simulate` to floating point at shipped
settings. Each arm is therefore **one named departure from the shipped
simulator**, not a second implementation that happens to disagree.

Two ranking constructions, always reported together:

| | dates | how |
|---|---|---|
| **weave** | 68 | CPCV path 0 — each panel date scored once, by the first split that held it out. The repo's own construction. Training blocks may sit after the test block. |
| **forward** | 35 | purged expanding walk-forward — fit on everything up to `i − purge`, rank date `i`. Fewer dates, but no fit ever sees a later session. |

A result that holds on one and not the other is reported as that.

---

## Why the stop, and why 3.5

### The exit layer is where the money goes

Section G of the September dossier, regenerated on the repaired simulator. Each
line adds one layer to a book that starts as the ranking alone. `vs bench` is
against an equal-weight hold of the eligible universe on the same windows.

| layer | weave | vs bench | forward | vs bench |
|---|---|---|---|---|
| 0 equal-weight eligible universe | +5.41% | — | +5.05% | — |
| 1 top 8, equal weight, no exits | +7.41% | **+2.00%** | +6.22% | **+1.16%** |
| 1a + refuse the names the book cannot open | +7.15% | +1.74% | +6.02% | +0.96% |
| 2 + risk-budget sizing | +5.65% | +0.24% | +5.21% | +0.16% |
| 3 + 2.5× ATR stop | +2.93% | −2.48% | +1.74% | −3.31% |
| 4 + 3R target | +2.17% | −3.24% | +1.34% | −3.71% |
| 5 + invalidation exit | +2.02% | −3.39% | +1.16% | −3.89% |
| 6 + costs — **shipped book** | +1.53% | **−3.88%** | +0.71% | **−4.35%** |

The ranking earns +2.0% and +1.2%. Everything after it is subtractive, and the
single largest item is the stop at −2.7% and −3.5%.

### The whole exit layer, priced against switching it off

| arm | weave | vs bench | Sharpe | forward | vs bench | Sharpe |
|---|---|---|---|---|---|---|
| shipped | +1.53% | −3.88% | 0.54 | +0.71% | −4.35% | 0.24 |
| no target | +2.26% | −3.15% | 0.65 | +1.09% | −3.96% | 0.33 |
| no invalidation | +1.68% | −3.73% | 0.56 | +0.88% | −4.17% | 0.30 |
| no stop | +2.66% | −2.76% | 0.87 | +1.69% | −3.37% | 0.48 |
| **no exits at all** | **+5.14%** | **−0.27%** | **1.11** | **+4.74%** | **−0.31%** | **1.07** |

With sizing, the rank band and full costs all still in place, removing the three
exits brings the book level with its own benchmark and roughly doubles its
Sharpe. **That is the finding.** The stop multiple is a corner of it.

### The stop frontier

Seven arms, all carrying costs, target and invalidation on.

| stop | weave net | worst cohort | worst schedule dd | Sharpe | forward net | worst cohort | worst dd | Sharpe |
|---|---|---|---|---|---|---|---|---|
| 1.5× | −0.42% | −8.2% | −24.8% | −0.19 | −1.35% | −8.4% | −19.3% | −0.60 |
| 2.0× | +0.45% | −8.5% | −23.1% | 0.17 | −0.63% | −8.2% | −19.8% | −0.25 |
| **2.5× (v1)** | **+1.53%** | **−8.2%** | **−19.1%** | **0.54** | **+0.71%** | **−8.0%** | **−13.0%** | **0.24** |
| 3.0× | +1.45% | −8.3% | −17.2% | 0.53 | +0.88% | −8.0% | −12.5% | 0.32 |
| **3.5× (v2)** | **+1.77%** | **−8.6%** | **−16.5%** | **0.67** | **+1.10%** | **−6.4%** | **−11.8%** | **0.39** |
| 5.0× | +1.73% | −6.7% | −15.8% | 0.73 | +1.14% | −6.6% | −7.4% | 0.45 |
| none | +2.66% | −10.8% | −16.7% | 0.87 | +1.69% | −10.9% | −14.2% | 0.48 |

Spearman correlation between the multiple and return: **+0.93** (weave),
**+1.00** (forward), with the OFF arm best on both. **3.5 is not a peak found by
searching — it is a step along a monotone relationship, stopped where the
config's own `search_range: [1.5, 3.5]` ends rather than where the improvement
does.**

### Two things the dossier's section H could not show

**A tighter stop makes the equity drawdown WORSE, not better.** Section H
reported the worst single *cohort*, on which the stop does help. On the worst
peak-to-trough of a *schedule* the ordering reverses: 1.5× gives −24.8% against
no stop's −16.7%. A stop truncates the right tail of every quarter and converts
noise into realised losses that then compound across quarters. The worst quarter
improves; the sequence of quarters gets worse. Both columns are now reported;
`phase_summary` gained `worst_schedule_drawdown` because it previously published
only the MEAN across schedules, which is an experience nobody had.

**The win at 5× is about the stop being flat, not about it being wide.** At 5×
ATR, **78%** of names sit on the `max_stop_distance_pct: 15.0` ceiling, so that
arm is a 15% flat stop for four names in five. Tested directly:

| arm | weave net | Sharpe | worst cohort | worst dd | forward net | Sharpe |
|---|---|---|---|---|---|---|
| 2.5× ATR (v1, vol-scaled, median 10.0%) | +1.53% | 0.54 | −8.2% | −19.1% | +0.71% | 0.24 |
| flat 10% stop | +1.09% | 0.42 | −8.4% | −19.7% | +0.82% | 0.27 |
| flat 12% stop | +1.58% | 0.61 | −8.4% | −17.6% | +1.19% | 0.41 |
| flat 15% stop | +1.75% | 0.77 | −6.7% | −15.2% | +1.15% | 0.48 |
| flat 20% stop | +1.66% | 0.96 | −4.6% | −7.6% | +1.06% | 0.58 |

A flat stop matches or beats the volatility-scaled one at every comparable
width. **The ATR scaling of the stop is not earning its keep**, and the arms
that do best are the ones where it barely operates. Part of the improvement at
20% is de-levering — position value is `risk_budget / stop_distance`, so a 20%
stop takes a 40%-of-slot position — but Sharpe is invariant to that, and Sharpe
still nearly doubles.

Changing `min_stop_distance_pct` / `max_stop_distance_pct` to make the stop flat
would leave both outside their declared search ranges. **v2 does not do it.**
It is recorded here as the more interesting question than the multiple.

---

## What was rejected, and why

Every arm below was looked at and could have been chosen. All of them are
charged to the Deflated Sharpe.

### The R-multiple label — REJECTED, and it was my idea

`research/REPAIR_LOG.md` Job 1 turned the engine-geometry label off for a
correct reason: a winner realises `+3 × (2.5·ATR/P)` and a loser
`−1 × (2.5·ATR/P)`, so `|label| ∝ ATR/P` and any volatility-correlated feature
predicts it by construction. Dividing the same outcome by its own stop distance
removes that proportionality exactly — a winner is +3R and a loser −1R for every
name — while keeping the label equal to the trade the engine takes.

Measured, mean within-date correlation between `|label|` and ATR/price:

| label | corr | |
|---|---|---|
| horizon return (shipped) | **+0.210** | the artefact is reduced, not removed |
| engine geometry, rupee return | **+0.292** | Job 1's finding, confirmed |
| engine geometry, R-multiple | **−0.036** | gone |

So the correction works. The label does not:

| fitted on | weave net | vs bench | Sharpe | forward net | vs bench | Sharpe |
|---|---|---|---|---|---|---|
| horizon return (shipped) | +1.53% | −3.87% | 0.54 | +0.71% | −4.34% | 0.24 |
| engine geometry, rupee | −0.73% | −5.71% | −0.33 | −0.20% | −5.57% | −0.13 |
| engine geometry, R-multiple | +0.51% | −4.48% | 0.19 | +0.27% | −5.10% | 0.13 |

**The shipped label wins on both constructions.** The reason is now identified:
there are two channels, and the R-multiple closes only one. The second is the
invalidation exit, which is a trend filter living inside the target — a name up
30% over the quarter that dipped through MA50 − 1.5×ATR in week three is booked
as a loss. Under the R-multiple label `mom_f` goes from t +3.61 to −0.89:
momentum stops pricing, exactly as Job 1 described. **`triple_barrier: false`
survives an adversarial challenge and should stay.**

### Full investment — REJECTED as an improvement, KEPT as a disclosure

The book runs about **75%** invested. That is not an accident and not a defect:

```
invested ≈ (risk_per_trade_pct / 100) × max_open_positions / median stop distance
         = 0.01 × 8 / 0.10 = 0.80
```

Three parameters set independently decide the book's exposure, and nothing in
the engine said so. Holding the same relative weights but scaling to full
investment:

| | weave | Sharpe | worst dd | forward | Sharpe | worst dd |
|---|---|---|---|---|---|---|
| shipped, 75% invested | +1.53% | 0.54 | −19.1% | +0.71% | 0.24 | −13.0% |
| same weights, 100% invested | +2.18% | 0.55 | −28.9% | +0.67% | 0.17 | −16.4% |

Return rises on one construction and falls on the other; **Sharpe does not move
and the drawdown deepens by ten points.** This is leverage, not alpha. The cash
is bought, and what it buys is a hard cap: `risk_per_trade_pct ×
max_open_positions` = 8% of capital if every position stops at once.

Nothing is changed. `deployed_frac` is now a first-class metric on
`phase_summary`, so the cash position is visible instead of arriving inside the
number labelled "position sizing".

### Training on the population the book can buy — DEFERRED to a later v2

`exits.tradeable_at_entry` is applied live by stage 3 and stage 6, and in the
label path inside `resolve_exits` — which `build_panel` reaches only when
`exit_rules` is not None, which under `triple_barrier: false` it is not. So the
model is fitted and ranked on a population about a fifth larger than the book
can open, and the simulator discovers the difference at fill time: **7.29 of 8
selected slots fill.**

The capability now exists (`build_panel(admission_rules=...)`,
`research_panel.ADMIT_ONLY_TRADEABLE`) and is guarded by a test. It ships **off**
because turning it on changes the population the model is fitted on and
therefore the traded coefficients, and because it belongs with a refit rather
than with a stop multiple. Cost of leaving it off: **−0.20 to −0.25% per
period**, both constructions.

It is not a config parameter because `config_hash` digests the schema's
effective values, so adding one — even left at its old default — would move
`baseline-v1@127d8a314ec49aa2`, and a correctness fix that renames the baseline
it is measured against is not one.

---

## The pre-committed forward hypotheses for v2

To be written into `data/ledger/forward_test.json` at registration, hashed with
it, and not editable afterwards. **v2's clock has not started.** The v1
registration is separate and is currently INVALID for three independent reasons
(see `docs/REAUDIT.md`, R1).

1. **PRIMARY** — factor attribution alpha, as v1.
2. **SECONDARY** — pooled rank IC of the shortlist, as v1.
3. **TERTIARY, and the only one that asks whether running the engine beats not
   running it.** Mean excess return of the paper book over an equal-weight hold
   of the eligible universe, same holding windows, positive at overlap-corrected
   t ≥ 2.0 over 18 months. **v2 is expected to fail this**, on the evidence
   above: it loses to its universe by 3.6–3.9 points per period on the selection
   period. It is registered for that reason.

`Registration.tertiary` now exists, is inside the fingerprint, and a window
carrying no benchmark-relative hypothesis is refused rather than graded.

---

## Trial accounting

Every arm in this document was looked at and could have been chosen.

| group | arms |
|---|---|
| stop frontier | 7 |
| flat-stop widths | 5 |
| exit components | 4 |
| sizing / exposure | 3 |
| label variants | 3 |
| admission population | 1 |
| combined v2 candidates | 6 |
| **added by this work** | **29** |

The registry stood at 67 recorded plus 20 carried. Any figure produced from this
work must be deflated at **116**, not 87. Recorded in advance in
`work/PREREGISTRATION.md`, which was committed before the arms were run and
which this document departs from in two places — the trial count is 29 rather
than the 16 estimated, and H4 was rejected. Both are stated rather than
reconciled.
