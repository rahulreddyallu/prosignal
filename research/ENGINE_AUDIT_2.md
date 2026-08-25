# ProSignal — second engine audit

The first audit asked whether the ranking carried information. It does, and
it answered that with leakage control, CPCV, PBO, DSR, an overlap
correction, and a factor attribution. Fourteen of the demands in this
directive were already met by that work and are not repeated here.

This audit asks a different question: **does the thing on the screen
correspond to the thing that was validated?**

It does not. That is the headline, and it is measured, not argued.

Assembled 2026-08-25 against `baseline-v1@39e9687c57bd12bd`.

---

## Finding 1 — the shortlist is not the portfolio

**Severity: highest. This is an architecture defect, not a tuning issue.**

Measured over 80 signal days, taking one run per day:

| | |
|---|---|
| Mean top-5 turnover per session | **89.0%** |
| Median | **100%** |
| Sessions where the list is entirely replaced | **84.8%** |
| Median sessions a name survives in the top 5 | **1** |
| Longest survival observed | 4 sessions |

A name enters the shortlist and is gone the next session. The card labels it
**Buy**, quotes a target 13% away and a hold of ~14 sessions — for a position
the list will not contain tomorrow.

**Why this is not simply "high turnover".** Stage 6 admits at rank ≤ 8 and
*holds while rank ≤ 16*. The strategy that was validated is therefore
patient: it opens a position and keeps it through rank drift. The shortlist
is a daily top-5 snapshot with no hysteresis applied to it at all. **They are
two different products, and the screen shows the one that was never
validated.**

If the displayed list were traded literally, at the engine's own 88 bps
round trip and 89% daily replacement of five slots, cost drag is on the
order of **~196% of capital per year**. That is not a strategy; it is a
description of why the display and the engine must be reconciled.

**What to do.** Apply the Stage 6 band to the presented slate: a name shown
today stays shown while it remains inside rank 16, and only leaves when the
engine would actually close it. The information already exists — `model_rank`
is on every card — and `presentation/selection.py` simply does not use it
that way.

---

## Finding 2 — the score does not order outcomes

**Severity: high.**

Composite score against realised net return, fully observed cohort only
(n=75, censored trades excluded):

| Score quartile | Range | n | Win rate | Mean return |
|---|---|---|---|---|
| q1 | 0.646–0.805 | 18 | 50.0% | **+2.15%** |
| q2 | 0.812–0.887 | 18 | 50.0% | +1.06% |
| q3 | 0.891–0.940 | 18 | 55.6% | **+3.17%** |
| q4 | **0.941–1.000** | 21 | 47.6% | **−0.70%** |

Not monotonic, and **the highest-scoring bucket is the worst**. The same
shape appears on the uncensored sample (q4: 41.7% win, −1.72%), so it is not
an artefact of the cohort filter.

Position within the list carries no information either:

| Rank shown | n | Mean return | Win rate |
|---|---|---|---|
| 1 | 65 | +0.44% | **44.6%** |
| 2 | 19 | +0.32% | 47.4% |
| 3 | 9 | +3.57% | 55.6% |

Rank 1 — the name presented first — has the **lowest** win rate of the three.

**Caveat, stated plainly.** n is 18–21 per bucket and 9–65 per rank. These are
suggestive, not conclusive; a monotonicity test at this sample size has very
little power. But the engine currently presents an *ordered* list, and there
is no evidence in the record that the order means anything.

**What to do.** Either demonstrate the ordering predicts, or stop implying it
does. Presenting five names as a ranked list when the ranking is unmeasured
is a claim the evidence does not support.

---

## Finding 3 — score stability

Median absolute session-over-session change in composite score for the same
name: **0.049** on a 0–1 scale; p90 **0.175**; max **0.390**.

A tenth of names move more than 17% of the full score range in one session.
Some of that is genuine cross-sectional re-ranking. How much is numerical is
not currently separable, because nothing decomposes a score change into
"the market moved" versus "the universe re-ranked around it".

**What to do.** Attribute score deltas. Until that exists, the 89% turnover
in Finding 1 cannot be diagnosed as signal or as noise.

---

## Finding 4 — what remains genuinely unmeasured

Nine of the directive's demands have no implementation and no measurement:

| Demand | State |
|---|---|
| Calibration curve / Brier score | **now measured above — it fails** |
| Per-prediction uncertainty | absent |
| Confidence separate from score | absent |
| Contradiction / coherence detection | absent |
| Top-k specific evaluation | **now measured above** |
| Rank stability / turnover | **now measured above — it fails** |
| Score stability | **now measured above** |
| Regime-sliced performance | absent |
| Feature-family ablation | absent |

Already present and adequate: signal decay (MAE/MFE are recorded per
outcome), out-of-distribution inputs (the staleness gate halts rather than
degrades), hard gates, abstention, model versioning, cost modelling.

---

## Prioritised roadmap

Ranked by expected impact × evidence strength ÷ complexity. Deliberately
short: the directive's own rule is that the engine must not become a feature
junkyard.

### P0 — reconcile the display with the strategy
Apply the Stage 6 hold band to the presented slate. **Evidence: Finding 1,
measured.** No new data, no new factor, no model change. This is the single
largest gap between what was validated and what is shown.

**Pass criteria:** median survival of a shown name rises from 1 session to
the band's implied dwell; realised cost drag falls to within the modelled 88
bps per position.

### P1 — stop presenting an unvalidated ordering
Either evidence the rank ordering or present the slate unordered.
**Evidence: Finding 2, measured.**

**Pass criteria:** a monotonicity test on rank vs forward return that clears
the project's own t ≥ 3 bar, on fully observed cohorts. If it cannot be
demonstrated, the ordering goes.

### P2 — separate confidence from score
The engine has one number where it needs three: opportunity, evidence
strength, data quality. Finding 2 shows the single number is not carrying
the first; it certainly is not carrying the other two.

**Pass criteria:** confidence buckets separate realised win rate where score
buckets do not.

### P3 — regime-sliced reporting
Every existing performance figure is an average across regimes the README
already shows the engine behaves differently in — momentum crashes
especially, where the risk family *amplifies* exposure (−9.82% vs −5.43%
without it).

**Pass criteria:** the crash-period behaviour already documented becomes
visible in routine reporting rather than only in a phase note.

### P4 — feature-family ablation
17 factors span 11 principal components with a participation ratio of 8.97.
The redundancy is measured; the *incremental value* of each family is not.

**Pass criteria:** each family either survives leave-one-out on the holdout,
or is removed.

---

## What this audit does not recommend

- **No new factors.** 17 already span 8.97 effective dimensions.
- **No new data feeds.** The existing OHLCV, delivery and index data is not
  fully exploited; Finding 1 is a presentation bug, not a data gap.
- **No ML upgrade.** With 6 independent holdout windows and 11 independent
  value observations, a more expressive model would fit noise faster.
- **No re-tuning.** PBO is 44.3%; configuration selection is already close to
  a coin flip.

The three findings above are all correctness issues in what the engine
already has. None of them is solved by making it bigger.

---

## Forward-test integrity

None of the P0–P4 work has been implemented. The forward test remains
registered at `ce83da4347d90323` with **zero observations**.

P0 and P1 change what is presented and, in P1's case, possibly what is
selected. Either would be a material change to the configuration under test.
If they are implemented, the forward test must be re-registered with a new
fingerprint and a clean boundary — which is exactly what the measurement
period mechanism exists to do.
