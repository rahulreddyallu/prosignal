# ProSignal — consolidated research record

One document for what was measured, what it means, and what it does not.
Everything here is drawn from `BASELINE_V1.json`, the phase notes beside it,
and the validation modules. Where a number is re-measured and disagrees with
what was recorded, both appear and the disagreement is the point.

Assembled 2026-08-25 against `baseline-v1@39e9687c57bd12bd`.

---

## 1. The finding, first

The ranking carries real out-of-sample information. Six self-defined factors
explain **73%** of it, and the residual alpha is **−1.01% per period at
t −0.38**.

| | |
|---|---|
| Holdout top-decile excess | **+4.35%** per 63-session period |
| Overlap-corrected t | **3.13** — clears the pre-set t ≥ 3.0 bar |
| CPCV ranking paths | 36, median Sharpe 0.50, **none negative** |
| Alpha vs six factors | **−1.01%**, t **−0.38** |
| R² of that regression | **0.730** |

Both halves are true simultaneously. The engine ranks; what it ranks on is
mostly momentum and a size tilt.

---

## 2. Evidence levels

Claims below are tagged, because they are not equally supported.

- **[L]** literature — established in published work, not re-derived here
- **[E]** ProSignal empirical — measured on this data, this code
- **[D]** design decision — chosen, with a reason, not validated
- **[H]** hypothesis — stated, untested

---

## 3. What the model is

A ridge regression on 17 cross-sectional factor ranks, `alpha = 20000`,
refit on every run against a 63-session forward-return label. **[D]**

Because it refits from stored history, **the store is the training set**. A
store at 1,900 sessions and one at 2,200 produce different coefficients from
identical code and identical config. Every ledger row now carries a model
fingerprint for exactly this reason. **[E]**

### Factor structure

| Measure | Value |
|---|---|
| Named factors | 17 |
| Principal components for 90% of variance | **11** |
| Participation ratio | **8.97** |
| Kaiser criterion | **5** |
| Momentum share of IC | **41%** |

The momentum block's 41% is **one latent factor**: a sign-pinned PC1 of the
three momentum factors reproduces their equal-weighted IC (+0.0755 against
+0.0752), and explains 59% of the block's variance. **[E]**

Nine momentum variants span **2.57 effective dimensions**. Diversification
cannot come from adding more momentum. **[E]**

---

## 4. Horizon

63 sessions was chosen from a surface, not tuned to a peak. **[E]**

| Horizon | Gross IC | Net annualised | Alpha (bp/session) |
|---|---|---|---|
| 21 | 0.0513 | 11.65% | 2.5 |
| 42 | 0.0712 | 11.45% | — |
| **63** | **0.0769** | **11.61%** | **2.3** |
| 84 | 0.0805 | 11.20% | — |
| 126 | 0.0847 | 9.93% | 2.0 |
| 189 | 0.0874 | 7.29% | 1.5 |

Net return is flat from 21 to 84 sessions — a **plateau, not a peak**. IC
rises monotonically with horizon while net return falls, which is cost and
turnover, not signal. **[E]**

Shortening does not rescue the sample: at H=21 independent observations
roughly triple but IC falls 0.0878 → 0.0584 and DSR collapses 0.994 → 0.003.
**More observations of a weaker signal is not more evidence.** **[E]**

---

## 5. Validation design

- **CPCV** — 10 groups, 3 test, purge 63, embargo 21, over 2018-11-27 →
  2025-02-07, a window that **contains the 2020 crash and rebound**. **[D]**
- **Purge = label horizon.** It was 21 against a 63-session label, leaving 42
  sessions of every training row's label window inside the test block. The
  loader now refuses `purge < horizon`. **[E]**
- **Holdout** — 378 sessions from 2025-02-07, untouched during development.
- **Trial accounting** — `cumulative_trials_logged: 0`. DSR is charged
  against a local count, not a global one. **[H]**

### The overlap correction

Labels are 63 sessions; panel dates are 21 apart. Every observation shares
two-thirds of its window with the next. Measured against simulated noise at
this exact scheme: **[E]**

| Statistic | False-positive rate at nominal 5% | Power |
|---|---|---|
| Naive t | **32.0%** | — |
| Newey–West, estimated | 19.5% | — |
| Newey–West, **analytic** | **10.0%** | 64% |
| Stationary bootstrap | **28.7%** | 81% |

Two consequences, both against convention:

1. Newey–West alone is insufficient — at n=15 it recovers an inflation of
   1.74 where the arithmetic gives 3.00, so the analytic factor is used.
2. **The block bootstrap does not work at this sample size.** 28.7% at n=15,
   15.5% at n=30, 8.5% at n=120, for every block length 2–8. Below
   `BOOTSTRAP_MIN_N = 30` the result is returned flagged and is not usable.

---

## 6. Results

### Holdout

| | Recorded | Re-measured |
|---|---|---|
| Top-decile excess | +3.45% | **+4.35%** |
| Rank IC | 0.0878 | **0.1069** |
| Naive t | 3.20 | 5.26 |
| **Corrected t** | — | **3.13** |
| Net annualised | 10.99% | — |
| DSR (11 trials) | 1.00 | — |

Fifteen reported periods; **six independent** 63-session windows.

### CPCV

| | Result |
|---|---|
| Ranking paths | 36, median Sharpe 0.50, min 0.16, **0% negative** |
| Pooled IC | 0.062 |
| Portfolio splits | 45, median Sharpe 0.84, min **−1.66**, **18% negative** |

No t-statistic is quoted from CPCV, deliberately: test dates recur across
splits and paths share training sets, so neither is a sample of independent
experiments. A path t of **+14.85** was computed during development and
discarded for this reason. **[E]**

### PBO

**44.3%** across seven factor sets. The bar is ≤50%, so it passes narrowly.
Selection among configurations is close to a coin flip. **[E]**

### Factor attribution — the decisive test

Six long-short factors built from the engine's own definitions. A hostile
test by construction.

| Factor set | R² | Alpha/period | Corrected t | Survives |
|---|---|---|---|---|
| MOM only | 0.201 | +3.67% | +2.49 | yes |
| MOM + SIZE | 0.342 | +1.94% | +0.85 | **no** |
| + LIQ | 0.581 | +1.16% | +0.59 | no |
| **All six** | **0.730** | **−1.01%** | **−0.38** | no |

Momentum is the only loading above t=2 (+3.66). **The alpha dies as soon as
size is priced**, consistent with a shortlist of small and mid caps. **[E]**

Not conclusive: 15 observations against 6 factors leaves 8 degrees of
freedom, and the estimate swings from +3.67% to −1.01% depending on which
factors enter. Directionally unfavourable, not settled.

---

## 7. Risk

The risk factor family **amplifies** momentum-crash exposure rather than
damping it. **[E]**

| Through the 2020 rebound | |
|---|---|
| Worst 63-session top-decile excess | **−13.14%** |
| Periods underperforming | 13 of 15 |
| Four strongest rebound dates, with risk family | **−9.82%** |
| Same dates, without it | **−5.43%** |
| Recovered by the regime gate | +0.62 points |

Low-beta and low-drawdown tilts point away from the beaten-down names that
lead a recovery. A risk control that makes the tail worse is a position, not
a control.

---

## 8. Sample-size ceilings, and why they cannot be raised

| Block | Factors | Independent obs at H=63 |
|---|---|---|
| Momentum | 3 | 30 |
| Risk | 4 | 30 |
| Liquidity | 2 | 30 |
| Reversal | 1 | 30 |
| Delivery | 2 | 26 |
| **Value** | **5** | **11** |

The model weights the value block as if it were as evidenced as the price
block. It is not, and this is the single largest known weakness. **[E]**

Every ceiling is external: **[E]**

- NSE **403s before 2016-01** — price history cannot reach further back
- Delivery **not re-servable before ~2021**
- Vendor statement bulk coverage begins **2023-06**
- Measured disappearance **4.3%/year** — a 26-year reconstruction would be
  missing **68%** of the companies that actually traded

---

## 9. Corrections made during this work

Recorded because they bound how much the surviving numbers should be trusted.

| Error | Effect |
|---|---|
| Walk-forward silently used `alpha=10` not 20000 | Ablation ordering changed on rerun |
| Stop-loss analysis measured per-position | Missed that tighter stops buy larger positions; the "89% of alpha" claim was wrong |
| Overlapping cohorts implied 3× leverage | Fixed with phase offsets |
| Path t of +14.85 nearly shipped | Paths share training sets |
| SVD sign unpinned | Inverted a PC1 conclusion |
| `phase_summary` hardcoded `sqrt(4)` | Correct only at H=63 |
| Headline t computed on a stale figure | 1.85 → corrected 3.13 |
| `pct_change` fill | **34,433 zero returns fabricated** across 263 names |
| Re-entries counted as separate positions | 137 closed trades → **86**; total 101% → **61.7%** |

---

## 10. What is and is not established

**Strongly established** — leakage control (purge at the label horizon,
embargo, untouched holdout, point-in-time universe with measured
survivorship); realistic size-dependent costs applied before every net
figure; out-of-sample ranking information across 36 CPCV paths, none
negative; momentum's 41% is one latent factor; 63 sessions is a plateau.

**Reasonably supported** — holdout excess +4.35% at corrected t 3.13, on six
independent windows.

**Preliminary** — the attribution, at 8 degrees of freedom.

**Not established** — that any alpha survives factor exposure; that the score
is a probability (no calibration exists); that it works live (never traded,
not even on paper); that this configuration is the right one (PBO 44.3%);
that the value factors are evidenced (n=11); that it survives a momentum
crash (it does not — the risk family amplifies it); that the regime windows
are right (UNVALIDATED, never searched); that the trial count is complete
(`cumulative_trials_logged: 0`).

---

## 11. What would change the answer

Ordered by uncertainty reduced per unit of effort.

1. **Complete the 18-month forward test.** The only route to independent
   observations. Everything else is secondary.
2. **Resolve the risk family's crash behaviour** — fix it, or state plainly
   that the engine carries unhedged momentum-crash exposure.
3. **Log cumulative trials**, so the DSR is charged globally.
4. **Downweight or drop the value block.** Weighting 5 factors on 11
   observations equally with factors on 30 is unsupported.
5. **Retire the composite fallback.** A scorer measured at t −0.11 should not
   be able to render cards at all.

---

## 12. The honest summary

> A competently built factor harvester whose incremental alpha is, on the
> evidence available, indistinguishable from zero.

The ranking works. The claim that it works *for a reason not already priced*
does not survive its own attribution test. The forward test exists to settle
that, and it has not run yet.
