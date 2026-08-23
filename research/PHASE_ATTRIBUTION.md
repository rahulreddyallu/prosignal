# Factor attribution and overlap-corrected significance

Run 2026-08-23 against the shipped configuration `39e9687c57bd12bd`, on the
sacred holdout: 15 dates, 2025-03-05 to 2026-05-19, fitted on the 73 panel
dates that end at least 1.5 horizons before the holdout opens.

## 1. The overlap correction changes every t in this project

The label is a 63-session forward return and panel dates are 21 sessions
apart, so each observation shares two thirds of its window with the next.
Measured against simulated noise at this exact sampling scheme:

| statistic | rejects a true null at nominal 5% | power against a real edge |
|---|---|---|
| naive t | **32.0%** | — |
| Newey-West, estimated inflation | 19.5% | — |
| Newey-West, analytic inflation | **10.0%** | 64% |
| stationary bootstrap, 95% CI | **28.7%** | 81% |

Two findings, both unwelcome.

**Newey-West underestimates the inflation at this sample size.** It recovers
1.74 where the arithmetic gives 3.00, because a lag-2 covariance estimated
from 15 observations is very noisy. The sampling scheme is known, so the
module now uses the analytic variance inflation factor and takes the larger of
the two when both are available. That halves the false-positive rate.

**The block bootstrap does not work here at all.** Its interval excludes zero
25-30% of the time on pure noise, for every mean block length from 2 to 8. The
failure is sample size, not tuning: 28.7% at n=15, 15.5% at n=30, 8.5% at
n=120. Resampling blocks of an already-overlapping series does not reproduce
the null when there are only fifteen blocks to draw from. `BOOTSTRAP_MIN_N =
30`, and below it the result is returned flagged `uncalibrated` and is not
usable as a test. This contradicts the standard recommendation to bootstrap,
and the recommendation is wrong at this n.

## 2. The holdout, re-measured

| | recorded in BASELINE_V1 | this run |
|---|---|---|
| top-decile excess per period | +3.45% | **+4.35%** |
| rank IC | 0.0878 | **0.1069** |
| naive t | 3.20 | 5.26 |
| overlap-corrected t | — | **3.13** |

The re-run is stronger, not weaker, and the corrected t **clears** the t >= 3.0
bar. The difference from the recorded figures comes from the training cutoff
and from configuration changes since the baseline was frozen. This corrects
the headline of the audit written earlier the same day, which used the recorded
3.20 and inferred a corrected 1.85.

## 3. The alpha does not survive the factors

Six long-short factors built from the engine's own definitions, same universe,
same dates, 30% tails. This is a hostile test by construction: the regressors
are as close to the strategy's own inputs as they can be.

| factor set | R² | alpha / period | corrected t | survives |
|---|---|---|---|---|
| MOM only | 0.201 | +3.67% | +2.49 | yes |
| MOM + SIZE | 0.342 | +1.94% | +0.85 | no |
| MOM + SIZE + LIQ | 0.581 | +1.16% | +0.59 | no |
| MOM + VALUE + LOWVOL | 0.202 | +3.81% | +1.57 | no |
| **all six** | **0.730** | **−1.01%** | **−0.38** | no |

Loadings in the full model, by contribution:

```
SIZE     beta +1.702   t +1.89    contributes +0.0917
LIQ      beta -0.968   t -1.07    contributes -0.0604
MOM      beta +1.184   t +3.66    contributes +0.0199
LOWVOL   beta -0.091   t -0.28    contributes +0.0026
VALUE    beta -0.297   t -0.07    contributes -0.0003
QUALITY  beta +4.169   t +0.98    contributes -0.0000
```

Momentum is the only loading with a t above 2. The alpha dies as soon as a
size factor enters, and the shortlist is consistent with that: the names it
surfaces are small and mid caps.

## 4. What this does and does not establish

**Establishes.** The ranking carries information: a raw excess of +4.35% per
period at a corrected t of 3.13, on a holdout that was never touched during
development. That is a real result and it clears the project's own bar.

**Establishes.** The excess is substantially factor exposure. Six factors
explain 73% of its variance, momentum loads at t 3.66, and the intercept is
negative once size is priced.

**Does not establish.** Which of those two readings is right. Fifteen
observations against six factors leaves eight degrees of freedom, the alpha
estimate moves from +3.67% to −1.01% depending on which factors are included,
and that instability is exactly what this sample size produces. The
attribution is directionally unfavourable and it is not conclusive.

The honest summary is that the engine ranks Indian equities in a way that
worked out of sample, and that most of what it captures appears to be
momentum and a small-cap tilt rather than something the factors do not
already price.
