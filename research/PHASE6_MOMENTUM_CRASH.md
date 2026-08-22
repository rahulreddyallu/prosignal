# Phase 6 — Momentum crash risk

**Question (Gap 6).** Do the current risk controls protect the model against a
momentum crash? **They do not. In the one real crash this sample contains they
made it worse, and the regime gate caught less than a fifth of the damage.**

## The condition, from the literature

Daniel & Moskowitz (2016) find momentum crashes are partly forecastable: they
occur in **panic states** — following market declines, when market volatility is
high — and are **contemporaneous with market rebounds**. The mechanism is the
option-like payoff of past losers, whose betas rise sharply after a decline, so
a book holding past winners is on the wrong side exactly when the rebound pays.

## The standard walk-forward cannot see it

The walk-forward needs 36 training dates, so its first test date is **2022-01**.
February–March 2020 is inside the training window. **Every validation result in
this repository — CPCV, holdout, PBO, portfolio — excludes the only textbook
momentum-crash setup in the sample.** That is not a small caveat.

Measured on the 2022-onward window the model looks *better* in panic states
(+3.57% excess against +1.74% normal, n = 8). That result is an artefact of the
window containing no real crash.

## Reached with a deliberately short training window

Shortening the training window to 15 dates makes 2020 testable. The fit is thin
and these are weak numbers — but a weak observation of the real event beats a
strong observation of a period without one.

| signal date | market fwd 63d | top-decile excess | without risk family |
|---|---|---|---|
| 2020-04-01 | +27.3% | **−12.31%** | −12.09% |
| 2020-05-07 | +13.5% | −7.79% | −7.17% |
| 2020-06-08 | +9.2% | −5.64% | +1.19% |
| 2020-09-03 | +14.3% | −6.47% | −2.38% |
| 2020-10-05 | +20.2% | **−11.01%** | −3.21% |
| 2020-11-03 | +26.2% | **−13.14%** | −5.13% |
| 2021-04-08 | +8.6% | −9.52% | −2.03% |

**The top decile underperformed the universe in 13 of 15 periods**, worst
−13.14%. On the four strongest-rebound dates (market +15% or more): mean excess
**−9.82%**, worst −13.14%.

## The risk family amplified the loss

| | mean excess on strong-rebound dates |
|---|---|
| shipped 17 factors | **−9.82%** |
| risk family removed | **−5.43%** |

The four risk factors — low `downside_vol`, low `beta_120`, low `max_dd_120`,
low `max5_21` — tilt *away* from high-beta beaten-down names. Those are exactly
the names that lead a rebound. **In a crash the risk controls are not neutral;
they are positioned on the wrong side and cost 4.4 additional points.**

This is coherent with the mechanism rather than a fluke, and it is the reverse
of the intuition that a risk factor is protective.

## The regime gate catches the acute phase and misses the rest

`no_new_entry_buckets` already contains `uptrend_highvol_rebound`, described in
the config as the Daniel & Moskowitz state. It fired:

| | n | mean excess | worst |
|---|---|---|---|
| blocked by the gate | 4 | −6.09% | −12.31% |
| allowed through | 11 | −3.77% | **−13.14%** |

| | mean excess |
|---|---|
| all 15 periods | −4.39% |
| gate-filtered | −3.77% |
| **protection** | **+0.62 points** |

By August 2020 volatility had normalised to `uptrend_midvol` while the rebound
ran for another eight months. **The D&M state is defined on trailing conditions;
the damage extends well past them.** The single worst period, 2020-11-03 at
−13.14%, was classified `uptrend_midvol` and allowed through.

## What was NOT done

The obvious response is to widen the no-entry buckets, or to add a rebound
detector, or to drop the risk family in high-volatility states. **None of that
was done, on fifteen observations of one event.**

Retuning regime buckets against a single crash is how a strategy acquires a
rule that fits 2020 and nothing else, and PBO is already 44.3%. The honest
output of a sample containing one crash is a documented exposure, not a fitted
defence.

## What this means

The engine has a **known, literature-predicted failure mode that its current
controls do not mitigate**:

- worst observed 63-session excess: **−13.14%**
- the risk family **costs** 4.4 points in that state rather than protecting
- the regime gate recovers **+0.62 points** of −4.39%
- and every other validation figure in this repository was computed on a window
  that excludes it

That last point is the one to carry forward. The CPCV distribution, the holdout
DSR and the portfolio spread are all conditioned on a period with no crash in it.
