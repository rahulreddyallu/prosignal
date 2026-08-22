# Phase 13 — Horizon surface and signal decay

**Questions.** Gap 3: is 63 sessions genuinely optimal, or did it merely happen
to be optimal in this sample? Gap 4: does exiting before 63 forfeit the
forecast return?

**Answers.** 63 is not uniquely optimal — it sits on a broad plateau. And no:
alpha accrues *fastest* early. What early exit forfeits is cost amortisation,
not alpha.

## Method, and the confound it avoids

The naive construction re-labels the panel at each horizon and compares. That
confounds two things. At H=189 a re-labelled panel has ~10 independent
observations against ~31 at H=63, so a long horizon can look better purely from
a smaller, noisier sample; and cost per unit time falls mechanically with
horizon whether or not the signal is any good.

This holds the **signal dates fixed**: the model is fitted once per date on a
purged expanding window, the score is taken at *t*, and that same score is
scored against forward returns at every horizon. Decay becomes a property of
the signal rather than of the sampling.

## Gross: IC rises with horizon, per-session alpha falls

45 signal dates, selection period:

| H | IC | t(IC) | excess | ann. excess | per-session |
|---|---|---|---|---|---|
| 1 | +0.0176 | +1.02 | −0.03% | −6.73% | −2.8 bp |
| 5 | +0.0458 | +2.81 | +0.27% | +14.80% | +5.5 bp |
| 21 | +0.0513 | +3.11 | +0.52% | +6.47% | +2.5 bp |
| 42 | +0.0712 | +5.79 | +1.10% | +6.78% | +2.6 bp |
| **63** | **+0.0769** | **+6.95** | **+1.44%** | **+5.88%** | **+2.3 bp** |
| 84 | +0.0805 | +7.43 | +1.73% | +5.28% | +2.1 bp |
| 126 | +0.0847 | +8.49 | +2.58% | +5.22% | +2.0 bp |
| 189 | +0.0874 | +10.38 | +2.86% | +3.83% | +1.5 bp |

**Raw IC rises monotonically and would nominate H = 189.** Per-session alpha
falls monotonically and would nominate H = 5. Neither is the answer, because
neither pays for trading.

## Net of cost, at portfolio level: a plateau

Real turnover, risk-based sizing, buffer bands, and a cost that scales with
participation:

| H | return/period | **ann. net** | Sharpe | max DD | new names/rebalance |
|---|---|---|---|---|---|
| 21 | +0.92% | **+11.65%** | +0.75 | −8.8% | 3.1 |
| 42 | +1.82% | **+11.45%** | +0.77 | −8.0% | 4.5 |
| **63** | +2.78% | **+11.61%** | **+0.78** | −10.6% | 5.6 |
| 84 | +3.60% | **+11.20%** | +0.73 | −9.5% | 6.1 |
| 126 | +4.85% | +9.93% | +0.61 | −7.7% | 6.8 |
| 189 | +5.42% | +7.29% | +0.47 | −5.3% | 7.1 |

**Annualised net return is flat from 21 to 84 sessions — 11.20% to 11.65%, a
spread of 0.45 percentage points.** That is the performance plateau §43 asks
for, and it is evidence of robustness rather than a tuned value.

Sharpe peaks at 63 and falls away on both sides — 0.75, 0.77, **0.78**, 0.73,
0.61, 0.47.

**Correction.** An earlier version of this table reported Sharpe rising
monotonically to 0.86 at H=126 and concluded that longer horizons were better
risk-adjusted. That was an annualisation bug in `phase_summary`, which
hardcoded `sqrt(4)` — correct only at H=63. At H=21 there are twelve periods a
year and the factor is `sqrt(12)`, so short horizons were understated by 1.73×
and long ones overstated. Corrected, the Sharpe plateau matches the return
plateau instead of contradicting it.

Note that the gross IC table would have chosen H = 189, and net of costs
H = 189 is the **worst** horizon tested. The confound was real.

## Should the horizon change?

Corrected, **63 has the highest Sharpe of every horizon tested** and sits at the
top of the return plateau. H = 84 costs 0.05 of Sharpe and 0.41 points of
return; H = 21 costs 0.03 of Sharpe.

**No change. 63 stays**, and now for a positive reason rather than a tie.

### Would a shorter horizon buy statistical power?

It is the obvious idea: H = 21 triples the independent observations, from 31 to
93, and the value block from 11 to 36. Tested end to end, it does not work:

| H | indep obs | selection IC | holdout IC | holdout t | holdout excess | DSR |
|---|---|---|---|---|---|---|
| 21 | 93 | +0.0616 | +0.0584 | +2.04 | +0.65% | **0.003** |
| 63 | 31 | +0.0876 | +0.0878 | +3.20 | +3.45% | **0.994** |

The signal is genuinely weaker at 21 sessions and the difference is
cost-independent — holdout IC 0.0584 against 0.0878. Gross annualised
top-decile excess is 7.8% at H = 21 against 13.8% at H = 63.

The portfolio-level equivalence above is real but does not mean what it appears
to. With buffer bands only 3.1 of 8 names turn over per rebalance at H = 21, so
the book is not running a 21-day strategy — hysteresis stretches the effective
hold well past the nominal horizon. **More observations of a weaker signal is
not more evidence.**

## Gap 4: what early exit actually forfeits

The prior understanding was that exiting before 63 sessions forfeits much of
the forecast return. That is true in *total* terms and misleading per unit time:

- per-session alpha is **highest early** — 5.5 bp at H = 5 against 2.3 bp at
  H = 63
- but annualised net return is flat across 21–84

So the alpha does not need time to accrue. **What shorter holding costs you is
round trips, and over 21–84 sessions the faster accrual and the higher cost
roughly cancel.** The reason to hold a quarter is cost amortisation, not
waiting for the forecast to mature.

This also means a dynamic early exit is not forfeiting alpha the way it
appeared to. It is spending cost. Whether it earns that back is a separate
question and one this data cannot answer at n = 31.
