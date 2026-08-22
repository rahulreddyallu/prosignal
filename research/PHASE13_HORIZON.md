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
| 21 | +0.92% | **+11.65%** | +0.44 | −8.8% | 3.1 |
| 42 | +1.82% | **+11.45%** | +0.63 | −8.0% | 4.5 |
| **63** | +2.78% | **+11.61%** | +0.78 | −10.6% | 5.6 |
| 84 | +3.60% | **+11.20%** | +0.84 | −9.5% | 6.1 |
| 126 | +4.85% | +9.93% | +0.86 | −7.7% | 6.8 |
| 189 | +5.42% | +7.29% | +0.81 | −5.3% | 7.1 |

**Annualised net return is flat from 21 to 84 sessions — 11.20% to 11.65%, a
spread of 0.45 percentage points.** That is the performance plateau §43 asks
for, and it is evidence of robustness rather than a tuned value.

Sharpe improves with horizon to about 126 (0.44 → 0.86) as the return per
rebalance grows against roughly constant per-period volatility. Drawdown
improves too. Past 126 the gross alpha has decayed enough that net return
falls away.

Note that the gross IC table would have chosen H = 189, and net of costs
H = 189 is the **worst** horizon tested. The confound was real.

## Should the horizon change?

H = 84 buys +0.06 of Sharpe and costs 0.41 points of annualised return against
H = 63. The CPCV distribution of portfolio Sharpe has **sd 0.83 across splits**.
A 0.06 difference is noise, and H = 126 rests on 15 independent observations
against 31 at H = 63.

**No change. 63 stays.**

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
