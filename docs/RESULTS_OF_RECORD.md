# RESULTS OF RECORD

> [!IMPORTANT]
> **This file is GENERATED. Do not edit it.** Run `prosignal research results` to regenerate it. Every figure below was produced by code from the store named in the stamp; a number that appears anywhere else in this repository and disagrees with this file is superseded by it, and `tests/test_readme_numbers.py` fails if README.md drifts.

## What produced these numbers

| | |
|---|---|
| generated at | `2026-09-06T05:56:24+00:00` |
| config version | `baseline-v2@25d9176dacd25857` |
| — parameters hash | `6828f4d19d68a1ac` |
| — store hash | `00333f4212937563` |
| — training-window hash | `4dc2dcfe23298c98` |
| shipped ranker | `v3_composite` |
| git commit | `36e8b2798f05` |
| engine version | `0.1.0` |
| data manifest digest | `86f8b3d6e8865906` |
| store fingerprint | delivery 1711s/4806n 2019-06-27..2026-09-03; fundamentals 740s/186n 2019-11-14..2025-03-11; indices 2218s/177n 2017-09-08..2026-09-03; prices 2219s/7112n 2017-09-08..2026-09-03 |
| panel span | 2019-05-23 → 2026-08-24 |
| panel rows | 200,190 |
| distinct signal dates | 359 |
| **independent observations** | **29.4** |
| horizon / stride | 63 / 5 sessions |
| cumulative trials charged | 639 |
| trials by v10 pass | pre-v10=619 |

**Read `independent observations` before any t-statistic below.** The panel has 200,190 rows and 29.4 independent 63-session windows. Every Sharpe, every information ratio and every deflated statistic in this engine is bounded by the second number, not the first.

## The ranking, judged apart from any book

The ordering is a different object from the book built on it, and this repository's history is largely the story of the two being confused. No naive `t` is quoted: signal dates are 5 sessions apart against a 63-session label, so observations overlap and the naive statistic is inflated by roughly `sqrt(VIF)`.

**`OUT_OF_SAMPLE` is the row a claim about the shipped model rests on.** The signs and weights were fitted over `v3.FIT_WINDOW`, which covers 269 of the panel's 359 signal dates, so a figure pooled across the whole panel is neither an in-sample fit statistic nor an out-of-sample result. Every published table quoted the pooled number.

`STABLE_MODEL` is a second and independent cut. `score_frame` re-caps the theme blend over the themes a name actually has, so a name scored on three themes and a name scored on five are combined by different weight vectors. The fundamentals feed reaches almost nobody at the start of the panel and most of the universe at the end, so `FULL_PANEL` averages across structurally different models with the weighting set by a data feed. `STABLE_MODEL` is the span over which every theme stays above +40% coverage -- the composite as it now stands, and there is much less of it.

`FULL_PANEL` is reported last rather than dropped. It is the longer record and the one every superseded figure came from.

| window | horizon | dates | rows | themes/name | rank IC | IC t (naive) | IC t (corrected) | quintile spread | spread t (corr.) | top-decile excess | top-decile t (corr.) | decile monotonicity | indep. obs | VIF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **OUT_OF_SAMPLE** | 5 | 90 | 67,002 | +4.80 | +0.0503 | +3.97 | **+3.97** | +0.34% | **+2.12** | +0.10% | **+0.92** | +0.156 | 90.0 | 1.00 |
| **OUT_OF_SAMPLE** | 21 | 87 | 64,539 | +4.79 | +0.0580 | +4.66 | **+2.28** | +0.99% | **+1.46** | +0.26% | **+0.59** | +0.264 | 21.5 | 4.17 |
| **OUT_OF_SAMPLE** | 42 | 83 | 61,283 | +4.79 | +0.0693 | +5.30 | **+1.86** | +1.79% | **+1.30** | +0.53% | **+0.62** | +0.313 | 10.8 | 8.15 |
| **OUT_OF_SAMPLE** | 63 | 78 | 57,367 | +4.79 | +0.0781 | +5.98 | **+1.73** | +2.28% | **+1.17** | +0.36% | **+0.29** | +0.330 | 7.1 | 11.94 |
| IN_SAMPLE | 5 | 269 | 131,870 | +4.31 | +0.0547 | +6.42 | **+6.42** | +0.51% | **+3.91** | +0.29% | **+3.98** | +0.172 | 269.0 | 1.00 |
| IN_SAMPLE | 21 | 269 | 131,247 | +4.31 | +0.0691 | +7.59 | **+3.70** | +1.60% | **+2.64** | +0.84% | **+2.64** | +0.214 | 64.8 | 4.22 |
| IN_SAMPLE | 42 | 269 | 130,517 | +4.31 | +0.0783 | +8.83 | **+3.06** | +2.88% | **+2.51** | +1.57% | **+2.43** | +0.279 | 32.9 | 8.34 |
| IN_SAMPLE | 63 | 269 | 129,920 | +4.31 | +0.0843 | +9.51 | **+2.70** | +4.06% | **+2.33** | +2.40% | **+2.44** | +0.315 | 22.3 | 12.42 |
| STABLE_MODEL | 5 | 153 | 112,499 | +4.75 | +0.0520 | +5.95 | **+5.95** | +0.39% | **+3.50** | +0.12% | **+1.57** | +0.190 | 153.0 | 1.00 |
| STABLE_MODEL | 21 | 150 | 109,727 | +4.75 | +0.0646 | +7.41 | **+3.62** | +1.22% | **+2.52** | +0.40% | **+1.29** | +0.258 | 36.5 | 4.20 |
| STABLE_MODEL | 42 | 146 | 106,105 | +4.75 | +0.0781 | +9.01 | **+3.13** | +2.31% | **+2.44** | +0.94% | **+1.54** | +0.341 | 18.3 | 8.27 |
| STABLE_MODEL | 63 | 141 | 101,909 | +4.74 | +0.0912 | +11.33 | **+3.24** | +3.39% | **+2.56** | +1.20% | **+1.37** | +0.409 | 12.1 | 12.24 |
| FULL_PANEL | 5 | 359 | 198,872 | +4.48 | +0.0536 | +7.53 | **+7.53** | +0.46% | **+4.43** | +0.24% | **+3.96** | +0.168 | 359.0 | 1.00 |
| FULL_PANEL | 21 | 356 | 195,786 | +4.47 | +0.0664 | +8.83 | **+4.30** | +1.45% | **+2.97** | +0.70% | **+2.64** | +0.226 | 85.5 | 4.22 |
| FULL_PANEL | 42 | 352 | 191,800 | +4.47 | +0.0762 | +10.24 | **+3.54** | +2.63% | **+2.79** | +1.33% | **+2.46** | +0.287 | 42.8 | 8.36 |
| FULL_PANEL | 63 | 347 | 187,287 | +4.46 | +0.0829 | +11.11 | **+3.15** | +3.66% | **+2.55** | +1.94% | **+2.34** | +0.318 | 28.5 | 12.47 |

### Where in the ordering the information actually is

`decile monotonicity` above compresses the whole shape into one rank correlation, and a profile that rises to D7 and falls away can score well there. Each row below is the mean excess of that decile over its own date's cross-section, averaged across dates. **The shipped book holds six names off the very top of D10.**

| window | horizon | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 | peak | D10−D6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **OUT_OF_SAMPLE** | 5 | -0.27% | -0.18% | -0.04% | +0.02% | +0.06% | +0.04% | +0.04% | +0.11% | +0.13% | +0.10% | D9 | +0.06% |
| **OUT_OF_SAMPLE** | 21 | -0.99% | -0.45% | -0.27% | -0.26% | +0.22% | +0.51% | +0.49% | +0.24% | +0.28% | +0.24% | D6 | -0.27% |
| **OUT_OF_SAMPLE** | 42 | -1.65% | -0.86% | -0.53% | -0.64% | +0.25% | +1.05% | +0.97% | +0.41% | +0.48% | +0.54% | D6 | -0.51% |
| **OUT_OF_SAMPLE** | 63 | -2.31% | -1.27% | -0.60% | -0.65% | +0.31% | +1.23% | +1.73% | +0.64% | +0.58% | +0.36% | D7 | -0.87% |
| IN_SAMPLE | 5 | -0.33% | -0.20% | -0.08% | -0.05% | -0.03% | +0.05% | +0.08% | +0.12% | +0.16% | +0.30% | D10 | +0.25% |
| IN_SAMPLE | 21 | -1.20% | -0.50% | -0.26% | -0.02% | -0.11% | +0.05% | +0.27% | +0.32% | +0.61% | +0.87% | D10 | +0.82% |
| IN_SAMPLE | 42 | -2.03% | -1.01% | -0.51% | -0.17% | -0.26% | +0.20% | +0.41% | +0.70% | +1.08% | +1.60% | D10 | +1.39% |
| IN_SAMPLE | 63 | -2.77% | -1.41% | -0.93% | -0.10% | -0.22% | +0.06% | +0.70% | +0.78% | +1.44% | +2.45% | D10 | +2.39% |
| STABLE_MODEL | 5 | -0.31% | -0.20% | -0.07% | -0.02% | +0.07% | +0.04% | +0.07% | +0.15% | +0.15% | +0.12% | D8 | +0.08% |
| STABLE_MODEL | 21 | -1.02% | -0.57% | -0.22% | -0.24% | +0.18% | +0.28% | +0.43% | +0.34% | +0.44% | +0.39% | D9 | +0.11% |
| STABLE_MODEL | 42 | -1.91% | -1.09% | -0.42% | -0.45% | +0.18% | +0.74% | +0.78% | +0.63% | +0.64% | +0.93% | D10 | +0.19% |
| STABLE_MODEL | 63 | -2.97% | -1.55% | -0.69% | -0.47% | +0.19% | +0.87% | +1.37% | +1.05% | +1.05% | +1.19% | D7 | +0.32% |
| FULL_PANEL | 5 | -0.32% | -0.20% | -0.07% | -0.03% | -0.01% | +0.05% | +0.07% | +0.11% | +0.16% | +0.25% | D10 | +0.21% |
| FULL_PANEL | 21 | -1.15% | -0.49% | -0.26% | -0.08% | -0.03% | +0.16% | +0.32% | +0.30% | +0.53% | +0.71% | D10 | +0.55% |
| FULL_PANEL | 42 | -1.94% | -0.97% | -0.52% | -0.28% | -0.14% | +0.40% | +0.54% | +0.64% | +0.94% | +1.35% | D10 | +0.95% |
| FULL_PANEL | 63 | -2.66% | -1.38% | -0.85% | -0.22% | -0.10% | +0.32% | +0.93% | +0.75% | +1.25% | +1.98% | D10 | +1.65% |

Read the `peak` and `D10−D6` columns against each other across the two windows. In sample the profile is monotone and D10 wins by a distance; out of sample it peaks in the middle of the upper half and D10 is not the best decile. The BOTTOM of the distribution generalises closely -- which is the Stambaugh-Yu-Yuan result, reproduced from a long-only panel that was never built to test it, and it is not a leg this engine can trade.

## The two book tables, re-run

README.md carried two performance tables that cannot both describe the same engine. Both configurations are re-run below against the store named in the stamp. **They are not averaged, and the more favourable one is not quoted.**

### RESULTS OF RECORD -- the shipped book against its own universe

**Status: SUPERSEDED** — the published claim is stated in `mean_excess` and `ir`, both of which are confounded with the risk budget and are no longer headline figures. Re-run on the current store the book deploys 20.3% of capital, so of its -16.3% raw annual excess, -16.2% is the cash it is not holding. The leverage-neutral reading is +6.74% a year on deployed capital at t +1.55 -- indistinguishable from zero. The old claim is neither confirmed nor refuted; it is expressed in a retired unit.

*Claimed in:* README.md, 'RESULTS OF RECORD'

*Configuration:* ranking.source=v3_composite (22 factors in 5 themes); 6 slots, entry rank 6, exit rank 18, horizon 63 sessions; stop 8xATR (armed=True), target 3R (armed=False), invalidation armed=False; absolute floor DISABLED; shipped cost model

> This arm **is** the configuration the engine ships.

**Headline — leverage-neutral.** The book does not hold all of its capital, so a raw comparison against a fully-invested benchmark measures the sizing knob as much as the signal:

| | value |
|---|---|
| **alpha on deployed capital (ann)** | **+6.74%** |
| t(alpha) | +1.55 |
| excess on deployed capital (ann) | -0.60% |
| capital actually deployed | 20.3% |

**Full reconciliation.** The rows marked *confounded* move with `risk_per_trade_pct` even when the ranking and the names are identical; they are retained so published figures can be traced, not because they measure anything:

| | book | benchmark (equal-weight eligible universe) |
|---|---|---|
| mean return / period | +0.33% | +1.69% |
| annualised | +4.0% | +20.3% |
| Sharpe | +0.73 | +0.88 |
| beta to benchmark | +0.13 | — |
| alpha / period *(scales with deployment)* | +0.11% | — |
| leverage-matched excess (ann) | -0.1% | — |
| mean excess / period *(confounded)* | -1.36% | — |
| information ratio *(confounded)* | -0.79 | — |
| periods beating the benchmark *(confounded)* | 37.0% | — |
| worst schedule drawdown | -12.2% | — |
| mean names held | 4.8 | — |
| periods scored | 351 | — |

**Gross and cost, separately** — netting them and keeping the last number hides which of the two is binding:

| | annualised |
|---|---|
| gross excess over the universe | -15.1% |
| cost drag | -1.2% |
| **net excess** | **-16.3%** |

*power: expected t = IR x sqrt(years) = -0.79 x sqrt(7.3) = -2.13; t=2.0 is unreachable at a non-positive IR*

**Claimed against measured**, every published figure, headline or not:

| figure | published claim | re-run | verdict | headline? |
|---|---|---|---|---|
| alpha on deployed capital (ann) | n/a | +6.74% | NOT_TESTABLE | yes |
| excess on deployed capital (ann) | n/a | -0.60% | NOT_TESTABLE | yes |
| information ratio [LEVERAGE-CONFOUNDED] | -83.00% | -79.21% | matches | no |
| mean excess / period [LEVERAGE-CONFOUNDED] | -4.23% | -1.36% | matches | no |
| periods beating the benchmark | +32.90% | +37.04% | matches | no |
| alpha / period (scales with deployment) | -0.67% | +0.11% | OPPOSITE SIGN | no |

### Tuning pass (2026-08-29) -- sector-neutral 6-1 momentum, 6 names

**Status: WITHDRAWN** — annualised alpha: claimed +0.203, measured +0.02095; Sharpe: claimed +1.59, measured +1.052; annualised book return: claimed +0.426, measured +0.05635

*Claimed in:* README.md, 'What changed in the tuning pass (2026-08-29)' (the section appeared twice) and config `expectancy:`

*Configuration:* ranking = sector-neutral rank of mom_6_1 (close[t-21]/close[t-147] - 1), ONE column; 6 slots, entry rank 6, exit rank 18, entries every 21 sessions, held to a 63-session backstop; disaster floor 8xATR clipped to 35% of entry; no profit target, no invalidation exit; shipped cost model. This is `ranking.source=measured_factor`, which is NOT what the engine ships.

> This arm is **not** what the engine ships. It is re-run here only to find out whether its published numbers reproduce.

**Headline — leverage-neutral.** The book does not hold all of its capital, so a raw comparison against a fully-invested benchmark measures the sizing knob as much as the signal:

| | value |
|---|---|
| **alpha on deployed capital (ann)** | **+12.92%** |
| t(alpha) | +2.97 |
| excess on deployed capital (ann) | +13.46% |
| capital actually deployed | 16.2% |

**Full reconciliation.** The rows marked *confounded* move with `risk_per_trade_pct` even when the ranking and the names are identical; they are retained so published figures can be traced, not because they measure anything:

| | book | benchmark (equal-weight eligible universe) |
|---|---|---|
| mean return / period | +0.47% | +1.77% |
| annualised | +5.6% | +21.3% |
| Sharpe | +1.05 | +0.91 |
| beta to benchmark | +0.17 | — |
| alpha / period *(scales with deployment)* | +0.17% | — |
| leverage-matched excess (ann) | +2.2% | — |
| mean excess / period *(confounded)* | -1.30% | — |
| information ratio *(confounded)* | -0.79 | — |
| periods beating the benchmark *(confounded)* | 37.7% | — |
| worst schedule drawdown | -10.4% | — |
| mean names held | 4.9 | — |
| periods scored | 355 | — |

**Gross and cost, separately** — netting them and keeping the last number hides which of the two is binding:

| | annualised |
|---|---|
| gross excess over the universe | -14.9% |
| cost drag | -0.8% |
| **net excess** | **-15.7%** |

*power: expected t = IR x sqrt(years) = -0.79 x sqrt(7.3) = -2.13; t=2.0 is unreachable at a non-positive IR*

**Claimed against measured**, every published figure, headline or not:

| figure | published claim | re-run | verdict | headline? |
|---|---|---|---|---|
| annualised alpha | +20.30% | +2.09% | outside 0.05 | yes |
| Sharpe | +1.59 | +1.05 | outside 0.5 | yes |
| annualised book return | +42.60% | +5.64% | outside 0.1 | yes |
| excess Sharpe | +1.12 | n/a | NOT_TESTABLE | no |

## What these numbers are not

- Both arms are priced with the SHIPPED cost model at the shipped impact coefficient. Gross excess, cost drag and net excess are reported separately on every arm; the cost question and the gross-edge question are different questions and netting them hides which one is binding.
- The book model is the repository's own cohort simulator (`portfolio_sim.simulate` / `phase_summary`), the same one `research portfolio` uses. It is NOT a bit-reproduction of the continuous weekly book the sealed holdouts evaluated, so magnitudes are not comparable to HOLDOUT_V3_A/B. Read the sign and the ordering.
- The panel spans the whole store, which OVERLAPS the surfaces both of these configurations were selected on. Neither arm is out-of-sample evidence; this is a reproduction check, not a validation.

---

*Generated by `prosignal research results` (docs/RESULTS_OF_RECORD.md). Regenerate; do not edit.*
