# RESULTS OF RECORD

> [!IMPORTANT]
> **This file is GENERATED. Do not edit it.** Run `prosignal research results` to regenerate it. Every figure below was produced by code from the store named in the stamp; a number that appears anywhere else in this repository and disagrees with this file is superseded by it, and `tests/test_readme_numbers.py` fails if README.md drifts.

## What produced these numbers

| | |
|---|---|
| generated at | `2026-09-07T14:18:38+00:00` |
| config version | `baseline-v2@2496c3bcbf1dd996` |
| — parameters hash | `6828f4d19d68a1ac` |
| — store hash | `4a3cf2f68f9950fe` |
| — training-window hash | `0682c59badec28c4` |
| shipped ranker | `v3_composite` |
| git commit | `2c0a10a52749` **(working tree dirty)** |
| engine version | `0.1.0` |
| data manifest digest | `808e9c104f856f7f` |
| store fingerprint | delivery 1712s/4808n 2019-06-27..2026-09-04; fundamentals 740s/186n 2019-11-14..2025-03-11; indices 2219s/177n 2017-09-08..2026-09-04; prices 2220s/7119n 2017-09-08..2026-09-04 |
| panel span | 2019-05-23 → 2026-08-03 |
| panel rows | 197,940 |
| distinct signal dates | 356 |
| **independent observations** | **29.2** |
| horizon / stride | 63 / 5 sessions |
| cumulative trials charged | 119 |
| trials by v10 pass | pre-v10=99 |

**Read `independent observations` before any t-statistic below.** The panel has 197,940 rows and 29.2 independent 63-session windows. Every Sharpe, every information ratio and every deflated statistic in this engine is bounded by the second number, not the first.

## The ranking, judged apart from any book

The ordering is a different object from the book built on it, and this repository's history is largely the story of the two being confused. No naive `t` is quoted: signal dates are 5 sessions apart against a 63-session label, so observations overlap and the naive statistic is inflated by roughly `sqrt(VIF)`.

| horizon | dates | rows | rank IC | IC t (naive) | IC t (corrected) | quintile spread | spread t (corr.) | top-decile excess | top-decile t (corr.) | decile monotonicity | indep. obs | VIF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 21 | 356 | 195,781 | +0.0545 | +8.24 | **+4.01** | +1.14% | **+2.60** | +0.49% | **+1.88** | +0.182 | 85.5 | 4.22 |
| 42 | 352 | 191,795 | +0.0582 | +9.03 | **+3.12** | +2.01% | **+2.46** | +0.93% | **+1.75** | +0.218 | 42.8 | 8.36 |
| 63 | 348 | 188,022 | +0.0618 | +9.52 | **+2.70** | +2.79% | **+2.20** | +1.33% | **+1.61** | +0.244 | 28.5 | 12.47 |

## The two book tables, re-run

README.md carried two performance tables that cannot both describe the same engine. Both configurations are re-run below against the store named in the stamp. **They are not averaged, and the more favourable one is not quoted.**

### RESULTS OF RECORD -- the shipped book against its own universe

**Status: REPRODUCED** — the direction and magnitude of the published headline claim survive a re-run on the current store. Note, and it is reported rather than dropped because it is not a headline figure: alpha / period claimed -0.0067 against +0.00107 measured (OPPOSITE SIGN). Alpha here is a near-zero residual of two different books with different betas, so its sign is not stable; the headline claim is the underperformance, and that reproduces.

*Claimed in:* README.md, 'RESULTS OF RECORD'

*Configuration:* ranking.source=v3_composite (22 factors in 5 themes); 6 slots, entry rank 6, exit rank 18, horizon 63 sessions; stop 8xATR (armed=True), target 3R (armed=False), invalidation armed=False; absolute floor DISABLED; shipped cost model

> This arm **is** the configuration the engine ships.

| | book | benchmark (equal-weight eligible universe) |
|---|---|---|
| mean return / period | +0.79% | +5.72% |
| annualised | +3.2% | +22.9% |
| Sharpe | +0.61 | +0.93 |
| mean excess / period | -4.93% | — |
| information ratio | -0.89 | — |
| beta to benchmark | +0.12 | — |
| alpha / period | +0.11% | — |
| periods beating the benchmark | 32.9% | — |
| worst schedule drawdown | -9.2% | — |
| mean names held | 4.7 | — |
| periods scored | 346 | — |

**Gross and cost, separately** — netting them and keeping the last number hides which of the two is binding:

| | annualised |
|---|---|
| gross excess over the universe | -19.3% |
| cost drag | -0.5% |
| **net excess** | **-19.7%** |

*power: expected t = IR x sqrt(years) = -0.89 x sqrt(7.2) = -2.40; t=2.0 is unreachable at a non-positive IR*

**Claimed against measured**, every published figure, headline or not:

| figure | published claim | re-run | verdict | headline? |
|---|---|---|---|---|
| information ratio | -83.00% | -89.37% | matches | yes |
| mean excess / period | -4.23% | -4.93% | matches | yes |
| periods beating the benchmark | +32.90% | +32.95% | matches | yes |
| alpha / period | -0.67% | +0.11% | OPPOSITE SIGN | no |

### Tuning pass (2026-08-29) -- sector-neutral 6-1 momentum, 6 names

**Status: WITHDRAWN** — annualised alpha: claimed +0.203, measured +0.01485; Sharpe: claimed +1.59, measured +0.9147; annualised book return: claimed +0.426, measured +0.0539

*Claimed in:* README.md, 'What changed in the tuning pass (2026-08-29)' (the section appeared twice) and config `expectancy:`

*Configuration:* ranking = sector-neutral rank of mom_6_1 (close[t-21]/close[t-147] - 1), ONE column; 6 slots, entry rank 6, exit rank 18, entries every 21 sessions, held to a 63-session backstop; disaster floor 8xATR clipped to 35% of entry; no profit target, no invalidation exit; shipped cost model. This is `ranking.source=measured_factor`, which is NOT what the engine ships.

> This arm is **not** what the engine ships. It is re-run here only to find out whether its published numbers reproduce.

| | book | benchmark (equal-weight eligible universe) |
|---|---|---|
| mean return / period | +1.35% | +5.74% |
| annualised | +5.4% | +22.9% |
| Sharpe | +0.91 | +0.93 |
| mean excess / period | -4.39% | — |
| information ratio | -0.84 | — |
| beta to benchmark | +0.17 | — |
| alpha / period | +0.37% | — |
| periods beating the benchmark | 32.8% | — |
| worst schedule drawdown | -8.2% | — |
| mean names held | 4.8 | — |
| periods scored | 348 | — |

**Gross and cost, separately** — netting them and keeping the last number hides which of the two is binding:

| | annualised |
|---|---|
| gross excess over the universe | -17.2% |
| cost drag | -0.4% |
| **net excess** | **-17.6%** |

*power: expected t = IR x sqrt(years) = -0.84 x sqrt(7.2) = -2.26; t=2.0 is unreachable at a non-positive IR*

**Claimed against measured**, every published figure, headline or not:

| figure | published claim | re-run | verdict | headline? |
|---|---|---|---|---|
| annualised alpha | +20.30% | +1.49% | outside 0.05 | yes |
| Sharpe | +1.59 | +91.47% | outside 0.5 | yes |
| annualised book return | +42.60% | +5.39% | outside 0.1 | yes |
| excess Sharpe | +1.12 | n/a | NOT_TESTABLE | no |

## What these numbers are not

- Both arms are priced with the SHIPPED cost model at the shipped impact coefficient. Gross excess, cost drag and net excess are reported separately on every arm; the cost question and the gross-edge question are different questions and netting them hides which one is binding.
- The book model is the repository's own cohort simulator (`portfolio_sim.simulate` / `phase_summary`), the same one `research portfolio` uses. It is NOT a bit-reproduction of the continuous weekly book the sealed holdouts evaluated, so magnitudes are not comparable to HOLDOUT_V3_A/B. Read the sign and the ordering.
- The panel spans the whole store, which OVERLAPS the surfaces both of these configurations were selected on. Neither arm is out-of-sample evidence; this is a reproduction check, not a validation.

---

*Generated by `prosignal research results` (docs/RESULTS_OF_RECORD.md). Regenerate; do not edit.*
