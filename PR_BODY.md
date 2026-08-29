## Remediate the August audit findings

Takes every finding from the August audit to a root cause, and re-audits the
result adversarially. **The verdict does not change: NOT READY.** That is the
outcome, not a shortfall — the engine now reports its own performance
accurately, and what it reports is negative.

18 commits · 22 files · +3,153 / −164 · suite **1,420 passed, 0 failed**
Base `a359622` → head `21aa26c`

---

### What this does not do

No parameter, gate, threshold or bar was changed. The diff to `config/` is
**comments only** and the config hash is unchanged at
`baseline-v1@127d8a314ec49aa2`, so the forward-test registration survives. Two
of the four headline statistics got *worse* as a direct result of fixing how
they were measured.

### The measurement defects

| | before | after |
|---|---|---|
| Deflated Sharpe | 1.000 — passed at 100,000 trials | **0.346, fails at 81** |
| CPCV sample size | 601 (split, date) pairs read as observations | 70 distinct dates → **23.6** independent |
| Live feature date | four sessions stale on every run | the decision date |
| Significance gate | estimated overlap correction (1.44–1.99) | analytic floor (2.97) |
| Universe price floor | applied to back-adjusted prices | quoted price — 4,905 cells recovered |
| Ranked population | 23.3% unbuyable; 1.55 of top 8 refused | masked in training *and* at the decision |
| Benchmark | none anywhere in the repository | equal-weight eligible universe, everywhere |
| Reported drawdown | mean of phases, −18.6% | worst single schedule, −20.6% |

The DSR was the sharpest of these. It ran on the pooled excess vector — every
test date counted about nine times — while its documented "conservative unit
variance" fallback was in fact `1/(n−1)`, which at n = 639 is 1.6e−3. Two
errors compounding in the same direction produced a multiple-testing defence
insensitive to multiple testing.

### The finding that matters

Nothing in the repository had ever compared the book to buying the market. With
a benchmark, adding each layer of the engine in turn over 70 out-of-sample
dates:

| | mean | vs benchmark | delta |
|---|---|---|---|
| equal-weight eligible universe | +4.68% | — | — |
| **ranking alone, no exits** | +5.73% | **+1.05%** | — |
| + risk-budget sizing | +4.68% | −0.00% | **−1.05%** |
| + 2.5× ATR stop | +2.05% | −2.63% | **−2.63%** |
| + 3R target | +1.62% | −3.06% | −0.42% |
| + invalidation exit | +1.50% | −3.18% | −0.12% |
| **+ costs — shipped book** | +0.83% | **−3.85%** | −0.67% |

The ranking earns +1.05%. Sizing gives back exactly that. The stop takes more
than twice what the ranking produced, and the stop test is monotone with no
interior optimum. The traded stop value was **not** changed — that is a
risk-tolerance decision for whoever holds the capital, and loosening it because
the backtest prefers it is the search this exercise forbids. What changed is
that the false "2.5× is near-optimal" justification was withdrawn and the price
is now stated.

### Two new findings raised while fixing

- **N1** — verifying the corporate-action fix returned *zero change* because
  reading `adj_factor` without a price column served a write-time placeholder
  of 1.0, indistinguishable from "no corporate actions". Re-measured: 4,905
  cells across 165 symbols. The store now raises.
- **N2** — the config declined uniqueness weighting on a stated measurement
  ("within-date sd is exactly 0.000 on every date") that is **false**: sd runs
  0.082–0.207, zero on none of 87 dates. The conclusion survives on a better
  argument; the weighting is refused explicitly rather than silently dropped.
  Recorded rather than deleted, because it would have added a third traded
  theme and that is exactly the result not to take.

### Left open

**W2** — the winner's curse in the traded coefficients. A correction was built
and **failed criterion 6 of its own pre-committed ship rule**, so it is
reported and not traded, and a guard fails if anyone wires it in without
re-deciding. What it reports: corrected for the selection the gate performs,
`delivery`'s implied true t is **+1.44**, below the gate it passed.

### The remediation was itself audited

An independent adversarial pass returned **REJECT** and eleven defects, most of
them introduced by the remediation. All fixed. The three worst:

- the admissible-population fix reached **1 session in 21** — refit runs every
  21 days and the cached path was never masked;
- the DSR repair could silently un-repair itself: `0.878 FAIL → 1.0000 PASS`
  from one omitted keyword;
- the selection correction conditioned on the wrong event, reporting a theme
  with **no real effect** as a genuine +1.0σ one. The pre-committed criterion
  could not see it because it averaged over both tails.

It also found a null test of mine that had become **vacuous** and three guards
that passed on deliberately reverted code. A mutation probe now reverts 14
fixes and catches **14 of 14**, each by the guard written for it.

### Review path

Start with `docs/FORWARD_TEST.md` and the three `work/audit/W*_failure_model.md`
files — each states the failure model and the acceptance criteria *before* the
fix, with the result appended after. `tests/test_remediation_guards.py` has 74
guards, each naming the defect it prevents.

Merging this does not make the system ready. It makes it honest.
