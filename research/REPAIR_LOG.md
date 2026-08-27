# ProSignal Repair — execution log

Branch `repair/label-and-families`, tagged `pre-repair-baseline` at commit `8281530`.
Runbook line numbers were written against `bf7702c`; every one re-verified at HEAD before editing.
Holdout unspent — no command in this repair was given `--include-holdout`.

---

## Job 0 — baseline captured

Rollback artifact: `research/coef_pre_repair.json`. Live model archived to
`data/curated/crosssec_model_versions/`.

Live coefficients matched the audit's verification block exactly, so the model had **not**
refitted since 2026-08-25 and the audit's figures stand:

| theme | coef | fm t |
|---|---|---|
| mom_f | 0.0 | −0.139 |
| reversal_f | −0.056746 | −5.864 |
| lottery_f | −0.083108 | −6.283 |
| risk_f | 0.0 | +1.862 |
| delivery_f | +0.025722 | +5.194 |

`dropped_for_coverage`: all twelve value/quality/log_mcap columns at 0.381 date-span against
the 0.60 floor — Job 10's premise, confirmed from the live file rather than assumed.

**Test suite before any change: 1362 passed, 10 skipped, 0 failed.**

### What the baseline actually showed

`research factors` — momentum priced NEGATIVELY under the barrier label:

    mom -0.0244 · reversal -0.1258 · lottery -0.1893 · risk +0.0212 · delivery +0.1158

`research estimator` — the production arm scored rank IC **+0.1903 (t +4.13)** against the
barrier label and beat the 1/N control by +0.0313, yet its **top-decile excess was negative,
−0.49% (t −0.87)**. High IC against a circular label, negative money. That one table is the
audit's thesis.

`research cpcv` — pooled top-decile excess −0.56%; **Deflated Sharpe 0.000 charging 81 trials,
FAIL**; worst of 36 paths Sharpe −0.24; **100% of paths below zero**; PBO **33% (PASS)**.

### Corrections to the runbook's Phase E gates

| Gate | Runbook | Measured |
|---|---|---|
| CPCV negative-path share | "no worse than the recorded 0%" | **100% of paths below zero** |
| PBO | 44.3% | **33%** |

The "no worse than 0%" gate is unsatisfiable — the pre-repair engine is already at 100%. Gates
are re-anchored to the measured baseline above.

---

## Job 1 — the training label. Carries the whole effect.

`config/parameters.yaml` — `labels.triple_barrier: true → false`. `barrier_source`,
`upper_sigma`, `lower_sigma` and `vol_window_sessions` left in place; inert while the flag is
false, and they document the geometry that was calibrated.

**The argument is algebraic and needs no data.** Under the engine's geometry a winner's
realised return is `+3 × (2.5·ATR/P)` and a loser's `−1 × (2.5·ATR/P)`. 91% of rows resolve at
one of the two, so `|label| ∝ ATR/P` — the label's *magnitude* is the name's volatility. Any
feature correlated with volatility predicts it by construction. That is why lottery reached
t −6.3 while momentum fell to t −0.14.

**Not an argument against the triple-barrier method.** In López de Prado (2018) the barrier
label is a *classification* target (−1/0/+1), and its volatility scaling exists precisely to
make that classification comparable across names of different volatility. Using the scaled
*magnitude* as a cross-sectional ranking target puts back exactly the heteroskedasticity the
scaling removes. The defect is the target, not the method.

Second channel: 39.5% of labels resolve as thesis invalidation, so a name that gains 30% over
the quarter but dips through MA50 − 1.5×ATR in week three is booked as a loss — momentum's own
drawdown profile, entered into the target with the sign reversed.

The geometry is untouched and still enforced where it belongs: `stage7_risk` places the stop,
`stage6_entry` holds the exit band, `exits.resolve_exits` scores the outcome.

### Job 1 result — measured

`research factors`, same builder, before and after one config line:

| family | before (barrier label) | after (forward return) |
|---|---|---|
| **mom** | **−0.0244** | **+0.0673** |
| reversal | −0.1258 | +0.0157 |
| lottery | −0.1893 | −0.0357 |
| risk | +0.0212 | +0.0349 |
| delivery | +0.1158 | +0.0354 |

`research estimator`, out of sample, purged walk-forward:

| | before | after |
|---|---|---|
| themes priced | 3 (reversal, lottery, delivery) | **2 (mom, delivery)** |
| mom in-sample t | −0.02 | **+3.57** |
| rank IC | +0.1903 | +0.0462 |
| **top-decile excess** | **−0.49% (t −0.87)** | **+1.07% (t +1.32)** |
| beats 1/N control | +0.0313 | +0.0175 |

Rank IC falls because the old figure was measured against a circular label. Lower IC, positive
money. Momentum's sign reverses and it becomes the strongest family. `reversal` and `lottery` —
the two themes the shipped model priced most confidently — collapse toward zero once the label
stops being proportional to volatility, which is what "the model discovered that volatility
predicts the label" means once you remove the volatility. `delivery` falls from +0.1158 to
+0.0354, so two thirds of the delivery signal was also label artifact.

**Cache trap.** `load_cached` validated fit date, feature set and estimator — never the label,
which was not stored in the blob at all. Handled by hand here (archive), made structural by
Guard A.

`config_version` moved `a30a8d4847080ddc → 1b2f891704ae3bb6`, which by the forward test's own
invalidation rules voids the registration opened 2026-08-27. Re-registered last, not first.

---

## Jobs 2 & 3 — family construction

`risk` split into `beta` and `drawdown`; `idio_skew` pulled out of `lottery` into its own
single-member `skew` theme. Seven price-derived families now (nine including value/quality).

Blast radius handled — every consumer keyed on family name:
`famamacbeth.THEME_PRIOR_SIGN`, `stage4._MODEL_CITE`, `evidence.FAMILY_MAP`,
`evidence.EVIDENCE_CATEGORIES`, `evidence._LEVEL_CATEGORIES`.
`stage8._exposure_themes` reads the priced themes off the card dynamically and needed nothing.

Priors: `beta: -1` on Frazzini & Pedersen (2014), and in this market Agarwalla, Jacob, Varma &
Vasudevan (2014), who find the BAB factor earns significant positive returns in India and
*dominates* the size, value and momentum factors. `drawdown: None` — no literature analogue for
a drawdown-depth cross-section. `skew: -1` on Bali, Cakici & Whitelaw (2011) and Boyer, Mitton &
Vorkink (2010).

### The audit's stated benefit for Job 2 does not survive Job 1

The audit justified splitting `risk` by reporting beta alone at t −3.67 and max_dd alone at
t +4.69 — "two significant signals averaged into an insignificant column". Both figures were
measured **under the barrier label**. Refitted on real data against the forward return:

    beta_f      t +0.52     gated out
    drawdown_f  t +0.51     gated out

Neither carries a signal in either direction. The split is still correct — averaging a
higher-is-riskier rank with a higher-is-safer rank under a common sign cancels the axis, and
that is indefensible on construction whatever the data says — but it **recovers nothing**. The
runbook said "expect no payoff"; the truth is stronger than that. The two significant signals it
promised to rescue were themselves label artifacts.

Recorded because the opposite mistake is easy: reading the split as validated because the
engine improved, when the improvement is entirely Job 1's.

`skew` reads t −1.84 — correct sign for its −1 prior, below the gate, visible and zeroed.
Exactly the outcome the runbook predicted and the reason for preferring a visible zero to an
invisible dilution.

---

## Job 4 — prox_52w skips the last 21 sessions

`prox_52w` used today's close, so the most recent month sat inside it — the same window
`resid_reversal` prices with the opposite sign, so momentum and reversal partially cancelled
through one factor (ρ +0.378 as shipped, −0.029 with the skip). It now reads the 252-session
high ending 21 sessions back.

George & Hwang (2004) do **not** skip — they measure nearness on the current price. The
deviation is deliberate and the comment says so rather than implying the paper's authority.

**Chain reaction, caught by the engine's own validator.** The factor now needs 273 sessions, so
`crosssec.MIN_LOOKBACK` moved 253 → 274 and the config loader refused the file:
`unexplained_jump_lookback_sessions` (260) no longer covered the longest feature lookback, which
would leave a corporate action in the 14-session gap invisible to Stage 1 and fully consumed by
the features. Raised 260 → 281, preserving the same 7-session margin.

Second consequence: `coverage.model_minimum` is `MIN_LOOKBACK + horizon + 60`, so the bar for
the model to fit at all moved **376 → 397 sessions**. A store of 330 sessions is now 67 short
rather than 46. That is an operator-facing number and the test asserts it literally.

`modelprint.MODEL_SOURCES` already covers `crosssec.py`, so the model fingerprint moves
automatically — no manual provenance note was needed, contrary to the runbook's suggestion.

---

## Jobs 5 & 6 — Stage 5 double counts

Disabled `gap_signal` (ρ +0.88 with idio_vol), `news_spike` (ρ +0.81 with lottery) and
`overextension` (ρ +1.00 with resid_reversal). Each restated a factor the fit already prices,
in the same direction; stacking a hand-set constant on a fitted coefficient produces an
effective loading nobody estimated, and pushing a loading past its estimated optimum strictly
lowers expected out-of-sample IC. Double-counting is not the conservative choice.

Kept: `earnings_distortion`, `corporate_action_distortion`, `liquidity_distortion`,
`data_integrity`. These measure **tradability**, which the ranking model genuinely cannot see,
and they are why Stage 5 should continue to exist.

Job 6 resolved the one straight contradiction: `beta_explained_move` imposed a hand-set
*negative* beta tilt worth up to 0.20 of a 0.35 penalty budget while the fitted model measured
beta's slope as *positive* and declined to price it. Resolved in favour of the estimator —
`beta` is now its own theme carrying the Frazzini–Pedersen prior, and the Stage 5 override is
off. A defence that contradicts your own estimator is a position taken without measuring it.

---

## Job 7 — penalty units

`score_after = max(score_before − total_penalty, 0)` where `score_before` is `composite_unit`,
a rank uniform on [0,1] by construction. A penalty of 0.10 does not remove "0.10 of expected
return"; there is no return in that unit. It demotes the name ten percentile points, and it
demotes by the same ten points on a tightly-bunched day as on a widely-dispersed one.

Option 1 taken: arithmetic unchanged, the language made honest at the arithmetic and on the
contract fields. Fields **not** renamed — they are serialised into the ledger and the API, and
renaming would break stored runs to fix a naming problem.

Option 2 — expressing a penalty as a fraction of the day's `prediction_dispersion` and applying
it to `composite_raw` before the rank transform — is the right design, is a real change to what
gets selected, and is deliberately **not** done here. It belongs in a measured trial.

---

## Job 8 — the refit-day attribution bug

`stage4_core_score` built the live feature frame **without `sectors` and `actions`** on the
accepted-refit branch, while the cached, rejected-refit and superseded branches all passed both.
So on a refit day the score came from sector-neutral ranks while the card's contributions,
member breakdown and redundancy report were computed from universe-wide ranks: the numbers on
the card did not sum to the number on the card. Fixed; all four call sites now pass an identical
keyword set.

---

## Job 9 — uniqueness weighting is inert, for two independent reasons

1. `fit_coefficients` forwards `weights` only on the **ridge** branch. `fama_macbeth()` never
   receives them and `_ols_slopes` is a plain unweighted `lstsq`, so under the shipped
   `estimator.method: fama_macbeth` the weights reach nothing.

2. More fundamentally there is now nothing to weight. Since `triple_barrier: false` every label
   runs the full horizon, `build_panel` assigns `held = 63` to every row, and uniqueness is
   computed *within symbol* over identically-shaped spans. **Measured on a rebuilt panel the
   within-date standard deviation of uniqueness is exactly 0.000 on every date.**

So implementing WLS in `famamacbeth._ols_slopes` would be a provable no-op, not an improvement.
The runbook offered "implement it or declare it honest" as a judgement call; it is not one.
Left `true` because it is read and honoured on the ridge and meta-label paths.

---

## Job 10 — the regime layer says what it did

`reachable_multipliers` correctly refuses to scale momentum down when no defensive family is
priced, because the weight would rotate into whatever else is weighted — `delivery` on the
shipped model — and delivery was never a crash stabiliser. That guard fires on every run:
value and quality are dropped upstream at 38% date-span against a 60% floor.

The diagnosis was computed, attached to the model, and **read by nothing** — only four
references, all inside `crossmodel.py`. The card printed a regime and gave the reader no way to
learn the multiplier never reached a score. That silence was the defect, not the guard.

The card now states it, including that the crash control which *does* bind is the entry gate
(`no_new_entry_buckets`), not a weighting. `score_with` — the cached path serving 20 of every 21
sessions — now attaches the reachability too; it previously set only
`regime_multipliers_applied`, so the honest note would have appeared on refit days alone.

The deeper question the runbook raises stands and is **not** resolved here: yfinance carries
~5 years of statements against a panel starting 2018, so value and quality cannot reach the 60%
floor. Either accept they are permanently absent and remove the regime multipliers as a
mechanism, or source deeper statement history. Left as a decision, flagged, not taken.

---

## Job 11 — guards

**A. Label fingerprint in the model cache.** `save_cache` now writes what the model was fitted
against — horizon, barrier source and the full exit geometry — and `load_cached` refuses a blob
whose label differs. A pre-fingerprint blob is treated as a mismatch rather than a pass:
refitting once on upgrade is cheap, scoring for six weeks against a retired label is not. Built
from the arguments the fit actually consumed, so it cannot drift from the model it describes.

**B. The significance cliff — implemented, and deliberately OFF.** `gated_shrink` zeroes any
theme below |t| 2.0 and keeps the rest near full strength, making a traded coefficient a step
function of a noisy statistic: `risk` read t +1.86 live and +2.45 on the rebuild. The continuous
form `t²/(t²+c)` with c = 4.0 puts the half-weight point exactly at |t| = 2, so it smooths the
shipped rule rather than replacing it, with a hard zero below |t| = 1.

It ships **off**. The argument for it is structural, but switching it on moves live
coefficients, which makes it an estimator change and therefore a trial — and the DSR is already
charging 81 trials and reading 0.000. This is the one item in the runbook that is not a
correctness fix, and it is left for `research estimator` to decide as a recorded comparison
rather than settled by whoever edits the config.

**C. Cross-family sign audit.** For every family, no two members may correlate negatively unless
one is declared in `NEGATED_IN_FAMILY`. Measured on a synthetic cross-section spanning beta 0.3
to 2.0 with idiosyncratic volatility varying alongside it, which reproduces the real panel's
structure closely enough to be a real test:

| pair | fixture | audit's real panel |
|---|---|---|
| lottery members | +0.80 … +0.95 | 0.48 … 0.68 |
| **beta vs max_dd** | **−0.829** | **−0.417** |
| idio_skew vs downside_vol | +0.125 | +0.04 |

A companion test asserts the retired `risk` shape still fails the guard, so it cannot pass
vacuously.

---

## Defects found during the repair, not in the runbook

**Job 12 — the decay monitor's kill criterion ignored theme sign.**

`validation/decay.py` compared a raw rolling t against `kill_t = 0.0` with no orientation, so
every theme priced negatively breached on every check, forever. The pre-repair baseline returned
**KILL: mom, reversal, lottery** — and `reversal` (t −4.78) and `lottery` (t −5.05) both carry
literature priors of −1 and were behaving exactly as those priors say they should.

A sign error inside a **pre-committed removal rule** is the worst place for one: the rule's
whole authority is that it was declared before the numbers were seen, and acting on it would
have deleted the two best-working themes plus momentum. It also made Phase E's decay gate
meaningless.

Fixed by orienting the comparison by the theme's expected direction — the literature prior where
one exists, the theme's own full-sample sign where none does. The rule itself is unchanged: same
floor, same required consecutive breaches. Only the direction "below" is measured in.

**A ZeroDivisionError on the production fit path.** `hierarchical_shrink` built its
inverse-variance pool as `1.0 / se²` while filtering only on `isfinite(se)`, so a theme with a
constant slope series (se = 0) raised `ZeroDivisionError` — caught upstream as "cross-sectional
model failed", taking the ranking down for the run. Unreachable while every family had two or
more members; single-member themes make it reachable. A zero standard error is a degenerate
column the panel could not move, not an infinitely precise measurement, so it is now excluded
from the pool rather than dominating it.

---

## State

**Test suite: 1384 passed, 9 skipped, 0 failed** (from 1362/10/0). Every changed assertion was
rewritten with what was measured and when, not deleted.

Real-data refit, 750-name universe, 43,555 training rows, split families:

| theme | coef | t |
|---|---|---|
| **mom_f** | **+0.02976** | **+4.12** |
| delivery_f | +0.01828 | +4.77 |
| lottery_f | −0.01618 | −2.26 |
| reversal_f | 0.0 | +1.69 |
| skew_f | 0.0 | −1.84 |
| beta_f | 0.0 | +0.52 |
| drawdown_f | 0.0 | +0.51 |

Label fingerprint on the fitted model: `{'horizon': 63, 'triple_barrier': False,
'source': 'forward_return'}`.

### Not yet done

- **Phase E gauntlet** — running; outputs land as `research/AFTER_ALL_*.txt`. No pre-repair
  baseline exists for `research portfolio` or `research spread`; those two are post-only.
- **End-to-end `analyse run`** — blocked. The store ends 2026-08-25 and Stage 1's staleness gate
  counts against today (2026-08-27), so every pipeline entry point refuses. Clearing it needs
  `prosignal data ingest`, which deepens the store and would break like-for-like comparability
  with the baselines captured above. Deliberately deferred until the gauntlet is complete.
- **Forward test re-registration** — must be last, after validation, with the discarded
  registration documented.
