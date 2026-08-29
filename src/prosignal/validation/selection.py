"""W2 -- the winner's curse in a coefficient that had to pass a gate.

THE PROBLEM. `gated_shrink` zeroes any theme whose |t| falls below
`significance_floor` and prices the rest. So every traded coefficient is a
SURVIVOR of a selection on its own t-statistic, and the estimate that survived
is biased away from zero by exactly the amount the selection removed. Quoting
`mom_f t +2.87` as if it were an unconditional measurement overstates it, and
the overstatement is largest for the themes that only just cleared -- which are
the ones the decision actually turns on.

THE CORRECTION. Observed `t` given a true non-centrality `m`, conditioned on
having passed `|t| >= c` with the sign it in fact showed, is a one-sided
truncated normal. Its maximum-likelihood estimate of `m` solves

    m + lambda(c - m) = t,        lambda(x) = phi(x) / (1 - Phi(x))

which is monotone in `t`, never exceeds `t`, and decays to `t` far above the
boundary. Clamped at zero to preserve sign.

ONE-SIDED, AND THAT IS THE WHOLE POINT. A first attempt conditioned two-sided,
on `|t| >= c`. A survivor arrives WITH ITS SIGN, so the conditioning event is
`t >= c`. Measured over positive-tail survivors at a true effect of zero, the
two-sided version returns `+1.00` and the one-sided version `+0.59`: a theme
with no real effect at all that scraped the gate on noise came back as a
genuine effect, because the correction could not tell which tail it had come
from.

AND `+0.59` IS NOT ZERO. Even the correct estimator over-reports a true zero on
the positive tail, because the survivors that clamp are exactly those with the
least evidence and a clamp cannot subtract more than all of the effect. That
residual is a property of conditioning on survival rather than a defect in the
solve, and it is the sharpest single argument against trading this number --
sharper than criterion 6 below, because it holds at the one true effect where
the right answer is not in doubt.

HOW THAT WAS FOUND. Not by reading. The first version of the guard averaged the
SIGNED corrections over `|t| >= c`, where the two tails cancel at zero effect
and both estimators return zero. Reverting the source to the two-sided form
left the test green. `simulate_recovery(..., positive_tail=True)` exists
because of that miss.

IT FAILS ITS OWN ACCEPTANCE CRITERIA, AND IT SHIPS ANYWAY -- AS A REPORT.
Six criteria were fixed before it was written. Five hold exactly. The sixth,
that it recover the truth in simulation, holds for true effects up to about
m = 2.5 and fails from m = 3.0 upward: `corrected` is convex in `t` near the
clamp, so by Jensen the average correction sits below the truth even though
every individual one is defensible. At large `m` the gate barely binds, and a
correction that conditions on a truncation which is not really operating
over-shrinks.

So the criterion was set in advance, it is not met, and the number is therefore
REPORTED AND NOT TRADED. `assert_not_traded` exists so that wiring it into the
scoring path fails a test rather than passing a review, and the decision to
adopt it has to be taken again, in the open, by a person.

WHAT IT SAYS ABOUT THIS ENGINE. On the shipped fit the two traded themes come
back at implied true `t` of about +2.20 and +1.44 against their own floor of
2.0. One of the two coefficients the engine trades does not clear the bar it
was selected by, once the selection is priced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..validation.metrics import norm_cdf

__all__ = [
    "SelectionCorrected", "correct_t", "correct_all", "ACCEPTANCE",
    "assert_not_traded", "simulate_recovery",
]

#: Written before the correction was, and not edited since. Criterion 6 is the
#: one that fails; it is kept in the list rather than removed, because a set of
#: acceptance criteria that only contains the ones that passed is not a set of
#: acceptance criteria.
ACCEPTANCE: Dict[str, str] = {
    "monotone": "corrected t is non-decreasing in observed t",
    "never_inflates": "corrected t never exceeds observed t",
    "decays": "the correction tends to zero far above the gate",
    "sign_preserving": "a positive survivor never corrects to a negative effect",
    "no_new_parameter": "the gate c is the only input; nothing is tuned",
    "recovers_truth": "in simulation, mean corrected t is within 0.25 of the "
                      "true effect for m in [0, 4]",
}


def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _lambda(x: float) -> float:
    """Inverse Mills ratio phi(x)/(1-Phi(x)), stable in the upper tail.

    The naive form loses all precision above x ~ 8, where 1-Phi(x) underflows;
    the asymptotic expansion takes over there. Without it the correction
    silently returns garbage for a theme at t = 10 -- which is the case where
    it should return almost nothing.
    """
    tail = 1.0 - norm_cdf(x)
    if tail > 1e-12:
        return _phi(x) / tail
    inv = 1.0 / x
    return x + inv - 2.0 * inv ** 3 + 10.0 * inv ** 5


@dataclass(frozen=True)
class SelectionCorrected:
    name: str
    observed_t: float
    corrected_t: float
    gate: float
    #: True when the gate did not actually bind -- |t| far above c -- so the
    #: correction is arithmetically present and economically nil.
    negligible: bool

    @property
    def shrinkage(self) -> float:
        return abs(self.observed_t) - abs(self.corrected_t)

    def clears(self, floor: float) -> bool:
        return abs(self.corrected_t) >= floor

    def describe(self, floor: Optional[float] = None) -> str:
        s = (f"{self.name}: t {self.observed_t:+.2f} observed, "
             f"{self.corrected_t:+.2f} once the gate it passed is priced")
        if floor is not None:
            s += f" -- {'clears' if self.clears(floor) else 'MISSES'} {floor:.1f}"
        return s


def correct_t(observed_t: float, gate: float, *, tol: float = 1e-10,
              max_iter: int = 200) -> float:
    """MLE of the true non-centrality for a coefficient that passed ``|t| >= gate``.

    Solved by bisection rather than Newton: the objective is monotone, so
    bisection cannot diverge, and 200 halvings on a bracket of width 20 is
    exact to far beyond the precision of the t it is given. A Newton step here
    would be faster and would occasionally leave the bracket.
    """
    if not math.isfinite(observed_t) or not math.isfinite(gate) or gate <= 0:
        return float("nan")
    sign = 1.0 if observed_t >= 0 else -1.0
    t = abs(float(observed_t))
    if t < gate:
        # Not a survivor; there is no selection to undo.
        return float(observed_t)

    # g(m) = m + lambda(gate - m) - t is increasing in m, so bisect on [0, t].
    def g(m: float) -> float:
        return m + _lambda(gate - m) - t

    if g(0.0) >= 0.0:
        # Even a true effect of zero produces an observed t at least this
        # large once you condition on having passed. Sign-preserving clamp.
        return 0.0
    lo, hi = 0.0, t
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if g(mid) < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return sign * 0.5 * (lo + hi)


def correct_all(t_stats: Dict[str, float], gate: float,
                *, only: Optional[Iterable[str]] = None
                ) -> List[SelectionCorrected]:
    """Correct every theme that survived the gate. Order is stable by name."""
    names = sorted(t_stats) if only is None else [n for n in sorted(t_stats)
                                                  if n in set(only)]
    out = []
    for n in names:
        t = float(t_stats[n])
        if not math.isfinite(t) or abs(t) < gate:
            continue
        c = correct_t(t, gate)
        out.append(SelectionCorrected(
            name=n, observed_t=t, corrected_t=c, gate=gate,
            negligible=abs(abs(t) - abs(c)) < 0.05))
    return out


def simulate_recovery(gate: float, true_m: Sequence[float], *,
                      draws: int = 20_000, seed: int = 20260829,
                      positive_tail: bool = False) -> Dict[float, float]:
    """Criterion 6, run rather than argued.

    For each true effect, draw t ~ N(m, 1), keep the draws that would have
    passed the gate, correct them, and return the mean. A correction that works
    returns roughly m. This is the criterion the correction fails, and running
    it is how that is known rather than suspected.

    TWO STATISTICS, AND THEY ANSWER DIFFERENT QUESTIONS. The default keeps
    ``|t| >= gate`` and averages the SIGNED corrections, which asks whether the
    estimator is unbiased for the signed effect. `positive_tail` keeps
    ``t >= gate`` only, which asks what a SURVIVING POSITIVE COEFFICIENT
    reports -- and that is what an engine reads off the card.

    The default statistic cannot distinguish one-sided conditioning from
    two-sided: at a true effect of zero both tails are symmetric and the signed
    mean cancels to zero under either. Mutation testing found that -- the
    two-sided estimator was reinstated in the source and the test stayed green.
    On the positive tail the two are far apart: at m = 0 the one-sided
    correction reports +0.59 and the two-sided one reports +1.00, which is the
    "a pure-noise survivor came back as a genuine +1.0 effect" this module was
    written to avoid.

    +0.59 is not zero either. On the positive tail even the correct estimator
    over-reports a true zero, because a clamp at zero cannot subtract more than
    all of the effect and the survivors that clamp are exactly the ones with
    the least evidence. That residual is a property of conditioning on
    survival, not a defect in the solve, and it is the sharpest reason the
    number is reported rather than traded.
    """
    rng = np.random.default_rng(seed)
    out: Dict[float, float] = {}
    for m in true_m:
        t = rng.normal(float(m), 1.0, draws)
        kept = t[t >= gate] if positive_tail else t[np.abs(t) >= gate]
        if kept.size < 100:
            out[float(m)] = float("nan")
            continue
        out[float(m)] = float(np.mean([correct_t(float(x), gate) for x in kept]))
    return out


#: Modules that may compute a corrected t. Anything else importing it is
#: wiring a reported number into a decision.
REPORTING_ONLY = (
    "prosignal.cli",
    "prosignal.presentation",
    "prosignal.validation",
)


def assert_not_traded(module_name: str) -> None:
    """Raise if the correction is being reached from the scoring path.

    The correction failed its sixth acceptance criterion. That criterion was
    fixed in advance and it is not met, so the number is reported and not
    traded -- and this is what makes that a property of the code rather than an
    intention in a document. Adopting it means deleting this call deliberately,
    which is a decision somebody has to make and sign.
    """
    if not any(module_name.startswith(p) for p in REPORTING_ONLY):
        raise RuntimeError(
            f"{module_name} is reaching the selection correction. It failed "
            f"acceptance criterion 'recovers_truth' -- in simulation it "
            f"under-recovers true effects at m >= 3.0, where the per-draw "
            f"correction is convex and Jensen's inequality drags the average "
            f"below the truth. It is REPORTED, not TRADED. Wiring it into a "
            f"score means re-deciding that in the open, not importing it."
        )
