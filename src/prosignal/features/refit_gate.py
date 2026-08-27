"""Promotion gate for a new set of model coefficients.

The model refits every 21 sessions and the new coefficients replace the live
ones. That is the one place where a corrupted upstream date, a feed change or a
bad adjustment can reach every future decision at once, silently, without
failing any test: the fit succeeds, the numbers look like numbers, and the
ranking quietly changes.

So a refit is proposed, not installed. It is compared against the coefficients
currently live and rejected if it moved in ways a normal 21-session refit does
not: a factor reversing sign, or a coefficient jumping by more than a stated
multiple. On rejection the previous coefficients stay live and the event goes to
the ledger, because a refit that had to be held back is a fact about the data
that someone needs to see.

The first fit has nothing to compare against and is always accepted. Thereafter
the gate can only keep the older model; it never loosens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["RefitVerdict", "review_refit", "SIGN_FLIP_FLOOR", "MAX_MAGNITUDE_RATIO"]

#: A coefficient smaller than this is noise around zero; its sign carries no
#: meaning and flipping it is not evidence of anything.
SIGN_FLIP_FLOOR = 0.002

#: How far a coefficient may move, as a multiple of its previous magnitude,
#: before the refit is treated as a different model rather than an update.
MAX_MAGNITUDE_RATIO = 5.0

#: How many factors may flip sign before the refit is rejected outright. One
#: marginal factor reversing is ordinary; several at once is a different fit.
MAX_SIGN_FLIPS = 2


@dataclass
class RefitVerdict:
    """Whether a proposed refit may go live, and why."""

    accepted: bool
    reasons: List[str] = field(default_factory=list)
    sign_flips: List[str] = field(default_factory=list)
    magnitude_jumps: List[str] = field(default_factory=list)
    compared_against: Optional[str] = None

    def summary(self) -> str:
        if self.accepted and not self.reasons:
            return "refit accepted"
        head = "refit accepted" if self.accepted else "refit REJECTED"
        return f"{head}: " + "; ".join(self.reasons)

    def to_dict(self) -> Dict[str, object]:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "sign_flips": list(self.sign_flips),
            "magnitude_jumps": list(self.magnitude_jumps),
            "compared_against": self.compared_against,
        }


def review_refit(
    proposed: Dict[str, float],
    previous: Optional[Dict[str, float]],
    previous_train_end: Optional[str] = None,
    *,
    proposed_estimator: Optional[str] = None,
    previous_estimator: Optional[str] = None,
) -> RefitVerdict:
    """Compare a proposed coefficient set against the one currently live.

    A CHANGE OF ESTIMATOR IS NOT A REFIT. The comparison below is arithmetic on
    coefficient magnitudes, and that only means something when both sets came
    out of the same estimator. Across a switch it means nothing: ridge spreads
    a penalty across collinear inputs while Fama-MacBeth gates on significance
    and shrinks what survives, so the same data produces different numbers by
    construction. On the recorded ridge-to-Fama-MacBeth change that read as a
    sign flip and a 5.4x jump -- indistinguishable, by magnitude alone, from
    the corrupted upstream date this gate exists to catch.

    So an estimator change is treated as a FIRST FIT for the new estimator:
    accepted, because there is nothing comparable to hold on to, and recorded
    loudly, because a model replacement passing through the promotion gate is
    exactly the event someone needs to see in the ledger.
    """
    if previous is None or not previous:
        return RefitVerdict(accepted=True, reasons=["no previous fit to compare against"])

    if (proposed_estimator is not None and previous_estimator is not None
            and str(proposed_estimator) != str(previous_estimator)):
        return RefitVerdict(
            accepted=True,
            reasons=[
                f"the estimator changed from {previous_estimator!r} to "
                f"{proposed_estimator!r}, so there is no comparable previous "
                f"fit. Reviewed as a first fit; the coefficient comparison was "
                f"NOT applied and this replacement is on the record"
            ],
            compared_against=previous_train_end,
        )

    if not proposed:
        return RefitVerdict(accepted=False, reasons=["proposed fit has no coefficients"],
                            compared_against=previous_train_end)

    shared = sorted(set(proposed) & set(previous))
    if not shared:
        # A disjoint feature set is a deliberate change to the model, not a
        # refit, and it has to be reviewed by a person rather than waved past.
        return RefitVerdict(
            accepted=False,
            reasons=["proposed fit shares no factors with the live model"],
            compared_against=previous_train_end,
        )

    flips: List[str] = []
    jumps: List[str] = []
    for name in shared:
        new = float(proposed[name])
        old = float(previous[name])
        if not np.isfinite(new):
            jumps.append(f"{name}: proposed value is not finite")
            continue
        if abs(old) >= SIGN_FLIP_FLOOR and abs(new) >= SIGN_FLIP_FLOOR and old * new < 0:
            flips.append(f"{name}: {old:+.5f} -> {new:+.5f}")
        if abs(old) >= SIGN_FLIP_FLOOR:
            ratio = abs(new) / abs(old)
            if ratio > MAX_MAGNITUDE_RATIO:
                jumps.append(f"{name}: |{old:+.5f}| -> |{new:+.5f}| ({ratio:.1f}x)")

    reasons: List[str] = []
    if len(flips) > MAX_SIGN_FLIPS:
        reasons.append(
            f"{len(flips)} factors reversed sign, above the {MAX_SIGN_FLIPS} allowed"
        )
    if jumps:
        reasons.append(
            f"{len(jumps)} coefficient(s) moved more than {MAX_MAGNITUDE_RATIO:g}x"
        )

    return RefitVerdict(
        accepted=not reasons,
        reasons=reasons or ["within tolerance of the live model"],
        sign_flips=flips,
        magnitude_jumps=jumps,
        compared_against=previous_train_end,
    )
