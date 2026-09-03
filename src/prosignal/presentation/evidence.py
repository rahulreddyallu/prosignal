"""Group the model's themes into categories a reader can act on.

WHAT THIS LAYER IS FOR. Named as the model names them -- `resid_mom`,
`prox_52w`, `deliv_pct` -- the engine's own columns are unreadable to anyone
who did not build them, and the old screen printed them raw:

    resid_mom: +1.55 sd vs the universe, raises the score by 0.0316
    (coefficient +0.02036)

That sentence is true, and it is the engine talking to itself.

WHY IT IS KEYED ON FAMILIES. The model does not fit individual factors. It
averages them into families and fits ONE coefficient per family -- see
`crossmodel.FAMILIES`. This table was keyed on the individual factors for one
release after that change, and the consequence was total: the key sets did not
intersect, so every category reported "Not available", `confirmation_count`
returned (0, 0), and the card printed "0 of 0 areas support this" on every
name of every run. Nothing raised. `FAMILY_MAP` is now asserted equal to
`crossmodel.FAMILIES` in the tests, so a family added without a label fails the
suite rather than vanishing from the screen.

WHY ORIENTATION COMES FROM THE FIT. Each category's reading is oriented by the
SIGN OF THE FITTED COEFFICIENT, not by a hardcoded expectation of which
direction is good. Those two diverged: `reversal` carries a positive literature
prior and a negative fitted coefficient, so a table asserting "high is
favourable" would have told a reader a name was strong on an axis that was
lowering its score. Orienting by the coefficient cannot drift from the model,
because it IS the model: positive always means "this theme is raising this
name's score", which is the question the card exists to answer.

WHAT A GATED THEME IS. The estimator sets a theme it could not measure past its
significance floor to EXACTLY zero. That is not a neutral reading -- it is a
theme that was not used. It is reported as its own state, never averaged into a
verdict, because "neutral" would say the model looked and found nothing while
in fact it declined to look.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..features.families import FAMILIES

__all__ = [
    "EVIDENCE_CATEGORIES", "FAMILY_MAP", "FACTOR_MAP", "COMPOSITE_KEYS",
    "MODEL_KEYS", "Category", "Signal",
    "build_evidence", "category_summary", "confirmation_count",
]

#: A coefficient at or below this is the estimator's "not used", not a small
#: weight. `gated_shrink` writes exactly 0.0, so the tolerance only guards
#: against float round-trips through JSON.
GATED = 1e-12

#: Family -> (category key, the reader's name for it).
#:
#: One family, one category. The previous table folded several families into a
#: shared "Momentum" heading, which would now be actively misleading: `mom` and
#: `reversal` are opposite sides of one axis and the fit prices them in
#: opposite directions, so a single momentum verdict would net a name's
#: medium-term strength against its recent fall and report the residue as
#: though it were a view.
FAMILY_MAP: Dict[str, Tuple[str, str]] = {
    "mom":      ("momentum",      "Medium-term momentum"),
    "reversal": ("reversal",      "Recent one-month move"),
    "lottery":  ("lottery",       "Lottery-like payoff shape"),
    "skew":     ("skew",          "Return skewness"),
    "beta":     ("beta",          "Market beta"),
    "drawdown": ("drawdown",      "Drawdown depth"),
    "delivery": ("participation", "Delivery-backed participation"),
    "value":    ("valuation",     "Valuation"),
    "quality":  ("quality",       "Business quality"),
}

#: The hand-weighted composite's factors. Reachable only when the fitted model
#: is unavailable AND `allow_composite_fallback` is on, which the shipped config
#: forbids -- but the path exists, so it renders rather than falling through to
#: an empty panel. These carry a fixed orientation because a composite weight is
#: always positive and therefore says nothing about direction.
FACTOR_MAP: Dict[str, Tuple[str, str, bool]] = {
    "momentum_12_1":            ("momentum",  "Twelve-month momentum", True),
    "sector_relative_strength": ("momentum",  "Strength vs sector and market", True),
    "value":                    ("valuation", "Valuation", True),
    "quality":                  ("quality",   "Business quality", True),
}

#: Keys that identify each scorer from its output alone. `_scorer_used` reads
#: these to tell the operator which model ranked the universe, and it must not
#: be keyed on anything that can drift from the engine.
MODEL_KEYS = frozenset(FAMILY_MAP)
COMPOSITE_KEYS = frozenset(FACTOR_MAP)

#: Display order: what moved the book first, what constrains it last.
EVIDENCE_CATEGORIES: Tuple[Tuple[str, str], ...] = (
    ("momentum",      "Momentum"),
    ("reversal",      "Recent Move"),
    ("participation", "Participation"),
    ("valuation",     "Valuation"),
    ("quality",       "Quality"),
    ("lottery",       "Payoff Shape"),
    ("skew",          "Skewness"),
    ("beta",          "Market Beta"),
    ("drawdown",      "Drawdown"),
)

_LABELS = dict(EVIDENCE_CATEGORIES)

#: Categories that read as a LEVEL rather than a direction. "Strong risk" is
#: ambiguous about whether that is good news; "Contained" is not.
#: `risk` split into `beta` and `drawdown`, and `skew` left `lottery`; all
#: five read as a LEVEL rather than a direction.
_LEVEL_CATEGORIES = frozenset({"beta", "drawdown", "lottery", "skew"})

#: Standardised-deviation thresholds for a verdict. A name inside +/-0.35 sd of
#: the universe is genuinely unremarkable on that axis and is called neutral
#: rather than being pushed into a direction it does not support.
_STRONG = 0.90
_MILD = 0.35


@dataclass(frozen=True)
class Signal:
    """One theme, in the reader's language."""

    label: str
    #: Standardised deviations from the universe mean, oriented by the FITTED
    #: coefficient so that positive always means "raises this name's score".
    oriented_sd: float
    raw_value: Optional[float]
    #: Absolute coefficient. How much this theme moves the score at all.
    weight: float
    reading: str


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    verdict: str
    detail: str
    signals: List[Signal] = field(default_factory=list)
    available: bool = True


def _verdict(sd: float, key: str) -> str:
    if key in _LEVEL_CATEGORIES:
        if sd >= _MILD:
            return "Contained"
        if sd > -_MILD:
            return "Moderate"
        return "Elevated"
    if sd >= _STRONG:
        return "Strong"
    if sd >= _MILD:
        return "Supportive"
    if sd > -_MILD:
        return "Neutral"
    if sd > -_STRONG:
        return "Soft"
    return "Weak"


def _reading(label: str, sd: float) -> str:
    """Describe one theme without repeating the standardised number.

    The magnitude is carried in `oriented_sd` for anyone who wants it. Prose
    that says "+1.55 sd" is the failure this layer exists to correct.
    """
    if abs(sd) <= _MILD:
        return f"{label} is close to the market average."
    strength = "well " if abs(sd) >= _STRONG else ""
    direction = "in this name's favour" if sd > 0 else "against this name"
    return f"{label} reads {strength}{direction}."


def _orientation(name: str, detail: Dict[str, Any]) -> Optional[float]:
    """+1 or -1: which way this theme pushes THIS engine's score.

    From the fitted coefficient for a family, and from the table for a
    composite factor, whose weight is always positive and therefore carries no
    direction. None means the orientation is unknown and the theme cannot be
    read honestly.
    """
    if name in FAMILY_MAP:
        weight = detail.get("weight")
        if weight is None:
            return None
        return 1.0 if float(weight) >= 0 else -1.0
    mapped = FACTOR_MAP.get(name)
    if mapped is None:
        return None
    return 1.0 if mapped[2] else -1.0


def build_evidence(factors: Dict[str, Any]) -> List[Category]:
    """Turn the engine's per-name theme block into reader-facing categories.

    `factors` is what the engine already serialises: raw value, standardised
    loading, model weight and an availability flag, keyed by whatever the
    scorer that ran uses.
    """
    grouped: Dict[str, List[Signal]] = {key: [] for key, _ in EVIDENCE_CATEGORIES}
    unavailable: Dict[str, int] = {key: 0 for key, _ in EVIDENCE_CATEGORIES}
    gated: Dict[str, bool] = {key: False for key, _ in EVIDENCE_CATEGORIES}

    for name, detail in (factors or {}).items():
        mapped = FAMILY_MAP.get(name)
        key, label = mapped if mapped else (FACTOR_MAP.get(name) or (None, None))[:2]
        if key is None:
            # An unmapped theme is a gap in this table, not something to
            # quietly drop into a bucket where it does not belong. The tests
            # assert FAMILY_MAP against the engine so this cannot happen for a
            # family; it can for a factor the composite adds.
            continue
        detail = detail or {}
        weight = detail.get("weight")
        # A theme the estimator gated out was NOT consulted and found neutral.
        # Averaging its zero into a verdict would say the opposite.
        if name in FAMILY_MAP and weight is not None and abs(float(weight)) <= GATED:
            gated[key] = True
            continue
        if not detail.get("available", False):
            unavailable[key] += 1
            continue
        sd = detail.get("standardised")
        direction = _orientation(name, detail)
        if not isinstance(sd, (int, float)) or direction is None:
            unavailable[key] += 1
            continue
        oriented = float(sd) * direction
        grouped[key].append(Signal(
            label=label,
            oriented_sd=oriented,
            raw_value=detail.get("raw"),
            weight=abs(float(weight or 0.0)),
            reading=_reading(label, oriented),
        ))

    out: List[Category] = []
    for key, label in EVIDENCE_CATEGORIES:
        signals = sorted(grouped[key], key=lambda s: -abs(s.oriented_sd))
        if not signals:
            # "Not used" and "Not available" are different facts and the card
            # must not blur them: one is a theme the estimator declined to
            # weight, the other is a theme whose inputs this name lacks.
            out.append(Category(
                key=key, label=label,
                verdict="Not used" if gated[key] else "Not available",
                detail=_missing_detail(key, unavailable[key], gated[key]),
                signals=[], available=False,
            ))
            continue
        # Weight the category by how much each theme actually moves the score.
        # An equal average would let a theme the model barely uses outvote the
        # one carrying the block.
        total_w = sum(s.weight for s in signals)
        if total_w > 0:
            agg = sum(s.oriented_sd * s.weight for s in signals) / total_w
        else:
            agg = sum(s.oriented_sd for s in signals) / len(signals)
        out.append(Category(
            key=key, label=label, verdict=_verdict(agg, key),
            detail=signals[0].reading, signals=signals, available=True,
        ))
    return out


def _missing_detail(key: str, count: int, was_gated: bool) -> str:
    if was_gated:
        return (
            "The model could not measure this theme past its significance "
            "floor on its own training window, so it was set to zero rather "
            "than given a weight the data did not support. It did not "
            "contribute to this name's ranking either way."
        )
    if key in ("valuation", "quality"):
        return (
            "No published financials covered enough of the panel to build this "
            "theme, so it did not contribute to this name's ranking."
        )
    if count:
        return "The inputs for this theme were not available for this name."
    return "This theme has no inputs for this name."


def category_summary(categories: Sequence[Category]) -> List[str]:
    """The few phrases the recommendation card shows before it is opened."""
    out: List[str] = []
    for cat in categories:
        if not cat.available or cat.verdict in {"Neutral", "Moderate"}:
            continue
        out.append(f"{cat.label} {cat.verdict.lower()}")
    return out


def confirmation_count(categories: Sequence[Category]) -> Tuple[int, int]:
    """How many PRICED themes raise this name's score, out of those measured.

    Deliberately NOT a count of independent confirmations. The themes are
    correlated by construction -- they are averages of overlapping factors --
    so reading "4 of 5 agree" as four independent votes would manufacture
    exactly the confidence this engine refuses to claim elsewhere. What this
    reports is a DECOMPOSITION of one number: of the themes the model actually
    priced, how many push this name up and how many push it down.

    Gated themes are excluded from both halves. A theme the estimator declined
    to use is not evidence for or against.
    """
    judged = [c for c in categories if c.available]
    positive = [c for c in judged
                if c.verdict in {"Strong", "Supportive", "Contained"}]
    return len(positive), len(judged)
