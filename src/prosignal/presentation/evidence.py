"""Group the model's factors into categories a reader can act on.

The engine fits seventeen cross-sectional factors. Named as the model names
them -- resid_mom, prox_52w, deliv_pct, amihud -- they are unreadable to
anyone who did not build them, and the old screen printed them raw:

    resid_mom: +1.55 sd vs the universe, raises the score by 0.0316
    (coefficient +0.02036)

That sentence is true, and it is the engine talking to itself.

WHAT THIS ENGINE ACTUALLY MEASURES. The categories below are drawn from the
factors that exist, not from a technical-analysis taxonomy the model does not
use. There is no RSI here, no MACD, no moving-average crossover, no EMA_50 --
this is a cross-sectional ranking model, and inventing indicator readings to
fill a familiar-looking layout would be fabricating evidence. Where a
conventional category has no factor behind it, it is absent rather than
padded, and Participation carries India-specific delivery data that a generic
"volume" heading would misdescribe.

A category's verdict comes from the standardised loadings of its own factors,
weighted the way the model weights them, so the reading a reader sees is the
one that moved the score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Factor -> (category, human name, whether a high reading is favourable).
#: Every factor the model fits appears exactly once. A factor absent from this
#: table would silently vanish from the evidence, so the mapping is asserted
#: complete against the engine in the tests.
FACTOR_MAP: Dict[str, Tuple[str, str, bool]] = {
    # Momentum -- the model's largest and most concentrated block.
    "resid_mom":      ("momentum", "Momentum vs the market", True),
    "mom_6_1":        ("momentum", "Six-month momentum", True),
    "resid_reversal": ("momentum", "Recent one-month move, market-adjusted", False),
    "idio_vol":       ("risk", "Stock-specific volatility", False),
    "idio_skew":      ("risk", "Lottery-like payoff shape", False),
    # Trend position -- where price sits in its own range.
    "prox_52w":       ("trend", "Position in the 52-week range", True),
    # Participation -- Indian delivery data, not just traded volume.
    "deliv_pct":      ("participation", "Share of trading taken to delivery", True),
    "deliv_trend":    ("participation", "Trend in delivered share", True),
    "turnover_ratio": ("participation", "Turnover relative to free float", True),
    # Valuation.
    "earnings_yield": ("valuation", "Earnings yield", True),
    "fcf_yield":      ("valuation", "Free cash flow yield", True),
    "ebitda_to_ev":   ("valuation", "Operating profit to enterprise value", True),
    "book_to_price":  ("valuation", "Book value to price", True),
    "sales_to_price": ("valuation", "Sales to price", True),
    # Risk -- every one of these is favourable when LOW.
    "downside_vol":   ("risk", "Downside volatility", False),
    "max_dd_120":     ("risk", "Deepest fall over six months", False),
    "max5_21":        ("risk", "Largest single-day spike", False),
    "beta_120":       ("risk", "Sensitivity to the market", False),
    "amihud":         ("risk", "Cost of trading size", False),
}

#: Display order. Momentum first because it carries the most weight in the fit.
EVIDENCE_CATEGORIES: Tuple[Tuple[str, str], ...] = (
    ("momentum", "Momentum"),
    ("trend", "Trend Position"),
    ("participation", "Participation"),
    ("valuation", "Valuation"),
    ("risk", "Risk"),
)

_LABELS = dict(EVIDENCE_CATEGORIES)

#: Standardised-deviation thresholds for a verdict. A name inside +/-0.35 sd of
#: the universe is genuinely unremarkable on that axis and is called neutral
#: rather than being pushed into a direction it does not support.
_STRONG = 0.90
_MILD = 0.35


@dataclass(frozen=True)
class Signal:
    """One factor, in the reader's language."""

    label: str
    #: Standardised deviations from the universe mean, oriented so that
    #: positive always means favourable regardless of the raw factor's sign.
    oriented_sd: float
    raw_value: Optional[float]
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


def _verdict(sd: float) -> str:
    if sd >= _STRONG:
        return "Strong"
    if sd >= _MILD:
        return "Supportive"
    if sd > -_MILD:
        return "Neutral"
    if sd > -_STRONG:
        return "Soft"
    return "Weak"


#: Risk reads as a level, not a direction: "Strong risk" would be ambiguous
#: about whether that is good news.
_RISK_VERDICT = {
    "Strong": "Contained", "Supportive": "Contained", "Neutral": "Moderate",
    "Soft": "Elevated", "Weak": "Elevated",
}


def _reading(label: str, sd: float, favourable_high: bool) -> str:
    """Describe one factor without repeating the standardised number.

    The magnitude is carried in `oriented_sd` for anyone who wants it. Prose
    that says "+1.55 sd" is the failure this layer exists to correct.
    """
    if abs(sd) <= _MILD:
        return f"{label} is close to the market average."
    strength = "well" if abs(sd) >= _STRONG else ""
    direction = "above" if (sd > 0) == favourable_high else "below"
    return f"{label} sits {strength} {direction} the market average.".replace("  ", " ")


def build_evidence(factors: Dict[str, Any]) -> List[Category]:
    """Turn the model's factor detail into reader-facing categories.

    `factors` is the per-name block the engine already serialises: raw value,
    standardised loading, model weight and an availability flag.
    """
    grouped: Dict[str, List[Signal]] = {key: [] for key, _ in EVIDENCE_CATEGORIES}
    unavailable: Dict[str, int] = {key: 0 for key, _ in EVIDENCE_CATEGORIES}

    for name, detail in (factors or {}).items():
        mapped = FACTOR_MAP.get(name)
        if mapped is None:
            # An unmapped factor is a gap in this table, not something to
            # quietly drop into a bucket where it does not belong.
            continue
        key, label, favourable_high = mapped
        if not (detail or {}).get("available", False):
            unavailable[key] += 1
            continue
        sd = detail.get("standardised")
        if not isinstance(sd, (int, float)):
            unavailable[key] += 1
            continue
        oriented = float(sd) if favourable_high else -float(sd)
        grouped[key].append(Signal(
            label=label,
            oriented_sd=oriented,
            raw_value=detail.get("raw"),
            weight=abs(float(detail.get("weight") or 0.0)),
            reading=_reading(label, oriented, favourable_high),
        ))

    out: List[Category] = []
    for key, label in EVIDENCE_CATEGORIES:
        signals = sorted(grouped[key], key=lambda s: -abs(s.oriented_sd))
        if not signals:
            out.append(Category(
                key=key, label=label, verdict="Not available",
                detail=_missing_detail(key, unavailable[key]),
                signals=[], available=False,
            ))
            continue
        # Weight the category by how much each factor actually moves the score.
        # An equal average would let a factor the model barely uses outvote the
        # one carrying the block.
        total_w = sum(s.weight for s in signals)
        if total_w > 0:
            agg = sum(s.oriented_sd * s.weight for s in signals) / total_w
        else:
            agg = sum(s.oriented_sd for s in signals) / len(signals)
        verdict = _verdict(agg)
        if key == "risk":
            verdict = _RISK_VERDICT[verdict]
        out.append(Category(
            key=key, label=label, verdict=verdict,
            detail=signals[0].reading, signals=signals, available=True,
        ))
    return out


def _missing_detail(key: str, count: int) -> str:
    if key == "valuation":
        return (
            "No published financials were current enough to use, so valuation "
            "did not contribute to this name's ranking."
        )
    if count:
        return "The inputs for this category were not available for this name."
    return "This category has no inputs for this name."


def category_summary(categories: Sequence[Category]) -> List[str]:
    """The few phrases the recommendation card shows before it is opened."""
    out: List[str] = []
    for cat in categories:
        if not cat.available or cat.verdict in {"Neutral", "Moderate"}:
            continue
        out.append(f"{cat.label} {cat.verdict.lower()}")
    return out


def confirmation_count(categories: Sequence[Category]) -> Tuple[int, int]:
    """How many categories back the name, out of those that could be judged.

    Reported as "4 of 5 areas support this" rather than a single blended
    number, because the blend is what hid the disagreement in the first place.
    """
    judged = [c for c in categories if c.available]
    positive = [c for c in judged
                if c.verdict in {"Strong", "Supportive", "Contained"}]
    return len(positive), len(judged)
